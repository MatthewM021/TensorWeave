from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .components import (
    CheapCausalRouter,
    ModelOutput,
    PredictiveModel,
    TokenPositionEmbedding,
    TreeTensorReducer,
    last_valid_state,
)


class FixedTreeTensorModel(PredictiveModel):
    def __init__(
        self,
        vocab_size,
        num_classes,
        pad_id,
        max_length,
        dimension=16,
        tensor_rank=16,
        use_disentanglers=False,
        adaptive_rank=False,
        dropout=0.0,
        share_levels=False,
    ):
        super().__init__()
        self.model_name = "fixed_mera" if use_disentanglers else "fixed_ttn"
        if adaptive_rank:
            self.model_name += "_adaptive"
        self.dimension = dimension
        self.use_disentanglers = use_disentanglers
        self.adaptive_rank = adaptive_rank
        self.embedding = TokenPositionEmbedding(
            vocab_size, dimension, max_length, pad_id, dropout
        )
        self.reducer = TreeTensorReducer(
            dimension,
            tensor_rank,
            max_length,
            use_disentanglers,
            adaptive_rank,
            dropout,
            share_levels,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, num_classes),
        )

    def forward(self, tokens, valid_mask, routes=None):
        states = self.embedding(tokens)
        root, _, diag = self.reducer(states, valid_mask.unsqueeze(-1).to(states.dtype))
        aux = {}
        if "orthogonality" in diag:
            aux["orthogonality"] = diag["orthogonality"]
        if "rank_regularizer" in diag:
            aux["rank_regularizer"] = diag["rank_regularizer"]
        return ModelOutput(self.head(root), aux, diag)

    def structural_metrics(self, sequence_length):
        depth = max(1, math.ceil(math.log2(max(2, sequence_length))))
        effective = float(self.dimension)
        if self.adaptive_rank:
            with torch.no_grad():
                effective = float(
                    torch.stack([g.effective_rank() for g in self.reducer.rank_gates])
                    .mean()
                    .cpu()
                )
        crossing = depth if self.use_disentanglers else 1
        return {
            "bond_dimension": float(self.dimension),
            "effective_rank": effective,
            "tree_depth": float(depth),
            "active_branches": 1.0,
            "cut_capacity_bits_proxy": crossing * math.log2(max(1.000001, effective)),
            "stream_state_scalars_proxy": depth * effective,
            "parameter_count": float(self.parameter_count),
        }


class RoutedTreeTensorModel(PredictiveModel):
    def __init__(
        self,
        vocab_size,
        num_classes,
        pad_id,
        max_length,
        branches,
        routing,
        dimension=16,
        tensor_rank=16,
        use_disentanglers=False,
        adaptive_rank=False,
        dropout=0.0,
        share_levels=False,
    ):
        super().__init__()
        if routing not in {"oracle", "learned"}:
            raise ValueError("routing must be oracle or learned")
        if branches & (branches - 1):
            raise ValueError("branches must be power of two")
        self.routing = routing
        self.branches = branches
        self.dimension = dimension
        self.adaptive_rank = adaptive_rank
        base = "routed_mera" if use_disentanglers else "routed_ttn"
        self.model_name = f"{base}_{routing}" + ("_adaptive" if adaptive_rank else "")
        self.embedding = TokenPositionEmbedding(
            vocab_size, dimension, max_length, pad_id, dropout
        )
        self.router = (
            CheapCausalRouter(vocab_size, dimension, branches, 3)
            if routing == "learned"
            else None
        )
        self.branch_reducer = TreeTensorReducer(
            dimension,
            tensor_rank,
            max_length,
            use_disentanglers,
            adaptive_rank,
            dropout,
            share_levels,
        )
        self.global_reducer = TreeTensorReducer(
            dimension,
            tensor_rank,
            branches,
            use_disentanglers,
            adaptive_rank,
            dropout,
            share_levels,
        )
        self.query_projection = nn.Linear(dimension, dimension)
        self.head = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, num_classes),
        )

    def set_router_temperature(self, temperature, hard=False):
        if self.router is not None:
            self.router.set_temperature(temperature, hard)

    def _oracle(self, routes, valid_mask, dtype):
        clipped = routes.clamp(0, self.branches - 1)
        onehot = F.one_hot(clipped, self.branches).to(dtype)
        return onehot * (routes >= 0).unsqueeze(-1).to(dtype) * valid_mask.unsqueeze(-1).to(dtype)

    def forward(self, tokens, valid_mask, routes=None):
        emb = self.embedding(tokens)
        router_diag: Dict[str, torch.Tensor] = {}
        if self.routing == "oracle":
            if routes is None:
                raise ValueError("oracle routing requires routes")
            assignments = self._oracle(routes, valid_mask, emb.dtype)
        else:
            assignments, router_diag = self.router(tokens, emb, valid_mask)
        masks = assignments.permute(0, 2, 1).unsqueeze(-1)
        states = emb.unsqueeze(1).expand(-1, self.branches, -1, -1)
        roots, root_masks, branch_diag = self.branch_reducer(states, masks)
        root, _, global_diag = self.global_reducer(roots, root_masks)
        logits = self.head(root + self.query_projection(last_valid_state(emb, valid_mask)))
        aux: Dict[str, torch.Tensor] = {}
        orth = [d["orthogonality"] for d in (branch_diag, global_diag) if "orthogonality" in d]
        if orth:
            aux["orthogonality"] = torch.stack(orth).mean()
        ranks = [d["rank_regularizer"] for d in (branch_diag, global_diag) if "rank_regularizer" in d]
        if ranks:
            aux["rank_regularizer"] = torch.stack(ranks).mean()
        if self.routing == "learned":
            aux["router_entropy"] = router_diag["router_entropy"]
            aux["router_balance"] = router_diag["router_balance"]
        diag = {"assignments": assignments, "branch_root_masks": root_masks}
        for prefix, values in (("branch", branch_diag), ("global", global_diag), ("router", router_diag)):
            for k, v in values.items():
                diag[f"{prefix}_{k}"] = v
        return ModelOutput(logits, aux, diag)

    def structural_metrics(self, sequence_length):
        depth = max(1, math.ceil(math.log2(max(2, sequence_length))))
        vals = []
        if self.adaptive_rank:
            with torch.no_grad():
                for reducer in (self.branch_reducer, self.global_reducer):
                    vals.extend(float(g.effective_rank().cpu()) for g in reducer.rank_gates)
        effective = sum(vals) / len(vals) if vals else float(self.dimension)
        return {
            "bond_dimension": float(self.dimension),
            "effective_rank": effective,
            "tree_depth": float(depth),
            "active_branches": float(self.branches),
            "cut_capacity_bits_proxy": self.branches * math.log2(max(1.000001, effective)),
            "stream_state_scalars_proxy": self.branches * depth * effective,
            "router_search_scalars_proxy": (
                sequence_length * self.branches * self.dimension if self.routing == "learned" else 0.0
            ),
            "parameter_count": float(self.parameter_count),
        }
