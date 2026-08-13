from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelOutput:
    logits: torch.Tensor
    aux_losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: Dict[str, torch.Tensor] = field(default_factory=dict)


class PredictiveModel(nn.Module):
    model_name = "predictive_model"

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def structural_metrics(self, sequence_length: int) -> Dict[str, float]:
        raise NotImplementedError


class TokenPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, dimension, max_length, pad_id, dropout=0.0):
        super().__init__()
        self.token = nn.Embedding(vocab_size, dimension, padding_idx=pad_id)
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / max(1, dimension))
        )
        positional = torch.zeros(max_length, dimension)
        positional[:, 0::2] = torch.sin(positions * frequencies)
        if dimension > 1:
            positional[:, 1::2] = torch.cos(
                positions * frequencies[: positional[:, 1::2].shape[1]]
            )
        self.register_buffer("position", positional, persistent=True)
        self.position_scale = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length
        self.dimension = dimension

    def forward(self, tokens):
        length = tokens.shape[1]
        if length > self.max_length:
            raise ValueError("sequence exceeds max_length")
        pos = self.position[:length].to(dtype=self.token.weight.dtype)
        return self.dropout(
            self.norm(self.token(tokens) + self.position_scale * pos[None, :, :])
        )


class TensorMerge(nn.Module):
    """CP-factorized three-index contraction plus a stable residual path."""

    def __init__(self, dimension: int, tensor_rank: int, dropout: float = 0.0):
        super().__init__()
        self.left = nn.Linear(dimension, tensor_rank, bias=False)
        self.right = nn.Linear(dimension, tensor_rank, bias=False)
        self.output = nn.Linear(tensor_rank, dimension, bias=False)
        self.left_residual = nn.Linear(dimension, dimension, bias=False)
        self.right_residual = nn.Linear(dimension, dimension, bias=False)
        self.gate = nn.Linear(dimension, dimension)
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.left.weight)
        nn.init.xavier_uniform_(self.right.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.eye_(self.left_residual.weight)
        nn.init.eye_(self.right_residual.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -0.5)

    def forward(self, left, right):
        interaction = self.output(
            torch.tanh(self.left(left)) * torch.tanh(self.right(right))
        )
        residual = 0.5 * (
            self.left_residual(left) + self.right_residual(right)
        )
        gate = torch.sigmoid(self.gate(0.5 * (left + right)))
        merged = residual + gate * interaction
        return self.norm(merged + self.dropout(F.gelu(merged)))


class PairDisentangler(nn.Module):
    """Two-site map initialized exactly as the identity."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.weight = nn.Parameter(torch.eye(2 * dimension))

    def forward(self, left, right):
        mixed = F.linear(torch.cat((left, right), dim=-1), self.weight)
        return mixed.split(self.dimension, dim=-1)

    def orthogonality_error(self):
        size = self.weight.shape[0]
        identity = torch.eye(size, device=self.weight.device, dtype=self.weight.dtype)
        gram = self.weight.T @ self.weight
        return torch.mean((gram - identity) ** 2)


class RankGate(nn.Module):
    def __init__(self, dimension: int, initial_logit: float = 2.5):
        super().__init__()
        self.logits = nn.Parameter(torch.full((dimension,), initial_logit))

    def values(self):
        return torch.sigmoid(self.logits)

    def forward(self, x):
        return x * self.values()

    def regularizer(self):
        return self.values().mean()

    def effective_rank(self):
        g = self.values()
        return g.sum().pow(2) / (g.pow(2).sum() + 1e-8)


class TreeTensorReducer(nn.Module):
    def __init__(
        self,
        dimension: int,
        tensor_rank: int,
        max_leaves: int,
        use_disentanglers: bool,
        adaptive_rank: bool,
        dropout: float = 0.0,
        share_levels: bool = False,
    ):
        super().__init__()
        self.dimension = dimension
        self.max_leaves = max_leaves
        self.max_levels = max(1, math.ceil(math.log2(max_leaves)))
        self.use_disentanglers = use_disentanglers
        self.adaptive_rank = adaptive_rank
        self.share_levels = share_levels
        count = 1 if share_levels else self.max_levels
        self.merges = nn.ModuleList(
            TensorMerge(dimension, tensor_rank, dropout) for _ in range(count)
        )
        self.disentanglers = (
            nn.ModuleList(PairDisentangler(dimension) for _ in range(count))
            if use_disentanglers
            else nn.ModuleList()
        )
        self.rank_gates = (
            nn.ModuleList(RankGate(dimension) for _ in range(count))
            if adaptive_rank
            else nn.ModuleList()
        )

    def _idx(self, level):
        return 0 if self.share_levels else min(level, len(self.merges) - 1)

    @staticmethod
    def _pad_even(states, masks):
        if states.shape[-2] % 2 == 0:
            return states, masks
        return (
            torch.cat((states, torch.zeros_like(states[..., :1, :])), dim=-2),
            torch.cat((masks, torch.zeros_like(masks[..., :1, :])), dim=-2),
        )

    @staticmethod
    def _disentangle(states, masks, module):
        n = states.shape[-2]
        count = max(0, (n - 1) // 2)
        if count == 0:
            return states, masks
        li = torch.arange(1, 1 + 2 * count, 2, device=states.device)
        ri = li + 1
        left = states.index_select(-2, li)
        right = states.index_select(-2, ri)
        lm = masks.index_select(-2, li)
        rm = masks.index_select(-2, ri)
        nl, nr = module(left, right)
        joint = lm * rm
        result = states.clone()
        result.index_copy_(-2, li, joint * nl + (1 - joint) * left)
        result.index_copy_(-2, ri, joint * nr + (1 - joint) * right)
        return result, masks

    def forward(self, states, masks):
        if states.shape[:-1] != masks.shape[:-1] or masks.shape[-1] != 1:
            raise ValueError("incompatible state/mask shapes")
        if states.shape[-2] > self.max_leaves:
            raise ValueError("too many leaves")
        current = states
        current_mask = masks.to(states.dtype)
        orth: List[torch.Tensor] = []
        rank_terms: List[torch.Tensor] = []
        eff: List[torch.Tensor] = []
        level = 0
        while current.shape[-2] > 1:
            current, current_mask = self._pad_even(current, current_mask)
            idx = self._idx(level)
            if self.use_disentanglers:
                module = self.disentanglers[idx]
                current, current_mask = self._disentangle(
                    current, current_mask, module
                )
                orth.append(module.orthogonality_error())
            left, right = current[..., 0::2, :], current[..., 1::2, :]
            lm, rm = current_mask[..., 0::2, :], current_mask[..., 1::2, :]
            both = lm * rm
            lo, ro = lm * (1 - rm), (1 - lm) * rm
            occ = both + lo + ro
            current = (
                both * self.merges[idx](left, right) + lo * left + ro * right
            ) / (occ + 1e-8)
            current_mask = lm + rm - lm * rm
            if self.adaptive_rank:
                gate = self.rank_gates[idx]
                current = gate(current)
                rank_terms.append(gate.regularizer())
                eff.append(gate.effective_rank())
            level += 1
        diag: Dict[str, torch.Tensor] = {
            "levels_used": torch.tensor(float(level), device=states.device),
            "effective_rank": torch.tensor(float(self.dimension), device=states.device),
        }
        if orth:
            diag["orthogonality"] = torch.stack(orth).mean()
        if rank_terms:
            diag["rank_regularizer"] = torch.stack(rank_terms).mean()
            diag["effective_rank"] = torch.stack(eff).mean()
        return current.squeeze(-2), current_mask.squeeze(-2), diag


class CheapCausalRouter(nn.Module):
    """O(BL) token-local router; no all-pairs attention."""

    def __init__(self, vocab_size, dimension, branches, context_lags=3):
        super().__init__()
        self.branches = branches
        self.context_lags = context_lags
        self.token_route_logits = nn.Embedding(vocab_size, branches)
        nn.init.normal_(self.token_route_logits.weight, std=0.20)
        self.lag_selector = nn.Embedding(vocab_size, context_lags + 1)
        nn.init.zeros_(self.lag_selector.weight)
        self.context_correction = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, branches),
        )
        nn.init.zeros_(self.context_correction[-1].weight)
        nn.init.zeros_(self.context_correction[-1].bias)
        self.temperature = 1.0
        self.hard = False

    def set_temperature(self, temperature: float, hard: bool = False):
        self.temperature = max(float(temperature), 1e-3)
        self.hard = bool(hard)

    def forward(self, tokens, embeddings, valid_mask):
        base = self.token_route_logits(tokens)
        candidates = [base]
        for lag in range(1, self.context_lags + 1):
            shifted = torch.zeros_like(base)
            shifted[:, lag:, :] = base[:, :-lag, :]
            candidates.append(shifted)
        stacked = torch.stack(candidates, dim=-2)
        lag_weights = torch.softmax(self.lag_selector(tokens), dim=-1).unsqueeze(-1)
        logits = (stacked * lag_weights).sum(dim=-2)
        logits = (logits + 0.25 * self.context_correction(embeddings)) / self.temperature
        probs = torch.softmax(logits, dim=-1)
        if self.hard:
            indices = probs.argmax(dim=-1)
            hard = F.one_hot(indices, self.branches).to(probs.dtype)
            assignments = hard - probs.detach() + probs
        else:
            assignments = probs
        valid = valid_mask.unsqueeze(-1).to(probs.dtype)
        assignments = assignments * valid
        token_count = valid.sum().clamp_min(1.0)
        entropy = -(probs.clamp_min(1e-8).log() * probs * valid).sum() / (
            token_count * math.log(max(2, self.branches))
        )
        load = assignments.sum(dim=(0, 1)) / token_count
        target = torch.full_like(load, 1.0 / self.branches)
        balance = self.branches * torch.mean((load - target) ** 2)
        active = torch.sum((load > 0.25 / self.branches).float())
        lagp = lag_weights.squeeze(-1)
        lag_entropy = -(lagp.clamp_min(1e-8).log() * lagp * valid).sum() / (
            token_count * math.log(self.context_lags + 1)
        )
        return assignments, {
            "router_entropy": entropy,
            "router_balance": balance,
            "router_active_branches": active,
            "router_lag_entropy": lag_entropy,
            "router_load": load,
        }


def last_valid_state(sequence, valid_mask):
    indices = valid_mask.long().sum(dim=1).clamp_min(1) - 1
    batch = torch.arange(sequence.shape[0], device=sequence.device)
    return sequence[batch, indices]
