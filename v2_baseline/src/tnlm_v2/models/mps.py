from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .components import ModelOutput, PredictiveModel


class MPSClassifier(PredictiveModel):
    def __init__(self, vocab_size, num_classes, pad_id, bond_dimension=16):
        super().__init__()
        self.model_name = "mps"
        self.bond_dimension = bond_dimension
        eye = torch.eye(bond_dimension).unsqueeze(0).repeat(vocab_size, 1, 1)
        core = eye + 0.03 * torch.randn_like(eye) / math.sqrt(bond_dimension)
        core[pad_id] = torch.eye(bond_dimension)
        self.core = nn.Parameter(core)
        self.left_boundary = nn.Parameter(torch.randn(bond_dimension))
        self.head = nn.Sequential(
            nn.LayerNorm(bond_dimension),
            nn.Linear(bond_dimension, bond_dimension),
            nn.Tanh(),
            nn.Linear(bond_dimension, num_classes),
        )

    def forward(self, tokens, valid_mask, routes=None):
        state = F.normalize(self.left_boundary[None, :].expand(tokens.shape[0], -1), dim=-1)
        for t in range(tokens.shape[1]):
            updated = torch.bmm(state.unsqueeze(1), self.core[tokens[:, t]]).squeeze(1)
            updated = F.normalize(updated, dim=-1)
            state = torch.where(valid_mask[:, t, None], updated, state)
        return ModelOutput(self.head(state))

    def structural_metrics(self, sequence_length):
        chi = float(self.bond_dimension)
        return {
            "bond_dimension": chi,
            "effective_rank": chi,
            "tree_depth": float(sequence_length),
            "active_branches": 1.0,
            "cut_capacity_bits_proxy": math.log2(max(1.000001, chi)),
            "stream_state_scalars_proxy": chi,
            "parameter_count": float(self.parameter_count),
        }
