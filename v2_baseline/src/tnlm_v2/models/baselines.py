from __future__ import annotations

import math
import torch
from torch import nn

from .components import ModelOutput, PredictiveModel, TokenPositionEmbedding, last_valid_state


class GRUBaseline(PredictiveModel):
    def __init__(self, vocab_size, num_classes, pad_id, max_length, dimension=32, layers=1, dropout=0.0):
        super().__init__()
        self.model_name = "gru"
        self.dimension = dimension
        self.layers = layers
        self.embedding = TokenPositionEmbedding(vocab_size, dimension, max_length, pad_id, dropout)
        self.gru = nn.GRU(
            dimension,
            dimension,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, num_classes),
        )

    def forward(self, tokens, valid_mask, routes=None):
        sequence, _ = self.gru(self.embedding(tokens))
        return ModelOutput(self.head(last_valid_state(sequence, valid_mask)))

    def structural_metrics(self, sequence_length):
        d = float(self.dimension)
        return {
            "bond_dimension": d,
            "effective_rank": d,
            "tree_depth": float(sequence_length),
            "active_branches": 1.0,
            "cut_capacity_bits_proxy": math.log2(max(1.000001, d)),
            "stream_state_scalars_proxy": self.layers * d,
            "parameter_count": float(self.parameter_count),
        }


class TransformerBaseline(PredictiveModel):
    def __init__(
        self,
        vocab_size,
        num_classes,
        pad_id,
        max_length,
        dimension=32,
        layers=2,
        heads=4,
        feedforward_multiplier=3,
        dropout=0.0,
    ):
        super().__init__()
        self.model_name = "transformer"
        self.dimension = dimension
        self.layers = layers
        self.heads = heads
        self.embedding = TokenPositionEmbedding(vocab_size, dimension, max_length, pad_id, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=feedforward_multiplier * dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(dimension), enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, num_classes),
        )

    def forward(self, tokens, valid_mask, routes=None):
        x = self.embedding(tokens)
        length = tokens.shape[1]
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1
        )
        encoded = self.encoder(x, mask=causal, src_key_padding_mask=~valid_mask)
        return ModelOutput(self.head(last_valid_state(encoded, valid_mask)))

    def structural_metrics(self, sequence_length):
        return {
            "bond_dimension": float(self.dimension),
            "effective_rank": float(self.dimension),
            "tree_depth": float(self.layers),
            "active_branches": float(self.heads),
            "cut_capacity_bits_proxy": None,
            "stream_state_scalars_proxy": 2.0 * self.layers * sequence_length * self.dimension,
            "attention_score_scalars_proxy": float(self.layers * self.heads * sequence_length**2),
            "parameter_count": float(self.parameter_count),
        }
