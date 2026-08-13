from __future__ import annotations

from .models import (
    FixedTreeTensorModel,
    GRUBaseline,
    MPSClassifier,
    RoutedTreeTensorModel,
    TransformerBaseline,
)

MODEL_NAMES = (
    "mps", "fixed_ttn", "fixed_mera", "fixed_ttn_adaptive", "fixed_mera_adaptive",
    "routed_ttn_oracle", "routed_ttn_oracle_widecore", "routed_ttn_learned", "routed_mera_oracle", "routed_mera_learned",
    "routed_ttn_oracle_adaptive", "routed_ttn_learned_adaptive",
    "routed_mera_oracle_adaptive", "routed_mera_learned_adaptive", "gru", "transformer",
)


def create_model(name, task_spec, max_length, settings=None):
    cfg = dict(settings or {})
    name = name.strip().lower()
    if name not in MODEL_NAMES:
        raise KeyError(name)
    d = int(cfg.get("tn_dimension", 16))
    r = int(cfg.get("tn_rank", d))
    common = dict(
        vocab_size=task_spec.vocab_size,
        num_classes=task_spec.num_classes,
        pad_id=task_spec.pad_id,
    )
    if name == "mps":
        return MPSClassifier(
            **common, bond_dimension=int(cfg.get("mps_bond_dimension", d))
        )
    if name.startswith("fixed_"):
        return FixedTreeTensorModel(
            **common,
            max_length=max_length,
            dimension=d,
            tensor_rank=r,
            use_disentanglers="mera" in name,
            adaptive_rank=name.endswith("_adaptive"),
            dropout=float(cfg.get("tn_dropout", 0.0)),
            share_levels=bool(cfg.get("share_levels", False)),
        )
    if name.startswith("routed_"):
        return RoutedTreeTensorModel(
            **common,
            max_length=max_length,
            branches=int(cfg.get("branches", task_spec.max_branches)),
            routing="oracle" if "_oracle" in name else "learned",
            dimension=d,
            tensor_rank=r,
            use_disentanglers="mera" in name,
            adaptive_rank=name.endswith("_adaptive"),
            dropout=float(cfg.get("tn_dropout", 0.0)),
            share_levels=bool(cfg.get("share_levels", False)),
        )
    bd = int(cfg.get("baseline_dimension", 32))
    if name == "gru":
        return GRUBaseline(
            **common,
            max_length=max_length,
            dimension=bd,
            layers=int(cfg.get("gru_layers", 1)),
            dropout=float(cfg.get("baseline_dropout", 0.0)),
        )
    return TransformerBaseline(
        **common,
        max_length=max_length,
        dimension=bd,
        layers=int(cfg.get("transformer_layers", 2)),
        heads=int(cfg.get("transformer_heads", 4)),
        feedforward_multiplier=int(cfg.get("transformer_ff_multiplier", 3)),
        dropout=float(cfg.get("baseline_dropout", 0.0)),
    )
