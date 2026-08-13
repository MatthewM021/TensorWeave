"""Strict construction helpers for reproducible V3 configurations."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from .forest import ForestConfig, RoutedTensorLanguageModel


_MODEL_KEYS = {field.name for field in fields(ForestConfig)}
_ROOT_KEYS = {
    "schema_version",
    "seed",
    "device",
    "dtype",
    "determinism",
    "model",
    "architecture",
    "readout",
    "validation",
}


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def load_forest_config(path: str | Path) -> ForestConfig:
    """Load a checked YAML file and return its executable model config."""

    source = Path(path)
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(document, "configuration")
    unknown_root = set(root) - _ROOT_KEYS
    if unknown_root:
        raise ValueError(f"unknown configuration keys: {sorted(unknown_root)}")
    version = root.get("schema_version")
    if isinstance(version, bool) or version != 1:
        raise ValueError("schema_version must be integer 1")
    model = _mapping(root.get("model"), "model")
    unknown_model = set(model) - _MODEL_KEYS
    missing_model = {"branches", "d_model", "cp_rank", "vocab_size"} - set(model)
    if unknown_model:
        raise ValueError(f"unknown model keys: {sorted(unknown_model)}")
    if missing_model:
        raise ValueError(f"missing model keys: {sorted(missing_model)}")

    architecture = _mapping(root.get("architecture", {}), "architecture")
    if set(architecture) != {"operator", "global_lane"}:
        raise ValueError("architecture must define only operator and global_lane")
    if architecture["operator"] != "scale_shared_cp_merge":
        raise ValueError("unsupported architecture operator")
    if architecture["global_lane"] is not True:
        raise ValueError("Milestone 1 requires the dedicated global lane")
    return ForestConfig(**dict(model))


def build_model(config: ForestConfig | str | Path) -> RoutedTensorLanguageModel:
    """Build a model from a validated dataclass or YAML path."""

    resolved = load_forest_config(config) if isinstance(config, (str, Path)) else config
    if not isinstance(resolved, ForestConfig):
        raise TypeError("config must be ForestConfig or a YAML path")
    return RoutedTensorLanguageModel(resolved)


__all__ = ["build_model", "load_forest_config"]
