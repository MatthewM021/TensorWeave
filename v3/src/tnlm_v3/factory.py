"""Strict construction helpers for reproducible V3 configurations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .forest import ForestConfig, RoutedTensorLanguageModel
from .binding import BindingArchitectureConfig, BindingModelConfig, RoutedBindingModel
from .data import BindingTaskConfig
from .routing import CurriculumSchedule, RoutingMode
from .training import BindingLossConfig


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
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
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


@dataclass(frozen=True)
class BindingExperimentConfig:
    """Executable, hashable configuration for a Milestone-2 smoke condition."""

    condition: RoutingMode
    model_seed: int
    data_seed: int
    episodes: int
    sequence_length: int
    steps: int
    learning_rate: float
    weight_decay: float
    max_gradient_norm: float
    task: BindingTaskConfig
    model: BindingModelConfig
    loss: BindingLossConfig

    def __post_init__(self) -> None:
        mode = RoutingMode(self.condition)
        object.__setattr__(self, "condition", mode)
        if mode is not self.model.routing_mode:
            raise ValueError("condition must match model.routing_mode")
        if not isinstance(self.task, BindingTaskConfig):
            raise TypeError("task must be a BindingTaskConfig")
        if self.model.task != BindingArchitectureConfig.from_task(self.task):
            raise ValueError("model architecture must match the experiment task")
        for name in ("model_seed", "data_seed", "episodes", "sequence_length", "steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.episodes <= 0 or self.sequence_length <= 0 or self.steps <= 0:
            raise ValueError("episodes, sequence_length, and steps must be positive")
        for name in ("learning_rate", "weight_decay", "max_gradient_norm"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"{name} must be a real number")
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.learning_rate == 0 or self.max_gradient_norm == 0:
            raise ValueError("learning_rate and max_gradient_norm must be positive")
        if not self.task.min_length <= self.sequence_length <= self.task.max_length:
            raise ValueError("sequence_length must be inside the task length range")
        if (
            mode is RoutingMode.CURRICULUM
            and self.model.curriculum_schedule is not None
            and self.model.curriculum_schedule.end_step > self.steps
        ):
            raise ValueError("curriculum guidance must finish within the training run")

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        unknown = sorted(set(mapping) - expected)
        raise ValueError(f"invalid {name} keys; missing={missing}, unknown={unknown}")


def load_binding_experiment_config(path: str | Path) -> BindingExperimentConfig:
    """Load one strict oracle/curriculum/latent smoke configuration."""

    document = _mapping(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")), "configuration"
    )
    _exact_keys(
        document,
        {
            "schema_version",
            "condition",
            "model_seed",
            "data_seed",
            "task",
            "model",
            "routing",
            "loss",
            "training",
        },
        "configuration",
    )
    version = document["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("schema_version must be integer 1")

    task_values = dict(_mapping(document["task"], "task"))
    task_keys = {field.name for field in fields(BindingTaskConfig)}
    _exact_keys(task_values, task_keys, "task")
    task_values["heldout_key_value_pairs"] = tuple(
        tuple(pair) for pair in task_values["heldout_key_value_pairs"]
    )
    task = BindingTaskConfig(**task_values)

    model_values = dict(_mapping(document["model"], "model"))
    _exact_keys(
        model_values,
        {
            "d_model",
            "cp_rank",
            "router_hidden_dim",
            "scale_feature_dim",
            "straight_through_route_surrogate",
        },
        "model",
    )
    routing = dict(_mapping(document["routing"], "routing"))
    _exact_keys(routing, {"mode", "curriculum_seed", "schedule"}, "routing")
    mode = RoutingMode(routing["mode"])
    schedule_values = routing["schedule"]
    if mode is RoutingMode.CURRICULUM:
        schedule = CurriculumSchedule(**dict(_mapping(schedule_values, "schedule")))
    else:
        if schedule_values is not None:
            raise ValueError("non-curriculum conditions require schedule: null")
        schedule = None
    model = BindingModelConfig(
        task=task,
        routing_mode=mode,
        curriculum_schedule=schedule,
        curriculum_seed=routing["curriculum_seed"],
        **model_values,
    )

    loss_values = dict(_mapping(document["loss"], "loss"))
    _exact_keys(
        loss_values,
        {field.name for field in fields(BindingLossConfig)},
        "loss",
    )
    loss = BindingLossConfig(**loss_values)
    training = dict(_mapping(document["training"], "training"))
    _exact_keys(
        training,
        {
            "episodes",
            "sequence_length",
            "steps",
            "learning_rate",
            "weight_decay",
            "max_gradient_norm",
        },
        "training",
    )
    return BindingExperimentConfig(
        condition=RoutingMode(document["condition"]),
        model_seed=document["model_seed"],
        data_seed=document["data_seed"],
        task=task,
        model=model,
        loss=loss,
        **training,
    )


def build_binding_model(
    config: BindingExperimentConfig | BindingModelConfig | str | Path,
) -> RoutedBindingModel:
    """Build a routed binding model from a dataclass or experiment YAML."""

    if isinstance(config, (str, Path)):
        resolved = load_binding_experiment_config(config).model
    elif isinstance(config, BindingExperimentConfig):
        resolved = config.model
    else:
        resolved = config
    if not isinstance(resolved, BindingModelConfig):
        raise TypeError("config must be a binding model/experiment config or YAML path")
    return RoutedBindingModel(resolved)


__all__ = [
    "BindingExperimentConfig",
    "build_binding_model",
    "build_model",
    "load_binding_experiment_config",
    "load_forest_config",
]
