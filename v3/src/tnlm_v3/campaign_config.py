"""Strict, stage-gated Milestone-4 campaign configuration and planning.

Promotion paths receive lexical validation here.  The execution runner must
also resolve them under its declared external output root, require a regular file,
hash the file bytes, and semantically validate the promoted screen
campaign, manifest, executable bundle, selected models, and seed freshness.
Those checks require external artifacts and therefore do not belong to this
pure configuration parser.

Screen selection is exhaustive: it includes every trainable model except the
routed-oracle reference stratum, plus every derived compact-rank candidate.
Additional models are allowed, but they follow that same inclusion policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

import yaml

from .data import BindingTaskConfig


_MAX_CONFIG_BYTES = 1024 * 1024
_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_FAMILIES = {"routed", "gru", "cached_transformer", "causal_ttn"}
_ROLES = {"trainable_source", "derived_compact"}
_ROUTING = {"oracle", "curriculum", "latent"}
_MODEL_KEYS = {
    "model_id",
    "family",
    "role",
    "routing_mode",
    "parent_model_id",
    "architecture",
    "export",
}
_ARCHITECTURE_KEYS = {
    "routed": {
        "d_model",
        "cp_rank",
        "router_hidden_dim",
        "scale_feature_dim",
        "straight_through_route_surrogate",
        "curriculum_seed",
        "schedule",
    },
    "gru": {"d_model", "hidden_dim", "num_layers"},
    "cached_transformer": {"d_model", "num_heads", "num_layers", "ff_dim"},
    "causal_ttn": {"d_model", "cp_rank", "scale_feature_dim"},
}
_COMMON_ROOT_KEYS = {
    "schema_version",
    "campaign_id",
    "stage",
    "description",
    "claim_eligible",
    "implementation_policy",
    "task",
    "models",
    "pairs",
    "data",
    "training",
    "quality",
    "statistics",
    "runtime",
}
_RUN_DOMAIN = "tnlm-v3-milestone4-run-v1"
_PLAN_DOMAIN = "tnlm-v3-milestone4-plan-v1"
_MAX_YAML_NODES = 10_000
_MAX_YAML_DEPTH = 32


class CampaignStage(str, Enum):
    PILOT = "pilot"
    SCREEN = "screen"
    CONFIRMATORY = "confirmatory"


def _plain_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if value > 2**63 - 1:
        raise ValueError(f"{name} exceeds the signed 64-bit range")
    return value


def _finite_real(
    value: object,
    name: str,
    *,
    minimum: float = 0.0,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < minimum or (positive and result == minimum):
        relation = "greater than" if positive else "at least"
        raise ValueError(f"{name} must be {relation} {minimum}")
    return result


def _literal_true(value: object, name: str) -> bool:
    if value is not True:
        raise ValueError(f"{name} must be literal true")
    return True


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase identifier")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError(f"{name} must be a nonempty bounded string")
    return value


def _sha(value: object, name: str, pattern: re.Pattern[str] = _HEX64) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        length = 40 if pattern is _HEX40 else 64
        raise ValueError(
            f"{name} must be {length} lowercase hexadecimal characters"
        )
    return value


def _exact(mapping: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        raise ValueError(
            f"invalid {name} keys; missing={missing}, unknown={unknown}"
        )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _sequence(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a YAML sequence")
    return value


def _immutable_mapping(value: Mapping[str, Any]) -> tuple[tuple[str, object], ...]:
    result: list[tuple[str, object]] = []
    for key in sorted(value):
        item = value[key]
        if isinstance(item, Mapping):
            normalized: object = _immutable_mapping(_mapping(item, key))
        elif isinstance(item, list):
            normalized = tuple(item)
        else:
            normalized = item
        result.append((key, normalized))
    return tuple(result)


def _thaw(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _thaw(asdict(value))
    if isinstance(value, dict):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_thaw(item) for item in value]
    if isinstance(value, tuple):
        if value and all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            for item in value
        ):
            return {key: _thaw(item) for key, item in value}
        return [_thaw(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _canonical(value: object) -> str:
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ImplementationPolicy:
    require_clean_head: bool
    require_committed_inputs: bool
    require_external_output_root: bool
    deterministic_algorithms: bool
    intraop_threads: int
    interop_threads: int

    def __post_init__(self) -> None:
        for name in (
            "require_clean_head",
            "require_committed_inputs",
            "require_external_output_root",
            "deterministic_algorithms",
        ):
            _literal_true(getattr(self, name), name)
        _plain_int(self.intraop_threads, "intraop_threads", minimum=1)
        _plain_int(self.interop_threads, "interop_threads", minimum=1)


@dataclass(frozen=True)
class CampaignModelSpec:
    model_id: str
    family: str
    role: str
    routing_mode: str | None
    parent_model_id: str | None
    architecture: tuple[tuple[str, object], ...]
    export: tuple[tuple[str, object], ...] | None

    def __post_init__(self) -> None:
        _identifier(self.model_id, "model_id")
        if self.family not in _FAMILIES:
            raise ValueError(f"unsupported model family {self.family!r}")
        if self.role not in _ROLES:
            raise ValueError(f"unsupported model role {self.role!r}")
        if self.routing_mode is not None and self.routing_mode not in _ROUTING:
            raise ValueError("unsupported routing_mode")
        if self.parent_model_id is not None:
            _identifier(self.parent_model_id, "parent_model_id")

    @property
    def architecture_values(self) -> dict[str, object]:
        return {key: _thaw(value) for key, value in self.architecture}

    @property
    def export_values(self) -> dict[str, object] | None:
        return (
            None
            if self.export is None
            else {key: _thaw(value) for key, value in self.export}
        )


@dataclass(frozen=True)
class CampaignPairSpec:
    pair_id: str
    model_seed: int
    train_seed: int
    validation_seed: int
    statistics_seed: int
    test_seed: int | None = None

    def __post_init__(self) -> None:
        _identifier(self.pair_id, "pair_id")
        for name in (
            "model_seed",
            "train_seed",
            "validation_seed",
            "statistics_seed",
        ):
            _plain_int(getattr(self, name), name)
        if self.test_seed is not None:
            _plain_int(self.test_seed, "test_seed")


@dataclass(frozen=True)
class TrainDataSpec:
    batch_size: int
    length_schedule: tuple[int, ...]
    deterministic_step_stream: bool

    def __post_init__(self) -> None:
        _plain_int(self.batch_size, "batch_size", minimum=1)
        if not isinstance(self.length_schedule, tuple) or not self.length_schedule:
            raise ValueError("length_schedule must be a nonempty tuple")
        for length in self.length_schedule:
            _plain_int(length, "training length", minimum=1)
        _literal_true(self.deterministic_step_stream, "deterministic_step_stream")
        if len(self.length_schedule) != self.batch_size:
            raise ValueError("length_schedule must contain one length per batch row")
        if len(set(self.length_schedule)) < 2:
            raise ValueError("length_schedule must contain mixed real lengths")


@dataclass(frozen=True)
class EvaluationDataSpec:
    lengths: tuple[int, ...]
    episodes_per_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.lengths, tuple) or not self.lengths:
            raise ValueError("evaluation lengths must be a nonempty tuple")
        for length in self.lengths:
            _plain_int(length, "evaluation length", minimum=1)
        if tuple(sorted(set(self.lengths))) != self.lengths:
            raise ValueError("evaluation lengths must be strictly increasing")
        _plain_int(self.episodes_per_length, "episodes_per_length", minimum=1)


@dataclass(frozen=True)
class CampaignDataSpec:
    generator_version: str
    train: TrainDataSpec
    validation: EvaluationDataSpec
    test: EvaluationDataSpec | None = None
    scaling: EvaluationDataSpec | None = None

    def __post_init__(self) -> None:
        _identifier(self.generator_version, "generator_version")


@dataclass(frozen=True)
class CampaignTrainingSpec:
    optimizer: str
    learning_rate: float
    weight_decay: float
    max_gradient_norm: float
    optimizer_steps: int
    train_token_budget: int
    checkpoint_interval: int
    dtype: str
    device: str

    def __post_init__(self) -> None:
        if self.optimizer != "adamw":
            raise ValueError("optimizer must be adamw")
        object.__setattr__(
            self,
            "learning_rate",
            _finite_real(self.learning_rate, "learning_rate", positive=True),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _finite_real(self.weight_decay, "weight_decay"),
        )
        object.__setattr__(
            self,
            "max_gradient_norm",
            _finite_real(
                self.max_gradient_norm, "max_gradient_norm", positive=True
            ),
        )
        _plain_int(self.optimizer_steps, "optimizer_steps", minimum=1)
        _plain_int(self.train_token_budget, "train_token_budget", minimum=1)
        _plain_int(self.checkpoint_interval, "checkpoint_interval", minimum=1)
        if self.checkpoint_interval > self.optimizer_steps:
            raise ValueError("checkpoint_interval cannot exceed optimizer_steps")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.device != "cpu":
            raise ValueError("device must be cpu")


@dataclass(frozen=True)
class CampaignSelectionSpec:
    candidates_by_family: tuple[tuple[str, tuple[str, ...]], ...]
    primary_metric: str
    direction: str
    tie_break: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.primary_metric != "macro_length_query_accuracy":
            raise ValueError("selection primary_metric is unsupported")
        if self.direction != "maximize":
            raise ValueError("selection direction must be maximize")
        if self.tie_break != ("smaller_parameter_count", "lexical_candidate_id"):
            raise ValueError("selection tie_break does not match the locked order")


@dataclass(frozen=True)
class CampaignQualitySpec:
    primary_reference_model_id: str
    metric: str
    max_absolute_drop: float
    operational_rule: str
    claim_rule: str

    def __post_init__(self) -> None:
        _identifier(self.primary_reference_model_id, "primary_reference_model_id")
        if self.metric != "macro_length_query_accuracy":
            raise ValueError("quality metric is unsupported")
        value = _finite_real(self.max_absolute_drop, "max_absolute_drop")
        if value > 1.0:
            raise ValueError("max_absolute_drop must lie in [0,1]")
        object.__setattr__(self, "max_absolute_drop", value)
        if self.operational_rule != "mean_paired_delta_gte_negative_margin":
            raise ValueError("unsupported quality operational_rule")
        if self.claim_rule != "one_sided_95pct_lower_bound_gt_negative_margin":
            raise ValueError("unsupported quality claim_rule")


@dataclass(frozen=True)
class CampaignStatisticsSpec:
    paired_unit: str
    confidence_level: float
    method: str
    resamples: int

    def __post_init__(self) -> None:
        if self.paired_unit != "pair_id":
            raise ValueError("paired_unit must be pair_id")
        level = _finite_real(self.confidence_level, "confidence_level")
        if level != 0.95:
            raise ValueError("confidence_level must be exactly 0.95")
        object.__setattr__(self, "confidence_level", level)
        if self.method != "paired_percentile_bootstrap_v1":
            raise ValueError("unsupported statistics method")
        _plain_int(self.resamples, "resamples", minimum=1000)


@dataclass(frozen=True)
class CampaignRuntimeSpec:
    semantics: str
    warmups: int
    timed_iterations: int
    process_repetitions: int
    raw_samples_required: bool
    condition: str

    def __post_init__(self) -> None:
        if self.semantics != "batch1_full_document_streaming":
            raise ValueError("unsupported runtime semantics")
        _plain_int(self.warmups, "warmups")
        _plain_int(self.timed_iterations, "timed_iterations", minimum=1)
        _plain_int(self.process_repetitions, "process_repetitions", minimum=1)
        _literal_true(self.raw_samples_required, "raw_samples_required")
        if self.condition != "operational_quality_gate":
            raise ValueError("unsupported runtime condition")


@dataclass(frozen=True)
class CampaignPromotionSpec:
    screen_campaign_id: str
    record_path: str
    record_sha256: str
    screen_manifest_sha256: str
    executable_bundle_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.screen_campaign_id, "screen_campaign_id")
        if not isinstance(self.record_path, str) or not self.record_path:
            raise ValueError("record_path must be a relative POSIX path")
        path = PurePosixPath(self.record_path)
        if (
            path.is_absolute()
            or "\\" in self.record_path
            or ":" in self.record_path
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != self.record_path
        ):
            raise ValueError("record_path must be a normalized relative POSIX path")
        _sha(self.record_sha256, "record_sha256")
        _sha(self.screen_manifest_sha256, "screen_manifest_sha256")
        _sha(self.executable_bundle_sha256, "executable_bundle_sha256")


@dataclass(frozen=True)
class Milestone4CampaignConfig:
    schema_version: int
    campaign_id: str
    stage: CampaignStage
    description: str
    claim_eligible: bool
    implementation_policy: ImplementationPolicy
    task: BindingTaskConfig
    models: tuple[CampaignModelSpec, ...]
    pairs: tuple[CampaignPairSpec, ...]
    data: CampaignDataSpec
    training: CampaignTrainingSpec
    quality: CampaignQualitySpec
    statistics: CampaignStatisticsSpec
    runtime: CampaignRuntimeSpec
    selection: CampaignSelectionSpec | None = None
    promotion: CampaignPromotionSpec | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("schema_version must be integer 1")
        _identifier(self.campaign_id, "campaign_id")
        if not isinstance(self.stage, CampaignStage):
            raise TypeError("stage must be CampaignStage")
        _text(self.description, "description")
        if not isinstance(self.claim_eligible, bool):
            raise TypeError("claim_eligible must be boolean")
        if self.claim_eligible is not (self.stage is CampaignStage.CONFIRMATORY):
            raise ValueError("claim_eligible must be true exactly for confirmatory")
        if type(self.implementation_policy) is not ImplementationPolicy:
            raise TypeError("implementation_policy has the wrong type")
        if type(self.task) is not BindingTaskConfig:
            raise TypeError("task has the wrong type")
        if not isinstance(self.models, tuple) or any(
            type(model) is not CampaignModelSpec for model in self.models
        ):
            raise TypeError("models must be a tuple of CampaignModelSpec")
        if not isinstance(self.pairs, tuple) or any(
            type(pair) is not CampaignPairSpec for pair in self.pairs
        ):
            raise TypeError("pairs must be a tuple of CampaignPairSpec")
        if type(self.data) is not CampaignDataSpec:
            raise TypeError("data has the wrong type")
        if type(self.training) is not CampaignTrainingSpec:
            raise TypeError("training has the wrong type")
        if type(self.quality) is not CampaignQualitySpec:
            raise TypeError("quality has the wrong type")
        if type(self.statistics) is not CampaignStatisticsSpec:
            raise TypeError("statistics has the wrong type")
        if type(self.runtime) is not CampaignRuntimeSpec:
            raise TypeError("runtime has the wrong type")
        if self.selection is not None and type(self.selection) is not CampaignSelectionSpec:
            raise TypeError("selection has the wrong type")
        if self.promotion is not None and type(self.promotion) is not CampaignPromotionSpec:
            raise TypeError("promotion has the wrong type")
        if not self.models or not self.pairs:
            raise ValueError("campaign requires models and pairs")
        object.__setattr__(self, "models", tuple(sorted(self.models, key=lambda x: x.model_id)))
        object.__setattr__(self, "pairs", tuple(sorted(self.pairs, key=lambda x: x.pair_id)))
        _unique((model.model_id for model in self.models), "model_id")
        _unique((pair.pair_id for pair in self.pairs), "pair_id")
        self._validate_stage()
        self._validate_data()
        self._validate_models()
        self._validate_pair_seeds()
        self._validate_selection()

    def _validate_stage(self) -> None:
        if self.stage is CampaignStage.PILOT:
            if self.selection is not None or self.promotion is not None:
                raise ValueError("pilot forbids selection and promotion")
        elif self.stage is CampaignStage.SCREEN:
            if self.selection is None or self.promotion is not None:
                raise ValueError("screen requires selection and forbids promotion")
        else:
            if self.selection is not None or self.promotion is None:
                raise ValueError("confirmatory requires promotion and forbids selection")
            if len(self.pairs) < 3:
                raise ValueError("confirmatory requires at least three pairs")
        for pair in self.pairs:
            if (pair.test_seed is not None) is not (
                self.stage is CampaignStage.CONFIRMATORY
            ):
                raise ValueError("test_seed must be present exactly for confirmatory")
        confirmatory = self.stage is CampaignStage.CONFIRMATORY
        if (self.data.test is not None) is not confirmatory or (
            self.data.scaling is not None
        ) is not confirmatory:
            raise ValueError("test and scaling data must be present exactly for confirmatory")

    def _validate_data(self) -> None:
        for length in self.data.train.length_schedule:
            if not self.task.min_length <= length <= self.task.max_length:
                raise ValueError("training length is outside task bounds")
        for spec in (
            self.data.validation,
            self.data.test,
            self.data.scaling,
        ):
            if spec is not None and any(
                not self.task.min_length <= length <= self.task.max_length
                for length in spec.lengths
            ):
                raise ValueError("evaluation length is outside task bounds")
        expected_budget = self.training.optimizer_steps * sum(
            self.data.train.length_schedule
        )
        if self.training.train_token_budget != expected_budget:
            raise ValueError("train_token_budget does not match the exact schedule")

    def _validate_models(self) -> None:
        by_id = {model.model_id: model for model in self.models}
        for model in self.models:
            architecture = model.architecture_values
            if set(architecture) != _ARCHITECTURE_KEYS[model.family]:
                raise ValueError(f"architecture keys do not match {model.family}")
            _validate_architecture(model, self.training.optimizer_steps)
            if model.role == "trainable_source":
                if model.parent_model_id is not None or model.export is not None:
                    raise ValueError("source model forbids parent and export")
                if model.family == "routed":
                    if model.routing_mode not in _ROUTING:
                        raise ValueError("routed source requires routing_mode")
                elif model.routing_mode is not None:
                    raise ValueError("non-routed source requires routing_mode: null")
            else:
                if model.family != "routed" or model.routing_mode != "curriculum":
                    raise ValueError("derived compact must be routed curriculum")
                if model.parent_model_id not in by_id or model.export is None:
                    raise ValueError("derived compact requires an in-config parent and export")
                parent = by_id[model.parent_model_id]
                if (
                    parent.role != "trainable_source"
                    or parent.family != "routed"
                    or parent.routing_mode != "curriculum"
                ):
                    raise ValueError("compact parent must be routed curriculum source")
                export = model.export_values
                assert export is not None
                if set(export) != {"selection_method", "target_cp_rank"}:
                    raise ValueError("invalid compact export keys")
                if export["selection_method"] != "parameter_energy_v1":
                    raise ValueError("unsupported compact selection_method")
                target = _plain_int(export["target_cp_rank"], "target_cp_rank", minimum=1)
                parent_arch = parent.architecture_values
                if target >= _plain_int(parent_arch["cp_rank"], "parent cp_rank", minimum=1):
                    raise ValueError("target_cp_rank must be smaller than parent rank")
                expected = dict(parent_arch)
                expected["cp_rank"] = target
                if architecture != expected:
                    raise ValueError("compact architecture must equal parent except CP rank")
        routed_source_modes = {
            model.routing_mode
            for model in self.models
            if model.family == "routed" and model.role == "trainable_source"
        }
        if not _ROUTING <= routed_source_modes:
            raise ValueError(
                "campaign requires routed oracle, curriculum, and latent sources"
            )
        source_families = {
            model.family
            for model in self.models
            if model.role == "trainable_source"
        }
        required_controls = {"gru", "cached_transformer", "causal_ttn"}
        if not required_controls <= source_families:
            raise ValueError(
                "campaign requires GRU, cached Transformer, and causal TTN sources"
            )
        if not any(model.role == "derived_compact" for model in self.models):
            raise ValueError("campaign requires a routed curriculum compact child")

    def _validate_pair_seeds(self) -> None:
        seen: dict[int, str] = {}
        for pair in self.pairs:
            values = {
                "model_seed": pair.model_seed,
                "train_seed": pair.train_seed,
                "validation_seed": pair.validation_seed,
                "statistics_seed": pair.statistics_seed,
            }
            if pair.test_seed is not None:
                values["test_seed"] = pair.test_seed
            if len(set(values.values())) != len(values):
                raise ValueError("all seeds within each pair must be distinct")
            for category, seed in values.items():
                owner = f"{pair.pair_id}.{category}"
                if seed in seen:
                    raise ValueError(
                        f"seed {seed} is reused by {seen[seed]} and {owner}"
                    )
                seen[seed] = owner

    def _validate_selection(self) -> None:
        model_by_id = {model.model_id: model for model in self.models}
        if self.quality.primary_reference_model_id not in model_by_id:
            raise ValueError("primary reference model is not in the campaign")
        reference = model_by_id[self.quality.primary_reference_model_id]
        if (
            reference.role != "trainable_source"
            or reference.family != "routed"
            or reference.routing_mode != "curriculum"
        ):
            raise ValueError(
                "primary reference must be a routed curriculum trainable source"
            )
        if self.selection is None:
            return
        seen: set[str] = set()
        for family, candidates in self.selection.candidates_by_family:
            if family not in _FAMILIES or not candidates:
                raise ValueError("selection family/candidates are invalid")
            if len(set(candidates)) != len(candidates):
                raise ValueError("selection candidates must be unique")
            for model_id in candidates:
                if model_id in seen or model_id not in model_by_id:
                    raise ValueError("selection candidate is duplicate or unknown")
                if model_by_id[model_id].family != family:
                    raise ValueError("selection candidate family mismatch")
                seen.add(model_id)
        required_candidates = {
            model.model_id
            for model in self.models
            if model.role == "derived_compact"
            or not (model.family == "routed" and model.routing_mode == "oracle")
        }
        if seen != required_candidates:
            raise ValueError(
                "screen selection must contain exactly every non-oracle source "
                "and every compact candidate"
            )

    def canonical_json(self) -> str:
        return _canonical(self)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _unique(values: Any, name: str) -> None:
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{name} values must be unique")


def _validate_architecture(model: CampaignModelSpec, optimizer_steps: int) -> None:
    values = model.architecture_values
    integer_fields = _ARCHITECTURE_KEYS[model.family] - {
        "straight_through_route_surrogate",
        "schedule",
    }
    for name in integer_fields:
        _plain_int(values[name], f"{model.model_id}.{name}", minimum=1 if name != "curriculum_seed" else 0)
    if model.family == "cached_transformer" and int(values["d_model"]) % int(values["num_heads"]):
        raise ValueError("cached Transformer d_model must be divisible by num_heads")
    if model.family != "routed":
        return
    if not isinstance(values["straight_through_route_surrogate"], bool):
        raise TypeError("straight_through_route_surrogate must be boolean")
    schedule = values["schedule"]
    if model.routing_mode == "curriculum":
        mapping = _mapping(schedule, "curriculum schedule")
        _exact(
            mapping,
            {"start_step", "end_step", "start_probability", "end_probability"},
            "curriculum schedule",
        )
        start = _plain_int(mapping["start_step"], "start_step")
        end = _plain_int(mapping["end_step"], "end_step", minimum=1)
        start_probability = _finite_real(mapping["start_probability"], "start_probability")
        end_probability = _finite_real(mapping["end_probability"], "end_probability")
        if not start < end <= optimizer_steps:
            raise ValueError("curriculum schedule must finish within training")
        if not 0 <= end_probability <= start_probability <= 1:
            raise ValueError("curriculum probabilities must decrease inside [0,1]")
    elif schedule is not None:
        raise ValueError("only curriculum routing may define a schedule")


@dataclass(frozen=True)
class ResolvedCampaignRun:
    run_id: str
    campaign_id: str
    stage: CampaignStage
    model_id: str
    pair_id: str
    family: str
    role: str
    routing_mode: str | None
    parent_model_id: str | None
    parent_run_id: str | None
    architecture: tuple[tuple[str, object], ...]
    export: tuple[tuple[str, object], ...] | None
    task: BindingTaskConfig
    data: CampaignDataSpec
    model_seed: int
    train_seed: int
    validation_seed: int
    statistics_seed: int
    test_seed: int | None
    training_required: bool
    training: CampaignTrainingSpec | None
    code_commit: str
    code_tree: str
    raw_config_sha256: str
    semantic_config_sha256: str
    executable_bundle_sha256: str

    def __post_init__(self) -> None:
        self._validate()

    def _identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        del payload["run_id"]
        return payload

    def _validate(self) -> None:
        _sha(self.run_id, "run_id")
        _identifier(self.campaign_id, "campaign_id")
        if not isinstance(self.stage, CampaignStage):
            raise TypeError("stage must be CampaignStage")
        _identifier(self.model_id, "model_id")
        _identifier(self.pair_id, "pair_id")
        if type(self.task) is not BindingTaskConfig:
            raise TypeError("task has the wrong type")
        if type(self.data) is not CampaignDataSpec:
            raise TypeError("data has the wrong type")
        model = CampaignModelSpec(
            model_id=self.model_id,
            family=self.family,
            role=self.role,
            routing_mode=self.routing_mode,
            parent_model_id=self.parent_model_id,
            architecture=self.architecture,
            export=self.export,
        )
        if set(model.architecture_values) != _ARCHITECTURE_KEYS[model.family]:
            raise ValueError("run architecture does not match its model family")
        for name in (
            "model_seed",
            "train_seed",
            "validation_seed",
            "statistics_seed",
        ):
            _plain_int(getattr(self, name), name)
        if self.test_seed is not None:
            _plain_int(self.test_seed, "test_seed")
        if (self.test_seed is not None) is not (
            self.stage is CampaignStage.CONFIRMATORY
        ):
            raise ValueError("run test_seed must be present exactly for confirmatory")
        if (self.data.test is not None) is not (
            self.stage is CampaignStage.CONFIRMATORY
        ) or (self.data.scaling is not None) is not (
            self.stage is CampaignStage.CONFIRMATORY
        ):
            raise ValueError("run test and scaling data must match its stage")
        _sha(self.code_commit, "code_commit", _HEX40)
        _sha(self.code_tree, "code_tree", _HEX40)
        _sha(self.raw_config_sha256, "raw_config_sha256")
        _sha(self.semantic_config_sha256, "semantic_config_sha256")
        _sha(self.executable_bundle_sha256, "executable_bundle_sha256")
        if self.role == "trainable_source":
            if self.training_required is not True:
                raise ValueError("source run must require training")
            if type(self.training) is not CampaignTrainingSpec:
                raise TypeError("source run requires CampaignTrainingSpec")
            if (
                self.parent_model_id is not None
                or self.parent_run_id is not None
                or self.export is not None
            ):
                raise ValueError("source run forbids parent and export")
            if self.family == "routed":
                if self.routing_mode not in _ROUTING:
                    raise ValueError("routed source run requires routing_mode")
            elif self.routing_mode is not None:
                raise ValueError("non-routed source run requires null routing")
            _validate_architecture(model, self.training.optimizer_steps)
        else:
            if self.training_required is not False or self.training is not None:
                raise ValueError("derived compact run cannot contain training")
            if (
                self.family != "routed"
                or self.routing_mode != "curriculum"
                or self.parent_model_id is None
                or self.parent_run_id is None
                or self.export is None
            ):
                raise ValueError("derived compact run requires complete lineage")
            _sha(self.parent_run_id, "parent_run_id")
            export = model.export_values
            assert export is not None
            if set(export) != {"selection_method", "target_cp_rank"}:
                raise ValueError("derived compact run export is invalid")
            if export["selection_method"] != "parameter_energy_v1":
                raise ValueError("derived compact run selection method is invalid")
            _plain_int(export["target_cp_rank"], "target_cp_rank", minimum=1)
        expected = _run_id(self._identity_payload())
        if self.run_id != expected:
            raise ValueError("run_id does not match the complete run payload")

    def canonical_json(self) -> str:
        return _canonical(self)


def _run_id(payload: Mapping[str, object]) -> str:
    material = _RUN_DOMAIN + "\0" + _canonical(dict(payload))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def resolve_campaign_plan(
    config: Milestone4CampaignConfig,
    code_commit: str,
    code_tree: str,
    raw_config_sha256: str,
    executable_bundle_sha256: str,
) -> tuple[ResolvedCampaignRun, ...]:
    """Resolve one immutable paired run per model and pair."""

    if type(config) is not Milestone4CampaignConfig:
        raise TypeError("config must be Milestone4CampaignConfig")
    config.__post_init__()
    _sha(code_commit, "code_commit", _HEX40)
    _sha(code_tree, "code_tree", _HEX40)
    _sha(raw_config_sha256, "raw_config_sha256")
    _sha(executable_bundle_sha256, "executable_bundle_sha256")
    semantic_config_sha256 = config.fingerprint()
    _sha(semantic_config_sha256, "semantic_config_sha256")
    if (
        config.promotion is not None
        and config.promotion.executable_bundle_sha256
        != executable_bundle_sha256
    ):
        raise ValueError("promoted executable bundle does not match resolved bundle")

    runs: list[ResolvedCampaignRun] = []
    run_by_model_pair: dict[tuple[str, str], str] = {}
    ordered_models = sorted(
        config.models,
        key=lambda model: (model.role == "derived_compact", model.model_id),
    )
    for model in ordered_models:
        for pair in config.pairs:
            parent_run_id = (
                run_by_model_pair.get((model.parent_model_id, pair.pair_id))
                if model.parent_model_id is not None
                else None
            )
            if model.role == "derived_compact" and parent_run_id is None:
                raise RuntimeError("compact parent run was not resolved")
            values = {
                "campaign_id": config.campaign_id,
                "stage": config.stage,
                "model_id": model.model_id,
                "pair_id": pair.pair_id,
                "family": model.family,
                "role": model.role,
                "routing_mode": model.routing_mode,
                "parent_model_id": model.parent_model_id,
                "parent_run_id": parent_run_id,
                "architecture": model.architecture,
                "export": model.export,
                "task": config.task,
                "data": config.data,
                "model_seed": pair.model_seed,
                "train_seed": pair.train_seed,
                "validation_seed": pair.validation_seed,
                "statistics_seed": pair.statistics_seed,
                "test_seed": pair.test_seed,
                "training_required": model.role == "trainable_source",
                "training": (
                    config.training if model.role == "trainable_source" else None
                ),
                "code_commit": code_commit,
                "code_tree": code_tree,
                "raw_config_sha256": raw_config_sha256,
                "semantic_config_sha256": semantic_config_sha256,
                "executable_bundle_sha256": executable_bundle_sha256,
            }
            identity = _run_id(values)
            run = ResolvedCampaignRun(
                run_id=identity,
                **values,
            )
            runs.append(run)
            run_by_model_pair[(model.model_id, pair.pair_id)] = identity
    _unique((run.run_id for run in runs), "run_id")
    return tuple(runs)


def campaign_plan_sha256(
    config: Milestone4CampaignConfig,
    runs: tuple[ResolvedCampaignRun, ...],
) -> str:
    if type(config) is not Milestone4CampaignConfig:
        raise TypeError("config must be Milestone4CampaignConfig")
    config.__post_init__()
    if not isinstance(runs, tuple) or any(
        type(run) is not ResolvedCampaignRun for run in runs
    ):
        raise TypeError("runs must be a tuple of ResolvedCampaignRun")
    if not runs:
        raise ValueError("campaign plan cannot be empty")
    for run in runs:
        run._validate()
    _unique((run.run_id for run in runs), "run_id")
    _unique(((run.model_id, run.pair_id) for run in runs), "model/pair")

    first = runs[0]
    common = (
        first.campaign_id,
        first.stage,
        first.task,
        first.data,
        first.code_commit,
        first.code_tree,
        first.raw_config_sha256,
        first.semantic_config_sha256,
        first.executable_bundle_sha256,
    )
    model_specs: dict[str, tuple[object, ...]] = {}
    pair_specs: dict[str, tuple[object, ...]] = {}
    by_key = {(run.model_id, run.pair_id): run for run in runs}
    for run in runs:
        if (
            run.campaign_id,
            run.stage,
            run.task,
            run.data,
            run.code_commit,
            run.code_tree,
            run.raw_config_sha256,
            run.semantic_config_sha256,
            run.executable_bundle_sha256,
        ) != common:
            raise ValueError("all runs must share campaign and provenance fields")
        model_spec = (
            run.family,
            run.role,
            run.routing_mode,
            run.parent_model_id,
            run.architecture,
            run.export,
            run.training_required,
            run.training,
        )
        if run.model_id in model_specs and model_specs[run.model_id] != model_spec:
            raise ValueError("model definition changes across pairs")
        model_specs[run.model_id] = model_spec
        pair_spec = (
            run.model_seed,
            run.train_seed,
            run.validation_seed,
            run.statistics_seed,
            run.test_seed,
        )
        if run.pair_id in pair_specs and pair_specs[run.pair_id] != pair_spec:
            raise ValueError("pair seeds change across models")
        pair_specs[run.pair_id] = pair_spec

    expected_keys = {
        (model_id, pair_id)
        for model_id in model_specs
        for pair_id in pair_specs
    }
    if set(by_key) != expected_keys:
        raise ValueError("campaign plan must contain the complete model/pair product")
    representative_models = {
        model_id: next(run for run in runs if run.model_id == model_id)
        for model_id in model_specs
    }
    routed_modes = {
        run.routing_mode
        for run in representative_models.values()
        if run.family == "routed" and run.role == "trainable_source"
    }
    source_families = {
        run.family
        for run in representative_models.values()
        if run.role == "trainable_source"
    }
    if not _ROUTING <= routed_modes:
        raise ValueError("campaign plan is missing a required routed source")
    if not {"gru", "cached_transformer", "causal_ttn"} <= source_families:
        raise ValueError("campaign plan is missing a required control source")
    if not any(
        run.role == "derived_compact" for run in representative_models.values()
    ):
        raise ValueError("campaign plan is missing a compact candidate")
    seen_seeds: set[int] = set()
    for seeds in pair_specs.values():
        active_seeds = tuple(seed for seed in seeds if seed is not None)
        if len(set(active_seeds)) != len(active_seeds):
            raise ValueError("campaign plan reuses a seed within a pair")
        if seen_seeds.intersection(active_seeds):
            raise ValueError("campaign plan reuses a seed across pairs")
        seen_seeds.update(active_seeds)
    source_training_by_pair: dict[str, CampaignTrainingSpec] = {}
    for run in runs:
        if run.role != "trainable_source":
            continue
        assert run.training is not None
        if run.data != config.data:
            raise ValueError("trainable source run data does not match the config")
        prior = source_training_by_pair.setdefault(run.pair_id, run.training)
        if run.training != prior or run.training != config.training:
            raise ValueError(
                "trainable source runs must share the exact config training spec"
            )
    for run in runs:
        if run.role != "derived_compact":
            continue
        assert run.parent_model_id is not None
        parent = by_key.get((run.parent_model_id, run.pair_id))
        if (
            parent is None
            or parent.role != "trainable_source"
            or parent.family != "routed"
            or parent.routing_mode != "curriculum"
            or run.parent_run_id != parent.run_id
        ):
            raise ValueError("derived compact parent lineage is invalid")
        export = _thaw(run.export)
        if not isinstance(export, dict):
            raise ValueError("derived compact export is invalid")
        expected_architecture = dict(_thaw(parent.architecture))
        expected_architecture["cp_rank"] = export.get("target_cp_rank")
        if _thaw(run.architecture) != expected_architecture:
            raise ValueError("derived compact architecture does not match parent")
    expected = resolve_campaign_plan(
        config,
        first.code_commit,
        first.code_tree,
        first.raw_config_sha256,
        first.executable_bundle_sha256,
    )
    if {run.run_id for run in runs} != {run.run_id for run in expected}:
        raise ValueError("campaign plan does not match the exact resolved config runs")
    ordered = sorted((_thaw(run) for run in runs), key=lambda item: item["run_id"])
    material = _PLAN_DOMAIN + "\0" + _canonical(ordered)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class _StrictLoader(yaml.SafeLoader):
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._campaign_node_count = 0
        self._campaign_depth = 0

    def compose_node(self, parent, index):  # type: ignore[no-untyped-def]
        if self.check_event(yaml.AliasEvent):
            raise ValueError("YAML aliases are forbidden")
        self._campaign_node_count += 1
        if self._campaign_node_count > _MAX_YAML_NODES:
            raise ValueError(f"YAML must contain at most {_MAX_YAML_NODES} nodes")
        self._campaign_depth += 1
        try:
            if self._campaign_depth > _MAX_YAML_DEPTH:
                raise ValueError(f"YAML nesting must not exceed {_MAX_YAML_DEPTH}")
            return super().compose_node(parent, index)
        finally:
            self._campaign_depth -= 1


def _unique_mapping(loader, node, deep=False):  # type: ignore[no-untyped-def]
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ValueError("YAML keys must be hashable") from error
        if duplicate:
            raise ValueError(f"duplicate YAML key {key!r} is forbidden")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _plain_yaml(value: object) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("YAML floating-point values must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _plain_yaml(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("YAML mapping keys must be strings")
            if key == "<<":
                raise ValueError("YAML merge keys are forbidden")
            _plain_yaml(item)
        return
    raise ValueError("YAML contains a nonstandard scalar or collection")


def _load_document(path: str | Path) -> Mapping[str, Any]:
    raw = Path(path).read_bytes()
    if len(raw) > _MAX_CONFIG_BYTES:
        raise ValueError(f"configuration must be at most {_MAX_CONFIG_BYTES} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("configuration must be valid UTF-8") from error
    try:
        document = yaml.load(text, Loader=_StrictLoader)
        _plain_yaml(document)
    except yaml.YAMLError as error:
        raise ValueError("configuration must be strict safe YAML") from error
    except RecursionError as error:
        raise ValueError("configuration exceeds the YAML depth limit") from error
    return _mapping(document, "configuration")


def _make_model(value: object) -> CampaignModelSpec:
    mapping = _mapping(value, "model")
    _exact(mapping, _MODEL_KEYS, "model")
    architecture = _mapping(mapping["architecture"], "architecture")
    export_value = mapping["export"]
    export = (
        None
        if export_value is None
        else _immutable_mapping(_mapping(export_value, "export"))
    )
    return CampaignModelSpec(
        model_id=mapping["model_id"],
        family=mapping["family"],
        role=mapping["role"],
        routing_mode=mapping["routing_mode"],
        parent_model_id=mapping["parent_model_id"],
        architecture=_immutable_mapping(architecture),
        export=export,
    )


def _make_pair(value: object, stage: CampaignStage) -> CampaignPairSpec:
    mapping = _mapping(value, "pair")
    common = {
        "pair_id",
        "model_seed",
        "train_seed",
        "validation_seed",
        "statistics_seed",
    }
    expected = common | ({"test_seed"} if stage is CampaignStage.CONFIRMATORY else set())
    _exact(mapping, expected, "pair")
    return CampaignPairSpec(**dict(mapping))


def _make_train_data(value: object) -> TrainDataSpec:
    mapping = _mapping(value, "train data")
    _exact(mapping, {"batch_size", "length_schedule", "deterministic_step_stream"}, "train data")
    return TrainDataSpec(
        batch_size=mapping["batch_size"],
        length_schedule=tuple(_sequence(mapping["length_schedule"], "length_schedule")),
        deterministic_step_stream=mapping["deterministic_step_stream"],
    )


def _make_evaluation(value: object, name: str) -> EvaluationDataSpec:
    mapping = _mapping(value, name)
    _exact(mapping, {"lengths", "episodes_per_length"}, name)
    return EvaluationDataSpec(
        lengths=tuple(_sequence(mapping["lengths"], f"{name}.lengths")),
        episodes_per_length=mapping["episodes_per_length"],
    )


def _make_selection(value: object) -> CampaignSelectionSpec:
    mapping = _mapping(value, "selection")
    _exact(mapping, {"candidates_by_family", "primary_metric", "direction", "tie_break"}, "selection")
    candidates = _mapping(mapping["candidates_by_family"], "candidates_by_family")
    normalized: list[tuple[str, tuple[str, ...]]] = []
    for family in sorted(candidates):
        normalized.append(
            (
                family,
                tuple(_sequence(candidates[family], f"candidates.{family}")),
            )
        )
    return CampaignSelectionSpec(
        candidates_by_family=tuple(normalized),
        primary_metric=mapping["primary_metric"],
        direction=mapping["direction"],
        tie_break=tuple(_sequence(mapping["tie_break"], "tie_break")),
    )


def load_milestone4_campaign_config(path: str | Path) -> Milestone4CampaignConfig:
    """Load a bounded, exact-key, stage-dependent Milestone-4 YAML config."""

    document = _load_document(path)
    raw_stage = document.get("stage")
    try:
        stage = CampaignStage(raw_stage)
    except (TypeError, ValueError) as error:
        raise ValueError("stage must be pilot, screen, or confirmatory") from error
    expected = set(_COMMON_ROOT_KEYS)
    if stage is CampaignStage.SCREEN:
        expected.add("selection")
    elif stage is CampaignStage.CONFIRMATORY:
        expected.add("promotion")
    _exact(document, expected, "configuration")

    policy_values = _mapping(document["implementation_policy"], "implementation_policy")
    _exact(policy_values, set(ImplementationPolicy.__dataclass_fields__), "implementation_policy")

    task_values = dict(_mapping(document["task"], "task"))
    _exact(task_values, set(BindingTaskConfig.__dataclass_fields__), "task")
    task_values["heldout_key_value_pairs"] = tuple(
        tuple(_sequence(pair, "heldout pair"))
        for pair in _sequence(task_values["heldout_key_value_pairs"], "heldout pairs")
    )
    task = BindingTaskConfig(**task_values)

    models = tuple(_make_model(item) for item in _sequence(document["models"], "models"))
    pairs = tuple(_make_pair(item, stage) for item in _sequence(document["pairs"], "pairs"))

    data_values = _mapping(document["data"], "data")
    data_expected = {"generator_version", "train", "validation"}
    if stage is CampaignStage.CONFIRMATORY:
        data_expected |= {"test", "scaling"}
    _exact(data_values, data_expected, "data")
    data = CampaignDataSpec(
        generator_version=data_values["generator_version"],
        train=_make_train_data(data_values["train"]),
        validation=_make_evaluation(data_values["validation"], "validation"),
        test=(
            _make_evaluation(data_values["test"], "test")
            if stage is CampaignStage.CONFIRMATORY
            else None
        ),
        scaling=(
            _make_evaluation(data_values["scaling"], "scaling")
            if stage is CampaignStage.CONFIRMATORY
            else None
        ),
    )

    def exact_dataclass(name: str, cls: type, value: object):
        mapping = _mapping(value, name)
        _exact(mapping, set(cls.__dataclass_fields__), name)
        return cls(**dict(mapping))

    selection = (
        _make_selection(document["selection"])
        if stage is CampaignStage.SCREEN
        else None
    )
    promotion = (
        exact_dataclass("promotion", CampaignPromotionSpec, document["promotion"])
        if stage is CampaignStage.CONFIRMATORY
        else None
    )
    return Milestone4CampaignConfig(
        schema_version=document["schema_version"],
        campaign_id=document["campaign_id"],
        stage=stage,
        description=document["description"],
        claim_eligible=document["claim_eligible"],
        implementation_policy=ImplementationPolicy(**dict(policy_values)),
        task=task,
        models=models,
        pairs=pairs,
        data=data,
        training=exact_dataclass("training", CampaignTrainingSpec, document["training"]),
        quality=exact_dataclass("quality", CampaignQualitySpec, document["quality"]),
        statistics=exact_dataclass("statistics", CampaignStatisticsSpec, document["statistics"]),
        runtime=exact_dataclass("runtime", CampaignRuntimeSpec, document["runtime"]),
        selection=selection,
        promotion=promotion,
    )


__all__ = [
    "CampaignDataSpec",
    "CampaignModelSpec",
    "CampaignPairSpec",
    "CampaignPromotionSpec",
    "CampaignQualitySpec",
    "CampaignRuntimeSpec",
    "CampaignSelectionSpec",
    "CampaignStage",
    "CampaignStatisticsSpec",
    "CampaignTrainingSpec",
    "EvaluationDataSpec",
    "ImplementationPolicy",
    "Milestone4CampaignConfig",
    "ResolvedCampaignRun",
    "TrainDataSpec",
    "campaign_plan_sha256",
    "load_milestone4_campaign_config",
    "resolve_campaign_plan",
]
