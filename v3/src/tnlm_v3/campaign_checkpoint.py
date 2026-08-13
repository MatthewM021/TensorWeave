"""Bounded, canonical, non-executable Milestone-4 training checkpoints.

The format is deliberately not a PyTorch archive: a fixed prefix binds strict
canonical JSON to one contiguous little-endian tensor payload.  Loading never
uses pickle, ``torch.load``, or optimizer/model state-dict callbacks.
Checksums detect corruption but do not authenticate authorship; callers must
also verify the content digest recorded by the campaign manifest and pass the
expected run/stream provenance digests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
import random
import re
import struct
import sys
from typing import Any, Mapping, Sequence, TypeAlias

import torch
from torch import Tensor, nn
import torch.nn.modules.module as module_hooks
import torch.optim.optimizer as optimizer_hooks

from .baselines import (
    CachedCausalTransformerBindingBaseline,
    CachedTransformerBindingBaselineConfig,
    RecurrentBindingBaseline,
    RecurrentBindingBaselineConfig,
)
from .binding import BindingArchitectureConfig, BindingModelConfig, RoutedBindingModel
from .causal_ttn import CausalCompleteTreeBindingBaseline, CausalTreeBindingBaselineConfig
from .routing import CurriculumSchedule, RoutingMode


CampaignCheckpointModel: TypeAlias = (
    RoutedBindingModel
    | RecurrentBindingBaseline
    | CachedCausalTransformerBindingBaseline
    | CausalCompleteTreeBindingBaseline
)

_MAGIC = b"TNLM4CK\x00"
_VERSION = 1
_PREFIX = struct.Struct("<8sIQQ32s32s")
_KIND = "tnlm_v3.campaign_checkpoint"
_DOMAIN = b"tnlm-v3-campaign-model-fingerprint-v1\x00"
_SHA = re.compile(r"[0-9a-f]{64}")
_MAX_HEADER = 2 * 1024 * 1024
# Keep the complete input inside the same order of magnitude as the compact
# artifact worker ceiling.  Decoding uses owned copies, so a conservative
# on-wire cap also bounds transient memory.
_MAX_PAYLOAD = 62 * 1024 * 1024
_MAX_ARTIFACT = _PREFIX.size + _MAX_HEADER + _MAX_PAYLOAD
_MAX_TENSORS = 16_384
_MAX_JSON_NODES = 40_000
_MAX_JSON_DEPTH = 32
_MAX_NAME_BYTES = 1024
_MAX_RANK = 8
_MAX_DIMENSION = (1 << 31) - 1
_MAX_MODEL_ELEMENTS = 8_000_000
_MAX_ARCHITECTURE = 4096
_MAX_WIDTH = 65_536
_MAX_LAYERS = 1024
# Non-capturable AdamW stores its step counter as float32.  Values through this
# bound (and their next update) remain exactly representable.
_MAX_GLOBAL_STEP = (1 << 24) - 1

_DTYPE_TO_NAME = {
    torch.bool: "bool",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.float32: "float32",
    torch.float64: "float64",
}
_NAME_TO_DTYPE = {value: key for key, value in _DTYPE_TO_NAME.items()}
_ROOT_KEYS = frozenset(
    {
        "artifact_kind",
        "schema_version",
        "model_family",
        "model_config",
        "model_dtype",
        "model_training",
        "model_fingerprint",
        "optimizer",
        "resume_state",
        "python_rng",
        "tensors",
    }
)
_TENSOR_KEYS = frozenset({"name", "dtype", "shape", "offset", "nbytes", "sha256"})
_OPTIMIZER_KEYS = frozenset(
    {
        "type",
        "parameter_names",
        "initialized_parameters",
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
    }
)
_RESUME_KEYS = frozenset(
    {
        "global_step",
        "data_cursor",
        "run_spec_sha256",
        "stream_prefix_sha256",
        "best_metric",
        "best_step",
    }
)
_TASK_KEYS = frozenset({"num_surface_keys", "value_cardinality", "branches"})
_ROUTED_KEYS = frozenset(
    {
        "task",
        "d_model",
        "cp_rank",
        "router_hidden_dim",
        "routing_mode",
        "curriculum_schedule",
        "curriculum_seed",
        "scale_feature_dim",
        "straight_through_route_surrogate",
    }
)
_SCHEDULE_KEYS = frozenset(
    {"start_step", "end_step", "start_probability", "end_probability"}
)
_GRU_KEYS = frozenset({"task", "d_model", "hidden_dim", "num_layers"})
_TRANSFORMER_KEYS = frozenset(
    {"task", "d_model", "num_heads", "num_layers", "ff_dim"}
)
_TTN_KEYS = frozenset({"task", "d_model", "cp_rank", "scale_feature_dim"})
_MODULE_HOOK_FIELDS = (
    "_backward_hooks",
    "_backward_pre_hooks",
    "_forward_hooks",
    "_forward_hooks_always_called",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_load_state_dict_post_hooks",
    "_load_state_dict_pre_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
)
_GLOBAL_MODULE_HOOKS = (
    "_global_backward_hooks",
    "_global_backward_pre_hooks",
    "_global_buffer_registration_hooks",
    "_global_forward_hooks",
    "_global_forward_hooks_always_called",
    "_global_forward_hooks_with_kwargs",
    "_global_forward_pre_hooks",
    "_global_module_registration_hooks",
    "_global_parameter_registration_hooks",
)
_GLOBAL_OPTIMIZER_HOOKS = ("_global_optimizer_pre_hooks", "_global_optimizer_post_hooks")
_OPTIMIZER_HOOK_FIELDS = (
    "_optimizer_load_state_dict_post_hooks",
    "_optimizer_load_state_dict_pre_hooks",
    "_optimizer_state_dict_post_hooks",
    "_optimizer_state_dict_pre_hooks",
    "_optimizer_step_post_hooks",
    "_optimizer_step_pre_hooks",
)
_MODULE_INTERNAL = frozenset(
    {
        "training",
        "_parameters",
        "_buffers",
        "_non_persistent_buffers_set",
        "_modules",
        "_is_full_backward_hook",
        *_MODULE_HOOK_FIELDS,
    }
)


@dataclass(frozen=True)
class CampaignResumeState:
    """Position and provenance required to continue one deterministic run."""

    global_step: int
    data_cursor: int
    run_spec_sha256: str
    stream_prefix_sha256: str
    best_metric: float | None = None
    best_step: int | None = None

    def __post_init__(self) -> None:
        _integer(
            self.global_step,
            "global_step",
            minimum=0,
            maximum=_MAX_GLOBAL_STEP,
        )
        _integer(
            self.data_cursor,
            "data_cursor",
            minimum=0,
            maximum=_MAX_GLOBAL_STEP,
        )
        if self.global_step != self.data_cursor:
            raise ValueError("global_step and data_cursor must match")
        _digest(self.run_spec_sha256, "run_spec_sha256")
        _digest(self.stream_prefix_sha256, "stream_prefix_sha256")
        if (self.best_metric is None) is not (self.best_step is None):
            raise ValueError("best_metric and best_step must be present together")
        if self.best_metric is not None:
            metric = _json_float(self.best_metric, "best_metric")
            if not 0.0 <= metric <= 1.0:
                raise ValueError("best_metric must lie in [0,1]")
            _integer(self.best_step, "best_step", minimum=0, maximum=self.global_step)


@dataclass(frozen=True)
class CampaignCheckpointContract:
    """Trusted executable identity supplied independently of checkpoint bytes."""

    model_family: str
    model_config: object
    model_dtype: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    foreach: bool | None
    fused: bool | None

    def __post_init__(self) -> None:
        expected = {
            "routed": BindingModelConfig,
            "gru": RecurrentBindingBaselineConfig,
            "cached_transformer": CachedTransformerBindingBaselineConfig,
            "causal_ttn": CausalTreeBindingBaselineConfig,
        }
        if self.model_family not in expected:
            raise ValueError("unsupported checkpoint contract model_family")
        if type(self.model_config) is not expected[self.model_family]:
            raise TypeError("checkpoint contract model_config has the wrong exact type")
        parsed = _config_from_dict(
            self.model_family, _config_to_dict(self.model_config)
        )
        if parsed != self.model_config:
            raise ValueError("checkpoint contract config is not canonical")
        if self.model_dtype not in {"float32", "float64"}:
            raise ValueError("checkpoint contract dtype must be float32 or float64")
        if self.optimizer != "adamw":
            raise ValueError("checkpoint contract optimizer must be adamw")
        for name, positive in (
            ("learning_rate", True),
            ("weight_decay", False),
            ("eps", True),
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"checkpoint contract {name} must be a finite float")
            if value < 0 or (positive and value == 0):
                raise ValueError(f"checkpoint contract {name} is outside its valid range")
        if type(self.betas) is not tuple or len(self.betas) != 2 or any(
            type(value) is not float
            or not math.isfinite(value)
            or not 0.0 <= value < 1.0
            for value in self.betas
        ):
            raise ValueError("checkpoint contract betas must be a finite float pair")
        for name in ("foreach", "fused"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise TypeError(f"checkpoint contract {name} must be boolean or null")
            if value is True:
                raise ValueError(f"checkpoint contract {name} must not be true")

    def canonical_json(self) -> str:
        return _canonical(
            {
                "model_family": self.model_family,
                "model_config": _config_to_dict(self.model_config),
                "model_dtype": self.model_dtype,
                "optimizer": self.optimizer,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "betas": list(self.betas),
                "eps": self.eps,
                "foreach": self.foreach,
                "fused": self.fused,
            }
        ).decode("utf-8")

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _json_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite JSON float")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    return value


def _keys(value: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"invalid {name} fields; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _reject_constant(value: str) -> object:
    raise ValueError(f"nonstandard JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _validate_json_tree(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ValueError("checkpoint JSON has too many nodes")
        if depth > _MAX_JSON_DEPTH:
            raise ValueError("checkpoint JSON is too deeply nested")
        if item is None or type(item) in (str, bool, int):
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("checkpoint JSON numbers must be finite")
            continue
        if type(item) is list:
            stack.extend((child, depth + 1) for child in item)
            continue
        if type(item) is dict and all(type(key) is str for key in item):
            stack.extend((child, depth + 1) for child in item.values())
            continue
        raise ValueError("checkpoint JSON contains a non-plain value")


def _parse_json(raw: bytes) -> Mapping[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("checkpoint header must be strict UTF-8 JSON") from error
    _validate_json_tree(value)
    root = _mapping(value, "checkpoint header")
    try:
        encoded = _canonical(root)
    except (UnicodeEncodeError, ValueError, TypeError) as error:
        raise ValueError("checkpoint header is not canonical JSON") from error
    if encoded != raw:
        raise ValueError("checkpoint header is not canonical JSON")
    return root


def _check_global_hooks() -> None:
    for name in _GLOBAL_MODULE_HOOKS:
        if bool(getattr(module_hooks, name, None)):
            raise ValueError(f"global PyTorch module hooks are unsupported ({name})")
    for name in _GLOBAL_OPTIMIZER_HOOKS:
        if bool(getattr(optimizer_hooks, name, None)):
            raise ValueError(f"global PyTorch optimizer hooks are unsupported ({name})")


def _plain_metadata(value: object) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("module metadata must be finite")
        return value
    if isinstance(value, Enum):
        return {"enum": f"{type(value).__module__}.{type(value).__qualname__}", "value": value.value}
    if hasattr(value, "__dataclass_fields__"):
        return _plain_metadata(asdict(value))
    if isinstance(value, (tuple, list, torch.Size)):
        return [_plain_metadata(item) for item in value]
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: _plain_metadata(item) for key, item in value.items()}
    raise ValueError(f"unsupported live module metadata type {type(value).__name__}")


def _module_metadata(module: nn.Module, name: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in module.__dict__.items():
        if key in _MODULE_INTERNAL:
            continue
        if callable(value):
            raise ValueError(f"instance-level executable callable at module {name!r}")
        result[key] = _plain_metadata(value)
    return result


def _construct_preserving_rng(
    config: object, *, device: str | torch.device
) -> CampaignCheckpointModel:
    with torch.random.fork_rng(devices=[], enabled=True):
        with torch.device(device):
            if type(config) is BindingModelConfig:
                return RoutedBindingModel(config)
            if type(config) is RecurrentBindingBaselineConfig:
                return RecurrentBindingBaseline(config)
            if type(config) is CachedTransformerBindingBaselineConfig:
                return CachedCausalTransformerBindingBaseline(config)
            if type(config) is CausalTreeBindingBaselineConfig:
                return CausalCompleteTreeBindingBaseline(config)
    raise TypeError("unsupported campaign model config")


def _family_and_config(model: CampaignCheckpointModel) -> tuple[str, object]:
    exact = {
        RoutedBindingModel: ("routed", BindingModelConfig),
        RecurrentBindingBaseline: ("gru", RecurrentBindingBaselineConfig),
        CachedCausalTransformerBindingBaseline: (
            "cached_transformer",
            CachedTransformerBindingBaselineConfig,
        ),
        CausalCompleteTreeBindingBaseline: ("causal_ttn", CausalTreeBindingBaselineConfig),
    }
    item = exact.get(type(model))
    if item is None:
        raise TypeError("model must have one exact supported campaign model type")
    family, config_type = item
    config = getattr(model, "config", None)
    if type(config) is not config_type:
        raise TypeError("model config has the wrong exact type")
    return family, config


def _validate_live_model(model: CampaignCheckpointModel) -> tuple[str, object, dict[str, Tensor]]:
    _check_global_hooks()
    family, config = _family_and_config(model)
    try:
        parsed_config = _config_from_dict(family, _config_to_dict(config))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("live model config is not canonically reconstructible") from error
    if parsed_config != config:
        raise ValueError("live model config is not canonically reconstructible")
    config = parsed_config
    modules = list(model.named_modules(remove_duplicate=False))
    parameters = list(model.named_parameters(remove_duplicate=False))
    buffers = list(model.named_buffers(remove_duplicate=False))
    for name, module in modules:
        if any(bool(getattr(module, field, None)) for field in _MODULE_HOOK_FIELDS):
            raise ValueError(f"runtime hooks are unsupported at module {name!r}")
    for name, parameter in parameters:
        if bool(getattr(parameter, "_backward_hooks", None)) or bool(
            getattr(parameter, "_post_accumulate_grad_hooks", None)
        ):
            raise ValueError(f"parameter hooks are unsupported at {name!r}")
        if parameter.grad is not None:
            raise ValueError(f"pending gradients are unsupported at {name!r}")
        if not parameter.requires_grad:
            raise ValueError("campaign checkpoint parameters must all require gradients")
    for name, buffer in buffers:
        if buffer.requires_grad or bool(getattr(buffer, "_backward_hooks", None)) or bool(
            getattr(buffer, "_post_accumulate_grad_hooks", None)
        ):
            raise ValueError(f"buffer gradient state is unsupported at {name!r}")
    try:
        schema = _construct_preserving_rng(config, device="meta")
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("model config cannot reconstruct a bounded architecture") from error
    expected_modules = list(schema.named_modules(remove_duplicate=False))
    expected_parameters = list(schema.named_parameters(remove_duplicate=False))
    expected_buffers = list(schema.named_buffers(remove_duplicate=False))
    for kind, actual, expected in (
        ("module", modules, expected_modules),
        ("parameter", parameters, expected_parameters),
        ("buffer", buffers, expected_buffers),
    ):
        if [name for name, _ in actual] != [name for name, _ in expected]:
            raise ValueError(f"model {kind} topology does not match its config")
        if len({id(item) for _, item in actual}) != len(actual):
            raise ValueError(f"aliased model {kind}s are unsupported")
        if any(
            type(actual_item) is not type(expected_item)
            for (_, actual_item), (_, expected_item) in zip(
                actual, expected, strict=True
            )
        ):
            raise ValueError(f"model {kind} tensor/module types do not match config")
    for (name, actual), (_, expected) in zip(modules, expected_modules, strict=True):
        if type(actual) is not type(expected):
            raise ValueError(f"model module type changed at {name!r}")
        if actual._non_persistent_buffers_set != expected._non_persistent_buffers_set:
            raise ValueError(f"buffer persistence metadata changed at {name!r}")
        if actual._is_full_backward_hook != expected._is_full_backward_hook:
            raise ValueError(f"backward-hook metadata changed at {name!r}")
        if _module_metadata(actual, name) != _module_metadata(expected, name):
            raise ValueError(f"live module metadata changed at {name!r}")
    state: dict[str, Tensor] = {}
    for name, tensor in (*parameters, *buffers):
        if tensor.layout is not torch.strided:
            raise ValueError("model state must contain only strided tensors")
        state[name] = tensor
    expected_state = {
        name: tensor
        for name, tensor in (*expected_parameters, *expected_buffers)
    }
    if set(state) != set(expected_state) or any(
        tuple(state[name].shape) != tuple(expected_state[name].shape) for name in state
    ):
        raise ValueError("model state shapes do not match its config")
    if sum(tensor.numel() for tensor in state.values()) > _MAX_MODEL_ELEMENTS:
        raise ValueError("model exceeds the checkpoint allocation limit")
    devices = {tensor.device for tensor in state.values()}
    if devices != {torch.device("cpu")}:
        raise ValueError("campaign checkpoints currently require CPU model state")
    dtypes = {parameter.dtype for _, parameter in parameters}
    if len(dtypes) != 1 or next(iter(dtypes)) not in (torch.float32, torch.float64):
        raise ValueError("model parameters must share float32 or float64 dtype")
    model_dtype = next(iter(dtypes))
    for name, tensor in state.items():
        if tensor.is_floating_point() and tensor.dtype != model_dtype:
            raise ValueError(f"floating model state dtype mismatch at {name!r}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"model state must be finite at {name!r}")
    return family, config, state


def _storage_token(tensor: Tensor) -> int:
    if tensor.numel() == 0:
        return id(tensor)
    storage = tensor.untyped_storage()
    if not tensor.is_contiguous() or tensor.storage_offset() != 0:
        raise ValueError("tensor views and noncontiguous tensors are unsupported")
    if storage.nbytes() != tensor.numel() * tensor.element_size():
        raise ValueError("tensor storage must be exact-sized and non-overlapping")
    return int(storage._cdata)


def _validate_storage_isolation(named: Sequence[tuple[str, Tensor]]) -> None:
    owners: dict[int, str] = {}
    for name, tensor in named:
        token = _storage_token(tensor)
        if tensor.numel() and token in owners:
            raise ValueError(f"shared tensor storage is unsupported: {owners[token]!r}, {name!r}")
        owners[token] = name


def _optimizer_document(
    optimizer: torch.optim.Optimizer,
    model: CampaignCheckpointModel,
    global_step: int,
) -> tuple[dict[str, object], list[tuple[str, Tensor]]]:
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("optimizer must be exactly torch.optim.AdamW")
    if any(callable(value) for value in optimizer.__dict__.values()):
        raise ValueError("instance-level optimizer callables are unsupported")
    if any(bool(getattr(optimizer, field, None)) for field in _OPTIMIZER_HOOK_FIELDS):
        raise ValueError("optimizer hooks are unsupported")
    parameters = list(model.named_parameters())
    names = [name for name, _ in parameters]
    if len(optimizer.param_groups) != 1:
        raise ValueError("campaign AdamW must have exactly one parameter group")
    group = optimizer.param_groups[0]
    group_parameters = list(group["params"])
    expected_parameters = [parameter for _, parameter in parameters]
    if len(group_parameters) != len(expected_parameters) or any(
        actual is not expected
        for actual, expected in zip(group_parameters, expected_parameters, strict=True)
    ):
        raise ValueError("optimizer parameters must exactly match model parameter order")
    required_group = {
        "params", "lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
        "foreach", "capturable", "differentiable", "fused",
    }
    allowed_group = required_group | {"decoupled_weight_decay"}
    if not required_group <= set(group) or not set(group) <= allowed_group:
        raise ValueError("optimizer parameter group has unsupported fields")
    if set(optimizer.defaults) != set(group) - {"params"}:
        raise ValueError("optimizer defaults do not match its sole parameter group")
    if group.get("amsgrad") is not False or group.get("maximize") is not False:
        raise ValueError("campaign AdamW requires amsgrad=false and maximize=false")
    if group.get("capturable") is not False or group.get("differentiable") is not False:
        raise ValueError("campaign AdamW requires capturable=false and differentiable=false")
    if (
        group.get("foreach") is not None
        and group.get("foreach") is not False
    ) or (
        group.get("fused") is not None and group.get("fused") is not False
    ):
        raise ValueError("campaign AdamW forbids foreach and fused execution")
    if "decoupled_weight_decay" in group and group["decoupled_weight_decay"] is not True:
        raise ValueError("AdamW must use decoupled weight decay")
    for name in ("lr", "eps", "weight_decay"):
        raw = group[name]
        if isinstance(raw, Tensor) or isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"optimizer {name} must be a scalar real")
        if not math.isfinite(float(raw)) or float(raw) < 0:
            raise ValueError(f"optimizer {name} must be finite and nonnegative")
    if float(group["lr"]) <= 0 or float(group["eps"]) <= 0:
        raise ValueError("optimizer lr and eps must be positive")
    betas = group["betas"]
    if not isinstance(betas, tuple) or len(betas) != 2:
        raise TypeError("optimizer betas must be a pair")
    if any(
        isinstance(value, (bool, Tensor)) or not isinstance(value, (int, float))
        for value in betas
    ):
        raise TypeError("optimizer betas must be scalar reals")
    beta_values = tuple(float(value) for value in betas)
    if any(not math.isfinite(value) or not 0.0 <= value < 1.0 for value in beta_values):
        raise ValueError("optimizer betas must lie in [0,1)")
    for name, expected in group.items():
        if name == "params":
            continue
        actual = optimizer.defaults[name]
        if isinstance(actual, Tensor) or callable(actual):
            raise ValueError("optimizer defaults must contain plain values")
        if name == "betas":
            if not isinstance(actual, tuple) or len(actual) != 2 or any(
                isinstance(item, (bool, Tensor))
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in actual
            ):
                raise ValueError("optimizer default betas are invalid")
            matches = tuple(float(item) for item in actual) == beta_values
        else:
            matches = type(actual) is type(expected) and actual == expected
        if not matches:
            raise ValueError("optimizer defaults do not match its sole parameter group")
    by_parameter = {parameter: name for name, parameter in parameters}
    if set(optimizer.state) - set(by_parameter):
        raise ValueError("optimizer state contains an unknown parameter")
    initialized: list[str] = []
    tensors: list[tuple[str, Tensor]] = []
    for parameter, name in ((parameter, name) for name, parameter in parameters):
        state = optimizer.state.get(parameter)
        if state is None or not state:
            continue
        expected = {"step", "exp_avg", "exp_avg_sq"}
        if set(state) != expected:
            raise ValueError(f"optimizer state fields are invalid for {name!r}")
        step, avg, square = state["step"], state["exp_avg"], state["exp_avg_sq"]
        if not all(type(item) is Tensor for item in (step, avg, square)):
            raise TypeError("optimizer state values must be exact tensors")
        for item in (step, avg, square):
            if item.requires_grad or item.grad is not None or bool(
                getattr(item, "_backward_hooks", None)
            ) or bool(getattr(item, "_post_accumulate_grad_hooks", None)):
                raise ValueError("optimizer tensors must not carry autograd state or hooks")
        if step.shape != () or step.dtype is not torch.float32:
            raise ValueError("optimizer step must be a float32 scalar tensor")
        if step.device.type != "cpu" or not bool(torch.isfinite(step)):
            raise ValueError("optimizer step must be a finite CPU scalar")
        step_value = float(step.item())
        if step_value != global_step:
            raise ValueError("optimizer step must equal checkpoint global_step")
        for item in (avg, square):
            if item.shape != parameter.shape or item.dtype != parameter.dtype or item.device != parameter.device:
                raise ValueError("optimizer moments must match their parameter")
            if not bool(torch.isfinite(item).all()):
                raise ValueError("optimizer moments must be finite")
        if bool((square < 0).any()):
            raise ValueError("optimizer squared moments must be nonnegative")
        initialized.append(name)
        tensors.extend(
            (
                (f"optimizer.{name}.step", step),
                (f"optimizer.{name}.exp_avg", avg),
                (f"optimizer.{name}.exp_avg_sq", square),
            )
        )
    document: dict[str, object] = {
        "type": "torch.optim.AdamW",
        "parameter_names": names,
        "initialized_parameters": initialized,
        "lr": float(group["lr"]),
        "betas": [float(beta_values[0]), float(beta_values[1])],
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "amsgrad": False,
        "maximize": False,
        "foreach": group.get("foreach"),
        "capturable": False,
        "differentiable": False,
        "fused": group.get("fused"),
    }
    if (global_step == 0 and initialized) or (
        global_step > 0 and initialized != names
    ):
        raise ValueError("optimizer initialization does not match global_step")
    return document, tensors


def _config_to_dict(config: object) -> dict[str, object]:
    value = asdict(config)
    if type(config) is BindingModelConfig:
        value["routing_mode"] = config.routing_mode.value
    return value


def _task_from_dict(value: object) -> BindingArchitectureConfig:
    document = _mapping(value, "task")
    _keys(document, _TASK_KEYS, "task")
    return BindingArchitectureConfig(
        _integer(document["num_surface_keys"], "num_surface_keys", minimum=2, maximum=_MAX_ARCHITECTURE),
        _integer(document["value_cardinality"], "value_cardinality", minimum=2, maximum=_MAX_ARCHITECTURE),
        _integer(document["branches"], "branches", minimum=2, maximum=_MAX_ARCHITECTURE),
    )


def _width(value: object, name: str) -> int:
    return _integer(value, name, minimum=1, maximum=_MAX_WIDTH)


def _config_from_dict(family: object, value: object) -> object:
    if not isinstance(family, str):
        raise TypeError("model_family must be a string")
    document = _mapping(value, "model_config")
    if family == "routed":
        _keys(document, _ROUTED_KEYS, "model_config")
        schedule_value = document["curriculum_schedule"]
        schedule = None
        if schedule_value is not None:
            encoded = _mapping(schedule_value, "curriculum_schedule")
            _keys(encoded, _SCHEDULE_KEYS, "curriculum_schedule")
            schedule = CurriculumSchedule(
                start_step=_integer(encoded["start_step"], "start_step", minimum=0),
                end_step=_integer(encoded["end_step"], "end_step", minimum=1),
                start_probability=_json_float(encoded["start_probability"], "start_probability"),
                end_probability=_json_float(encoded["end_probability"], "end_probability"),
            )
        mode_value = document["routing_mode"]
        if not isinstance(mode_value, str):
            raise TypeError("routing_mode must be a string")
        surrogate = document["straight_through_route_surrogate"]
        if type(surrogate) is not bool:
            raise TypeError("straight_through_route_surrogate must be boolean")
        return BindingModelConfig(
            task=_task_from_dict(document["task"]),
            d_model=_width(document["d_model"], "d_model"),
            cp_rank=_width(document["cp_rank"], "cp_rank"),
            router_hidden_dim=_width(document["router_hidden_dim"], "router_hidden_dim"),
            routing_mode=RoutingMode(mode_value),
            curriculum_schedule=schedule,
            curriculum_seed=_integer(document["curriculum_seed"], "curriculum_seed"),
            scale_feature_dim=_width(document["scale_feature_dim"], "scale_feature_dim"),
            straight_through_route_surrogate=surrogate,
        )
    if family == "gru":
        _keys(document, _GRU_KEYS, "model_config")
        return RecurrentBindingBaselineConfig(
            task=_task_from_dict(document["task"]),
            d_model=_width(document["d_model"], "d_model"),
            hidden_dim=_width(document["hidden_dim"], "hidden_dim"),
            num_layers=_integer(document["num_layers"], "num_layers", minimum=1, maximum=_MAX_LAYERS),
        )
    if family == "cached_transformer":
        _keys(document, _TRANSFORMER_KEYS, "model_config")
        return CachedTransformerBindingBaselineConfig(
            task=_task_from_dict(document["task"]),
            d_model=_width(document["d_model"], "d_model"),
            num_heads=_width(document["num_heads"], "num_heads"),
            num_layers=_integer(document["num_layers"], "num_layers", minimum=1, maximum=_MAX_LAYERS),
            ff_dim=_width(document["ff_dim"], "ff_dim"),
        )
    if family == "causal_ttn":
        _keys(document, _TTN_KEYS, "model_config")
        return CausalTreeBindingBaselineConfig(
            task=_task_from_dict(document["task"]),
            d_model=_width(document["d_model"], "d_model"),
            cp_rank=_width(document["cp_rank"], "cp_rank"),
            scale_feature_dim=_width(document["scale_feature_dim"], "scale_feature_dim"),
        )
    raise ValueError("unsupported model_family")


def _portable_bytes(tensor: Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    if sys.byteorder == "little" or value.element_size() == 1:
        return raw
    width = value.element_size()
    return b"".join(raw[index : index + width][::-1] for index in range(0, len(raw), width))


def _native_bytes(raw: bytes, width: int) -> bytes:
    if sys.byteorder == "little" or width == 1:
        return raw
    return b"".join(raw[index : index + width][::-1] for index in range(0, len(raw), width))


def _tensor_records(named: Sequence[tuple[str, Tensor]]) -> tuple[list[dict[str, object]], bytes]:
    if len(named) > _MAX_TENSORS:
        raise ValueError("checkpoint contains too many tensors")
    records: list[dict[str, object]] = []
    parts: list[bytes] = []
    offset = 0
    prior: str | None = None
    for name, tensor in sorted(named, key=lambda item: item[0]):
        if prior is not None and name <= prior:
            raise ValueError("checkpoint tensor names must be unique")
        prior = name
        if not name or len(name.encode("utf-8")) > _MAX_NAME_BYTES:
            raise ValueError("checkpoint tensor name is invalid")
        dtype = _DTYPE_TO_NAME.get(tensor.dtype)
        if dtype is None or tensor.layout is not torch.strided:
            raise TypeError("unsupported checkpoint tensor dtype or layout")
        if tensor.ndim > _MAX_RANK or any(int(size) > _MAX_DIMENSION for size in tensor.shape):
            raise ValueError("checkpoint tensor shape is unsupported")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("checkpoint tensor must be finite")
        raw = _portable_bytes(tensor)
        if len(raw) > _MAX_PAYLOAD - offset:
            raise ValueError("checkpoint payload exceeds the safety limit")
        records.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": [int(size) for size in tensor.shape],
                "offset": offset,
                "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        parts.append(raw)
        offset += len(raw)
    return records, b"".join(parts)


def _training_flags(model: CampaignCheckpointModel) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for name, module in model.named_modules():
        if type(module.training) is not bool:
            raise TypeError("module training flags must be boolean")
        result[name] = module.training
    return result


def _python_rng_to_json(state: tuple[Any, ...]) -> dict[str, object]:
    version, values, gaussian = state
    if type(version) is not int or not isinstance(values, tuple) or len(values) != 625:
        raise ValueError("unsupported Python RNG state")
    encoded_gaussian = None if gaussian is None else float(gaussian)
    if encoded_gaussian is not None and not math.isfinite(encoded_gaussian):
        raise ValueError("Python RNG Gaussian cache must be finite")
    return {"version": version, "state": list(values), "gaussian": encoded_gaussian}


def _python_rng_from_json(value: object) -> tuple[Any, ...]:
    document = _mapping(value, "python_rng")
    _keys(document, frozenset({"version", "state", "gaussian"}), "python_rng")
    version = _integer(document["version"], "python_rng.version", minimum=1, maximum=3)
    values = document["state"]
    if type(values) is not list or len(values) != 625:
        raise ValueError("python_rng.state must contain exactly 625 integers")
    normalized = tuple(_integer(item, "python_rng state", minimum=0, maximum=0xFFFFFFFF) for item in values)
    gaussian = document["gaussian"]
    if gaussian is not None:
        gaussian = _json_float(gaussian, "python_rng.gaussian")
    state = (version, normalized, gaussian)
    probe = random.Random()
    try:
        probe.setstate(state)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid Python RNG state") from error
    return state


def _resume_to_dict(state: CampaignResumeState) -> dict[str, object]:
    return asdict(state)


def _resume_from_dict(value: object) -> CampaignResumeState:
    document = _mapping(value, "resume_state")
    _keys(document, _RESUME_KEYS, "resume_state")
    return CampaignResumeState(**dict(document))


def campaign_checkpoint_contract(
    model: CampaignCheckpointModel,
    optimizer: torch.optim.Optimizer,
) -> CampaignCheckpointContract:
    """Derive the trusted executable contract from a fresh planned run.

    The optimizer must be uninitialized and the model must have no pending
    gradients.  Runners should derive this object from their independently
    resolved run specification, never from the checkpoint being inspected.
    """

    family, config, _ = _validate_live_model(model)
    group, _ = _optimizer_document(optimizer, model, global_step=0)
    dtype = _DTYPE_TO_NAME[next(iter(model.parameters())).dtype]
    return CampaignCheckpointContract(
        model_family=family,
        model_config=config,
        model_dtype=dtype,
        optimizer="adamw",
        learning_rate=group["lr"],
        weight_decay=group["weight_decay"],
        betas=tuple(group["betas"]),
        eps=group["eps"],
        foreach=group["foreach"],
        fused=group["fused"],
    )


def _validate_contract_header(
    contract: CampaignCheckpointContract,
    root: Mapping[str, object],
) -> None:
    if type(contract) is not CampaignCheckpointContract:
        raise TypeError("expected_contract must be exactly CampaignCheckpointContract")
    contract.__post_init__()
    optimizer = _mapping(root["optimizer"], "optimizer")
    actual = {
        "model_family": root["model_family"],
        "model_config": root["model_config"],
        "model_dtype": root["model_dtype"],
        "optimizer": "adamw" if optimizer.get("type") == "torch.optim.AdamW" else optimizer.get("type"),
        "learning_rate": optimizer.get("lr"),
        "weight_decay": optimizer.get("weight_decay"),
        "betas": optimizer.get("betas"),
        "eps": optimizer.get("eps"),
        "foreach": optimizer.get("foreach"),
        "fused": optimizer.get("fused"),
    }
    expected = json.loads(contract.canonical_json())
    if actual != expected:
        raise ValueError("checkpoint executable contract does not match expectation")


def _model_fingerprint_from_parts(family: str, config: object, state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(_DOMAIN)
    digest.update(_canonical({"family": family, "config": _config_to_dict(config)}))
    for name in sorted(state):
        tensor = state[name]
        raw = _portable_bytes(tensor)
        descriptor = _canonical(
            {"name": name, "dtype": _DTYPE_TO_NAME[tensor.dtype], "shape": list(tensor.shape)}
        )
        digest.update(struct.pack("<Q", len(descriptor)))
        digest.update(descriptor)
        digest.update(struct.pack("<Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def campaign_model_fingerprint(model: CampaignCheckpointModel) -> str:
    """Hash exact executable config and every parameter/buffer bit."""

    family, config, state = _validate_live_model(model)
    _validate_storage_isolation(list(state.items()))
    return _model_fingerprint_from_parts(family, config, state)


def serialize_campaign_checkpoint(
    model: CampaignCheckpointModel,
    optimizer: torch.optim.Optimizer,
    resume_state: CampaignResumeState,
) -> bytes:
    """Serialize a validated training boundary without executable payloads.

    A boundary has no pending gradients.  Callers should use
    ``optimizer.zero_grad(set_to_none=True)`` after the completed step; this
    function rejects gradients instead of silently discarding resume state.
    """

    if type(resume_state) is not CampaignResumeState:
        raise TypeError("resume_state must be exactly CampaignResumeState")
    resume_state.__post_init__()
    family, config, state = _validate_live_model(model)
    optimizer_document, optimizer_tensors = _optimizer_document(
        optimizer, model, resume_state.global_step
    )
    named = [(f"model.{name}", tensor) for name, tensor in state.items()]
    named.extend(optimizer_tensors)
    named.append(("rng.torch_cpu", torch.random.get_rng_state()))
    _validate_storage_isolation(named)
    records, payload = _tensor_records(named)
    model_dtype = _DTYPE_TO_NAME[next(iter(model.parameters())).dtype]
    root = {
        "artifact_kind": _KIND,
        "schema_version": _VERSION,
        "model_family": family,
        "model_config": _config_to_dict(config),
        "model_dtype": model_dtype,
        "model_training": _training_flags(model),
        "model_fingerprint": _model_fingerprint_from_parts(family, config, state),
        "optimizer": optimizer_document,
        "resume_state": _resume_to_dict(resume_state),
        "python_rng": _python_rng_to_json(random.getstate()),
        "tensors": records,
    }
    header = _canonical(root)
    if len(header) > _MAX_HEADER:
        raise ValueError("checkpoint header exceeds the safety limit")
    prefix = _PREFIX.pack(
        _MAGIC,
        _VERSION,
        len(header),
        len(payload),
        hashlib.sha256(header).digest(),
        hashlib.sha256(payload).digest(),
    )
    return prefix + header + payload


def _shape(value: object, name: str) -> tuple[int, ...]:
    if type(value) is not list or len(value) > _MAX_RANK:
        raise ValueError(f"{name} must be a bounded shape array")
    shape = tuple(_integer(item, name, minimum=0, maximum=_MAX_DIMENSION) for item in value)
    product = 1
    for size in shape:
        product *= size
        if product > _MAX_PAYLOAD:
            raise ValueError("declared checkpoint tensor is too large")
    return shape


def _decode_records(value: object, payload: bytes) -> dict[str, Tensor]:
    if type(value) is not list or len(value) > _MAX_TENSORS:
        raise ValueError("tensors must be a bounded array")
    decoded: dict[str, Tensor] = {}
    prior: str | None = None
    expected_offset = 0
    for index, item in enumerate(value):
        record = _mapping(item, f"tensor record {index}")
        _keys(record, _TENSOR_KEYS, f"tensor record {index}")
        name = record["name"]
        if not isinstance(name, str) or not name or len(name.encode("utf-8")) > _MAX_NAME_BYTES:
            raise ValueError("tensor name is invalid")
        if prior is not None and name <= prior:
            raise ValueError("tensor names must be unique and sorted")
        prior = name
        dtype_name = record["dtype"]
        if not isinstance(dtype_name, str) or dtype_name not in _NAME_TO_DTYPE:
            raise ValueError("tensor dtype is unsupported")
        dtype = _NAME_TO_DTYPE[dtype_name]
        shape = _shape(record["shape"], f"tensor {name} shape")
        offset = _integer(record["offset"], "tensor offset", minimum=0)
        nbytes = _integer(record["nbytes"], "tensor nbytes", minimum=0)
        digest = _digest(record["sha256"], "tensor sha256")
        width = torch.empty((), device="cpu", dtype=dtype).element_size()
        numel = math.prod(shape)
        if nbytes != numel * width:
            raise ValueError("tensor byte count does not match shape and dtype")
        if offset != expected_offset or nbytes > len(payload) - offset:
            raise ValueError("tensor payload ranges must be contiguous and bounded")
        raw = payload[offset : offset + nbytes]
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"tensor checksum mismatch for {name!r}")
        try:
            tensor = torch.frombuffer(bytearray(_native_bytes(raw, width)), dtype=dtype).clone().reshape(shape)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"tensor {name!r} cannot be decoded") from error
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"tensor {name!r} must be finite")
        decoded[name] = tensor
        expected_offset += nbytes
    if expected_offset != len(payload):
        raise ValueError("tensor records do not consume the complete payload")
    return decoded


def _parse_training_flags(value: object, expected: set[str]) -> dict[str, bool]:
    document = _mapping(value, "model_training")
    if set(document) != expected:
        raise ValueError("model_training keys do not match architecture")
    result: dict[str, bool] = {}
    for name, item in document.items():
        if type(item) is not bool:
            raise TypeError("model_training values must be boolean")
        result[name] = item
    return result


def _parse_optimizer(
    value: object,
    model: CampaignCheckpointModel,
    tensors: Mapping[str, Tensor],
    global_step: int,
) -> tuple[torch.optim.AdamW, set[str]]:
    document = _mapping(value, "optimizer")
    _keys(document, _OPTIMIZER_KEYS, "optimizer")
    if document["type"] != "torch.optim.AdamW":
        raise ValueError("optimizer type is unsupported")
    parameters = list(model.named_parameters())
    names = [name for name, _ in parameters]
    if document["parameter_names"] != names:
        raise ValueError("optimizer parameter order does not match model")
    initialized = document["initialized_parameters"]
    if type(initialized) is not list or any(not isinstance(name, str) for name in initialized):
        raise TypeError("initialized_parameters must be a string array")
    if any(name not in names for name in initialized) or len(initialized) != len(set(initialized)):
        raise ValueError("initialized_parameters are invalid")
    if initialized != [name for name in names if name in set(initialized)]:
        raise ValueError("initialized_parameters must follow model parameter order")
    if (global_step == 0 and initialized) or (global_step > 0 and initialized != names):
        raise ValueError("optimizer initialization does not match global_step")
    for field in ("amsgrad", "maximize", "capturable", "differentiable"):
        if document[field] is not False:
            raise ValueError(f"optimizer {field} must be false")
    if (
        document["foreach"] is not None and document["foreach"] is not False
    ) or (
        document["fused"] is not None and document["fused"] is not False
    ):
        raise ValueError("optimizer foreach and fused must be null or false")
    lr = _json_float(document["lr"], "optimizer.lr")
    eps = _json_float(document["eps"], "optimizer.eps")
    weight_decay = _json_float(document["weight_decay"], "optimizer.weight_decay")
    betas = document["betas"]
    if type(betas) is not list or len(betas) != 2:
        raise TypeError("optimizer.betas must be a two-element array")
    beta_pair = (_json_float(betas[0], "beta1"), _json_float(betas[1], "beta2"))
    if lr <= 0 or eps <= 0 or weight_decay < 0 or any(not 0 <= beta < 1 for beta in beta_pair):
        raise ValueError("optimizer hyperparameters are outside their valid range")
    optimizer = torch.optim.AdamW(
        (parameter for _, parameter in parameters),
        lr=lr,
        betas=beta_pair,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=False,
        maximize=False,
        foreach=document["foreach"],
        capturable=False,
        differentiable=False,
        fused=document["fused"],
    )
    used: set[str] = set()
    by_name = dict(parameters)
    for name in initialized:
        parameter = by_name[name]
        prefix = f"optimizer.{name}."
        keys = {suffix: tensors[prefix + suffix] for suffix in ("step", "exp_avg", "exp_avg_sq") if prefix + suffix in tensors}
        if set(keys) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"optimizer tensors are incomplete for {name!r}")
        step, avg, square = keys["step"], keys["exp_avg"], keys["exp_avg_sq"]
        if step.shape != () or step.dtype is not torch.float32:
            raise ValueError("optimizer step tensor is invalid")
        step_value = float(step.item())
        if step_value != global_step:
            raise ValueError("optimizer step tensor does not match global_step")
        for item in (avg, square):
            if item.shape != parameter.shape or item.dtype != parameter.dtype:
                raise ValueError("optimizer moment tensor does not match its parameter")
        if bool((square < 0).any()):
            raise ValueError("optimizer squared moments must be nonnegative")
        optimizer.state[parameter] = {"step": step, "exp_avg": avg, "exp_avg_sq": square}
        used.update(prefix + suffix for suffix in keys)
    return optimizer, used


def deserialize_campaign_checkpoint(
    blob: bytes | bytearray | memoryview,
    *,
    expected_run_spec_sha256: str,
    expected_stream_prefix_sha256: str,
    expected_contract: CampaignCheckpointContract,
    device: str | torch.device | None = None,
) -> tuple[CampaignCheckpointModel, torch.optim.AdamW, CampaignResumeState]:
    """Load a canonical checkpoint and transactionally install its RNG state."""

    _check_global_hooks()
    if device is not None:
        try:
            target = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("invalid checkpoint target device") from error
        if target != torch.device("cpu"):
            raise ValueError("campaign checkpoints currently support only CPU restore")
    if type(blob) not in (bytes, bytearray, memoryview):
        raise TypeError("blob must be bytes-like")
    try:
        view = memoryview(blob)
        if not view.contiguous:
            raise TypeError("blob must be contiguous")
        size = view.nbytes
    except (TypeError, ValueError) as error:
        raise TypeError("blob must expose one contiguous byte buffer") from error
    if size > _MAX_ARTIFACT:
        raise ValueError("checkpoint exceeds the total safety limit")
    artifact = view.tobytes()
    if len(artifact) > _MAX_ARTIFACT:
        raise ValueError("checkpoint exceeds the total safety limit")
    if len(artifact) < _PREFIX.size:
        raise ValueError("checkpoint is truncated")
    magic, version, header_size, payload_size, header_hash, payload_hash = _PREFIX.unpack_from(artifact)
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("unsupported checkpoint magic or version")
    if header_size > _MAX_HEADER or payload_size > _MAX_PAYLOAD:
        raise ValueError("checkpoint section exceeds its safety limit")
    if _PREFIX.size + header_size + payload_size != len(artifact):
        raise ValueError("checkpoint is truncated or has trailing bytes")
    header = artifact[_PREFIX.size : _PREFIX.size + header_size]
    payload = artifact[_PREFIX.size + header_size :]
    if hashlib.sha256(header).digest() != header_hash or hashlib.sha256(payload).digest() != payload_hash:
        raise ValueError("checkpoint checksum mismatch")
    root = _parse_json(header)
    _keys(root, _ROOT_KEYS, "checkpoint header")
    if root["artifact_kind"] != _KIND or _integer(root["schema_version"], "schema_version") != _VERSION:
        raise ValueError("unsupported checkpoint schema")
    _validate_contract_header(expected_contract, root)
    resume = _resume_from_dict(root["resume_state"])
    if resume.run_spec_sha256 != _digest(expected_run_spec_sha256, "expected_run_spec_sha256"):
        raise ValueError("checkpoint run specification does not match expectation")
    if resume.stream_prefix_sha256 != _digest(expected_stream_prefix_sha256, "expected_stream_prefix_sha256"):
        raise ValueError("checkpoint stream prefix does not match expectation")
    config = _config_from_dict(root["model_family"], root["model_config"])
    dtype_name = root["model_dtype"]
    if dtype_name not in ("float32", "float64"):
        raise ValueError("model_dtype must be float32 or float64")
    try:
        schema = _construct_preserving_rng(config, device="meta")
        schema = schema.to(dtype=_NAME_TO_DTYPE[dtype_name])
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("checkpoint model config exceeds allocation limits") from error
    expected_state = {name: tensor for name, tensor in (*schema.named_parameters(), *schema.named_buffers())}
    if sum(tensor.numel() for tensor in expected_state.values()) > _MAX_MODEL_ELEMENTS:
        raise ValueError("checkpoint model exceeds allocation limit")
    training = _parse_training_flags(root["model_training"], {name for name, _ in schema.named_modules()})
    tensors = _decode_records(root["tensors"], payload)
    expected_model_names = {f"model.{name}" for name in expected_state}
    actual_model_names = {name for name in tensors if name.startswith("model.")}
    if actual_model_names != expected_model_names:
        raise ValueError("checkpoint model tensor keys do not match architecture")
    for name, expected in expected_state.items():
        tensor = tensors[f"model.{name}"]
        if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
            raise ValueError(f"checkpoint model tensor does not match {name!r}")
    torch_rng = tensors.get("rng.torch_cpu")
    if torch_rng is None or torch_rng.dtype is not torch.uint8 or torch_rng.ndim != 1:
        raise ValueError("checkpoint CPU RNG tensor is missing or invalid")
    try:
        generator = torch.Generator(device="cpu")
        generator.set_state(torch_rng)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("checkpoint CPU RNG state is invalid") from error
    python_rng = _python_rng_from_json(root["python_rng"])
    caller_torch_rng = torch.random.get_rng_state()
    caller_python_rng = random.getstate()
    try:
        model = _construct_preserving_rng(config, device="cpu").to(
            device="cpu", dtype=_NAME_TO_DTYPE[dtype_name]
        )
        with torch.no_grad():
            destinations = {name: tensor for name, tensor in (*model.named_parameters(), *model.named_buffers())}
            for name, destination in destinations.items():
                destination.copy_(tensors[f"model.{name}"])
        for name, module in model.named_modules():
            module.training = training[name]
        optimizer, used_optimizer = _parse_optimizer(
            root["optimizer"], model, tensors, resume.global_step
        )
        allowed = expected_model_names | used_optimizer | {"rng.torch_cpu"}
        if set(tensors) != allowed:
            raise ValueError("checkpoint contains unexpected tensors")
        family, live_config, live_state = _validate_live_model(model)
        _validate_storage_isolation(
            [(f"model.{name}", tensor) for name, tensor in live_state.items()]
            + [(name, tensor) for name, tensor in tensors.items() if name.startswith("optimizer.")]
        )
        fingerprint = _digest(root["model_fingerprint"], "model_fingerprint")
        if _model_fingerprint_from_parts(family, live_config, live_state) != fingerprint:
            raise ValueError("checkpoint model fingerprint mismatch")
        _optimizer_document(optimizer, model, resume.global_step)
        torch.random.set_rng_state(torch_rng)
        random.setstate(python_rng)
    except BaseException:
        torch.random.set_rng_state(caller_torch_rng)
        random.setstate(caller_python_rng)
        raise
    return model, optimizer, resume


__all__ = [
    "CampaignCheckpointContract",
    "CampaignCheckpointModel",
    "CampaignResumeState",
    "campaign_checkpoint_contract",
    "campaign_model_fingerprint",
    "deserialize_campaign_checkpoint",
    "serialize_campaign_checkpoint",
]
