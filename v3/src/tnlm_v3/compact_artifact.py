"""Deterministic, non-executable compact binding-model artifacts.

The format deliberately does not use ``pickle`` or ``torch.save``. A fixed
binary prefix checksum-binds a canonical JSON header and a contiguous
little-endian tensor payload. These unkeyed digests detect corruption; they do
not authenticate authorship. The loader validates the complete schema and all
declared sizes before constructing a model.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import re
import struct
import sys
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .binding import (
    BindingArchitectureConfig,
    BindingModelConfig,
    RoutedBindingModel,
)
from .model_export import CompactExportManifest
from .routing import CurriculumSchedule, RoutingMode
from .truncation import CPRankSelection, model_state_fingerprint


_MAGIC = b"TNLM3CM\x00"
_FORMAT_VERSION = 1
_PREFIX = struct.Struct("<8sIQQ32s32s")
_ARTIFACT_KIND = "tnlm_v3.compact_binding_model"
_HEADER_KEYS = frozenset(
    {
        "artifact_kind",
        "schema_version",
        "model_config",
        "manifest",
        "selection",
        "exported_model_fingerprint",
        "module_training",
        "parameter_requires_grad",
        "tensors",
    }
)
_MODEL_CONFIG_KEYS = frozenset(
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
_TASK_KEYS = frozenset(
    {"num_surface_keys", "value_cardinality", "branches"}
)
_SCHEDULE_KEYS = frozenset(
    {"start_step", "end_step", "start_probability", "end_probability"}
)
_SELECTION_KEYS = frozenset(
    {
        "schema_version",
        "source_model_fingerprint",
        "method",
        "nominal_rank",
        "exported_rank",
        "retained_indices",
        "channel_scores",
        "calibration_fingerprint",
    }
)
_TENSOR_KEYS = frozenset(
    {"name", "dtype", "shape", "offset", "nbytes", "sha256"}
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
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
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}
_MAX_HEADER_BYTES = 1 << 20
# Keep a complete artifact (payload + the 1 MiB header ceiling + prefix) below
# the independent replay worker's 64 MiB input ceiling.
_MAX_PAYLOAD_BYTES = 62 * 1024 * 1024
_MAX_TENSORS = 4096
_MAX_TENSOR_RANK = 8
_MAX_DIMENSION = (1 << 31) - 1
_MAX_NAME_BYTES = 1024
_MAX_ARCHITECTURE_CARDINALITY = 4096
_MAX_MODEL_DIMENSION = 65536
_MAX_MODEL_ELEMENTS = 16_000_000
_MAX_ARTIFACT_BYTES = _PREFIX.size + _MAX_HEADER_BYTES + _MAX_PAYLOAD_BYTES


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_int(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _strict_real(value: object, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be a JSON float")
    result = value
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError(f"{name} must be an object with string keys")
    return value


def _keys(
    value: Mapping[str, object], expected: frozenset[str], name: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"invalid {name} fields; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant {value!r} is forbidden")


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _parse_canonical_json(value: bytes) -> Mapping[str, object]:
    try:
        text = value.decode("utf-8")
        parsed = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("artifact header must be valid strict UTF-8 JSON") from error
    root = _mapping(parsed, "artifact header")
    try:
        canonical = _canonical_json(root)
    except UnicodeEncodeError as error:
        raise ValueError("artifact header contains invalid Unicode") from error
    if canonical != value:
        raise ValueError("artifact header is not canonical JSON")
    return root


def _portable_tensor_bytes(tensor: Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    if sys.byteorder == "little" or value.element_size() == 1:
        return raw
    width = value.element_size()
    return b"".join(
        raw[index : index + width][::-1]
        for index in range(0, len(raw), width)
    )


def _native_tensor_bytes(raw: bytes, element_size: int) -> bytes:
    if sys.byteorder == "little" or element_size == 1:
        return raw
    return b"".join(
        raw[index : index + element_size][::-1]
        for index in range(0, len(raw), element_size)
    )


def _model_config_to_dict(config: BindingModelConfig) -> dict[str, object]:
    schedule = config.curriculum_schedule
    return {
        "task": asdict(config.task),
        "d_model": config.d_model,
        "cp_rank": config.cp_rank,
        "router_hidden_dim": config.router_hidden_dim,
        "routing_mode": config.routing_mode.value,
        "curriculum_schedule": None if schedule is None else schedule.as_dict(),
        "curriculum_seed": config.curriculum_seed,
        "scale_feature_dim": config.scale_feature_dim,
        "straight_through_route_surrogate": (
            config.straight_through_route_surrogate
        ),
    }


def _model_config_from_dict(value: object) -> BindingModelConfig:
    document = _mapping(value, "model_config")
    _keys(document, _MODEL_CONFIG_KEYS, "model_config")
    task_value = _mapping(document["task"], "model_config.task")
    _keys(task_value, _TASK_KEYS, "model_config.task")
    task = BindingArchitectureConfig(
        num_surface_keys=_strict_int(
            task_value["num_surface_keys"],
            "num_surface_keys",
            minimum=2,
            maximum=_MAX_ARCHITECTURE_CARDINALITY,
        ),
        value_cardinality=_strict_int(
            task_value["value_cardinality"],
            "value_cardinality",
            minimum=2,
            maximum=_MAX_ARCHITECTURE_CARDINALITY,
        ),
        branches=_strict_int(
            task_value["branches"],
            "branches",
            minimum=2,
            maximum=_MAX_ARCHITECTURE_CARDINALITY,
        ),
    )

    schedule_value = document["curriculum_schedule"]
    schedule: CurriculumSchedule | None
    if schedule_value is None:
        schedule = None
    else:
        encoded = _mapping(schedule_value, "curriculum_schedule")
        _keys(encoded, _SCHEDULE_KEYS, "curriculum_schedule")
        schedule = CurriculumSchedule(
            start_step=_strict_int(
                encoded["start_step"], "schedule.start_step", minimum=0
            ),
            end_step=_strict_int(
                encoded["end_step"], "schedule.end_step", minimum=0
            ),
            start_probability=_strict_real(
                encoded["start_probability"], "schedule.start_probability"
            ),
            end_probability=_strict_real(
                encoded["end_probability"], "schedule.end_probability"
            ),
        )
    mode_value = document["routing_mode"]
    if not isinstance(mode_value, str):
        raise TypeError("routing_mode must be a string")
    try:
        mode = RoutingMode(mode_value)
    except ValueError as error:
        raise ValueError("routing_mode is unsupported") from error
    surrogate = document["straight_through_route_surrogate"]
    if type(surrogate) is not bool:
        raise TypeError("straight_through_route_surrogate must be boolean")
    return BindingModelConfig(
        task=task,
        d_model=_strict_int(
            document["d_model"],
            "d_model",
            minimum=1,
            maximum=_MAX_MODEL_DIMENSION,
        ),
        cp_rank=_strict_int(
            document["cp_rank"],
            "cp_rank",
            minimum=1,
            maximum=_MAX_MODEL_DIMENSION,
        ),
        router_hidden_dim=_strict_int(
            document["router_hidden_dim"],
            "router_hidden_dim",
            minimum=1,
            maximum=_MAX_MODEL_DIMENSION,
        ),
        routing_mode=mode,
        curriculum_schedule=schedule,
        curriculum_seed=_strict_int(document["curriculum_seed"], "curriculum_seed"),
        scale_feature_dim=_strict_int(
            document["scale_feature_dim"],
            "scale_feature_dim",
            minimum=1,
            maximum=_MAX_MODEL_DIMENSION,
        ),
        straight_through_route_surrogate=surrogate,
    )


def _selection_to_dict(selection: CPRankSelection) -> dict[str, object]:
    return {
        "schema_version": selection.schema_version,
        "source_model_fingerprint": selection.source_model_fingerprint,
        "method": selection.method,
        "nominal_rank": selection.nominal_rank,
        "exported_rank": selection.exported_rank,
        "retained_indices": list(selection.retained_indices),
        "channel_scores": list(selection.channel_scores),
        "calibration_fingerprint": selection.calibration_fingerprint,
    }


def _selection_from_dict(value: object) -> CPRankSelection:
    document = _mapping(value, "selection")
    _keys(document, _SELECTION_KEYS, "selection")
    indices = document["retained_indices"]
    scores = document["channel_scores"]
    if type(indices) is not list:
        raise TypeError("selection.retained_indices must be an array")
    if type(scores) is not list:
        raise TypeError("selection.channel_scores must be an array")
    return CPRankSelection(
        schema_version=_strict_int(
            document["schema_version"], "selection.schema_version"
        ),
        source_model_fingerprint=_strict_sha256(
            document["source_model_fingerprint"],
            "selection.source_model_fingerprint",
        ),
        method=(
            document["method"]
            if isinstance(document["method"], str)
            else _raise_type("selection.method must be a string")
        ),
        nominal_rank=_strict_int(
            document["nominal_rank"], "selection.nominal_rank"
        ),
        exported_rank=_strict_int(
            document["exported_rank"], "selection.exported_rank"
        ),
        retained_indices=tuple(
            _strict_int(item, "selection retained index") for item in indices
        ),
        channel_scores=tuple(
            _strict_real(item, "selection channel score") for item in scores
        ),
        calibration_fingerprint=(
            None
            if document["calibration_fingerprint"] is None
            else _strict_sha256(
                document["calibration_fingerprint"],
                "selection.calibration_fingerprint",
            )
        ),
    )


def _raise_type(message: str):
    raise TypeError(message)


def _parameter_count(model: RoutedBindingModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _raw_tensor_bytes(model: RoutedBindingModel) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
    )


def _validate_relationships(
    model: RoutedBindingModel,
    manifest: CompactExportManifest,
    selection: CPRankSelection,
) -> str:
    if not isinstance(model, RoutedBindingModel):
        raise TypeError("model must be a RoutedBindingModel")
    if not isinstance(manifest, CompactExportManifest):
        raise TypeError("manifest must be a CompactExportManifest")
    if not isinstance(selection, CPRankSelection):
        raise TypeError("selection must be a CPRankSelection")
    fingerprint = model_state_fingerprint(model)
    checks = {
        "manifest exported model fingerprint": (
            manifest.exported_model_fingerprint == fingerprint
        ),
        "manifest exported config fingerprint": (
            manifest.exported_config_fingerprint == model.config.fingerprint()
        ),
        "manifest source model fingerprint": (
            manifest.source_model_fingerprint
            == selection.source_model_fingerprint
        ),
        "manifest selection fingerprint": (
            manifest.selection_fingerprint == selection.fingerprint()
        ),
        "manifest nominal rank": manifest.nominal_cp_rank == selection.nominal_rank,
        "manifest exported rank": (
            manifest.exported_cp_rank
            == selection.exported_rank
            == model.config.cp_rank
        ),
        "manifest retained indices": (
            manifest.retained_indices == selection.retained_indices
        ),
        "manifest exported parameters": (
            manifest.exported_parameter_count == _parameter_count(model)
        ),
        "manifest exported raw bytes": (
            manifest.exported_raw_tensor_bytes == _raw_tensor_bytes(model)
        ),
        "manifest exported state d_model": (
            manifest.exported_state_d_model == model.config.d_model
        ),
        "manifest exported state paths": (
            manifest.exported_state_paths == model.forest.paths
        ),
        "manifest exported state scalars": (
            manifest.exported_state_scalars_per_slot == model.config.d_model
        ),
        "manifest exported operation proxy": (
            manifest.exported_operation_count_proxy_per_merge
            == model.forest.merge.structural_metrics(merge_count=1)[
                "operation_count_proxy_per_merge"
            ]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("compact artifact relationship mismatch: " + ", ".join(failed))
    return fingerprint


def _training_flags(model: RoutedBindingModel) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for name, module in model.named_modules():
        if type(module.training) is not bool:
            raise TypeError(f"module {name!r} training flag must be boolean")
        result[name] = module.training
    return result


def _requires_grad_flags(model: RoutedBindingModel) -> dict[str, bool]:
    return {name: parameter.requires_grad for name, parameter in model.named_parameters()}


def _validate_serializable_topology(model: RoutedBindingModel) -> None:
    """Reject aliases or extra structure that config-only loading cannot rebuild."""

    try:
        with torch.device("meta"):
            schema = RoutedBindingModel(model.config)
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("model configuration cannot reconstruct its architecture") from error

    actual_modules = list(model.named_modules(remove_duplicate=False))
    actual_parameters = list(model.named_parameters(remove_duplicate=False))
    actual_buffers = list(model.named_buffers(remove_duplicate=False))
    collections = (
        ("module", actual_modules, list(schema.named_modules(remove_duplicate=False))),
        (
            "parameter",
            actual_parameters,
            list(schema.named_parameters(remove_duplicate=False)),
        ),
        ("buffer", actual_buffers, list(schema.named_buffers(remove_duplicate=False))),
    )
    for name, actual, expected in collections:
        actual_names = [item_name for item_name, _ in actual]
        expected_names = [item_name for item_name, _ in expected]
        if actual_names != expected_names:
            raise ValueError(f"model {name} topology is not reconstructible from config")
        identities = [id(item) for _, item in actual]
        if len(identities) != len(set(identities)):
            raise ValueError(f"aliased model {name}s are unsupported by artifact version 1")
    storage_owners: dict[tuple[str, int], str] = {}
    for name, tensor in (*actual_parameters, *actual_buffers):
        if tensor.numel() == 0:
            continue
        storage_key = (str(tensor.device), tensor.untyped_storage().data_ptr())
        prior = storage_owners.get(storage_key)
        if prior is not None:
            raise ValueError(
                "shared tensor storage is unsupported by artifact version 1: "
                f"{prior!r} and {name!r}"
            )
        storage_owners[storage_key] = name
    actual_state = model.state_dict()
    expected_state = schema.state_dict()
    if list(actual_state) != list(expected_state) or any(
        tuple(actual_state[name].shape) != tuple(expected_state[name].shape)
        for name in expected_state
    ):
        raise ValueError("model state topology is not reconstructible from config")


def serialize_compact_binding_model(
    model: RoutedBindingModel,
    manifest: CompactExportManifest,
    selection: CPRankSelection,
) -> bytes:
    """Serialize a validated compact model to canonical non-executable bytes."""

    fingerprint = _validate_relationships(model, manifest, selection)
    # Apply the exact loader-side configuration schema and allocation ceiling
    # so serialization can never emit an artifact that this version rejects.
    parsed_config = _model_config_from_dict(_model_config_to_dict(model.config))
    if parsed_config != model.config:
        raise ValueError("model configuration is not canonically reconstructible")
    _validate_model_allocation_budget(parsed_config)
    _validate_serializable_topology(model)
    state = model.state_dict()
    if len(state) > _MAX_TENSORS:
        raise ValueError("model has too many state tensors for artifact version 1")

    payload_parts: list[bytes] = []
    tensor_records: list[dict[str, object]] = []
    offset = 0
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, Tensor) or tensor.layout is not torch.strided:
            raise TypeError("model state must contain only strided tensors")
        if len(name.encode("utf-8")) > _MAX_NAME_BYTES:
            raise ValueError("model-state tensor name is too long")
        dtype_name = _DTYPE_TO_NAME.get(tensor.dtype)
        if dtype_name is None:
            raise TypeError(f"unsupported artifact tensor dtype {tensor.dtype}")
        if tensor.ndim > _MAX_TENSOR_RANK or any(
            int(dimension) > _MAX_DIMENSION for dimension in tensor.shape
        ):
            raise ValueError(f"tensor {name} has unsupported shape")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"tensor {name} must be finite")
        raw = _portable_tensor_bytes(tensor)
        if len(raw) > _MAX_PAYLOAD_BYTES - offset:
            raise ValueError("compact artifact payload is too large")
        tensor_records.append(
            {
                "name": name,
                "dtype": dtype_name,
                "shape": [int(dimension) for dimension in tensor.shape],
                "offset": offset,
                "nbytes": len(raw),
                "sha256": _sha256(raw),
            }
        )
        payload_parts.append(raw)
        offset += len(raw)
    payload = b"".join(payload_parts)
    header_value = {
        "artifact_kind": _ARTIFACT_KIND,
        "schema_version": _FORMAT_VERSION,
        "model_config": _model_config_to_dict(model.config),
        "manifest": manifest.to_dict(),
        "selection": _selection_to_dict(selection),
        "exported_model_fingerprint": fingerprint,
        "module_training": _training_flags(model),
        "parameter_requires_grad": _requires_grad_flags(model),
        "tensors": tensor_records,
    }
    header = _canonical_json(header_value)
    if len(header) > _MAX_HEADER_BYTES:
        raise ValueError("compact artifact header is too large")
    prefix = _PREFIX.pack(
        _MAGIC,
        _FORMAT_VERSION,
        len(header),
        len(payload),
        hashlib.sha256(header).digest(),
        hashlib.sha256(payload).digest(),
    )
    return prefix + header + payload


def _parse_device(value: object) -> torch.device:
    if value is None:
        return torch.device("cpu")
    if not isinstance(value, (str, torch.device)):
        raise TypeError("device must be None, a string, or torch.device")
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("invalid artifact target device") from error
    if device.type == "cpu":
        if device.index is not None:
            raise ValueError("CPU artifact device must not have an index")
        return device
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA artifact device requested but CUDA is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError("CUDA artifact device index is unavailable")
        return torch.device("cuda", index)
    raise ValueError("artifact target device must be CPU or CUDA")


def _parse_bool_mapping(
    value: object, name: str
) -> dict[str, bool]:
    document = _mapping(value, name)
    result: dict[str, bool] = {}
    for key, item in document.items():
        if len(key.encode("utf-8")) > _MAX_NAME_BYTES:
            raise ValueError(f"{name} key is too long")
        if type(item) is not bool:
            raise TypeError(f"{name}.{key} must be boolean")
        result[key] = item
    return result


def _checked_shape(value: object, name: str) -> tuple[int, ...]:
    if type(value) is not list or len(value) > _MAX_TENSOR_RANK:
        raise ValueError(f"{name} must be an array with bounded rank")
    return tuple(
        _strict_int(
            dimension,
            f"{name} dimension",
            minimum=0,
            maximum=_MAX_DIMENSION,
        )
        for dimension in value
    )


def _shape_numel(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
        if result > _MAX_PAYLOAD_BYTES:
            raise ValueError("declared tensor shape is too large")
    return result


def _decode_tensors(
    value: object,
    payload: bytes,
    expected_state: Mapping[str, Tensor],
    parameter_names: frozenset[str],
    device: torch.device,
) -> dict[str, Tensor]:
    if type(value) is not list or len(value) > _MAX_TENSORS:
        raise ValueError("tensors must be a bounded JSON array")
    decoded: dict[str, Tensor] = {}
    expected_offset = 0
    prior_name: str | None = None
    for index, item in enumerate(value):
        record = _mapping(item, f"tensor record {index}")
        _keys(record, _TENSOR_KEYS, f"tensor record {index}")
        name = record["name"]
        if not isinstance(name, str) or not name:
            raise TypeError("tensor name must be a nonempty string")
        if len(name.encode("utf-8")) > _MAX_NAME_BYTES:
            raise ValueError("tensor name is too long")
        if prior_name is not None and name <= prior_name:
            raise ValueError("tensor records must have unique sorted names")
        prior_name = name
        if name not in expected_state:
            raise ValueError(f"artifact contains an unexpected tensor {name}")
        dtype_name = record["dtype"]
        if not isinstance(dtype_name, str) or dtype_name not in _NAME_TO_DTYPE:
            raise ValueError(f"tensor {name} has unsupported dtype")
        dtype = _NAME_TO_DTYPE[dtype_name]
        shape = _checked_shape(record["shape"], f"tensor {name} shape")
        if shape != tuple(expected_state[name].shape):
            raise ValueError(f"artifact tensor {name} has an architecture-invalid shape")
        if name in parameter_names and dtype not in (torch.float32, torch.float64):
            raise ValueError(f"artifact parameter {name} must be float32 or float64")
        offset = _strict_int(record["offset"], f"tensor {name} offset", minimum=0)
        nbytes = _strict_int(record["nbytes"], f"tensor {name} nbytes", minimum=0)
        digest = _strict_sha256(record["sha256"], f"tensor {name} sha256")
        expected_nbytes = _shape_numel(shape) * torch.empty((), dtype=dtype).element_size()
        if nbytes != expected_nbytes:
            raise ValueError(f"tensor {name} byte count does not match shape and dtype")
        if offset != expected_offset or nbytes > len(payload) - offset:
            raise ValueError("tensor payload ranges must be contiguous and in bounds")
        raw = payload[offset : offset + nbytes]
        if _sha256(raw) != digest:
            raise ValueError(f"tensor {name} checksum mismatch")
        native = _native_tensor_bytes(raw, torch.empty((), dtype=dtype).element_size())
        try:
            flat = torch.frombuffer(bytearray(native), dtype=dtype).clone()
            tensor = flat.reshape(shape).to(device=device)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"tensor {name} could not be reconstructed") from error
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"tensor {name} must be finite")
        decoded[name] = tensor
        expected_offset += nbytes
    if expected_offset != len(payload):
        raise ValueError("tensor records do not consume the complete payload")
    if set(decoded) != set(expected_state):
        missing = sorted(set(expected_state) - set(decoded))
        extra = sorted(set(decoded) - set(expected_state))
        raise ValueError(f"artifact tensor keys mismatch; missing={missing}, extra={extra}")
    return decoded


def _construct_model_preserving_rng(config: BindingModelConfig) -> RoutedBindingModel:
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        return RoutedBindingModel(config)
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _meta_model_schema(config: BindingModelConfig) -> RoutedBindingModel:
    """Build and size-check the architecture without real tensor allocation."""

    try:
        with torch.device("meta"):
            schema = RoutedBindingModel(config)
        state = schema.state_dict()
        elements = sum(tensor.numel() for tensor in state.values())
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("artifact model configuration has invalid dimensions") from error
    if len(state) > _MAX_TENSORS or elements > _MAX_MODEL_ELEMENTS:
        raise ValueError("artifact model configuration exceeds the allocation limit")
    return schema


def _validate_model_allocation_budget(config: BindingModelConfig) -> None:
    """Check the complete state shape on the meta device before allocation."""

    _meta_model_schema(config)


def deserialize_compact_binding_model(
    blob: bytes | bytearray | memoryview,
    expected_source_fingerprint: str | None = None,
    device: str | torch.device | None = None,
    *,
    expected_manifest_fingerprint: str | None = None,
    expected_selection_fingerprint: str | None = None,
) -> tuple[RoutedBindingModel, CompactExportManifest, CPRankSelection]:
    """Strictly load canonical compact-model bytes without executing code."""

    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("blob must be bytes-like")
    try:
        blob_size = memoryview(blob).nbytes
    except (TypeError, ValueError) as error:
        raise TypeError("blob must expose one contiguous byte buffer") from error
    if blob_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("compact artifact exceeds the total safety limit")
    artifact = bytes(blob)
    if len(artifact) < _PREFIX.size:
        raise ValueError("compact artifact is truncated")
    magic, version, header_length, payload_length, header_hash, payload_hash = (
        _PREFIX.unpack_from(artifact)
    )
    if magic != _MAGIC:
        raise ValueError("invalid compact artifact magic")
    if version != _FORMAT_VERSION:
        raise ValueError("unsupported compact artifact version")
    if header_length > _MAX_HEADER_BYTES:
        raise ValueError("compact artifact header exceeds the safety limit")
    if payload_length > _MAX_PAYLOAD_BYTES:
        raise ValueError("compact artifact payload exceeds the safety limit")
    expected_total = _PREFIX.size + header_length + payload_length
    if expected_total != len(artifact):
        raise ValueError("compact artifact is truncated or has trailing bytes")
    header_start = _PREFIX.size
    payload_start = header_start + header_length
    header = artifact[header_start:payload_start]
    payload = artifact[payload_start:]
    if hashlib.sha256(header).digest() != header_hash:
        raise ValueError("compact artifact header checksum mismatch")
    if hashlib.sha256(payload).digest() != payload_hash:
        raise ValueError("compact artifact payload checksum mismatch")

    root = _parse_canonical_json(header)
    _keys(root, _HEADER_KEYS, "artifact header")
    if root["artifact_kind"] != _ARTIFACT_KIND:
        raise ValueError("unsupported compact artifact kind")
    if _strict_int(root["schema_version"], "schema_version") != _FORMAT_VERSION:
        raise ValueError("unsupported compact artifact schema_version")
    exported_fingerprint = _strict_sha256(
        root["exported_model_fingerprint"], "exported_model_fingerprint"
    )
    config = _model_config_from_dict(root["model_config"])
    _validate_model_allocation_budget(config)
    manifest = CompactExportManifest.from_dict(
        _mapping(root["manifest"], "manifest")
    )
    selection = _selection_from_dict(root["selection"])
    if expected_source_fingerprint is not None:
        expected = _strict_sha256(
            expected_source_fingerprint, "expected_source_fingerprint"
        )
        if manifest.source_model_fingerprint != expected:
            raise ValueError("artifact source fingerprint does not match expectation")
    if expected_manifest_fingerprint is not None:
        expected = _strict_sha256(
            expected_manifest_fingerprint, "expected_manifest_fingerprint"
        )
        if manifest.fingerprint() != expected:
            raise ValueError("artifact manifest fingerprint does not match expectation")
    if expected_selection_fingerprint is not None:
        expected = _strict_sha256(
            expected_selection_fingerprint, "expected_selection_fingerprint"
        )
        if selection.fingerprint() != expected:
            raise ValueError("artifact selection fingerprint does not match expectation")
    target_device = _parse_device(device)
    training = _parse_bool_mapping(root["module_training"], "module_training")
    requires_grad = _parse_bool_mapping(
        root["parameter_requires_grad"], "parameter_requires_grad"
    )

    schema = _meta_model_schema(config)
    expected_modules = {name for name, _ in schema.named_modules()}
    expected_parameters = {name for name, _ in schema.named_parameters()}
    if set(training) != expected_modules:
        raise ValueError("module_training keys do not match reconstructed architecture")
    if set(requires_grad) != expected_parameters:
        raise ValueError(
            "parameter_requires_grad keys do not match reconstructed architecture"
        )
    decoded = _decode_tensors(
        root["tensors"],
        payload,
        schema.state_dict(),
        frozenset(name for name, _ in schema.named_parameters()),
        target_device,
    )
    model = _construct_model_preserving_rng(config).to(device=target_device)
    try:
        model.load_state_dict(decoded, strict=True, assign=True)
    except RuntimeError as error:
        raise ValueError("artifact tensors do not fit reconstructed architecture") from error
    for name, module in model.named_modules():
        module.training = training[name]
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(requires_grad[name])

    actual_fingerprint = _validate_relationships(model, manifest, selection)
    if actual_fingerprint != exported_fingerprint:
        raise ValueError("artifact exported model fingerprint mismatch")
    return model, manifest, selection


__all__ = [
    "deserialize_compact_binding_model",
    "serialize_compact_binding_model",
]
