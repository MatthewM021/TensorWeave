"""Strict configuration and evidence helpers for the Milestone-3 audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import struct
import sys
import tempfile
from typing import Mapping

import torch
import yaml
from torch import Tensor

from .data import BindingModelInputs
from .factory import load_binding_experiment_config
from .forest import ForestState


_SCHEMA_VERSION = 1
_METHOD = "parameter_energy_v1"
_MAX_CONFIG_BYTES = 64 * 1024
_MAX_SEQUENCE_LENGTH = 4096
_MAX_LENGTH_ENTRIES = 256
_MAX_SEED = (1 << 63) - 1
_MAX_TORCH_THREADS = 64
_MAX_WARMUP_ITERATIONS = 100
_MAX_TIMED_ITERATIONS = 1000
_MAX_FLOAT32_TOLERANCE = 0.01
_HASH_DTYPES = frozenset(
    (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    )
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects aliases and duplicate mapping keys."""

    def compose_node(self, parent, index):  # type: ignore[no-untyped-def]
        if self.check_event(yaml.AliasEvent):
            raise ValueError("YAML aliases are forbidden")
        return super().compose_node(parent, index)


def _construct_unique_mapping(loader, node, deep=False):  # type: ignore[no-untyped-def]
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ValueError("YAML mapping keys must be hashable") from error
        if duplicate:
            raise ValueError(f"duplicate YAML key {key!r} is forbidden")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _plain_positive_int(
    value: object, name: str, *, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _finite_nonnegative_real(
    value: object, name: str, *, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _strict_relative_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    path = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or ".." in path.parts
        or "\\" in value
        or ":" in value
    ):
        raise ValueError(f"{name} must be a normalized repository-relative POSIX path")
    if path.as_posix() != value:
        raise ValueError(f"{name} must be normalized")
    return value


@dataclass(frozen=True)
class ExportAuditConfig:
    """Predeclared deterministic implementation-audit settings."""

    schema_version: int
    source_config: str
    target_cp_rank: int
    selection_method: str
    calibration_split: str
    calibration_seed: int
    calibration_lengths: tuple[int, ...]
    evaluation_split: str
    evaluation_seed: int
    evaluation_lengths: tuple[int, ...]
    torch_threads: int
    warmup_iterations: int
    timed_iterations: int
    float32_rtol: float
    float32_atol: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != _SCHEMA_VERSION
        ):
            raise ValueError("schema_version must be integer 1")
        object.__setattr__(
            self, "source_config", _strict_relative_path(self.source_config, "source_config")
        )
        _plain_positive_int(self.target_cp_rank, "target_cp_rank")
        if self.selection_method != _METHOD:
            raise ValueError(f"selection_method must be {_METHOD!r}")
        if self.calibration_split != "validation":
            raise ValueError("calibration_split must be 'validation'")
        if self.evaluation_split != "eval":
            raise ValueError("evaluation_split must be 'eval'")
        for name in ("calibration_seed", "evaluation_seed"):
            _plain_positive_int(getattr(self, name), name, maximum=_MAX_SEED)
        _plain_positive_int(
            self.torch_threads, "torch_threads", maximum=_MAX_TORCH_THREADS
        )
        _plain_positive_int(
            self.warmup_iterations,
            "warmup_iterations",
            maximum=_MAX_WARMUP_ITERATIONS,
        )
        _plain_positive_int(
            self.timed_iterations,
            "timed_iterations",
            maximum=_MAX_TIMED_ITERATIONS,
        )
        for name in ("calibration_lengths", "evaluation_lengths"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise TypeError(f"{name} must be a nonempty tuple")
            if len(values) > _MAX_LENGTH_ENTRIES:
                raise ValueError(f"{name} contains too many entries")
            for value in values:
                _plain_positive_int(
                    value, f"{name} entry", maximum=_MAX_SEQUENCE_LENGTH
                )
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
        if max(self.calibration_lengths) >= max(self.evaluation_lengths):
            raise ValueError("evaluation must include real updates beyond calibration length")
        object.__setattr__(
            self,
            "float32_rtol",
            _finite_nonnegative_real(
                self.float32_rtol,
                "float32_rtol",
                maximum=_MAX_FLOAT32_TOLERANCE,
            ),
        )
        object.__setattr__(
            self,
            "float32_atol",
            _finite_nonnegative_real(
                self.float32_atol,
                "float32_atol",
                maximum=_MAX_FLOAT32_TOLERANCE,
            ),
        )

    def canonical_json(self) -> str:
        return json.dumps(
            asdict(self), allow_nan=False, sort_keys=True, separators=(",", ":")
        )

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_export_audit_config(path: str | Path) -> ExportAuditConfig:
    """Load the exact schema and validate it against its source experiment."""

    resolved = Path(path).resolve()
    repository_root = Path(__file__).resolve().parents[3]
    if not resolved.is_relative_to(repository_root) or not resolved.is_file():
        raise ValueError("export-audit config must be a file inside this repository")
    raw = resolved.read_bytes()
    if len(raw) > _MAX_CONFIG_BYTES:
        raise ValueError("export-audit config exceeds the byte limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("export-audit config must be UTF-8") from error
    try:
        document = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise ValueError("export-audit config is invalid YAML") from error
    expected = {field.name for field in fields(ExportAuditConfig)}
    if not isinstance(document, Mapping) or set(document) != expected:
        missing = sorted(expected - set(document)) if isinstance(document, Mapping) else sorted(expected)
        unknown = sorted(set(document) - expected) if isinstance(document, Mapping) else []
        raise ValueError(f"invalid export-audit keys; missing={missing}, unknown={unknown}")
    values = dict(document)
    for name in ("calibration_lengths", "evaluation_lengths"):
        raw = values[name]
        if type(raw) is not list:
            raise TypeError(f"{name} must be a YAML sequence")
        values[name] = tuple(raw)
    config = ExportAuditConfig(**values)
    source_path = (repository_root / config.source_config).resolve()
    if not source_path.is_relative_to(repository_root) or not source_path.is_file():
        raise ValueError("source_config does not resolve to a repository file")
    source = load_binding_experiment_config(source_path)
    if config.target_cp_rank >= source.model.cp_rank:
        raise ValueError("target_cp_rank must be smaller than source CP rank")
    if min(config.calibration_lengths) < source.task.min_length or max(
        config.calibration_lengths
    ) > source.task.max_length:
        raise ValueError("calibration lengths must lie inside the source training range")
    if min(config.evaluation_lengths) < source.task.min_length:
        raise ValueError("evaluation lengths must satisfy the task grammar minimum")
    return config


def validate_finite_json(value: object, path: str = "$") -> None:
    """Reject values outside strict finite JSON before evidence is written."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float at {path}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string JSON key at {path}")
            validate_finite_json(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_finite_json(child, f"{path}[{index}]")
        return
    raise TypeError(f"unsupported JSON value {type(value).__name__} at {path}")


def atomic_write_json(path: str | Path, value: dict[str, object]) -> dict[str, object]:
    """Atomically replace a strict JSON record and return a detached snapshot."""

    if type(value) is not dict:
        raise TypeError("JSON evidence root must be an object")
    validate_finite_json(value)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    snapshot = json.loads(encoded)
    if not isinstance(snapshot, dict):
        raise TypeError("JSON evidence root must be an object")
    return snapshot


def _canonical_tensor_bytes(tensor: Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    if sys.byteorder == "little" or value.element_size() == 1:
        return raw
    width = value.element_size()
    return b"".join(
        raw[index : index + width][::-1]
        for index in range(0, len(raw), width)
    )


def tensor_sha256(tensor: Tensor) -> str:
    """Hash dtype, shape, and portable bytes for one strided tensor."""

    if not isinstance(tensor, Tensor) or tensor.layout is not torch.strided:
        raise TypeError("tensor must be a strided torch.Tensor")
    if tensor.dtype not in _HASH_DTYPES:
        raise TypeError(f"unsupported evidence tensor dtype {tensor.dtype}")
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise ValueError("evidence tensor must be finite")
    digest = hashlib.sha256()
    dtype = str(tensor.dtype).encode("ascii")
    digest.update(struct.pack("<Q", len(dtype)))
    digest.update(dtype)
    digest.update(struct.pack("<Q", tensor.ndim))
    for dimension in tensor.shape:
        digest.update(struct.pack("<Q", int(dimension)))
    digest.update(_canonical_tensor_bytes(tensor))
    return digest.hexdigest()


def binding_inputs_sha256(inputs: BindingModelInputs) -> str:
    if not isinstance(inputs, BindingModelInputs):
        raise TypeError("inputs must be BindingModelInputs")
    digest = hashlib.sha256()
    for field in fields(BindingModelInputs):
        digest.update(field.name.encode("ascii"))
        digest.update(bytes.fromhex(tensor_sha256(getattr(inputs, field.name))))
    return digest.hexdigest()


def forest_state_sha256(state: ForestState) -> str:
    if not isinstance(state, ForestState):
        raise TypeError("state must be ForestState")
    digest = hashlib.sha256()
    for name in ("slots", "occupied", "counts", "valid_steps"):
        digest.update(name.encode("ascii"))
        digest.update(bytes.fromhex(tensor_sha256(getattr(state, name))))
    return digest.hexdigest()


__all__ = [
    "ExportAuditConfig",
    "atomic_write_json",
    "binding_inputs_sha256",
    "forest_state_sha256",
    "load_export_audit_config",
    "tensor_sha256",
    "validate_finite_json",
]
