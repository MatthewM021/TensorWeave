#!/usr/bin/env python3
"""Replay a compact binding-model artifact in an independent Python process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import tempfile
from typing import Mapping

import torch
from torch import Tensor

from tnlm_v3.compact_artifact import deserialize_compact_binding_model
from tnlm_v3.data import BindingModelInputs
from tnlm_v3.truncation import model_state_fingerprint


_FIXTURE_SCHEMA_VERSION = 1
_RESULT_SCHEMA_VERSION = 1
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_FIXTURE_BYTES = 8 * 1024 * 1024
_MAX_BATCH_SIZE = 256
_MAX_TIME_STEPS = 4096
_MAX_INPUT_ELEMENTS = 4096
_TORCH_THREADS = 1
_INPUT_DTYPES = {
    "token_ids": torch.int64,
    "event_kinds": torch.int64,
    "primary_key_ids": torch.int64,
    "secondary_key_ids": torch.int64,
    "arguments": torch.int64,
    "valid_mask": torch.bool,
}
_HASH_DOMAIN = b"tnlm_v3.compact_replay_tensor.v1\x00"
_HEX_DIGITS = frozenset("0123456789abcdef")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(character not in _HEX_DIGITS for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _read_bounded(path: Path, limit: int, name: str) -> bytes:
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{name} exceeds the {limit}-byte safety limit")
    return data


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant {value!r} is forbidden")


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("replay record contains a non-finite float")
    if isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _finite(dict(value))
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _strict_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return value


def _strict_keys(
    value: Mapping[str, object], expected: set[str], name: str
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"invalid {name} fields; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _load_fixture(data: bytes) -> BindingModelInputs:
    if len(data) > _MAX_FIXTURE_BYTES:
        raise ValueError(
            f"fixture exceeds the {_MAX_FIXTURE_BYTES}-byte safety limit"
        )
    try:
        text = data.decode("utf-8", errors="strict")
        document = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("fixture must be valid UTF-8 strict JSON") from error
    root = _strict_mapping(document, "fixture")
    _strict_keys(root, {"schema_version", "inputs"}, "fixture")
    version = root["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("fixture schema_version must be integer 1")
    encoded_inputs = _strict_mapping(root["inputs"], "fixture inputs")
    _strict_keys(encoded_inputs, set(_INPUT_DTYPES), "fixture inputs")

    tensors: dict[str, Tensor] = {}
    common_shape: tuple[int, int] | None = None
    for name, expected_dtype in _INPUT_DTYPES.items():
        encoded = _strict_mapping(encoded_inputs[name], f"input {name}")
        _strict_keys(encoded, {"dtype", "shape", "values"}, f"input {name}")
        expected_name = "bool" if expected_dtype is torch.bool else "int64"
        if encoded["dtype"] != expected_name:
            raise ValueError(f"input {name} must declare dtype {expected_name}")
        shape_value = encoded["shape"]
        if (
            type(shape_value) is not list
            or len(shape_value) != 2
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
                for dimension in shape_value
            )
        ):
            raise ValueError(
                f"input {name} shape must contain two nonnegative integers"
            )
        shape = (shape_value[0], shape_value[1])
        if not 1 <= shape[0] <= _MAX_BATCH_SIZE:
            raise ValueError(
                f"input {name} batch dimension must lie in "
                f"[1,{_MAX_BATCH_SIZE}]"
            )
        if not 1 <= shape[1] <= _MAX_TIME_STEPS:
            raise ValueError(
                f"input {name} time dimension must lie in "
                f"[1,{_MAX_TIME_STEPS}]"
            )
        if shape[0] * shape[1] > _MAX_INPUT_ELEMENTS:
            raise ValueError(
                f"input {name} exceeds the {_MAX_INPUT_ELEMENTS}-element "
                "safety limit"
            )
        if common_shape is None:
            common_shape = shape
        elif shape != common_shape:
            raise ValueError("all fixture input tensors must share shape [N,T]")
        values = encoded["values"]
        if type(values) is not list or len(values) != shape[0]:
            raise ValueError(f"input {name} values must have shape {shape}")
        for row in values:
            if type(row) is not list or len(row) != shape[1]:
                raise ValueError(f"input {name} values must have shape {shape}")
            if expected_dtype is torch.bool:
                if any(type(current) is not bool for current in row):
                    raise ValueError(f"input {name} values must be booleans")
            elif any(
                isinstance(current, bool) or not isinstance(current, int)
                for current in row
            ):
                raise ValueError(f"input {name} values must be integers")
        try:
            tensor = torch.tensor(values, dtype=expected_dtype)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"input {name} has invalid values") from error
        if tuple(tensor.shape) != shape:
            raise ValueError(f"input {name} values do not match declared shape")
        tensors[name] = tensor
    assert common_shape is not None
    return BindingModelInputs(**tensors)


def _canonical_tensor_hash(tensor: Tensor) -> str:
    if not isinstance(tensor, Tensor) or tensor.layout is not torch.strided:
        raise TypeError("replay outputs must be strided tensors")
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    if sys.byteorder != "little" and value.element_size() > 1:
        width = value.element_size()
        raw = b"".join(
            raw[index : index + width][::-1]
            for index in range(0, len(raw), width)
        )
    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    dtype = str(value.dtype).encode("ascii")
    digest.update(struct.pack("<Q", len(dtype)))
    digest.update(dtype)
    digest.update(struct.pack("<Q", value.ndim))
    for dimension in value.shape:
        digest.update(struct.pack("<Q", int(dimension)))
    digest.update(raw)
    return digest.hexdigest()


def _output_hashes(output) -> dict[str, str]:
    forest = output.forest_state
    router = output.router_state
    return {
        "routes": _canonical_tensor_hash(output.routes),
        "route_logits": _canonical_tensor_hash(output.route_logits),
        "route_probabilities": _canonical_tensor_hash(
            output.route_probabilities
        ),
        "value_logits": _canonical_tensor_hash(output.value_logits),
        "forest_slots": _canonical_tensor_hash(forest.slots),
        "forest_occupied": _canonical_tensor_hash(forest.occupied),
        "forest_counts": _canonical_tensor_hash(forest.counts),
        "forest_valid_steps": _canonical_tensor_hash(forest.valid_steps),
        "router_prototypes": _canonical_tensor_hash(router.prototypes),
        "router_occupied": _canonical_tensor_hash(router.occupied),
        "router_ages": _canonical_tensor_hash(router.ages),
        "router_loads": _canonical_tensor_hash(router.loads),
        "router_global_state": _canonical_tensor_hash(router.global_state),
        "router_global_occupied": _canonical_tensor_hash(router.global_occupied),
        "router_global_load": _canonical_tensor_hash(router.global_load),
        "router_valid_steps": _canonical_tensor_hash(router.valid_steps),
        **{
            f"diagnostic:{name}": _canonical_tensor_hash(value)
            for name, value in sorted(output.diagnostics.items())
        },
    }


def _raw_tensor_bytes(model: torch.nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-artifact-sha256", required=True)
    parser.add_argument("--expected-source-fingerprint", required=True)
    parser.add_argument("--expected-manifest-fingerprint", required=True)
    parser.add_argument("--expected-selection-fingerprint", required=True)
    args = parser.parse_args()
    for name in ("artifact", "fixture", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            parser.error(f"--{name} must be an absolute path")
        setattr(args, name, value.resolve())
    if len({args.artifact, args.fixture, args.output}) != 3:
        parser.error("--artifact, --fixture, and --output must be distinct paths")
    for name in (
        "expected_artifact_sha256",
        "expected_source_fingerprint",
        "expected_manifest_fingerprint",
        "expected_selection_fingerprint",
    ):
        try:
            setattr(args, name, _strict_sha256(getattr(args, name), f"--{name.replace('_', '-')}"))
        except ValueError as error:
            parser.error(str(error))
    return args


def main() -> None:
    args = _parse_args()
    record: dict[str, object] = {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "status": "in_progress",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid(),
    }
    _atomic_json(args.output, record)
    try:
        artifact_bytes = _read_bounded(
            args.artifact, _MAX_ARTIFACT_BYTES, "artifact"
        )
        record["artifact_sha256"] = _sha256(artifact_bytes)
        record["expected_provenance"] = {
            "artifact_sha256": args.expected_artifact_sha256,
            "source_model_fingerprint": args.expected_source_fingerprint,
            "manifest_fingerprint": args.expected_manifest_fingerprint,
            "selection_fingerprint": args.expected_selection_fingerprint,
        }
        fixture_bytes = _read_bounded(
            args.fixture, _MAX_FIXTURE_BYTES, "fixture"
        )
        record["fixture_sha256"] = _sha256(fixture_bytes)
        if record["artifact_sha256"] != args.expected_artifact_sha256:
            raise ValueError("artifact SHA-256 does not match the trusted expectation")
        inputs = _load_fixture(fixture_bytes)
        loaded = deserialize_compact_binding_model(
            artifact_bytes,
            expected_source_fingerprint=args.expected_source_fingerprint,
            expected_manifest_fingerprint=args.expected_manifest_fingerprint,
            expected_selection_fingerprint=args.expected_selection_fingerprint,
        )
        if not isinstance(loaded, tuple) or len(loaded) != 3:
            raise TypeError(
                "deserialize_compact_binding_model must return "
                "model, manifest, selection"
            )
        model, manifest, selection = loaded
        model.eval()
        torch.set_num_threads(_TORCH_THREADS)
        with torch.no_grad():
            streaming = model(inputs, implementation="streaming")
            parallel = model(inputs, implementation="parallel")
        streaming_hashes = _output_hashes(streaming)
        parallel_hashes = _output_hashes(parallel)
        record.update(
            {
                "status": "passed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_fingerprint": model_state_fingerprint(model),
                "source_model_fingerprint": (
                    manifest.source_model_fingerprint
                ),
                "manifest_fingerprint": manifest.fingerprint(),
                "selection_fingerprint": selection.fingerprint(),
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "raw_tensor_bytes": _raw_tensor_bytes(model),
                "streaming": streaming_hashes,
                "parallel": parallel_hashes,
                "streaming_parallel_hashes_equal": (
                    streaming_hashes == parallel_hashes
                ),
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "torch_threads": torch.get_num_threads(),
                    "cuda_available": torch.cuda.is_available(),
                },
            }
        )
        _atomic_json(args.output, record)
    except Exception as error:
        record.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        _atomic_json(args.output, record)
        raise
    print(f"COMPACT_ARTIFACT_REPLAY_PASSED: {args.output}")


if __name__ == "__main__":
    main()
