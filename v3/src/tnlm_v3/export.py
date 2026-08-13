"""Canonical serialization for the runtime tensor-forest state.

The format deliberately avoids pickle and ``torch.save``.  Its wire layout is::

    magic | version | uint64_le(header_length) | canonical_json | tensor_bytes

The JSON is compact, has sorted keys, and describes fixed-order, contiguous,
little-endian tensor payloads.  A payload digest detects otherwise invisible
corruption of tensor data.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from collections.abc import Mapping
from typing import Any, Optional

import torch

from .forest import ForestState


_MAGIC = b"TNLMFST\x00"
_VERSION = 1
_PREFIX = struct.Struct("<8sBQ")
_SCHEMA = "tnlm_v3.forest_state"
_TENSOR_ORDER = ("slots", "occupied", "counts", "valid_steps")
_DTYPE_NAMES = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bool: "bool",
    torch.int64: "int64",
}
_DTYPES = {name: dtype for dtype, name in _DTYPE_NAMES.items()}
_ITEM_SIZES = {"float32": 4, "float64": 8, "bool": 1, "int64": 8}
_REQUIRED_HEADER_KEYS = {
    "config_fingerprint",
    "payload_bytes",
    "payload_sha256",
    "schema",
    "tensors",
    "version",
}
_REQUIRED_TENSOR_KEYS = {"dtype", "nbytes", "offset", "shape"}
_EXPECTED_RANKS = {"slots": 4, "occupied": 3, "counts": 2, "valid_steps": 1}
_MAX_HEADER_BYTES = 1_000_000
_MAX_TENSOR_DIMENSION = (1 << 63) - 1


def _product(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    """Return canonical little-endian bytes from a contiguous CPU tensor."""

    tensor = tensor.detach().cpu().contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    if sys.byteorder == "little" or tensor.element_size() == 1:
        return raw
    width = tensor.element_size()
    return b"".join(raw[index : index + width][::-1] for index in range(0, len(raw), width))


def _retained_scale_count(occupied: torch.Tensor) -> int:
    active_by_scale = occupied.any(dim=(0, 1))
    active = torch.nonzero(active_by_scale, as_tuple=False)
    return max(1, int(active[-1, 0]) + 1) if active.numel() else 1


def _validate_binary_occupancy(occupied: torch.Tensor, counts: torch.Tensor) -> None:
    scales = occupied.shape[2]
    if not 1 <= scales <= 63:
        raise ValueError("state must contain between 1 and 63 scales")
    scale_ids = torch.arange(scales, device=counts.device, dtype=torch.int64)
    expected = ((counts.unsqueeze(-1) >> scale_ids) & 1).bool()
    if not torch.equal(occupied, expected):
        raise ValueError("occupied must equal the binary pattern encoded by counts")
    if scales < 63 and torch.any(counts >= (1 << scales)):
        raise ValueError("state lacks scale capacity for its counts")


def _routed_counts_fit_valid_steps(
    counts: torch.Tensor, valid_steps: torch.Tensor
) -> bool:
    rows = counts.detach().cpu().tolist()
    clocks = valid_steps.detach().cpu().tolist()
    return all(
        sum(int(value) for value in row) <= int(clock)
        for row, clock in zip(rows, clocks, strict=True)
    )


def _validate_state(state: ForestState) -> None:
    if not isinstance(state.slots, torch.Tensor) or state.slots.ndim != 4:
        raise ValueError("slots must be a rank-4 tensor [N,P,S,D]")
    if state.slots.dtype not in (torch.float32, torch.float64):
        raise ValueError("slots dtype must be float32 or float64")
    if not isinstance(state.occupied, torch.Tensor) or state.occupied.dtype != torch.bool:
        raise ValueError("occupied must be a bool tensor")
    if not isinstance(state.counts, torch.Tensor) or state.counts.dtype != torch.int64:
        raise ValueError("counts must be an int64 tensor")
    if not isinstance(state.valid_steps, torch.Tensor) or state.valid_steps.dtype != torch.int64:
        raise ValueError("valid_steps must be an int64 tensor")

    batches, paths, scales, _ = state.slots.shape
    if batches <= 0 or paths <= 0 or scales <= 0 or state.slots.shape[3] <= 0:
        raise ValueError("slots must have positive N, P, S, and D dimensions")
    if tuple(state.occupied.shape) != (batches, paths, scales):
        raise ValueError("occupied shape must equal slots.shape[:3]")
    if tuple(state.counts.shape) != (batches, paths):
        raise ValueError("counts shape must be [N,P]")
    if tuple(state.valid_steps.shape) != (batches,):
        raise ValueError("valid_steps shape must be [N]")
    devices = {
        state.slots.device,
        state.occupied.device,
        state.counts.device,
        state.valid_steps.device,
    }
    if len(devices) != 1:
        raise ValueError("all forest-state tensors must share one device")
    if torch.any(state.counts < 0) or torch.any(state.valid_steps < 0):
        raise ValueError("counts and valid_steps must be nonnegative")
    if not _routed_counts_fit_valid_steps(state.counts, state.valid_steps):
        raise ValueError("routed counts cannot exceed valid_steps")
    _validate_binary_occupancy(state.occupied, state.counts)
    inactive_values = state.slots.masked_select(~state.occupied.unsqueeze(-1))
    if torch.any(inactive_values != 0):
        raise ValueError("unoccupied state slots must be zero")


def serialize_forest_state(state: ForestState, config_fingerprint: str) -> bytes:
    """Serialize ``state`` into deterministic, portable, non-executable bytes.

    Completely empty trailing scales are omitted across all batch/path lanes;
    at least one scale is always retained.
    """

    if not isinstance(config_fingerprint, str) or not config_fingerprint:
        raise ValueError("config_fingerprint must be a nonempty string")
    _validate_state(state)

    scales = _retained_scale_count(state.occupied)
    tensors = {
        "slots": state.slots[:, :, :scales, :],
        "occupied": state.occupied[:, :, :scales],
        "counts": state.counts,
        "valid_steps": state.valid_steps,
    }

    payload_parts: list[bytes] = []
    metadata: dict[str, dict[str, Any]] = {}
    offset = 0
    for name in _TENSOR_ORDER:
        tensor = tensors[name]
        dtype_name = _DTYPE_NAMES[tensor.dtype]
        raw = _raw_bytes(tensor)
        shape = [int(value) for value in tensor.shape]
        expected = _product(shape) * _ITEM_SIZES[dtype_name]
        if len(raw) != expected:
            raise RuntimeError(f"unexpected byte count for {name}")
        metadata[name] = {
            "dtype": dtype_name,
            "nbytes": len(raw),
            "offset": offset,
            "shape": shape,
        }
        payload_parts.append(raw)
        offset += len(raw)

    payload = b"".join(payload_parts)
    header = {
        "config_fingerprint": config_fingerprint,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "schema": _SCHEMA,
        "tensors": metadata,
        "version": _VERSION,
    }
    header_bytes = _canonical_json(header)
    return _PREFIX.pack(_MAGIC, _VERSION, len(header_bytes)) + header_bytes + payload


def _require_exact_mapping_keys(
    value: object, expected: set[str], description: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"invalid {description} schema")
    return value


def _parse_nonnegative_int(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} must be a nonnegative integer")
    return value


def _parse_shape(value: object, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"invalid shape for {name}")
    shape = [_parse_nonnegative_int(dimension, f"{name} shape dimension") for dimension in value]
    if len(shape) != _EXPECTED_RANKS[name]:
        raise ValueError(f"invalid rank for {name}")
    if any(dimension > _MAX_TENSOR_DIMENSION for dimension in shape):
        raise ValueError(f"shape dimension is too large for {name}")
    return shape


def _decode_tensor(
    payload: bytes, name: str, metadata: Mapping[str, Any]
) -> torch.Tensor:
    dtype_name = metadata["dtype"]
    if not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
        raise ValueError(f"unsupported dtype for {name}")
    shape = _parse_shape(metadata["shape"], name)
    offset = _parse_nonnegative_int(metadata["offset"], f"{name} offset")
    nbytes = _parse_nonnegative_int(metadata["nbytes"], f"{name} nbytes")
    expected = _product(shape) * _ITEM_SIZES[dtype_name]
    if nbytes != expected or offset + nbytes > len(payload):
        raise ValueError(f"invalid byte range for {name}")

    raw = bytearray(payload[offset : offset + nbytes])
    width = _ITEM_SIZES[dtype_name]
    if dtype_name == "bool" and any(value not in (0, 1) for value in raw):
        raise ValueError(f"noncanonical boolean payload for {name}")
    if sys.byteorder != "little" and width > 1:
        raw = bytearray(
            b"".join(
                raw[index : index + width][::-1]
                for index in range(0, len(raw), width)
            )
        )
    if nbytes == 0:
        return torch.empty(shape, dtype=_DTYPES[dtype_name])
    tensor = torch.frombuffer(raw, dtype=_DTYPES[dtype_name])
    return tensor.reshape(shape).clone()


def deserialize_forest_state(
    blob: bytes,
    expected_config_fingerprint: Optional[str] = None,
    device: Optional[torch.device | str] = None,
) -> ForestState:
    """Validate and deserialize a canonical forest-state blob."""

    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("blob must be bytes-like")
    blob = bytes(blob)
    if len(blob) < _PREFIX.size:
        raise ValueError("truncated forest-state prefix")

    magic, prefix_version, header_length = _PREFIX.unpack_from(blob)
    if magic != _MAGIC:
        raise ValueError("invalid forest-state magic")
    if prefix_version != _VERSION:
        raise ValueError("unsupported forest-state version")
    header_end = _PREFIX.size + header_length
    if header_length > _MAX_HEADER_BYTES:
        raise ValueError("forest-state header exceeds the size limit")
    if header_end > len(blob):
        raise ValueError("truncated forest-state header")

    header_bytes = blob[_PREFIX.size : header_end]
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid forest-state header") from error
    header = _require_exact_mapping_keys(header, _REQUIRED_HEADER_KEYS, "header")
    if _canonical_json(header) != header_bytes:
        raise ValueError("header is not canonical JSON")
    header_version = header["version"]
    if (
        header["schema"] != _SCHEMA
        or isinstance(header_version, bool)
        or not isinstance(header_version, int)
        or header_version != _VERSION
    ):
        raise ValueError("unsupported forest-state schema or version")

    fingerprint = header["config_fingerprint"]
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("invalid config fingerprint")
    if expected_config_fingerprint is not None and fingerprint != expected_config_fingerprint:
        raise ValueError("forest-state config fingerprint mismatch")

    payload_bytes = _parse_nonnegative_int(header["payload_bytes"], "payload_bytes")
    if len(blob) != header_end + payload_bytes:
        raise ValueError("truncated payload or trailing data")
    payload = blob[header_end:]
    digest = header["payload_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("invalid payload digest")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("forest-state payload checksum mismatch")

    tensor_metadata = _require_exact_mapping_keys(
        header["tensors"], set(_TENSOR_ORDER), "tensor metadata"
    )
    tensors: dict[str, torch.Tensor] = {}
    next_offset = 0
    for name in _TENSOR_ORDER:
        metadata = _require_exact_mapping_keys(
            tensor_metadata[name], _REQUIRED_TENSOR_KEYS, f"{name} metadata"
        )
        offset = _parse_nonnegative_int(metadata["offset"], f"{name} offset")
        nbytes = _parse_nonnegative_int(metadata["nbytes"], f"{name} nbytes")
        if offset != next_offset:
            raise ValueError("tensor payloads must be contiguous and in fixed order")
        tensors[name] = _decode_tensor(payload, name, metadata)
        next_offset += nbytes
    if next_offset != payload_bytes:
        raise ValueError("tensor metadata does not cover the complete payload")

    if tensors["slots"].dtype not in (torch.float32, torch.float64):
        raise ValueError("slots dtype must be float32 or float64")
    if tensors["occupied"].dtype != torch.bool:
        raise ValueError("occupied dtype must be bool")
    if tensors["counts"].dtype != torch.int64 or tensors["valid_steps"].dtype != torch.int64:
        raise ValueError("count tensors must be int64")
    if tensors["slots"].ndim != 4 or tensors["slots"].shape[2] < 1:
        raise ValueError("invalid slots shape")

    batches, paths, scales, dimension = tensors["slots"].shape
    if batches <= 0 or paths <= 0 or scales <= 0 or dimension <= 0:
        raise ValueError("slots dimensions must all be positive")
    if tuple(tensors["occupied"].shape) != (batches, paths, scales):
        raise ValueError("occupied shape does not match slots")
    if tuple(tensors["counts"].shape) != (batches, paths):
        raise ValueError("counts shape does not match slots")
    if tuple(tensors["valid_steps"].shape) != (batches,):
        raise ValueError("valid_steps shape does not match slots")
    if torch.any(tensors["counts"] < 0) or torch.any(tensors["valid_steps"] < 0):
        raise ValueError("counts and valid_steps must be nonnegative")
    if not _routed_counts_fit_valid_steps(
        tensors["counts"], tensors["valid_steps"]
    ):
        raise ValueError("routed counts cannot exceed valid_steps")
    _validate_binary_occupancy(tensors["occupied"], tensors["counts"])

    if device is not None:
        tensors = {name: tensor.to(device=device) for name, tensor in tensors.items()}
    state = ForestState(
        slots=tensors["slots"],
        occupied=tensors["occupied"],
        counts=tensors["counts"],
        valid_steps=tensors["valid_steps"],
    )
    _validate_state(state)
    return state


__all__ = ["deserialize_forest_state", "serialize_forest_state"]
