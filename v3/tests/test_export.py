from __future__ import annotations

import json
import struct

import pytest
import torch


forest = pytest.importorskip("tnlm_v3.forest")

from tnlm_v3.export import deserialize_forest_state, serialize_forest_state


ForestState = forest.ForestState
PREFIX = struct.Struct("<8sBQ")


def make_state(dtype: torch.dtype = torch.float32, scales: int = 6) -> ForestState:
    slots = torch.zeros((2, 3, scales, 4), dtype=dtype)
    occupied = torch.zeros((2, 3, scales), dtype=torch.bool)
    occupied[0, 0, 0] = True
    occupied[0, 1, 2] = True
    occupied[1, 2, 3] = True
    slots[0, 0, 0] = torch.tensor([1, 2, 3, 4], dtype=dtype)
    slots[0, 1, 2] = torch.tensor([5, 6, 7, 8], dtype=dtype)
    slots[1, 2, 3] = torch.tensor([9, 10, 11, 12], dtype=dtype)
    return ForestState(
        slots=slots,
        occupied=occupied,
        counts=torch.tensor([[1, 4, 0], [0, 0, 8]], dtype=torch.int64),
        valid_steps=torch.tensor([5, 8], dtype=torch.int64),
    )


def assert_state_equal(actual: ForestState, expected: ForestState) -> None:
    assert torch.equal(actual.slots, expected.slots)
    assert torch.equal(actual.occupied, expected.occupied)
    assert torch.equal(actual.counts, expected.counts)
    assert torch.equal(actual.valid_steps, expected.valid_steps)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_roundtrip_is_byte_deterministic_and_preserves_dtype(dtype: torch.dtype) -> None:
    state = make_state(dtype)
    first = serialize_forest_state(state, "config-sha256")
    second = serialize_forest_state(state, "config-sha256")
    assert first == second

    restored = deserialize_forest_state(first, "config-sha256")
    assert restored.slots.dtype == dtype
    assert_state_equal(
        restored,
        ForestState(
            slots=state.slots[:, :, :4, :],
            occupied=state.occupied[:, :, :4],
            counts=state.counts,
            valid_steps=state.valid_steps,
        ),
    )
    assert serialize_forest_state(restored, "config-sha256") == first


def test_empty_capacity_retains_exactly_one_scale() -> None:
    state = make_state(scales=5)
    state.occupied.zero_()
    state.slots.zero_()
    state.counts.zero_()
    restored = deserialize_forest_state(serialize_forest_state(state, "empty"), "empty")
    assert restored.slots.shape == (2, 3, 1, 4)
    assert restored.occupied.shape == (2, 3, 1)


def test_serialized_state_resumes_to_exact_uninterrupted_result() -> None:
    torch.manual_seed(73)
    model = forest.ScaleSharedBinaryForest(
        d_model=4, branches=2, cp_rank=3
    ).to(dtype=torch.float64)
    events = torch.randn(2, 17, 4, dtype=torch.float64)
    routes = torch.randint(-1, 3, (2, 17), dtype=torch.int64)
    valid = torch.rand(2, 17) > 0.2
    uninterrupted = model.reduce_streaming(events, routes, valid).state
    prefix = model.reduce_streaming(events[:, :8], routes[:, :8], valid[:, :8]).state
    restored = deserialize_forest_state(
        serialize_forest_state(prefix, "resume"), "resume"
    )
    resumed = model.reduce_streaming(
        events[:, 8:], routes[:, 8:], valid[:, 8:], initial_state=restored
    ).state

    torch.testing.assert_close(resumed.slots, uninterrupted.slots, rtol=0, atol=0)
    assert torch.equal(resumed.occupied, uninterrupted.occupied)
    assert torch.equal(resumed.counts, uninterrupted.counts)
    assert torch.equal(resumed.valid_steps, uninterrupted.valid_steps)


def test_device_argument_is_applied() -> None:
    restored = deserialize_forest_state(
        serialize_forest_state(make_state(), "device"), "device", device="cpu"
    )
    assert {restored.slots.device.type, restored.occupied.device.type} == {"cpu"}


@pytest.mark.parametrize("cut", [1, PREFIX.size - 1, PREFIX.size + 4])
def test_truncated_blobs_are_rejected(cut: int) -> None:
    blob = serialize_forest_state(make_state(), "truncated")
    with pytest.raises(ValueError):
        deserialize_forest_state(blob[:cut])
    with pytest.raises(ValueError):
        deserialize_forest_state(blob[:-1])


def test_corrupt_payload_and_trailing_data_are_rejected() -> None:
    blob = bytearray(serialize_forest_state(make_state(), "corrupt"))
    blob[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        deserialize_forest_state(blob)

    clean = serialize_forest_state(make_state(), "corrupt")
    with pytest.raises(ValueError, match="trailing"):
        deserialize_forest_state(clean + b"x")


def test_wrong_fingerprint_is_rejected() -> None:
    blob = serialize_forest_state(make_state(), "right")
    with pytest.raises(ValueError, match="fingerprint"):
        deserialize_forest_state(blob, "wrong")


def test_logically_inconsistent_state_is_rejected() -> None:
    state = make_state()
    state.occupied[0, 0, 1] = True
    with pytest.raises(ValueError, match="binary pattern"):
        serialize_forest_state(state, "inconsistent")


def test_missing_scale_capacity_and_nonzero_inactive_slots_are_rejected() -> None:
    insufficient = ForestState(
        slots=torch.zeros(1, 1, 1, 2),
        occupied=torch.zeros(1, 1, 1, dtype=torch.bool),
        counts=torch.tensor([[2]], dtype=torch.int64),
        valid_steps=torch.tensor([2], dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="capacity"):
        serialize_forest_state(insufficient, "insufficient")

    inactive = make_state()
    inactive.slots[0, 2, 0, 0] = 1
    with pytest.raises(ValueError, match="unoccupied"):
        serialize_forest_state(inactive, "inactive")


def test_noncanonical_or_invalid_schema_header_is_rejected() -> None:
    blob = serialize_forest_state(make_state(), "schema")
    magic, version, header_length = PREFIX.unpack_from(blob)
    header_end = PREFIX.size + header_length
    header = json.loads(blob[PREFIX.size:header_end])
    header["extra"] = True
    altered_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    altered = PREFIX.pack(magic, version, len(altered_header)) + altered_header + blob[header_end:]
    with pytest.raises(ValueError, match="schema"):
        deserialize_forest_state(altered)

    del header["extra"]
    noncanonical_header = json.dumps(header, sort_keys=True).encode()
    noncanonical = (
        PREFIX.pack(magic, version, len(noncanonical_header))
        + noncanonical_header
        + blob[header_end:]
    )
    with pytest.raises(ValueError):
        deserialize_forest_state(noncanonical)


def _replace_header(blob: bytes, mutate) -> bytes:
    magic, version, header_length = PREFIX.unpack_from(blob)
    header_end = PREFIX.size + header_length
    header = json.loads(blob[PREFIX.size:header_end])
    mutate(header)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return PREFIX.pack(magic, version, len(encoded)) + encoded + blob[header_end:]


def test_boolean_header_version_and_invalid_tensor_rank_are_rejected() -> None:
    blob = serialize_forest_state(make_state(), "strict-header")
    boolean_version = _replace_header(blob, lambda header: header.__setitem__("version", True))
    with pytest.raises(ValueError, match="version"):
        deserialize_forest_state(boolean_version)

    wrong_rank = _replace_header(
        blob, lambda header: header["tensors"]["slots"].__setitem__("shape", [0, 0])
    )
    with pytest.raises(ValueError, match="rank"):
        deserialize_forest_state(wrong_rank)


def test_routed_count_check_cannot_overflow_int64() -> None:
    maximum = torch.iinfo(torch.int64).max
    state = ForestState(
        slots=torch.zeros(1, 2, 63, 1),
        occupied=torch.ones(1, 2, 63, dtype=torch.bool),
        counts=torch.tensor([[maximum, maximum]], dtype=torch.int64),
        valid_steps=torch.tensor([maximum], dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="valid_steps"):
        serialize_forest_state(state, "overflow")


def test_magic_version_and_tensor_metadata_are_validated() -> None:
    blob = serialize_forest_state(make_state(), "metadata")
    magic, version, header_length = PREFIX.unpack_from(blob)
    header_end = PREFIX.size + header_length
    header = json.loads(blob[PREFIX.size:header_end])
    payload = blob[header_end:]

    with pytest.raises(ValueError, match="magic"):
        deserialize_forest_state(PREFIX.pack(b"badmagic", version, header_length) + blob[PREFIX.size:])
    with pytest.raises(ValueError, match="version"):
        deserialize_forest_state(PREFIX.pack(magic, version + 1, header_length) + blob[PREFIX.size:])

    header["tensors"]["slots"]["dtype"] = "float16"
    altered_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    altered = PREFIX.pack(magic, version, len(altered_header)) + altered_header + payload
    with pytest.raises(ValueError, match="dtype"):
        deserialize_forest_state(altered)


def test_tensor_payload_is_little_endian() -> None:
    state = make_state(torch.float32)
    blob = serialize_forest_state(state, "endian")
    _, _, header_length = PREFIX.unpack_from(blob)
    header_end = PREFIX.size + header_length
    header = json.loads(blob[PREFIX.size:header_end])
    slots = header["tensors"]["slots"]
    start = header_end + slots["offset"]
    # The first occupied scalar is exactly float32 1.0.
    assert blob[start : start + 4] == struct.pack("<f", 1.0)
