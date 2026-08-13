from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import struct

import pytest
import torch
from torch import nn

from tnlm_v3.binding import (
    BindingArchitectureConfig,
    BindingModelConfig,
    BindingModelOutput,
    RoutedBindingModel,
)
from tnlm_v3.compact_artifact import (
    deserialize_compact_binding_model,
    serialize_compact_binding_model,
)
from tnlm_v3.data import (
    BindingModelInputs,
    BindingTaskConfig,
    collate_binding_episodes,
    generate_binding_episodes,
)
from tnlm_v3.model_export import (
    CompactExportManifest,
    export_compact_binding_model,
)
from tnlm_v3.routing import RoutingMode
from tnlm_v3.truncation import (
    CPRankSelection,
    model_state_fingerprint,
    select_cp_rank_by_parameter_energy,
)


_MAGIC = b"TNLM3CM\x00"
_VERSION = 1
_PREFIX = struct.Struct("<8sIQQ32s32s")
_MAX_HEADER_BYTES = 1 << 20
_MAX_PAYLOAD_BYTES = 62 * 1024 * 1024


def _make_export() -> tuple[
    RoutedBindingModel, CompactExportManifest, CPRankSelection
]:
    torch.manual_seed(771)
    source = RoutedBindingModel(
        BindingModelConfig(
            task=BindingArchitectureConfig(
                num_surface_keys=4,
                value_cardinality=3,
                branches=2,
            ),
            d_model=6,
            cp_rank=9,
            router_hidden_dim=7,
            routing_mode=RoutingMode.LATENT,
            curriculum_seed=37,
            scale_feature_dim=4,
            straight_through_route_surrogate=False,
        )
    ).to(dtype=torch.float64)
    selection = select_cp_rank_by_parameter_energy(
        source,
        target_rank=4,
        calibration_fingerprint="a" * 64,
    )
    compact, manifest = export_compact_binding_model(source, selection)

    # Module mode is per-module state, not merely the root's train/eval bit.
    compact.eval()
    compact.encoder.train()
    compact.encoder.event_projection.eval()
    compact.router.train()
    compact.router.branch_scorer.eval()
    compact.forest.train()
    compact.forest.merge.eval()
    compact.readout.train()

    for index, parameter in enumerate(compact.parameters()):
        parameter.requires_grad_(index % 3 != 1)
    return compact, manifest, selection


def _make_inputs() -> BindingModelInputs:
    task = BindingTaskConfig(
        num_surface_keys=4,
        value_cardinality=3,
        branches=2,
        max_live_bindings=2,
        min_length=10,
        max_length=18,
        heldout_key_value_pairs=((0, 0),),
        global_distractor_probability=0.5,
    )
    return collate_binding_episodes(
        generate_binding_episodes(
            task,
            count=3,
            seed=883,
            split="eval",
            lengths=[15, 17, 18],
        )
    ).inputs


def _assert_output_bit_exact(
    actual: BindingModelOutput, expected: BindingModelOutput
) -> None:
    for name in ("value_logits", "routes", "route_logits", "route_probabilities"):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name
    for name in ("slots", "occupied", "counts", "valid_steps"):
        assert torch.equal(
            getattr(actual.forest_state, name), getattr(expected.forest_state, name)
        ), f"forest_state.{name}"
    for name in (
        "prototypes",
        "occupied",
        "ages",
        "loads",
        "global_state",
        "global_occupied",
        "global_load",
        "valid_steps",
    ):
        assert torch.equal(
            getattr(actual.router_state, name), getattr(expected.router_state, name)
        ), f"router_state.{name}"
    assert actual.diagnostics.keys() == expected.diagnostics.keys()
    for name in actual.diagnostics:
        assert torch.equal(actual.diagnostics[name], expected.diagnostics[name]), name


def _parts(blob: bytes) -> tuple[dict[str, object], bytes, bytes]:
    magic, version, header_length, payload_length, _, _ = _PREFIX.unpack_from(blob)
    assert magic == _MAGIC
    assert version == _VERSION
    start = _PREFIX.size
    header_bytes = blob[start : start + header_length]
    payload = blob[start + header_length :]
    assert len(payload) == payload_length
    return json.loads(header_bytes), header_bytes, payload


def _pack(
    header: dict[str, object],
    payload: bytes,
    *,
    header_bytes: bytes | None = None,
    magic: bytes = _MAGIC,
    version: int = _VERSION,
    header_hash: bytes | None = None,
    payload_hash: bytes | None = None,
    declared_header_length: int | None = None,
    declared_payload_length: int | None = None,
) -> bytes:
    if header_bytes is None:
        header_bytes = json.dumps(
            header,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    prefix = _PREFIX.pack(
        magic,
        version,
        len(header_bytes)
        if declared_header_length is None
        else declared_header_length,
        len(payload)
        if declared_payload_length is None
        else declared_payload_length,
        hashlib.sha256(header_bytes).digest()
        if header_hash is None
        else header_hash,
        hashlib.sha256(payload).digest()
        if payload_hash is None
        else payload_hash,
    )
    return prefix + header_bytes + payload


def _changed_header(blob: bytes, mutate) -> bytes:
    header, _, payload = _parts(blob)
    mutate(header)
    return _pack(header, payload)


@pytest.fixture()
def exported() -> tuple[
    RoutedBindingModel, CompactExportManifest, CPRankSelection, bytes
]:
    model, manifest, selection = _make_export()
    return model, manifest, selection, serialize_compact_binding_model(
        model, manifest, selection
    )


def test_serialization_is_canonical_and_deterministic() -> None:
    model, manifest, selection = _make_export()

    first = serialize_compact_binding_model(model, manifest, selection)
    second = serialize_compact_binding_model(model, manifest, selection)
    loaded, loaded_manifest, loaded_selection = deserialize_compact_binding_model(first)
    third = serialize_compact_binding_model(
        loaded, loaded_manifest, loaded_selection
    )

    assert first == second == third
    header, header_bytes, _ = _parts(first)
    assert header_bytes == json.dumps(
        header,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert [record["name"] for record in header["tensors"]] == sorted(
        record["name"] for record in header["tensors"]
    )


def test_round_trip_is_bit_exact_and_preserves_all_non_tensor_state() -> None:
    model, manifest, selection = _make_export()
    inputs = _make_inputs()
    original_modes = {name: module.training for name, module in model.named_modules()}
    original_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    original_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }

    loaded, loaded_manifest, loaded_selection = deserialize_compact_binding_model(
        serialize_compact_binding_model(model, manifest, selection),
        expected_source_fingerprint=selection.source_model_fingerprint,
        device=torch.device("cpu"),
    )

    assert loaded_manifest == manifest
    assert loaded_selection == selection
    assert loaded.config == model.config
    assert model_state_fingerprint(loaded) == model_state_fingerprint(model)
    assert {name: module.training for name, module in loaded.named_modules()} == (
        original_modes
    )
    assert {
        name: parameter.requires_grad
        for name, parameter in loaded.named_parameters()
    } == original_grad
    assert loaded.state_dict().keys() == original_state.keys()
    for name, tensor in loaded.state_dict().items():
        assert tensor.device.type == "cpu"
        assert tensor.dtype == original_state[name].dtype
        assert torch.equal(tensor, original_state[name]), name

    with torch.no_grad():
        for implementation in ("streaming", "parallel"):
            expected = model(inputs, implementation=implementation)
            actual = loaded(inputs, implementation=implementation)
            _assert_output_bit_exact(actual, expected)


def test_serialize_and_deserialize_preserve_cpu_rng_state() -> None:
    model, manifest, selection = _make_export()
    torch.manual_seed(9103)
    before_serialize = torch.random.get_rng_state().clone()
    blob = serialize_compact_binding_model(model, manifest, selection)
    assert torch.equal(torch.random.get_rng_state(), before_serialize)

    torch.manual_seed(9104)
    before_deserialize = torch.random.get_rng_state().clone()
    deserialize_compact_binding_model(blob)
    assert torch.equal(torch.random.get_rng_state(), before_deserialize)


def test_bytes_bytearray_and_memoryview_are_all_accepted(exported) -> None:
    _, _, _, blob = exported
    for value in (blob, bytearray(blob), memoryview(blob)):
        loaded, _, _ = deserialize_compact_binding_model(value)
        assert loaded.config.cp_rank == 4


def test_trusted_provenance_fingerprints_are_enforced(exported) -> None:
    _, manifest, selection, blob = exported
    deserialize_compact_binding_model(
        blob,
        expected_source_fingerprint=selection.source_model_fingerprint,
        expected_manifest_fingerprint=manifest.fingerprint(),
        expected_selection_fingerprint=selection.fingerprint(),
    )
    with pytest.raises(ValueError, match="source fingerprint"):
        deserialize_compact_binding_model(
            blob, expected_source_fingerprint="0" * 64
        )
    for invalid in (7, True, "A" * 64, "0" * 63):
        with pytest.raises((TypeError, ValueError)):
            deserialize_compact_binding_model(
                blob, expected_source_fingerprint=invalid  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="manifest fingerprint"):
        deserialize_compact_binding_model(
            blob, expected_manifest_fingerprint="0" * 64
        )
    with pytest.raises(ValueError, match="selection fingerprint"):
        deserialize_compact_binding_model(
            blob, expected_selection_fingerprint="0" * 64
        )
    for invalid in (7, True, "A" * 64, "0" * 63):
        with pytest.raises((TypeError, ValueError)):
            deserialize_compact_binding_model(
                blob,
                expected_manifest_fingerprint=invalid,  # type: ignore[arg-type]
            )
        with pytest.raises((TypeError, ValueError)):
            deserialize_compact_binding_model(
                blob,
                expected_selection_fingerprint=invalid,  # type: ignore[arg-type]
            )


@pytest.mark.parametrize("cut", [0, 1, _PREFIX.size - 1])
def test_truncated_prefix_is_rejected(exported, cut: int) -> None:
    _, _, _, blob = exported
    with pytest.raises(ValueError, match="truncated"):
        deserialize_compact_binding_model(blob[:cut])


def test_truncated_payload_and_trailing_bytes_are_rejected(exported) -> None:
    _, _, _, blob = exported
    with pytest.raises(ValueError, match="truncated or has trailing bytes"):
        deserialize_compact_binding_model(blob[:-1])
    with pytest.raises(ValueError, match="truncated or has trailing bytes"):
        deserialize_compact_binding_model(blob + b"\x00")


def test_bad_magic_and_prefix_version_are_rejected(exported) -> None:
    _, _, _, blob = exported
    header, _, payload = _parts(blob)
    with pytest.raises(ValueError, match="magic"):
        deserialize_compact_binding_model(
            _pack(header, payload, magic=b"NOTTNLM\x00")
        )
    with pytest.raises(ValueError, match="version"):
        deserialize_compact_binding_model(_pack(header, payload, version=2))


def test_header_schema_version_is_independently_enforced(exported) -> None:
    _, _, _, blob = exported
    forged = _changed_header(blob, lambda header: header.__setitem__("schema_version", 2))
    with pytest.raises(ValueError, match="schema_version"):
        deserialize_compact_binding_model(forged)


def test_header_and_payload_checksums_are_enforced(exported) -> None:
    _, _, _, blob = exported
    header, header_bytes, payload = _parts(blob)
    with pytest.raises(ValueError, match="header checksum"):
        deserialize_compact_binding_model(
            _pack(header, payload, header_hash=b"\x00" * 32)
        )
    with pytest.raises(ValueError, match="payload checksum"):
        deserialize_compact_binding_model(
            _pack(header, payload, payload_hash=b"\x00" * 32)
        )

    corrupt_header = bytearray(blob)
    corrupt_header[_PREFIX.size + len(header_bytes) // 2] ^= 1
    with pytest.raises(ValueError, match="header checksum"):
        deserialize_compact_binding_model(corrupt_header)

    corrupt_payload = bytearray(blob)
    corrupt_payload[-1] ^= 1
    with pytest.raises(ValueError, match="payload checksum"):
        deserialize_compact_binding_model(corrupt_payload)


def test_duplicate_json_keys_are_rejected_even_with_a_valid_header_hash(exported) -> None:
    _, _, _, blob = exported
    header, header_bytes, payload = _parts(blob)
    duplicate = (
        b'{"artifact_kind":"tnlm_v3.compact_binding_model",' + header_bytes[1:]
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        deserialize_compact_binding_model(
            _pack(header, payload, header_bytes=duplicate)
        )


def test_noncanonical_json_is_rejected_even_with_a_valid_header_hash(exported) -> None:
    _, _, _, blob = exported
    header, _, payload = _parts(blob)
    noncanonical = json.dumps(header, sort_keys=True, indent=2).encode("utf-8")
    with pytest.raises(ValueError, match="not canonical JSON"):
        deserialize_compact_binding_model(
            _pack(header, payload, header_bytes=noncanonical)
        )


def test_missing_and_extra_header_fields_are_rejected(exported) -> None:
    _, _, _, blob = exported
    missing = _changed_header(blob, lambda header: header.pop("manifest"))
    extra = _changed_header(blob, lambda header: header.__setitem__("code", "run me"))
    for forged in (missing, extra):
        with pytest.raises(ValueError, match="invalid artifact header fields"):
            deserialize_compact_binding_model(forged)


def test_forged_artifact_kind_and_exported_fingerprint_are_rejected(exported) -> None:
    _, _, _, blob = exported
    kind = _changed_header(
        blob, lambda header: header.__setitem__("artifact_kind", "forged")
    )
    fingerprint = _changed_header(
        blob,
        lambda header: header.__setitem__("exported_model_fingerprint", "0" * 64),
    )
    with pytest.raises(ValueError, match="kind"):
        deserialize_compact_binding_model(kind)
    with pytest.raises(ValueError, match="exported model fingerprint"):
        deserialize_compact_binding_model(fingerprint)


def test_forged_manifest_is_rejected(exported) -> None:
    _, _, _, blob = exported

    def mutate(header) -> None:
        header["manifest"]["exported_parameter_count"] += 1

    forged = _changed_header(blob, mutate)
    with pytest.raises(ValueError, match="manifest exported parameters"):
        deserialize_compact_binding_model(forged)


def test_forged_selection_is_rejected(exported) -> None:
    _, _, _, blob = exported

    def mutate(header) -> None:
        header["selection"]["calibration_fingerprint"] = "b" * 64

    forged = _changed_header(blob, mutate)
    with pytest.raises(ValueError, match="manifest selection fingerprint"):
        deserialize_compact_binding_model(forged)


def test_forged_model_config_is_rejected(exported) -> None:
    _, _, _, blob = exported

    def mutate(header) -> None:
        header["model_config"]["router_hidden_dim"] += 1

    forged = _changed_header(blob, mutate)
    with pytest.raises(ValueError, match="architecture-invalid shape"):
        deserialize_compact_binding_model(forged)


def test_forged_tensor_name_is_rejected(exported) -> None:
    _, _, _, blob = exported

    def mutate(header) -> None:
        header["tensors"][0]["name"] = "!forged"

    with pytest.raises(ValueError, match="unexpected tensor|tensor keys mismatch"):
        deserialize_compact_binding_model(_changed_header(blob, mutate))


def test_duplicate_and_unsorted_tensor_names_are_rejected(exported) -> None:
    _, _, _, blob = exported

    def duplicate(header) -> None:
        header["tensors"][1]["name"] = header["tensors"][0]["name"]

    def unsorted(header) -> None:
        header["tensors"][0], header["tensors"][1] = (
            header["tensors"][1],
            header["tensors"][0],
        )

    for mutate in (duplicate, unsorted):
        with pytest.raises(ValueError, match="unique sorted names|contiguous"):
            deserialize_compact_binding_model(_changed_header(blob, mutate))


def test_forged_tensor_dtype_is_rejected(exported) -> None:
    _, _, _, blob = exported

    def mutate(header) -> None:
        record = header["tensors"][0]
        record["dtype"] = "float32" if record["dtype"] != "float32" else "float64"

    with pytest.raises(ValueError, match="byte count does not match"):
        deserialize_compact_binding_model(_changed_header(blob, mutate))


def test_forged_tensor_shape_is_rejected(exported) -> None:
    _, _, _, blob = exported

    def mutate(header) -> None:
        for record in header["tensors"]:
            shape = record["shape"]
            if len(shape) == 2 and shape[0] != shape[1]:
                record["shape"] = list(reversed(shape))
                return
        raise AssertionError("fixture lacks a non-square rank-two tensor")

    with pytest.raises(ValueError, match="architecture-invalid shape"):
        deserialize_compact_binding_model(_changed_header(blob, mutate))


def test_forged_tensor_offset_is_rejected(exported) -> None:
    _, _, _, blob = exported

    def mutate(header) -> None:
        header["tensors"][1]["offset"] += 1

    with pytest.raises(ValueError, match="contiguous and in bounds"):
        deserialize_compact_binding_model(_changed_header(blob, mutate))


def test_forged_tensor_nbytes_is_rejected(exported) -> None:
    _, _, _, blob = exported

    def mutate(header) -> None:
        header["tensors"][0]["nbytes"] += 1

    with pytest.raises(ValueError, match="byte count does not match"):
        deserialize_compact_binding_model(_changed_header(blob, mutate))


def test_forged_per_tensor_checksum_is_rejected(exported) -> None:
    _, _, _, blob = exported

    def mutate(header) -> None:
        header["tensors"][0]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="tensor .* checksum mismatch"):
        deserialize_compact_binding_model(_changed_header(blob, mutate))


def test_inner_tensor_checksum_detects_payload_rehashed_at_outer_layer(exported) -> None:
    _, _, _, blob = exported
    header, _, payload = _parts(blob)
    corrupt = bytearray(payload)
    corrupt[0] ^= 1
    with pytest.raises(ValueError, match="tensor .* checksum mismatch"):
        deserialize_compact_binding_model(_pack(header, bytes(corrupt)))


def test_tensor_record_missing_or_extra_fields_are_rejected(exported) -> None:
    _, _, _, blob = exported

    def missing(header) -> None:
        header["tensors"][0].pop("sha256")

    def extra(header) -> None:
        header["tensors"][0]["executable"] = True

    for mutate in (missing, extra):
        with pytest.raises(ValueError, match="invalid tensor record 0 fields"):
            deserialize_compact_binding_model(_changed_header(blob, mutate))


def test_forged_module_modes_and_requires_grad_maps_are_rejected(exported) -> None:
    _, _, _, blob = exported

    def missing_mode(header) -> None:
        header["module_training"].pop(next(iter(header["module_training"])))

    def extra_grad(header) -> None:
        header["parameter_requires_grad"]["forged.parameter"] = False

    def nonboolean_mode(header) -> None:
        header["module_training"][next(iter(header["module_training"]))] = 1

    for mutate, match in (
        (missing_mode, "module_training keys"),
        (extra_grad, "parameter_requires_grad keys"),
        (nonboolean_mode, "must be boolean"),
    ):
        with pytest.raises((TypeError, ValueError), match=match):
            deserialize_compact_binding_model(_changed_header(blob, mutate))


def test_declared_oversized_header_and_payload_are_rejected_before_allocation(
    exported,
) -> None:
    _, _, _, blob = exported
    header, _, payload = _parts(blob)
    oversized_header = _pack(
        header,
        b"",
        declared_header_length=_MAX_HEADER_BYTES + 1,
        declared_payload_length=0,
    )
    oversized_payload = _pack(
        header,
        b"",
        declared_header_length=0,
        declared_payload_length=_MAX_PAYLOAD_BYTES + 1,
    )
    with pytest.raises(ValueError, match="header exceeds the safety limit"):
        deserialize_compact_binding_model(oversized_header)
    with pytest.raises(ValueError, match="payload exceeds the safety limit"):
        deserialize_compact_binding_model(oversized_payload)


@pytest.mark.parametrize("blob", [None, 1, "artifact", object(), [1, 2]])
def test_invalid_blob_types_are_rejected(blob) -> None:
    with pytest.raises(TypeError, match="bytes-like"):
        deserialize_compact_binding_model(blob)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "device",
    [1, object(), "cpu:0", "meta", "not-a-device"],
)
def test_invalid_devices_are_rejected(exported, device) -> None:
    _, _, _, blob = exported
    with pytest.raises((TypeError, ValueError), match="device|Device"):
        deserialize_compact_binding_model(blob, device=device)


def test_serializer_rejects_wrong_object_types() -> None:
    model, manifest, selection = _make_export()
    with pytest.raises(TypeError, match="model"):
        serialize_compact_binding_model(object(), manifest, selection)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="manifest"):
        serialize_compact_binding_model(model, object(), selection)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="selection"):
        serialize_compact_binding_model(model, manifest, object())  # type: ignore[arg-type]


def test_serializer_rejects_manifest_or_selection_from_another_export() -> None:
    model, manifest, selection = _make_export()
    other_model, other_manifest, other_selection = _make_export()
    with torch.no_grad():
        other_model.readout.output.weight.add_(1.0)

    # The second helper begins from the same seed, so explicitly make its
    # relationship records foreign by taking a fresh source/export seed.
    torch.manual_seed(772)
    foreign_source = RoutedBindingModel(
        BindingModelConfig(
            task=BindingArchitectureConfig(4, 3, 2),
            d_model=6,
            cp_rank=9,
            router_hidden_dim=7,
            routing_mode=RoutingMode.LATENT,
            curriculum_seed=37,
            scale_feature_dim=4,
            straight_through_route_surrogate=False,
        )
    ).double()
    foreign_selection = select_cp_rank_by_parameter_energy(
        foreign_source, target_rank=4, calibration_fingerprint="a" * 64
    )
    _, foreign_manifest = export_compact_binding_model(
        foreign_source, foreign_selection
    )

    with pytest.raises(ValueError, match="relationship mismatch"):
        serialize_compact_binding_model(model, foreign_manifest, selection)
    with pytest.raises(ValueError, match="relationship mismatch"):
        serialize_compact_binding_model(model, manifest, foreign_selection)


def test_serializer_rejects_forged_selection_scores_and_indices() -> None:
    model, manifest, selection = _make_export()
    forged_scores = replace(
        selection,
        channel_scores=(selection.channel_scores[0] + 1.0,)
        + selection.channel_scores[1:],
    )
    swapped = list(selection.retained_indices)
    discarded = next(
        index
        for index in range(selection.nominal_rank)
        if index not in selection.retained_indices
    )
    swapped[-1] = discarded
    forged_indices = replace(selection, retained_indices=tuple(sorted(swapped)))

    with pytest.raises(ValueError, match="manifest selection fingerprint"):
        serialize_compact_binding_model(model, manifest, forged_scores)
    with pytest.raises(ValueError, match="manifest selection fingerprint|retained indices"):
        serialize_compact_binding_model(model, manifest, forged_indices)


def test_loader_rejects_forged_selection_scores_and_indices(exported) -> None:
    _, _, _, blob = exported

    def scores(header) -> None:
        header["selection"]["channel_scores"][0] += 1.0

    def indices(header) -> None:
        retained = header["selection"]["retained_indices"]
        nominal = header["selection"]["nominal_rank"]
        replacement = next(index for index in range(nominal) if index not in retained)
        retained[-1] = replacement
        retained.sort()

    for mutate in (scores, indices):
        with pytest.raises(ValueError, match="relationship mismatch"):
            deserialize_compact_binding_model(_changed_header(blob, mutate))


def test_serializer_rejects_mixed_parameter_dtypes() -> None:
    model, manifest, selection = _make_export()
    _, parameter = next(iter(model.named_parameters()))
    assert parameter.dtype is torch.float64
    parameter.data = parameter.data.float()

    with pytest.raises((TypeError, ValueError), match="dtype|floating"):
        serialize_compact_binding_model(model, manifest, selection)


def test_serializer_rejects_tied_parameter_aliases() -> None:
    model, manifest, selection = _make_export()
    assert (
        model.encoder.primary_embedding.weight.shape
        == model.encoder.secondary_embedding.weight.shape
    )
    model.encoder.secondary_embedding.weight = model.encoder.primary_embedding.weight
    tied_manifest = replace(
        manifest,
        exported_model_fingerprint=model_state_fingerprint(model),
        exported_parameter_count=sum(
            parameter.numel() for parameter in model.parameters()
        ),
    )

    with pytest.raises((TypeError, ValueError), match="alias|tied|shared"):
        serialize_compact_binding_model(model, tied_manifest, selection)


def test_serializer_rejects_distinct_parameters_with_shared_storage() -> None:
    model, manifest, selection = _make_export()
    primary = model.encoder.primary_embedding.weight
    model.encoder.secondary_embedding.weight = nn.Parameter(primary.data)
    shared_manifest = replace(
        manifest,
        exported_model_fingerprint=model_state_fingerprint(model),
    )
    with pytest.raises(ValueError, match="shared tensor storage"):
        serialize_compact_binding_model(model, shared_manifest, selection)


def test_serializer_rejects_non_boolean_module_training_state() -> None:
    model, manifest, selection = _make_export()
    model.encoder.training = 1  # type: ignore[assignment]

    with pytest.raises((TypeError, ValueError), match="training|boolean"):
        serialize_compact_binding_model(model, manifest, selection)


def test_serializer_applies_the_same_config_caps_as_the_loader() -> None:
    source = RoutedBindingModel(
        BindingModelConfig(
            task=BindingArchitectureConfig(4097, 2, 2),
            d_model=2,
            cp_rank=2,
            router_hidden_dim=2,
            routing_mode=RoutingMode.LATENT,
            scale_feature_dim=1,
        )
    )
    selection = select_cp_rank_by_parameter_energy(source, target_rank=1)
    model, manifest = export_compact_binding_model(source, selection)

    with pytest.raises(ValueError, match="num_surface_keys must be at most 4096"):
        serialize_compact_binding_model(model, manifest, selection)
