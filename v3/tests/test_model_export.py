from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from tnlm_v3.binding import (
    BindingArchitectureConfig,
    BindingModelConfig,
    RoutedBindingModel,
)
from tnlm_v3.data import (
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
    build_dense_selected_reference,
    model_state_fingerprint,
    select_cp_rank_by_parameter_energy,
)


CP_TENSORS = {
    "forest.merge.left.weight",
    "forest.merge.right.weight",
    "forest.merge.scale_to_rank.weight",
    "forest.merge.global_rank",
    "forest.merge.output.weight",
}


def make_model() -> RoutedBindingModel:
    torch.manual_seed(301)
    model = RoutedBindingModel(
        BindingModelConfig(
            task=BindingArchitectureConfig(
                num_surface_keys=4,
                value_cardinality=3,
                branches=2,
            ),
            d_model=5,
            cp_rank=11,
            router_hidden_dim=7,
            routing_mode=RoutingMode.LATENT,
            scale_feature_dim=3,
        )
    ).to(dtype=torch.float64)
    # Exercise exact copying of non-default requires-grad state as well as data.
    model.readout.global_type.requires_grad_(False)
    return model


def export_model(
    *, target_rank: int = 4
) -> tuple[RoutedBindingModel, RoutedBindingModel, CompactExportManifest]:
    source = make_model()
    selection = select_cp_rank_by_parameter_energy(
        source, target_rank=target_rank, calibration_fingerprint="a" * 64
    )
    compact, manifest = export_compact_binding_model(source, selection)
    return source, compact, manifest


def test_export_physically_slices_exactly_the_five_cp_tensors() -> None:
    source = make_model()
    selection = select_cp_rank_by_parameter_energy(source, target_rank=4)
    retained = torch.tensor(selection.retained_indices, dtype=torch.int64)

    compact, manifest = export_compact_binding_model(source, selection)
    source_merge = source.forest.merge
    compact_merge = compact.forest.merge

    assert compact.config is not source.config
    assert compact.config.cp_rank == selection.exported_rank == 4
    assert compact.forest.cp_rank == compact_merge.cp_rank == 4
    assert compact_merge.left.weight.shape == (4, source.config.d_model)
    assert compact_merge.right.weight.shape == (4, source.config.d_model)
    assert compact_merge.scale_to_rank.weight.shape == (
        4,
        source.config.scale_feature_dim,
    )
    assert compact_merge.global_rank.shape == (4,)
    assert compact_merge.output.weight.shape == (source.config.d_model, 4)
    assert torch.equal(
        compact_merge.left.weight,
        source_merge.left.weight.index_select(0, retained),
    )
    assert torch.equal(
        compact_merge.right.weight,
        source_merge.right.weight.index_select(0, retained),
    )
    assert torch.equal(
        compact_merge.scale_to_rank.weight,
        source_merge.scale_to_rank.weight.index_select(0, retained),
    )
    assert torch.equal(
        compact_merge.global_rank,
        source_merge.global_rank.index_select(0, retained),
    )
    assert torch.equal(
        compact_merge.output.weight,
        source_merge.output.weight.index_select(1, retained),
    )
    assert manifest.retained_indices == selection.retained_indices


def test_every_state_tensor_outside_cp_axes_is_an_exact_independent_copy() -> None:
    source, compact, _ = export_model()
    source_state = source.state_dict()
    compact_state = compact.state_dict()

    assert source_state.keys() == compact_state.keys()
    for name in source_state:
        if name not in CP_TENSORS:
            assert torch.equal(source_state[name], compact_state[name]), name
        assert source_state[name].data_ptr() != compact_state[name].data_ptr(), name

    source_parameters = dict(source.named_parameters())
    compact_parameters = dict(compact.named_parameters())
    assert source_parameters.keys() == compact_parameters.keys()
    for name in source_parameters:
        assert (
            source_parameters[name].requires_grad
            == compact_parameters[name].requires_grad
        ), name


def test_manifest_records_real_savings_but_no_state_width_saving() -> None:
    source, compact, manifest = export_model(target_rank=3)

    assert manifest.nominal_cp_rank == source.config.cp_rank == 11
    assert manifest.exported_cp_rank == compact.config.cp_rank == 3
    assert manifest.exported_parameter_count < manifest.source_parameter_count
    assert manifest.exported_raw_tensor_bytes < manifest.source_raw_tensor_bytes
    assert (
        manifest.exported_operation_count_proxy_per_merge
        < manifest.source_operation_count_proxy_per_merge
    )
    assert manifest.source_parameter_count == sum(
        parameter.numel() for parameter in source.parameters()
    )
    assert manifest.exported_parameter_count == sum(
        parameter.numel() for parameter in compact.parameters()
    )
    assert manifest.source_raw_tensor_bytes == sum(
        tensor.numel() * tensor.element_size()
        for tensor in source.state_dict().values()
    )
    assert manifest.exported_raw_tensor_bytes == sum(
        tensor.numel() * tensor.element_size()
        for tensor in compact.state_dict().values()
    )

    assert manifest.state_interface_unchanged is True
    assert manifest.source_state_d_model == manifest.exported_state_d_model == 5
    assert manifest.source_state_paths == manifest.exported_state_paths == 3
    assert (
        manifest.source_state_scalars_per_slot
        == manifest.exported_state_scalars_per_slot
        == 5
    )
    source_state = source.forest.initial_state(2)
    compact_state = compact.forest.initial_state(2)
    assert source_state.slots.shape == compact_state.slots.shape == (2, 3, 1, 5)


def test_manifest_strict_json_round_trip_and_fingerprint() -> None:
    _, _, manifest = export_model()
    payload = manifest.to_dict()

    assert type(payload["retained_indices"]) is list
    assert CompactExportManifest.from_dict(payload) == manifest
    assert json.loads(manifest.canonical_json()) == payload
    assert len(manifest.fingerprint()) == 64
    assert manifest.fingerprint() == manifest.fingerprint()

    with pytest.raises(ValueError, match="invalid compact-export manifest fields"):
        CompactExportManifest.from_dict({**payload, "unexpected": 1})
    missing = dict(payload)
    missing.pop("source_model_fingerprint")
    with pytest.raises(ValueError, match="invalid compact-export manifest fields"):
        CompactExportManifest.from_dict(missing)
    with pytest.raises(TypeError, match="JSON array"):
        CompactExportManifest.from_dict(
            {**payload, "retained_indices": tuple(payload["retained_indices"])}
        )


def test_wrong_source_fingerprint_is_rejected_before_export() -> None:
    source = make_model()
    selection = select_cp_rank_by_parameter_energy(source, target_rank=4)
    wrong = replace(selection, source_model_fingerprint="0" * 64)

    with pytest.raises(ValueError, match="source_model_fingerprint"):
        export_compact_binding_model(source, wrong)


def test_export_has_no_mask_buffer_or_original_rank_cp_tensor() -> None:
    source, compact, _ = export_model()
    nominal_rank = source.config.cp_rank
    merge_state = compact.forest.merge.state_dict()

    assert list(compact.forest.merge.named_buffers()) == []
    assert all(
        word not in name.lower()
        for name in merge_state
        for word in ("mask", "retained", "original", "nominal")
    )
    assert all(nominal_rank not in tensor.shape for tensor in merge_state.values())
    assert compact.config.cp_rank != nominal_rank


@pytest.mark.parametrize("training", [False, True])
def test_source_is_unchanged_and_mode_is_preserved(training: bool) -> None:
    source = make_model()
    source.train(training)
    selection = select_cp_rank_by_parameter_energy(source, target_rank=4)
    before_fingerprint = model_state_fingerprint(source)
    before_state = {
        name: tensor.detach().clone() for name, tensor in source.state_dict().items()
    }

    compact, manifest = export_compact_binding_model(source, selection)

    assert source.config.cp_rank == 11
    assert source.forest.cp_rank == source.forest.merge.cp_rank == 11
    assert model_state_fingerprint(source) == before_fingerprint
    assert all(
        torch.equal(before_state[name], tensor)
        for name, tensor in source.state_dict().items()
    )
    assert source.training is training
    assert compact.training is training
    assert compact.forest.merge.training is source.forest.merge.training
    assert manifest.source_model_fingerprint == before_fingerprint
    assert manifest.exported_model_fingerprint == model_state_fingerprint(compact)


def test_compact_merge_matches_dense_selected_reference() -> None:
    source = make_model().eval()
    selection = select_cp_rank_by_parameter_energy(source, target_rank=4)
    dense = build_dense_selected_reference(source, selection)
    compact, _ = export_compact_binding_model(source, selection)
    generator = torch.Generator().manual_seed(91)
    left = torch.randn(6, source.config.d_model, generator=generator, dtype=torch.float64)
    right = torch.randn(6, source.config.d_model, generator=generator, dtype=torch.float64)
    scales = torch.arange(6, dtype=torch.float64)
    global_path = torch.tensor([False, True, False, True, False, True])

    expected = dense.forest.merge(left, right, scales, global_path)
    actual = compact.forest.merge(left, right, scales, global_path)

    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("implementation", ["streaming", "parallel"])
def test_complete_compact_model_matches_dense_selected_reference(
    implementation: str,
) -> None:
    source = make_model().eval()
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
    batch = collate_binding_episodes(
        generate_binding_episodes(
            task,
            count=2,
            seed=431,
            split="eval",
            lengths=[17, 18],
        )
    )
    selection = select_cp_rank_by_parameter_energy(source, target_rank=4)
    dense = build_dense_selected_reference(source, selection).eval()
    compact, _ = export_compact_binding_model(source, selection)
    compact.eval()

    expected = dense(batch.inputs, implementation=implementation)
    actual = compact(batch.inputs, implementation=implementation)

    assert torch.equal(actual.routes, expected.routes)
    torch.testing.assert_close(actual.route_logits, expected.route_logits, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.route_probabilities,
        expected.route_probabilities,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(actual.value_logits, expected.value_logits, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        actual.forest_state.slots,
        expected.forest_state.slots,
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.equal(actual.forest_state.occupied, expected.forest_state.occupied)
    assert torch.equal(actual.forest_state.counts, expected.forest_state.counts)
    assert torch.equal(
        actual.forest_state.valid_steps, expected.forest_state.valid_steps
    )
