from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from tnlm_v3.binding import (
    BindingArchitectureConfig,
    BindingModelConfig,
    RoutedBindingModel,
)
from tnlm_v3.routing import RoutingMode
from tnlm_v3.truncation import (
    CPRankSelection,
    build_dense_selected_reference,
    model_state_fingerprint,
    select_cp_rank_by_parameter_energy,
)


def make_model(rank: int = 4) -> RoutedBindingModel:
    torch.manual_seed(813)
    return RoutedBindingModel(
        BindingModelConfig(
            task=BindingArchitectureConfig(
                num_surface_keys=5,
                value_cardinality=4,
                branches=3,
            ),
            d_model=6,
            cp_rank=rank,
            router_hidden_dim=7,
            routing_mode=RoutingMode.LATENT,
            scale_feature_dim=3,
        )
    ).double()


def valid_selection(**changes) -> CPRankSelection:
    values = dict(
        schema_version=1,
        source_model_fingerprint="a" * 64,
        method="parameter_energy_v1",
        nominal_rank=4,
        exported_rank=2,
        retained_indices=(0, 2),
        channel_scores=(4.0, 3.0, 2.0, 1.0),
        calibration_fingerprint=None,
    )
    values.update(changes)
    return CPRankSelection(**values)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"schema_version": True}, ValueError),
        ({"schema_version": 2}, ValueError),
        ({"source_model_fingerprint": "A" * 64}, ValueError),
        ({"method": "energy"}, ValueError),
        ({"nominal_rank": True}, TypeError),
        ({"exported_rank": 4}, ValueError),
        ({"retained_indices": [0, 2]}, TypeError),
        ({"retained_indices": (2, 0)}, ValueError),
        ({"retained_indices": (0, 0)}, ValueError),
        ({"retained_indices": (0, 4)}, ValueError),
        ({"channel_scores": [4.0, 3.0, 2.0, 1.0]}, TypeError),
        ({"channel_scores": (4.0, 3.0, float("inf"), 1.0)}, ValueError),
        ({"channel_scores": (4.0, 3.0, -1.0, 1.0)}, ValueError),
        ({"calibration_fingerprint": "short"}, ValueError),
    ],
)
def test_selection_schema_is_strict(changes, error) -> None:
    with pytest.raises(error):
        valid_selection(**changes)


def test_parameter_energy_is_exact_and_ties_use_lower_index() -> None:
    model = make_model()
    merge = model.forest.merge
    with torch.no_grad():
        merge.left.weight.fill_(1)
        merge.right.weight.fill_(1)
        merge.output.weight.fill_(1)
        merge.scale_to_rank.weight.zero_()
        merge.global_rank.zero_()
        # Channels 0 and 1 tie. Channel 2 has four times their output energy;
        # channel 3 has five times their augmented conditioning energy.
        merge.output.weight[:, 2].fill_(2)
        merge.global_rank[3] = 2

    selection = select_cp_rank_by_parameter_energy(model, target_rank=3)
    base = 6.0 * 6.0 * 6.0
    assert selection.channel_scores == pytest.approx(
        (base, base, 4.0 * base, 5.0 * base), rel=0, abs=0
    )
    assert selection.retained_indices == (0, 2, 3)
    assert selection.exported_rank == 3


def test_zero_energy_ties_are_deterministic_and_repeat_exactly() -> None:
    model = make_model()
    merge = model.forest.merge
    with torch.no_grad():
        merge.left.weight.zero_()
        merge.right.weight.zero_()
        merge.scale_to_rank.weight.zero_()
        merge.global_rank.zero_()
        merge.output.weight.zero_()

    first = select_cp_rank_by_parameter_energy(model, target_rank=2)
    second = select_cp_rank_by_parameter_energy(model, target_rank=2)
    assert first == second
    assert first.fingerprint() == second.fingerprint()
    assert first.retained_indices == (0, 1)
    assert first.channel_scores == (0.0, 0.0, 0.0, 0.0)


def test_model_fingerprint_is_exact_repeatable_and_state_sensitive() -> None:
    model = make_model()
    model.register_buffer("fingerprint_scalar", torch.tensor(1.0))
    clone = copy.deepcopy(model)
    first = model_state_fingerprint(model)
    assert first == model_state_fingerprint(model)
    assert first == model_state_fingerprint(clone)
    assert len(first) == 64

    with torch.no_grad():
        clone.encoder.event_projection.bias[0].add_(1)
    assert model_state_fingerprint(clone) != first


def test_model_fingerprint_rejects_live_architecture_drift() -> None:
    model = make_model()
    model.router.mode = RoutingMode.ORACLE
    with pytest.raises(ValueError, match="architecture"):
        model_state_fingerprint(model)

    model = make_model()
    model.router.temperature = 0.5
    with pytest.raises(ValueError, match="architecture"):
        model_state_fingerprint(model)

    model = make_model()
    model.forest.merge.scale_feature_dim += 1
    with pytest.raises(ValueError, match="architecture"):
        model_state_fingerprint(model)


def test_dense_reference_rejects_a_selection_for_another_model() -> None:
    model = make_model()
    selection = select_cp_rank_by_parameter_energy(model, target_rank=2)
    with torch.no_grad():
        model.readout.output.bias[0].add_(1)
    with pytest.raises(ValueError, match="fingerprint"):
        build_dense_selected_reference(model, selection)


def test_dense_reference_rejects_forged_scores_and_retained_indices() -> None:
    model = make_model()
    selection = select_cp_rank_by_parameter_energy(model, target_rank=2)
    forged_scores = replace(
        selection,
        channel_scores=tuple(0.0 for _ in range(selection.nominal_rank)),
    )
    with pytest.raises(ValueError, match="scores"):
        build_dense_selected_reference(model, forged_scores)

    alternative = next(
        index
        for index in range(selection.nominal_rank)
        if index not in selection.retained_indices
    )
    forged_indices = replace(
        selection,
        retained_indices=tuple(sorted((selection.retained_indices[0], alternative))),
    )
    with pytest.raises(ValueError, match="retained indices"):
        build_dense_selected_reference(model, forged_indices)


def test_dense_reference_zeros_only_discarded_cp_tensors_without_mutation() -> None:
    model = make_model()
    source_before = {name: value.clone() for name, value in model.state_dict().items()}
    selection = select_cp_rank_by_parameter_energy(model, target_rank=2)
    reference = build_dense_selected_reference(model, selection)
    reference_again = build_dense_selected_reference(model, selection)

    for name, value in model.state_dict().items():
        assert torch.equal(value, source_before[name])
    for name, value in reference.state_dict().items():
        assert torch.equal(value, reference_again.state_dict()[name])

    discarded = sorted(set(range(selection.nominal_rank)) - set(selection.retained_indices))
    kept = list(selection.retained_indices)
    source_merge = model.forest.merge
    selected_merge = reference.forest.merge
    for attribute in ("left", "right", "scale_to_rank"):
        source_weight = getattr(source_merge, attribute).weight
        selected_weight = getattr(selected_merge, attribute).weight
        assert torch.equal(selected_weight[kept], source_weight[kept])
        assert not bool(selected_weight[discarded].any())
    assert torch.equal(
        selected_merge.output.weight[:, kept], source_merge.output.weight[:, kept]
    )
    assert not bool(selected_merge.output.weight[:, discarded].any())
    assert torch.equal(selected_merge.global_rank[kept], source_merge.global_rank[kept])
    assert not bool(selected_merge.global_rank[discarded].any())

    cp_names = {
        "forest.merge.left.weight",
        "forest.merge.right.weight",
        "forest.merge.scale_to_rank.weight",
        "forest.merge.global_rank",
        "forest.merge.output.weight",
    }
    for name, source_value in source_before.items():
        if name not in cp_names:
            assert torch.equal(reference.state_dict()[name], source_value)


def test_discarded_parameter_perturbations_cannot_change_selected_reference() -> None:
    source = make_model()
    selection = select_cp_rank_by_parameter_energy(source, target_rank=2)
    changed = copy.deepcopy(source)
    discarded = sorted(set(range(selection.nominal_rank)) - set(selection.retained_indices))
    merge = changed.forest.merge
    with torch.no_grad():
        merge.left.weight[discarded].normal_(mean=100, std=20)
        merge.right.weight[discarded].normal_(mean=-100, std=20)
        merge.scale_to_rank.weight[discarded].fill_(500)
        merge.global_rank[discarded].fill_(-700)
        merge.output.weight[:, discarded].normal_(mean=900, std=30)

    changed_selection = select_cp_rank_by_parameter_energy(
        changed,
        target_rank=selection.exported_rank,
    )
    assert changed_selection.retained_indices == selection.retained_indices
    first = build_dense_selected_reference(source, selection)
    second = build_dense_selected_reference(changed, changed_selection)
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])

    torch.manual_seed(91)
    left = torch.randn(11, source.config.d_model, dtype=torch.float64)
    right = torch.randn(11, source.config.d_model, dtype=torch.float64)
    scales = torch.arange(11, dtype=torch.float64)
    global_path = torch.arange(11).remainder(2).bool()
    actual = first.forest.merge(left, right, scales, global_path)
    repeated = second.forest.merge(left, right, scales, global_path)
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)


def test_target_rank_must_request_a_real_reduction() -> None:
    model = make_model()
    for target in (0, model.config.cp_rank, model.config.cp_rank + 1):
        with pytest.raises(ValueError, match="target_rank"):
            select_cp_rank_by_parameter_energy(model, target_rank=target)
    with pytest.raises(TypeError, match="target_rank"):
        select_cp_rank_by_parameter_energy(model, target_rank=True)
