from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from tnlm_v3.operators import ScaleSharedCPMerge, slice_cp_merge


CP_PARAMETERS = {
    "left.weight",
    "right.weight",
    "scale_to_rank.weight",
    "global_rank",
    "output.weight",
}


def make_source() -> ScaleSharedCPMerge:
    torch.manual_seed(4703)
    source = ScaleSharedCPMerge(
        d_model=6,
        cp_rank=5,
        scale_feature_dim=4,
    ).to(dtype=torch.float64)
    source.eval()
    return source


def dense_selected_reference(
    source: ScaleSharedCPMerge, retained_indices: tuple[int, ...]
) -> ScaleSharedCPMerge:
    dense = copy.deepcopy(source)
    discarded = sorted(set(range(source.cp_rank)) - set(retained_indices))
    index = torch.tensor(discarded, device=source.left.weight.device)
    with torch.no_grad():
        dense.left.weight.index_fill_(0, index, 0)
        dense.right.weight.index_fill_(0, index, 0)
        dense.scale_to_rank.weight.index_fill_(0, index, 0)
        dense.global_rank.index_fill_(0, index, 0)
        dense.output.weight.index_fill_(1, index, 0)
    return dense


@pytest.mark.parametrize(
    "global_path",
    (
        False,
        True,
        torch.tensor([[False, True, False]], dtype=torch.bool),
    ),
)
def test_compact_operator_matches_dense_selected_across_broadcasts(
    global_path,
) -> None:
    source = make_source()
    retained = (0, 2, 4)
    dense = dense_selected_reference(source, retained)
    compact = slice_cp_merge(source, retained)
    generator = torch.Generator().manual_seed(901)
    left = torch.randn(2, 3, 6, generator=generator, dtype=torch.float64)
    right = torch.randn(2, 3, 6, generator=generator, dtype=torch.float64)
    scale = torch.tensor([[0.0], [7.0]], dtype=torch.float64)

    expected = dense(left, right, scale=scale, global_path=global_path)
    actual = compact(left, right, scale=scale, global_path=global_path)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_slice_is_physical_smaller_and_does_not_mutate_source() -> None:
    source = make_source()
    source.left.weight.requires_grad_(False)
    before = {
        name: value.detach().clone() for name, value in source.state_dict().items()
    }

    compact = slice_cp_merge(source, (1, 3))

    assert compact.cp_rank == 2
    assert compact.left.out_features == 2
    assert compact.right.out_features == 2
    assert compact.scale_to_rank.out_features == 2
    assert compact.output.in_features == 2
    assert compact.global_rank.shape == (2,)
    assert list(compact.named_buffers()) == []
    assert compact.left.weight.dtype == source.left.weight.dtype
    assert compact.left.weight.device == source.left.weight.device
    assert compact.training is source.training
    assert compact.left.weight.requires_grad is False

    source_parameters = sum(parameter.numel() for parameter in source.parameters())
    compact_parameters = sum(parameter.numel() for parameter in compact.parameters())
    assert compact_parameters < source_parameters
    assert compact.structural_metrics()["merge_parameter_count"] < (
        source.structural_metrics()["merge_parameter_count"]
    )
    assert compact.structural_metrics()["operation_count_proxy_per_merge"] < (
        source.structural_metrics()["operation_count_proxy_per_merge"]
    )
    source_bytes = sum(
        value.numel() * value.element_size() for value in source.state_dict().values()
    )
    compact_bytes = sum(
        value.numel() * value.element_size() for value in compact.state_dict().values()
    )
    assert compact_bytes < source_bytes
    for name, value in source.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_non_cp_parameters_are_copied_exactly() -> None:
    source = make_source()
    source.gate.bias.requires_grad_(False)
    compact = slice_cp_merge(source, (0, 3, 4))
    source_parameters = dict(source.named_parameters())
    compact_parameters = dict(compact.named_parameters())

    for name, parameter in source_parameters.items():
        if name in CP_PARAMETERS:
            continue
        assert torch.equal(compact_parameters[name], parameter), name
        assert compact_parameters[name].dtype == parameter.dtype
        assert compact_parameters[name].device == parameter.device
        assert compact_parameters[name].requires_grad == parameter.requires_grad


def test_slice_preserves_rng_and_nested_module_training_flags() -> None:
    source = make_source()
    source.train()
    source.gate.eval()
    source.norm.eval()
    torch.manual_seed(9123)
    before = torch.random.get_rng_state().clone()

    compact = slice_cp_merge(source, (0, 2, 4))

    assert torch.equal(torch.random.get_rng_state(), before)
    assert compact.training is True
    assert compact.gate.training is False
    assert compact.norm.training is False
    source_modes = {name: module.training for name, module in source.named_modules()}
    compact_modes = {name: module.training for name, module in compact.named_modules()}
    assert compact_modes == source_modes


def test_compact_operator_backward_is_finite() -> None:
    compact = slice_cp_merge(make_source(), (0, 2, 3))
    left = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    right = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)

    loss = compact(
        left,
        right,
        scale=torch.arange(4, dtype=torch.float64),
        global_path=torch.tensor([False, True, False, True]),
    ).square().mean()
    loss.backward()

    assert left.grad is not None and bool(torch.isfinite(left.grad).all())
    assert right.grad is not None and bool(torch.isfinite(right.grad).all())
    for parameter in compact.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert bool(torch.isfinite(parameter.grad).all())


class DerivedMerge(ScaleSharedCPMerge):
    pass


@pytest.mark.parametrize(
    "source",
    (nn.Linear(2, 2), DerivedMerge(3, 3)),
)
def test_slice_rejects_non_exact_source_type(source: nn.Module) -> None:
    with pytest.raises(TypeError, match="exactly"):
        slice_cp_merge(source, (0,))


@pytest.mark.parametrize(
    ("indices", "match"),
    (
        ((), "at least one"),
        ((0, 0), "sorted and unique"),
        ((2, 1), "sorted and unique"),
        ((-1,), "outside"),
        ((5,), "outside"),
        ((0, 1, 2, 3, 4), "smaller"),
    ),
)
def test_slice_rejects_invalid_index_sets(indices, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        slice_cp_merge(make_source(), indices)


@pytest.mark.parametrize("indices", ((True,), (1.5,), "01", None))
def test_slice_rejects_non_integer_index_inputs(indices) -> None:
    with pytest.raises(TypeError, match="integers"):
        slice_cp_merge(make_source(), indices)
