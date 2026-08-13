import pytest
import torch

from tnlm_v3.operators import ScaleSharedCPMerge, analytic_scale_features


def test_analytic_scale_features_preserve_shape_dtype_and_device():
    scales = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    features = analytic_scale_features(scales, feature_dim=7)

    assert features.shape == (2, 3, 7)
    assert features.dtype == scales.dtype
    assert features.device == scales.device
    assert torch.isfinite(features).all()


def test_analytic_scale_features_remain_finite_for_large_scales():
    scales = torch.tensor([0.0, 1.0, 1.0e12, 1.0e300], dtype=torch.float64)
    features = analytic_scale_features(scales, feature_dim=9)

    assert features.shape == (4, 9)
    assert torch.isfinite(features).all()


@pytest.mark.parametrize("scale", [3, 3.0, torch.tensor(3), torch.tensor(3.0)])
def test_merge_supports_scalar_scale_forms(scale):
    torch.manual_seed(1)
    merge = ScaleSharedCPMerge(d_model=6, cp_rank=5)
    left = torch.randn(2, 4, 6)
    right = torch.randn(2, 4, 6)

    output = merge(left, right, scale)

    assert output.shape == left.shape
    assert torch.isfinite(output).all()


def test_merge_broadcasts_scale_and_global_path():
    torch.manual_seed(2)
    merge = ScaleSharedCPMerge(d_model=8, cp_rank=4, scale_feature_dim=6)
    left = torch.randn(3, 5, 8)
    right = torch.randn(3, 5, 8)
    scales = torch.arange(5).reshape(1, 5)
    global_path = torch.tensor([[False], [True], [False]])

    output = merge(left, right, scales, global_path)

    assert output.shape == (3, 5, 8)
    assert torch.isfinite(output).all()


def test_merge_backpropagates_to_inputs_and_all_parameter_groups():
    torch.manual_seed(3)
    merge = ScaleSharedCPMerge(d_model=7, cp_rank=5, scale_feature_dim=4)
    left = torch.randn(11, 7, requires_grad=True)
    right = torch.randn(11, 7, requires_grad=True)

    loss = merge(
        left,
        right,
        scale=torch.arange(11, dtype=left.dtype),
        global_path=torch.arange(11).remainder(2).bool(),
    ).square().mean()
    loss.backward()

    assert left.grad is not None and torch.isfinite(left.grad).all()
    assert right.grad is not None and torch.isfinite(right.grad).all()
    assert all(parameter.grad is not None for parameter in merge.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in merge.parameters())


def test_merge_is_chronologically_noncommutative():
    torch.manual_seed(4)
    merge = ScaleSharedCPMerge(d_model=6, cp_rank=5)
    left = torch.randn(9, 6)
    right = torch.randn(9, 6)

    forward_order = merge(left, right, scale=2)
    reversed_order = merge(right, left, scale=2)

    assert not torch.allclose(forward_order, reversed_order)


def test_parameterization_is_independent_of_scale_values():
    torch.manual_seed(5)
    merge = ScaleSharedCPMerge(d_model=8, cp_rank=3, scale_feature_dim=5)
    before_keys = tuple(merge.state_dict().keys())
    before_count = sum(parameter.numel() for parameter in merge.parameters())
    left = torch.randn(4, 8)
    right = torch.randn(4, 8)

    merge(left, right, scale=0)
    merge(left, right, scale=torch.tensor([1, 17, 10_000, 1_000_000]))

    assert tuple(merge.state_dict().keys()) == before_keys
    assert sum(parameter.numel() for parameter in merge.parameters()) == before_count
    assert not any("table" in key or "level" in key for key in before_keys)


def test_merge_supports_float64_forward_and_backward():
    torch.manual_seed(6)
    merge = ScaleSharedCPMerge(
        d_model=5, cp_rank=4, scale_feature_dim=3
    ).double()
    left = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    scales = torch.tensor([[0.0, 7.0, 1.0e100]], dtype=torch.float64)

    output = merge(left, right, scales, global_path=True)
    output.sum().backward()

    assert output.dtype == torch.float64
    assert output.shape == left.shape
    assert torch.isfinite(output).all()
    assert left.grad is not None and torch.isfinite(left.grad).all()
    assert right.grad is not None and torch.isfinite(right.grad).all()
