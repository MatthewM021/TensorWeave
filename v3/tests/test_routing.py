import pytest
import torch

from tnlm_v3.routing import (
    NULL_ROUTE,
    route_counts,
    routed_lane_mask,
    validate_routes,
)


def test_validate_routes_accepts_local_global_and_valid_null_routes():
    routes = torch.tensor([[NULL_ROUTE, 0, 1, 2]])
    valid_mask = torch.ones_like(routes, dtype=torch.bool)

    assert validate_routes(routes, valid_mask, branches=2) is None


def test_invalid_padding_routes_are_ignored_and_have_no_lane():
    routes = torch.tensor([[0, 999_999, -999_999, 2]])
    valid_mask = torch.tensor([[True, False, False, True]])

    validate_routes(routes, valid_mask, branches=2)
    mask = routed_lane_mask(routes, valid_mask, branches=2)

    assert mask.dtype is torch.bool
    assert mask.shape == (1, 4, 3)
    assert torch.equal(mask[0, 0], torch.tensor([True, False, False]))
    assert not bool(mask[0, 1].any())
    assert not bool(mask[0, 2].any())
    assert torch.equal(mask[0, 3], torch.tensor([False, False, True]))


@pytest.mark.parametrize("route", [-2, 3, 100])
def test_validate_routes_rejects_out_of_range_route_at_valid_position(route):
    routes = torch.tensor([[route]])
    valid_mask = torch.tensor([[True]])

    with pytest.raises(ValueError, match="valid route IDs"):
        validate_routes(routes, valid_mask, branches=2)


def test_routed_lane_mask_excludes_null_and_padding():
    routes = torch.tensor(
        [
            [NULL_ROUTE, 0, 1, 3, 2],
            [3, 2, NULL_ROUTE, 0, 12345],
        ]
    )
    valid_mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, True, False],
        ]
    )

    actual = routed_lane_mask(routes, valid_mask, branches=3)
    expected = torch.tensor(
        [
            [
                [False, False, False, False],
                [True, False, False, False],
                [False, True, False, False],
                [False, False, False, True],
                [False, False, False, False],
            ],
            [
                [False, False, False, True],
                [False, False, True, False],
                [False, False, False, False],
                [True, False, False, False],
                [False, False, False, False],
            ],
        ]
    )

    assert torch.equal(actual, expected)


def test_route_counts_are_int64_and_include_global_lane():
    routes = torch.tensor(
        [
            [0, 0, 1, 2, NULL_ROUTE, 999],
            [2, 1, 1, NULL_ROUTE, 0, -999],
        ],
        dtype=torch.int32,
    )
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, True, True, False],
        ]
    )

    counts = route_counts(routes, valid_mask, branches=2)

    assert counts.dtype is torch.int64
    assert counts.shape == (2, 3)
    assert torch.equal(counts, torch.tensor([[2, 1, 1], [1, 2, 1]]))


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32, torch.float64])
def test_validate_routes_requires_integer_route_dtype(dtype):
    routes = torch.zeros((1, 2), dtype=dtype)
    valid_mask = torch.ones((1, 2), dtype=torch.bool)

    with pytest.raises(TypeError, match="integer dtype"):
        validate_routes(routes, valid_mask, branches=2)


def test_validate_routes_requires_matching_two_dimensional_shapes():
    with pytest.raises(ValueError, match="same shape"):
        validate_routes(
            torch.zeros((2, 3), dtype=torch.int64),
            torch.ones((2, 2), dtype=torch.bool),
            branches=2,
        )

    with pytest.raises(ValueError, match=r"\[batch, time\]"):
        validate_routes(
            torch.zeros(3, dtype=torch.int64),
            torch.ones(3, dtype=torch.bool),
            branches=2,
        )


@pytest.mark.parametrize("branches", [0, -1])
def test_validate_routes_requires_positive_branch_count(branches):
    with pytest.raises(ValueError, match="positive"):
        validate_routes(
            torch.zeros((1, 1), dtype=torch.int64),
            torch.ones((1, 1), dtype=torch.bool),
            branches=branches,
        )
