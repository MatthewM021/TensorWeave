from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn

from tnlm_v3.forest import (
    ForestConfig,
    ForestReadout,
    RoutedTensorLanguageModel,
    ScaleSharedBinaryForest,
)
from tnlm_v3.routing import NULL_ROUTE


class ChronologyMerge(nn.Module):
    """Non-associative merge that makes operand order directly observable."""

    def forward(self, left, right, *, scale, global_path=False):
        del scale, global_path
        return left * 10.0 + right


class CountingMerge(ChronologyMerge):
    def __init__(self):
        super().__init__()
        self.elements = 0

    def forward(self, left, right, *, scale, global_path=False):
        self.elements += left.numel() // left.shape[-1]
        return super().forward(
            left, right, scale=scale, global_path=global_path
        )


def make_forest(*, merge=None, dtype=torch.float64):
    forest = ScaleSharedBinaryForest(
        d_model=3,
        branches=2,
        cp_rank=4,
        scale_feature_dim=8,
        merge=merge,
    )
    return forest.to(dtype=dtype)


@pytest.mark.parametrize(
    "length", [0, 1, 2, 3, 4, 5, 7, 8, 13, 31, 32, 33, 65]
)
def test_binary_occupancy_and_exact_merge_count(length):
    forest = make_forest(merge=ChronologyMerge())
    events = torch.arange(1, length * 3 + 1, dtype=torch.float64).reshape(1, length, 3)
    routes = torch.zeros(1, length, dtype=torch.int64)
    valid = torch.ones(1, length, dtype=torch.bool)

    streamed = forest.reduce_streaming(events, routes, valid)
    parallel = forest.reduce_parallel(events, routes, valid)

    expected = [(length >> scale) & 1 for scale in range(streamed.state.scales)]
    assert streamed.state.occupied[0, 0].tolist() == [bool(value) for value in expected]
    assert not bool(streamed.state.occupied[0, 1:].any())
    assert int(streamed.merge_count) == length - length.bit_count()
    assert int(parallel.merge_count) == length - length.bit_count()
    torch.testing.assert_close(streamed.state.slots, parallel.state.slots, rtol=0, atol=0)
    assert torch.equal(streamed.state.occupied, parallel.state.occupied)


def test_sparse_routes_use_chronologically_packed_subsequences():
    forest = ScaleSharedBinaryForest(
        d_model=1,
        branches=2,
        cp_rank=1,
        merge=ChronologyMerge(),
    ).to(dtype=torch.float64)
    events = torch.tensor([[[1.0], [9.0], [2.0], [8.0], [3.0], [4.0], [5.0], [6.0]]])
    routes = torch.tensor([[0, 1, 0, 1, 0, 0, 0, 0]])
    valid = torch.ones_like(routes, dtype=torch.bool)

    streamed = forest.reduce_streaming(events, routes, valid).state
    parallel = forest.reduce_parallel(events, routes, valid).state

    # Lane 0 receives [1,2,3,4,5,6]: scale 2 is ((1,2),(3,4))=154,
    # and scale 1 is (5,6)=56.  This catches wrong singleton promotion.
    assert streamed.occupied[0, 0].tolist() == [False, True, True]
    assert streamed.slots[0, 0, 1, 0].item() == 56.0
    assert streamed.slots[0, 0, 2, 0].item() == 154.0
    torch.testing.assert_close(streamed.slots, parallel.slots, rtol=0, atol=0)


def test_streaming_and_parallel_real_operator_match_values_and_gradients():
    torch.manual_seed(4)
    forest = make_forest()
    events_a = torch.randn(2, 13, 3, dtype=torch.float64, requires_grad=True)
    events_b = events_a.detach().clone().requires_grad_(True)
    routes = torch.tensor(
        [
            [0, 1, 0, 2, -1, 0, 1, 0, 0, 2, 1, 0, 1],
            [1, 1, 0, -1, 2, 0, 1, 1, 0, 0, 2, 1, 0],
        ],
        dtype=torch.int64,
    )
    valid = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    streamed = forest.reduce_streaming(events_a, routes, valid)
    parallel = forest.reduce_parallel(events_b, routes, valid)
    torch.testing.assert_close(streamed.state.slots, parallel.state.slots, rtol=1e-9, atol=1e-10)
    assert torch.equal(streamed.state.occupied, parallel.state.occupied)
    assert torch.equal(streamed.state.counts, parallel.state.counts)
    assert torch.equal(streamed.state.valid_steps, parallel.state.valid_steps)
    assert int(streamed.merge_count) == int(parallel.merge_count)

    parameters = tuple(forest.parameters())
    loss_a = (streamed.state.slots.square()).sum()
    grads_a = torch.autograd.grad(loss_a, (events_a, *parameters))
    loss_b = (parallel.state.slots.square()).sum()
    grads_b = torch.autograd.grad(loss_b, (events_b, *parameters))
    for left, right in zip(grads_a, grads_b, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-10)


def test_parallel_prefix_builder_matches_every_streaming_prefix():
    torch.manual_seed(6)
    forest = make_forest()
    events = torch.randn(2, 19, 3, dtype=torch.float64)
    routes = torch.randint(-1, 3, (2, 19), dtype=torch.int64)
    valid = torch.rand(2, 19) > 0.25
    parallel = forest.reduce_parallel_prefixes(events, routes, valid)

    assert len(parallel.states) == events.shape[1]
    for length, state in enumerate(parallel.states, start=1):
        streamed = forest.reduce_streaming(
            events[:, :length], routes[:, :length], valid[:, :length]
        ).state
        # The prefix builder uses final-sequence capacity; trim its all-empty
        # tail before comparing against a dynamically grown streaming state.
        trimmed = state.slots[:, :, : streamed.scales]
        trimmed_occupied = state.occupied[:, :, : streamed.scales]
        torch.testing.assert_close(trimmed, streamed.slots, rtol=1e-9, atol=1e-10)
        assert torch.equal(trimmed_occupied, streamed.occupied)
        assert torch.equal(state.counts, streamed.counts)
        assert torch.equal(state.valid_steps, streamed.valid_steps)


def test_padding_is_total_noop_and_null_is_clock_only():
    forest = make_forest()
    initial = forest.initial_state(2, dtype=torch.float64)
    event = torch.randn(2, 3, dtype=torch.float64)

    padded = forest.step(
        initial,
        event,
        route=torch.tensor([999, -999]),
        valid=torch.tensor([False, False]),
    ).state
    torch.testing.assert_close(padded.slots, initial.slots, rtol=0, atol=0)
    assert torch.equal(padded.occupied, initial.occupied)
    assert torch.equal(padded.counts, initial.counts)
    assert torch.equal(padded.valid_steps, initial.valid_steps)

    queried = forest.step(
        initial,
        event,
        route=torch.tensor([NULL_ROUTE, NULL_ROUTE]),
        valid=torch.tensor([True, True]),
    ).state
    assert torch.equal(queried.counts, initial.counts)
    assert queried.valid_steps.tolist() == [1, 1]
    assert not bool(queried.occupied.any())


def test_global_route_has_its_own_lane_and_state_input_is_not_mutated():
    forest = make_forest()
    initial = forest.initial_state(1, dtype=torch.float64)
    before = initial.slots.clone()
    result = forest.step(
        initial,
        torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
        route=torch.tensor([forest.branches]),
        valid=torch.tensor([True]),
    ).state
    torch.testing.assert_close(initial.slots, before, rtol=0, atol=0)
    assert not bool(initial.occupied.any())
    assert result.counts.tolist() == [[0, 0, 1]]
    assert result.occupied[0, forest.branches, 0]


def test_runtime_scale_growth_has_no_parameters_and_obeys_memory_bound():
    forest = ScaleSharedBinaryForest(
        d_model=1, branches=3, cp_rank=1, merge=ChronologyMerge()
    ).to(dtype=torch.float64)
    keys_before = tuple(forest.state_dict())
    parameter_count = sum(parameter.numel() for parameter in forest.parameters())
    length = 257
    events = torch.ones(1, length, 1, dtype=torch.float64)
    routes = torch.arange(length).remainder(4).unsqueeze(0)
    valid = torch.ones(1, length, dtype=torch.bool)
    state = forest.reduce_streaming(events, routes, valid).state

    assert state.scales == int(state.counts.max()).bit_length()
    assert int(state.occupied.sum()) == sum(
        int(value).bit_count() for value in state.counts.flatten().tolist()
    )
    assert state.occupied.numel() <= forest.paths * math.ceil(math.log2(length + 1))
    assert tuple(forest.state_dict()) == keys_before
    assert sum(parameter.numel() for parameter in forest.parameters()) == parameter_count


def test_executed_merge_work_equals_reported_binary_carries():
    merge = CountingMerge()
    forest = ScaleSharedBinaryForest(
        d_model=1, branches=3, cp_rank=1, merge=merge
    ).to(dtype=torch.float64)
    length = 257
    events = torch.ones(2, length, 1, dtype=torch.float64)
    routes = torch.stack(
        (torch.zeros(length, dtype=torch.int64), torch.arange(length).remainder(4))
    )
    valid = torch.ones(2, length, dtype=torch.bool)
    run = forest.reduce_streaming(events, routes, valid)

    assert merge.elements == int(run.merge_count)
    expected = sum(
        int(count) - int(count).bit_count()
        for count in run.state.counts.flatten().tolist()
    )
    assert merge.elements == expected
    metrics = forest.structural_metrics(run.state, merge_count=run.merge_count)
    assert metrics["executed_merge_count"] == expected
    assert metrics["active_slots"] == int(run.state.occupied.sum())
    assert metrics["nominal_rank"] == forest.cp_rank
    assert metrics["effective_rank"] == forest.cp_rank
    assert metrics["exported_rank"] == forest.cp_rank


def test_no_merge_is_executed_for_padding_null_or_first_insert():
    merge = CountingMerge()
    forest = ScaleSharedBinaryForest(
        d_model=1, branches=3, cp_rank=1, merge=merge
    ).to(dtype=torch.float64)
    state = forest.initial_state(2, dtype=torch.float64)
    event = torch.ones(2, 1, dtype=torch.float64)

    for routes, valid in (
        (torch.tensor([999, -999]), torch.tensor([False, False])),
        (torch.tensor([NULL_ROUTE, NULL_ROUTE]), torch.tensor([True, True])),
        (torch.tensor([0, 3]), torch.tensor([True, True])),
    ):
        run = forest.step(state, event, routes, valid)
        assert int(run.merge_count) == 0
        assert merge.elements == 0
        state = run.state


def test_resume_from_checkpoint_matches_uninterrupted_streaming():
    torch.manual_seed(9)
    forest = make_forest()
    events = torch.randn(2, 17, 3, dtype=torch.float64)
    routes = torch.randint(-1, 3, (2, 17), dtype=torch.int64)
    valid = torch.rand(2, 17) > 0.2
    full = forest.reduce_streaming(events, routes, valid).state
    prefix = forest.reduce_streaming(events[:, :7], routes[:, :7], valid[:, :7]).state
    resumed = forest.reduce_streaming(
        events[:, 7:], routes[:, 7:], valid[:, 7:], initial_state=prefix
    ).state

    torch.testing.assert_close(full.slots, resumed.slots, rtol=0, atol=0)
    assert torch.equal(full.occupied, resumed.occupied)
    assert torch.equal(full.counts, resumed.counts)
    assert torch.equal(full.valid_steps, resumed.valid_steps)


def make_model() -> RoutedTensorLanguageModel:
    return RoutedTensorLanguageModel(
        ForestConfig(
            branches=3,
            d_model=6,
            cp_rank=4,
            vocab_size=29,
            output_size=11,
            scale_feature_dim=8,
        )
    ).to(dtype=torch.float64)


def test_model_streaming_parallel_logits_state_and_gradients_match():
    torch.manual_seed(20)
    streaming_model = make_model()
    parallel_model = copy.deepcopy(streaming_model)
    tokens = torch.randint(1, 29, (2, 9))
    routes = torch.tensor(
        [[0, 1, 2, 3, -1, 0, 1, 2, 0], [2, 0, 1, -1, 3, 2, 0, 1, 2]]
    )
    valid = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 1, 1]],
        dtype=torch.bool,
    )
    left = streaming_model(tokens, routes, valid, implementation="streaming")
    right = parallel_model(tokens, routes, valid, implementation="parallel")
    torch.testing.assert_close(left.logits, right.logits, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(left.state.slots, right.state.slots, rtol=1e-9, atol=1e-10)

    target = torch.randn_like(left.logits)
    left_loss = (left.logits * target).sum()
    right_loss = (right.logits * target).sum()
    left_grads = torch.autograd.grad(left_loss, tuple(streaming_model.parameters()))
    right_grads = torch.autograd.grad(right_loss, tuple(parallel_model.parameters()))
    for grad_left, grad_right in zip(left_grads, right_grads, strict=True):
        torch.testing.assert_close(grad_left, grad_right, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("implementation", ["streaming", "parallel"])
def test_logits_are_causal_under_future_token_and_route_changes(implementation):
    torch.manual_seed(31)
    model = make_model().eval()
    tokens = torch.randint(1, 29, (1, 8))
    routes = torch.tensor([[0, 1, 2, -1, 3, 0, 1, 2]])
    valid = torch.ones_like(tokens, dtype=torch.bool)
    baseline = model(tokens, routes, valid, implementation=implementation).logits
    changed_tokens = tokens.clone()
    changed_routes = routes.clone()
    changed_tokens[:, 5:] = torch.tensor([[27, 26, 25]])
    changed_routes[:, 5:] = torch.tensor([[2, 3, -1]])
    changed = model(
        changed_tokens, changed_routes, valid, implementation=implementation
    ).logits
    torch.testing.assert_close(baseline[:, :5], changed[:, :5], rtol=0, atol=0)


def test_interspersed_padding_does_not_change_valid_event_results():
    torch.manual_seed(40)
    model = make_model().eval()
    compact_tokens = torch.tensor([[3, 4, 5, 6, 7]])
    compact_routes = torch.tensor([[0, 1, -1, 3, 0]])
    compact_valid = torch.ones_like(compact_tokens, dtype=torch.bool)
    padded_tokens = torch.tensor([[3, 20, 4, 21, 5, 6, 22, 7]])
    padded_routes = torch.tensor([[0, 999, 1, -999, -1, 3, 222, 0]])
    padded_valid = torch.tensor([[1, 0, 1, 0, 1, 1, 0, 1]], dtype=torch.bool)

    compact = model(compact_tokens, compact_routes, compact_valid)
    padded = model(padded_tokens, padded_routes, padded_valid)
    torch.testing.assert_close(
        compact.logits,
        padded.logits[:, padded_valid[0]],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(compact.state.slots, padded.state.slots, rtol=0, atol=0)
    assert torch.equal(compact.state.counts, padded.state.counts)
    assert torch.equal(compact.state.valid_steps, padded.state.valid_steps)


@pytest.mark.parametrize("implementation", ["streaming", "parallel"])
def test_padding_token_ids_are_never_observed_even_when_out_of_vocabulary(implementation):
    torch.manual_seed(44)
    model = make_model().eval()
    compact_tokens = torch.tensor([[3, 4, 5]])
    compact_routes = torch.tensor([[0, 1, -1]])
    compact_valid = torch.ones_like(compact_tokens, dtype=torch.bool)
    padded_tokens = torch.tensor([[3, -99999, 4, 99999, 5]])
    padded_routes = torch.tensor([[0, 888, 1, -777, -1]])
    padded_valid = torch.tensor([[1, 0, 1, 0, 1]], dtype=torch.bool)

    compact = model(
        compact_tokens, compact_routes, compact_valid, implementation=implementation
    )
    padded = model(
        padded_tokens, padded_routes, padded_valid, implementation=implementation
    )
    torch.testing.assert_close(
        compact.logits,
        padded.logits[:, padded_valid[0]],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(compact.state.slots, padded.state.slots, rtol=0, atol=0)


def test_float32_streaming_parallel_parity_uses_declared_tolerance():
    torch.manual_seed(45)
    model = make_model().float().eval()
    tokens = torch.randint(1, 29, (2, 17))
    routes = torch.randint(-1, 4, (2, 17))
    valid = torch.rand(2, 17) > 0.2
    streamed = model(tokens, routes, valid, implementation="streaming")
    parallel = model(tokens, routes, valid, implementation="parallel")
    torch.testing.assert_close(streamed.logits, parallel.logits, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(streamed.state.slots, parallel.state.slots, rtol=2e-5, atol=2e-6)


def test_local_branch_permutation_equivariance_and_readout_invariance():
    torch.manual_seed(50)
    model = make_model().eval()
    events = torch.randn(2, 12, 6, dtype=torch.float64)
    routes = torch.randint(-1, 4, (2, 12))
    valid = torch.ones(2, 12, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 1])
    permuted_routes = routes.clone()
    for old in range(3):
        permuted_routes[routes == old] = permutation[old]
    original = model.forest.reduce_streaming(events, routes, valid).state
    permuted = model.forest.reduce_streaming(events, permuted_routes, valid).state

    torch.testing.assert_close(
        original.slots[:, :3], permuted.slots[:, permutation], rtol=1e-9, atol=1e-10
    )
    torch.testing.assert_close(original.slots[:, 3], permuted.slots[:, 3], rtol=0, atol=0)
    query = torch.randn(2, 6, dtype=torch.float64)
    torch.testing.assert_close(
        model.readout(original, query),
        model.readout(permuted, query),
        rtol=1e-9,
        atol=1e-10,
    )


def test_configuration_and_checkpoint_are_length_independent():
    config = ForestConfig(branches=2, d_model=4, cp_rank=3, vocab_size=13)
    assert "length" not in config.canonical_json()
    model = RoutedTensorLanguageModel(config)
    checkpoint = copy.deepcopy(model.state_dict())
    keys = tuple(checkpoint)
    count = sum(parameter.numel() for parameter in model.parameters())

    for runtime_length in (1, 33, 257, 2048):
        # Runtime length is deliberately not passed to construction.
        reloaded = RoutedTensorLanguageModel(config)
        reloaded.load_state_dict(checkpoint, strict=True)
        assert tuple(reloaded.state_dict()) == keys
        assert sum(parameter.numel() for parameter in reloaded.parameters()) == count
        assert runtime_length >= 1


def test_loaded_checkpoint_executes_beyond_257_updates_without_new_parameters():
    torch.manual_seed(60)
    config = ForestConfig(
        branches=2, d_model=4, cp_rank=3, vocab_size=17, output_size=5
    )
    source = RoutedTensorLanguageModel(config)
    checkpoint = copy.deepcopy(source.state_dict())
    model = RoutedTensorLanguageModel(config)
    model.load_state_dict(checkpoint, strict=True)
    keys_before = tuple(model.state_dict())
    count_before = sum(parameter.numel() for parameter in model.parameters())
    length = 258
    tokens = torch.randint(1, config.vocab_size, (1, length))
    routes = torch.arange(length).remainder(config.paths).unsqueeze(0)
    valid = torch.ones(1, length, dtype=torch.bool)

    with torch.no_grad():
        output = model(tokens, routes, valid, implementation="streaming")

    assert output.logits.shape == (1, length, config.resolved_output_size)
    assert output.state.scales >= 7
    assert tuple(model.state_dict()) == keys_before
    assert sum(parameter.numel() for parameter in model.parameters()) == count_before


def test_batch_members_are_independent():
    torch.manual_seed(62)
    model = make_model().eval()
    tokens = torch.randint(1, 29, (2, 10))
    routes = torch.randint(-1, 4, (2, 10))
    valid = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]],
        dtype=torch.bool,
    )
    together = model(tokens, routes, valid)
    first = model(tokens[:1], routes[:1], valid[:1])
    second = model(tokens[1:], routes[1:], valid[1:])

    torch.testing.assert_close(together.logits[:1], first.logits, rtol=0, atol=0)
    torch.testing.assert_close(together.logits[1:], second.logits, rtol=0, atol=0)
    together_first = model.forest.state_for_batch(together.state, 0)
    together_second = model.forest.state_for_batch(together.state, 1)
    torch.testing.assert_close(together_first.slots, first.state.slots, rtol=0, atol=0)
    torch.testing.assert_close(
        together_second.slots,
        second.state.slots,
        rtol=0,
        atol=0,
    )
    assert torch.equal(together_second.counts, second.state.counts)


def test_empty_readout_is_finite_and_query_dependent():
    torch.manual_seed(61)
    readout = ForestReadout(d_model=4, branches=2, output_size=3).to(torch.float64)
    forest = ScaleSharedBinaryForest(d_model=4, branches=2, cp_rank=2).to(torch.float64)
    state = forest.initial_state(2, dtype=torch.float64)
    query = torch.randn(2, 4, dtype=torch.float64)
    logits = readout(state, query)
    assert torch.isfinite(logits).all()
    assert not torch.equal(logits[0], logits[1])
