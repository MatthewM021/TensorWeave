from __future__ import annotations

import copy
from dataclasses import fields, replace
import io
import inspect
import math
from typing import get_type_hints

import pytest
import torch
from torch import Tensor, nn

from tnlm_v3.baselines import BaselineBindingOutput
from tnlm_v3.binding import BindingArchitectureConfig
from tnlm_v3.causal_ttn import (
    CausalCompleteTreeBindingBaseline,
    CausalTreeBindingBaselineConfig,
    CausalTreeBindingState,
)
from tnlm_v3.data import (
    BindingEventKind,
    BindingModelInputs,
    BindingTaskConfig,
    collate_binding_episodes,
    generate_binding_episodes,
)
from tnlm_v3.operators import ScaleSharedCPMerge


def task(*, max_length: int = 300) -> BindingTaskConfig:
    return BindingTaskConfig(
        num_surface_keys=5,
        value_cardinality=4,
        branches=3,
        max_live_bindings=3,
        min_length=10,
        max_length=max_length,
        heldout_key_value_pairs=((0, 0),),
        global_distractor_probability=0.5,
    )


def batch_inputs(lengths: tuple[int, ...] = (17, 13)) -> BindingModelInputs:
    episodes = generate_binding_episodes(
        task(max_length=max(lengths)),
        count=len(lengths),
        seed=18_071,
        split="train",
        lengths=lengths,
    )
    return collate_binding_episodes(episodes).inputs


def sliced(inputs: BindingModelInputs, start: int, stop: int) -> BindingModelInputs:
    return BindingModelInputs(
        **{
            field.name: getattr(inputs, field.name)[:, start:stop]
            for field in fields(inputs)
        }
    )


def make_model(
    *,
    d_model: int = 8,
    cp_rank: int = 5,
    dtype: torch.dtype = torch.float64,
) -> CausalCompleteTreeBindingBaseline:
    torch.manual_seed(918)
    return CausalCompleteTreeBindingBaseline(
        CausalTreeBindingBaselineConfig(
            task(),
            d_model=d_model,
            cp_rank=cp_rank,
            scale_feature_dim=4,
        )
    ).to(dtype=dtype)


def assert_state_equal(
    left: CausalTreeBindingState,
    right: CausalTreeBindingState,
) -> None:
    assert torch.equal(left.slots, right.slots)
    assert torch.equal(left.occupied, right.occupied)
    assert torch.equal(left.valid_steps, right.valid_steps)


def assert_state_storage_independent(
    left: CausalTreeBindingState,
    right: CausalTreeBindingState,
) -> None:
    assert left is not right
    assert left.slots.data_ptr() != right.slots.data_ptr()
    assert left.occupied.data_ptr() != right.occupied.data_ptr()
    assert left.valid_steps.data_ptr() != right.valid_steps.data_ptr()


class RecordingChronologyMerge(nn.Module):
    """Non-associative merge that rejects dummy or reversed children."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[int, float, float]] = []
        self.elements = 0

    def forward(
        self,
        left: Tensor,
        right: Tensor,
        *,
        scale: int,
        global_path: bool | Tensor = False,
    ) -> Tensor:
        assert left.shape == right.shape
        assert left.shape[-1] == 1
        assert bool((left > 0).all())
        assert bool((right > 0).all())
        assert bool(torch.as_tensor(global_path).all())
        for older, newer in zip(
            left.reshape(-1).tolist(), right.reshape(-1).tolist(), strict=True
        ):
            self.calls.append((int(scale), float(older), float(newer)))
        self.elements += left.numel()
        return left * 10.0 + right


class SumMerge(nn.Module):
    def forward(
        self,
        left: Tensor,
        right: Tensor,
        *,
        scale: int,
        global_path: bool | Tensor = False,
    ) -> Tensor:
        del scale, global_path
        return left + right


def core_model(merge: nn.Module) -> CausalCompleteTreeBindingBaseline:
    model = make_model(d_model=1, cp_rank=1)
    model.merge = merge
    return model


def append_scalar_prefix(
    model: CausalCompleteTreeBindingBaseline,
    length: int,
) -> CausalTreeBindingState:
    state = model.initial_state(1, device="cpu", dtype=torch.float64)
    for value in range(1, length + 1):
        state, _ = model._append(
            state,
            torch.tensor([[float(value)]], dtype=torch.float64),
            torch.tensor([True]),
        )
    return state


def scalar_dense_complete_tree(length: int) -> float:
    nodes = [float(value) for value in range(1, length + 1)]
    while len(nodes) > 1:
        next_nodes: list[float] = []
        for index in range(0, len(nodes), 2):
            if index + 1 == len(nodes):
                next_nodes.append(nodes[index])
            else:
                next_nodes.append(nodes[index] * 10.0 + nodes[index + 1])
        nodes = next_nodes
    return nodes[0] if nodes else 0.0


def dense_complete_tree(
    leaves: list[Tensor], merge: nn.Module
) -> Tensor:
    if not leaves:
        raise ValueError("dense reference requires at least one leaf")
    current = leaves
    scale = 0
    while len(current) > 1:
        next_level: list[Tensor] = []
        for index in range(0, len(current), 2):
            if index + 1 == len(current):
                next_level.append(current[index])
            else:
                next_level.append(
                    merge(
                        current[index],
                        current[index + 1],
                        scale=scale,
                        global_path=True,
                    )
                )
        current = next_level
        scale += 1
    return current[0]


@pytest.mark.parametrize(
    "length", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65]
)
def test_complete_tree_matches_independent_dense_reference(length: int) -> None:
    merge = RecordingChronologyMerge()
    model = core_model(merge)
    state = append_scalar_prefix(model, length)
    root, root_merges = model._prefix_root(
        state, torch.tensor([length > 0], dtype=torch.bool)
    )

    assert root.item() == scalar_dense_complete_tree(length)
    assert merge.elements == max(length - 1, 0)
    assert int(root_merges) == max(length.bit_count() - 1, 0)


def test_length_seven_is_complete_tree_not_left_folded_forest() -> None:
    merge = RecordingChronologyMerge()
    model = core_model(merge)
    state = append_scalar_prefix(model, 7)
    root, _ = model._prefix_root(state, torch.tensor([True]))

    assert state.occupied.tolist() == [[True, True, True]]
    assert state.slots[0, :, 0].tolist() == [7.0, 56.0, 154.0]
    assert root.item() == 2107.0
    assert root.item() != 15967.0
    assert merge.calls == [
        (0, 1.0, 2.0),
        (0, 3.0, 4.0),
        (1, 12.0, 34.0),
        (0, 5.0, 6.0),
        (1, 56.0, 7.0),
        (2, 154.0, 567.0),
    ]


@pytest.mark.parametrize("length", [1, 3, 5, 7, 9])
def test_singletons_are_identity_promoted_without_dummy_merges(length: int) -> None:
    merge = RecordingChronologyMerge()
    model = core_model(merge)
    state = append_scalar_prefix(model, length)
    root, _ = model._prefix_root(state, torch.tensor([True]))

    assert root.item() == scalar_dense_complete_tree(length)
    assert merge.elements == length - 1
    if length == 1:
        assert merge.calls == []
        assert root.item() == 1.0


def test_incremental_roots_match_dense_prefix_values_and_gradients() -> None:
    torch.manual_seed(77)
    incremental = make_model(d_model=4, cp_rank=3)
    reference = copy.deepcopy(incremental)
    events_a = torch.randn(1, 9, 4, dtype=torch.float64, requires_grad=True)
    events_b = events_a.detach().clone().requires_grad_(True)
    valid = torch.tensor([[1, 0, 1, 1, 0, 1, 1, 1, 1]], dtype=torch.bool)

    state = incremental.initial_state(1, device="cpu", dtype=torch.float64)
    seen_b: list[Tensor] = []
    roots_a: list[Tensor] = []
    roots_b: list[Tensor] = []
    for index in range(events_a.shape[1]):
        state, _ = incremental._append(
            state, events_a[:, index], valid[:, index]
        )
        if bool(valid[0, index]):
            root, _ = incremental._prefix_root(state, valid[:, index])
            roots_a.append(root)
            seen_b.append(events_b[:, index])
            roots_b.append(dense_complete_tree(list(seen_b), reference.merge))

    stacked_a = torch.stack(roots_a, dim=1)
    stacked_b = torch.stack(roots_b, dim=1)
    torch.testing.assert_close(stacked_a, stacked_b, rtol=1e-9, atol=1e-10)
    weight = torch.randn_like(stacked_a)
    gradients_a = torch.autograd.grad(
        (stacked_a * weight).sum(),
        (events_a, *incremental.merge.parameters()),
    )
    gradients_b = torch.autograd.grad(
        (stacked_b * weight).sum(),
        (events_b, *reference.merge.parameters()),
    )
    for left, right in zip(gradients_a, gradients_b, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_step_and_many_cut_chunk_resume_are_identical(
    dtype: torch.dtype,
) -> None:
    model = make_model(dtype=dtype).eval()
    inputs = batch_inputs((17, 13))
    full = model(inputs)

    state = model.initial_state(2, device="cpu", dtype=dtype)
    stepped_logits: list[Tensor] = []
    for index in range(17):
        output = model.step(sliced(inputs, index, index + 1), state)
        state = output.final_state
        stepped_logits.append(output.value_logits)
    assert torch.equal(full.value_logits, torch.cat(stepped_logits, dim=1))
    assert_state_equal(full.final_state, state)

    state = model.initial_state(2, device="cpu", dtype=dtype)
    chunk_logits: list[Tensor] = []
    cuts = (0, 1, 3, 7, 8, 11, 16, 17)
    for start, stop in zip(cuts[:-1], cuts[1:], strict=True):
        output = model(sliced(inputs, start, stop), state)
        state = output.final_state
        chunk_logits.append(output.value_logits)
    assert torch.equal(full.value_logits, torch.cat(chunk_logits, dim=1))
    assert_state_equal(full.final_state, state)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_model_and_tensor_state_roundtrip_resume_exactly(dtype: torch.dtype) -> None:
    model = make_model(dtype=dtype).eval()
    inputs = batch_inputs((17, 13))
    prefix = model(sliced(inputs, 0, 7))
    expected = model(sliced(inputs, 7, 17), prefix.final_state)

    buffer = io.BytesIO()
    torch.save(
        {
            "model_state": model.state_dict(),
            "slots": prefix.final_state.slots,
            "occupied": prefix.final_state.occupied,
            "valid_steps": prefix.final_state.valid_steps,
        },
        buffer,
    )
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=True)

    resumed_model = make_model(dtype=dtype).eval()
    resumed_model.load_state_dict(restored["model_state"], strict=True)
    resumed_state = CausalTreeBindingState(
        slots=restored["slots"],
        occupied=restored["occupied"],
        valid_steps=restored["valid_steps"],
    )
    actual = resumed_model(sliced(inputs, 7, 17), resumed_state)
    assert torch.equal(expected.value_logits, actual.value_logits)
    assert_state_equal(expected.final_state, actual.final_state)


@pytest.mark.parametrize("invalid", [None, True, 1, "float32", torch.int64])
def test_initial_state_rejects_nonfloating_or_non_dtype_values(invalid: object) -> None:
    with pytest.raises(TypeError, match="floating-point torch dtype"):
        make_model().initial_state(1, device="cpu", dtype=invalid)  # type: ignore[arg-type]


def test_future_fields_and_validity_cannot_affect_prefix_values_or_gradients() -> None:
    inputs = batch_inputs((17, 17))
    cut = 7
    changed_values: dict[str, Tensor] = {}
    for field in fields(inputs):
        tensor = getattr(inputs, field.name).clone()
        if field.name == "valid_mask":
            changed_values[field.name] = tensor
            continue
        if field.name == "token_ids":
            tensor[:, cut:] = 1 + tensor[:, cut:].remainder(
                BindingArchitectureConfig.from_task(task()).vocab_size - 1
            )
        elif field.name == "event_kinds":
            tensor[:, cut:] = 1 + tensor[:, cut:].remainder(
                len(BindingEventKind) - 1
            )
        elif field.name in ("primary_key_ids", "secondary_key_ids"):
            tensor[:, cut:] = (tensor[:, cut:] + 1).remainder(
                task().num_surface_keys + 1
            )
        else:
            tensor[:, cut:] = (tensor[:, cut:] + 1).remainder(
                task().value_cardinality + 1
            )
        changed_values[field.name] = tensor
    changed = BindingModelInputs(**changed_values)

    first = make_model()
    second = copy.deepcopy(first)
    left = first(inputs).value_logits[:, :cut]
    right = second(changed).value_logits[:, :cut]
    assert torch.equal(left, right)
    gradients_left = torch.autograd.grad(left.square().sum(), tuple(first.parameters()))
    gradients_right = torch.autograd.grad(
        right.square().sum(), tuple(second.parameters())
    )
    for expected, actual in zip(gradients_left, gradients_right, strict=True):
        assert torch.equal(expected, actual)

    future_padding = replace(
        changed,
        valid_mask=torch.cat(
            (
                changed.valid_mask[:, :cut],
                torch.zeros_like(changed.valid_mask[:, cut:]),
            ),
            dim=1,
        ),
    )
    assert torch.equal(
        make_model().eval()(inputs).value_logits[:, :cut],
        make_model().eval()(future_padding).value_logits[:, :cut],
    )


def test_padding_garbage_is_exact_forward_and_backward_noop() -> None:
    inputs = batch_inputs((13, 10))
    garbage_values = {
        "token_ids": 2**60,
        "event_kinds": 2**60,
        "primary_key_ids": -(2**60),
        "secondary_key_ids": 2**60,
        "arguments": -(2**60),
    }
    values: dict[str, Tensor] = {}
    for field in fields(inputs):
        tensor = getattr(inputs, field.name).clone()
        if field.name != "valid_mask":
            tensor[~inputs.valid_mask] = garbage_values[field.name]
        values[field.name] = tensor
    garbage = BindingModelInputs(**values)

    first = make_model()
    second = copy.deepcopy(first)
    output_a = first(inputs)
    output_b = second(garbage)
    assert torch.equal(output_a.value_logits, output_b.value_logits)
    assert_state_equal(output_a.final_state, output_b.final_state)
    gradients_a = torch.autograd.grad(
        output_a.value_logits.square().sum(), tuple(first.parameters())
    )
    gradients_b = torch.autograd.grad(
        output_b.value_logits.square().sum(), tuple(second.parameters())
    )
    for expected, actual in zip(gradients_a, gradients_b, strict=True):
        assert torch.equal(expected, actual)


def test_interspersed_padding_matches_chronologically_compacted_events() -> None:
    compact = sliced(batch_inputs((10,)), 0, 7)
    positions = torch.tensor([0, 2, 3, 5, 7, 8, 10])
    padded_values: dict[str, Tensor] = {}
    for field in fields(compact):
        source = getattr(compact, field.name)
        if field.name == "valid_mask":
            target = torch.zeros(1, 11, dtype=torch.bool)
        else:
            target = torch.full((1, 11), 2**60, dtype=torch.int64)
        target[:, positions] = source
        padded_values[field.name] = target
    padded = BindingModelInputs(**padded_values)

    model = make_model().eval()
    compact_output = model(compact)
    padded_output = model(padded)
    assert torch.equal(
        compact_output.value_logits, padded_output.value_logits[:, positions]
    )
    assert not bool(
        padded_output.value_logits[:, ~padded.valid_mask[0]].any()
    )
    assert_state_equal(compact_output.final_state, padded_output.final_state)


def test_all_padding_and_empty_chunks_are_differentiable_storage_independent_noops() -> None:
    model = make_model()
    inputs = batch_inputs((10, 10))
    prefix = model(sliced(inputs, 0, 4)).final_state
    padding = replace(
        sliced(inputs, 4, 5),
        token_ids=torch.full((2, 1), 2**60, dtype=torch.int64),
        event_kinds=torch.full((2, 1), 2**60, dtype=torch.int64),
        primary_key_ids=torch.full((2, 1), -(2**60), dtype=torch.int64),
        secondary_key_ids=torch.full((2, 1), 2**60, dtype=torch.int64),
        arguments=torch.full((2, 1), -(2**60), dtype=torch.int64),
        valid_mask=torch.zeros(2, 1, dtype=torch.bool),
    )

    model.zero_grad(set_to_none=True)
    padded = model.step(padding, prefix)
    assert not bool(padded.value_logits.any())
    assert padded.value_logits.requires_grad
    padded.value_logits.sum().backward()
    assert all(
        parameter.grad is None or not bool(parameter.grad.any())
        for parameter in model.parameters()
    )
    assert_state_equal(prefix, padded.final_state)
    assert_state_storage_independent(prefix, padded.final_state)

    model.zero_grad(set_to_none=True)
    empty = model(sliced(inputs, 4, 4), prefix)
    assert empty.value_logits.shape == (2, 0, task().value_cardinality)
    assert empty.value_logits.requires_grad
    empty.value_logits.sum().backward()
    assert all(
        parameter.grad is None or not bool(parameter.grad.any())
        for parameter in model.parameters()
    )
    assert_state_equal(prefix, empty.final_state)
    assert_state_storage_independent(prefix, empty.final_state)


def test_mixed_batch_padding_preserves_inactive_row_during_capacity_growth() -> None:
    model = make_model()
    inputs = batch_inputs((10, 10))
    prefix = model(sliced(inputs, 0, 3)).final_state
    one = sliced(inputs, 3, 4)
    valid = one.valid_mask.clone()
    valid[1] = False
    one = replace(
        one,
        token_ids=one.token_ids.masked_fill(~valid, 2**60),
        event_kinds=one.event_kinds.masked_fill(~valid, 2**60),
        primary_key_ids=one.primary_key_ids.masked_fill(~valid, -(2**60)),
        secondary_key_ids=one.secondary_key_ids.masked_fill(~valid, 2**60),
        arguments=one.arguments.masked_fill(~valid, -(2**60)),
        valid_mask=valid,
    )
    output = model.step(one, prefix)

    assert output.final_state.valid_steps.tolist() == [4, 3]
    assert output.final_state.scales == 3
    assert torch.equal(
        output.final_state.slots[1, : prefix.scales], prefix.slots[1]
    )
    assert torch.equal(
        output.final_state.occupied[1, : prefix.scales], prefix.occupied[1]
    )
    assert not bool(output.final_state.slots[1, prefix.scales :].any())
    assert not bool(output.final_state.occupied[1, prefix.scales :].any())
    assert not bool(output.value_logits[1].any())


def test_valid_step_does_not_mutate_or_alias_source_state() -> None:
    model = make_model()
    inputs = batch_inputs((10, 10))
    prefix = model(sliced(inputs, 0, 3)).final_state
    snapshot = CausalTreeBindingState(
        prefix.slots.clone(), prefix.occupied.clone(), prefix.valid_steps.clone()
    )
    output = model.step(sliced(inputs, 3, 4), prefix).final_state

    assert_state_equal(prefix, snapshot)
    assert_state_storage_independent(prefix, output)


def test_suffix_gradients_flow_through_occupied_prefix_slots() -> None:
    model = make_model()
    inputs = batch_inputs((12,))
    prefix = model(sliced(inputs, 0, 6))
    prefix.final_state.slots.retain_grad()
    suffix = model(sliced(inputs, 6, 9), prefix.final_state)
    suffix.value_logits.square().sum().backward()

    gradient = prefix.final_state.slots.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    occupied_gradient = gradient.masked_select(
        prefix.final_state.occupied.unsqueeze(-1)
    )
    assert bool(occupied_gradient.any())


@pytest.mark.parametrize("length", [0, 1, 2, 3, 5, 7, 8, 9, 31, 32, 33, 65])
def test_state_is_minimal_canonical_binary_prefix(length: int) -> None:
    model = core_model(SumMerge())
    state = model.initial_state(1, device="cpu", dtype=torch.float64)
    for _ in range(length):
        state, _ = model._append(
            state, torch.ones(1, 1, dtype=torch.float64), torch.tensor([True])
        )
    expected = [bool((length >> scale) & 1) for scale in range(state.scales)]

    assert state.scales == max(1, length.bit_length())
    assert state.occupied[0].tolist() == expected
    assert int(state.occupied.sum()) == length.bit_count()
    assert not bool(state.slots.masked_select(~state.occupied.unsqueeze(-1)).any())


def test_state_validation_rejects_every_noncanonical_form() -> None:
    model = make_model(d_model=4)
    inputs = batch_inputs((10,))
    state = model(sliced(inputs, 0, 5)).final_state

    wrong_occupancy = state.occupied.clone()
    wrong_occupancy[0, 1] = True
    with pytest.raises(ValueError, match="occupancy"):
        model.step(sliced(inputs, 5, 6), replace(state, occupied=wrong_occupancy))

    with pytest.raises(ValueError, match="capacity"):
        model.step(
            sliced(inputs, 5, 6),
            replace(
                state,
                slots=torch.nn.functional.pad(state.slots, (0, 0, 0, 1)),
                occupied=torch.nn.functional.pad(state.occupied, (0, 1)),
            ),
        )
    with pytest.raises(ValueError, match="capacity"):
        model.step(
            sliced(inputs, 5, 6),
            replace(state, slots=state.slots[:, :2], occupied=state.occupied[:, :2]),
        )

    nonzero_unused = state.slots.clone()
    nonzero_unused[0, 1, 0] = 1.0
    with pytest.raises(ValueError, match="unoccupied"):
        model.step(sliced(inputs, 5, 6), replace(state, slots=nonzero_unused))

    nonfinite = state.slots.clone()
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model.step(sliced(inputs, 5, 6), replace(state, slots=nonfinite))

    with pytest.raises(ValueError, match="nonnegative"):
        model.step(
            sliced(inputs, 5, 6),
            replace(state, valid_steps=torch.tensor([-1], dtype=torch.int64)),
        )
    with pytest.raises(ValueError, match="int64"):
        model.step(
            sliced(inputs, 5, 6),
            replace(state, valid_steps=state.valid_steps.to(torch.int32)),
        )


def test_int64_overflow_is_rejected_and_padding_at_max_is_a_noop() -> None:
    model = make_model(d_model=4)
    maximum = torch.iinfo(torch.int64).max
    state = CausalTreeBindingState(
        slots=torch.ones(1, 63, 4, dtype=torch.float64),
        occupied=torch.ones(1, 63, dtype=torch.bool),
        valid_steps=torch.tensor([maximum], dtype=torch.int64),
    )
    one = sliced(batch_inputs((10,)), 0, 1)
    with pytest.raises(ValueError, match="overflow"):
        model.step(one, state)

    padding = replace(one, valid_mask=torch.zeros_like(one.valid_mask))
    output = model.step(padding, state)
    assert not bool(output.value_logits.any())
    assert_state_equal(output.final_state, state)
    assert_state_storage_independent(output.final_state, state)


def test_forward_boundary_has_no_routes_branches_labels_or_evaluation() -> None:
    model = make_model()
    inputs = batch_inputs((10,))
    assert set(inspect.signature(model.forward).parameters) == {
        "inputs",
        "initial_state",
    }
    assert set(inspect.signature(model.step).parameters) == {"inputs", "state"}
    assert {field.name for field in fields(BaselineBindingOutput)} == {
        "value_logits",
        "final_state",
        "diagnostics",
    }
    assert {field.name for field in fields(CausalTreeBindingState)} == {
        "slots",
        "occupied",
        "valid_steps",
    }
    assert (
        get_type_hints(model.step)["return"]
        == BaselineBindingOutput[CausalTreeBindingState]
    )
    assert not hasattr(model, "router")
    assert all(
        word not in name
        for name in model.state_dict()
        for word in ("route", "branch", "target", "evaluation")
    )
    with pytest.raises(TypeError):
        model(inputs, route_labels=torch.zeros_like(inputs.token_ids))


def test_historical_length_tied_construction_cannot_return() -> None:
    short_config = CausalTreeBindingBaselineConfig(
        task(max_length=18), d_model=4, cp_rank=3, scale_feature_dim=4
    )
    long_config = CausalTreeBindingBaselineConfig(
        task(max_length=4096), d_model=4, cp_rank=3, scale_feature_dim=4
    )
    assert short_config.canonical_json() == long_config.canonical_json()
    assert short_config.fingerprint() == long_config.fingerprint()
    assert all(
        "length" not in field.name and "leaves" not in field.name
        for field in fields(CausalTreeBindingBaselineConfig)
    )

    torch.manual_seed(12)
    source = CausalCompleteTreeBindingBaseline(short_config)
    checkpoint = copy.deepcopy(source.state_dict())
    model = CausalCompleteTreeBindingBaseline(long_config)
    model.load_state_dict(checkpoint, strict=True)
    keys_before = tuple(model.state_dict())
    count_before = sum(parameter.numel() for parameter in model.parameters())
    assert sum(
        isinstance(module, ScaleSharedCPMerge) for module in model.modules()
    ) == 1
    assert not any(
        "level" in name or "position" in name
        for name, _ in (*model.named_parameters(), *model.named_buffers())
    )

    one = sliced(batch_inputs((10,)), 0, 1)
    long_inputs = BindingModelInputs(
        **{
            field.name: getattr(one, field.name).repeat(1, 257)
            for field in fields(one)
        }
    )
    with torch.no_grad():
        output = model(long_inputs)
    assert output.value_logits.shape == (1, 257, task().value_cardinality)
    assert output.final_state.scales == (257).bit_length()
    assert torch.isfinite(output.value_logits).all()
    assert tuple(model.state_dict()) == keys_before
    assert sum(parameter.numel() for parameter in model.parameters()) == count_before


def test_diagnostics_and_structural_metrics_report_real_tree_work_and_state() -> None:
    model = make_model(d_model=4, cp_rank=3)
    inputs = sliced(batch_inputs((10,)), 0, 7)
    output = model(inputs)

    assert output.final_state.valid_steps.tolist() == [7]
    assert output.final_state.occupied.tolist() == [[True, True, True]]
    assert int(output.diagnostics["update_merge_count"]) == 4
    assert int(output.diagnostics["readout_merge_count"]) == 5
    assert int(output.diagnostics["active_slots"]) == 3
    assert int(output.diagnostics["allocated_scales"]) == 3

    total_merges = sum(
        int(output.diagnostics[key])
        for key in ("update_merge_count", "readout_merge_count")
    )
    metrics = model.structural_metrics(
        output.final_state, merge_count=total_merges
    )
    assert metrics["parameter_count"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert metrics["state_scalars_per_occupied_slot"] == 4
    assert metrics["occupancy_scalars_per_allocated_slot"] == 1
    assert metrics["state_counter_scalars_per_batch_row"] == 1
    assert metrics["tree_lanes"] == 1
    assert metrics["active_slots"] == 3
    assert metrics["allocated_slots"] == 3
    assert metrics["active_state_elements"] == 12
    assert metrics["allocated_state_elements"] == 12
    assert metrics["active_state_logical_scalars"] == 16
    assert metrics["allocated_state_logical_scalars"] == 16
    assert metrics["active_state_bytes"] == 107
    assert metrics["allocated_state_bytes"] == 107
    assert metrics["operation_count_proxy"] == (
        metrics["operation_count_proxy_per_merge"] * total_merges
    )

    with pytest.raises(TypeError):
        model.structural_metrics(merge_count=True)
    with pytest.raises(TypeError):
        model.structural_metrics(merge_count=1.5)
    with pytest.raises(ValueError):
        model.structural_metrics(merge_count=-1)


def test_real_model_has_finite_gradients_for_every_parameter_group() -> None:
    model = make_model()
    output = model(batch_inputs((10, 10)))
    output.value_logits.square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
