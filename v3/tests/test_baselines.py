from __future__ import annotations

from dataclasses import fields, replace
import inspect

import pytest
import torch

from tnlm_v3.baselines import (
    BaselineBindingOutput,
    CachedCausalTransformerBindingBaseline,
    CachedTransformerBindingBaselineConfig,
    CachedTransformerBindingState,
    RecurrentBindingBaseline,
    RecurrentBindingBaselineConfig,
    RecurrentBindingState,
)
from tnlm_v3.binding import BindingArchitectureConfig
from tnlm_v3.data import (
    BindingModelInputs,
    BindingTaskConfig,
    collate_binding_episodes,
    generate_binding_episodes,
)


def task(*, max_length: int = 24) -> BindingTaskConfig:
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


def batch_inputs(lengths: tuple[int, ...] = (12, 10)) -> BindingModelInputs:
    episodes = generate_binding_episodes(
        task(max_length=max(lengths)),
        count=len(lengths),
        seed=9102,
        split="train",
        lengths=lengths,
    )
    return collate_binding_episodes(episodes).inputs


def sliced(inputs: BindingModelInputs, start: int, stop: int) -> BindingModelInputs:
    return BindingModelInputs(
        **{field.name: getattr(inputs, field.name)[:, start:stop] for field in fields(inputs)}
    )


def models() -> tuple[RecurrentBindingBaseline, CachedCausalTransformerBindingBaseline]:
    architecture = BindingArchitectureConfig.from_task(task())
    torch.manual_seed(123)
    gru = RecurrentBindingBaseline(
        RecurrentBindingBaselineConfig(architecture, d_model=12, hidden_dim=16, num_layers=2)
    )
    torch.manual_seed(123)
    transformer = CachedCausalTransformerBindingBaseline(
        CachedTransformerBindingBaselineConfig(
            architecture, d_model=12, num_heads=3, num_layers=2, ff_dim=20
        )
    )
    return gru, transformer


@pytest.mark.parametrize(
    "config",
    [
        RecurrentBindingBaselineConfig(task()),
        CachedTransformerBindingBaselineConfig(task()),
    ],
)
def test_configs_are_sanitized_length_independent_and_hashable(config: object) -> None:
    assert isinstance(config.task, BindingArchitectureConfig)
    assert set(config.task.__dict__) == {
        "num_surface_keys",
        "value_cardinality",
        "branches",
    }
    canonical = config.canonical_json()
    assert "heldout" not in canonical and "max_length" not in canonical
    assert config.fingerprint() == config.fingerprint()
    assert len(config.fingerprint()) == 64


@pytest.mark.parametrize(
    ("constructor", "changes", "error"),
    [
        (RecurrentBindingBaselineConfig, {"d_model": True}, ValueError),
        (RecurrentBindingBaselineConfig, {"num_layers": 0}, ValueError),
        (CachedTransformerBindingBaselineConfig, {"num_heads": 0}, ValueError),
        (CachedTransformerBindingBaselineConfig, {"d_model": 10, "num_heads": 3}, ValueError),
        (CachedTransformerBindingBaselineConfig, {"ff_dim": "12"}, ValueError),
    ],
)
def test_configs_reject_invalid_dimensions(constructor: type, changes: dict, error: type) -> None:
    with pytest.raises(error):
        constructor(task(), **changes)


@pytest.mark.parametrize("model", models())
def test_full_step_and_chunked_resume_agree(model: torch.nn.Module) -> None:
    inputs = batch_inputs()
    model.eval()
    full = model(inputs)
    state = model.initial_state(
        inputs.token_ids.shape[0],
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )
    pieces = []
    for index in range(inputs.token_ids.shape[1]):
        output = model.step(sliced(inputs, index, index + 1), state)
        assert isinstance(output, BaselineBindingOutput)
        state = output.final_state
        pieces.append(output.value_logits)
    stepped = torch.cat(pieces, dim=1)
    assert torch.equal(full.value_logits, stepped)
    cut = 5
    prefix = model(sliced(inputs, 0, cut))
    suffix = model(sliced(inputs, cut, inputs.token_ids.shape[1]), prefix.final_state)
    assert torch.equal(full.value_logits[:, :cut], prefix.value_logits)
    assert torch.equal(full.value_logits[:, cut:], suffix.value_logits)
    assert torch.equal(full.final_state.valid_steps, suffix.final_state.valid_steps)
    if isinstance(full.final_state, RecurrentBindingState):
        assert torch.equal(full.final_state.hidden, suffix.final_state.hidden)
    else:
        assert torch.equal(full.final_state.occupied, suffix.final_state.occupied)
        for left, right in zip(full.final_state.keys, suffix.final_state.keys, strict=True):
            assert torch.equal(left, right)
        for left, right in zip(full.final_state.values, suffix.final_state.values, strict=True):
            assert torch.equal(left, right)


@pytest.mark.parametrize("model", models())
def test_float64_full_step_and_chunked_resume_agree(model: torch.nn.Module) -> None:
    model = model.double().eval()
    inputs = batch_inputs((12, 10))
    full = model(inputs)
    prefix = model(sliced(inputs, 0, 5))
    suffix = model(sliced(inputs, 5, 12), prefix.final_state)
    state = model.initial_state(2, device="cpu", dtype=torch.float64)
    pieces: list[torch.Tensor] = []
    for index in range(12):
        result = model.step(sliced(inputs, index, index + 1), state)
        state = result.final_state
        pieces.append(result.value_logits)
    assert torch.equal(full.value_logits, torch.cat(pieces, dim=1))
    assert torch.equal(full.value_logits[:, 5:], suffix.value_logits)
    assert torch.equal(full.final_state.valid_steps, suffix.final_state.valid_steps)


@pytest.mark.parametrize("model", models())
def test_prefix_is_independent_of_future_visible_fields(model: torch.nn.Module) -> None:
    inputs = batch_inputs((12, 12))
    cut = 6
    changed = BindingModelInputs(
        **{
            field.name: (
                torch.cat(
                    (
                        getattr(inputs, field.name)[:, :cut],
                        torch.flip(getattr(inputs, field.name)[:, cut:], dims=(1,)),
                    ),
                    dim=1,
                )
                if field.name != "valid_mask"
                else getattr(inputs, field.name)
            )
            for field in fields(inputs)
        }
    )
    model.eval()
    assert torch.equal(model(inputs).value_logits[:, :cut], model(changed).value_logits[:, :cut])


@pytest.mark.parametrize("model", models())
def test_padding_garbage_is_total_forward_and_backward_noop(model: torch.nn.Module) -> None:
    inputs = batch_inputs((12, 10))
    garbage_values = {
        "token_ids": 2**60,
        "event_kinds": 2**60,
        "primary_key_ids": -(2**60),
        "secondary_key_ids": 2**60,
        "arguments": -(2**60),
    }
    values = {}
    for field in fields(inputs):
        tensor = getattr(inputs, field.name).clone()
        if field.name != "valid_mask":
            tensor[~inputs.valid_mask] = garbage_values[field.name]
        values[field.name] = tensor
    garbage = BindingModelInputs(**values)
    model.zero_grad(set_to_none=True)
    first = model(inputs)
    first.value_logits.square().sum().backward()
    gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
    model.zero_grad(set_to_none=True)
    second = model(garbage)
    second.value_logits.square().sum().backward()
    assert torch.equal(first.value_logits, second.value_logits)
    assert torch.equal(first.final_state.valid_steps, second.final_state.valid_steps)
    for expected, parameter in zip(gradients, model.parameters(), strict=True):
        assert torch.equal(expected, parameter.grad)


@pytest.mark.parametrize("model", models())
def test_all_padding_step_is_exact_state_noop(model: torch.nn.Module) -> None:
    valid = batch_inputs((10, 10))
    prefix = model(sliced(valid, 0, 4))
    pad = BindingModelInputs(
        token_ids=torch.full((2, 1), 2**60, dtype=torch.int64),
        event_kinds=torch.full((2, 1), 2**60, dtype=torch.int64),
        primary_key_ids=torch.full((2, 1), -(2**60), dtype=torch.int64),
        secondary_key_ids=torch.full((2, 1), 2**60, dtype=torch.int64),
        arguments=torch.full((2, 1), -(2**60), dtype=torch.int64),
        valid_mask=torch.zeros(2, 1, dtype=torch.bool),
    )
    model.zero_grad(set_to_none=True)
    output = model.step(pad, prefix.final_state)
    assert not bool(output.value_logits.any())
    assert output.diagnostics["valid_events"].item() == 0
    output.value_logits.sum().backward()
    assert all(
        parameter.grad is None or not bool(parameter.grad.any())
        for parameter in model.parameters()
    )
    assert torch.equal(output.final_state.valid_steps, prefix.final_state.valid_steps)
    assert output.final_state.valid_steps.data_ptr() != prefix.final_state.valid_steps.data_ptr()
    if isinstance(prefix.final_state, RecurrentBindingState):
        assert torch.equal(output.final_state.hidden, prefix.final_state.hidden)
        assert output.final_state.hidden.data_ptr() != prefix.final_state.hidden.data_ptr()
    else:
        assert output.final_state.occupied.shape == prefix.final_state.occupied.shape
        assert torch.equal(output.final_state.occupied, prefix.final_state.occupied)
        assert output.final_state.occupied.data_ptr() != prefix.final_state.occupied.data_ptr()
        for before, after in zip(prefix.final_state.keys, output.final_state.keys, strict=True):
            assert torch.equal(before, after)
            assert before.data_ptr() != after.data_ptr()
        for before, after in zip(prefix.final_state.values, output.final_state.values, strict=True):
            assert torch.equal(before, after)
            assert before.data_ptr() != after.data_ptr()


@pytest.mark.parametrize("model", models())
def test_empty_forward_is_differentiable_independent_state_noop(
    model: torch.nn.Module,
) -> None:
    inputs = batch_inputs((10, 10))
    prefix = model(sliced(inputs, 0, 3))
    model.zero_grad(set_to_none=True)
    output = model(sliced(inputs, 3, 3), prefix.final_state)
    assert output.value_logits.shape == (2, 0, model.config.task.value_cardinality)
    assert output.value_logits.requires_grad
    assert output.diagnostics["valid_events"].item() == 0
    output.value_logits.sum().backward()
    assert all(
        parameter.grad is None or not bool(parameter.grad.any())
        for parameter in model.parameters()
    )
    assert output.final_state is not prefix.final_state
    assert torch.equal(output.final_state.valid_steps, prefix.final_state.valid_steps)
    assert output.final_state.valid_steps.data_ptr() != prefix.final_state.valid_steps.data_ptr()
    if isinstance(prefix.final_state, RecurrentBindingState):
        assert torch.equal(output.final_state.hidden, prefix.final_state.hidden)
        assert output.final_state.hidden.data_ptr() != prefix.final_state.hidden.data_ptr()
    else:
        assert torch.equal(output.final_state.occupied, prefix.final_state.occupied)
        assert output.final_state.occupied.data_ptr() != prefix.final_state.occupied.data_ptr()
        for before, after in zip(
            (*prefix.final_state.keys, *prefix.final_state.values),
            (*output.final_state.keys, *output.final_state.values),
            strict=True,
        ):
            assert torch.equal(before, after)
            assert before.data_ptr() != after.data_ptr()


def test_transformer_cache_is_per_layer_and_grows_by_real_events() -> None:
    transformer = models()[1]
    inputs = batch_inputs((12, 10))
    output = transformer(inputs)
    state = output.final_state
    assert isinstance(state, CachedTransformerBindingState)
    assert len(state.keys) == len(state.values) == transformer.config.num_layers
    assert state.occupied.shape == (2, 12)
    assert state.valid_steps.tolist() == [12, 10]
    assert state.occupied.sum(1).tolist() == [12, 10]
    expected = (
        2,
        transformer.config.num_heads,
        12,
        transformer.config.d_model // transformer.config.num_heads,
    )
    assert all(tensor.shape == expected for tensor in (*state.keys, *state.values))


def test_transformer_output_depends_on_cached_prefix() -> None:
    transformer = models()[1]
    inputs = batch_inputs((12, 12))
    first = transformer(sliced(inputs, 0, 6))
    altered = replace(
        first.final_state,
        values=tuple(torch.zeros_like(value) for value in first.final_state.values),
    )
    normal = transformer(sliced(inputs, 6, 7), first.final_state)
    changed = transformer(sliced(inputs, 6, 7), altered)
    assert not torch.equal(normal.value_logits, changed.value_logits)


def test_transformer_suffix_gradients_flow_through_every_prefix_cache() -> None:
    transformer = models()[1]
    inputs = batch_inputs((12, 12))
    prefix = transformer(sliced(inputs, 0, 6))
    for item in (*prefix.final_state.keys, *prefix.final_state.values):
        item.retain_grad()
    suffix = transformer(sliced(inputs, 6, 8), prefix.final_state)
    suffix.value_logits.square().sum().backward()
    for item in (*prefix.final_state.keys, *prefix.final_state.values):
        assert item.grad is not None
        assert bool(torch.isfinite(item.grad).all())
        assert bool(item.grad.any())


def test_transformer_rejects_noncanonical_cache_occupancy_and_capacity() -> None:
    transformer = models()[1]
    inputs = batch_inputs((10, 10))
    state = transformer(sliced(inputs, 0, 3)).final_state
    hole = state.occupied.clone()
    hole[0] = torch.tensor([True, False, True])
    with pytest.raises(ValueError, match="packed prefix"):
        transformer.step(sliced(inputs, 3, 4), replace(state, occupied=hole))
    padded_keys = tuple(torch.nn.functional.pad(value, (0, 0, 0, 1)) for value in state.keys)
    padded_values = tuple(torch.nn.functional.pad(value, (0, 0, 0, 1)) for value in state.values)
    with pytest.raises(ValueError, match="capacity"):
        transformer.step(
            sliced(inputs, 3, 4),
            replace(
                state,
                keys=padded_keys,
                values=padded_values,
                occupied=torch.nn.functional.pad(state.occupied, (0, 1)),
            ),
        )


def test_transformer_rejects_payload_in_unoccupied_cache_entries() -> None:
    transformer = models()[1]
    inputs = batch_inputs((10, 10))
    state = transformer(sliced(inputs, 0, 3)).final_state
    mixed_counts = state.valid_steps.clone()
    mixed_counts[1] = 1
    mixed_occupied = state.occupied.clone()
    mixed_occupied[1, 1:] = False
    bad_keys = tuple(item.clone() for item in state.keys)
    bad_keys[0][1, :, 1:, :] = 12345.0
    malformed = replace(
        state,
        keys=bad_keys,
        occupied=mixed_occupied,
        valid_steps=mixed_counts,
    )
    with pytest.raises(ValueError, match="unoccupied cache entries must be zero"):
        transformer.step(sliced(inputs, 3, 4), malformed)


def test_baselines_reject_nonfinite_persistent_state() -> None:
    gru, transformer = models()
    inputs = batch_inputs((10, 10))
    gru_state = gru(sliced(inputs, 0, 3)).final_state
    bad_hidden = gru_state.hidden.clone()
    bad_hidden[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="recurrent state is invalid"):
        gru.step(sliced(inputs, 3, 4), replace(gru_state, hidden=bad_hidden))

    transformer_state = transformer(sliced(inputs, 0, 3)).final_state
    bad_values = tuple(item.clone() for item in transformer_state.values)
    bad_values[0][0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="cached key/value tensor is invalid"):
        transformer.step(
            sliced(inputs, 3, 4),
            replace(transformer_state, values=bad_values),
        )


@pytest.mark.parametrize("model", models())
def test_valid_step_returns_storage_independent_state(model: torch.nn.Module) -> None:
    inputs = batch_inputs((10, 10))
    prefix = model(sliced(inputs, 0, 3)).final_state
    if isinstance(prefix, RecurrentBindingState):
        snapshots = (prefix.hidden.clone(), prefix.valid_steps.clone())
    else:
        snapshots = tuple(
            item.clone()
            for item in (
                *prefix.keys,
                *prefix.values,
                prefix.occupied,
                prefix.valid_steps,
            )
        )
    output = model.step(sliced(inputs, 3, 4), prefix).final_state
    assert output is not prefix
    assert output.valid_steps.data_ptr() != prefix.valid_steps.data_ptr()
    if isinstance(prefix, RecurrentBindingState):
        assert isinstance(output, RecurrentBindingState)
        assert output.hidden.data_ptr() != prefix.hidden.data_ptr()
        assert torch.equal(prefix.hidden, snapshots[0])
        assert torch.equal(prefix.valid_steps, snapshots[1])
    else:
        assert isinstance(output, CachedTransformerBindingState)
        current = (*prefix.keys, *prefix.values, prefix.occupied, prefix.valid_steps)
        returned = (*output.keys, *output.values, output.occupied, output.valid_steps)
        for before, after, snapshot in zip(current, returned, snapshots, strict=True):
            assert before.data_ptr() != after.data_ptr()
            assert torch.equal(before, snapshot)


@pytest.mark.parametrize("model", models())
def test_state_counters_reject_int64_overflow(model: torch.nn.Module) -> None:
    inputs = sliced(batch_inputs((10, 10)), 0, 1)
    state = model.initial_state(2, device="cpu", dtype=next(model.parameters()).dtype)
    state.valid_steps[:] = torch.iinfo(torch.int64).max
    with pytest.raises(ValueError, match="overflow"):
        model.step(inputs, state)


def test_recurrent_max_counter_still_allows_padding_noop() -> None:
    model = models()[0]
    state = model.initial_state(2, device="cpu", dtype=next(model.parameters()).dtype)
    state.valid_steps[:] = torch.iinfo(torch.int64).max
    inputs = sliced(batch_inputs((10, 10)), 0, 1)
    padded = replace(inputs, valid_mask=torch.zeros_like(inputs.valid_mask))
    output = model.step(padded, state)
    assert output.final_state is not state
    assert torch.equal(output.final_state.hidden, state.hidden)
    assert torch.equal(output.final_state.valid_steps, state.valid_steps)
    assert output.diagnostics["valid_events"].item() == 0


@pytest.mark.parametrize("model", models())
def test_gradients_are_finite_and_parameter_count_is_length_independent(
    model: torch.nn.Module,
) -> None:
    short = batch_inputs((10, 10))
    long = batch_inputs((24, 24))
    count = sum(parameter.numel() for parameter in model.parameters())
    output = model(short)
    output.value_logits.sum().backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == count
    assert model(long).value_logits.shape == (2, 24, model.config.task.value_cardinality)
    assert not any("length" in name for name, _ in model.named_parameters())
    metrics = model.structural_metrics()
    assert metrics["parameter_count"] == count


def test_forward_api_cannot_accept_labels_routes_or_evaluation_metadata() -> None:
    for model in models():
        parameters = inspect.signature(model.forward).parameters
        assert set(parameters) == {"inputs", "initial_state"}
        assert all(word not in parameters for word in ("routes", "targets", "labels", "evaluation"))
