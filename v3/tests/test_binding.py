from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from tnlm_v3.binding import BindingArchitectureConfig, BindingModelConfig, RoutedBindingModel
from tnlm_v3.data import (
    BindingEventKind,
    BindingModelInputs,
    BindingTaskConfig,
    collate_binding_episodes,
    generate_binding_episodes,
)
from tnlm_v3.routing import CurriculumSchedule, RoutingMode, route_counts


def make_task(**overrides) -> BindingTaskConfig:
    values = dict(
        num_surface_keys=6,
        value_cardinality=5,
        branches=3,
        max_live_bindings=3,
        min_length=10,
        max_length=18,
        heldout_key_value_pairs=((0, 0),),
    )
    values.update(overrides)
    return BindingTaskConfig(**values)


def make_batch(task: BindingTaskConfig, *, count=2, length=14, split="train"):
    episodes = generate_binding_episodes(
        task,
        count=count,
        seed=31415,
        split=split,
        lengths=[length] * count,
    )
    return collate_binding_episodes(episodes)


def make_config(mode: RoutingMode, *, task=None) -> BindingModelConfig:
    return BindingModelConfig(
        task=task or make_task(),
        d_model=8,
        cp_rank=4,
        router_hidden_dim=7,
        routing_mode=mode,
        curriculum_schedule=(
            CurriculumSchedule(0, 20, 1.0, 0.0)
            if mode is RoutingMode.CURRICULUM
            else None
        ),
        curriculum_seed=91,
    )


def test_oracle_routes_are_operationally_applied_to_forest_counts():
    torch.manual_seed(1)
    task = make_task()
    batch = make_batch(task)
    model = RoutedBindingModel(make_config(RoutingMode.ORACLE, task=task)).double()
    output = model(batch.inputs, route_labels=batch.evaluation.oracle_routes)

    assert torch.equal(output.routes, batch.evaluation.oracle_routes)
    expected = route_counts(output.routes, batch.inputs.valid_mask, task.branches)
    assert torch.equal(output.forest_state.counts, expected)
    assert int(output.diagnostics["forest_active_slots"]) == int(
        output.forest_state.occupied.sum()
    )


def test_curriculum_evaluation_is_bit_exact_label_independent():
    torch.manual_seed(2)
    task = make_task()
    batch = make_batch(task, split="eval")
    model = RoutedBindingModel(make_config(RoutingMode.CURRICULUM, task=task)).double()
    model.eval()
    first = model(
        batch.inputs,
        route_labels=batch.evaluation.oracle_routes,
        training_step=0,
    )
    poisoned = torch.full_like(batch.evaluation.oracle_routes, 987654321)
    second = model(batch.inputs, route_labels=poisoned, training_step=19)

    assert torch.equal(first.routes, second.routes)
    torch.testing.assert_close(first.route_logits, second.route_logits, rtol=0, atol=0)
    torch.testing.assert_close(first.value_logits, second.value_logits, rtol=0, atol=0)
    assert int(first.diagnostics["guided_events"]) == 0


def test_latent_mode_rejects_route_labels_and_never_has_guidance():
    task = make_task()
    batch = make_batch(task)
    model = RoutedBindingModel(make_config(RoutingMode.LATENT, task=task)).double()
    with pytest.raises(ValueError, match="must not receive"):
        model(batch.inputs, route_labels=batch.evaluation.oracle_routes)
    output = model(batch.inputs)
    assert int(output.diagnostics["guided_events"]) == 0
    assert float(output.diagnostics["guidance_probability"]) == 0.0


@pytest.mark.parametrize("mode", [RoutingMode.CURRICULUM, RoutingMode.LATENT])
def test_autonomous_routes_and_logits_are_prefix_causal(mode):
    torch.manual_seed(3)
    task = make_task()
    batch = make_batch(task, count=1, length=18)
    model = RoutedBindingModel(make_config(mode, task=task)).double().eval()
    baseline = model(batch.inputs)
    boundary = 9
    inputs = batch.inputs
    changed = BindingModelInputs(
        token_ids=inputs.token_ids.clone(),
        event_kinds=inputs.event_kinds.clone(),
        primary_key_ids=inputs.primary_key_ids.clone(),
        secondary_key_ids=inputs.secondary_key_ids.clone(),
        arguments=inputs.arguments.clone(),
        valid_mask=inputs.valid_mask.clone(),
    )
    changed.token_ids[:, boundary:] = torch.flip(changed.token_ids[:, boundary:], (1,))
    changed.event_kinds[:, boundary:] = int(BindingEventKind.DISTRACTOR)
    changed.primary_key_ids[:, boundary:] = 0
    changed.secondary_key_ids[:, boundary:] = 0
    changed.arguments[:, boundary:] = 0
    altered = model(changed)

    assert torch.equal(baseline.routes[:, :boundary], altered.routes[:, :boundary])
    torch.testing.assert_close(
        baseline.route_logits[:, :boundary],
        altered.route_logits[:, :boundary],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        baseline.value_logits[:, :boundary],
        altered.value_logits[:, :boundary],
        rtol=0,
        atol=0,
    )


def test_padding_payload_is_a_total_model_noop():
    torch.manual_seed(4)
    task = make_task(max_length=18)
    episodes = generate_binding_episodes(
        task, count=1, seed=22, split="eval", lengths=[10]
    )
    compact = collate_binding_episodes(episodes)
    padded = collate_binding_episodes(episodes, pad_to_length=18)
    mask = ~padded.inputs.valid_mask
    corrupted = BindingModelInputs(
        token_ids=padded.inputs.token_ids.masked_fill(mask, 999999),
        event_kinds=padded.inputs.event_kinds.masked_fill(mask, -999),
        primary_key_ids=padded.inputs.primary_key_ids.masked_fill(mask, 8888),
        secondary_key_ids=padded.inputs.secondary_key_ids.masked_fill(mask, -8888),
        arguments=padded.inputs.arguments.masked_fill(mask, 7777),
        valid_mask=padded.inputs.valid_mask,
    )
    model = RoutedBindingModel(make_config(RoutingMode.LATENT, task=task)).double().eval()
    left = model(compact.inputs)
    right = model(corrupted)
    length = compact.padded_length

    assert torch.equal(left.routes, right.routes[:, :length])
    torch.testing.assert_close(left.value_logits, right.value_logits[:, :length], rtol=0, atol=0)
    torch.testing.assert_close(left.forest_state.slots, right.forest_state.slots, rtol=0, atol=0)
    assert torch.equal(left.router_state.valid_steps, right.router_state.valid_steps)


@pytest.mark.parametrize("implementation", ["streaming", "parallel"])
def test_straight_through_surrogate_carries_query_gradient_to_latent_router(
    implementation,
):
    torch.manual_seed(5)
    task = make_task()
    batch = make_batch(task, count=2, length=14)
    model = RoutedBindingModel(make_config(RoutingMode.LATENT, task=task)).double()
    output = model(batch.inputs, implementation=implementation)
    query_mask = batch.inputs.event_kinds == int(BindingEventKind.QUERY)
    loss = F.cross_entropy(
        output.value_logits[query_mask], batch.evaluation.targets[query_mask]
    )
    loss.backward()

    router_gradients = [parameter.grad for parameter in model.router.parameters()]
    assert any(
        gradient is not None and bool((gradient.abs() > 0).any())
        for gradient in router_gradients
    )


@pytest.mark.parametrize("mode", [RoutingMode.ORACLE, RoutingMode.LATENT])
def test_binding_streaming_parallel_parity_in_evaluation(mode):
    torch.manual_seed(6)
    task = make_task()
    batch = make_batch(task, count=2, length=14)
    model = RoutedBindingModel(make_config(mode, task=task)).double().eval()
    labels = batch.evaluation.oracle_routes if mode is RoutingMode.ORACLE else None
    streaming = model(batch.inputs, route_labels=labels, implementation="streaming")
    parallel = model(batch.inputs, route_labels=labels, implementation="parallel")

    assert torch.equal(streaming.routes, parallel.routes)
    torch.testing.assert_close(streaming.value_logits, parallel.value_logits, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(streaming.forest_state.slots, parallel.forest_state.slots, rtol=1e-9, atol=1e-10)


def test_data_max_length_does_not_change_model_parameterization():
    short = make_config(RoutingMode.LATENT, task=make_task(max_length=18))
    long = make_config(RoutingMode.LATENT, task=make_task(max_length=2048))
    short_model = RoutedBindingModel(short)
    long_model = RoutedBindingModel(long)
    short_shapes = {name: tuple(value.shape) for name, value in short_model.state_dict().items()}
    long_shapes = {name: tuple(value.shape) for name, value in long_model.state_dict().items()}
    assert short_shapes == long_shapes
    assert sum(p.numel() for p in short_model.parameters()) == sum(
        p.numel() for p in long_model.parameters()
    )


def test_model_config_drops_generator_split_metadata() -> None:
    task = make_task()
    config = make_config(RoutingMode.LATENT, task=task)
    assert isinstance(config.task, BindingArchitectureConfig)
    assert not hasattr(config.task, "heldout_key_value_pairs")
    assert not hasattr(config.task, "global_distractor_probability")
    assert config.task.vocab_size == task.vocab_size


def test_router_feature_consumes_visible_distractor_scope_bit() -> None:
    task = make_task()
    model = RoutedBindingModel(make_config(RoutingMode.LATENT, task=task)).double()
    inputs = BindingModelInputs(
        token_ids=torch.ones(1, 1, dtype=torch.int64),
        event_kinds=torch.full((1, 1), int(BindingEventKind.DISTRACTOR)),
        primary_key_ids=torch.zeros(1, 1, dtype=torch.int64),
        secondary_key_ids=torch.zeros(1, 1, dtype=torch.int64),
        arguments=torch.ones(1, 1, dtype=torch.int64),
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    changed = replace(inputs, arguments=torch.full((1, 1), 2, dtype=torch.int64))
    _, first = model.encoder(inputs)
    _, second = model.encoder(changed)
    assert not torch.equal(first, second)
