from __future__ import annotations

from dataclasses import replace

import torch

from tnlm_v3.binding import BindingModelConfig, RoutedBindingModel
from tnlm_v3.data import (
    BindingEvaluation,
    BindingModelInputs,
    BindingTaskConfig,
    collate_binding_episodes,
    generate_binding_episodes,
)
from tnlm_v3.routing import CurriculumSchedule, RoutingMode
from tnlm_v3.training import (
    BindingLossConfig,
    compute_binding_loss,
    evaluate_binding_model,
    train_binding_step,
)


class _NoRouteLabels:
    def __init__(self, evaluation: BindingEvaluation) -> None:
        self._evaluation = evaluation
        self.reads = 0

    @property
    def oracle_routes(self):
        self.reads += 1
        raise AssertionError("latent training accessed route labels")

    def __getattr__(self, name):
        return getattr(self._evaluation, name)


def fixture(mode: RoutingMode):
    task = BindingTaskConfig(
        num_surface_keys=5,
        value_cardinality=4,
        branches=3,
        max_live_bindings=3,
        min_length=10,
        max_length=14,
        heldout_key_value_pairs=((0, 0),),
    )
    batch = collate_binding_episodes(
        generate_binding_episodes(
            task, count=2, seed=177, split="train", lengths=[12, 14]
        )
    )
    config = BindingModelConfig(
        task=task,
        d_model=8,
        cp_rank=4,
        router_hidden_dim=7,
        routing_mode=mode,
        curriculum_schedule=(
            CurriculumSchedule(0, 10) if mode is RoutingMode.CURRICULUM else None
        ),
    )
    return batch, RoutedBindingModel(config).double()


def test_latent_loss_is_route_label_independent_and_has_zero_supervision():
    torch.manual_seed(10)
    batch, model = fixture(RoutingMode.LATENT)
    output = model(batch.inputs)
    first = compute_binding_loss(
        output,
        batch.inputs,
        batch.evaluation,
        routing_mode=RoutingMode.LATENT,
    )
    poisoned = replace(
        batch.evaluation,
        oracle_routes=torch.full_like(batch.evaluation.oracle_routes, 123456),
    )
    second = compute_binding_loss(
        output,
        batch.inputs,
        poisoned,
        routing_mode=RoutingMode.LATENT,
    )
    torch.testing.assert_close(first.total, second.total, rtol=0, atol=0)
    assert first.route_curriculum.item() == 0.0
    assert first.route_supervision_count == 0


def test_latent_training_does_not_access_route_label_property():
    torch.manual_seed(101)
    batch, model = fixture(RoutingMode.LATENT)
    guarded = _NoRouteLabels(batch.evaluation)
    guarded_batch = replace(batch, evaluation=guarded)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _, loss = train_binding_step(model, guarded_batch, optimizer, training_step=1)
    assert guarded.reads == 0
    assert loss.route_supervision_count == 0
    assert torch.isfinite(loss.total)


def test_curriculum_route_objective_has_finite_router_gradients():
    torch.manual_seed(11)
    batch, model = fixture(RoutingMode.CURRICULUM)
    output = model(
        batch.inputs, route_labels=batch.evaluation.oracle_routes, training_step=0
    )
    loss = compute_binding_loss(
        output,
        batch.inputs,
        batch.evaluation,
        routing_mode=RoutingMode.CURRICULUM,
    )
    loss.total.backward()
    assert loss.route_supervision_count == int(batch.inputs.valid_mask.sum())
    assert torch.isfinite(loss.route_curriculum)
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and bool((parameter.grad.abs() > 0).any())
        for parameter in model.router.parameters()
    )


def test_curriculum_route_objective_ends_with_guidance_schedule():
    torch.manual_seed(111)
    batch, model = fixture(RoutingMode.CURRICULUM)
    output = model(
        batch.inputs, route_labels=batch.evaluation.oracle_routes, training_step=10
    )
    loss = compute_binding_loss(
        output,
        batch.inputs,
        batch.evaluation,
        routing_mode=RoutingMode.CURRICULUM,
    )
    assert int(output.diagnostics["guided_events"]) == 0
    assert loss.route_supervision_count == 0
    assert loss.route_curriculum.item() == 0.0
    assert torch.isfinite(loss.total)


def test_empty_curriculum_batch_has_finite_zero_route_loss():
    _, model = fixture(RoutingMode.CURRICULUM)
    inputs = BindingModelInputs(
        token_ids=torch.empty(1, 0, dtype=torch.int64),
        event_kinds=torch.empty(1, 0, dtype=torch.int64),
        primary_key_ids=torch.empty(1, 0, dtype=torch.int64),
        secondary_key_ids=torch.empty(1, 0, dtype=torch.int64),
        arguments=torch.empty(1, 0, dtype=torch.int64),
        valid_mask=torch.empty(1, 0, dtype=torch.bool),
    )
    evaluation = BindingEvaluation(
        oracle_routes=torch.empty(1, 0, dtype=torch.int64),
        targets=torch.empty(1, 0, dtype=torch.int64),
        dependency_parents=torch.empty(1, 0, 2, dtype=torch.int64),
        generation_ids=torch.empty(1, 0, dtype=torch.int64),
        live_binding_counts=torch.empty(1, 0, dtype=torch.int64),
        heldout_combination_mask=torch.empty(1, 0, dtype=torch.bool),
    )
    output = model(inputs, route_labels=evaluation.oracle_routes, training_step=0)
    loss = compute_binding_loss(
        output, inputs, evaluation, routing_mode=RoutingMode.CURRICULUM
    )
    assert loss.query_count == 0 and loss.route_supervision_count == 0
    assert loss.route_curriculum.item() == 0.0
    assert torch.isfinite(loss.total)


def test_one_training_step_updates_model_and_reports_all_components():
    torch.manual_seed(12)
    batch, model = fixture(RoutingMode.ORACLE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    _, loss = train_binding_step(
        model,
        batch,
        optimizer,
        training_step=0,
        loss_config=BindingLossConfig(),
    )
    assert torch.isfinite(loss.total)
    assert loss.query_count > 0
    assert loss.persistence_pair_count > 0
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )


def test_routed_nonfinite_gradients_fail_before_optimizer_mutation() -> None:
    torch.manual_seed(121)
    batch, model = fixture(RoutingMode.ORACLE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    def poison(tensor: torch.Tensor) -> torch.Tensor:
        return torch.full_like(tensor, float("nan")) if tensor.is_floating_point() else tensor

    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor, poison):
        try:
            train_binding_step(model, batch, optimizer, training_step=0)
        except ValueError as error:
            assert "finite" in str(error)
        else:
            raise AssertionError("nonfinite routed gradients were accepted")
    assert not optimizer.state
    assert all(
        torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )


def test_evaluation_contract_is_autonomous_for_curriculum():
    torch.manual_seed(13)
    batch, model = fixture(RoutingMode.CURRICULUM)
    _, summary = evaluate_binding_model(model, batch)
    assert 0.0 <= summary.query.accuracy <= 1.0
    assert 0.0 <= summary.route_recovery.accuracy <= 1.0
    assert 0.0 <= summary.route_consistency.consistency <= 1.0
    assert summary.router_load.local_event_count > 0
    assert not model.training


def test_loss_weights_validate_finite_nonnegative_values():
    for value in (-1.0, float("inf"), float("nan")):
        try:
            BindingLossConfig(query_weight=value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid loss weight was accepted")


def test_tiny_deterministic_oracle_batch_overfits() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(123)
    task = BindingTaskConfig(
        num_surface_keys=5,
        value_cardinality=4,
        branches=3,
        max_live_bindings=3,
        min_length=10,
        max_length=12,
        heldout_key_value_pairs=((0, 0),),
    )
    batch = collate_binding_episodes(
        generate_binding_episodes(
            task, count=4, seed=777, split="train", lengths=[10] * 4
        )
    )
    model = RoutedBindingModel(
        BindingModelConfig(
            task=task,
            d_model=16,
            cp_rank=8,
            router_hidden_dim=12,
            routing_mode=RoutingMode.ORACLE,
        )
    ).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
    for step in range(1, 26):
        train_binding_step(
            model,
            batch,
            optimizer,
            training_step=step,
            max_gradient_norm=5.0,
        )
    _, summary = evaluate_binding_model(model, batch)
    assert summary.query.accuracy == 1.0
