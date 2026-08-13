"""Losses, optimization steps, and evaluation for dynamic binding."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F

from .benchmark import (
    QueryAccuracy,
    RouteConsistencySummary,
    RouteRecoverySummary,
    RouterLoadSummary,
    document_local_route_consistency,
    exact_query_accuracy,
    per_document_route_recovery,
    summarize_router_load,
)
from .binding import BindingModelOutput, RoutedBindingModel
from .data import (
    BindingBatch,
    BindingEvaluation,
    BindingEventKind,
    BindingModelInputs,
)
from .routing import NULL_ROUTE, RoutingMode


@dataclass(frozen=True)
class BindingLossConfig:
    query_weight: float = 1.0
    route_curriculum_weight: float = 1.0
    router_balance_weight: float = 0.02
    router_entropy_weight: float = 0.002
    route_persistence_weight: float = 0.02

    def __post_init__(self) -> None:
        for name, raw in self.__dict__.items():
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"{name} must be a finite nonnegative number")
            if not math.isfinite(float(raw)) or float(raw) < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")


@dataclass
class BindingLoss:
    total: Tensor
    query: Tensor
    route_curriculum: Tensor
    router_balance: Tensor
    router_entropy: Tensor
    route_persistence: Tensor
    query_count: int
    route_supervision_count: int
    persistence_pair_count: int


@dataclass(frozen=True)
class BindingEvaluationSummary:
    query: QueryAccuracy
    seen_query: QueryAccuracy
    heldout_query: QueryAccuracy
    route_recovery: RouteRecoverySummary
    route_consistency: RouteConsistencySummary
    router_load: RouterLoadSummary


def _zero(reference: Tensor) -> Tensor:
    # Summing masked logits near ``-finfo.max`` can overflow before a later
    # multiplication by zero. A scalar constant is the correct empty loss.
    return reference.new_zeros(())


def _label_classes(routes: Tensor, branches: int) -> Tensor:
    return torch.where(
        routes == NULL_ROUTE,
        torch.full_like(routes, branches + 1),
        routes,
    ).to(torch.int64)


def _persistence_loss(
    probabilities: Tensor,
    inputs: BindingModelInputs,
    branches: int,
) -> tuple[Tensor, int]:
    """Visible-key consistency, reset after an explicit invalidation event."""

    local = probabilities[:, :, :branches]
    local = local / local.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(local.dtype).tiny
    )
    losses: list[Tensor] = []
    for batch_index in range(local.shape[0]):
        previous: dict[int, Tensor] = {}
        for time_index in range(local.shape[1]):
            if not bool(inputs.valid_mask[batch_index, time_index]):
                continue
            key = int(inputs.primary_key_ids[batch_index, time_index].item())
            if key <= 0:
                continue
            kind = int(inputs.event_kinds[batch_index, time_index].item())
            current = local[batch_index, time_index]
            if key in previous:
                losses.append((current - previous[key]).square().mean())
            if kind == int(BindingEventKind.INVALIDATE):
                previous.pop(key, None)
            else:
                previous[key] = current
    if not losses:
        return _zero(probabilities), 0
    return torch.stack(losses).mean(), len(losses)


def compute_binding_loss(
    output: BindingModelOutput,
    inputs: BindingModelInputs,
    evaluation: BindingEvaluation,
    *,
    routing_mode: RoutingMode | str,
    config: BindingLossConfig = BindingLossConfig(),
) -> BindingLoss:
    """Compute declared Milestone-2 objectives without latent route labels.

    Query targets are predictive labels and are used in every condition. Raw
    oracle route labels enter the objective only in curriculum mode. Oracle
    routing consumes labels as its architectural upper bound; fully latent
    routing neither reads nor supervises from them.
    """

    mode = RoutingMode(routing_mode)
    if output.value_logits.shape[:2] != inputs.valid_mask.shape:
        raise ValueError("value logits and model inputs have incompatible shapes")
    if evaluation.targets.shape != inputs.valid_mask.shape:
        raise ValueError("query targets must have shape [N,T]")
    query_mask = inputs.valid_mask & (
        inputs.event_kinds == int(BindingEventKind.QUERY)
    )
    query_count = int(query_mask.sum().item())
    query = (
        F.cross_entropy(output.value_logits[query_mask], evaluation.targets[query_mask])
        if query_count
        else _zero(output.value_logits)
    )

    route_curriculum = _zero(output.route_logits)
    route_supervision_count = 0
    branches = output.route_logits.shape[-1] - 2
    if branches <= 0:
        raise ValueError("binding router must expose local, global, and null classes")
    if mode is RoutingMode.CURRICULUM:
        guidance_mask = output.diagnostics.get("guidance_mask")
        if (
            not isinstance(guidance_mask, Tensor)
            or guidance_mask.shape != inputs.valid_mask.shape
            or guidance_mask.dtype != torch.bool
            or guidance_mask.device != inputs.valid_mask.device
        ):
            raise ValueError("curriculum output requires a boolean guidance_mask")
        supervision_mask = inputs.valid_mask & guidance_mask
        route_supervision_count = int(supervision_mask.sum().item())
        if route_supervision_count:
            labels = _label_classes(evaluation.oracle_routes, branches)
            route_curriculum = F.cross_entropy(
                output.route_logits[supervision_mask], labels[supervision_mask]
            )

    entity = inputs.valid_mask & (inputs.primary_key_ids > 0)
    local_probabilities = output.route_probabilities[:, :, :branches]
    if bool(entity.any()):
        local = local_probabilities[entity]
        local = local / local.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(local.dtype).tiny
        )
        mean_load = local.mean(dim=0)
        target = torch.full_like(mean_load, 1.0 / branches)
        balance = branches * (mean_load - target).square().mean()
        entropy = -(
            local
            * local.clamp_min(torch.finfo(local.dtype).tiny).log()
        ).sum(dim=-1).mean()
    else:
        balance = _zero(output.route_probabilities)
        entropy = _zero(output.route_probabilities)
    persistence, persistence_count = _persistence_loss(
        output.route_probabilities, inputs, branches
    )

    total = (
        config.query_weight * query
        + config.route_curriculum_weight * route_curriculum
        + config.router_balance_weight * balance
        + config.router_entropy_weight * entropy
        + config.route_persistence_weight * persistence
    )
    return BindingLoss(
        total=total,
        query=query,
        route_curriculum=route_curriculum,
        router_balance=balance,
        router_entropy=entropy,
        route_persistence=persistence,
        query_count=query_count,
        route_supervision_count=route_supervision_count,
        persistence_pair_count=persistence_count,
    )


def train_binding_step(
    model: RoutedBindingModel,
    batch: BindingBatch,
    optimizer: torch.optim.Optimizer,
    *,
    training_step: int,
    loss_config: BindingLossConfig = BindingLossConfig(),
    max_gradient_norm: float | None = 1.0,
) -> tuple[BindingModelOutput, BindingLoss]:
    """Run one deterministic optimization step under the model's route mode."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    mode = model.config.routing_mode
    route_labels = (
        batch.evaluation.oracle_routes
        if mode in (RoutingMode.ORACLE, RoutingMode.CURRICULUM)
        else None
    )
    output = model(
        batch.inputs, route_labels=route_labels, training_step=training_step
    )
    loss = compute_binding_loss(
        output,
        batch.inputs,
        batch.evaluation,
        routing_mode=mode,
        config=loss_config,
    )
    loss.total.backward()
    if max_gradient_norm is not None:
        if max_gradient_norm <= 0 or not math.isfinite(float(max_gradient_norm)):
            raise ValueError("max_gradient_norm must be positive and finite")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
    optimizer.step()
    return output, loss


@torch.no_grad()
def evaluate_binding_model(
    model: RoutedBindingModel,
    batch: BindingBatch,
) -> tuple[BindingModelOutput, BindingEvaluationSummary]:
    """Evaluate autonomously except in the explicitly oracle condition."""

    model.eval()
    labels = (
        batch.evaluation.oracle_routes
        if model.config.routing_mode is RoutingMode.ORACLE
        else None
    )
    output = model(batch.inputs, route_labels=labels)
    n, t = batch.inputs.valid_mask.shape
    document_ids = torch.arange(n, device=output.routes.device).unsqueeze(1).expand(n, t)
    query_mask = batch.inputs.valid_mask & (
        batch.inputs.event_kinds == int(BindingEventKind.QUERY)
    )
    route_recovery = per_document_route_recovery(
        output.routes,
        batch.evaluation.oracle_routes,
        document_ids,
        batch.inputs.valid_mask,
        model.config.task.branches,
    )
    consistency = document_local_route_consistency(
        output.routes,
        document_ids,
        batch.inputs.primary_key_ids,
        batch.evaluation.generation_ids,
        batch.inputs.valid_mask,
        model.config.task.branches,
    )
    local = output.route_probabilities[:, :, : model.config.task.branches]
    local = local / local.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(local.dtype).tiny
    )
    load = summarize_router_load(
        output.routes,
        batch.inputs.valid_mask,
        model.config.task.branches,
        local_probabilities=local,
        document_ids=document_ids,
    )
    summary = BindingEvaluationSummary(
        query=exact_query_accuracy(
            output.value_logits, batch.evaluation.targets, query_mask
        ),
        seen_query=exact_query_accuracy(
            output.value_logits,
            batch.evaluation.targets,
            query_mask & ~batch.evaluation.heldout_combination_mask,
        ),
        heldout_query=exact_query_accuracy(
            output.value_logits,
            batch.evaluation.targets,
            query_mask & batch.evaluation.heldout_combination_mask,
        ),
        route_recovery=route_recovery,
        route_consistency=consistency,
        router_load=load,
    )
    return output, summary


__all__ = [
    "BindingEvaluationSummary",
    "BindingLoss",
    "BindingLossConfig",
    "compute_binding_loss",
    "evaluate_binding_model",
    "train_binding_step",
]
