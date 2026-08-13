"""Dependency-light diagnostics for V3 routing and query evaluation.

All event-wise inputs use shape ``[N, T]``. Route IDs ``0..B-1`` are local,
``B`` is the global lane, and :data:`tnlm_v3.routing.NULL_ROUTE` is the valid
read-only/null route. Unless stated otherwise, null, global, and padded events
are excluded from local-routing metrics.

This module accepts tensors directly so it does not depend on a particular
dataset or training-loop API. Empty selections return finite zero-valued
metrics together with an explicit count of zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import torch
from torch import Tensor

from .routing import NULL_ROUTE, validate_routes


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@dataclass(frozen=True)
class DocumentRouteMatch:
    """Optimal predicted-to-true branch alignment for one document."""

    document_id: int
    correct: int
    local_event_count: int
    accuracy: float
    predicted_to_true: tuple[int, ...]


@dataclass(frozen=True)
class RouteRecoverySummary:
    """Micro/macro route recovery after independent document alignments."""

    correct: int
    local_event_count: int
    accuracy: float
    macro_accuracy: float
    document_count: int
    documents: tuple[DocumentRouteMatch, ...]


@dataclass(frozen=True)
class ConsistencyGroup:
    """Modal-route agreement for one document-local binding generation."""

    document_id: int
    key_id: int
    generation: int
    modal_route: int
    consistent_events: int
    local_event_count: int
    consistency: float


@dataclass(frozen=True)
class RouteConsistencySummary:
    """Route persistence grouped by ``(document, key, generation)``."""

    consistent_events: int
    local_event_count: int
    consistency: float
    group_count: int
    fully_consistent_groups: int
    groups: tuple[ConsistencyGroup, ...]


@dataclass(frozen=True)
class RouterLoadSummary:
    """Hard local/global/null loads and optional assignment diagnostics."""

    branch_counts: tuple[int, ...]
    branch_fractions: tuple[float, ...]
    local_event_count: int
    global_event_count: int
    null_event_count: int
    valid_event_count: int
    global_event_fraction: float
    null_event_fraction: float
    active_branches: int
    collapsed: bool
    document_count: int
    collapsed_document_count: int
    collapsed_document_fraction: float
    mean_active_branches_per_document: float
    max_load_fraction: float
    load_entropy: float
    normalized_load_entropy: float
    mean_assignment_entropy: float
    normalized_mean_assignment_entropy: float
    assignment_entropy_count: int


@dataclass(frozen=True)
class QueryAccuracy:
    """Exact class accuracy on query positions only."""

    correct: int
    query_count: int
    accuracy: float


@dataclass(frozen=True)
class OracleGap:
    """Signed accuracy gap, ``oracle_accuracy - autonomous_accuracy``."""

    oracle_accuracy: float
    autonomous_accuracy: float
    gap: float


@dataclass(frozen=True)
class EvaluationRunOutputs:
    """Model-visible outputs used by the evaluation-label audit.

    Evaluation labels are intentionally not a field. The caller changes labels
    outside the model, reruns with identical model-visible inputs, and gives
    this helper only the resulting logits and optional hard routes.
    """

    logits: Tensor
    routes: Tensor | None = None


@dataclass(frozen=True)
class LabelIndependenceAudit:
    """Comparison result for two label-independent evaluation runs."""

    passed: bool
    logits_equal: bool
    routes_equal: bool
    logit_element_count: int
    route_element_count: int
    logit_mismatch_count: int
    route_mismatch_count: int
    max_abs_logit_difference: float


def _branch_count(branches: int) -> int:
    if isinstance(branches, bool):
        raise TypeError("branches must be a positive integer")
    try:
        value = operator.index(branches)
    except TypeError as error:
        raise TypeError("branches must be a positive integer") from error
    if value <= 0:
        raise ValueError("branches must be positive")
    return value


def _require_integer_tensor(name: str, value: Tensor, shape: torch.Size) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use an integer dtype")


def _require_same_device(reference: Tensor, **values: Tensor) -> None:
    for name, value in values.items():
        if value.device != reference.device:
            raise ValueError(f"{name} must be on the same device as routes")


def _validate_route_inputs(
    predicted_routes: Tensor,
    valid_mask: Tensor,
    branches: int,
    **integer_fields: Tensor,
) -> int:
    branch_count = _branch_count(branches)
    validate_routes(predicted_routes, valid_mask, branch_count)
    for name, value in integer_fields.items():
        _require_integer_tensor(name, value, predicted_routes.shape)
    _require_same_device(predicted_routes, valid_mask=valid_mask, **integer_fields)
    return branch_count


def _optimal_assignment(
    confusion: list[list[int]],
) -> tuple[int, tuple[int, ...]]:
    """Return an exact maximum-weight row-to-column assignment using DP."""

    size = len(confusion)
    # mask -> (score, columns selected for completed rows)
    dynamic: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for row in range(size):
        updated: dict[int, tuple[int, tuple[int, ...]]] = {}
        for mask, (score, assignment) in dynamic.items():
            for column in range(size):
                bit = 1 << column
                if mask & bit:
                    continue
                candidate = (
                    score + int(confusion[row][column]),
                    assignment + (column,),
                )
                next_mask = mask | bit
                previous = updated.get(next_mask)
                if previous is None or candidate[0] > previous[0] or (
                    candidate[0] == previous[0]
                    and candidate[1] < previous[1]
                ):
                    updated[next_mask] = candidate
        dynamic = updated
    return dynamic[(1 << size) - 1]


def per_document_route_recovery(
    predicted_routes: Tensor,
    true_routes: Tensor,
    document_ids: Tensor,
    valid_mask: Tensor,
    branches: int,
) -> RouteRecoverySummary:
    """Compute exact route recovery with a separate permutation per document.

    The denominator contains valid events whose *true* route is local. A null
    or global prediction for one of those events is an error; events whose true
    route is null/global are excluded. This prevents non-local predictions from
    escaping the recovery denominator.
    """

    branch_count = _validate_route_inputs(
        predicted_routes,
        valid_mask,
        branches,
        true_routes=true_routes,
        document_ids=document_ids,
    )
    validate_routes(true_routes, valid_mask, branch_count)
    true_local = valid_mask & (true_routes >= 0) & (true_routes < branch_count)
    document_values = torch.unique(document_ids[true_local]).detach().cpu().tolist()
    matches: list[DocumentRouteMatch] = []

    for raw_document_id in document_values:
        document_id = int(raw_document_id)
        selected = true_local & (document_ids == document_id)
        local_event_count = int(selected.sum().item())
        predicted_local = (
            selected
            & (predicted_routes >= 0)
            & (predicted_routes < branch_count)
        )
        encoded = (
            predicted_routes[predicted_local].to(torch.int64) * branch_count
            + true_routes[predicted_local].to(torch.int64)
        )
        confusion_tensor = torch.bincount(
            encoded,
            minlength=branch_count * branch_count,
        ).reshape(branch_count, branch_count)
        correct, assignment = _optimal_assignment(
            confusion_tensor.detach().cpu().tolist()
        )
        matches.append(
            DocumentRouteMatch(
                document_id=document_id,
                correct=correct,
                local_event_count=local_event_count,
                accuracy=correct / local_event_count,
                predicted_to_true=assignment,
            )
        )

    total = sum(match.local_event_count for match in matches)
    correct = sum(match.correct for match in matches)
    document_count = len(matches)
    return RouteRecoverySummary(
        correct=correct,
        local_event_count=total,
        accuracy=correct / total if total else 0.0,
        macro_accuracy=(
            sum(match.accuracy for match in matches) / document_count
            if document_count
            else 0.0
        ),
        document_count=document_count,
        documents=tuple(matches),
    )


def document_local_route_consistency(
    predicted_routes: Tensor,
    document_ids: Tensor,
    key_ids: Tensor,
    generation_ids: Tensor,
    valid_mask: Tensor,
    branches: int,
) -> RouteConsistencySummary:
    """Measure modal-route agreement per document-local binding generation."""

    branch_count = _validate_route_inputs(
        predicted_routes,
        valid_mask,
        branches,
        document_ids=document_ids,
        key_ids=key_ids,
        generation_ids=generation_ids,
    )
    local = (
        valid_mask
        & (predicted_routes >= 0)
        & (predicted_routes < branch_count)
    )
    grouped: dict[tuple[int, int, int], list[int]] = {}
    values = zip(
        document_ids[local].detach().cpu().tolist(),
        key_ids[local].detach().cpu().tolist(),
        generation_ids[local].detach().cpu().tolist(),
        predicted_routes[local].detach().cpu().tolist(),
        strict=True,
    )
    for raw_document, raw_key, raw_generation, raw_route in values:
        group_key = (int(raw_document), int(raw_key), int(raw_generation))
        counts = grouped.setdefault(group_key, [0] * branch_count)
        counts[int(raw_route)] += 1

    groups: list[ConsistencyGroup] = []
    for (document_id, key_id, generation), counts in sorted(grouped.items()):
        local_event_count = sum(counts)
        consistent_events = max(counts)
        modal_route = min(
            route for route, count in enumerate(counts) if count == consistent_events
        )
        groups.append(
            ConsistencyGroup(
                document_id=document_id,
                key_id=key_id,
                generation=generation,
                modal_route=modal_route,
                consistent_events=consistent_events,
                local_event_count=local_event_count,
                consistency=consistent_events / local_event_count,
            )
        )

    event_count = sum(group.local_event_count for group in groups)
    consistent = sum(group.consistent_events for group in groups)
    return RouteConsistencySummary(
        consistent_events=consistent,
        local_event_count=event_count,
        consistency=consistent / event_count if event_count else 0.0,
        group_count=len(groups),
        fully_consistent_groups=sum(
            group.consistent_events == group.local_event_count for group in groups
        ),
        groups=tuple(groups),
    )


def summarize_router_load(
    routes: Tensor,
    valid_mask: Tensor,
    branches: int,
    *,
    local_probabilities: Tensor | None = None,
    document_ids: Tensor | None = None,
) -> RouterLoadSummary:
    """Summarize hard route loads and optional local assignment probabilities.

    ``local_probabilities``, when supplied, has shape ``[N,T,B]`` and contains
    a probability distribution over local branches. It is inspected only at
    valid positions whose hard route is local, so padding/null/global payloads
    remain irrelevant.
    """

    branch_count = _validate_route_inputs(routes, valid_mask, branches)
    if document_ids is not None:
        _require_integer_tensor("document_ids", document_ids, routes.shape)
        _require_same_device(routes, document_ids=document_ids)
    local = valid_mask & (routes >= 0) & (routes < branch_count)
    counts_tensor = torch.bincount(
        routes[local].to(torch.int64), minlength=branch_count
    )
    counts = tuple(int(value) for value in counts_tensor.detach().cpu().tolist())
    event_count = sum(counts)
    global_event_count = int((valid_mask & (routes == branch_count)).sum().item())
    null_event_count = int((valid_mask & (routes == NULL_ROUTE)).sum().item())
    valid_event_count = int(valid_mask.sum().item())
    fractions = tuple(
        count / event_count if event_count else 0.0 for count in counts
    )
    active_branches = sum(count > 0 for count in counts)
    load_entropy = -sum(
        fraction * math.log(fraction) for fraction in fractions if fraction > 0
    )
    normalizer = math.log(branch_count) if branch_count > 1 else 0.0

    document_count = 0
    collapsed_document_count = 0
    active_branches_by_document: list[int] = []
    if document_ids is not None:
        for raw_document in torch.unique(document_ids[valid_mask]).detach().cpu().tolist():
            selected = local & (document_ids == int(raw_document))
            local_routes = routes[selected].to(torch.int64)
            active = int(torch.unique(local_routes).numel()) if local_routes.numel() else 0
            active_branches_by_document.append(active)
            collapsed_document_count += int(bool(local_routes.numel()) and active <= 1)
        document_count = len(active_branches_by_document)

    mean_assignment_entropy = 0.0
    assignment_entropy_count = 0
    if local_probabilities is not None:
        if not isinstance(local_probabilities, Tensor):
            raise TypeError("local_probabilities must be a torch.Tensor")
        expected_shape = routes.shape + (branch_count,)
        if local_probabilities.shape != expected_shape:
            raise ValueError(
                f"local_probabilities must have shape {tuple(expected_shape)}"
            )
        if not local_probabilities.is_floating_point():
            raise TypeError("local_probabilities must use a floating-point dtype")
        if local_probabilities.device != routes.device:
            raise ValueError("local_probabilities must be on the same device as routes")
        selected = local_probabilities[local]
        assignment_entropy_count = int(selected.shape[0])
        if assignment_entropy_count:
            if not bool(torch.isfinite(selected).all()) or bool((selected < 0).any()):
                raise ValueError("selected local probabilities must be finite and nonnegative")
            sums = selected.sum(dim=-1)
            if not bool(
                torch.allclose(
                    sums,
                    torch.ones_like(sums),
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )
            ):
                raise ValueError("selected local probabilities must sum to one")
            safe_log = selected.clamp_min(torch.finfo(selected.dtype).tiny).log()
            terms = torch.where(
                selected > 0,
                selected * safe_log,
                torch.zeros_like(selected),
            )
            mean_assignment_entropy = float((-terms.sum(dim=-1).mean()).detach().cpu())

    return RouterLoadSummary(
        branch_counts=counts,
        branch_fractions=fractions,
        local_event_count=event_count,
        global_event_count=global_event_count,
        null_event_count=null_event_count,
        valid_event_count=valid_event_count,
        global_event_fraction=(
            global_event_count / valid_event_count if valid_event_count else 0.0
        ),
        null_event_fraction=(
            null_event_count / valid_event_count if valid_event_count else 0.0
        ),
        active_branches=active_branches,
        collapsed=bool(event_count and branch_count > 1 and active_branches <= 1),
        document_count=document_count,
        collapsed_document_count=collapsed_document_count,
        collapsed_document_fraction=(
            collapsed_document_count / document_count if document_count else 0.0
        ),
        mean_active_branches_per_document=(
            sum(active_branches_by_document) / document_count if document_count else 0.0
        ),
        max_load_fraction=max(fractions, default=0.0),
        load_entropy=load_entropy,
        normalized_load_entropy=(load_entropy / normalizer if normalizer else 0.0),
        mean_assignment_entropy=mean_assignment_entropy,
        normalized_mean_assignment_entropy=(
            mean_assignment_entropy / normalizer if normalizer else 0.0
        ),
        assignment_entropy_count=assignment_entropy_count,
    )


def exact_query_accuracy(
    predictions: Tensor,
    targets: Tensor,
    query_mask: Tensor,
) -> QueryAccuracy:
    """Return exact class accuracy over query positions.

    ``targets`` and ``query_mask`` have shape ``[N,T]``. ``predictions`` may
    contain integer class IDs with that shape or floating logits with shape
    ``[N,T,C]``. Non-query targets are ignored and may contain sentinel values.
    """

    if not isinstance(targets, Tensor) or targets.ndim != 2:
        raise ValueError("targets must be a tensor with shape [N,T]")
    _require_integer_tensor("targets", targets, targets.shape)
    if not isinstance(query_mask, Tensor) or query_mask.shape != targets.shape:
        raise ValueError("query_mask must have the same shape as targets")
    if query_mask.dtype != torch.bool:
        raise TypeError("query_mask must use torch.bool")
    if query_mask.device != targets.device:
        raise ValueError("targets and query_mask must share a device")
    if not isinstance(predictions, Tensor):
        raise TypeError("predictions must be a torch.Tensor")
    if predictions.device != targets.device:
        raise ValueError("predictions and targets must share a device")

    if predictions.shape == targets.shape:
        if predictions.dtype not in _INTEGER_DTYPES:
            raise TypeError("class-ID predictions must use an integer dtype")
        predicted_ids = predictions
        selected_predictions = predicted_ids[query_mask]
        selected_targets = targets[query_mask]
        if selected_predictions.numel() and (
            bool((selected_predictions < 0).any())
            or bool((selected_targets < 0).any())
        ):
            raise ValueError("query class IDs and targets must be nonnegative")
    elif (
        predictions.ndim == targets.ndim + 1
        and predictions.shape[:-1] == targets.shape
        and predictions.shape[-1] > 0
    ):
        if not predictions.is_floating_point():
            raise TypeError("logit predictions must use a floating-point dtype")
        selected_logits = predictions[query_mask]
        if selected_logits.numel() and not bool(torch.isfinite(selected_logits).all()):
            raise ValueError("query logits must be finite")
        selected_targets = targets[query_mask]
        if selected_targets.numel() and not bool(
            ((selected_targets >= 0) & (selected_targets < predictions.shape[-1])).all()
        ):
            raise ValueError("query targets must index the logit class dimension")
        predicted_ids = predictions.argmax(dim=-1)
    else:
        raise ValueError("predictions must have shape [N,T] or [N,T,C]")

    query_count = int(query_mask.sum().item())
    correct = int((predicted_ids[query_mask] == targets[query_mask]).sum().item())
    return QueryAccuracy(
        correct=correct,
        query_count=query_count,
        accuracy=correct / query_count if query_count else 0.0,
    )


def compute_oracle_gap(
    oracle_accuracy: float,
    autonomous_accuracy: float,
) -> OracleGap:
    """Return the signed oracle-minus-autonomous accuracy gap."""

    values: list[float] = []
    for name, raw in (
        ("oracle_accuracy", oracle_accuracy),
        ("autonomous_accuracy", autonomous_accuracy),
    ):
        if isinstance(raw, bool):
            raise TypeError(f"{name} must be a real number")
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be a real number") from error
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and inside [0,1]")
        values.append(value)
    return OracleGap(
        oracle_accuracy=values[0],
        autonomous_accuracy=values[1],
        gap=values[0] - values[1],
    )


def _validate_audit_outputs(name: str, outputs: EvaluationRunOutputs) -> None:
    if not isinstance(outputs, EvaluationRunOutputs):
        raise TypeError(f"{name} must be EvaluationRunOutputs")
    if not isinstance(outputs.logits, Tensor) or not outputs.logits.is_floating_point():
        raise TypeError(f"{name}.logits must be a floating-point tensor")
    if not bool(torch.isfinite(outputs.logits).all()):
        raise ValueError(f"{name}.logits must be finite")
    if outputs.routes is not None:
        if not isinstance(outputs.routes, Tensor) or outputs.routes.dtype not in _INTEGER_DTYPES:
            raise TypeError(f"{name}.routes must be an integer tensor when supplied")


def audit_evaluation_label_independence(
    reference: EvaluationRunOutputs,
    relabeled: EvaluationRunOutputs,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> LabelIndependenceAudit:
    """Compare outputs from runs differing only in evaluation-only labels.

    Labels are deliberately absent from this API. Both runs must be produced
    with identical model-visible inputs; the audit checks floating outputs
    within the declared tolerance and hard routes exactly.
    """

    _validate_audit_outputs("reference", reference)
    _validate_audit_outputs("relabeled", relabeled)
    for name, raw in (("atol", atol), ("rtol", rtol)):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"{name} must be a nonnegative finite number")
        if not math.isfinite(float(raw)) or float(raw) < 0:
            raise ValueError(f"{name} must be a nonnegative finite number")
    if reference.logits.shape != relabeled.logits.shape:
        raise ValueError("logit shapes must match")
    if reference.logits.dtype != relabeled.logits.dtype:
        raise ValueError("logit dtypes must match")
    if reference.logits.device != relabeled.logits.device:
        raise ValueError("logits must share a device")
    if (reference.routes is None) != (relabeled.routes is None):
        raise ValueError("routes must be supplied for both runs or neither")

    close = torch.isclose(
        reference.logits,
        relabeled.logits,
        atol=float(atol),
        rtol=float(rtol),
        equal_nan=False,
    )
    logit_mismatches = int((~close).sum().item())
    logit_count = int(reference.logits.numel())
    if logit_count:
        difference = (reference.logits - relabeled.logits).abs()
        max_difference = float(difference.max().detach().cpu())
    else:
        max_difference = 0.0

    routes_equal = True
    route_count = 0
    route_mismatches = 0
    if reference.routes is not None and relabeled.routes is not None:
        if reference.routes.shape != relabeled.routes.shape:
            raise ValueError("route shapes must match")
        if reference.routes.dtype != relabeled.routes.dtype:
            raise ValueError("route dtypes must match")
        if reference.routes.device != relabeled.routes.device:
            raise ValueError("routes must share a device")
        route_count = int(reference.routes.numel())
        route_mismatches = int((reference.routes != relabeled.routes).sum().item())
        routes_equal = route_mismatches == 0

    logits_equal = logit_mismatches == 0
    return LabelIndependenceAudit(
        passed=logits_equal and routes_equal,
        logits_equal=logits_equal,
        routes_equal=routes_equal,
        logit_element_count=logit_count,
        route_element_count=route_count,
        logit_mismatch_count=logit_mismatches,
        route_mismatch_count=route_mismatches,
        max_abs_logit_difference=max_difference,
    )


__all__ = [
    "ConsistencyGroup",
    "DocumentRouteMatch",
    "EvaluationRunOutputs",
    "LabelIndependenceAudit",
    "OracleGap",
    "QueryAccuracy",
    "RouteConsistencySummary",
    "RouteRecoverySummary",
    "RouterLoadSummary",
    "audit_evaluation_label_independence",
    "compute_oracle_gap",
    "document_local_route_consistency",
    "exact_query_accuracy",
    "per_document_route_recovery",
    "summarize_router_load",
]
