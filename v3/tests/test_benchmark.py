import math

import pytest
import torch

from tnlm_v3.benchmark import (
    EvaluationRunOutputs,
    audit_evaluation_label_independence,
    compute_oracle_gap,
    document_local_route_consistency,
    exact_query_accuracy,
    per_document_route_recovery,
    summarize_router_load,
)
from tnlm_v3.routing import NULL_ROUTE


def _routes_from_confusion(confusion):
    predicted = []
    truth = []
    for predicted_route, row in enumerate(confusion):
        for true_route, count in enumerate(row):
            predicted.extend([predicted_route] * count)
            truth.extend([true_route] * count)
    return torch.tensor([predicted]), torch.tensor([truth])


def test_route_recovery_uses_exact_assignment_when_greedy_would_fail():
    # Largest-cell greedy chooses 9 + 7 + 0 = 16. The exact assignment is
    # row0->col1, row1->col0, row2->col2 for 8 + 8 + 6 = 22.
    confusion = [[9, 8, 0], [8, 0, 0], [0, 7, 6]]
    predicted, truth = _routes_from_confusion(confusion)
    documents = torch.zeros_like(truth)
    valid = torch.ones_like(truth, dtype=torch.bool)

    metrics = per_document_route_recovery(
        predicted, truth, documents, valid, branches=3
    )

    assert metrics.correct == 22
    assert metrics.local_event_count == 38
    assert metrics.accuracy == pytest.approx(22 / 38)
    assert metrics.document_count == 1
    assert metrics.documents[0].predicted_to_true == (1, 0, 2)


def test_route_recovery_aligns_each_document_independently():
    predicted = torch.tensor(
        [
            [1, 1, 0, 0, NULL_ROUTE, 3],
            [2, 0, 1, 2, 3, NULL_ROUTE],
        ]
    )
    truth = torch.tensor(
        [
            [0, 0, 1, 1, NULL_ROUTE, 3],
            [0, 1, 2, NULL_ROUTE, 3, NULL_ROUTE],
        ]
    )
    documents = torch.tensor([[10] * 6, [20] * 6])
    valid = torch.ones_like(truth, dtype=torch.bool)

    metrics = per_document_route_recovery(
        predicted, truth, documents, valid, branches=3
    )

    assert metrics.correct == 7
    assert metrics.local_event_count == 7
    assert metrics.accuracy == 1.0
    assert metrics.macro_accuracy == 1.0
    assert [item.document_id for item in metrics.documents] == [10, 20]


def test_true_local_prediction_of_null_or_global_is_counted_as_error():
    predicted = torch.tensor([[NULL_ROUTE, 2, 0, 1]])
    truth = torch.tensor([[0, 1, NULL_ROUTE, 2]])
    documents = torch.zeros_like(truth)
    valid = torch.ones_like(truth, dtype=torch.bool)

    metrics = per_document_route_recovery(
        predicted, truth, documents, valid, branches=2
    )

    assert metrics.local_event_count == 2
    assert metrics.correct == 0
    assert metrics.accuracy == 0.0


def test_consistency_separates_rebinding_generations_and_excludes_nonlocal_events():
    routes = torch.tensor([[0, 0, 1, 1, NULL_ROUTE, 2, 999]])
    documents = torch.tensor([[4, 4, 4, 4, 4, 4, -999]])
    keys = torch.tensor([[7, 7, 7, 7, 7, 7, -999]])
    generations = torch.tensor([[0, 0, 1, 1, 1, 1, -999]])
    valid = torch.tensor([[True, True, True, True, True, True, False]])

    metrics = document_local_route_consistency(
        routes, documents, keys, generations, valid, branches=2
    )

    assert metrics.local_event_count == 4
    assert metrics.consistent_events == 4
    assert metrics.consistency == 1.0
    assert metrics.group_count == 2
    assert metrics.fully_consistent_groups == 2
    assert [(group.generation, group.modal_route) for group in metrics.groups] == [
        (0, 0),
        (1, 1),
    ]


def test_consistency_reports_modal_agreement_and_deterministic_tie_break():
    routes = torch.tensor([[1, 0, 1, 0, 1]])
    fields = torch.zeros_like(routes)
    valid = torch.ones_like(routes, dtype=torch.bool)

    metrics = document_local_route_consistency(
        routes, fields, fields, fields, valid, branches=2
    )

    assert metrics.consistency == 3 / 5
    assert metrics.groups[0].modal_route == 1

    tied = document_local_route_consistency(
        routes[:, :4], fields[:, :4], fields[:, :4], fields[:, :4], valid[:, :4], 2
    )
    assert tied.groups[0].modal_route == 0


def test_router_summary_excludes_null_global_and_padding():
    routes = torch.tensor([[0, 0, 1, NULL_ROUTE, 3, 2, 999]])
    valid = torch.tensor([[True, True, True, True, True, True, False]])

    metrics = summarize_router_load(routes, valid, branches=3)

    assert metrics.branch_counts == (2, 1, 1)
    assert metrics.branch_fractions == (0.5, 0.25, 0.25)
    assert metrics.local_event_count == 4
    assert metrics.global_event_count == 1
    assert metrics.null_event_count == 1
    assert metrics.valid_event_count == 6
    assert metrics.active_branches == 3
    assert not metrics.collapsed
    assert metrics.max_load_fraction == 0.5
    assert metrics.load_entropy == pytest.approx(-(0.5 * math.log(0.5) + 0.5 * math.log(0.25)))
    assert 0.0 < metrics.normalized_load_entropy <= 1.0


def test_router_summary_detects_collapse_and_computes_assignment_entropy():
    routes = torch.tensor([[1, 1, NULL_ROUTE, 3]])
    valid = torch.ones_like(routes, dtype=torch.bool)
    probabilities = torch.tensor(
        [[[0.5, 0.5, 0.0], [0.25, 0.25, 0.5], [float("nan")] * 3, [-9.0] * 3]]
    )

    metrics = summarize_router_load(
        routes,
        valid,
        branches=3,
        local_probabilities=probabilities,
    )

    assert metrics.collapsed
    assert metrics.active_branches == 1
    assert metrics.assignment_entropy_count == 2
    expected = (math.log(2.0) + 1.5 * math.log(2.0)) / 2
    assert metrics.mean_assignment_entropy == pytest.approx(expected)


def test_router_summary_reports_document_level_collapse() -> None:
    routes = torch.tensor([[0, 0], [1, 1], [2, 2]])
    valid = torch.ones_like(routes, dtype=torch.bool)
    documents = torch.tensor([[10, 10], [20, 20], [30, 30]])
    metrics = summarize_router_load(
        routes, valid, branches=3, document_ids=documents
    )
    assert not metrics.collapsed
    assert metrics.document_count == 3
    assert metrics.collapsed_document_count == 3
    assert metrics.collapsed_document_fraction == 1.0
    assert metrics.mean_active_branches_per_document == 1.0


def test_query_accuracy_accepts_logits_and_ignores_nonquery_targets():
    logits = torch.tensor(
        [[[0.1, 0.9], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6]]]
    )
    targets = torch.tensor([[-999, 0, -999, 0]])
    query_mask = torch.tensor([[False, True, False, True]])

    metrics = exact_query_accuracy(logits, targets, query_mask)

    assert metrics.correct == 1
    assert metrics.query_count == 2
    assert metrics.accuracy == 0.5


def test_integer_query_accuracy_rejects_negative_sentinel_matches():
    with pytest.raises(ValueError, match="nonnegative"):
        exact_query_accuracy(
            torch.tensor([[-100]]),
            torch.tensor([[-100]]),
            torch.tensor([[True]]),
        )


def test_oracle_gap_is_signed_and_validated():
    metrics = compute_oracle_gap(0.95, 0.70)
    assert metrics.gap == pytest.approx(0.25)
    assert compute_oracle_gap(0.7, 0.8).gap == pytest.approx(-0.1)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        compute_oracle_gap(1.1, 0.8)


def test_empty_masks_have_finite_zero_metrics():
    routes = torch.tensor([[NULL_ROUTE, 2, 999]])
    truth = torch.tensor([[NULL_ROUTE, 2, -999]])
    documents = torch.tensor([[4, 4, -999]])
    valid = torch.tensor([[True, True, False]])

    recovery = per_document_route_recovery(
        routes, truth, documents, valid, branches=2
    )
    consistency = document_local_route_consistency(
        routes,
        documents,
        torch.zeros_like(routes),
        torch.zeros_like(routes),
        valid,
        branches=2,
    )
    load = summarize_router_load(routes, valid, branches=2)
    query = exact_query_accuracy(
        torch.zeros((1, 3), dtype=torch.int64),
        torch.full((1, 3), -999, dtype=torch.int64),
        torch.zeros((1, 3), dtype=torch.bool),
    )

    assert recovery.local_event_count == 0 and recovery.accuracy == 0.0
    assert recovery.document_count == 0 and recovery.documents == ()
    assert consistency.local_event_count == 0 and consistency.consistency == 0.0
    assert consistency.group_count == 0 and consistency.groups == ()
    assert load.local_event_count == 0 and load.load_entropy == 0.0
    assert load.branch_fractions == (0.0, 0.0) and not load.collapsed
    assert query.query_count == 0 and query.accuracy == 0.0


def test_evaluation_label_independence_passes_when_outputs_do_not_change():
    # Evaluation labels deliberately remain outside EvaluationRunOutputs.
    labels_a = torch.tensor([[0, 1]])
    labels_b = torch.tensor([[1, 0]])
    assert not torch.equal(labels_a, labels_b)
    logits = torch.tensor([[[0.2, 0.8], [0.7, 0.3]]])
    routes = torch.tensor([[0, 1]])

    audit = audit_evaluation_label_independence(
        EvaluationRunOutputs(logits=logits, routes=routes),
        EvaluationRunOutputs(logits=logits.clone(), routes=routes.clone()),
    )

    assert audit.passed
    assert audit.logits_equal and audit.routes_equal
    assert audit.logit_mismatch_count == 0
    assert audit.route_mismatch_count == 0


def test_evaluation_label_independence_fails_for_logit_or_route_changes():
    reference = EvaluationRunOutputs(
        logits=torch.tensor([[0.0, 1.0]]),
        routes=torch.tensor([[0, 1]]),
    )
    changed = EvaluationRunOutputs(
        logits=torch.tensor([[0.0, 1.01]]),
        routes=torch.tensor([[1, 1]]),
    )

    audit = audit_evaluation_label_independence(
        reference,
        changed,
        atol=1.0e-4,
        rtol=0.0,
    )

    assert not audit.passed
    assert not audit.logits_equal and not audit.routes_equal
    assert audit.logit_mismatch_count == 1
    assert audit.route_mismatch_count == 1
    assert audit.max_abs_logit_difference == pytest.approx(0.01)


def test_diagnostics_validate_shapes_and_dtypes():
    routes = torch.zeros((1, 2), dtype=torch.int64)
    valid = torch.ones((1, 2), dtype=torch.bool)
    with pytest.raises(TypeError, match="document_ids"):
        per_document_route_recovery(
            routes,
            routes,
            torch.zeros((1, 2), dtype=torch.float32),
            valid,
            branches=2,
        )
    with pytest.raises(ValueError, match="query_mask"):
        exact_query_accuracy(
            routes,
            routes,
            torch.ones((1, 1), dtype=torch.bool),
        )
