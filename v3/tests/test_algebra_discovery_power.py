from __future__ import annotations

from dataclasses import fields

import pytest

import tnlm_v3.algebra_discovery_power as power
from tnlm_v3.algebra_discovery import (
    PrototypeAddress,
    SequenceSelectionMode,
    TraceSupervisedCorpus,
)
from tnlm_v3.algebra_discovery_power import (
    PairLocalExceptionPowerReport,
    PowerConditionResult,
    PowerControlBudget,
    PowerControlCondition,
    PowerControlCorpus,
    PowerControlLimitError,
    PowerTraceManifestRow,
    PowerTraceRole,
    build_power_control_corpus,
    default_power_control_budget,
    run_pair_local_exception_power_control,
    run_power_condition,
)
from tnlm_v3.data import BindingEventKind


def _query_answers(bundle: PowerControlCorpus, row: PowerTraceManifestRow) -> tuple[int, ...]:
    trace = bundle.trace_corpus.traces[row.trace_index]
    return tuple(
        target for target in trace.sequence.query_targets if target is not None
    )


def _budget_with(**updates: int) -> PowerControlBudget:
    base = default_power_control_budget()
    values = {
        field.name: getattr(base, field.name)
        for field in fields(base)
        if field.name != "budget_sha256"
    }
    values.update(updates)
    return PowerControlBudget(**values, budget_sha256=power._sha256(values))


def _rehash_manifest_row(
    row: PowerTraceManifestRow, **updates: object
) -> PowerTraceManifestRow:
    values = {
        field.name: getattr(row, field.name)
        for field in fields(row)
        if field.name != "row_sha256"
    }
    values.update(updates)
    temporary = object.__new__(PowerTraceManifestRow)
    for name, value in values.items():
        object.__setattr__(temporary, name, value)
    object.__setattr__(temporary, "row_sha256", "0" * 64)
    return PowerTraceManifestRow(
        **values,
        row_sha256=power._sha256(power._manifest_row_payload(temporary)),
    )


def _rehash_corpus(
    bundle: PowerControlCorpus, **updates: object
) -> PowerControlCorpus:
    values = {
        field.name: getattr(bundle, field.name)
        for field in fields(bundle)
        if field.name != "certificate_sha256"
    }
    values.update(updates)
    temporary = object.__new__(PowerControlCorpus)
    for name, value in values.items():
        object.__setattr__(temporary, name, value)
    object.__setattr__(temporary, "certificate_sha256", "0" * 64)
    return PowerControlCorpus(
        **values,
        certificate_sha256=power._sha256(power._corpus_payload(temporary)),
    )


def _rehash_result(
    result: PowerConditionResult, **updates: object
) -> PowerConditionResult:
    values = {
        field.name: getattr(result, field.name)
        for field in fields(result)
        if field.name != "result_sha256"
    }
    values.update(updates)
    temporary = object.__new__(PowerConditionResult)
    for name, value in values.items():
        object.__setattr__(temporary, name, value)
    object.__setattr__(temporary, "result_sha256", "0" * 64)
    return PowerConditionResult(
        **values,
        result_sha256=power._sha256(power._condition_result_payload(temporary)),
    )


@pytest.fixture(scope="module")
def paired_corpora() -> tuple[PowerControlCorpus, PowerControlCorpus]:
    return (
        build_power_control_corpus(
            PowerControlCondition.OBSERVED_PAIR_LOCAL_EXCEPTION
        ),
        build_power_control_corpus(PowerControlCondition.NO_EXCEPTION),
    )


@pytest.fixture(scope="module")
def report() -> PairLocalExceptionPowerReport:
    return run_pair_local_exception_power_control()


def test_paired_corpora_are_program_matched_balanced_and_firewalled(
    paired_corpora: tuple[PowerControlCorpus, PowerControlCorpus],
):
    positive, negative = paired_corpora
    assert positive.program_manifest_sha256 == negative.program_manifest_sha256
    assert positive.trace_corpus.corpus_sha256 != negative.trace_corpus.corpus_sha256
    assert (
        positive.train_trace_count,
        positive.validation_trace_count,
        positive.train_event_count,
        positive.validation_event_count,
        positive.train_query_count,
        positive.validation_query_count,
    ) == (42, 42, 341, 355, 225, 231)
    assert positive.train_output_class_counts == (75, 75, 75)
    assert positive.validation_output_class_counts == (77, 77, 77)
    assert negative.train_output_class_counts == (75, 75, 75)
    assert negative.validation_output_class_counts == (77, 77, 77)
    universe = {
        (key, value)
        for key in range(positive.design.num_surface_keys)
        for value in range(positive.design.value_cardinality)
    }
    assert set(positive.observed_cells) == universe - {positive.design.outer_cell}
    assert positive.observed_cells == negative.observed_cells
    assert positive.design.outer_cell not in positive.observed_cells
    assert positive.outer_firewall_passed and negative.outer_firewall_passed
    assert not positive.outer_test_results_used
    assert not positive.exception_declaration_received_by_estimator
    assert not positive.outer_identifier_received_by_estimator


def test_exception_changes_balanced_dynamic_behavior_not_visible_programs(
    paired_corpora: tuple[PowerControlCorpus, PowerControlCorpus],
):
    positive, negative = paired_corpora
    positive_rows = tuple(
        row
        for row in positive.trace_manifest
        if row.role is PowerTraceRole.DYNAMIC
        and row.primary_key == 1
        and row.source_value == 0
    )
    negative_rows = tuple(
        row
        for row in negative.trace_manifest
        if row.role is PowerTraceRole.DYNAMIC
        and row.primary_key == 1
        and row.source_value == 0
    )
    assert len(positive_rows) == len(negative_rows) == 4
    for positive_row, negative_row in zip(positive_rows, negative_rows, strict=True):
        assert positive_row.program_sha256 == negative_row.program_sha256
        assert positive_row.exception_opportunity_count == 1
        assert positive_row.exception_applied_count == 1
        assert negative_row.exception_opportunity_count == 1
        assert negative_row.exception_applied_count == 0
        positive_answers = _query_answers(positive, positive_row)
        negative_answers = _query_answers(negative, negative_row)
        if positive_row.split == "train":
            assert positive_answers == (2, 1, 0)
            assert negative_answers == (1, 0, 2)
        else:
            assert positive_answers == (2, 1, 0, 0, 1, 2)
            assert negative_answers == (1, 0, 2, 2, 0, 1)
            assert positive_row.validation_crosslink
        assert sorted(positive_answers) == sorted(negative_answers)
    for trace in positive.trace_corpus.traces:
        for event in trace.sequence.events:
            if event.kind is BindingEventKind.UPDATE:
                assert event.argument in (0, 1)


def test_positive_control_selects_and_realizes_the_exact_semantic_exception(
    report: PairLocalExceptionPowerReport,
):
    result = report.positive
    assert result.selection.mode is SequenceSelectionMode.OBSERVED_EXCEPTION_POWER_CONTROL
    assert result.selection.selected_residual_penalty == 4
    assert result.selection.primary_score_best_penalties == (4,)
    assert not result.selection.primary_score_tied
    audits = {row.residual_penalty: row for row in result.direct_penalty_audits}
    assert (
        audits[4].training_mistakes,
        audits[4].attained_training_objective,
        audits[4].validation_mistakes,
    ) == (0, 4, 0)
    assert (
        audits[16].training_mistakes,
        audits[16].attained_training_objective,
        audits[16].validation_mistakes,
    ) == (6, 6, 12)
    expected_override = (
        (1, PrototypeAddress("update", 0, 0), 2),
    )
    assert audits[4].model.local_overrides == expected_override
    assert audits[16].model.local_overrides == ()
    assert audits[4].canonical_shared_table_realized
    assert audits[16].canonical_shared_table_realized
    assert audits[4].semantic_decomposition_gauge_fixed
    assert audits[4].expected_exception_override_realized
    assert result.selection.final_model == audits[4].model
    assert result.selected_sequence_validation_margin == 36
    assert result.direct_full_validation_margin == 12
    assert result.crosslink_winning_cells == ((2, 0), (2, 1), (2, 2))


def test_every_positive_fold_candidate_attains_its_analytic_optimum(
    report: PairLocalExceptionPowerReport,
):
    rows = report.positive.fold_optimum_certificates
    assert len(rows) == 22
    assert all(row.global_optimum_certified_for_frozen_control for row in rows)
    assert all(
        row.attained_penalized_objective == row.penalized_objective_lower_bound
        for row in rows
    )
    identifying = tuple(row for row in rows if not row.self_pseudoheldout_nonidentifying)
    assert all(row.retained_exception_opportunity_count == 2 for row in identifying)
    assert all(row.hard_shared_mistake_lower_bound == 6 for row in identifying)
    assert all(row.surviving_isolation_witness_floor >= 7 for row in identifying)
    for row in identifying:
        if row.residual_penalty == 4:
            assert (row.attained_training_mistakes, row.attained_override_count) == (0, 1)
        else:
            assert (row.attained_training_mistakes, row.attained_override_count) == (6, 0)


def test_self_pseudoheldout_fold_is_nonidentifying_and_not_the_direct_margin(
    report: PairLocalExceptionPowerReport,
):
    result = report.positive
    assert result.self_pseudoheldout_cells == ((1, 0), (1, 1), (1, 2))
    self_rows = tuple(
        row
        for row in result.fold_optimum_certificates
        if row.self_pseudoheldout_nonidentifying
    )
    assert len(self_rows) == 6
    assert all(row.retained_exception_opportunity_count == 0 for row in self_rows)
    assert all(row.penalized_objective_lower_bound == 0 for row in self_rows)
    for fold in result.selection.folds:
        if fold.pseudoheldout_cell in result.self_pseudoheldout_cells:
            assert (
                fold.candidates[0].all_validation_query_mistakes
                == fold.candidates[1].all_validation_query_mistakes
            )
    audits = {row.residual_penalty: row for row in result.direct_penalty_audits}
    assert audits[4].validation_query_count == audits[16].validation_query_count == 231
    assert audits[16].validation_mistakes - audits[4].validation_mistakes == 12
    assert not result.unseen_exception_prediction_claimed


def test_no_exception_negative_control_prefers_strong_penalty_without_overrides(
    report: PairLocalExceptionPowerReport,
):
    result = report.negative
    assert result.selection.selected_residual_penalty == 16
    assert result.selection.primary_score_best_penalties == (4, 16)
    assert result.selection.primary_score_tied
    assert result.selected_sequence_validation_margin == 0
    assert result.direct_full_validation_margin == 0
    assert result.crosslink_winning_cells == ()
    assert all(row.model.local_overrides == () for row in result.direct_penalty_audits)
    assert all(row.training_mistakes == 0 for row in result.direct_penalty_audits)
    assert all(row.validation_mistakes == 0 for row in result.direct_penalty_audits)
    assert all(row.canonical_shared_table_realized for row in result.direct_penalty_audits)
    assert all(
        row.attained_penalized_objective == 0
        for row in result.fold_optimum_certificates
    )


def test_rehashed_provenance_and_result_forgeries_are_rejected(
    paired_corpora: tuple[PowerControlCorpus, PowerControlCorpus],
    report: PairLocalExceptionPowerReport,
):
    positive, _ = paired_corpora
    original = positive.trace_manifest[0]
    forged_row = _rehash_manifest_row(original, role=PowerTraceRole.BALANCE)
    forged_manifest = (forged_row, *positive.trace_manifest[1:])
    with pytest.raises(ValueError, match="frozen recipe"):
        _rehash_corpus(positive, trace_manifest=forged_manifest)
    with pytest.raises(ValueError, match="margin is inconsistent"):
        _rehash_result(
            report.positive,
            direct_full_validation_margin=report.positive.direct_full_validation_margin + 1,
        )
    with pytest.raises(ValueError, match="does not reproduce"):
        _rehash_corpus(positive, train_query_count=positive.train_query_count + 1)


def test_budgets_fail_before_corpus_or_fit_work(monkeypatch: pytest.MonkeyPatch):
    def forbidden_materialization(*args: object, **kwargs: object):
        raise AssertionError("materialization must not start")

    monkeypatch.setattr(power, "_materialize_control", forbidden_materialization)
    with pytest.raises(PowerControlLimitError, match="max_events"):
        build_power_control_corpus(
            PowerControlCondition.NO_EXCEPTION,
            budget=_budget_with(max_events=695),
        )

    def forbidden_selection(*args: object, **kwargs: object):
        raise AssertionError("selection must not start")

    monkeypatch.setattr(power, "select_sequence_algebra", forbidden_selection)
    with pytest.raises(PowerControlLimitError, match="max_fit_calls"):
        run_pair_local_exception_power_control(
            budget=_budget_with(max_fit_calls=47)
        )


def test_selector_boundary_receives_no_exception_outer_or_test_metadata(
    monkeypatch: pytest.MonkeyPatch,
    paired_corpora: tuple[PowerControlCorpus, PowerControlCorpus],
):
    positive, _ = paired_corpora

    class BoundaryReached(RuntimeError):
        pass

    def inspect_boundary(corpus: object, **kwargs: object):
        assert type(corpus) is TraceSupervisedCorpus
        assert not hasattr(corpus, "condition")
        assert not hasattr(corpus, "exception")
        assert not hasattr(corpus, "outer_cell")
        assert kwargs["residual_penalties"] == (4, 16)
        assert kwargs["mode"] is SequenceSelectionMode.OBSERVED_EXCEPTION_POWER_CONTROL
        forbidden = {
            "exception",
            "condition",
            "outer_cell",
            "heldout",
            "test_results",
            "expected_override",
        }
        assert forbidden.isdisjoint(kwargs)
        raise BoundaryReached

    monkeypatch.setattr(power, "select_sequence_algebra", inspect_boundary)
    with pytest.raises(BoundaryReached):
        run_power_condition(positive)


def test_report_scope_and_hashes_are_closed(report: PairLocalExceptionPowerReport):
    assert len(report.report_sha256) == 64
    assert report.matched_visible_programs
    assert report.balanced_output_classes
    assert report.observed_exception_seen_in_train_and_validation
    assert report.self_pseudoheldout_nonidentifying_disclosed
    assert not report.outer_results_used_for_tuning
    assert not report.unseen_exception_prediction_claimed
    assert not report.representation_discovery_performed
    assert not report.confirmatory_claim_permitted
