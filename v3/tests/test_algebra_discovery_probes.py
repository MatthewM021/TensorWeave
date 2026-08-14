from __future__ import annotations

from dataclasses import replace

import pytest

import tnlm_v3.algebra_discovery_probes as probe_module
from tnlm_v3.algebra_discovery import (
    VisibleEvent as TrainingEvent,
    VisibleSequence as TrainingSequence,
    fit_sequence_algebra,
    make_sequence_corpus,
)
from tnlm_v3.algebra_discovery_probes import (
    ProbeBudgetExceededError,
    ProbeCellRotation,
    ProbeFamily,
    ProbePathRelation,
    ProbeProtocolStatus,
    VisibleProbeProgram,
    build_balanced_probe_suite,
    cyclic_cell_rotation_inventory,
    evaluate_probe_suite,
    evaluate_shortcut_controls,
)
from tnlm_v3.data import BindingEventKind


KEYS = 5
VALUES = 4
OUTER_CELL = (0, 0)
ALL_CELLS = tuple((key, value) for key in range(KEYS) for value in range(VALUES))
INNER_CELLS = tuple(cell for cell in ALL_CELLS if cell != OUTER_CELL)


class ExactVisiblePredictor:
    def predict_queries(self, program: VisibleProbeProgram) -> tuple[int, ...]:
        state: list[int | None] = [None] * program.num_surface_keys
        answers: list[int] = []
        for event in program.events:
            key = event.primary_key
            if event.kind is BindingEventKind.BIND:
                assert state[key] is None
                state[key] = event.argument
            elif event.kind is BindingEventKind.UPDATE:
                assert state[key] is not None
                state[key] = (
                    state[key] + event.argument + 1
                ) % program.value_cardinality
            elif event.kind is BindingEventKind.COPY:
                assert state[key] is not None
                assert state[event.secondary_key] is not None
                state[key] = state[event.secondary_key]
            elif event.kind is BindingEventKind.INVALIDATE:
                assert state[key] is not None
                state[key] = None
            elif event.kind is BindingEventKind.QUERY:
                assert state[key] is not None
                answers.append(state[key])
        return tuple(answers)


class BoundarySpy(ExactVisiblePredictor):
    def __init__(self) -> None:
        self.call_count = 0

    def predict_queries(self, program: VisibleProbeProgram) -> tuple[int, ...]:
        assert type(program) is VisibleProbeProgram
        assert program.query_targets == (None,) * len(program.events)
        for forbidden in (
            "case_id",
            "family",
            "probe_pair",
            "base_probe_pair",
            "expected_answers",
            "query_roles",
            "support_audit",
            "equivalence_group",
            "path_relation",
            "cell_rotation",
        ):
            assert not hasattr(program, forbidden)
        self.call_count += 1
        return super().predict_queries(program)


@pytest.fixture(scope="module")
def inner_suite():
    return build_balanced_probe_suite(
        KEYS,
        VALUES,
        INNER_CELLS,
        forbidden_pairs=(OUTER_CELL,),
    )


@pytest.fixture(scope="module")
def actual_suite():
    return build_balanced_probe_suite(KEYS, VALUES, (OUTER_CELL,))


@pytest.fixture(scope="module")
def rotated_suite():
    return build_balanced_probe_suite(
        KEYS,
        VALUES,
        (OUTER_CELL,),
        cell_rotations=cyclic_cell_rotation_inventory(KEYS, VALUES),
    )


def _event(
    kind: BindingEventKind,
    primary_key: int = -1,
    secondary_key: int = -1,
    argument: int = -1,
) -> TrainingEvent:
    return TrainingEvent(kind, primary_key, secondary_key, argument)


def _controlled_training_sequences() -> tuple[TrainingSequence, ...]:
    sequences: list[TrainingSequence] = []
    for key, value in INNER_CELLS:
        sequences.append(
            TrainingSequence(
                events=(
                    _event(BindingEventKind.BIND, key, argument=value),
                    _event(BindingEventKind.QUERY, key),
                ),
                query_targets=(None, value),
            )
        )
    for source in range(VALUES):
        for transform in range(VALUES - 1):
            target = (source + transform + 1) % VALUES
            sequences.append(
                TrainingSequence(
                    events=(
                        _event(BindingEventKind.BIND, 1, argument=source),
                        _event(BindingEventKind.QUERY, 1),
                        _event(BindingEventKind.UPDATE, 1, argument=transform),
                        _event(BindingEventKind.QUERY, 1),
                    ),
                    query_targets=(None, source, None, target),
                )
            )
    for source in range(VALUES):
        sequences.append(
            TrainingSequence(
                events=(
                    _event(BindingEventKind.BIND, 1, argument=0),
                    _event(BindingEventKind.BIND, 2, argument=source),
                    _event(BindingEventKind.QUERY, 2),
                    _event(BindingEventKind.COPY, 1, secondary_key=2),
                    _event(BindingEventKind.QUERY, 1),
                ),
                query_targets=(None, None, source, None, source),
            )
        )
    return tuple(sequences)


@pytest.fixture(scope="module")
def learned_model():
    corpus = make_sequence_corpus(
        KEYS,
        VALUES,
        split="train",
        sequences=_controlled_training_sequences(),
    )
    assert (len(corpus.sequences), corpus.event_count, corpus.query_count) == (
        35,
        106,
        51,
    )
    model = fit_sequence_algebra(
        corpus,
        residual_penalty=16,
        seed=0,
        restart_count=3,
        max_sweeps=8,
    )
    assert model.fit.training_mistakes == 0
    assert model.local_overrides == ()
    assert model.fit.canonical_state_supervision_used_by_estimator is False
    assert model.fit.exact_executor_used_for_fitting is False
    assert model.fit.heldout_identifier_received_by_estimator is False
    assert model.fit.evaluation_metadata_received_by_estimator is False
    return model


def test_all_nineteen_rotations_are_balanced_and_never_touch_outer_cell(
    inner_suite,
) -> None:
    assert inner_suite.protocol_status is (
        ProbeProtocolStatus.RETROSPECTIVE_PROTOCOL_REHEARSAL
    )
    assert inner_suite.probe_pairs == INNER_CELLS
    assert inner_suite.forbidden_pairs == (OUTER_CELL,)
    assert len(inner_suite.cases) == 285
    assert sum(case.program.query_count for case in inner_suite.cases) == 1_704
    assert inner_suite.balance.every_family_output_balanced
    assert inner_suite.balance.rotated_cells_equally_weighted
    assert {count for _, count in inner_suite.balance.pair_case_counts} == {15}
    assert {count for _, count in inner_suite.balance.pair_focal_query_counts} == {
        24
    }
    assert tuple(family for family, _ in inner_suite.balance.family_class_counts) == (
        tuple(ProbeFamily)
    )
    assert all(
        len(set(counts)) == 1
        for _, counts in inner_suite.balance.family_class_counts
    )
    assert all(case.support_audit.passed for case in inner_suite.cases)
    assert all(
        OUTER_CELL not in case.support_audit.touched_pairs
        for case in inner_suite.cases
    )
    assert {
        case.probe_entry_kind
        for case in inner_suite.cases
        if case.family is ProbeFamily.FIRST_ENTRY_BIND
    } == {BindingEventKind.BIND}
    assert {
        case.probe_entry_kind
        for case in inner_suite.cases
        if case.family is ProbeFamily.FIRST_ENTRY_UPDATE
    } == {BindingEventKind.UPDATE}
    assert {
        case.probe_entry_kind
        for case in inner_suite.cases
        if case.family is ProbeFamily.FIRST_ENTRY_COPY
    } == {BindingEventKind.COPY}


def test_actual_cell_has_a_first_class_undiluted_pair_result(rotated_suite) -> None:
    evaluation = evaluate_probe_suite(ExactVisiblePredictor(), rotated_suite)
    actual = evaluation.result_for_pair(OUTER_CELL)
    assert len(rotated_suite.cases) == 300
    assert len(evaluation.pair_results) == 20
    assert evaluation.query_count == sum(
        row.query_count for row in evaluation.pair_results
    )
    assert (
        actual.case_count,
        actual.query_count,
        actual.focal_query_count,
        actual.correct_count,
        actual.focal_correct_count,
        actual.exact_case_count,
    ) == (15, 96, 24, 96, 24, 15)
    assert actual.accuracy == actual.focal_accuracy == 1.0
    assert actual.query_count < evaluation.query_count
    with pytest.raises(KeyError, match="not present"):
        evaluation.result_for_pair((KEYS, 0))


def test_suite_digest_balance_and_expected_answers_reject_forgery(
    inner_suite,
) -> None:
    with pytest.raises(ValueError, match="suite_sha256"):
        replace(inner_suite, suite_sha256="0" * 64)

    forged_counts = tuple(
        (pair, count + 1) for pair, count in inner_suite.balance.pair_case_counts
    )
    forged_balance = replace(
        inner_suite.balance,
        pair_case_counts=forged_counts,
    )
    with pytest.raises(ValueError, match="balance certificate"):
        replace(inner_suite, balance=forged_balance)

    case = inner_suite.cases[0]
    forged_answers = ((case.expected_answers[0] + 1) % VALUES,) + (
        case.expected_answers[1:]
    )
    with pytest.raises(ValueError, match="trusted program execution"):
        replace(case, expected_answers=forged_answers)


def test_evaluation_manifest_summary_equivalence_and_digest_reject_forgery(
    actual_suite,
) -> None:
    evaluation = evaluate_probe_suite(ExactVisiblePredictor(), actual_suite)
    first = evaluation.case_results[0]
    forged_first = replace(
        first,
        expected_answers=(0,) * len(first.expected_answers),
        predicted_answers=(0,) * len(first.predicted_answers),
    )
    forged_rows = (forged_first, *evaluation.case_results[1:])
    forged_manifest = probe_module._scoring_manifest_sha256_from_results(forged_rows)
    with pytest.raises(ValueError, match="bound suite"):
        replace(
            evaluation,
            case_results=forged_rows,
            scoring_manifest_sha256=forged_manifest,
        )

    equality = next(
        row
        for row in evaluation.equivalence_results
        if row.relation is ProbePathRelation.EQUAL
    )
    changed = replace(
        equality,
        predicted_focal_answers=tuple(
            tuple((value + 1) % VALUES for value in answers)
            for answers in equality.predicted_focal_answers
        ),
    )
    changed_rows = tuple(
        changed if row.group == equality.group else row
        for row in evaluation.equivalence_results
    )
    with pytest.raises(ValueError, match="equivalence results"):
        replace(evaluation, equivalence_results=changed_rows)

    with pytest.raises(ValueError, match="evaluation_sha256"):
        replace(evaluation, evaluation_sha256="0" * 64)


def test_predictor_callback_receives_only_target_free_visible_program(
    actual_suite,
) -> None:
    spy = BoundarySpy()
    evaluation = evaluate_probe_suite(spy, actual_suite)
    assert spy.call_count == len(actual_suite.cases) == 15
    assert evaluation.accuracy == evaluation.focal_accuracy == 1.0


def test_all_four_declared_shortcuts_fail_focal_discovery(actual_suite) -> None:
    rows = evaluate_shortcut_controls(actual_suite)
    assert tuple(row.name for row in rows) == (
        "constant_class_0",
        "last_visible_argument",
        "latest_bind_argument_for_query_key",
        "source_key_bind_echo",
    )
    assert all(row.evaluation.accuracy < 1.0 for row in rows)
    assert all(row.evaluation.focal_accuracy < 1.0 for row in rows)
    assert all(len(row.evaluation.family_results) == len(ProbeFamily) for row in rows)
    constant = rows[0].evaluation
    assert constant.accuracy == 1 / VALUES
    assert all(result.accuracy == 1 / VALUES for result in constant.family_results)
    assert constant.path_consistency < 1.0


def test_copy_order_is_a_certified_inequality_not_an_equivalence(
    actual_suite,
) -> None:
    exact = evaluate_probe_suite(ExactVisiblePredictor(), actual_suite)
    inequalities = tuple(
        row
        for row in exact.equivalence_results
        if row.relation is ProbePathRelation.NOT_EQUAL
    )
    assert len(inequalities) == exact.inequality_group_count == 1
    row = inequalities[0]
    assert row.family is ProbeFamily.COPY_ORDER_NONCOMMUTATION
    assert row.member_count == 2
    assert not row.expected_consistent
    assert not row.predicted_consistent
    assert row.expected_relation_satisfied
    assert row.predicted_relation_satisfied
    assert exact.inequality_satisfied_group_count == 1
    assert exact.all_copy_order_inequalities_satisfied

    constant = evaluate_shortcut_controls(actual_suite)[0].evaluation
    assert constant.inequality_satisfied_group_count == 0
    assert not constant.all_copy_order_inequalities_satisfied


def test_cell_rotations_are_injective_attributions_not_metamorphic_claims(
    rotated_suite,
) -> None:
    assert "does not certify program-level" in (ProbeCellRotation.__doc__ or "")
    assert "not program-level conjugacy" in (
        build_balanced_probe_suite.__doc__ or ""
    )
    inventory = cyclic_cell_rotation_inventory(KEYS, VALUES)
    assert len(inventory) == KEYS * VALUES == 20
    assert {row.apply_pair(OUTER_CELL) for row in inventory} == set(ALL_CELLS)
    for row in inventory:
        assert tuple(sorted(row.key_permutation)) == tuple(range(KEYS))
        assert row.value_permutation == tuple(
            (value + row.value_offset) % VALUES for value in range(VALUES)
        )
    assert rotated_suite.base_probe_pairs == (OUTER_CELL,)
    assert rotated_suite.cell_rotations == inventory
    assert rotated_suite.probe_pairs == ALL_CELLS
    assert all(
        case.cell_rotation.apply_pair(case.base_probe_pair) == case.probe_pair
        for case in rotated_suite.cases
    )

    with pytest.raises(ValueError, match="unique"):
        build_balanced_probe_suite(
            KEYS,
            VALUES,
            (OUTER_CELL,),
            cell_rotations=(inventory[0], inventory[0]),
        )
    with pytest.raises(ValueError, match="value_permutation"):
        ProbeCellRotation(tuple(range(KEYS)), 1, tuple(range(VALUES)))
    with pytest.raises(ValueError, match="suite_sha256"):
        replace(
            rotated_suite,
            cell_rotations=tuple(reversed(rotated_suite.cell_rotations)),
        )


def test_probe_programs_use_only_generator_supported_update_arguments(
    inner_suite,
) -> None:
    update_arguments = {
        event.argument
        for case in inner_suite.cases
        for event in case.program.events
        if event.kind is BindingEventKind.UPDATE
    }
    assert update_arguments == {0, 1, 2}
    assert VALUES - 1 not in update_arguments


def test_probe_work_budgets_fail_before_materializing_large_requests() -> None:
    with pytest.raises(ProbeBudgetExceededError, match="max_cell_rotations"):
        cyclic_cell_rotation_inventory(100, 100, max_cell_rotations=20)
    with pytest.raises(ProbeBudgetExceededError, match="max_cases"):
        build_balanced_probe_suite(KEYS, VALUES, INNER_CELLS, max_cases=20)
    with pytest.raises(ProbeBudgetExceededError, match="max_probe_events"):
        build_balanced_probe_suite(
            KEYS,
            VALUES,
            (OUTER_CELL,),
            long_neutral_cycles=1_000_000,
            max_probe_events=1_000,
        )
    with pytest.raises(ProbeBudgetExceededError, match="max_probe_work"):
        build_balanced_probe_suite(
            KEYS,
            VALUES,
            (OUTER_CELL,),
            max_probe_work=100,
        )


def test_trace_only_learned_model_solves_the_actual_cell_probe_suite(
    learned_model,
    actual_suite,
) -> None:
    evaluation = evaluate_probe_suite(learned_model, actual_suite)
    actual = evaluation.result_for_pair(OUTER_CELL)
    assert (evaluation.correct_count, evaluation.query_count) == (96, 96)
    assert evaluation.accuracy == evaluation.focal_accuracy == 1.0
    assert evaluation.macro_family_accuracy == 1.0
    assert evaluation.macro_family_focal_accuracy == 1.0
    assert all(
        row.exact_case_count == row.case_count
        for row in evaluation.family_results
    )
    assert (actual.correct_count, actual.query_count) == (96, 96)
    assert actual.focal_accuracy == 1.0
    assert actual.exact_case_count == actual.case_count == 15
    assert evaluation.path_consistency == 1.0
    assert evaluation.all_path_relations_satisfied
    assert evaluation.all_copy_order_inequalities_satisfied
