from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass, replace
import hashlib
import inspect
from itertools import permutations, product
from pathlib import Path
import re

import pytest

import tnlm_v3.opaque_predictive_state as opaque_module
from tnlm_v3.opaque_predictive_state import (
    FullDiagnosticCoordinate,
    OpaqueDiagnosticInput,
    OpaqueDiagnosticScope,
    OpaqueExperimentStatus,
    OpaqueHankelBudgets,
    OpaqueMembershipAnswer,
    OpaquePredictiveStateLimitError,
    PostactiveDiscoveryResult,
    Rational,
    ToyOpaqueEnvironmentResult,
    build_toy_controller_environment,
    fit_passive_opaque_hankel,
    fit_postactive_opaque_hankel,
    release_committed_membership_answer,
    run_toy_opaque_hankel_experiment,
)


CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
TOKEN_PATTERN = re.compile(r"[0-9a-f]{32}")
CONTROLLER_NONCES = (
    "4701c24d779b2fdeeb8cd483628724ca31d594d42067990fe6d6b7dcac6adcfc",
    "62b939d3def73879797f2af76e39bcf4a463e37f8a87f1d32b7463e6ed1b7bf1",
    "b8788825db3933ad4e7aa3cf140d0a1fa3f4d5dd17270c9195c3c73ff7e45994",
    "3d1ec4129fb92f87b1878b35d25bb5e6670179b977dfdba574759d7f7fa20581",
    "2bf12926b5d1f86484cf50787c8471a93992784eb04f958bfd835e5420e59911",
    "afc32aa3ad63947352cb4ec4c31a7b7a0844e607bf39302d0545c4e67ffc489e",
    "3215fd87853097bbc32517638b9e01f8b42aa4eb8bcf49efc16015f21dc45293",
    "add1a5336bf5f3378f9b3a1f91ea98e4123a7d6ae9541e299e37a293c0b3809e",
)


def _controller_nonce(cell: tuple[int, int], block: int) -> str:
    return CONTROLLER_NONCES[block * len(CELLS) + CELLS.index(cell)]


def _controller(cell: tuple[int, int], block: int, **kwargs):
    return build_toy_controller_environment(
        cell,
        block,
        controller_nonce=_controller_nonce(cell, block),
        **kwargs,
    )


def _public_rows_under_bijection(
    learner_input: OpaqueDiagnosticInput,
    omitted_cell: tuple[int, int],
    event_order: tuple[int, ...],
    query_order: tuple[int, ...],
    answer_order: tuple[int, ...],
    history_key_order: tuple[int, ...],
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    """Rebuild a hypothesis using only public alphabets and toy dimensions."""

    event_by_cell = dict(
        zip(
            CELLS,
            tuple(learner_input.event_tokens[index] for index in event_order),
            strict=True,
        )
    )
    query_by_key = dict(
        zip(
            range(2),
            tuple(learner_input.query_tokens[index] for index in query_order),
            strict=True,
        )
    )
    answer_by_value = dict(
        zip(
            (-1, 0, 1),
            tuple(learner_input.answer_tokens[index] for index in answer_order),
            strict=True,
        )
    )
    query_key_by_token = {token: key for key, token in query_by_key.items()}
    rows = []
    for state in product((-1, 0, 1), repeat=2):
        if state[omitted_cell[0]] == omitted_cell[1]:
            continue
        word = tuple(
            event_by_cell[(key, state[key])]
            for key in history_key_order
            if state[key] >= 0
        )
        answers = tuple(
            answer_by_value[state[query_key_by_token[token]]]
            for token in learner_input.query_tokens
        )
        rows.append((word, answers))
    return tuple(sorted(rows))


def _publicly_compatible_environment_hypotheses(
    learner_input: OpaqueDiagnosticInput,
) -> tuple[tuple[int, tuple[int, int]], ...]:
    """Enumerate coordinate hypotheses without nonce or controller access."""

    observed = tuple((row.word, row.answers) for row in learner_input.passive_rows)
    compatible_cells = []
    for cell in CELLS:
        found = any(
            _public_rows_under_bijection(
                learner_input,
                cell,
                event_order,
                query_order,
                answer_order,
                history_key_order,
            )
            == observed
            for event_order in permutations(range(4))
            for query_order in permutations(range(2))
            for answer_order in permutations(range(3))
            for history_key_order in permutations(range(2))
        )
        if found:
            compatible_cells.append(cell)
    return tuple((block, cell) for block in range(2) for cell in compatible_cells)


@pytest.fixture(scope="module")
def report():
    return run_toy_opaque_hankel_experiment(controller_nonces=CONTROLLER_NONCES)


def test_learner_surface_is_opaque_atomic_and_controller_free() -> None:
    controller = _controller((0, 0), 0)
    learner_input = controller.learner_input
    assert type(learner_input) is OpaqueDiagnosticInput
    assert set(field.name for field in fields(learner_input)) == {
        "scope",
        "vocabulary_contract",
        "event_tokens",
        "query_tokens",
        "answer_tokens",
        "passive_rows",
        "candidate_words",
        "budgets",
        "passive_table_sha256",
        "candidate_pool_sha256",
        "input_sha256",
        "schema",
    }
    forbidden_fields = {
        "key",
        "value",
        "state",
        "heldout",
        "omitted",
        "semantic",
        "executor",
        "argument",
        "event_kind",
        "controller",
    }
    assert not forbidden_fields & set(field.name for field in fields(learner_input))
    assert all(
        TOKEN_PATTERN.fullmatch(token)
        for token in (
            learner_input.event_tokens
            + learner_input.query_tokens
            + learner_input.answer_tokens
        )
    )
    assert len(
        set(
            learner_input.event_tokens
            + learner_input.query_tokens
            + learner_input.answer_tokens
        )
    ) == 9
    assert all(
        len(word) == 1 and word[0] in learner_input.event_tokens
        for word in learner_input.candidate_words
    )
    assert "(0, 0)" not in repr(learner_input)
    assert "(0, 0)" not in repr(controller)
    assert "_material" not in repr(controller)
    assert CONTROLLER_NONCES[0] not in repr(learner_input)
    input_payload = opaque_module._input_payload(learner_input)
    assert not forbidden_fields & set(input_payload)
    assert "controller_nonce" not in input_payload

    passive_signature = inspect.signature(fit_passive_opaque_hankel)
    assert tuple(passive_signature.parameters) == ("value",)
    passive_syntax = ast.parse(inspect.getsource(fit_passive_opaque_hankel))
    passive_identifiers = {
        node.id for node in ast.walk(passive_syntax) if isinstance(node, ast.Name)
    } | {
        node.attr
        for node in ast.walk(passive_syntax)
        if isinstance(node, ast.Attribute)
    }
    assert "controller" not in passive_identifiers
    assert "_material" not in passive_identifiers

    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "tnlm_v3"
        / "opaque_predictive_state.py"
    )
    syntax = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = tuple(
        node
        for node in ast.walk(syntax)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    assert all(
        not isinstance(node, ast.ImportFrom) or node.level == 0 for node in imports
    )
    assert "exact_algebra" not in source_path.read_text(encoding="utf-8")
    assert "from .binding" not in source_path.read_text(encoding="utf-8")


def test_opaque_ids_are_not_the_old_bruteforceable_coordinate_hashes(report) -> None:
    for environment in report.environments:
        old_tokens = {
            hashlib.sha256(
                (
                    "tnlm-v3-phase3-toy/"
                    f"{environment.environment_index}/{namespace}/{index}"
                ).encode("utf-8")
            ).hexdigest()[:32]
            for namespace, count in (
                ("atomic-event", 4),
                ("terminal-query", 2),
                ("categorical-answer", 3),
            )
            for index in range(count)
        }
        actual_tokens = set(
            environment.learner_input.event_tokens
            + environment.learner_input.query_tokens
            + environment.learner_input.answer_tokens
        )
        assert not old_tokens & actual_tokens
        assert environment._controller_nonce not in repr(environment.learner_input)


def test_nonce_keyed_bijections_do_not_depend_on_cell_or_relabel_block() -> None:
    nonce = CONTROLLER_NONCES[0]
    controllers = tuple(
        build_toy_controller_environment(
            cell,
            block,
            controller_nonce=nonce,
        )
        for block in range(2)
        for cell in CELLS
    )
    assert len({row.learner_input.event_tokens for row in controllers}) == 1
    assert len({row.learner_input.query_tokens for row in controllers}) == 1
    assert len({row.learner_input.answer_tokens for row in controllers}) == 1
    assert len({row._material.event_by_cell for row in controllers}) == 1
    assert len({row._material.query_by_key for row in controllers}) == 1
    assert len({row._material.answer_by_value for row in controllers}) == 1
    assert len({row._material.history_key_order for row in controllers}) == 1


def test_public_eight_hypothesis_attack_cannot_deanonymize_any_input(report) -> None:
    expected_version_space = tuple(
        (block, cell) for block in range(2) for cell in CELLS
    )
    for environment in report.environments:
        learner_only = environment.learner_input
        compatible = _publicly_compatible_environment_hypotheses(learner_only)
        assert compatible == expected_version_space
        assert (environment.relabel_block, environment.pseudoheldout_cell) in compatible
        assert len(compatible) == 8


def test_all_pseudoheldout_rotations_have_exact_passive_nonidentifiability(
    report,
) -> None:
    assert report.passive_ranks == (4,) * 8
    assert tuple(
        (row.relabel_block, row.pseudoheldout_cell) for row in report.environments
    ) == tuple((block, cell) for block in range(2) for cell in CELLS)
    for environment in report.environments:
        learner_input = environment.learner_input
        passive = environment.passive
        commitment = passive.commitment
        assert len(learner_input.passive_rows) == 6
        assert len(passive.model.rank_certificate.matrix) == 6
        assert len(passive.model.rank_certificate.matrix[0]) == 6
        assert passive.exact_rank == 4
        assert passive.ambiguity_dimension == 1
        assert passive.exact_rank_evaluation_count == 14
        assert passive.model.basis_determinant.numerator != 0
        assert passive.span_answer_tuples == tuple(
            sorted(row.answers for row in learner_input.passive_rows)
        )
        assert len(commitment.compatible_completions) == 3
        coverage = commitment.vocabulary_coverage
        assert len(coverage.observed_slots) == 3
        assert coverage.missing_slot not in coverage.observed_slots
        missing_query_index = learner_input.query_tokens.index(
            coverage.missing_slot[0]
        )
        assert all(
            witness.augmented_rank.rank == 5
            and witness.augmented_rank.witness_determinant.numerator != 0
            and witness.answers[missing_query_index] == coverage.missing_slot[1]
            and witness.augmented_rank.matrix[:-1]
            == passive.model.rank_certificate.matrix
            for witness in commitment.compatible_completions
        )
        compatible_answers = tuple(
            witness.answers for witness in commitment.compatible_completions
        )
        assert commitment.twin_answers[0] != commitment.twin_answers[1]
        assert all(twin in compatible_answers for twin in commitment.twin_answers)
        assert commitment.selected_word == learner_input.unlabeled_candidate_words[0]
        assert commitment.selected_word not in {
            row.word for row in learner_input.passive_rows
        }
        assert commitment.known_control_word in {
            row.word for row in learner_input.passive_rows
        }
        assert commitment.known_control_rank.rank == 4
        assert commitment.preactive_membership_calls == 0


def test_fixed_pool_commitment_precedes_and_does_not_choose_an_answer() -> None:
    controller = _controller((1, 0), 1)
    learner_input = controller.learner_input
    first = fit_passive_opaque_hankel(learner_input)
    second = fit_passive_opaque_hankel(learner_input)
    assert first == second
    assert first.commitment.candidate_pool_sha256 == learner_input.candidate_pool_sha256
    assert first.commitment.preactive_membership_calls == 0
    assert not hasattr(first.commitment, "selected_answer")
    assert "answer_sha256" not in opaque_module._commitment_payload(first.commitment)
    assert tuple(
        row.answers for row in first.commitment.compatible_completions
    ) == tuple(sorted(row.answers for row in first.commitment.compatible_completions))

    active_answer = release_committed_membership_answer(controller, first)
    assert active_answer.commitment_sha256 == first.commitment.commitment_sha256
    assert active_answer.word == first.commitment.selected_word
    assert active_answer.answers in tuple(
        row.answers for row in first.commitment.compatible_completions
    )


def test_known_and_aliased_answers_do_not_fake_information_gain() -> None:
    controller = _controller((0, 1), 0)
    learner_input = controller.learner_input
    passive = fit_passive_opaque_hankel(learner_input)
    assert passive.commitment.known_control_rank.rank == passive.exact_rank == 4

    aliased_answers = passive.span_answer_tuples[0]
    aliased_matrix = passive.model.rank_certificate.matrix + (
        opaque_module._diagnostic_vector(learner_input, aliased_answers),
    )
    aliased_rank = opaque_module._make_rank_certificate(aliased_matrix)
    assert aliased_rank.rank == 4
    assert all(
        witness.augmented_rank.rank == 5
        for witness in passive.commitment.compatible_completions
    )


def test_one_membership_response_restores_rank_five_and_all_diagnostics(
    report,
) -> None:
    assert report.postactive_ranks == (5,) * 8
    for environment in report.environments:
        passive = environment.passive
        answer = environment.active_answer
        postactive = environment.postactive
        assert answer.response_ordinal == 1
        assert len(answer.answers) == 2
        assert postactive.active_membership_calls == 1
        assert postactive.rebuild_count == 1
        assert postactive.exact_rank == 5
        assert postactive.model.basis_determinant.numerator != 0
        assert len(postactive.model.source_words) == 7
        assert postactive.full_diagnostic_count == 9
        assert tuple(row.answers for row in postactive.full_diagnostics) == tuple(
            product(
                environment.learner_input.answer_tokens,
                repeat=len(environment.learner_input.query_tokens),
            )
        )
        opened_witness = passive.commitment.compatible_completions[
            postactive.compatible_completion_index
        ]
        assert opened_witness.answers == answer.answers
        assert (
            opened_witness.witness_sha256
            == postactive.compatible_completion_witness_sha256
        )
        for row in postactive.full_diagnostics:
            assert len(row.coordinates) == 5
            assert opaque_module._row_times_matrix(
                row.coordinates, postactive.model.readout_matrix
            ) == row.diagnostic_row


@pytest.mark.parametrize(
    ("cell", "block", "completion_index"),
    tuple(
        (cell, block, completion_index)
        for block in range(2)
        for cell in CELLS
        for completion_index in range(3)
    ),
)
def test_postactive_rebuild_is_total_over_every_committed_compatible_answer(
    cell: tuple[int, int], block: int, completion_index: int
) -> None:
    controller = _controller(cell, block)
    learner_input = controller.learner_input
    passive = fit_passive_opaque_hankel(learner_input)
    witness = passive.commitment.compatible_completions[completion_index]
    answer_payload = {
        "schema": "tnlm-v3-opaque-membership-answer-v1",
        "input_sha256": learner_input.input_sha256,
        "commitment_sha256": passive.commitment.commitment_sha256,
        "word": passive.commitment.selected_word,
        "answers": witness.answers,
        "response_ordinal": 1,
    }
    answer = OpaqueMembershipAnswer(
        input_sha256=learner_input.input_sha256,
        commitment_sha256=passive.commitment.commitment_sha256,
        word=passive.commitment.selected_word,
        answers=witness.answers,
        response_ordinal=1,
        answer_sha256=opaque_module._sha256(answer_payload),
    )
    postactive = fit_postactive_opaque_hankel(learner_input, passive, answer)

    assert postactive.compatible_completion_index == completion_index
    assert postactive.compatible_completion_witness_sha256 == witness.witness_sha256
    assert postactive.exact_rank == 5
    assert postactive.full_diagnostic_count == 9
    assert all(
        opaque_module._row_times_matrix(
            row.coordinates, postactive.model.readout_matrix
        )
        == row.diagnostic_row
        for row in postactive.full_diagnostics
    )


def test_controller_attribution_confirms_selected_word_is_each_missing_token(
    report,
) -> None:
    for environment in report.environments:
        controller = build_toy_controller_environment(
            environment.pseudoheldout_cell,
            environment.relabel_block,
            controller_nonce=_controller_nonce(
                environment.pseudoheldout_cell, environment.relabel_block
            ),
            budgets=environment.learner_input.budgets,
        )
        event_lookup = dict(controller._material.event_by_cell)
        assert environment.passive.commitment.selected_word == (
            event_lookup[environment.pseudoheldout_cell],
        )
        assert environment.active_answer == release_committed_membership_answer(
            controller, environment.passive
        )


def test_opaque_relabel_blocks_are_disjoint_and_rationally_similar(report) -> None:
    assert len(report.similarities) == 4
    by_key = {
        (row.relabel_block, row.pseudoheldout_cell): row
        for row in report.environments
    }
    for cell, certificate in zip(CELLS, report.similarities, strict=True):
        left = by_key[(0, cell)]
        right = by_key[(1, cell)]
        left_tokens = set(
            left.learner_input.event_tokens
            + left.learner_input.query_tokens
            + left.learner_input.answer_tokens
        )
        right_tokens = set(
            right.learner_input.event_tokens
            + right.learner_input.query_tokens
            + right.learner_input.answer_tokens
        )
        assert not left_tokens & right_tokens
        assert left.learner_input.input_sha256 != right.learner_input.input_sha256
        assert left.passive.commitment.selected_word != right.passive.commitment.selected_word
        assert left.active_answer.answers != right.active_answer.answers
        assert certificate.pseudoheldout_cell == cell
        assert certificate.determinant.numerator != 0
        assert certificate.fit_equation_count == 5
        assert certificate.test_equation_count == 4
        assert certificate.fit_state_set_sha256 != certificate.test_state_set_sha256
        assert certificate == opaque_module._make_similarity_certificate(left, right)

        controller_left = _controller(cell, 0)
        controller_right = _controller(cell, 1)
        assert controller_left._material.controller_nonce != (
            controller_right._material.controller_nonce
        )


def test_nested_hashes_and_semantic_invariants_reject_forgery(report) -> None:
    environment = report.environments[0]
    with pytest.raises(ValueError, match="input_sha256"):
        replace(environment.learner_input, input_sha256="0" * 64)
    with pytest.raises(ValueError, match="determinant"):
        replace(
            environment.passive.model.rank_certificate,
            witness_determinant=Rational(2),
        )
    with pytest.raises(ValueError, match="report_sha256"):
        replace(report, report_sha256="0" * 64)

    first_witness, second_witness = (
        environment.passive.commitment.compatible_completions[:2]
    )
    rebound_witness_payload = {
        "schema": first_witness.schema,
        "answers": second_witness.answers,
        "answer_tokens": first_witness.answer_tokens,
        "augmented_rank_sha256": first_witness.augmented_rank.certificate_sha256,
    }
    with pytest.raises(ValueError, match="do not encode"):
        opaque_module.PassiveCompletionWitness(
            answers=second_witness.answers,
            answer_tokens=first_witness.answer_tokens,
            augmented_rank=first_witness.augmented_rank,
            witness_sha256=opaque_module._sha256(rebound_witness_payload),
        )

    first = environment.postactive.full_diagnostics[0]
    second = environment.postactive.full_diagnostics[1]
    rebound_coordinate_payload = {
        "schema": first.schema,
        "answers": second.answers,
        "answer_tokens": first.answer_tokens,
        "diagnostic_row": first.diagnostic_row,
        "coordinates": first.coordinates,
    }
    with pytest.raises(ValueError, match="do not encode"):
        FullDiagnosticCoordinate(
            answers=second.answers,
            answer_tokens=first.answer_tokens,
            diagnostic_row=first.diagnostic_row,
            coordinates=first.coordinates,
            coordinate_sha256=opaque_module._sha256(
                rebound_coordinate_payload
            ),
        )
    corrupted_coordinates = (
        Rational(first.coordinates[0].numerator + 1, first.coordinates[0].denominator),
    ) + first.coordinates[1:]
    corrupted_payload = {
        "schema": first.schema,
        "answers": first.answers,
        "answer_tokens": first.answer_tokens,
        "diagnostic_row": first.diagnostic_row,
        "coordinates": corrupted_coordinates,
    }
    corrupted_row = FullDiagnosticCoordinate(
        answers=first.answers,
        answer_tokens=first.answer_tokens,
        diagnostic_row=first.diagnostic_row,
        coordinates=corrupted_coordinates,
        coordinate_sha256=opaque_module._sha256(corrupted_payload),
    )
    corrupted_rows = (corrupted_row,) + environment.postactive.full_diagnostics[1:]
    with pytest.raises(ValueError, match="does not reconstruct"):
        PostactiveDiscoveryResult(
            input_sha256=environment.postactive.input_sha256,
            passive_result_sha256=environment.postactive.passive_result_sha256,
            answer_sha256=environment.postactive.answer_sha256,
            model=environment.postactive.model,
            full_diagnostics=corrupted_rows,
            compatible_completion_index=(
                environment.postactive.compatible_completion_index
            ),
            compatible_completion_witness_sha256=(
                environment.postactive.compatible_completion_witness_sha256
            ),
            active_membership_calls=1,
            rebuild_count=1,
            result_sha256=environment.postactive.result_sha256,
        )

    changed_cell = (0, 1)
    forged_environment_payload = {
        "schema": environment.schema,
        "environment_index": 1,
        "pseudoheldout_cell": changed_cell,
        "relabel_block": 0,
        "learner_input_sha256": environment.learner_input.input_sha256,
        "controller_commitment_sha256": environment.controller_commitment_sha256,
        "passive_result_sha256": environment.passive.result_sha256,
        "active_answer_sha256": environment.active_answer.answer_sha256,
        "postactive_result_sha256": environment.postactive.result_sha256,
        "controller_nonce_sha256": opaque_module._sha256(
            environment._controller_nonce
        ),
    }
    with pytest.raises(ValueError, match="deterministic controller"):
        ToyOpaqueEnvironmentResult(
            environment_index=1,
            pseudoheldout_cell=changed_cell,
            relabel_block=0,
            learner_input=environment.learner_input,
            controller_commitment_sha256=environment.controller_commitment_sha256,
            passive=environment.passive,
            active_answer=environment.active_answer,
            postactive=environment.postactive,
            _controller_nonce=environment._controller_nonce,
            environment_sha256=opaque_module._sha256(forged_environment_payload),
        )


@pytest.mark.parametrize(
    ("field_name", "too_small"),
    (
        ("max_word_length", 1),
        ("max_suffix_test_candidates", 5),
        ("max_oracle_evaluations", 5),
        ("max_active_candidate_words", 3),
        ("max_basis_dimension", 4),
        ("max_complete_diagnostic_rows", 8),
    ),
)
def test_analytic_budgets_fail_before_construction(
    field_name: str, too_small: int
) -> None:
    budgets = replace(OpaqueHankelBudgets(), **{field_name: too_small})
    with pytest.raises(OpaquePredictiveStateLimitError, match=field_name):
        _controller((0, 0), 0, budgets=budgets)


def test_budget_schema_rejects_zero_response_and_protocol_ceiling_expansion() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        replace(OpaqueHankelBudgets(), max_active_responses=0)
    with pytest.raises(ValueError, match="exceeds"):
        replace(OpaqueHankelBudgets(), max_oracle_evaluations=50_001)


def test_controller_nonce_inventory_is_explicit_strict_and_unique() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        build_toy_controller_environment(
            (0, 0), 0, controller_nonce="0" * 32
        )
    with pytest.raises(ValueError, match="distinct"):
        run_toy_opaque_hankel_experiment(
            controller_nonces=(CONTROLLER_NONCES[0],) * 8
        )
    with pytest.raises(TypeError):
        run_toy_opaque_hankel_experiment(
            controller_nonces=list(CONTROLLER_NONCES)  # type: ignore[arg-type]
        )


def _walk_values(value: object):
    yield value
    if is_dataclass(value):
        for item in fields(value):
            yield from _walk_values(getattr(value, item.name))
    elif isinstance(value, tuple):
        for item in value:
            yield from _walk_values(item)


def test_report_is_exact_deterministic_and_honestly_scoped(report) -> None:
    repeated = run_toy_opaque_hankel_experiment(
        controller_nonces=CONTROLLER_NONCES
    )
    assert repeated == report
    assert repeated.report_sha256 == report.report_sha256
    assert report.passed
    assert report.status is (
        OpaqueExperimentStatus.SYNTHETIC_PROTOCOL_IMPLEMENTATION_REHEARSAL
    )
    assert report.scope is OpaqueDiagnosticScope.ZERO_SUFFIX_ABSENCE_DIAGNOSTIC_BLOCK
    assert "zero_suffix" in report.scope.value
    assert "operator" not in report.scope.value
    assert "factor" not in report.scope.value
    assert all(nonce not in repr(report) for nonce in CONTROLLER_NONCES)
    assert not any(type(value) is float for value in _walk_values(report))


def test_postactive_fit_rejects_uncommitted_or_second_response() -> None:
    controller = _controller((1, 1), 1)
    passive = fit_passive_opaque_hankel(controller.learner_input)
    answer = release_committed_membership_answer(controller, passive)
    with pytest.raises(ValueError, match="exactly one"):
        replace(answer, response_ordinal=2)
    other_controller = _controller((1, 0), 1)
    other_passive = fit_passive_opaque_hankel(other_controller.learner_input)
    other_answer = release_committed_membership_answer(other_controller, other_passive)
    with pytest.raises(ValueError, match="does not match"):
        fit_postactive_opaque_hankel(
            controller.learner_input,
            passive,
            other_answer,
        )
