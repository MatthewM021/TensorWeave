from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass, replace
import inspect
from itertools import product
import re

import pytest

import tnlm_v3.opaque_partial_operators as partial_module


CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
CONTROLLER_NONCES = (
    "68e1a01aa80a042b82e04f36dfe696a91f4ffd9f82d5f311a027d73adbd2a9a5",
    "a57aebc4843c69b6d5fdff38d49523b3dedbf7cf55ccdd822745f7e14c14304e",
    "323b4597f9e3dfb9e7f954aab5873b8730321dd8476a2992f81e93cc7a44b046",
    "45d4ff9e973a3c9fda9a41ae7ab8c827f9e7299a2ece6f402ab32dc0fd598718",
    "a3d1591df9b8fe82666f78e885f578cfd5677df28dfa6bc2daa5cfa468d87203",
    "0ec845c35cfe807b28f8d012391d9e811211a780e3fd22510ad33a6059dcf5fc",
    "422d894dade5755a90355dea9dec66be8a8943509365c79727046eb895042bb4",
    "be187a3a14f4d575944e93a407f0e1c53142a9db4c27abc0dcb6bbb3d7712df3",
    "219803a3b13f67dbb90f23749b785d71836918b7d5d0cee78d74d85ff0319490",
    "8e9f16318b215a2e117e34bc75cbc649b602dac9d8ce5ffcdace6950c8971ce2",
)
TOKEN_PATTERN = re.compile(r"[0-9a-f]{32}")
EXPECTED_LEGAL_SOURCE_RANKS = {
    "bind": (3, 3, 3, 3),
    "update": (4, 4),
    "copy": (3, 3),
    "invalidate": (4, 4),
}
EXPECTED_TOTAL_EXTENSION_NULLITIES = {
    "bind": (10, 10, 10, 10),
    "update": (5, 5),
    "copy": (10, 10),
    "invalidate": (5, 5),
}


def _full_controller(block: int):
    return partial_module.build_full_support_controller_environment(
        block,
        controller_nonce=CONTROLLER_NONCES[block],
    )


def _omission_controller(cell: tuple[int, int], block: int):
    return partial_module.build_omission_controller_environment(
        cell,
        block,
        controller_nonce=CONTROLLER_NONCES[2 + block * len(CELLS) + CELLS.index(cell)],
    )


@pytest.fixture(scope="module")
def full_controllers():
    return tuple(_full_controller(block) for block in range(2))


@pytest.fixture(scope="module")
def omission_controllers():
    return tuple(
        _omission_controller(cell, block)
        for block in range(2)
        for cell in CELLS
    )


@pytest.fixture(scope="module")
def omission_fits(omission_controllers):
    rows = []
    for controller in omission_controllers:
        learner_input = controller.learner_input
        passive = partial_module.fit_passive_partial_operators(learner_input)
        first = partial_module.release_first_active_response(controller, passive)
        checkpoint = partial_module.analyze_one_response_checkpoint(
            learner_input, passive, first
        )
        remaining = partial_module.release_remaining_active_responses(
            controller, passive, checkpoint
        )
        active = partial_module.fit_active_partial_operators(
            learner_input,
            passive,
            first,
            checkpoint,
            remaining,
        )
        rows.append((controller, passive, first, checkpoint, remaining, active))
    return tuple(rows)


@pytest.fixture(scope="module")
def report():
    return partial_module.run_toy_partial_operator_experiment(
        controller_nonces=CONTROLLER_NONCES
    )


def _walk_values(value: object):
    yield value
    if is_dataclass(value):
        for item in fields(value):
            yield from _walk_values(getattr(value, item.name))
    elif isinstance(value, tuple):
        for item in value:
            yield from _walk_values(item)


def test_public_learner_schema_contains_only_opaque_partial_contract_data() -> None:
    learner_fields = {field.name for field in fields(partial_module.PartialOperatorLearnerInput)}
    assert learner_fields == {
        "event_tokens",
        "query_tokens",
        "answer_tokens",
        "passive_state_observations",
        "passive_edge_observations",
        "candidate_edge_requests",
        "defined_edge_requests",
        "undefined_edge_requests",
        "budgets",
        "domain_mask_is_learner_visible_supervision",
        "undefined_words_are_absent_constraints",
        "illegal_words_encoded_as_zero_or_dead",
        "state_changing_event_subalphabet_only",
        "diagnostic_queries_are_output_channels_not_events",
        "categorical_state_space_declared_as_full_product",
        "full_product_state_grammar_counted_as_supervision",
        "passive_table_sha256",
        "domain_mask_sha256",
        "candidate_pool_sha256",
        "input_sha256",
        "schema",
    }
    forbidden = {
        "key",
        "value",
        "state",
        "cell",
        "heldout",
        "omitted",
        "semantic",
        "family",
        "executor",
        "controller",
        "nonce",
        "event_kind",
        "argument",
    }
    assert not forbidden.intersection(learner_fields)


def test_default_budget_can_hold_both_frozen_protocol_arms() -> None:
    budgets = partial_module.PartialOperatorBudgets()
    assert budgets.max_domain_edges == 90
    assert budgets.max_passive_edges == 44
    assert budgets.max_active_responses == 15
    assert budgets.max_categorical_labels == 30
    assert budgets.max_sealed_edges == 8
    assert budgets.max_long_probes == 12


def test_membership_response_contains_only_two_target_labels() -> None:
    response_fields = {
        field.name for field in fields(partial_module.OpaqueMembershipResponse)
    }
    assert response_fields == {
        "request",
        "target_answers",
        "response_ordinal",
        "response_sha256",
        "schema",
    }
    assert "source_answers" not in response_fields
    commitment_fields = {
        field.name for field in fields(partial_module.ActiveAcquisitionCommitment)
    }
    assert {
        "active_batch_call_count",
        "expected_response_count",
        "expected_returned_categorical_token_count",
    } <= commitment_fields


@pytest.mark.parametrize(
    ("field_name", "too_small"),
    (
        ("max_word_length", 1),
        ("max_event_tokens", 9),
        ("max_domain_edges", 89),
        ("max_passive_edges", 20),
        ("max_active_responses", 14),
        ("max_categorical_labels", 29),
        ("max_sealed_edges", 7),
        ("max_long_probes", 11),
        ("max_basis_dimension", 4),
    ),
)
def test_frozen_budgets_have_hostile_underflow_cases(
    field_name: str, too_small: int
) -> None:
    # Construction functions are tested below for fail-before-work behavior.
    # Here the exact smaller values remain constructible so each boundary can
    # be exercised rather than silently rounded back up.
    budgets = replace(
        partial_module.PartialOperatorBudgets(), **{field_name: too_small}
    )
    assert getattr(budgets, field_name) == too_small


def test_partial_operator_schema_forbids_a_total_matrix_surface() -> None:
    operator_fields = {
        field.name
        for field in fields(partial_module.ExactPartialOperatorCertificate)
    }
    realization_fields = {
        field.name for field in fields(partial_module.ExactPartialRealization)
    }
    active_fields = {
        field.name for field in fields(partial_module.ActivePartialDiscovery)
    }
    assert "total_operator" in operator_fields
    assert "total_operator" in realization_fields
    assert "total_operator" in active_fields
    assert "total_operator_matrix" not in operator_fields | realization_fields | active_fields


def test_two_full_controls_have_the_exact_44_edge_support(full_controllers) -> None:
    assert len(full_controllers) == 2
    assert {controller.relabel_block for controller in full_controllers} == {0, 1}
    for controller in full_controllers:
        learner_input = controller.learner_input
        assert controller.kind is partial_module.EnvironmentKind.FULL_SUPPORT_CONTROL
        assert controller.pseudoheldout_cell is None
        assert len(learner_input.passive_state_observations) == 9
        assert len(learner_input.passive_edge_observations) == 44
        assert len(learner_input.candidate_edge_requests) == 0
        assert len(learner_input.defined_edge_requests) == 44
        assert len(learner_input.undefined_edge_requests) == 46
        assert controller.passive_semantic_family_counts == (
            ("bind", 12),
            ("copy", 8),
            ("invalidate", 12),
            ("update", 12),
        )
        assert not controller.active_responses
        assert not controller.sealed_edge_programs
        assert len(controller.long_path_programs) == 12

        passive = partial_module.fit_passive_partial_operators(learner_input)
        assert passive.passive_rank_certificate.rank == 5
        assert passive.realization.ambient_rank == 5
        assert passive.realization.restricted_maps_complete
        assert len(passive.realization.operator_certificates) == 10
        assert passive.compatible_omission_hypothesis_count is None
        assert not passive.omission_hypothesis_analysis_performed_inside_estimator
        assert passive.active_commitment.expected_response_count == 0
        assert passive.active_commitment.active_batch_call_count == 0


def test_all_eight_omission_inputs_have_the_exact_21_15_8_partition(
    omission_controllers,
) -> None:
    assert len(omission_controllers) == 8
    assert {
        (controller.relabel_block, controller.pseudoheldout_cell)
        for controller in omission_controllers
    } == {(block, cell) for block in range(2) for cell in CELLS}
    for controller in omission_controllers:
        learner_input = controller.learner_input
        assert controller.kind is partial_module.EnvironmentKind.ROTATED_OMISSION
        assert len(learner_input.passive_state_observations) == 6
        assert len(learner_input.passive_edge_observations) == 21
        assert len(learner_input.candidate_edge_requests) == 23
        assert len(learner_input.defined_edge_requests) == 44
        assert len(learner_input.undefined_edge_requests) == 46
        assert len(controller.active_responses) == 15
        assert len(controller.sealed_edge_programs) == 8
        assert len(controller.long_path_programs) == 12
        assert controller.passive_semantic_family_counts == (
            ("bind", 7),
            ("copy", 3),
            ("invalidate", 7),
            ("update", 4),
        )
        passive_hashes = {
            row.request.request_sha256
            for row in learner_input.passive_edge_observations
        }
        candidate_hashes = {
            row.request_sha256 for row in learner_input.candidate_edge_requests
        }
        defined_hashes = {
            row.request_sha256 for row in learner_input.defined_edge_requests
        }
        undefined_hashes = {
            row.request_sha256 for row in learner_input.undefined_edge_requests
        }
        assert passive_hashes.isdisjoint(candidate_hashes)
        assert passive_hashes | candidate_hashes == defined_hashes
        assert defined_hashes.isdisjoint(undefined_hashes)


def test_learner_inputs_bind_definedness_without_zero_or_dead_encoding(
    full_controllers, omission_controllers
) -> None:
    for controller in full_controllers + omission_controllers:
        learner_input = controller.learner_input
        assert learner_input.domain_mask_is_learner_visible_supervision
        assert learner_input.undefined_words_are_absent_constraints
        assert not learner_input.illegal_words_encoded_as_zero_or_dead
        assert learner_input.state_changing_event_subalphabet_only
        assert learner_input.diagnostic_queries_are_output_channels_not_events
        assert learner_input.categorical_state_space_declared_as_full_product
        assert learner_input.full_product_state_grammar_counted_as_supervision

        event_tokens = set(learner_input.event_tokens)
        query_tokens = set(learner_input.query_tokens)
        answer_tokens = set(learner_input.answer_tokens)
        assert (len(event_tokens), len(query_tokens), len(answer_tokens)) == (10, 2, 3)
        assert event_tokens.isdisjoint(query_tokens | answer_tokens)
        assert query_tokens.isdisjoint(answer_tokens)
        assert all(
            TOKEN_PATTERN.fullmatch(token)
            for token in event_tokens | query_tokens | answer_tokens
        )
        for row in (
            learner_input.defined_edge_requests
            + learner_input.undefined_edge_requests
            + learner_input.candidate_edge_requests
        ):
            assert row.event_token in event_tokens
            assert row.program == row.source_word + (row.event_token,)
            assert set(row.program) <= event_tokens
        assert controller.trusted_controller_nonce not in repr(learner_input)


def test_public_learner_functions_have_no_controller_or_semantic_arguments() -> None:
    functions = (
        partial_module.fit_passive_partial_operators,
        partial_module.analyze_one_response_checkpoint,
        partial_module.fit_active_partial_operators,
        partial_module.publicly_compatible_omission_hypothesis_count,
    )
    forbidden_argument_fragments = {
        "cell",
        "heldout",
        "omitted",
        "semantic",
        "family",
        "executor",
        "controller",
        "nonce",
        "key",
        "value",
    }
    forbidden_names = {
        "_semantic_step",
        "_execute_semantic",
        "_canonical_to_raw_maps",
        "_map_state_to_raw",
        "_map_action_to_raw",
        "ToyPartialControllerEnvironment",
    }
    for function in functions:
        parameter_names = set(inspect.signature(function).parameters)
        assert not any(
            fragment in name
            for fragment in forbidden_argument_fragments
            for name in parameter_names
        )
        tree = ast.parse(inspect.getsource(function))
        used_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert used_names.isdisjoint(forbidden_names)


def test_pure_fit_call_graph_does_not_reach_semantic_hypothesis_enumerator(
    monkeypatch, omission_controllers
) -> None:
    controller = omission_controllers[0]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pure learner reached controller/evaluator semantics")

    monkeypatch.setattr(partial_module, "_semantic_step", forbidden)
    monkeypatch.setattr(
        partial_module,
        "enumerate_publicly_compatible_omission_hypotheses",
        forbidden,
    )
    monkeypatch.setattr(
        partial_module,
        "publicly_compatible_omission_hypothesis_count",
        forbidden,
    )
    passive = partial_module.fit_passive_partial_operators(controller.learner_input)
    first = partial_module.release_first_active_response(controller, passive)
    checkpoint = partial_module.analyze_one_response_checkpoint(
        controller.learner_input, passive, first
    )
    remaining = partial_module.release_remaining_active_responses(
        controller, passive, checkpoint
    )
    active = partial_module.fit_active_partial_operators(
        controller.learner_input,
        passive,
        first,
        checkpoint,
        remaining,
    )
    assert passive.compatible_omission_hypothesis_count is None
    assert active.active_rank_certificate.rank == 5


def test_passive_omission_rank_is_four_with_the_exact_missing_source_spans(
    omission_fits,
) -> None:
    expected_observed_ranks = {
        ("bind", 0, 0): 0,
        ("bind", 0, 1): 3,
        ("bind", 1, 0): 2,
        ("bind", 1, 1): 2,
        ("update", 0, -1): 0,
        ("update", 1, -1): 3,
        ("copy", 0, 1): 1,
        ("copy", 1, 0): 2,
        ("invalidate", 0, -1): 3,
        ("invalidate", 1, -1): 3,
    }
    for controller, passive, *_ in omission_fits:
        assert passive.passive_edge_count == 21
        assert passive.passive_rank_certificate.rank == 4
        assert passive.realization.ambient_rank == 4
        assert not passive.realization.restricted_maps_complete
        assert not passive.realization.operator_certificates
        assert passive.compatible_omission_hypothesis_count is None
        assert not passive.omission_hypothesis_analysis_performed_inside_estimator
        ranks_by_token = {
            row.event_token: row.observed_source_rank
            for row in passive.observed_event_ranks
        }
        assert {
            action: ranks_by_token[token]
            for action, token in controller.canonical_event_tokens
        } == expected_observed_ranks


def test_first_response_raises_rank_but_cannot_close_the_partial_operators(
    omission_fits,
) -> None:
    for controller, passive, first, checkpoint, remaining, _ in omission_fits:
        commitment = passive.active_commitment
        assert commitment.selected_before_any_active_answer
        assert commitment.active_basis_supplied_by_trusted_controller
        assert commitment.controller_order_encodes_semantically_designed_basis
        assert not commitment.learner_selected_acquisition_basis
        assert not commitment.learner_selection_used_semantic_roles
        assert not commitment.learner_selection_used_controller_nonce
        assert not commitment.selection_used_sealed_answers
        assert commitment.active_batch_call_count == 2
        assert commitment.expected_response_count == 15
        assert commitment.expected_returned_categorical_token_count == 30
        assert first.request.request_sha256 == commitment.selected_request_sha256s[0]
        assert first.response_ordinal == 1
        assert len(first.target_answers) == 2
        assert len(remaining) == 14

        known_source_words = {
            row.word for row in controller.learner_input.passive_state_observations
        }
        for row in controller.learner_input.passive_edge_observations:
            known_source_words.add(row.request.source_word)
            known_source_words.add(row.request.program)
        assert first.request.source_word in known_source_words
        assert first.request.program not in known_source_words

        assert checkpoint.response_count == 1
        assert checkpoint.returned_categorical_token_count == 2
        assert checkpoint.rank_certificate.rank == 5
        assert checkpoint.rank_after_first_response == 5
        assert not checkpoint.operator_maps_identified
        assert checkpoint.unidentified_event_token_count == 9
        assert checkpoint.unanswered_candidate_request_count == 22
        assert checkpoint.compatible_outcome_analysis_used_first_response_labels
        assert not checkpoint.compatible_outcome_analysis_used_responses_2_through_15
        assert not checkpoint.actual_next_response_read
        assert not checkpoint.sealed_answer_read
        assert checkpoint.differing_restricted_map_witness
        assert checkpoint.compatible_outcome_a != checkpoint.compatible_outcome_b
        assert len(checkpoint.event_rank_deficits) == 10
        assert sum(
            row.remaining_source_rank_deficit > 0
            for row in checkpoint.event_rank_deficits
        ) == 9
        assert sum(
            row.remaining_source_rank_deficit
            for row in checkpoint.event_rank_deficits
        ) == 14
        assert checkpoint.aggregate_remaining_source_rank_deficit == 14
        assert checkpoint.branch_source_rank == 2
        assert checkpoint.branch_a_linear_system_consistent
        assert checkpoint.branch_b_linear_system_consistent
        assert checkpoint.differing_image_row_index == 1
        assert any(value.numerator for value in checkpoint.differing_image_delta)
        assert checkpoint.compatible_outcomes_are_exact_unobserved_full_product_rows
        assert set(checkpoint.compatible_outcome_a) <= set(
            controller.learner_input.answer_tokens
        )
        assert set(checkpoint.compatible_outcome_b) <= set(
            controller.learner_input.answer_tokens
        )
        known_rows = {
            row.answers
            for row in controller.learner_input.passive_state_observations
        }
        known_rows.update(
            row.source_answers
            for row in controller.learner_input.passive_edge_observations
        )
        known_rows.update(
            row.target_answers
            for row in controller.learner_input.passive_edge_observations
        )
        known_rows.add(first.target_answers)
        full_product = set(product(controller.learner_input.answer_tokens, repeat=2))
        assert {
            checkpoint.compatible_outcome_a,
            checkpoint.compatible_outcome_b,
        } == full_product - known_rows


def test_fifteen_two_label_responses_close_every_restricted_map(
    omission_fits,
) -> None:
    expected_domain_rank_by_family = {
        "bind": 3,
        "update": 4,
        "copy": 3,
        "invalidate": 4,
    }
    expected_extension_by_family = {
        "bind": 10,
        "update": 5,
        "copy": 10,
        "invalidate": 5,
    }
    for controller, passive, first, _, remaining, active in omission_fits:
        responses = (first,) + remaining
        assert tuple(row.response_ordinal for row in responses) == tuple(range(1, 16))
        assert tuple(row.request.request_sha256 for row in responses) == (
            passive.active_commitment.selected_request_sha256s
        )
        assert all(len(row.target_answers) == 2 for row in responses)
        assert active.active_batch_call_count == 2
        assert active.active_response_count == 15
        assert active.returned_categorical_token_count == 30
        assert active.returned_target_label_fields_per_response == 2
        assert active.active_rank_certificate.rank == 5
        assert active.realization.ambient_rank == 5
        assert active.realization.restricted_maps_complete
        assert active.restricted_legal_domain_maps_identified
        assert len(active.operator_certificates) == 10
        assert active.aggregate_total_extension_nullity == 80
        assert active.legal_domain_rank_multiset == (
            3,
            3,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
        )
        assert active.total_extension_nullity_multiset == (
            5,
            5,
            5,
            5,
            10,
            10,
            10,
            10,
            10,
            10,
        )
        operator_by_token = {
            row.event_token: row for row in active.operator_certificates
        }
        for action, token in controller.canonical_event_tokens:
            row = operator_by_token[token]
            family = action[0]
            assert row.ambient_rank == 5
            assert row.legal_domain_rank == expected_domain_rank_by_family[family]
            assert row.observed_source_rank == row.legal_domain_rank
            assert row.restricted_nullity == 0
            assert row.total_extension_nullity == expected_extension_by_family[family]
            assert len(row.off_domain_annihilator_basis) == 5 - row.legal_domain_rank
            assert row.restricted_map_identified
            assert not row.off_domain_extension_identified
            assert not row.undefined_inputs_encoded_as_zero_or_dead
            assert row.total_operator is None
        assert active.total_operator is None
        assert active.realization.total_operator is None


def test_full_controls_recover_the_same_legal_source_ranks_without_acquisition(
    full_controllers,
) -> None:
    expected = {"bind": (3, 10), "update": (4, 5), "copy": (3, 10), "invalidate": (4, 5)}
    for controller in full_controllers:
        passive = partial_module.fit_passive_partial_operators(controller.learner_input)
        ranks = {row.event_token: row for row in passive.observed_event_ranks}
        operators = {
            row.event_token: row for row in passive.realization.operator_certificates
        }
        for action, token in controller.canonical_event_tokens:
            family = action[0]
            domain_rank, extension_nullity = expected[family]
            assert ranks[token].observed_source_rank == domain_rank
            assert operators[token].legal_domain_rank == domain_rank
            assert operators[token].observed_source_rank == domain_rank
            assert operators[token].total_extension_nullity == extension_nullity
            assert operators[token].total_operator is None


def test_active_response_sources_are_known_before_each_two_label_return(
    omission_fits,
) -> None:
    for controller, _, first, _, remaining, _ in omission_fits:
        known = {
            row.word: row.answers
            for row in controller.learner_input.passive_state_observations
        }
        for row in controller.learner_input.passive_edge_observations:
            known.setdefault(row.request.source_word, row.source_answers)
            known.setdefault(row.request.program, row.target_answers)
        for response in (first,) + remaining:
            assert response.request.source_word in known
            assert response.request.program not in known
            assert not hasattr(response, "source_answers")
            known[response.request.program] = response.target_answers


def test_eight_sealed_edges_and_twelve_long_programs_are_exact(
    omission_fits,
) -> None:
    for controller, *_, active in omission_fits:
        sealed = controller.sealed_edge_programs
        long = controller.long_path_programs
        assert len(sealed) == 8
        assert len(long) == 12
        assert all(row.probe_kind == "heldout_legal_edge" for row in sealed)
        assert all(row.probe_kind == "long_path" for row in long)
        predictions = {
            row.program_sha256: active.realization.predict_answers(
                row.program,
                initial_answers=controller.initial_answers,
            )
            for row in sealed + long
        }
        assert all(
            predictions[row.program_sha256] == row.expected_answers
            for row in sealed + long
        )
        assert sum(len(row.expected_answers) for row in sealed + long) == 40

        relation_rows: dict[str, list[tuple[str, ...]]] = {}
        expectations: dict[str, str] = {}
        for row in long:
            if row.relation_group is None:
                assert row.relation_expectation == "none"
                continue
            relation_rows.setdefault(row.relation_group, []).append(
                predictions[row.program_sha256]
            )
            expectations[row.relation_group] = row.relation_expectation
        assert {name: len(rows) for name, rows in relation_rows.items()} == {
            "alternate_path_equal": 2,
            "copy_order_noncommutes": 2,
            "update_order_commutes": 2,
        }
        for name, rows in relation_rows.items():
            if expectations[name] == "equal":
                assert rows[0] == rows[1]
            else:
                assert expectations[name] == "not_equal"
                assert rows[0] != rows[1]


def test_full_support_partial_maps_replay_all_44_legal_edges(full_controllers) -> None:
    for controller in full_controllers:
        realization = partial_module.fit_passive_partial_operators(
            controller.learner_input
        ).realization
        assert len(controller.learner_input.passive_edge_observations) == 44
        for observation in controller.learner_input.passive_edge_observations:
            source = realization.answers_to_coordinates(observation.source_answers)
            target = realization.apply_event(
                source, observation.request.event_token
            )
            assert realization.coordinates_to_answers(target) == observation.target_answers


def test_every_learner_input_admits_all_eight_controller_arm_hypotheses(
    omission_controllers,
) -> None:
    expected = {(cell, block) for cell in CELLS for block in range(2)}
    for controller in omission_controllers:
        certificate = (
            partial_module.enumerate_publicly_compatible_omission_hypotheses(
                controller.learner_input
            )
        )
        assert certificate.learner_input_sha256 == controller.learner_input.input_sha256
        assert certificate.compatible_hypothesis_count == 8
        assert len(certificate.witnesses) == 8
        assert {
            (row.inferred_omitted_cell, row.compatible_relabel_block)
            for row in certificate.witnesses
        } == expected
        assert certificate.exhaustive_finite_bijection_enumeration
        assert not certificate.active_answers_used
        assert not certificate.sealed_answers_used
        assert not certificate.controller_nonce_used
        assert not certificate.actual_omitted_identifier_used
        assert (
            partial_module.publicly_compatible_omission_hypothesis_count(
                controller.learner_input
            )
            == 8
        )


def test_undefined_requests_cannot_be_reintroduced_as_zero_or_dead(
    omission_controllers,
) -> None:
    learner_input = omission_controllers[0].learner_input
    with pytest.raises(ValueError, match="illegal_words_encoded_as_zero_or_dead"):
        replace(learner_input, illegal_words_encoded_as_zero_or_dead=True)
    with pytest.raises(ValueError, match="undefined_words_are_absent_constraints"):
        replace(learner_input, undefined_words_are_absent_constraints=False)

    undefined = learner_input.undefined_edge_requests[0]
    original = learner_input.passive_edge_observations[0]
    fabricated = partial_module._make_observation(
        undefined,
        original.source_answers,
        original.target_answers,
    )
    rows = (fabricated,) + learner_input.passive_edge_observations[1:]
    with pytest.raises(ValueError, match="absent from the defined mask"):
        partial_module._make_learner_input(
            event_tokens=learner_input.event_tokens,
            query_tokens=learner_input.query_tokens,
            answer_tokens=learner_input.answer_tokens,
            passive_state_observations=learner_input.passive_state_observations,
            passive_edge_observations=rows,
            candidate_edge_requests=learner_input.candidate_edge_requests,
            defined_edge_requests=learner_input.defined_edge_requests,
            undefined_edge_requests=learner_input.undefined_edge_requests,
            budgets=learner_input.budgets,
        )


def test_event_query_and_answer_roles_cannot_be_confused(omission_controllers) -> None:
    learner_input = omission_controllers[0].learner_input
    original = learner_input.defined_edge_requests[0]
    bad_request = partial_module._make_request(
        original.source_word, learner_input.query_tokens[0]
    )
    defined = (bad_request,) + learner_input.defined_edge_requests[1:]
    with pytest.raises(ValueError, match="undeclared event token"):
        partial_module._make_learner_input(
            event_tokens=learner_input.event_tokens,
            query_tokens=learner_input.query_tokens,
            answer_tokens=learner_input.answer_tokens,
            passive_state_observations=learner_input.passive_state_observations,
            passive_edge_observations=learner_input.passive_edge_observations,
            candidate_edge_requests=learner_input.candidate_edge_requests,
            defined_edge_requests=defined,
            undefined_edge_requests=learner_input.undefined_edge_requests,
            budgets=learner_input.budgets,
        )

    original_observation = learner_input.passive_edge_observations[0]
    bad_observation = partial_module._make_observation(
        original_observation.request,
        original_observation.source_answers,
        (learner_input.query_tokens[0], original_observation.target_answers[1]),
    )
    observations = (bad_observation,) + learner_input.passive_edge_observations[1:]
    with pytest.raises(ValueError, match="undeclared answer token|contradictory"):
        partial_module._make_learner_input(
            event_tokens=learner_input.event_tokens,
            query_tokens=learner_input.query_tokens,
            answer_tokens=learner_input.answer_tokens,
            passive_state_observations=learner_input.passive_state_observations,
            passive_edge_observations=observations,
            candidate_edge_requests=learner_input.candidate_edge_requests,
            defined_edge_requests=learner_input.defined_edge_requests,
            undefined_edge_requests=learner_input.undefined_edge_requests,
            budgets=learner_input.budgets,
        )


def test_partial_artifacts_reject_a_forged_total_operator(omission_fits) -> None:
    active = omission_fits[0][-1]
    zero = tuple(
        tuple(partial_module.Rational(0) for _ in range(5)) for _ in range(5)
    )
    with pytest.raises(ValueError, match="total operator|Contract B"):
        replace(active.operator_certificates[0], total_operator=zero)
    with pytest.raises(ValueError, match="total operator"):
        replace(active.realization, total_operator=zero)
    with pytest.raises(ValueError, match="total operator"):
        replace(active, total_operator=zero)


def test_one_response_checkpoint_forgery_is_reconstructed_before_active_fit(
    omission_fits,
) -> None:
    controller, passive, first, checkpoint, remaining, _ = omission_fits[0]
    alternatives = tuple(product(controller.learner_input.answer_tokens, repeat=2))
    forged_a, forged_b = next(
        (left, right)
        for left in alternatives
        for right in alternatives
        if left != right
        and (left, right)
        != (checkpoint.compatible_outcome_a, checkpoint.compatible_outcome_b)
    )
    forged_payload = checkpoint._payload(include_checkpoint_sha=False)
    forged_payload["compatible_outcome_a"] = forged_a
    forged_payload["compatible_outcome_b"] = forged_b
    forged = replace(
        checkpoint,
        compatible_outcome_a=forged_a,
        compatible_outcome_b=forged_b,
        checkpoint_sha256=partial_module._sha256(forged_payload),
    )
    assert forged != checkpoint
    with pytest.raises(ValueError, match="staged pure analysis"):
        partial_module.fit_active_partial_operators(
            controller.learner_input,
            passive,
            first,
            forged,
            remaining,
        )
    with pytest.raises(ValueError, match="differing_restricted_map_witness"):
        replace(checkpoint, differing_restricted_map_witness=False)


def test_one_response_ambiguity_witness_must_target_an_unobserved_program(
    omission_controllers,
) -> None:
    controller = omission_controllers[0]
    learner_input = controller.learner_input
    witness_program = learner_input.candidate_edge_requests[1].program
    replacement_state = partial_module.OpaqueStateObservation(
        witness_program,
        learner_input.passive_state_observations[0].answers,
    )
    with pytest.raises(ValueError, match="source representatives|already known|unobserved"):
        partial_module._make_learner_input(
            event_tokens=learner_input.event_tokens,
            query_tokens=learner_input.query_tokens,
            answer_tokens=learner_input.answer_tokens,
            passive_state_observations=(replacement_state,)
            + learner_input.passive_state_observations[1:],
            passive_edge_observations=learner_input.passive_edge_observations,
            candidate_edge_requests=learner_input.candidate_edge_requests,
            defined_edge_requests=learner_input.defined_edge_requests,
            undefined_edge_requests=learner_input.undefined_edge_requests,
            budgets=learner_input.budgets,
        )


def test_repeated_word_diagnostics_must_reconcile_globally(
    omission_controllers,
) -> None:
    learner_input = omission_controllers[0].learner_input
    original = learner_input.passive_edge_observations[0]
    alternative = next(
        token
        for token in learner_input.answer_tokens
        if token != original.source_answers[0]
    )
    forged = partial_module._make_observation(
        original.request,
        (alternative, original.source_answers[1]),
        original.target_answers,
    )
    observations = (forged,) + learner_input.passive_edge_observations[1:]
    with pytest.raises(ValueError, match="contradictory diagnostics"):
        partial_module._make_learner_input(
            event_tokens=learner_input.event_tokens,
            query_tokens=learner_input.query_tokens,
            answer_tokens=learner_input.answer_tokens,
            passive_state_observations=learner_input.passive_state_observations,
            passive_edge_observations=observations,
            candidate_edge_requests=learner_input.candidate_edge_requests,
            defined_edge_requests=learner_input.defined_edge_requests,
            undefined_edge_requests=learner_input.undefined_edge_requests,
            budgets=learner_input.budgets,
        )


@pytest.mark.parametrize(
    ("field_name", "too_small"),
    (
        ("max_word_length", 4),
        ("max_event_tokens", 9),
        ("max_domain_edges", 89),
        ("max_passive_edges", 20),
        ("max_active_responses", 14),
        ("max_categorical_labels", 29),
        ("max_sealed_edges", 7),
        ("max_long_probes", 11),
        ("max_basis_dimension", 4),
        ("max_exact_rank_evaluations", 511),
        ("max_rational_bit_length", 63),
    ),
)
def test_every_structural_budget_underflow_fails_before_controller_work(
    monkeypatch, field_name: str, too_small: int
) -> None:
    budgets = replace(
        partial_module.PartialOperatorBudgets(), **{field_name: too_small}
    )

    def materialization_started(*_args, **_kwargs):
        raise AssertionError("controller materialized opaque tokens before budget preflight")

    monkeypatch.setattr(partial_module, "_opaque_token", materialization_started)
    with pytest.raises(partial_module.OpaquePartialOperatorLimitError):
        partial_module.build_omission_controller_environment(
            (0, 0),
            0,
            controller_nonce=CONTROLLER_NONCES[2],
            budgets=budgets,
        )


def test_hashes_reject_local_tampering(omission_fits) -> None:
    controller, passive, first, checkpoint, _, active = omission_fits[0]
    request = controller.learner_input.defined_edge_requests[0]
    other_event = next(
        token
        for token in controller.learner_input.event_tokens
        if token != request.event_token
    )
    with pytest.raises(ValueError, match="request_sha256"):
        replace(
            request,
            event_token=other_event,
            program=request.source_word + (other_event,),
        )
    with pytest.raises(ValueError, match="response_sha256"):
        replace(
            first,
            target_answers=tuple(reversed(first.target_answers)),
        )
    with pytest.raises(ValueError, match="passive discovery digest"):
        replace(passive, observed_event_ranks=tuple(reversed(passive.observed_event_ranks)))
    with pytest.raises(ValueError, match="checkpoint digest"):
        replace(
            checkpoint,
            compatible_outcome_a=checkpoint.compatible_outcome_b,
            compatible_outcome_b=checkpoint.compatible_outcome_a,
        )
    with pytest.raises(ValueError, match="active discovery digest"):
        replace(
            active,
            response_sha256s=(active.response_sha256s[0],)
            + tuple(reversed(active.response_sha256s[1:])),
        )


def test_controller_and_fit_are_exactly_deterministic(
    full_controllers, omission_controllers
) -> None:
    rebuilt_full = tuple(_full_controller(block) for block in range(2))
    rebuilt_omissions = tuple(
        _omission_controller(cell, block)
        for block in range(2)
        for cell in CELLS
    )
    assert rebuilt_full == full_controllers
    assert rebuilt_omissions == omission_controllers
    all_controllers = full_controllers + omission_controllers
    assert len({row.controller_sha256 for row in all_controllers}) == 10
    assert len({row.learner_input.input_sha256 for row in all_controllers}) == 10
    token_alphabets = [
        set(row.learner_input.event_tokens)
        | set(row.learner_input.query_tokens)
        | set(row.learner_input.answer_tokens)
        for row in all_controllers
    ]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(token_alphabets)
        for right in token_alphabets[index + 1 :]
    )
    assert all(
        partial_module.fit_passive_partial_operators(rebuilt.learner_input)
        == partial_module.fit_passive_partial_operators(original.learner_input)
        for rebuilt, original in zip(
            rebuilt_full + rebuilt_omissions,
            all_controllers,
            strict=True,
        )
    )


def test_nonce_inventory_is_explicit_strict_and_unique() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        partial_module.build_omission_controller_environment(
            (0, 0), 0, controller_nonce="0" * 32
        )
    assert len(CONTROLLER_NONCES) == len(set(CONTROLLER_NONCES)) == 10


def test_top_level_report_pins_the_exact_ten_environment_inventory(report) -> None:
    assert report.environment_count == 10
    assert report.full_support_control_count == 2
    assert report.rotated_omission_count == 8
    assert len(report.environments) == 10
    assert report.passive_edge_total == 256
    assert report.active_response_total == 120
    assert report.active_returned_categorical_token_total == 240
    assert report.sealed_edge_total == 64
    assert report.long_path_total == 120
    assert report.sealed_program_total == 184
    assert report.similarity_pair_count == 5
    assert report.passive_rank_sequence == (5, 5) + (4,) * 8
    assert report.postactive_rank_sequence == (5,) * 10
    assert report.aggregate_total_extension_nullity_per_environment == (80,) * 10
    assert report.all_environment_behavior_passed

    expected_schedule = (
        (partial_module.EnvironmentKind.FULL_SUPPORT_CONTROL, None, 0),
        (partial_module.EnvironmentKind.FULL_SUPPORT_CONTROL, None, 1),
    ) + tuple(
        (partial_module.EnvironmentKind.ROTATED_OMISSION, cell, block)
        for cell in CELLS
        for block in range(2)
    )
    assert tuple(
        (
            row.controller.kind,
            row.controller.pseudoheldout_cell,
            row.controller.relabel_block,
        )
        for row in report.environments
    ) == expected_schedule


def test_report_states_the_narrow_supplied_structure_scope_honestly(report) -> None:
    assert report.scope is (
        partial_module.PartialOperatorScope.GUARDED_ABSENCE_AWARE_PARTIAL_OPERATORS
    )
    assert report.status is (
        partial_module.PartialOperatorStatus.SYNTHETIC_PROTOCOL_IMPLEMENTATION_REHEARSAL
    )
    assert report.supplied_full_product_state_grammar
    assert report.supplied_semantic_active_excitation_basis
    assert not report.active_basis_learned_or_selected_by_estimator
    assert report.exact_partial_legal_domain_operator_claim
    assert not report.total_wfa_operator_claim
    assert not report.assumption_free_representation_discovery_claim
    assert report.learner_boundaries_use_only_opaque_inputs
    assert not report.trusted_controller_and_learner_are_process_isolated
    assert report.contextual_nested_certificates_reconstructed_by_report
    assert not any(type(value) is float for value in _walk_values(report))


def test_every_authoritative_environment_reconstructs_and_rejects_all_46_undefined_pairs(
    report,
) -> None:
    for index, environment in enumerate(report.environments):
        assert environment.authoritative_environment_reconstructed_pure_fit
        assert environment.nested_certificates_are_contextual_content_links
        assert not environment.nested_certificates_independently_authenticated
        assert not environment.learner_received_controller_material
        assert environment.model_behavior_passed
        assert environment.legal_edge_partition_total == 44
        undefined = environment.undefined_domain_rejection
        assert undefined.undefined_edge_count == 46
        assert undefined.rejected_edge_count == 46
        assert len(undefined.rows) == 46
        assert len({row.request_sha256 for row in undefined.rows}) == 46
        assert all(row.rejected_as_outside_legal_domain for row in undefined.rows)
        assert undefined.undefined_words_treated_as_absent_constraints
        assert not undefined.zero_or_dead_state_filling_used

        evaluation = environment.sealed_evaluation
        assert evaluation.long_path_program_count == 12
        assert evaluation.satisfied_path_relation_count == 3
        assert evaluation.all_exact
        if index < 2:
            assert (
                environment.passive_edge_count,
                environment.active_response_count,
                environment.active_returned_categorical_token_count,
                environment.sealed_edge_count,
                evaluation.total_program_count,
                evaluation.categorical_label_prediction_count,
            ) == (44, 0, 0, 0, 12, 24)
            assert environment.compatible_omission_certificate is None
            assert environment.one_response_checkpoint is None
            assert environment.active is None
        else:
            assert (
                environment.passive_edge_count,
                environment.active_response_count,
                environment.active_returned_categorical_token_count,
                environment.sealed_edge_count,
                evaluation.total_program_count,
                evaluation.categorical_label_prediction_count,
            ) == (21, 15, 30, 8, 20, 40)
            assert environment.compatible_omission_certificate is not None
            assert (
                environment.compatible_omission_certificate.compatible_hypothesis_count
                == 8
            )
            assert environment.one_response_checkpoint is not None
            assert environment.active is not None


def test_five_partial_similarity_certificates_use_one_fit5_test4_gauge(report) -> None:
    assert len(report.similarities) == 5
    assert tuple(row.paired_omitted_cell for row in report.similarities) == (
        None,
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    )
    for similarity in report.similarities:
        assert similarity.gauge_fit_state_count == 5
        assert similarity.disjoint_test_state_count == 4
        assert len(similarity.gauge_fit_state_indices) == 5
        assert len(similarity.disjoint_test_state_indices) == 4
        assert set(similarity.gauge_fit_state_indices).isdisjoint(
            similarity.disjoint_test_state_indices
        )
        assert set(
            similarity.gauge_fit_state_indices
            + similarity.disjoint_test_state_indices
        ) == set(range(9))
        fit_pairs = tuple(
            similarity.canonical_state_coordinate_pairs[index]
            for index in similarity.gauge_fit_state_indices
        )
        test_pairs = tuple(
            similarity.canonical_state_coordinate_pairs[index]
            for index in similarity.disjoint_test_state_indices
        )
        assert partial_module._rank_profile(
            partial_module._matrix(row[0] for row in fit_pairs)
        )[0] == 5
        derived_gauge = partial_module._matmul(
            partial_module._inverse(
                partial_module._matrix(row[0] for row in fit_pairs)
            ),
            partial_module._matrix(row[1] for row in fit_pairs),
        )
        assert derived_gauge == similarity.global_change_of_basis
        assert similarity.gauge_fit_rows_sha256 == partial_module._sha256(fit_pairs)
        assert similarity.disjoint_test_rows_sha256 == partial_module._sha256(
            test_pairs
        )
        assert all(
            partial_module._row_times_matrix(left, derived_gauge) == right
            for left, right in test_pairs
        )
        assert similarity.state_row_count == 9
        assert similarity.restricted_legal_edge_count == 44
        assert similarity.event_map_count == 10
        assert similarity.one_global_gauge_used
        assert similarity.every_state_row_transforms
        assert similarity.every_restricted_graph_edge_transforms
        assert not similarity.arbitrary_off_domain_fill_compared
        assert not similarity.total_operator_compared
        assert similarity.controller_supplied_state_event_correspondence
        assert not similarity.correspondence_learned_by_opaque_estimator
        assert similarity.correspondence_used_only_postfit


def test_similarity_readouts_are_exact_linear_postfit_witnesses(report) -> None:
    for similarity in report.similarities:
        assert similarity.every_categorical_readout_corresponds
        assert len(similarity.categorical_readout_correspondence) == 9
        assert len(similarity.canonical_diagnostic_feature_pairs) == 9
        assert all(
            partial_module._row_times_matrix(
                right_feature,
                similarity.right_to_left_diagnostic_feature_alignment,
            )
            == left_feature
            for left_feature, right_feature in similarity.canonical_diagnostic_feature_pairs
        )
        assert similarity.right_readout_aligned_matrix == partial_module._matmul(
            similarity.right_readout_matrix,
            similarity.right_to_left_diagnostic_feature_alignment,
        )
        assert similarity.left_readout_matrix == partial_module._matmul(
            similarity.global_change_of_basis,
            similarity.right_readout_aligned_matrix,
        )
        assert similarity.exact_linear_readout_equation_holds


def test_report_is_byte_for_byte_deterministic(report) -> None:
    repeated = partial_module.run_toy_partial_operator_experiment(
        controller_nonces=CONTROLLER_NONCES
    )
    assert repeated == report
    assert repeated.report_sha256 == report.report_sha256


def test_active_request_inventory_is_committed_independently_of_outcomes(
    omission_fits,
) -> None:
    controller, passive, *_ = omission_fits[0]
    learner_input = controller.learner_input
    commitment = passive.active_commitment
    assert commitment.selected_request_sha256s == tuple(
        row.request_sha256 for row in learner_input.candidate_edge_requests[:15]
    )
    assert commitment.sealed_request_sha256s == tuple(
        row.request_sha256 for row in learner_input.candidate_edge_requests[15:]
    )
    alternative_outcomes = tuple(
        partial_module._make_response(
            request,
            (
                learner_input.answer_tokens[index % 3],
                learner_input.answer_tokens[(index + 1) % 3],
            ),
            index + 1,
        )
        for index, request in enumerate(
            learner_input.candidate_edge_requests[:15]
        )
    )
    assert tuple(
        row.request.request_sha256 for row in alternative_outcomes
    ) == commitment.selected_request_sha256s
    assert (
        partial_module.fit_passive_partial_operators(learner_input).active_commitment
        == commitment
    )


def test_all_total_operator_slots_remain_none_in_the_authoritative_report(report) -> None:
    total_operator_slot_count = 0

    def walk(value: object) -> None:
        nonlocal total_operator_slot_count
        if is_dataclass(value):
            for item in fields(value):
                child = getattr(value, item.name)
                if item.name == "total_operator":
                    total_operator_slot_count += 1
                    assert child is None
                walk(child)
        elif isinstance(value, tuple):
            for child in value:
                walk(child)

    walk(report)
    assert total_operator_slot_count > 100


def test_environment_parent_rejects_rehashed_nested_checkpoint_and_probe_forgeries(
    report,
) -> None:
    environment = report.environments[2]
    checkpoint = environment.one_response_checkpoint
    assert checkpoint is not None
    alternatives = tuple(
        product(environment.controller.learner_input.answer_tokens, repeat=2)
    )
    forged_a, forged_b = next(
        (left, right)
        for left in alternatives
        for right in alternatives
        if left != right
        and (left, right)
        != (checkpoint.compatible_outcome_a, checkpoint.compatible_outcome_b)
    )
    checkpoint_payload = checkpoint._payload(include_checkpoint_sha=False)
    checkpoint_payload["compatible_outcome_a"] = forged_a
    checkpoint_payload["compatible_outcome_b"] = forged_b
    forged_checkpoint = replace(
        checkpoint,
        compatible_outcome_a=forged_a,
        compatible_outcome_b=forged_b,
        checkpoint_sha256=partial_module._sha256(checkpoint_payload),
    )
    with pytest.raises(ValueError, match="checkpoint fails reconstruction"):
        replace(environment, one_response_checkpoint=forged_checkpoint)

    evaluation = environment.sealed_evaluation
    original_prediction = evaluation.predictions[0]
    fake_answers = next(
        row
        for row in alternatives
        if row != original_prediction.expected_answers
    )
    forged_prediction = replace(
        original_prediction,
        predicted_answers=fake_answers,
        expected_answers=fake_answers,
        exact=True,
    )
    forged_predictions = (forged_prediction,) + evaluation.predictions[1:]
    evaluation_payload = evaluation._payload(include_evaluation_sha=False)
    evaluation_payload["predictions"] = [
        row.__dict__ for row in forged_predictions
    ]
    forged_evaluation = replace(
        evaluation,
        predictions=forged_predictions,
        evaluation_sha256=partial_module._sha256(evaluation_payload),
    )
    with pytest.raises(ValueError, match="sealed evaluation fails replay"):
        replace(environment, sealed_evaluation=forged_evaluation)


def test_report_parent_rejects_rehashed_similarity_correspondence_forgery(report) -> None:
    similarity = report.similarities[0]
    correspondence = list(similarity.event_token_correspondence)
    correspondence[0] = (correspondence[0][0], correspondence[1][1])
    correspondence[1] = (correspondence[1][0], similarity.event_token_correspondence[0][1])
    forged_rows = tuple(correspondence)
    similarity_payload = similarity._payload(include_certificate_sha=False)
    similarity_payload["event_token_correspondence"] = forged_rows
    forged_similarity = replace(
        similarity,
        event_token_correspondence=forged_rows,
        certificate_sha256=partial_module._sha256(similarity_payload),
    )
    with pytest.raises(ValueError, match="five reconstructed block similarities"):
        replace(
            report,
            similarities=(forged_similarity,) + report.similarities[1:],
        )


def test_report_schedule_and_relabel_forgeries_fail_before_environment_replay(
    monkeypatch, report
) -> None:
    def replay_started(*_args, **_kwargs):
        raise AssertionError("expensive environment replay ran before schedule rejection")

    monkeypatch.setattr(partial_module, "run_toy_partial_environment", replay_started)
    with pytest.raises(ValueError, match="environment schedule"):
        replace(
            report,
            environments=(report.environments[1], report.environments[0])
            + report.environments[2:],
        )

    original = report.environments[2].controller
    controller_payload = original._payload(include_controller_sha=False)
    controller_payload["relabel_block"] = 1 - original.relabel_block
    forged_controller = replace(
        original,
        relabel_block=1 - original.relabel_block,
        controller_sha256=partial_module._sha256(controller_payload),
    )
    # The trusted environment can be rebuilt coherently around the forged
    # metadata; only the report's frozen controller schedule may authenticate it.
    monkeypatch.undo()
    forged_environment = partial_module.run_toy_partial_environment(forged_controller)
    monkeypatch.setattr(partial_module, "run_toy_partial_environment", replay_started)
    forged_environments = (
        report.environments[:2]
        + (forged_environment,)
        + report.environments[3:]
    )
    with pytest.raises(ValueError, match="environment schedule"):
        replace(report, environments=forged_environments)


def test_invalid_report_nonce_inventory_fails_before_controller_construction(
    monkeypatch,
) -> None:
    def construction_started(*_args, **_kwargs):
        raise AssertionError("controller construction ran before nonce validation")

    monkeypatch.setattr(
        partial_module,
        "build_full_support_controller_environment",
        construction_started,
    )
    with pytest.raises(ValueError, match="distinct"):
        partial_module.run_toy_partial_operator_experiment(
            controller_nonces=(CONTROLLER_NONCES[0],) * 10
        )
