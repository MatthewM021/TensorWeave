from __future__ import annotations

import ast
from dataclasses import fields, replace
import inspect
import os

import pytest

import tnlm_v3.opaque_active_discovery as active_module
import tnlm_v3.opaque_partial_operators as partial_module


PRIMARY_CELL = (0, 0)
PRIMARY_BLOCK = 0
DEVELOPMENT_CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
DEVELOPMENT_OMISSION_NONCE_FIXTURES = (
    "323b4597f9e3dfb9e7f954aab5873b8730321dd8476a2992f81e93cc7a44b046",
    "45d4ff9e973a3c9fda9a41ae7ab8c827f9e7299a2ece6f402ab32dc0fd598718",
    "a3d1591df9b8fe82666f78e885f578cfd5677df28dfa6bc2daa5cfa468d87203",
    "0ec845c35cfe807b28f8d012391d9e811211a780e3fd22510ad33a6059dcf5fc",
    "422d894dade5755a90355dea9dec66be8a8943509365c79727046eb895042bb4",
    "be187a3a14f4d575944e93a407f0e1c53142a9db4c27abc0dcb6bbb3d7712df3",
    "219803a3b13f67dbb90f23749b785d71836918b7d5d0cee78d74d85ff0319490",
    "8e9f16318b215a2e117e34bc75cbc649b602dac9d8ce5ffcdace6950c8971ce2",
)
DEVELOPMENT_CONTROLLER_NONCE_FIXTURE = DEVELOPMENT_OMISSION_NONCE_FIXTURES[0]


@pytest.fixture(scope="module")
def primary_controller():
    return partial_module.build_omission_controller_environment(
        PRIMARY_CELL,
        PRIMARY_BLOCK,
        controller_nonce=DEVELOPMENT_CONTROLLER_NONCE_FIXTURE,
    )


@pytest.fixture(scope="module")
def primary_input(primary_controller):
    return active_module.make_opaque_active_input(primary_controller.learner_input)


@pytest.fixture(scope="module")
def primary_run(primary_controller, primary_input):
    token_to_action = {
        token: action for action, token in primary_controller.canonical_event_tokens
    }
    state_to_answers = dict(primary_controller.canonical_state_answers)
    committed_choice_hashes: list[str] = []

    def trusted_development_provider(choice):
        # This is deliberately outside the learner boundary.  It records the
        # already-created commitment before consulting the synthetic executor.
        committed_choice_hashes.append(choice.choice_sha256)
        semantic_state = (-1, -1)
        for token in choice.chosen_request.program:
            semantic_state = partial_module._semantic_step(
                semantic_state, token_to_action[token]
            )
            assert semantic_state is not None
        return state_to_answers[semantic_state]

    result = active_module.run_opaque_active_discovery(
        primary_input, trusted_development_provider
    )
    return result, tuple(committed_choice_hashes)


def _score_payload(**overrides: object) -> dict[str, object]:
    answers_a = ("0" * 32, "0" * 32)
    answers_b = ("1" * 32, "1" * 32)
    payload: dict[str, object] = {
        "schema": "tnlm-v3-opaque-active-candidate-score-v1",
        "frontier_target_unlock": 0,
        "request_sha256": "2" * 64,
        "event_token": "3" * 32,
        "outcome_bucket_rows": (
            (answers_a, 1, 2, 4),
            (answers_b, 1, 3, 6),
        ),
        "observed_source_rank_before": 1,
        "observed_source_rank_after": 2,
        "compatible_outcome_count": 2,
        # Post-frontier map outcomes all share the sole completed source
        # assignment; assignment buckets therefore overlap while event/global
        # version buckets partition.
        "source_assignment_count_before": 1,
        "selected_event_version_count_before": 5,
        "global_version_mass_before": 10,
        "worst_case_surviving_event_version_count": 3,
        "worst_posterior_global_version_product": 6,
        "exact_restricted_nullity_drop": 5,
    }
    payload.update(overrides)
    return payload


def _make_score(**overrides: object):
    payload = _score_payload(**overrides)
    return active_module.OpaqueActiveCandidateScore(
        **payload,
        score_sha256=active_module._sha256(payload),
    )


def _trusted_development_answer(controller, choice) -> tuple[str, ...]:
    token_to_action = {
        token: action for action, token in controller.canonical_event_tokens
    }
    state_to_answers = dict(controller.canonical_state_answers)
    semantic_state = (-1, -1)
    for token in choice.chosen_request.program:
        semantic_state = partial_module._semantic_step(
            semantic_state, token_to_action[token]
        )
        assert semantic_state is not None
    return state_to_answers[semantic_state]


def test_active_input_surface_contains_only_the_declared_opaque_supervision() -> None:
    assert {field.name for field in fields(active_module.OpaqueActiveLearnerInput)} == {
        "event_tokens",
        "query_tokens",
        "answer_tokens",
        "passive_state_observations",
        "passive_edge_observations",
        "canonical_candidate_requests",
        "canonical_defined_requests",
        "canonical_undefined_requests",
        "budgets",
        "passive_table_sha256",
        "domain_mask_sha256",
        "canonical_candidate_set_sha256",
        "candidate_order_discarded",
        "domain_mask_is_visible_supervision",
        "full_product_diagnostic_gauge_is_visible_supervision",
        "mask_source_representatives_bijectively_cover_full_product",
        "source_bijection_counted_as_supervision",
        "input_sha256",
        "schema",
    }


def test_input_erases_t1_order_and_canonicalizes_every_visible_collection(
    primary_controller,
) -> None:
    source = primary_controller.learner_input
    forward = active_module.make_opaque_active_input(source)
    reversed_candidates = active_module.make_opaque_active_input(
        source,
        candidate_requests=tuple(reversed(source.candidate_edge_requests)),
    )
    assert forward == reversed_candidates
    assert tuple(
        sorted(
            forward.passive_state_observations,
            key=lambda row: active_module._sha256(active_module._state_payload(row)),
        )
    ) == forward.passive_state_observations
    assert tuple(
        sorted(
            forward.passive_edge_observations,
            key=lambda row: row.observation_sha256,
        )
    ) == forward.passive_edge_observations
    for rows in (
        forward.canonical_candidate_requests,
        forward.canonical_defined_requests,
        forward.canonical_undefined_requests,
    ):
        assert tuple(sorted(rows, key=lambda row: row.request_sha256)) == rows
    assert not hasattr(forward, "candidate_edge_requests")
    assert source.input_sha256 not in repr(forward)
    assert primary_controller.trusted_controller_nonce not in repr(forward)


def test_primitive_row_constructor_is_equivalent_and_has_no_upstream_surface(
    primary_controller,
) -> None:
    source = primary_controller.learner_input
    primitive = active_module.make_opaque_active_input_from_rows(
        event_tokens=source.event_tokens,
        query_tokens=source.query_tokens,
        answer_tokens=source.answer_tokens,
        passive_state_observations=tuple(
            reversed(source.passive_state_observations)
        ),
        passive_edge_observations=tuple(
            reversed(source.passive_edge_observations)
        ),
        candidate_requests=tuple(reversed(source.candidate_edge_requests)),
        defined_requests=tuple(reversed(source.defined_edge_requests)),
        undefined_requests=tuple(reversed(source.undefined_edge_requests)),
    )
    adapted = active_module.make_opaque_active_input(source)
    assert primitive == adapted
    assert source.input_sha256 not in repr(primitive)
    assert primary_controller.trusted_controller_nonce not in repr(primitive)
    assert "source" not in inspect.signature(
        active_module.make_opaque_active_input_from_rows
    ).parameters


def test_reordered_material_cannot_be_reintroduced_after_canonicalization(
    primary_input,
) -> None:
    with pytest.raises(ValueError, match="canonical opaque-content order"):
        replace(
            primary_input,
            passive_state_observations=tuple(
                reversed(primary_input.passive_state_observations)
            ),
        )
    with pytest.raises(ValueError, match="canonical observation-digest order"):
        replace(
            primary_input,
            passive_edge_observations=tuple(
                reversed(primary_input.passive_edge_observations)
            ),
        )
    with pytest.raises(ValueError, match="canonical SHA order"):
        replace(
            primary_input,
            canonical_candidate_requests=tuple(
                reversed(primary_input.canonical_candidate_requests)
            ),
        )


def test_omission_input_has_the_exact_all_action_mask_and_partition(primary_input) -> None:
    assert (
        len(primary_input.event_tokens),
        len(primary_input.query_tokens),
        len(primary_input.answer_tokens),
    ) == (10, 2, 3)
    assert set(primary_input.event_tokens).isdisjoint(primary_input.query_tokens)
    assert len(primary_input.passive_state_observations) == 6
    assert len(primary_input.passive_edge_observations) == 21
    assert len(primary_input.canonical_candidate_requests) == 23
    assert len(primary_input.canonical_defined_requests) == 44
    assert len(primary_input.canonical_undefined_requests) == 46

    defined = {
        (row.source_word, row.event_token)
        for row in primary_input.canonical_defined_requests
    }
    undefined = {
        (row.source_word, row.event_token)
        for row in primary_input.canonical_undefined_requests
    }
    sources = {word for word, _ in defined | undefined}
    assert len(sources) == 9
    assert not defined & undefined
    assert defined | undefined == {
        (word, event) for word in sources for event in primary_input.event_tokens
    }
    assert {
        row.request.request_sha256
        for row in primary_input.passive_edge_observations
    } | {
        row.request_sha256 for row in primary_input.canonical_candidate_requests
    } == {row.request_sha256 for row in primary_input.canonical_defined_requests}
    for row in (
        primary_input.canonical_defined_requests
        + primary_input.canonical_undefined_requests
    ):
        assert row.program == row.source_word + (row.event_token,)
        assert set(row.program) <= set(primary_input.event_tokens)


def test_undefined_mask_rows_have_no_answer_or_totalized_surface(primary_input) -> None:
    request_fields = {
        field.name for field in fields(partial_module.OpaqueEdgeRequest)
    }
    assert not request_fields & {
        "answer",
        "answers",
        "target_answers",
        "zero",
        "dead",
        "no_op",
    }
    assert all(
        not hasattr(row, "target_answers")
        for row in primary_input.canonical_undefined_requests
    )
    assert "total_operator" in {
        field.name for field in fields(active_module.AutonomousRestrictedOperator)
    }
    assert "total_operator" in {
        field.name for field in fields(active_module.AutonomousPartialModel)
    }
    assert "total_wfa_claimed" in {
        field.name for field in fields(active_module.AutonomousPartialOperatorResult)
    }


@pytest.mark.parametrize(
    ("budget_overrides", "message"),
    (
        ({"max_active_calls": 13}, "active-call budget"),
        ({"max_returned_categorical_tokens": 27}, "returned-label budget"),
        ({"max_candidate_score_rows": 239}, "candidate-score budget"),
        ({"max_exact_rank_evaluations": 749}, "exact-rank budget"),
    ),
)
def test_structural_budget_underflow_fails_before_state_or_version_work(
    monkeypatch,
    primary_controller,
    budget_overrides,
    message,
) -> None:
    budgets = active_module.OpaqueActiveDiscoveryBudgets(**budget_overrides)
    learner_input = active_module.make_opaque_active_input(
        primary_controller.learner_input,
        budgets=budgets,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("expensive state/version work ran before budget rejection")

    monkeypatch.setattr(active_module, "_make_state", forbidden)
    monkeypatch.setattr(active_module, "_conditional_global_blocks", forbidden)
    with pytest.raises(active_module.OpaqueActiveDiscoveryLimitError, match=message):
        active_module.initialize_opaque_active_discovery(learner_input)


def test_candidate_score_binds_exact_assignment_event_and_global_partitions() -> None:
    score = _make_score()
    assert score.source_assignment_count_before == 1
    assert score.selected_event_version_count_before == 5
    assert score.global_version_mass_before == 10
    assert score.worst_case_surviving_event_version_count == 3
    assert score.worst_posterior_global_version_product == 6
    # Once source assignment is complete the same sole assignment can appear
    # in every categorical map-outcome branch.  It is not a disjoint mass.
    assert sum(row[1] for row in score.outcome_bucket_rows) == 2
    frontier = _make_score(
        frontier_target_unlock=1,
        source_assignment_count_before=2,
    )
    assert sum(row[1] for row in frontier.outcome_bucket_rows) == 2
    with pytest.raises(ValueError, match="source assignments"):
        _make_score(
            frontier_target_unlock=1,
            source_assignment_count_before=3,
        )
    with pytest.raises(ValueError, match="selected-event versions"):
        _make_score(selected_event_version_count_before=6)
    with pytest.raises(ValueError, match="global version mass"):
        _make_score(global_version_mass_before=11)
    with pytest.raises(ValueError, match="worst selected-event bucket"):
        _make_score(worst_case_surviving_event_version_count=2)
    with pytest.raises(ValueError, match="global version mass"):
        _make_score(worst_posterior_global_version_product=4)


def test_score_schema_has_no_sum_squared_proxy_field() -> None:
    assert "sum_squared_outcome_bucket_sizes" not in {
        field.name for field in fields(active_module.OpaqueActiveCandidateScore)
    }


def test_pure_active_api_has_no_controller_or_semantic_parameters_or_calls() -> None:
    pure_functions = (
        active_module.initialize_opaque_active_discovery,
        active_module.choose_next_opaque_edge,
        active_module.incorporate_opaque_response,
        active_module.incorporate_opaque_structural_inference,
        active_module.finalize_opaque_active_discovery,
        active_module.predict_defined_suffix,
    )
    forbidden_fragments = {
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
        "salt",
        "commitment_root",
        "sealed_answer",
    }
    forbidden_names = {
        "_semantic_step",
        "_execute_semantic",
        "_ACTIONS",
        "_STATES",
        "ToyPartialControllerEnvironment",
        "build_omission_controller_environment",
        "release_first_active_response",
        "release_remaining_active_responses",
    }
    for function in pure_functions:
        assert not any(
            fragment in parameter
            for parameter in inspect.signature(function).parameters
            for fragment in forbidden_fragments
        )
        tree = ast.parse(inspect.getsource(function))
        used_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert used_names.isdisjoint(forbidden_names)


def test_transitive_pure_call_graph_and_runtime_do_not_reach_controller_semantics(
    monkeypatch,
    primary_input,
    primary_run,
) -> None:
    module_tree = ast.parse(inspect.getsource(active_module))
    definitions = {
        node.name: node
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    references = {
        name: {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and child.id in definitions
        }
        for name, node in definitions.items()
    }
    frontier = [
        "initialize_opaque_active_discovery",
        "choose_next_opaque_edge",
        "incorporate_opaque_response",
        "incorporate_opaque_structural_inference",
        "finalize_opaque_active_discovery",
        "predict_defined_suffix",
    ]
    reachable: set[str] = set()
    while frontier:
        name = frontier.pop()
        if name in reachable:
            continue
        reachable.add(name)
        frontier.extend(references.get(name, ()))
    assert reachable.isdisjoint(
        {
            "_semantic_step",
            "_execute_semantic",
            "build_omission_controller_environment",
            "release_first_active_response",
            "release_remaining_active_responses",
        }
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pure learner reached trusted controller semantics")

    for name in (
        "_semantic_step",
        "_execute_semantic",
        "build_omission_controller_environment",
        "release_first_active_response",
        "release_remaining_active_responses",
    ):
        monkeypatch.setattr(partial_module, name, forbidden)
    recorded_result, _ = primary_run
    prefix = active_module.initialize_opaque_active_discovery(primary_input)
    for recorded in recorded_result.final_state.steps:
        choice = active_module.choose_next_opaque_edge(primary_input, prefix)
        assert choice == recorded.choice
        prefix = (
            active_module.incorporate_opaque_response(
                primary_input, prefix, choice, recorded.response
            )
            if type(recorded) is active_module.OpaqueActiveStep
            else active_module.incorporate_opaque_structural_inference(
                primary_input, prefix, choice
            )
        )
    replayed = active_module.finalize_opaque_active_discovery(primary_input, prefix)
    assert replayed == recorded_result


def test_current_and_counterfactual_answers_share_one_preanswer_commitment(
    primary_input,
    primary_run,
) -> None:
    state = active_module.initialize_opaque_active_discovery(primary_input)
    choice = active_module.choose_next_opaque_edge(primary_input, state)
    assert choice == primary_run[0].final_state.steps[0].choice
    assert choice is not None and choice.requires_membership_response
    assert len(choice.exact_outcome_branches) == 3
    counterfactual_states = []
    for branch in choice.exact_outcome_branches:
        response = active_module.make_opaque_active_response(
            primary_input,
            state,
            choice,
            branch.target_answers,
        )
        assert response.prior_choice_sha256 == choice.choice_sha256
        counterfactual_states.append(
            active_module.incorporate_opaque_response(
                primary_input,
                state,
                choice,
                response,
            )
        )
    assert len({row.state_sha256 for row in counterfactual_states}) == 3
    assert active_module.choose_next_opaque_edge(primary_input, state) == choice


def test_out_of_order_response_and_stale_sum_squares_choice_fail_closed(
    primary_input,
    primary_run,
) -> None:
    result, _ = primary_run
    initial = active_module.initialize_opaque_active_discovery(primary_input)
    first_choice = active_module.choose_next_opaque_edge(primary_input, initial)
    assert first_choice is not None
    second_response = next(
        row.response
        for row in result.final_state.steps[1:]
        if type(row) is active_module.OpaqueActiveStep
    )
    with pytest.raises(ValueError, match="response does not answer the committed request"):
        active_module.incorporate_opaque_response(
            primary_input,
            initial,
            first_choice,
            second_response,
        )

    postfront_choice = result.final_state.steps[3].choice
    stale_proxy_order = tuple(
        sorted(
            postfront_choice.eligible_scores,
            key=lambda row: (
                row.worst_case_surviving_event_version_count,
                sum(bucket[2] ** 2 for bucket in row.outcome_bucket_rows),
                -row.compatible_outcome_count,
                row.request_sha256,
            ),
        )
    )
    assert stale_proxy_order[0].request_sha256 != (
        postfront_choice.chosen_request.request_sha256
    )
    with pytest.raises(ValueError, match="deterministic selection order"):
        replace(postfront_choice, eligible_scores=stale_proxy_order)


def test_primary_fixture_has_the_exact_frozen_14q_1i_version_trace(primary_run) -> None:
    result, committed_choice_hashes = primary_run
    steps = result.final_state.steps
    assert tuple(
        "Q" if type(row) is active_module.OpaqueActiveStep else "I"
        for row in steps
    ) == ("Q", "Q", "I") + ("Q",) * 12
    chosen_scores = tuple(row.choice.eligible_scores[0] for row in steps)
    assert tuple(row.global_version_mass_before for row in chosen_scores) == (
        410_090_902_188_462,
        3_215_218_050,
        1_607_609_025,
        1_607_609_025,
        178_623_225,
        19_847_025,
        2_205_225,
        245_025,
        27_225,
        3_600,
        576,
        144,
        18,
        6,
        2,
    )
    assert tuple(row.compatible_outcome_count for row in chosen_scores) == (
        3,
        2,
        1,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        4,
        9,
        3,
        3,
        2,
    )
    assert tuple(row.source_assignment_count_before for row in chosen_scores) == (
        6,
        2,
        1,
    ) + (1,) * 12
    assert tuple(
        row.worst_posterior_global_version_product for row in chosen_scores
    ) == (
        205_043_843_485_206,
        1_607_609_025,
        1_607_609_025,
        178_623_225,
        19_847_025,
        2_205_225,
        245_025,
        27_225,
        3_600,
        576,
        144,
        36,
        6,
        2,
        1,
    )
    assert tuple(
        row.worst_case_surviving_event_version_count for row in chosen_scores[3:]
    ) == (1, 1, 9, 1, 121, 16, 4, 1, 4, 1, 1, 1)
    assert sorted(row[3] for row in chosen_scores[0].outcome_bucket_rows) == [
        3_215_218_050,
        205_043_843_485_206,
        205_043_843_485_206,
    ]
    assert committed_choice_hashes == tuple(
        row.choice.choice_sha256
        for row in steps
        if type(row) is active_module.OpaqueActiveStep
    )


def test_every_choice_recomputes_the_frozen_causal_selection_tuple(primary_run) -> None:
    result, _ = primary_run
    saw_sum_squares_disagreement = False
    for step in result.final_state.steps:
        choice = step.choice
        rows = choice.eligible_scores
        if choice.singleton_fixed_point_applied:
            expected = min(rows, key=lambda row: row.request_sha256)
        elif choice.source_assignment_incomplete_before_choice:
            expected = min(
                rows,
                key=lambda row: (-row.compatible_outcome_count, row.request_sha256),
            )
        else:
            expected = min(
                rows,
                key=lambda row: (
                    row.worst_posterior_global_version_product,
                    row.worst_case_surviving_event_version_count,
                    -row.compatible_outcome_count,
                    row.request_sha256,
                ),
            )
            stale_sum_squares_proxy = min(
                rows,
                key=lambda row: (
                    row.worst_case_surviving_event_version_count,
                    sum(bucket[2] ** 2 for bucket in row.outcome_bucket_rows),
                    -row.compatible_outcome_count,
                    row.request_sha256,
                ),
            )
            saw_sum_squares_disagreement |= (
                stale_sum_squares_proxy.request_sha256 != expected.request_sha256
            )
        assert expected.request_sha256 == choice.chosen_request.request_sha256
        for score in rows:
            assignment_buckets = tuple(row[1] for row in score.outcome_bucket_rows)
            event_buckets = tuple(row[2] for row in score.outcome_bucket_rows)
            global_buckets = tuple(row[3] for row in score.outcome_bucket_rows)
            if score.frontier_target_unlock:
                assert sum(assignment_buckets) == score.source_assignment_count_before
            else:
                assert all(
                    value <= score.source_assignment_count_before
                    for value in assignment_buckets
                )
            assert sum(event_buckets) == score.selected_event_version_count_before
            assert sum(global_buckets) == score.global_version_mass_before
            assert max(event_buckets) == score.worst_case_surviving_event_version_count
            assert max(global_buckets) == score.worst_posterior_global_version_product
    assert saw_sum_squares_disagreement


def test_actual_branch_mass_is_the_next_causal_version_mass(primary_run) -> None:
    result, _ = primary_run
    steps = result.final_state.steps
    for index, step in enumerate(steps):
        target_answers = (
            step.response.target_answers
            if type(step) is active_module.OpaqueActiveStep
            else step.inference.inferred_target_answers
        )
        branch = next(
            row
            for row in step.choice.exact_outcome_branches
            if row.target_answers == target_answers
        )
        score = step.choice.eligible_scores[0]
        assert branch.source_assignment_count_before == score.source_assignment_count_before
        assert branch.selected_event_version_count_before == score.selected_event_version_count_before
        assert branch.global_version_mass_before == score.global_version_mass_before
        expected_next_mass = (
            steps[index + 1].choice.eligible_scores[0].global_version_mass_before
            if index + 1 < len(steps)
            else 1
        )
        assert branch.global_version_mass_after == expected_next_mass
    assert steps[2].choice.singleton_fixed_point_applied
    assert steps[2].inference.inference_kind == (
        "full_product_source_bijection_singleton"
    )
    assert steps[2].inference.returned_categorical_token_count == 0
    assert not hasattr(steps[2].inference, "response_ordinal")


def test_every_choice_cross_links_each_branch_to_its_exact_score_bucket(
    primary_run,
) -> None:
    result, _ = primary_run
    for step in result.final_state.steps:
        choice = step.choice
        score = choice.eligible_scores[0]
        buckets = {
            answers: (assignments, event_versions, global_mass)
            for answers, assignments, event_versions, global_mass
            in score.outcome_bucket_rows
        }
        assert set(buckets) == {
            branch.target_answers for branch in choice.exact_outcome_branches
        }
        for branch in choice.exact_outcome_branches:
            assert (
                branch.source_assignment_count_before,
                branch.selected_event_version_count_before,
                branch.global_version_mass_before,
            ) == (
                score.source_assignment_count_before,
                score.selected_event_version_count_before,
                score.global_version_mass_before,
            )
            assert (
                branch.source_assignment_count_after,
                branch.selected_event_version_count_after,
                branch.global_version_mass_after,
            ) == buckets[branch.target_answers]


def test_authoritative_prefix_replay_reconstructs_every_choice_and_state(
    primary_input,
    primary_run,
) -> None:
    result, _ = primary_run
    prefix = active_module.initialize_opaque_active_discovery(primary_input)
    for recorded in result.final_state.steps:
        choice = active_module.choose_next_opaque_edge(primary_input, prefix)
        assert choice == recorded.choice
        for name in (
            "selected_before_current_response",
            "deterministic_sha_tiebreak",
        ):
            assert getattr(choice, name)
        for name in (
            "current_response_labels_used",
            "future_response_labels_used",
            "sealed_answers_used",
            "controller_candidate_order_used",
            "semantic_roles_used",
        ):
            assert not getattr(choice, name)
        if type(recorded) is active_module.OpaqueActiveStep:
            prefix = active_module.incorporate_opaque_response(
                primary_input,
                prefix,
                choice,
                recorded.response,
            )
        else:
            prefix = active_module.incorporate_opaque_structural_inference(
                primary_input,
                prefix,
                choice,
            )
        assert prefix.steps[-1] == recorded
    assert prefix == result.final_state
    assert active_module.choose_next_opaque_edge(primary_input, prefix) is None
    active_module.validate_opaque_active_state(primary_input, prefix)


def test_primary_result_closes_only_the_guarded_partial_maps(primary_controller, primary_run) -> None:
    result, _ = primary_run
    assert (
        result.active_call_count,
        result.structural_inference_count,
        result.returned_categorical_token_count,
        result.unopened_candidate_count,
    ) == (14, 1, 28, 8)
    assert result.primary_omission_realized_14q_1i_8sealed
    assert result.strict_causal_minimax_selector_used
    assert not result.posthoc_truth_specific_13_query_teaching_set_used_by_selector
    assert not result.global_query_minimality_claimed
    assert result.model.total_operator is None
    assert all(row.total_operator is None for row in result.model.operators)
    assert all(row.categorical_version_count == 1 for row in result.model.operators)
    assert all(row.raw_restricted_linear_nullity == 0 for row in result.model.operators)
    assert result.aggregate_raw_restricted_linear_nullity == 0
    assert result.aggregate_total_extension_nullity == 80
    assert sum(
        value for _, value in result.final_state.inference_event_rank_increments
    ) == 1
    assert all(
        row.direct_passive_source_rank
        + row.response_source_rank_increment
        + row.inference_source_rank_increment
        == row.final_observed_source_rank
        == row.legal_domain_rank
        for row in result.model.operators
    )
    selected = {
        row.choice.chosen_request.request_sha256 for row in result.final_state.steps
    }
    t1_supplied_basis = {
        row.request_sha256
        for row in primary_controller.learner_input.candidate_edge_requests[:15]
    }
    assert selected != t1_supplied_basis

    guarded = result.guarded_language
    assert (
        guarded.state_count,
        guarded.defined_transition_count,
        guarded.undefined_pair_count,
    ) == (9, 44, 46)
    assert guarded.exact_guard_rejection_count == 46
    assert guarded.restricted_domain_span_control_rejection_count == 46
    assert guarded.all_defined_targets_are_declared_states
    assert guarded.all_undefined_pairs_rejected
    assert guarded.all_undefined_pairs_rejected_by_exact_guard
    assert guarded.all_undefined_pairs_outside_restricted_domain_span
    assert guarded.arbitrary_length_legal_suffix_induction
    assert not guarded.arbitrary_suffix_without_definedness_guard_claimed
    assert not guarded.total_wfa_claimed
    assert not guarded.unqueried_edge_answers_used_to_fit_or_select
    for row in guarded.defined_transitions:
        assert row.admitted_by_exact_categorical_guard
        assert row.prediction_is_unique_categorical_version
        assert active_module.predict_defined_suffix(
            result,
            row.source_answers,
            (row.event_token,),
        ) == row.predicted_target_answers
    assert all(
        row.rejected_by_exact_categorical_guard
        and row.rejected_by_restricted_domain_span_control
        for row in guarded.undefined_pairs
    )


def test_exact_mask_guard_rejects_before_any_restricted_operator_application(
    monkeypatch,
    primary_run,
) -> None:
    result, _ = primary_run
    undefined = result.guarded_language.undefined_pairs[0]
    apply_calls = 0

    def forbidden_apply(*_args, **_kwargs):
        nonlocal apply_calls
        apply_calls += 1
        raise AssertionError("undefined pair reached restricted linear algebra")

    monkeypatch.setattr(
        active_module.AutonomousRestrictedOperator,
        "apply_row",
        forbidden_apply,
    )
    with pytest.raises(ValueError, match="exact definedness mask"):
        result.model.predict_defined_suffix(
            undefined.source_answers,
            (undefined.event_token,),
        )
    assert apply_calls == 0


@pytest.mark.parametrize(
    (
        "budget_overrides",
        "branch_index",
        "minimum_unopened",
        "expected_reason",
        "expected_calls",
    ),
    (
        (
            {"max_active_calls": 14},
            0,
            0,
            "active_call_budget_exhausted",
            14,
        ),
        (
            {"max_returned_categorical_tokens": 28},
            0,
            0,
            "returned_categorical_token_budget_exhausted",
            14,
        ),
        (
            {"max_structural_inferences": 1},
            -1,
            0,
            "structural_inference_budget_exhausted",
            8,
        ),
        (
            {},
            0,
            23,
            "sealed_candidate_quota_reached",
            0,
        ),
    ),
)
def test_every_runtime_cap_returns_typed_not_identified_before_blocked_answer(
    primary_controller,
    budget_overrides,
    branch_index,
    minimum_unopened,
    expected_reason,
    expected_calls,
) -> None:
    budgets = active_module.OpaqueActiveDiscoveryBudgets(**budget_overrides)
    learner_input = active_module.make_opaque_active_input(
        primary_controller.learner_input,
        budgets=budgets,
    )
    provider_calls: list[str] = []

    def compatible_counterfactual_provider(choice):
        provider_calls.append(choice.choice_sha256)
        return choice.exact_outcome_branches[branch_index].target_answers

    result = active_module.run_opaque_active_discovery(
        learner_input,
        compatible_counterfactual_provider,
        minimum_unopened_candidate_count=minimum_unopened,
    )
    assert type(result) is active_module.OpaqueActiveNotIdentifiedResult
    assert result.identification_status == "not_identified_budget_or_sealed_quota"
    assert result.stop_reason == expected_reason
    assert len(provider_calls) == result.active_call_count == expected_calls
    assert not result.blocked_choice_response_provider_called
    assert not result.blocked_choice_answer_opened
    assert result.blocked_choice.choice_sha256 not in provider_calls
    assert result.unopened_candidate_count >= result.minimum_unopened_candidate_count


def test_all_first_response_branches_separate_actual_identification_from_typed_stops(
    primary_controller,
) -> None:
    """The official cap succeeds only on the actual first-answer branch.

    The other two pre-answer-compatible branches are deliberately continued
    with compatible categorical responses.  They must retain honest residual
    version mass and stop at the sealed quota without a fifteenth oracle call;
    they are not allowed to borrow the actual branch's identification result.
    """

    budgets = active_module.OpaqueActiveDiscoveryBudgets(
        max_active_calls=14,
        max_structural_inferences=1,
        max_returned_categorical_tokens=28,
    )
    learner_input = active_module.make_opaque_active_input(
        primary_controller.learner_input,
        budgets=budgets,
    )
    initial_state = active_module.initialize_opaque_active_discovery(learner_input)
    first_choice = active_module.choose_next_opaque_edge(
        learner_input, initial_state
    )
    assert first_choice is not None
    actual_first_answers = _trusted_development_answer(
        primary_controller, first_choice
    )
    first_branch_answers = tuple(
        branch.target_answers for branch in first_choice.exact_outcome_branches
    )
    assert len(first_branch_answers) == 3
    assert first_branch_answers.count(actual_first_answers) == 1
    actual_branch_index = first_branch_answers.index(actual_first_answers)

    for first_branch_index in range(3):
        provider_calls: list[str] = []

        def compatible_provider(choice):
            provider_calls.append(choice.choice_sha256)
            if len(provider_calls) == 1:
                return choice.exact_outcome_branches[
                    first_branch_index
                ].target_answers
            controller_answers = _trusted_development_answer(
                primary_controller, choice
            )
            compatible = tuple(
                branch.target_answers
                for branch in choice.exact_outcome_branches
            )
            return (
                controller_answers
                if controller_answers in compatible
                else compatible[0]
            )

        result = active_module.run_opaque_active_discovery(
            learner_input,
            compatible_provider,
            minimum_unopened_candidate_count=8,
        )
        assert len(provider_calls) == 14
        if first_branch_index == actual_branch_index:
            assert type(result) is active_module.AutonomousPartialOperatorResult
            assert result.identification_status == "identified"
            assert result.primary_omission_realized_14q_1i_8sealed
            continue

        assert type(result) is active_module.OpaqueActiveNotIdentifiedResult
        assert result.identification_status == "not_identified_budget_or_sealed_quota"
        assert result.stop_reason == "sealed_candidate_quota_reached"
        assert (
            result.active_call_count,
            result.structural_inference_count,
            result.returned_categorical_token_count,
            result.unopened_candidate_count,
        ) == (14, 1, 28, 8)
        assert not result.blocked_choice_response_provider_called
        assert not result.blocked_choice_answer_opened
        assert result.blocked_choice is not None
        assert result.blocked_choice.choice_sha256 not in provider_calls

        # The stopped branch remains genuinely nonidentified: the causal
        # posterior has 363 global versions, including a 121-version selected
        # event bucket, and several event source domains remain rank deficient.
        blocked_scores = result.blocked_choice.eligible_scores
        assert blocked_scores
        assert {
            score.global_version_mass_before for score in blocked_scores
        } == {363}
        assert max(
            score.selected_event_version_count_before for score in blocked_scores
        ) >= 121
        assert sorted(
            rank for _, rank in result.final_state.observed_event_ranks
        ) == [1, 3, 3, 3, 3, 3, 4, 4, 5, 5]


def test_validation_replay_ceiling_uses_cached_prefixes_but_fails_closed_uncached(
    monkeypatch,
    primary_controller,
) -> None:
    budgets = active_module.OpaqueActiveDiscoveryBudgets(
        max_validation_replay_decisions=1
    )
    learner_input = active_module.make_opaque_active_input(
        primary_controller.learner_input,
        budgets=budgets,
    )
    token_to_action = {
        token: action for action, token in primary_controller.canonical_event_tokens
    }
    state_to_answers = dict(primary_controller.canonical_state_answers)

    def trusted_provider(choice):
        semantic_state = (-1, -1)
        for token in choice.chosen_request.program:
            semantic_state = partial_module._semantic_step(
                semantic_state, token_to_action[token]
            )
            assert semantic_state is not None
        return state_to_answers[semantic_state]

    isolated_cache: set[tuple[str, str]] = set()
    monkeypatch.setattr(active_module, "_VALIDATED_STATE_CACHE", isolated_cache)
    result = active_module.run_opaque_active_discovery(
        learner_input, trusted_provider
    )
    assert type(result) is active_module.AutonomousPartialOperatorResult
    active_module.validate_opaque_active_state(learner_input, result.final_state)
    isolated_cache.clear()
    with pytest.raises(
        active_module.OpaqueActiveDiscoveryLimitError,
        match="validation replay.*ceiling",
    ):
        active_module.validate_opaque_active_state(
            learner_input, result.final_state
        )


def test_public_exports_bind_primitive_input_and_typed_nonidentification() -> None:
    assert {
        "make_opaque_active_input_from_rows",
        "make_opaque_candidate_removal_negative_control",
        "OpaqueActiveNotIdentifiedResult",
        "OpaqueActiveCandidatePoolExhaustedError",
    } <= set(active_module.__all__)


def test_removing_a_zero_passive_bind_cocircuit_yields_not_identified(
    primary_controller,
    primary_input,
) -> None:
    direct_ranks = dict(
        active_module.initialize_opaque_active_discovery(
            primary_input
        ).direct_passive_event_ranks
    )
    candidates_by_event = {
        token: tuple(
            row
            for row in primary_input.canonical_candidate_requests
            if row.event_token == token
        )
        for token in primary_input.event_tokens
    }
    zero_passive_three_edge_event = next(
        token
        for token, rows in candidates_by_event.items()
        if direct_ranks[token] == 0
        and len(
            tuple(
                row
                for row in primary_input.canonical_defined_requests
                if row.event_token == token
            )
        )
        == 3
        and len(rows) == 3
    )
    removed = candidates_by_event[zero_passive_three_edge_event][0]
    negative_input = active_module.make_opaque_candidate_removal_negative_control(
        primary_input, removed.request_sha256
    )
    assert not negative_input.candidate_pool_complete
    assert negative_input.single_candidate_removal_negative_control
    assert negative_input.missing_candidate_request_sha256s == (
        removed.request_sha256,
    )

    token_to_action = {
        token: action for action, token in primary_controller.canonical_event_tokens
    }
    state_to_answers = dict(primary_controller.canonical_state_answers)
    provider_calls: list[str] = []

    def trusted_provider(choice):
        provider_calls.append(choice.choice_sha256)
        semantic_state = (-1, -1)
        for token in choice.chosen_request.program:
            semantic_state = partial_module._semantic_step(
                semantic_state, token_to_action[token]
            )
            assert semantic_state is not None
        return state_to_answers[semantic_state]

    result = active_module.run_opaque_active_discovery(
        negative_input,
        trusted_provider,
        minimum_unopened_candidate_count=0,
    )
    assert type(result) is active_module.OpaqueActiveNotIdentifiedResult
    assert result.stop_reason == "candidate_pool_exhausted_before_identification"
    assert result.blocked_choice is None
    assert not result.blocked_choice_response_provider_called
    assert not result.blocked_choice_answer_opened
    assert len(provider_calls) == result.active_call_count
    assert dict(result.final_state.observed_event_ranks)[
        zero_passive_three_edge_event
    ] == 2


@pytest.mark.skipif(
    os.environ.get("TNLM_RUN_SLOW_PHASE3_T2") != "1",
    reason="set TNLM_RUN_SLOW_PHASE3_T2=1 for all eight development rotations",
)
def test_all_eight_development_rotations_share_the_exact_primary_closure() -> None:
    reference_kinds: tuple[str, ...] | None = None
    reference_global_trace: tuple[int, ...] | None = None
    reference_outcome_trace: tuple[int, ...] | None = None
    for block in range(2):
        for cell_index, cell in enumerate(DEVELOPMENT_CELLS):
            environment_index = block * len(DEVELOPMENT_CELLS) + cell_index
            controller = partial_module.build_omission_controller_environment(
                cell,
                block,
                controller_nonce=DEVELOPMENT_OMISSION_NONCE_FIXTURES[
                    environment_index
                ],
            )
            learner_input = active_module.make_opaque_active_input(
                controller.learner_input
            )
            result = active_module.run_opaque_active_discovery(
                learner_input,
                lambda choice, controller=controller: _trusted_development_answer(
                    controller, choice
                ),
            )
            assert type(result) is active_module.AutonomousPartialOperatorResult
            assert (
                result.active_call_count,
                result.structural_inference_count,
                result.returned_categorical_token_count,
                result.unopened_candidate_count,
            ) == (14, 1, 28, 8)
            assert result.primary_omission_realized_14q_1i_8sealed
            assert (
                result.model.ambient_rank,
                result.model.diagnostic_rank,
                result.model.categorical_state_count,
            ) == (5, 5, 9)
            assert all(
                count == 1 for _, count in result.final_event_version_counts
            )
            assert result.model.total_operator is None
            assert all(row.total_operator is None for row in result.model.operators)
            assert (
                result.guarded_language.defined_transition_count,
                result.guarded_language.undefined_pair_count,
                result.guarded_language.exact_guard_rejection_count,
                result.guarded_language.restricted_domain_span_control_rejection_count,
            ) == (44, 46, 46, 46)
            kinds = tuple(
                "Q" if type(row) is active_module.OpaqueActiveStep else "I"
                for row in result.final_state.steps
            )
            global_trace = tuple(
                row.choice.eligible_scores[0].global_version_mass_before
                for row in result.final_state.steps
            )
            outcome_trace = tuple(
                row.choice.eligible_scores[0].compatible_outcome_count
                for row in result.final_state.steps
            )
            if reference_kinds is None:
                reference_kinds = kinds
                reference_global_trace = global_trace
                reference_outcome_trace = outcome_trace
            assert kinds == reference_kinds == ("Q", "Q", "I") + ("Q",) * 12
            assert global_trace == reference_global_trace
            assert outcome_trace == reference_outcome_trace
