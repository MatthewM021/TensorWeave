from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import os

import pytest

import tnlm_v3.opaque_active_discovery as active_module
import tnlm_v3.opaque_active_teaching_control as teaching_module
import tnlm_v3.opaque_partial_operators as partial_module


CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
DEVELOPMENT_NONCES = (
    "323b4597f9e3dfb9e7f954aab5873b8730321dd8476a2992f81e93cc7a44b046",
    "45d4ff9e973a3c9fda9a41ae7ab8c827f9e7299a2ece6f402ab32dc0fd598718",
    "a3d1591df9b8fe82666f78e885f578cfd5677df28dfa6bc2daa5cfa468d87203",
    "0ec845c35cfe807b28f8d012391d9e811211a780e3fd22510ad33a6059dcf5fc",
    "422d894dade5755a90355dea9dec66be8a8943509365c79727046eb895042bb4",
    "be187a3a14f4d575944e93a407f0e1c53142a9db4c27abc0dcb6bbb3d7712df3",
    "219803a3b13f67dbb90f23749b785d71836918b7d5d0cee78d74d85ff0319490",
    "8e9f16318b215a2e117e34bc75cbc649b602dac9d8ce5ffcdace6950c8971ce2",
)


def _fixture(index: int = 0):
    controller = partial_module.build_omission_controller_environment(
        CELLS[index % 4],
        index // 4,
        controller_nonce=DEVELOPMENT_NONCES[index],
    )
    learner_input = active_module.make_opaque_active_input(controller.learner_input)
    answers = {
        row.request.request_sha256: row.target_answers
        for row in controller.active_responses
    }
    answers.update(
        {
            request.request_sha256: sealed.expected_answers
            for request, sealed in zip(
                controller.learner_input.candidate_edge_requests[15:],
                controller.sealed_edge_programs,
                strict=True,
            )
        }
    )
    return learner_input, answers


@pytest.fixture(scope="module")
def primary_control():
    learner_input, answers = _fixture()
    result = teaching_module.discover_postfit_teaching_control(
        learner_input, answers
    )
    return learner_input, answers, result


def test_teaching_module_is_postfit_only_and_has_no_semantic_controller_imports() -> None:
    tree = ast.parse(inspect.getsource(teaching_module))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(
        fragment in name
        for name in imports
        for fragment in ("controller", "semantic", "executor")
    )
    assert "opaque_active_teaching_control" not in inspect.getsource(active_module)


def test_exact_13q_2i_8_partition_and_claim_boundary(primary_control) -> None:
    _, _, result = primary_control
    assert (result.query_count, result.inference_count, result.unopened_count) == (
        13,
        2,
        8,
    )
    assert result.counterfactual_returned_categorical_label_count == 26
    assert result.new_membership_calls_made == 0
    assert result.aggregate_independent_rank_gains == 15
    assert result.aggregate_image_coordinate_constraints == 75
    assert result.truth_aware_postfit_control
    assert result.truth_specific_noncausal_control
    assert result.query_count_is_counterfactual
    assert not result.selection_eligible
    assert not result.confirmatory_claim_eligible
    assert not result.global_query_minimality_claimed
    assert not result.arbitrary_total_operator_constructed
    assert result.total_operator is None


def test_two_inferences_are_answer_free_singletons_with_exact_rank_gain(
    primary_control,
) -> None:
    _, _, result = primary_control
    inferred = tuple(
        row
        for row in result.teaching_decisions
        if row.acquisition_kind == "singleton_inferred"
    )
    assert tuple(row.inference_kind for row in inferred) == (
        "full_product_source_bijection_singleton",
        "categorical_restricted_map_singleton_rank_completion",
    )
    assert all(row.returned_categorical_label_count == 0 for row in inferred)
    assert all(row.inference_value_derived_without_truth_lookup for row in inferred)
    assert all(row.inference_value_verified_against_postfit_truth for row in inferred)
    assert all(
        row.observed_source_rank_after == row.observed_source_rank_before + 1
        for row in inferred
    )
    assert not inferred[0].selected_event_map_already_closed_before_inference
    assert inferred[1].selected_event_map_already_closed_before_inference


def test_truth_selected_rows_bind_all_eligible_actual_posterior_scores(
    primary_control,
) -> None:
    _, _, result = primary_control
    queried = tuple(
        row for row in result.teaching_decisions if row.acquisition_kind == "queried"
    )
    assert len(queried) == 13
    assert all(row.complete_truth_map_used_to_select_request for row in queried)
    assert all(row.complete_truth_map_used_to_supply_target for row in queried)
    assert all(row.returned_categorical_label_count == 2 for row in queried)
    assert result.truth_score_rows_evaluated == sum(
        len(row.eligible_truth_scores) for row in queried
    )


def test_every_restricted_map_and_raw_legal_domain_rank_close(primary_control) -> None:
    _, _, result = primary_control
    assert len(result.final_event_version_rows) == 10
    assert all(count == 1 for _, count, _ in result.final_event_version_rows)
    assert len(result.final_event_rank_rows) == 10
    assert all(observed == legal for _, observed, legal in result.final_event_rank_rows)
    assert result.restricted_categorical_maps_closed


def test_primary_is_reconstructed_first_and_remains_unchanged(primary_control) -> None:
    _, _, result = primary_control
    assert len(result.primary_queried_request_sha256s) == 14
    assert len(result.primary_inferred_request_sha256s) == 1
    assert result.primary_reconstructed_before_teaching_search
    assert result.primary_result_unchanged_by_control
    assert result.primary_posthoc_selector_flag_remains_false
    assert not result.primary_selector_received_teaching_choices
    assert result.teaching_trace_differs_from_primary_trace


def test_answer_map_order_is_erased_and_authoritative_replay_is_exact(
    primary_control,
) -> None:
    learner_input, answers, result = primary_control
    reversed_answers = dict(reversed(tuple(answers.items())))
    repeated = teaching_module.discover_postfit_teaching_control(
        learner_input, reversed_answers
    )
    assert repeated == result
    teaching_module.validate_postfit_teaching_control(
        learner_input, answers, result
    )


def test_wrong_postfit_answer_map_cannot_validate_the_frozen_control(
    primary_control,
) -> None:
    learner_input, answers, result = primary_control
    altered = dict(answers)
    request = learner_input.canonical_candidate_requests[0]
    original = altered[request.request_sha256]
    altered[request.request_sha256] = next(
        row
        for row in sorted(
            __import__("itertools").product(learner_input.answer_tokens, repeat=2)
        )
        if row != original
    )
    with pytest.raises(ValueError):
        teaching_module.validate_postfit_teaching_control(
            learner_input, altered, result
        )


def test_budget_underflow_fails_before_postfit_search() -> None:
    learner_input, answers = _fixture()
    with pytest.raises(
        active_module.OpaqueActiveDiscoveryLimitError,
        match="max_queries",
    ):
        teaching_module.discover_postfit_teaching_control(
            learner_input,
            answers,
            budgets=teaching_module.PostfitTeachingControlBudgets(
                max_queries=12
            ),
        )


def test_claim_and_total_operator_forgery_is_rejected(primary_control) -> None:
    _, _, result = primary_control
    with pytest.raises(ValueError, match="global_query_minimality_claimed"):
        replace(result, global_query_minimality_claimed=True)
    with pytest.raises(ValueError, match="total operator"):
        replace(result, total_operator=((1,),))


@pytest.mark.skipif(
    os.environ.get("TNLM_RUN_SLOW_PHASE3_T2") != "1",
    reason="set TNLM_RUN_SLOW_PHASE3_T2=1 for all eight development teaching controls",
)
def test_all_eight_development_omissions_realize_13q_2i() -> None:
    hashes: list[str] = []
    for index in range(8):
        learner_input, answers = _fixture(index)
        result = teaching_module.discover_postfit_teaching_control(
            learner_input, answers
        )
        assert (result.query_count, result.inference_count, result.unopened_count) == (
            13,
            2,
            8,
        )
        assert all(
            observed == legal
            for _, observed, legal in result.final_event_rank_rows
        )
        hashes.append(result.result_sha256)
    assert len(set(hashes)) == 8
