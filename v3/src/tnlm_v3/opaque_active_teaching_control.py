"""Truth-aware, postfit teaching-set control for Phase III-T2.

This module is deliberately outside the causal learner.  Its input includes
the complete 23-row candidate answer map, so its choices are ineligible as
active-selection evidence.  It reconstructs the primary opaque selector
first, then uses the already-open truth to find a deterministic 13-query
teaching trace.  One full-product source label is inferred on the way to
closure; after closure, one further unopened edge is verified as a categorical
singleton, leaving eight sealed rows.  The control establishes a compact
truth-specific teaching set, not global query minimality and not a total WFA.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence

from . import opaque_active_discovery as _active
from .opaque_partial_operators import OpaqueEdgeRequest


_ANSWER_MAP_COUNT = 23
_QUERY_COUNT = 13
_INFERENCE_COUNT = 2
_UNOPENED_COUNT = 8
_DECISION_COUNT = 15
_RETURNED_LABEL_COUNT = 26
_EVENT_COUNT = 10

_SCORE_SCHEMA = "tnlm-v3-t2-postfit-teaching-truth-score-v1"
_DECISION_SCHEMA = "tnlm-v3-t2-postfit-teaching-decision-v1"
_RESULT_SCHEMA = "tnlm-v3-t2-postfit-teaching-control-v1"


def _sha256(value: object) -> str:
    return _active._sha256(value)


def _require_sha256(name: str, value: object) -> str:
    return _active._require_sha256(name, value)


def _plain_int(name: str, value: object, minimum: int = 0) -> int:
    return _active._plain_int(name, value, minimum)


def _require_bool(name: str, value: object) -> bool:
    return _active._require_bool(name, value)


def _request_payload(request: OpaqueEdgeRequest) -> dict[str, object]:
    return _active._request_payload(request)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _active._jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class PostfitTeachingControlBudgets:
    max_candidate_answer_rows: int = 23
    max_decisions: int = 15
    max_queries: int = 13
    max_inferences: int = 2
    max_truth_score_rows: int = 240
    max_exact_rank_evaluations: int = 480
    max_version_space_recomputations: int = 17
    max_conditional_assignment_blocks: int = 32
    max_cumulative_basis_image_candidates: int = 3_000_000
    max_certificate_bytes: int = 4_000_000

    def __post_init__(self) -> None:
        ceilings = {
            "max_candidate_answer_rows": 23,
            "max_decisions": 15,
            "max_queries": 13,
            "max_inferences": 2,
            "max_truth_score_rows": 240,
            "max_exact_rank_evaluations": 480,
            "max_version_space_recomputations": 17,
            "max_conditional_assignment_blocks": 32,
            "max_cumulative_basis_image_candidates": 3_000_000,
            "max_certificate_bytes": 4_000_000,
        }
        for name, ceiling in ceilings.items():
            value = _plain_int(name, getattr(self, name), 1)
            if value > ceiling:
                raise ValueError(f"{name} exceeds the frozen postfit-control ceiling")

    def payload(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, order=True)
class PostfitTruthScore:
    actual_posterior_global_version_mass: int
    actual_selected_event_version_count: int
    request_sha256: str
    actual_source_assignment_count: int
    observed_source_rank_before: int
    observed_source_rank_after: int
    source_frontier_phase: bool
    score_sha256: str
    schema: str = _SCORE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SCORE_SCHEMA:
            raise ValueError("unknown postfit truth-score schema")
        _plain_int(
            "actual_posterior_global_version_mass",
            self.actual_posterior_global_version_mass,
            1,
        )
        _plain_int(
            "actual_selected_event_version_count",
            self.actual_selected_event_version_count,
            1,
        )
        _require_sha256("request_sha256", self.request_sha256)
        _plain_int(
            "actual_source_assignment_count",
            self.actual_source_assignment_count,
            1,
        )
        before = _plain_int("observed_source_rank_before", self.observed_source_rank_before)
        after = _plain_int("observed_source_rank_after", self.observed_source_rank_after)
        if after != before + 1:
            raise ValueError("a teaching query must add one observed source direction")
        _require_bool("source_frontier_phase", self.source_frontier_phase)
        if _require_sha256("score_sha256", self.score_sha256) != _sha256(
            self._payload(False)
        ):
            raise ValueError("postfit truth-score digest mismatch")

    def _payload(self, include_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "actual_posterior_global_version_mass": self.actual_posterior_global_version_mass,
            "actual_selected_event_version_count": self.actual_selected_event_version_count,
            "request_sha256": self.request_sha256,
            "actual_source_assignment_count": self.actual_source_assignment_count,
            "observed_source_rank_before": self.observed_source_rank_before,
            "observed_source_rank_after": self.observed_source_rank_after,
            "source_frontier_phase": self.source_frontier_phase,
        }
        if include_sha:
            payload["score_sha256"] = self.score_sha256
        return payload


def _make_truth_score(
    request: OpaqueEdgeRequest,
    *,
    global_mass_after: int,
    event_versions_after: int,
    source_assignments_after: int,
    rank_before: int,
    rank_after: int,
    source_frontier_phase: bool,
) -> PostfitTruthScore:
    kwargs = {
        "actual_posterior_global_version_mass": global_mass_after,
        "actual_selected_event_version_count": event_versions_after,
        "request_sha256": request.request_sha256,
        "actual_source_assignment_count": source_assignments_after,
        "observed_source_rank_before": rank_before,
        "observed_source_rank_after": rank_after,
        "source_frontier_phase": source_frontier_phase,
        "schema": _SCORE_SCHEMA,
    }
    return PostfitTruthScore(**kwargs, score_sha256=_sha256(kwargs))


@dataclass(frozen=True)
class PostfitTeachingDecision:
    ordinal: int
    request: OpaqueEdgeRequest
    acquisition_kind: str
    inference_kind: str | None
    target_answers: tuple[str, ...]
    source_assignment_count_before: int
    source_assignment_count_after: int
    selected_event_version_count_before: int
    selected_event_version_count_after: int
    global_version_mass_before: int
    global_version_mass_after: int
    observed_source_rank_before: int
    observed_source_rank_after: int
    eligible_truth_scores: tuple[PostfitTruthScore, ...]
    eligible_singleton_request_sha256s: tuple[str, ...]
    complete_truth_map_used_to_select_request: bool
    complete_truth_map_used_to_supply_target: bool
    inference_value_derived_without_truth_lookup: bool
    inference_value_verified_against_postfit_truth: bool
    returned_categorical_label_count: int
    restricted_maps_closed_after_decision: bool
    selected_event_map_already_closed_before_inference: bool
    decision_sha256: str
    schema: str = _DECISION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _DECISION_SCHEMA:
            raise ValueError("unknown postfit teaching-decision schema")
        _plain_int("ordinal", self.ordinal, 1)
        if type(self.request) is not OpaqueEdgeRequest:
            raise TypeError("teaching decision requires an exact opaque request")
        _active._validate_answers("teaching target answers", self.target_answers)
        for name in (
            "source_assignment_count_before",
            "source_assignment_count_after",
            "selected_event_version_count_before",
            "selected_event_version_count_after",
            "global_version_mass_before",
            "global_version_mass_after",
        ):
            _plain_int(name, getattr(self, name), 1)
        before = _plain_int("observed_source_rank_before", self.observed_source_rank_before)
        after = _plain_int("observed_source_rank_after", self.observed_source_rank_after)
        if after != before + 1:
            raise ValueError("every teaching decision must add one source-rank direction")
        for name in (
            "complete_truth_map_used_to_select_request",
            "complete_truth_map_used_to_supply_target",
            "inference_value_derived_without_truth_lookup",
            "inference_value_verified_against_postfit_truth",
            "restricted_maps_closed_after_decision",
            "selected_event_map_already_closed_before_inference",
        ):
            _require_bool(name, getattr(self, name))
        if self.acquisition_kind == "queried":
            if self.inference_kind is not None:
                raise ValueError("queried teaching row cannot name an inference kind")
            if not self.eligible_truth_scores:
                raise ValueError("truth-aware query must bind every eligible score")
            frontier_flags = {
                row.source_frontier_phase for row in self.eligible_truth_scores
            }
            if len(frontier_flags) != 1:
                raise ValueError("one query score table cannot mix frontier phases")
            source_frontier = next(iter(frontier_flags))
            expected_scores = tuple(
                sorted(
                    self.eligible_truth_scores,
                    key=(
                        (lambda row: (
                            row.actual_source_assignment_count,
                            row.request_sha256,
                        ))
                        if source_frontier
                        else (lambda row: (
                            row.actual_posterior_global_version_mass,
                            row.actual_selected_event_version_count,
                            row.request_sha256,
                        ))
                    ),
                )
            )
            if self.eligible_truth_scores != expected_scores:
                raise ValueError("truth-aware scores use the wrong deterministic order")
            if expected_scores[0].request_sha256 != self.request.request_sha256:
                raise ValueError("queried row is not the truth-specific best score")
            if self.eligible_singleton_request_sha256s:
                raise ValueError("a query cannot coexist with a pending singleton")
            if not (
                self.complete_truth_map_used_to_select_request
                and self.complete_truth_map_used_to_supply_target
                and not self.inference_value_derived_without_truth_lookup
                and not self.inference_value_verified_against_postfit_truth
                and self.returned_categorical_label_count == 2
                and after == before + 1
            ):
                raise ValueError("queried-row truth/provenance flags are inconsistent")
        elif self.acquisition_kind == "singleton_inferred":
            allowed = {
                "full_product_source_bijection_singleton",
                "categorical_restricted_map_singleton_rank_completion",
            }
            if self.inference_kind not in allowed:
                raise ValueError("unknown teaching singleton kind")
            if self.eligible_truth_scores:
                raise ValueError("singleton propagation must precede truth-scored queries")
            if tuple(sorted(self.eligible_singleton_request_sha256s)) != self.eligible_singleton_request_sha256s:
                raise ValueError("singleton pool must use canonical request order")
            if not self.eligible_singleton_request_sha256s or self.eligible_singleton_request_sha256s[0] != self.request.request_sha256:
                raise ValueError("inference must select the first opaque singleton")
            if not (
                not self.complete_truth_map_used_to_select_request
                and not self.complete_truth_map_used_to_supply_target
                and self.inference_value_derived_without_truth_lookup
                and self.inference_value_verified_against_postfit_truth
                and self.returned_categorical_label_count == 0
            ):
                raise ValueError("inferred-row truth/provenance flags are inconsistent")
            if self.selected_event_map_already_closed_before_inference != (
                self.inference_kind
                == "categorical_restricted_map_singleton_rank_completion"
            ):
                raise ValueError("event-closed singleton disclosure mismatch")
            if self.selected_event_map_already_closed_before_inference and not (
                self.global_version_mass_before == self.global_version_mass_after
                and self.selected_event_version_count_before
                == self.selected_event_version_count_after
                == self.source_assignment_count_before
                == self.source_assignment_count_after
                == 1
                and after == before + 1
            ):
                raise ValueError("event-closed inference must add one observed rank direction")
        else:
            raise ValueError("unknown teaching acquisition kind")
        if self.restricted_maps_closed_after_decision != (
            self.global_version_mass_after == 1
            and self.source_assignment_count_after == 1
        ):
            raise ValueError("restricted-map closure flag mismatch")
        if _require_sha256("decision_sha256", self.decision_sha256) != _sha256(
            self._payload(False)
        ):
            raise ValueError("teaching-decision digest mismatch")

    def _payload(self, include_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "ordinal": self.ordinal,
            "request": _request_payload(self.request),
            "acquisition_kind": self.acquisition_kind,
            "inference_kind": self.inference_kind,
            "target_answers": self.target_answers,
            "source_assignment_count_before": self.source_assignment_count_before,
            "source_assignment_count_after": self.source_assignment_count_after,
            "selected_event_version_count_before": self.selected_event_version_count_before,
            "selected_event_version_count_after": self.selected_event_version_count_after,
            "global_version_mass_before": self.global_version_mass_before,
            "global_version_mass_after": self.global_version_mass_after,
            "observed_source_rank_before": self.observed_source_rank_before,
            "observed_source_rank_after": self.observed_source_rank_after,
            "eligible_truth_scores": [row._payload(True) for row in self.eligible_truth_scores],
            "eligible_singleton_request_sha256s": self.eligible_singleton_request_sha256s,
            "complete_truth_map_used_to_select_request": self.complete_truth_map_used_to_select_request,
            "complete_truth_map_used_to_supply_target": self.complete_truth_map_used_to_supply_target,
            "inference_value_derived_without_truth_lookup": self.inference_value_derived_without_truth_lookup,
            "inference_value_verified_against_postfit_truth": self.inference_value_verified_against_postfit_truth,
            "returned_categorical_label_count": self.returned_categorical_label_count,
            "restricted_maps_closed_after_decision": self.restricted_maps_closed_after_decision,
            "selected_event_map_already_closed_before_inference": self.selected_event_map_already_closed_before_inference,
        }
        if include_sha:
            payload["decision_sha256"] = self.decision_sha256
        return payload


def _make_decision(**kwargs: object) -> PostfitTeachingDecision:
    payload = {"schema": _DECISION_SCHEMA, **kwargs}
    payload["request"] = _request_payload(kwargs["request"])  # type: ignore[arg-type]
    payload["eligible_truth_scores"] = [
        row._payload(True) for row in kwargs["eligible_truth_scores"]  # type: ignore[union-attr]
    ]
    return PostfitTeachingDecision(
        **kwargs,  # type: ignore[arg-type]
        schema=_DECISION_SCHEMA,
        decision_sha256=_sha256(payload),
    )


@dataclass(frozen=True)
class PostfitTeachingControlResult:
    learner_input_sha256: str
    complete_candidate_answer_rows: tuple[tuple[str, tuple[str, ...]], ...]
    complete_candidate_answer_map_sha256: str
    primary_reconstruction_result_sha256: str
    primary_request_sha256s: tuple[str, ...]
    primary_queried_request_sha256s: tuple[str, ...]
    primary_inferred_request_sha256s: tuple[str, ...]
    teaching_decisions: tuple[PostfitTeachingDecision, ...]
    queried_request_sha256s: tuple[str, ...]
    inferred_request_sha256s: tuple[str, ...]
    unopened_request_sha256s: tuple[str, ...]
    final_event_version_rows: tuple[tuple[str, int, str], ...]
    final_event_rank_rows: tuple[tuple[str, int, int], ...]
    query_count: int
    inference_count: int
    unopened_count: int
    returned_categorical_label_count: int
    counterfactual_membership_query_count: int
    counterfactual_returned_categorical_label_count: int
    new_membership_calls_made: int
    aggregate_independent_rank_gains: int
    aggregate_image_coordinate_constraints: int
    truth_score_rows_evaluated: int
    exact_rank_evaluations: int
    version_space_recomputations: int
    conditional_assignment_blocks_evaluated: int
    basis_image_candidates_planned: int
    closure_first_achieved_after_query_count: int
    event_closed_rank_completion_inference_count: int
    primary_reconstructed_before_teaching_search: bool
    teaching_trace_differs_from_primary_trace: bool
    truth_aware_postfit_control: bool
    truth_specific_noncausal_control: bool
    complete_answer_map_available_before_teaching_selection: bool
    query_count_is_counterfactual: bool
    primary_selector_received_teaching_choices: bool
    primary_result_unchanged_by_control: bool
    primary_posthoc_selector_flag_remains_false: bool
    selection_eligible: bool
    confirmatory_claim_eligible: bool
    global_query_minimality_claimed: bool
    restricted_categorical_maps_closed: bool
    arbitrary_total_operator_constructed: bool
    total_operator: None
    budgets: PostfitTeachingControlBudgets
    result_sha256: str
    schema: str = _RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _RESULT_SCHEMA:
            raise ValueError("unknown postfit teaching-control schema")
        _require_sha256("learner_input_sha256", self.learner_input_sha256)
        if type(self.budgets) is not PostfitTeachingControlBudgets:
            raise TypeError("teaching budgets must be exact")
        if len(self.complete_candidate_answer_rows) != _ANSWER_MAP_COUNT:
            raise ValueError("teaching control must bind all 23 postfit answers")
        if tuple(sorted(self.complete_candidate_answer_rows)) != self.complete_candidate_answer_rows:
            raise ValueError("candidate answer rows must use canonical request order")
        if len({digest for digest, _ in self.complete_candidate_answer_rows}) != _ANSWER_MAP_COUNT:
            raise ValueError("candidate answer rows contain duplicate requests")
        for digest, answers in self.complete_candidate_answer_rows:
            _require_sha256("candidate answer request", digest)
            _active._validate_answers("candidate answer", answers)
        if _require_sha256(
            "complete_candidate_answer_map_sha256",
            self.complete_candidate_answer_map_sha256,
        ) != _sha256(self.complete_candidate_answer_rows):
            raise ValueError("complete candidate-answer digest mismatch")
        _require_sha256(
            "primary_reconstruction_result_sha256",
            self.primary_reconstruction_result_sha256,
        )
        if len(self.teaching_decisions) != _DECISION_COUNT or tuple(
            row.ordinal for row in self.teaching_decisions
        ) != tuple(range(1, _DECISION_COUNT + 1)):
            raise ValueError("teaching trace must contain 15 contiguous decisions")
        if len({row.request.request_sha256 for row in self.teaching_decisions}) != _DECISION_COUNT:
            raise ValueError("teaching trace cannot acquire one request twice")
        queried = tuple(
            sorted(
                row.request.request_sha256
                for row in self.teaching_decisions
                if row.acquisition_kind == "queried"
            )
        )
        inferred = tuple(
            sorted(
                row.request.request_sha256
                for row in self.teaching_decisions
                if row.acquisition_kind == "singleton_inferred"
            )
        )
        if self.queried_request_sha256s != queried or len(queried) != _QUERY_COUNT:
            raise ValueError("teaching queried partition mismatch")
        if self.inferred_request_sha256s != inferred or len(inferred) != _INFERENCE_COUNT:
            raise ValueError("teaching inferred partition mismatch")
        answer_requests = {digest for digest, _ in self.complete_candidate_answer_rows}
        if len(self.unopened_request_sha256s) != _UNOPENED_COUNT or tuple(
            sorted(self.unopened_request_sha256s)
        ) != self.unopened_request_sha256s:
            raise ValueError("teaching unopened partition mismatch")
        if set(queried) | set(inferred) | set(self.unopened_request_sha256s) != answer_requests:
            raise ValueError("teaching partitions do not cover the candidate answer map")
        if set(queried).intersection(inferred) or set(queried).intersection(
            self.unopened_request_sha256s
        ) or set(inferred).intersection(self.unopened_request_sha256s):
            raise ValueError("teaching candidate partitions overlap")
        answer_map = dict(self.complete_candidate_answer_rows)
        if any(answer_map[row.request.request_sha256] != row.target_answers for row in self.teaching_decisions):
            raise ValueError("teaching decision target differs from postfit truth")
        if (
            self.query_count != _QUERY_COUNT
            or self.inference_count != _INFERENCE_COUNT
            or self.unopened_count != _UNOPENED_COUNT
            or self.returned_categorical_label_count != _RETURNED_LABEL_COUNT
        ):
            raise ValueError("teaching 13+2+8 accounting mismatch")
        inference_kinds = tuple(
            sorted(
                row.inference_kind
                for row in self.teaching_decisions
                if row.inference_kind is not None
            )
        )
        if inference_kinds != (
            "categorical_restricted_map_singleton_rank_completion",
            "full_product_source_bijection_singleton",
        ):
            raise ValueError("teaching control must contain the two declared singleton kinds")
        if len(self.final_event_version_rows) != _EVENT_COUNT or any(
            count != 1 for _, count, _ in self.final_event_version_rows
        ):
            raise ValueError("all ten categorical restricted maps must close")
        if tuple(sorted(self.final_event_version_rows)) != self.final_event_version_rows or len(
            {token for token, _, _ in self.final_event_version_rows}
        ) != _EVENT_COUNT:
            raise ValueError("final event-version rows must be canonical and unique")
        for token, _, digest in self.final_event_version_rows:
            _active._require_token("final event token", token)
            _require_sha256("final event version digest", digest)
        if len(self.final_event_rank_rows) != _EVENT_COUNT:
            raise ValueError("final teaching control must bind ten event-rank rows")
        for token, observed, legal in self.final_event_rank_rows:
            _active._require_token("final event-rank token", token)
            if _plain_int("final observed rank", observed, 1) != _plain_int(
                "final legal rank", legal, 1
            ):
                raise ValueError("every final observed source rank must close its legal domain")
        if tuple(token for token, _, _ in self.final_event_rank_rows) != tuple(
            token for token, _, _ in self.final_event_version_rows
        ):
            raise ValueError("final rank/version rows use inconsistent event order")
        if (
            self.counterfactual_membership_query_count != _QUERY_COUNT
            or self.counterfactual_returned_categorical_label_count
            != _RETURNED_LABEL_COUNT
            or self.new_membership_calls_made != 0
            or self.aggregate_independent_rank_gains != _DECISION_COUNT
            or self.aggregate_image_coordinate_constraints
            != _DECISION_COUNT * 5
        ):
            raise ValueError("counterfactual query/rank accounting mismatch")
        for name in (
            "truth_score_rows_evaluated",
            "exact_rank_evaluations",
            "version_space_recomputations",
            "conditional_assignment_blocks_evaluated",
            "basis_image_candidates_planned",
        ):
            _plain_int(name, getattr(self, name), 1)
        if self.truth_score_rows_evaluated > self.budgets.max_truth_score_rows:
            raise ValueError("truth-score work exceeds its bound")
        if self.exact_rank_evaluations > self.budgets.max_exact_rank_evaluations:
            raise ValueError("rank work exceeds its bound")
        if self.version_space_recomputations > self.budgets.max_version_space_recomputations:
            raise ValueError("version-space work exceeds its bound")
        if self.conditional_assignment_blocks_evaluated > self.budgets.max_conditional_assignment_blocks:
            raise ValueError("conditional-assignment work exceeds its bound")
        if self.basis_image_candidates_planned > self.budgets.max_cumulative_basis_image_candidates:
            raise ValueError("basis-image planning work exceeds its bound")
        if self.closure_first_achieved_after_query_count != _QUERY_COUNT:
            raise ValueError("truth-specific closure must first occur after query 13")
        if self.event_closed_rank_completion_inference_count != 1:
            raise ValueError("exactly one inference must complete an already closed event rank")
        if len(self.primary_queried_request_sha256s) != 14 or len(
            self.primary_inferred_request_sha256s
        ) != 1:
            raise ValueError("primary reconstruction must retain the 14Q+1I trace")
        if self.primary_request_sha256s != tuple(
            row
            for row in self.primary_request_sha256s
        ) or len(self.primary_request_sha256s) != 15:
            raise ValueError("primary reconstruction must bind 15 decisions")
        if (
            len(set(self.primary_request_sha256s)) != 15
            or set(self.primary_queried_request_sha256s).intersection(
                self.primary_inferred_request_sha256s
            )
            or set(self.primary_queried_request_sha256s)
            | set(self.primary_inferred_request_sha256s)
            != set(self.primary_request_sha256s)
        ):
            raise ValueError("primary query/inference partitions are inconsistent")
        if self.teaching_trace_differs_from_primary_trace != (
            tuple(row.request.request_sha256 for row in self.teaching_decisions)
            != self.primary_request_sha256s
        ):
            raise ValueError("primary/teaching trace-difference flag mismatch")
        for name, value, required in (
            ("primary_reconstructed_before_teaching_search", self.primary_reconstructed_before_teaching_search, True),
            ("teaching_trace_differs_from_primary_trace", self.teaching_trace_differs_from_primary_trace, True),
            ("truth_aware_postfit_control", self.truth_aware_postfit_control, True),
            ("truth_specific_noncausal_control", self.truth_specific_noncausal_control, True),
            ("complete_answer_map_available_before_teaching_selection", self.complete_answer_map_available_before_teaching_selection, True),
            ("query_count_is_counterfactual", self.query_count_is_counterfactual, True),
            ("primary_selector_received_teaching_choices", self.primary_selector_received_teaching_choices, False),
            ("primary_result_unchanged_by_control", self.primary_result_unchanged_by_control, True),
            ("primary_posthoc_selector_flag_remains_false", self.primary_posthoc_selector_flag_remains_false, True),
            ("selection_eligible", self.selection_eligible, False),
            ("confirmatory_claim_eligible", self.confirmatory_claim_eligible, False),
            ("global_query_minimality_claimed", self.global_query_minimality_claimed, False),
            ("restricted_categorical_maps_closed", self.restricted_categorical_maps_closed, True),
            ("arbitrary_total_operator_constructed", self.arbitrary_total_operator_constructed, False),
        ):
            if _require_bool(name, value) is not required:
                raise ValueError(f"{name} must be {required}")
        if self.total_operator is not None:
            raise ValueError("postfit teaching control cannot serialize a total operator")
        if _require_sha256("result_sha256", self.result_sha256) != _sha256(
            self._payload(False)
        ):
            raise ValueError("postfit teaching-control digest mismatch")

    def _payload(self, include_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "learner_input_sha256": self.learner_input_sha256,
            "complete_candidate_answer_rows": self.complete_candidate_answer_rows,
            "complete_candidate_answer_map_sha256": self.complete_candidate_answer_map_sha256,
            "primary_reconstruction_result_sha256": self.primary_reconstruction_result_sha256,
            "primary_request_sha256s": self.primary_request_sha256s,
            "primary_queried_request_sha256s": self.primary_queried_request_sha256s,
            "primary_inferred_request_sha256s": self.primary_inferred_request_sha256s,
            "teaching_decisions": [row._payload(True) for row in self.teaching_decisions],
            "queried_request_sha256s": self.queried_request_sha256s,
            "inferred_request_sha256s": self.inferred_request_sha256s,
            "unopened_request_sha256s": self.unopened_request_sha256s,
            "final_event_version_rows": self.final_event_version_rows,
            "final_event_rank_rows": self.final_event_rank_rows,
            "query_count": self.query_count,
            "inference_count": self.inference_count,
            "unopened_count": self.unopened_count,
            "returned_categorical_label_count": self.returned_categorical_label_count,
            "counterfactual_membership_query_count": self.counterfactual_membership_query_count,
            "counterfactual_returned_categorical_label_count": self.counterfactual_returned_categorical_label_count,
            "new_membership_calls_made": self.new_membership_calls_made,
            "aggregate_independent_rank_gains": self.aggregate_independent_rank_gains,
            "aggregate_image_coordinate_constraints": self.aggregate_image_coordinate_constraints,
            "truth_score_rows_evaluated": self.truth_score_rows_evaluated,
            "exact_rank_evaluations": self.exact_rank_evaluations,
            "version_space_recomputations": self.version_space_recomputations,
            "conditional_assignment_blocks_evaluated": self.conditional_assignment_blocks_evaluated,
            "basis_image_candidates_planned": self.basis_image_candidates_planned,
            "closure_first_achieved_after_query_count": self.closure_first_achieved_after_query_count,
            "event_closed_rank_completion_inference_count": self.event_closed_rank_completion_inference_count,
            "primary_reconstructed_before_teaching_search": self.primary_reconstructed_before_teaching_search,
            "teaching_trace_differs_from_primary_trace": self.teaching_trace_differs_from_primary_trace,
            "truth_aware_postfit_control": self.truth_aware_postfit_control,
            "truth_specific_noncausal_control": self.truth_specific_noncausal_control,
            "complete_answer_map_available_before_teaching_selection": self.complete_answer_map_available_before_teaching_selection,
            "query_count_is_counterfactual": self.query_count_is_counterfactual,
            "primary_selector_received_teaching_choices": self.primary_selector_received_teaching_choices,
            "primary_result_unchanged_by_control": self.primary_result_unchanged_by_control,
            "primary_posthoc_selector_flag_remains_false": self.primary_posthoc_selector_flag_remains_false,
            "selection_eligible": self.selection_eligible,
            "confirmatory_claim_eligible": self.confirmatory_claim_eligible,
            "global_query_minimality_claimed": self.global_query_minimality_claimed,
            "restricted_categorical_maps_closed": self.restricted_categorical_maps_closed,
            "arbitrary_total_operator_constructed": self.arbitrary_total_operator_constructed,
            "total_operator": self.total_operator,
            "budgets": self.budgets.payload(),
        }
        if include_sha:
            payload["result_sha256"] = self.result_sha256
        return payload

    def payload(self) -> dict[str, object]:
        return self._payload(True)


def _canonical_answer_rows(
    learner_input: _active.OpaqueActiveLearnerInput,
    candidate_answer_map: Mapping[str, tuple[str, ...]],
    budgets: PostfitTeachingControlBudgets,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if type(candidate_answer_map) is not dict:
        raise TypeError("candidate_answer_map must be an exact primitive dict")
    if budgets.max_candidate_answer_rows < _ANSWER_MAP_COUNT:
        raise _active.OpaqueActiveDiscoveryLimitError(
            "postfit candidate-answer budget is below 23 before map access"
        )
    expected = {
        row.request_sha256 for row in learner_input.canonical_candidate_requests
    }
    if set(candidate_answer_map) != expected or len(candidate_answer_map) != _ANSWER_MAP_COUNT:
        raise ValueError("postfit answer map must exactly cover all 23 candidates")
    rows: list[tuple[str, tuple[str, ...]]] = []
    for digest in sorted(expected):
        _require_sha256("candidate answer key", digest)
        answers = candidate_answer_map[digest]
        if type(answers) is not tuple:
            raise TypeError("candidate answer values must be exact tuples")
        _active._validate_answers(
            "candidate answer", answers, learner_input.answer_tokens
        )
        rows.append((digest, answers))
    return tuple(rows)


def _rank_change(
    learner_input: _active.OpaqueActiveLearnerInput,
    edges: Sequence[_active._KnownEdge],
    request: OpaqueEdgeRequest,
    source_answers: tuple[str, ...],
) -> tuple[int, int]:
    source_rows, _ = _active._observed_event_rows(
        learner_input, edges, request.event_token
    )
    before = _active._rank(source_rows)
    after = _active._rank(
        source_rows
        + (_active._diagnostic_row(source_answers, learner_input.answer_tokens),)
    )
    return before, after


def discover_postfit_teaching_control(
    learner_input: _active.OpaqueActiveLearnerInput,
    candidate_answer_map: Mapping[str, tuple[str, ...]],
    *,
    budgets: PostfitTeachingControlBudgets | None = None,
) -> PostfitTeachingControlResult:
    """Build the deterministic truth-specific 13Q+2I postfit control."""

    if type(learner_input) is not _active.OpaqueActiveLearnerInput:
        raise TypeError("learner_input must be exact OpaqueActiveLearnerInput")
    if not learner_input.candidate_pool_complete or len(
        learner_input.canonical_candidate_requests
    ) != _ANSWER_MAP_COUNT:
        raise ValueError("teaching control requires the complete 23-request omission pool")
    selected_budgets = PostfitTeachingControlBudgets() if budgets is None else budgets
    if type(selected_budgets) is not PostfitTeachingControlBudgets:
        raise TypeError("budgets must be exact PostfitTeachingControlBudgets")
    requirements = {
        "max_decisions": _DECISION_COUNT,
        "max_queries": _QUERY_COUNT,
        "max_inferences": _INFERENCE_COUNT,
        "max_truth_score_rows": sum(
            _ANSWER_MAP_COUNT - index for index in range(_DECISION_COUNT)
        ),
        "max_exact_rank_evaluations": 2
        * sum(_ANSWER_MAP_COUNT - index for index in range(_DECISION_COUNT)),
        "max_version_space_recomputations": 17,
        "max_conditional_assignment_blocks": 32,
        "max_cumulative_basis_image_candidates": 3_000_000,
        "max_certificate_bytes": 128_000,
    }
    for name, required in requirements.items():
        if getattr(selected_budgets, name) < required:
            raise _active.OpaqueActiveDiscoveryLimitError(
                f"{name} is below the analytic postfit-control requirement"
            )
    answer_rows = _canonical_answer_rows(
        learner_input, candidate_answer_map, selected_budgets
    )
    truth = dict(answer_rows)

    # Reconstruct the causal primary before any teaching-set choice is made.
    primary = _active.run_opaque_active_discovery(
        learner_input,
        lambda choice: truth[choice.chosen_request.request_sha256],
    )
    if type(primary) is not _active.AutonomousPartialOperatorResult:
        raise ValueError("postfit truth does not reconstruct an identified primary arm")
    primary_steps = primary.final_state.steps
    if primary.active_call_count != 14 or primary.structural_inference_count != 1:
        raise ValueError("teaching control requires the frozen 14Q+1I primary trace")
    primary_requests = tuple(row.choice.chosen_request.request_sha256 for row in primary_steps)
    primary_queried = tuple(
        row.choice.chosen_request.request_sha256
        for row in primary_steps
        if type(row) is _active.OpaqueActiveStep
    )
    primary_inferred = tuple(
        row.choice.chosen_request.request_sha256
        for row in primary_steps
        if type(row) is _active.OpaqueStructuralInferenceStep
    )

    known, initial_edges = _active._known_material(learner_input, ())
    edges: list[_active._KnownEdge] = list(initial_edges)
    acquired: set[str] = set()
    decisions: list[PostfitTeachingDecision] = []
    mask_words = set(_active._mask_source_words(learner_input))
    pool = learner_input.canonical_candidate_requests
    query_count = 0
    inference_count = 0
    truth_score_work = 0
    rank_work = 0
    version_work = 0
    assignment_block_work = 0
    basis_candidate_work = 0
    closure_query_count: int | None = None

    while True:
        assignments = _active._source_assignments(learner_input, known)
        planned_basis_rows = _active._planned_basis_image_candidates(
            learner_input, known, assignments
        )
        if (
            len(assignments)
            > learner_input.budgets.max_conditional_assignment_blocks_per_choice
            or sum(planned_basis_rows)
            > learner_input.budgets.max_basis_image_candidates_per_choice
            or max(planned_basis_rows, default=0)
            > learner_input.budgets.max_materialized_versions_per_assignment
        ):
            raise _active.OpaqueActiveDiscoveryLimitError(
                "teaching exact-version work exceeds the learner-input analytic ceiling"
            )
        if (
            assignment_block_work + len(assignments)
            > selected_budgets.max_conditional_assignment_blocks
            or basis_candidate_work + sum(planned_basis_rows)
            > selected_budgets.max_cumulative_basis_image_candidates
        ):
            raise _active.OpaqueActiveDiscoveryLimitError(
                "teaching cumulative exact-version work ceiling would be exceeded"
            )
        assignment_block_work += len(assignments)
        basis_candidate_work += sum(planned_basis_rows)
        source_incomplete = any(word not in known for word in mask_words)
        scores: list[
            tuple[
                PostfitTruthScore,
                OpaqueEdgeRequest,
                tuple[str, ...],
                int,
                int,
                int,
                int,
                int,
                int,
            ]
        ] = []
        singleton_rows: list[
            tuple[
                OpaqueEdgeRequest,
                tuple[str, ...],
                int,
                int,
                int,
                int,
                int,
                int,
            ]
        ] = []
        version_work += 1
        if version_work > selected_budgets.max_version_space_recomputations:
            raise _active.OpaqueActiveDiscoveryLimitError(
                "postfit version-space recomputation ceiling exceeded"
            )
        if source_incomplete:
            blocks = _active._conditional_global_blocks(
                learner_input, known, edges, assignments
            )
            global_before = sum(row.global_version_mass for row in blocks)
            for request in pool:
                if (
                    request.request_sha256 in acquired
                    or request.source_word not in known
                    or request.program not in mask_words
                    or request.program in known
                ):
                    continue
                before_rank, after_rank = _rank_change(
                    learner_input, edges, request, known[request.source_word]
                )
                rank_work += 2
                if after_rank != before_rank + 1:
                    continue
                partitions: dict[
                    tuple[str, ...], list[_active._ConditionalGlobalVersionBlock]
                ] = {}
                for block in blocks:
                    partitions.setdefault(
                        dict(block.source_assignment)[request.program], []
                    ).append(block)
                selected_event_before = sum(
                    block.event_count(request.event_token) for block in blocks
                )
                if len(partitions) == 1:
                    target, selected = next(iter(partitions.items()))
                    singleton_rows.append(
                        (
                            request,
                            target,
                            len(blocks),
                            len(selected),
                            selected_event_before,
                            sum(
                                row.event_count(request.event_token)
                                for row in selected
                            ),
                            global_before,
                            sum(row.global_version_mass for row in selected),
                        )
                    )
                    continue
                target = truth[request.request_sha256]
                selected = partitions.get(target)
                if not selected:
                    raise ValueError("postfit truth is outside a source-assignment branch")
                score = _make_truth_score(
                    request,
                    global_mass_after=sum(
                        row.global_version_mass for row in selected
                    ),
                    event_versions_after=sum(
                        row.event_count(request.event_token) for row in selected
                    ),
                    source_assignments_after=len(selected),
                    rank_before=before_rank,
                    rank_after=after_rank,
                    source_frontier_phase=True,
                )
                scores.append(
                    (
                        score,
                        request,
                        target,
                        len(blocks),
                        len(selected),
                        selected_event_before,
                        score.actual_selected_event_version_count,
                        global_before,
                        score.actual_posterior_global_version_mass,
                    )
                )
        else:
            versions_by_event = _active._filtered_versions(
                learner_input, known, edges
            )
            rank_closed = all(
                _active._rank(
                    _active._observed_event_rows(
                        learner_input, edges, token
                    )[0]
                )
                == len(rows[0].domain_basis_image_rows)
                for token, rows in versions_by_event.items()
            )
            if all(
                len(rows) == 1 for rows in versions_by_event.values()
            ) and rank_closed:
                closure_query_count = query_count
                break
            global_before = 1
            for rows in versions_by_event.values():
                global_before *= len(rows)
            for request in pool:
                if request.request_sha256 in acquired or request.source_word not in known:
                    continue
                event_versions = versions_by_event[request.event_token]
                request_index = event_versions[0].legal_request_sha256s.index(
                    request.request_sha256
                )
                partitions: dict[
                    tuple[str, ...], list[_active.ExactCategoricalRestrictedMapVersion]
                ] = {}
                for version in event_versions:
                    partitions.setdefault(
                        version.predicted_target_answers[request_index], []
                    ).append(version)
                before_rank, after_rank = _rank_change(
                    learner_input, edges, request, known[request.source_word]
                )
                rank_work += 2
                if after_rank != before_rank + 1:
                    continue
                if len(partitions) == 1:
                    target, selected = next(iter(partitions.items()))
                    singleton_rows.append(
                        (
                            request,
                            target,
                            1,
                            1,
                            len(event_versions),
                            len(selected),
                            global_before,
                            len(selected)
                            * (global_before // len(event_versions)),
                        )
                    )
                    continue
                target = truth[request.request_sha256]
                selected = partitions.get(target)
                if not selected:
                    raise ValueError("postfit truth is outside a categorical-map branch")
                score = _make_truth_score(
                    request,
                    global_mass_after=len(selected)
                    * (global_before // len(event_versions)),
                    event_versions_after=len(selected),
                    source_assignments_after=1,
                    rank_before=before_rank,
                    rank_after=after_rank,
                    source_frontier_phase=False,
                )
                scores.append(
                    (
                        score,
                        request,
                        target,
                        1,
                        1,
                        len(event_versions),
                        len(selected),
                        global_before,
                        score.actual_posterior_global_version_mass,
                    )
                )
        pending_count = len(pool) - len(acquired)
        if truth_score_work + pending_count > selected_budgets.max_truth_score_rows:
            raise _active.OpaqueActiveDiscoveryLimitError(
                "postfit truth-score ceiling would be crossed"
            )
        if rank_work > selected_budgets.max_exact_rank_evaluations:
            raise _active.OpaqueActiveDiscoveryLimitError(
                "postfit exact-rank ceiling exceeded"
            )
        if len(decisions) >= _DECISION_COUNT:
            raise ValueError("teaching search exceeded 15 decisions before closure")
        if singleton_rows:
            singleton_rows.sort(key=lambda row: row[0].request_sha256)
            (
                request,
                target,
                assignments_before,
                assignments_after,
                event_before,
                event_after,
                global_before,
                global_after,
            ) = singleton_rows[0]
            before_rank, after_rank = _rank_change(
                learner_input, edges, request, known[request.source_word]
            )
            singleton_shas = tuple(
                row[0].request_sha256 for row in singleton_rows
            )
            inference_kind = (
                "full_product_source_bijection_singleton"
                if source_incomplete
                else "categorical_restricted_map_singleton_rank_completion"
            )
            if target != truth[request.request_sha256]:
                raise ValueError("singleton inference disagrees with postfit truth")
            inference_count += 1
            decision = _make_decision(
                ordinal=len(decisions) + 1,
                request=request,
                acquisition_kind="singleton_inferred",
                inference_kind=inference_kind,
                target_answers=target,
                source_assignment_count_before=assignments_before,
                source_assignment_count_after=assignments_after,
                selected_event_version_count_before=event_before,
                selected_event_version_count_after=event_after,
                global_version_mass_before=global_before,
                global_version_mass_after=global_after,
                observed_source_rank_before=before_rank,
                observed_source_rank_after=after_rank,
                eligible_truth_scores=(),
                eligible_singleton_request_sha256s=singleton_shas,
                complete_truth_map_used_to_select_request=False,
                complete_truth_map_used_to_supply_target=False,
                inference_value_derived_without_truth_lookup=True,
                inference_value_verified_against_postfit_truth=True,
                returned_categorical_label_count=0,
                restricted_maps_closed_after_decision=(
                    assignments_after == global_after == 1
                ),
                selected_event_map_already_closed_before_inference=(
                    not source_incomplete and event_before == event_after == 1
                ),
            )
        else:
            if not scores:
                raise ValueError("truth-aware teaching search has no eligible request")
            truth_score_work += len(scores)
            scores.sort(
                key=(
                    (lambda row: (
                        row[0].actual_source_assignment_count,
                        row[0].request_sha256,
                    ))
                    if source_incomplete
                    else (lambda row: (
                        row[0].actual_posterior_global_version_mass,
                        row[0].actual_selected_event_version_count,
                        row[0].request_sha256,
                    ))
                )
            )
            (
                _,
                request,
                target,
                assignments_before,
                assignments_after,
                event_before,
                event_after,
                global_before,
                global_after,
            ) = scores[0]
            query_count += 1
            if query_count > selected_budgets.max_queries:
                raise ValueError("truth-specific teaching search did not close in 13 queries")
            decision = _make_decision(
                ordinal=len(decisions) + 1,
                request=request,
                acquisition_kind="queried",
                inference_kind=None,
                target_answers=target,
                source_assignment_count_before=assignments_before,
                source_assignment_count_after=assignments_after,
                selected_event_version_count_before=event_before,
                selected_event_version_count_after=event_after,
                global_version_mass_before=global_before,
                global_version_mass_after=global_after,
                observed_source_rank_before=scores[0][0].observed_source_rank_before,
                observed_source_rank_after=scores[0][0].observed_source_rank_after,
                eligible_truth_scores=tuple(row[0] for row in scores),
                eligible_singleton_request_sha256s=(),
                complete_truth_map_used_to_select_request=True,
                complete_truth_map_used_to_supply_target=True,
                inference_value_derived_without_truth_lookup=False,
                inference_value_verified_against_postfit_truth=False,
                returned_categorical_label_count=2,
                restricted_maps_closed_after_decision=(
                    assignments_after == global_after == 1
                ),
                selected_event_map_already_closed_before_inference=False,
            )
        decisions.append(decision)
        edges.append(
            _active._KnownEdge(
                request=request,
                source_answers=known[request.source_word],
                target_answers=target,
            )
        )
        previous = known.setdefault(request.program, target)
        if previous != target:
            raise ValueError("teaching decision contradicts a known word diagnostic")
        acquired.add(request.request_sha256)

    if closure_query_count != _QUERY_COUNT or inference_count != _INFERENCE_COUNT:
        raise ValueError("truth-specific search did not realize 13Q+2I at first closure")
    final_versions = _active._filtered_versions(learner_input, known, edges)
    version_work += 1
    if version_work > selected_budgets.max_version_space_recomputations:
        raise _active.OpaqueActiveDiscoveryLimitError(
            "final teaching version-space ceiling exceeded"
        )
    if any(len(rows) != 1 for rows in final_versions.values()):
        raise ValueError("final 13Q+2I teaching set does not close every event")
    if query_count != _QUERY_COUNT or inference_count != _INFERENCE_COUNT:
        raise ValueError("teaching control did not realize exactly 13Q+2I")

    queried = tuple(
        sorted(
            row.request.request_sha256
            for row in decisions
            if row.acquisition_kind == "queried"
        )
    )
    inferred = tuple(
        sorted(
            row.request.request_sha256
            for row in decisions
            if row.acquisition_kind == "singleton_inferred"
        )
    )
    unopened = tuple(
        sorted(
            row.request_sha256
            for row in pool
            if row.request_sha256 not in acquired
        )
    )
    version_rows = tuple(
        (
            token,
            len(rows),
            rows[0].version_sha256,
        )
        for token, rows in sorted(final_versions.items())
    )
    rank_rows = tuple(
        (
            token,
            _active._rank(
                _active._observed_event_rows(learner_input, edges, token)[0]
            ),
            len(rows[0].domain_basis_image_rows),
        )
        for token, rows in sorted(final_versions.items())
    )
    kwargs = {
        "learner_input_sha256": learner_input.input_sha256,
        "complete_candidate_answer_rows": answer_rows,
        "complete_candidate_answer_map_sha256": _sha256(answer_rows),
        "primary_reconstruction_result_sha256": primary.result_sha256,
        "primary_request_sha256s": primary_requests,
        "primary_queried_request_sha256s": primary_queried,
        "primary_inferred_request_sha256s": primary_inferred,
        "teaching_decisions": tuple(decisions),
        "queried_request_sha256s": queried,
        "inferred_request_sha256s": inferred,
        "unopened_request_sha256s": unopened,
        "final_event_version_rows": version_rows,
        "final_event_rank_rows": rank_rows,
        "query_count": query_count,
        "inference_count": inference_count,
        "unopened_count": len(unopened),
        "returned_categorical_label_count": 2 * query_count,
        "counterfactual_membership_query_count": query_count,
        "counterfactual_returned_categorical_label_count": 2 * query_count,
        "new_membership_calls_made": 0,
        "aggregate_independent_rank_gains": sum(
            row.observed_source_rank_after - row.observed_source_rank_before
            for row in decisions
        ),
        "aggregate_image_coordinate_constraints": 5 * len(decisions),
        "truth_score_rows_evaluated": truth_score_work,
        "exact_rank_evaluations": rank_work,
        "version_space_recomputations": version_work,
        "conditional_assignment_blocks_evaluated": assignment_block_work,
        "basis_image_candidates_planned": basis_candidate_work,
        "closure_first_achieved_after_query_count": closure_query_count,
        "event_closed_rank_completion_inference_count": 1,
        "primary_reconstructed_before_teaching_search": True,
        "teaching_trace_differs_from_primary_trace": tuple(
            row.request.request_sha256 for row in decisions
        )
        != primary_requests,
        "truth_aware_postfit_control": True,
        "truth_specific_noncausal_control": True,
        "complete_answer_map_available_before_teaching_selection": True,
        "query_count_is_counterfactual": True,
        "primary_selector_received_teaching_choices": False,
        "primary_result_unchanged_by_control": True,
        "primary_posthoc_selector_flag_remains_false": (
            not primary.posthoc_truth_specific_13_query_teaching_set_used_by_selector
        ),
        "selection_eligible": False,
        "confirmatory_claim_eligible": False,
        "global_query_minimality_claimed": False,
        "restricted_categorical_maps_closed": True,
        "arbitrary_total_operator_constructed": False,
        "total_operator": None,
        "budgets": selected_budgets,
        "schema": _RESULT_SCHEMA,
    }
    payload = {"schema": _RESULT_SCHEMA, **{key: value for key, value in kwargs.items() if key != "schema"}}
    payload["teaching_decisions"] = [row._payload(True) for row in decisions]
    payload["budgets"] = selected_budgets.payload()
    result = PostfitTeachingControlResult(
        **kwargs,
        result_sha256=_sha256(payload),
    )
    if len(_canonical_bytes(result.payload())) > selected_budgets.max_certificate_bytes:
        raise _active.OpaqueActiveDiscoveryLimitError(
            "postfit teaching certificate exceeds its byte budget"
        )
    return result


def validate_postfit_teaching_control(
    learner_input: _active.OpaqueActiveLearnerInput,
    candidate_answer_map: Mapping[str, tuple[str, ...]],
    result: PostfitTeachingControlResult,
) -> None:
    """Authoritatively reconstruct the answer-dependent control certificate."""

    if type(result) is not PostfitTeachingControlResult:
        raise TypeError("result must be exact PostfitTeachingControlResult")
    reconstructed = discover_postfit_teaching_control(
        learner_input,
        candidate_answer_map,
        budgets=result.budgets,
    )
    if reconstructed != result:
        raise ValueError("postfit teaching control fails authoritative reconstruction")


__all__ = (
    "PostfitTeachingControlBudgets",
    "PostfitTeachingControlResult",
    "PostfitTeachingDecision",
    "PostfitTruthScore",
    "discover_postfit_teaching_control",
    "validate_postfit_teaching_control",
)
