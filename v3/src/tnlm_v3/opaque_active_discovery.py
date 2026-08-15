"""Opaque, response-adaptive discovery of guarded partial operators.

This Phase-III-T2 prototype removes the semantically ordered acquisition basis
used by :mod:`tnlm_v3.opaque_partial_operators`.  Its pure learner receives an
unordered set of opaque legal-edge requests, an opaque diagnostic table, and
the learner-visible 9x10 definedness mask.  It canonicalizes the request set by
content hash and commits each request *before* the corresponding two-label
membership response is released.

The learned object is deliberately partial.  An event operator is represented
only by its exact action on the span of its legal source states.  Illegal words
remain undefined; no zero/dead encoding and no arbitrary total 5x5 extension
is constructed.  Once all restricted maps close, a finite guarded-transition
certificate establishes behavior for arbitrary-length suffixes by induction,
provided every traversed edge is learner-mask-defined.  This is not a total
WFA and not assumption-free representation discovery: the two-query full-
product diagnostic gauge and definedness mask are supplied supervision.

Only opaque learner types are imported from the frozen T1 module.  The pure
selection/fitting call graph contains no controller builders, semantic action
tables, canonical states, omitted-cell identifiers, nonces, or T1 candidate
order.  A caller/controller may answer a committed request, but is outside the
learner boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import gcd
import hashlib
import json
from typing import Callable, Iterable, Sequence

from .opaque_partial_operators import (
    OpaqueEdgeObservation,
    OpaqueEdgeRequest,
    OpaqueStateObservation,
    PartialOperatorLearnerInput,
    Rational,
)


OpaqueWord = tuple[str, ...]
RationalVector = tuple[Rational, ...]
RationalMatrix = tuple[RationalVector, ...]

_TOKEN_LENGTH = 32
_STATE_COUNT = 9
_EVENT_COUNT = 10
_QUERY_COUNT = 2
_ANSWER_COUNT = 3
_LEGAL_EDGE_COUNT = 44
_UNDEFINED_EDGE_COUNT = 46
_OMISSION_PASSIVE_EDGE_COUNT = 21
_OMISSION_CANDIDATE_COUNT = 23
_EXPECTED_INDEPENDENT_ACQUISITIONS = 15
_EXPECTED_ACTIVE_CALLS = 14
_EXPECTED_STRUCTURAL_INFERENCES = 1
_EXPECTED_RETURNED_LABELS = 28
_FROZEN_RESULT_CERTIFICATE_UPPER_BOUND = 8_000_000
_AMBIENT_RANK = 5

_INPUT_SCHEMA = "tnlm-v3-opaque-autonomous-active-input-v1"
_RESPONSE_SCHEMA = "tnlm-v3-opaque-autonomous-active-response-v1"
_INFERENCE_SCHEMA = "tnlm-v3-opaque-autonomous-structural-inference-v1"
_BRANCH_SCHEMA = "tnlm-v3-opaque-exact-outcome-branch-v1"
_SCORE_SCHEMA = "tnlm-v3-opaque-active-candidate-score-v1"
_CHOICE_SCHEMA = "tnlm-v3-opaque-active-choice-v1"
_STEP_SCHEMA = "tnlm-v3-opaque-active-step-v1"
_STATE_SCHEMA = "tnlm-v3-opaque-active-state-v1"
_OPERATOR_SCHEMA = "tnlm-v3-opaque-autonomous-restricted-operator-v1"
_MODEL_SCHEMA = "tnlm-v3-opaque-autonomous-partial-model-v1"
_LANGUAGE_SCHEMA = "tnlm-v3-opaque-finite-guarded-language-v1"
_RESULT_SCHEMA = "tnlm-v3-opaque-autonomous-partial-result-v1"
_NOT_IDENTIFIED_SCHEMA = "tnlm-v3-opaque-autonomous-not-identified-v1"


class OpaqueActiveDiscoveryLimitError(RuntimeError):
    """Raised before a frozen T2 analytic work ceiling is crossed."""


class OpaqueActiveCandidatePoolExhaustedError(RuntimeError):
    """Raised when a declared negative-control pool cannot close the versions."""


def _plain_int(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact boolean")
    return value


def _require_sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_token(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != _TOKEN_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a 128-bit lowercase opaque token")
    return value


def _jsonable(value: object) -> object:
    if isinstance(value, Rational):
        return [value.numerator, value.denominator]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if value is None or type(value) in (str, int, bool):
        return value
    raise TypeError(f"{type(value).__name__} is not canonically encodable")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _request_payload(request: OpaqueEdgeRequest) -> dict[str, object]:
    return {
        "schema": request.schema,
        "source_word": request.source_word,
        "event_token": request.event_token,
        "program": request.program,
        "request_sha256": request.request_sha256,
    }


def _observation_payload(observation: OpaqueEdgeObservation) -> dict[str, object]:
    return {
        "schema": observation.schema,
        "request": _request_payload(observation.request),
        "source_answers": observation.source_answers,
        "target_answers": observation.target_answers,
        "observation_sha256": observation.observation_sha256,
    }


def _state_payload(observation: OpaqueStateObservation) -> dict[str, object]:
    return {"word": observation.word, "answers": observation.answers}


def _validate_answers(
    name: str,
    answers: object,
    answer_tokens: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if not isinstance(answers, tuple) or len(answers) != _QUERY_COUNT:
        raise ValueError(f"{name} must contain exactly two labels")
    for token in answers:
        _require_token(name, token)
    if answer_tokens is not None and any(token not in answer_tokens for token in answers):
        raise ValueError(f"{name} contains an undeclared answer token")
    return answers


def _as_fraction(value: Rational | Fraction | int) -> Fraction:
    if type(value) is Rational:
        return Fraction(value.numerator, value.denominator)
    if type(value) is Fraction:
        return value
    if type(value) is int:
        return Fraction(value)
    raise TypeError("exact arithmetic accepts only Rational, Fraction, or int")


def _rational(value: Rational | Fraction | int) -> Rational:
    fraction = _as_fraction(value)
    return Rational(fraction.numerator, fraction.denominator)


def _matrix(rows: Iterable[Iterable[Rational | Fraction | int]]) -> RationalMatrix:
    result = tuple(tuple(_rational(value) for value in row) for row in rows)
    if not result or not result[0]:
        raise ValueError("exact matrix must be nonempty")
    width = len(result[0])
    if any(len(row) != width for row in result):
        raise ValueError("exact matrix must be rectangular")
    return result


def _rank_profile(rows: Sequence[Sequence[Rational | Fraction | int]]) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    if not rows:
        return 0, (), ()
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("rank input must be a nonempty-width rectangle")
    work = [[_as_fraction(value) for value in row] for row in rows]
    source_rows = list(range(len(work)))
    pivot_rows: list[int] = []
    pivot_columns: list[int] = []
    rank = 0
    for column in range(width):
        pivot = next((row for row in range(rank, len(work)) if work[row][column]), None)
        if pivot is None:
            continue
        if pivot != rank:
            work[rank], work[pivot] = work[pivot], work[rank]
            source_rows[rank], source_rows[pivot] = source_rows[pivot], source_rows[rank]
        pivot_value = work[rank][column]
        work[rank] = [value / pivot_value for value in work[rank]]
        for row in range(len(work)):
            if row == rank or not work[row][column]:
                continue
            factor = work[row][column]
            work[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(work[row], work[rank], strict=True)
            ]
        pivot_rows.append(source_rows[rank])
        pivot_columns.append(column)
        rank += 1
        if rank == len(work):
            break
    return rank, tuple(pivot_rows), tuple(pivot_columns)


def _rank(rows: Sequence[Sequence[Rational | Fraction | int]]) -> int:
    return _rank_profile(rows)[0]


def _diagnostic_row(
    answers: tuple[str, ...], answer_tokens: tuple[str, ...]
) -> RationalVector:
    _validate_answers("diagnostic answers", answers, answer_tokens)
    baseline = min(answer_tokens)
    contrasts = tuple(token for token in sorted(answer_tokens) if token != baseline)
    return tuple(
        [Rational(1)]
        + [Rational(int(answers[0] == token)) for token in contrasts]
        + [Rational(int(answers[1] == token)) for token in contrasts]
    )


def _row_coordinates(
    row: Sequence[Rational | Fraction | int],
    basis_rows: Sequence[Sequence[Rational | Fraction | int]],
) -> tuple[Fraction, ...]:
    if not basis_rows:
        if any(_as_fraction(value) for value in row):
            raise ValueError("nonzero row is outside the empty span")
        return ()
    basis = [[_as_fraction(value) for value in values] for values in basis_rows]
    target = [_as_fraction(value) for value in row]
    dimension = len(basis)
    width = len(target)
    if any(len(values) != width for values in basis):
        raise ValueError("basis/row width mismatch")
    if _rank(basis) != dimension:
        raise ValueError("basis rows must be independent")
    transposed = [[basis[i][j] for i in range(dimension)] for j in range(width)]
    _, pivot_equations, _ = _rank_profile(transposed)
    if len(pivot_equations) != dimension:
        raise ValueError("basis coordinate system is singular")
    square = [[basis[i][column] for i in range(dimension)] for column in pivot_equations]
    rhs = [target[column] for column in pivot_equations]
    augmented = [square[i] + [rhs[i]] for i in range(dimension)]
    for column in range(dimension):
        pivot = next((row_index for row_index in range(column, dimension) if augmented[row_index][column]), None)
        if pivot is None:
            raise ValueError("basis coordinate system is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row_index in range(dimension):
            if row_index == column or not augmented[row_index][column]:
                continue
            factor = augmented[row_index][column]
            augmented[row_index] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(augmented[row_index], augmented[column], strict=True)
            ]
    coordinates = tuple(augmented[index][-1] for index in range(dimension))
    reconstructed = [
        sum(coordinates[i] * basis[i][j] for i in range(dimension))
        for j in range(width)
    ]
    if reconstructed != target:
        raise ValueError("row is outside the supplied span")
    return coordinates


def _map_is_consistent(
    source_rows: Sequence[Sequence[Rational | Fraction | int]],
    image_rows: Sequence[Sequence[Rational | Fraction | int]],
) -> bool:
    if len(source_rows) != len(image_rows):
        raise ValueError("source/image constraint counts differ")
    if not source_rows:
        return True
    _, pivots, _ = _rank_profile(source_rows)
    source_basis = [source_rows[index] for index in pivots]
    image_basis = [image_rows[index] for index in pivots]
    for source, image in zip(source_rows, image_rows, strict=True):
        coordinates = _row_coordinates(source, source_basis)
        reconstructed = tuple(
            sum(coordinates[i] * _as_fraction(image_basis[i][column]) for i in range(len(coordinates)))
            for column in range(len(image_basis[0]))
        )
        if reconstructed != tuple(_as_fraction(value) for value in image):
            return False
    return True


@dataclass(frozen=True)
class OpaqueActiveDiscoveryBudgets:
    max_word_length: int = 16
    max_candidate_requests: int = 23
    max_active_calls: int = 19
    max_structural_inferences: int = 4
    max_returned_categorical_tokens: int = 38
    max_outcome_branches_per_choice: int = 9
    max_candidate_score_rows: int = 437
    max_exact_rank_evaluations: int = 20_000
    max_conditional_assignment_blocks_per_choice: int = 6
    max_basis_image_candidates_per_choice: int = 700_000
    max_materialized_versions_per_assignment: int = 150_000
    max_validation_replay_decisions: int = 23
    max_suffix_events_per_prediction: int = 4_096
    max_certificate_bytes: int = 8_000_000

    def __post_init__(self) -> None:
        ceilings = {
            "max_word_length": 16,
            "max_candidate_requests": 23,
            "max_active_calls": 19,
            "max_structural_inferences": 4,
            "max_returned_categorical_tokens": 38,
            "max_outcome_branches_per_choice": 9,
            "max_candidate_score_rows": 437,
            "max_exact_rank_evaluations": 20_000,
            "max_conditional_assignment_blocks_per_choice": 6,
            "max_basis_image_candidates_per_choice": 700_000,
            "max_materialized_versions_per_assignment": 150_000,
            "max_validation_replay_decisions": 23,
            "max_suffix_events_per_prediction": 4_096,
            "max_certificate_bytes": 8_000_000,
        }
        for name, ceiling in ceilings.items():
            value = _plain_int(name, getattr(self, name), 1)
            if value > ceiling:
                raise ValueError(f"{name} exceeds the frozen T2 prototype ceiling")

    def payload(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class OpaqueActiveLearnerInput:
    event_tokens: tuple[str, ...]
    query_tokens: tuple[str, ...]
    answer_tokens: tuple[str, ...]
    passive_state_observations: tuple[OpaqueStateObservation, ...]
    passive_edge_observations: tuple[OpaqueEdgeObservation, ...]
    canonical_candidate_requests: tuple[OpaqueEdgeRequest, ...]
    canonical_defined_requests: tuple[OpaqueEdgeRequest, ...]
    canonical_undefined_requests: tuple[OpaqueEdgeRequest, ...]
    budgets: OpaqueActiveDiscoveryBudgets
    passive_table_sha256: str
    domain_mask_sha256: str
    canonical_candidate_set_sha256: str
    candidate_order_discarded: bool
    domain_mask_is_visible_supervision: bool
    full_product_diagnostic_gauge_is_visible_supervision: bool
    mask_source_representatives_bijectively_cover_full_product: bool
    source_bijection_counted_as_supervision: bool
    input_sha256: str
    schema: str = _INPUT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _INPUT_SCHEMA:
            raise ValueError("unknown autonomous active-input schema")
        if type(self.budgets) is not OpaqueActiveDiscoveryBudgets:
            raise TypeError("budgets must be exact OpaqueActiveDiscoveryBudgets")
        for name, values, expected in (
            ("event_tokens", self.event_tokens, _EVENT_COUNT),
            ("query_tokens", self.query_tokens, _QUERY_COUNT),
            ("answer_tokens", self.answer_tokens, _ANSWER_COUNT),
        ):
            if not isinstance(values, tuple) or len(values) != expected or len(set(values)) != expected:
                raise ValueError(f"{name} must contain exactly {expected} unique tokens")
            for token in values:
                _require_token(name, token)
        if len(set(self.event_tokens + self.query_tokens + self.answer_tokens)) != 15:
            raise ValueError("opaque vocabularies must be disjoint")
        for name, rows in (
            ("passive_state_observations", self.passive_state_observations),
            ("passive_edge_observations", self.passive_edge_observations),
            ("canonical_candidate_requests", self.canonical_candidate_requests),
            ("canonical_defined_requests", self.canonical_defined_requests),
            ("canonical_undefined_requests", self.canonical_undefined_requests),
        ):
            if not isinstance(rows, tuple):
                raise TypeError(f"{name} must be an exact tuple")
        if tuple(
            sorted(self.passive_state_observations, key=lambda row: _sha256(_state_payload(row)))
        ) != self.passive_state_observations:
            raise ValueError("passive state rows must use canonical opaque-content order")
        if tuple(
            sorted(
                self.passive_edge_observations,
                key=lambda row: row.observation_sha256,
            )
        ) != self.passive_edge_observations:
            raise ValueError("passive edge rows must use canonical observation-digest order")
        passive_count = len(self.passive_edge_observations)
        if passive_count not in (_OMISSION_PASSIVE_EDGE_COUNT, _LEGAL_EDGE_COUNT):
            raise ValueError("passive table must contain 21 omission or 44 control edges")
        full_candidate_count = _LEGAL_EDGE_COUNT - passive_count
        if full_candidate_count not in (0, _OMISSION_CANDIDATE_COUNT):
            raise ValueError("unsupported candidate-set size")
        supplied_candidate_count = len(self.canonical_candidate_requests)
        if supplied_candidate_count not in (
            full_candidate_count,
            max(0, full_candidate_count - 1),
        ):
            raise ValueError("candidate set must be complete or remove exactly one row")
        if full_candidate_count == 0 and supplied_candidate_count != 0:
            raise ValueError("full-support controls cannot contain candidates")
        if len(self.canonical_defined_requests) != _LEGAL_EDGE_COUNT:
            raise ValueError("defined mask must contain exactly 44 requests")
        if len(self.canonical_undefined_requests) != _UNDEFINED_EDGE_COUNT:
            raise ValueError("undefined mask must contain exactly 46 requests")
        if len(self.canonical_candidate_requests) > self.budgets.max_candidate_requests:
            raise OpaqueActiveDiscoveryLimitError("candidate-request budget exceeded")
        canonical_sets = (
            self.canonical_candidate_requests,
            self.canonical_defined_requests,
            self.canonical_undefined_requests,
        )
        for rows in canonical_sets:
            if tuple(sorted(rows, key=lambda row: row.request_sha256)) != rows:
                raise ValueError("request collections must use canonical SHA order")
            if len({row.request_sha256 for row in rows}) != len(rows):
                raise ValueError("request collection contains duplicates")
            for row in rows:
                if type(row) is not OpaqueEdgeRequest:
                    raise TypeError("request collections require exact OpaqueEdgeRequest rows")
                if row.event_token not in self.event_tokens:
                    raise ValueError("request contains an undeclared event token")
                if len(row.program) > self.budgets.max_word_length:
                    raise OpaqueActiveDiscoveryLimitError("word-length budget exceeded")
                if any(token not in self.event_tokens for token in row.source_word + row.program):
                    raise ValueError("request word contains a token outside the event alphabet")
        mask_pairs = {
            (row.source_word, row.event_token)
            for row in self.canonical_defined_requests + self.canonical_undefined_requests
        }
        mask_words = {word for word, _ in mask_pairs}
        if len(mask_words) != _STATE_COUNT or mask_pairs != {
            (word, event) for word in mask_words for event in self.event_tokens
        }:
            raise ValueError("definedness mask must be the exact opaque 9x10 grid")
        expected_passive_state_count = (
            _STATE_COUNT if passive_count == _LEGAL_EDGE_COUNT else 6
        )
        if len(self.passive_state_observations) != expected_passive_state_count:
            raise ValueError(
                "passive state table must contain six omission or nine control representatives"
            )
        passive_state_words = tuple(row.word for row in self.passive_state_observations)
        if len(set(passive_state_words)) != expected_passive_state_count:
            raise ValueError("passive state representative words must be distinct")
        if not set(passive_state_words).issubset(mask_words):
            raise ValueError("passive state words must be definedness-mask source representatives")
        defined_hashes = {row.request_sha256 for row in self.canonical_defined_requests}
        passive_hashes = {row.request.request_sha256 for row in self.passive_edge_observations}
        candidate_hashes = {row.request_sha256 for row in self.canonical_candidate_requests}
        if len(passive_hashes) != passive_count:
            raise ValueError("passive edge table contains duplicate requests")
        full_candidate_hashes = defined_hashes - passive_hashes
        if not candidate_hashes.issubset(full_candidate_hashes):
            raise ValueError("candidate pool contains a non-complement request")
        missing_candidate_hashes = tuple(sorted(full_candidate_hashes - candidate_hashes))
        if len(missing_candidate_hashes) > 1:
            raise ValueError("only one-row candidate-removal controls are supported")
        for digest in missing_candidate_hashes:
            _require_sha256("missing candidate request digest", digest)
        for row in self.passive_state_observations:
            if type(row) is not OpaqueStateObservation:
                raise TypeError("passive state rows must be exact OpaqueStateObservation")
            _validate_answers("passive state answers", row.answers, self.answer_tokens)
            if len(row.word) > self.budgets.max_word_length:
                raise OpaqueActiveDiscoveryLimitError("passive state word exceeds word budget")
            if any(token not in self.event_tokens for token in row.word):
                raise ValueError("passive state word contains an undeclared event token")
        known: dict[OpaqueWord, tuple[str, ...]] = {}
        for row in self.passive_state_observations:
            previous = known.setdefault(row.word, row.answers)
            if previous != row.answers:
                raise ValueError("passive word has contradictory diagnostics")
        for row in self.passive_edge_observations:
            if type(row) is not OpaqueEdgeObservation:
                raise TypeError("passive edge rows must be exact OpaqueEdgeObservation")
            if row.request.request_sha256 not in defined_hashes:
                raise ValueError("passive edge is absent from the defined mask")
            for word, answers in (
                (row.request.source_word, row.source_answers),
                (row.request.program, row.target_answers),
            ):
                _validate_answers("passive edge answers", answers, self.answer_tokens)
                previous = known.setdefault(word, answers)
                if previous != answers:
                    raise ValueError("passive word has contradictory diagnostics")
        for name, value, required in (
            ("candidate_order_discarded", self.candidate_order_discarded, True),
            ("domain_mask_is_visible_supervision", self.domain_mask_is_visible_supervision, True),
            ("full_product_diagnostic_gauge_is_visible_supervision", self.full_product_diagnostic_gauge_is_visible_supervision, True),
            ("mask_source_representatives_bijectively_cover_full_product", self.mask_source_representatives_bijectively_cover_full_product, True),
            ("source_bijection_counted_as_supervision", self.source_bijection_counted_as_supervision, True),
        ):
            if _require_bool(name, value) is not required:
                raise ValueError(f"{name} must be {required}")
        passive_payload = {
            "states": [_state_payload(row) for row in self.passive_state_observations],
            "edges": [_observation_payload(row) for row in self.passive_edge_observations],
        }
        domain_payload = {
            "defined": [_request_payload(row) for row in self.canonical_defined_requests],
            "undefined": [_request_payload(row) for row in self.canonical_undefined_requests],
        }
        candidate_payload = [_request_payload(row) for row in self.canonical_candidate_requests]
        if _require_sha256("passive_table_sha256", self.passive_table_sha256) != _sha256(passive_payload):
            raise ValueError("passive-table digest mismatch")
        if _require_sha256("domain_mask_sha256", self.domain_mask_sha256) != _sha256(domain_payload):
            raise ValueError("domain-mask digest mismatch")
        if _require_sha256("canonical_candidate_set_sha256", self.canonical_candidate_set_sha256) != _sha256(candidate_payload):
            raise ValueError("canonical candidate-set digest mismatch")
        if _require_sha256("input_sha256", self.input_sha256) != _sha256(self._payload(False)):
            raise ValueError("autonomous active-input digest mismatch")

    @property
    def is_full_support_control(self) -> bool:
        return not self.canonical_candidate_requests

    @property
    def missing_candidate_request_sha256s(self) -> tuple[str, ...]:
        defined = {row.request_sha256 for row in self.canonical_defined_requests}
        passive = {
            row.request.request_sha256 for row in self.passive_edge_observations
        }
        candidates = {
            row.request_sha256 for row in self.canonical_candidate_requests
        }
        return tuple(sorted(defined - passive - candidates))

    @property
    def candidate_pool_complete(self) -> bool:
        return not self.missing_candidate_request_sha256s

    @property
    def single_candidate_removal_negative_control(self) -> bool:
        return len(self.missing_candidate_request_sha256s) == 1

    def _payload(self, include_input_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "event_tokens": self.event_tokens,
            "query_tokens": self.query_tokens,
            "answer_tokens": self.answer_tokens,
            "passive_state_observations": [_state_payload(row) for row in self.passive_state_observations],
            "passive_edge_observations": [_observation_payload(row) for row in self.passive_edge_observations],
            "canonical_candidate_requests": [_request_payload(row) for row in self.canonical_candidate_requests],
            "canonical_defined_requests": [_request_payload(row) for row in self.canonical_defined_requests],
            "canonical_undefined_requests": [_request_payload(row) for row in self.canonical_undefined_requests],
            "budgets": self.budgets.payload(),
            "passive_table_sha256": self.passive_table_sha256,
            "domain_mask_sha256": self.domain_mask_sha256,
            "canonical_candidate_set_sha256": self.canonical_candidate_set_sha256,
            "candidate_order_discarded": self.candidate_order_discarded,
            "domain_mask_is_visible_supervision": self.domain_mask_is_visible_supervision,
            "full_product_diagnostic_gauge_is_visible_supervision": self.full_product_diagnostic_gauge_is_visible_supervision,
            "mask_source_representatives_bijectively_cover_full_product": self.mask_source_representatives_bijectively_cover_full_product,
            "source_bijection_counted_as_supervision": self.source_bijection_counted_as_supervision,
        }
        if include_input_sha:
            payload["input_sha256"] = self.input_sha256
        return payload

    def payload(self) -> dict[str, object]:
        return self._payload(True)


def make_opaque_active_input_from_rows(
    *,
    event_tokens: Sequence[str],
    query_tokens: Sequence[str],
    answer_tokens: Sequence[str],
    passive_state_observations: Sequence[OpaqueStateObservation],
    passive_edge_observations: Sequence[OpaqueEdgeObservation],
    candidate_requests: Sequence[OpaqueEdgeRequest],
    defined_requests: Sequence[OpaqueEdgeRequest],
    undefined_requests: Sequence[OpaqueEdgeRequest],
    budgets: OpaqueActiveDiscoveryBudgets | None = None,
    allow_incomplete_candidate_pool: bool = False,
) -> OpaqueActiveLearnerInput:
    """Construct the fresh T2 boundary from unordered primitive opaque rows.

    No upstream container, digest, row order, controller nonce, semantic role,
    or omission identifier is retained.  Every collection is canonicalized by
    its learner-visible content before the new T2 commitments are computed.
    """

    selected_budgets = OpaqueActiveDiscoveryBudgets() if budgets is None else budgets
    if type(selected_budgets) is not OpaqueActiveDiscoveryBudgets:
        raise TypeError("budgets must be exact OpaqueActiveDiscoveryBudgets")
    candidates = tuple(sorted(tuple(candidate_requests), key=lambda row: row.request_sha256))
    defined = tuple(sorted(tuple(defined_requests), key=lambda row: row.request_sha256))
    undefined = tuple(sorted(tuple(undefined_requests), key=lambda row: row.request_sha256))
    passive_states = tuple(
        sorted(
            tuple(passive_state_observations),
            key=lambda row: _sha256(_state_payload(row)),
        )
    )
    passive_edges = tuple(
        sorted(
            tuple(passive_edge_observations),
            key=lambda row: row.observation_sha256,
        )
    )
    passive_payload = {
        "states": [_state_payload(row) for row in passive_states],
        "edges": [_observation_payload(row) for row in passive_edges],
    }
    passive_request_hashes = {
        row.request.request_sha256 for row in passive_edges
    }
    defined_request_hashes = {row.request_sha256 for row in defined}
    candidate_request_hashes = {row.request_sha256 for row in candidates}
    missing_candidate_hashes = tuple(
        sorted(defined_request_hashes - passive_request_hashes - candidate_request_hashes)
    )
    if missing_candidate_hashes and not allow_incomplete_candidate_pool:
        raise ValueError(
            "an incomplete candidate pool requires the explicit negative-control constructor"
        )
    if allow_incomplete_candidate_pool and len(missing_candidate_hashes) != 1:
        raise ValueError("incomplete candidate pool must remove exactly one request")
    domain_payload = {
        "defined": [_request_payload(row) for row in defined],
        "undefined": [_request_payload(row) for row in undefined],
    }
    candidate_payload = [_request_payload(row) for row in candidates]
    kwargs: dict[str, object] = {
        "event_tokens": tuple(event_tokens),
        "query_tokens": tuple(query_tokens),
        "answer_tokens": tuple(answer_tokens),
        "passive_state_observations": passive_states,
        "passive_edge_observations": passive_edges,
        "canonical_candidate_requests": candidates,
        "canonical_defined_requests": defined,
        "canonical_undefined_requests": undefined,
        "budgets": selected_budgets,
        "passive_table_sha256": _sha256(passive_payload),
        "domain_mask_sha256": _sha256(domain_payload),
        "canonical_candidate_set_sha256": _sha256(candidate_payload),
        "candidate_order_discarded": True,
        "domain_mask_is_visible_supervision": True,
        "full_product_diagnostic_gauge_is_visible_supervision": True,
        "mask_source_representatives_bijectively_cover_full_product": True,
        "source_bijection_counted_as_supervision": True,
        "schema": _INPUT_SCHEMA,
    }
    payload = {"schema": _INPUT_SCHEMA, **{key: value for key, value in kwargs.items() if key != "schema"}}
    payload["budgets"] = selected_budgets.payload()
    payload["passive_state_observations"] = passive_payload["states"]
    payload["passive_edge_observations"] = passive_payload["edges"]
    payload["canonical_candidate_requests"] = candidate_payload
    payload["canonical_defined_requests"] = domain_payload["defined"]
    payload["canonical_undefined_requests"] = domain_payload["undefined"]
    return OpaqueActiveLearnerInput(
        **kwargs,  # type: ignore[arg-type]
        input_sha256=_sha256(payload),
    )


def make_opaque_active_input(
    source: PartialOperatorLearnerInput,
    *,
    candidate_requests: Sequence[OpaqueEdgeRequest] | None = None,
    budgets: OpaqueActiveDiscoveryBudgets | None = None,
) -> OpaqueActiveLearnerInput:
    """Compatibility adapter from frozen T1; no T1 object/hash survives."""

    if type(source) is not PartialOperatorLearnerInput:
        raise TypeError("source must be exact PartialOperatorLearnerInput")
    supplied = tuple(
        source.candidate_edge_requests
        if candidate_requests is None
        else candidate_requests
    )
    if {row.request_sha256 for row in supplied} != {
        row.request_sha256 for row in source.candidate_edge_requests
    } or len(supplied) != len(source.candidate_edge_requests):
        raise ValueError(
            "candidate_requests must be an exact permutation of the opaque candidate set"
        )
    return make_opaque_active_input_from_rows(
        event_tokens=source.event_tokens,
        query_tokens=source.query_tokens,
        answer_tokens=source.answer_tokens,
        passive_state_observations=source.passive_state_observations,
        passive_edge_observations=source.passive_edge_observations,
        candidate_requests=supplied,
        defined_requests=source.defined_edge_requests,
        undefined_requests=source.undefined_edge_requests,
        budgets=budgets,
    )


def make_opaque_candidate_removal_negative_control(
    learner_input: OpaqueActiveLearnerInput,
    removed_request_sha256: str,
) -> OpaqueActiveLearnerInput:
    """Remove exactly one opaque request from a complete omission pool.

    This is a marked loss-of-excitation control, never a primary learner
    input.  The missing request is already inferable from the visible legal
    mask and passive table; no semantic sidecar is added.
    """

    if type(learner_input) is not OpaqueActiveLearnerInput:
        raise TypeError("learner_input must be exact OpaqueActiveLearnerInput")
    if not learner_input.candidate_pool_complete:
        raise ValueError("candidate-removal control requires a complete source pool")
    _require_sha256("removed_request_sha256", removed_request_sha256)
    candidates = tuple(
        row
        for row in learner_input.canonical_candidate_requests
        if row.request_sha256 != removed_request_sha256
    )
    if len(candidates) != len(learner_input.canonical_candidate_requests) - 1:
        raise ValueError("removed request is absent from the candidate pool")
    return make_opaque_active_input_from_rows(
        event_tokens=learner_input.event_tokens,
        query_tokens=learner_input.query_tokens,
        answer_tokens=learner_input.answer_tokens,
        passive_state_observations=learner_input.passive_state_observations,
        passive_edge_observations=learner_input.passive_edge_observations,
        candidate_requests=candidates,
        defined_requests=learner_input.canonical_defined_requests,
        undefined_requests=learner_input.canonical_undefined_requests,
        budgets=learner_input.budgets,
        allow_incomplete_candidate_pool=True,
    )


@dataclass(frozen=True)
class OpaqueActiveMembershipResponse:
    request: OpaqueEdgeRequest
    target_answers: tuple[str, ...]
    response_ordinal: int
    prior_choice_sha256: str
    returned_categorical_token_count: int
    response_sha256: str
    schema: str = _RESPONSE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _RESPONSE_SCHEMA:
            raise ValueError("unknown autonomous membership-response schema")
        if type(self.request) is not OpaqueEdgeRequest:
            raise TypeError("response request must be exact OpaqueEdgeRequest")
        _validate_answers("target_answers", self.target_answers)
        _plain_int("response_ordinal", self.response_ordinal, 1)
        _require_sha256("prior_choice_sha256", self.prior_choice_sha256)
        if self.returned_categorical_token_count != 2:
            raise ValueError("one membership response returns exactly two target labels")
        if _require_sha256("response_sha256", self.response_sha256) != _sha256(self._payload(False)):
            raise ValueError("autonomous response digest mismatch")

    def _payload(self, include_response_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "request": _request_payload(self.request),
            "target_answers": self.target_answers,
            "response_ordinal": self.response_ordinal,
            "prior_choice_sha256": self.prior_choice_sha256,
            "returned_categorical_token_count": self.returned_categorical_token_count,
        }
        if include_response_sha:
            payload["response_sha256"] = self.response_sha256
        return payload


@dataclass(frozen=True)
class OpaqueStructuralInference:
    choice_sha256: str
    request: OpaqueEdgeRequest
    inferred_target_answers: tuple[str, ...]
    decision_ordinal: int
    inference_kind: str
    returned_categorical_token_count: int
    inference_sha256: str
    schema: str = _INFERENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _INFERENCE_SCHEMA:
            raise ValueError("unknown structural-inference schema")
        _require_sha256("choice_sha256", self.choice_sha256)
        if type(self.request) is not OpaqueEdgeRequest:
            raise TypeError("inference request must be exact OpaqueEdgeRequest")
        _validate_answers("inferred_target_answers", self.inferred_target_answers)
        _plain_int("decision_ordinal", self.decision_ordinal, 1)
        if self.inference_kind not in (
            "full_product_source_bijection_singleton",
            "categorical_restricted_map_singleton",
        ):
            raise ValueError("unknown structural-inference kind")
        if self.returned_categorical_token_count != 0:
            raise ValueError("structural inference cannot return oracle labels")
        if _require_sha256("inference_sha256", self.inference_sha256) != _sha256(
            self._payload(False)
        ):
            raise ValueError("structural-inference digest mismatch")

    def _payload(self, include_inference_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "choice_sha256": self.choice_sha256,
            "request": _request_payload(self.request),
            "inferred_target_answers": self.inferred_target_answers,
            "decision_ordinal": self.decision_ordinal,
            "inference_kind": self.inference_kind,
            "returned_categorical_token_count": self.returned_categorical_token_count,
        }
        if include_inference_sha:
            payload["inference_sha256"] = self.inference_sha256
        return payload


@dataclass(frozen=True)
class ExactOutcomeBranch:
    target_answers: tuple[str, ...]
    augmented_source_rows: RationalMatrix
    augmented_image_rows: RationalMatrix
    source_rank_before: int
    source_rank_after: int
    source_assignment_count_before: int
    source_assignment_count_after: int
    selected_event_version_count_before: int
    selected_event_version_count_after: int
    global_version_mass_before: int
    global_version_mass_after: int
    compatible_versions_before_sha256: str
    compatible_versions_after_sha256: str
    outcome_is_declared_full_product_row: bool
    exact_linear_constraints_consistent: bool
    branch_sha256: str
    schema: str = _BRANCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _BRANCH_SCHEMA:
            raise ValueError("unknown exact-outcome branch schema")
        _validate_answers("branch target answers", self.target_answers)
        if len(self.augmented_source_rows) != len(self.augmented_image_rows):
            raise ValueError("branch source/image row counts differ")
        before = _plain_int("source_rank_before", self.source_rank_before)
        after = _plain_int("source_rank_after", self.source_rank_after)
        actual_after = _rank(self.augmented_source_rows)
        if actual_after != after or before + 1 != after:
            raise ValueError("outcome branch must add exactly one source direction")
        assignment_before = _plain_int(
            "source_assignment_count_before", self.source_assignment_count_before, 1
        )
        assignment_after = _plain_int(
            "source_assignment_count_after", self.source_assignment_count_after, 1
        )
        event_before = _plain_int(
            "selected_event_version_count_before",
            self.selected_event_version_count_before,
            1,
        )
        event_after = _plain_int(
            "selected_event_version_count_after",
            self.selected_event_version_count_after,
            1,
        )
        mass_before = _plain_int(
            "global_version_mass_before", self.global_version_mass_before, 1
        )
        mass_after = _plain_int(
            "global_version_mass_after", self.global_version_mass_after, 1
        )
        if assignment_after > assignment_before:
            raise ValueError("an outcome branch cannot enlarge source assignments")
        if event_after > event_before or mass_after > mass_before:
            raise ValueError("an outcome branch cannot enlarge the compatible version set")
        _require_sha256(
            "compatible_versions_before_sha256",
            self.compatible_versions_before_sha256,
        )
        _require_sha256(
            "compatible_versions_after_sha256",
            self.compatible_versions_after_sha256,
        )
        if not _require_bool(
            "outcome_is_declared_full_product_row",
            self.outcome_is_declared_full_product_row,
        ):
            raise ValueError("branch output must be a declared categorical codebook row")
        consistent = _map_is_consistent(self.augmented_source_rows, self.augmented_image_rows)
        if _require_bool("exact_linear_constraints_consistent", self.exact_linear_constraints_consistent) != consistent or not consistent:
            raise ValueError("outcome branch is not an exact compatible continuation")
        if _require_sha256("branch_sha256", self.branch_sha256) != _sha256(self._payload(False)):
            raise ValueError("outcome-branch digest mismatch")

    def _payload(self, include_branch_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "target_answers": self.target_answers,
            "augmented_source_rows": self.augmented_source_rows,
            "augmented_image_rows": self.augmented_image_rows,
            "source_rank_before": self.source_rank_before,
            "source_rank_after": self.source_rank_after,
            "source_assignment_count_before": self.source_assignment_count_before,
            "source_assignment_count_after": self.source_assignment_count_after,
            "selected_event_version_count_before": self.selected_event_version_count_before,
            "selected_event_version_count_after": self.selected_event_version_count_after,
            "global_version_mass_before": self.global_version_mass_before,
            "global_version_mass_after": self.global_version_mass_after,
            "compatible_versions_before_sha256": self.compatible_versions_before_sha256,
            "compatible_versions_after_sha256": self.compatible_versions_after_sha256,
            "outcome_is_declared_full_product_row": self.outcome_is_declared_full_product_row,
            "exact_linear_constraints_consistent": self.exact_linear_constraints_consistent,
        }
        if include_branch_sha:
            payload["branch_sha256"] = self.branch_sha256
        return payload


@dataclass(frozen=True, order=True)
class OpaqueActiveCandidateScore:
    frontier_target_unlock: int
    request_sha256: str
    event_token: str
    outcome_bucket_rows: tuple[
        tuple[tuple[str, ...], int, int, int], ...
    ]
    observed_source_rank_before: int
    observed_source_rank_after: int
    compatible_outcome_count: int
    source_assignment_count_before: int
    selected_event_version_count_before: int
    global_version_mass_before: int
    worst_case_surviving_event_version_count: int
    worst_posterior_global_version_product: int
    exact_restricted_nullity_drop: int
    score_sha256: str
    schema: str = _SCORE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SCORE_SCHEMA:
            raise ValueError("unknown candidate-score schema")
        if self.frontier_target_unlock not in (0, 1):
            raise ValueError("frontier_target_unlock must be binary")
        _require_sha256("request_sha256", self.request_sha256)
        _require_token("event_token", self.event_token)
        if not isinstance(self.outcome_bucket_rows, tuple) or not self.outcome_bucket_rows:
            raise ValueError("candidate score must bind nonempty outcome buckets")
        if tuple(sorted(self.outcome_bucket_rows)) != self.outcome_bucket_rows:
            raise ValueError("candidate outcome buckets must use canonical answer order")
        for answers, assignment_count, event_version_count, global_mass in self.outcome_bucket_rows:
            _validate_answers("candidate outcome", answers)
            _plain_int("candidate outcome assignment count", assignment_count, 1)
            _plain_int("candidate outcome event-version count", event_version_count, 1)
            _plain_int("candidate outcome global mass", global_mass, 1)
        if self.observed_source_rank_after != self.observed_source_rank_before + 1:
            raise ValueError("eligible candidate must add one source direction")
        if self.compatible_outcome_count != len(self.outcome_bucket_rows):
            raise ValueError("compatible outcome count disagrees with bucket rows")
        if self.compatible_outcome_count < 1:
            raise ValueError("eligible acquisition must have at least one compatible outcome")
        assignments = _plain_int(
            "source_assignment_count_before", self.source_assignment_count_before, 1
        )
        event_versions = _plain_int(
            "selected_event_version_count_before",
            self.selected_event_version_count_before,
            1,
        )
        global_before = _plain_int(
            "global_version_mass_before", self.global_version_mass_before, 1
        )
        worst = _plain_int(
            "worst_case_surviving_event_version_count",
            self.worst_case_surviving_event_version_count,
            1,
        )
        posterior = _plain_int(
            "worst_posterior_global_version_product",
            self.worst_posterior_global_version_product,
            1,
        )
        assignment_buckets = tuple(row[1] for row in self.outcome_bucket_rows)
        event_buckets = tuple(row[2] for row in self.outcome_bucket_rows)
        global_buckets = tuple(row[3] for row in self.outcome_bucket_rows)
        if any(value > assignments for value in assignment_buckets):
            raise ValueError("candidate outcome supports more source assignments than exist")
        if self.frontier_target_unlock and sum(assignment_buckets) != assignments:
            raise ValueError("frontier outcomes must partition source assignments exactly")
        if sum(event_buckets) != event_versions:
            raise ValueError("candidate buckets do not partition selected-event versions")
        if sum(global_buckets) != global_before or max(global_buckets) != posterior:
            raise ValueError("candidate buckets do not partition the global version mass")
        if max(event_buckets) != worst:
            raise ValueError("worst selected-event bucket mismatch")
        if worst > global_before or global_before < event_versions or posterior > global_before:
            raise ValueError("candidate version-space score arithmetic is impossible")
        if self.exact_restricted_nullity_drop != _AMBIENT_RANK:
            raise ValueError("one independent edge must reduce restricted nullity by five")
        if _require_sha256("score_sha256", self.score_sha256) != _sha256(self._payload(False)):
            raise ValueError("candidate-score digest mismatch")

    def _payload(self, include_score_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "frontier_target_unlock": self.frontier_target_unlock,
            "request_sha256": self.request_sha256,
            "event_token": self.event_token,
            "outcome_bucket_rows": self.outcome_bucket_rows,
            "observed_source_rank_before": self.observed_source_rank_before,
            "observed_source_rank_after": self.observed_source_rank_after,
            "compatible_outcome_count": self.compatible_outcome_count,
            "source_assignment_count_before": self.source_assignment_count_before,
            "selected_event_version_count_before": self.selected_event_version_count_before,
            "global_version_mass_before": self.global_version_mass_before,
            "worst_case_surviving_event_version_count": self.worst_case_surviving_event_version_count,
            "worst_posterior_global_version_product": self.worst_posterior_global_version_product,
            "exact_restricted_nullity_drop": self.exact_restricted_nullity_drop,
        }
        if include_score_sha:
            payload["score_sha256"] = self.score_sha256
        return payload


@dataclass(frozen=True)
class OpaqueActiveChoiceCertificate:
    learner_input_sha256: str
    prior_state_sha256: str
    prior_response_history_sha256: str
    canonical_pending_pool_sha256: str
    known_word_table_sha256: str
    known_observed_edge_sha256: str
    decision_ordinal: int
    next_membership_call_ordinal: int
    chosen_request: OpaqueEdgeRequest
    eligible_scores: tuple[OpaqueActiveCandidateScore, ...]
    exact_outcome_branches: tuple[ExactOutcomeBranch, ...]
    selection_rule: str
    source_assignment_incomplete_before_choice: bool
    singleton_fixed_point_applied: bool
    chosen_source_available: bool
    chosen_target_expands_frontier: bool
    requires_membership_response: bool
    unique_structurally_inferred_answers: tuple[str, ...] | None
    selected_before_current_response: bool
    current_response_labels_used: bool
    future_response_labels_used: bool
    sealed_answers_used: bool
    controller_candidate_order_used: bool
    semantic_roles_used: bool
    deterministic_sha_tiebreak: bool
    rank_evaluations_this_choice: int
    conditional_assignment_blocks_evaluated: int
    basis_image_candidates_planned: int
    max_potential_version_rows_per_assignment: int
    exact_version_enumeration_uses_cached_numeric_templates: bool
    choice_sha256: str
    schema: str = _CHOICE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _CHOICE_SCHEMA:
            raise ValueError("unknown active-choice schema")
        _require_sha256("learner_input_sha256", self.learner_input_sha256)
        _require_sha256("prior_state_sha256", self.prior_state_sha256)
        _require_sha256("prior_response_history_sha256", self.prior_response_history_sha256)
        _require_sha256("canonical_pending_pool_sha256", self.canonical_pending_pool_sha256)
        _require_sha256("known_word_table_sha256", self.known_word_table_sha256)
        _require_sha256("known_observed_edge_sha256", self.known_observed_edge_sha256)
        _plain_int("decision_ordinal", self.decision_ordinal, 1)
        _plain_int("next_membership_call_ordinal", self.next_membership_call_ordinal, 1)
        if type(self.chosen_request) is not OpaqueEdgeRequest:
            raise TypeError("chosen_request must be exact OpaqueEdgeRequest")
        if not self.eligible_scores:
            raise ValueError("a choice must bind all eligible candidate scores")
        if any(type(row) is not OpaqueActiveCandidateScore for row in self.eligible_scores):
            raise TypeError("eligible score rows must be exact")
        _require_bool(
            "source_assignment_incomplete_before_choice",
            self.source_assignment_incomplete_before_choice,
        )
        _require_bool(
            "singleton_fixed_point_applied", self.singleton_fixed_point_applied
        )
        if not _require_bool("chosen_source_available", self.chosen_source_available):
            raise ValueError("a committed request source must already be available")
        _require_bool("chosen_target_expands_frontier", self.chosen_target_expands_frontier)
        if self.chosen_target_expands_frontier != bool(
            self.eligible_scores[0].frontier_target_unlock
        ):
            raise ValueError("frontier-expansion flag disagrees with chosen score")
        score_key = (
            (lambda row: row.request_sha256)
            if self.singleton_fixed_point_applied
            else (
                (lambda row: (-row.compatible_outcome_count, row.request_sha256))
                if self.source_assignment_incomplete_before_choice
                else (
                lambda row: (
                    row.worst_posterior_global_version_product,
                    row.worst_case_surviving_event_version_count,
                    -row.compatible_outcome_count,
                    row.request_sha256,
                )
                )
            )
        )
        if tuple(sorted(self.eligible_scores, key=score_key)) != self.eligible_scores:
            raise ValueError("eligible scores must use the deterministic selection order")
        if self.eligible_scores[0].request_sha256 != self.chosen_request.request_sha256:
            raise ValueError("chosen request must be the first deterministic score row")
        if len(self.exact_outcome_branches) != self.eligible_scores[0].compatible_outcome_count:
            raise ValueError("chosen outcome branches do not match compatible count")
        if len({row.target_answers for row in self.exact_outcome_branches}) != len(self.exact_outcome_branches):
            raise ValueError("outcome branches must use distinct target diagnostics")
        chosen_score = self.eligible_scores[0]
        buckets = {row[0]: row[1:] for row in chosen_score.outcome_bucket_rows}
        for branch in self.exact_outcome_branches:
            assignment_after, event_after, mass_after = buckets[branch.target_answers]
            if (
                branch.source_assignment_count_before
                != chosen_score.source_assignment_count_before
                or branch.selected_event_version_count_before
                != chosen_score.selected_event_version_count_before
                or branch.global_version_mass_before
                != chosen_score.global_version_mass_before
                or branch.source_assignment_count_after != assignment_after
                or branch.selected_event_version_count_after != event_after
                or branch.global_version_mass_after != mass_after
            ):
                raise ValueError("outcome branch counts do not cross-link to chosen score bucket")
        if self.selection_rule != "opaque_source_frontier_then_exact_categorical_map_minimax_then_request_sha":
            raise ValueError("unknown autonomous active-selection rule")
        requires_response = len(self.exact_outcome_branches) > 1
        if _require_bool("requires_membership_response", self.requires_membership_response) != requires_response:
            raise ValueError("membership-response requirement must follow exact branch count")
        expected_inference = (
            None if requires_response else self.exact_outcome_branches[0].target_answers
        )
        if self.unique_structurally_inferred_answers != expected_inference:
            raise ValueError("unique structural inference does not match the sole exact branch")
        for name, value, required in (
            ("selected_before_current_response", self.selected_before_current_response, True),
            ("current_response_labels_used", self.current_response_labels_used, False),
            ("future_response_labels_used", self.future_response_labels_used, False),
            ("sealed_answers_used", self.sealed_answers_used, False),
            ("controller_candidate_order_used", self.controller_candidate_order_used, False),
            ("semantic_roles_used", self.semantic_roles_used, False),
            ("deterministic_sha_tiebreak", self.deterministic_sha_tiebreak, True),
        ):
            if _require_bool(name, value) is not required:
                raise ValueError(f"{name} must be {required}")
        _plain_int("rank_evaluations_this_choice", self.rank_evaluations_this_choice, 1)
        _plain_int(
            "conditional_assignment_blocks_evaluated",
            self.conditional_assignment_blocks_evaluated,
            1,
        )
        _plain_int(
            "basis_image_candidates_planned",
            self.basis_image_candidates_planned,
            1,
        )
        _plain_int(
            "max_potential_version_rows_per_assignment",
            self.max_potential_version_rows_per_assignment,
            1,
        )
        if not _require_bool(
            "exact_version_enumeration_uses_cached_numeric_templates",
            self.exact_version_enumeration_uses_cached_numeric_templates,
        ):
            raise ValueError("exact version enumeration must disclose numeric template caching")
        if _require_sha256("choice_sha256", self.choice_sha256) != _sha256(self._payload(False)):
            raise ValueError("active-choice digest mismatch")

    def _payload(self, include_choice_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "learner_input_sha256": self.learner_input_sha256,
            "prior_state_sha256": self.prior_state_sha256,
            "prior_response_history_sha256": self.prior_response_history_sha256,
            "canonical_pending_pool_sha256": self.canonical_pending_pool_sha256,
            "known_word_table_sha256": self.known_word_table_sha256,
            "known_observed_edge_sha256": self.known_observed_edge_sha256,
            "decision_ordinal": self.decision_ordinal,
            "next_membership_call_ordinal": self.next_membership_call_ordinal,
            "chosen_request": _request_payload(self.chosen_request),
            "eligible_scores": [row._payload(True) for row in self.eligible_scores],
            "exact_outcome_branches": [row._payload(True) for row in self.exact_outcome_branches],
            "selection_rule": self.selection_rule,
            "source_assignment_incomplete_before_choice": self.source_assignment_incomplete_before_choice,
            "singleton_fixed_point_applied": self.singleton_fixed_point_applied,
            "chosen_source_available": self.chosen_source_available,
            "chosen_target_expands_frontier": self.chosen_target_expands_frontier,
            "requires_membership_response": self.requires_membership_response,
            "unique_structurally_inferred_answers": self.unique_structurally_inferred_answers,
            "selected_before_current_response": self.selected_before_current_response,
            "current_response_labels_used": self.current_response_labels_used,
            "future_response_labels_used": self.future_response_labels_used,
            "sealed_answers_used": self.sealed_answers_used,
            "controller_candidate_order_used": self.controller_candidate_order_used,
            "semantic_roles_used": self.semantic_roles_used,
            "deterministic_sha_tiebreak": self.deterministic_sha_tiebreak,
            "rank_evaluations_this_choice": self.rank_evaluations_this_choice,
            "conditional_assignment_blocks_evaluated": self.conditional_assignment_blocks_evaluated,
            "basis_image_candidates_planned": self.basis_image_candidates_planned,
            "max_potential_version_rows_per_assignment": self.max_potential_version_rows_per_assignment,
            "exact_version_enumeration_uses_cached_numeric_templates": self.exact_version_enumeration_uses_cached_numeric_templates,
        }
        if include_choice_sha:
            payload["choice_sha256"] = self.choice_sha256
        return payload


@dataclass(frozen=True)
class OpaqueActiveStep:
    choice: OpaqueActiveChoiceCertificate
    response: OpaqueActiveMembershipResponse
    step_sha256: str
    schema: str = _STEP_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _STEP_SCHEMA:
            raise ValueError("unknown active-step schema")
        if type(self.choice) is not OpaqueActiveChoiceCertificate:
            raise TypeError("step choice must be exact")
        if type(self.response) is not OpaqueActiveMembershipResponse:
            raise TypeError("step response must be exact")
        if not self.choice.requires_membership_response:
            raise ValueError("a unique structural inference cannot consume a response")
        if self.response.request != self.choice.chosen_request:
            raise ValueError("response does not answer the committed request")
        if self.response.response_ordinal != self.choice.next_membership_call_ordinal:
            raise ValueError("response ordinal differs from committed choice")
        if self.response.prior_choice_sha256 != self.choice.choice_sha256:
            raise ValueError("response does not bind the prior choice certificate")
        if self.response.target_answers not in {
            branch.target_answers for branch in self.choice.exact_outcome_branches
        }:
            raise ValueError("response target is outside the committed exact branches")
        if _require_sha256("step_sha256", self.step_sha256) != _sha256(self._payload(False)):
            raise ValueError("active-step digest mismatch")

    def _payload(self, include_step_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "choice": self.choice._payload(True),
            "response": self.response._payload(True),
        }
        if include_step_sha:
            payload["step_sha256"] = self.step_sha256
        return payload


@dataclass(frozen=True)
class OpaqueStructuralInferenceStep:
    choice: OpaqueActiveChoiceCertificate
    inference: OpaqueStructuralInference
    step_sha256: str
    schema: str = _STEP_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _STEP_SCHEMA:
            raise ValueError("unknown structural-inference-step schema")
        if type(self.choice) is not OpaqueActiveChoiceCertificate:
            raise TypeError("inference-step choice must be exact")
        if type(self.inference) is not OpaqueStructuralInference:
            raise TypeError("inference-step inference must be exact")
        if self.choice.requires_membership_response:
            raise ValueError("ambiguous active choice cannot be inferred")
        if self.inference.choice_sha256 != self.choice.choice_sha256:
            raise ValueError("inference does not bind the prior choice")
        if self.inference.request != self.choice.chosen_request:
            raise ValueError("inference does not apply to the committed request")
        if self.inference.decision_ordinal != self.choice.decision_ordinal:
            raise ValueError("inference decision ordinal mismatch")
        if self.inference.inferred_target_answers != self.choice.unique_structurally_inferred_answers:
            raise ValueError("inference answer differs from the unique exact branch")
        if _require_sha256("step_sha256", self.step_sha256) != _sha256(self._payload(False)):
            raise ValueError("structural-inference-step digest mismatch")

    def _payload(self, include_step_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "choice": self.choice._payload(True),
            "inference": self.inference._payload(True),
        }
        if include_step_sha:
            payload["step_sha256"] = self.step_sha256
        return payload


@dataclass(frozen=True)
class OpaqueActiveDiscoveryState:
    learner_input_sha256: str
    steps: tuple[OpaqueActiveStep | OpaqueStructuralInferenceStep, ...]
    known_word_count: int
    known_mask_source_count: int
    known_word_table_sha256: str
    known_observed_edge_sha256: str
    canonical_pending_pool_sha256: str
    direct_passive_event_ranks: tuple[tuple[str, int], ...]
    response_event_rank_increments: tuple[tuple[str, int], ...]
    inference_event_rank_increments: tuple[tuple[str, int], ...]
    observed_event_ranks: tuple[tuple[str, int], ...]
    returned_categorical_token_count: int
    active_call_count: int
    structural_inference_count: int
    exact_rank_evaluations: int
    candidate_score_rows_evaluated: int
    response_history_sha256: str
    state_sha256: str
    schema: str = _STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _STATE_SCHEMA:
            raise ValueError("unknown active-state schema")
        _require_sha256("learner_input_sha256", self.learner_input_sha256)
        if not isinstance(self.steps, tuple) or any(
            type(row) not in (OpaqueActiveStep, OpaqueStructuralInferenceStep)
            for row in self.steps
        ):
            raise TypeError("state steps must be exact response/inference step rows")
        if tuple(row.choice.decision_ordinal for row in self.steps) != tuple(range(1, len(self.steps) + 1)):
            raise ValueError("state decision ordinals must be contiguous")
        response_rows = tuple(row for row in self.steps if type(row) is OpaqueActiveStep)
        inference_rows = tuple(
            row for row in self.steps if type(row) is OpaqueStructuralInferenceStep
        )
        if tuple(row.response.response_ordinal for row in response_rows) != tuple(
            range(1, len(response_rows) + 1)
        ):
            raise ValueError("membership-response ordinals must be contiguous")
        if len({row.choice.chosen_request.request_sha256 for row in self.steps}) != len(self.steps):
            raise ValueError("state cannot query the same request twice")
        _plain_int("known_word_count", self.known_word_count)
        _plain_int("known_mask_source_count", self.known_mask_source_count)
        _require_sha256("known_word_table_sha256", self.known_word_table_sha256)
        _require_sha256("known_observed_edge_sha256", self.known_observed_edge_sha256)
        _require_sha256("canonical_pending_pool_sha256", self.canonical_pending_pool_sha256)
        for name, rows in (
            ("direct_passive_event_ranks", self.direct_passive_event_ranks),
            ("response_event_rank_increments", self.response_event_rank_increments),
            ("inference_event_rank_increments", self.inference_event_rank_increments),
        ):
            if not isinstance(rows, tuple) or len(rows) != _EVENT_COUNT:
                raise ValueError(f"{name} must bind ten event rows")
            for token, value in rows:
                _require_token(name, token)
                _plain_int(name, value)
        if not isinstance(self.observed_event_ranks, tuple) or len(self.observed_event_ranks) != _EVENT_COUNT:
            raise ValueError("state must bind ten event-rank rows")
        for token, rank in self.observed_event_ranks:
            _require_token("observed event token", token)
            _plain_int("observed event rank", rank)
        direct = dict(self.direct_passive_event_ranks)
        response = dict(self.response_event_rank_increments)
        inference = dict(self.inference_event_rank_increments)
        observed = dict(self.observed_event_ranks)
        expected_tokens = set(direct)
        if not (
            set(response) == expected_tokens
            and set(inference) == expected_tokens
            and set(observed) == expected_tokens
        ):
            raise ValueError("event-rank provenance rows use inconsistent token inventories")
        if any(
            direct[token] + response[token] + inference[token] != observed[token]
            for token in expected_tokens
        ):
            raise ValueError("direct/response/inference event-rank arithmetic mismatch")
        if self.active_call_count != len(response_rows):
            raise ValueError("active-call count mismatch")
        if self.structural_inference_count != len(inference_rows):
            raise ValueError("structural-inference count mismatch")
        if self.returned_categorical_token_count != 2 * self.active_call_count:
            raise ValueError("returned-token count must be twice the active-call count")
        _plain_int("exact_rank_evaluations", self.exact_rank_evaluations)
        _plain_int("candidate_score_rows_evaluated", self.candidate_score_rows_evaluated)
        if self.exact_rank_evaluations != sum(
            row.choice.rank_evaluations_this_choice for row in self.steps
        ):
            raise ValueError("state exact-rank work is not recomputed from choice history")
        if self.candidate_score_rows_evaluated != sum(
            len(row.choice.eligible_scores) for row in self.steps
        ):
            raise ValueError("state candidate-score work is not recomputed from choice history")
        if _require_sha256("response_history_sha256", self.response_history_sha256) != _sha256(
            tuple(row.response.response_sha256 for row in response_rows)
        ):
            raise ValueError("response-history digest mismatch")
        if _require_sha256("state_sha256", self.state_sha256) != _sha256(self._payload(False)):
            raise ValueError("active-state digest mismatch")

    def _payload(self, include_state_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "learner_input_sha256": self.learner_input_sha256,
            "steps": [row._payload(True) for row in self.steps],
            "known_word_count": self.known_word_count,
            "known_mask_source_count": self.known_mask_source_count,
            "known_word_table_sha256": self.known_word_table_sha256,
            "known_observed_edge_sha256": self.known_observed_edge_sha256,
            "canonical_pending_pool_sha256": self.canonical_pending_pool_sha256,
            "direct_passive_event_ranks": self.direct_passive_event_ranks,
            "response_event_rank_increments": self.response_event_rank_increments,
            "inference_event_rank_increments": self.inference_event_rank_increments,
            "observed_event_ranks": self.observed_event_ranks,
            "returned_categorical_token_count": self.returned_categorical_token_count,
            "active_call_count": self.active_call_count,
            "structural_inference_count": self.structural_inference_count,
            "exact_rank_evaluations": self.exact_rank_evaluations,
            "candidate_score_rows_evaluated": self.candidate_score_rows_evaluated,
            "response_history_sha256": self.response_history_sha256,
        }
        if include_state_sha:
            payload["state_sha256"] = self.state_sha256
        return payload


@dataclass(frozen=True)
class _KnownEdge:
    request: OpaqueEdgeRequest
    source_answers: tuple[str, ...]
    target_answers: tuple[str, ...]


def _mask_source_words(learner_input: OpaqueActiveLearnerInput) -> tuple[OpaqueWord, ...]:
    return tuple(
        sorted(
            {
                row.source_word
                for row in learner_input.canonical_defined_requests
                + learner_input.canonical_undefined_requests
            }
        )
    )


def _full_product_answers(
    learner_input: OpaqueActiveLearnerInput,
) -> tuple[tuple[str, ...], ...]:
    return tuple(sorted(product(learner_input.answer_tokens, repeat=_QUERY_COUNT)))


def _known_material(
    learner_input: OpaqueActiveLearnerInput,
    steps: Sequence[OpaqueActiveStep | OpaqueStructuralInferenceStep],
) -> tuple[dict[OpaqueWord, tuple[str, ...]], tuple[_KnownEdge, ...]]:
    known: dict[OpaqueWord, tuple[str, ...]] = {}

    def reconcile(word: OpaqueWord, answers: tuple[str, ...]) -> None:
        _validate_answers("known diagnostic", answers, learner_input.answer_tokens)
        previous = known.setdefault(word, answers)
        if previous != answers:
            raise ValueError("active history gives one opaque word contradictory diagnostics")

    for row in learner_input.passive_state_observations:
        reconcile(row.word, row.answers)
    edges: list[_KnownEdge] = []
    for row in learner_input.passive_edge_observations:
        reconcile(row.request.source_word, row.source_answers)
        reconcile(row.request.program, row.target_answers)
        edges.append(_KnownEdge(row.request, row.source_answers, row.target_answers))
    acquired: set[str] = set()
    response_ordinal = 0
    for decision_ordinal, row in enumerate(steps, 1):
        if row.choice.decision_ordinal != decision_ordinal:
            raise ValueError("active history has a noncontiguous decision ordinal")
        request = row.choice.chosen_request
        if request.request_sha256 in acquired:
            raise ValueError("active history acquires the same request twice")
        if request.source_word not in known:
            raise ValueError("active history uses a source diagnostic not yet known")
        if type(row) is OpaqueActiveStep:
            response_ordinal += 1
            if row.response.response_ordinal != response_ordinal:
                raise ValueError("active history has a noncontiguous response ordinal")
            target_answers = row.response.target_answers
        else:
            target_answers = row.inference.inferred_target_answers
        source_answers = known[request.source_word]
        reconcile(request.program, target_answers)
        edges.append(_KnownEdge(request, source_answers, target_answers))
        acquired.add(request.request_sha256)
    return known, tuple(edges)


def _source_assignments(
    learner_input: OpaqueActiveLearnerInput,
    known: dict[OpaqueWord, tuple[str, ...]],
) -> tuple[tuple[tuple[OpaqueWord, tuple[str, ...]], ...], ...]:
    source_words = _mask_source_words(learner_input)
    known_rows = {word: known[word] for word in source_words if word in known}
    if len(set(known_rows.values())) != len(known_rows):
        raise ValueError(
            "supplied full-product source-representative bijection forbids duplicate known rows"
        )
    all_answers = _full_product_answers(learner_input)
    if any(answers not in all_answers for answers in known_rows.values()):
        raise ValueError("known mask-source diagnostic is outside the full product")
    unknown_words = tuple(word for word in source_words if word not in known_rows)
    missing_answers = tuple(row for row in all_answers if row not in known_rows.values())
    if len(unknown_words) != len(missing_answers):
        raise ValueError("mask-source/full-product bijection has inconsistent cardinality")
    from itertools import permutations

    assignments = tuple(
        tuple(sorted(tuple(known_rows.items()) + tuple(zip(unknown_words, permutation, strict=True))))
        for permutation in permutations(missing_answers)
    )
    return tuple(sorted(assignments, key=_sha256))


def _observed_event_rows(
    learner_input: OpaqueActiveLearnerInput,
    edges: Sequence[_KnownEdge],
    event_token: str,
) -> tuple[tuple[RationalVector, ...], tuple[RationalVector, ...]]:
    selected = tuple(row for row in edges if row.request.event_token == event_token)
    return (
        tuple(_diagnostic_row(row.source_answers, learner_input.answer_tokens) for row in selected),
        tuple(_diagnostic_row(row.target_answers, learner_input.answer_tokens) for row in selected),
    )


def _observed_rank_rows(
    learner_input: OpaqueActiveLearnerInput,
    edges: Sequence[_KnownEdge],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (
            token,
            _rank(_observed_event_rows(learner_input, edges, token)[0]),
        )
        for token in sorted(learner_input.event_tokens)
    )


def _make_state(
    learner_input: OpaqueActiveLearnerInput,
    steps: tuple[OpaqueActiveStep | OpaqueStructuralInferenceStep, ...],
) -> OpaqueActiveDiscoveryState:
    known, edges = _known_material(learner_input, steps)
    mask_words = _mask_source_words(learner_input)
    response_rows = tuple(row for row in steps if type(row) is OpaqueActiveStep)
    inference_rows = tuple(row for row in steps if type(row) is OpaqueStructuralInferenceStep)
    direct_edges = tuple(
        _KnownEdge(row.request, row.source_answers, row.target_answers)
        for row in learner_input.passive_edge_observations
    )
    direct_ranks = _observed_rank_rows(learner_input, direct_edges)
    response_increments = {token: 0 for token in learner_input.event_tokens}
    inference_increments = {token: 0 for token in learner_input.event_tokens}
    prefix_edges: list[_KnownEdge] = list(direct_edges)
    for step, edge in zip(steps, edges[len(direct_edges):], strict=True):
        token = edge.request.event_token
        before = _rank(_observed_event_rows(learner_input, prefix_edges, token)[0])
        prefix_edges.append(edge)
        after = _rank(_observed_event_rows(learner_input, prefix_edges, token)[0])
        target = response_increments if type(step) is OpaqueActiveStep else inference_increments
        target[token] += after - before
    acquired = {row.choice.chosen_request.request_sha256 for row in steps}
    pending = tuple(
        row.request_sha256
        for row in learner_input.canonical_candidate_requests
        if row.request_sha256 not in acquired
    )
    known_word_payload = tuple(sorted(known.items()))
    known_edge_payload = tuple(
        sorted(
            (
                row.request.request_sha256,
                row.source_answers,
                row.target_answers,
            )
            for row in edges
        )
    )
    kwargs: dict[str, object] = {
        "learner_input_sha256": learner_input.input_sha256,
        "steps": steps,
        "known_word_count": len(known),
        "known_mask_source_count": sum(word in known for word in mask_words),
        "known_word_table_sha256": _sha256(known_word_payload),
        "known_observed_edge_sha256": _sha256(known_edge_payload),
        "canonical_pending_pool_sha256": _sha256(pending),
        "direct_passive_event_ranks": direct_ranks,
        "response_event_rank_increments": tuple(
            (token, response_increments[token])
            for token in sorted(learner_input.event_tokens)
        ),
        "inference_event_rank_increments": tuple(
            (token, inference_increments[token])
            for token in sorted(learner_input.event_tokens)
        ),
        "observed_event_ranks": _observed_rank_rows(learner_input, edges),
        "returned_categorical_token_count": 2 * len(response_rows),
        "active_call_count": len(response_rows),
        "structural_inference_count": len(inference_rows),
        "exact_rank_evaluations": sum(
            row.choice.rank_evaluations_this_choice for row in steps
        ),
        "candidate_score_rows_evaluated": sum(
            len(row.choice.eligible_scores) for row in steps
        ),
        "response_history_sha256": _sha256(
            tuple(row.response.response_sha256 for row in response_rows)
        ),
        "schema": _STATE_SCHEMA,
    }
    payload = {"schema": _STATE_SCHEMA, **{key: value for key, value in kwargs.items() if key != "schema"}}
    payload["steps"] = [row._payload(True) for row in steps]
    return OpaqueActiveDiscoveryState(
        **kwargs,  # type: ignore[arg-type]
        state_sha256=_sha256(payload),
    )


def initialize_opaque_active_discovery(
    learner_input: OpaqueActiveLearnerInput,
) -> OpaqueActiveDiscoveryState:
    if type(learner_input) is not OpaqueActiveLearnerInput:
        raise TypeError("learner_input must be exact OpaqueActiveLearnerInput")
    if learner_input.canonical_candidate_requests:
        if learner_input.budgets.max_active_calls < _EXPECTED_ACTIVE_CALLS:
            raise OpaqueActiveDiscoveryLimitError(
                "active-call budget is below the realized primary-arm requirement 14"
            )
        if learner_input.budgets.max_returned_categorical_tokens < _EXPECTED_RETURNED_LABELS:
            raise OpaqueActiveDiscoveryLimitError(
                "returned-label budget is below the realized primary-arm requirement 28"
            )
        if learner_input.budgets.max_structural_inferences < _EXPECTED_STRUCTURAL_INFERENCES:
            raise OpaqueActiveDiscoveryLimitError(
                "structural-inference budget is below the primary-arm requirement"
            )
        candidate_count = len(learner_input.canonical_candidate_requests)
        decision_cap = min(
            candidate_count,
            learner_input.budgets.max_active_calls
            + learner_input.budgets.max_structural_inferences,
        )
        planned_score_rows = sum(candidate_count - index for index in range(decision_cap))
        planned_rank_evaluations = sum(
            2 * (candidate_count - index)
            + 2 * learner_input.budgets.max_outcome_branches_per_choice
            for index in range(decision_cap)
        )
        if learner_input.budgets.max_candidate_score_rows < planned_score_rows:
            raise OpaqueActiveDiscoveryLimitError(
                "candidate-score budget is below the analytic transcript ceiling"
            )
        if learner_input.budgets.max_exact_rank_evaluations < planned_rank_evaluations:
            raise OpaqueActiveDiscoveryLimitError(
                "exact-rank budget is below the analytic transcript ceiling"
            )
    return _make_state(learner_input, ())


@dataclass(frozen=True)
class ExactCategoricalRestrictedMapVersion:
    event_token: str
    legal_request_sha256s: tuple[str, ...]
    predicted_target_answers: tuple[tuple[str, ...], ...]
    domain_basis_image_rows: RationalMatrix
    version_sha256: str

    def __post_init__(self) -> None:
        _require_token("event_token", self.event_token)
        if len(self.legal_request_sha256s) != len(self.predicted_target_answers):
            raise ValueError("version prediction/request counts differ")
        for digest in self.legal_request_sha256s:
            _require_sha256("legal request digest", digest)
        for answers in self.predicted_target_answers:
            _validate_answers("version target answers", answers)
        if _require_sha256("version_sha256", self.version_sha256) != _sha256(
            {
                "event_token": self.event_token,
                "legal_request_sha256s": self.legal_request_sha256s,
                "predicted_target_answers": self.predicted_target_answers,
                "domain_basis_image_rows": self.domain_basis_image_rows,
            }
        ):
            raise ValueError("categorical restricted-map version digest mismatch")


_BASE_VERSION_CACHE: dict[
    tuple[str, str],
    tuple[tuple[str, tuple[ExactCategoricalRestrictedMapVersion, ...]], ...],
] = {}

_VERSION_TEMPLATE_CACHE: dict[
    tuple[tuple[Fraction, ...], ...],
    tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
] = {}
_COORDINATE_LABEL_CACHE: dict[
    tuple[Fraction, ...], tuple[int | None, ...]
] = {}


def _universal_codebook_rows() -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(map(Fraction, (1, int(a == 1), int(a == 2), int(b == 1), int(b == 2))))
        for a, b in product(range(_ANSWER_COUNT), repeat=_QUERY_COUNT)
    )


def _coordinate_label_table(coordinates: tuple[Fraction, ...]) -> tuple[int | None, ...]:
    """Return the exact categorical image for every basis-label assignment.

    The diagnostic gauge is an affine one-hot encoding.  Precomputing the
    three-way categorical lookup removes Fraction arithmetic from the much
    larger ``9**domain_rank`` pair-image enumeration without changing the
    exact version set.
    """

    cached = _COORDINATE_LABEL_CACHE.get(coordinates)
    if cached is not None:
        return cached
    values: list[int | None] = []
    for labels in product(range(_ANSWER_COUNT), repeat=len(coordinates)):
        contrast_one = sum(
            coefficient for coefficient, label in zip(coordinates, labels, strict=True)
            if label == 1
        )
        contrast_two = sum(
            coefficient for coefficient, label in zip(coordinates, labels, strict=True)
            if label == 2
        )
        if contrast_one == 0 and contrast_two == 0:
            values.append(0)
        elif contrast_one == 1 and contrast_two == 0:
            values.append(1)
        elif contrast_one == 0 and contrast_two == 1:
            values.append(2)
        else:
            values.append(None)
    frozen = tuple(values)
    _COORDINATE_LABEL_CACHE[coordinates] = frozen
    return frozen


def _categorical_version_templates(
    source_rows: RationalMatrix,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    key = tuple(
        tuple(value.as_fraction() for value in row) for row in source_rows
    )
    cached = _VERSION_TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    domain_rank, pivot_indices, _ = _rank_profile(source_rows)
    domain_basis = tuple(source_rows[index] for index in pivot_indices)
    coordinates = tuple(_row_coordinates(row, domain_basis) for row in source_rows)
    coordinate_tables = tuple(_coordinate_label_table(row) for row in coordinates)
    label_tuple_count = _ANSWER_COUNT ** domain_rank
    valid: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for image_indices in product(range(_STATE_COUNT), repeat=domain_rank):
        first_label_index = 0
        second_label_index = 0
        for image_index in image_indices:
            first_label, second_label = divmod(image_index, _ANSWER_COUNT)
            first_label_index = first_label_index * _ANSWER_COUNT + first_label
            second_label_index = second_label_index * _ANSWER_COUNT + second_label
        if first_label_index >= label_tuple_count or second_label_index >= label_tuple_count:
            raise AssertionError("categorical label tuple index escaped its exact range")
        predictions: list[int] = []
        for table in coordinate_tables:
            first_prediction = table[first_label_index]
            second_prediction = table[second_label_index]
            if first_prediction is None or second_prediction is None:
                break
            predictions.append(first_prediction * _ANSWER_COUNT + second_prediction)
        else:
            valid.append((tuple(image_indices), tuple(predictions)))
    frozen = tuple(valid)
    _VERSION_TEMPLATE_CACHE[key] = frozen
    return frozen


def _enumerate_base_versions(
    learner_input: OpaqueActiveLearnerInput,
    known: dict[OpaqueWord, tuple[str, ...]],
) -> tuple[tuple[str, tuple[ExactCategoricalRestrictedMapVersion, ...]], ...]:
    source_words = _mask_source_words(learner_input)
    if any(word not in known for word in source_words):
        raise ValueError("categorical map versions require all nine mask-source diagnostics")
    source_assignment_payload = tuple((word, known[word]) for word in source_words)
    cache_key = (learner_input.input_sha256, _sha256(source_assignment_payload))
    cached = _BASE_VERSION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    codebook_answers = _full_product_answers(learner_input)
    codebook_rows = tuple(
        _diagnostic_row(answers, learner_input.answer_tokens)
        for answers in codebook_answers
    )
    if tuple(
        tuple(value.as_fraction() for value in row) for row in codebook_rows
    ) != _universal_codebook_rows():
        raise AssertionError("opaque diagnostic relabel changed the numeric codebook geometry")
    result: list[tuple[str, tuple[ExactCategoricalRestrictedMapVersion, ...]]] = []
    for event_token in sorted(learner_input.event_tokens):
        requests = tuple(
            row
            for row in learner_input.canonical_defined_requests
            if row.event_token == event_token
        )
        source_rows = tuple(
            _diagnostic_row(known[row.source_word], learner_input.answer_tokens)
            for row in requests
        )
        domain_rank, pivot_indices, _ = _rank_profile(source_rows)
        versions: list[ExactCategoricalRestrictedMapVersion] = []
        templates = _categorical_version_templates(source_rows)
        for image_indices, prediction_indices in templates:
            image_basis_rows = tuple(codebook_rows[index] for index in image_indices)
            predictions = tuple(codebook_answers[index] for index in prediction_indices)
            payload = {
                "event_token": event_token,
                "legal_request_sha256s": tuple(row.request_sha256 for row in requests),
                "predicted_target_answers": predictions,
                "domain_basis_image_rows": image_basis_rows,
            }
            versions.append(
                ExactCategoricalRestrictedMapVersion(
                    **payload,
                    version_sha256=_sha256(payload),
                )
            )
        canonical = tuple(sorted(versions, key=lambda row: row.version_sha256))
        if not canonical:
            raise ValueError("declared codebook admits no restricted-map version")
        result.append((event_token, canonical))
    frozen = tuple(result)
    _BASE_VERSION_CACHE[cache_key] = frozen
    return frozen


def _filtered_versions(
    learner_input: OpaqueActiveLearnerInput,
    known: dict[OpaqueWord, tuple[str, ...]],
    edges: Sequence[_KnownEdge],
) -> dict[str, tuple[ExactCategoricalRestrictedMapVersion, ...]]:
    base = dict(_enumerate_base_versions(learner_input, known))
    result: dict[str, tuple[ExactCategoricalRestrictedMapVersion, ...]] = {}
    for event_token, versions in base.items():
        constraints: dict[str, tuple[str, ...]] = {}
        for row in edges:
            if row.request.event_token != event_token:
                continue
            previous = constraints.setdefault(
                row.request.request_sha256, row.target_answers
            )
            if previous != row.target_answers:
                raise ValueError("known edge constraints contradict each other")
        # Literal word identity is learner-visible: if the exact target program
        # already has a diagnostic anywhere in the causal known-word table,
        # every compatible map version must predict that same row even when
        # this edge was not itself acquired.
        for request in learner_input.canonical_defined_requests:
            if request.event_token != event_token or request.program not in known:
                continue
            expected = known[request.program]
            previous = constraints.setdefault(request.request_sha256, expected)
            if previous != expected:
                raise ValueError("known target-word equality contradicts an edge response")
        request_indices = {
            digest: index
            for index, digest in enumerate(versions[0].legal_request_sha256s)
        }
        filtered = tuple(
            version
            for version in versions
            if all(
                version.predicted_target_answers[request_indices[digest]] == answers
                for digest, answers in constraints.items()
            )
        )
        if not filtered:
            raise ValueError("observed answers eliminate every exact categorical map version")
        result[event_token] = filtered
    return result


@dataclass(frozen=True)
class _ConditionalGlobalVersionBlock:
    source_assignment: tuple[tuple[OpaqueWord, tuple[str, ...]], ...]
    event_version_factors: tuple[tuple[str, int, str], ...]
    global_version_mass: int
    block_sha256: str

    def __post_init__(self) -> None:
        expected_mass = 1
        for token, count, digest in self.event_version_factors:
            _require_token("conditional event token", token)
            expected_mass *= _plain_int("conditional event version count", count, 1)
            _require_sha256("conditional event version digest", digest)
        if self.global_version_mass != expected_mass:
            raise ValueError("conditional global-version mass factorization mismatch")
        if self.block_sha256 != _sha256(self.payload(False)):
            raise ValueError("conditional global-version block digest mismatch")

    def payload(self, include_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_assignment": self.source_assignment,
            "event_version_factors": self.event_version_factors,
            "global_version_mass": self.global_version_mass,
        }
        if include_sha:
            payload["block_sha256"] = self.block_sha256
        return payload

    def event_count(self, event_token: str) -> int:
        return {token: count for token, count, _ in self.event_version_factors}[
            event_token
        ]


def _filtered_template_summary(
    learner_input: OpaqueActiveLearnerInput,
    known: dict[OpaqueWord, tuple[str, ...]],
    edges: Sequence[_KnownEdge],
    event_token: str,
) -> tuple[int, str]:
    requests = tuple(
        row
        for row in learner_input.canonical_defined_requests
        if row.event_token == event_token
    )
    source_rows = tuple(
        _diagnostic_row(known[row.source_word], learner_input.answer_tokens)
        for row in requests
    )
    templates = _categorical_version_templates(source_rows)
    answer_rows = _full_product_answers(learner_input)
    answer_index = {answers: index for index, answers in enumerate(answer_rows)}
    request_indices = {
        row.request_sha256: index for index, row in enumerate(requests)
    }
    constraints: dict[str, tuple[str, ...]] = {}
    for edge in edges:
        if edge.request.event_token == event_token:
            constraints[edge.request.request_sha256] = edge.target_answers
    for request in requests:
        if request.program in known:
            previous = constraints.setdefault(
                request.request_sha256, known[request.program]
            )
            if previous != known[request.program]:
                raise ValueError("conditional known-word constraint contradiction")
    filtered = tuple(
        template
        for template in templates
        if all(
            template[1][request_indices[digest]] == answer_index[answers]
            for digest, answers in constraints.items()
        )
    )
    if not filtered:
        raise ValueError("conditional assignment eliminates every map template")
    return len(filtered), _sha256(filtered)


def _conditional_global_blocks(
    learner_input: OpaqueActiveLearnerInput,
    known: dict[OpaqueWord, tuple[str, ...]],
    edges: Sequence[_KnownEdge],
    assignments: Sequence[tuple[tuple[OpaqueWord, tuple[str, ...]], ...]],
) -> tuple[_ConditionalGlobalVersionBlock, ...]:
    blocks: list[_ConditionalGlobalVersionBlock] = []
    for assignment in assignments:
        completed_known = dict(known)
        for word, answers in assignment:
            previous = completed_known.setdefault(word, answers)
            if previous != answers:
                raise ValueError("source-assignment block contradicts known diagnostics")
        event_rows = tuple(
            (
                token,
                *_filtered_template_summary(
                    learner_input, completed_known, edges, token
                ),
            )
            for token in sorted(learner_input.event_tokens)
        )
        mass = 1
        for _, count, _ in event_rows:
            mass *= count
        payload = {
            "source_assignment": assignment,
            "event_version_factors": event_rows,
            "global_version_mass": mass,
        }
        blocks.append(
            _ConditionalGlobalVersionBlock(
                source_assignment=assignment,
                event_version_factors=event_rows,
                global_version_mass=mass,
                block_sha256=_sha256(payload),
            )
        )
    return tuple(sorted(blocks, key=lambda row: row.block_sha256))


def _planned_basis_image_candidates(
    learner_input: OpaqueActiveLearnerInput,
    known: dict[OpaqueWord, tuple[str, ...]],
    assignments: Sequence[tuple[tuple[OpaqueWord, tuple[str, ...]], ...]],
) -> tuple[int, ...]:
    totals: list[int] = []
    for assignment in assignments:
        total = 0
        completed = dict(known)
        completed.update(dict(assignment))
        for token in learner_input.event_tokens:
            source_rows = tuple(
                _diagnostic_row(completed[row.source_word], learner_input.answer_tokens)
                for row in learner_input.canonical_defined_requests
                if row.event_token == token
            )
            total += _STATE_COUNT ** _rank(source_rows)
        totals.append(total)
    return tuple(totals)


def _make_candidate_score(
    request: OpaqueEdgeRequest,
    *,
    frontier: bool,
    before_rank: int,
    after_rank: int,
    outcome_buckets: Sequence[tuple[tuple[str, ...], int, int, int]],
    source_assignment_count_before: int,
    selected_event_version_count_before: int,
    global_version_mass_before: int,
) -> OpaqueActiveCandidateScore:
    buckets = tuple(sorted(outcome_buckets))
    event_counts = tuple(row[2] for row in buckets)
    global_counts = tuple(row[3] for row in buckets)
    kwargs = {
        "frontier_target_unlock": int(frontier),
        "request_sha256": request.request_sha256,
        "event_token": request.event_token,
        "outcome_bucket_rows": buckets,
        "observed_source_rank_before": before_rank,
        "observed_source_rank_after": after_rank,
        "compatible_outcome_count": len(buckets),
        "source_assignment_count_before": source_assignment_count_before,
        "selected_event_version_count_before": selected_event_version_count_before,
        "global_version_mass_before": global_version_mass_before,
        "worst_case_surviving_event_version_count": max(event_counts),
        "worst_posterior_global_version_product": max(global_counts),
        "exact_restricted_nullity_drop": _AMBIENT_RANK,
        "schema": _SCORE_SCHEMA,
    }
    payload = {"schema": _SCORE_SCHEMA, **{key: value for key, value in kwargs.items() if key != "schema"}}
    return OpaqueActiveCandidateScore(**kwargs, score_sha256=_sha256(payload))


def _compatible_version_digest(value: object) -> str:
    if type(value) is ExactCategoricalRestrictedMapVersion:
        return value.version_sha256
    if type(value) is _ConditionalGlobalVersionBlock:
        return value.block_sha256
    return _sha256(value)


def _make_outcome_branch(
    learner_input: OpaqueActiveLearnerInput,
    edges: Sequence[_KnownEdge],
    request: OpaqueEdgeRequest,
    source_answers: tuple[str, ...],
    outcome: tuple[str, ...],
    *,
    versions_before: Sequence[object],
    versions_after: Sequence[object],
    source_assignment_count_before: int,
    source_assignment_count_after: int,
    selected_event_version_count_before: int,
    selected_event_version_count_after: int,
    global_version_mass_before: int,
    global_version_mass_after: int,
) -> ExactOutcomeBranch:
    source_rows, image_rows = _observed_event_rows(
        learner_input, edges, request.event_token
    )
    before_rank = _rank(source_rows)
    augmented_source = source_rows + (
        _diagnostic_row(source_answers, learner_input.answer_tokens),
    )
    augmented_image = image_rows + (
        _diagnostic_row(outcome, learner_input.answer_tokens),
    )
    kwargs = {
        "target_answers": outcome,
        "augmented_source_rows": augmented_source,
        "augmented_image_rows": augmented_image,
        "source_rank_before": before_rank,
        "source_rank_after": _rank(augmented_source),
        "source_assignment_count_before": source_assignment_count_before,
        "source_assignment_count_after": source_assignment_count_after,
        "selected_event_version_count_before": selected_event_version_count_before,
        "selected_event_version_count_after": selected_event_version_count_after,
        "global_version_mass_before": global_version_mass_before,
        "global_version_mass_after": global_version_mass_after,
        "compatible_versions_before_sha256": _sha256(
            tuple(_compatible_version_digest(value) for value in versions_before)
        ),
        "compatible_versions_after_sha256": _sha256(
            tuple(_compatible_version_digest(value) for value in versions_after)
        ),
        "outcome_is_declared_full_product_row": outcome in _full_product_answers(learner_input),
        "exact_linear_constraints_consistent": _map_is_consistent(
            augmented_source, augmented_image
        ),
        "schema": _BRANCH_SCHEMA,
    }
    payload = {"schema": _BRANCH_SCHEMA, **{key: value for key, value in kwargs.items() if key != "schema"}}
    return ExactOutcomeBranch(**kwargs, branch_sha256=_sha256(payload))


def _choose_next_opaque_edge_core(
    learner_input: OpaqueActiveLearnerInput,
    state: OpaqueActiveDiscoveryState,
) -> OpaqueActiveChoiceCertificate | None:
    """Commit the next opaque request without reading its response.

    Source labels are first closed under the supplied nine-state full-product
    bijection.  Thereafter the learner enumerates every exact restricted map
    whose legal images remain in the nine-row categorical codebook and applies
    the frozen minimax score ``(worst posterior global product, worst selected-
    event bucket, -distinct outcomes, request SHA)``.
    """

    if type(learner_input) is not OpaqueActiveLearnerInput:
        raise TypeError("learner_input must be exact OpaqueActiveLearnerInput")
    if type(state) is not OpaqueActiveDiscoveryState:
        raise TypeError("state must be exact OpaqueActiveDiscoveryState")
    if state.learner_input_sha256 != learner_input.input_sha256:
        raise ValueError("state belongs to a different learner input")
    reconstructed = _make_state(learner_input, state.steps)
    if reconstructed != state:
        raise ValueError("active state fails deterministic material reconstruction")
    acquired = {row.choice.chosen_request.request_sha256 for row in state.steps}
    pending_count = sum(
        row.request_sha256 not in acquired
        for row in learner_input.canonical_candidate_requests
    )
    if (
        state.candidate_score_rows_evaluated + pending_count
        > learner_input.budgets.max_candidate_score_rows
    ):
        raise OpaqueActiveDiscoveryLimitError(
            "candidate-score budget would be crossed by the next selection pass"
        )
    conservative_next_rank_work = (
        2 * pending_count
        + 2 * learner_input.budgets.max_outcome_branches_per_choice
    )
    if (
        state.exact_rank_evaluations + conservative_next_rank_work
        > learner_input.budgets.max_exact_rank_evaluations
    ):
        raise OpaqueActiveDiscoveryLimitError(
            "exact-rank budget would be crossed by the next selection pass"
        )
    known, edges = _known_material(learner_input, state.steps)
    mask_words = _mask_source_words(learner_input)
    assignments = _source_assignments(learner_input, known)
    if (
        len(assignments)
        > learner_input.budgets.max_conditional_assignment_blocks_per_choice
    ):
        raise OpaqueActiveDiscoveryLimitError(
            "conditional source-assignment block budget exceeded"
        )
    planned_basis_candidates_by_assignment = _planned_basis_image_candidates(
        learner_input, known, assignments
    )
    planned_basis_candidates = sum(planned_basis_candidates_by_assignment)
    if (
        planned_basis_candidates
        > learner_input.budgets.max_basis_image_candidates_per_choice
    ):
        raise OpaqueActiveDiscoveryLimitError(
            "categorical basis-image enumeration budget would be exceeded"
        )
    if (
        max(planned_basis_candidates_by_assignment, default=0)
        > learner_input.budgets.max_materialized_versions_per_assignment
    ):
        raise OpaqueActiveDiscoveryLimitError(
            "per-assignment categorical version-row ceiling would be exceeded"
        )
    source_incomplete = any(word not in known for word in mask_words)
    score_rows: list[tuple[OpaqueActiveCandidateScore, OpaqueEdgeRequest, dict[tuple[str, ...], tuple[object, ...]]]] = []
    rank_evaluations = 0
    if source_incomplete:
        conditional_blocks = _conditional_global_blocks(
            learner_input, known, edges, assignments
        )
        global_mass_before = sum(row.global_version_mass for row in conditional_blocks)
        for request in learner_input.canonical_candidate_requests:
            if request.request_sha256 in acquired or request.source_word not in known:
                continue
            if request.program not in mask_words or request.program in known:
                continue
            source_rows, _ = _observed_event_rows(
                learner_input, edges, request.event_token
            )
            before_rank = _rank(source_rows)
            after_rank = _rank(
                source_rows
                + (_diagnostic_row(known[request.source_word], learner_input.answer_tokens),)
            )
            rank_evaluations += 2
            if after_rank != before_rank + 1:
                continue
            partitions: dict[tuple[str, ...], list[_ConditionalGlobalVersionBlock]] = {}
            for block in conditional_blocks:
                outcome = dict(block.source_assignment)[request.program]
                partitions.setdefault(outcome, []).append(block)
            selected_event_before = sum(
                block.event_count(request.event_token) for block in conditional_blocks
            )
            score = _make_candidate_score(
                request,
                frontier=True,
                before_rank=before_rank,
                after_rank=after_rank,
                outcome_buckets=tuple(
                    (
                        outcome,
                        len(rows),
                        sum(block.event_count(request.event_token) for block in rows),
                        sum(block.global_version_mass for block in rows),
                    )
                    for outcome, rows in partitions.items()
                ),
                source_assignment_count_before=len(conditional_blocks),
                selected_event_version_count_before=selected_event_before,
                global_version_mass_before=global_mass_before,
            )
            score_rows.append(
                (score, request, {key: tuple(value) for key, value in partitions.items()})
            )
        frontier_singletons = [
            row for row in score_rows if row[0].compatible_outcome_count == 1
        ]
        if frontier_singletons:
            score_rows = sorted(
                frontier_singletons, key=lambda row: row[0].request_sha256
            )
        else:
            score_rows.sort(
                key=lambda row: (
                    -row[0].compatible_outcome_count,
                    row[0].request_sha256,
                )
            )
    else:
        versions_by_event = _filtered_versions(learner_input, known, edges)
        if all(len(rows) == 1 for rows in versions_by_event.values()):
            return None
        global_version_count = 1
        for rows in versions_by_event.values():
            global_version_count *= len(rows)
        singleton_rows: list[
            tuple[
                OpaqueActiveCandidateScore,
                OpaqueEdgeRequest,
                dict[tuple[str, ...], tuple[object, ...]],
            ]
        ] = []
        for request in learner_input.canonical_candidate_requests:
            if request.request_sha256 in acquired or request.source_word not in known:
                continue
            event_versions = versions_by_event[request.event_token]
            if len(event_versions) == 1:
                continue
            request_index = event_versions[0].legal_request_sha256s.index(
                request.request_sha256
            )
            partitions: dict[tuple[str, ...], list[object]] = {}
            for version in event_versions:
                outcome = version.predicted_target_answers[request_index]
                partitions.setdefault(outcome, []).append(version)
            source_rows, _ = _observed_event_rows(
                learner_input, edges, request.event_token
            )
            before_rank = _rank(source_rows)
            after_rank = _rank(
                source_rows
                + (_diagnostic_row(known[request.source_word], learner_input.answer_tokens),)
            )
            rank_evaluations += 2
            if after_rank != before_rank + 1:
                if len(partitions) == 1:
                    # A singleton on an already observed source direction is
                    # merely a redundant word label, not an independent
                    # structural acquisition needed for operator closure.
                    continue
                raise ValueError(
                    "categorical map disagreement occurred without a new exact source direction"
                )
            score = _make_candidate_score(
                request,
                frontier=False,
                before_rank=before_rank,
                after_rank=after_rank,
                outcome_buckets=tuple(
                    (
                        outcome,
                        1,
                        len(rows),
                        len(rows)
                        * (global_version_count // len(event_versions)),
                    )
                    for outcome, rows in partitions.items()
                ),
                source_assignment_count_before=1,
                selected_event_version_count_before=len(event_versions),
                global_version_mass_before=global_version_count,
            )
            scored = (
                score,
                request,
                {key: tuple(value) for key, value in partitions.items()},
            )
            if len(partitions) == 1:
                singleton_rows.append(scored)
            else:
                score_rows.append(scored)
        if singleton_rows:
            # Exact singleton propagation is a fixed point applied before any
            # ambiguous membership choice.  It consumes no returned labels.
            score_rows = sorted(
                singleton_rows, key=lambda row: row[0].request_sha256
            )
        else:
            score_rows.sort(
                key=lambda row: (
                    row[0].worst_posterior_global_version_product,
                    row[0].worst_case_surviving_event_version_count,
                    -row[0].compatible_outcome_count,
                    row[0].request_sha256,
                )
            )
    if not score_rows:
        if learner_input.single_candidate_removal_negative_control:
            raise OpaqueActiveCandidatePoolExhaustedError(
                "negative-control candidate pool exhausted before identification"
            )
        raise ValueError("active version space is incomplete but no opaque acquisition is eligible")
    chosen_score, request, partitions = score_rows[0]
    versions_before = tuple(
        sorted(
            (item for values in partitions.values() for item in values),
            key=_compatible_version_digest,
        )
    )
    branches = tuple(
        _make_outcome_branch(
            learner_input,
            edges,
            request,
            known[request.source_word],
            outcome,
            versions_before=versions_before,
            versions_after=tuple(sorted(rows, key=_compatible_version_digest)),
            source_assignment_count_before=chosen_score.source_assignment_count_before,
            source_assignment_count_after=next(
                row[1]
                for row in chosen_score.outcome_bucket_rows
                if row[0] == outcome
            ),
            selected_event_version_count_before=chosen_score.selected_event_version_count_before,
            selected_event_version_count_after=next(
                row[2]
                for row in chosen_score.outcome_bucket_rows
                if row[0] == outcome
            ),
            global_version_mass_before=chosen_score.global_version_mass_before,
            global_version_mass_after=next(
                row[3]
                for row in chosen_score.outcome_bucket_rows
                if row[0] == outcome
            ),
        )
        for outcome, rows in sorted(partitions.items())
    )
    exact_rank_evaluations = rank_evaluations + 2 * len(branches)
    next_call = state.active_call_count + 1
    kwargs: dict[str, object] = {
        "learner_input_sha256": learner_input.input_sha256,
        "prior_state_sha256": state.state_sha256,
        "prior_response_history_sha256": state.response_history_sha256,
        "canonical_pending_pool_sha256": state.canonical_pending_pool_sha256,
        "known_word_table_sha256": state.known_word_table_sha256,
        "known_observed_edge_sha256": state.known_observed_edge_sha256,
        "decision_ordinal": len(state.steps) + 1,
        "next_membership_call_ordinal": next_call,
        "chosen_request": request,
        "eligible_scores": tuple(row[0] for row in score_rows),
        "exact_outcome_branches": branches,
        "selection_rule": "opaque_source_frontier_then_exact_categorical_map_minimax_then_request_sha",
        "source_assignment_incomplete_before_choice": source_incomplete,
        "singleton_fixed_point_applied": chosen_score.compatible_outcome_count == 1,
        "chosen_source_available": request.source_word in known,
        "chosen_target_expands_frontier": bool(chosen_score.frontier_target_unlock),
        "requires_membership_response": len(branches) > 1,
        "unique_structurally_inferred_answers": (
            None if len(branches) > 1 else branches[0].target_answers
        ),
        "selected_before_current_response": True,
        "current_response_labels_used": False,
        "future_response_labels_used": False,
        "sealed_answers_used": False,
        "controller_candidate_order_used": False,
        "semantic_roles_used": False,
        "deterministic_sha_tiebreak": True,
        "rank_evaluations_this_choice": exact_rank_evaluations,
        "conditional_assignment_blocks_evaluated": len(assignments),
        "basis_image_candidates_planned": planned_basis_candidates,
        "max_potential_version_rows_per_assignment": max(
            planned_basis_candidates_by_assignment
        ),
        "exact_version_enumeration_uses_cached_numeric_templates": True,
        "schema": _CHOICE_SCHEMA,
    }
    payload = {"schema": _CHOICE_SCHEMA, **{key: value for key, value in kwargs.items() if key != "schema"}}
    payload["chosen_request"] = _request_payload(request)
    payload["eligible_scores"] = [row._payload(True) for row in kwargs["eligible_scores"]]  # type: ignore[union-attr]
    payload["exact_outcome_branches"] = [row._payload(True) for row in branches]
    return OpaqueActiveChoiceCertificate(
        **kwargs,  # type: ignore[arg-type]
        choice_sha256=_sha256(payload),
    )


_VALIDATED_STATE_CACHE: set[tuple[str, str]] = set()


def validate_opaque_active_state(
    learner_input: OpaqueActiveLearnerInput,
    state: OpaqueActiveDiscoveryState,
) -> None:
    """Authoritatively replay every pre-response choice against its prefix."""

    if type(learner_input) is not OpaqueActiveLearnerInput:
        raise TypeError("learner_input must be exact OpaqueActiveLearnerInput")
    if type(state) is not OpaqueActiveDiscoveryState:
        raise TypeError("state must be exact OpaqueActiveDiscoveryState")
    if state.learner_input_sha256 != learner_input.input_sha256:
        raise ValueError("state belongs to a different learner input")
    cache_key = (learner_input.input_sha256, state.state_sha256)
    if cache_key in _VALIDATED_STATE_CACHE:
        return
    # Normal staged use validates each predecessor before incorporating one
    # more step.  Start from the longest such authoritative prefix so replay
    # work is separately bounded and does not silently re-enumerate the whole
    # transcript on every call.  An externally reconstructed certificate with
    # no cached ancestry still receives one bounded full replay.
    prefix = _make_state(learner_input, ())
    replay_start = 0
    for index in range(len(state.steps), -1, -1):
        candidate = _make_state(learner_input, state.steps[:index])
        if (learner_input.input_sha256, candidate.state_sha256) in _VALIDATED_STATE_CACHE:
            prefix = candidate
            replay_start = index
            break
    replay_count = len(state.steps) - replay_start
    if replay_count > learner_input.budgets.max_validation_replay_decisions:
        raise OpaqueActiveDiscoveryLimitError(
            "uncached causal validation replay-decision ceiling would be exceeded"
        )
    for row in state.steps[replay_start:]:
        expected = _choose_next_opaque_edge_core(learner_input, prefix)
        if expected is None or row.choice != expected:
            raise ValueError("active history choice fails causal prefix reconstruction")
        if type(row) is OpaqueActiveStep:
            if not expected.requires_membership_response:
                raise ValueError("response was released for a singleton inference")
            if row.response.prior_choice_sha256 != expected.choice_sha256:
                raise ValueError("response does not bind its causal choice")
            if row.response.target_answers not in {
                branch.target_answers for branch in expected.exact_outcome_branches
            }:
                raise ValueError("response is outside the prefix-compatible version space")
        else:
            if expected.requires_membership_response:
                raise ValueError("ambiguous prefix choice was laundered as inference")
            if (
                row.inference.inferred_target_answers
                != expected.unique_structurally_inferred_answers
            ):
                raise ValueError("inference differs from the unique prefix branch")
        prefix = _make_state(learner_input, prefix.steps + (row,))
    if prefix != state:
        raise ValueError("active state fails authoritative causal reconstruction")
    _VALIDATED_STATE_CACHE.add(cache_key)


def choose_next_opaque_edge(
    learner_input: OpaqueActiveLearnerInput,
    state: OpaqueActiveDiscoveryState,
) -> OpaqueActiveChoiceCertificate | None:
    validate_opaque_active_state(learner_input, state)
    return _choose_next_opaque_edge_core(learner_input, state)


def make_opaque_active_response(
    learner_input: OpaqueActiveLearnerInput,
    state: OpaqueActiveDiscoveryState,
    choice: OpaqueActiveChoiceCertificate,
    target_answers: tuple[str, ...],
) -> OpaqueActiveMembershipResponse:
    expected = choose_next_opaque_edge(learner_input, state)
    if expected is None or type(choice) is not OpaqueActiveChoiceCertificate or choice != expected:
        raise ValueError("response requires the exact next pre-response choice commitment")
    if not choice.requires_membership_response:
        raise ValueError("a singleton branch must be incorporated as structural inference")
    answers = _validate_answers("target_answers", target_answers, learner_input.answer_tokens)
    if answers not in {row.target_answers for row in choice.exact_outcome_branches}:
        raise ValueError("response is outside the exact compatible outcome version space")
    if state.active_call_count >= learner_input.budgets.max_active_calls:
        raise OpaqueActiveDiscoveryLimitError("active-call budget exceeded before response")
    if state.returned_categorical_token_count + 2 > learner_input.budgets.max_returned_categorical_tokens:
        raise OpaqueActiveDiscoveryLimitError("returned-label budget exceeded before response")
    kwargs = {
        "request": choice.chosen_request,
        "target_answers": answers,
        "response_ordinal": choice.next_membership_call_ordinal,
        "prior_choice_sha256": choice.choice_sha256,
        "returned_categorical_token_count": 2,
        "schema": _RESPONSE_SCHEMA,
    }
    payload = {"schema": _RESPONSE_SCHEMA, **{key: value for key, value in kwargs.items() if key != "schema"}}
    payload["request"] = _request_payload(choice.chosen_request)
    return OpaqueActiveMembershipResponse(
        **kwargs,
        response_sha256=_sha256(payload),
    )


def incorporate_opaque_response(
    learner_input: OpaqueActiveLearnerInput,
    state: OpaqueActiveDiscoveryState,
    choice: OpaqueActiveChoiceCertificate,
    response: OpaqueActiveMembershipResponse,
) -> OpaqueActiveDiscoveryState:
    expected = choose_next_opaque_edge(learner_input, state)
    if expected is None or choice != expected:
        raise ValueError("choice is not the exact next pre-response commitment")
    if type(response) is not OpaqueActiveMembershipResponse:
        raise TypeError("response must be exact OpaqueActiveMembershipResponse")
    if response.prior_choice_sha256 != choice.choice_sha256 or response.request != choice.chosen_request:
        raise ValueError("response does not answer the committed request")
    if response.target_answers not in {row.target_answers for row in choice.exact_outcome_branches}:
        raise ValueError("response target is outside the exact compatible branches")
    payload = {
        "schema": _STEP_SCHEMA,
        "choice": choice._payload(True),
        "response": response._payload(True),
    }
    step = OpaqueActiveStep(
        choice=choice,
        response=response,
        step_sha256=_sha256(payload),
    )
    extended = _make_state(
        learner_input,
        state.steps + (step,),
    )
    _VALIDATED_STATE_CACHE.add((learner_input.input_sha256, extended.state_sha256))
    return extended


def incorporate_opaque_structural_inference(
    learner_input: OpaqueActiveLearnerInput,
    state: OpaqueActiveDiscoveryState,
    choice: OpaqueActiveChoiceCertificate,
) -> OpaqueActiveDiscoveryState:
    expected = choose_next_opaque_edge(learner_input, state)
    if expected is None or choice != expected:
        raise ValueError("choice is not the exact next structural-inference commitment")
    if choice.requires_membership_response or choice.unique_structurally_inferred_answers is None:
        raise ValueError("ambiguous choice cannot be structurally inferred")
    if state.structural_inference_count >= learner_input.budgets.max_structural_inferences:
        raise OpaqueActiveDiscoveryLimitError("structural-inference budget exceeded")
    inference_kind = (
        "full_product_source_bijection_singleton"
        if choice.source_assignment_incomplete_before_choice
        else "categorical_restricted_map_singleton"
    )
    inference_kwargs = {
        "choice_sha256": choice.choice_sha256,
        "request": choice.chosen_request,
        "inferred_target_answers": choice.unique_structurally_inferred_answers,
        "decision_ordinal": choice.decision_ordinal,
        "inference_kind": inference_kind,
        "returned_categorical_token_count": 0,
        "schema": _INFERENCE_SCHEMA,
    }
    inference_payload = {
        "schema": _INFERENCE_SCHEMA,
        **{key: value for key, value in inference_kwargs.items() if key != "schema"},
    }
    inference_payload["request"] = _request_payload(choice.chosen_request)
    inference = OpaqueStructuralInference(
        **inference_kwargs,
        inference_sha256=_sha256(inference_payload),
    )
    step_payload = {
        "schema": _STEP_SCHEMA,
        "choice": choice._payload(True),
        "inference": inference._payload(True),
    }
    step = OpaqueStructuralInferenceStep(
        choice=choice,
        inference=inference,
        step_sha256=_sha256(step_payload),
    )
    extended = _make_state(
        learner_input,
        state.steps + (step,),
    )
    _VALIDATED_STATE_CACHE.add((learner_input.input_sha256, extended.state_sha256))
    return extended


def _right_nullspace_basis(
    rows: Sequence[Sequence[Rational | Fraction | int]],
) -> RationalMatrix:
    if not rows:
        return _matrix(
            tuple(Rational(int(i == j)) for j in range(_AMBIENT_RANK))
            for i in range(_AMBIENT_RANK)
        )
    work = [[_as_fraction(value) for value in row] for row in rows]
    width = len(work[0])
    pivot_columns: list[int] = []
    rank = 0
    for column in range(width):
        pivot = next((row for row in range(rank, len(work)) if work[row][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        value = work[rank][column]
        work[rank] = [entry / value for entry in work[rank]]
        for row in range(len(work)):
            if row == rank or not work[row][column]:
                continue
            factor = work[row][column]
            work[row] = [
                entry - factor * pivot_entry
                for entry, pivot_entry in zip(work[row], work[rank], strict=True)
            ]
        pivot_columns.append(column)
        rank += 1
        if rank == len(work):
            break
    free_columns = tuple(column for column in range(width) if column not in pivot_columns)
    basis: list[tuple[Fraction, ...]] = []
    for free in free_columns:
        vector = [Fraction(0) for _ in range(width)]
        vector[free] = Fraction(1)
        for row, pivot in reversed(tuple(enumerate(pivot_columns))):
            vector[pivot] = -sum(
                work[row][column] * vector[column]
                for column in free_columns
            )
        basis.append(tuple(vector))
    if not basis:
        # Some mathematically compatible counterfactual labelings make one
        # guarded legal domain span all five diagnostic directions.  Its
        # off-domain annihilator is the exact empty basis; the categorical
        # legality mask still keeps the public operator partial.
        return ()
    return _matrix(basis)


@dataclass(frozen=True)
class AutonomousRestrictedOperator:
    event_token: str
    legal_request_sha256s: tuple[str, ...]
    legal_source_answers: tuple[tuple[str, ...], ...]
    legal_source_rows: RationalMatrix
    domain_basis_indices: tuple[int, ...]
    domain_basis_rows: RationalMatrix
    image_basis_rows: RationalMatrix
    off_domain_annihilator_basis: RationalMatrix
    legal_domain_rank: int
    direct_passive_source_rank: int
    response_source_rank_increment: int
    inference_source_rank_increment: int
    final_observed_source_rank: int
    raw_restricted_linear_nullity: int
    categorical_version_count: int
    total_extension_nullity: int
    categorical_restricted_map_identified: bool
    raw_linear_map_identified: bool
    off_domain_extension_identified: bool
    total_operator: None
    operator_sha256: str
    schema: str = _OPERATOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _OPERATOR_SCHEMA:
            raise ValueError("unknown autonomous restricted-operator schema")
        _require_token("event_token", self.event_token)
        if len(self.legal_request_sha256s) != len(self.legal_source_answers):
            raise ValueError("legal request/source-answer counts differ")
        for digest in self.legal_request_sha256s:
            _require_sha256("legal request digest", digest)
        for answers in self.legal_source_answers:
            _validate_answers("legal source answers", answers)
        if len(self.legal_source_rows) != len(self.legal_request_sha256s):
            raise ValueError("legal source-row count mismatch")
        domain_rank = _rank(self.legal_source_rows)
        if self.legal_domain_rank != domain_rank:
            raise ValueError("legal-domain rank mismatch")
        if self.domain_basis_indices != _rank_profile(self.legal_source_rows)[1]:
            raise ValueError("domain-basis pivot profile mismatch")
        expected_basis = tuple(self.legal_source_rows[index] for index in self.domain_basis_indices)
        if self.domain_basis_rows != expected_basis:
            raise ValueError("domain basis does not match legal source pivots")
        if len(self.image_basis_rows) != domain_rank:
            raise ValueError("image basis must have one row per domain direction")
        if _rank(self.off_domain_annihilator_basis) != _AMBIENT_RANK - domain_rank:
            raise ValueError("off-domain annihilator basis rank mismatch")
        for source in self.domain_basis_rows:
            for annihilator in self.off_domain_annihilator_basis:
                if sum(
                    left.as_fraction() * right.as_fraction()
                    for left, right in zip(source, annihilator, strict=True)
                ):
                    raise ValueError("off-domain annihilator does not vanish on legal domain")
        direct = _plain_int("direct_passive_source_rank", self.direct_passive_source_rank)
        response = _plain_int("response_source_rank_increment", self.response_source_rank_increment)
        inference = _plain_int("inference_source_rank_increment", self.inference_source_rank_increment)
        if self.final_observed_source_rank != direct + response + inference:
            raise ValueError("operator rank-provenance arithmetic mismatch")
        if self.raw_restricted_linear_nullity != _AMBIENT_RANK * (
            domain_rank - self.final_observed_source_rank
        ):
            raise ValueError("raw restricted-linear nullity mismatch")
        if self.categorical_version_count != 1:
            raise ValueError("final categorical restricted-map version must be unique")
        if self.total_extension_nullity != _AMBIENT_RANK * (_AMBIENT_RANK - domain_rank):
            raise ValueError("total-extension nullity mismatch")
        if not _require_bool(
            "categorical_restricted_map_identified",
            self.categorical_restricted_map_identified,
        ):
            raise ValueError("final categorical restricted map must be identified")
        if _require_bool("raw_linear_map_identified", self.raw_linear_map_identified) != (
            self.final_observed_source_rank == domain_rank
        ):
            raise ValueError("raw-linear identification flag mismatch")
        if _require_bool(
            "off_domain_extension_identified", self.off_domain_extension_identified
        ) != (domain_rank == _AMBIENT_RANK):
            raise ValueError("off-domain identification must follow whether a complement exists")
        if self.total_operator is not None:
            raise ValueError("Contract B forbids a total operator")
        if _require_sha256("operator_sha256", self.operator_sha256) != _sha256(
            self._payload(False)
        ):
            raise ValueError("autonomous restricted-operator digest mismatch")

    def _payload(self, include_operator_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "event_token": self.event_token,
            "legal_request_sha256s": self.legal_request_sha256s,
            "legal_source_answers": self.legal_source_answers,
            "legal_source_rows": self.legal_source_rows,
            "domain_basis_indices": self.domain_basis_indices,
            "domain_basis_rows": self.domain_basis_rows,
            "image_basis_rows": self.image_basis_rows,
            "off_domain_annihilator_basis": self.off_domain_annihilator_basis,
            "legal_domain_rank": self.legal_domain_rank,
            "direct_passive_source_rank": self.direct_passive_source_rank,
            "response_source_rank_increment": self.response_source_rank_increment,
            "inference_source_rank_increment": self.inference_source_rank_increment,
            "final_observed_source_rank": self.final_observed_source_rank,
            "raw_restricted_linear_nullity": self.raw_restricted_linear_nullity,
            "categorical_version_count": self.categorical_version_count,
            "total_extension_nullity": self.total_extension_nullity,
            "categorical_restricted_map_identified": self.categorical_restricted_map_identified,
            "raw_linear_map_identified": self.raw_linear_map_identified,
            "off_domain_extension_identified": self.off_domain_extension_identified,
            "total_operator": self.total_operator,
        }
        if include_operator_sha:
            payload["operator_sha256"] = self.operator_sha256
        return payload

    def apply_row(self, source_row: RationalVector) -> RationalVector:
        coordinates = _row_coordinates(source_row, self.domain_basis_rows)
        return tuple(
            Rational(
                sum(
                    coordinates[index]
                    * self.image_basis_rows[index][column].as_fraction()
                    for index in range(self.legal_domain_rank)
                ).numerator,
                sum(
                    coordinates[index]
                    * self.image_basis_rows[index][column].as_fraction()
                    for index in range(self.legal_domain_rank)
                ).denominator,
            )
            for column in range(_AMBIENT_RANK)
        )


@dataclass(frozen=True)
class AutonomousPartialModel:
    event_tokens: tuple[str, ...]
    query_tokens: tuple[str, ...]
    answer_tokens: tuple[str, ...]
    diagnostic_codebook: tuple[tuple[tuple[str, ...], RationalVector], ...]
    mask_source_answer_rows: tuple[tuple[OpaqueWord, tuple[str, ...]], ...]
    defined_categorical_guard_pairs: tuple[tuple[tuple[str, ...], str], ...]
    operators: tuple[AutonomousRestrictedOperator, ...]
    ambient_rank: int
    diagnostic_rank: int
    categorical_state_count: int
    restricted_maps_complete: bool
    total_operator: None
    model_sha256: str
    schema: str = _MODEL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _MODEL_SCHEMA:
            raise ValueError("unknown autonomous partial-model schema")
        if len(self.event_tokens) != _EVENT_COUNT or len(set(self.event_tokens)) != _EVENT_COUNT:
            raise ValueError("model must contain ten event tokens")
        if len(self.query_tokens) != _QUERY_COUNT or len(self.answer_tokens) != _ANSWER_COUNT:
            raise ValueError("model diagnostic vocabulary mismatch")
        if self.ambient_rank != _AMBIENT_RANK or self.diagnostic_rank != _AMBIENT_RANK:
            raise ValueError("model must use the supplied exact rank-five diagnostic gauge")
        if self.categorical_state_count != _STATE_COUNT:
            raise ValueError("model must bind all nine categorical states")
        if len(self.diagnostic_codebook) != _STATE_COUNT:
            raise ValueError("diagnostic codebook must contain nine rows")
        expected_answers = tuple(sorted(product(self.answer_tokens, repeat=_QUERY_COUNT)))
        if tuple(answers for answers, _ in self.diagnostic_codebook) != expected_answers:
            raise ValueError("diagnostic codebook must be the exact full product")
        if _rank(tuple(row for _, row in self.diagnostic_codebook)) != _AMBIENT_RANK:
            raise ValueError("diagnostic codebook must have exact rank five")
        if len(self.mask_source_answer_rows) != _STATE_COUNT:
            raise ValueError("model must bind nine mask-source representatives")
        if len({word for word, _ in self.mask_source_answer_rows}) != _STATE_COUNT:
            raise ValueError("mask-source representatives must be unique")
        if len({answers for _, answers in self.mask_source_answer_rows}) != _STATE_COUNT:
            raise ValueError("mask-source diagnostics must bijectively cover the full product")
        if len(self.defined_categorical_guard_pairs) != _LEGAL_EDGE_COUNT:
            raise ValueError("model must bind exactly 44 categorical guard pairs")
        if tuple(sorted(self.defined_categorical_guard_pairs)) != self.defined_categorical_guard_pairs:
            raise ValueError("categorical guard pairs must use canonical order")
        if len(set(self.defined_categorical_guard_pairs)) != _LEGAL_EDGE_COUNT:
            raise ValueError("categorical guard pairs must be unique")
        codebook_answers = {answers for answers, _ in self.diagnostic_codebook}
        if any(
            answers not in codebook_answers or event_token not in self.event_tokens
            for answers, event_token in self.defined_categorical_guard_pairs
        ):
            raise ValueError("categorical guard contains an undeclared state/event token")
        if len(codebook_answers) * len(self.event_tokens) - len(
            self.defined_categorical_guard_pairs
        ) != _UNDEFINED_EDGE_COUNT:
            raise ValueError("categorical guard complement must contain exactly 46 pairs")
        if len(self.operators) != _EVENT_COUNT:
            raise ValueError("model must contain ten restricted operators")
        if tuple(row.event_token for row in self.operators) != tuple(sorted(self.event_tokens)):
            raise ValueError("operators must use canonical event-token order")
        guard_by_event = {
            token: {answers for answers, event in self.defined_categorical_guard_pairs if event == token}
            for token in self.event_tokens
        }
        if any(
            guard_by_event[row.event_token] != set(row.legal_source_answers)
            for row in self.operators
        ):
            raise ValueError("categorical guard does not match restricted-operator legal sources")
        complete = all(row.categorical_restricted_map_identified for row in self.operators)
        if _require_bool("restricted_maps_complete", self.restricted_maps_complete) != complete:
            raise ValueError("restricted-map completion flag mismatch")
        if self.total_operator is not None:
            raise ValueError("model cannot contain a total operator")
        if _require_sha256("model_sha256", self.model_sha256) != _sha256(self._payload(False)):
            raise ValueError("autonomous partial-model digest mismatch")

    def _payload(self, include_model_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "event_tokens": self.event_tokens,
            "query_tokens": self.query_tokens,
            "answer_tokens": self.answer_tokens,
            "diagnostic_codebook": self.diagnostic_codebook,
            "mask_source_answer_rows": self.mask_source_answer_rows,
            "defined_categorical_guard_pairs": self.defined_categorical_guard_pairs,
            "operators": [row._payload(True) for row in self.operators],
            "ambient_rank": self.ambient_rank,
            "diagnostic_rank": self.diagnostic_rank,
            "categorical_state_count": self.categorical_state_count,
            "restricted_maps_complete": self.restricted_maps_complete,
            "total_operator": self.total_operator,
        }
        if include_model_sha:
            payload["model_sha256"] = self.model_sha256
        return payload

    def predict_defined_suffix(
        self,
        source_answers: tuple[str, ...],
        suffix: Sequence[str],
        *,
        max_events: int = 4_096,
    ) -> tuple[str, ...]:
        answers = _validate_answers("source_answers", source_answers, self.answer_tokens)
        if not isinstance(suffix, (tuple, list)):
            raise TypeError("suffix must be a tuple or list")
        if len(suffix) > max_events:
            raise OpaqueActiveDiscoveryLimitError("suffix prediction budget exceeded")
        row_by_answers = dict(self.diagnostic_codebook)
        answers_by_row = {row: value for value, row in self.diagnostic_codebook}
        current = row_by_answers[answers]
        current_answers = answers
        operators = {row.event_token: row for row in self.operators}
        exact_guard = set(self.defined_categorical_guard_pairs)
        for event_token in suffix:
            _require_token("suffix event token", event_token)
            if event_token not in operators:
                raise ValueError("suffix contains an undeclared event token")
            if (current_answers, event_token) not in exact_guard:
                raise ValueError("suffix traverses a pair excluded by the exact definedness mask")
            current = operators[event_token].apply_row(current)
            if current not in answers_by_row:
                raise ValueError("restricted operator left the declared categorical state space")
            current_answers = answers_by_row[current]
        return current_answers


@dataclass(frozen=True, order=True)
class GuardedDefinedTransition:
    request_sha256: str
    event_token: str
    source_answers: tuple[str, ...]
    predicted_target_answers: tuple[str, ...]
    admitted_by_exact_categorical_guard: bool
    prediction_is_unique_categorical_version: bool


@dataclass(frozen=True, order=True)
class GuardedUndefinedPair:
    request_sha256: str
    event_token: str
    source_answers: tuple[str, ...]
    rejected_by_exact_categorical_guard: bool
    rejected_by_restricted_domain_span_control: bool


@dataclass(frozen=True)
class GuardedLanguageCertificate:
    model_sha256: str
    initial_word: OpaqueWord
    initial_answers: tuple[str, ...]
    diagnostic_readout_rows: tuple[tuple[tuple[str, ...], RationalVector], ...]
    defined_transitions: tuple[GuardedDefinedTransition, ...]
    undefined_pairs: tuple[GuardedUndefinedPair, ...]
    state_count: int
    defined_transition_count: int
    undefined_pair_count: int
    exact_guard_rejection_count: int
    restricted_domain_span_control_rejection_count: int
    all_defined_targets_are_declared_states: bool
    all_undefined_pairs_rejected: bool
    all_undefined_pairs_rejected_by_exact_guard: bool
    all_undefined_pairs_outside_restricted_domain_span: bool
    arbitrary_length_legal_suffix_induction: bool
    arbitrary_suffix_without_definedness_guard_claimed: bool
    total_wfa_claimed: bool
    unqueried_edge_answers_used_to_fit_or_select: bool
    certificate_sha256: str
    schema: str = _LANGUAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _LANGUAGE_SCHEMA:
            raise ValueError("unknown guarded-language certificate schema")
        _require_sha256("model_sha256", self.model_sha256)
        if self.initial_word != ():
            raise ValueError("guarded-language initial representative must be the empty word")
        _validate_answers("initial_answers", self.initial_answers)
        if len(self.diagnostic_readout_rows) != _STATE_COUNT:
            raise ValueError("guarded certificate must bind nine exact readout rows")
        if len(self.defined_transitions) != _LEGAL_EDGE_COUNT:
            raise ValueError("guarded certificate must bind all 44 defined transitions")
        if len(self.undefined_pairs) != _UNDEFINED_EDGE_COUNT:
            raise ValueError("guarded certificate must bind all 46 undefined pairs")
        if self.state_count != _STATE_COUNT or self.defined_transition_count != _LEGAL_EDGE_COUNT or self.undefined_pair_count != _UNDEFINED_EDGE_COUNT:
            raise ValueError("guarded-language census mismatch")
        if self.exact_guard_rejection_count != _UNDEFINED_EDGE_COUNT:
            raise ValueError("exact categorical guard must reject all 46 undefined pairs")
        if self.restricted_domain_span_control_rejection_count != _UNDEFINED_EDGE_COUNT:
            raise ValueError("restricted-domain span control must reject all 46 undefined pairs")
        if not all(row.admitted_by_exact_categorical_guard for row in self.defined_transitions):
            raise ValueError("every defined transition must be admitted by the exact guard")
        if not all(row.prediction_is_unique_categorical_version for row in self.defined_transitions):
            raise ValueError("every defined transition must follow the unique categorical map")
        if not all(row.rejected_by_exact_categorical_guard for row in self.undefined_pairs):
            raise ValueError("every undefined pair must be rejected by the exact guard")
        if not all(
            row.rejected_by_restricted_domain_span_control for row in self.undefined_pairs
        ):
            raise ValueError("every undefined pair must pass the independent span-rejection control")
        readout_answers = {answers for answers, _ in self.diagnostic_readout_rows}
        if readout_answers != set(product(sorted({token for answers in readout_answers for token in answers}), repeat=_QUERY_COUNT)):
            raise ValueError("guarded readout rows must form an exact three-label full product")
        defined_pairs = {
            (row.source_answers, row.event_token) for row in self.defined_transitions
        }
        undefined_pairs = {
            (row.source_answers, row.event_token) for row in self.undefined_pairs
        }
        if len(defined_pairs) != _LEGAL_EDGE_COUNT or len(undefined_pairs) != _UNDEFINED_EDGE_COUNT:
            raise ValueError("guarded transition pairs must be unique")
        if defined_pairs.intersection(undefined_pairs):
            raise ValueError("defined and undefined categorical guards overlap")
        event_tokens = {event for _, event in defined_pairs | undefined_pairs}
        if len(event_tokens) != _EVENT_COUNT or defined_pairs | undefined_pairs != {
            (answers, event) for answers in readout_answers for event in event_tokens
        }:
            raise ValueError("guarded rows must partition the exact 9x10 categorical grid")
        if any(
            row.predicted_target_answers not in readout_answers
            for row in self.defined_transitions
        ):
            raise ValueError("defined transition target is outside the readout codebook")
        for name, value, required in (
            ("all_defined_targets_are_declared_states", self.all_defined_targets_are_declared_states, True),
            ("all_undefined_pairs_rejected", self.all_undefined_pairs_rejected, True),
            ("all_undefined_pairs_rejected_by_exact_guard", self.all_undefined_pairs_rejected_by_exact_guard, True),
            ("all_undefined_pairs_outside_restricted_domain_span", self.all_undefined_pairs_outside_restricted_domain_span, True),
            ("arbitrary_length_legal_suffix_induction", self.arbitrary_length_legal_suffix_induction, True),
            ("arbitrary_suffix_without_definedness_guard_claimed", self.arbitrary_suffix_without_definedness_guard_claimed, False),
            ("total_wfa_claimed", self.total_wfa_claimed, False),
            ("unqueried_edge_answers_used_to_fit_or_select", self.unqueried_edge_answers_used_to_fit_or_select, False),
        ):
            if _require_bool(name, value) is not required:
                raise ValueError(f"{name} must be {required}")
        if _require_sha256("certificate_sha256", self.certificate_sha256) != _sha256(
            self._payload(False)
        ):
            raise ValueError("guarded-language certificate digest mismatch")

    def _payload(self, include_certificate_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "model_sha256": self.model_sha256,
            "initial_word": self.initial_word,
            "initial_answers": self.initial_answers,
            "diagnostic_readout_rows": self.diagnostic_readout_rows,
            "defined_transitions": [row.__dict__ for row in self.defined_transitions],
            "undefined_pairs": [row.__dict__ for row in self.undefined_pairs],
            "state_count": self.state_count,
            "defined_transition_count": self.defined_transition_count,
            "undefined_pair_count": self.undefined_pair_count,
            "exact_guard_rejection_count": self.exact_guard_rejection_count,
            "restricted_domain_span_control_rejection_count": self.restricted_domain_span_control_rejection_count,
            "all_defined_targets_are_declared_states": self.all_defined_targets_are_declared_states,
            "all_undefined_pairs_rejected": self.all_undefined_pairs_rejected,
            "all_undefined_pairs_rejected_by_exact_guard": self.all_undefined_pairs_rejected_by_exact_guard,
            "all_undefined_pairs_outside_restricted_domain_span": self.all_undefined_pairs_outside_restricted_domain_span,
            "arbitrary_length_legal_suffix_induction": self.arbitrary_length_legal_suffix_induction,
            "arbitrary_suffix_without_definedness_guard_claimed": self.arbitrary_suffix_without_definedness_guard_claimed,
            "total_wfa_claimed": self.total_wfa_claimed,
            "unqueried_edge_answers_used_to_fit_or_select": self.unqueried_edge_answers_used_to_fit_or_select,
        }
        if include_certificate_sha:
            payload["certificate_sha256"] = self.certificate_sha256
        return payload


@dataclass(frozen=True)
class OpaqueActiveNotIdentifiedResult:
    """A bounded causal stop emitted before one additional answer is opened."""

    learner_input_sha256: str
    final_state: OpaqueActiveDiscoveryState
    blocked_choice: OpaqueActiveChoiceCertificate | None
    identification_status: str
    stop_reason: str
    active_call_count: int
    structural_inference_count: int
    returned_categorical_token_count: int
    unopened_candidate_count: int
    minimum_unopened_candidate_count: int
    blocked_choice_response_provider_called: bool
    blocked_choice_answer_opened: bool
    result_sha256: str
    schema: str = _NOT_IDENTIFIED_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _NOT_IDENTIFIED_SCHEMA:
            raise ValueError("unknown not-identified result schema")
        _require_sha256("learner_input_sha256", self.learner_input_sha256)
        if type(self.final_state) is not OpaqueActiveDiscoveryState:
            raise TypeError("not-identified final_state must be exact active state")
        if self.blocked_choice is not None and type(
            self.blocked_choice
        ) is not OpaqueActiveChoiceCertificate:
            raise TypeError("blocked_choice must be an exact committed choice or none")
        if self.final_state.learner_input_sha256 != self.learner_input_sha256:
            raise ValueError("not-identified state belongs to another learner input")
        if self.blocked_choice is not None and (
            self.blocked_choice.learner_input_sha256 != self.learner_input_sha256
        ):
            raise ValueError("blocked choice belongs to another learner input")
        allowed_reasons = {
            "active_call_budget_exhausted",
            "returned_categorical_token_budget_exhausted",
            "structural_inference_budget_exhausted",
            "sealed_candidate_quota_reached",
            "candidate_pool_exhausted_before_identification",
        }
        if self.stop_reason not in allowed_reasons:
            raise ValueError("unknown bounded not-identified stop reason")
        expected_status = (
            "not_identified_candidate_pool_incomplete"
            if self.stop_reason == "candidate_pool_exhausted_before_identification"
            else "not_identified_budget_or_sealed_quota"
        )
        if self.identification_status != expected_status:
            raise ValueError("not-identified status does not match its stop reason")
        if (
            self.stop_reason == "candidate_pool_exhausted_before_identification"
        ) != (self.blocked_choice is None):
            raise ValueError("only candidate-pool exhaustion may omit a blocked choice")
        if self.active_call_count != self.final_state.active_call_count:
            raise ValueError("not-identified active-call count mismatch")
        if self.structural_inference_count != self.final_state.structural_inference_count:
            raise ValueError("not-identified inference count mismatch")
        if (
            self.returned_categorical_token_count
            != self.final_state.returned_categorical_token_count
        ):
            raise ValueError("not-identified returned-label count mismatch")
        _plain_int("unopened_candidate_count", self.unopened_candidate_count)
        _plain_int(
            "minimum_unopened_candidate_count",
            self.minimum_unopened_candidate_count,
        )
        if self.unopened_candidate_count < self.minimum_unopened_candidate_count:
            raise ValueError("not-identified stop already crossed the sealed quota")
        for name, value in (
            (
                "blocked_choice_response_provider_called",
                self.blocked_choice_response_provider_called,
            ),
            ("blocked_choice_answer_opened", self.blocked_choice_answer_opened),
        ):
            if _require_bool(name, value):
                raise ValueError(f"{name} must be false at a fail-before-answer stop")
        if _require_sha256("result_sha256", self.result_sha256) != _sha256(
            self._payload(False)
        ):
            raise ValueError("not-identified result digest mismatch")

    def _payload(self, include_result_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "learner_input_sha256": self.learner_input_sha256,
            "final_state": self.final_state._payload(True),
            "blocked_choice": (
                None
                if self.blocked_choice is None
                else self.blocked_choice._payload(True)
            ),
            "identification_status": self.identification_status,
            "stop_reason": self.stop_reason,
            "active_call_count": self.active_call_count,
            "structural_inference_count": self.structural_inference_count,
            "returned_categorical_token_count": self.returned_categorical_token_count,
            "unopened_candidate_count": self.unopened_candidate_count,
            "minimum_unopened_candidate_count": self.minimum_unopened_candidate_count,
            "blocked_choice_response_provider_called": self.blocked_choice_response_provider_called,
            "blocked_choice_answer_opened": self.blocked_choice_answer_opened,
        }
        if include_result_sha:
            payload["result_sha256"] = self.result_sha256
        return payload


def _enforce_result_certificate_budget(
    learner_input: OpaqueActiveLearnerInput,
    result: "AutonomousPartialOperatorResult | OpaqueActiveNotIdentifiedResult",
) -> None:
    size = len(_canonical_bytes(result._payload(True)))
    if size > learner_input.budgets.max_certificate_bytes:
        raise OpaqueActiveDiscoveryLimitError(
            "canonical final-result certificate exceeds byte budget"
        )


def _make_not_identified_result(
    learner_input: OpaqueActiveLearnerInput,
    state: OpaqueActiveDiscoveryState,
    choice: OpaqueActiveChoiceCertificate | None,
    *,
    stop_reason: str,
    minimum_unopened_candidate_count: int,
) -> OpaqueActiveNotIdentifiedResult:
    unopened = len(learner_input.canonical_candidate_requests) - len(state.steps)
    kwargs = {
        "learner_input_sha256": learner_input.input_sha256,
        "final_state": state,
        "blocked_choice": choice,
        "identification_status": (
            "not_identified_candidate_pool_incomplete"
            if stop_reason == "candidate_pool_exhausted_before_identification"
            else "not_identified_budget_or_sealed_quota"
        ),
        "stop_reason": stop_reason,
        "active_call_count": state.active_call_count,
        "structural_inference_count": state.structural_inference_count,
        "returned_categorical_token_count": state.returned_categorical_token_count,
        "unopened_candidate_count": unopened,
        "minimum_unopened_candidate_count": minimum_unopened_candidate_count,
        "blocked_choice_response_provider_called": False,
        "blocked_choice_answer_opened": False,
        "schema": _NOT_IDENTIFIED_SCHEMA,
    }
    payload = {
        "schema": _NOT_IDENTIFIED_SCHEMA,
        **{key: value for key, value in kwargs.items() if key != "schema"},
    }
    payload["final_state"] = state._payload(True)
    payload["blocked_choice"] = None if choice is None else choice._payload(True)
    result = OpaqueActiveNotIdentifiedResult(
        **kwargs,
        result_sha256=_sha256(payload),
    )
    _enforce_result_certificate_budget(learner_input, result)
    return result


@dataclass(frozen=True)
class AutonomousPartialOperatorResult:
    learner_input_sha256: str
    final_state: OpaqueActiveDiscoveryState
    model: AutonomousPartialModel
    guarded_language: GuardedLanguageCertificate
    active_call_count: int
    structural_inference_count: int
    returned_categorical_token_count: int
    unopened_candidate_count: int
    final_event_version_counts: tuple[tuple[str, int], ...]
    aggregate_raw_restricted_linear_nullity: int
    aggregate_total_extension_nullity: int
    primary_omission_realized_14q_1i_8sealed: bool
    strict_causal_minimax_selector_used: bool
    posthoc_truth_specific_13_query_teaching_set_used_by_selector: bool
    global_query_minimality_claimed: bool
    supplied_definedness_mask_and_full_product_gauge: bool
    assumption_free_representation_discovery_claimed: bool
    total_wfa_claimed: bool
    result_sha256: str
    schema: str = _RESULT_SCHEMA

    @property
    def identification_status(self) -> str:
        return "identified"

    def __post_init__(self) -> None:
        if self.schema != _RESULT_SCHEMA:
            raise ValueError("unknown autonomous result schema")
        _require_sha256("learner_input_sha256", self.learner_input_sha256)
        if type(self.final_state) is not OpaqueActiveDiscoveryState:
            raise TypeError("final_state must be exact OpaqueActiveDiscoveryState")
        if type(self.model) is not AutonomousPartialModel:
            raise TypeError("model must be exact AutonomousPartialModel")
        if type(self.guarded_language) is not GuardedLanguageCertificate:
            raise TypeError("guarded_language must be exact GuardedLanguageCertificate")
        if self.model.model_sha256 != self.guarded_language.model_sha256:
            raise ValueError("guarded certificate does not bind the final model")
        if self.guarded_language.diagnostic_readout_rows != self.model.diagnostic_codebook:
            raise ValueError("guarded certificate readouts differ from the final model")
        guarded_defined = {
            (row.source_answers, row.event_token)
            for row in self.guarded_language.defined_transitions
        }
        if guarded_defined != set(self.model.defined_categorical_guard_pairs):
            raise ValueError("guarded certificate does not match the model's exact guard")
        operators = {row.event_token: row for row in self.model.operators}
        codebook_rows = dict(self.model.diagnostic_codebook)
        for transition in self.guarded_language.defined_transitions:
            predicted = operators[transition.event_token].apply_row(
                codebook_rows[transition.source_answers]
            )
            if predicted != codebook_rows[transition.predicted_target_answers]:
                raise ValueError("guarded transition disagrees with the restricted operator")
        if self.active_call_count != self.final_state.active_call_count:
            raise ValueError("result active-call count mismatch")
        if self.structural_inference_count != self.final_state.structural_inference_count:
            raise ValueError("result inference count mismatch")
        if self.returned_categorical_token_count != self.final_state.returned_categorical_token_count:
            raise ValueError("result returned-token count mismatch")
        if any(count != 1 for _, count in self.final_event_version_counts):
            raise ValueError("every final event must have one categorical restricted-map version")
        if self.aggregate_raw_restricted_linear_nullity != sum(
            row.raw_restricted_linear_nullity for row in self.model.operators
        ):
            raise ValueError("aggregate raw restricted-nullity mismatch")
        if self.aggregate_total_extension_nullity != sum(
            row.total_extension_nullity for row in self.model.operators
        ):
            raise ValueError("aggregate total-extension nullity mismatch")
        for name, value, required in (
            ("strict_causal_minimax_selector_used", self.strict_causal_minimax_selector_used, True),
            ("posthoc_truth_specific_13_query_teaching_set_used_by_selector", self.posthoc_truth_specific_13_query_teaching_set_used_by_selector, False),
            ("global_query_minimality_claimed", self.global_query_minimality_claimed, False),
            ("supplied_definedness_mask_and_full_product_gauge", self.supplied_definedness_mask_and_full_product_gauge, True),
            ("assumption_free_representation_discovery_claimed", self.assumption_free_representation_discovery_claimed, False),
            ("total_wfa_claimed", self.total_wfa_claimed, False),
        ):
            if _require_bool(name, value) is not required:
                raise ValueError(f"{name} must be {required}")
        if _require_sha256("result_sha256", self.result_sha256) != _sha256(self._payload(False)):
            raise ValueError("autonomous result digest mismatch")

    def _payload(self, include_result_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "learner_input_sha256": self.learner_input_sha256,
            "final_state": self.final_state._payload(True),
            "model": self.model._payload(True),
            "guarded_language": self.guarded_language._payload(True),
            "active_call_count": self.active_call_count,
            "structural_inference_count": self.structural_inference_count,
            "returned_categorical_token_count": self.returned_categorical_token_count,
            "unopened_candidate_count": self.unopened_candidate_count,
            "final_event_version_counts": self.final_event_version_counts,
            "aggregate_raw_restricted_linear_nullity": self.aggregate_raw_restricted_linear_nullity,
            "aggregate_total_extension_nullity": self.aggregate_total_extension_nullity,
            "primary_omission_realized_14q_1i_8sealed": self.primary_omission_realized_14q_1i_8sealed,
            "strict_causal_minimax_selector_used": self.strict_causal_minimax_selector_used,
            "posthoc_truth_specific_13_query_teaching_set_used_by_selector": self.posthoc_truth_specific_13_query_teaching_set_used_by_selector,
            "global_query_minimality_claimed": self.global_query_minimality_claimed,
            "supplied_definedness_mask_and_full_product_gauge": self.supplied_definedness_mask_and_full_product_gauge,
            "assumption_free_representation_discovery_claimed": self.assumption_free_representation_discovery_claimed,
            "total_wfa_claimed": self.total_wfa_claimed,
        }
        if include_result_sha:
            payload["result_sha256"] = self.result_sha256
        return payload


def _build_autonomous_model(
    learner_input: OpaqueActiveLearnerInput,
    state: OpaqueActiveDiscoveryState,
    known: dict[OpaqueWord, tuple[str, ...]],
    versions_by_event: dict[str, tuple[ExactCategoricalRestrictedMapVersion, ...]],
) -> AutonomousPartialModel:
    direct = dict(state.direct_passive_event_ranks)
    response = dict(state.response_event_rank_increments)
    inference = dict(state.inference_event_rank_increments)
    final = dict(state.observed_event_ranks)
    operators: list[AutonomousRestrictedOperator] = []
    for event_token in sorted(learner_input.event_tokens):
        versions = versions_by_event[event_token]
        if len(versions) != 1:
            raise ValueError("cannot build an operator from a non-singleton version space")
        version = versions[0]
        requests_by_sha = {
            row.request_sha256: row
            for row in learner_input.canonical_defined_requests
            if row.event_token == event_token
        }
        requests = tuple(requests_by_sha[digest] for digest in version.legal_request_sha256s)
        source_answers = tuple(known[row.source_word] for row in requests)
        source_rows = tuple(
            _diagnostic_row(answers, learner_input.answer_tokens)
            for answers in source_answers
        )
        domain_rank, pivot_indices, _ = _rank_profile(source_rows)
        domain_basis = tuple(source_rows[index] for index in pivot_indices)
        image_basis = tuple(
            _diagnostic_row(
                version.predicted_target_answers[index], learner_input.answer_tokens
            )
            for index in pivot_indices
        )
        kwargs: dict[str, object] = {
            "event_token": event_token,
            "legal_request_sha256s": version.legal_request_sha256s,
            "legal_source_answers": source_answers,
            "legal_source_rows": source_rows,
            "domain_basis_indices": pivot_indices,
            "domain_basis_rows": domain_basis,
            "image_basis_rows": image_basis,
            "off_domain_annihilator_basis": _right_nullspace_basis(domain_basis),
            "legal_domain_rank": domain_rank,
            "direct_passive_source_rank": direct[event_token],
            "response_source_rank_increment": response[event_token],
            "inference_source_rank_increment": inference[event_token],
            "final_observed_source_rank": final[event_token],
            "raw_restricted_linear_nullity": _AMBIENT_RANK
            * (domain_rank - final[event_token]),
            "categorical_version_count": 1,
            "total_extension_nullity": _AMBIENT_RANK
            * (_AMBIENT_RANK - domain_rank),
            "categorical_restricted_map_identified": True,
            "raw_linear_map_identified": final[event_token] == domain_rank,
            "off_domain_extension_identified": domain_rank == _AMBIENT_RANK,
            "total_operator": None,
            "schema": _OPERATOR_SCHEMA,
        }
        payload = {"schema": _OPERATOR_SCHEMA, **{key: value for key, value in kwargs.items() if key != "schema"}}
        operator = AutonomousRestrictedOperator(
            **kwargs,  # type: ignore[arg-type]
            operator_sha256=_sha256(payload),
        )
        for source_row, expected_answers in zip(
            source_rows, version.predicted_target_answers, strict=True
        ):
            predicted = operator.apply_row(source_row)
            expected = _diagnostic_row(expected_answers, learner_input.answer_tokens)
            if predicted != expected:
                raise ValueError("operator basis does not reconstruct its unique version")
        operators.append(operator)
    codebook = tuple(
        (
            answers,
            _diagnostic_row(answers, learner_input.answer_tokens),
        )
        for answers in _full_product_answers(learner_input)
    )
    mask_rows = tuple((word, known[word]) for word in _mask_source_words(learner_input))
    guard_pairs = tuple(
        sorted(
            (known[request.source_word], request.event_token)
            for request in learner_input.canonical_defined_requests
        )
    )
    kwargs = {
        "event_tokens": learner_input.event_tokens,
        "query_tokens": learner_input.query_tokens,
        "answer_tokens": learner_input.answer_tokens,
        "diagnostic_codebook": codebook,
        "mask_source_answer_rows": mask_rows,
        "defined_categorical_guard_pairs": guard_pairs,
        "operators": tuple(operators),
        "ambient_rank": _AMBIENT_RANK,
        "diagnostic_rank": _rank(tuple(row for _, row in codebook)),
        "categorical_state_count": len(codebook),
        "restricted_maps_complete": True,
        "total_operator": None,
        "schema": _MODEL_SCHEMA,
    }
    payload = {"schema": _MODEL_SCHEMA, **{key: value for key, value in kwargs.items() if key != "schema"}}
    payload["operators"] = [row._payload(True) for row in operators]
    return AutonomousPartialModel(
        **kwargs,
        model_sha256=_sha256(payload),
    )


def _build_guarded_language_certificate(
    learner_input: OpaqueActiveLearnerInput,
    model: AutonomousPartialModel,
    known: dict[OpaqueWord, tuple[str, ...]],
) -> GuardedLanguageCertificate:
    operators = {row.event_token: row for row in model.operators}
    answers_by_row = {row: answers for answers, row in model.diagnostic_codebook}
    exact_guard = set(model.defined_categorical_guard_pairs)
    defined_rows: list[GuardedDefinedTransition] = []
    for request in learner_input.canonical_defined_requests:
        source_answers = known[request.source_word]
        if (source_answers, request.event_token) not in exact_guard:
            raise ValueError("declared defined transition is absent from the exact model guard")
        source_row = _diagnostic_row(source_answers, learner_input.answer_tokens)
        predicted_row = operators[request.event_token].apply_row(source_row)
        predicted_answers = answers_by_row.get(predicted_row)
        if predicted_answers is None:
            raise ValueError("defined transition leaves the nine-state categorical codebook")
        defined_rows.append(
            GuardedDefinedTransition(
                request_sha256=request.request_sha256,
                event_token=request.event_token,
                source_answers=source_answers,
                predicted_target_answers=predicted_answers,
                admitted_by_exact_categorical_guard=True,
                prediction_is_unique_categorical_version=True,
            )
        )
    undefined_rows: list[GuardedUndefinedPair] = []
    for request in learner_input.canonical_undefined_requests:
        source_answers = known[request.source_word]
        if (source_answers, request.event_token) in exact_guard:
            raise ValueError("declared undefined pair is admitted by the exact model guard")
        source_row = _diagnostic_row(source_answers, learner_input.answer_tokens)
        rejected = False
        try:
            operators[request.event_token].apply_row(source_row)
        except ValueError:
            rejected = True
        if not rejected:
            raise ValueError("undefined mask pair lies in the learned legal-domain span")
        undefined_rows.append(
            GuardedUndefinedPair(
                request_sha256=request.request_sha256,
                event_token=request.event_token,
                source_answers=source_answers,
                rejected_by_exact_categorical_guard=True,
                rejected_by_restricted_domain_span_control=True,
            )
        )
    empty_words = [word for word in _mask_source_words(learner_input) if word == ()]
    if len(empty_words) != 1 or () not in known:
        raise ValueError("definedness mask must contain one known empty-word initial state")
    kwargs = {
        "model_sha256": model.model_sha256,
        "initial_word": (),
        "initial_answers": known[()],
        "diagnostic_readout_rows": model.diagnostic_codebook,
        "defined_transitions": tuple(defined_rows),
        "undefined_pairs": tuple(undefined_rows),
        "state_count": _STATE_COUNT,
        "defined_transition_count": _LEGAL_EDGE_COUNT,
        "undefined_pair_count": _UNDEFINED_EDGE_COUNT,
        "exact_guard_rejection_count": len(undefined_rows),
        "restricted_domain_span_control_rejection_count": len(undefined_rows),
        "all_defined_targets_are_declared_states": True,
        "all_undefined_pairs_rejected": True,
        "all_undefined_pairs_rejected_by_exact_guard": True,
        "all_undefined_pairs_outside_restricted_domain_span": True,
        "arbitrary_length_legal_suffix_induction": True,
        "arbitrary_suffix_without_definedness_guard_claimed": False,
        "total_wfa_claimed": False,
        "unqueried_edge_answers_used_to_fit_or_select": False,
        "schema": _LANGUAGE_SCHEMA,
    }
    payload = {"schema": _LANGUAGE_SCHEMA, **{key: value for key, value in kwargs.items() if key != "schema"}}
    payload["defined_transitions"] = [row.__dict__ for row in defined_rows]
    payload["undefined_pairs"] = [row.__dict__ for row in undefined_rows]
    return GuardedLanguageCertificate(
        **kwargs,
        certificate_sha256=_sha256(payload),
    )


def finalize_opaque_active_discovery(
    learner_input: OpaqueActiveLearnerInput,
    state: OpaqueActiveDiscoveryState,
) -> AutonomousPartialOperatorResult:
    validate_opaque_active_state(learner_input, state)
    if _choose_next_opaque_edge_core(learner_input, state) is not None:
        raise ValueError("active discovery cannot finalize before version-space closure")
    known, edges = _known_material(learner_input, state.steps)
    if any(word not in known for word in _mask_source_words(learner_input)):
        raise ValueError("finalization requires all nine mask-source diagnostics")
    versions = _filtered_versions(learner_input, known, edges)
    if any(len(rows) != 1 for rows in versions.values()):
        raise ValueError("finalization requires one categorical version per event")
    model = _build_autonomous_model(learner_input, state, known, versions)
    guarded = _build_guarded_language_certificate(learner_input, model, known)
    unopened = len(learner_input.canonical_candidate_requests) - len(state.steps)
    primary = (
        bool(learner_input.canonical_candidate_requests)
        and state.active_call_count == _EXPECTED_ACTIVE_CALLS
        and state.structural_inference_count == _EXPECTED_STRUCTURAL_INFERENCES
        and state.returned_categorical_token_count == _EXPECTED_RETURNED_LABELS
        and unopened == 8
    )
    kwargs = {
        "learner_input_sha256": learner_input.input_sha256,
        "final_state": state,
        "model": model,
        "guarded_language": guarded,
        "active_call_count": state.active_call_count,
        "structural_inference_count": state.structural_inference_count,
        "returned_categorical_token_count": state.returned_categorical_token_count,
        "unopened_candidate_count": unopened,
        "final_event_version_counts": tuple(
            (token, len(rows)) for token, rows in sorted(versions.items())
        ),
        "aggregate_raw_restricted_linear_nullity": sum(
            row.raw_restricted_linear_nullity for row in model.operators
        ),
        "aggregate_total_extension_nullity": sum(
            row.total_extension_nullity for row in model.operators
        ),
        "primary_omission_realized_14q_1i_8sealed": primary,
        "strict_causal_minimax_selector_used": True,
        "posthoc_truth_specific_13_query_teaching_set_used_by_selector": False,
        "global_query_minimality_claimed": False,
        "supplied_definedness_mask_and_full_product_gauge": True,
        "assumption_free_representation_discovery_claimed": False,
        "total_wfa_claimed": False,
        "schema": _RESULT_SCHEMA,
    }
    payload = {"schema": _RESULT_SCHEMA, **{key: value for key, value in kwargs.items() if key != "schema"}}
    payload["final_state"] = state._payload(True)
    payload["model"] = model._payload(True)
    payload["guarded_language"] = guarded._payload(True)
    result = AutonomousPartialOperatorResult(
        **kwargs,
        result_sha256=_sha256(payload),
    )
    _enforce_result_certificate_budget(learner_input, result)
    return result


def run_opaque_active_discovery(
    learner_input: OpaqueActiveLearnerInput,
    response_provider: Callable[
        [OpaqueActiveChoiceCertificate], tuple[str, ...] | OpaqueActiveMembershipResponse
    ],
    *,
    minimum_unopened_candidate_count: int | None = None,
) -> AutonomousPartialOperatorResult | OpaqueActiveNotIdentifiedResult:
    """Run the causal learner; the provider is called only after commitment."""

    if not callable(response_provider):
        raise TypeError("response_provider must be callable")
    if learner_input.budgets.max_certificate_bytes < _FROZEN_RESULT_CERTIFICATE_UPPER_BOUND:
        raise OpaqueActiveDiscoveryLimitError(
            "certificate byte budget is below the frozen conservative upper bound"
        )
    minimum_unopened = (
        (
            8
            if learner_input.canonical_candidate_requests
            and learner_input.candidate_pool_complete
            else 0
        )
        if minimum_unopened_candidate_count is None
        else _plain_int(
            "minimum_unopened_candidate_count",
            minimum_unopened_candidate_count,
        )
    )
    if minimum_unopened > len(learner_input.canonical_candidate_requests):
        raise ValueError("minimum unopened quota exceeds the candidate pool")
    state = initialize_opaque_active_discovery(learner_input)
    while True:
        try:
            choice = choose_next_opaque_edge(learner_input, state)
        except OpaqueActiveCandidatePoolExhaustedError:
            if not learner_input.single_candidate_removal_negative_control:
                raise
            return _make_not_identified_result(
                learner_input,
                state,
                None,
                stop_reason="candidate_pool_exhausted_before_identification",
                minimum_unopened_candidate_count=minimum_unopened,
            )
        if choice is None:
            break
        unopened_before = len(learner_input.canonical_candidate_requests) - len(
            state.steps
        )
        if unopened_before - 1 < minimum_unopened:
            return _make_not_identified_result(
                learner_input,
                state,
                choice,
                stop_reason="sealed_candidate_quota_reached",
                minimum_unopened_candidate_count=minimum_unopened,
            )
        if not choice.requires_membership_response:
            if (
                state.structural_inference_count + 1
                > learner_input.budgets.max_structural_inferences
            ):
                return _make_not_identified_result(
                    learner_input,
                    state,
                    choice,
                    stop_reason="structural_inference_budget_exhausted",
                    minimum_unopened_candidate_count=minimum_unopened,
                )
            state = incorporate_opaque_structural_inference(
                learner_input, state, choice
            )
            continue
        if state.active_call_count + 1 > learner_input.budgets.max_active_calls:
            return _make_not_identified_result(
                learner_input,
                state,
                choice,
                stop_reason="active_call_budget_exhausted",
                minimum_unopened_candidate_count=minimum_unopened,
            )
        if (
            state.returned_categorical_token_count + _QUERY_COUNT
            > learner_input.budgets.max_returned_categorical_tokens
        ):
            return _make_not_identified_result(
                learner_input,
                state,
                choice,
                stop_reason="returned_categorical_token_budget_exhausted",
                minimum_unopened_candidate_count=minimum_unopened,
            )
        provided = response_provider(choice)
        response = (
            provided
            if type(provided) is OpaqueActiveMembershipResponse
            else make_opaque_active_response(
                learner_input,
                state,
                choice,
                tuple(provided),
            )
        )
        state = incorporate_opaque_response(
            learner_input, state, choice, response
        )
    return finalize_opaque_active_discovery(learner_input, state)


def predict_defined_suffix(
    result: AutonomousPartialOperatorResult,
    source_answers: tuple[str, ...],
    suffix: Sequence[str],
    *,
    max_events: int | None = None,
) -> tuple[str, ...]:
    if type(result) is not AutonomousPartialOperatorResult:
        raise TypeError("result must be exact AutonomousPartialOperatorResult")
    limit = 4_096 if max_events is None else _plain_int("max_events", max_events, 1)
    return result.model.predict_defined_suffix(
        source_answers,
        suffix,
        max_events=limit,
    )


__all__ = (
    "AutonomousPartialModel",
    "AutonomousPartialOperatorResult",
    "AutonomousRestrictedOperator",
    "ExactCategoricalRestrictedMapVersion",
    "ExactOutcomeBranch",
    "GuardedDefinedTransition",
    "GuardedLanguageCertificate",
    "GuardedUndefinedPair",
    "OpaqueActiveCandidateScore",
    "OpaqueActiveChoiceCertificate",
    "OpaqueActiveCandidatePoolExhaustedError",
    "OpaqueActiveDiscoveryBudgets",
    "OpaqueActiveDiscoveryLimitError",
    "OpaqueActiveDiscoveryState",
    "OpaqueActiveLearnerInput",
    "OpaqueActiveMembershipResponse",
    "OpaqueActiveNotIdentifiedResult",
    "OpaqueActiveStep",
    "OpaqueStructuralInference",
    "OpaqueStructuralInferenceStep",
    "finalize_opaque_active_discovery",
    "incorporate_opaque_response",
    "incorporate_opaque_structural_inference",
    "initialize_opaque_active_discovery",
    "make_opaque_active_input",
    "make_opaque_active_input_from_rows",
    "make_opaque_candidate_removal_negative_control",
    "make_opaque_active_response",
    "predict_defined_suffix",
    "run_opaque_active_discovery",
    "validate_opaque_active_state",
    "choose_next_opaque_edge",
)
