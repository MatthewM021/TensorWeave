"""Exact opaque-token partial-operator discovery for the Phase-III-T1 toy.

The observable contract is the guarded, absence-aware ``K=2,V=2,M=2``
binding behavior.  The learner boundary contains only opaque words, opaque
categorical diagnostics, an explicit definedness mask, and content hashes.
It receives no semantic state, key, value, event family, omitted cell,
executor, nonce, or controller mapping.

Contract B is partial: an illegal program has no output.  Consequently this
module never serializes or claims a total 5x5 WFA event matrix.  It identifies
each event's exact linear map on the span of its legal source states and
reports the remaining off-domain extension nullity.  Undefined words are
constraints absent from the series, not zeros, ``ABSENT`` values, or a dead
state.

The trusted controller and pure learner live in one module for this bounded
synthetic protocol rehearsal.  The separation is an audited argument/API
boundary, not cryptographic process isolation.  Nested digests are contextual
content links; the enclosing environment/report constructors reconstruct all
learner results before treating them as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from itertools import product
from math import gcd
import hashlib
import json
from typing import Iterable, Mapping, Sequence


OpaqueWord = tuple[str, ...]
SemanticState = tuple[int, int]
SemanticCell = tuple[int, int]
SemanticAction = tuple[str, int, int]

_TOKEN_LENGTH = 32
_FULL_RANK = 5
_PASSIVE_RANK = 4
_STATE_COUNT = 9
_EVENT_COUNT = 10
_QUERY_COUNT = 2
_ANSWER_COUNT = 3
_PASSIVE_EDGE_COUNT = 21
_ACTIVE_RESPONSE_COUNT = 15
_SEALED_EDGE_COUNT = 8
_LEGAL_EDGE_COUNT = 44
_UNDEFINED_EDGE_COUNT = _STATE_COUNT * _EVENT_COUNT - _LEGAL_EDGE_COUNT
_LONG_PROBE_COUNT = 12
_ENVIRONMENT_COUNT = 10

_INPUT_SCHEMA = "tnlm-v3-toy-opaque-partial-operators-input-v1"
_REQUEST_SCHEMA = "tnlm-v3-toy-opaque-edge-request-v1"
_OBSERVATION_SCHEMA = "tnlm-v3-toy-opaque-edge-observation-v1"
_RESPONSE_SCHEMA = "tnlm-v3-toy-opaque-edge-response-v1"
_RANK_SCHEMA = "tnlm-v3-toy-exact-partial-rank-v1"
_OPERATOR_SCHEMA = "tnlm-v3-toy-exact-partial-event-operator-v1"
_REALIZATION_SCHEMA = "tnlm-v3-toy-exact-partial-realization-v1"
_COMMITMENT_SCHEMA = "tnlm-v3-toy-partial-active-commitment-v1"
_PASSIVE_SCHEMA = "tnlm-v3-toy-opaque-partial-passive-v1"
_ONE_RESPONSE_SCHEMA = "tnlm-v3-toy-opaque-one-response-checkpoint-v1"
_ACTIVE_SCHEMA = "tnlm-v3-toy-opaque-partial-active-v1"
_ENVIRONMENT_SCHEMA = "tnlm-v3-toy-opaque-partial-environment-v1"
_SIMILARITY_SCHEMA = "tnlm-v3-toy-opaque-partial-similarity-v1"
_REPORT_SCHEMA = "tnlm-v3-toy-opaque-partial-report-v1"
_HYPOTHESIS_SCHEMA = "tnlm-v3-toy-opaque-omission-hypothesis-v1"
_HYPOTHESIS_CERTIFICATE_SCHEMA = (
    "tnlm-v3-toy-opaque-compatible-omission-certificate-v1"
)
_UNDEFINED_REJECTION_SCHEMA = "tnlm-v3-toy-opaque-undefined-domain-rejection-v1"


class OpaquePartialOperatorLimitError(RuntimeError):
    """Raised before a frozen Phase-III-T1 work ceiling is crossed."""


class PartialOperatorScope(str, Enum):
    GUARDED_ABSENCE_AWARE_PARTIAL_OPERATORS = (
        "guarded_absence_aware_partial_predictive_operators_only"
    )


class PartialOperatorStatus(str, Enum):
    SYNTHETIC_PROTOCOL_IMPLEMENTATION_REHEARSAL = (
        "synthetic_protocol_implementation_rehearsal"
    )


class EnvironmentKind(str, Enum):
    FULL_SUPPORT_CONTROL = "full_support_control"
    ROTATED_OMISSION = "rotated_omission"


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


def _require_nonce(value: object) -> str:
    return _require_sha256("trusted controller nonce", value)


def _jsonable(value: object) -> object:
    if isinstance(value, Rational):
        return [value.numerator, value.denominator]
    if isinstance(value, Enum):
        return value.value
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


@dataclass(frozen=True, order=True)
class Rational:
    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise TypeError("rational components must be exact integers")
        if self.denominator <= 0:
            raise ValueError("rational denominator must be positive")
        if gcd(abs(self.numerator), self.denominator) != 1:
            raise ValueError("rational must be reduced")

    @classmethod
    def from_fraction(cls, value: Fraction) -> Rational:
        return cls(value.numerator, value.denominator)

    def as_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


RationalVector = tuple[Rational, ...]
RationalMatrix = tuple[RationalVector, ...]


def _rat(value: int | Fraction | Rational) -> Rational:
    if type(value) is Rational:
        return value
    if type(value) is int:
        return Rational(value)
    if type(value) is Fraction:
        return Rational.from_fraction(value)
    raise TypeError("exact arithmetic accepts only int, Fraction, or Rational")


def _matrix(rows: Iterable[Iterable[int | Fraction | Rational]]) -> RationalMatrix:
    result = tuple(tuple(_rat(value) for value in row) for row in rows)
    _matrix_shape(result)
    return result


def _matrix_shape(matrix: object) -> tuple[int, int]:
    if not isinstance(matrix, tuple) or not matrix:
        raise TypeError("matrix must be a nonempty tuple")
    width: int | None = None
    for row in matrix:
        if not isinstance(row, tuple) or not row:
            raise TypeError("matrix rows must be nonempty tuples")
        if any(type(value) is not Rational for value in row):
            raise TypeError("matrix entries must be Rational")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("matrix must be rectangular")
    assert width is not None
    return len(matrix), width


def _fractions(matrix: RationalMatrix) -> list[list[Fraction]]:
    return [[value.as_fraction() for value in row] for row in matrix]


def _rank_profile(matrix: RationalMatrix) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    rows, columns = _matrix_shape(matrix)
    work = _fractions(matrix)
    source_rows = list(range(rows))
    pivot_rows: list[int] = []
    pivot_columns: list[int] = []
    rank = 0
    for column in range(columns):
        pivot = next((row for row in range(rank, rows) if work[row][column]), None)
        if pivot is None:
            continue
        if pivot != rank:
            work[rank], work[pivot] = work[pivot], work[rank]
            source_rows[rank], source_rows[pivot] = source_rows[pivot], source_rows[rank]
        pivot_value = work[rank][column]
        work[rank] = [value / pivot_value for value in work[rank]]
        for row in range(rows):
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
        if rank == rows:
            break
    return rank, tuple(pivot_rows), tuple(pivot_columns)


def _determinant(matrix: RationalMatrix) -> Rational:
    rows, columns = _matrix_shape(matrix)
    if rows != columns:
        raise ValueError("determinant requires a square matrix")
    work = _fractions(matrix)
    result = Fraction(1)
    sign = 1
    for column in range(columns):
        pivot = next((row for row in range(column, rows) if work[row][column]), None)
        if pivot is None:
            return Rational(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            sign *= -1
        pivot_value = work[column][column]
        result *= pivot_value
        for row in range(column + 1, rows):
            if not work[row][column]:
                continue
            factor = work[row][column] / pivot_value
            for inner in range(column, columns):
                work[row][inner] -= factor * work[column][inner]
    return Rational.from_fraction(sign * result)


def _inverse(matrix: RationalMatrix) -> RationalMatrix:
    rows, columns = _matrix_shape(matrix)
    if rows != columns:
        raise ValueError("inverse requires a square matrix")
    work = [
        row + [Fraction(int(i == j)) for j in range(columns)]
        for i, row in enumerate(_fractions(matrix))
    ]
    for column in range(columns):
        pivot = next((row for row in range(column, rows) if work[row][column]), None)
        if pivot is None:
            raise ValueError("matrix is singular")
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
        pivot_value = work[column][column]
        work[column] = [value / pivot_value for value in work[column]]
        for row in range(rows):
            if row == column or not work[row][column]:
                continue
            factor = work[row][column]
            work[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(work[row], work[column], strict=True)
            ]
    return _matrix(row[columns:] for row in work)


def _row_times_matrix(row: RationalVector, matrix: RationalMatrix) -> RationalVector:
    rows, columns = _matrix_shape(matrix)
    if len(row) != rows:
        raise ValueError("row and matrix dimensions do not align")
    left = [value.as_fraction() for value in row]
    right = _fractions(matrix)
    return tuple(
        Rational.from_fraction(sum(left[i] * right[i][j] for i in range(rows)))
        for j in range(columns)
    )


def _matmul(left: RationalMatrix, right: RationalMatrix) -> RationalMatrix:
    left_rows, inner = _matrix_shape(left)
    right_rows, right_columns = _matrix_shape(right)
    if inner != right_rows:
        raise ValueError("matrix dimensions do not align")
    a, b = _fractions(left), _fractions(right)
    return _matrix(
        tuple(sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(right_columns))
        for i in range(left_rows)
    )


@dataclass(frozen=True)
class PartialOperatorBudgets:
    max_word_length: int = 16
    max_event_tokens: int = 16
    max_domain_edges: int = 90
    max_passive_edges: int = 44
    max_active_responses: int = 15
    max_categorical_labels: int = 30
    max_sealed_edges: int = 8
    max_long_probes: int = 12
    max_basis_dimension: int = 10
    max_exact_rank_evaluations: int = 50_000
    max_rational_bit_length: int = 4_096

    def __post_init__(self) -> None:
        hard = {
            "max_word_length": 16,
            "max_event_tokens": 16,
            "max_domain_edges": 90,
            "max_passive_edges": 44,
            "max_active_responses": 15,
            "max_categorical_labels": 30,
            "max_sealed_edges": 8,
            "max_long_probes": 12,
            "max_basis_dimension": 10,
            "max_exact_rank_evaluations": 50_000,
            "max_rational_bit_length": 4_096,
        }
        for name, ceiling in hard.items():
            value = _plain_int(name, getattr(self, name), 1)
            if value > ceiling:
                raise ValueError(f"{name} exceeds the frozen Phase-III-T1 ceiling")

    def payload(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, order=True)
class OpaqueStateObservation:
    word: OpaqueWord
    answers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.word, tuple):
            raise TypeError("state word must be an exact tuple")
        for token in self.word:
            _require_token("state word token", token)
        if not isinstance(self.answers, tuple) or len(self.answers) != _QUERY_COUNT:
            raise ValueError("state observation must contain two categorical answers")
        for token in self.answers:
            _require_token("state answer", token)


@dataclass(frozen=True, order=True)
class OpaqueEdgeRequest:
    source_word: OpaqueWord
    event_token: str
    program: OpaqueWord
    request_sha256: str
    schema: str = _REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _REQUEST_SCHEMA:
            raise ValueError("unknown opaque edge-request schema")
        if not isinstance(self.source_word, tuple) or not isinstance(self.program, tuple):
            raise TypeError("edge words must be exact tuples")
        for token in self.source_word + self.program:
            _require_token("edge word token", token)
        _require_token("edge event token", self.event_token)
        if self.program != self.source_word + (self.event_token,):
            raise ValueError("edge program must append exactly its event token")
        expected = _sha256(
            {
                "schema": self.schema,
                "source_word": self.source_word,
                "event_token": self.event_token,
                "program": self.program,
            }
        )
        if _require_sha256("request_sha256", self.request_sha256) != expected:
            raise ValueError("request_sha256 does not bind the edge request")


def _make_request(source_word: OpaqueWord, event_token: str) -> OpaqueEdgeRequest:
    payload = {
        "schema": _REQUEST_SCHEMA,
        "source_word": source_word,
        "event_token": event_token,
        "program": source_word + (event_token,),
    }
    return OpaqueEdgeRequest(
        source_word=source_word,
        event_token=event_token,
        program=source_word + (event_token,),
        request_sha256=_sha256(payload),
    )


@dataclass(frozen=True, order=True)
class OpaqueEdgeObservation:
    request: OpaqueEdgeRequest
    source_answers: tuple[str, ...]
    target_answers: tuple[str, ...]
    observation_sha256: str
    schema: str = _OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _OBSERVATION_SCHEMA:
            raise ValueError("unknown opaque edge-observation schema")
        if type(self.request) is not OpaqueEdgeRequest:
            raise TypeError("request must be exact OpaqueEdgeRequest")
        for name, answers in (
            ("source_answers", self.source_answers),
            ("target_answers", self.target_answers),
        ):
            if not isinstance(answers, tuple) or len(answers) != _QUERY_COUNT:
                raise ValueError(f"{name} must contain two labels")
            for token in answers:
                _require_token(name, token)
        expected = _sha256(
            {
                "schema": self.schema,
                "request_sha256": self.request.request_sha256,
                "source_answers": self.source_answers,
                "target_answers": self.target_answers,
            }
        )
        if _require_sha256("observation_sha256", self.observation_sha256) != expected:
            raise ValueError("observation digest mismatch")


def _make_observation(
    request: OpaqueEdgeRequest,
    source_answers: tuple[str, ...],
    target_answers: tuple[str, ...],
) -> OpaqueEdgeObservation:
    payload = {
        "schema": _OBSERVATION_SCHEMA,
        "request_sha256": request.request_sha256,
        "source_answers": source_answers,
        "target_answers": target_answers,
    }
    return OpaqueEdgeObservation(
        request=request,
        source_answers=source_answers,
        target_answers=target_answers,
        observation_sha256=_sha256(payload),
    )


def _validate_answers(name: str, answers: object) -> tuple[str, ...]:
    if not isinstance(answers, tuple) or len(answers) != _QUERY_COUNT:
        raise ValueError(f"{name} must contain exactly two categorical labels")
    for token in answers:
        _require_token(name, token)
    return answers


@dataclass(frozen=True, order=True)
class OpaqueMembershipResponse:
    """The two *target* labels returned by one active membership call.

    The source labels are deliberately absent.  The source word must already
    have a diagnostic in passive data or an earlier response; consequently 15
    responses contain exactly 30 returned categorical tokens, not 60.
    """

    request: OpaqueEdgeRequest
    target_answers: tuple[str, ...]
    response_ordinal: int
    response_sha256: str
    schema: str = _RESPONSE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _RESPONSE_SCHEMA:
            raise ValueError("unknown opaque membership-response schema")
        if type(self.request) is not OpaqueEdgeRequest:
            raise TypeError("request must be exact OpaqueEdgeRequest")
        _validate_answers("target_answers", self.target_answers)
        _plain_int("response_ordinal", self.response_ordinal, 1)
        expected = _sha256(
            {
                "schema": self.schema,
                "request_sha256": self.request.request_sha256,
                "target_answers": self.target_answers,
                "response_ordinal": self.response_ordinal,
            }
        )
        if _require_sha256("response_sha256", self.response_sha256) != expected:
            raise ValueError("response_sha256 does not bind the membership response")


def _make_response(
    request: OpaqueEdgeRequest,
    target_answers: tuple[str, ...],
    ordinal: int,
) -> OpaqueMembershipResponse:
    payload = {
        "schema": _RESPONSE_SCHEMA,
        "request_sha256": request.request_sha256,
        "target_answers": target_answers,
        "response_ordinal": ordinal,
    }
    return OpaqueMembershipResponse(
        request=request,
        target_answers=target_answers,
        response_ordinal=ordinal,
        response_sha256=_sha256(payload),
    )


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


def _response_payload(response: OpaqueMembershipResponse) -> dict[str, object]:
    return {
        "schema": response.schema,
        "request": _request_payload(response.request),
        "target_answers": response.target_answers,
        "response_ordinal": response.response_ordinal,
        "response_sha256": response.response_sha256,
    }


@dataclass(frozen=True)
class PartialOperatorLearnerInput:
    """Pure opaque learner boundary.

    The explicit defined/undefined edge mask is supervision.  No claim in this
    module treats guarded definedness as something inferred from answer-only
    strings.  Conversely, the mask contains no categorical target answer for
    an unobserved edge.
    """

    event_tokens: tuple[str, ...]
    query_tokens: tuple[str, ...]
    answer_tokens: tuple[str, ...]
    passive_state_observations: tuple[OpaqueStateObservation, ...]
    passive_edge_observations: tuple[OpaqueEdgeObservation, ...]
    candidate_edge_requests: tuple[OpaqueEdgeRequest, ...]
    defined_edge_requests: tuple[OpaqueEdgeRequest, ...]
    undefined_edge_requests: tuple[OpaqueEdgeRequest, ...]
    budgets: PartialOperatorBudgets
    domain_mask_is_learner_visible_supervision: bool
    undefined_words_are_absent_constraints: bool
    illegal_words_encoded_as_zero_or_dead: bool
    state_changing_event_subalphabet_only: bool
    diagnostic_queries_are_output_channels_not_events: bool
    categorical_state_space_declared_as_full_product: bool
    full_product_state_grammar_counted_as_supervision: bool
    passive_table_sha256: str
    domain_mask_sha256: str
    candidate_pool_sha256: str
    input_sha256: str
    schema: str = _INPUT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _INPUT_SCHEMA:
            raise ValueError("unknown partial-operator learner-input schema")
        if type(self.budgets) is not PartialOperatorBudgets:
            raise TypeError("budgets must be exact PartialOperatorBudgets")
        vocabularies = (
            ("event_tokens", self.event_tokens, _EVENT_COUNT),
            ("query_tokens", self.query_tokens, _QUERY_COUNT),
            ("answer_tokens", self.answer_tokens, _ANSWER_COUNT),
        )
        all_tokens: list[str] = []
        for name, values, expected_count in vocabularies:
            if not isinstance(values, tuple) or len(values) != expected_count:
                raise ValueError(f"{name} must have exactly {expected_count} entries")
            for token in values:
                _require_token(name, token)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
            all_tokens.extend(values)
        if len(set(all_tokens)) != len(all_tokens):
            raise ValueError("opaque event/query/answer vocabularies must be disjoint")
        if len(self.event_tokens) > self.budgets.max_event_tokens:
            raise OpaquePartialOperatorLimitError("event-token budget exceeded")
        for name, value, required in (
            (
                "domain_mask_is_learner_visible_supervision",
                self.domain_mask_is_learner_visible_supervision,
                True,
            ),
            (
                "undefined_words_are_absent_constraints",
                self.undefined_words_are_absent_constraints,
                True,
            ),
            (
                "illegal_words_encoded_as_zero_or_dead",
                self.illegal_words_encoded_as_zero_or_dead,
                False,
            ),
            (
                "state_changing_event_subalphabet_only",
                self.state_changing_event_subalphabet_only,
                True,
            ),
            (
                "diagnostic_queries_are_output_channels_not_events",
                self.diagnostic_queries_are_output_channels_not_events,
                True,
            ),
            (
                "categorical_state_space_declared_as_full_product",
                self.categorical_state_space_declared_as_full_product,
                True,
            ),
            (
                "full_product_state_grammar_counted_as_supervision",
                self.full_product_state_grammar_counted_as_supervision,
                True,
            ),
        ):
            if _require_bool(name, value) is not required:
                raise ValueError(f"{name} must be {required}")
        for name, values in (
            ("passive_state_observations", self.passive_state_observations),
            ("passive_edge_observations", self.passive_edge_observations),
            ("candidate_edge_requests", self.candidate_edge_requests),
            ("defined_edge_requests", self.defined_edge_requests),
            ("undefined_edge_requests", self.undefined_edge_requests),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be an exact tuple")
        if len(self.defined_edge_requests) != _LEGAL_EDGE_COUNT:
            raise ValueError("defined mask must contain exactly 44 legal edges")
        if len(self.undefined_edge_requests) != _UNDEFINED_EDGE_COUNT:
            raise ValueError("undefined mask must contain exactly 46 illegal edges")
        if len(self.defined_edge_requests) + len(self.undefined_edge_requests) > self.budgets.max_domain_edges:
            raise OpaquePartialOperatorLimitError("definedness-mask edge budget exceeded")
        passive_count = len(self.passive_edge_observations)
        if passive_count not in (_PASSIVE_EDGE_COUNT, _LEGAL_EDGE_COUNT):
            raise ValueError("passive table must be the 21-edge omission slice or 44-edge control")
        if passive_count > self.budgets.max_passive_edges:
            raise OpaquePartialOperatorLimitError("passive-edge budget exceeded")
        expected_state_count = _STATE_COUNT if passive_count == _LEGAL_EDGE_COUNT else 6
        if len(self.passive_state_observations) != expected_state_count:
            raise ValueError(
                "passive state table must contain nine control states or six omission states"
            )
        if len({row.word for row in self.passive_state_observations}) != expected_state_count:
            raise ValueError("passive state observations must use distinct representative words")
        expected_candidate_count = _LEGAL_EDGE_COUNT - passive_count
        if len(self.candidate_edge_requests) != expected_candidate_count:
            raise ValueError("candidate pool must be the exact complement of passive edges")
        if expected_candidate_count not in (0, _ACTIVE_RESPONSE_COUNT + _SEALED_EDGE_COUNT):
            raise ValueError("unsupported candidate-pool partition")
        if self.budgets.max_basis_dimension < _FULL_RANK:
            raise OpaquePartialOperatorLimitError("basis-dimension budget is below rank five")
        if self.budgets.max_exact_rank_evaluations < 512:
            raise OpaquePartialOperatorLimitError(
                "exact-rank evaluation budget is below the conservative fit preflight"
            )
        if self.budgets.max_rational_bit_length < 64:
            raise OpaquePartialOperatorLimitError(
                "rational bit-length budget is below the conservative exact-arithmetic preflight"
            )
        if expected_candidate_count:
            if self.budgets.max_active_responses < _ACTIVE_RESPONSE_COUNT:
                raise OpaquePartialOperatorLimitError("active-response budget is below 15")
            if self.budgets.max_categorical_labels < 2 * _ACTIVE_RESPONSE_COUNT:
                raise OpaquePartialOperatorLimitError("categorical-label budget is below 30")
            if self.budgets.max_sealed_edges < _SEALED_EDGE_COUNT:
                raise OpaquePartialOperatorLimitError("sealed-edge budget is below eight")

        event_set = set(self.event_tokens)
        all_request_rows = self.defined_edge_requests + self.undefined_edge_requests
        for request in all_request_rows + self.candidate_edge_requests:
            if type(request) is not OpaqueEdgeRequest:
                raise TypeError("edge masks must contain exact OpaqueEdgeRequest rows")
            if request.event_token not in event_set:
                raise ValueError("edge request contains an undeclared event token")
            if len(request.program) > self.budgets.max_word_length:
                raise OpaquePartialOperatorLimitError("word-length budget exceeded")
            if any(token not in event_set for token in request.program):
                raise ValueError("program contains a token outside the state-changing alphabet")
        defined_pairs = {
            (row.source_word, row.event_token): row.request_sha256
            for row in self.defined_edge_requests
        }
        undefined_pairs = {
            (row.source_word, row.event_token): row.request_sha256
            for row in self.undefined_edge_requests
        }
        if len(defined_pairs) != _LEGAL_EDGE_COUNT or len(undefined_pairs) != _UNDEFINED_EDGE_COUNT:
            raise ValueError("definedness masks contain duplicate source/event pairs")
        if set(defined_pairs).intersection(undefined_pairs):
            raise ValueError("an edge cannot be both defined and undefined")
        mask_pairs = set(defined_pairs) | set(undefined_pairs)
        mask_source_words = {word for word, _ in mask_pairs}
        if len(mask_source_words) != _STATE_COUNT:
            raise ValueError("definedness mask must use exactly nine source representatives")
        expected_cartesian = {
            (word, token)
            for word in mask_source_words
            for token in self.event_tokens
        }
        if mask_pairs != expected_cartesian:
            raise ValueError("definedness mask must be the exact 9x10 source/event Cartesian grid")
        if not {
            row.word for row in self.passive_state_observations
        }.issubset(mask_source_words):
            raise ValueError("passive state rows must use definedness-mask source representatives")
        defined_hashes = {row.request_sha256 for row in self.defined_edge_requests}
        passive_hashes: set[str] = set()
        word_answers: dict[OpaqueWord, tuple[str, ...]] = {}

        def reconcile(word: OpaqueWord, answers: tuple[str, ...]) -> None:
            _validate_answers("reconciled answers", answers)
            previous = word_answers.setdefault(word, answers)
            if previous != answers:
                raise ValueError("the same opaque word has contradictory diagnostics")
            if any(answer not in self.answer_tokens for answer in answers):
                raise ValueError("observation uses an undeclared answer token")

        for state in self.passive_state_observations:
            if type(state) is not OpaqueStateObservation:
                raise TypeError("state table must contain exact OpaqueStateObservation rows")
            if len(state.word) > self.budgets.max_word_length:
                raise OpaquePartialOperatorLimitError("state-word budget exceeded")
            if any(token not in event_set for token in state.word):
                raise ValueError("state word contains an undeclared event token")
            reconcile(state.word, state.answers)
        for observation in self.passive_edge_observations:
            if type(observation) is not OpaqueEdgeObservation:
                raise TypeError("passive table must contain exact OpaqueEdgeObservation rows")
            digest = observation.request.request_sha256
            if digest not in defined_hashes:
                raise ValueError("passive observation is absent from the defined mask")
            if digest in passive_hashes:
                raise ValueError("duplicate passive edge observation")
            passive_hashes.add(digest)
            reconcile(observation.request.source_word, observation.source_answers)
            reconcile(observation.request.program, observation.target_answers)
        candidate_hashes = tuple(row.request_sha256 for row in self.candidate_edge_requests)
        if len(set(candidate_hashes)) != len(candidate_hashes):
            raise ValueError("candidate pool contains duplicate requests")
        if set(candidate_hashes) != defined_hashes - passive_hashes:
            raise ValueError("candidate pool must exactly complement passive observations")
        if set(candidate_hashes).intersection(passive_hashes):
            raise ValueError("candidate and passive edges must be disjoint")
        if expected_candidate_count:
            structurally_known_words = set(word_answers)
            for ordinal, request in enumerate(
                self.candidate_edge_requests[:_ACTIVE_RESPONSE_COUNT], 1
            ):
                if request.source_word not in structurally_known_words:
                    raise ValueError(
                        f"active request {ordinal} uses a source unavailable before its response"
                    )
                structurally_known_words.add(request.program)
            for request in self.candidate_edge_requests[_ACTIVE_RESPONSE_COUNT:]:
                if request.source_word not in structurally_known_words:
                    raise ValueError("sealed request source is unavailable after active acquisition")

        passive_payload = {
            "state_observations": [
                {"word": row.word, "answers": row.answers}
                for row in self.passive_state_observations
            ],
            "edge_observations": [
                _observation_payload(row) for row in self.passive_edge_observations
            ],
        }
        domain_payload = {
            "defined": [_request_payload(row) for row in self.defined_edge_requests],
            "undefined": [_request_payload(row) for row in self.undefined_edge_requests],
        }
        candidate_payload = [_request_payload(row) for row in self.candidate_edge_requests]
        if _require_sha256("passive_table_sha256", self.passive_table_sha256) != _sha256(passive_payload):
            raise ValueError("passive-table digest mismatch")
        if _require_sha256("domain_mask_sha256", self.domain_mask_sha256) != _sha256(domain_payload):
            raise ValueError("definedness-mask digest mismatch")
        if _require_sha256("candidate_pool_sha256", self.candidate_pool_sha256) != _sha256(candidate_payload):
            raise ValueError("candidate-pool digest mismatch")
        expected_input = _sha256(self._payload(include_input_sha=False))
        if _require_sha256("input_sha256", self.input_sha256) != expected_input:
            raise ValueError("input_sha256 does not bind the complete learner input")

    @property
    def is_full_support_control(self) -> bool:
        return len(self.passive_edge_observations) == _LEGAL_EDGE_COUNT

    def _payload(self, *, include_input_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "event_tokens": self.event_tokens,
            "query_tokens": self.query_tokens,
            "answer_tokens": self.answer_tokens,
            "passive_state_observations": [
                {"word": row.word, "answers": row.answers}
                for row in self.passive_state_observations
            ],
            "passive_edge_observations": [
                _observation_payload(row) for row in self.passive_edge_observations
            ],
            "candidate_edge_requests": [
                _request_payload(row) for row in self.candidate_edge_requests
            ],
            "defined_edge_requests": [
                _request_payload(row) for row in self.defined_edge_requests
            ],
            "undefined_edge_requests": [
                _request_payload(row) for row in self.undefined_edge_requests
            ],
            "budgets": self.budgets.payload(),
            "domain_mask_is_learner_visible_supervision": self.domain_mask_is_learner_visible_supervision,
            "undefined_words_are_absent_constraints": self.undefined_words_are_absent_constraints,
            "illegal_words_encoded_as_zero_or_dead": self.illegal_words_encoded_as_zero_or_dead,
            "state_changing_event_subalphabet_only": self.state_changing_event_subalphabet_only,
            "diagnostic_queries_are_output_channels_not_events": self.diagnostic_queries_are_output_channels_not_events,
            "categorical_state_space_declared_as_full_product": self.categorical_state_space_declared_as_full_product,
            "full_product_state_grammar_counted_as_supervision": self.full_product_state_grammar_counted_as_supervision,
            "passive_table_sha256": self.passive_table_sha256,
            "domain_mask_sha256": self.domain_mask_sha256,
            "candidate_pool_sha256": self.candidate_pool_sha256,
        }
        if include_input_sha:
            payload["input_sha256"] = self.input_sha256
        return payload

    def payload(self) -> dict[str, object]:
        return self._payload(include_input_sha=True)


def _make_learner_input(
    *,
    event_tokens: tuple[str, ...],
    query_tokens: tuple[str, ...],
    answer_tokens: tuple[str, ...],
    passive_state_observations: tuple[OpaqueStateObservation, ...],
    passive_edge_observations: tuple[OpaqueEdgeObservation, ...],
    candidate_edge_requests: tuple[OpaqueEdgeRequest, ...],
    defined_edge_requests: tuple[OpaqueEdgeRequest, ...],
    undefined_edge_requests: tuple[OpaqueEdgeRequest, ...],
    budgets: PartialOperatorBudgets,
) -> PartialOperatorLearnerInput:
    passive_payload = {
        "state_observations": [
            {"word": row.word, "answers": row.answers}
            for row in passive_state_observations
        ],
        "edge_observations": [
            _observation_payload(row) for row in passive_edge_observations
        ],
    }
    domain_payload = {
        "defined": [_request_payload(row) for row in defined_edge_requests],
        "undefined": [_request_payload(row) for row in undefined_edge_requests],
    }
    candidate_payload = [_request_payload(row) for row in candidate_edge_requests]
    kwargs: dict[str, object] = {
        "event_tokens": event_tokens,
        "query_tokens": query_tokens,
        "answer_tokens": answer_tokens,
        "passive_state_observations": passive_state_observations,
        "passive_edge_observations": passive_edge_observations,
        "candidate_edge_requests": candidate_edge_requests,
        "defined_edge_requests": defined_edge_requests,
        "undefined_edge_requests": undefined_edge_requests,
        "budgets": budgets,
        "domain_mask_is_learner_visible_supervision": True,
        "undefined_words_are_absent_constraints": True,
        "illegal_words_encoded_as_zero_or_dead": False,
        "state_changing_event_subalphabet_only": True,
        "diagnostic_queries_are_output_channels_not_events": True,
        "categorical_state_space_declared_as_full_product": True,
        "full_product_state_grammar_counted_as_supervision": True,
        "passive_table_sha256": _sha256(passive_payload),
        "domain_mask_sha256": _sha256(domain_payload),
        "candidate_pool_sha256": _sha256(candidate_payload),
        "schema": _INPUT_SCHEMA,
    }
    unhashed = {
        "schema": kwargs["schema"],
        **{key: value for key, value in kwargs.items() if key != "schema"},
    }
    # Convert nested exact rows through the same public payload used by validation.
    temporary_payload = {
        "schema": _INPUT_SCHEMA,
        "event_tokens": event_tokens,
        "query_tokens": query_tokens,
        "answer_tokens": answer_tokens,
        "passive_state_observations": passive_payload["state_observations"],
        "passive_edge_observations": passive_payload["edge_observations"],
        "candidate_edge_requests": candidate_payload,
        "defined_edge_requests": domain_payload["defined"],
        "undefined_edge_requests": domain_payload["undefined"],
        "budgets": budgets.payload(),
        "domain_mask_is_learner_visible_supervision": True,
        "undefined_words_are_absent_constraints": True,
        "illegal_words_encoded_as_zero_or_dead": False,
        "state_changing_event_subalphabet_only": True,
        "diagnostic_queries_are_output_channels_not_events": True,
        "categorical_state_space_declared_as_full_product": True,
        "full_product_state_grammar_counted_as_supervision": True,
        "passive_table_sha256": kwargs["passive_table_sha256"],
        "domain_mask_sha256": kwargs["domain_mask_sha256"],
        "candidate_pool_sha256": kwargs["candidate_pool_sha256"],
    }
    del unhashed
    return PartialOperatorLearnerInput(
        **kwargs,  # type: ignore[arg-type]
        input_sha256=_sha256(temporary_payload),
    )


def _diagnostic_row(
    answers: tuple[str, ...], answer_tokens: tuple[str, ...]
) -> RationalVector:
    _validate_answers("diagnostic answers", answers)
    baseline = min(answer_tokens)
    contrasts = tuple(token for token in sorted(answer_tokens) if token != baseline)
    return tuple(
        [Rational(1)]
        + [Rational(int(answers[0] == token)) for token in contrasts]
        + [Rational(int(answers[1] == token)) for token in contrasts]
    )


def _row_coordinates(row: RationalVector, basis_rows: RationalMatrix) -> RationalVector:
    dimension, width = _matrix_shape(basis_rows)
    if len(row) != width:
        raise ValueError("row width does not match realization basis")
    transposed = _matrix(tuple(basis_rows[i][j] for i in range(dimension)) for j in range(width))
    rank, pivot_rows, _ = _rank_profile(transposed)
    if rank != dimension:
        raise ValueError("basis rows are linearly dependent")
    pivot_columns = pivot_rows
    square = _matrix(
        tuple(basis_rows[i][column] for column in pivot_columns)
        for i in range(dimension)
    )
    selected = tuple(row[column] for column in pivot_columns)
    coordinates = _row_times_matrix(selected, _inverse(square))
    reconstructed = tuple(
        Rational.from_fraction(
            sum(
                coordinates[i].as_fraction() * basis_rows[i][j].as_fraction()
                for i in range(dimension)
            )
        )
        for j in range(width)
    )
    if reconstructed != row:
        raise ValueError("row is outside the supplied realization span")
    return coordinates


@dataclass(frozen=True)
class ExactRankCertificate:
    matrix: RationalMatrix
    rank: int
    pivot_row_indices: tuple[int, ...]
    pivot_column_indices: tuple[int, ...]
    nonsingular_minor_determinant: Rational
    matrix_sha256: str
    certificate_sha256: str
    schema: str = _RANK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _RANK_SCHEMA:
            raise ValueError("unknown exact-rank certificate schema")
        rows, columns = _matrix_shape(self.matrix)
        expected_rank, row_indices, column_indices = _rank_profile(self.matrix)
        if self.rank != expected_rank:
            raise ValueError("rank certificate reports an incorrect rank")
        if self.pivot_row_indices != row_indices or self.pivot_column_indices != column_indices:
            raise ValueError("rank pivot profile mismatch")
        if len(row_indices) != expected_rank:
            raise ValueError("rank pivot count mismatch")
        if expected_rank:
            minor = _matrix(
                tuple(self.matrix[row][column] for column in column_indices)
                for row in row_indices
            )
            determinant = _determinant(minor)
            if determinant.numerator == 0:
                raise ValueError("rank witness minor is singular")
        else:
            determinant = Rational(1)
        if self.nonsingular_minor_determinant != determinant:
            raise ValueError("rank witness determinant mismatch")
        if _require_sha256("matrix_sha256", self.matrix_sha256) != _sha256(self.matrix):
            raise ValueError("rank matrix digest mismatch")
        expected = _sha256(self._payload(include_certificate_sha=False))
        if _require_sha256("certificate_sha256", self.certificate_sha256) != expected:
            raise ValueError("rank certificate digest mismatch")
        if rows < expected_rank or columns < expected_rank:
            raise ValueError("invalid rank dimensions")

    def _payload(self, *, include_certificate_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "matrix": self.matrix,
            "rank": self.rank,
            "pivot_row_indices": self.pivot_row_indices,
            "pivot_column_indices": self.pivot_column_indices,
            "nonsingular_minor_determinant": self.nonsingular_minor_determinant,
            "matrix_sha256": self.matrix_sha256,
        }
        if include_certificate_sha:
            payload["certificate_sha256"] = self.certificate_sha256
        return payload


def _rank_certificate(matrix: RationalMatrix) -> ExactRankCertificate:
    rank, rows, columns = _rank_profile(matrix)
    determinant = (
        _determinant(_matrix(tuple(matrix[row][column] for column in columns) for row in rows))
        if rank
        else Rational(1)
    )
    kwargs = {
        "matrix": matrix,
        "rank": rank,
        "pivot_row_indices": rows,
        "pivot_column_indices": columns,
        "nonsingular_minor_determinant": determinant,
        "matrix_sha256": _sha256(matrix),
        "schema": _RANK_SCHEMA,
    }
    payload = {"schema": _RANK_SCHEMA, **{key: value for key, value in kwargs.items() if key != "schema"}}
    return ExactRankCertificate(**kwargs, certificate_sha256=_sha256(payload))


@dataclass(frozen=True)
class ObservedEventRank:
    event_token: str
    observation_count: int
    observed_source_rank: int

    def __post_init__(self) -> None:
        _require_token("event_token", self.event_token)
        _plain_int("observation_count", self.observation_count)
        _plain_int("observed_source_rank", self.observed_source_rank)
        if self.observed_source_rank > self.observation_count:
            raise ValueError("observed source rank cannot exceed observation count")


@dataclass(frozen=True)
class ExactPartialOperatorCertificate:
    event_token: str
    ambient_rank: int
    legal_domain_rank: int
    observed_source_rank: int
    restricted_nullity: int
    total_extension_nullity: int
    domain_basis_coordinates: RationalMatrix
    image_basis_coordinates: RationalMatrix
    off_domain_annihilator_basis: RationalMatrix
    restricted_map_identified: bool
    off_domain_extension_identified: bool
    undefined_inputs_encoded_as_zero_or_dead: bool
    total_operator: None
    certificate_sha256: str
    schema: str = _OPERATOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _OPERATOR_SCHEMA:
            raise ValueError("unknown partial-operator certificate schema")
        _require_token("event_token", self.event_token)
        ambient = _plain_int("ambient_rank", self.ambient_rank, 1)
        domain = _plain_int("legal_domain_rank", self.legal_domain_rank, 1)
        observed = _plain_int("observed_source_rank", self.observed_source_rank)
        if not (observed <= domain <= ambient):
            raise ValueError("operator ranks must satisfy observed <= domain <= ambient")
        if self.restricted_nullity != ambient * (domain - observed):
            raise ValueError("restricted nullity arithmetic mismatch")
        if self.total_extension_nullity != ambient * (ambient - domain):
            raise ValueError("total-extension nullity arithmetic mismatch")
        source_rows, source_width = _matrix_shape(self.domain_basis_coordinates)
        image_rows, image_width = _matrix_shape(self.image_basis_coordinates)
        if source_rows != domain or image_rows != domain:
            raise ValueError("partial-map bases must contain legal_domain_rank rows")
        if source_width != ambient or image_width != ambient:
            raise ValueError("partial-map coordinates must use the ambient realization")
        if _rank_profile(self.domain_basis_coordinates)[0] != domain:
            raise ValueError("operator domain basis is dependent")
        annihilator_rows, annihilator_width = _matrix_shape(
            self.off_domain_annihilator_basis
        )
        if annihilator_rows != ambient - domain or annihilator_width != ambient:
            raise ValueError("off-domain annihilator basis shape mismatch")
        if _rank_profile(self.off_domain_annihilator_basis)[0] != ambient - domain:
            raise ValueError("off-domain annihilator rows must be independent")
        products = _matmul(
            self.domain_basis_coordinates,
            _matrix(
                tuple(
                    self.off_domain_annihilator_basis[row][column]
                    for row in range(annihilator_rows)
                )
                for column in range(ambient)
            ),
        )
        if any(value.numerator for row in products for value in row):
            raise ValueError("annihilator witness does not vanish on the legal domain")
        if _require_bool("restricted_map_identified", self.restricted_map_identified) != (observed == domain):
            raise ValueError("restricted-map identification flag is inconsistent")
        if _require_bool("off_domain_extension_identified", self.off_domain_extension_identified):
            raise ValueError("off-domain total extensions are never identified")
        if _require_bool(
            "undefined_inputs_encoded_as_zero_or_dead",
            self.undefined_inputs_encoded_as_zero_or_dead,
        ):
            raise ValueError("undefined inputs cannot be zero/dead encoded")
        if self.total_operator is not None:
            raise ValueError("Contract B forbids a serialized total operator")
        expected = _sha256(self._payload(include_certificate_sha=False))
        if _require_sha256("certificate_sha256", self.certificate_sha256) != expected:
            raise ValueError("partial-operator certificate digest mismatch")

    def _payload(self, *, include_certificate_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "event_token": self.event_token,
            "ambient_rank": self.ambient_rank,
            "legal_domain_rank": self.legal_domain_rank,
            "observed_source_rank": self.observed_source_rank,
            "restricted_nullity": self.restricted_nullity,
            "total_extension_nullity": self.total_extension_nullity,
            "domain_basis_coordinates": self.domain_basis_coordinates,
            "image_basis_coordinates": self.image_basis_coordinates,
            "off_domain_annihilator_basis": self.off_domain_annihilator_basis,
            "restricted_map_identified": self.restricted_map_identified,
            "off_domain_extension_identified": self.off_domain_extension_identified,
            "undefined_inputs_encoded_as_zero_or_dead": self.undefined_inputs_encoded_as_zero_or_dead,
            "total_operator": self.total_operator,
        }
        if include_certificate_sha:
            payload["certificate_sha256"] = self.certificate_sha256
        return payload


@dataclass(frozen=True)
class ExactPartialRealization:
    """A minimal opaque predictive realization with partial event maps only."""

    event_tokens: tuple[str, ...]
    query_tokens: tuple[str, ...]
    answer_tokens: tuple[str, ...]
    ambient_rank: int
    rank_certificate: ExactRankCertificate
    basis_words: tuple[OpaqueWord, ...]
    basis_diagnostic_rows: RationalMatrix
    coordinate_answer_codebook: tuple[tuple[RationalVector, tuple[str, ...]], ...]
    operator_certificates: tuple[ExactPartialOperatorCertificate, ...]
    restricted_maps_complete: bool
    total_operator: None
    model_sha256: str
    schema: str = _REALIZATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _REALIZATION_SCHEMA:
            raise ValueError("unknown exact partial-realization schema")
        if len(self.event_tokens) != _EVENT_COUNT or len(set(self.event_tokens)) != _EVENT_COUNT:
            raise ValueError("realization must bind the ten state-changing event tokens")
        if len(self.query_tokens) != _QUERY_COUNT or len(self.answer_tokens) != _ANSWER_COUNT:
            raise ValueError("realization vocabulary size mismatch")
        for token in self.event_tokens + self.query_tokens + self.answer_tokens:
            _require_token("realization token", token)
        if type(self.rank_certificate) is not ExactRankCertificate:
            raise TypeError("rank_certificate must be exact ExactRankCertificate")
        if self.ambient_rank != self.rank_certificate.rank:
            raise ValueError("realization ambient rank disagrees with rank certificate")
        if len(self.basis_words) != self.ambient_rank:
            raise ValueError("basis word count must equal ambient rank")
        rows, width = _matrix_shape(self.basis_diagnostic_rows)
        if rows != self.ambient_rank or width != _FULL_RANK:
            raise ValueError("diagnostic basis shape mismatch")
        if _rank_profile(self.basis_diagnostic_rows)[0] != self.ambient_rank:
            raise ValueError("diagnostic basis rows are dependent")
        seen_coordinates: set[RationalVector] = set()
        seen_answers: set[tuple[str, ...]] = set()
        for coordinates, answers in self.coordinate_answer_codebook:
            if not isinstance(coordinates, tuple) or len(coordinates) != self.ambient_rank:
                raise ValueError("codebook coordinate width mismatch")
            if any(type(value) is not Rational for value in coordinates):
                raise TypeError("codebook coordinates must be exact rationals")
            _validate_answers("codebook answers", answers)
            if coordinates in seen_coordinates or answers in seen_answers:
                raise ValueError("codebook must be bijective on known categorical states")
            seen_coordinates.add(coordinates)
            seen_answers.add(answers)
        if len(self.operator_certificates) not in (0, _EVENT_COUNT):
            raise ValueError("realization must contain zero or ten partial-map certificates")
        if len({row.event_token for row in self.operator_certificates}) != len(self.operator_certificates):
            raise ValueError("duplicate partial operator token")
        complete = (
            len(self.operator_certificates) == _EVENT_COUNT
            and all(row.restricted_map_identified for row in self.operator_certificates)
        )
        if _require_bool("restricted_maps_complete", self.restricted_maps_complete) != complete:
            raise ValueError("restricted-map completion flag mismatch")
        if self.total_operator is not None:
            raise ValueError("a total operator cannot be attached to a partial realization")
        expected = _sha256(self._payload(include_model_sha=False))
        if _require_sha256("model_sha256", self.model_sha256) != expected:
            raise ValueError("partial-realization digest mismatch")

    def _payload(self, *, include_model_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "event_tokens": self.event_tokens,
            "query_tokens": self.query_tokens,
            "answer_tokens": self.answer_tokens,
            "ambient_rank": self.ambient_rank,
            "rank_certificate": self.rank_certificate._payload(include_certificate_sha=True),
            "basis_words": self.basis_words,
            "basis_diagnostic_rows": self.basis_diagnostic_rows,
            "coordinate_answer_codebook": self.coordinate_answer_codebook,
            "operator_certificates": [
                row._payload(include_certificate_sha=True)
                for row in self.operator_certificates
            ],
            "restricted_maps_complete": self.restricted_maps_complete,
            "total_operator": self.total_operator,
        }
        if include_model_sha:
            payload["model_sha256"] = self.model_sha256
        return payload

    def answers_to_coordinates(self, answers: tuple[str, ...]) -> RationalVector:
        row = _diagnostic_row(answers, self.answer_tokens)
        return _row_coordinates(row, self.basis_diagnostic_rows)

    def coordinates_to_answers(self, coordinates: RationalVector) -> tuple[str, ...]:
        matches = [answers for row, answers in self.coordinate_answer_codebook if row == coordinates]
        if len(matches) != 1:
            raise ValueError("predicted coordinate is not a uniquely known categorical state")
        return matches[0]

    def apply_event(
        self, source_coordinates: RationalVector, event_token: str
    ) -> RationalVector:
        matches = [row for row in self.operator_certificates if row.event_token == event_token]
        if len(matches) != 1:
            raise ValueError("partial operator is absent for this event token")
        operator = matches[0]
        coefficients = _row_coordinates(
            source_coordinates, operator.domain_basis_coordinates
        )
        return tuple(
            Rational.from_fraction(
                sum(
                    coefficients[i].as_fraction()
                    * operator.image_basis_coordinates[i][j].as_fraction()
                    for i in range(operator.legal_domain_rank)
                )
            )
            for j in range(self.ambient_rank)
        )

    def predict_answers(
        self,
        program: Sequence[str],
        *,
        initial_answers: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not self.restricted_maps_complete:
            raise ValueError("prediction requires all restricted partial maps")
        if not isinstance(program, (tuple, list)):
            raise TypeError("program must be a tuple or list of opaque event tokens")
        current = self.answers_to_coordinates(initial_answers)
        for token in program:
            _require_token("program token", token)
            if token not in self.event_tokens:
                raise ValueError("program contains an undeclared event token")
            current = self.apply_event(current, token)
        return self.coordinates_to_answers(current)


@dataclass(frozen=True)
class ActiveAcquisitionCommitment:
    learner_input_sha256: str
    candidate_pool_sha256: str
    selected_request_sha256s: tuple[str, ...]
    sealed_request_sha256s: tuple[str, ...]
    selection_rule: str
    selected_before_any_active_answer: bool
    active_basis_supplied_by_trusted_controller: bool
    controller_order_encodes_semantically_designed_basis: bool
    learner_selected_acquisition_basis: bool
    learner_selection_used_semantic_roles: bool
    learner_selection_used_controller_nonce: bool
    selection_used_sealed_answers: bool
    active_batch_call_count: int
    expected_response_count: int
    expected_returned_categorical_token_count: int
    commitment_sha256: str
    schema: str = _COMMITMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _COMMITMENT_SCHEMA:
            raise ValueError("unknown active-acquisition commitment schema")
        _require_sha256("learner_input_sha256", self.learner_input_sha256)
        _require_sha256("candidate_pool_sha256", self.candidate_pool_sha256)
        for digest in self.selected_request_sha256s + self.sealed_request_sha256s:
            _require_sha256("request commitment digest", digest)
        selected_count = len(self.selected_request_sha256s)
        sealed_count = len(self.sealed_request_sha256s)
        if (selected_count, sealed_count) not in ((0, 0), (_ACTIVE_RESPONSE_COUNT, _SEALED_EDGE_COUNT)):
            raise ValueError("active commitment must encode the frozen 15/8 split or a control")
        if len(set(self.selected_request_sha256s + self.sealed_request_sha256s)) != selected_count + sealed_count:
            raise ValueError("active and sealed commitment rows overlap")
        if self.selection_rule != "trusted_controller_supplied_fixed_15_edge_legal_domain_excitation_basis":
            raise ValueError("unknown active acquisition rule")
        has_active = selected_count > 0
        for name, value, required in (
            ("selected_before_any_active_answer", self.selected_before_any_active_answer, True),
            ("active_basis_supplied_by_trusted_controller", self.active_basis_supplied_by_trusted_controller, has_active),
            ("controller_order_encodes_semantically_designed_basis", self.controller_order_encodes_semantically_designed_basis, has_active),
            ("learner_selected_acquisition_basis", self.learner_selected_acquisition_basis, False),
            ("learner_selection_used_semantic_roles", self.learner_selection_used_semantic_roles, False),
            ("learner_selection_used_controller_nonce", self.learner_selection_used_controller_nonce, False),
            ("selection_used_sealed_answers", self.selection_used_sealed_answers, False),
        ):
            if _require_bool(name, value) is not required:
                raise ValueError(f"{name} must be {required}")
        expected_calls = 2 * int(selected_count > 0)
        if self.active_batch_call_count != expected_calls:
            raise ValueError("active batch call count mismatch")
        if self.expected_response_count != selected_count:
            raise ValueError("active response count mismatch")
        if self.expected_returned_categorical_token_count != 2 * selected_count:
            raise ValueError("active categorical-token count mismatch")
        expected = _sha256(self._payload(include_commitment_sha=False))
        if _require_sha256("commitment_sha256", self.commitment_sha256) != expected:
            raise ValueError("active commitment digest mismatch")

    def _payload(self, *, include_commitment_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "learner_input_sha256": self.learner_input_sha256,
            "candidate_pool_sha256": self.candidate_pool_sha256,
            "selected_request_sha256s": self.selected_request_sha256s,
            "sealed_request_sha256s": self.sealed_request_sha256s,
            "selection_rule": self.selection_rule,
            "selected_before_any_active_answer": self.selected_before_any_active_answer,
            "active_basis_supplied_by_trusted_controller": self.active_basis_supplied_by_trusted_controller,
            "controller_order_encodes_semantically_designed_basis": self.controller_order_encodes_semantically_designed_basis,
            "learner_selected_acquisition_basis": self.learner_selected_acquisition_basis,
            "learner_selection_used_semantic_roles": self.learner_selection_used_semantic_roles,
            "learner_selection_used_controller_nonce": self.learner_selection_used_controller_nonce,
            "selection_used_sealed_answers": self.selection_used_sealed_answers,
            "active_batch_call_count": self.active_batch_call_count,
            "expected_response_count": self.expected_response_count,
            "expected_returned_categorical_token_count": self.expected_returned_categorical_token_count,
        }
        if include_commitment_sha:
            payload["commitment_sha256"] = self.commitment_sha256
        return payload


@dataclass(frozen=True)
class PassivePartialDiscovery:
    scope: PartialOperatorScope
    status: PartialOperatorStatus
    learner_input_sha256: str
    passive_edge_count: int
    passive_rank_certificate: ExactRankCertificate
    observed_event_ranks: tuple[ObservedEventRank, ...]
    realization: ExactPartialRealization
    active_commitment: ActiveAcquisitionCommitment
    compatible_omission_hypothesis_count: None
    omission_hypothesis_analysis_performed_inside_estimator: bool
    definedness_mask_counted_as_supervision: bool
    passive_answers_only: bool
    semantic_roles_received: bool
    controller_nonce_received: bool
    executor_received: bool
    result_sha256: str
    schema: str = _PASSIVE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _PASSIVE_SCHEMA:
            raise ValueError("unknown passive partial-discovery schema")
        if self.scope is not PartialOperatorScope.GUARDED_ABSENCE_AWARE_PARTIAL_OPERATORS:
            raise ValueError("passive result scope mismatch")
        if self.status is not PartialOperatorStatus.SYNTHETIC_PROTOCOL_IMPLEMENTATION_REHEARSAL:
            raise ValueError("passive result status mismatch")
        _require_sha256("learner_input_sha256", self.learner_input_sha256)
        if self.passive_edge_count not in (_PASSIVE_EDGE_COUNT, _LEGAL_EDGE_COUNT):
            raise ValueError("passive edge count mismatch")
        if type(self.passive_rank_certificate) is not ExactRankCertificate:
            raise TypeError("passive_rank_certificate must be exact")
        expected_rank = _FULL_RANK if self.passive_edge_count == _LEGAL_EDGE_COUNT else _PASSIVE_RANK
        if self.passive_rank_certificate.rank != expected_rank:
            raise ValueError("passive rank does not match protocol arm")
        if type(self.realization) is not ExactPartialRealization:
            raise TypeError("realization must be exact ExactPartialRealization")
        if self.realization.ambient_rank != expected_rank:
            raise ValueError("passive realization rank mismatch")
        if type(self.active_commitment) is not ActiveAcquisitionCommitment:
            raise TypeError("active_commitment must be exact")
        if self.active_commitment.learner_input_sha256 != self.learner_input_sha256:
            raise ValueError("active commitment refers to another learner input")
        if len(self.observed_event_ranks) != _EVENT_COUNT:
            raise ValueError("one observed-rank row is required per event token")
        if len({row.event_token for row in self.observed_event_ranks}) != _EVENT_COUNT:
            raise ValueError("observed-rank event tokens must be unique")
        if self.compatible_omission_hypothesis_count is not None:
            raise ValueError("pure estimator cannot attach semantic omission hypotheses")
        for name, value, required in (
            ("definedness_mask_counted_as_supervision", self.definedness_mask_counted_as_supervision, True),
            ("passive_answers_only", self.passive_answers_only, True),
            ("omission_hypothesis_analysis_performed_inside_estimator", self.omission_hypothesis_analysis_performed_inside_estimator, False),
            ("semantic_roles_received", self.semantic_roles_received, False),
            ("controller_nonce_received", self.controller_nonce_received, False),
            ("executor_received", self.executor_received, False),
        ):
            if _require_bool(name, value) is not required:
                raise ValueError(f"{name} must be {required}")
        expected = _sha256(self._payload(include_result_sha=False))
        if _require_sha256("result_sha256", self.result_sha256) != expected:
            raise ValueError("passive discovery digest mismatch")

    def _payload(self, *, include_result_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "scope": self.scope,
            "status": self.status,
            "learner_input_sha256": self.learner_input_sha256,
            "passive_edge_count": self.passive_edge_count,
            "passive_rank_certificate": self.passive_rank_certificate._payload(include_certificate_sha=True),
            "observed_event_ranks": [row.__dict__ for row in self.observed_event_ranks],
            "realization": self.realization._payload(include_model_sha=True),
            "active_commitment": self.active_commitment._payload(include_commitment_sha=True),
            "compatible_omission_hypothesis_count": self.compatible_omission_hypothesis_count,
            "omission_hypothesis_analysis_performed_inside_estimator": self.omission_hypothesis_analysis_performed_inside_estimator,
            "definedness_mask_counted_as_supervision": self.definedness_mask_counted_as_supervision,
            "passive_answers_only": self.passive_answers_only,
            "semantic_roles_received": self.semantic_roles_received,
            "controller_nonce_received": self.controller_nonce_received,
            "executor_received": self.executor_received,
        }
        if include_result_sha:
            payload["result_sha256"] = self.result_sha256
        return payload


@dataclass(frozen=True)
class OneResponseEventDeficit:
    event_token: str
    legal_domain_rank: int
    observed_source_rank: int
    remaining_source_rank_deficit: int

    def __post_init__(self) -> None:
        _require_token("event_token", self.event_token)
        domain = _plain_int("legal_domain_rank", self.legal_domain_rank, 1)
        observed = _plain_int("observed_source_rank", self.observed_source_rank)
        if observed > domain:
            raise ValueError("observed rank exceeds legal-domain rank")
        if self.remaining_source_rank_deficit != domain - observed:
            raise ValueError("one-response source-rank deficit arithmetic mismatch")


@dataclass(frozen=True)
class OneResponseCheckpoint:
    learner_input_sha256: str
    passive_result_sha256: str
    commitment_sha256: str
    first_response_sha256: str
    response_count: int
    returned_categorical_token_count: int
    rank_certificate: ExactRankCertificate
    rank_after_first_response: int
    operator_maps_identified: bool
    unidentified_event_token_count: int
    event_rank_deficits: tuple[OneResponseEventDeficit, ...]
    aggregate_remaining_source_rank_deficit: int
    unanswered_candidate_request_count: int
    compatible_outcome_witness_request_sha256: str
    compatible_outcome_witness_event_token: str
    preexisting_witness_event_observation_count: int
    compatible_outcome_a: tuple[str, ...]
    compatible_outcome_b: tuple[str, ...]
    branch_source_coordinates: RationalMatrix
    branch_a_image_coordinates: RationalMatrix
    branch_b_image_coordinates: RationalMatrix
    branch_source_rank: int
    branch_a_constraint_system_sha256: str
    branch_b_constraint_system_sha256: str
    branch_a_linear_system_consistent: bool
    branch_b_linear_system_consistent: bool
    differing_image_row_index: int
    differing_image_delta: RationalVector
    compatible_outcome_analysis_used_first_response_labels: bool
    compatible_outcome_analysis_used_responses_2_through_15: bool
    compatible_outcomes_are_exact_unobserved_full_product_rows: bool
    actual_next_response_read: bool
    sealed_answer_read: bool
    differing_restricted_map_witness: bool
    checkpoint_sha256: str
    schema: str = _ONE_RESPONSE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _ONE_RESPONSE_SCHEMA:
            raise ValueError("unknown one-response checkpoint schema")
        for name, digest in (
            ("learner_input_sha256", self.learner_input_sha256),
            ("passive_result_sha256", self.passive_result_sha256),
            ("commitment_sha256", self.commitment_sha256),
            ("first_response_sha256", self.first_response_sha256),
            ("compatible_outcome_witness_request_sha256", self.compatible_outcome_witness_request_sha256),
        ):
            _require_sha256(name, digest)
        _require_token(
            "compatible_outcome_witness_event_token",
            self.compatible_outcome_witness_event_token,
        )
        if self.preexisting_witness_event_observation_count != 1:
            raise ValueError("branch witness must include the one existing same-event edge")
        if self.response_count != 1 or self.returned_categorical_token_count != 2:
            raise ValueError("one-response accounting must be exactly one call row/two labels")
        if self.rank_after_first_response != _FULL_RANK or self.rank_certificate.rank != _FULL_RANK:
            raise ValueError("the first response must raise the diagnostic rank to five")
        if _require_bool("operator_maps_identified", self.operator_maps_identified):
            raise ValueError("one response cannot identify all restricted maps")
        if len(self.event_rank_deficits) != _EVENT_COUNT:
            raise ValueError("one deficit row is required per event token")
        if len({row.event_token for row in self.event_rank_deficits}) != _EVENT_COUNT:
            raise ValueError("one-response deficit event tokens must be unique")
        derived_unidentified = sum(
            row.remaining_source_rank_deficit > 0
            for row in self.event_rank_deficits
        )
        if self.unidentified_event_token_count != derived_unidentified or derived_unidentified != 9:
            raise ValueError("exactly nine event maps must remain unidentified")
        derived_deficit = sum(
            row.remaining_source_rank_deficit for row in self.event_rank_deficits
        )
        if self.aggregate_remaining_source_rank_deficit != derived_deficit or derived_deficit != 14:
            raise ValueError("one-response aggregate source-rank deficit must be 14")
        if self.unanswered_candidate_request_count != _ACTIVE_RESPONSE_COUNT + _SEALED_EDGE_COUNT - 1:
            raise ValueError("unanswered candidate count mismatch")
        _validate_answers("compatible_outcome_a", self.compatible_outcome_a)
        _validate_answers("compatible_outcome_b", self.compatible_outcome_b)
        if self.compatible_outcome_a == self.compatible_outcome_b:
            raise ValueError("compatible outcomes must be distinct")
        source_rows, source_width = _matrix_shape(self.branch_source_coordinates)
        a_rows, a_width = _matrix_shape(self.branch_a_image_coordinates)
        b_rows, b_width = _matrix_shape(self.branch_b_image_coordinates)
        if source_rows != 2 or a_rows != 2 or b_rows != 2:
            raise ValueError("one-response branch witness must use two edge constraints")
        if source_width != _FULL_RANK or a_width != _FULL_RANK or b_width != _FULL_RANK:
            raise ValueError("branch witnesses must use rank-five coordinates")
        actual_source_rank = _rank_profile(self.branch_source_coordinates)[0]
        if self.branch_source_rank != actual_source_rank or actual_source_rank != 2:
            raise ValueError("second response must add an independent source direction")
        expected_a_hash = _sha256(
            {
                "source_coordinates": self.branch_source_coordinates,
                "image_coordinates": self.branch_a_image_coordinates,
            }
        )
        expected_b_hash = _sha256(
            {
                "source_coordinates": self.branch_source_coordinates,
                "image_coordinates": self.branch_b_image_coordinates,
            }
        )
        if _require_sha256("branch_a_constraint_system_sha256", self.branch_a_constraint_system_sha256) != expected_a_hash:
            raise ValueError("branch-A constraint-system digest mismatch")
        if _require_sha256("branch_b_constraint_system_sha256", self.branch_b_constraint_system_sha256) != expected_b_hash:
            raise ValueError("branch-B constraint-system digest mismatch")
        if not _require_bool("branch_a_linear_system_consistent", self.branch_a_linear_system_consistent):
            raise ValueError("branch A must be an exact compatible linear system")
        if not _require_bool("branch_b_linear_system_consistent", self.branch_b_linear_system_consistent):
            raise ValueError("branch B must be an exact compatible linear system")
        if not (0 <= self.differing_image_row_index < 2):
            raise ValueError("differing image-row index lies outside the branch witness")
        actual_delta = tuple(
            Rational.from_fraction(
                self.branch_a_image_coordinates[self.differing_image_row_index][column].as_fraction()
                - self.branch_b_image_coordinates[self.differing_image_row_index][column].as_fraction()
            )
            for column in range(_FULL_RANK)
        )
        if self.differing_image_delta != actual_delta or not any(
            value.numerator for value in actual_delta
        ):
            raise ValueError("branch maps must have a concrete nonzero image difference")
        for name, value, required in (
            ("compatible_outcome_analysis_used_first_response_labels", self.compatible_outcome_analysis_used_first_response_labels, True),
            ("compatible_outcome_analysis_used_responses_2_through_15", self.compatible_outcome_analysis_used_responses_2_through_15, False),
            ("compatible_outcomes_are_exact_unobserved_full_product_rows", self.compatible_outcomes_are_exact_unobserved_full_product_rows, True),
            ("actual_next_response_read", self.actual_next_response_read, False),
            ("sealed_answer_read", self.sealed_answer_read, False),
            ("differing_restricted_map_witness", self.differing_restricted_map_witness, True),
        ):
            if _require_bool(name, value) is not required:
                raise ValueError(f"{name} must be {required}")
        expected = _sha256(self._payload(include_checkpoint_sha=False))
        if _require_sha256("checkpoint_sha256", self.checkpoint_sha256) != expected:
            raise ValueError("one-response checkpoint digest mismatch")

    def _payload(self, *, include_checkpoint_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "learner_input_sha256": self.learner_input_sha256,
            "passive_result_sha256": self.passive_result_sha256,
            "commitment_sha256": self.commitment_sha256,
            "first_response_sha256": self.first_response_sha256,
            "response_count": self.response_count,
            "returned_categorical_token_count": self.returned_categorical_token_count,
            "rank_certificate": self.rank_certificate._payload(include_certificate_sha=True),
            "rank_after_first_response": self.rank_after_first_response,
            "operator_maps_identified": self.operator_maps_identified,
            "unidentified_event_token_count": self.unidentified_event_token_count,
            "event_rank_deficits": [row.__dict__ for row in self.event_rank_deficits],
            "aggregate_remaining_source_rank_deficit": self.aggregate_remaining_source_rank_deficit,
            "unanswered_candidate_request_count": self.unanswered_candidate_request_count,
            "compatible_outcome_witness_request_sha256": self.compatible_outcome_witness_request_sha256,
            "compatible_outcome_witness_event_token": self.compatible_outcome_witness_event_token,
            "preexisting_witness_event_observation_count": self.preexisting_witness_event_observation_count,
            "compatible_outcome_a": self.compatible_outcome_a,
            "compatible_outcome_b": self.compatible_outcome_b,
            "branch_source_coordinates": self.branch_source_coordinates,
            "branch_a_image_coordinates": self.branch_a_image_coordinates,
            "branch_b_image_coordinates": self.branch_b_image_coordinates,
            "branch_source_rank": self.branch_source_rank,
            "branch_a_constraint_system_sha256": self.branch_a_constraint_system_sha256,
            "branch_b_constraint_system_sha256": self.branch_b_constraint_system_sha256,
            "branch_a_linear_system_consistent": self.branch_a_linear_system_consistent,
            "branch_b_linear_system_consistent": self.branch_b_linear_system_consistent,
            "differing_image_row_index": self.differing_image_row_index,
            "differing_image_delta": self.differing_image_delta,
            "compatible_outcome_analysis_used_first_response_labels": self.compatible_outcome_analysis_used_first_response_labels,
            "compatible_outcome_analysis_used_responses_2_through_15": self.compatible_outcome_analysis_used_responses_2_through_15,
            "compatible_outcomes_are_exact_unobserved_full_product_rows": self.compatible_outcomes_are_exact_unobserved_full_product_rows,
            "actual_next_response_read": self.actual_next_response_read,
            "sealed_answer_read": self.sealed_answer_read,
            "differing_restricted_map_witness": self.differing_restricted_map_witness,
        }
        if include_checkpoint_sha:
            payload["checkpoint_sha256"] = self.checkpoint_sha256
        return payload


@dataclass(frozen=True)
class ActivePartialDiscovery:
    scope: PartialOperatorScope
    status: PartialOperatorStatus
    learner_input_sha256: str
    passive_result_sha256: str
    commitment_sha256: str
    response_sha256s: tuple[str, ...]
    active_batch_call_count: int
    active_response_count: int
    returned_categorical_token_count: int
    returned_target_label_fields_per_response: int
    one_response_checkpoint: OneResponseCheckpoint
    active_rank_certificate: ExactRankCertificate
    realization: ExactPartialRealization
    operator_certificates: tuple[ExactPartialOperatorCertificate, ...]
    restricted_legal_domain_maps_identified: bool
    aggregate_total_extension_nullity: int
    legal_domain_rank_multiset: tuple[int, ...]
    total_extension_nullity_multiset: tuple[int, ...]
    total_operator: None
    sealed_answers_received_during_fit: bool
    semantic_roles_received: bool
    controller_nonce_received: bool
    executor_received: bool
    result_sha256: str
    schema: str = _ACTIVE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _ACTIVE_SCHEMA:
            raise ValueError("unknown active partial-discovery schema")
        if self.scope is not PartialOperatorScope.GUARDED_ABSENCE_AWARE_PARTIAL_OPERATORS:
            raise ValueError("active result scope mismatch")
        if self.status is not PartialOperatorStatus.SYNTHETIC_PROTOCOL_IMPLEMENTATION_REHEARSAL:
            raise ValueError("active result status mismatch")
        for name, digest in (
            ("learner_input_sha256", self.learner_input_sha256),
            ("passive_result_sha256", self.passive_result_sha256),
            ("commitment_sha256", self.commitment_sha256),
        ):
            _require_sha256(name, digest)
        if len(self.response_sha256s) != _ACTIVE_RESPONSE_COUNT:
            raise ValueError("active result must bind exactly 15 response rows")
        for digest in self.response_sha256s:
            _require_sha256("response_sha256", digest)
        if len(set(self.response_sha256s)) != _ACTIVE_RESPONSE_COUNT:
            raise ValueError("active response digests must be unique")
        if self.active_batch_call_count != 2:
            raise ValueError("the frozen staged acquisition uses one 1-row and one 14-row call")
        if self.active_response_count != _ACTIVE_RESPONSE_COUNT:
            raise ValueError("active response count must be 15")
        if self.returned_categorical_token_count != 2 * _ACTIVE_RESPONSE_COUNT:
            raise ValueError("15 responses must return exactly 30 categorical tokens")
        if self.returned_target_label_fields_per_response != 2:
            raise ValueError("each active response contains only two target-label fields")
        if type(self.one_response_checkpoint) is not OneResponseCheckpoint:
            raise TypeError("one_response_checkpoint must be exact")
        if self.one_response_checkpoint.learner_input_sha256 != self.learner_input_sha256:
            raise ValueError("one-response checkpoint refers to another learner input")
        if self.one_response_checkpoint.passive_result_sha256 != self.passive_result_sha256:
            raise ValueError("one-response checkpoint refers to another passive fit")
        if self.one_response_checkpoint.commitment_sha256 != self.commitment_sha256:
            raise ValueError("one-response checkpoint refers to another commitment")
        if self.one_response_checkpoint.first_response_sha256 != self.response_sha256s[0]:
            raise ValueError("one-response checkpoint is not bound to response one")
        if self.active_rank_certificate.rank != _FULL_RANK:
            raise ValueError("active realization must have rank five")
        if self.active_rank_certificate != self.realization.rank_certificate:
            raise ValueError("active rank certificate must equal the realization certificate")
        if type(self.realization) is not ExactPartialRealization or self.realization.ambient_rank != _FULL_RANK:
            raise ValueError("active realization must be exact rank five")
        if self.operator_certificates != self.realization.operator_certificates:
            raise ValueError("operator certificates must equal the realization maps")
        if len(self.operator_certificates) != _EVENT_COUNT:
            raise ValueError("active result must contain ten partial maps")
        if not all(row.restricted_map_identified for row in self.operator_certificates):
            raise ValueError("every restricted legal-domain map must be identified")
        domain_multiset = tuple(sorted(row.legal_domain_rank for row in self.operator_certificates))
        extension_multiset = tuple(sorted(row.total_extension_nullity for row in self.operator_certificates))
        if domain_multiset != (3, 3, 3, 3, 3, 3, 4, 4, 4, 4):
            raise ValueError("legal-domain ranks must be six rank-3 and four rank-4 maps")
        if extension_multiset != (5, 5, 5, 5, 10, 10, 10, 10, 10, 10):
            raise ValueError("total-extension nullities must be four 5s and six 10s")
        if self.legal_domain_rank_multiset != domain_multiset:
            raise ValueError("reported legal-domain rank multiset mismatch")
        if self.total_extension_nullity_multiset != extension_multiset:
            raise ValueError("reported extension-nullity multiset mismatch")
        if self.aggregate_total_extension_nullity != 80:
            raise ValueError("aggregate total-extension nullity must be exactly 80")
        if sum(extension_multiset) != self.aggregate_total_extension_nullity:
            raise ValueError("aggregate total-extension nullity arithmetic mismatch")
        if not _require_bool(
            "restricted_legal_domain_maps_identified",
            self.restricted_legal_domain_maps_identified,
        ):
            raise ValueError("restricted maps must be identified after the frozen batch")
        if self.total_operator is not None:
            raise ValueError("active result must not serialize a total operator")
        for name, value in (
            ("sealed_answers_received_during_fit", self.sealed_answers_received_during_fit),
            ("semantic_roles_received", self.semantic_roles_received),
            ("controller_nonce_received", self.controller_nonce_received),
            ("executor_received", self.executor_received),
        ):
            if _require_bool(name, value):
                raise ValueError(f"{name} must be False")
        expected = _sha256(self._payload(include_result_sha=False))
        if _require_sha256("result_sha256", self.result_sha256) != expected:
            raise ValueError("active discovery digest mismatch")

    def _payload(self, *, include_result_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "scope": self.scope,
            "status": self.status,
            "learner_input_sha256": self.learner_input_sha256,
            "passive_result_sha256": self.passive_result_sha256,
            "commitment_sha256": self.commitment_sha256,
            "response_sha256s": self.response_sha256s,
            "active_batch_call_count": self.active_batch_call_count,
            "active_response_count": self.active_response_count,
            "returned_categorical_token_count": self.returned_categorical_token_count,
            "returned_target_label_fields_per_response": self.returned_target_label_fields_per_response,
            "one_response_checkpoint": self.one_response_checkpoint._payload(include_checkpoint_sha=True),
            "active_rank_certificate": self.active_rank_certificate._payload(include_certificate_sha=True),
            "realization": self.realization._payload(include_model_sha=True),
            "operator_certificates": [row._payload(include_certificate_sha=True) for row in self.operator_certificates],
            "restricted_legal_domain_maps_identified": self.restricted_legal_domain_maps_identified,
            "aggregate_total_extension_nullity": self.aggregate_total_extension_nullity,
            "legal_domain_rank_multiset": self.legal_domain_rank_multiset,
            "total_extension_nullity_multiset": self.total_extension_nullity_multiset,
            "total_operator": self.total_operator,
            "sealed_answers_received_during_fit": self.sealed_answers_received_during_fit,
            "semantic_roles_received": self.semantic_roles_received,
            "controller_nonce_received": self.controller_nonce_received,
            "executor_received": self.executor_received,
        }
        if include_result_sha:
            payload["result_sha256"] = self.result_sha256
        return payload


# The semantic names below exist only in the trusted synthetic controller.
# None is accepted by any fit/analyse function.
_A = -1
_STATES: tuple[SemanticState, ...] = tuple(product((_A, 0, 1), repeat=2))
_B00: SemanticAction = ("bind", 0, 0)
_B01: SemanticAction = ("bind", 0, 1)
_B10: SemanticAction = ("bind", 1, 0)
_B11: SemanticAction = ("bind", 1, 1)
_U0: SemanticAction = ("update", 0, -1)
_U1: SemanticAction = ("update", 1, -1)
_C01: SemanticAction = ("copy", 0, 1)
_C10: SemanticAction = ("copy", 1, 0)
_I0: SemanticAction = ("invalidate", 0, -1)
_I1: SemanticAction = ("invalidate", 1, -1)
_ACTIONS: tuple[SemanticAction, ...] = (
    _B00,
    _B01,
    _B10,
    _B11,
    _U0,
    _U1,
    _C01,
    _C10,
    _I0,
    _I1,
)


def _semantic_step(state: SemanticState, action: SemanticAction) -> SemanticState | None:
    family, first, second = action
    result = list(state)
    if family == "bind":
        if state[first] != _A:
            return None
        result[first] = second
    elif family == "update":
        if state[first] == _A:
            return None
        result[first] = 1 - state[first]
    elif family == "copy":
        destination, source = first, second
        if state[destination] == _A or state[source] == _A:
            return None
        result[destination] = state[source]
    elif family == "invalidate":
        if state[first] == _A:
            return None
        result[first] = _A
    else:
        raise ValueError("unknown trusted-controller action")
    return result[0], result[1]


def _execute_semantic(program: Sequence[SemanticAction]) -> SemanticState | None:
    state: SemanticState = (_A, _A)
    for action in program:
        state = _semantic_step(state, action)  # type: ignore[assignment]
        if state is None:
            return None
    return state


def _state_contains_cell(state: SemanticState, cell: SemanticCell) -> bool:
    key, value = cell
    return state[key] == value


def _canonical_representative(state: SemanticState) -> tuple[SemanticAction, ...]:
    return tuple(
        ("bind", key, value)
        for key, value in enumerate(state)
        if value != _A
    )


def _semantic_legal_edges() -> tuple[tuple[SemanticState, SemanticAction, SemanticState], ...]:
    rows: list[tuple[SemanticState, SemanticAction, SemanticState]] = []
    for state in _STATES:
        for action in _ACTIONS:
            target = _semantic_step(state, action)
            if target is not None:
                rows.append((state, action, target))
    if len(rows) != _LEGAL_EDGE_COUNT:
        raise AssertionError("trusted toy legal-edge census drifted")
    return tuple(rows)


_LEGAL_EDGES = _semantic_legal_edges()


_ACTIVE_CANONICAL_PROGRAMS: tuple[tuple[SemanticAction, ...], ...] = (
    (_B00,),
    (_B10, _B00),
    (_B11, _B00),
    (_B00, _B10),
    (_B00, _B11),
    (_B00, _U0),
    (_B01, _U0),
    (_B00, _B10, _U0),
    (_B00, _B11, _U0),
    (_B00, _B10, _U1),
    (_B00, _B10, _C01),
    (_B00, _B11, _C01),
    (_B00, _B10, _C10),
    (_B00, _I0),
    (_B00, _B10, _I1),
)

_SEALED_EDGE_CANONICAL_PROGRAMS: tuple[tuple[SemanticAction, ...], ...] = (
    (_B01, _B10, _U0),
    (_B01, _B11, _U0),
    (_B00, _B11, _U1),
    (_B01, _B10, _C01),
    (_B00, _B11, _C10),
    (_B00, _B10, _I0),
    (_B00, _B11, _I0),
    (_B00, _B11, _I1),
)

_LONG_CANONICAL_PROGRAMS: tuple[tuple[SemanticAction, ...], ...] = (
    (_B00, _U0, _U0),
    (_B00, _B10, _U0, _U0),
    (_B00, _B10, _U0, _U1),
    (_B00, _B10, _U1, _U0),
    (_B00, _B11, _C01, _C10),
    (_B00, _B11, _C10, _C01),
    (_B00, _B10, _I0, _B01),
    (_B00, _B11, _I1, _B10),
    (_B00, _B10, _C01, _U1),
    (_B00, _B11, _C10, _U0),
    (_B00, _U0, _B10),
    (_B00, _B11, _C01, _I1, _B10),
)


def _assert_frozen_semantic_partition() -> None:
    omitted = (0, 0)
    passive = {
        (source, action)
        for source, action, target in _LEGAL_EDGES
        if not _state_contains_cell(source, omitted)
        and not _state_contains_cell(target, omitted)
    }
    active: set[tuple[SemanticState, SemanticAction]] = set()
    for program in _ACTIVE_CANONICAL_PROGRAMS:
        source = _execute_semantic(program[:-1])
        target = _execute_semantic(program)
        if source is None or target is None:
            raise AssertionError("frozen active program is illegal")
        active.add((source, program[-1]))
    sealed: set[tuple[SemanticState, SemanticAction]] = set()
    for program in _SEALED_EDGE_CANONICAL_PROGRAMS:
        source = _execute_semantic(program[:-1])
        target = _execute_semantic(program)
        if source is None or target is None:
            raise AssertionError("frozen sealed program is illegal")
        sealed.add((source, program[-1]))
    legal = {(source, action) for source, action, _ in _LEGAL_EDGES}
    if len(passive) != 21 or len(active) != 15 or len(sealed) != 8:
        raise AssertionError("frozen 21/15/8 partition count drifted")
    if passive | active | sealed != legal or passive & active or passive & sealed or active & sealed:
        raise AssertionError("frozen 21/15/8 partition no longer covers the legal graph")
    for program in _LONG_CANONICAL_PROGRAMS:
        if _execute_semantic(program) is None:
            raise AssertionError("frozen long/path probe is illegal")


_assert_frozen_semantic_partition()


def _opaque_token(nonce: str, category: str, role: object) -> str:
    return hashlib.sha256(
        _canonical_bytes({"nonce": nonce, "category": category, "role": role})
    ).hexdigest()[:_TOKEN_LENGTH]


def _nonce_order(nonce: str, category: str, values: Sequence[object]) -> tuple[object, ...]:
    def encodable(value: object) -> object:
        if isinstance(value, OpaqueEdgeObservation):
            return _observation_payload(value)
        if isinstance(value, OpaqueEdgeRequest):
            return _request_payload(value)
        if isinstance(value, OpaqueStateObservation):
            return {"word": value.word, "answers": value.answers}
        return value

    return tuple(
        sorted(
            values,
            key=lambda value: hashlib.sha256(
                _canonical_bytes(
                    {
                        "nonce": nonce,
                        "category": category,
                        "value": encodable(value),
                    }
                )
            ).digest(),
        )
    )


def _canonical_to_raw_maps(
    *,
    omitted_cell: SemanticCell | None,
    relabel_block: int,
) -> tuple[dict[int, int], dict[tuple[int, int], int]]:
    if relabel_block not in (0, 1):
        raise ValueError("relabel_block must be zero or one")
    if omitted_cell is None:
        key_map = {0: relabel_block, 1: 1 - relabel_block}
        value_map = {
            (canonical_key, value): value ^ relabel_block
            for canonical_key in (0, 1)
            for value in (0, 1)
        }
        return key_map, value_map
    key, value = omitted_cell
    if key not in (0, 1) or value not in (0, 1):
        raise ValueError("omitted cell must lie in K=2,V=2")
    key_map = {0: key, 1: 1 - key}
    value_map = {
        (0, canonical_value): canonical_value ^ value
        for canonical_value in (0, 1)
    }
    value_map.update(
        {
            (1, canonical_value): canonical_value ^ relabel_block
            for canonical_value in (0, 1)
        }
    )
    return key_map, value_map


def _map_state_to_raw(
    state: SemanticState,
    key_map: Mapping[int, int],
    value_map: Mapping[tuple[int, int], int],
) -> SemanticState:
    result = [_A, _A]
    for canonical_key, canonical_value in enumerate(state):
        if canonical_value != _A:
            result[key_map[canonical_key]] = value_map[(canonical_key, canonical_value)]
    return result[0], result[1]


def _map_action_to_raw(
    action: SemanticAction,
    key_map: Mapping[int, int],
    value_map: Mapping[tuple[int, int], int],
) -> SemanticAction:
    family, first, second = action
    if family == "bind":
        return family, key_map[first], value_map[(first, second)]
    if family in ("update", "invalidate"):
        return family, key_map[first], -1
    if family == "copy":
        return family, key_map[first], key_map[second]
    raise ValueError("unknown canonical action")


def _known_answer_table(
    learner_input: PartialOperatorLearnerInput,
    responses: Sequence[OpaqueMembershipResponse] = (),
) -> dict[OpaqueWord, tuple[str, ...]]:
    if type(learner_input) is not PartialOperatorLearnerInput:
        raise TypeError("learner_input must be exact PartialOperatorLearnerInput")
    table: dict[OpaqueWord, tuple[str, ...]] = {}

    def add(word: OpaqueWord, answers: tuple[str, ...]) -> None:
        _validate_answers("known answers", answers)
        if any(token not in learner_input.answer_tokens for token in answers):
            raise ValueError("answer outside learner vocabulary")
        previous = table.setdefault(word, answers)
        if previous != answers:
            raise ValueError("contradictory answers for the same opaque word")

    for row in learner_input.passive_state_observations:
        add(row.word, row.answers)
    for row in learner_input.passive_edge_observations:
        add(row.request.source_word, row.source_answers)
        add(row.request.program, row.target_answers)
    expected_selected = learner_input.candidate_edge_requests[:_ACTIVE_RESPONSE_COUNT]
    if responses and learner_input.is_full_support_control:
        raise ValueError("full-support controls accept no membership responses")
    if len(responses) > len(expected_selected):
        raise ValueError("too many active responses")
    for index, response in enumerate(responses, 1):
        if type(response) is not OpaqueMembershipResponse:
            raise TypeError("responses must be exact OpaqueMembershipResponse rows")
        expected_request = expected_selected[index - 1]
        if response.request != expected_request or response.response_ordinal != index:
            raise ValueError("active responses must follow the committed request order")
        if response.request.source_word not in table:
            raise ValueError("active response source diagnostic was not previously known")
        add(response.request.program, response.target_answers)
    return table


def _known_rank_certificate(
    learner_input: PartialOperatorLearnerInput,
    responses: Sequence[OpaqueMembershipResponse] = (),
) -> ExactRankCertificate:
    table = _known_answer_table(learner_input, responses)
    rows = _matrix(
        _diagnostic_row(table[word], learner_input.answer_tokens)
        for word in sorted(table)
    )
    return _rank_certificate(rows)


def _make_active_commitment(
    learner_input: PartialOperatorLearnerInput,
) -> ActiveAcquisitionCommitment:
    selected = learner_input.candidate_edge_requests[:_ACTIVE_RESPONSE_COUNT]
    sealed = learner_input.candidate_edge_requests[_ACTIVE_RESPONSE_COUNT:]
    kwargs = {
        "learner_input_sha256": learner_input.input_sha256,
        "candidate_pool_sha256": learner_input.candidate_pool_sha256,
        "selected_request_sha256s": tuple(row.request_sha256 for row in selected),
        "sealed_request_sha256s": tuple(row.request_sha256 for row in sealed),
        "selection_rule": "trusted_controller_supplied_fixed_15_edge_legal_domain_excitation_basis",
        "selected_before_any_active_answer": True,
        "active_basis_supplied_by_trusted_controller": bool(selected),
        "controller_order_encodes_semantically_designed_basis": bool(selected),
        "learner_selected_acquisition_basis": False,
        "learner_selection_used_semantic_roles": False,
        "learner_selection_used_controller_nonce": False,
        "selection_used_sealed_answers": False,
        "active_batch_call_count": 2 * int(bool(selected)),
        "expected_response_count": len(selected),
        "expected_returned_categorical_token_count": 2 * len(selected),
        "schema": _COMMITMENT_SCHEMA,
    }
    payload = {"schema": _COMMITMENT_SCHEMA, **{key: value for key, value in kwargs.items() if key != "schema"}}
    return ActiveAcquisitionCommitment(
        **kwargs,
        commitment_sha256=_sha256(payload),
    )


@dataclass(frozen=True)
class OpaqueOmissionHypothesisWitness:
    """One semantic decoding compatible with only the public passive input."""

    query_position_to_key: tuple[int, int]
    answer_token_to_value_by_key: tuple[
        tuple[int, tuple[tuple[str, int], ...]], ...
    ]
    event_token_to_action: tuple[tuple[str, SemanticAction], ...]
    source_word_to_state: tuple[tuple[OpaqueWord, SemanticState], ...]
    inferred_omitted_cell: SemanticCell
    compatible_relabel_block: int
    query_position_to_hypothesized_raw_key: tuple[int, int]
    opaque_event_token_to_hypothesized_raw_action: tuple[
        tuple[str, SemanticAction], ...
    ]
    source_word_to_hypothesized_raw_state: tuple[
        tuple[OpaqueWord, SemanticState], ...
    ]
    witness_sha256: str
    schema: str = _HYPOTHESIS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _HYPOTHESIS_SCHEMA:
            raise ValueError("unknown omission-hypothesis schema")
        if tuple(sorted(self.query_position_to_key)) != (0, 1):
            raise ValueError("query-position assignment must be a key permutation")
        if tuple(key for key, _ in self.answer_token_to_value_by_key) != (0, 1):
            raise ValueError("hypothesis must contain one answer decoding per key")
        for _, rows in self.answer_token_to_value_by_key:
            if len(rows) != _ANSWER_COUNT or {value for _, value in rows} != {_A, 0, 1}:
                raise ValueError("each key-local answer-token decoding must be bijective")
            for token, _ in rows:
                _require_token("hypothesis answer token", token)
        if len(self.event_token_to_action) != _EVENT_COUNT:
            raise ValueError("hypothesis must decode all ten event tokens")
        if {action for _, action in self.event_token_to_action} != set(_ACTIONS):
            raise ValueError("event-token decoding must be bijective")
        for token, _ in self.event_token_to_action:
            _require_token("hypothesis event token", token)
        if len(self.source_word_to_state) != _STATE_COUNT:
            raise ValueError("hypothesis must decode all nine source representatives")
        if {state for _, state in self.source_word_to_state} != set(_STATES):
            raise ValueError("source representatives must decode bijectively")
        if self.inferred_omitted_cell not in ((0, 0), (0, 1), (1, 0), (1, 1)):
            raise ValueError("hypothesis omitted cell lies outside K=2,V=2")
        if self.compatible_relabel_block not in (0, 1):
            raise ValueError("compatible relabel block must be zero or one")
        key_map, value_map = _canonical_to_raw_maps(
            omitted_cell=self.inferred_omitted_cell,
            relabel_block=self.compatible_relabel_block,
        )
        expected_query_keys = tuple(
            key_map[key] for key in self.query_position_to_key
        )
        if self.query_position_to_hypothesized_raw_key != expected_query_keys:
            raise ValueError("hypothesized raw query-key isomorphism mismatch")
        expected_event_rows = tuple(
            sorted(
                (
                    token,
                    _map_action_to_raw(action, key_map, value_map),
                )
                for token, action in self.event_token_to_action
            )
        )
        if self.opaque_event_token_to_hypothesized_raw_action != expected_event_rows:
            raise ValueError("hypothesized raw event isomorphism mismatch")
        expected_state_rows = tuple(
            sorted(
                (
                    word,
                    _map_state_to_raw(state, key_map, value_map),
                )
                for word, state in self.source_word_to_state
            )
        )
        if self.source_word_to_hypothesized_raw_state != expected_state_rows:
            raise ValueError("hypothesized raw history/state isomorphism mismatch")
        expected = _sha256(self._payload(include_witness_sha=False))
        if _require_sha256("witness_sha256", self.witness_sha256) != expected:
            raise ValueError("omission-hypothesis digest mismatch")

    def _payload(self, *, include_witness_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "query_position_to_key": self.query_position_to_key,
            "answer_token_to_value_by_key": self.answer_token_to_value_by_key,
            "event_token_to_action": self.event_token_to_action,
            "source_word_to_state": self.source_word_to_state,
            "inferred_omitted_cell": self.inferred_omitted_cell,
            "compatible_relabel_block": self.compatible_relabel_block,
            "query_position_to_hypothesized_raw_key": self.query_position_to_hypothesized_raw_key,
            "opaque_event_token_to_hypothesized_raw_action": self.opaque_event_token_to_hypothesized_raw_action,
            "source_word_to_hypothesized_raw_state": self.source_word_to_hypothesized_raw_state,
        }
        if include_witness_sha:
            payload["witness_sha256"] = self.witness_sha256
        return payload


@dataclass(frozen=True)
class CompatibleOmissionHypothesisCertificate:
    learner_input_sha256: str
    witnesses: tuple[OpaqueOmissionHypothesisWitness, ...]
    compatible_hypothesis_count: int
    exhaustive_finite_bijection_enumeration: bool
    active_answers_used: bool
    sealed_answers_used: bool
    controller_nonce_used: bool
    actual_omitted_identifier_used: bool
    certificate_sha256: str
    schema: str = _HYPOTHESIS_CERTIFICATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _HYPOTHESIS_CERTIFICATE_SCHEMA:
            raise ValueError("unknown compatible-omission certificate schema")
        _require_sha256("learner_input_sha256", self.learner_input_sha256)
        if self.compatible_hypothesis_count != len(self.witnesses):
            raise ValueError("compatible-hypothesis count mismatch")
        if len({row.witness_sha256 for row in self.witnesses}) != len(self.witnesses):
            raise ValueError("compatible hypotheses must be distinct")
        for name, value, required in (
            ("exhaustive_finite_bijection_enumeration", self.exhaustive_finite_bijection_enumeration, True),
            ("active_answers_used", self.active_answers_used, False),
            ("sealed_answers_used", self.sealed_answers_used, False),
            ("controller_nonce_used", self.controller_nonce_used, False),
            ("actual_omitted_identifier_used", self.actual_omitted_identifier_used, False),
        ):
            if _require_bool(name, value) is not required:
                raise ValueError(f"{name} must be {required}")
        expected = _sha256(self._payload(include_certificate_sha=False))
        if _require_sha256("certificate_sha256", self.certificate_sha256) != expected:
            raise ValueError("compatible-omission certificate digest mismatch")

    def _payload(self, *, include_certificate_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "learner_input_sha256": self.learner_input_sha256,
            "witnesses": [row._payload(include_witness_sha=True) for row in self.witnesses],
            "compatible_hypothesis_count": self.compatible_hypothesis_count,
            "exhaustive_finite_bijection_enumeration": self.exhaustive_finite_bijection_enumeration,
            "active_answers_used": self.active_answers_used,
            "sealed_answers_used": self.sealed_answers_used,
            "controller_nonce_used": self.controller_nonce_used,
            "actual_omitted_identifier_used": self.actual_omitted_identifier_used,
        }
        if include_certificate_sha:
            payload["certificate_sha256"] = self.certificate_sha256
        return payload


def enumerate_publicly_compatible_omission_hypotheses(
    learner_input: PartialOperatorLearnerInput,
) -> CompatibleOmissionHypothesisCertificate:
    """Exhaustively enumerate semantic bijections compatible with passive data.

    The enumeration consumes only ``PartialOperatorLearnerInput``.  It tries
    both query-to-key bijections, all six answer-label bijections, every
    passive-consistent event/action matching, and verifies the exact 9x10
    definedness grid plus every visible transition.  It never consumes an
    active/sealed answer, controller nonce, or actual omission identifier.
    """

    if type(learner_input) is not PartialOperatorLearnerInput:
        raise TypeError("learner_input must be exact PartialOperatorLearnerInput")
    if learner_input.is_full_support_control:
        kwargs = {
            "learner_input_sha256": learner_input.input_sha256,
            "witnesses": (),
            "compatible_hypothesis_count": 0,
            "exhaustive_finite_bijection_enumeration": True,
            "active_answers_used": False,
            "sealed_answers_used": False,
            "controller_nonce_used": False,
            "actual_omitted_identifier_used": False,
            "schema": _HYPOTHESIS_CERTIFICATE_SCHEMA,
        }
        payload = {"schema": _HYPOTHESIS_CERTIFICATE_SCHEMA, **{key: value for key, value in kwargs.items() if key != "schema"}}
        return CompatibleOmissionHypothesisCertificate(
            **kwargs, certificate_sha256=_sha256(payload)
        )
    known = _known_answer_table(learner_input)
    mask_status = {
        (row.source_word, row.event_token): True
        for row in learner_input.defined_edge_requests
    }
    mask_status.update(
        {
            (row.source_word, row.event_token): False
            for row in learner_input.undefined_edge_requests
        }
    )
    source_words = tuple(sorted({word for word, _ in mask_status}))
    if () not in source_words:
        raise ValueError("opaque protocol must include the empty source representative")
    observations = learner_input.passive_edge_observations
    witnesses: list[OpaqueOmissionHypothesisWitness] = []
    answer_permutations = tuple(
        tuple(zip(learner_input.answer_tokens, values, strict=True))
        for values in __import__("itertools").permutations((_A, 0, 1))
    )
    query_permutations = ((0, 1), (1, 0))
    for query_assignment in query_permutations:
        for key0_rows in answer_permutations:
            for key1_rows in answer_permutations:
                answer_maps = {0: dict(key0_rows), 1: dict(key1_rows)}

                def decode(answers: tuple[str, ...]) -> SemanticState:
                    state = [_A, _A]
                    for position, token in enumerate(answers):
                        key = query_assignment[position]
                        state[key] = answer_maps[key][token]
                    return state[0], state[1]

                known_states = {word: decode(answers) for word, answers in known.items()}
                if len(set(known_states.values())) != len(set(known.values())):
                    continue
                if known_states.get(()) != (_A, _A):
                    continue
                candidates: dict[str, tuple[SemanticAction, ...]] = {}
                for event_token in learner_input.event_tokens:
                    compatible: list[SemanticAction] = []
                    for action in _ACTIONS:
                        good = True
                        for word, source_state in known_states.items():
                            status = mask_status.get((word, event_token))
                            if status is not None and status != (
                                _semantic_step(source_state, action) is not None
                            ):
                                good = False
                                break
                        if not good:
                            continue
                        for observation in observations:
                            if observation.request.event_token != event_token:
                                continue
                            source = decode(observation.source_answers)
                            target = decode(observation.target_answers)
                            if _semantic_step(source, action) != target:
                                good = False
                                break
                        if good:
                            compatible.append(action)
                    if not compatible:
                        break
                    candidates[event_token] = tuple(compatible)
                if len(candidates) != _EVENT_COUNT:
                    continue
                ordered_tokens = tuple(
                    sorted(learner_input.event_tokens, key=lambda token: (len(candidates[token]), token))
                )

                def search(
                    index: int,
                    mapping: dict[str, SemanticAction],
                    used: set[SemanticAction],
                ) -> None:
                    if index < len(ordered_tokens):
                        token = ordered_tokens[index]
                        for action in candidates[token]:
                            if action in used:
                                continue
                            mapping[token] = action
                            used.add(action)
                            search(index + 1, mapping, used)
                            used.remove(action)
                            del mapping[token]
                        return
                    decoded_sources: dict[OpaqueWord, SemanticState] = {}
                    for word in source_words:
                        state: SemanticState | None = (_A, _A)
                        for token in word:
                            state = _semantic_step(state, mapping[token]) if state is not None else None
                            if state is None:
                                return
                        assert state is not None
                        decoded_sources[word] = state
                    if set(decoded_sources.values()) != set(_STATES):
                        return
                    if any(
                        decoded_sources[word] != state
                        for word, state in known_states.items()
                        if word in decoded_sources
                    ):
                        return
                    for (word, token), status in mask_status.items():
                        if status != (
                            _semantic_step(decoded_sources[word], mapping[token]) is not None
                        ):
                            return
                    passive_state_set = set(known_states.values())
                    compatible_cells = tuple(
                        cell
                        for cell in ((0, 0), (0, 1), (1, 0), (1, 1))
                        if passive_state_set
                        == {
                            state
                            for state in _STATES
                            if not _state_contains_cell(state, cell)
                        }
                    )
                    if len(compatible_cells) != 1:
                        return
                    common_rows = {
                        "query_position_to_key": query_assignment,
                        "answer_token_to_value_by_key": (
                            (0, tuple(sorted(answer_maps[0].items()))),
                            (1, tuple(sorted(answer_maps[1].items()))),
                        ),
                        "event_token_to_action": tuple(sorted(mapping.items())),
                        "source_word_to_state": tuple(sorted(decoded_sources.items())),
                        "inferred_omitted_cell": compatible_cells[0],
                    }
                    # The two frozen controller relabel blocks are latent
                    # coordinate conventions after opaque retagging.  They
                    # induce the same public relation and therefore remain two
                    # distinct compatible controller hypotheses per cell.
                    for compatible_relabel_block in (0, 1):
                        hypothesis_key_map, hypothesis_value_map = _canonical_to_raw_maps(
                            omitted_cell=compatible_cells[0],
                            relabel_block=compatible_relabel_block,
                        )
                        rows = {
                            **common_rows,
                            "compatible_relabel_block": compatible_relabel_block,
                            "query_position_to_hypothesized_raw_key": tuple(
                                hypothesis_key_map[key]
                                for key in query_assignment
                            ),
                            "opaque_event_token_to_hypothesized_raw_action": tuple(
                                sorted(
                                    (
                                        token,
                                        _map_action_to_raw(
                                            action,
                                            hypothesis_key_map,
                                            hypothesis_value_map,
                                        ),
                                    )
                                    for token, action in mapping.items()
                                )
                            ),
                            "source_word_to_hypothesized_raw_state": tuple(
                                sorted(
                                    (
                                        word,
                                        _map_state_to_raw(
                                            state,
                                            hypothesis_key_map,
                                            hypothesis_value_map,
                                        ),
                                    )
                                    for word, state in decoded_sources.items()
                                )
                            ),
                            "schema": _HYPOTHESIS_SCHEMA,
                        }
                        payload = {"schema": _HYPOTHESIS_SCHEMA, **{key: value for key, value in rows.items() if key != "schema"}}
                        witnesses.append(
                            OpaqueOmissionHypothesisWitness(
                                **rows,
                                witness_sha256=_sha256(payload),
                            )
                        )

                search(0, {}, set())
    unique = {row.witness_sha256: row for row in witnesses}
    ordered_witnesses = tuple(unique[digest] for digest in sorted(unique))
    kwargs = {
        "learner_input_sha256": learner_input.input_sha256,
        "witnesses": ordered_witnesses,
        "compatible_hypothesis_count": len(ordered_witnesses),
        "exhaustive_finite_bijection_enumeration": True,
        "active_answers_used": False,
        "sealed_answers_used": False,
        "controller_nonce_used": False,
        "actual_omitted_identifier_used": False,
        "schema": _HYPOTHESIS_CERTIFICATE_SCHEMA,
    }
    payload = {
        "schema": _HYPOTHESIS_CERTIFICATE_SCHEMA,
        "learner_input_sha256": learner_input.input_sha256,
        "witnesses": [row._payload(include_witness_sha=True) for row in ordered_witnesses],
        "compatible_hypothesis_count": len(ordered_witnesses),
        "exhaustive_finite_bijection_enumeration": True,
        "active_answers_used": False,
        "sealed_answers_used": False,
        "controller_nonce_used": False,
        "actual_omitted_identifier_used": False,
    }
    return CompatibleOmissionHypothesisCertificate(
        **kwargs,
        certificate_sha256=_sha256(payload),
    )


def publicly_compatible_omission_hypothesis_count(
    learner_input: PartialOperatorLearnerInput,
) -> int:
    if learner_input.is_full_support_control:
        return 1
    certificate = enumerate_publicly_compatible_omission_hypotheses(learner_input)
    if certificate.compatible_hypothesis_count != 8:
        raise ValueError(
            "passive opaque input does not admit exactly the frozen eight hypotheses"
        )
    return certificate.compatible_hypothesis_count


def _right_nullspace_rows(matrix: RationalMatrix) -> RationalMatrix:
    rows, columns = _matrix_shape(matrix)
    work = _fractions(matrix)
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(columns):
        pivot = next(
            (row for row in range(pivot_row, rows) if work[row][column]),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][column]
        work[pivot_row] = [value / scale for value in work[pivot_row]]
        for row in range(rows):
            if row == pivot_row or not work[row][column]:
                continue
            factor = work[row][column]
            work[row] = [
                value - factor * entry
                for value, entry in zip(work[row], work[pivot_row], strict=True)
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    free_columns = [column for column in range(columns) if column not in pivot_columns]
    basis: list[tuple[Fraction, ...]] = []
    for free in free_columns:
        vector = [Fraction(0) for _ in range(columns)]
        vector[free] = Fraction(1)
        for row, pivot in reversed(tuple(enumerate(pivot_columns))):
            vector[pivot] = -sum(
                work[row][column] * vector[column]
                for column in free_columns
            )
        basis.append(tuple(vector))
    if not basis:
        raise ValueError("frozen partial domains must have a nontrivial off-domain nullspace")
    return _matrix(basis)


def _build_realization(
    learner_input: PartialOperatorLearnerInput,
    responses: Sequence[OpaqueMembershipResponse],
    *,
    require_complete_maps: bool,
) -> tuple[ExactPartialRealization, tuple[ObservedEventRank, ...]]:
    """Pure exact fitting from learner-visible diagnostics and edge rows."""

    known = _known_answer_table(learner_input, responses)
    ordered_words = tuple(sorted(known))
    answer_matrix = _matrix(
        _diagnostic_row(known[word], learner_input.answer_tokens)
        for word in ordered_words
    )
    rank_certificate = _rank_certificate(answer_matrix)
    rank = rank_certificate.rank
    basis_words = tuple(ordered_words[index] for index in rank_certificate.pivot_row_indices)
    basis_rows = _matrix(answer_matrix[index] for index in rank_certificate.pivot_row_indices)
    coordinate_by_word = {
        word: _row_coordinates(
            _diagnostic_row(answers, learner_input.answer_tokens), basis_rows
        )
        for word, answers in known.items()
    }
    coordinate_codebook_rows = {
        coordinates: answers
        for word, coordinates in coordinate_by_word.items()
        for answers in (known[word],)
    }
    if len(coordinate_codebook_rows) != len(set(known.values())):
        raise ValueError("opaque diagnostic feature gauge collapsed distinct answer pairs")
    codebook = tuple(
        sorted(
            coordinate_codebook_rows.items(),
            key=lambda row: _canonical_bytes(row),
        )
    )

    observations: tuple[OpaqueEdgeObservation, ...] = (
        learner_input.passive_edge_observations
        + tuple(
            _make_observation(
                response.request,
                known[response.request.source_word],
                response.target_answers,
            )
            for response in responses
        )
    )
    observed_event_ranks: list[ObservedEventRank] = []
    operator_certificates: list[ExactPartialOperatorCertificate] = []
    for event_token in learner_input.event_tokens:
        event_observations = tuple(
            row for row in observations if row.request.event_token == event_token
        )
        if event_observations:
            observed_sources = _matrix(
                coordinate_by_word[row.request.source_word]
                for row in event_observations
            )
            observed_rank, observed_pivot_rows, _ = _rank_profile(observed_sources)
        else:
            observed_rank, observed_pivot_rows = 0, ()
        observed_event_ranks.append(
            ObservedEventRank(
                event_token=event_token,
                observation_count=len(event_observations),
                observed_source_rank=observed_rank,
            )
        )
        if not require_complete_maps:
            continue
        domain_requests = tuple(
            row
            for row in learner_input.defined_edge_requests
            if row.event_token == event_token
        )
        if any(row.source_word not in coordinate_by_word for row in domain_requests):
            raise ValueError("active answers do not label every legal-domain source word")
        domain_rows = _matrix(
            coordinate_by_word[row.source_word] for row in domain_requests
        )
        domain_rank = _rank_profile(domain_rows)[0]
        if observed_rank != domain_rank:
            raise ValueError("observed edges do not close the event's legal-domain map")
        chosen_observations = tuple(
            event_observations[index] for index in observed_pivot_rows
        )
        domain_basis = _matrix(
            coordinate_by_word[row.request.source_word]
            for row in chosen_observations
        )
        image_basis = _matrix(
            coordinate_by_word[row.request.program]
            for row in chosen_observations
        )
        # Recompute exact agreement on every observed edge, including dependent rows.
        for observation in event_observations:
            source = coordinate_by_word[observation.request.source_word]
            coefficients = _row_coordinates(source, domain_basis)
            predicted = tuple(
                Rational.from_fraction(
                    sum(
                        coefficients[i].as_fraction()
                        * image_basis[i][j].as_fraction()
                        for i in range(domain_rank)
                    )
                )
                for j in range(rank)
            )
            if predicted != coordinate_by_word[observation.request.program]:
                raise ValueError("observed edges are inconsistent with a linear partial map")
        annihilator = _right_nullspace_rows(domain_basis)
        kwargs = {
            "event_token": event_token,
            "ambient_rank": rank,
            "legal_domain_rank": domain_rank,
            "observed_source_rank": observed_rank,
            "restricted_nullity": rank * (domain_rank - observed_rank),
            "total_extension_nullity": rank * (rank - domain_rank),
            "domain_basis_coordinates": domain_basis,
            "image_basis_coordinates": image_basis,
            "off_domain_annihilator_basis": annihilator,
            "restricted_map_identified": True,
            "off_domain_extension_identified": False,
            "undefined_inputs_encoded_as_zero_or_dead": False,
            "total_operator": None,
            "schema": _OPERATOR_SCHEMA,
        }
        payload = {"schema": _OPERATOR_SCHEMA, **{key: value for key, value in kwargs.items() if key != "schema"}}
        operator_certificates.append(
            ExactPartialOperatorCertificate(
                **kwargs,
                certificate_sha256=_sha256(payload),
            )
        )

    complete = len(operator_certificates) == _EVENT_COUNT
    realization_kwargs = {
        "event_tokens": learner_input.event_tokens,
        "query_tokens": learner_input.query_tokens,
        "answer_tokens": learner_input.answer_tokens,
        "ambient_rank": rank,
        "rank_certificate": rank_certificate,
        "basis_words": basis_words,
        "basis_diagnostic_rows": basis_rows,
        "coordinate_answer_codebook": codebook,
        "operator_certificates": tuple(operator_certificates),
        "restricted_maps_complete": complete,
        "total_operator": None,
        "schema": _REALIZATION_SCHEMA,
    }
    realization_payload = {
        "schema": _REALIZATION_SCHEMA,
        "event_tokens": learner_input.event_tokens,
        "query_tokens": learner_input.query_tokens,
        "answer_tokens": learner_input.answer_tokens,
        "ambient_rank": rank,
        "rank_certificate": rank_certificate._payload(include_certificate_sha=True),
        "basis_words": basis_words,
        "basis_diagnostic_rows": basis_rows,
        "coordinate_answer_codebook": codebook,
        "operator_certificates": [
            row._payload(include_certificate_sha=True)
            for row in operator_certificates
        ],
        "restricted_maps_complete": complete,
        "total_operator": None,
    }
    realization = ExactPartialRealization(
        **realization_kwargs,
        model_sha256=_sha256(realization_payload),
    )
    rational_values: list[Rational] = []
    rational_values.extend(value for row in answer_matrix for value in row)
    rational_values.extend(value for row in basis_rows for value in row)
    for coordinates, _ in codebook:
        rational_values.extend(coordinates)
    for operator in operator_certificates:
        for matrix in (
            operator.domain_basis_coordinates,
            operator.image_basis_coordinates,
            operator.off_domain_annihilator_basis,
        ):
            rational_values.extend(value for row in matrix for value in row)
    maximum_bits = max(
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        for value in rational_values
    )
    if maximum_bits > learner_input.budgets.max_rational_bit_length:
        raise OpaquePartialOperatorLimitError("exact rational bit-length budget exceeded")
    return realization, tuple(observed_event_ranks)


def fit_passive_partial_operators(
    learner_input: PartialOperatorLearnerInput,
) -> PassivePartialDiscovery:
    """Fit the trace/diagnostic-only passive realization."""

    if type(learner_input) is not PartialOperatorLearnerInput:
        raise TypeError("learner_input must be exact PartialOperatorLearnerInput")
    complete = learner_input.is_full_support_control
    realization, observed_ranks = _build_realization(
        learner_input, (), require_complete_maps=complete
    )
    expected_rank = _FULL_RANK if complete else _PASSIVE_RANK
    if realization.ambient_rank != expected_rank:
        raise ValueError("passive answer table has the wrong exact rank")
    commitment = _make_active_commitment(learner_input)
    kwargs = {
        "scope": PartialOperatorScope.GUARDED_ABSENCE_AWARE_PARTIAL_OPERATORS,
        "status": PartialOperatorStatus.SYNTHETIC_PROTOCOL_IMPLEMENTATION_REHEARSAL,
        "learner_input_sha256": learner_input.input_sha256,
        "passive_edge_count": len(learner_input.passive_edge_observations),
        "passive_rank_certificate": realization.rank_certificate,
        "observed_event_ranks": observed_ranks,
        "realization": realization,
        "active_commitment": commitment,
        "compatible_omission_hypothesis_count": None,
        "omission_hypothesis_analysis_performed_inside_estimator": False,
        "definedness_mask_counted_as_supervision": True,
        "passive_answers_only": True,
        "semantic_roles_received": False,
        "controller_nonce_received": False,
        "executor_received": False,
        "schema": _PASSIVE_SCHEMA,
    }
    payload = {
        "schema": _PASSIVE_SCHEMA,
        "scope": kwargs["scope"],
        "status": kwargs["status"],
        "learner_input_sha256": learner_input.input_sha256,
        "passive_edge_count": kwargs["passive_edge_count"],
        "passive_rank_certificate": realization.rank_certificate._payload(include_certificate_sha=True),
        "observed_event_ranks": [row.__dict__ for row in observed_ranks],
        "realization": realization._payload(include_model_sha=True),
        "active_commitment": commitment._payload(include_commitment_sha=True),
        "compatible_omission_hypothesis_count": kwargs["compatible_omission_hypothesis_count"],
        "omission_hypothesis_analysis_performed_inside_estimator": False,
        "definedness_mask_counted_as_supervision": True,
        "passive_answers_only": True,
        "semantic_roles_received": False,
        "controller_nonce_received": False,
        "executor_received": False,
    }
    return PassivePartialDiscovery(**kwargs, result_sha256=_sha256(payload))


def analyze_one_response_checkpoint(
    learner_input: PartialOperatorLearnerInput,
    passive: PassivePartialDiscovery,
    first_response: OpaqueMembershipResponse,
) -> OneResponseCheckpoint:
    """Freeze the rank/nonidentification analysis before responses 2--15 exist.

    The API accepts exactly one response object.  Its compatible-outcome twin
    is selected only from the declared answer vocabulary and the unanswered
    opaque request pool; neither the actual second answer nor a sealed answer
    is an argument to this function.
    """

    if type(learner_input) is not PartialOperatorLearnerInput:
        raise TypeError("learner_input must be exact PartialOperatorLearnerInput")
    if type(passive) is not PassivePartialDiscovery:
        raise TypeError("passive must be exact PassivePartialDiscovery")
    if type(first_response) is not OpaqueMembershipResponse:
        raise TypeError("first_response must be exact OpaqueMembershipResponse")
    if learner_input.is_full_support_control:
        raise ValueError("full-support controls require no active checkpoint")
    reconstructed_passive = fit_passive_partial_operators(learner_input)
    if passive != reconstructed_passive:
        raise ValueError("passive result is not the pure fit of this learner input")
    selected = learner_input.candidate_edge_requests[:_ACTIVE_RESPONSE_COUNT]
    if first_response.request != selected[0] or first_response.response_ordinal != 1:
        raise ValueError("checkpoint must use the first committed response")
    prefix_realization, prefix_observed_ranks = _build_realization(
        learner_input, (first_response,), require_complete_maps=False
    )
    rank_certificate = prefix_realization.rank_certificate
    if rank_certificate.rank != _FULL_RANK:
        raise ValueError("the frozen first response does not raise rank 4 to rank 5")
    witness_request = selected[1]
    if witness_request.event_token != first_response.request.event_token:
        raise ValueError("one-response ambiguity witness must concern the same event map")
    preexisting_same_event = tuple(
        row
        for row in learner_input.passive_edge_observations
        if row.request.event_token == witness_request.event_token
    ) + (
        _make_observation(
            first_response.request,
            _known_answer_table(learner_input)[first_response.request.source_word],
            first_response.target_answers,
        ),
    )
    if len(preexisting_same_event) != 1:
        raise ValueError("branch witness requires exactly one existing same-event constraint")
    known = _known_answer_table(learner_input, (first_response,))
    if witness_request.program in known:
        raise ValueError("compatible-outcome witness target program is already known")
    full_product_rows = set(product(learner_input.answer_tokens, repeat=2))
    unobserved_rows = tuple(
        sorted(full_product_rows - set(known.values()), key=_canonical_bytes)
    )
    if len(unobserved_rows) != 2:
        raise ValueError("rank-five prefix must leave exactly two full-product rows unobserved")
    outcome_a, outcome_b = unobserved_rows
    mask_source_words = tuple(
        sorted({row.source_word for row in learner_input.defined_edge_requests})
    )
    unlabeled_sources = tuple(word for word in mask_source_words if word not in known)
    if len(unlabeled_sources) != 2:
        raise ValueError("one-response prefix must leave exactly two source representatives unlabeled")
    observed_rank_by_token = {
        row.event_token: row.observed_source_rank for row in prefix_observed_ranks
    }
    domain_rank_profiles: list[tuple[tuple[str, int], ...]] = []
    for assigned_rows in (
        (unobserved_rows[0], unobserved_rows[1]),
        (unobserved_rows[1], unobserved_rows[0]),
    ):
        assigned = dict(zip(unlabeled_sources, assigned_rows, strict=True))
        coordinate_by_source = {
            word: prefix_realization.answers_to_coordinates(
                known[word] if word in known else assigned[word]
            )
            for word in mask_source_words
        }
        profile: list[tuple[str, int]] = []
        for event_token in learner_input.event_tokens:
            domain_rows = _matrix(
                coordinate_by_source[row.source_word]
                for row in learner_input.defined_edge_requests
                if row.event_token == event_token
            )
            profile.append((event_token, _rank_profile(domain_rows)[0]))
        domain_rank_profiles.append(tuple(profile))
    if domain_rank_profiles[0] != domain_rank_profiles[1]:
        raise ValueError("legal-domain ranks depend on an unopened categorical assignment")
    event_rank_deficits = tuple(
        OneResponseEventDeficit(
            event_token=event_token,
            legal_domain_rank=domain_rank,
            observed_source_rank=observed_rank_by_token[event_token],
            remaining_source_rank_deficit=(
                domain_rank - observed_rank_by_token[event_token]
            ),
        )
        for event_token, domain_rank in domain_rank_profiles[0]
    )
    unidentified_count = sum(
        row.remaining_source_rank_deficit > 0 for row in event_rank_deficits
    )
    aggregate_deficit = sum(
        row.remaining_source_rank_deficit for row in event_rank_deficits
    )
    if unidentified_count != 9 or aggregate_deficit != 14:
        raise ValueError("one-response learner-visible domain-rank census drifted")

    first_source = prefix_realization.answers_to_coordinates(
        known[first_response.request.source_word]
    )
    second_source = prefix_realization.answers_to_coordinates(
        known[witness_request.source_word]
    )
    first_image = prefix_realization.answers_to_coordinates(
        first_response.target_answers
    )
    branch_source = _matrix((first_source, second_source))
    branch_a_images = _matrix(
        (first_image, prefix_realization.answers_to_coordinates(outcome_a))
    )
    branch_b_images = _matrix(
        (first_image, prefix_realization.answers_to_coordinates(outcome_b))
    )
    branch_source_rank = _rank_profile(branch_source)[0]
    if branch_source_rank != 2:
        raise ValueError("the second committed request does not add a source direction")
    differing_delta = tuple(
        Rational.from_fraction(
            branch_a_images[1][column].as_fraction()
            - branch_b_images[1][column].as_fraction()
        )
        for column in range(_FULL_RANK)
    )
    if not any(value.numerator for value in differing_delta):
        raise ValueError("compatible outcome branches do not differ")
    branch_a_hash = _sha256(
        {"source_coordinates": branch_source, "image_coordinates": branch_a_images}
    )
    branch_b_hash = _sha256(
        {"source_coordinates": branch_source, "image_coordinates": branch_b_images}
    )
    kwargs = {
        "learner_input_sha256": learner_input.input_sha256,
        "passive_result_sha256": passive.result_sha256,
        "commitment_sha256": passive.active_commitment.commitment_sha256,
        "first_response_sha256": first_response.response_sha256,
        "response_count": 1,
        "returned_categorical_token_count": 2,
        "rank_certificate": rank_certificate,
        "rank_after_first_response": _FULL_RANK,
        "operator_maps_identified": False,
        "unidentified_event_token_count": unidentified_count,
        "event_rank_deficits": event_rank_deficits,
        "aggregate_remaining_source_rank_deficit": aggregate_deficit,
        "unanswered_candidate_request_count": _ACTIVE_RESPONSE_COUNT + _SEALED_EDGE_COUNT - 1,
        "compatible_outcome_witness_request_sha256": witness_request.request_sha256,
        "compatible_outcome_witness_event_token": witness_request.event_token,
        "preexisting_witness_event_observation_count": len(preexisting_same_event),
        "compatible_outcome_a": outcome_a,
        "compatible_outcome_b": outcome_b,
        "branch_source_coordinates": branch_source,
        "branch_a_image_coordinates": branch_a_images,
        "branch_b_image_coordinates": branch_b_images,
        "branch_source_rank": branch_source_rank,
        "branch_a_constraint_system_sha256": branch_a_hash,
        "branch_b_constraint_system_sha256": branch_b_hash,
        "branch_a_linear_system_consistent": branch_source_rank == 2,
        "branch_b_linear_system_consistent": branch_source_rank == 2,
        "differing_image_row_index": 1,
        "differing_image_delta": differing_delta,
        "compatible_outcome_analysis_used_first_response_labels": True,
        "compatible_outcome_analysis_used_responses_2_through_15": False,
        "compatible_outcomes_are_exact_unobserved_full_product_rows": True,
        "actual_next_response_read": False,
        "sealed_answer_read": False,
        "differing_restricted_map_witness": True,
        "schema": _ONE_RESPONSE_SCHEMA,
    }
    payload = {
        "schema": _ONE_RESPONSE_SCHEMA,
        "learner_input_sha256": learner_input.input_sha256,
        "passive_result_sha256": passive.result_sha256,
        "commitment_sha256": passive.active_commitment.commitment_sha256,
        "first_response_sha256": first_response.response_sha256,
        "response_count": 1,
        "returned_categorical_token_count": 2,
        "rank_certificate": rank_certificate._payload(include_certificate_sha=True),
        "rank_after_first_response": _FULL_RANK,
        "operator_maps_identified": False,
        "unidentified_event_token_count": unidentified_count,
        "event_rank_deficits": [row.__dict__ for row in event_rank_deficits],
        "aggregate_remaining_source_rank_deficit": aggregate_deficit,
        "unanswered_candidate_request_count": _ACTIVE_RESPONSE_COUNT + _SEALED_EDGE_COUNT - 1,
        "compatible_outcome_witness_request_sha256": witness_request.request_sha256,
        "compatible_outcome_witness_event_token": witness_request.event_token,
        "preexisting_witness_event_observation_count": len(preexisting_same_event),
        "compatible_outcome_a": outcome_a,
        "compatible_outcome_b": outcome_b,
        "branch_source_coordinates": branch_source,
        "branch_a_image_coordinates": branch_a_images,
        "branch_b_image_coordinates": branch_b_images,
        "branch_source_rank": branch_source_rank,
        "branch_a_constraint_system_sha256": branch_a_hash,
        "branch_b_constraint_system_sha256": branch_b_hash,
        "branch_a_linear_system_consistent": branch_source_rank == 2,
        "branch_b_linear_system_consistent": branch_source_rank == 2,
        "differing_image_row_index": 1,
        "differing_image_delta": differing_delta,
        "compatible_outcome_analysis_used_first_response_labels": True,
        "compatible_outcome_analysis_used_responses_2_through_15": False,
        "compatible_outcomes_are_exact_unobserved_full_product_rows": True,
        "actual_next_response_read": False,
        "sealed_answer_read": False,
        "differing_restricted_map_witness": True,
    }
    return OneResponseCheckpoint(**kwargs, checkpoint_sha256=_sha256(payload))


def fit_active_partial_operators(
    learner_input: PartialOperatorLearnerInput,
    passive: PassivePartialDiscovery,
    first_response: OpaqueMembershipResponse,
    one_response_checkpoint: OneResponseCheckpoint,
    remaining_responses: Sequence[OpaqueMembershipResponse],
) -> ActivePartialDiscovery:
    """Fit all restricted maps after the staged 1+14 controller release."""

    if type(learner_input) is not PartialOperatorLearnerInput:
        raise TypeError("learner_input must be exact PartialOperatorLearnerInput")
    if learner_input.is_full_support_control:
        raise ValueError("full-support controls require no active fit")
    reconstructed_passive = fit_passive_partial_operators(learner_input)
    if type(passive) is not PassivePartialDiscovery or passive != reconstructed_passive:
        raise ValueError("passive result is not the pure fit of this learner input")
    reconstructed_checkpoint = analyze_one_response_checkpoint(
        learner_input, passive, first_response
    )
    if (
        type(one_response_checkpoint) is not OneResponseCheckpoint
        or one_response_checkpoint != reconstructed_checkpoint
    ):
        raise ValueError("one-response checkpoint is not the staged pure analysis")
    if not isinstance(remaining_responses, (tuple, list)):
        raise TypeError("remaining_responses must be an exact sequence")
    if len(remaining_responses) != _ACTIVE_RESPONSE_COUNT - 1:
        raise ValueError("the second controller release must contain 14 responses")
    responses = (first_response,) + tuple(remaining_responses)
    if len(responses) > learner_input.budgets.max_active_responses:
        raise OpaquePartialOperatorLimitError("active-response budget exceeded")
    if 2 * len(responses) > learner_input.budgets.max_categorical_labels:
        raise OpaquePartialOperatorLimitError("categorical-label budget exceeded")
    selected = learner_input.candidate_edge_requests[:_ACTIVE_RESPONSE_COUNT]
    for index, (response, request) in enumerate(zip(responses, selected, strict=True), 1):
        if type(response) is not OpaqueMembershipResponse:
            raise TypeError("active rows must be exact OpaqueMembershipResponse")
        if response.request != request or response.response_ordinal != index:
            raise ValueError("response order does not match the precommitted active basis")
    realization, observed_ranks = _build_realization(
        learner_input, responses, require_complete_maps=True
    )
    if realization.ambient_rank != _FULL_RANK:
        raise ValueError("active answers must produce the exact rank-five realization")
    operators = realization.operator_certificates
    if len(operators) != _EVENT_COUNT:
        raise AssertionError("active realization did not produce ten restricted maps")
    domain_multiset = tuple(sorted(row.legal_domain_rank for row in operators))
    extension_multiset = tuple(sorted(row.total_extension_nullity for row in operators))
    if sum(row.observed_source_rank for row in observed_ranks) != sum(
        row.legal_domain_rank for row in operators
    ):
        raise ValueError("15 responses did not close every observed legal-domain span")
    kwargs = {
        "scope": PartialOperatorScope.GUARDED_ABSENCE_AWARE_PARTIAL_OPERATORS,
        "status": PartialOperatorStatus.SYNTHETIC_PROTOCOL_IMPLEMENTATION_REHEARSAL,
        "learner_input_sha256": learner_input.input_sha256,
        "passive_result_sha256": passive.result_sha256,
        "commitment_sha256": passive.active_commitment.commitment_sha256,
        "response_sha256s": tuple(row.response_sha256 for row in responses),
        "active_batch_call_count": 2,
        "active_response_count": _ACTIVE_RESPONSE_COUNT,
        "returned_categorical_token_count": 2 * _ACTIVE_RESPONSE_COUNT,
        "returned_target_label_fields_per_response": 2,
        "one_response_checkpoint": one_response_checkpoint,
        "active_rank_certificate": realization.rank_certificate,
        "realization": realization,
        "operator_certificates": operators,
        "restricted_legal_domain_maps_identified": True,
        "aggregate_total_extension_nullity": sum(
            row.total_extension_nullity for row in operators
        ),
        "legal_domain_rank_multiset": domain_multiset,
        "total_extension_nullity_multiset": extension_multiset,
        "total_operator": None,
        "sealed_answers_received_during_fit": False,
        "semantic_roles_received": False,
        "controller_nonce_received": False,
        "executor_received": False,
        "schema": _ACTIVE_SCHEMA,
    }
    payload = {
        "schema": _ACTIVE_SCHEMA,
        "scope": kwargs["scope"],
        "status": kwargs["status"],
        "learner_input_sha256": learner_input.input_sha256,
        "passive_result_sha256": passive.result_sha256,
        "commitment_sha256": passive.active_commitment.commitment_sha256,
        "response_sha256s": kwargs["response_sha256s"],
        "active_batch_call_count": 2,
        "active_response_count": _ACTIVE_RESPONSE_COUNT,
        "returned_categorical_token_count": 2 * _ACTIVE_RESPONSE_COUNT,
        "returned_target_label_fields_per_response": 2,
        "one_response_checkpoint": one_response_checkpoint._payload(include_checkpoint_sha=True),
        "active_rank_certificate": realization.rank_certificate._payload(include_certificate_sha=True),
        "realization": realization._payload(include_model_sha=True),
        "operator_certificates": [row._payload(include_certificate_sha=True) for row in operators],
        "restricted_legal_domain_maps_identified": True,
        "aggregate_total_extension_nullity": kwargs["aggregate_total_extension_nullity"],
        "legal_domain_rank_multiset": domain_multiset,
        "total_extension_nullity_multiset": extension_multiset,
        "total_operator": None,
        "sealed_answers_received_during_fit": False,
        "semantic_roles_received": False,
        "controller_nonce_received": False,
        "executor_received": False,
    }
    return ActivePartialDiscovery(**kwargs, result_sha256=_sha256(payload))


@dataclass(frozen=True)
class OpaqueSealedProgram:
    program: OpaqueWord
    expected_answers: tuple[str, ...]
    probe_kind: str
    relation_group: str | None
    relation_expectation: str
    program_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.program, tuple) or not self.program:
            raise ValueError("sealed program must be a nonempty tuple")
        for token in self.program:
            _require_token("sealed program token", token)
        _validate_answers("sealed expected_answers", self.expected_answers)
        if self.probe_kind not in ("heldout_legal_edge", "long_path"):
            raise ValueError("unknown sealed probe kind")
        if self.relation_expectation not in ("none", "equal", "not_equal"):
            raise ValueError("unknown path relation expectation")
        if self.relation_expectation == "none" and self.relation_group is not None:
            raise ValueError("standalone probe cannot have a relation group")
        if self.relation_expectation != "none" and (
            type(self.relation_group) is not str or not self.relation_group
        ):
            raise ValueError("relational probe requires a group")
        expected = _sha256(
            {
                "program": self.program,
                "expected_answers": self.expected_answers,
                "probe_kind": self.probe_kind,
                "relation_group": self.relation_group,
                "relation_expectation": self.relation_expectation,
            }
        )
        if _require_sha256("program_sha256", self.program_sha256) != expected:
            raise ValueError("sealed program digest mismatch")

    def payload(self) -> dict[str, object]:
        return {
            "program": self.program,
            "expected_answers": self.expected_answers,
            "probe_kind": self.probe_kind,
            "relation_group": self.relation_group,
            "relation_expectation": self.relation_expectation,
            "program_sha256": self.program_sha256,
        }


def _make_sealed_program(
    program: OpaqueWord,
    answers: tuple[str, ...],
    *,
    probe_kind: str,
    relation_group: str | None = None,
    relation_expectation: str = "none",
) -> OpaqueSealedProgram:
    payload = {
        "program": program,
        "expected_answers": answers,
        "probe_kind": probe_kind,
        "relation_group": relation_group,
        "relation_expectation": relation_expectation,
    }
    return OpaqueSealedProgram(
        **payload,
        program_sha256=_sha256(payload),
    )


@dataclass(frozen=True)
class ToyPartialControllerEnvironment:
    """Trusted controller material; never a learner-function argument."""

    kind: EnvironmentKind
    relabel_block: int
    pseudoheldout_cell: SemanticCell | None
    trusted_controller_nonce: str
    learner_input: PartialOperatorLearnerInput
    initial_answers: tuple[str, ...]
    active_responses: tuple[OpaqueMembershipResponse, ...]
    sealed_edge_programs: tuple[OpaqueSealedProgram, ...]
    long_path_programs: tuple[OpaqueSealedProgram, ...]
    canonical_state_answers: tuple[tuple[SemanticState, tuple[str, ...]], ...]
    canonical_event_tokens: tuple[tuple[SemanticAction, str], ...]
    passive_semantic_family_counts: tuple[tuple[str, int], ...]
    full_semantic_family_counts: tuple[tuple[str, int], ...]
    controller_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in (EnvironmentKind.FULL_SUPPORT_CONTROL, EnvironmentKind.ROTATED_OMISSION):
            raise ValueError("unknown controller environment kind")
        if self.relabel_block not in (0, 1):
            raise ValueError("relabel_block must be zero or one")
        _require_nonce(self.trusted_controller_nonce)
        if type(self.learner_input) is not PartialOperatorLearnerInput:
            raise TypeError("learner_input must be exact PartialOperatorLearnerInput")
        _validate_answers("initial_answers", self.initial_answers)
        if self.kind is EnvironmentKind.FULL_SUPPORT_CONTROL:
            if self.pseudoheldout_cell is not None:
                raise ValueError("full-support control cannot name an omitted cell")
            if not self.learner_input.is_full_support_control:
                raise ValueError("full-support controller must expose 44 passive edges")
            if self.active_responses or self.sealed_edge_programs:
                raise ValueError("full-support control has zero active/sealed edge rows")
        else:
            if self.pseudoheldout_cell not in ((0, 0), (0, 1), (1, 0), (1, 1)):
                raise ValueError("rotated omission requires one K=2,V=2 cell")
            if self.learner_input.is_full_support_control:
                raise ValueError("omission controller must expose only 21 passive edges")
            if len(self.active_responses) != _ACTIVE_RESPONSE_COUNT:
                raise ValueError("omission controller must contain 15 hidden active responses")
            if len(self.sealed_edge_programs) != _SEALED_EDGE_COUNT:
                raise ValueError("omission controller must contain eight sealed edge programs")
        if len(self.long_path_programs) != _LONG_PROBE_COUNT:
            raise ValueError("controller must contain exactly 12 long/path probes")
        if len({row.program for row in self.long_path_programs}) != _LONG_PROBE_COUNT:
            raise ValueError("long/path programs must be unique")
        if set(row.program for row in self.long_path_programs).intersection(
            row.request.program for row in self.active_responses
        ):
            raise ValueError("long/path programs cannot replay an active acquisition word")
        if set(row.program for row in self.long_path_programs).intersection(
            row.program for row in self.sealed_edge_programs
        ):
            raise ValueError("long/path programs must differ from heldout edge programs")
        learner_visible_labeled_words = {
            row.word for row in self.learner_input.passive_state_observations
        }
        learner_visible_labeled_words.update(
            row.request.source_word
            for row in self.learner_input.passive_edge_observations
        )
        learner_visible_labeled_words.update(
            row.request.program
            for row in self.learner_input.passive_edge_observations
        )
        learner_visible_labeled_words.update(
            row.request.program for row in self.active_responses
        )
        if set(row.program for row in self.long_path_programs).intersection(
            learner_visible_labeled_words
        ):
            raise ValueError("long/path outputs must be absent from all fit-time labeled words")
        committed_sealed_programs = tuple(
            row.program
            for row in self.learner_input.candidate_edge_requests[
                _ACTIVE_RESPONSE_COUNT:
            ]
        )
        if tuple(row.program for row in self.sealed_edge_programs) != committed_sealed_programs:
            raise ValueError("sealed edge programs must exactly match the committed eight requests")
        if len(self.canonical_state_answers) != _STATE_COUNT:
            raise ValueError("controller must bind all nine canonical state diagnostics")
        if len({state for state, _ in self.canonical_state_answers}) != _STATE_COUNT:
            raise ValueError("canonical state diagnostics must be unique")
        if len(self.canonical_event_tokens) != _EVENT_COUNT:
            raise ValueError("controller must bind all ten canonical event roles")
        if len({token for _, token in self.canonical_event_tokens}) != _EVENT_COUNT:
            raise ValueError("canonical event-token mapping must be bijective")
        expected_full = (("bind", 12), ("copy", 8), ("invalidate", 12), ("update", 12))
        if self.full_semantic_family_counts != expected_full:
            raise ValueError("full legal-edge family census mismatch")
        expected_passive = (
            expected_full
            if self.kind is EnvironmentKind.FULL_SUPPORT_CONTROL
            else (("bind", 7), ("copy", 3), ("invalidate", 7), ("update", 4))
        )
        if self.passive_semantic_family_counts != expected_passive:
            raise ValueError("passive legal-edge family census mismatch")
        expected = _sha256(self._payload(include_controller_sha=False))
        if _require_sha256("controller_sha256", self.controller_sha256) != expected:
            raise ValueError("trusted controller digest mismatch")

    def _payload(self, *, include_controller_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": self.kind,
            "relabel_block": self.relabel_block,
            "pseudoheldout_cell": self.pseudoheldout_cell,
            "trusted_controller_nonce": self.trusted_controller_nonce,
            "learner_input": self.learner_input.payload(),
            "initial_answers": self.initial_answers,
            "active_responses": [_response_payload(row) for row in self.active_responses],
            "sealed_edge_programs": [row.payload() for row in self.sealed_edge_programs],
            "long_path_programs": [row.payload() for row in self.long_path_programs],
            "canonical_state_answers": self.canonical_state_answers,
            "canonical_event_tokens": self.canonical_event_tokens,
            "passive_semantic_family_counts": self.passive_semantic_family_counts,
            "full_semantic_family_counts": self.full_semantic_family_counts,
        }
        if include_controller_sha:
            payload["controller_sha256"] = self.controller_sha256
        return payload


def _preflight_controller_budgets(
    budgets: PartialOperatorBudgets,
    *,
    omission: bool,
) -> None:
    if budgets.max_word_length < max(len(row) for row in _LONG_CANONICAL_PROGRAMS):
        raise OpaquePartialOperatorLimitError("word budget is below the frozen long/path maximum")
    if budgets.max_event_tokens < _EVENT_COUNT:
        raise OpaquePartialOperatorLimitError("event-token budget is below ten")
    if budgets.max_domain_edges < _STATE_COUNT * _EVENT_COUNT:
        raise OpaquePartialOperatorLimitError("domain-mask budget is below 90")
    required_passive = _PASSIVE_EDGE_COUNT if omission else _LEGAL_EDGE_COUNT
    if budgets.max_passive_edges < required_passive:
        raise OpaquePartialOperatorLimitError("passive-edge budget is below the protocol arm")
    if budgets.max_basis_dimension < _FULL_RANK:
        raise OpaquePartialOperatorLimitError("basis budget is below exact rank five")
    if budgets.max_exact_rank_evaluations < 512:
        raise OpaquePartialOperatorLimitError("exact-rank budget is below conservative preflight")
    if budgets.max_rational_bit_length < 64:
        raise OpaquePartialOperatorLimitError("rational bit-length budget is below conservative preflight")
    if budgets.max_long_probes < _LONG_PROBE_COUNT:
        raise OpaquePartialOperatorLimitError("long-probe budget is below 12")
    if omission:
        if budgets.max_active_responses < _ACTIVE_RESPONSE_COUNT:
            raise OpaquePartialOperatorLimitError("active-response budget is below 15")
        if budgets.max_categorical_labels < 2 * _ACTIVE_RESPONSE_COUNT:
            raise OpaquePartialOperatorLimitError("categorical-label budget is below 30")
        if budgets.max_sealed_edges < _SEALED_EDGE_COUNT:
            raise OpaquePartialOperatorLimitError("sealed-edge budget is below eight")


def _build_controller_environment(
    *,
    kind: EnvironmentKind,
    omitted_cell: SemanticCell | None,
    relabel_block: int,
    controller_nonce: str,
    budgets: PartialOperatorBudgets,
) -> ToyPartialControllerEnvironment:
    nonce = _require_nonce(controller_nonce)
    omission = kind is EnvironmentKind.ROTATED_OMISSION
    _preflight_controller_budgets(budgets, omission=omission)
    key_map, value_map = _canonical_to_raw_maps(
        omitted_cell=omitted_cell,
        relabel_block=relabel_block,
    )
    raw_actions = tuple(
        _map_action_to_raw(action, key_map, value_map) for action in _ACTIONS
    )
    if len(set(raw_actions)) != _EVENT_COUNT:
        raise AssertionError("semantic relabel must preserve the action alphabet")
    raw_event_token = {
        action: _opaque_token(nonce, "event", action)
        for action in raw_actions
    }
    canonical_event_token = {
        canonical: raw_event_token[raw]
        for canonical, raw in zip(_ACTIONS, raw_actions, strict=True)
    }
    raw_query_token = {
        key: _opaque_token(nonce, "query", key) for key in (0, 1)
    }
    raw_answer_token = {
        value: _opaque_token(nonce, "answer", value) for value in (_A, 0, 1)
    }
    event_tokens = tuple(
        value for value in _nonce_order(nonce, "event-vocabulary-order", tuple(raw_event_token.values()))
    )
    query_tokens = tuple(
        value for value in _nonce_order(nonce, "query-vocabulary-order", tuple(raw_query_token.values()))
    )
    answer_tokens = tuple(
        value for value in _nonce_order(nonce, "answer-vocabulary-order", tuple(raw_answer_token.values()))
    )
    query_key_by_token = {token: key for key, token in raw_query_token.items()}

    def answers_for_state(state: SemanticState) -> tuple[str, ...]:
        raw_state = _map_state_to_raw(state, key_map, value_map)
        return tuple(
            raw_answer_token[raw_state[query_key_by_token[token]]]
            for token in query_tokens
        )

    def opaque_word(program: Sequence[SemanticAction]) -> OpaqueWord:
        return tuple(canonical_event_token[action] for action in program)

    representatives = {
        state: opaque_word(_canonical_representative(state)) for state in _STATES
    }
    request_by_pair = {
        (state, action): _make_request(
            representatives[state], canonical_event_token[action]
        )
        for state in _STATES
        for action in _ACTIONS
    }
    legal_pairs = {(source, action) for source, action, _ in _LEGAL_EDGES}
    defined_requests = tuple(
        request_by_pair[pair]
        for pair in _nonce_order(nonce, "defined-mask-row-order", tuple(legal_pairs))
    )
    illegal_pairs = tuple(
        (state, action)
        for state in _STATES
        for action in _ACTIONS
        if (state, action) not in legal_pairs
    )
    undefined_requests = tuple(
        request_by_pair[pair]
        for pair in _nonce_order(nonce, "undefined-mask-row-order", illegal_pairs)
    )

    canonical_omitted = (0, 0)
    passive_edges = tuple(
        (source, action, target)
        for source, action, target in _LEGAL_EDGES
        if not omission
        or (
            not _state_contains_cell(source, canonical_omitted)
            and not _state_contains_cell(target, canonical_omitted)
        )
    )
    passive_edge_rows: list[OpaqueEdgeObservation] = []
    for source, action, target in passive_edges:
        request = request_by_pair[(source, action)]
        passive_edge_rows.append(
            _make_observation(request, answers_for_state(source), answers_for_state(target))
        )
    passive_edge_observations = tuple(
        row
        for row in _nonce_order(
            nonce, "passive-edge-row-order", tuple(passive_edge_rows)
        )
    )
    passive_states = tuple(
        state
        for state in _STATES
        if not omission or not _state_contains_cell(state, canonical_omitted)
    )
    passive_state_observations = tuple(
        OpaqueStateObservation(representatives[state], answers_for_state(state))
        for state in _nonce_order(nonce, "passive-state-row-order", passive_states)
    )

    active_pairs: list[tuple[SemanticState, SemanticAction, SemanticState]] = []
    for program in _ACTIVE_CANONICAL_PROGRAMS:
        source = _execute_semantic(program[:-1])
        target = _execute_semantic(program)
        assert source is not None and target is not None
        active_pairs.append((source, program[-1], target))
    sealed_pairs: list[tuple[SemanticState, SemanticAction, SemanticState]] = []
    for program in _SEALED_EDGE_CANONICAL_PROGRAMS:
        source = _execute_semantic(program[:-1])
        target = _execute_semantic(program)
        assert source is not None and target is not None
        sealed_pairs.append((source, program[-1], target))
    if omission:
        candidate_requests = tuple(
            request_by_pair[(source, action)]
            for source, action, _ in active_pairs + sealed_pairs
        )
    else:
        candidate_requests = ()
    learner_input = _make_learner_input(
        event_tokens=event_tokens,  # type: ignore[arg-type]
        query_tokens=query_tokens,  # type: ignore[arg-type]
        answer_tokens=answer_tokens,  # type: ignore[arg-type]
        passive_state_observations=passive_state_observations,
        passive_edge_observations=passive_edge_observations,
        candidate_edge_requests=candidate_requests,
        defined_edge_requests=defined_requests,
        undefined_edge_requests=undefined_requests,
        budgets=budgets,
    )
    active_responses = (
        tuple(
            _make_response(
                request_by_pair[(source, action)],
                answers_for_state(target),
                ordinal,
            )
            for ordinal, (source, action, target) in enumerate(active_pairs, 1)
        )
        if omission
        else ()
    )
    sealed_edge_programs = (
        tuple(
            _make_sealed_program(
                opaque_word(program),
                answers_for_state(target),
                probe_kind="heldout_legal_edge",
            )
            for program, (_, _, target) in zip(
                _SEALED_EDGE_CANONICAL_PROGRAMS, sealed_pairs, strict=True
            )
        )
        if omission
        else ()
    )
    relation_rows: dict[int, tuple[str | None, str]] = {
        2: ("update_order_commutes", "equal"),
        3: ("update_order_commutes", "equal"),
        4: ("copy_order_noncommutes", "not_equal"),
        5: ("copy_order_noncommutes", "not_equal"),
        9: ("alternate_path_equal", "equal"),
        11: ("alternate_path_equal", "equal"),
    }
    long_path_programs: list[OpaqueSealedProgram] = []
    for index, program in enumerate(_LONG_CANONICAL_PROGRAMS):
        target = _execute_semantic(program)
        assert target is not None
        relation_group, expectation = relation_rows.get(index, (None, "none"))
        long_path_programs.append(
            _make_sealed_program(
                opaque_word(program),
                answers_for_state(target),
                probe_kind="long_path",
                relation_group=relation_group,
                relation_expectation=expectation,
            )
        )
    family_counts: dict[str, int] = {}
    for _, action, _ in passive_edges:
        family_counts[action[0]] = family_counts.get(action[0], 0) + 1
    full_counts: dict[str, int] = {}
    for _, action, _ in _LEGAL_EDGES:
        full_counts[action[0]] = full_counts.get(action[0], 0) + 1
    kwargs = {
        "kind": kind,
        "relabel_block": relabel_block,
        "pseudoheldout_cell": omitted_cell,
        "trusted_controller_nonce": nonce,
        "learner_input": learner_input,
        "initial_answers": answers_for_state((_A, _A)),
        "active_responses": active_responses,
        "sealed_edge_programs": sealed_edge_programs,
        "long_path_programs": tuple(long_path_programs),
        "canonical_state_answers": tuple(
            (state, answers_for_state(state)) for state in _STATES
        ),
        "canonical_event_tokens": tuple(
            (action, canonical_event_token[action]) for action in _ACTIONS
        ),
        "passive_semantic_family_counts": tuple(sorted(family_counts.items())),
        "full_semantic_family_counts": tuple(sorted(full_counts.items())),
    }
    payload = {
        "kind": kind,
        "relabel_block": relabel_block,
        "pseudoheldout_cell": omitted_cell,
        "trusted_controller_nonce": nonce,
        "learner_input": learner_input.payload(),
        "initial_answers": kwargs["initial_answers"],
        "active_responses": [_response_payload(row) for row in active_responses],
        "sealed_edge_programs": [row.payload() for row in sealed_edge_programs],
        "long_path_programs": [row.payload() for row in long_path_programs],
        "canonical_state_answers": kwargs["canonical_state_answers"],
        "canonical_event_tokens": kwargs["canonical_event_tokens"],
        "passive_semantic_family_counts": kwargs["passive_semantic_family_counts"],
        "full_semantic_family_counts": kwargs["full_semantic_family_counts"],
    }
    return ToyPartialControllerEnvironment(
        **kwargs,
        controller_sha256=_sha256(payload),
    )


def build_full_support_controller_environment(
    relabel_block: int,
    *,
    controller_nonce: str,
    budgets: PartialOperatorBudgets | None = None,
) -> ToyPartialControllerEnvironment:
    return _build_controller_environment(
        kind=EnvironmentKind.FULL_SUPPORT_CONTROL,
        omitted_cell=None,
        relabel_block=relabel_block,
        controller_nonce=controller_nonce,
        budgets=PartialOperatorBudgets() if budgets is None else budgets,
    )


def build_omission_controller_environment(
    pseudoheldout_cell: SemanticCell,
    relabel_block: int,
    *,
    controller_nonce: str,
    budgets: PartialOperatorBudgets | None = None,
) -> ToyPartialControllerEnvironment:
    return _build_controller_environment(
        kind=EnvironmentKind.ROTATED_OMISSION,
        omitted_cell=pseudoheldout_cell,
        relabel_block=relabel_block,
        controller_nonce=controller_nonce,
        budgets=PartialOperatorBudgets() if budgets is None else budgets,
    )


def release_first_active_response(
    controller: ToyPartialControllerEnvironment,
    passive: PassivePartialDiscovery,
) -> OpaqueMembershipResponse:
    """Trusted staged call one: release response 1 and nothing else."""

    if type(controller) is not ToyPartialControllerEnvironment:
        raise TypeError("controller must be exact ToyPartialControllerEnvironment")
    if controller.kind is not EnvironmentKind.ROTATED_OMISSION:
        raise ValueError("full-support controls have no active responses")
    reconstructed = fit_passive_partial_operators(controller.learner_input)
    if type(passive) is not PassivePartialDiscovery or passive != reconstructed:
        raise ValueError("passive result does not match the controller learner input")
    return controller.active_responses[0]


def release_remaining_active_responses(
    controller: ToyPartialControllerEnvironment,
    passive: PassivePartialDiscovery,
    checkpoint: OneResponseCheckpoint,
) -> tuple[OpaqueMembershipResponse, ...]:
    """Trusted staged call two: release rows 2--15 after checkpoint freeze."""

    if type(controller) is not ToyPartialControllerEnvironment:
        raise TypeError("controller must be exact ToyPartialControllerEnvironment")
    if controller.kind is not EnvironmentKind.ROTATED_OMISSION:
        raise ValueError("full-support controls have no active responses")
    first = controller.active_responses[0]
    reconstructed = analyze_one_response_checkpoint(
        controller.learner_input, passive, first
    )
    if type(checkpoint) is not OneResponseCheckpoint or checkpoint != reconstructed:
        raise ValueError("remaining responses cannot open before the exact checkpoint")
    return controller.active_responses[1:]


@dataclass(frozen=True)
class SealedProgramPrediction:
    program_sha256: str
    probe_kind: str
    predicted_answers: tuple[str, ...]
    expected_answers: tuple[str, ...]
    exact: bool

    def __post_init__(self) -> None:
        _require_sha256("program_sha256", self.program_sha256)
        if self.probe_kind not in ("heldout_legal_edge", "long_path"):
            raise ValueError("unknown sealed prediction kind")
        _validate_answers("predicted_answers", self.predicted_answers)
        _validate_answers("expected_answers", self.expected_answers)
        if _require_bool("exact", self.exact) != (
            self.predicted_answers == self.expected_answers
        ):
            raise ValueError("sealed prediction exactness flag mismatch")


@dataclass(frozen=True)
class PathRelationEvaluation:
    relation_group: str
    expectation: str
    program_sha256s: tuple[str, str]
    predicted_answer_rows: tuple[tuple[str, ...], tuple[str, ...]]
    satisfied: bool

    def __post_init__(self) -> None:
        if type(self.relation_group) is not str or not self.relation_group:
            raise ValueError("path relation group must be nonempty")
        if self.expectation not in ("equal", "not_equal"):
            raise ValueError("path relation expectation must be equal/not_equal")
        if len(self.program_sha256s) != 2:
            raise ValueError("path relation must contain exactly two programs")
        for digest in self.program_sha256s:
            _require_sha256("path program digest", digest)
        if len(self.predicted_answer_rows) != 2:
            raise ValueError("path relation must contain exactly two prediction rows")
        for answers in self.predicted_answer_rows:
            _validate_answers("path predicted answers", answers)
        expected = (
            self.predicted_answer_rows[0] == self.predicted_answer_rows[1]
            if self.expectation == "equal"
            else self.predicted_answer_rows[0] != self.predicted_answer_rows[1]
        )
        if _require_bool("satisfied", self.satisfied) != expected:
            raise ValueError("path relation satisfaction mismatch")


@dataclass(frozen=True)
class SealedPartialEvaluation:
    model_sha256: str
    controller_sha256: str
    predictions: tuple[SealedProgramPrediction, ...]
    path_relations: tuple[PathRelationEvaluation, ...]
    sealed_edge_program_count: int
    long_path_program_count: int
    total_program_count: int
    categorical_label_prediction_count: int
    exact_program_count: int
    satisfied_path_relation_count: int
    all_exact: bool
    evaluation_sha256: str

    def __post_init__(self) -> None:
        _require_sha256("model_sha256", self.model_sha256)
        _require_sha256("controller_sha256", self.controller_sha256)
        edge_count = sum(row.probe_kind == "heldout_legal_edge" for row in self.predictions)
        long_count = sum(row.probe_kind == "long_path" for row in self.predictions)
        if self.sealed_edge_program_count != edge_count:
            raise ValueError("sealed edge prediction count mismatch")
        if self.long_path_program_count != long_count or long_count != _LONG_PROBE_COUNT:
            raise ValueError("long/path prediction count mismatch")
        if self.total_program_count != len(self.predictions):
            raise ValueError("total sealed program count mismatch")
        if self.categorical_label_prediction_count != 2 * len(self.predictions):
            raise ValueError("sealed categorical-label prediction count mismatch")
        exact_count = sum(row.exact for row in self.predictions)
        if self.exact_program_count != exact_count:
            raise ValueError("exact sealed-program count mismatch")
        relation_count = sum(row.satisfied for row in self.path_relations)
        if len(self.path_relations) != 3 or relation_count != self.satisfied_path_relation_count:
            raise ValueError("path relation aggregate mismatch")
        expected_all = exact_count == len(self.predictions) and relation_count == 3
        if _require_bool("all_exact", self.all_exact) != expected_all:
            raise ValueError("sealed aggregate exactness mismatch")
        expected = _sha256(self._payload(include_evaluation_sha=False))
        if _require_sha256("evaluation_sha256", self.evaluation_sha256) != expected:
            raise ValueError("sealed evaluation digest mismatch")

    def _payload(self, *, include_evaluation_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "model_sha256": self.model_sha256,
            "controller_sha256": self.controller_sha256,
            "predictions": [row.__dict__ for row in self.predictions],
            "path_relations": [row.__dict__ for row in self.path_relations],
            "sealed_edge_program_count": self.sealed_edge_program_count,
            "long_path_program_count": self.long_path_program_count,
            "total_program_count": self.total_program_count,
            "categorical_label_prediction_count": self.categorical_label_prediction_count,
            "exact_program_count": self.exact_program_count,
            "satisfied_path_relation_count": self.satisfied_path_relation_count,
            "all_exact": self.all_exact,
        }
        if include_evaluation_sha:
            payload["evaluation_sha256"] = self.evaluation_sha256
        return payload


def _evaluate_sealed_programs(
    controller: ToyPartialControllerEnvironment,
    model: ExactPartialRealization,
) -> SealedPartialEvaluation:
    if type(controller) is not ToyPartialControllerEnvironment:
        raise TypeError("controller must be exact ToyPartialControllerEnvironment")
    if type(model) is not ExactPartialRealization or not model.restricted_maps_complete:
        raise ValueError("sealed evaluation requires complete restricted maps")
    programs = controller.sealed_edge_programs + controller.long_path_programs
    prediction_rows: list[SealedProgramPrediction] = []
    for row in programs:
        predicted = model.predict_answers(
            row.program, initial_answers=controller.initial_answers
        )
        prediction_rows.append(
            SealedProgramPrediction(
                program_sha256=row.program_sha256,
                probe_kind=row.probe_kind,
                predicted_answers=predicted,
                expected_answers=row.expected_answers,
                exact=predicted == row.expected_answers,
            )
        )
    predictions = tuple(prediction_rows)
    prediction_by_sha = {row.program_sha256: row for row in predictions}
    relation_groups: dict[str, list[OpaqueSealedProgram]] = {}
    for row in controller.long_path_programs:
        if row.relation_group is not None:
            relation_groups.setdefault(row.relation_group, []).append(row)
    path_relations: list[PathRelationEvaluation] = []
    for group in sorted(relation_groups):
        rows = relation_groups[group]
        if len(rows) != 2 or rows[0].relation_expectation != rows[1].relation_expectation:
            raise ValueError("sealed relation group must contain exactly one declared pair")
        expectation = rows[0].relation_expectation
        answer_rows = tuple(
            prediction_by_sha[row.program_sha256].predicted_answers for row in rows
        )
        satisfied = (
            answer_rows[0] == answer_rows[1]
            if expectation == "equal"
            else answer_rows[0] != answer_rows[1]
        )
        path_relations.append(
            PathRelationEvaluation(
                relation_group=group,
                expectation=expectation,
                program_sha256s=(rows[0].program_sha256, rows[1].program_sha256),
                predicted_answer_rows=(answer_rows[0], answer_rows[1]),
                satisfied=satisfied,
            )
        )
    kwargs = {
        "model_sha256": model.model_sha256,
        "controller_sha256": controller.controller_sha256,
        "predictions": predictions,
        "path_relations": tuple(path_relations),
        "sealed_edge_program_count": len(controller.sealed_edge_programs),
        "long_path_program_count": len(controller.long_path_programs),
        "total_program_count": len(programs),
        "categorical_label_prediction_count": 2 * len(programs),
        "exact_program_count": sum(row.exact for row in predictions),
        "satisfied_path_relation_count": sum(row.satisfied for row in path_relations),
        "all_exact": all(row.exact for row in predictions) and all(
            row.satisfied for row in path_relations
        ),
    }
    payload = {
        "model_sha256": model.model_sha256,
        "controller_sha256": controller.controller_sha256,
        "predictions": [row.__dict__ for row in predictions],
        "path_relations": [row.__dict__ for row in path_relations],
        "sealed_edge_program_count": kwargs["sealed_edge_program_count"],
        "long_path_program_count": kwargs["long_path_program_count"],
        "total_program_count": kwargs["total_program_count"],
        "categorical_label_prediction_count": kwargs["categorical_label_prediction_count"],
        "exact_program_count": kwargs["exact_program_count"],
        "satisfied_path_relation_count": kwargs["satisfied_path_relation_count"],
        "all_exact": kwargs["all_exact"],
    }
    return SealedPartialEvaluation(
        **kwargs,
        evaluation_sha256=_sha256(payload),
    )


@dataclass(frozen=True)
class UndefinedEdgeRejection:
    request_sha256: str
    event_token: str
    source_coordinates: RationalVector
    rejected_as_outside_legal_domain: bool

    def __post_init__(self) -> None:
        _require_sha256("request_sha256", self.request_sha256)
        _require_token("event_token", self.event_token)
        if len(self.source_coordinates) != _FULL_RANK or any(
            type(value) is not Rational for value in self.source_coordinates
        ):
            raise ValueError("undefined source coordinate must be exact rank five")
        if not _require_bool(
            "rejected_as_outside_legal_domain",
            self.rejected_as_outside_legal_domain,
        ):
            raise ValueError("every undefined edge must be rejected")


@dataclass(frozen=True)
class UndefinedDomainRejectionCertificate:
    learner_input_sha256: str
    model_sha256: str
    rows: tuple[UndefinedEdgeRejection, ...]
    undefined_edge_count: int
    rejected_edge_count: int
    undefined_words_treated_as_absent_constraints: bool
    zero_or_dead_state_filling_used: bool
    certificate_sha256: str
    schema: str = _UNDEFINED_REJECTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _UNDEFINED_REJECTION_SCHEMA:
            raise ValueError("unknown undefined-domain rejection schema")
        _require_sha256("learner_input_sha256", self.learner_input_sha256)
        _require_sha256("model_sha256", self.model_sha256)
        if len(self.rows) != _UNDEFINED_EDGE_COUNT:
            raise ValueError("undefined-domain certificate must contain all 46 rows")
        if len({row.request_sha256 for row in self.rows}) != _UNDEFINED_EDGE_COUNT:
            raise ValueError("undefined-domain rejection requests must be unique")
        if self.undefined_edge_count != _UNDEFINED_EDGE_COUNT:
            raise ValueError("undefined-edge count must be 46")
        actual_rejected = sum(row.rejected_as_outside_legal_domain for row in self.rows)
        if self.rejected_edge_count != actual_rejected or actual_rejected != _UNDEFINED_EDGE_COUNT:
            raise ValueError("all 46 undefined edges must be rejected")
        if not _require_bool(
            "undefined_words_treated_as_absent_constraints",
            self.undefined_words_treated_as_absent_constraints,
        ):
            raise ValueError("undefined words must be absent constraints")
        if _require_bool(
            "zero_or_dead_state_filling_used", self.zero_or_dead_state_filling_used
        ):
            raise ValueError("undefined-domain certificate forbids zero/dead filling")
        expected = _sha256(self._payload(include_certificate_sha=False))
        if _require_sha256("certificate_sha256", self.certificate_sha256) != expected:
            raise ValueError("undefined-domain rejection digest mismatch")

    def _payload(self, *, include_certificate_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "learner_input_sha256": self.learner_input_sha256,
            "model_sha256": self.model_sha256,
            "rows": [row.__dict__ for row in self.rows],
            "undefined_edge_count": self.undefined_edge_count,
            "rejected_edge_count": self.rejected_edge_count,
            "undefined_words_treated_as_absent_constraints": self.undefined_words_treated_as_absent_constraints,
            "zero_or_dead_state_filling_used": self.zero_or_dead_state_filling_used,
        }
        if include_certificate_sha:
            payload["certificate_sha256"] = self.certificate_sha256
        return payload


def _certify_undefined_domain_rejections(
    controller: ToyPartialControllerEnvironment,
    model: ExactPartialRealization,
) -> UndefinedDomainRejectionCertificate:
    responses: tuple[OpaqueMembershipResponse, ...] = (
        controller.active_responses
        if controller.kind is EnvironmentKind.ROTATED_OMISSION
        else ()
    )
    known = _known_answer_table(controller.learner_input, responses)
    rows: list[UndefinedEdgeRejection] = []
    for request in controller.learner_input.undefined_edge_requests:
        if request.source_word not in known:
            raise ValueError("final answer table lacks an undefined-mask source diagnostic")
        coordinates = model.answers_to_coordinates(known[request.source_word])
        rejected = False
        try:
            model.apply_event(coordinates, request.event_token)
        except ValueError:
            rejected = True
        if not rejected:
            raise ValueError("learned partial map accepted an undefined source/event edge")
        rows.append(
            UndefinedEdgeRejection(
                request_sha256=request.request_sha256,
                event_token=request.event_token,
                source_coordinates=coordinates,
                rejected_as_outside_legal_domain=True,
            )
        )
    kwargs = {
        "learner_input_sha256": controller.learner_input.input_sha256,
        "model_sha256": model.model_sha256,
        "rows": tuple(rows),
        "undefined_edge_count": _UNDEFINED_EDGE_COUNT,
        "rejected_edge_count": _UNDEFINED_EDGE_COUNT,
        "undefined_words_treated_as_absent_constraints": True,
        "zero_or_dead_state_filling_used": False,
        "schema": _UNDEFINED_REJECTION_SCHEMA,
    }
    payload = {
        "schema": _UNDEFINED_REJECTION_SCHEMA,
        "learner_input_sha256": controller.learner_input.input_sha256,
        "model_sha256": model.model_sha256,
        "rows": [row.__dict__ for row in rows],
        "undefined_edge_count": _UNDEFINED_EDGE_COUNT,
        "rejected_edge_count": _UNDEFINED_EDGE_COUNT,
        "undefined_words_treated_as_absent_constraints": True,
        "zero_or_dead_state_filling_used": False,
    }
    return UndefinedDomainRejectionCertificate(
        **kwargs,
        certificate_sha256=_sha256(payload),
    )


@dataclass(frozen=True)
class ToyPartialEnvironmentResult:
    controller: ToyPartialControllerEnvironment
    passive: PassivePartialDiscovery
    compatible_omission_certificate: CompatibleOmissionHypothesisCertificate | None
    one_response_checkpoint: OneResponseCheckpoint | None
    active: ActivePartialDiscovery | None
    final_model: ExactPartialRealization
    sealed_evaluation: SealedPartialEvaluation
    undefined_domain_rejection: UndefinedDomainRejectionCertificate
    passive_edge_count: int
    active_response_count: int
    active_returned_categorical_token_count: int
    sealed_edge_count: int
    long_path_count: int
    legal_edge_partition_total: int
    authoritative_environment_reconstructed_pure_fit: bool
    nested_certificates_are_contextual_content_links: bool
    nested_certificates_independently_authenticated: bool
    learner_received_controller_material: bool
    model_behavior_passed: bool
    result_sha256: str
    schema: str = _ENVIRONMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _ENVIRONMENT_SCHEMA:
            raise ValueError("unknown toy partial environment-result schema")
        if type(self.controller) is not ToyPartialControllerEnvironment:
            raise TypeError("controller must be exact ToyPartialControllerEnvironment")
        reconstructed_passive = fit_passive_partial_operators(
            self.controller.learner_input
        )
        if type(self.passive) is not PassivePartialDiscovery or self.passive != reconstructed_passive:
            raise ValueError("environment passive result fails pure reconstruction")
        if self.controller.kind is EnvironmentKind.ROTATED_OMISSION:
            reconstructed_hypotheses = enumerate_publicly_compatible_omission_hypotheses(
                self.controller.learner_input
            )
            if (
                type(self.compatible_omission_certificate)
                is not CompatibleOmissionHypothesisCertificate
                or self.compatible_omission_certificate != reconstructed_hypotheses
                or reconstructed_hypotheses.compatible_hypothesis_count != 8
            ):
                raise ValueError("environment omission-hypothesis certificate fails reconstruction")
            first = self.controller.active_responses[0]
            reconstructed_checkpoint = analyze_one_response_checkpoint(
                self.controller.learner_input, self.passive, first
            )
            if (
                type(self.one_response_checkpoint) is not OneResponseCheckpoint
                or self.one_response_checkpoint != reconstructed_checkpoint
            ):
                raise ValueError("environment one-response checkpoint fails reconstruction")
            reconstructed_active = fit_active_partial_operators(
                self.controller.learner_input,
                self.passive,
                first,
                reconstructed_checkpoint,
                self.controller.active_responses[1:],
            )
            if type(self.active) is not ActivePartialDiscovery or self.active != reconstructed_active:
                raise ValueError("environment active result fails pure reconstruction")
            expected_model = reconstructed_active.realization
            expected_counts = (21, 15, 30, 8, 12, 44)
        else:
            if self.compatible_omission_certificate is not None:
                raise ValueError("full control cannot attach omission hypotheses")
            if self.one_response_checkpoint is not None or self.active is not None:
                raise ValueError("full control has no one-response/active result")
            expected_model = reconstructed_passive.realization
            expected_counts = (44, 0, 0, 0, 12, 44)
        if self.final_model != expected_model:
            raise ValueError("environment final model does not equal reconstructed pure fit")
        reconstructed_evaluation = _evaluate_sealed_programs(
            self.controller, expected_model
        )
        if (
            type(self.sealed_evaluation) is not SealedPartialEvaluation
            or self.sealed_evaluation != reconstructed_evaluation
        ):
            raise ValueError("environment sealed evaluation fails replay")
        reconstructed_undefined = _certify_undefined_domain_rejections(
            self.controller, expected_model
        )
        if (
            type(self.undefined_domain_rejection)
            is not UndefinedDomainRejectionCertificate
            or self.undefined_domain_rejection != reconstructed_undefined
        ):
            raise ValueError("environment undefined-domain rejection fails replay")
        actual_counts = (
            self.passive_edge_count,
            self.active_response_count,
            self.active_returned_categorical_token_count,
            self.sealed_edge_count,
            self.long_path_count,
            self.legal_edge_partition_total,
        )
        if actual_counts != expected_counts:
            raise ValueError("environment edge/label partition mismatch")
        for name, value, required in (
            ("authoritative_environment_reconstructed_pure_fit", self.authoritative_environment_reconstructed_pure_fit, True),
            ("nested_certificates_are_contextual_content_links", self.nested_certificates_are_contextual_content_links, True),
            ("nested_certificates_independently_authenticated", self.nested_certificates_independently_authenticated, False),
            ("learner_received_controller_material", self.learner_received_controller_material, False),
            ("model_behavior_passed", self.model_behavior_passed, reconstructed_evaluation.all_exact),
        ):
            if _require_bool(name, value) is not required:
                raise ValueError(f"{name} must be {required}")
        expected = _sha256(self._payload(include_result_sha=False))
        if _require_sha256("result_sha256", self.result_sha256) != expected:
            raise ValueError("environment result digest mismatch")

    def _payload(self, *, include_result_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "controller": self.controller._payload(include_controller_sha=True),
            "passive": self.passive._payload(include_result_sha=True),
            "compatible_omission_certificate": (
                None
                if self.compatible_omission_certificate is None
                else self.compatible_omission_certificate._payload(include_certificate_sha=True)
            ),
            "one_response_checkpoint": (
                None
                if self.one_response_checkpoint is None
                else self.one_response_checkpoint._payload(include_checkpoint_sha=True)
            ),
            "active": (
                None if self.active is None else self.active._payload(include_result_sha=True)
            ),
            "final_model": self.final_model._payload(include_model_sha=True),
            "sealed_evaluation": self.sealed_evaluation._payload(include_evaluation_sha=True),
            "undefined_domain_rejection": self.undefined_domain_rejection._payload(include_certificate_sha=True),
            "passive_edge_count": self.passive_edge_count,
            "active_response_count": self.active_response_count,
            "active_returned_categorical_token_count": self.active_returned_categorical_token_count,
            "sealed_edge_count": self.sealed_edge_count,
            "long_path_count": self.long_path_count,
            "legal_edge_partition_total": self.legal_edge_partition_total,
            "authoritative_environment_reconstructed_pure_fit": self.authoritative_environment_reconstructed_pure_fit,
            "nested_certificates_are_contextual_content_links": self.nested_certificates_are_contextual_content_links,
            "nested_certificates_independently_authenticated": self.nested_certificates_independently_authenticated,
            "learner_received_controller_material": self.learner_received_controller_material,
            "model_behavior_passed": self.model_behavior_passed,
        }
        if include_result_sha:
            payload["result_sha256"] = self.result_sha256
        return payload


def run_toy_partial_environment(
    controller: ToyPartialControllerEnvironment,
) -> ToyPartialEnvironmentResult:
    if type(controller) is not ToyPartialControllerEnvironment:
        raise TypeError("controller must be exact ToyPartialControllerEnvironment")
    passive = fit_passive_partial_operators(controller.learner_input)
    if controller.kind is EnvironmentKind.ROTATED_OMISSION:
        hypotheses = enumerate_publicly_compatible_omission_hypotheses(
            controller.learner_input
        )
        if hypotheses.compatible_hypothesis_count != 8:
            raise ValueError("omission input fails the eight-hypothesis opacity gate")
        first = release_first_active_response(controller, passive)
        checkpoint = analyze_one_response_checkpoint(
            controller.learner_input, passive, first
        )
        remaining = release_remaining_active_responses(
            controller, passive, checkpoint
        )
        active = fit_active_partial_operators(
            controller.learner_input,
            passive,
            first,
            checkpoint,
            remaining,
        )
        final_model = active.realization
        counts = (21, 15, 30, 8, 12, 44)
    else:
        hypotheses = None
        checkpoint = None
        active = None
        final_model = passive.realization
        counts = (44, 0, 0, 0, 12, 44)
    evaluation = _evaluate_sealed_programs(controller, final_model)
    undefined_rejection = _certify_undefined_domain_rejections(
        controller, final_model
    )
    kwargs = {
        "controller": controller,
        "passive": passive,
        "compatible_omission_certificate": hypotheses,
        "one_response_checkpoint": checkpoint,
        "active": active,
        "final_model": final_model,
        "sealed_evaluation": evaluation,
        "undefined_domain_rejection": undefined_rejection,
        "passive_edge_count": counts[0],
        "active_response_count": counts[1],
        "active_returned_categorical_token_count": counts[2],
        "sealed_edge_count": counts[3],
        "long_path_count": counts[4],
        "legal_edge_partition_total": counts[5],
        "authoritative_environment_reconstructed_pure_fit": True,
        "nested_certificates_are_contextual_content_links": True,
        "nested_certificates_independently_authenticated": False,
        "learner_received_controller_material": False,
        "model_behavior_passed": evaluation.all_exact,
        "schema": _ENVIRONMENT_SCHEMA,
    }
    payload = {
        "schema": _ENVIRONMENT_SCHEMA,
        "controller": controller._payload(include_controller_sha=True),
        "passive": passive._payload(include_result_sha=True),
        "compatible_omission_certificate": (
            None if hypotheses is None else hypotheses._payload(include_certificate_sha=True)
        ),
        "one_response_checkpoint": (
            None if checkpoint is None else checkpoint._payload(include_checkpoint_sha=True)
        ),
        "active": None if active is None else active._payload(include_result_sha=True),
        "final_model": final_model._payload(include_model_sha=True),
        "sealed_evaluation": evaluation._payload(include_evaluation_sha=True),
        "undefined_domain_rejection": undefined_rejection._payload(include_certificate_sha=True),
        "passive_edge_count": counts[0],
        "active_response_count": counts[1],
        "active_returned_categorical_token_count": counts[2],
        "sealed_edge_count": counts[3],
        "long_path_count": counts[4],
        "legal_edge_partition_total": counts[5],
        "authoritative_environment_reconstructed_pure_fit": True,
        "nested_certificates_are_contextual_content_links": True,
        "nested_certificates_independently_authenticated": False,
        "learner_received_controller_material": False,
        "model_behavior_passed": evaluation.all_exact,
    }
    return ToyPartialEnvironmentResult(
        **kwargs,
        result_sha256=_sha256(payload),
    )


@dataclass(frozen=True)
class PartialOperatorSimilarityCertificate:
    left_environment_result_sha256: str
    right_environment_result_sha256: str
    left_model_sha256: str
    right_model_sha256: str
    environment_kind: EnvironmentKind
    paired_omitted_cell: SemanticCell | None
    global_change_of_basis: RationalMatrix
    inverse_change_of_basis: RationalMatrix
    canonical_state_coordinate_pairs: tuple[
        tuple[RationalVector, RationalVector], ...
    ]
    gauge_fit_state_indices: tuple[int, ...]
    disjoint_test_state_indices: tuple[int, ...]
    gauge_fit_state_count: int
    disjoint_test_state_count: int
    gauge_fit_rows_sha256: str
    disjoint_test_rows_sha256: str
    categorical_readout_correspondence: tuple[
        tuple[tuple[str, ...], tuple[str, ...]], ...
    ]
    canonical_diagnostic_feature_pairs: tuple[
        tuple[RationalVector, RationalVector], ...
    ]
    right_to_left_diagnostic_feature_alignment: RationalMatrix
    left_readout_matrix: RationalMatrix
    right_readout_matrix: RationalMatrix
    right_readout_aligned_matrix: RationalMatrix
    exact_linear_readout_equation_holds: bool
    event_token_correspondence: tuple[tuple[str, str], ...]
    restricted_graph_witness_sha256: str
    state_row_count: int
    restricted_legal_edge_count: int
    event_map_count: int
    one_global_gauge_used: bool
    every_state_row_transforms: bool
    every_categorical_readout_corresponds: bool
    every_restricted_graph_edge_transforms: bool
    arbitrary_off_domain_fill_compared: bool
    total_operator_compared: bool
    controller_supplied_state_event_correspondence: bool
    correspondence_learned_by_opaque_estimator: bool
    correspondence_used_only_postfit: bool
    certificate_sha256: str
    schema: str = _SIMILARITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SIMILARITY_SCHEMA:
            raise ValueError("unknown partial-operator similarity schema")
        for name, digest in (
            ("left_environment_result_sha256", self.left_environment_result_sha256),
            ("right_environment_result_sha256", self.right_environment_result_sha256),
            ("left_model_sha256", self.left_model_sha256),
            ("right_model_sha256", self.right_model_sha256),
            ("restricted_graph_witness_sha256", self.restricted_graph_witness_sha256),
        ):
            _require_sha256(name, digest)
        if self.environment_kind not in (
            EnvironmentKind.FULL_SUPPORT_CONTROL,
            EnvironmentKind.ROTATED_OMISSION,
        ):
            raise ValueError("similarity environment kind mismatch")
        if self.environment_kind is EnvironmentKind.FULL_SUPPORT_CONTROL:
            if self.paired_omitted_cell is not None:
                raise ValueError("full-control similarity cannot name an omitted cell")
        elif self.paired_omitted_cell not in ((0, 0), (0, 1), (1, 0), (1, 1)):
            raise ValueError("omission similarity requires a K=2,V=2 cell")
        rows, columns = _matrix_shape(self.global_change_of_basis)
        inverse_rows, inverse_columns = _matrix_shape(self.inverse_change_of_basis)
        if (rows, columns, inverse_rows, inverse_columns) != (5, 5, 5, 5):
            raise ValueError("similarity gauge and inverse must be 5x5")
        identity = _matrix(
            tuple(Rational(int(row == column)) for column in range(5))
            for row in range(5)
        )
        if _matmul(self.global_change_of_basis, self.inverse_change_of_basis) != identity:
            raise ValueError("similarity gauge inverse mismatch")
        if _matmul(self.inverse_change_of_basis, self.global_change_of_basis) != identity:
            raise ValueError("similarity inverse is not two-sided")
        if len(self.canonical_state_coordinate_pairs) != _STATE_COUNT:
            raise ValueError("similarity must bind all nine state rows")
        for left, right in self.canonical_state_coordinate_pairs:
            if _row_times_matrix(left, self.global_change_of_basis) != right:
                raise ValueError("a canonical state row violates the single global gauge")
        if (
            len(self.gauge_fit_state_indices) != _FULL_RANK
            or len(self.disjoint_test_state_indices) != _STATE_COUNT - _FULL_RANK
        ):
            raise ValueError("similarity must expose an exact 5-fit/4-test partition")
        if set(self.gauge_fit_state_indices).intersection(self.disjoint_test_state_indices):
            raise ValueError("similarity gauge fit and test states must be disjoint")
        if set(self.gauge_fit_state_indices + self.disjoint_test_state_indices) != set(range(_STATE_COUNT)):
            raise ValueError("similarity fit/test state indices must partition all nine states")
        if self.gauge_fit_state_count != _FULL_RANK or self.disjoint_test_state_count != 4:
            raise ValueError("similarity fit/test counts must be exactly five/four")
        fit_rows = tuple(
            self.canonical_state_coordinate_pairs[index]
            for index in self.gauge_fit_state_indices
        )
        test_rows = tuple(
            self.canonical_state_coordinate_pairs[index]
            for index in self.disjoint_test_state_indices
        )
        if _require_sha256("gauge_fit_rows_sha256", self.gauge_fit_rows_sha256) != _sha256(fit_rows):
            raise ValueError("similarity gauge-fit row digest mismatch")
        if _require_sha256("disjoint_test_rows_sha256", self.disjoint_test_rows_sha256) != _sha256(test_rows):
            raise ValueError("similarity disjoint-test row digest mismatch")
        if any(_row_times_matrix(left, self.global_change_of_basis) != right for left, right in test_rows):
            raise ValueError("a disjoint test state violates the fitted global gauge")
        left_fit_matrix = _matrix(row[0] for row in fit_rows)
        right_fit_matrix = _matrix(row[1] for row in fit_rows)
        if _rank_profile(left_fit_matrix)[0] != _FULL_RANK:
            raise ValueError("five gauge-fit left rows must be independent")
        derived_gauge = _matmul(_inverse(left_fit_matrix), right_fit_matrix)
        if self.global_change_of_basis != derived_gauge:
            raise ValueError("global gauge must be derived only from the five fit rows")
        if len(self.categorical_readout_correspondence) != _STATE_COUNT:
            raise ValueError("similarity must bind all nine categorical readouts")
        for left, right in self.categorical_readout_correspondence:
            _validate_answers("left similarity readout", left)
            _validate_answers("right similarity readout", right)
        if len(self.canonical_diagnostic_feature_pairs) != _STATE_COUNT:
            raise ValueError("similarity must bind all nine diagnostic feature pairs")
        for left_feature, right_feature in self.canonical_diagnostic_feature_pairs:
            if _row_times_matrix(
                right_feature,
                self.right_to_left_diagnostic_feature_alignment,
            ) != left_feature:
                raise ValueError("right diagnostic feature does not align to left gauge")
        for matrix in (
            self.right_to_left_diagnostic_feature_alignment,
            self.left_readout_matrix,
            self.right_readout_matrix,
            self.right_readout_aligned_matrix,
        ):
            if _matrix_shape(matrix) != (5, 5):
                raise ValueError("linear readout witnesses must be 5x5")
        expected_aligned = _matmul(
            self.right_readout_matrix,
            self.right_to_left_diagnostic_feature_alignment,
        )
        if self.right_readout_aligned_matrix != expected_aligned:
            raise ValueError("aligned right readout matrix mismatch")
        if _matmul(
            self.global_change_of_basis,
            self.right_readout_aligned_matrix,
        ) != self.left_readout_matrix:
            raise ValueError("exact G O_R(aligned) = O_L readout equation fails")
        if not _require_bool(
            "exact_linear_readout_equation_holds",
            self.exact_linear_readout_equation_holds,
        ):
            raise ValueError("exact linear readout equation must hold")
        if len(self.event_token_correspondence) != _EVENT_COUNT:
            raise ValueError("similarity must bind all ten event maps")
        if len({left for left, _ in self.event_token_correspondence}) != _EVENT_COUNT:
            raise ValueError("left event-token correspondence is not bijective")
        if len({right for _, right in self.event_token_correspondence}) != _EVENT_COUNT:
            raise ValueError("right event-token correspondence is not bijective")
        for left, right in self.event_token_correspondence:
            _require_token("left similarity event token", left)
            _require_token("right similarity event token", right)
        if (
            self.state_row_count != _STATE_COUNT
            or self.restricted_legal_edge_count != _LEGAL_EDGE_COUNT
            or self.event_map_count != _EVENT_COUNT
        ):
            raise ValueError("similarity census mismatch")
        for name, value, required in (
            ("one_global_gauge_used", self.one_global_gauge_used, True),
            ("every_state_row_transforms", self.every_state_row_transforms, True),
            ("every_categorical_readout_corresponds", self.every_categorical_readout_corresponds, True),
            ("every_restricted_graph_edge_transforms", self.every_restricted_graph_edge_transforms, True),
            ("arbitrary_off_domain_fill_compared", self.arbitrary_off_domain_fill_compared, False),
            ("total_operator_compared", self.total_operator_compared, False),
            ("controller_supplied_state_event_correspondence", self.controller_supplied_state_event_correspondence, True),
            ("correspondence_learned_by_opaque_estimator", self.correspondence_learned_by_opaque_estimator, False),
            ("correspondence_used_only_postfit", self.correspondence_used_only_postfit, True),
        ):
            if _require_bool(name, value) is not required:
                raise ValueError(f"{name} must be {required}")
        expected = _sha256(self._payload(include_certificate_sha=False))
        if _require_sha256("certificate_sha256", self.certificate_sha256) != expected:
            raise ValueError("partial similarity certificate digest mismatch")

    def _payload(self, *, include_certificate_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "left_environment_result_sha256": self.left_environment_result_sha256,
            "right_environment_result_sha256": self.right_environment_result_sha256,
            "left_model_sha256": self.left_model_sha256,
            "right_model_sha256": self.right_model_sha256,
            "environment_kind": self.environment_kind,
            "paired_omitted_cell": self.paired_omitted_cell,
            "global_change_of_basis": self.global_change_of_basis,
            "inverse_change_of_basis": self.inverse_change_of_basis,
            "canonical_state_coordinate_pairs": self.canonical_state_coordinate_pairs,
            "gauge_fit_state_indices": self.gauge_fit_state_indices,
            "disjoint_test_state_indices": self.disjoint_test_state_indices,
            "gauge_fit_state_count": self.gauge_fit_state_count,
            "disjoint_test_state_count": self.disjoint_test_state_count,
            "gauge_fit_rows_sha256": self.gauge_fit_rows_sha256,
            "disjoint_test_rows_sha256": self.disjoint_test_rows_sha256,
            "categorical_readout_correspondence": self.categorical_readout_correspondence,
            "canonical_diagnostic_feature_pairs": self.canonical_diagnostic_feature_pairs,
            "right_to_left_diagnostic_feature_alignment": self.right_to_left_diagnostic_feature_alignment,
            "left_readout_matrix": self.left_readout_matrix,
            "right_readout_matrix": self.right_readout_matrix,
            "right_readout_aligned_matrix": self.right_readout_aligned_matrix,
            "exact_linear_readout_equation_holds": self.exact_linear_readout_equation_holds,
            "event_token_correspondence": self.event_token_correspondence,
            "restricted_graph_witness_sha256": self.restricted_graph_witness_sha256,
            "state_row_count": self.state_row_count,
            "restricted_legal_edge_count": self.restricted_legal_edge_count,
            "event_map_count": self.event_map_count,
            "one_global_gauge_used": self.one_global_gauge_used,
            "every_state_row_transforms": self.every_state_row_transforms,
            "every_categorical_readout_corresponds": self.every_categorical_readout_corresponds,
            "every_restricted_graph_edge_transforms": self.every_restricted_graph_edge_transforms,
            "arbitrary_off_domain_fill_compared": self.arbitrary_off_domain_fill_compared,
            "total_operator_compared": self.total_operator_compared,
            "controller_supplied_state_event_correspondence": self.controller_supplied_state_event_correspondence,
            "correspondence_learned_by_opaque_estimator": self.correspondence_learned_by_opaque_estimator,
            "correspondence_used_only_postfit": self.correspondence_used_only_postfit,
        }
        if include_certificate_sha:
            payload["certificate_sha256"] = self.certificate_sha256
        return payload


def _build_similarity_certificate(
    left: ToyPartialEnvironmentResult,
    right: ToyPartialEnvironmentResult,
) -> PartialOperatorSimilarityCertificate:
    if type(left) is not ToyPartialEnvironmentResult or type(right) is not ToyPartialEnvironmentResult:
        raise TypeError("similarity inputs must be exact environment results")
    if left.controller.kind is not right.controller.kind:
        raise ValueError("similarity pair must use the same environment kind")
    if left.controller.pseudoheldout_cell != right.controller.pseudoheldout_cell:
        raise ValueError("omission similarity pair must use the same omitted cell")
    if (left.controller.relabel_block, right.controller.relabel_block) != (0, 1):
        raise ValueError("similarity pair must be the frozen block-0/block-1 pair")
    left_answers = dict(left.controller.canonical_state_answers)
    right_answers = dict(right.controller.canonical_state_answers)
    state_pairs = tuple(
        (
            left.final_model.answers_to_coordinates(left_answers[state]),
            right.final_model.answers_to_coordinates(right_answers[state]),
        )
        for state in _STATES
    )
    left_state_matrix = _matrix(row[0] for row in state_pairs)
    rank, pivot_rows, _ = _rank_profile(left_state_matrix)
    if rank != _FULL_RANK:
        raise ValueError("left state rows do not span the rank-five realization")
    left_basis = _matrix(state_pairs[index][0] for index in pivot_rows)
    right_basis = _matrix(state_pairs[index][1] for index in pivot_rows)
    if _rank_profile(right_basis)[0] != _FULL_RANK:
        raise ValueError("paired right state rows do not form a common basis")
    gauge = _matmul(_inverse(left_basis), right_basis)
    inverse_gauge = _inverse(gauge)
    if any(_row_times_matrix(left_row, gauge) != right_row for left_row, right_row in state_pairs):
        raise ValueError("no single global state-row similarity exists")
    left_event = dict(left.controller.canonical_event_tokens)
    right_event = dict(right.controller.canonical_event_tokens)
    graph_rows: list[dict[str, object]] = []
    for source, action, target in _LEGAL_EDGES:
        left_source = left.final_model.answers_to_coordinates(left_answers[source])
        right_source = right.final_model.answers_to_coordinates(right_answers[source])
        left_target = left.final_model.answers_to_coordinates(left_answers[target])
        right_target = right.final_model.answers_to_coordinates(right_answers[target])
        predicted_left = left.final_model.apply_event(left_source, left_event[action])
        predicted_right = right.final_model.apply_event(right_source, right_event[action])
        if predicted_left != left_target or predicted_right != right_target:
            raise ValueError("a learned restricted graph edge is incorrect")
        if _row_times_matrix(predicted_left, gauge) != predicted_right:
            raise ValueError("a restricted graph edge violates the global gauge")
        graph_rows.append(
            {
                "left_source": left_source,
                "right_source": right_source,
                "left_event_token": left_event[action],
                "right_event_token": right_event[action],
                "left_target": left_target,
                "right_target": right_target,
            }
        )
    event_correspondence = tuple(
        (left_event[action], right_event[action]) for action in _ACTIONS
    )
    readout_correspondence = tuple(
        (left_answers[state], right_answers[state]) for state in _STATES
    )
    diagnostic_feature_pairs = tuple(
        (
            _diagnostic_row(left_answers[state], left.final_model.answer_tokens),
            _diagnostic_row(right_answers[state], right.final_model.answer_tokens),
        )
        for state in _STATES
    )
    right_feature_basis = _matrix(
        diagnostic_feature_pairs[index][1] for index in pivot_rows
    )
    left_feature_basis = _matrix(
        diagnostic_feature_pairs[index][0] for index in pivot_rows
    )
    feature_alignment = _matmul(_inverse(right_feature_basis), left_feature_basis)
    if any(
        _row_times_matrix(right_row, feature_alignment) != left_row
        for left_row, right_row in diagnostic_feature_pairs
    ):
        raise ValueError("categorical diagnostic features do not admit one alignment")
    right_readout_aligned = _matmul(
        right.final_model.basis_diagnostic_rows,
        feature_alignment,
    )
    if _matmul(gauge, right_readout_aligned) != left.final_model.basis_diagnostic_rows:
        raise ValueError("global state gauge and diagnostic readouts are inconsistent")
    test_rows = tuple(index for index in range(_STATE_COUNT) if index not in pivot_rows)
    kwargs = {
        "left_environment_result_sha256": left.result_sha256,
        "right_environment_result_sha256": right.result_sha256,
        "left_model_sha256": left.final_model.model_sha256,
        "right_model_sha256": right.final_model.model_sha256,
        "environment_kind": left.controller.kind,
        "paired_omitted_cell": left.controller.pseudoheldout_cell,
        "global_change_of_basis": gauge,
        "inverse_change_of_basis": inverse_gauge,
        "canonical_state_coordinate_pairs": state_pairs,
        "gauge_fit_state_indices": pivot_rows,
        "disjoint_test_state_indices": test_rows,
        "gauge_fit_state_count": _FULL_RANK,
        "disjoint_test_state_count": 4,
        "gauge_fit_rows_sha256": _sha256(tuple(state_pairs[index] for index in pivot_rows)),
        "disjoint_test_rows_sha256": _sha256(tuple(state_pairs[index] for index in test_rows)),
        "categorical_readout_correspondence": readout_correspondence,
        "canonical_diagnostic_feature_pairs": diagnostic_feature_pairs,
        "right_to_left_diagnostic_feature_alignment": feature_alignment,
        "left_readout_matrix": left.final_model.basis_diagnostic_rows,
        "right_readout_matrix": right.final_model.basis_diagnostic_rows,
        "right_readout_aligned_matrix": right_readout_aligned,
        "exact_linear_readout_equation_holds": True,
        "event_token_correspondence": event_correspondence,
        "restricted_graph_witness_sha256": _sha256(graph_rows),
        "state_row_count": _STATE_COUNT,
        "restricted_legal_edge_count": _LEGAL_EDGE_COUNT,
        "event_map_count": _EVENT_COUNT,
        "one_global_gauge_used": True,
        "every_state_row_transforms": True,
        "every_categorical_readout_corresponds": True,
        "every_restricted_graph_edge_transforms": True,
        "arbitrary_off_domain_fill_compared": False,
        "total_operator_compared": False,
        "controller_supplied_state_event_correspondence": True,
        "correspondence_learned_by_opaque_estimator": False,
        "correspondence_used_only_postfit": True,
        "schema": _SIMILARITY_SCHEMA,
    }
    payload = {"schema": _SIMILARITY_SCHEMA, **{key: value for key, value in kwargs.items() if key != "schema"}}
    return PartialOperatorSimilarityCertificate(
        **kwargs,
        certificate_sha256=_sha256(payload),
    )


@dataclass(frozen=True)
class ToyPartialOperatorReport:
    scope: PartialOperatorScope
    status: PartialOperatorStatus
    budgets: PartialOperatorBudgets
    controller_nonces: tuple[str, ...]
    environments: tuple[ToyPartialEnvironmentResult, ...]
    similarities: tuple[PartialOperatorSimilarityCertificate, ...]
    environment_count: int
    full_support_control_count: int
    rotated_omission_count: int
    passive_edge_total: int
    active_response_total: int
    active_returned_categorical_token_total: int
    sealed_edge_total: int
    long_path_total: int
    sealed_program_total: int
    undefined_edge_rejection_total: int
    similarity_pair_count: int
    passive_rank_sequence: tuple[int, ...]
    postactive_rank_sequence: tuple[int, ...]
    aggregate_total_extension_nullity_per_environment: tuple[int, ...]
    all_environment_behavior_passed: bool
    supplied_full_product_state_grammar: bool
    supplied_semantic_active_excitation_basis: bool
    active_basis_learned_or_selected_by_estimator: bool
    exact_partial_legal_domain_operator_claim: bool
    total_wfa_operator_claim: bool
    assumption_free_representation_discovery_claim: bool
    learner_boundaries_use_only_opaque_inputs: bool
    trusted_controller_and_learner_are_process_isolated: bool
    contextual_nested_certificates_reconstructed_by_report: bool
    similarity_correspondences_supplied_by_trusted_controller: bool
    similarity_correspondences_learned_by_opaque_estimator: bool
    report_sha256: str
    schema: str = _REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _REPORT_SCHEMA:
            raise ValueError("unknown toy partial-operator report schema")
        if self.scope is not PartialOperatorScope.GUARDED_ABSENCE_AWARE_PARTIAL_OPERATORS:
            raise ValueError("report scope mismatch")
        if self.status is not PartialOperatorStatus.SYNTHETIC_PROTOCOL_IMPLEMENTATION_REHEARSAL:
            raise ValueError("report status mismatch")
        if type(self.budgets) is not PartialOperatorBudgets:
            raise TypeError("report budgets must be exact PartialOperatorBudgets")
        if len(self.controller_nonces) != _ENVIRONMENT_COUNT:
            raise ValueError("report must bind exactly ten controller nonces")
        for nonce in self.controller_nonces:
            _require_nonce(nonce)
        if len(set(self.controller_nonces)) != _ENVIRONMENT_COUNT:
            raise ValueError("report controller nonces must be distinct")
        if len(self.environments) != _ENVIRONMENT_COUNT or self.environment_count != _ENVIRONMENT_COUNT:
            raise ValueError("report must contain exactly ten environments")
        for result in self.environments:
            if type(result) is not ToyPartialEnvironmentResult:
                raise TypeError("environment rows must be exact")
        expected_schedule = (
            (EnvironmentKind.FULL_SUPPORT_CONTROL, None, 0),
            (EnvironmentKind.FULL_SUPPORT_CONTROL, None, 1),
        ) + tuple(
            (EnvironmentKind.ROTATED_OMISSION, cell, block)
            for cell in ((0, 0), (0, 1), (1, 0), (1, 1))
            for block in (0, 1)
        )
        actual_schedule = tuple(
            (
                result.controller.kind,
                result.controller.pseudoheldout_cell,
                result.controller.relabel_block,
            )
            for result in self.environments
        )
        if actual_schedule != expected_schedule:
            raise ValueError("environment schedule must be 2 controls plus 4x2 omissions")
        if tuple(row.controller.trusted_controller_nonce for row in self.environments) != self.controller_nonces:
            raise ValueError("environment controllers do not match the report nonce commitment")
        rebuilt_controllers: list[ToyPartialControllerEnvironment] = []
        for index, (kind, cell, block) in enumerate(expected_schedule):
            if kind is EnvironmentKind.FULL_SUPPORT_CONTROL:
                rebuilt = build_full_support_controller_environment(
                    block,
                    controller_nonce=self.controller_nonces[index],
                    budgets=self.budgets,
                )
            else:
                assert cell is not None
                rebuilt = build_omission_controller_environment(
                    cell,
                    block,
                    controller_nonce=self.controller_nonces[index],
                    budgets=self.budgets,
                )
            rebuilt_controllers.append(rebuilt)
        if tuple(rebuilt_controllers) != tuple(row.controller for row in self.environments):
            raise ValueError("controller material fails exact schedule/nonce/budget reconstruction")
        # Only after the cheap schedule/nonce/controller firewall succeeds do
        # we replay fits, active answers, and post-fit evaluations.
        for result in self.environments:
            reconstructed = run_toy_partial_environment(result.controller)
            if result != reconstructed:
                raise ValueError("report environment fails authoritative reconstruction")
        expected_similarities = (
            _build_similarity_certificate(self.environments[0], self.environments[1]),
        ) + tuple(
            _build_similarity_certificate(
                self.environments[2 + 2 * index],
                self.environments[3 + 2 * index],
            )
            for index in range(4)
        )
        if self.similarities != expected_similarities:
            raise ValueError("report must contain exactly the five reconstructed block similarities")
        expected_scalars = {
            "full_support_control_count": 2,
            "rotated_omission_count": 8,
            "passive_edge_total": 2 * 44 + 8 * 21,
            "active_response_total": 8 * 15,
            "active_returned_categorical_token_total": 8 * 30,
            "sealed_edge_total": 8 * 8,
            "long_path_total": 10 * 12,
            "sealed_program_total": 2 * 12 + 8 * 20,
            "undefined_edge_rejection_total": 10 * 46,
            "similarity_pair_count": 5,
        }
        for name, expected in expected_scalars.items():
            if getattr(self, name) != expected:
                raise ValueError(f"report aggregate {name} mismatch")
        expected_passive_ranks = (5, 5) + (4,) * 8
        expected_postactive_ranks = (5,) * 10
        if self.passive_rank_sequence != expected_passive_ranks:
            raise ValueError("report passive-rank sequence mismatch")
        if self.postactive_rank_sequence != expected_postactive_ranks:
            raise ValueError("report postactive-rank sequence mismatch")
        if self.aggregate_total_extension_nullity_per_environment != (80,) * 10:
            raise ValueError("every environment must retain total-extension nullity 80")
        for name, value, required in (
            ("all_environment_behavior_passed", self.all_environment_behavior_passed, True),
            ("supplied_full_product_state_grammar", self.supplied_full_product_state_grammar, True),
            ("supplied_semantic_active_excitation_basis", self.supplied_semantic_active_excitation_basis, True),
            ("active_basis_learned_or_selected_by_estimator", self.active_basis_learned_or_selected_by_estimator, False),
            ("exact_partial_legal_domain_operator_claim", self.exact_partial_legal_domain_operator_claim, True),
            ("total_wfa_operator_claim", self.total_wfa_operator_claim, False),
            ("assumption_free_representation_discovery_claim", self.assumption_free_representation_discovery_claim, False),
            ("learner_boundaries_use_only_opaque_inputs", self.learner_boundaries_use_only_opaque_inputs, True),
            ("trusted_controller_and_learner_are_process_isolated", self.trusted_controller_and_learner_are_process_isolated, False),
            ("contextual_nested_certificates_reconstructed_by_report", self.contextual_nested_certificates_reconstructed_by_report, True),
            ("similarity_correspondences_supplied_by_trusted_controller", self.similarity_correspondences_supplied_by_trusted_controller, True),
            ("similarity_correspondences_learned_by_opaque_estimator", self.similarity_correspondences_learned_by_opaque_estimator, False),
        ):
            if _require_bool(name, value) is not required:
                raise ValueError(f"{name} must be {required}")
        expected = _sha256(self._payload(include_report_sha=False))
        if _require_sha256("report_sha256", self.report_sha256) != expected:
            raise ValueError("toy partial-operator report digest mismatch")

    def _payload(self, *, include_report_sha: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "scope": self.scope,
            "status": self.status,
            "budgets": self.budgets.payload(),
            "controller_nonces": self.controller_nonces,
            "environments": [row._payload(include_result_sha=True) for row in self.environments],
            "similarities": [row._payload(include_certificate_sha=True) for row in self.similarities],
        }
        for name in (
            "environment_count",
            "full_support_control_count",
            "rotated_omission_count",
            "passive_edge_total",
            "active_response_total",
            "active_returned_categorical_token_total",
            "sealed_edge_total",
            "long_path_total",
            "sealed_program_total",
            "undefined_edge_rejection_total",
            "similarity_pair_count",
            "passive_rank_sequence",
            "postactive_rank_sequence",
            "aggregate_total_extension_nullity_per_environment",
            "all_environment_behavior_passed",
            "supplied_full_product_state_grammar",
            "supplied_semantic_active_excitation_basis",
            "active_basis_learned_or_selected_by_estimator",
            "exact_partial_legal_domain_operator_claim",
            "total_wfa_operator_claim",
            "assumption_free_representation_discovery_claim",
            "learner_boundaries_use_only_opaque_inputs",
            "trusted_controller_and_learner_are_process_isolated",
            "contextual_nested_certificates_reconstructed_by_report",
            "similarity_correspondences_supplied_by_trusted_controller",
            "similarity_correspondences_learned_by_opaque_estimator",
        ):
            payload[name] = getattr(self, name)
        if include_report_sha:
            payload["report_sha256"] = self.report_sha256
        return payload


def run_toy_partial_operator_experiment(
    *,
    controller_nonces: Sequence[str],
    budgets: PartialOperatorBudgets | None = None,
) -> ToyPartialOperatorReport:
    if not isinstance(controller_nonces, (tuple, list)) or len(controller_nonces) != _ENVIRONMENT_COUNT:
        raise ValueError("exactly ten controller nonces are required")
    nonces = tuple(_require_nonce(value) for value in controller_nonces)
    if len(set(nonces)) != _ENVIRONMENT_COUNT:
        raise ValueError("controller nonces must be distinct")
    selected_budgets = PartialOperatorBudgets() if budgets is None else budgets
    _preflight_controller_budgets(selected_budgets, omission=True)
    controllers: list[ToyPartialControllerEnvironment] = [
        build_full_support_controller_environment(
            block,
            controller_nonce=nonces[block],
            budgets=selected_budgets,
        )
        for block in (0, 1)
    ]
    nonce_index = 2
    for cell in ((0, 0), (0, 1), (1, 0), (1, 1)):
        for block in (0, 1):
            controllers.append(
                build_omission_controller_environment(
                    cell,
                    block,
                    controller_nonce=nonces[nonce_index],
                    budgets=selected_budgets,
                )
            )
            nonce_index += 1
    environments = tuple(run_toy_partial_environment(row) for row in controllers)
    similarities = (
        _build_similarity_certificate(environments[0], environments[1]),
    ) + tuple(
        _build_similarity_certificate(
            environments[2 + 2 * index], environments[3 + 2 * index]
        )
        for index in range(4)
    )
    passive_ranks = tuple(row.passive.passive_rank_certificate.rank for row in environments)
    postactive_ranks = tuple(row.final_model.ambient_rank for row in environments)
    extension_nullities = tuple(
        sum(operator.total_extension_nullity for operator in row.final_model.operator_certificates)
        for row in environments
    )
    kwargs = {
        "scope": PartialOperatorScope.GUARDED_ABSENCE_AWARE_PARTIAL_OPERATORS,
        "status": PartialOperatorStatus.SYNTHETIC_PROTOCOL_IMPLEMENTATION_REHEARSAL,
        "budgets": selected_budgets,
        "controller_nonces": nonces,
        "environments": environments,
        "similarities": similarities,
        "environment_count": 10,
        "full_support_control_count": 2,
        "rotated_omission_count": 8,
        "passive_edge_total": 2 * 44 + 8 * 21,
        "active_response_total": 8 * 15,
        "active_returned_categorical_token_total": 8 * 30,
        "sealed_edge_total": 8 * 8,
        "long_path_total": 10 * 12,
        "sealed_program_total": 2 * 12 + 8 * 20,
        "undefined_edge_rejection_total": 10 * 46,
        "similarity_pair_count": 5,
        "passive_rank_sequence": passive_ranks,
        "postactive_rank_sequence": postactive_ranks,
        "aggregate_total_extension_nullity_per_environment": extension_nullities,
        "all_environment_behavior_passed": all(row.model_behavior_passed for row in environments),
        "supplied_full_product_state_grammar": True,
        "supplied_semantic_active_excitation_basis": True,
        "active_basis_learned_or_selected_by_estimator": False,
        "exact_partial_legal_domain_operator_claim": True,
        "total_wfa_operator_claim": False,
        "assumption_free_representation_discovery_claim": False,
        "learner_boundaries_use_only_opaque_inputs": True,
        "trusted_controller_and_learner_are_process_isolated": False,
        "contextual_nested_certificates_reconstructed_by_report": True,
        "similarity_correspondences_supplied_by_trusted_controller": True,
        "similarity_correspondences_learned_by_opaque_estimator": False,
        "schema": _REPORT_SCHEMA,
    }
    payload = {
        "schema": _REPORT_SCHEMA,
        "scope": kwargs["scope"],
        "status": kwargs["status"],
        "budgets": selected_budgets.payload(),
        "controller_nonces": nonces,
        "environments": [row._payload(include_result_sha=True) for row in environments],
        "similarities": [row._payload(include_certificate_sha=True) for row in similarities],
    }
    for name in (
        "environment_count",
        "full_support_control_count",
        "rotated_omission_count",
        "passive_edge_total",
        "active_response_total",
        "active_returned_categorical_token_total",
        "sealed_edge_total",
        "long_path_total",
        "sealed_program_total",
        "undefined_edge_rejection_total",
        "similarity_pair_count",
        "passive_rank_sequence",
        "postactive_rank_sequence",
        "aggregate_total_extension_nullity_per_environment",
        "all_environment_behavior_passed",
        "supplied_full_product_state_grammar",
        "supplied_semantic_active_excitation_basis",
        "active_basis_learned_or_selected_by_estimator",
        "exact_partial_legal_domain_operator_claim",
        "total_wfa_operator_claim",
        "assumption_free_representation_discovery_claim",
        "learner_boundaries_use_only_opaque_inputs",
        "trusted_controller_and_learner_are_process_isolated",
        "contextual_nested_certificates_reconstructed_by_report",
        "similarity_correspondences_supplied_by_trusted_controller",
        "similarity_correspondences_learned_by_opaque_estimator",
    ):
        payload[name] = kwargs[name]
    return ToyPartialOperatorReport(
        **kwargs,
        report_sha256=_sha256(payload),
    )


__all__ = (
    "ActiveAcquisitionCommitment",
    "ActivePartialDiscovery",
    "CompatibleOmissionHypothesisCertificate",
    "EnvironmentKind",
    "ExactPartialOperatorCertificate",
    "ExactPartialRealization",
    "ExactRankCertificate",
    "OpaqueEdgeObservation",
    "OpaqueEdgeRequest",
    "OpaqueMembershipResponse",
    "OpaqueOmissionHypothesisWitness",
    "OpaquePartialOperatorLimitError",
    "OpaqueSealedProgram",
    "OpaqueStateObservation",
    "OneResponseCheckpoint",
    "OneResponseEventDeficit",
    "ObservedEventRank",
    "PartialOperatorBudgets",
    "PartialOperatorLearnerInput",
    "PartialOperatorScope",
    "PartialOperatorSimilarityCertificate",
    "PartialOperatorStatus",
    "PassivePartialDiscovery",
    "PathRelationEvaluation",
    "SealedPartialEvaluation",
    "SealedProgramPrediction",
    "Rational",
    "ToyPartialControllerEnvironment",
    "ToyPartialEnvironmentResult",
    "ToyPartialOperatorReport",
    "UndefinedDomainRejectionCertificate",
    "UndefinedEdgeRejection",
    "analyze_one_response_checkpoint",
    "build_full_support_controller_environment",
    "build_omission_controller_environment",
    "enumerate_publicly_compatible_omission_hypotheses",
    "fit_active_partial_operators",
    "fit_passive_partial_operators",
    "publicly_compatible_omission_hypothesis_count",
    "release_first_active_response",
    "release_remaining_active_responses",
    "run_toy_partial_environment",
    "run_toy_partial_operator_experiment",
)
