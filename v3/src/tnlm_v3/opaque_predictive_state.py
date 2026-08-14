"""Exact opaque-symbol discovery for the smallest absence-diagnostic block.

This module implements the bounded ``K=2, V=2, M=2`` debugging experiment
from the Phase-III protocol.  Its claim is intentionally narrow: it identifies
the rank of the finite, zero-suffix, multi-output absence-diagnostic Hankel
block.  It does not identify arbitrary-suffix transition operators, a full
weighted automaton, a tensor factorization, or a geometry of language.

The learner-facing value is :class:`OpaqueDiagnosticInput`.  It contains only
opaque atomic event, query, and answer identifiers; opaque words; observed
answer tuples; a domain table; a frozen candidate order; and budgets.  It has
no key/value coordinates, event arguments, canonical states, omitted-cell
identifier, semantic mapping, or executor callback.  A trusted in-process
controller exists in this same small implementation to generate and score the
synthetic control.  That is an argument-boundary rehearsal, not an operating-
system process-isolation certificate.  Relabel identifiers and the semantic-
to-token bijections require explicit controller-supplied 256-bit nonces and
are not derived from the small semantic coordinate domain, the omitted cell,
or the relabel-block index.  The traversal used to serialize commuting multi-
key histories is nonce-keyed as well.  Only sorted opaque alphabets are
exposed; these controller mappings remain outside the learner input.  This is
honest-code/API isolation, not cryptographic protection against an adversary
that can inspect controller memory, source internals, or the Python call stack.
Nested passive, postactive, and similarity digests are context-bound content
links; only environment/report reconstruction is the authoritative semantic
evidence boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from itertools import product
from math import gcd
import hashlib
import json
from typing import Iterable, Sequence


OpaqueWord = tuple[str, ...]
SemanticCell = tuple[int, int]
SemanticState = tuple[int, int]

_INPUT_SCHEMA = "tnlm-v3-toy-opaque-diagnostic-input-v1"
_RANK_SCHEMA = "tnlm-v3-exact-rank-certificate-v1"
_MODEL_SCHEMA = "tnlm-v3-exact-diagnostic-model-v1"
_COVERAGE_SCHEMA = "tnlm-v3-opaque-vocabulary-coverage-v1"
_COMPLETION_SCHEMA = "tnlm-v3-passive-completion-witness-v1"
_COMMITMENT_SCHEMA = "tnlm-v3-opaque-acquisition-commitment-v1"
_PASSIVE_SCHEMA = "tnlm-v3-toy-opaque-passive-result-v1"
_ANSWER_SCHEMA = "tnlm-v3-opaque-membership-answer-v1"
_COORDINATE_SCHEMA = "tnlm-v3-full-diagnostic-coordinate-v1"
_POSTACTIVE_SCHEMA = "tnlm-v3-toy-opaque-postactive-result-v1"
_ENVIRONMENT_SCHEMA = "tnlm-v3-toy-opaque-environment-result-v1"
_SIMILARITY_SCHEMA = "tnlm-v3-toy-opaque-similarity-v1"
_REPORT_SCHEMA = "tnlm-v3-toy-opaque-hankel-report-v1"

_TOY_EVENT_COUNT = 4
_TOY_QUERY_COUNT = 2
_TOY_ANSWER_COUNT = 3
_TOY_PASSIVE_ROW_COUNT = 6
_TOY_FULL_ROW_COUNT = 9
_TOY_PASSIVE_RANK = 4
_TOY_FULL_RANK = 5
_TOY_RELABEL_BLOCKS = 2

_HARD_MAX_WORD_LENGTH = 16
_HARD_MAX_SUFFIX_TEST_CANDIDATES = 64
_HARD_MAX_ORACLE_EVALUATIONS = 50_000
_HARD_MAX_ACTIVE_CANDIDATE_WORDS = 4_096
_HARD_MAX_ACTIVE_RESPONSES = 1
_HARD_MAX_BASIS_DIMENSION = 10
_HARD_MAX_COMPLETE_DIAGNOSTIC_ROWS = 9

_SEMANTIC_CELLS: tuple[SemanticCell, ...] = (
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
)
_FULL_SEMANTIC_STATES: tuple[SemanticState, ...] = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (-1, 1),
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
)
_SIMILARITY_FIT_STATES = _FULL_SEMANTIC_STATES[:5]
_SIMILARITY_TEST_STATES = _FULL_SEMANTIC_STATES[5:]


class OpaquePredictiveStateLimitError(RuntimeError):
    """Raised before a frozen toy-experiment resource limit is exceeded."""


class OpaqueDiagnosticScope(str, Enum):
    """Closed scientific scope for this deliberately smallest experiment."""

    ZERO_SUFFIX_ABSENCE_DIAGNOSTIC_BLOCK = (
        "zero_suffix_multioutput_absence_diagnostic_block_only"
    )


class OpaqueExperimentStatus(str, Enum):
    """Closed protocol status; this module is a synthetic implementation run."""

    SYNTHETIC_PROTOCOL_IMPLEMENTATION_REHEARSAL = (
        "synthetic_protocol_implementation_rehearsal"
    )


class OpaqueVocabularyContract(str, Enum):
    """Supplied nonsemantic coverage promise used by active selection."""

    DISTINCT_NONEMPTY_QUERY_ANSWER_SLOTS = (
        "atomic_tokens_cover_distinct_nonempty_query_answer_slots"
    )


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


def _require_controller_nonce(value: object) -> str:
    """Validate controller-only 256-bit relabel material.

    Entropy is a controller responsibility; the learner never receives this
    value.  Requiring an explicit nonce prevents this module from deriving
    identifiers from the tiny public semantic-coordinate domain.
    """

    return _require_sha256("trusted controller nonce", value)


def _require_opaque_token(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a 128-bit lowercase opaque identifier")
    return value


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
    raise TypeError(f"value of type {type(value).__name__} is not canonically encodable")


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
    """A strict canonical rational scalar suitable for immutable evidence."""

    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise TypeError("rational numerator and denominator must be exact integers")
        if self.denominator <= 0:
            raise ValueError("rational denominator must be positive")
        if gcd(abs(self.numerator), self.denominator) != 1:
            raise ValueError("rational must be in lowest terms")

    @classmethod
    def from_fraction(cls, value: Fraction) -> Rational:
        if type(value) is not Fraction:
            raise TypeError("value must be an exact Fraction")
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
        raise TypeError("matrix must be a nonempty tuple of rows")
    width: int | None = None
    for row in matrix:
        if not isinstance(row, tuple) or not row:
            raise TypeError("matrix rows must be nonempty tuples")
        if any(type(value) is not Rational for value in row):
            raise TypeError("matrix entries must be exact Rational values")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("matrix must be rectangular")
    assert width is not None
    return len(matrix), width


def _fraction_rows(matrix: RationalMatrix) -> list[list[Fraction]]:
    return [[value.as_fraction() for value in row] for row in matrix]


def _rank_profile(matrix: RationalMatrix) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    row_count, column_count = _matrix_shape(matrix)
    work = _fraction_rows(matrix)
    original_rows = list(range(row_count))
    pivot_rows: list[int] = []
    pivot_columns: list[int] = []
    rank = 0
    for column in range(column_count):
        pivot = next(
            (row for row in range(rank, row_count) if work[row][column] != 0),
            None,
        )
        if pivot is None:
            continue
        if pivot != rank:
            work[rank], work[pivot] = work[pivot], work[rank]
            original_rows[rank], original_rows[pivot] = (
                original_rows[pivot],
                original_rows[rank],
            )
        pivot_value = work[rank][column]
        work[rank] = [value / pivot_value for value in work[rank]]
        for row in range(row_count):
            if row == rank:
                continue
            factor = work[row][column]
            if factor != 0:
                work[row] = [
                    value - factor * pivot_entry
                    for value, pivot_entry in zip(work[row], work[rank], strict=True)
                ]
        pivot_rows.append(original_rows[rank])
        pivot_columns.append(column)
        rank += 1
        if rank == row_count:
            break
    return rank, tuple(pivot_rows), tuple(pivot_columns)


def _determinant(matrix: RationalMatrix) -> Rational:
    rows, columns = _matrix_shape(matrix)
    if rows != columns:
        raise ValueError("determinant requires a square matrix")
    work = _fraction_rows(matrix)
    result = Fraction(1)
    sign = 1
    for column in range(columns):
        pivot = next(
            (row for row in range(column, rows) if work[row][column] != 0),
            None,
        )
        if pivot is None:
            return Rational(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            sign *= -1
        pivot_value = work[column][column]
        result *= pivot_value
        for row in range(column + 1, rows):
            if work[row][column] == 0:
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
        row
        + [Fraction(int(row_index == column)) for column in range(columns)]
        for row_index, row in enumerate(_fraction_rows(matrix))
    ]
    for column in range(columns):
        pivot = next(
            (row for row in range(column, rows) if work[row][column] != 0),
            None,
        )
        if pivot is None:
            raise ValueError("matrix is singular")
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
        pivot_value = work[column][column]
        work[column] = [value / pivot_value for value in work[column]]
        for row in range(rows):
            if row == column:
                continue
            factor = work[row][column]
            if factor != 0:
                work[row] = [
                    value - factor * pivot_entry
                    for value, pivot_entry in zip(
                        work[row], work[column], strict=True
                    )
                ]
    return _matrix(row[columns:] for row in work)


def _matmul(left: RationalMatrix, right: RationalMatrix) -> RationalMatrix:
    left_rows, inner = _matrix_shape(left)
    right_rows, right_columns = _matrix_shape(right)
    if inner != right_rows:
        raise ValueError("matrix dimensions do not align")
    left_fraction = _fraction_rows(left)
    right_fraction = _fraction_rows(right)
    return _matrix(
        tuple(
            sum(
                left_fraction[row][index] * right_fraction[index][column]
                for index in range(inner)
            )
            for column in range(right_columns)
        )
        for row in range(left_rows)
    )


def _row_times_matrix(row: RationalVector, matrix: RationalMatrix) -> RationalVector:
    _, columns = _matrix_shape(matrix)
    if len(row) != len(matrix):
        raise ValueError("row and matrix dimensions do not align")
    fractions = [value.as_fraction() for value in row]
    right = _fraction_rows(matrix)
    return tuple(
        Rational.from_fraction(
            sum(fractions[index] * right[index][column] for index in range(len(row)))
        )
        for column in range(columns)
    )


def _submatrix(
    matrix: RationalMatrix,
    row_indices: Sequence[int],
    column_indices: Sequence[int],
) -> RationalMatrix:
    return tuple(
        tuple(matrix[row][column] for column in column_indices) for row in row_indices
    )


@dataclass(frozen=True)
class OpaqueHankelBudgets:
    """Frozen toy caps, each no larger than the Phase-III protocol ceiling."""

    max_word_length: int = _HARD_MAX_WORD_LENGTH
    max_suffix_test_candidates: int = _HARD_MAX_SUFFIX_TEST_CANDIDATES
    max_oracle_evaluations: int = _HARD_MAX_ORACLE_EVALUATIONS
    max_active_candidate_words: int = _HARD_MAX_ACTIVE_CANDIDATE_WORDS
    max_active_responses: int = _HARD_MAX_ACTIVE_RESPONSES
    max_basis_dimension: int = _HARD_MAX_BASIS_DIMENSION
    max_complete_diagnostic_rows: int = _HARD_MAX_COMPLETE_DIAGNOSTIC_ROWS

    def __post_init__(self) -> None:
        for name, hard_max in (
            ("max_word_length", _HARD_MAX_WORD_LENGTH),
            ("max_suffix_test_candidates", _HARD_MAX_SUFFIX_TEST_CANDIDATES),
            ("max_oracle_evaluations", _HARD_MAX_ORACLE_EVALUATIONS),
            ("max_active_candidate_words", _HARD_MAX_ACTIVE_CANDIDATE_WORDS),
            ("max_active_responses", _HARD_MAX_ACTIVE_RESPONSES),
            ("max_basis_dimension", _HARD_MAX_BASIS_DIMENSION),
            ("max_complete_diagnostic_rows", _HARD_MAX_COMPLETE_DIAGNOSTIC_ROWS),
        ):
            value = _plain_int(name, getattr(self, name), 1)
            if value > hard_max:
                raise ValueError(f"{name} exceeds the frozen Phase-III toy ceiling")

    def payload(self) -> dict[str, int]:
        return {
            "max_word_length": self.max_word_length,
            "max_suffix_test_candidates": self.max_suffix_test_candidates,
            "max_oracle_evaluations": self.max_oracle_evaluations,
            "max_active_candidate_words": self.max_active_candidate_words,
            "max_active_responses": self.max_active_responses,
            "max_basis_dimension": self.max_basis_dimension,
            "max_complete_diagnostic_rows": self.max_complete_diagnostic_rows,
        }


@dataclass(frozen=True, order=True)
class OpaqueDiagnosticRow:
    """One opaque history and both opaque terminal-query answers."""

    word: OpaqueWord
    answers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.word, tuple):
            raise TypeError("word must be an exact tuple")
        if any(_require_opaque_token("word token", token) != token for token in self.word):
            raise AssertionError("unreachable token validation failure")
        if not isinstance(self.answers, tuple) or not self.answers:
            raise TypeError("answers must be a nonempty tuple")
        if any(
            _require_opaque_token("answer token", token) != token
            for token in self.answers
        ):
            raise AssertionError("unreachable answer validation failure")


def _row_payload(row: OpaqueDiagnosticRow) -> dict[str, object]:
    return {"word": row.word, "answers": row.answers}


@dataclass(frozen=True)
class OpaqueDiagnosticInput:
    """The complete learner-facing object; deliberately contains no semantics."""

    scope: OpaqueDiagnosticScope
    vocabulary_contract: OpaqueVocabularyContract
    event_tokens: tuple[str, ...]
    query_tokens: tuple[str, ...]
    answer_tokens: tuple[str, ...]
    passive_rows: tuple[OpaqueDiagnosticRow, ...]
    candidate_words: tuple[OpaqueWord, ...]
    budgets: OpaqueHankelBudgets
    passive_table_sha256: str
    candidate_pool_sha256: str
    input_sha256: str
    schema: str = _INPUT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _INPUT_SCHEMA:
            raise ValueError("unknown opaque diagnostic input schema")
        if type(self.scope) is not OpaqueDiagnosticScope:
            raise TypeError("scope must be exact OpaqueDiagnosticScope")
        if self.vocabulary_contract is not (
            OpaqueVocabularyContract.DISTINCT_NONEMPTY_QUERY_ANSWER_SLOTS
        ):
            raise ValueError("unknown opaque atomic-vocabulary coverage contract")
        if type(self.budgets) is not OpaqueHankelBudgets:
            raise TypeError("budgets must be exact OpaqueHankelBudgets")
        for name, values, expected in (
            ("event_tokens", self.event_tokens, _TOY_EVENT_COUNT),
            ("query_tokens", self.query_tokens, _TOY_QUERY_COUNT),
            ("answer_tokens", self.answer_tokens, _TOY_ANSWER_COUNT),
        ):
            if not isinstance(values, tuple) or len(values) != expected:
                raise ValueError(f"{name} must contain exactly {expected} tokens")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
            for token in values:
                _require_opaque_token(name, token)
        all_tokens = self.event_tokens + self.query_tokens + self.answer_tokens
        if len(set(all_tokens)) != len(all_tokens):
            raise ValueError("event, query, and answer token namespaces must be disjoint")
        if not isinstance(self.passive_rows, tuple) or len(self.passive_rows) != 6:
            raise ValueError("toy passive table must contain exactly six rows")
        if any(type(row) is not OpaqueDiagnosticRow for row in self.passive_rows):
            raise TypeError("passive rows must be exact OpaqueDiagnosticRow values")
        if tuple(sorted(self.passive_rows, key=lambda row: row.word)) != self.passive_rows:
            raise ValueError("passive rows must be serialized in word order")
        words = tuple(row.word for row in self.passive_rows)
        if len(set(words)) != len(words):
            raise ValueError("passive words must be unique")
        if () not in words:
            raise ValueError("passive table must contain the empty history")
        if len({row.answers for row in self.passive_rows}) != _TOY_PASSIVE_ROW_COUNT:
            raise ValueError("passive diagnostic answer rows must be unique")
        for row in self.passive_rows:
            if len(row.word) > self.budgets.max_word_length:
                raise OpaquePredictiveStateLimitError("passive word exceeds budget")
            if any(token not in self.event_tokens for token in row.word):
                raise ValueError("passive word contains a token outside the alphabet")
            if len(row.answers) != len(self.query_tokens):
                raise ValueError("passive answers must align to opaque query tokens")
            if any(token not in self.answer_tokens for token in row.answers):
                raise ValueError("passive answer is outside the opaque output alphabet")
        expected_candidates = tuple((token,) for token in self.event_tokens)
        if self.candidate_words != expected_candidates:
            raise ValueError("candidate pool must be all atomic events in byte order")
        if len(self.candidate_words) > self.budgets.max_active_candidate_words:
            raise OpaquePredictiveStateLimitError("candidate pool exceeds budget")
        if any(len(word) > self.budgets.max_word_length for word in self.candidate_words):
            raise OpaquePredictiveStateLimitError("candidate word exceeds budget")
        missing_candidates = tuple(word for word in self.candidate_words if word not in words)
        if len(missing_candidates) != 1:
            raise ValueError("toy passive table must omit exactly one atomic event word")
        if len(self.query_tokens) * len(self.answer_tokens) > (
            self.budgets.max_suffix_test_candidates
        ):
            raise OpaquePredictiveStateLimitError("diagnostic tests exceed suffix budget")
        if len(self.passive_rows) > self.budgets.max_oracle_evaluations:
            raise OpaquePredictiveStateLimitError("passive table exceeds oracle budget")
        if _TOY_FULL_ROW_COUNT > self.budgets.max_complete_diagnostic_rows:
            raise OpaquePredictiveStateLimitError("full diagnostic table exceeds budget")
        passive_hash = _sha256(tuple(_row_payload(row) for row in self.passive_rows))
        candidate_hash = _sha256(self.candidate_words)
        if _require_sha256("passive_table_sha256", self.passive_table_sha256) != passive_hash:
            raise ValueError("passive_table_sha256 does not bind passive rows")
        if _require_sha256("candidate_pool_sha256", self.candidate_pool_sha256) != candidate_hash:
            raise ValueError("candidate_pool_sha256 does not bind candidate order")
        expected_input_hash = _sha256(_input_payload(self))
        if _require_sha256("input_sha256", self.input_sha256) != expected_input_hash:
            raise ValueError("input_sha256 does not bind the sanitized learner input")

    @property
    def unlabeled_candidate_words(self) -> tuple[OpaqueWord, ...]:
        observed = {row.word for row in self.passive_rows}
        return tuple(word for word in self.candidate_words if word not in observed)


def _input_payload(value: OpaqueDiagnosticInput) -> dict[str, object]:
    return {
        "schema": value.schema,
        "scope": value.scope,
        "vocabulary_contract": value.vocabulary_contract,
        "event_tokens": value.event_tokens,
        "query_tokens": value.query_tokens,
        "answer_tokens": value.answer_tokens,
        "passive_rows": tuple(_row_payload(row) for row in value.passive_rows),
        "candidate_words": value.candidate_words,
        "budgets": value.budgets.payload(),
        "passive_table_sha256": value.passive_table_sha256,
        "candidate_pool_sha256": value.candidate_pool_sha256,
    }


@dataclass(frozen=True)
class ExactRankCertificate:
    """An exact witness minor and deterministic rational rank transcript."""

    matrix: RationalMatrix
    rank: int
    pivot_row_indices: tuple[int, ...]
    pivot_column_indices: tuple[int, ...]
    witness_determinant: Rational
    matrix_sha256: str
    certificate_sha256: str
    schema: str = _RANK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _RANK_SCHEMA:
            raise ValueError("unknown rank certificate schema")
        rows, columns = _matrix_shape(self.matrix)
        rank = _plain_int("rank", self.rank, 1)
        if rank > min(rows, columns):
            raise ValueError("rank exceeds matrix dimensions")
        computed_rank, pivot_rows, pivot_columns = _rank_profile(self.matrix)
        if (
            rank != computed_rank
            or self.pivot_row_indices != pivot_rows
            or self.pivot_column_indices != pivot_columns
        ):
            raise ValueError("rank or deterministic pivot transcript is incorrect")
        if type(self.witness_determinant) is not Rational:
            raise TypeError("witness_determinant must be exact Rational")
        witness = _submatrix(self.matrix, pivot_rows, pivot_columns)
        determinant = _determinant(witness)
        if determinant.numerator == 0 or determinant != self.witness_determinant:
            raise ValueError("rank witness determinant is incorrect or singular")
        matrix_hash = _sha256(self.matrix)
        if _require_sha256("matrix_sha256", self.matrix_sha256) != matrix_hash:
            raise ValueError("matrix_sha256 does not bind the matrix")
        certificate_hash = _sha256(_rank_payload(self))
        if (
            _require_sha256("certificate_sha256", self.certificate_sha256)
            != certificate_hash
        ):
            raise ValueError("rank certificate hash mismatch")


def _rank_payload(value: ExactRankCertificate) -> dict[str, object]:
    return {
        "schema": value.schema,
        "matrix_sha256": value.matrix_sha256,
        "rank": value.rank,
        "pivot_row_indices": value.pivot_row_indices,
        "pivot_column_indices": value.pivot_column_indices,
        "witness_determinant": value.witness_determinant,
    }


def _make_rank_certificate(matrix: RationalMatrix) -> ExactRankCertificate:
    rank, pivot_rows, pivot_columns = _rank_profile(matrix)
    if rank < 1:
        raise ValueError("zero-rank blocks are outside this toy experiment")
    determinant = _determinant(_submatrix(matrix, pivot_rows, pivot_columns))
    matrix_hash = _sha256(matrix)
    provisional = {
        "schema": _RANK_SCHEMA,
        "matrix_sha256": matrix_hash,
        "rank": rank,
        "pivot_row_indices": pivot_rows,
        "pivot_column_indices": pivot_columns,
        "witness_determinant": determinant,
    }
    return ExactRankCertificate(
        matrix=matrix,
        rank=rank,
        pivot_row_indices=pivot_rows,
        pivot_column_indices=pivot_columns,
        witness_determinant=determinant,
        matrix_sha256=matrix_hash,
        certificate_sha256=_sha256(provisional),
    )


@dataclass(frozen=True)
class ExactDiagnosticModel:
    """Basis-coordinate factorization of one finite diagnostic block."""

    source_words: tuple[OpaqueWord, ...]
    rank_certificate: ExactRankCertificate
    basis_words: tuple[OpaqueWord, ...]
    basis_determinant: Rational
    model_sha256: str
    schema: str = _MODEL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _MODEL_SCHEMA:
            raise ValueError("unknown exact diagnostic model schema")
        if not isinstance(self.source_words, tuple) or len(self.source_words) != len(
            self.rank_certificate.matrix
        ):
            raise ValueError("source words must align to rank-certificate rows")
        if len(set(self.source_words)) != len(self.source_words):
            raise ValueError("source words must be unique")
        for word in self.source_words:
            if not isinstance(word, tuple):
                raise TypeError("source words must be tuples")
            for token in word:
                _require_opaque_token("source word token", token)
        if type(self.rank_certificate) is not ExactRankCertificate:
            raise TypeError("rank_certificate must be exact ExactRankCertificate")
        expected_basis_words = tuple(
            self.source_words[index]
            for index in self.rank_certificate.pivot_row_indices
        )
        if self.basis_words != expected_basis_words:
            raise ValueError("basis words disagree with deterministic pivot rows")
        if type(self.basis_determinant) is not Rational:
            raise TypeError("basis_determinant must be exact Rational")
        if self.basis_determinant != self.rank_certificate.witness_determinant:
            raise ValueError("basis determinant disagrees with rank witness")
        expected_hash = _sha256(_model_payload(self))
        if _require_sha256("model_sha256", self.model_sha256) != expected_hash:
            raise ValueError("model_sha256 does not bind the model")

    @property
    def rank(self) -> int:
        return self.rank_certificate.rank

    @property
    def readout_matrix(self) -> RationalMatrix:
        return tuple(
            self.rank_certificate.matrix[index]
            for index in self.rank_certificate.pivot_row_indices
        )


def _model_payload(value: ExactDiagnosticModel) -> dict[str, object]:
    return {
        "schema": value.schema,
        "source_words": value.source_words,
        "rank_certificate_sha256": value.rank_certificate.certificate_sha256,
        "basis_words": value.basis_words,
        "basis_determinant": value.basis_determinant,
    }


def _make_model(
    source_words: tuple[OpaqueWord, ...], matrix: RationalMatrix
) -> ExactDiagnosticModel:
    rank_certificate = _make_rank_certificate(matrix)
    basis_words = tuple(
        source_words[index] for index in rank_certificate.pivot_row_indices
    )
    payload = {
        "schema": _MODEL_SCHEMA,
        "source_words": source_words,
        "rank_certificate_sha256": rank_certificate.certificate_sha256,
        "basis_words": basis_words,
        "basis_determinant": rank_certificate.witness_determinant,
    }
    return ExactDiagnosticModel(
        source_words=source_words,
        rank_certificate=rank_certificate,
        basis_words=basis_words,
        basis_determinant=rank_certificate.witness_determinant,
        model_sha256=_sha256(payload),
    )


def _coordinates(model: ExactDiagnosticModel, row: RationalVector) -> RationalVector:
    _, width = _matrix_shape(model.rank_certificate.matrix)
    if not isinstance(row, tuple) or len(row) != width:
        raise ValueError("diagnostic row width disagrees with model")
    if any(type(value) is not Rational for value in row):
        raise TypeError("diagnostic row entries must be Rational")
    pivot_columns = model.rank_certificate.pivot_column_indices
    basis_minor = _submatrix(
        model.readout_matrix,
        tuple(range(model.rank)),
        pivot_columns,
    )
    pivot_row = tuple(row[column] for column in pivot_columns)
    result = _row_times_matrix(pivot_row, _inverse(basis_minor))
    if _row_times_matrix(result, model.readout_matrix) != row:
        raise ValueError("row is outside the model's exact predictive span")
    return result


def _in_span(model: ExactDiagnosticModel, row: RationalVector) -> bool:
    try:
        _coordinates(model, row)
    except ValueError:
        return False
    return True


def _diagnostic_vector_from_tokens(
    answer_tokens: tuple[str, ...], answers: tuple[str, ...]
) -> RationalVector:
    if (
        not isinstance(answer_tokens, tuple)
        or len(answer_tokens) != _TOY_ANSWER_COUNT
        or answer_tokens != tuple(sorted(set(answer_tokens)))
    ):
        raise ValueError("answer_tokens must be the sorted three-token alphabet")
    for token in answer_tokens:
        _require_opaque_token("diagnostic answer alphabet", token)
    if not isinstance(answers, tuple) or len(answers) != _TOY_QUERY_COUNT:
        raise ValueError("answers must align to opaque query tokens")
    width = len(answers) * len(answer_tokens)
    row = [Rational(0) for _ in range(width)]
    for query_index, answer in enumerate(answers):
        if answer not in answer_tokens:
            raise ValueError("answer is outside the opaque output alphabet")
        answer_index = answer_tokens.index(answer)
        row[query_index * len(answer_tokens) + answer_index] = Rational(1)
    return tuple(row)


def _diagnostic_vector(
    value: OpaqueDiagnosticInput, answers: tuple[str, ...]
) -> RationalVector:
    if len(answers) != len(value.query_tokens):
        raise ValueError("answers must align to opaque query tokens")
    return _diagnostic_vector_from_tokens(value.answer_tokens, answers)


def _all_answer_tuples(value: OpaqueDiagnosticInput) -> tuple[tuple[str, ...], ...]:
    return tuple(product(value.answer_tokens, repeat=len(value.query_tokens)))


def _rows_matrix(rows: Sequence[OpaqueDiagnosticRow], value: OpaqueDiagnosticInput) -> RationalMatrix:
    return tuple(_diagnostic_vector(value, row.answers) for row in rows)


@dataclass(frozen=True)
class PassiveCompletionWitness:
    """One exact passive-consistent answer to the selected opaque word."""

    answers: tuple[str, ...]
    answer_tokens: tuple[str, ...]
    augmented_rank: ExactRankCertificate
    witness_sha256: str
    schema: str = _COMPLETION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _COMPLETION_SCHEMA:
            raise ValueError("unknown passive completion witness schema")
        if not isinstance(self.answers, tuple) or not self.answers:
            raise TypeError("completion answers must be a nonempty tuple")
        for answer in self.answers:
            _require_opaque_token("completion answer", answer)
        expected_row = _diagnostic_vector_from_tokens(
            self.answer_tokens, self.answers
        )
        if type(self.augmented_rank) is not ExactRankCertificate:
            raise TypeError("augmented_rank must be exact ExactRankCertificate")
        if self.augmented_rank.matrix[-1] != expected_row:
            raise ValueError(
                "completion answers do not encode the appended diagnostic row"
            )
        expected_hash = _sha256(_completion_payload(self))
        if _require_sha256("witness_sha256", self.witness_sha256) != expected_hash:
            raise ValueError("completion witness hash mismatch")


def _completion_payload(value: PassiveCompletionWitness) -> dict[str, object]:
    return {
        "schema": value.schema,
        "answers": value.answers,
        "answer_tokens": value.answer_tokens,
        "augmented_rank_sha256": value.augmented_rank.certificate_sha256,
    }


def _make_completion_witness(
    answers: tuple[str, ...],
    answer_tokens: tuple[str, ...],
    rank: ExactRankCertificate,
) -> PassiveCompletionWitness:
    payload = {
        "schema": _COMPLETION_SCHEMA,
        "answers": answers,
        "answer_tokens": answer_tokens,
        "augmented_rank_sha256": rank.certificate_sha256,
    }
    return PassiveCompletionWitness(
        answers=answers,
        answer_tokens=answer_tokens,
        augmented_rank=rank,
        witness_sha256=_sha256(payload),
    )


@dataclass(frozen=True)
class OpaqueVocabularyCoverage:
    """Behavioral proof of three observed slots and one opaque missing slot."""

    empty_answers: tuple[str, ...]
    observed_slots: tuple[tuple[str, str], ...]
    missing_slot: tuple[str, str]
    coverage_sha256: str
    schema: str = _COVERAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _COVERAGE_SCHEMA:
            raise ValueError("unknown opaque vocabulary coverage schema")
        if not isinstance(self.empty_answers, tuple) or len(self.empty_answers) != 2:
            raise ValueError("coverage needs both empty-history answers")
        for answer in self.empty_answers:
            _require_opaque_token("empty-history answer", answer)
        if (
            not isinstance(self.observed_slots, tuple)
            or len(self.observed_slots) != 3
            or self.observed_slots != tuple(sorted(set(self.observed_slots)))
        ):
            raise ValueError("coverage must contain three sorted distinct slots")
        for query, answer in self.observed_slots:
            _require_opaque_token("coverage query", query)
            _require_opaque_token("coverage answer", answer)
        if type(self.missing_slot) is not tuple or len(self.missing_slot) != 2:
            raise TypeError("missing_slot must be an opaque query/answer pair")
        _require_opaque_token("missing-slot query", self.missing_slot[0])
        _require_opaque_token("missing-slot answer", self.missing_slot[1])
        if self.missing_slot in self.observed_slots:
            raise ValueError("missing slot is already observed")
        expected_hash = _sha256(_coverage_payload(self))
        if _require_sha256("coverage_sha256", self.coverage_sha256) != expected_hash:
            raise ValueError("coverage_sha256 does not bind the coverage certificate")


def _coverage_payload(value: OpaqueVocabularyCoverage) -> dict[str, object]:
    return {
        "schema": value.schema,
        "empty_answers": value.empty_answers,
        "observed_slots": value.observed_slots,
        "missing_slot": value.missing_slot,
    }


@dataclass(frozen=True)
class OpaqueAcquisitionCommitment:
    """Answer-free fixed-pool commitment emitted before membership access."""

    input_sha256: str
    candidate_pool_sha256: str
    vocabulary_coverage: OpaqueVocabularyCoverage
    selected_word: OpaqueWord
    compatible_completions: tuple[PassiveCompletionWitness, ...]
    twin_answers: tuple[tuple[str, ...], tuple[str, ...]]
    known_control_word: OpaqueWord
    known_control_rank: ExactRankCertificate
    preactive_membership_calls: int
    commitment_sha256: str
    schema: str = _COMMITMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _COMMITMENT_SCHEMA:
            raise ValueError("unknown opaque acquisition commitment schema")
        _require_sha256("input_sha256", self.input_sha256)
        _require_sha256("candidate_pool_sha256", self.candidate_pool_sha256)
        if type(self.vocabulary_coverage) is not OpaqueVocabularyCoverage:
            raise TypeError(
                "vocabulary_coverage must be exact OpaqueVocabularyCoverage"
            )
        for name, word in (
            ("selected_word", self.selected_word),
            ("known_control_word", self.known_control_word),
        ):
            if not isinstance(word, tuple) or len(word) != 1:
                raise ValueError(f"{name} must be one opaque atomic event")
            _require_opaque_token(name, word[0])
        if self.selected_word == self.known_control_word:
            raise ValueError("selected acquisition and known control must differ")
        if not isinstance(self.compatible_completions, tuple) or len(
            self.compatible_completions
        ) < 2:
            raise ValueError("at least two compatible completions are required")
        if any(
            type(row) is not PassiveCompletionWitness
            for row in self.compatible_completions
        ):
            raise TypeError("compatible completions have the wrong type")
        completion_answers = tuple(row.answers for row in self.compatible_completions)
        if completion_answers != tuple(sorted(set(completion_answers))):
            raise ValueError("compatible completions must be sorted and unique")
        if (
            not isinstance(self.twin_answers, tuple)
            or len(self.twin_answers) != 2
            or self.twin_answers[0] == self.twin_answers[1]
            or any(answers not in completion_answers for answers in self.twin_answers)
        ):
            raise ValueError("twins must be two disagreeing compatible completions")
        if type(self.known_control_rank) is not ExactRankCertificate:
            raise TypeError("known_control_rank must be exact ExactRankCertificate")
        if _plain_int(
            "preactive_membership_calls", self.preactive_membership_calls, 0
        ) != 0:
            raise ValueError("preactive commitment must make zero membership calls")
        expected_hash = _sha256(_commitment_payload(self))
        if _require_sha256("commitment_sha256", self.commitment_sha256) != expected_hash:
            raise ValueError("commitment_sha256 does not bind the commitment")


def _commitment_payload(value: OpaqueAcquisitionCommitment) -> dict[str, object]:
    return {
        "schema": value.schema,
        "input_sha256": value.input_sha256,
        "candidate_pool_sha256": value.candidate_pool_sha256,
        "vocabulary_coverage_sha256": value.vocabulary_coverage.coverage_sha256,
        "selected_word": value.selected_word,
        "compatible_completion_sha256s": tuple(
            row.witness_sha256 for row in value.compatible_completions
        ),
        "twin_answers": value.twin_answers,
        "known_control_word": value.known_control_word,
        "known_control_rank_sha256": value.known_control_rank.certificate_sha256,
        "preactive_membership_calls": value.preactive_membership_calls,
    }


@dataclass(frozen=True)
class PassiveDiscoveryResult:
    """Exact rank-four result and one-dimensional acquisition commitment.

    Its digest is a context-bound content link, not a standalone semantic
    certificate.  Authoritative validation reconstructs this object from its
    :class:`OpaqueDiagnosticInput` inside :class:`ToyOpaqueEnvironmentResult`.
    """

    input_sha256: str
    model: ExactDiagnosticModel
    span_answer_tuples: tuple[tuple[str, ...], ...]
    commitment: OpaqueAcquisitionCommitment
    exact_rank_evaluation_count: int
    result_sha256: str
    schema: str = _PASSIVE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _PASSIVE_SCHEMA:
            raise ValueError("unknown passive discovery result schema")
        _require_sha256("input_sha256", self.input_sha256)
        if type(self.model) is not ExactDiagnosticModel:
            raise TypeError("model must be exact ExactDiagnosticModel")
        if self.model.rank != _TOY_PASSIVE_RANK:
            raise ValueError("toy passive model must have exact rank four")
        if len(self.model.source_words) != _TOY_PASSIVE_ROW_COUNT:
            raise ValueError("passive model must contain the complete six-row support")
        if (
            not isinstance(self.span_answer_tuples, tuple)
            or len(self.span_answer_tuples) != _TOY_PASSIVE_ROW_COUNT
            or self.span_answer_tuples != tuple(sorted(set(self.span_answer_tuples)))
        ):
            raise ValueError("passive span answers must be six sorted unique rows")
        if type(self.commitment) is not OpaqueAcquisitionCommitment:
            raise TypeError("commitment must be exact OpaqueAcquisitionCommitment")
        if self.commitment.input_sha256 != self.input_sha256:
            raise ValueError("passive result and commitment input hashes disagree")
        if self.commitment.known_control_rank.rank != _TOY_PASSIVE_RANK:
            raise ValueError("known-entry control must remain rank four")
        if len(self.commitment.compatible_completions) != 3 or any(
            row.augmented_rank.rank != _TOY_FULL_RANK
            for row in self.commitment.compatible_completions
        ):
            raise ValueError("all three compatible answers must raise rank four to five")
        passive_matrix = self.model.rank_certificate.matrix
        augmented_matrices = tuple(
            row.augmented_rank.matrix
            for row in self.commitment.compatible_completions
        )
        if any(
            len(matrix) != len(passive_matrix) + 1
            or matrix[:-1] != passive_matrix
            for matrix in augmented_matrices
        ):
            raise ValueError("completion witnesses must share the exact passive block")
        if len({matrix[-1] for matrix in augmented_matrices}) != 3:
            raise ValueError("completion witnesses must disagree on the selected row")
        known_matrix = self.commitment.known_control_rank.matrix
        if (
            len(known_matrix) != len(passive_matrix) + 1
            or known_matrix[:-1] != passive_matrix
            or known_matrix[-1] not in passive_matrix
        ):
            raise ValueError("known control must append one aliased passive row")
        if _plain_int(
            "exact_rank_evaluation_count", self.exact_rank_evaluation_count, 1
        ) != 14:
            raise ValueError("toy passive search must perform exactly 14 rank evaluations")
        expected_hash = _sha256(_passive_payload(self))
        if _require_sha256("result_sha256", self.result_sha256) != expected_hash:
            raise ValueError("passive result hash mismatch")

    @property
    def exact_rank(self) -> int:
        return self.model.rank

    @property
    def ambiguity_dimension(self) -> int:
        return _TOY_FULL_RANK - self.model.rank


def _passive_payload(value: PassiveDiscoveryResult) -> dict[str, object]:
    return {
        "schema": value.schema,
        "input_sha256": value.input_sha256,
        "model_sha256": value.model.model_sha256,
        "span_answer_tuples": value.span_answer_tuples,
        "commitment_sha256": value.commitment.commitment_sha256,
        "exact_rank_evaluation_count": value.exact_rank_evaluation_count,
    }


def fit_passive_opaque_hankel(value: OpaqueDiagnosticInput) -> PassiveDiscoveryResult:
    """Fit and commit using only the sanitized passive table and vocabulary.

    No callback, controller object, omitted identifier, or active answer is in
    this function's argument surface.  Candidate compatibility is defined by
    categorical vocabulary coverage and exact row-span disagreement.
    """

    if type(value) is not OpaqueDiagnosticInput:
        raise TypeError("fit_passive_opaque_hankel accepts only OpaqueDiagnosticInput")
    passive_matrix = _rows_matrix(value.passive_rows, value)
    model = _make_model(tuple(row.word for row in value.passive_rows), passive_matrix)
    if model.rank > value.budgets.max_basis_dimension:
        raise OpaquePredictiveStateLimitError("passive basis exceeds dimension budget")
    if model.rank != _TOY_PASSIVE_RANK:
        raise ValueError("passive diagnostic support did not have exact rank four")

    answer_inventory = _all_answer_tuples(value)
    span_answers = tuple(
        answers
        for answers in answer_inventory
        if _in_span(model, _diagnostic_vector(value, answers))
    )
    observed_answers = tuple(sorted(row.answers for row in value.passive_rows))
    if span_answers != observed_answers:
        raise ValueError(
            "passive support is not complete for its exact categorical row span"
        )
    observed_by_word = {row.word: row for row in value.passive_rows}
    empty_answers = observed_by_word[()].answers
    observed_slots: list[tuple[str, str]] = []
    for word in value.candidate_words:
        row = observed_by_word.get(word)
        if row is None:
            continue
        changed_queries = tuple(
            index
            for index, (answer, empty_answer) in enumerate(
                zip(row.answers, empty_answers, strict=True)
            )
            if answer != empty_answer
        )
        if len(changed_queries) != 1:
            raise ValueError(
                "each observed atomic token must expose one nonempty query slot"
            )
        query_index = changed_queries[0]
        observed_slots.append(
            (value.query_tokens[query_index], row.answers[query_index])
        )
    observed_slot_tuple = tuple(sorted(observed_slots))
    all_slots = tuple(
        sorted(
            (query, answer)
            for query_index, query in enumerate(value.query_tokens)
            for answer in value.answer_tokens
            if answer != empty_answers[query_index]
        )
    )
    missing_slots = tuple(slot for slot in all_slots if slot not in observed_slot_tuple)
    if (
        len(observed_slot_tuple) != 3
        or len(set(observed_slot_tuple)) != 3
        or len(all_slots) != 4
        or len(missing_slots) != 1
    ):
        raise ValueError(
            "opaque atomic vocabulary does not certify three-of-four slot coverage"
        )
    missing_slot = missing_slots[0]
    coverage_payload = {
        "schema": _COVERAGE_SCHEMA,
        "empty_answers": empty_answers,
        "observed_slots": observed_slot_tuple,
        "missing_slot": missing_slot,
    }
    coverage = OpaqueVocabularyCoverage(
        empty_answers=empty_answers,
        observed_slots=observed_slot_tuple,
        missing_slot=missing_slot,
        coverage_sha256=_sha256(coverage_payload),
    )
    missing_query_index = value.query_tokens.index(missing_slot[0])
    coverage_answers = tuple(
        answers
        for answers in answer_inventory
        if answers[missing_query_index] == missing_slot[1]
    )
    outside_answers = tuple(
        answers for answers in answer_inventory if answers not in span_answers
    )
    if coverage_answers != outside_answers or len(coverage_answers) != 3:
        raise ValueError(
            "coverage-compatible answers must equal the exact disagreement space"
        )

    selected_word: OpaqueWord | None = None
    compatible: tuple[PassiveCompletionWitness, ...] = ()
    rank_evaluations = 1 + len(answer_inventory)
    for candidate_word in value.candidate_words:
        if candidate_word in observed_by_word:
            continue
        witnesses: list[PassiveCompletionWitness] = []
        for answers in coverage_answers:
            augmented = passive_matrix + (_diagnostic_vector(value, answers),)
            rank = _make_rank_certificate(augmented)
            rank_evaluations += 1
            if rank.rank == model.rank + 1:
                witnesses.append(
                    _make_completion_witness(answers, value.answer_tokens, rank)
                )
        if witnesses and len(witnesses) == len(coverage_answers):
            selected_word = candidate_word
            compatible = tuple(witnesses)
            break
    if selected_word is None:
        raise ValueError("no frozen candidate has answer-independent rank gain")

    deficient = missing_query_index
    other = 1 - deficient
    empty_preserving = list(empty_answers)
    empty_preserving[deficient] = missing_slot[1]
    alternative = list(empty_preserving)
    alternative[other] = next(
        answer for answer in value.answer_tokens if answer != empty_answers[other]
    )
    twins = (tuple(empty_preserving), tuple(alternative))
    compatible_answers = tuple(row.answers for row in compatible)
    if any(answers not in compatible_answers for answers in twins):
        raise ValueError("behaviorally selected twins left the compatible version space")

    known_control_word = next(
        word for word in value.candidate_words if word in observed_by_word
    )
    known_control_matrix = passive_matrix + (
        _diagnostic_vector(value, observed_by_word[known_control_word].answers),
    )
    known_control_rank = _make_rank_certificate(known_control_matrix)
    rank_evaluations += 1
    if known_control_rank.rank != model.rank:
        raise ValueError("repeating a known diagnostic unexpectedly changed rank")

    commitment_payload = {
        "schema": _COMMITMENT_SCHEMA,
        "input_sha256": value.input_sha256,
        "candidate_pool_sha256": value.candidate_pool_sha256,
        "vocabulary_coverage_sha256": coverage.coverage_sha256,
        "selected_word": selected_word,
        "compatible_completion_sha256s": tuple(
            row.witness_sha256 for row in compatible
        ),
        "twin_answers": twins,
        "known_control_word": known_control_word,
        "known_control_rank_sha256": known_control_rank.certificate_sha256,
        "preactive_membership_calls": 0,
    }
    commitment = OpaqueAcquisitionCommitment(
        input_sha256=value.input_sha256,
        candidate_pool_sha256=value.candidate_pool_sha256,
        vocabulary_coverage=coverage,
        selected_word=selected_word,
        compatible_completions=compatible,
        twin_answers=twins,
        known_control_word=known_control_word,
        known_control_rank=known_control_rank,
        preactive_membership_calls=0,
        commitment_sha256=_sha256(commitment_payload),
    )
    passive_payload = {
        "schema": _PASSIVE_SCHEMA,
        "input_sha256": value.input_sha256,
        "model_sha256": model.model_sha256,
        "span_answer_tuples": span_answers,
        "commitment_sha256": commitment.commitment_sha256,
        "exact_rank_evaluation_count": rank_evaluations,
    }
    return PassiveDiscoveryResult(
        input_sha256=value.input_sha256,
        model=model,
        span_answer_tuples=span_answers,
        commitment=commitment,
        exact_rank_evaluation_count=rank_evaluations,
        result_sha256=_sha256(passive_payload),
    )


@dataclass(frozen=True)
class OpaqueMembershipAnswer:
    """The sole label-bearing response, released only after commitment."""

    input_sha256: str
    commitment_sha256: str
    word: OpaqueWord
    answers: tuple[str, ...]
    response_ordinal: int
    answer_sha256: str
    schema: str = _ANSWER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _ANSWER_SCHEMA:
            raise ValueError("unknown opaque membership answer schema")
        _require_sha256("input_sha256", self.input_sha256)
        _require_sha256("commitment_sha256", self.commitment_sha256)
        if not isinstance(self.word, tuple) or len(self.word) != 1:
            raise ValueError("membership answer word must be one atomic event")
        _require_opaque_token("membership word", self.word[0])
        if not isinstance(self.answers, tuple) or len(self.answers) != 2:
            raise ValueError("membership response must contain both query answers")
        for answer in self.answers:
            _require_opaque_token("membership answer", answer)
        if _plain_int("response_ordinal", self.response_ordinal, 1) != 1:
            raise ValueError("toy experiment permits exactly one active response")
        expected_hash = _sha256(_answer_payload(self))
        if _require_sha256("answer_sha256", self.answer_sha256) != expected_hash:
            raise ValueError("answer_sha256 does not bind the membership response")


def _answer_payload(value: OpaqueMembershipAnswer) -> dict[str, object]:
    return {
        "schema": value.schema,
        "input_sha256": value.input_sha256,
        "commitment_sha256": value.commitment_sha256,
        "word": value.word,
        "answers": value.answers,
        "response_ordinal": value.response_ordinal,
    }


@dataclass(frozen=True, order=True)
class FullDiagnosticCoordinate:
    """Exact coordinates and reconstruction for one categorical diagnostic."""

    answers: tuple[str, ...]
    answer_tokens: tuple[str, ...]
    diagnostic_row: RationalVector
    coordinates: RationalVector
    coordinate_sha256: str
    schema: str = _COORDINATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _COORDINATE_SCHEMA:
            raise ValueError("unknown full diagnostic coordinate schema")
        if not isinstance(self.answers, tuple) or len(self.answers) != 2:
            raise ValueError("full diagnostic answers must contain two outputs")
        for answer in self.answers:
            _require_opaque_token("full diagnostic answer", answer)
        expected_row = _diagnostic_vector_from_tokens(
            self.answer_tokens, self.answers
        )
        if not isinstance(self.diagnostic_row, tuple) or not isinstance(
            self.coordinates, tuple
        ):
            raise TypeError("diagnostic row and coordinates must be tuples")
        if any(type(value) is not Rational for value in self.diagnostic_row):
            raise TypeError("diagnostic row entries must be Rational")
        if any(type(value) is not Rational for value in self.coordinates):
            raise TypeError("coordinate entries must be Rational")
        if self.diagnostic_row != expected_row:
            raise ValueError(
                "full diagnostic answers do not encode the diagnostic row"
            )
        expected_hash = _sha256(_coordinate_payload(self))
        if _require_sha256("coordinate_sha256", self.coordinate_sha256) != expected_hash:
            raise ValueError("coordinate hash mismatch")


def _coordinate_payload(value: FullDiagnosticCoordinate) -> dict[str, object]:
    return {
        "schema": value.schema,
        "answers": value.answers,
        "answer_tokens": value.answer_tokens,
        "diagnostic_row": value.diagnostic_row,
        "coordinates": value.coordinates,
    }


@dataclass(frozen=True)
class PostactiveDiscoveryResult:
    """Rank-five from-scratch rebuild and complete diagnostic reconstruction.

    Its digest is a context-bound content link.  Completion attribution and
    answer linkage become authoritative only when the enclosing environment
    reconstructs this result from the learner input and opened answer.
    """

    input_sha256: str
    passive_result_sha256: str
    answer_sha256: str
    model: ExactDiagnosticModel
    full_diagnostics: tuple[FullDiagnosticCoordinate, ...]
    compatible_completion_index: int
    compatible_completion_witness_sha256: str
    active_membership_calls: int
    rebuild_count: int
    result_sha256: str
    schema: str = _POSTACTIVE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _POSTACTIVE_SCHEMA:
            raise ValueError("unknown postactive discovery result schema")
        _require_sha256("input_sha256", self.input_sha256)
        _require_sha256("passive_result_sha256", self.passive_result_sha256)
        _require_sha256("answer_sha256", self.answer_sha256)
        if type(self.model) is not ExactDiagnosticModel or self.model.rank != 5:
            raise ValueError("postactive model must be an exact rank-five model")
        if (
            not isinstance(self.full_diagnostics, tuple)
            or len(self.full_diagnostics) != _TOY_FULL_ROW_COUNT
            or any(
                type(row) is not FullDiagnosticCoordinate
                for row in self.full_diagnostics
            )
        ):
            raise ValueError("postactive result must contain all nine diagnostics")
        answers = tuple(row.answers for row in self.full_diagnostics)
        if answers != tuple(sorted(set(answers))):
            raise ValueError("full diagnostics must be sorted and unique")
        diagnostic_rows = tuple(row.diagnostic_row for row in self.full_diagnostics)
        if len(set(diagnostic_rows)) != _TOY_FULL_ROW_COUNT:
            raise ValueError("full diagnostic vectors must be distinct")
        for row in self.full_diagnostics:
            if (
                len(row.diagnostic_row) != _TOY_QUERY_COUNT * _TOY_ANSWER_COUNT
                or sum(value.as_fraction() for value in row.diagnostic_row)
                != _TOY_QUERY_COUNT
                or any(value not in (Rational(0), Rational(1)) for value in row.diagnostic_row)
            ):
                raise ValueError("full diagnostic row is not exact two-channel one-hot")
            if len(row.coordinates) != self.model.rank:
                raise ValueError("diagnostic coordinate dimension is incorrect")
            if _row_times_matrix(row.coordinates, self.model.readout_matrix) != (
                row.diagnostic_row
            ):
                raise ValueError("full diagnostic coordinate does not reconstruct")
        if _plain_int(
            "compatible_completion_index", self.compatible_completion_index, 0
        ) >= 3:
            raise ValueError("compatible completion index must be zero, one, or two")
        _require_sha256(
            "compatible_completion_witness_sha256",
            self.compatible_completion_witness_sha256,
        )
        if _plain_int("active_membership_calls", self.active_membership_calls, 0) != 1:
            raise ValueError("postactive result must consume exactly one response")
        if _plain_int("rebuild_count", self.rebuild_count, 0) != 1:
            raise ValueError("postactive result permits one from-scratch rebuild")
        expected_hash = _sha256(_postactive_payload(self))
        if _require_sha256("result_sha256", self.result_sha256) != expected_hash:
            raise ValueError("postactive result hash mismatch")

    @property
    def exact_rank(self) -> int:
        return self.model.rank

    @property
    def full_diagnostic_count(self) -> int:
        return len(self.full_diagnostics)


def _postactive_payload(value: PostactiveDiscoveryResult) -> dict[str, object]:
    return {
        "schema": value.schema,
        "input_sha256": value.input_sha256,
        "passive_result_sha256": value.passive_result_sha256,
        "answer_sha256": value.answer_sha256,
        "model_sha256": value.model.model_sha256,
        "full_diagnostic_sha256s": tuple(
            row.coordinate_sha256 for row in value.full_diagnostics
        ),
        "compatible_completion_index": value.compatible_completion_index,
        "compatible_completion_witness_sha256": (
            value.compatible_completion_witness_sha256
        ),
        "active_membership_calls": value.active_membership_calls,
        "rebuild_count": value.rebuild_count,
    }


def fit_postactive_opaque_hankel(
    value: OpaqueDiagnosticInput,
    passive: PassiveDiscoveryResult,
    answer: OpaqueMembershipAnswer,
) -> PostactiveDiscoveryResult:
    """Rebuild once from the passive table plus the committed two-query answer."""

    if type(value) is not OpaqueDiagnosticInput:
        raise TypeError("value must be exact OpaqueDiagnosticInput")
    if type(passive) is not PassiveDiscoveryResult:
        raise TypeError("passive must be exact PassiveDiscoveryResult")
    if type(answer) is not OpaqueMembershipAnswer:
        raise TypeError("answer must be exact OpaqueMembershipAnswer")
    if passive != fit_passive_opaque_hankel(value):
        raise ValueError("passive result is not the deterministic fit of this input")
    commitment = passive.commitment
    if (
        answer.input_sha256 != value.input_sha256
        or answer.commitment_sha256 != commitment.commitment_sha256
        or answer.word != commitment.selected_word
    ):
        raise ValueError("active answer does not match the frozen commitment")
    compatible_answers = tuple(
        row.answers for row in commitment.compatible_completions
    )
    if answer.answers not in compatible_answers:
        raise ValueError("active answer was not in the committed version space")
    completion_matches = tuple(
        index
        for index, witness in enumerate(commitment.compatible_completions)
        if witness.answers == answer.answers
    )
    if len(completion_matches) != 1:
        raise ValueError("active answer must select one unique committed completion")
    completion_index = completion_matches[0]
    completion_witness = commitment.compatible_completions[completion_index]
    if value.budgets.max_active_responses < 1:
        raise OpaquePredictiveStateLimitError("active response budget is exhausted")

    augmented_rows = tuple(
        sorted(
            value.passive_rows
            + (OpaqueDiagnosticRow(answer.word, answer.answers),),
            key=lambda row: row.word,
        )
    )
    model = _make_model(
        tuple(row.word for row in augmented_rows),
        _rows_matrix(augmented_rows, value),
    )
    if model.rank > value.budgets.max_basis_dimension:
        raise OpaquePredictiveStateLimitError("postactive basis exceeds dimension budget")
    if model.rank != _TOY_FULL_RANK:
        raise ValueError("active response did not restore exact rank five")

    full_rows: list[FullDiagnosticCoordinate] = []
    for answers in _all_answer_tuples(value):
        diagnostic_row = _diagnostic_vector(value, answers)
        coordinates = _coordinates(model, diagnostic_row)
        coordinate_payload = {
            "schema": _COORDINATE_SCHEMA,
            "answers": answers,
            "answer_tokens": value.answer_tokens,
            "diagnostic_row": diagnostic_row,
            "coordinates": coordinates,
        }
        full_rows.append(
            FullDiagnosticCoordinate(
                answers=answers,
                answer_tokens=value.answer_tokens,
                diagnostic_row=diagnostic_row,
                coordinates=coordinates,
                coordinate_sha256=_sha256(coordinate_payload),
            )
        )
    post_payload = {
        "schema": _POSTACTIVE_SCHEMA,
        "input_sha256": value.input_sha256,
        "passive_result_sha256": passive.result_sha256,
        "answer_sha256": answer.answer_sha256,
        "model_sha256": model.model_sha256,
        "full_diagnostic_sha256s": tuple(
            row.coordinate_sha256 for row in full_rows
        ),
        "compatible_completion_index": completion_index,
        "compatible_completion_witness_sha256": completion_witness.witness_sha256,
        "active_membership_calls": 1,
        "rebuild_count": 1,
    }
    return PostactiveDiscoveryResult(
        input_sha256=value.input_sha256,
        passive_result_sha256=passive.result_sha256,
        answer_sha256=answer.answer_sha256,
        model=model,
        full_diagnostics=tuple(full_rows),
        compatible_completion_index=completion_index,
        compatible_completion_witness_sha256=completion_witness.witness_sha256,
        active_membership_calls=1,
        rebuild_count=1,
        result_sha256=_sha256(post_payload),
    )


def _normalize_semantic_cell(value: object) -> SemanticCell:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError("controller pseudoheldout cell must be an exact pair")
    key, symbol = value
    _plain_int("controller pseudoheldout key", key, 0)
    _plain_int("controller pseudoheldout value", symbol, 0)
    if (key, symbol) not in _SEMANTIC_CELLS:
        raise ValueError("controller pseudoheldout cell is outside K=2,V=2")
    return key, symbol


def _opaque_pool(namespace: str, controller_nonce: str, count: int) -> tuple[str, ...]:
    nonce = bytes.fromhex(_require_controller_nonce(controller_nonce))
    return tuple(
        sorted(
            hashlib.sha256(
                nonce
                + b"\x00tnlm-v3-phase3-toy\x00"
                + namespace.encode("ascii")
                + b"\x00"
                + index.to_bytes(4, "big")
            ).hexdigest()[:32]
            for index in range(count)
        )
    )


def _nonce_keyed_permutation(
    namespace: str, controller_nonce: str, count: int
) -> tuple[int, ...]:
    """Return a private deterministic bijection independent of environment labels.

    The semantic position is used only as an input to a nonce-keyed ordering.
    Neither the pseudoheldout cell nor the relabel-block index participates, so
    the learner-visible omission pattern cannot be matched against a public
    eight-environment permutation schedule.
    """

    nonce = bytes.fromhex(_require_controller_nonce(controller_nonce))
    if type(namespace) is not str or not namespace:
        raise TypeError("permutation namespace must be a nonempty string")
    size = _plain_int("permutation size", count, 1)
    ranked = sorted(
        (
            hashlib.sha256(
                nonce
                + b"\x00tnlm-v3-phase3-toy-private-bijection\x00"
                + namespace.encode("ascii")
                + b"\x00"
                + index.to_bytes(4, "big")
            ).digest(),
            index,
        )
        for index in range(size)
    )
    return tuple(index for _, index in ranked)


@dataclass(frozen=True)
class _ControllerMaterial:
    controller_nonce: str = field(repr=False)
    omitted_cell: SemanticCell = field(repr=False)
    relabel_block: int = field(repr=False)
    event_by_cell: tuple[tuple[SemanticCell, str], ...] = field(repr=False)
    query_by_key: tuple[tuple[int, str], ...] = field(repr=False)
    answer_by_value: tuple[tuple[int, str], ...] = field(repr=False)
    history_key_order: tuple[int, ...] = field(repr=False)
    full_rows: tuple[tuple[SemanticState, OpaqueDiagnosticRow], ...] = field(
        repr=False
    )


def _controller_payload(
    material: _ControllerMaterial, input_sha256: str
) -> dict[str, object]:
    return {
        "controller_schema": "tnlm-v3-toy-opaque-controller-private-v1",
        "controller_nonce_sha256": _sha256(material.controller_nonce),
        "omitted_cell": material.omitted_cell,
        "relabel_block": material.relabel_block,
        "event_by_cell": material.event_by_cell,
        "query_by_key": material.query_by_key,
        "answer_by_value": material.answer_by_value,
        "history_key_order": material.history_key_order,
        "full_rows": tuple(
            {"state": state, "row": _row_payload(row)}
            for state, row in material.full_rows
        ),
        "sanitized_input_sha256": input_sha256,
    }


@dataclass(frozen=True, repr=False)
class ToyControllerEnvironment:
    """Controller-owned wrapper; never accepted by either learner function.

    Its public representation exposes only content commitments.  Controller
    semantics are private implementation fields and are absent from
    :class:`OpaqueDiagnosticInput` and from that input's canonical hash.
    """

    learner_input: OpaqueDiagnosticInput
    controller_commitment_sha256: str
    _material: _ControllerMaterial = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.learner_input) is not OpaqueDiagnosticInput:
            raise TypeError("learner_input must be exact OpaqueDiagnosticInput")
        if type(self._material) is not _ControllerMaterial:
            raise TypeError("private controller material has the wrong type")
        expected = _sha256(
            _controller_payload(self._material, self.learner_input.input_sha256)
        )
        if (
            _require_sha256(
                "controller_commitment_sha256", self.controller_commitment_sha256
            )
            != expected
        ):
            raise ValueError("controller commitment does not bind private material")

    def __repr__(self) -> str:
        return (
            "ToyControllerEnvironment(learner_input_sha256="
            f"'{self.learner_input.input_sha256}', "
            "controller_commitment_sha256="
            f"'{self.controller_commitment_sha256}')"
        )


def _check_controller_budgets(budgets: OpaqueHankelBudgets) -> None:
    requirements = (
        ("max_word_length", budgets.max_word_length, 2),
        (
            "max_suffix_test_candidates",
            budgets.max_suffix_test_candidates,
            _TOY_QUERY_COUNT * _TOY_ANSWER_COUNT,
        ),
        ("max_oracle_evaluations", budgets.max_oracle_evaluations, 6),
        (
            "max_active_candidate_words",
            budgets.max_active_candidate_words,
            _TOY_EVENT_COUNT,
        ),
        ("max_active_responses", budgets.max_active_responses, 1),
        ("max_basis_dimension", budgets.max_basis_dimension, _TOY_FULL_RANK),
        (
            "max_complete_diagnostic_rows",
            budgets.max_complete_diagnostic_rows,
            _TOY_FULL_ROW_COUNT,
        ),
    )
    for name, available, required in requirements:
        if available < required:
            raise OpaquePredictiveStateLimitError(
                f"{name}={available} cannot cover analytic requirement {required}"
            )


def build_toy_controller_environment(
    pseudoheldout_cell: SemanticCell,
    relabel_block: int,
    *,
    controller_nonce: str,
    budgets: OpaqueHankelBudgets | None = None,
) -> ToyControllerEnvironment:
    """Build one trusted rotation while returning only a sanitized learner view."""

    omitted = _normalize_semantic_cell(pseudoheldout_cell)
    block = _plain_int("relabel_block", relabel_block, 0)
    if block >= _TOY_RELABEL_BLOCKS:
        raise ValueError("toy experiment has exactly two relabel blocks")
    nonce = _require_controller_nonce(controller_nonce)
    selected_budgets = OpaqueHankelBudgets() if budgets is None else budgets
    if type(selected_budgets) is not OpaqueHankelBudgets:
        raise TypeError("budgets must be exact OpaqueHankelBudgets or None")
    _check_controller_budgets(selected_budgets)

    event_pool = _opaque_pool("atomic-event", nonce, _TOY_EVENT_COUNT)
    query_pool = _opaque_pool("terminal-query", nonce, _TOY_QUERY_COUNT)
    answer_pool = _opaque_pool("categorical-answer", nonce, _TOY_ANSWER_COUNT)
    event_order = _nonce_keyed_permutation(
        "event-semantic-bijection", nonce, _TOY_EVENT_COUNT
    )
    query_order = _nonce_keyed_permutation(
        "query-semantic-bijection", nonce, _TOY_QUERY_COUNT
    )
    answer_order = _nonce_keyed_permutation(
        "answer-semantic-bijection", nonce, _TOY_ANSWER_COUNT
    )
    history_key_order = _nonce_keyed_permutation(
        "history-key-traversal", nonce, _TOY_QUERY_COUNT
    )
    event_by_cell = tuple(
        (cell, event_pool[event_order[index]])
        for index, cell in enumerate(_SEMANTIC_CELLS)
    )
    query_by_key = tuple(
        (key, query_pool[query_order[key]]) for key in range(_TOY_QUERY_COUNT)
    )
    semantic_answers = (-1, 0, 1)
    answer_by_value = tuple(
        (value, answer_pool[answer_order[index]])
        for index, value in enumerate(semantic_answers)
    )
    event_lookup = dict(event_by_cell)
    query_lookup = dict(query_by_key)
    answer_lookup = dict(answer_by_value)
    query_tokens = tuple(sorted(query_pool))
    query_key_by_token = {token: key for key, token in query_by_key}

    full_rows: list[tuple[SemanticState, OpaqueDiagnosticRow]] = []
    for state in _FULL_SEMANTIC_STATES:
        word = tuple(
            event_lookup[(key, state[key])]
            for key in history_key_order
            if state[key] >= 0
        )
        answers = tuple(
            answer_lookup[state[query_key_by_token[token]]] for token in query_tokens
        )
        full_rows.append((state, OpaqueDiagnosticRow(word, answers)))
    passive_rows = tuple(
        sorted(
            (
                row
                for state, row in full_rows
                if state[omitted[0]] != omitted[1]
            ),
            key=lambda row: row.word,
        )
    )
    candidates = tuple((token,) for token in sorted(event_pool))
    passive_hash = _sha256(tuple(_row_payload(row) for row in passive_rows))
    candidate_hash = _sha256(candidates)
    input_hash = _sha256(
        {
            "schema": _INPUT_SCHEMA,
            "scope": OpaqueDiagnosticScope.ZERO_SUFFIX_ABSENCE_DIAGNOSTIC_BLOCK,
            "vocabulary_contract": (
                OpaqueVocabularyContract.DISTINCT_NONEMPTY_QUERY_ANSWER_SLOTS
            ),
            "event_tokens": tuple(sorted(event_pool)),
            "query_tokens": query_tokens,
            "answer_tokens": tuple(sorted(answer_pool)),
            "passive_rows": tuple(_row_payload(row) for row in passive_rows),
            "candidate_words": candidates,
            "budgets": selected_budgets.payload(),
            "passive_table_sha256": passive_hash,
            "candidate_pool_sha256": candidate_hash,
        }
    )
    learner_input = OpaqueDiagnosticInput(
        scope=OpaqueDiagnosticScope.ZERO_SUFFIX_ABSENCE_DIAGNOSTIC_BLOCK,
        vocabulary_contract=(
            OpaqueVocabularyContract.DISTINCT_NONEMPTY_QUERY_ANSWER_SLOTS
        ),
        event_tokens=tuple(sorted(event_pool)),
        query_tokens=query_tokens,
        answer_tokens=tuple(sorted(answer_pool)),
        passive_rows=passive_rows,
        candidate_words=candidates,
        budgets=selected_budgets,
        passive_table_sha256=passive_hash,
        candidate_pool_sha256=candidate_hash,
        input_sha256=input_hash,
    )
    material = _ControllerMaterial(
        controller_nonce=nonce,
        omitted_cell=omitted,
        relabel_block=block,
        event_by_cell=event_by_cell,
        query_by_key=query_by_key,
        answer_by_value=answer_by_value,
        history_key_order=history_key_order,
        full_rows=tuple(full_rows),
    )
    controller_hash = _sha256(_controller_payload(material, learner_input.input_sha256))
    return ToyControllerEnvironment(
        learner_input=learner_input,
        controller_commitment_sha256=controller_hash,
        _material=material,
    )


def release_committed_membership_answer(
    controller: ToyControllerEnvironment,
    passive: PassiveDiscoveryResult,
) -> OpaqueMembershipAnswer:
    """Release one two-query response after validating the answer-free commit."""

    if type(controller) is not ToyControllerEnvironment:
        raise TypeError("controller must be exact ToyControllerEnvironment")
    if type(passive) is not PassiveDiscoveryResult:
        raise TypeError("passive must be exact PassiveDiscoveryResult")
    expected_passive = fit_passive_opaque_hankel(controller.learner_input)
    if passive != expected_passive:
        raise ValueError("passive result does not match the controller input")
    event_lookup = dict(controller._material.event_by_cell)
    expected_word = (event_lookup[controller._material.omitted_cell],)
    if passive.commitment.selected_word != expected_word:
        raise ValueError("opaque acquisition did not select the omitted atomic token")
    row = next(
        row
        for state, row in controller._material.full_rows
        if state[controller._material.omitted_cell[0]]
        == controller._material.omitted_cell[1]
        and state[1 - controller._material.omitted_cell[0]] == -1
    )
    if row.word != expected_word:
        raise AssertionError("controller singleton representative is inconsistent")
    payload = {
        "schema": _ANSWER_SCHEMA,
        "input_sha256": controller.learner_input.input_sha256,
        "commitment_sha256": passive.commitment.commitment_sha256,
        "word": row.word,
        "answers": row.answers,
        "response_ordinal": 1,
    }
    return OpaqueMembershipAnswer(
        input_sha256=controller.learner_input.input_sha256,
        commitment_sha256=passive.commitment.commitment_sha256,
        word=row.word,
        answers=row.answers,
        response_ordinal=1,
        answer_sha256=_sha256(payload),
    )


@dataclass(frozen=True)
class ToyOpaqueEnvironmentResult:
    """Authoritative reconstruction around two sanitized learner calls.

    Nested result hashes are context-bound links; this enclosing object reruns
    their learner/controller derivations before binding the environment.
    """

    environment_index: int
    pseudoheldout_cell: SemanticCell
    relabel_block: int
    learner_input: OpaqueDiagnosticInput
    controller_commitment_sha256: str
    passive: PassiveDiscoveryResult
    active_answer: OpaqueMembershipAnswer
    postactive: PostactiveDiscoveryResult
    _controller_nonce: str = field(repr=False, compare=False)
    environment_sha256: str
    schema: str = _ENVIRONMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _ENVIRONMENT_SCHEMA:
            raise ValueError("unknown toy opaque environment result schema")
        index = _plain_int("environment_index", self.environment_index, 0)
        cell = _normalize_semantic_cell(self.pseudoheldout_cell)
        block = _plain_int("relabel_block", self.relabel_block, 0)
        nonce = _require_controller_nonce(self._controller_nonce)
        if index != block * len(_SEMANTIC_CELLS) + _SEMANTIC_CELLS.index(cell):
            raise ValueError("environment index disagrees with rotation and block")
        if type(self.learner_input) is not OpaqueDiagnosticInput:
            raise TypeError("learner_input must be exact OpaqueDiagnosticInput")
        expected_controller = build_toy_controller_environment(
            cell,
            block,
            controller_nonce=nonce,
            budgets=self.learner_input.budgets,
        )
        if (
            self.learner_input != expected_controller.learner_input
            or self.controller_commitment_sha256
            != expected_controller.controller_commitment_sha256
        ):
            raise ValueError("environment input disagrees with deterministic controller")
        expected_passive = fit_passive_opaque_hankel(self.learner_input)
        if self.passive != expected_passive:
            raise ValueError("environment passive artifact is not reproducible")
        expected_answer = release_committed_membership_answer(
            expected_controller, expected_passive
        )
        if self.active_answer != expected_answer:
            raise ValueError("environment active answer is not the committed response")
        expected_postactive = fit_postactive_opaque_hankel(
            self.learner_input, expected_passive, expected_answer
        )
        if self.postactive != expected_postactive:
            raise ValueError("environment postactive artifact is not reproducible")
        expected_hash = _sha256(_environment_payload(self))
        if _require_sha256("environment_sha256", self.environment_sha256) != expected_hash:
            raise ValueError("environment_sha256 does not bind the environment")

    @property
    def passed(self) -> bool:
        return (
            self.passive.exact_rank == _TOY_PASSIVE_RANK
            and self.passive.ambiguity_dimension == 1
            and self.postactive.exact_rank == _TOY_FULL_RANK
            and self.postactive.full_diagnostic_count == _TOY_FULL_ROW_COUNT
            and self.passive.commitment.preactive_membership_calls == 0
            and self.postactive.active_membership_calls == 1
        )


def _environment_payload(value: ToyOpaqueEnvironmentResult) -> dict[str, object]:
    return {
        "schema": value.schema,
        "environment_index": value.environment_index,
        "pseudoheldout_cell": value.pseudoheldout_cell,
        "relabel_block": value.relabel_block,
        "learner_input_sha256": value.learner_input.input_sha256,
        "controller_commitment_sha256": value.controller_commitment_sha256,
        "passive_result_sha256": value.passive.result_sha256,
        "active_answer_sha256": value.active_answer.answer_sha256,
        "postactive_result_sha256": value.postactive.result_sha256,
        "controller_nonce_sha256": _sha256(value._controller_nonce),
    }


def _run_toy_environment(
    pseudoheldout_cell: SemanticCell,
    relabel_block: int,
    controller_nonce: str,
    budgets: OpaqueHankelBudgets,
) -> ToyOpaqueEnvironmentResult:
    controller = build_toy_controller_environment(
        pseudoheldout_cell,
        relabel_block,
        controller_nonce=controller_nonce,
        budgets=budgets,
    )
    passive = fit_passive_opaque_hankel(controller.learner_input)
    answer = release_committed_membership_answer(controller, passive)
    postactive = fit_postactive_opaque_hankel(
        controller.learner_input, passive, answer
    )
    environment_index = relabel_block * len(_SEMANTIC_CELLS) + _SEMANTIC_CELLS.index(
        pseudoheldout_cell
    )
    payload = {
        "schema": _ENVIRONMENT_SCHEMA,
        "environment_index": environment_index,
        "pseudoheldout_cell": pseudoheldout_cell,
        "relabel_block": relabel_block,
        "learner_input_sha256": controller.learner_input.input_sha256,
        "controller_commitment_sha256": controller.controller_commitment_sha256,
        "passive_result_sha256": passive.result_sha256,
        "active_answer_sha256": answer.answer_sha256,
        "postactive_result_sha256": postactive.result_sha256,
        "controller_nonce_sha256": _sha256(controller_nonce),
    }
    return ToyOpaqueEnvironmentResult(
        environment_index=environment_index,
        pseudoheldout_cell=pseudoheldout_cell,
        relabel_block=relabel_block,
        learner_input=controller.learner_input,
        controller_commitment_sha256=controller.controller_commitment_sha256,
        passive=passive,
        active_answer=answer,
        postactive=postactive,
        _controller_nonce=controller_nonce,
        environment_sha256=_sha256(payload),
    )


def _controller_answers_for_state(
    controller: ToyControllerEnvironment, state: SemanticState
) -> tuple[str, ...]:
    return next(row.answers for candidate, row in controller._material.full_rows if candidate == state)


def _state_coordinate(
    controller: ToyControllerEnvironment,
    environment: ToyOpaqueEnvironmentResult,
    state: SemanticState,
) -> RationalVector:
    answers = _controller_answers_for_state(controller, state)
    return _coordinates(
        environment.postactive.model,
        _diagnostic_vector(environment.learner_input, answers),
    )


def _aligned_readout_matrix(
    controller: ToyControllerEnvironment,
    environment: ToyOpaqueEnvironmentResult,
) -> RationalMatrix:
    query_lookup = dict(controller._material.query_by_key)
    answer_lookup = dict(controller._material.answer_by_value)
    column_indices: list[int] = []
    for key in range(_TOY_QUERY_COUNT):
        query_position = environment.learner_input.query_tokens.index(query_lookup[key])
        for value in (-1, 0, 1):
            answer_position = environment.learner_input.answer_tokens.index(
                answer_lookup[value]
            )
            column_indices.append(
                query_position * len(environment.learner_input.answer_tokens)
                + answer_position
            )
    return tuple(
        tuple(row[column] for column in column_indices)
        for row in environment.postactive.model.readout_matrix
    )


@dataclass(frozen=True)
class RationalSimilarityCertificate:
    """One context-bound exact gauge map between paired opaque relabelings.

    This row alone does not prove its environment attribution.  The enclosing
    :class:`ToyOpaqueHankelReport` reconstructs the map from both authoritative
    environment results and is the evidence boundary.
    """

    pseudoheldout_cell: SemanticCell
    environment_a_sha256: str
    environment_b_sha256: str
    similarity_matrix: RationalMatrix
    determinant: Rational
    fit_state_set_sha256: str
    test_state_set_sha256: str
    fit_equation_count: int
    test_equation_count: int
    certificate_sha256: str
    schema: str = _SIMILARITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SIMILARITY_SCHEMA:
            raise ValueError("unknown rational similarity certificate schema")
        _normalize_semantic_cell(self.pseudoheldout_cell)
        _require_sha256("environment_a_sha256", self.environment_a_sha256)
        _require_sha256("environment_b_sha256", self.environment_b_sha256)
        rows, columns = _matrix_shape(self.similarity_matrix)
        if rows != columns or rows != _TOY_FULL_RANK:
            raise ValueError("toy similarity matrix must be exact 5x5")
        if type(self.determinant) is not Rational:
            raise TypeError("similarity determinant must be exact Rational")
        if self.determinant.numerator == 0 or self.determinant != _determinant(
            self.similarity_matrix
        ):
            raise ValueError("similarity map is singular or determinant is wrong")
        _require_sha256("fit_state_set_sha256", self.fit_state_set_sha256)
        _require_sha256("test_state_set_sha256", self.test_state_set_sha256)
        if _plain_int("fit_equation_count", self.fit_equation_count, 1) != len(
            _SIMILARITY_FIT_STATES
        ):
            raise ValueError("similarity fit count is not the frozen five states")
        if _plain_int("test_equation_count", self.test_equation_count, 1) != len(
            _SIMILARITY_TEST_STATES
        ):
            raise ValueError("similarity test count is not the disjoint four states")
        expected_hash = _sha256(_similarity_payload(self))
        if _require_sha256("certificate_sha256", self.certificate_sha256) != expected_hash:
            raise ValueError("similarity certificate hash mismatch")


def _similarity_payload(value: RationalSimilarityCertificate) -> dict[str, object]:
    return {
        "schema": value.schema,
        "pseudoheldout_cell": value.pseudoheldout_cell,
        "environment_a_sha256": value.environment_a_sha256,
        "environment_b_sha256": value.environment_b_sha256,
        "similarity_matrix": value.similarity_matrix,
        "determinant": value.determinant,
        "fit_state_set_sha256": value.fit_state_set_sha256,
        "test_state_set_sha256": value.test_state_set_sha256,
        "fit_equation_count": value.fit_equation_count,
        "test_equation_count": value.test_equation_count,
    }


def _make_similarity_certificate(
    environment_a: ToyOpaqueEnvironmentResult,
    environment_b: ToyOpaqueEnvironmentResult,
) -> RationalSimilarityCertificate:
    if (
        environment_a.pseudoheldout_cell != environment_b.pseudoheldout_cell
        or environment_a.relabel_block != 0
        or environment_b.relabel_block != 1
    ):
        raise ValueError("similarity requires paired blocks of one rotation")
    cell = environment_a.pseudoheldout_cell
    controller_a = build_toy_controller_environment(
        cell,
        0,
        controller_nonce=environment_a._controller_nonce,
        budgets=environment_a.learner_input.budgets,
    )
    controller_b = build_toy_controller_environment(
        cell,
        1,
        controller_nonce=environment_b._controller_nonce,
        budgets=environment_b.learner_input.budgets,
    )
    fit_a = tuple(
        _state_coordinate(controller_a, environment_a, state)
        for state in _SIMILARITY_FIT_STATES
    )
    fit_b = tuple(
        _state_coordinate(controller_b, environment_b, state)
        for state in _SIMILARITY_FIT_STATES
    )
    fit_a_matrix = tuple(fit_a)
    fit_b_matrix = tuple(fit_b)
    if _determinant(fit_a_matrix).numerator == 0 or _determinant(
        fit_b_matrix
    ).numerator == 0:
        raise ValueError("frozen similarity fit states are not independent")
    similarity = _matmul(_inverse(fit_a_matrix), fit_b_matrix)
    for state in _FULL_SEMANTIC_STATES:
        left = _state_coordinate(controller_a, environment_a, state)
        right = _state_coordinate(controller_b, environment_b, state)
        if _row_times_matrix(left, similarity) != right:
            raise ValueError("one similarity map does not align all diagnostic states")
    aligned_a = _aligned_readout_matrix(controller_a, environment_a)
    aligned_b = _aligned_readout_matrix(controller_b, environment_b)
    if _matmul(similarity, aligned_b) != aligned_a:
        raise ValueError("one similarity map does not align the opaque readouts")
    determinant = _determinant(similarity)
    fit_hash = _sha256(_SIMILARITY_FIT_STATES)
    test_hash = _sha256(_SIMILARITY_TEST_STATES)
    provisional = {
        "schema": _SIMILARITY_SCHEMA,
        "pseudoheldout_cell": cell,
        "environment_a_sha256": environment_a.environment_sha256,
        "environment_b_sha256": environment_b.environment_sha256,
        "similarity_matrix": similarity,
        "determinant": determinant,
        "fit_state_set_sha256": fit_hash,
        "test_state_set_sha256": test_hash,
        "fit_equation_count": len(_SIMILARITY_FIT_STATES),
        "test_equation_count": len(_SIMILARITY_TEST_STATES),
    }
    return RationalSimilarityCertificate(
        pseudoheldout_cell=cell,
        environment_a_sha256=environment_a.environment_sha256,
        environment_b_sha256=environment_b.environment_sha256,
        similarity_matrix=similarity,
        determinant=determinant,
        fit_state_set_sha256=fit_hash,
        test_state_set_sha256=test_hash,
        fit_equation_count=len(_SIMILARITY_FIT_STATES),
        test_equation_count=len(_SIMILARITY_TEST_STATES),
        certificate_sha256=_sha256(provisional),
    )


@dataclass(frozen=True)
class ToyOpaqueHankelReport:
    """All four pseudoheldout rotations under two opaque relabel blocks."""

    status: OpaqueExperimentStatus
    scope: OpaqueDiagnosticScope
    budgets: OpaqueHankelBudgets
    environments: tuple[ToyOpaqueEnvironmentResult, ...]
    similarities: tuple[RationalSimilarityCertificate, ...]
    report_sha256: str
    schema: str = _REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _REPORT_SCHEMA:
            raise ValueError("unknown toy opaque Hankel report schema")
        if self.status is not (
            OpaqueExperimentStatus.SYNTHETIC_PROTOCOL_IMPLEMENTATION_REHEARSAL
        ):
            raise ValueError("toy report must retain its closed rehearsal status")
        if self.scope is not OpaqueDiagnosticScope.ZERO_SUFFIX_ABSENCE_DIAGNOSTIC_BLOCK:
            raise ValueError("toy report may claim only the zero-suffix diagnostic block")
        if type(self.budgets) is not OpaqueHankelBudgets:
            raise TypeError("budgets must be exact OpaqueHankelBudgets")
        if (
            not isinstance(self.environments, tuple)
            or len(self.environments) != len(_SEMANTIC_CELLS) * _TOY_RELABEL_BLOCKS
            or any(type(row) is not ToyOpaqueEnvironmentResult for row in self.environments)
        ):
            raise ValueError("report must contain all eight exact environments")
        if tuple(row.environment_index for row in self.environments) != tuple(range(8)):
            raise ValueError("environment rows must be in frozen index order")
        expected_inventory = tuple(
            (block, cell)
            for block in range(_TOY_RELABEL_BLOCKS)
            for cell in _SEMANTIC_CELLS
        )
        actual_inventory = tuple(
            (row.relabel_block, row.pseudoheldout_cell) for row in self.environments
        )
        if actual_inventory != expected_inventory:
            raise ValueError("pseudoheldout rotation/relabel inventory is incomplete")
        if any(row.learner_input.budgets != self.budgets for row in self.environments):
            raise ValueError("environment budgets disagree with report budgets")
        if any(not row.passed for row in self.environments):
            raise ValueError("every toy environment must pass exactly")
        input_hashes = tuple(row.learner_input.input_sha256 for row in self.environments)
        if len(set(input_hashes)) != len(input_hashes):
            raise ValueError("every opaque environment input must be independently relabeled")
        nonce_hashes = tuple(
            _sha256(row._controller_nonce) for row in self.environments
        )
        if len(set(nonce_hashes)) != len(nonce_hashes):
            raise ValueError("every environment needs distinct controller relabel material")
        all_token_sets = tuple(
            frozenset(
                row.learner_input.event_tokens
                + row.learner_input.query_tokens
                + row.learner_input.answer_tokens
            )
            for row in self.environments
        )
        if any(
            all_token_sets[left] & all_token_sets[right]
            for left in range(len(all_token_sets))
            for right in range(left + 1, len(all_token_sets))
        ):
            raise ValueError("opaque identifiers must be fresh in every environment")
        if (
            not isinstance(self.similarities, tuple)
            or len(self.similarities) != len(_SEMANTIC_CELLS)
            or any(
                type(row) is not RationalSimilarityCertificate
                for row in self.similarities
            )
        ):
            raise ValueError("report must contain four exact similarity certificates")
        by_key = {
            (row.relabel_block, row.pseudoheldout_cell): row
            for row in self.environments
        }
        expected_similarities = tuple(
            _make_similarity_certificate(by_key[(0, cell)], by_key[(1, cell)])
            for cell in _SEMANTIC_CELLS
        )
        if self.similarities != expected_similarities:
            raise ValueError("similarity certificates do not reconstruct from environments")
        expected_hash = _sha256(_report_payload(self))
        if _require_sha256("report_sha256", self.report_sha256) != expected_hash:
            raise ValueError("report_sha256 does not bind the complete report")

    @property
    def passed(self) -> bool:
        return len(self.environments) == 8 and all(row.passed for row in self.environments)

    @property
    def passive_ranks(self) -> tuple[int, ...]:
        return tuple(row.passive.exact_rank for row in self.environments)

    @property
    def postactive_ranks(self) -> tuple[int, ...]:
        return tuple(row.postactive.exact_rank for row in self.environments)


def _report_payload(value: ToyOpaqueHankelReport) -> dict[str, object]:
    return {
        "schema": value.schema,
        "status": value.status,
        "scope": value.scope,
        "budgets": value.budgets.payload(),
        "environment_sha256s": tuple(
            row.environment_sha256 for row in value.environments
        ),
        "similarity_sha256s": tuple(
            row.certificate_sha256 for row in value.similarities
        ),
    }


def run_toy_opaque_hankel_experiment(
    *,
    controller_nonces: tuple[str, ...],
    budgets: OpaqueHankelBudgets | None = None,
) -> ToyOpaqueHankelReport:
    """Run deterministically under eight explicit trusted-controller nonces.

    Replaying with the same nonce tuple is byte/content deterministic.  The
    nonce tuple is controller-side relabel material and is never copied into a
    learner input or learner-artifact hash payload.
    """

    selected_budgets = OpaqueHankelBudgets() if budgets is None else budgets
    if type(selected_budgets) is not OpaqueHankelBudgets:
        raise TypeError("budgets must be exact OpaqueHankelBudgets or None")
    _check_controller_budgets(selected_budgets)
    if type(controller_nonces) is not tuple:
        raise TypeError("controller_nonces must be an exact tuple")
    if len(controller_nonces) != len(_SEMANTIC_CELLS) * _TOY_RELABEL_BLOCKS:
        raise ValueError("controller_nonces must be an exact eight-item tuple")
    normalized_nonces = tuple(
        _require_controller_nonce(nonce) for nonce in controller_nonces
    )
    if len(set(normalized_nonces)) != len(normalized_nonces):
        raise ValueError("controller_nonces must be distinct")
    environments = tuple(
        _run_toy_environment(
            cell,
            block,
            normalized_nonces[block * len(_SEMANTIC_CELLS) + cell_index],
            selected_budgets,
        )
        for block in range(_TOY_RELABEL_BLOCKS)
        for cell_index, cell in enumerate(_SEMANTIC_CELLS)
    )
    by_key = {
        (row.relabel_block, row.pseudoheldout_cell): row for row in environments
    }
    similarities = tuple(
        _make_similarity_certificate(by_key[(0, cell)], by_key[(1, cell)])
        for cell in _SEMANTIC_CELLS
    )
    provisional = {
        "schema": _REPORT_SCHEMA,
        "status": OpaqueExperimentStatus.SYNTHETIC_PROTOCOL_IMPLEMENTATION_REHEARSAL,
        "scope": OpaqueDiagnosticScope.ZERO_SUFFIX_ABSENCE_DIAGNOSTIC_BLOCK,
        "budgets": selected_budgets.payload(),
        "environment_sha256s": tuple(row.environment_sha256 for row in environments),
        "similarity_sha256s": tuple(
            row.certificate_sha256 for row in similarities
        ),
    }
    return ToyOpaqueHankelReport(
        status=OpaqueExperimentStatus.SYNTHETIC_PROTOCOL_IMPLEMENTATION_REHEARSAL,
        scope=OpaqueDiagnosticScope.ZERO_SUFFIX_ABSENCE_DIAGNOSTIC_BLOCK,
        budgets=selected_budgets,
        environments=environments,
        similarities=similarities,
        report_sha256=_sha256(provisional),
    )


__all__ = [
    "ExactDiagnosticModel",
    "ExactRankCertificate",
    "FullDiagnosticCoordinate",
    "OpaqueAcquisitionCommitment",
    "OpaqueDiagnosticInput",
    "OpaqueDiagnosticRow",
    "OpaqueDiagnosticScope",
    "OpaqueExperimentStatus",
    "OpaqueHankelBudgets",
    "OpaqueMembershipAnswer",
    "OpaquePredictiveStateLimitError",
    "OpaqueVocabularyContract",
    "OpaqueVocabularyCoverage",
    "PassiveCompletionWitness",
    "PassiveDiscoveryResult",
    "PostactiveDiscoveryResult",
    "Rational",
    "RationalSimilarityCertificate",
    "ToyControllerEnvironment",
    "ToyOpaqueEnvironmentResult",
    "ToyOpaqueHankelReport",
    "build_toy_controller_environment",
    "fit_passive_opaque_hankel",
    "fit_postactive_opaque_hankel",
    "release_committed_membership_answer",
    "run_toy_opaque_hankel_experiment",
]
