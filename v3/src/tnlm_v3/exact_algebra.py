"""Exact finite algebra for the dynamic-binding benchmark.

This module deliberately separates three objects which are easy to conflate:

* the strict, guarded event grammar used by the generator;
* a total affine completion that agrees with every grammar-valid history; and
* diagnostic feature matrices, which are not called Hankel matrices.

All arithmetic is integer or exact finite-field/rational arithmetic.  Nothing
in this module estimates a rank with floating point tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
import math
from typing import Iterable, Sequence

from .data import (
    BindingEpisode,
    BindingEventKind,
    BindingTaskConfig,
    IGNORE_QUERY_TARGET,
)


IntegerMatrix = tuple[tuple[int, ...], ...]
DEFAULT_RANK_PRIMES = (1_000_000_007, 1_000_000_009)


class EnumerationLimitError(RuntimeError):
    """Raised before an exact enumeration would exceed its declared budget."""


class IllegalActionError(ValueError):
    """Raised for an action outside a guarded non-sink contract."""


def _plain_int(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class BindingAlgebraSpec:
    """The task fields that alter its exact semantic algebra."""

    num_surface_keys: int
    value_cardinality: int
    max_live_bindings: int
    heldout_key_value_pairs: tuple[tuple[int, int], ...] = ()
    branches: int | None = None

    def __post_init__(self) -> None:
        _plain_int("num_surface_keys", self.num_surface_keys, 1)
        _plain_int("value_cardinality", self.value_cardinality, 2)
        _plain_int("max_live_bindings", self.max_live_bindings, 0)
        if self.max_live_bindings > self.num_surface_keys:
            raise ValueError("max_live_bindings cannot exceed num_surface_keys")
        normalized: list[tuple[int, int]] = []
        for pair in self.heldout_key_value_pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError("held-out pairs must be two-integer tuples")
            key, value = pair
            _plain_int("held-out key", key, 0)
            _plain_int("held-out value", value, 0)
            if key >= self.num_surface_keys or value >= self.value_cardinality:
                raise ValueError("held-out pair is outside the algebra vocabulary")
            normalized.append((key, value))
        if len(set(normalized)) != len(normalized):
            raise ValueError("held-out pairs must be unique")
        object.__setattr__(self, "heldout_key_value_pairs", tuple(normalized))
        branch_count = (
            self.max_live_bindings if self.branches is None else self.branches
        )
        _plain_int("branches", branch_count, 1)
        if branch_count < self.max_live_bindings:
            raise ValueError("branches cannot be smaller than max_live_bindings")
        object.__setattr__(self, "branches", branch_count)

    @classmethod
    def from_task(cls, task: BindingTaskConfig) -> "BindingAlgebraSpec":
        if not isinstance(task, BindingTaskConfig):
            raise TypeError("task must be BindingTaskConfig")
        return cls(
            num_surface_keys=task.num_surface_keys,
            value_cardinality=task.value_cardinality,
            max_live_bindings=task.max_live_bindings,
            heldout_key_value_pairs=task.heldout_key_value_pairs,
            branches=task.branches,
        )


ConfigLike = BindingAlgebraSpec | BindingTaskConfig


def _spec(config: ConfigLike) -> BindingAlgebraSpec:
    if isinstance(config, BindingAlgebraSpec):
        return config
    if isinstance(config, BindingTaskConfig):
        return BindingAlgebraSpec.from_task(config)
    raise TypeError("config must be BindingAlgebraSpec or BindingTaskConfig")


@dataclass(frozen=True, order=True)
class SemanticState:
    """A key-indexed memory; ``-1`` denotes an absent binding."""

    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            raise TypeError("values must be a tuple")
        for value in self.values:
            _plain_int("semantic value", value, -1)

    @property
    def live_count(self) -> int:
        return sum(value >= 0 for value in self.values)

    @property
    def active_keys(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.values) if value >= 0)


def _validate_state(
    spec: BindingAlgebraSpec, state: SemanticState, *, enforce_cap: bool = True
) -> None:
    if not isinstance(state, SemanticState):
        raise TypeError("state must be SemanticState")
    if len(state.values) != spec.num_surface_keys:
        raise ValueError("state has the wrong number of keys")
    if any(value >= spec.value_cardinality for value in state.values):
        raise ValueError("state contains a value outside the vocabulary")
    if enforce_cap and state.live_count > spec.max_live_bindings:
        raise ValueError("state exceeds the live-binding cap")


@dataclass(frozen=True, order=True)
class AlgebraAction:
    """One canonical model-visible event signature, using raw zero-based IDs."""

    kind: BindingEventKind
    primary_key: int = -1
    secondary_key: int = -1
    argument: int = -1

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BindingEventKind):
            try:
                object.__setattr__(self, "kind", BindingEventKind(self.kind))
            except (TypeError, ValueError) as error:
                raise TypeError("kind must be BindingEventKind") from error
        for name in ("primary_key", "secondary_key", "argument"):
            _plain_int(name, getattr(self, name), -1)

    @classmethod
    def bind(cls, key: int, value: int) -> "AlgebraAction":
        return cls(BindingEventKind.BIND, primary_key=key, argument=value)

    @classmethod
    def update(cls, key: int, transform: int) -> "AlgebraAction":
        return cls(BindingEventKind.UPDATE, primary_key=key, argument=transform)

    @classmethod
    def copy(cls, destination: int, source: int) -> "AlgebraAction":
        return cls(
            BindingEventKind.COPY,
            primary_key=destination,
            secondary_key=source,
        )

    @classmethod
    def invalidate(cls, key: int) -> "AlgebraAction":
        return cls(BindingEventKind.INVALIDATE, primary_key=key)

    @classmethod
    def query(cls, key: int) -> "AlgebraAction":
        return cls(BindingEventKind.QUERY, primary_key=key)

    @classmethod
    def distractor(cls, scope: int) -> "AlgebraAction":
        return cls(BindingEventKind.DISTRACTOR, argument=scope)


# Descriptive alias used by the public research API.
VisibleAction = AlgebraAction


class AlgebraContract(str, Enum):
    """Named observable contracts for guarded semantic transitions."""

    STRICT = "strict"
    STRICT_GRAMMAR = "strict"
    ABSENCE_AWARE = "absence_aware"
    PROMISED = "promised_valid_query"
    PROMISED_VALID_QUERY = "promised_valid_query"


@dataclass(frozen=True)
class ActionResult:
    defined: bool
    state: SemanticState | None
    query_target: int | None = None

    @property
    def legal(self) -> bool:
        return self.defined

    @property
    def dead(self) -> bool:
        return not self.defined and self.state is None


def _validate_action(spec: BindingAlgebraSpec, action: AlgebraAction) -> None:
    if not isinstance(action, AlgebraAction):
        raise TypeError("action must be AlgebraAction")
    key_ok = 0 <= action.primary_key < spec.num_surface_keys
    source_ok = 0 <= action.secondary_key < spec.num_surface_keys
    argument_ok = 0 <= action.argument < spec.value_cardinality
    if action.kind in (BindingEventKind.BIND, BindingEventKind.UPDATE):
        valid = key_ok and action.secondary_key == -1 and argument_ok
    elif action.kind is BindingEventKind.COPY:
        valid = (
            key_ok
            and source_ok
            and action.primary_key != action.secondary_key
            and action.argument == -1
        )
    elif action.kind in (BindingEventKind.INVALIDATE, BindingEventKind.QUERY):
        valid = key_ok and action.secondary_key == -1 and action.argument == -1
    elif action.kind is BindingEventKind.DISTRACTOR:
        valid = (
            action.primary_key == -1
            and action.secondary_key == -1
            and action.argument in (0, 1)
        )
    else:
        valid = False
    if not valid:
        raise ValueError("action is not a canonical visible event")


def semantic_state_count(config: ConfigLike) -> int:
    """Number of capped key/value memories, excluding an error sink."""

    spec = _spec(config)
    return sum(
        math.comb(spec.num_surface_keys, live) * spec.value_cardinality**live
        for live in range(spec.max_live_bindings + 1)
    )


def train_semantic_state_count(config: ConfigLike) -> int:
    """Count states containing none of the configured held-out key/value pairs."""

    spec = _spec(config)
    heldout_per_key = [0] * spec.num_surface_keys
    for key, _ in spec.heldout_key_value_pairs:
        heldout_per_key[key] += 1
    choices = [spec.value_cardinality - count for count in heldout_per_key]
    coefficients = [1] + [0] * spec.max_live_bindings
    for active_choices in choices:
        for degree in range(spec.max_live_bindings, 0, -1):
            coefficients[degree] += coefficients[degree - 1] * active_choices
    return sum(coefficients)


def enumerate_semantic_states(
    config: ConfigLike,
    *,
    exclude_heldout: bool = False,
    max_states: int | None = None,
) -> tuple[SemanticState, ...]:
    """Enumerate capped states in deterministic occupancy/key/value order."""

    spec = _spec(config)
    expected = (
        train_semantic_state_count(spec) if exclude_heldout else semantic_state_count(spec)
    )
    if max_states is not None:
        _plain_int("max_states", max_states, 0)
        if expected > max_states:
            raise EnumerationLimitError(
                f"exact state enumeration requires {expected} states; limit is {max_states}"
            )
    heldout = set(spec.heldout_key_value_pairs) if exclude_heldout else set()
    states: list[SemanticState] = []
    for live in range(spec.max_live_bindings + 1):
        for keys in combinations(range(spec.num_surface_keys), live):
            allowed = [
                tuple(
                    value
                    for value in range(spec.value_cardinality)
                    if (key, value) not in heldout
                )
                for key in keys
            ]
            for active_values in product(*allowed):
                values = [-1] * spec.num_surface_keys
                for key, value in zip(keys, active_values, strict=True):
                    values[key] = value
                states.append(SemanticState(tuple(values)))
    if len(states) != expected:
        raise RuntimeError("analytic and enumerated state counts disagree")
    return tuple(states)


def canonical_visible_actions(
    config: ConfigLike,
    *,
    full_syntactic: bool = False,
    include_identity_updates: bool | None = None,
) -> tuple[AlgebraAction, ...]:
    """Return generator-supported (default) or all syntactically legal events.

    The generator never emits the no-change UPDATE argument ``V-1``.  Setting
    ``full_syntactic=True`` includes those K additional signatures (67 versus
    72 in the validation-screen configuration).
    """

    if include_identity_updates is not None:
        if not isinstance(include_identity_updates, bool):
            raise TypeError("include_identity_updates must be bool or None")
        if full_syntactic and not include_identity_updates:
            raise ValueError("conflicting full-syntactic update flags")
        full_syntactic = include_identity_updates
    spec = _spec(config)
    result: list[AlgebraAction] = []
    result.extend(
        AlgebraAction.bind(key, value)
        for key in range(spec.num_surface_keys)
        for value in range(spec.value_cardinality)
    )
    result.extend(
        AlgebraAction.update(key, transform)
        for key in range(spec.num_surface_keys)
        for transform in range(spec.value_cardinality)
        if full_syntactic or transform != spec.value_cardinality - 1
    )
    result.extend(
        AlgebraAction.copy(destination, source)
        for destination in range(spec.num_surface_keys)
        for source in range(spec.num_surface_keys)
        if destination != source
    )
    result.extend(
        AlgebraAction.invalidate(key) for key in range(spec.num_surface_keys)
    )
    result.extend(AlgebraAction.query(key) for key in range(spec.num_surface_keys))
    result.extend((AlgebraAction.distractor(0), AlgebraAction.distractor(1)))
    return tuple(result)


def canonical_visible_action_count(
    config: ConfigLike,
    *,
    full_syntactic: bool = False,
    include_identity_updates: bool | None = None,
) -> int:
    if include_identity_updates is not None:
        if not isinstance(include_identity_updates, bool):
            raise TypeError("include_identity_updates must be bool or None")
        if full_syntactic and not include_identity_updates:
            raise ValueError("conflicting full-syntactic update flags")
        full_syntactic = include_identity_updates
    spec = _spec(config)
    updates = spec.value_cardinality if full_syntactic else spec.value_cardinality - 1
    return (
        spec.num_surface_keys * spec.value_cardinality
        + spec.num_surface_keys * updates
        + spec.num_surface_keys * (spec.num_surface_keys - 1)
        + 2 * spec.num_surface_keys
        + 2
    )


def action_is_legal(
    config: ConfigLike,
    state: SemanticState,
    action: AlgebraAction,
) -> bool:
    """Return whether an action satisfies the strict generator guard."""

    spec = _spec(config)
    _validate_state(spec, state)
    _validate_action(spec, action)
    values = state.values
    if action.kind is BindingEventKind.BIND:
        return values[action.primary_key] < 0 and (
            state.live_count < spec.max_live_bindings
        )
    if action.kind in (
        BindingEventKind.UPDATE,
        BindingEventKind.INVALIDATE,
        BindingEventKind.QUERY,
    ):
        return values[action.primary_key] >= 0
    if action.kind is BindingEventKind.COPY:
        return (
            values[action.primary_key] >= 0
            and values[action.secondary_key] >= 0
        )
    return True


def apply_action(
    config: ConfigLike,
    state: SemanticState | None,
    action: AlgebraAction,
    *,
    contract: AlgebraContract | str = AlgebraContract.STRICT,
) -> ActionResult:
    """Apply one guarded event under the declared observable contract.

    The strict contract represents illegal histories by ``state=None`` and
    keeps that state absorbing. ``ABSENCE_AWARE`` rejects illegal calls,
    while ``PROMISED`` is the natural total completion on the full register
    product used by the homogeneous and symbolic composition operators.
    """

    spec = _spec(config)
    mode = AlgebraContract(contract)
    _validate_action(spec, action)
    if state is None:
        if mode is AlgebraContract.STRICT:
            return ActionResult(False, None, None)
        raise TypeError("only the strict grammar accepts the dead state")
    if mode is AlgebraContract.PROMISED:
        _validate_state(spec, state, enforce_cap=False)
    else:
        _validate_state(spec, state)
        legal = action_is_legal(spec, state, action)
        if not legal:
            if mode is AlgebraContract.STRICT:
                return ActionResult(False, None, None)
            raise IllegalActionError("action is outside the guarded semantic domain")
    values = list(state.values)
    kind = action.kind
    key = action.primary_key
    source = action.secondary_key

    query_target: int | None = None
    if kind is BindingEventKind.BIND:
        values[key] = action.argument
    elif kind is BindingEventKind.UPDATE:
        if values[key] >= 0:
            values[key] = (
                values[key] + action.argument + 1
            ) % spec.value_cardinality
    elif kind is BindingEventKind.COPY:
        values[key] = values[source]
    elif kind is BindingEventKind.INVALIDATE:
        values[key] = -1
    elif kind is BindingEventKind.QUERY:
        query_target = values[key] if values[key] >= 0 else None
    return ActionResult(True, SemanticState(tuple(values)), query_target)


def full_homogeneous_dimension(config: ConfigLike) -> int:
    """Dimension of the total absent-plus-V-one-hot affine completion."""

    spec = _spec(config)
    return 1 + spec.num_surface_keys * spec.value_cardinality


def promised_query_realization_upper_bound(config: ConfigLike) -> int:
    """Upper bound for query behavior under the generator's validity promise.

    Value zero can share the all-zero code with absence because an absent key is
    never updated, copied from, invalidated, or queried before a fresh bind.
    This is an upper bound, not a minimality claim.
    """

    spec = _spec(config)
    return 1 + spec.num_surface_keys * (spec.value_cardinality - 1)


def encode_homogeneous_state(
    config: ConfigLike, state: SemanticState
) -> tuple[int, ...]:
    spec = _spec(config)
    _validate_state(spec, state, enforce_cap=False)
    vector = [0] * full_homogeneous_dimension(spec)
    vector[0] = 1
    for key, value in enumerate(state.values):
        if value >= 0:
            vector[1 + key * spec.value_cardinality + value] = 1
    return tuple(vector)


def decode_homogeneous_state(
    config: ConfigLike, vector: Sequence[int]
) -> SemanticState:
    spec = _spec(config)
    if len(vector) != full_homogeneous_dimension(spec) or vector[0] != 1:
        raise ValueError("vector is not a homogeneous semantic state")
    values: list[int] = []
    for key in range(spec.num_surface_keys):
        start = 1 + key * spec.value_cardinality
        block = tuple(vector[start : start + spec.value_cardinality])
        if any(value not in (0, 1) for value in block) or sum(block) > 1:
            raise ValueError("homogeneous state blocks must be zero or one-hot")
        values.append(block.index(1) if 1 in block else -1)
    state = SemanticState(tuple(values))
    _validate_state(spec, state, enforce_cap=False)
    return state


def _identity_matrix(size: int) -> list[list[int]]:
    return [[int(row == column) for column in range(size)] for row in range(size)]


def homogeneous_operator(config: ConfigLike, action: AlgebraAction) -> IntegerMatrix:
    """Integer matrix for the total affine completion of one visible action."""

    spec = _spec(config)
    _validate_action(spec, action)
    size = full_homogeneous_dimension(spec)
    matrix = _identity_matrix(size)
    if action.kind in (BindingEventKind.QUERY, BindingEventKind.DISTRACTOR):
        return tuple(tuple(row) for row in matrix)
    key = action.primary_key
    start = 1 + key * spec.value_cardinality
    for row in range(start, start + spec.value_cardinality):
        matrix[row] = [0] * size
    if action.kind is BindingEventKind.BIND:
        matrix[start + action.argument][0] = 1
    elif action.kind is BindingEventKind.UPDATE:
        delta = (action.argument + 1) % spec.value_cardinality
        for old in range(spec.value_cardinality):
            matrix[start + (old + delta) % spec.value_cardinality][start + old] = 1
    elif action.kind is BindingEventKind.COPY:
        source = 1 + action.secondary_key * spec.value_cardinality
        for value in range(spec.value_cardinality):
            matrix[start + value][source + value] = 1
    elif action.kind is BindingEventKind.INVALIDATE:
        pass
    return tuple(tuple(row) for row in matrix)


def apply_integer_matrix(
    matrix: Sequence[Sequence[int]], vector: Sequence[int]
) -> tuple[int, ...]:
    if not matrix:
        if vector:
            raise ValueError("empty matrix requires an empty vector")
        return ()
    width = len(matrix[0])
    if len(vector) != width or any(len(row) != width for row in matrix):
        raise ValueError("matrix must be square and match the vector")
    return tuple(sum(value * vector[column] for column, value in enumerate(row)) for row in matrix)


def multiply_integer_matrices(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]
) -> IntegerMatrix:
    if not left or not right:
        if left or right:
            raise ValueError("cannot multiply an empty and nonempty matrix")
        return ()
    rows = len(left)
    inner = len(right)
    columns = len(right[0])
    if any(len(row) != inner for row in left) or any(
        len(row) != columns for row in right
    ):
        raise ValueError("matrix dimensions are incompatible")
    return tuple(
        tuple(
            sum(left[row][pivot] * right[pivot][column] for pivot in range(inner))
            for column in range(columns)
        )
        for row in range(rows)
    )


class RegisterExpressionKind(str, Enum):
    ABSENT = "absent"
    CONSTANT = "constant"
    SOURCE = "source"


@dataclass(frozen=True, order=True)
class RegisterExpression:
    kind: RegisterExpressionKind
    value: int = -1
    source: int = -1
    offset: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RegisterExpressionKind):
            object.__setattr__(self, "kind", RegisterExpressionKind(self.kind))
        for name in ("value", "source", "offset"):
            _plain_int(name, getattr(self, name), -1 if name != "offset" else 0)

    @classmethod
    def absent(cls) -> "RegisterExpression":
        return cls(RegisterExpressionKind.ABSENT)

    @classmethod
    def constant(cls, value: int) -> "RegisterExpression":
        return cls(RegisterExpressionKind.CONSTANT, value=value)

    @classmethod
    def from_source(cls, source: int, offset: int = 0) -> "RegisterExpression":
        return cls(RegisterExpressionKind.SOURCE, source=source, offset=offset)


@dataclass(frozen=True)
class SegmentTransformer:
    """Canonical K-register normal form for a whole event segment."""

    value_cardinality: int
    outputs: tuple[RegisterExpression, ...]

    def __post_init__(self) -> None:
        _plain_int("value_cardinality", self.value_cardinality, 2)
        if not isinstance(self.outputs, tuple) or not self.outputs:
            raise ValueError("outputs must be a nonempty tuple")
        keys = len(self.outputs)
        for expression in self.outputs:
            if not isinstance(expression, RegisterExpression):
                raise TypeError("outputs must contain RegisterExpression values")
            if expression.kind is RegisterExpressionKind.ABSENT:
                valid = (
                    expression.value == -1
                    and expression.source == -1
                    and expression.offset == 0
                )
            elif expression.kind is RegisterExpressionKind.CONSTANT:
                valid = (
                    0 <= expression.value < self.value_cardinality
                    and expression.source == -1
                    and expression.offset == 0
                )
            else:
                valid = (
                    expression.value == -1
                    and 0 <= expression.source < keys
                    and 0 <= expression.offset < self.value_cardinality
                )
            if not valid:
                raise ValueError("register expression is not canonical")

    @classmethod
    def identity(cls, config: ConfigLike) -> "SegmentTransformer":
        spec = _spec(config)
        return cls(
            spec.value_cardinality,
            tuple(
                RegisterExpression.from_source(key)
                for key in range(spec.num_surface_keys)
            ),
        )

    @classmethod
    def for_action(
        cls, config: ConfigLike, action: AlgebraAction
    ) -> "SegmentTransformer":
        spec = _spec(config)
        _validate_action(spec, action)
        outputs = list(cls.identity(spec).outputs)
        key = action.primary_key
        if action.kind is BindingEventKind.BIND:
            outputs[key] = RegisterExpression.constant(action.argument)
        elif action.kind is BindingEventKind.UPDATE:
            outputs[key] = RegisterExpression.from_source(
                key, (action.argument + 1) % spec.value_cardinality
            )
        elif action.kind is BindingEventKind.COPY:
            outputs[key] = RegisterExpression.from_source(action.secondary_key)
        elif action.kind is BindingEventKind.INVALIDATE:
            outputs[key] = RegisterExpression.absent()
        return cls(spec.value_cardinality, tuple(outputs))

    def compose(self, before: "SegmentTransformer") -> "SegmentTransformer":
        """Return ``self o before`` by exact symbolic substitution."""

        if not isinstance(before, SegmentTransformer):
            raise TypeError("before must be SegmentTransformer")
        if self.value_cardinality != before.value_cardinality or len(
            self.outputs
        ) != len(before.outputs):
            raise ValueError("transformers have incompatible signatures")
        modulus = self.value_cardinality
        result: list[RegisterExpression] = []
        for outer in self.outputs:
            if outer.kind is not RegisterExpressionKind.SOURCE:
                result.append(outer)
                continue
            inner = before.outputs[outer.source]
            if inner.kind is RegisterExpressionKind.ABSENT:
                result.append(inner)
            elif inner.kind is RegisterExpressionKind.CONSTANT:
                result.append(RegisterExpression.constant((inner.value + outer.offset) % modulus))
            else:
                result.append(
                    RegisterExpression.from_source(
                        inner.source, (inner.offset + outer.offset) % modulus
                    )
                )
        return SegmentTransformer(modulus, tuple(result))

    def then(self, following: "SegmentTransformer") -> "SegmentTransformer":
        """Compose chronologically: apply ``self`` and then ``following``."""

        if not isinstance(following, SegmentTransformer):
            raise TypeError("following must be SegmentTransformer")
        return following.compose(self)

    def apply(self, state: SemanticState) -> SemanticState:
        if len(state.values) != len(self.outputs):
            raise ValueError("state and transformer have incompatible key counts")
        values: list[int] = []
        for expression in self.outputs:
            if expression.kind is RegisterExpressionKind.ABSENT:
                values.append(-1)
            elif expression.kind is RegisterExpressionKind.CONSTANT:
                values.append(expression.value)
            else:
                source = state.values[expression.source]
                values.append(
                    -1 if source < 0 else (source + expression.offset) % self.value_cardinality
                )
        return SemanticState(tuple(values))

    def to_homogeneous_operator(self) -> IntegerMatrix:
        keys = len(self.outputs)
        size = 1 + keys * self.value_cardinality
        matrix = _identity_matrix(size)
        for key, expression in enumerate(self.outputs):
            start = 1 + key * self.value_cardinality
            for row in range(start, start + self.value_cardinality):
                matrix[row] = [0] * size
            if expression.kind is RegisterExpressionKind.CONSTANT:
                matrix[start + expression.value][0] = 1
            elif expression.kind is RegisterExpressionKind.SOURCE:
                source = 1 + expression.source * self.value_cardinality
                for old in range(self.value_cardinality):
                    output = start + (
                        old + expression.offset
                    ) % self.value_cardinality
                    matrix[output][source + old] = 1
        return tuple(tuple(row) for row in matrix)

    def to_integer_matrix(self) -> IntegerMatrix:
        """Alias emphasizing that every symbolic coefficient is integral."""

        return self.to_homogeneous_operator()


def transformer_for_actions(
    config: ConfigLike, actions: Iterable[AlgebraAction]
) -> SegmentTransformer:
    result = SegmentTransformer.identity(config)
    for action in actions:
        result = SegmentTransformer.for_action(config, action).compose(result)
    return result


def homogeneous_state_feature_matrix(
    config: ConfigLike,
    *,
    exclude_heldout: bool = False,
    max_states: int | None = None,
) -> IntegerMatrix:
    """Rows of the proposed homogeneous realization coordinates."""

    states = enumerate_semantic_states(
        config, exclude_heldout=exclude_heldout, max_states=max_states
    )
    return tuple(encode_homogeneous_state(config, state) for state in states)


def diagnostic_probe_matrix(
    config: ConfigLike,
    *,
    exclude_heldout: bool = False,
    max_states: int | None = None,
) -> IntegerMatrix:
    """Rows against observable per-key ``ABSENT``/value diagnostics.

    This is the base observable block used by the absence-aware behavioural
    contract, not by itself a full arbitrary-suffix Hankel matrix.
    """

    spec = _spec(config)
    states = enumerate_semantic_states(
        spec, exclude_heldout=exclude_heldout, max_states=max_states
    )
    outcome_count = spec.value_cardinality + 1
    rows: list[tuple[int, ...]] = []
    for state in states:
        row = [0] * (spec.num_surface_keys * outcome_count)
        for key, value in enumerate(state.values):
            row[key * outcome_count + value + 1] = 1
        rows.append(tuple(row))
    return tuple(rows)


@lru_cache(maxsize=None)
def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _rectangular_integer_matrix(
    matrix: Sequence[Sequence[int]],
) -> list[list[int]]:
    rows = [list(row) for row in matrix]
    if not rows:
        return []
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("matrix rows must have equal length")
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("exact rank requires integer entries")
    return rows


def _rational_rank(rows: list[list[int]]) -> int:
    if not rows or not rows[0]:
        return 0
    work = [[Fraction(value) for value in row] for row in rows]
    height, width = len(work), len(work[0])
    rank = 0
    for column in range(width):
        pivot = next(
            (row for row in range(rank, height) if work[row][column] != 0), None
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        for row in range(rank + 1, height):
            if work[row][column] == 0:
                continue
            factor = work[row][column] / pivot_value
            for index in range(column, width):
                work[row][index] -= factor * work[rank][index]
        rank += 1
        if rank == height:
            break
    return rank


def _finite_field_rank(rows: list[list[int]], prime: int) -> int:
    if not rows or not rows[0]:
        return 0
    work = [[value % prime for value in row] for row in rows]
    height, width = len(work), len(work[0])
    rank = 0
    for column in range(width):
        pivot = next(
            (row for row in range(rank, height) if work[row][column]), None
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = pow(work[rank][column], prime - 2, prime)
        for index in range(column, width):
            work[rank][index] = work[rank][index] * inverse % prime
        for row in range(rank + 1, height):
            factor = work[row][column]
            if factor:
                for index in range(column, width):
                    work[row][index] = (
                        work[row][index] - factor * work[rank][index]
                    ) % prime
        rank += 1
        if rank == height:
            break
    return rank


@dataclass(frozen=True)
class ExactRankResult:
    rational_rank: int
    finite_field_ranks: tuple[tuple[int, int], ...]

    @property
    def rank(self) -> int:
        """Concise alias for the exact rational rank."""

        return self.rational_rank

    @property
    def agrees_across_fields(self) -> bool:
        return all(rank == self.rational_rank for _, rank in self.finite_field_ranks)


def exact_matrix_rank(
    matrix: Sequence[Sequence[int]],
    *,
    primes: Sequence[int] = DEFAULT_RANK_PRIMES,
) -> ExactRankResult:
    """Compute exact Q-rank and independent ranks over declared prime fields."""

    rows = _rectangular_integer_matrix(matrix)
    checked: list[int] = []
    for prime in primes:
        _plain_int("rank modulus", prime, 2)
        if not _is_prime(prime):
            raise ValueError("finite-field rank moduli must be prime")
        if prime in checked:
            raise ValueError("finite-field rank moduli must be unique")
        checked.append(prime)
    return ExactRankResult(
        rational_rank=_rational_rank(rows),
        finite_field_ranks=tuple(
            (prime, _finite_field_rank(rows, prime)) for prime in checked
        ),
    )


def diagnostic_probe_rank(
    config: ConfigLike,
    *,
    exclude_heldout: bool = False,
    max_states: int | None = None,
    primes: Sequence[int] = DEFAULT_RANK_PRIMES,
) -> ExactRankResult:
    return exact_matrix_rank(
        diagnostic_probe_matrix(
            config, exclude_heldout=exclude_heldout, max_states=max_states
        ),
        primes=primes,
    )


def strict_grammar_rank_upper_bound(config: ConfigLike) -> int:
    """Structural upper bound for the declared strict diagnostic series.

    For each exact nonempty occupancy set, suffix observations lie in one
    constant direction plus one ``V-1`` contrast family per live key.  Empty
    and sink cells contribute one direction each.
    """

    spec = _spec(config)
    return 2 + sum(
        math.comb(spec.num_surface_keys, live)
        * (1 + live * (spec.value_cardinality - 1))
        for live in range(1, spec.max_live_bindings + 1)
    )


@dataclass(frozen=True)
class StrictGrammarRankCertificate:
    """Bounded finite certificate for the strict diagnostic real rank."""

    rank: int
    semantic_states_with_sink: int
    supported_actions: int
    base_observations: int
    gf2_lower_bound: int
    structural_upper_bound: int
    transition_cell_evaluations: int


def _gf2_insert(basis: dict[int, int], vector: int) -> int | None:
    reduced = vector
    while reduced:
        pivot = reduced.bit_length() - 1
        prior = basis.get(pivot)
        if prior is None:
            basis[pivot] = reduced
            return reduced
        reduced ^= prior
    return None


def _precompose_gf2(vector: int, transition: Sequence[int]) -> int:
    result = 0
    for source, destination in enumerate(transition):
        if (vector >> destination) & 1:
            result |= 1 << source
    return result


def strict_grammar_rank_certificate(
    config: ConfigLike,
    *,
    max_states: int = 2_000,
    max_actions: int = 128,
    max_transition_cell_evaluations: int = 20_000_000,
) -> StrictGrammarRankCertificate:
    """Certify strict rank using generator closure over exact GF(2) bitsets.

    A rank-``r`` integer observation minor that remains nonzero modulo two is
    also nonzero over the rationals, so the closed GF(2) rank is a real-rank
    lower bound.  Matching the independently derived structural upper bound
    certifies equality without relying on a floating-point tolerance.
    """

    spec = _spec(config)
    for name, value in (
        ("max_states", max_states),
        ("max_actions", max_actions),
        ("max_transition_cell_evaluations", max_transition_cell_evaluations),
    ):
        _plain_int(name, value, 1)
    states_with_sink = semantic_state_count(spec) + 1
    actions = canonical_visible_actions(spec)
    upper = strict_grammar_rank_upper_bound(spec)
    if states_with_sink > max_states:
        raise EnumerationLimitError(
            f"strict closure needs {states_with_sink} states; limit is {max_states}"
        )
    if len(actions) > max_actions:
        raise EnumerationLimitError(
            f"strict closure needs {len(actions)} actions; limit is {max_actions}"
        )
    estimated = states_with_sink * len(actions) * upper
    if estimated > max_transition_cell_evaluations:
        raise EnumerationLimitError(
            "strict closure work estimate exceeds limit: "
            f"{estimated} > {max_transition_cell_evaluations}"
        )

    states = enumerate_semantic_states(spec, max_states=max_states - 1)
    state_index = {state: index for index, state in enumerate(states)}
    sink = len(states)
    transitions: list[tuple[int, ...]] = []
    for action in actions:
        mapping: list[int] = []
        for state in states:
            result = apply_action(
                spec, state, action, contract=AlgebraContract.STRICT
            )
            mapping.append(
                sink if result.state is None else state_index[result.state]
            )
        mapping.append(sink)
        transitions.append(tuple(mapping))

    observations: list[int] = [1 << sink]
    for key in range(spec.num_surface_keys):
        for outcome in range(-1, spec.value_cardinality):
            vector = 0
            for index, state in enumerate(states):
                if state.values[key] == outcome:
                    vector |= 1 << index
            observations.append(vector)

    basis: dict[int, int] = {}
    queue: list[int] = []
    for observation in observations:
        inserted = _gf2_insert(basis, observation)
        if inserted is not None:
            queue.append(inserted)
    cursor = 0
    evaluations = 0
    while cursor < len(queue) and len(basis) < upper:
        vector = queue[cursor]
        cursor += 1
        for transition in transitions:
            evaluations += states_with_sink
            if evaluations > max_transition_cell_evaluations:
                raise EnumerationLimitError(
                    "strict closure exhausted its transition-cell budget"
                )
            inserted = _gf2_insert(
                basis, _precompose_gf2(vector, transition)
            )
            if inserted is not None:
                queue.append(inserted)
                if len(basis) > upper:
                    raise AssertionError(
                        "strict closure exceeded its structural upper bound"
                    )
                if len(basis) == upper:
                    break
    lower = len(basis)
    if lower != upper:
        raise RuntimeError(
            "finite closure did not certify the structural upper bound: "
            f"{lower} != {upper}"
        )
    return StrictGrammarRankCertificate(
        rank=upper,
        semantic_states_with_sink=states_with_sink,
        supported_actions=len(actions),
        base_observations=len(observations),
        gf2_lower_bound=lower,
        structural_upper_bound=upper,
        transition_cell_evaluations=evaluations,
    )


def strict_grammar_hankel_rank(
    config: ConfigLike,
    *,
    max_states: int = 2_000,
    max_actions: int = 128,
    max_transition_cell_evaluations: int = 20_000_000,
) -> int:
    """Return the computationally certified strict diagnostic rank."""

    return strict_grammar_rank_certificate(
        config,
        max_states=max_states,
        max_actions=max_actions,
        max_transition_cell_evaluations=max_transition_cell_evaluations,
    ).rank


def oracle_lane_state_count(
    config: ConfigLike,
    branches: int | None = None,
    *,
    exclude_heldout: bool = False,
) -> int:
    """Count raw semantic states after attaching injective oracle lane labels."""

    spec = _spec(config)
    if not isinstance(exclude_heldout, bool):
        raise TypeError("exclude_heldout must be bool")
    branch_count = spec.branches if branches is None else branches
    if branch_count is None:
        raise AssertionError("algebra spec did not normalize branches")
    _plain_int("branches", branch_count, 1)
    if branch_count < spec.max_live_bindings:
        raise ValueError("branches cannot be smaller than max_live_bindings")
    heldout = set(spec.heldout_key_value_pairs) if exclude_heldout else set()
    allowed = tuple(
        spec.value_cardinality
        - sum((key, value) in heldout for value in range(spec.value_cardinality))
        for key in range(spec.num_surface_keys)
    )
    total = 0
    for live in range(spec.max_live_bindings + 1):
        for keys in combinations(range(spec.num_surface_keys), live):
            values = 1
            for key in keys:
                values *= allowed[key]
            total += math.perm(branch_count, live) * values
    return total


def lane_permutation_quotient_state_count(
    config: ConfigLike, *, exclude_heldout: bool = False
) -> int:
    """Oracle lane labels modulo global branch permutation are pure gauge."""

    if not isinstance(exclude_heldout, bool):
        raise TypeError("exclude_heldout must be bool")
    return (
        train_semantic_state_count(config)
        if exclude_heldout
        else semantic_state_count(config)
    )


def _episode_action(episode: BindingEpisode, index: int) -> AlgebraAction:
    return AlgebraAction(
        kind=BindingEventKind(int(episode.inputs.event_kinds[index])),
        primary_key=int(episode.inputs.primary_key_ids[index]) - 1,
        secondary_key=int(episode.inputs.secondary_key_ids[index]) - 1,
        argument=int(episode.inputs.arguments[index]) - 1,
    )


def replay_episode(config: ConfigLike, episode: BindingEpisode) -> SemanticState:
    """Strictly replay one generated episode and verify every query target."""

    spec = _spec(config)
    if not isinstance(episode, BindingEpisode):
        raise TypeError("episode must be BindingEpisode")
    state = SemanticState((-1,) * spec.num_surface_keys)
    for index in range(episode.length):
        if not bool(episode.inputs.valid_mask[index]):
            raise ValueError("an unpadded BindingEpisode cannot contain padding")
        action = _episode_action(episode, index)
        result = apply_action(spec, state, action, contract=AlgebraContract.STRICT)
        if not result.defined or result.state is None:
            raise ValueError(f"episode violates strict grammar at event {index}")
        expected = int(episode.evaluation.targets[index])
        if action.kind is BindingEventKind.QUERY:
            if result.query_target != expected:
                raise ValueError(f"query target mismatch at event {index}")
        elif expected != IGNORE_QUERY_TARGET:
            raise ValueError(f"non-query target is not ignored at event {index}")
        state = result.state
        if state.live_count != int(episode.evaluation.live_binding_counts[index]):
            raise ValueError(f"live-binding count mismatch at event {index}")
    return state


__all__ = [
    "ActionResult",
    "AlgebraAction",
    "AlgebraContract",
    "BindingAlgebraSpec",
    "DEFAULT_RANK_PRIMES",
    "EnumerationLimitError",
    "ExactRankResult",
    "IllegalActionError",
    "RegisterExpression",
    "RegisterExpressionKind",
    "SegmentTransformer",
    "SemanticState",
    "StrictGrammarRankCertificate",
    "VisibleAction",
    "action_is_legal",
    "apply_action",
    "apply_integer_matrix",
    "canonical_visible_action_count",
    "canonical_visible_actions",
    "decode_homogeneous_state",
    "diagnostic_probe_matrix",
    "diagnostic_probe_rank",
    "encode_homogeneous_state",
    "enumerate_semantic_states",
    "exact_matrix_rank",
    "full_homogeneous_dimension",
    "homogeneous_operator",
    "homogeneous_state_feature_matrix",
    "lane_permutation_quotient_state_count",
    "multiply_integer_matrices",
    "oracle_lane_state_count",
    "promised_query_realization_upper_bound",
    "replay_episode",
    "semantic_state_count",
    "strict_grammar_hankel_rank",
    "strict_grammar_rank_certificate",
    "strict_grammar_rank_upper_bound",
    "train_semantic_state_count",
    "transformer_for_actions",
]
