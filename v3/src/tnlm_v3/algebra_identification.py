"""Exact identifiability analysis for held-out binding coordinates.

The validation screen removes one key/value coordinate from every training
state.  This module distinguishes two questions:

* what the complete training transition system identifies without structural
  assumptions; and
* what becomes uniquely determined after declaring an exact symmetry or
  compositional law.

The calculations use the canonical homogeneous realization from
``exact_algebra``.  Coordinate-specific witnesses are therefore statements in
that fixed gauge; the behavioural non-identifiability conclusion is gauge
independent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from typing import Sequence

from .data import BindingTaskConfig
from .exact_algebra import (
    AlgebraAction,
    AlgebraContract,
    BindingAlgebraSpec,
    IntegerMatrix,
    SemanticState,
    apply_integer_matrix,
    apply_action,
    canonical_visible_actions,
    decode_homogeneous_state,
    diagnostic_probe_rank,
    encode_homogeneous_state,
    enumerate_semantic_states,
    exact_matrix_rank,
    full_homogeneous_dimension,
    homogeneous_operator,
    homogeneous_state_feature_matrix,
    multiply_integer_matrices,
    semantic_state_count,
    train_semantic_state_count,
)


ConfigLike = BindingAlgebraSpec | BindingTaskConfig


class IdentificationLimitError(RuntimeError):
    """Raised before an exact identification calculation exceeds its budget."""


@dataclass(frozen=True)
class CyclicCompletionAnalysis:
    """Completions of the partially observed successor action at one key."""

    key: int
    heldout_value: int
    observed_edges: tuple[tuple[int, int], ...]
    unrestricted_function_completions: int
    permutation_completions: tuple[tuple[int, ...], ...]
    transitive_cyclic_completions: tuple[tuple[int, ...], ...]

    @property
    def cyclic_completion_is_unique(self) -> bool:
        return len(self.transitive_cyclic_completions) == 1


@dataclass(frozen=True)
class NonidentifiabilityWitness:
    """Two train-equivalent linear systems that disagree on held-out input."""

    train_states_checked: int
    heldout_state: tuple[int, ...]
    true_query_value: int
    alternative_query_value: int
    update_argument: int
    true_updated_value: int
    alternative_updated_value: int
    readouts_equal_on_all_train_states: bool
    operators_equal_on_all_train_states: bool
    unseen_bind_train_transitions: int
    true_bound_value: int
    alternative_bound_value: int


@dataclass(frozen=True)
class SymmetryRecoveryAnalysis:
    """Canonical identities under supplied universal task symmetries."""

    assumes_universal_exact_operator_identities: bool
    identities_are_inferred_from_train_constraints: bool
    source_key: int
    heldout_key: int
    key_permutation: tuple[int, ...]
    key_equivariance_recovers_bind: bool
    key_equivariance_recovers_update: bool
    key_equivariance_recovers_query_readout: bool
    key_equivariance_recovers_all_copy_operators: bool
    source_value: int
    heldout_value: int
    value_shift: int
    value_equivariance_recovers_bind: bool
    value_equivariance_recovers_query_row: bool
    update_commutes_with_value_shift: bool


@dataclass(frozen=True)
class ActionConstraintAnalysis:
    """Optimistic exact-state constraints for one unrestricted event matrix."""

    kind: str
    primary_key: int
    secondary_key: int
    argument: int
    observed_transitions: int
    source_rank: int
    parameter_count: int
    constraint_rank: int
    nullity: int


@dataclass(frozen=True)
class QueryConstraintAnalysis:
    """Exact-state constraints for one key-specific V-class readout."""

    key: int
    observed_queries: int
    source_rank: int
    parameter_count: int
    constraint_rank: int
    nullity: int


@dataclass(frozen=True)
class UnrestrictedSystemAnalysis:
    """Rank/nullity of the complete train-support linear design."""

    optimistic_latent_states_and_successors_are_observed: bool
    action_count: int
    observed_action_count: int
    observed_transitions: int
    operator_parameter_count: int
    operator_constraint_rank: int
    operator_nullity: int
    query_parameter_count: int
    query_constraint_rank: int
    query_nullity: int
    total_parameter_count: int
    total_constraint_rank: int
    total_nullity: int
    actions: tuple[ActionConstraintAnalysis, ...]
    queries: tuple[QueryConstraintAnalysis, ...]


@dataclass(frozen=True)
class PairCompletionAnalysis:
    """What common key/value completion assumptions identify."""

    pair_count: int
    observed_pair_count: int
    arbitrary_pair_table_rank: int
    arbitrary_pair_table_nullity: int
    additive_parameter_count: int
    additive_design_rank: int
    additive_parameter_gauge_dimension: int
    additive_heldout_behavior_nullity: int
    localized_pair_interaction_witness_exists: bool
    generic_rank_variety_dimensions: tuple[tuple[int, int], ...]
    generic_low_rank_requires_nondegenerate_anchor: bool


@dataclass(frozen=True)
class StructuredHypothesisAnalysis:
    """Exact canonical-block design rank for one sharing hypothesis."""

    hypothesis: str
    parameter_count: int
    constraint_rank: int
    nullity: int
    bind_nullity: int
    update_nullity: int
    copy_nullity: int
    query_nullity: int
    heldout_behavior_identified: bool
    assumes_canonical_local_block_supervision: bool


@dataclass(frozen=True)
class HeldoutIdentificationReport:
    """Canonical exact report for one held-out key/value direction."""

    schema: str
    task_fingerprint: str
    heldout_pair: tuple[int, int]
    homogeneous_dimension: int
    full_state_count: int
    train_state_count: int
    full_feature_rank: int
    train_feature_rank: int
    full_diagnostic_rank: int
    train_diagnostic_rank: int
    missing_feature_indices: tuple[int, ...]
    feature_nullity: int
    missing_column_scalar_readout_nullity: int
    missing_column_multiclass_readout_nullity: int
    unrestricted_operator_nullity_lower_bound_per_action: int
    unrestricted_affine_operator_nullity_lower_bound_per_action: int
    unseen_bind_action_matrix_freedom: int
    unseen_bind_action_affine_freedom: int
    witness: NonidentifiabilityWitness
    unrestricted_system: UnrestrictedSystemAnalysis
    pair_completion: PairCompletionAnalysis
    structured_hypotheses: tuple[StructuredHypothesisAnalysis, ...]
    cyclic_update_completion: CyclicCompletionAnalysis
    symmetry_recovery: SymmetryRecoveryAnalysis
    data_alone_recovers_heldout_direction: bool
    cyclic_update_law_alone_recovers_full_transducer: bool
    declared_universal_operator_equivariance_recovers_heldout_direction: bool
    exact_equivariance_is_supplied_not_inferred: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _spec(config: ConfigLike) -> BindingAlgebraSpec:
    if isinstance(config, BindingAlgebraSpec):
        return config
    if isinstance(config, BindingTaskConfig):
        return BindingAlgebraSpec.from_task(config)
    raise TypeError("config must be BindingAlgebraSpec or BindingTaskConfig")


def _plain_int(name: str, value: int, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def heldout_feature_indices(config: ConfigLike) -> tuple[int, ...]:
    """Return homogeneous coordinate indices removed from the train span."""

    spec = _spec(config)
    return tuple(
        1 + key * spec.value_cardinality + value
        for key, value in spec.heldout_key_value_pairs
    )


def query_readout_matrix(config: ConfigLike, key: int) -> IntegerMatrix:
    """Return the exact V-class query readout for one key."""

    spec = _spec(config)
    _plain_int("key", key)
    if key >= spec.num_surface_keys:
        raise ValueError("key is outside the task vocabulary")
    dimension = full_homogeneous_dimension(spec)
    rows = [[0] * dimension for _ in range(spec.value_cardinality)]
    start = 1 + key * spec.value_cardinality
    for value in range(spec.value_cardinality):
        rows[value][start + value] = 1
    return tuple(tuple(row) for row in rows)


def adversarial_query_readout_matrix(
    config: ConfigLike,
    *,
    key: int,
    heldout_value: int,
    alternative_value: int,
) -> IntegerMatrix:
    """Alter only the missing coordinate's class without changing train output."""

    spec = _spec(config)
    for name, value in (
        ("key", key),
        ("heldout_value", heldout_value),
        ("alternative_value", alternative_value),
    ):
        _plain_int(name, value)
    if key >= spec.num_surface_keys:
        raise ValueError("key is outside the task vocabulary")
    if (
        heldout_value >= spec.value_cardinality
        or alternative_value >= spec.value_cardinality
    ):
        raise ValueError("value is outside the task vocabulary")
    if heldout_value == alternative_value:
        raise ValueError("alternative value must differ from the held-out value")
    if (key, heldout_value) not in spec.heldout_key_value_pairs:
        raise ValueError("the selected coordinate is not held out")
    rows = [list(row) for row in query_readout_matrix(spec, key)]
    coordinate = 1 + key * spec.value_cardinality + heldout_value
    rows[heldout_value][coordinate] = 0
    rows[alternative_value][coordinate] = 1
    return tuple(tuple(row) for row in rows)


def apply_query_readout(
    readout: Sequence[Sequence[int]], state_vector: Sequence[int]
) -> tuple[int, ...]:
    """Apply an exact integer query readout."""

    return apply_integer_matrix(readout, state_vector)


def adversarial_update_operator(
    config: ConfigLike,
    *,
    key: int,
    heldout_value: int,
    argument: int,
    alternative_result: int,
) -> IntegerMatrix:
    """Change one unseen input column of an otherwise exact update operator."""

    spec = _spec(config)
    for name, value in (
        ("key", key),
        ("heldout_value", heldout_value),
        ("argument", argument),
        ("alternative_result", alternative_result),
    ):
        _plain_int(name, value)
    if key >= spec.num_surface_keys:
        raise ValueError("key is outside the task vocabulary")
    if any(
        value >= spec.value_cardinality
        for value in (heldout_value, argument, alternative_result)
    ):
        raise ValueError("value or argument is outside the task vocabulary")
    if (key, heldout_value) not in spec.heldout_key_value_pairs:
        raise ValueError("the selected coordinate is not held out")
    correct = (heldout_value + argument + 1) % spec.value_cardinality
    if alternative_result == correct:
        raise ValueError("alternative result must differ from the exact update")
    rows = [
        list(row)
        for row in homogeneous_operator(spec, AlgebraAction.update(key, argument))
    ]
    column = 1 + key * spec.value_cardinality + heldout_value
    for row in rows:
        row[column] = 0
    output = 1 + key * spec.value_cardinality + alternative_result
    rows[output][column] = 1
    return tuple(tuple(row) for row in rows)


def adversarial_bind_operator(
    config: ConfigLike,
    *,
    key: int,
    heldout_value: int,
    alternative_value: int,
) -> IntegerMatrix:
    """Alias the unseen bind signature to a different observed value."""

    spec = _spec(config)
    for name, value in (
        ("key", key),
        ("heldout_value", heldout_value),
        ("alternative_value", alternative_value),
    ):
        _plain_int(name, value)
    if key >= spec.num_surface_keys:
        raise ValueError("key is outside the task vocabulary")
    if (
        heldout_value >= spec.value_cardinality
        or alternative_value >= spec.value_cardinality
    ):
        raise ValueError("value is outside the task vocabulary")
    if heldout_value == alternative_value:
        raise ValueError("alternative value must differ from the held-out value")
    if (key, heldout_value) not in spec.heldout_key_value_pairs:
        raise ValueError("the selected coordinate is not held out")
    return homogeneous_operator(spec, AlgebraAction.bind(key, alternative_value))


def _is_single_cycle(permutation: tuple[int, ...]) -> bool:
    current = 0
    visited: set[int] = set()
    for _ in range(len(permutation)):
        visited.add(current)
        current = permutation[current]
    return current == 0 and len(visited) == len(permutation)


def cyclic_successor_completion_analysis(
    config: ConfigLike,
    *,
    key: int,
    max_permutations: int = 100_000,
) -> CyclicCompletionAnalysis:
    """Enumerate completions of UPDATE argument zero on a held-out key.

    The three nested hypothesis classes are arbitrary functions, permutations,
    and transitive generators of a cyclic action.  Only the last encodes the
    intended ``C_V`` law.
    """

    spec = _spec(config)
    _plain_int("key", key)
    _plain_int("max_permutations", max_permutations, 1)
    if key >= spec.num_surface_keys:
        raise ValueError("key is outside the task vocabulary")
    heldout_values = tuple(
        value
        for heldout_key, value in spec.heldout_key_value_pairs
        if heldout_key == key
    )
    if len(heldout_values) != 1:
        raise ValueError("analysis requires exactly one held-out value at the key")
    heldout = heldout_values[0]
    permutation_budget = 1
    for factor in range(2, spec.value_cardinality + 1):
        if permutation_budget > max_permutations // factor:
            raise IdentificationLimitError(
                "completion enumeration exceeds the declared permutation limit"
            )
        permutation_budget *= factor
    observed: list[tuple[int, int]] = []
    for value in range(spec.value_cardinality):
        result = (value + 1) % spec.value_cardinality
        if (
            (key, value) not in spec.heldout_key_value_pairs
            and (key, result) not in spec.heldout_key_value_pairs
        ):
            observed.append((value, result))
    unconstrained = spec.value_cardinality ** (
        spec.value_cardinality - len(observed)
    )
    compatible = tuple(
        candidate
        for candidate in permutations(range(spec.value_cardinality))
        if all(candidate[source] == target for source, target in observed)
    )
    cyclic = tuple(candidate for candidate in compatible if _is_single_cycle(candidate))
    return CyclicCompletionAnalysis(
        key=key,
        heldout_value=heldout,
        observed_edges=tuple(observed),
        unrestricted_function_completions=unconstrained,
        permutation_completions=compatible,
        transitive_cyclic_completions=cyclic,
    )


def key_permutation_operator(
    config: ConfigLike, permutation: Sequence[int]
) -> IntegerMatrix:
    """Return the homogeneous operator that relabels key roles."""

    spec = _spec(config)
    mapping = tuple(permutation)
    for index, value in enumerate(mapping):
        _plain_int(f"key permutation entry {index}", value)
    if sorted(mapping) != list(range(spec.num_surface_keys)):
        raise ValueError("key permutation must contain every key exactly once")
    dimension = full_homogeneous_dimension(spec)
    matrix = [[0] * dimension for _ in range(dimension)]
    matrix[0][0] = 1
    for key in range(spec.num_surface_keys):
        for value in range(spec.value_cardinality):
            source = 1 + key * spec.value_cardinality + value
            target = 1 + mapping[key] * spec.value_cardinality + value
            matrix[target][source] = 1
    return tuple(tuple(row) for row in matrix)


def value_shift_operator(config: ConfigLike, shift: int) -> IntegerMatrix:
    """Return the homogeneous action of a global cyclic value relabeling."""

    spec = _spec(config)
    _plain_int("shift", shift)
    shift %= spec.value_cardinality
    dimension = full_homogeneous_dimension(spec)
    matrix = [[0] * dimension for _ in range(dimension)]
    matrix[0][0] = 1
    for key in range(spec.num_surface_keys):
        for value in range(spec.value_cardinality):
            source = 1 + key * spec.value_cardinality + value
            target = 1 + key * spec.value_cardinality + (
                value + shift
            ) % spec.value_cardinality
            matrix[target][source] = 1
    return tuple(tuple(row) for row in matrix)


def _transpose(matrix: IntegerMatrix) -> IntegerMatrix:
    return tuple(tuple(column) for column in zip(*matrix))


def _symmetry_recovery(
    spec: BindingAlgebraSpec, heldout_key: int, heldout_value: int
) -> SymmetryRecoveryAnalysis:
    source_key = next(
        key for key in range(spec.num_surface_keys) if key != heldout_key
    )
    permutation = list(range(spec.num_surface_keys))
    permutation[source_key], permutation[heldout_key] = (
        permutation[heldout_key],
        permutation[source_key],
    )
    key_action = key_permutation_operator(spec, permutation)
    key_inverse = _transpose(key_action)

    def key_conjugate(action: AlgebraAction) -> IntegerMatrix:
        return multiply_integer_matrices(
            key_action,
            multiply_integer_matrices(
                homogeneous_operator(spec, action), key_inverse
            ),
        )

    bind_recovered = key_conjugate(
        AlgebraAction.bind(source_key, heldout_value)
    ) == homogeneous_operator(spec, AlgebraAction.bind(heldout_key, heldout_value))
    update_recovered = key_conjugate(
        AlgebraAction.update(source_key, 0)
    ) == homogeneous_operator(spec, AlgebraAction.update(heldout_key, 0))
    derived_readout = multiply_integer_matrices(
        query_readout_matrix(spec, source_key), key_inverse
    )
    readout_recovered = derived_readout == query_readout_matrix(spec, heldout_key)
    copies_recovered = all(
        key_conjugate(AlgebraAction.copy(destination, source))
        == homogeneous_operator(
            spec,
            AlgebraAction.copy(
                permutation[destination], permutation[source]
            ),
        )
        for destination in range(spec.num_surface_keys)
        for source in range(spec.num_surface_keys)
        if destination != source
    )

    source_value = next(
        value
        for value in range(spec.value_cardinality)
        if value != heldout_value
    )
    shift = (heldout_value - source_value) % spec.value_cardinality
    value_action = value_shift_operator(spec, shift)
    value_inverse = value_shift_operator(spec, -shift % spec.value_cardinality)
    derived_bind = multiply_integer_matrices(
        value_action,
        multiply_integer_matrices(
            homogeneous_operator(spec, AlgebraAction.bind(heldout_key, source_value)),
            value_inverse,
        ),
    )
    value_bind_recovered = derived_bind == homogeneous_operator(
        spec, AlgebraAction.bind(heldout_key, heldout_value)
    )
    # r_source * S_(source-heldout) selects the held-out coordinate.
    observed_row = (query_readout_matrix(spec, heldout_key)[source_value],)
    row_shift = value_shift_operator(
        spec, (source_value - heldout_value) % spec.value_cardinality
    )
    derived_row = multiply_integer_matrices(observed_row, row_shift)[0]
    true_row = query_readout_matrix(spec, heldout_key)[heldout_value]
    value_readout_recovered = derived_row == true_row
    update = homogeneous_operator(spec, AlgebraAction.update(heldout_key, 0))
    update_commutes = multiply_integer_matrices(
        update, value_action
    ) == multiply_integer_matrices(value_action, update)
    return SymmetryRecoveryAnalysis(
        assumes_universal_exact_operator_identities=True,
        identities_are_inferred_from_train_constraints=False,
        source_key=source_key,
        heldout_key=heldout_key,
        key_permutation=tuple(permutation),
        key_equivariance_recovers_bind=bind_recovered,
        key_equivariance_recovers_update=update_recovered,
        key_equivariance_recovers_query_readout=readout_recovered,
        key_equivariance_recovers_all_copy_operators=copies_recovered,
        source_value=source_value,
        heldout_value=heldout_value,
        value_shift=shift,
        value_equivariance_recovers_bind=value_bind_recovered,
        value_equivariance_recovers_query_row=value_readout_recovered,
        update_commutes_with_value_shift=update_commutes,
    )


def _state_is_train_valid(
    spec: BindingAlgebraSpec, state: SemanticState
) -> bool:
    return all(
        state.values[key] != value
        for key, value in spec.heldout_key_value_pairs
    )


def unrestricted_train_system_analysis(
    config: ConfigLike, *, max_states: int = 1_000_000
) -> UnrestrictedSystemAnalysis:
    """Build the exact unrestricted linear design over complete train support.

    This is intentionally optimistic: it pretends the canonical latent state
    and exact successor vector are supervised for every legal train edge.
    End-to-end query-only training cannot be more identifiable than this
    design.
    """

    spec = _spec(config)
    _plain_int("max_states", max_states, 1)
    train_count = train_semantic_state_count(spec)
    if train_count > max_states:
        raise IdentificationLimitError("train state count exceeds max_states")
    states = enumerate_semantic_states(
        spec, exclude_heldout=True, max_states=max_states
    )
    dimension = full_homogeneous_dimension(spec)
    action_rows: list[ActionConstraintAnalysis] = []
    total_transitions = 0
    for action in canonical_visible_actions(spec):
        source_vectors: list[tuple[int, ...]] = []
        for state in states:
            result = apply_action(
                spec, state, action, contract=AlgebraContract.STRICT
            )
            if result.state is None or not _state_is_train_valid(
                spec, result.state
            ):
                continue
            source_vectors.append(encode_homogeneous_state(spec, state))
        source_rank = exact_matrix_rank(source_vectors).rational_rank
        transitions = len(source_vectors)
        parameter_count = dimension * dimension
        constraint_rank = dimension * source_rank
        action_rows.append(
            ActionConstraintAnalysis(
                kind=action.kind.name,
                primary_key=action.primary_key,
                secondary_key=action.secondary_key,
                argument=action.argument,
                observed_transitions=transitions,
                source_rank=source_rank,
                parameter_count=parameter_count,
                constraint_rank=constraint_rank,
                nullity=parameter_count - constraint_rank,
            )
        )
        total_transitions += transitions

    query_rows: list[QueryConstraintAnalysis] = []
    for key in range(spec.num_surface_keys):
        source_vectors = [
            encode_homogeneous_state(spec, state)
            for state in states
            if state.values[key] >= 0
        ]
        source_rank = exact_matrix_rank(source_vectors).rational_rank
        parameter_count = spec.value_cardinality * dimension
        constraint_rank = spec.value_cardinality * source_rank
        query_rows.append(
            QueryConstraintAnalysis(
                key=key,
                observed_queries=len(source_vectors),
                source_rank=source_rank,
                parameter_count=parameter_count,
                constraint_rank=constraint_rank,
                nullity=parameter_count - constraint_rank,
            )
        )

    operator_parameters = sum(row.parameter_count for row in action_rows)
    operator_rank = sum(row.constraint_rank for row in action_rows)
    query_parameters = sum(row.parameter_count for row in query_rows)
    query_rank = sum(row.constraint_rank for row in query_rows)
    return UnrestrictedSystemAnalysis(
        optimistic_latent_states_and_successors_are_observed=True,
        action_count=len(action_rows),
        observed_action_count=sum(row.observed_transitions > 0 for row in action_rows),
        observed_transitions=total_transitions,
        operator_parameter_count=operator_parameters,
        operator_constraint_rank=operator_rank,
        operator_nullity=operator_parameters - operator_rank,
        query_parameter_count=query_parameters,
        query_constraint_rank=query_rank,
        query_nullity=query_parameters - query_rank,
        total_parameter_count=operator_parameters + query_parameters,
        total_constraint_rank=operator_rank + query_rank,
        total_nullity=(operator_parameters - operator_rank)
        + (query_parameters - query_rank),
        actions=tuple(action_rows),
        queries=tuple(query_rows),
    )


def pair_completion_analysis(config: ConfigLike) -> PairCompletionAnalysis:
    """Compare arbitrary, additive, and generic low-rank pair completion."""

    spec = _spec(config)
    pairs = tuple(
        (key, value)
        for key in range(spec.num_surface_keys)
        for value in range(spec.value_cardinality)
    )
    heldout = set(spec.heldout_key_value_pairs)
    observed = tuple(pair for pair in pairs if pair not in heldout)

    # A general pair table has one independent coordinate per cell.
    arbitrary_rows = []
    for pair in observed:
        row = [0] * len(pairs)
        row[pairs.index(pair)] = 1
        arbitrary_rows.append(tuple(row))
    arbitrary_rank = exact_matrix_rank(arbitrary_rows).rational_rank

    # Additive functions F(k,v)=a_k+b_v have the familiar one-dimensional
    # parameter gauge (a+c,b-c) but a uniquely determined missing behavior.
    additive_rows = []
    for key, value in observed:
        row = [0] * (spec.num_surface_keys + spec.value_cardinality)
        row[key] = 1
        row[spec.num_surface_keys + value] = 1
        additive_rows.append(tuple(row))
    additive_rank = exact_matrix_rank(additive_rows).rational_rank
    heldout_rows = []
    for key, value in spec.heldout_key_value_pairs:
        row = [0] * (spec.num_surface_keys + spec.value_cardinality)
        row[key] = 1
        row[spec.num_surface_keys + value] = 1
        heldout_rows.append(tuple(row))
    full_additive_rank = exact_matrix_rank(
        tuple(additive_rows) + tuple(heldout_rows)
    ).rational_rank
    max_rank = min(spec.num_surface_keys, spec.value_cardinality)
    varieties = tuple(
        (
            rank,
            rank
            * (spec.num_surface_keys + spec.value_cardinality - rank),
        )
        for rank in range(1, max_rank + 1)
    )
    return PairCompletionAnalysis(
        pair_count=len(pairs),
        observed_pair_count=len(observed),
        arbitrary_pair_table_rank=arbitrary_rank,
        arbitrary_pair_table_nullity=len(pairs) - arbitrary_rank,
        additive_parameter_count=(
            spec.num_surface_keys + spec.value_cardinality
        ),
        additive_design_rank=additive_rank,
        additive_parameter_gauge_dimension=(
            spec.num_surface_keys
            + spec.value_cardinality
            - additive_rank
        ),
        additive_heldout_behavior_nullity=(
            full_additive_rank - additive_rank
        ),
        localized_pair_interaction_witness_exists=bool(heldout),
        generic_rank_variety_dimensions=varieties,
        generic_low_rank_requires_nondegenerate_anchor=True,
    )


def structured_hypothesis_analyses(
    config: ConfigLike,
) -> tuple[StructuredHypothesisAnalysis, ...]:
    """Compare exact local-block sharing hypotheses on train support.

    The designs expose each canonical local value block and its exact target.
    They are therefore optimistic identifiability controls, not neural-network
    parameter counts.  The formulas apply to the report's one-held-out-pair,
    at-least-two-key setting.
    """

    spec = _spec(config)
    if len(spec.heldout_key_value_pairs) != 1:
        raise ValueError("structured comparison requires one held-out pair")
    if spec.num_surface_keys < 2:
        raise ValueError("structured comparison requires at least two keys")
    if spec.max_live_bindings < 2:
        raise ValueError(
            "structured comparison requires two live bindings for copy support"
        )
    keys = spec.num_surface_keys
    values = spec.value_cardinality
    block = values * values

    local_bind_parameters = keys * block
    local_update_parameters = keys * (values - 1) * block
    local_copy_parameters = keys * (keys - 1) * block
    local_query_parameters = keys * block
    local_bind_nullity = values
    local_update_nullity = 2 * (values - 1) * values
    local_copy_nullity = 2 * (keys - 1) * values
    local_query_nullity = values
    local_parameters = (
        local_bind_parameters
        + local_update_parameters
        + local_copy_parameters
        + local_query_parameters
    )
    local_nullity = (
        local_bind_nullity
        + local_update_nullity
        + local_copy_nullity
        + local_query_nullity
    )

    shared_transition_parameters = (
        local_bind_parameters
        + (values - 1) * block
        + block
        + local_query_parameters
    )
    # With exactly two keys, the only observed value-0 source is the non-heldout
    # key, and copying it into the heldout key would itself leave train support.
    # A third key supplies a clean source/destination pair and closes this cell.
    shared_copy_nullity = values if keys == 2 else 0
    fully_shared_parameters = (values + 2) * block
    cyclic_per_key_parameters = 2 * keys * values
    cyclic_shared_parameters = 2 * values

    def row(
        hypothesis: str,
        parameters: int,
        *,
        bind: int = 0,
        update: int = 0,
        copy: int = 0,
        query: int = 0,
    ) -> StructuredHypothesisAnalysis:
        nullity = bind + update + copy + query
        return StructuredHypothesisAnalysis(
            hypothesis=hypothesis,
            parameter_count=parameters,
            constraint_rank=parameters - nullity,
            nullity=nullity,
            bind_nullity=bind,
            update_nullity=update,
            copy_nullity=copy,
            query_nullity=query,
            heldout_behavior_identified=nullity == 0,
            assumes_canonical_local_block_supervision=True,
        )

    return (
        row(
            "key_local_bind_update_copy_query_blocks",
            local_parameters,
            bind=local_bind_nullity,
            update=local_update_nullity,
            copy=local_copy_nullity,
            query=local_query_nullity,
        ),
        row(
            "shared_update_and_copy_prototypes_only",
            shared_transition_parameters,
            bind=local_bind_nullity,
            copy=shared_copy_nullity,
            query=local_query_nullity,
        ),
        row(
            "shared_bind_update_copy_query_prototypes",
            fully_shared_parameters,
            copy=shared_copy_nullity,
        ),
        row(
            "per_key_calibrated_cyclic_bind_query_orbits",
            cyclic_per_key_parameters,
        ),
        row(
            "shared_calibrated_cyclic_bind_query_orbits",
            cyclic_shared_parameters,
        ),
    )


def _nonidentifiability_witness(
    spec: BindingAlgebraSpec, key: int, value: int
) -> NonidentifiabilityWitness:
    train_states = enumerate_semantic_states(
        spec, exclude_heldout=True, max_states=train_semantic_state_count(spec)
    )
    alternative_value = (value + 1) % spec.value_cardinality
    true_readout = query_readout_matrix(spec, key)
    alternative_readout = adversarial_query_readout_matrix(
        spec,
        key=key,
        heldout_value=value,
        alternative_value=alternative_value,
    )
    readouts_equal = all(
        apply_query_readout(true_readout, encode_homogeneous_state(spec, state))
        == apply_query_readout(
            alternative_readout, encode_homogeneous_state(spec, state)
        )
        for state in train_states
    )
    update_argument = 0
    true_updated = (value + update_argument + 1) % spec.value_cardinality
    alternative_updated = (true_updated + 1) % spec.value_cardinality
    true_operator = homogeneous_operator(
        spec, AlgebraAction.update(key, update_argument)
    )
    alternative_operator = adversarial_update_operator(
        spec,
        key=key,
        heldout_value=value,
        argument=update_argument,
        alternative_result=alternative_updated,
    )
    operators_equal = all(
        apply_integer_matrix(true_operator, encode_homogeneous_state(spec, state))
        == apply_integer_matrix(
            alternative_operator, encode_homogeneous_state(spec, state)
        )
        for state in train_states
    )
    heldout_values = [-1] * spec.num_surface_keys
    heldout_values[key] = value
    heldout_state = SemanticState(tuple(heldout_values))
    encoded = encode_homogeneous_state(spec, heldout_state)
    true_query = apply_query_readout(true_readout, encoded)
    alternative_query = apply_query_readout(alternative_readout, encoded)
    if true_query == alternative_query:
        raise AssertionError("readout witness does not separate the held-out state")
    true_state = decode_homogeneous_state(
        spec, apply_integer_matrix(true_operator, encoded)
    )
    alternative_state = decode_homogeneous_state(
        spec, apply_integer_matrix(alternative_operator, encoded)
    )
    if true_state == alternative_state:
        raise AssertionError("operator witness does not separate the held-out state")
    empty = encode_homogeneous_state(
        spec, SemanticState((-1,) * spec.num_surface_keys)
    )
    true_bind = homogeneous_operator(spec, AlgebraAction.bind(key, value))
    alternative_bind = adversarial_bind_operator(
        spec,
        key=key,
        heldout_value=value,
        alternative_value=alternative_value,
    )
    true_bound = decode_homogeneous_state(
        spec, apply_integer_matrix(true_bind, empty)
    )
    alternative_bound = decode_homogeneous_state(
        spec, apply_integer_matrix(alternative_bind, empty)
    )
    unseen_bind_train_transitions = 0
    for state in train_states:
        result = apply_action(
            spec,
            state,
            AlgebraAction.bind(key, value),
            contract=AlgebraContract.STRICT,
        )
        if result.state is not None and _state_is_train_valid(spec, result.state):
            unseen_bind_train_transitions += 1
    if unseen_bind_train_transitions != 0:
        raise AssertionError("held-out bind unexpectedly appears in train support")
    if true_bound == alternative_bound:
        raise AssertionError("bind witness does not separate the held-out event")
    return NonidentifiabilityWitness(
        train_states_checked=len(train_states),
        heldout_state=heldout_state.values,
        true_query_value=true_query.index(1),
        alternative_query_value=alternative_query.index(1),
        update_argument=update_argument,
        true_updated_value=true_state.values[key],
        alternative_updated_value=alternative_state.values[key],
        readouts_equal_on_all_train_states=readouts_equal,
        operators_equal_on_all_train_states=operators_equal,
        unseen_bind_train_transitions=unseen_bind_train_transitions,
        true_bound_value=true_bound.values[key],
        alternative_bound_value=alternative_bound.values[key],
    )


def analyze_heldout_identification(
    config: ConfigLike,
    *,
    max_states: int = 1_000_000,
    max_permutations: int = 100_000,
) -> HeldoutIdentificationReport:
    """Produce the exact missing-direction identification report."""

    spec = _spec(config)
    _plain_int("max_states", max_states, 1)
    if len(spec.heldout_key_value_pairs) != 1:
        raise ValueError("analysis currently requires exactly one held-out pair")
    if spec.num_surface_keys < 2:
        raise ValueError("analysis requires at least two keys for key symmetry")
    if spec.max_live_bindings < 2:
        raise ValueError("analysis requires two live bindings for copy support")
    if semantic_state_count(spec) > max_states:
        raise IdentificationLimitError("semantic state count exceeds max_states")
    key, value = spec.heldout_key_value_pairs[0]
    full_features = homogeneous_state_feature_matrix(
        spec, max_states=max_states
    )
    train_features = homogeneous_state_feature_matrix(
        spec, exclude_heldout=True, max_states=max_states
    )
    full_feature_rank = exact_matrix_rank(full_features).rational_rank
    train_feature_rank = exact_matrix_rank(train_features).rational_rank
    full_diagnostic_rank = diagnostic_probe_rank(
        spec, max_states=max_states
    ).rational_rank
    train_diagnostic_rank = diagnostic_probe_rank(
        spec, exclude_heldout=True, max_states=max_states
    ).rational_rank
    dimension = full_homogeneous_dimension(spec)
    nullity = dimension - train_feature_rank
    directions = heldout_feature_indices(spec)
    if nullity != len(directions):
        raise AssertionError("held-out coordinates do not match train-span nullity")
    witness = _nonidentifiability_witness(spec, key, value)
    unrestricted = unrestricted_train_system_analysis(
        spec, max_states=max_states
    )
    pair_completion = pair_completion_analysis(spec)
    structured = structured_hypothesis_analyses(spec)
    cyclic = cyclic_successor_completion_analysis(
        spec, key=key, max_permutations=max_permutations
    )
    symmetry = _symmetry_recovery(spec, key, value)
    symmetry_complete = all(
        (
            symmetry.key_equivariance_recovers_bind,
            symmetry.key_equivariance_recovers_update,
            symmetry.key_equivariance_recovers_query_readout,
            symmetry.key_equivariance_recovers_all_copy_operators,
            symmetry.value_equivariance_recovers_bind,
            symmetry.value_equivariance_recovers_query_row,
            symmetry.update_commutes_with_value_shift,
        )
    )
    return HeldoutIdentificationReport(
        schema="tnlm-v3-heldout-identification-v1",
        task_fingerprint=(
            config.fingerprint()
            if isinstance(config, BindingTaskConfig)
            else "binding-algebra-spec-no-task-fingerprint"
        ),
        heldout_pair=(key, value),
        homogeneous_dimension=dimension,
        full_state_count=semantic_state_count(spec),
        train_state_count=train_semantic_state_count(spec),
        full_feature_rank=full_feature_rank,
        train_feature_rank=train_feature_rank,
        full_diagnostic_rank=full_diagnostic_rank,
        train_diagnostic_rank=train_diagnostic_rank,
        missing_feature_indices=directions,
        feature_nullity=nullity,
        missing_column_scalar_readout_nullity=nullity,
        missing_column_multiclass_readout_nullity=(
            spec.value_cardinality * nullity
        ),
        unrestricted_operator_nullity_lower_bound_per_action=dimension * nullity,
        unrestricted_affine_operator_nullity_lower_bound_per_action=(
            (dimension - 1) * nullity
        ),
        unseen_bind_action_matrix_freedom=dimension * dimension,
        unseen_bind_action_affine_freedom=dimension * (dimension - 1),
        witness=witness,
        unrestricted_system=unrestricted,
        pair_completion=pair_completion,
        structured_hypotheses=structured,
        cyclic_update_completion=cyclic,
        symmetry_recovery=symmetry,
        data_alone_recovers_heldout_direction=False,
        cyclic_update_law_alone_recovers_full_transducer=False,
        declared_universal_operator_equivariance_recovers_heldout_direction=(
            symmetry_complete
        ),
        exact_equivariance_is_supplied_not_inferred=True,
    )


__all__ = [
    "ActionConstraintAnalysis",
    "CyclicCompletionAnalysis",
    "HeldoutIdentificationReport",
    "IdentificationLimitError",
    "NonidentifiabilityWitness",
    "PairCompletionAnalysis",
    "QueryConstraintAnalysis",
    "SymmetryRecoveryAnalysis",
    "StructuredHypothesisAnalysis",
    "UnrestrictedSystemAnalysis",
    "adversarial_query_readout_matrix",
    "adversarial_bind_operator",
    "adversarial_update_operator",
    "analyze_heldout_identification",
    "apply_query_readout",
    "cyclic_successor_completion_analysis",
    "heldout_feature_indices",
    "key_permutation_operator",
    "pair_completion_analysis",
    "query_readout_matrix",
    "unrestricted_train_system_analysis",
    "structured_hypothesis_analyses",
    "value_shift_operator",
]
