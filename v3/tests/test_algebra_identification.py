from __future__ import annotations

from collections import Counter

import pytest

from tnlm_v3.algebra_identification import (
    HeldoutIdentificationReport,
    IdentificationLimitError,
    adversarial_bind_operator,
    adversarial_query_readout_matrix,
    adversarial_update_operator,
    analyze_heldout_identification,
    apply_query_readout,
    cyclic_successor_completion_analysis,
    heldout_feature_indices,
    key_permutation_operator,
    pair_completion_analysis,
    query_readout_matrix,
    structured_hypothesis_analyses,
    unrestricted_train_system_analysis,
    value_shift_operator,
)
from tnlm_v3.data import BindingTaskConfig
from tnlm_v3.exact_algebra import (
    AlgebraAction,
    BindingAlgebraSpec,
    SemanticState,
    apply_integer_matrix,
    decode_homogeneous_state,
    encode_homogeneous_state,
    enumerate_semantic_states,
    homogeneous_operator,
    multiply_integer_matrices,
)


def screen_task() -> BindingTaskConfig:
    return BindingTaskConfig(
        num_surface_keys=5,
        value_cardinality=4,
        branches=3,
        max_live_bindings=3,
        min_length=10,
        max_length=512,
        heldout_key_value_pairs=((0, 0),),
    )


@pytest.fixture(scope="module")
def report() -> HeldoutIdentificationReport:
    return analyze_heldout_identification(
        screen_task(), max_states=821, max_permutations=24
    )


def test_screen_report_has_the_exact_missing_direction_and_system_totals(
    report: HeldoutIdentificationReport,
) -> None:
    assert report.schema == "tnlm-v3-heldout-identification-v1"
    assert report.task_fingerprint == screen_task().fingerprint()
    assert report.heldout_pair == (0, 0)
    assert report.homogeneous_dimension == 21
    assert (report.full_state_count, report.train_state_count) == (821, 708)
    assert (report.full_feature_rank, report.train_feature_rank) == (21, 20)
    assert (report.full_diagnostic_rank, report.train_diagnostic_rank) == (21, 20)
    assert report.missing_feature_indices == heldout_feature_indices(screen_task()) == (
        1,
    )
    assert report.feature_nullity == 1
    assert report.missing_column_scalar_readout_nullity == 1
    assert report.missing_column_multiclass_readout_nullity == 4
    assert report.unrestricted_operator_nullity_lower_bound_per_action == 21
    assert report.unrestricted_affine_operator_nullity_lower_bound_per_action == 20
    assert report.unseen_bind_action_matrix_freedom == 441
    assert report.unseen_bind_action_affine_freedom == 420

    system = report.unrestricted_system
    assert system.optimistic_latent_states_and_successors_are_observed
    assert (system.action_count, system.observed_action_count) == (67, 66)
    assert system.observed_transitions == 16_107
    assert (
        system.operator_parameter_count,
        system.operator_constraint_rank,
        system.operator_nullity,
    ) == (29_547, 24_675, 4_872)
    assert (
        system.query_parameter_count,
        system.query_constraint_rank,
        system.query_nullity,
    ) == (420, 380, 40)
    assert (
        system.total_parameter_count,
        system.total_constraint_rank,
        system.total_nullity,
    ) == (29_967, 25_055, 4_912)
    assert system.total_nullity == (
        system.total_parameter_count - system.total_constraint_rank
    )
    assert not report.data_alone_recovers_heldout_direction
    assert not report.cyclic_update_law_alone_recovers_full_transducer
    assert (
        report.declared_universal_operator_equivariance_recovers_heldout_direction
    )
    assert report.exact_equivariance_is_supplied_not_inferred


def test_unrestricted_action_source_rank_groups_are_exact(
    report: HeldoutIdentificationReport,
) -> None:
    observed = Counter(
        (
            row.kind,
            row.observed_transitions,
            row.source_rank,
            row.constraint_rank,
            row.nullity,
        )
        for row in report.unrestricted_system.actions
    )
    expected = Counter(
        {
            ("BIND", 0, 0, 0, 441): 1,
            ("BIND", 113, 17, 357, 84): 3,
            ("BIND", 100, 16, 336, 105): 16,
            ("UPDATE", 226, 18, 378, 63): 3,
            ("UPDATE", 400, 19, 399, 42): 12,
            ("COPY", 117, 17, 357, 84): 4,
            ("COPY", 156, 18, 378, 63): 4,
            ("COPY", 192, 18, 378, 63): 12,
            ("INVALIDATE", 339, 19, 399, 42): 1,
            ("INVALIDATE", 400, 19, 399, 42): 4,
            ("QUERY", 339, 19, 399, 42): 1,
            ("QUERY", 400, 19, 399, 42): 4,
            ("DISTRACTOR", 708, 20, 420, 21): 2,
        }
    )
    assert observed == expected

    unseen_bind = next(
        row
        for row in report.unrestricted_system.actions
        if (row.kind, row.primary_key, row.argument) == ("BIND", 0, 0)
    )
    assert unseen_bind.parameter_count == unseen_bind.nullity == 21**2
    assert unseen_bind.observed_transitions == unseen_bind.source_rank == 0

    queries = report.unrestricted_system.queries
    assert [row.observed_queries for row in queries] == [339, 400, 400, 400, 400]
    assert all(
        (
            row.source_rank,
            row.parameter_count,
            row.constraint_rank,
            row.nullity,
        )
        == (19, 84, 76, 8)
        for row in queries
    )


def test_pair_completion_separates_arbitrary_additive_and_low_rank_claims(
    report: HeldoutIdentificationReport,
) -> None:
    pair = report.pair_completion
    assert pair == pair_completion_analysis(screen_task())
    assert (pair.pair_count, pair.observed_pair_count) == (20, 19)
    assert (pair.arbitrary_pair_table_rank, pair.arbitrary_pair_table_nullity) == (
        19,
        1,
    )
    assert (
        pair.additive_parameter_count,
        pair.additive_design_rank,
        pair.additive_parameter_gauge_dimension,
        pair.additive_heldout_behavior_nullity,
    ) == (9, 8, 1, 0)
    assert pair.localized_pair_interaction_witness_exists
    assert pair.generic_rank_variety_dimensions == (
        (1, 8),
        (2, 14),
        (3, 18),
        (4, 20),
    )
    assert pair.generic_low_rank_requires_nondegenerate_anchor


def test_structured_hypotheses_pinpoint_which_sharing_laws_close_the_gap(
    report: HeldoutIdentificationReport,
) -> None:
    rows = report.structured_hypotheses
    assert rows == structured_hypothesis_analyses(screen_task())
    assert [row.hypothesis for row in rows] == [
        "key_local_bind_update_copy_query_blocks",
        "shared_update_and_copy_prototypes_only",
        "shared_bind_update_copy_query_prototypes",
        "per_key_calibrated_cyclic_bind_query_orbits",
        "shared_calibrated_cyclic_bind_query_orbits",
    ]
    assert [
        (row.parameter_count, row.constraint_rank, row.nullity) for row in rows
    ] == [
        (720, 656, 64),
        (224, 216, 8),
        (96, 96, 0),
        (40, 40, 0),
        (8, 8, 0),
    ]
    assert [
        (
            row.bind_nullity,
            row.update_nullity,
            row.copy_nullity,
            row.query_nullity,
        )
        for row in rows
    ] == [
        (4, 24, 32, 4),
        (4, 0, 0, 4),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    ]
    assert [row.heldout_behavior_identified for row in rows] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert all(row.assumes_canonical_local_block_supervision for row in rows)


@pytest.mark.parametrize(
    ("values", "full_parameters", "full_rank", "full_nullity"),
    ((2, 16, 14, 2), (4, 96, 92, 4)),
)
def test_two_key_shared_copy_cell_remains_unidentified(
    values: int,
    full_parameters: int,
    full_rank: int,
    full_nullity: int,
) -> None:
    spec = BindingAlgebraSpec(
        num_surface_keys=2,
        value_cardinality=values,
        max_live_bindings=2,
        heldout_key_value_pairs=((0, 0),),
        branches=2,
    )
    rows = {
        row.hypothesis: row for row in structured_hypothesis_analyses(spec)
    }
    full = rows["shared_bind_update_copy_query_prototypes"]
    assert (
        full.parameter_count,
        full.constraint_rank,
        full.nullity,
        full.copy_nullity,
        full.heldout_behavior_identified,
    ) == (full_parameters, full_rank, full_nullity, values, False)
    transitions = rows["shared_update_and_copy_prototypes_only"]
    assert transitions.copy_nullity == values
    assert transitions.nullity == 3 * values


def test_explicit_readout_and_update_witnesses_match_all_train_states_but_fail_heldout(
    report: HeldoutIdentificationReport,
) -> None:
    task = screen_task()
    train_states = enumerate_semantic_states(
        task, exclude_heldout=True, max_states=708
    )
    heldout_state = SemanticState((0, -1, -1, -1, -1))
    heldout_vector = encode_homogeneous_state(task, heldout_state)

    exact_readout = query_readout_matrix(task, 0)
    adversarial_readout = adversarial_query_readout_matrix(
        task, key=0, heldout_value=0, alternative_value=1
    )
    assert all(
        apply_query_readout(exact_readout, encode_homogeneous_state(task, state))
        == apply_query_readout(
            adversarial_readout, encode_homogeneous_state(task, state)
        )
        for state in train_states
    )
    assert apply_query_readout(exact_readout, heldout_vector) == (1, 0, 0, 0)
    assert apply_query_readout(adversarial_readout, heldout_vector) == (0, 1, 0, 0)

    exact_update = homogeneous_operator(task, AlgebraAction.update(0, 0))
    adversarial_update = adversarial_update_operator(
        task,
        key=0,
        heldout_value=0,
        argument=0,
        alternative_result=2,
    )
    assert all(
        apply_integer_matrix(exact_update, encode_homogeneous_state(task, state))
        == apply_integer_matrix(
            adversarial_update, encode_homogeneous_state(task, state)
        )
        for state in train_states
    )
    assert decode_homogeneous_state(
        task, apply_integer_matrix(exact_update, heldout_vector)
    ).values == (1, -1, -1, -1, -1)
    assert decode_homogeneous_state(
        task, apply_integer_matrix(adversarial_update, heldout_vector)
    ).values == (2, -1, -1, -1, -1)

    empty_vector = encode_homogeneous_state(
        task, SemanticState((-1, -1, -1, -1, -1))
    )
    exact_bind = homogeneous_operator(task, AlgebraAction.bind(0, 0))
    adversarial_bind = adversarial_bind_operator(
        task, key=0, heldout_value=0, alternative_value=1
    )
    assert decode_homogeneous_state(
        task, apply_integer_matrix(exact_bind, empty_vector)
    ).values == (0, -1, -1, -1, -1)
    assert decode_homogeneous_state(
        task, apply_integer_matrix(adversarial_bind, empty_vector)
    ).values == (1, -1, -1, -1, -1)

    witness = report.witness
    assert witness.train_states_checked == 708
    assert witness.heldout_state == heldout_state.values
    assert (witness.true_query_value, witness.alternative_query_value) == (0, 1)
    assert (witness.true_updated_value, witness.alternative_updated_value) == (
        1,
        2,
    )
    assert witness.readouts_equal_on_all_train_states
    assert witness.operators_equal_on_all_train_states
    assert witness.unseen_bind_train_transitions == 0
    assert (witness.true_bound_value, witness.alternative_bound_value) == (0, 1)


def test_cyclic_completion_reduces_sixteen_functions_to_two_permutations_to_one_cycle(
    report: HeldoutIdentificationReport,
) -> None:
    completion = report.cyclic_update_completion
    assert completion == cyclic_successor_completion_analysis(
        screen_task(), key=0, max_permutations=24
    )
    assert completion.observed_edges == ((1, 2), (2, 3))
    assert completion.unrestricted_function_completions == 16
    assert completion.permutation_completions == (
        (0, 2, 3, 1),
        (1, 2, 3, 0),
    )
    assert completion.transitive_cyclic_completions == ((1, 2, 3, 0),)
    assert completion.cyclic_completion_is_unique


def test_key_and_value_symmetries_recover_the_heldout_transducer(
    report: HeldoutIdentificationReport,
) -> None:
    recovery = report.symmetry_recovery
    assert recovery.assumes_universal_exact_operator_identities
    assert not recovery.identities_are_inferred_from_train_constraints
    assert (recovery.source_key, recovery.heldout_key) == (1, 0)
    assert recovery.key_permutation == (1, 0, 2, 3, 4)
    assert recovery.key_equivariance_recovers_bind
    assert recovery.key_equivariance_recovers_update
    assert recovery.key_equivariance_recovers_query_readout
    assert recovery.key_equivariance_recovers_all_copy_operators
    assert (recovery.source_value, recovery.heldout_value, recovery.value_shift) == (
        1,
        0,
        3,
    )
    assert recovery.value_equivariance_recovers_bind
    assert recovery.value_equivariance_recovers_query_row
    assert recovery.update_commutes_with_value_shift


def test_public_symmetry_operators_obey_exact_conjugation_and_covariance() -> None:
    task = screen_task()
    swap = key_permutation_operator(task, (1, 0, 2, 3, 4))

    def key_conjugate(action: AlgebraAction) -> tuple[tuple[int, ...], ...]:
        return multiply_integer_matrices(
            swap,
            multiply_integer_matrices(homogeneous_operator(task, action), swap),
        )

    assert key_conjugate(AlgebraAction.bind(1, 0)) == homogeneous_operator(
        task, AlgebraAction.bind(0, 0)
    )
    assert key_conjugate(AlgebraAction.update(1, 0)) == homogeneous_operator(
        task, AlgebraAction.update(0, 0)
    )
    assert key_conjugate(AlgebraAction.copy(1, 2)) == homogeneous_operator(
        task, AlgebraAction.copy(0, 2)
    )
    assert multiply_integer_matrices(query_readout_matrix(task, 1), swap) == (
        query_readout_matrix(task, 0)
    )

    shift_to_heldout = value_shift_operator(task, 3)
    inverse_shift = value_shift_operator(task, 1)
    recovered_bind = multiply_integer_matrices(
        shift_to_heldout,
        multiply_integer_matrices(
            homogeneous_operator(task, AlgebraAction.bind(0, 1)), inverse_shift
        ),
    )
    assert recovered_bind == homogeneous_operator(task, AlgebraAction.bind(0, 0))

    update = homogeneous_operator(task, AlgebraAction.update(0, 0))
    assert multiply_integer_matrices(update, shift_to_heldout) == (
        multiply_integer_matrices(shift_to_heldout, update)
    )
    observed_query_row = (query_readout_matrix(task, 0)[1],)
    recovered_query_row = multiply_integer_matrices(
        observed_query_row, inverse_shift
    )[0]
    assert recovered_query_row == query_readout_matrix(task, 0)[0]


def test_identification_limits_and_public_argument_contracts_fail_closed() -> None:
    task = screen_task()
    with pytest.raises(IdentificationLimitError, match="semantic state count"):
        analyze_heldout_identification(task, max_states=820)
    with pytest.raises(IdentificationLimitError, match="train state count"):
        unrestricted_train_system_analysis(task, max_states=707)
    with pytest.raises(IdentificationLimitError, match="permutation limit"):
        cyclic_successor_completion_analysis(
            task, key=0, max_permutations=23
        )

    with pytest.raises(TypeError, match="max_states must be an integer"):
        analyze_heldout_identification(task, max_states=True)
    with pytest.raises(TypeError, match="key must be an integer"):
        cyclic_successor_completion_analysis(task, key=True)
    with pytest.raises(TypeError, match="shift must be an integer"):
        value_shift_operator(task, True)
    with pytest.raises(ValueError, match="outside the task vocabulary"):
        query_readout_matrix(task, 5)
    with pytest.raises(ValueError, match="every key exactly once"):
        key_permutation_operator(task, (0, 0, 2, 3, 4))
    with pytest.raises(TypeError, match="entry 0 must be an integer"):
        key_permutation_operator(task, (False, 1, 2, 3, 4))
    with pytest.raises(TypeError, match="entry 0 must be an integer"):
        key_permutation_operator(task, (0.0, 1, 2, 3, 4))
    with pytest.raises(TypeError, match="key must be an integer"):
        adversarial_bind_operator(
            task,
            key=True,
            heldout_value=0,
            alternative_value=1,
        )
    with pytest.raises(ValueError, match="must differ"):
        adversarial_bind_operator(
            task,
            key=0,
            heldout_value=0,
            alternative_value=0,
        )
    with pytest.raises(TypeError, match="config must be"):
        heldout_feature_indices(object())  # type: ignore[arg-type]

    no_holdout = BindingTaskConfig(
        num_surface_keys=5,
        value_cardinality=4,
        branches=3,
        max_live_bindings=3,
        min_length=10,
        max_length=512,
        heldout_key_value_pairs=(),
    )
    with pytest.raises(ValueError, match="exactly one held-out pair"):
        analyze_heldout_identification(no_holdout, max_states=821)
    assert not pair_completion_analysis(
        BindingAlgebraSpec(2, 2, 2, (), branches=2)
    ).localized_pair_interaction_witness_exists
    with pytest.raises(ValueError, match="at least two keys"):
        analyze_heldout_identification(
            BindingAlgebraSpec(1, 2, 1, ((0, 0),), branches=1),
            max_states=3,
        )
    with pytest.raises(ValueError, match="two live bindings"):
        analyze_heldout_identification(
            BindingAlgebraSpec(2, 2, 1, ((0, 0),), branches=1),
            max_states=5,
        )
    huge_alphabet = BindingAlgebraSpec(
        2, 100_000, 2, ((0, 0),), branches=2
    )
    with pytest.raises(IdentificationLimitError, match="permutation limit"):
        cyclic_successor_completion_analysis(
            huge_alphabet, key=0, max_permutations=1
        )


def test_identification_report_is_deterministic(
    report: HeldoutIdentificationReport,
) -> None:
    repeated = analyze_heldout_identification(
        screen_task(), max_states=821, max_permutations=24
    )
    assert repeated == report
    assert repeated.to_dict() == report.to_dict()
