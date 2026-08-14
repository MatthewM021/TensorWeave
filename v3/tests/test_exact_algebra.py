from __future__ import annotations

import math

import pytest

from tnlm_v3.data import (
    BindingEventKind,
    BindingTaskConfig,
    generate_binding_episode,
)
from tnlm_v3.exact_algebra import (
    AlgebraAction,
    AlgebraContract,
    EnumerationLimitError,
    SegmentTransformer,
    SemanticState,
    apply_action,
    apply_integer_matrix,
    canonical_visible_action_count,
    canonical_visible_actions,
    diagnostic_probe_matrix,
    diagnostic_probe_rank,
    encode_homogeneous_state,
    enumerate_semantic_states,
    exact_matrix_rank,
    full_homogeneous_dimension,
    homogeneous_operator,
    homogeneous_state_feature_matrix,
    lane_permutation_quotient_state_count,
    multiply_integer_matrices,
    oracle_lane_state_count,
    promised_query_realization_upper_bound,
    replay_episode,
    semantic_state_count,
    strict_grammar_hankel_rank,
    strict_grammar_rank_certificate,
    strict_grammar_rank_upper_bound,
    train_semantic_state_count,
    transformer_for_actions,
)


def smoke_task() -> BindingTaskConfig:
    return BindingTaskConfig(
        num_surface_keys=2,
        value_cardinality=2,
        branches=2,
        max_live_bindings=2,
        min_length=10,
        max_length=512,
        heldout_key_value_pairs=(),
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


def test_smallest_complete_smoke_has_nine_states_and_rank_five() -> None:
    task = smoke_task()
    states = enumerate_semantic_states(task, max_states=9)
    matrix = diagnostic_probe_matrix(task, max_states=9)

    assert len(states) == semantic_state_count(task) == 9
    rank = exact_matrix_rank(matrix)
    assert rank.rational_rank == 5
    assert rank.agrees_across_fields
    assert diagnostic_probe_rank(task, max_states=9).rational_rank == 5
    assert full_homogeneous_dimension(task) == 5


def test_screen_semantics_has_821_states_and_rank_21() -> None:
    task = screen_task()
    states = enumerate_semantic_states(task, max_states=821)
    diagnostics = diagnostic_probe_matrix(task, max_states=821)
    features = homogeneous_state_feature_matrix(task, max_states=821)

    assert len(states) == semantic_state_count(task) == 821
    assert len(diagnostics) == len(features) == 821
    assert len(diagnostics[0]) == 25
    assert len(features[0]) == 21
    diagnostic_rank = exact_matrix_rank(diagnostics)
    feature_rank = exact_matrix_rank(features)
    assert diagnostic_rank.rational_rank == feature_rank.rational_rank == 21
    assert diagnostic_rank.agrees_across_fields
    assert feature_rank.agrees_across_fields
    assert diagnostic_probe_rank(task, max_states=821).rational_rank == 21
    assert full_homogeneous_dimension(task) == 21


def test_heldout_train_restriction_removes_one_predictive_direction() -> None:
    task = screen_task()
    states = enumerate_semantic_states(
        task, exclude_heldout=True, max_states=708
    )
    matrix = diagnostic_probe_matrix(
        task, exclude_heldout=True, max_states=708
    )

    assert len(states) == train_semantic_state_count(task) == 708
    assert exact_matrix_rank(matrix).rational_rank == 20
    assert diagnostic_probe_rank(
        task, exclude_heldout=True, max_states=708
    ).rational_rank == 20


def test_contract_dimensions_are_named_and_distinct() -> None:
    task = screen_task()

    assert promised_query_realization_upper_bound(task) == 16
    assert full_homogeneous_dimension(task) == 21
    assert strict_grammar_hankel_rank(task) == 192


def test_strict_rank_has_a_bounded_exact_certificate() -> None:
    task = screen_task()
    certificate = strict_grammar_rank_certificate(task)

    assert certificate.rank == 192
    assert certificate.semantic_states_with_sink == 822
    assert certificate.supported_actions == 67
    assert certificate.base_observations == 26
    assert certificate.gf2_lower_bound == 192
    assert certificate.structural_upper_bound == 192
    assert certificate.transition_cell_evaluations == 1_187_790
    assert strict_grammar_rank_upper_bound(task) == certificate.rank


def test_screen_supported_alphabet_excludes_identity_updates() -> None:
    task = screen_task()
    supported = canonical_visible_actions(task)
    full_syntactic = canonical_visible_actions(
        task, full_syntactic=True
    )

    assert len(supported) == canonical_visible_action_count(task) == 67
    assert len(full_syntactic) == canonical_visible_action_count(
        task, full_syntactic=True
    ) == 72
    supported_updates = {
        action.argument
        for action in supported
        if action.kind is BindingEventKind.UPDATE
    }
    full_update_arguments = {
        action.argument
        for action in full_syntactic
        if action.kind is BindingEventKind.UPDATE
    }
    assert supported_updates == {0, 1, 2}
    assert full_update_arguments == {0, 1, 2, 3}


def expected_strict_legality(
    task: BindingTaskConfig, state: SemanticState, action: AlgebraAction
) -> bool:
    if action.kind is BindingEventKind.BIND:
        return (
            state.values[action.primary_key] < 0
            and state.live_count < task.max_live_bindings
        )
    if action.kind in (
        BindingEventKind.UPDATE,
        BindingEventKind.INVALIDATE,
        BindingEventKind.QUERY,
    ):
        return state.values[action.primary_key] >= 0
    if action.kind is BindingEventKind.COPY:
        return (
            state.values[action.primary_key] >= 0
            and state.values[action.secondary_key] >= 0
        )
    return True


def test_homogeneous_and_symbolic_actions_match_every_legal_transition() -> None:
    task = screen_task()
    states = enumerate_semantic_states(task, max_states=821)
    actions = canonical_visible_actions(task)
    operators = {
        action: homogeneous_operator(task, action) for action in actions
    }
    transformers = {
        action: SegmentTransformer.for_action(task, action) for action in actions
    }

    for state in states:
        encoded = encode_homogeneous_state(task, state)
        for action in actions:
            promised = apply_action(
                task, state, action, contract=AlgebraContract.PROMISED
            )
            assert promised.defined and promised.state is not None
            promised_encoded = apply_integer_matrix(
                operators[action], encoded
            )
            assert promised_encoded == encode_homogeneous_state(
                task, promised.state
            )
            assert transformers[action].apply(state) == promised.state

            strict = apply_action(
                task, state, action, contract=AlgebraContract.STRICT
            )
            assert strict.defined is expected_strict_legality(task, state, action)
            if not strict.defined:
                assert strict.state is None
                assert strict.query_target is None
                continue

            assert strict.state is not None
            assert promised.state == strict.state
            assert promised.query_target == strict.query_target
            assert promised_encoded == encode_homogeneous_state(
                task, strict.state
            )
            assert transformers[action].apply(state) == strict.state
            assert (
                transformers[action].to_homogeneous_operator()
                == operators[action]
            )
            if action.kind is BindingEventKind.QUERY:
                assert strict.query_target == state.values[action.primary_key]
            else:
                assert strict.query_target is None


def test_symbolic_segment_composition_is_associative_and_matches_products() -> None:
    task = screen_task()
    actions = (
        AlgebraAction.bind(0, 1),
        AlgebraAction.bind(1, 2),
        AlgebraAction.copy(0, 1),
        AlgebraAction.update(0, 0),
    )
    transformers = tuple(
        SegmentTransformer.for_action(task, action) for action in actions
    )
    left_associated = (
        transformers[3]
        .compose(transformers[2])
        .compose(transformers[1])
        .compose(transformers[0])
    )
    right_associated = transformers[3].compose(
        transformers[2].compose(
            transformers[1].compose(transformers[0])
        )
    )
    summarized = transformer_for_actions(task, actions)

    assert left_associated == right_associated == summarized

    matrix = SegmentTransformer.identity(task).to_homogeneous_operator()
    for action in actions:
        matrix = multiply_integer_matrices(
            homogeneous_operator(task, action), matrix
        )
    assert summarized.to_homogeneous_operator() == matrix

    state = SemanticState((-1, -1, -1, -1, -1))
    sequential = state
    for action in actions:
        transition = apply_action(task, sequential, action)
        assert transition.state is not None
        sequential = transition.state
    assert summarized.apply(state) == sequential
    assert apply_integer_matrix(matrix, encode_homogeneous_state(task, state)) == (
        encode_homogeneous_state(task, sequential)
    )


def test_disjoint_updates_commute_but_copy_then_source_update_does_not() -> None:
    task = screen_task()
    state = SemanticState((0, 1, -1, -1, -1))
    update_zero = AlgebraAction.update(0, 0)
    update_one = AlgebraAction.update(1, 0)
    copy_zero_from_one = AlgebraAction.copy(0, 1)

    disjoint_left = transformer_for_actions(
        task, (update_zero, update_one)
    )
    disjoint_right = transformer_for_actions(
        task, (update_one, update_zero)
    )
    assert disjoint_left == disjoint_right
    assert disjoint_left.apply(state) == SemanticState((1, 2, -1, -1, -1))

    copy_then_update = transformer_for_actions(
        task, (copy_zero_from_one, update_one)
    )
    update_then_copy = transformer_for_actions(
        task, (update_one, copy_zero_from_one)
    )
    assert copy_then_update != update_then_copy
    assert copy_then_update.apply(state) == SemanticState(
        (1, 2, -1, -1, -1)
    )
    assert update_then_copy.apply(state) == SemanticState(
        (2, 2, -1, -1, -1)
    )


def test_strict_contract_exposes_illegality_while_total_operator_stays_separate() -> None:
    task = smoke_task()
    empty = SemanticState((-1, -1))
    illegal_query = AlgebraAction.query(0)

    strict = apply_action(
        task,
        empty,
        illegal_query,
        contract=AlgebraContract.STRICT,
    )
    assert not strict.defined
    assert strict.state is None
    for action in canonical_visible_actions(task):
        absorbing = apply_action(
            task,
            strict.state,
            action,
            contract=AlgebraContract.STRICT,
        )
        assert not absorbing.defined
        assert absorbing.state is None
        assert absorbing.query_target is None

    promised = apply_action(
        task, empty, illegal_query, contract=AlgebraContract.PROMISED
    )
    assert promised.defined
    assert promised.state == empty
    assert promised.query_target is None

    live_destination = SemanticState((0, -1))
    illegal_copy = AlgebraAction.copy(0, 1)
    strict_copy = apply_action(
        task,
        live_destination,
        illegal_copy,
        contract=AlgebraContract.STRICT,
    )
    promised_copy = apply_action(
        task,
        live_destination,
        illegal_copy,
        contract=AlgebraContract.PROMISED,
    )
    assert not strict_copy.defined and strict_copy.state is None
    assert promised_copy.defined
    assert promised_copy.state == empty

    assert apply_integer_matrix(
        homogeneous_operator(task, illegal_query),
        encode_homogeneous_state(task, empty),
    ) == encode_homogeneous_state(task, empty)


def test_promised_completion_remains_closed_after_crossing_the_live_cap() -> None:
    task = screen_task()
    capped = SemanticState((0, 1, 2, -1, -1))
    bind_fourth = AlgebraAction.bind(3, 3)
    update_fourth = AlgebraAction.update(3, 0)

    strict = apply_action(
        task, capped, bind_fourth, contract=AlgebraContract.STRICT
    )
    promised = apply_action(
        task, capped, bind_fourth, contract=AlgebraContract.PROMISED
    )
    assert not strict.defined and strict.state is None
    assert promised.defined
    assert promised.state == SemanticState((0, 1, 2, 3, -1))
    assert promised.state.live_count == 4

    continued = apply_action(
        task,
        promised.state,
        update_fourth,
        contract=AlgebraContract.PROMISED,
    )
    assert continued.defined
    assert continued.state == SemanticState((0, 1, 2, 0, -1))
    assert transformer_for_actions(
        task, (bind_fourth, update_fourth)
    ).apply(capped) == continued.state


@pytest.mark.parametrize("split", ("train", "eval"))
def test_exact_replay_matches_generated_queries_through_length_512(
    split: str,
) -> None:
    task = screen_task()
    episode = generate_binding_episode(
        task,
        length=512,
        seed=8_149,
        split=split,
        document_index=3,
    )
    final_state = replay_episode(task, episode)

    assert isinstance(final_state, SemanticState)
    assert int(
        (episode.inputs.event_kinds == int(BindingEventKind.QUERY)).sum()
    ) > 0
    if split == "eval":
        query_mask = episode.inputs.event_kinds == int(BindingEventKind.QUERY)
        assert bool(
            (
                query_mask
                & episode.evaluation.heldout_combination_mask
            ).any()
        )


def test_lane_labels_are_a_larger_gauge_dependent_state_space() -> None:
    task = screen_task()

    assert lane_permutation_quotient_state_count(task) == 821
    assert oracle_lane_state_count(task) == 4_861
    assert oracle_lane_state_count(task, exclude_heldout=True) == 4_186
    direct_formula = sum(
        math.comb(task.num_surface_keys, live_count)
        * task.value_cardinality**live_count
        * math.perm(task.branches, live_count)
        for live_count in range(task.max_live_bindings + 1)
    )
    assert direct_formula == 4_861


def test_enumeration_budget_fails_before_oversized_materialization() -> None:
    task = screen_task()

    with pytest.raises(EnumerationLimitError):
        enumerate_semantic_states(task, max_states=820)
    with pytest.raises(EnumerationLimitError):
        diagnostic_probe_matrix(task, max_states=820)
    with pytest.raises(EnumerationLimitError):
        strict_grammar_rank_certificate(task, max_states=821)
    with pytest.raises(EnumerationLimitError):
        strict_grammar_rank_certificate(task, max_actions=66)
    with pytest.raises(EnumerationLimitError):
        strict_grammar_rank_certificate(
            task, max_transition_cell_evaluations=1
        )
