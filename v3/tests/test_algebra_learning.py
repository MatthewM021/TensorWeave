from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

import tnlm_v3.algebra_learning as learning
from tnlm_v3.algebra_learning import (
    AlgebraLearnabilityReport,
    AlgebraLearningLimitError,
    CanonicalPrototypeObservation,
    CanonicalSupervisionSource,
    HypothesisLearnability,
    LearnedSharedPrototypeTransducer,
    LearningHypothesis,
    PrototypeCell,
    SharedPrototypeCoverage,
    SharedPrototypeFitEvaluation,
    SharedPrototypeCoverageSweep,
    analyze_algebra_learnability,
    evaluate_shared_prototype_transducer,
    fit_shared_prototype_transducer,
    generate_and_fit_shared_prototype_transducer,
    generate_shared_prototype_coverage,
    oracle_canonical_observations_from_episodes,
    pseudoheldout_fold_certificates,
    shared_prototype_cells,
    shared_prototype_coverage_from_episodes,
    shared_prototype_parameter_count,
    sweep_shared_prototype_coverage,
)
from tnlm_v3.data import (
    BindingEventKind,
    BindingTaskConfig,
    generate_binding_episode,
    generate_binding_episodes,
)
from tnlm_v3.exact_algebra import BindingAlgebraSpec


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


def external_observations_for_coverage(
    coverage: SharedPrototypeCoverage,
) -> tuple[CanonicalPrototypeObservation, ...]:
    """Replay visible fields locally into an independent canonical record."""

    task = screen_task()
    assert coverage.task_fingerprint == task.fingerprint()
    assert coverage.split == "train"
    episodes = generate_binding_episodes(
        task,
        count=coverage.document_count,
        seed=coverage.seed,
        split="train",
        lengths=(coverage.document_length,) * coverage.document_count,
    )
    result: list[CanonicalPrototypeObservation] = []
    for episode in episodes:
        state = [-1] * task.num_surface_keys
        for event_index in range(episode.length):
            kind = BindingEventKind(int(episode.inputs.event_kinds[event_index]))
            primary = int(episode.inputs.primary_key_ids[event_index]) - 1
            secondary = int(episode.inputs.secondary_key_ids[event_index]) - 1
            argument = int(episode.inputs.arguments[event_index]) - 1
            pre_state = tuple(state)
            cell_label: str | None = None
            query_target: int | None = None
            if kind is BindingEventKind.BIND:
                cell_label = f"bind:source:{argument}"
                state[primary] = argument
            elif kind is BindingEventKind.UPDATE:
                source = state[primary]
                cell_label = f"update:{argument}:source:{source}"
                state[primary] = (source + argument + 1) % task.value_cardinality
            elif kind is BindingEventKind.COPY:
                source = state[secondary]
                cell_label = f"copy:source:{source}"
                state[primary] = source
            elif kind is BindingEventKind.INVALIDATE:
                state[primary] = -1
            elif kind is BindingEventKind.QUERY:
                query_target = state[primary]
                cell_label = f"query:source:{query_target}"
            post_state = tuple(state)
            if cell_label is None:
                continue
            target_value = (
                query_target if kind is BindingEventKind.QUERY else state[primary]
            )
            assert target_value is not None
            target = tuple(
                int(index == target_value)
                for index in range(task.value_cardinality)
            )
            result.append(
                CanonicalPrototypeObservation(
                    task_fingerprint=task.fingerprint(),
                    sample_sha256=coverage.sample_sha256,
                    split="train",
                    supervision_source=(
                        CanonicalSupervisionSource.EXTERNAL_TRANSITION_RECORD
                    ),
                    cell_label=cell_label,
                    target=target,
                    source_document_id=episode.document_id,
                    event_index=event_index,
                    action_kind=int(kind),
                    primary_key=primary,
                    secondary_key=secondary,
                    argument=argument,
                    pre_state=pre_state,
                    post_state=post_state,
                    query_target=query_target,
                )
            )
    assert tuple(
        (
            row.source_document_id,
            row.event_index,
            row.cell_label,
            row.action_kind,
            row.primary_key,
            row.secondary_key,
            row.argument,
        )
        for row in result
    ) == coverage.cell_event_locations
    return tuple(result)


@pytest.fixture(scope="module")
def report() -> AlgebraLearnabilityReport:
    return analyze_algebra_learnability(
        screen_task(), max_states=821, max_permutations=24
    )


@pytest.fixture(scope="module")
def sparse_coverage() -> SharedPrototypeCoverage:
    return generate_shared_prototype_coverage(
        screen_task(),
        split="train",
        seed=0,
        document_count=1,
        document_length=64,
    )


@pytest.fixture(scope="module")
def full_coverage() -> SharedPrototypeCoverage:
    return generate_shared_prototype_coverage(
        screen_task(),
        split="train",
        seed=0,
        document_count=8,
        document_length=64,
    )


@pytest.fixture(scope="module")
def full_fit() -> LearnedSharedPrototypeTransducer:
    return generate_and_fit_shared_prototype_transducer(
        screen_task(),
        split="train",
        seed=0,
        document_count=8,
        document_length=64,
    )


@pytest.fixture(scope="module")
def sparse_fit() -> LearnedSharedPrototypeTransducer:
    return generate_and_fit_shared_prototype_transducer(
        screen_task(),
        split="train",
        seed=0,
        document_count=1,
        document_length=64,
    )


def test_exact_learnability_boundary_and_claim_scope(
    report: AlgebraLearnabilityReport,
) -> None:
    assert report.schema == "tnlm-v3-algebra-learnability-v1"
    assert report.task_fingerprint == screen_task().fingerprint()
    assert report.heldout_pair == (0, 0)
    assert (report.full_state_count, report.train_state_count) == (821, 708)
    assert (report.full_feature_rank, report.train_feature_rank) == (21, 20)
    assert report.passive_unrestricted_nullity == 4_912
    assert not report.passive_uniform_recovery_possible
    assert report.zero_support_identical_likelihood_witness_exists
    assert report.one_separating_membership_probe == (
        "BIND(0,0)",
        "QUERY(0)",
    )

    # This report is deliberately a canonical-state system-identification
    # control.  It must not be misreported as latent sequence learning or as
    # automatic discovery of the sharing law.
    assert not report.sequence_only_representation_learning_performed
    assert not report.automatic_hypothesis_selection_performed
    assert all(row.canonical_state_supervision_assumed for row in report.hypotheses)
    assert all(not row.sharing_law_selected_from_data for row in report.hypotheses)

    assert report.report_sha256 == (
        "c890990e0e97cf32b963c99711f5560c848bfa4761b0c5d01aae9aa8ca85936f"
    )
    repeated = analyze_algebra_learnability(
        screen_task(), max_states=821, max_permutations=24
    )
    assert repeated == report


def test_hypothesis_ladder_has_exact_ranks_and_supplied_law_flags(
    report: AlgebraLearnabilityReport,
) -> None:
    assert [row.hypothesis for row in report.hypotheses] == list(
        LearningHypothesis
    )
    assert [
        (row.parameter_count, row.constraint_rank, row.nullity)
        for row in report.hypotheses
    ] == [
        (29_967, 25_055, 4_912),
        (720, 656, 64),
        (224, 216, 8),
        (96, 96, 0),
        (40, 40, 0),
        (8, 8, 0),
    ]
    assert [
        row.heldout_behavior_identified_conditionally
        for row in report.hypotheses
    ] == [False, False, False, True, True, True]
    assert [row.universal_law_supplied for row in report.hypotheses] == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]


def test_shared_prototype_vocabulary_is_exact_and_value_generic() -> None:
    task = screen_task()
    cells = shared_prototype_cells(task)

    assert len(cells) == 24
    assert shared_prototype_parameter_count(task) == 96
    assert tuple(cell.label for cell in cells) == (
        "bind:source:0",
        "bind:source:1",
        "bind:source:2",
        "bind:source:3",
        "update:0:source:0",
        "update:0:source:1",
        "update:0:source:2",
        "update:0:source:3",
        "update:1:source:0",
        "update:1:source:1",
        "update:1:source:2",
        "update:1:source:3",
        "update:2:source:0",
        "update:2:source:1",
        "update:2:source:2",
        "update:2:source:3",
        "copy:source:0",
        "copy:source:1",
        "copy:source:2",
        "copy:source:3",
        "query:source:0",
        "query:source:1",
        "query:source:2",
        "query:source:3",
    )

    binary = BindingAlgebraSpec(2, 2, 2, ((0, 0),), branches=2)
    assert len(shared_prototype_cells(binary)) == 8
    assert shared_prototype_parameter_count(binary) == 16


def test_every_observed_pair_is_a_full_rank_pseudoheldout_fold(
    report: AlgebraLearnabilityReport,
) -> None:
    folds = report.pseudoheldout_folds
    expected_pairs = tuple(
        (key, value)
        for key in range(5)
        for value in range(4)
        if (key, value) != (0, 0)
    )

    assert folds == pseudoheldout_fold_certificates(screen_task())
    assert len(folds) == 19
    assert tuple(row.pseudoheldout_pair for row in folds) == expected_pairs
    assert all(row.real_heldout_pair == (0, 0) for row in folds)
    assert all(
        (
            row.supplied_shared_parameter_count,
            row.supplied_shared_design_rank,
            row.supplied_shared_nullity,
        )
        == (96, 96, 0)
        for row in folds
    )
    assert all(row.supplied_shared_identifies_pseudoheldout for row in folds)
    assert all(not row.automatic_hypothesis_selection_performed for row in folds)

    with pytest.raises(ValueError, match="one real held-out pair"):
        pseudoheldout_fold_certificates(
            replace(screen_task(), heldout_key_value_pairs=())
        )
    with pytest.raises(ValueError, match="one real held-out pair"):
        pseudoheldout_fold_certificates(
            replace(
                screen_task(), heldout_key_value_pairs=((0, 0), (1, 1))
            )
        )
    with pytest.raises(ValueError, match="two live keys"):
        pseudoheldout_fold_certificates(
            BindingAlgebraSpec(
                num_surface_keys=2,
                value_cardinality=2,
                max_live_bindings=1,
                heldout_key_value_pairs=((0, 0),),
                branches=1,
            )
        )


def test_two_key_pseudoheldout_ranks_respect_copy_successor_exclusions() -> None:
    binary = pseudoheldout_fold_certificates(
        BindingAlgebraSpec(
            num_surface_keys=2,
            value_cardinality=2,
            max_live_bindings=2,
            heldout_key_value_pairs=((0, 0),),
            branches=2,
        )
    )
    assert [
        (
            row.pseudoheldout_pair,
            row.supplied_shared_design_rank,
            row.supplied_shared_nullity,
            row.supplied_shared_identifies_pseudoheldout,
        )
        for row in binary
    ] == [
        ((0, 1), 12, 4, False),
        ((1, 0), 6, 10, False),
        ((1, 1), 8, 8, False),
    ]

    quaternary = pseudoheldout_fold_certificates(
        BindingAlgebraSpec(
            num_surface_keys=2,
            value_cardinality=4,
            max_live_bindings=2,
            heldout_key_value_pairs=((0, 0),),
            branches=2,
        )
    )
    assert [
        (
            row.pseudoheldout_pair,
            row.supplied_shared_design_rank,
            row.supplied_shared_nullity,
            row.supplied_shared_identifies_pseudoheldout,
        )
        for row in quaternary
    ] == [
        ((0, 1), 88, 8, False),
        ((0, 2), 88, 8, False),
        ((0, 3), 88, 8, False),
        ((1, 0), 60, 36, False),
        ((1, 1), 80, 16, False),
        ((1, 2), 80, 16, False),
        ((1, 3), 80, 16, False),
    ]


def test_finite_coverage_has_exact_sparse_and_saturated_ranks(
    sparse_coverage: SharedPrototypeCoverage,
    full_coverage: SharedPrototypeCoverage,
) -> None:
    sparse = sparse_coverage
    assert sparse.schema == "tnlm-v3-shared-prototype-coverage-v1"
    assert sparse.task_fingerprint == screen_task().fingerprint()
    assert (sparse.split, sparse.seed) == ("train", 0)
    assert (sparse.document_count, sparse.document_length, sparse.event_count) == (
        1,
        64,
        64,
    )
    assert (
        sparse.parameter_count,
        sparse.design_rank,
        sparse.nullity,
        sparse.observed_cell_count,
        sparse.total_cell_count,
    ) == (96, 68, 28, 17, 24)
    assert sparse.missing_cells == (
        "bind:source:0",
        "update:0:source:1",
        "update:0:source:3",
        "update:1:source:0",
        "update:2:source:0",
        "update:2:source:1",
        "update:2:source:2",
    )
    assert len(sparse.cell_observation_counts) == 24
    assert sum(count for _, count in sparse.cell_observation_counts) == 47
    assert sparse.smallest_nonzero_singular_value == 1.0
    assert sparse.largest_singular_value == pytest.approx(math.sqrt(11.0))
    assert sparse.condition_number == pytest.approx(math.sqrt(11.0))
    assert not sparse.full_rank
    assert sparse.sample_sha256 == (
        "a32b74ca764b151198a96e9c62e88b32962fb94a4bd383745759b50dc2b75f65"
    )

    full = full_coverage
    assert (full.document_count, full.document_length, full.event_count) == (
        8,
        64,
        512,
    )
    assert (full.parameter_count, full.design_rank, full.nullity) == (96, 96, 0)
    assert (full.observed_cell_count, full.total_cell_count) == (24, 24)
    assert full.missing_cells == ()
    assert full.smallest_nonzero_singular_value == 2.0
    assert full.largest_singular_value == pytest.approx(math.sqrt(52.0))
    assert full.condition_number == pytest.approx(math.sqrt(13.0))
    assert full.full_rank
    assert full.canonical_state_supervision_assumed
    assert not full.used_heldout_labels_for_fit
    assert not full.used_heldout_labels_for_selection
    assert full.sample_sha256 == (
        "91f7a9cb288f136c79ab10f9cd3ba73a4177f70c96a0316298e3e3227551f6e9"
    )

    repeated = generate_shared_prototype_coverage(
        screen_task(),
        split="train",
        seed=0,
        document_count=8,
        document_length=64,
    )
    assert repeated == full


def test_finite_coverage_sweep_preserves_seed_order_and_exact_ranks() -> None:
    sweep = sweep_shared_prototype_coverage(
        screen_task(),
        split="train",
        seeds=iter(range(5)),
        document_count=1,
        document_length=64,
    )
    assert sweep.schema == "tnlm-v3-shared-prototype-coverage-sweep-v1"
    assert sweep.seeds == (0, 1, 2, 3, 4)
    assert [row.design_rank for row in sweep.results] == [68, 84, 64, 84, 60]
    assert (sweep.full_rank_count, sweep.minimum_rank, sweep.maximum_rank) == (
        0,
        60,
        84,
    )
    assert sweep.mean_rank == 72.0
    assert sweep == sweep_shared_prototype_coverage(
        screen_task(),
        split="train",
        seeds=range(5),
        document_count=1,
        document_length=64,
    )


def test_saturated_sample_learns_all_tables_without_supplying_cyclic_law(
    full_fit: LearnedSharedPrototypeTransducer,
    full_coverage: SharedPrototypeCoverage,
) -> None:
    model = full_fit
    assert model.schema == "tnlm-v3-learned-shared-prototype-transducer-v1"
    assert model.task_fingerprint == screen_task().fingerprint()
    assert model.value_cardinality == 4
    assert model.coverage == full_coverage
    assert len(model.cell_outputs) == 24
    assert all(output is not None for _, output in model.cell_outputs)
    assert model.canonical_state_supervision_assumed
    assert model.universal_key_sharing_supplied
    assert not model.cyclic_law_supplied
    assert model.supervision_source is CanonicalSupervisionSource.EXACT_EXECUTOR_CONTROL
    assert len(model.observation_sha256) == 64
    assert not model.used_heldout_labels_for_fit
    assert not model.automatic_hypothesis_selection_performed

    expected: dict[str, tuple[int, ...]] = {}
    for cell in shared_prototype_cells(screen_task()):
        if cell.family == "update":
            assert cell.transform is not None
            value = (cell.source_value + cell.transform + 1) % 4
        else:
            value = cell.source_value
        expected[cell.label] = tuple(int(index == value) for index in range(4))
    assert dict(model.cell_outputs) == expected

    evaluation = evaluate_shared_prototype_transducer(screen_task(), model)
    assert evaluation == SharedPrototypeFitEvaluation(
        schema="tnlm-v3-shared-prototype-fit-evaluation-v1",
        evaluated_cell_count=24,
        exact_cell_count=24,
        missing_cell_count=0,
        all_value_tables_exact=True,
        all_valid_programs_exact_given_supplied_state_contract=True,
        heldout_semantics_used_only_for_postfit_evaluation=True,
    )


def test_direct_fitter_consumes_external_records_without_calling_executor(
    monkeypatch: pytest.MonkeyPatch,
    full_coverage: SharedPrototypeCoverage,
) -> None:
    observations = external_observations_for_coverage(full_coverage)

    def forbidden_executor(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("direct fitting called the exact executor")

    monkeypatch.setattr(learning, "apply_action", forbidden_executor)
    monkeypatch.setattr(learning, "generate_binding_episodes", forbidden_executor)
    monkeypatch.setattr(
        learning, "_oracle_observations_from_validated_episodes", forbidden_executor
    )
    with pytest.raises(
        TypeError, match="unexpected keyword argument.*supervision_source"
    ):
        fit_shared_prototype_transducer(
            screen_task(),
            observations,
            coverage=full_coverage,
            supervision_source=CanonicalSupervisionSource.EXACT_EXECUTOR_CONTROL,
        )  # type: ignore[call-arg]
    model = fit_shared_prototype_transducer(
        screen_task(),
        observations,
        coverage=full_coverage,
    )
    repeated = fit_shared_prototype_transducer(
        screen_task(),
        observations,
        coverage=full_coverage,
    )

    assert repeated == model
    assert model.coverage == full_coverage
    assert model.supervision_source is (
        CanonicalSupervisionSource.EXTERNAL_TRANSITION_RECORD
    )
    assert model.observation_sha256 == repeated.observation_sha256
    assert all(output is not None for _, output in model.cell_outputs)


def test_oracle_observations_cannot_enter_the_public_external_fit_path(
    sparse_coverage: SharedPrototypeCoverage,
) -> None:
    task = screen_task()
    episodes = generate_binding_episodes(
        task,
        count=1,
        seed=0,
        split="train",
        lengths=(64,),
    )
    oracle_observations = oracle_canonical_observations_from_episodes(
        task,
        episodes,
        split="train",
        seed=0,
        document_length=64,
    )
    assert oracle_observations
    assert all(
        row.supervision_source is CanonicalSupervisionSource.EXACT_EXECUTOR_CONTROL
        for row in oracle_observations
    )
    with pytest.raises(ValueError, match="provenance|fit entry path"):
        fit_shared_prototype_transducer(
            task,
            oracle_observations,
            coverage=sparse_coverage,
        )

    external = external_observations_for_coverage(sparse_coverage)
    forged_source = replace(
        external[0],
        supervision_source=CanonicalSupervisionSource.EXACT_EXECUTOR_CONTROL,
    )
    with pytest.raises(ValueError, match="provenance|fit entry path"):
        fit_shared_prototype_transducer(
            task,
            (forged_source, *external[1:]),
            coverage=sparse_coverage,
        )
    with pytest.raises(TypeError, match="supervision_source|exact"):
        replace(
            external[0],
            supervision_source="external_canonical_transition_record",
        )


def test_direct_fitter_rejects_non_one_hot_wrong_count_and_duplicate_records(
    full_coverage: SharedPrototypeCoverage,
) -> None:
    observations = external_observations_for_coverage(full_coverage)
    kwargs = {"coverage": full_coverage}

    non_one_hot = (
        replace(observations[0], target=(1, 1, 0, 0)),
        *observations[1:],
    )
    with pytest.raises(ValueError, match="one-hot"):
        fit_shared_prototype_transducer(
            screen_task(), non_one_hot, **kwargs  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="coverage counts|sample inventory"):
        fit_shared_prototype_transducer(
            screen_task(), observations[:-1], **kwargs  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="duplicate an event|sample inventory"):
        fit_shared_prototype_transducer(
            screen_task(),
            observations + (observations[0],),
            **kwargs,  # type: ignore[arg-type]
        )


def test_direct_fitter_rejects_forged_provenance_actions_and_locations(
    full_coverage: SharedPrototypeCoverage,
) -> None:
    observations = external_observations_for_coverage(full_coverage)

    def fit(rows: tuple[CanonicalPrototypeObservation, ...]) -> None:
        fit_shared_prototype_transducer(
            screen_task(),
            rows,
            coverage=full_coverage,
        )

    with pytest.raises(ValueError, match="task fingerprint"):
        fit((replace(observations[0], task_fingerprint="0" * 64), *observations[1:]))
    with pytest.raises(ValueError, match="sample digest"):
        fit((replace(observations[0], sample_sha256="0" * 64), *observations[1:]))
    with pytest.raises(ValueError, match="train split"):
        replace(observations[0], split="validation")

    wrong_action = replace(
        observations[0],
        action_kind=int(BindingEventKind.DISTRACTOR),
    )
    with pytest.raises(ValueError, match="bound to its sample event"):
        fit((wrong_action, *observations[1:]))

    wrong_location = replace(
        observations[0],
        event_index=observations[0].event_index + full_coverage.event_count + 1,
    )
    with pytest.raises(ValueError, match="bound to its sample event"):
        fit((wrong_location, *observations[1:]))

    invented_location = replace(
        observations[0], source_document_id="invented-document"
    )
    with pytest.raises(ValueError, match="bound to its sample event"):
        fit((invented_location, *observations[1:]))

    with pytest.raises(
        ValueError, match="coverage counts|cover the sample events|sample inventory"
    ):
        fit(observations[:-1])


def test_direct_fitter_rejects_heldout_and_internally_inconsistent_records(
    full_coverage: SharedPrototypeCoverage,
) -> None:
    observations = external_observations_for_coverage(full_coverage)

    def fit_replacement(index: int, row: CanonicalPrototypeObservation) -> None:
        fit_shared_prototype_transducer(
            screen_task(),
            (*observations[:index], row, *observations[index + 1 :]),
            coverage=full_coverage,
        )

    pre_index = next(
        index for index, row in enumerate(observations) if row.pre_state[0] >= 0
    )
    heldout_pre = list(observations[pre_index].pre_state)
    heldout_pre[0] = 0
    with pytest.raises(ValueError, match="pre_state contains a held-out pair"):
        fit_replacement(
            pre_index,
            replace(observations[pre_index], pre_state=tuple(heldout_pre)),
        )

    post_index = next(
        index for index, row in enumerate(observations) if row.post_state[0] >= 0
    )
    heldout_post = list(observations[post_index].post_state)
    heldout_post[0] = 0
    with pytest.raises(ValueError, match="post_state contains a held-out pair"):
        fit_replacement(
            post_index,
            replace(observations[post_index], post_state=tuple(heldout_post)),
        )

    query_index = next(
        index
        for index, row in enumerate(observations)
        if row.action_kind == int(BindingEventKind.QUERY)
    )
    query = observations[query_index]
    changed_value = next(
        value
        for value in range(4)
        if value != query.pre_state[query.primary_key]
        and (query.primary_key, value) != (0, 0)
    )
    changed_post = list(query.post_state)
    changed_post[query.primary_key] = changed_value
    with pytest.raises(ValueError, match="query observations must preserve"):
        fit_replacement(
            query_index, replace(query, post_state=tuple(changed_post))
        )

    wrong_query_target = (int(query.query_target) + 1) % 4
    with pytest.raises(ValueError, match="target disagrees"):
        fit_replacement(
            query_index, replace(query, query_target=wrong_query_target)
        )

    wrong_target_value = (int(query.query_target) + 1) % 4
    wrong_target = tuple(
        int(index == wrong_target_value) for index in range(4)
    )
    with pytest.raises(ValueError, match="target disagrees"):
        fit_replacement(query_index, replace(query, target=wrong_target))

    nonquery_index = next(
        index
        for index, row in enumerate(observations)
        if row.action_kind != int(BindingEventKind.QUERY)
    )
    with pytest.raises(ValueError, match="only query observations"):
        fit_replacement(
            nonquery_index,
            replace(observations[nonquery_index], query_target=1),
        )


def test_rank_deficient_sample_leaves_cells_unlearned_and_postfit_inexact(
    sparse_fit: LearnedSharedPrototypeTransducer,
    sparse_coverage: SharedPrototypeCoverage,
) -> None:
    assert sparse_fit.coverage == sparse_coverage
    missing = tuple(
        label for label, output in sparse_fit.cell_outputs if output is None
    )
    assert missing == sparse_coverage.missing_cells
    assert len(missing) == 7

    evaluation = evaluate_shared_prototype_transducer(screen_task(), sparse_fit)
    assert (
        evaluation.evaluated_cell_count,
        evaluation.exact_cell_count,
        evaluation.missing_cell_count,
    ) == (24, 17, 7)
    assert not evaluation.all_value_tables_exact
    assert not evaluation.all_valid_programs_exact_given_supplied_state_contract
    assert evaluation.heldout_semantics_used_only_for_postfit_evaluation


def test_postfit_evaluation_rejects_task_or_model_mismatch(
    full_fit: LearnedSharedPrototypeTransducer,
) -> None:
    same_algebra_different_task_fingerprint = replace(
        screen_task(), global_distractor_probability=0.25
    )
    with pytest.raises(ValueError, match="task fingerprint"):
        evaluate_shared_prototype_transducer(
            same_algebra_different_task_fingerprint, full_fit
        )
    with pytest.raises(TypeError, match="exact LearnedSharedPrototypeTransducer"):
        evaluate_shared_prototype_transducer(
            screen_task(), object()  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="cell inventories"):
        replace(full_fit, cell_outputs=full_fit.cell_outputs[:-1])


def test_seen_only_split_firewall_rejects_eval_and_provenance_relabelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = screen_task()
    calls: list[object] = []

    def forbidden_generation(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        raise AssertionError("generation should not have been reached")

    monkeypatch.setattr(learning, "generate_binding_episodes", forbidden_generation)
    with pytest.raises(ValueError, match="train or validation"):
        generate_shared_prototype_coverage(
            task,
            split="eval",  # type: ignore[arg-type]
            seed=0,
            document_count=1,
            document_length=64,
        )
    with pytest.raises(AlgebraLearningLimitError, match="max_events"):
        generate_shared_prototype_coverage(
            task,
            split="train",
            seed=0,
            document_count=1,
            document_length=64,
            max_events=63,
        )
    assert calls == []

    monkeypatch.undo()
    eval_episode = generate_binding_episode(
        task, length=64, seed=0, split="eval", document_index=0
    )
    assert bool(eval_episode.evaluation.heldout_combination_mask.any())
    with pytest.raises(ValueError, match="held-out|seen-only|deterministic sample"):
        shared_prototype_coverage_from_episodes(
            task,
            (replace(eval_episode, split="train"),),
            split="train",
            seed=0,
            document_length=64,
        )

    other_task = replace(task, heldout_key_value_pairs=((1, 1),))
    wrong_provenance = generate_binding_episode(
        other_task, length=64, seed=0, split="train", document_index=0
    )
    with pytest.raises(ValueError, match="fingerprint|config"):
        shared_prototype_coverage_from_episodes(
            task,
            (wrong_provenance,),
            split="train",
            seed=0,
            document_length=64,
        )


def test_coverage_uses_visible_actions_but_never_evaluation_labels() -> None:
    task = screen_task()
    episode = generate_binding_episode(
        task, length=64, seed=91, split="validation", document_index=0
    )
    expected = shared_prototype_coverage_from_episodes(
        task,
        (episode,),
        split="validation",
        seed=91,
        document_length=64,
    )
    poisoned_evaluation = replace(
        episode.evaluation,
        oracle_routes=torch.full_like(episode.evaluation.oracle_routes, 999),
        targets=torch.full_like(episode.evaluation.targets, 999),
        dependency_parents=torch.full_like(
            episode.evaluation.dependency_parents, 999
        ),
        generation_ids=torch.full_like(episode.evaluation.generation_ids, 999),
        live_binding_counts=torch.full_like(
            episode.evaluation.live_binding_counts, 999
        ),
        heldout_combination_mask=torch.ones_like(
            episode.evaluation.heldout_combination_mask
        ),
    )
    poisoned = replace(episode, evaluation=poisoned_evaluation)
    actual = shared_prototype_coverage_from_episodes(
        task,
        (poisoned,),
        split="validation",
        seed=91,
        document_length=64,
    )
    assert actual == expected

    invalid_mask = episode.inputs.valid_mask.clone()
    invalid_mask[0] = False
    with pytest.raises(ValueError, match="invalid events|deterministic sample"):
        shared_prototype_coverage_from_episodes(
            task,
            (replace(episode, inputs=replace(episode.inputs, valid_mask=invalid_mask)),),
            split="validation",
            seed=91,
            document_length=64,
        )

    event_kinds = episode.inputs.event_kinds.clone()
    event_kinds[0] = int(BindingEventKind.QUERY)
    with pytest.raises(
        ValueError, match="strict algebra|deterministic sample|source_value"
    ):
        shared_prototype_coverage_from_episodes(
            task,
            (replace(episode, inputs=replace(episode.inputs, event_kinds=event_kinds)),),
            split="validation",
            seed=91,
            document_length=64,
        )


def test_public_arguments_and_work_budgets_fail_closed() -> None:
    task = screen_task()
    episode = generate_binding_episode(
        task, length=64, seed=3, split="train", document_index=0
    )

    with pytest.raises(TypeError, match="config must be exact"):
        shared_prototype_cells(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config must be exact"):
        generate_shared_prototype_coverage(  # type: ignore[arg-type]
            BindingAlgebraSpec.from_task(task),
            split="train",
            seed=0,
            document_count=1,
            document_length=64,
        )
    with pytest.raises(TypeError, match="seed must be an exact integer"):
        generate_shared_prototype_coverage(
            task,
            split="train",
            seed=True,  # type: ignore[arg-type]
            document_count=1,
            document_length=64,
        )
    with pytest.raises(ValueError, match="document_count must be at least 1"):
        generate_shared_prototype_coverage(
            task,
            split="train",
            seed=0,
            document_count=0,
            document_length=64,
        )
    with pytest.raises(ValueError, match="document_length must be at least 10"):
        generate_shared_prototype_coverage(
            task,
            split="train",
            seed=0,
            document_count=1,
            document_length=9,
        )
    with pytest.raises(ValueError, match="exceeds config.max_length"):
        generate_shared_prototype_coverage(
            task,
            split="train",
            seed=0,
            document_count=1,
            document_length=513,
        )
    with pytest.raises(AlgebraLearningLimitError, match="max_events"):
        shared_prototype_coverage_from_episodes(
            task,
            (episode,),
            split="train",
            seed=3,
            document_length=64,
            max_events=63,
        )
    with pytest.raises(ValueError, match="nonempty sequence"):
        shared_prototype_coverage_from_episodes(
            task,
            (),
            split="train",
            seed=0,
            document_length=64,
        )
    with pytest.raises(TypeError, match="exact BindingEpisode"):
        shared_prototype_coverage_from_episodes(
            task,
            (object(),),  # type: ignore[arg-type]
            split="train",
            seed=0,
            document_length=64,
        )

    with pytest.raises(ValueError, match="cannot be empty"):
        sweep_shared_prototype_coverage(
            task,
            split="train",
            seeds=(),
            document_count=1,
            document_length=64,
        )
    with pytest.raises(ValueError, match="unique"):
        sweep_shared_prototype_coverage(
            task,
            split="train",
            seeds=(0, 0),
            document_count=1,
            document_length=64,
        )
    with pytest.raises(TypeError, match="exact nonnegative integers"):
        sweep_shared_prototype_coverage(
            task,
            split="train",
            seeds=(True,),  # type: ignore[arg-type]
            document_count=1,
            document_length=64,
        )
    with pytest.raises(AlgebraLearningLimitError, match="max_seeds"):
        sweep_shared_prototype_coverage(
            task,
            split="train",
            seeds=range(5),
            document_count=1,
            document_length=64,
            max_seeds=4,
        )
    with pytest.raises(AlgebraLearningLimitError, match="max_total_events"):
        sweep_shared_prototype_coverage(
            task,
            split="train",
            seeds=range(5),
            document_count=1,
            document_length=64,
            max_total_events=319,
        )


def test_public_certificate_dataclasses_reject_inconsistent_claims(
    report: AlgebraLearnabilityReport,
    full_coverage: SharedPrototypeCoverage,
) -> None:
    with pytest.raises(TypeError, match="parameter_count must be an exact integer"):
        HypothesisLearnability(
            hypothesis=LearningHypothesis.SHARED_CYCLIC,
            parameter_count=True,  # type: ignore[arg-type]
            constraint_rank=1,
            nullity=0,
            heldout_behavior_identified_conditionally=True,
            canonical_state_supervision_assumed=True,
            universal_law_supplied=True,
            sharing_law_selected_from_data=False,
        )
    with pytest.raises(ValueError, match="partition parameters"):
        replace(report.hypotheses[-1], nullity=1)
    with pytest.raises(TypeError, match="exact boolean"):
        replace(
            report.hypotheses[-1],
            sharing_law_selected_from_data=1,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="unknown shared prototype family"):
        PrototypeCell("erase", None, 0)
    with pytest.raises(ValueError, match="require a nonnegative transform"):
        PrototypeCell("update", None, 0)
    with pytest.raises(ValueError, match="only update"):
        PrototypeCell("bind", 0, 0)
    with pytest.raises(TypeError, match="source_value must be an exact integer"):
        PrototypeCell("query", None, True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="design rank and nullity"):
        replace(full_coverage, nullity=1)
    with pytest.raises(ValueError, match="full_rank disagrees"):
        replace(full_coverage, full_rank=False)
    with pytest.raises(ValueError, match="partition the design|observed_cell_count"):
        replace(full_coverage, observed_cell_count=23)
    with pytest.raises(ValueError, match="cannot use held-out labels"):
        replace(full_coverage, used_heldout_labels_for_fit=True)
    with pytest.raises(ValueError, match="positive finite"):
        replace(full_coverage, condition_number=float("inf"))

    sweep = sweep_shared_prototype_coverage(
        screen_task(),
        split="train",
        seeds=(0, 1),
        document_count=1,
        document_length=64,
    )
    with pytest.raises(ValueError, match="one coverage result"):
        replace(sweep, seeds=(0,))
    with pytest.raises(ValueError, match="seed order"):
        replace(sweep, results=tuple(reversed(sweep.results)))
    with pytest.raises(ValueError, match="full-rank count"):
        replace(sweep, full_rank_count=2)
    with pytest.raises(ValueError, match="rank extrema"):
        replace(sweep, minimum_rank=0)

    with pytest.raises(ValueError, match="rank and nullity"):
        replace(report.pseudoheldout_folds[0], supplied_shared_nullity=1)
    with pytest.raises(ValueError, match="does not perform hypothesis selection"):
        replace(
            report.pseudoheldout_folds[0],
            automatic_hypothesis_selection_performed=True,
        )
    with pytest.raises(ValueError, match="passive recovery"):
        replace(report, passive_uniform_recovery_possible=True)
    with pytest.raises(ValueError, match="Phase-I witness"):
        replace(report, zero_support_identical_likelihood_witness_exists=False)
    with pytest.raises(ValueError, match="does not select"):
        replace(report, automatic_hypothesis_selection_performed=True)
    with pytest.raises(ValueError, match="oracle-state analysis"):
        replace(report, sequence_only_representation_learning_performed=True)


def test_coverage_certificate_rejects_hostile_primitives_and_derived_claims(
    full_coverage: SharedPrototypeCoverage,
) -> None:
    with pytest.raises(ValueError, match="task_fingerprint|SHA-256"):
        replace(full_coverage, task_fingerprint="A" * 64)
    with pytest.raises(ValueError, match="sample_sha256|SHA-256"):
        replace(full_coverage, sample_sha256="not-a-sha")
    with pytest.raises(TypeError, match="seed|exact integer"):
        replace(full_coverage, seed=True)
    with pytest.raises(ValueError, match="document_count|nonnegative"):
        replace(full_coverage, document_count=-1)
    with pytest.raises(TypeError, match="event_count|exact integer"):
        replace(full_coverage, event_count=True)
    with pytest.raises(ValueError, match="event_count|document"):
        replace(full_coverage, event_count=full_coverage.event_count + 1)

    counts = list(full_coverage.cell_observation_counts)
    counts[0] = (counts[0][0], True)  # type: ignore[list-item]
    with pytest.raises(TypeError, match="observation count|exact integer"):
        replace(full_coverage, cell_observation_counts=tuple(counts))

    counts = list(full_coverage.cell_observation_counts)
    counts[-1] = (counts[0][0], counts[-1][1])
    with pytest.raises(ValueError, match="unique|labels|inventory"):
        replace(full_coverage, cell_observation_counts=tuple(counts))

    with pytest.raises(ValueError, match="observed|missing"):
        replace(
            full_coverage,
            observed_cell_count=23,
            missing_cells=(full_coverage.cell_observation_counts[0][0],),
        )
    with pytest.raises(ValueError, match="design rank|observed"):
        replace(
            full_coverage,
            design_rank=92,
            nullity=4,
            full_rank=False,
        )
    with pytest.raises(ValueError, match="singular|condition"):
        replace(
            full_coverage,
            condition_number=full_coverage.condition_number + 1.0,
        )
    with pytest.raises(TypeError, match="canonical_state_supervision|exact boolean"):
        replace(full_coverage, canonical_state_supervision_assumed=1)
    with pytest.raises(ValueError, match="canonical-state|supervision"):
        replace(full_coverage, canonical_state_supervision_assumed=False)
    with pytest.raises(TypeError, match="used_heldout_labels_for_fit|exact boolean"):
        replace(full_coverage, used_heldout_labels_for_fit=1)


def test_coverage_sweep_rejects_hostile_primitives_and_nested_mismatch() -> None:
    sweep = sweep_shared_prototype_coverage(
        screen_task(),
        split="train",
        seeds=(0, 1),
        document_count=1,
        document_length=64,
    )

    with pytest.raises(ValueError, match="task_fingerprint|SHA-256"):
        replace(sweep, task_fingerprint="Z" * 64)
    with pytest.raises(TypeError, match="document_count|exact integer"):
        replace(sweep, document_count=True)
    with pytest.raises(TypeError, match="seeds|tuple"):
        replace(sweep, seeds=[0, 1])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="seed|exact integer"):
        replace(sweep, seeds=(True, 1))
    with pytest.raises(ValueError, match="mean rank|exact float"):
        replace(sweep, mean_rank=76)
    with pytest.raises(ValueError, match="mean_rank|finite"):
        replace(sweep, mean_rank=float("nan"))
    with pytest.raises(TypeError, match="results|tuple"):
        replace(sweep, results=list(sweep.results))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SharedPrototypeCoverage|coverage record"):
        replace(sweep, results=(object(), sweep.results[1]))

    wrong_task = replace(sweep.results[0], task_fingerprint="0" * 64)
    with pytest.raises(
        ValueError, match="task fingerprint|task_fingerprint|metadata"
    ):
        replace(sweep, results=(wrong_task, sweep.results[1]))
    wrong_split = replace(sweep.results[0], split="validation")
    with pytest.raises(ValueError, match="split|metadata"):
        replace(sweep, results=(wrong_split, sweep.results[1]))
    wrong_document_count = replace(
        sweep.results[0], document_count=2, event_count=128
    )
    with pytest.raises(ValueError, match="document_count|document count"):
        replace(sweep, results=(wrong_document_count, sweep.results[1]))


def test_pseudoheldout_and_report_certificates_reject_hostile_claims(
    report: AlgebraLearnabilityReport,
) -> None:
    fold = report.pseudoheldout_folds[0]
    with pytest.raises(ValueError, match="real_heldout_pair|exact integers"):
        replace(fold, real_heldout_pair=[0, 0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pseudoheldout_pair|exact integer"):
        replace(fold, pseudoheldout_pair=(True, 1))
    with pytest.raises(ValueError, match="must differ|distinct"):
        replace(fold, pseudoheldout_pair=fold.real_heldout_pair)
    with pytest.raises(TypeError, match="parameter_count|exact integer"):
        replace(fold, supplied_shared_parameter_count=True)
    with pytest.raises(ValueError, match="nonnegative"):
        replace(
            fold,
            supplied_shared_design_rank=97,
            supplied_shared_nullity=-1,
        )
    with pytest.raises(ValueError, match="identif|nullity"):
        replace(fold, supplied_shared_identifies_pseudoheldout=False)
    with pytest.raises(TypeError, match="automatic_hypothesis|exact boolean"):
        replace(fold, automatic_hypothesis_selection_performed=0)

    with pytest.raises(ValueError, match="task_fingerprint|SHA-256"):
        replace(report, task_fingerprint="uppercase" + "A" * 55)
    with pytest.raises(ValueError, match="heldout_pair|exact integers"):
        replace(report, heldout_pair=[0, 0])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="full_state_count|exact integer"):
        replace(report, full_state_count=True)
    with pytest.raises(ValueError, match="nonnegative"):
        replace(report, passive_unrestricted_nullity=-1)
    with pytest.raises(
        ValueError, match="train_state_count|full_state_count|state count"
    ):
        replace(report, train_state_count=report.full_state_count + 1)
    with pytest.raises(
        ValueError, match="train_feature_rank|full_feature_rank|feature rank"
    ):
        replace(report, train_feature_rank=report.full_feature_rank + 1)
    with pytest.raises(TypeError, match="one_separating_membership_probe|tuple"):
        replace(report, one_separating_membership_probe=["BIND", "QUERY"])
    with pytest.raises(TypeError, match="hypotheses|tuple"):
        replace(report, hypotheses=list(report.hypotheses))
    with pytest.raises(
        ValueError, match="passive|nullity|hypothesis|report_sha256"
    ):
        replace(report, passive_unrestricted_nullity=4_911)

    wrong_real = replace(report.pseudoheldout_folds[0], real_heldout_pair=(1, 1))
    with pytest.raises(ValueError, match="heldout|fold"):
        replace(
            report,
            pseudoheldout_folds=(wrong_real, *report.pseudoheldout_folds[1:]),
        )
    with pytest.raises(TypeError, match="passive_uniform|exact boolean"):
        replace(report, passive_uniform_recovery_possible=0)
    with pytest.raises(ValueError, match="report_sha256|digest|SHA-256"):
        replace(report, report_sha256="0" * 64)
    with pytest.raises(ValueError, match="report_sha256|SHA-256"):
        replace(report, report_sha256="invalid")


def test_fit_evaluation_certificate_rejects_hostile_counts_and_booleans(
    full_fit: LearnedSharedPrototypeTransducer,
) -> None:
    evaluation = evaluate_shared_prototype_transducer(screen_task(), full_fit)

    with pytest.raises(TypeError, match="evaluated_cell_count|exact integer"):
        replace(evaluation, evaluated_cell_count=True)
    with pytest.raises(ValueError, match="nonnegative"):
        replace(
            evaluation,
            exact_cell_count=-1,
            all_value_tables_exact=False,
            all_valid_programs_exact_given_supplied_state_contract=False,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        replace(evaluation, missing_cell_count=-1)
    with pytest.raises(ValueError, match="counts are inconsistent"):
        replace(evaluation, missing_cell_count=1)
    with pytest.raises(TypeError, match="all_value_tables_exact|exact boolean"):
        replace(evaluation, all_value_tables_exact=1)
    with pytest.raises(TypeError, match="programs_exact|exact boolean"):
        replace(
            evaluation,
            all_valid_programs_exact_given_supplied_state_contract=1,
        )
    with pytest.raises(
        TypeError, match="heldout_semantics_used_only_for_postfit|exact boolean"
    ):
        replace(evaluation, heldout_semantics_used_only_for_postfit_evaluation=1)
