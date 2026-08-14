from __future__ import annotations

import ast
from dataclasses import fields, replace
import inspect
import os
from pathlib import Path

import pytest
import torch

import tnlm_v3.algebra_discovery as discovery
from tnlm_v3.algebra_discovery import (
    LearnedSequenceAlgebra,
    SequenceAlgebraSelectionResult,
    SequenceCorpus,
    SequenceDiscoveryLimitError,
    SequenceSelectionMode,
    TraceSupervisedCorpus,
    VisibleEvent,
    VisibleSequence,
    fit_sequence_algebra,
    make_sequence_corpus,
    make_trace_supervised_corpus,
    make_trace_supervised_sequence,
    run_outer_rotation,
    select_sequence_algebra,
    sequence_corpus_from_episodes,
    visible_sequence_from_episode,
)
from tnlm_v3.algebra_discovery_probes import (
    VisibleProbeEvent,
    VisibleProbeProgram,
)
from tnlm_v3.data import (
    BindingEpisode,
    BindingEvaluation,
    BindingEventKind,
    BindingTaskConfig,
    generate_binding_episode,
    generate_binding_episodes,
)


def _direct_sequence(
    key: int,
    target: int,
    *,
    argument: int | None = None,
) -> VisibleSequence:
    argument = target if argument is None else argument
    return VisibleSequence(
        events=(
            VisibleEvent(
                BindingEventKind.BIND,
                primary_key=key,
                argument=argument,
            ),
            VisibleEvent(BindingEventKind.QUERY, primary_key=key),
        ),
        query_targets=(None, target),
    )


def _direct_trace(
    keys: int,
    values: int,
    key: int,
    target: int,
    split: str,
    *,
    argument: int | None = None,
):
    sequence = _direct_sequence(key, target, argument=argument)
    state = ((key, target),)
    return make_trace_supervised_sequence(
        sequence,
        split=split,
        pre_event_cells=((), state),
        post_event_cells=(state, state),
        query_dependency_cells=((), state),
        num_surface_keys=keys,
        value_cardinality=values,
    )


def _direct_trace_corpus(
    keys: int,
    values: int,
    omitted: tuple[int, int],
) -> TraceSupervisedCorpus:
    cells = tuple(
        (key, value)
        for key in range(keys)
        for value in range(values)
        if (key, value) != omitted
    )
    traces = tuple(
        _direct_trace(keys, values, key, value, split)
        for split in ("train", "validation")
        for key, value in cells
    )
    return make_trace_supervised_corpus(keys, values, traces)


def _attest_episode(
    episode: BindingEpisode,
    *,
    keys: int,
    values: int,
):
    """External semantic audit used only to build trusted test-controller data."""

    sequence = visible_sequence_from_episode(episode)
    state: dict[int, int] = {}
    lineage: dict[int, set[tuple[int, int]]] = {}
    pre_rows: list[tuple[tuple[int, int], ...]] = []
    post_rows: list[tuple[tuple[int, int], ...]] = []
    dependency_rows: list[tuple[tuple[int, int], ...]] = []
    for event, target in zip(
        sequence.events, sequence.query_targets, strict=True
    ):
        pre_rows.append(tuple(sorted(state.items())))
        dependencies: tuple[tuple[int, int], ...] = ()
        key = event.primary_key
        if event.kind is BindingEventKind.BIND:
            state[key] = event.argument
            lineage[key] = {(key, event.argument)}
        elif event.kind is BindingEventKind.UPDATE:
            new_value = (state[key] + event.argument + 1) % values
            state[key] = new_value
            lineage[key] = set(lineage[key]) | {(key, new_value)}
        elif event.kind is BindingEventKind.COPY:
            new_value = state[event.secondary_key]
            state[key] = new_value
            lineage[key] = set(lineage[event.secondary_key]) | {(key, new_value)}
        elif event.kind is BindingEventKind.INVALIDATE:
            del state[key]
            del lineage[key]
        elif event.kind is BindingEventKind.QUERY:
            if target != state[key]:
                raise AssertionError("external replay disagrees with query label")
            dependencies = tuple(sorted(lineage[key] | {(key, state[key])}))
        post_rows.append(tuple(sorted(state.items())))
        dependency_rows.append(dependencies)
    return make_trace_supervised_sequence(
        sequence,
        split=episode.split,
        pre_event_cells=tuple(pre_rows),
        post_event_cells=tuple(post_rows),
        query_dependency_cells=tuple(dependency_rows),
        num_surface_keys=keys,
        value_cardinality=values,
    )


@pytest.fixture(scope="module")
def direct_fit() -> LearnedSequenceAlgebra:
    sequences = tuple(
        _direct_sequence(key, value)
        for key in range(2)
        for value in range(2)
    )
    corpus = make_sequence_corpus(2, 2, split="train", sequences=sequences)
    return fit_sequence_algebra(
        corpus,
        residual_penalty=1,
        restart_count=1,
        max_sweeps=2,
    )


@pytest.fixture(scope="module")
def full_nineteen_result() -> SequenceAlgebraSelectionResult:
    corpus = _direct_trace_corpus(5, 4, (0, 0))
    return select_sequence_algebra(
        corpus,
        residual_penalties=(0, 1),
        seed=0,
        restart_count=1,
        max_sweeps=1,
    )


def test_pure_estimator_has_no_forbidden_import_or_input_channel():
    source_path = Path(discovery.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(name.endswith("exact_algebra") for name in imported_modules)

    forbidden = {
        "heldout_pair",
        "heldout_pairs",
        "heldout_mask",
        "oracle_routes",
        "dependency_parents",
        "generation_ids",
        "generation_seed",
        "task_fingerprint",
        "config_fingerprint",
        "document_id",
        "pre_state",
        "post_state",
    }
    assert forbidden.isdisjoint(inspect.signature(fit_sequence_algebra).parameters)
    assert forbidden.isdisjoint(field.name for field in fields(SequenceCorpus))
    estimator_source = inspect.getsource(fit_sequence_algebra)
    assert "exact_executor" not in estimator_source
    assert "ExternalTraceAttestation" not in estimator_source
    initializer_signature = inspect.signature(
        discovery._trace_supervised_initial_outputs
    )
    assert tuple(initializer_signature.parameters) == ("corpus", "inventory", "seed")
    initializer_source = inspect.getsource(
        discovery._trace_supervised_initial_outputs
    )
    assert "ExternalTraceAttestation" not in initializer_source
    assert "exact_algebra" not in initializer_source


def test_trace_only_initializer_is_disclosed_and_uses_observed_answers():
    corpus = make_sequence_corpus(
        2,
        2,
        split="train",
        sequences=(
            _direct_sequence(0, 1, argument=0),
            _direct_sequence(1, 0, argument=1),
        ),
    )
    model = fit_sequence_algebra(
        corpus,
        residual_penalty=16,
        seed=9,
        restart_count=1,
        max_sweeps=1,
    )
    assert dict(model.shared_outputs)[discovery.PrototypeAddress("bind", None, 0)] == 1
    assert dict(model.shared_outputs)[discovery.PrototypeAddress("bind", None, 1)] == 0
    certificate = model.fit
    assert certificate.schema == "tnlm-v3-sequence-algebra-fit-v2"
    assert certificate.trace_supervised_initializer_used
    assert certificate.trace_supervised_initializer_vote_count == 2
    assert certificate.trace_supervised_initializer_covered_address_count == 2
    assert certificate.trace_supervised_initializer_conflicting_address_count == 0
    assert certificate.trace_supervised_initializer_round_count == 2
    assert certificate.random_restart_count == 0
    assert not certificate.initializer_received_canonical_state
    assert not certificate.initializer_received_exact_executor


def test_pairwise_trace_search_resolves_a_real_censored_fold_local_minimum():
    task = BindingTaskConfig(
        num_surface_keys=5,
        value_cardinality=4,
        branches=3,
        max_live_bindings=3,
        min_length=10,
        max_length=2048,
        heldout_key_value_pairs=((0, 0),),
    )
    traces = tuple(
        _attest_episode(episode, keys=5, values=4)
        for episode in generate_binding_episodes(
            task,
            count=16,
            seed=18,
            split="train",
            lengths=(64,) * 16,
        )
    )
    pseudo_cell = (4, 0)
    retained = tuple(
        trace
        for trace in traces
        if pseudo_cell
        not in (
            discovery._trace_states(trace)
            | discovery._trace_dependencies(trace)
        )
    )
    assert len(retained) == 4
    corpus = make_sequence_corpus(
        5,
        4,
        split="train",
        sequences=tuple(trace.sequence for trace in retained),
    )
    coordinate_only = fit_sequence_algebra(
        corpus,
        residual_penalty=16,
        seed=15,
        restart_count=1,
        max_sweeps=4,
        max_pairwise_rounds=0,
    )
    resolved = fit_sequence_algebra(
        corpus,
        residual_penalty=16,
        seed=15,
        restart_count=1,
        max_sweeps=4,
        max_pairwise_rounds=2,
    )
    assert coordinate_only.fit.training_mistakes > 0
    assert coordinate_only.fit.residual_override_count == 0
    assert resolved.fit.training_mistakes == 0
    assert resolved.fit.residual_override_count == 0
    assert resolved.fit.pairwise_improvement_count == 1
    assert resolved.fit.pairwise_uncertain_address_count == 2
    assert resolved.fit.pairwise_max_search_address_count == 3
    assert resolved.fit.pairwise_objective_evaluations == 48


def test_corpus_hash_exact_types_and_fit_work_budgets():
    sequence = _direct_sequence(0, 1)
    corpus = make_sequence_corpus(2, 2, split="train", sequences=(sequence,))
    with pytest.raises(ValueError, match="does not bind"):
        replace(corpus, sample_sha256="0" * 64)
    with pytest.raises(TypeError, match="exact integer"):
        make_sequence_corpus(True, 2, split="train", sequences=(sequence,))
    with pytest.raises(TypeError, match="exact BindingEventKind"):
        VisibleEvent(1, primary_key=0, argument=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="TRAIN data only"):
        fit_sequence_algebra(
            make_sequence_corpus(
                2, 2, split="validation", sequences=(sequence,)
            ),
            residual_penalty=0,
        )
    with pytest.raises(SequenceDiscoveryLimitError, match="max_events"):
        fit_sequence_algebra(
            corpus,
            residual_penalty=0,
            max_events=1,
        )
    with pytest.raises(SequenceDiscoveryLimitError, match="scored_event_work"):
        fit_sequence_algebra(
            corpus,
            residual_penalty=0,
            restart_count=2,
            max_sweeps=4,
            max_scored_event_work=1,
        )
    with pytest.raises(SequenceDiscoveryLimitError, match="objective_evaluations"):
        fit_sequence_algebra(
            corpus,
            residual_penalty=0,
            restart_count=2,
            max_sweeps=4,
            max_objective_evaluations=1,
            max_scored_event_work=10_000,
        )


def test_selector_aggregate_work_budget_fails_before_first_fit(monkeypatch):
    corpus = _direct_trace_corpus(2, 2, (0, 0))
    called = False

    def forbidden_fit(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("fit must not run before aggregate budget rejection")

    monkeypatch.setattr(discovery, "fit_sequence_algebra", forbidden_fit)
    with pytest.raises(
        SequenceDiscoveryLimitError, match="aggregate_scored_event_work"
    ):
        select_sequence_algebra(
            corpus,
            residual_penalties=(0, 1),
            restart_count=1,
            max_sweeps=1,
            max_aggregate_scored_event_work=1,
        )
    assert not called


def test_fit_and_selector_objective_budgets_fail_before_scoring(monkeypatch):
    sequence = _direct_sequence(0, 1)
    fit_corpus = make_sequence_corpus(
        2, 2, split="train", sequences=(sequence,)
    )
    scored = False

    def forbidden_score(*args, **kwargs):
        nonlocal scored
        scored = True
        raise AssertionError("objective score must not run after failed preflight")

    monkeypatch.setattr(discovery, "_corpus_mistakes", forbidden_score)
    with pytest.raises(
        SequenceDiscoveryLimitError, match="before work"
    ):
        fit_sequence_algebra(
            fit_corpus,
            residual_penalty=0,
            restart_count=2,
            max_sweeps=4,
            max_objective_evaluations=1,
            max_scored_event_work=10_000,
        )
    assert not scored

    called_fit = False

    def forbidden_fit(*args, **kwargs):
        nonlocal called_fit
        called_fit = True
        raise AssertionError("selector fit must not run after failed preflight")

    monkeypatch.setattr(discovery, "fit_sequence_algebra", forbidden_fit)
    with pytest.raises(
        SequenceDiscoveryLimitError,
        match="max_objective_evaluations_per_fit before fitting",
    ):
        select_sequence_algebra(
            _direct_trace_corpus(2, 2, (0, 0)),
            residual_penalties=(0, 1),
            restart_count=2,
            max_sweeps=4,
            max_objective_evaluations_per_fit=1,
        )
    assert not called_fit


def test_forbidden_episode_metadata_poisoning_is_erased_and_fit_invariant():
    task = BindingTaskConfig(
        num_surface_keys=5,
        value_cardinality=4,
        branches=3,
        max_live_bindings=3,
        min_length=10,
        max_length=64,
        heldout_key_value_pairs=((0, 0),),
    )
    episode = generate_binding_episode(
        task, length=32, seed=7, split="train", document_index=0
    )
    evaluation = episode.evaluation
    poisoned_evaluation = BindingEvaluation(
        oracle_routes=evaluation.oracle_routes + 97,
        targets=evaluation.targets.clone(),
        dependency_parents=torch.full_like(evaluation.dependency_parents, 777),
        generation_ids=torch.full_like(evaluation.generation_ids, 888),
        live_binding_counts=torch.full_like(evaluation.live_binding_counts, 999),
        heldout_combination_mask=~evaluation.heldout_combination_mask,
    )
    poisoned = replace(
        episode,
        evaluation=poisoned_evaluation,
        document_id="poisoned-evaluation-only-id",
        generation_seed=2**62,
        config_fingerprint="f" * 64,
    )
    assert visible_sequence_from_episode(poisoned) == visible_sequence_from_episode(
        episode
    )
    clean_corpus = sequence_corpus_from_episodes(
        5, 4, (episode,), split="train"
    )
    poisoned_corpus = sequence_corpus_from_episodes(
        5, 4, (poisoned,), split="train"
    )
    assert poisoned_corpus == clean_corpus
    fit_kwargs = dict(
        residual_penalty=4,
        seed=3,
        restart_count=1,
        max_sweeps=1,
    )
    assert fit_sequence_algebra(
        clean_corpus, **fit_kwargs
    ) == fit_sequence_algebra(poisoned_corpus, **fit_kwargs)


def test_trace_firewall_rejects_ghost_and_future_dependency_attacks():
    direct = _direct_sequence(0, 1)
    state = ((0, 1),)
    ghost = ((1, 0),)
    with pytest.raises(ValueError, match="begin with an empty state"):
        make_trace_supervised_sequence(
            direct,
            split="train",
            pre_event_cells=(ghost, state),
            post_event_cells=(tuple(sorted((*ghost, *state))), state),
            query_dependency_cells=((), state),
            num_surface_keys=2,
            value_cardinality=2,
        )

    sequence = VisibleSequence(
        events=(
            VisibleEvent(BindingEventKind.BIND, primary_key=0, argument=1),
            VisibleEvent(BindingEventKind.QUERY, primary_key=0),
            VisibleEvent(BindingEventKind.INVALIDATE, primary_key=0),
            VisibleEvent(BindingEventKind.BIND, primary_key=1, argument=1),
            VisibleEvent(BindingEventKind.QUERY, primary_key=1),
        ),
        query_targets=(None, 1, None, None, 1),
    )
    state0 = ((0, 1),)
    state1 = ((1, 1),)
    with pytest.raises(ValueError, match="first occupied later"):
        make_trace_supervised_sequence(
            sequence,
            split="train",
            pre_event_cells=((), state0, state0, (), state1),
            post_event_cells=(state0, state0, (), state1, state1),
            query_dependency_cells=(
                (),
                ((0, 1), (1, 1)),
                (),
                (),
                state1,
            ),
            num_surface_keys=2,
            value_cardinality=2,
        )


def test_trace_digest_bypass_and_outer_contamination_are_rejected():
    clean = _direct_trace_corpus(2, 2, (0, 0))
    first = clean.traces[0]
    forged = replace(first.attestation, attestation_sha256="0" * 64)
    forged_trace = replace(first, attestation=forged)
    with pytest.raises(ValueError, match="attestation"):
        make_trace_supervised_corpus(
            2, 2, (forged_trace, *clean.traces[1:])
        )

    contaminated = make_trace_supervised_corpus(
        2,
        2,
        tuple(
            _direct_trace(2, 2, key, value, split)
            for split in ("train", "validation")
            for key in range(2)
            for value in range(2)
        ),
    )
    with pytest.raises(ValueError, match="outer omission count"):
        select_sequence_algebra(
            contaminated,
            residual_penalties=(0, 1),
            restart_count=1,
            max_sweeps=1,
        )


def test_full_nineteen_fold_rotation_discloses_exact_tie_and_mdl_choice(
    full_nineteen_result: SequenceAlgebraSelectionResult,
):
    result = full_nineteen_result
    assert len(result.folds) == 19
    assert tuple(fold.pseudoheldout_cell for fold in result.folds) == tuple(
        (key, value)
        for key in range(5)
        for value in range(4)
        if (key, value) != (0, 0)
    )
    assert all(fold.whole_sequence_pre_post_censoring for fold in result.folds)
    assert all(fold.descendant_query_scoring for fold in result.folds)
    assert all(not fold.estimator_received_oracle_metadata for fold in result.folds)
    assert [row.pseudo_query_mistakes for row in result.aggregate_scores] == [0, 0]
    assert result.primary_score_best_penalties == (0, 1)
    assert result.primary_score_tied
    assert result.selected_residual_penalty == 1
    assert "description length" in result.selection_basis
    assert result.final_model.fit.training_mistakes == 0
    assert not result.final_model.local_overrides


def test_zero_penalty_is_explicitly_not_unregularized(
    full_nineteen_result: SequenceAlgebraSelectionResult,
):
    zero_rows = tuple(
        candidate
        for fold in full_nineteen_result.folds
        for candidate in fold.candidates
        if candidate.residual_penalty == 0
    )
    assert zero_rows
    # Fit one zero-explicit-penalty model so the optimizer's implicit MDL
    # tie-break is inspected directly rather than inferred from selection.
    corpus = make_sequence_corpus(
        2,
        2,
        split="train",
        sequences=tuple(
            _direct_sequence(key, value)
            for key in range(2)
            for value in range(2)
        ),
    )
    model = fit_sequence_algebra(
        corpus,
        residual_penalty=0,
        restart_count=1,
        max_sweeps=1,
    )
    assert model.fit.residual_penalty == 0
    assert model.fit.override_count_lexicographic_tiebreak
    assert model.fit.deterministic_table_lexicographic_tiebreak
    assert model.fit.zero_explicit_penalty_still_uses_minimum_override_tiebreak
    assert "override_count" in model.fit.optimization_tiebreak


def test_fit_provenance_and_claim_scope_are_caller_attested_only(
    direct_fit: LearnedSequenceAlgebra,
):
    certificate = direct_fit.fit
    assert certificate.direct_split_caller_attested
    assert not certificate.heldout_label_absence_independently_certified
    assert not certificate.heldout_identifier_received_by_estimator
    assert not certificate.evaluation_metadata_received_by_estimator
    assert not certificate.canonical_state_supervision_used_by_estimator
    assert not certificate.exact_executor_used_for_fitting
    assert certificate.trace_supervised_initializer_used
    assert certificate.trace_supervised_initializer_vote_count == 4
    assert certificate.trace_supervised_initializer_covered_address_count == 2
    assert certificate.trace_supervised_initializer_conflicting_address_count == 0
    assert certificate.trace_supervised_initializer_round_count == 2
    assert not certificate.initializer_received_canonical_state
    assert not certificate.initializer_received_exact_executor
    assert certificate.supplied_addressable_register_interface
    assert certificate.supplied_identity_query_gauge
    assert not certificate.representation_discovery_performed


def test_direct_fit_and_prediction_protocol_expose_no_outer_metadata(
    full_nineteen_result: SequenceAlgebraSelectionResult,
):
    model = full_nineteen_result.final_model
    program = VisibleProbeProgram(
        num_surface_keys=5,
        value_cardinality=4,
        events=(
            VisibleProbeEvent.bind(0, 0),
            VisibleProbeEvent.query(0),
        ),
    )
    assert model.predict_queries(program) == (0,)
    assert model.predict(program) == (None, 0)
    assert not any(
        "heldout" in field.name or "outer" in field.name
        for field in fields(SequenceCorpus)
    )
    assert not any(
        "heldout" in name or "outer" in name
        for name in inspect.signature(model.predict_queries).parameters
    )
    assert full_nineteen_result.outer_unobserved_identifier_inferred_and_used_by_trusted_firewall
    assert not full_nineteen_result.outer_unobserved_identifier_received_by_estimator
    assert not full_nineteen_result.outer_unobserved_labels_used_for_fit_or_selection
    assert not full_nineteen_result.outer_unobserved_identifier_used_in_candidate_ordering


def test_outer_rotation_binds_unique_omissions_and_complete_coverage():
    corpora = tuple(
        _direct_trace_corpus(2, 2, omitted)
        for omitted in ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    rotation = run_outer_rotation(
        corpora,
        require_complete_single_cell_rotation=True,
        residual_penalties=(0, 1),
        restart_count=1,
        max_sweeps=1,
    )
    assert rotation.environment_count == 4
    assert rotation.omitted_cell_sets == (
        ((0, 0),),
        ((0, 1),),
        ((1, 0),),
        ((1, 1),),
    )
    assert rotation.unique_omission_sets
    assert rotation.complete_single_cell_rotation
    assert not rotation.confirmatory_claim_permitted

    same_omission_different_digest = make_trace_supervised_corpus(
        2, 2, tuple(reversed(corpora[0].traces))
    )
    assert same_omission_different_digest.corpus_sha256 != corpora[0].corpus_sha256
    with pytest.raises(ValueError, match="repeats an omitted environment"):
        run_outer_rotation(
            (corpora[0], same_omission_different_digest),
            residual_penalties=(0, 1),
            restart_count=1,
            max_sweeps=1,
        )

    with pytest.raises(ValueError, match="committed to its declared omission"):
        replace(
            rotation,
            omitted_cell_sets=tuple(reversed(rotation.omitted_cell_sets)),
        )
    with pytest.raises(ValueError, match="distinct selection results"):
        replace(
            rotation,
            results=(rotation.results[0],) * rotation.environment_count,
        )


def test_pseudo_fold_optimizer_seed_is_absolute_and_outer_independent():
    base_seed = 7
    results = tuple(
        select_sequence_algebra(
            _direct_trace_corpus(2, 2, outer),
            residual_penalties=(0, 1),
            seed=base_seed,
            restart_count=1,
            max_sweeps=1,
            max_pairwise_rounds=0,
        )
        for outer in ((0, 0), (0, 1))
    )
    shared_pseudo_cell = (1, 1)
    folds = tuple(
        next(
            fold
            for fold in result.folds
            if fold.pseudoheldout_cell == shared_pseudo_cell
        )
        for result in results
    )
    assert tuple(fold.optimizer_seed for fold in folds) == (10, 10)
    assert all(
        not result.outer_unobserved_identifier_used_in_candidate_ordering
        for result in results
    )


def test_outer_rotation_aggregate_budget_fails_before_first_environment_fit(
    monkeypatch,
):
    corpora = (
        _direct_trace_corpus(2, 2, (0, 0)),
        _direct_trace_corpus(2, 2, (0, 1)),
    )
    called = False

    def forbidden_selector(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("no environment may fit after outer budget failure")

    monkeypatch.setattr(discovery, "select_sequence_algebra", forbidden_selector)
    with pytest.raises(
        SequenceDiscoveryLimitError, match="aggregate scored-event work"
    ):
        run_outer_rotation(
            corpora,
            max_outer_aggregate_scored_event_work=1,
            residual_penalties=(0, 1),
            restart_count=1,
            max_sweeps=1,
        )
    assert not called


def test_model_fit_and_selection_certificate_forgery_is_rejected(
    direct_fit: LearnedSequenceAlgebra,
    full_nineteen_result: SequenceAlgebraSelectionResult,
):
    with pytest.raises(ValueError):
        replace(
            direct_fit.fit,
            training_mistakes=direct_fit.fit.training_mistakes + 1,
        )
    with pytest.raises(ValueError, match="model_fingerprint"):
        replace(direct_fit, model_fingerprint="0" * 64)
    with pytest.raises(ValueError):
        replace(
            full_nineteen_result,
            selected_residual_penalty=0,
        )

    original = full_nineteen_result
    first = original.aggregate_scores[0]
    forged_first = replace(
        first,
        pseudo_query_mistakes=first.pseudo_query_mistakes + 1,
    )
    forged_aggregates = (forged_first, *original.aggregate_scores[1:])
    forged_fields = {
        name: getattr(original, name)
        for name in original.__dataclass_fields__
        if name != "result_sha256"
    }
    forged_fields["aggregate_scores"] = forged_aggregates
    forged_digest = discovery._sha256(
        discovery._selection_payload_fields(forged_fields)
    )
    with pytest.raises(ValueError, match="do not reproduce"):
        SequenceAlgebraSelectionResult(
            **forged_fields,
            result_sha256=forged_digest,
        )


def _observed_exception_power_corpus() -> TraceSupervisedCorpus:
    # Semantic cells A=(0,1), B=(1,0), C=(1,1); (0,0) stays outer.
    mapping = {
        (0, 1): 1,
        (1, 0): 1,  # BIND argument 1 -> local output 0 on key 1.
        (1, 1): 0,
    }
    train = tuple(
        _direct_trace(2, 2, key, target, "train", argument=argument)
        for (key, target), argument in mapping.items()
    )
    validation = []
    exception = (1, 0)
    for (key, target), argument in mapping.items():
        first = _direct_sequence(key, target, argument=argument)
        events = (
            *first.events,
            VisibleEvent(BindingEventKind.INVALIDATE, primary_key=key),
            VisibleEvent(BindingEventKind.BIND, primary_key=1, argument=1),
            VisibleEvent(BindingEventKind.QUERY, primary_key=1),
            VisibleEvent(BindingEventKind.QUERY, primary_key=1),
            VisibleEvent(BindingEventKind.QUERY, primary_key=1),
            VisibleEvent(BindingEventKind.INVALIDATE, primary_key=1),
            VisibleEvent(BindingEventKind.BIND, primary_key=0, argument=1),
            VisibleEvent(BindingEventKind.QUERY, primary_key=0),
        )
        targets = (*first.query_targets, None, None, 0, 0, 0, None, None, 1)
        sequence = VisibleSequence(events=events, query_targets=targets)
        first_state = ((key, target),)
        second_state = (exception,)
        normal_state = ((0, 1),)
        validation.append(
            make_trace_supervised_sequence(
                sequence,
                split="validation",
                pre_event_cells=(
                    (),
                    first_state,
                    first_state,
                    (),
                    second_state,
                    second_state,
                    second_state,
                    second_state,
                    (),
                    normal_state,
                ),
                post_event_cells=(
                    first_state,
                    first_state,
                    (),
                    second_state,
                    second_state,
                    second_state,
                    second_state,
                    (),
                    normal_state,
                    normal_state,
                ),
                query_dependency_cells=(
                    (),
                    first_state,
                    (),
                    (),
                    second_state,
                    second_state,
                    second_state,
                    (),
                    (),
                    normal_state,
                ),
                num_surface_keys=2,
                value_cardinality=2,
            )
        )
    return make_trace_supervised_corpus(2, 2, (*train, *validation))


def test_observed_exception_power_control_selects_nonzero_residual_capacity():
    result = select_sequence_algebra(
        _observed_exception_power_corpus(),
        residual_penalties=(0, 16),
        mode=SequenceSelectionMode.OBSERVED_EXCEPTION_POWER_CONTROL,
        seed=0,
        restart_count=4,
        max_sweeps=6,
    )
    scores = {row.residual_penalty: row for row in result.aggregate_scores}
    assert scores[0].all_validation_query_mistakes < (
        scores[16].all_validation_query_mistakes
    )
    assert scores[0].total_residual_override_count > 0
    assert result.selected_residual_penalty == 0
    assert result.observed_exception_power_control_only
    assert not result.confirmatory_claim_permitted


@pytest.mark.skipif(
    os.environ.get("TNLM_RUN_SLOW_DISCOVERY") != "1",
    reason="opt-in nominal passive exact-tie selector regression",
)
def test_nominal_passive_exact_tie_selector_regression():
    task = BindingTaskConfig(
        num_surface_keys=5,
        value_cardinality=4,
        branches=3,
        max_live_bindings=3,
        min_length=10,
        max_length=2048,
        heldout_key_value_pairs=((0, 0),),
        global_distractor_probability=0.5,
    )
    train = generate_binding_episodes(
        task,
        count=32,
        seed=17,
        split="train",
        lengths=(64,) * 32,
    )
    validation = generate_binding_episodes(
        task,
        count=32,
        seed=23,
        split="validation",
        lengths=(64,) * 32,
    )
    traces = tuple(
        _attest_episode(episode, keys=5, values=4)
        for episode in (*train, *validation)
    )
    result = select_sequence_algebra(
        make_trace_supervised_corpus(5, 4, traces),
        residual_penalties=(4, 16),
        seed=0,
        restart_count=2,
        max_sweeps=4,
        max_aggregate_scored_event_work=2_000_000_000,
    )
    scores = {row.residual_penalty: row for row in result.aggregate_scores}
    assert (
        scores[4].pseudo_query_mistakes,
        scores[4].all_validation_query_mistakes,
        scores[4].total_residual_override_count,
    ) == (0, 0, 0)
    assert (
        scores[16].pseudo_query_mistakes,
        scores[16].all_validation_query_mistakes,
        scores[16].total_residual_override_count,
    ) == (0, 0, 0)
    assert result.selected_residual_penalty == 16
    assert result.primary_score_tied
    assert result.final_model.fit.training_mistakes == 0
    assert not result.final_model.local_overrides
