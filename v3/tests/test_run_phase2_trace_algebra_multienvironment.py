from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tnlm_v3.algebra_discovery import SequenceDiscoveryLimitError


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_phase2_trace_algebra_multienvironment.py"
    )
    spec = importlib.util.spec_from_file_location("phase2_trace_multi", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _synthetic_preopen(module, protocol, index: int, *, passed: bool = True):
    body = {
        "schema": module.TEST_FIXTURE_ENVIRONMENT_SCHEMA,
        "environment": module._environment_spec_payload(protocol.schedule[index]),
        "selected_residual_penalty": protocol.residual_penalties[index % 2],
        "candidate_count": 38,
        "admissible_candidate_count": 38 - int(not passed),
        "final_training_mistakes": 0,
        "final_local_override_count": 0,
        "environment_preopen_gate_passed": passed,
        "trace_corpus_sha256": _digest(f"corpus-{index}"),
    }
    return module._seal_body(body)


@dataclass(frozen=True)
class _Candidate:
    residual_penalty: int
    pseudo_query_count: int
    training_mistakes: int
    residual_override_count: int
    model_fingerprint: str


@dataclass(frozen=True)
class _Fold:
    pseudoheldout_cell: tuple[int, int]
    optimizer_seed: int
    candidates: tuple[_Candidate, ...]


@dataclass(frozen=True)
class _Fit:
    seed: int
    restart_count: int
    max_sweeps: int
    max_pairwise_rounds: int
    residual_penalty: int
    training_query_count: int
    training_mistakes: int
    residual_override_count: int


@dataclass(frozen=True)
class _Model:
    fit: _Fit
    model_fingerprint: str


@dataclass(frozen=True)
class _Selection:
    source_corpus_sha256: str
    result_sha256: str
    selected_residual_penalty: int
    folds: tuple[_Fold, ...]
    final_model: _Model


@dataclass(frozen=True)
class _Rotation:
    omitted_cell_sets: tuple[tuple[tuple[int, int], ...], ...]
    results: tuple[_Selection, ...]
    result_sha256: str


def _fake_selection(module, protocol, index: int, *, unclean: bool = False):
    spec = protocol.schedule[index]
    universe = {
        (key, value)
        for key in range(protocol.num_surface_keys)
        for value in range(protocol.value_cardinality)
    }
    folds = []
    for fold_index, cell in enumerate(sorted(universe - {spec.outer_cell})):
        absolute = cell[0] * protocol.value_cardinality + cell[1]
        folds.append(
            _Fold(
                pseudoheldout_cell=cell,
                optimizer_seed=spec.optimizer_seed + absolute,
                candidates=tuple(
                    _Candidate(
                        residual_penalty=penalty,
                        pseudo_query_count=16,
                        training_mistakes=0,
                        residual_override_count=int(
                            unclean and fold_index == 0 and penalty == 4
                        ),
                        model_fingerprint=_digest(f"candidate-{index}-{cell}-{penalty}"),
                    )
                    for penalty in protocol.residual_penalties
                ),
            )
        )
    fit = _Fit(
        seed=spec.optimizer_seed,
        restart_count=protocol.restart_count,
        max_sweeps=protocol.max_sweeps,
        max_pairwise_rounds=protocol.max_pairwise_rounds,
        residual_penalty=16,
        training_query_count=100,
        training_mistakes=0,
        residual_override_count=0,
    )
    return _Selection(
        source_corpus_sha256=_digest(f"fake-corpus-{index}"),
        result_sha256=_digest(f"fake-selection-{index}"),
        selected_residual_penalty=16,
        folds=tuple(folds),
        final_model=_Model(fit=fit, model_fingerprint=_digest(f"model-{index}")),
    )


def _fake_prerequisites(module, protocol):
    manifest = module.ImplementationManifest(
        protocol.implementation_manifest_relative_path,
        _digest("manifest"),
        tuple((path, _digest(path)) for path in protocol.implementation_required_paths),
    )
    runtime = module.validate_runtime(protocol)
    power = module.PowerControlCommitment(
        protocol.power_control_relative_path,
        _digest("power-file"),
        _digest("power-record"),
    )
    return manifest, runtime, power


def test_v3_protocol_expands_exact_crossed_schedule_source_closure_and_runtime() -> None:
    module = _load_script()
    protocol = module.load_frozen_protocol()
    assert protocol.protocol_sha256 == module.PROTOCOL_SHA256
    assert len(protocol.schedule) == 40
    assert protocol.documents_per_split == 24
    assert protocol.document_length == 64
    assert protocol.residual_penalties == (4, 16)
    assert (protocol.restart_count, protocol.max_sweeps, protocol.max_pairwise_rounds) == (
        1,
        4,
        2,
    )
    assert protocol.minimum_pseudo_dependent_queries_per_fold == 16
    assert protocol.required_admissible_fold_candidates == 1_520
    assert protocol.expected_python_version == "3.12.13"
    assert protocol.expected_torch_version == "2.13.0+cpu"
    assert protocol.expected_pyyaml_version == "6.0.3"

    universe = {(key, value) for key in range(5) for value in range(4)}
    assert {row.outer_cell for row in protocol.schedule[:20]} == universe
    assert {row.outer_cell for row in protocol.schedule[20:]} == universe
    assert [row.seed_pair_index for row in protocol.schedule[:20]] == list(range(20))
    assert [row.seed_pair_index for row in protocol.schedule[20:]] == [
        (index + 7) % 20 for index in range(20)
    ]
    assert Counter(row.outer_cell for row in protocol.schedule) == {
        cell: 2 for cell in universe
    }
    assert Counter(row.seed_pair_index for row in protocol.schedule) == {
        index: 2 for index in range(20)
    }
    source_root = Path(__file__).resolve().parents[1] / "src" / "tnlm_v3"
    expected_sources = {
        path.relative_to(Path(__file__).resolve().parents[2]).as_posix()
        for path in source_root.glob("*.py")
    }
    assert len(expected_sources) == 27
    assert expected_sources <= set(protocol.implementation_required_paths)
    assert "v3/pyproject.toml" in protocol.implementation_required_paths
    assert len(protocol.implementation_required_paths) == 33


def test_preflight_freezes_exact_primary_replay_and_optimizer_work() -> None:
    module = _load_script()
    protocol, preflight = module.preflight_campaign()
    assert preflight.environment_count == 40
    assert preflight.primary_generated_event_count == 122_880
    assert preflight.prototype_count == 20
    assert preflight.planned_objective_evaluations_per_fit == 8_001
    assert preflight.fit_calls_per_environment == 39
    assert preflight.fold_candidate_count == 1_520
    assert preflight.scored_event_work_per_environment == 479_408_640
    assert preflight.total_scored_event_work == 19_176_345_600
    assert protocol.max_primary_generated_events_total == 122_880
    assert protocol.max_deterministic_replay_generated_events_total == 491_520
    assert protocol.max_all_generation_work_total == 614_400


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("max_total_primary_generated_events", 3_071, "primary generated-event"),
        ("max_total_scored_event_work", 479_408_639, "scored-event"),
    ),
)
def test_preopen_budget_fails_before_prerequisite_generation_or_fit(
    monkeypatch: pytest.MonkeyPatch,
    keyword: str,
    value: int,
    message: str,
) -> None:
    module = _load_script()

    def forbidden(*args, **kwargs):
        raise AssertionError("expensive or external prerequisite work was reached")

    monkeypatch.setattr(module, "_prerequisites", forbidden)
    monkeypatch.setattr(module, "_build_frozen_trace_corpus", forbidden)
    monkeypatch.setattr(module, "run_outer_rotation", forbidden)
    with pytest.raises(SequenceDiscoveryLimitError, match=message):
        module.build_preopen_environment_record(0, **{keyword: value})


def test_production_power_record_validates_exact_positive_and_null_gates() -> None:
    module = _load_script()
    protocol = module.load_frozen_protocol()
    commitment = module.load_power_control_commitment(protocol)
    assert commitment.relative_path == "v3_recovery/PHASE2_ALGEBRA_POWER_CONTROL_V1.json"
    assert commitment.file_sha256 == (
        "4fa8d54c636ae693fdca7931ebb3e2095488eb8cc74f8d7f49e120e4c6e230a6"
    )
    assert commitment.record_sha256 == (
        "831bf991a99ad51b841a916f919df94def448c8f722fd9e1f06769e873c1913f"
    )
    with pytest.raises(ValueError, match="preregistered path"):
        module.load_power_control_commitment(protocol, Path(__file__))


def test_runtime_record_is_exact_and_platform_bound() -> None:
    module = _load_script()
    protocol = module.load_frozen_protocol()
    runtime = module.validate_runtime(protocol)
    assert runtime[:4] == ("3.12.13", "2.13.0+cpu", "6.0.3", "cpu")
    assert runtime.platform
    assert runtime.machine


def test_real_selection_and_rotation_dataclasses_round_trip_and_reject_forgery() -> None:
    module = _load_script()
    from tnlm_v3.algebra_discovery import (
        OuterRotationResult,
        SequenceAlgebraSelectionResult,
        VisibleEvent,
        VisibleSequence,
        make_trace_supervised_corpus,
        make_trace_supervised_sequence,
        run_outer_rotation,
    )
    from tnlm_v3.data import BindingEventKind

    def trace(key: int, value: int, split: str):
        sequence = VisibleSequence(
            events=(
                VisibleEvent(BindingEventKind.BIND, primary_key=key, argument=value),
                VisibleEvent(BindingEventKind.QUERY, primary_key=key),
            ),
            query_targets=(None, value),
        )
        state = ((key, value),)
        return make_trace_supervised_sequence(
            sequence,
            split=split,
            pre_event_cells=((), state),
            post_event_cells=(state, state),
            query_dependency_cells=((), state),
            num_surface_keys=2,
            value_cardinality=2,
        )

    corpus = make_trace_supervised_corpus(
        2,
        2,
        tuple(
            trace(key, value, split)
            for split in ("train", "validation")
            for key in range(2)
            for value in range(2)
            if (key, value) != (0, 0)
        ),
    )
    rotation = run_outer_rotation(
        (corpus,),
        residual_penalties=(4, 16),
        seed=0,
        restart_count=1,
        max_sweeps=1,
        max_pairwise_rounds=0,
    )
    final_replay = module._replay_final_model_on_direct_train(
        corpus, rotation.results[0]
    )
    assert final_replay["final_train_fit_independently_verified"]
    assert final_replay["replayed_training_mistakes"] == 0
    assert final_replay["training_sample_sha256_matches_fit_certificate"]
    selection_json = module._jsonable(rotation.results[0])
    assert module._decode_dataclass(
        SequenceAlgebraSelectionResult, selection_json, "selection"
    ) == rotation.results[0]
    rotation_json = module._jsonable(rotation)
    rotation_json.pop("results")
    assert module._decode_dataclass(
        OuterRotationResult,
        {**rotation_json, "results": [selection_json]},
        "rotation",
    ) == rotation
    forged = json.loads(json.dumps(selection_json))
    forged["final_model"]["fit"]["training_mistakes"] += 1
    with pytest.raises(ValueError):
        module._decode_dataclass(
            SequenceAlgebraSelectionResult, forged, "forged_selection"
        )


def test_synthetic_aggregate_is_canonical_complete_and_never_evidence() -> None:
    module = _load_script()
    protocol = module.load_frozen_protocol()
    records = tuple(
        _synthetic_preopen(module, protocol, index)
        for index in range(len(protocol.schedule))
    )
    aggregate = module._aggregate_preopen_records_for_test_only(records)
    assert aggregate == module._aggregate_preopen_records_for_test_only(
        tuple(reversed(records))
    )
    material = dict(aggregate)
    digest = material.pop("record_sha256")
    assert digest == hashlib.sha256(module._canonical_bytes(material)).hexdigest()
    assert aggregate["schema"] == module.TEST_FIXTURE_CAMPAIGN_SCHEMA
    assert aggregate["aggregate"]["environment_count"] == 40
    assert aggregate["aggregate"]["fold_candidate_count"] == 1_520
    assert aggregate["aggregate"]["admissible_fold_candidate_count"] == 1_520
    assert aggregate["aggregate"]["selected_penalty_counts"] == {"4": 20, "16": 20}
    assert aggregate["aggregate"]["passing_preopen_environment_count"] == 40
    assert not aggregate["acceptance"]["outer_open_permitted"]
    assert not aggregate["acceptance"]["scientific_evidence_permitted"]
    assert not aggregate["claims"]["representation_discovery_performed"]
    assert not aggregate["claims"]["secret_law_discovery_performed"]
    assert not aggregate["claims"]["confirmatory_claim_permitted"]


def test_synthetic_aggregate_rejects_missing_duplicate_mismatch_and_forgery() -> None:
    module = _load_script()
    protocol = module.load_frozen_protocol()
    records = [
        _synthetic_preopen(module, protocol, index)
        for index in range(len(protocol.schedule))
    ]
    with pytest.raises(ValueError, match="every frozen environment"):
        module._aggregate_preopen_records_for_test_only(records[:-1])
    with pytest.raises(ValueError, match="duplicate environment"):
        module._aggregate_preopen_records_for_test_only([*records[:-1], records[0]])
    mismatched = dict(records[0])
    mismatched.pop("record_sha256")
    mismatched["environment"] = dict(mismatched["environment"])
    mismatched["environment"]["outer_cell"] = [4, 3]
    with pytest.raises(ValueError, match="frozen schedule"):
        module._aggregate_preopen_records_for_test_only(
            [module._seal_body(mismatched), *records[1:]]
        )
    forged = json.loads(json.dumps(records[0]))
    forged["environment_preopen_gate_passed"] = False
    with pytest.raises(ValueError, match="record_sha256"):
        module._aggregate_preopen_records_for_test_only([forged, *records[1:]])


def test_production_aggregate_rejects_synthetic_schema_before_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    protocol = module.load_frozen_protocol()
    records = tuple(
        _synthetic_preopen(module, protocol, index)
        for index in range(len(protocol.schedule))
    )
    monkeypatch.setattr(
        module, "_prerequisites", lambda *args: _fake_prerequisites(module, protocol)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic schema reached corpus replay")

    monkeypatch.setattr(module, "_build_frozen_trace_corpus", forbidden)
    with pytest.raises(ValueError, match="non-evidence shard schemas"):
        module.aggregate_preopen_records(records)


@pytest.mark.parametrize("unclean", (False, True))
def test_preopen_shard_never_constructs_or_evaluates_any_probe(
    monkeypatch: pytest.MonkeyPatch,
    unclean: bool,
) -> None:
    module = _load_script()
    protocol = module.load_frozen_protocol()
    spec = protocol.schedule[0]
    selection = _fake_selection(module, protocol, 0, unclean=unclean)
    corpus = SimpleNamespace(corpus_sha256=selection.source_corpus_sha256)
    rotation = _Rotation(
        omitted_cell_sets=((spec.outer_cell,),),
        results=(selection,),
        result_sha256=_digest("rotation"),
    )
    monkeypatch.setattr(
        module, "_prerequisites", lambda *args: _fake_prerequisites(module, protocol)
    )
    monkeypatch.setattr(
        module, "_build_frozen_trace_corpus", lambda *args: (object(), corpus)
    )
    captured: dict[str, object] = {}

    def rotation_call(*args, **kwargs):
        captured.update(kwargs)
        return rotation

    monkeypatch.setattr(module, "run_outer_rotation", rotation_call)
    monkeypatch.setattr(
        module,
        "_replay_final_model_on_direct_train",
        lambda *args: {"final_train_fit_independently_verified": True},
    )
    monkeypatch.setattr(
        module,
        "_validate_preopen_material",
        lambda record, *_args, **_kwargs: SimpleNamespace(index=0),
    )

    def forbidden_probe(*args, **kwargs):
        raise AssertionError("preopen path constructed or evaluated a probe")

    for name in (
        "_postfit_report",
        "build_balanced_probe_suite",
        "cyclic_cell_rotation_inventory",
        "evaluate_probe_suite",
        "evaluate_shortcut_controls",
    ):
        monkeypatch.setattr(module, name, forbidden_probe)
    record = module.build_preopen_environment_record(0)
    assert record["schema"] == module.PREOPEN_ENVIRONMENT_SCHEMA
    assert "postfit" not in record
    assert record["claims"]["outer_probe_suite_constructed"] is False
    assert record["claims"]["outer_probe_answers_read"] is False
    assert record["claims"]["outer_or_rotated_probe_evaluation_performed"] is False
    assert captured["max_pairwise_rounds"] == 2
    assert captured["max_objective_evaluations_per_fit"] == 8_001
    assert record["seen_fit_gate"]["environment_preopen_gate_passed"] is (not unclean)
    assert record["execution_status"] == (
        "preopen_gate_failed_outer_open_forbidden"
        if unclean
        else "preopen_gate_passed_waiting_for_terminal_campaign_aggregate"
    )


def test_replay_budget_fails_before_first_corpus_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    protocol = module.load_frozen_protocol()
    # Production-shaped enough to reach the campaign-wide replay preflight.
    records = tuple(
        module._seal_body(
            {
                "schema": module.PREOPEN_ENVIRONMENT_SCHEMA,
                "environment": module._environment_spec_payload(protocol.schedule[index]),
            }
        )
        for index in range(len(protocol.schedule))
    )
    monkeypatch.setattr(
        module, "_prerequisites", lambda *args: _fake_prerequisites(module, protocol)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("replay began before campaign-wide budget rejection")

    monkeypatch.setattr(module, "_build_frozen_trace_corpus", forbidden)
    with pytest.raises(SequenceDiscoveryLimitError, match="validation replay"):
        module.aggregate_preopen_records(
            records,
            max_validation_replay_generated_events=122_879,
        )


def test_batch_open_rejects_failed_terminal_before_first_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    protocol = module.load_frozen_protocol()
    prerequisites = _fake_prerequisites(module, protocol)
    fake_validated = tuple(SimpleNamespace(index=index) for index in range(40))
    terminal = module._seal_body(
        {
            "schema": module.PREOPEN_AGGREGATE_SCHEMA,
            "generation_work_accounting": {
                "prior_resume_validation_replay_events": 0,
                "terminal_aggregate_validation_replay_events": 122_880,
                "cumulative_validation_replay_events_through_terminal_preopen": 122_880,
            },
            "acceptance": {"outer_open_permitted": False},
        }
    )
    monkeypatch.setattr(module, "_prerequisites", lambda *args: prerequisites)
    monkeypatch.setattr(
        module,
        "_validate_complete_preopen_records",
        lambda *args, **kwargs: (fake_validated, 122_880),
    )
    monkeypatch.setattr(
        module,
        "_build_terminal_preopen_from_validated",
        lambda *args, **kwargs: terminal,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("failed terminal aggregate opened a probe")

    monkeypatch.setattr(module, "_build_open_environment_record", forbidden)
    with pytest.raises(ValueError, match="forbids every outer probe"):
        module.open_campaign(terminal, tuple({} for _ in range(40)))


def test_batch_open_validates_complete_terminal_before_opening_all_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    protocol = module.load_frozen_protocol()
    prerequisites = _fake_prerequisites(module, protocol)
    fake_validated = tuple(SimpleNamespace(index=index) for index in range(40))
    terminal = module._seal_body(
        {
            "schema": module.PREOPEN_AGGREGATE_SCHEMA,
            "generation_work_accounting": {
                "prior_resume_validation_replay_events": 0,
                "terminal_aggregate_validation_replay_events": 122_880,
                "cumulative_validation_replay_events_through_terminal_preopen": 122_880,
            },
            "acceptance": {"outer_open_permitted": True},
        }
    )
    validation_complete = False

    def validate_all(*args, **kwargs):
        nonlocal validation_complete
        validation_complete = True
        return fake_validated, 122_880

    opened: list[int] = []

    def open_one(_protocol, _manifest, _runtime, _power, _terminal, row):
        assert validation_complete
        opened.append(row.index)
        return module._seal_body({"schema": module.OPEN_ENVIRONMENT_SCHEMA, "index": row.index})

    monkeypatch.setattr(module, "_prerequisites", lambda *args: prerequisites)
    monkeypatch.setattr(module, "_validate_complete_preopen_records", validate_all)
    monkeypatch.setattr(
        module,
        "_build_terminal_preopen_from_validated",
        lambda *args, **kwargs: terminal,
    )
    monkeypatch.setattr(module, "_build_open_environment_record", open_one)
    monkeypatch.setattr(
        module,
        "_build_open_campaign_record",
        lambda *args, **kwargs: module._seal_body(
            {"schema": module.CAMPAIGN_SCHEMA, "opened": list(opened)}
        ),
    )
    result = module.open_campaign(terminal, tuple({} for _ in range(40)))
    assert opened == list(range(40))
    assert result["opened"] == list(range(40))


def _perfect_evaluation(module) -> dict[str, object]:
    return {
        "query_count": module.EXPECTED_ACTUAL_QUERY_COUNT,
        "correct_count": module.EXPECTED_ACTUAL_QUERY_COUNT,
        "focal_query_count": module.EXPECTED_ACTUAL_FOCAL_QUERY_COUNT,
        "focal_correct_count": module.EXPECTED_ACTUAL_FOCAL_QUERY_COUNT,
        "path_consistency": 1.0,
        "path_relations": [
            {
                "relation": "equal",
                "predicted_focal_answers": [[0], [0]],
            }
            for _ in range(module.EXPECTED_PATH_RELATION_COUNT)
        ],
    }


def test_shortcut_and_rotated_controls_are_hard_acceptance_gates() -> None:
    module = _load_script()
    protocol = module.load_frozen_protocol()
    selection = _fake_selection(module, protocol, 0)
    gate = module._fold_seen_fit_report(
        protocol,
        selection,
        {"final_train_fit_independently_verified": True},
    )
    actual = _perfect_evaluation(module)
    rotated = {
        **actual,
        "query_count": module.EXPECTED_ROTATED_QUERY_COUNT,
        "correct_count": module.EXPECTED_ROTATED_QUERY_COUNT,
        "focal_query_count": module.EXPECTED_ROTATED_FOCAL_QUERY_COUNT,
        "focal_correct_count": module.EXPECTED_ROTATED_FOCAL_QUERY_COUNT,
    }
    postfit = {
        "transition_table": {
            "supported_entry_count": 20,
            "exact_entry_count": 20,
        },
        "actual_cell_probe": {"case_count": 15, "evaluation": actual},
        "exact_probe_family_count": 12,
        "shortcut_controls": [
            {
                "name": "shortcut",
                "evaluation": {"correct_count": 95, "focal_correct_count": 23},
            }
        ],
        "balanced_rotated_cell_control": {
            "case_count": 300,
            "evaluation": rotated,
        },
    }
    assert module._environment_acceptance(
        protocol, selection, gate, postfit
    )["environment_passed"]
    postfit["shortcut_controls"][0]["evaluation"] = {
        "correct_count": 96,
        "focal_correct_count": 24,
    }
    rejected = module._environment_acceptance(protocol, selection, gate, postfit)
    assert not rejected["all_shortcut_controls_strictly_worse_overall_and_focal"]
    assert not rejected["environment_passed"]


def test_source_exposes_only_two_phase_cli_and_scoped_claims() -> None:
    module = _load_script()
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_phase2_trace_algebra_multienvironment.py"
    )
    source = path.read_text(encoding="utf-8")
    assert '"preopen-environment"' in source
    assert '"aggregate-preopen"' in source
    assert '"open"' in source
    assert "def build_environment_record(" not in source
    assert "def build_preopen_environment_record(" in source
    assert '"representation_discovery_performed": False' in source
    assert '"secret_law_discovery_performed": False' in source
    assert '"confirmatory_claim_permitted": False' in source
    assert '"numeric_seed_reuse_is_common_random_number_matching": False' in source
    assert module.PREOPEN_ENVIRONMENT_SCHEMA != module.OPEN_ENVIRONMENT_SCHEMA


@pytest.mark.skipif(
    os.environ.get("TNLM_RUN_SLOW_MULTIENVIRONMENT") != "1",
    reason="set TNLM_RUN_SLOW_MULTIENVIRONMENT=1 for the frozen 40-environment run",
)
def test_full_v3_two_phase_campaign_opt_in_only(tmp_path: Path) -> None:
    module = _load_script()
    terminal = module.run_all_preopen_with_resume(
        tmp_path / "preopen",
        tmp_path / "terminal-preopen.json",
    )
    assert terminal["acceptance"]["outer_open_permitted"]
    paths = tuple(sorted((tmp_path / "preopen").glob("preopen-environment-*.json")))
    records = tuple(
        module._read_json(path, max_record_bytes=module.DEFAULT_MAX_ENVIRONMENT_RECORD_BYTES)
        for path in paths
    )
    campaign = module.open_campaign(terminal, records)
    assert campaign["acceptance"]["campaign_passed"]
    module.validate_campaign_record(campaign, terminal, records)
