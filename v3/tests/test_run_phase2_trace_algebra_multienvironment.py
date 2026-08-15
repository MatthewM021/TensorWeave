from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tnlm_v3.algebra_discovery import SequenceDiscoveryLimitError


_POST_FREEZE_PACKAGE_SOURCE_SHA256 = {
    "src/tnlm_v3/opaque_active_discovery.py": (
        "4af57946adfbb7dc704a9fa7f30cede0b0cda2873cddf6cd22e3f9df59e320b9"
    ),
    "src/tnlm_v3/opaque_active_discovery_protocol.py": (
        "b84c0579eb76f8bc6ea90aded979c56983b1d0b01df243dc2e09ead4cb97af06"
    ),
    "src/tnlm_v3/opaque_active_teaching_control.py": (
        "5c0ca90c0bdeb92f338ef89dc5bc275e5ca2d6e0dd3eb56a924333c97f1238ae"
    ),
    "src/tnlm_v3/opaque_partial_operators.py": (
        "6efd6dad92e8c3c22fb787071dee599363a61a11b4d560f60d7a5d0fb20e9738"
    ),
}


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


def test_v4_protocol_expands_fresh_crossed_schedule_source_closure_and_runtime() -> None:
    module = _load_script()
    protocol = module.load_frozen_protocol()
    assert protocol.protocol_sha256 == (
        "966601e683b647b8f68ed0e99c1deca7a449f977027b5579fd6617522d42ec7b"
    )
    assert protocol.protocol_sha256 == module.PROTOCOL_SHA256
    assert protocol.execution_ready
    assert json.loads(module.default_protocol_path().read_text(encoding="utf-8"))[
        "execution_blocker"
    ] is None
    assert protocol.power_control_expected_file_sha256 == (
        "31cc623ad890582e21c1c2c414f14fb87ecc149e93b504fe28bbb3913dba00e3"
    )
    assert protocol.schedule_sha256 == (
        "a429725af7c72f20cb5d8755c664dcdaf552bf60176e05e68579da85e95838ae"
    )
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
    assert {row.train_seed for row in protocol.schedule} == set(range(40_000, 40_020))
    assert {row.validation_seed for row in protocol.schedule} == set(
        range(50_000, 50_020)
    )
    assert {row.optimizer_seed for row in protocol.schedule} == set(
        range(60_000, 60_020)
    )
    assert not (
        {row.train_seed for row in protocol.schedule}
        & set(range(10_000, 10_020))
    )
    source_root = Path(__file__).resolve().parents[1] / "src" / "tnlm_v3"
    expected_sources = {
        path.relative_to(Path(__file__).resolve().parents[2]).as_posix()
        for path in source_root.glob("*.py")
    }
    assert expected_sources <= set(protocol.implementation_required_paths)
    assert set(protocol.implementation_required_paths) == expected_sources | {
        "v3/configs/milestone4/validation_screen_v1.yaml",
        "v3/configs/phase2/outer_rotation_v4.json",
        "v3/pyproject.toml",
        "v3/scripts/run_phase2_algebra_power_control.py",
        "v3/scripts/run_phase2_trace_algebra_experiment.py",
        "v3/scripts/run_phase2_trace_algebra_multienvironment.py",
    }
    assert "v3/pyproject.toml" in protocol.implementation_required_paths
    assert "v3/configs/phase2/outer_rotation_v3.json" not in (
        protocol.implementation_required_paths
    )
    assert protocol.predecessor_evidence_commitment_sha256 == hashlib.sha256(
        module._canonical_bytes(module._expected_v3_predecessor_commitment())
    ).hexdigest()
    assert protocol.probe_source_sha256 == (
        "b4f2b45e53db9fd75fd506d487b68c4cdecbd408ba4246d5b0b68d7136449c80"
    )


def test_v4_is_forward_only_and_binds_the_immutable_failed_v3_evidence() -> None:
    module = _load_script()
    v3_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "phase2"
        / "outer_rotation_v3.json"
    )
    with pytest.raises(ValueError, match="immutable V3 evidence protocol"):
        module.load_frozen_protocol(v3_path)
    protocol = module.load_frozen_protocol()
    record = module._protocol_record(protocol)
    predecessor = record["v3_predecessor_evidence_commitment"]
    assert predecessor["v3_open_campaign"]["campaign_passed"] is False
    assert predecessor["v3_open_campaign"]["passing_environment_count"] == 10
    assert predecessor["v3_open_campaign"]["all_realized_probe_answers_correct"]
    assert predecessor["v3_formal_result_remains_failed_and_is_not_rewritten"]
    assert predecessor["v4_was_designed_after_v3_outer_results_were_opened"]
    assert not predecessor["v3_labels_models_or_scores_used_for_v4_fit_or_selection"]


def test_v4_phase_zero_probe_inventory_is_uniform_and_exactly_hash_bound() -> None:
    module = _load_script()
    protocol = module.load_frozen_protocol()
    from tnlm_v3.algebra_discovery_probes import (
        ProbeFamily,
        ProbeQueryRole,
        build_balanced_probe_suite,
        cyclic_cell_rotation_inventory,
    )

    expected_hashes = {
        cell: (actual_sha, rotated_sha)
        for cell, actual_sha, rotated_sha in protocol.probe_suite_hashes
    }
    assert protocol.probe_family_names == tuple(family.value for family in ProbeFamily)
    assert protocol.probe_family_query_counts == (
        4,
        4,
        4,
        4,
        8,
        12,
        4,
        8,
        16,
        8,
        8,
        16,
    )
    for cell in sorted(expected_hashes):
        actual = build_balanced_probe_suite(5, 4, (cell,))
        rotated = build_balanced_probe_suite(
            5,
            4,
            (cell,),
            cell_rotations=cyclic_cell_rotation_inventory(5, 4, anchor_key=cell[0]),
        )
        module._validate_constructed_probe_instrument(
            protocol,
            next(row for row in protocol.schedule if row.outer_cell == cell),
            actual,
            rotated,
        )
        assert (actual.suite_sha256, rotated.suite_sha256) == expected_hashes[cell]
        assert len(actual.cases) == 15
        assert sum(len(case.expected_answers) for case in actual.cases) == 96
        assert sum(
            sum(role is ProbeQueryRole.FOCAL for role in case.query_roles)
            for case in actual.cases
        ) == 24
        assert all(
            len(case.expected_answers)
            == 4 * sum(role is ProbeQueryRole.FOCAL for role in case.query_roles)
            for case in actual.cases
        )
        assert len(rotated.cases) == 300
        assert sum(len(case.expected_answers) for case in rotated.cases) == 1_920
        assert rotated.balance.pair_case_counts == tuple(
            (pair, 15) for pair in sorted(expected_hashes)
        )
        assert rotated.balance.pair_focal_query_counts == tuple(
            (pair, 24) for pair in sorted(expected_hashes)
        )


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
    assert protocol.postopen_model_query_evaluations_per_environment == 2_400
    assert protocol.max_primary_postopen_model_query_evaluations_total == 96_000
    assert (
        protocol.max_validation_replay_postopen_model_query_evaluations_total
        == 96_000
    )
    assert protocol.max_all_postopen_model_query_evaluations_total == 192_000


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


def test_production_power_v2_is_immutable_and_fails_closed_on_one_file_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    protocol = module.load_frozen_protocol()
    assert protocol.power_control_relative_path == (
        "v3_recovery/PHASE2_ALGEBRA_POWER_CONTROL_V2.json"
    )
    record_path = module.default_power_control_record_path()
    payload = record_path.read_bytes()
    expected_file_sha256 = (
        "31cc623ad890582e21c1c2c414f14fb87ecc149e93b504fe28bbb3913dba00e3"
    )
    expected_record_sha256 = (
        "74a8d6ee5b519f8e1bc840a6e3838952ff967065ca7b11f642d447c5e9123b12"
    )
    assert len(payload) == 1_303_121
    assert hashlib.sha256(payload).hexdigest() == expected_file_sha256
    assert expected_file_sha256 == protocol.power_control_expected_file_sha256

    document = json.loads(payload)
    assert payload == (
        json.dumps(
            document,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    material = dict(document)
    assert material.pop("record_sha256") == expected_record_sha256
    assert hashlib.sha256(module._canonical_bytes(material)).hexdigest() == (
        expected_record_sha256
    )

    recorded_sources = document["source_file_sha256"]
    assert isinstance(recorded_sources, dict)
    assert len(recorded_sources) == 30
    v3_root = Path(__file__).resolve().parents[1]
    for relative, expected_digest in recorded_sources.items():
        assert hashlib.sha256((v3_root / relative).read_bytes()).hexdigest() == (
            expected_digest
        )

    power_runner = module._load_power_runner()
    current_sources = power_runner._source_hashes()
    added_sources = _POST_FREEZE_PACKAGE_SOURCE_SHA256
    assert set(current_sources) - set(recorded_sources) == set(added_sources)
    assert not set(recorded_sources) - set(current_sources)
    assert all(current_sources[path] == digest for path, digest in recorded_sources.items())
    assert {path: current_sources[path] for path in added_sources} == added_sources

    def forbidden(*args, **kwargs):
        raise AssertionError("fit or probe work was reached before source-drift rejection")

    monkeypatch.setattr(
        power_runner, "run_pair_local_exception_power_control", forbidden
    )
    monkeypatch.setattr(module, "_load_power_runner", lambda: power_runner)
    for name in (
        "_build_frozen_trace_corpus",
        "run_outer_rotation",
        "build_balanced_probe_suite",
        "evaluate_probe_suite",
        "evaluate_shortcut_controls",
    ):
        monkeypatch.setattr(module, name, forbidden)
    with pytest.raises(ValueError) as error:
        module.load_power_control_commitment(protocol)
    assert str(error.value) == "source hashes do not match the executing source tree"
    with pytest.raises(ValueError, match="preregistered path"):
        module.load_power_control_commitment(protocol, Path(__file__))


def test_production_v4_manifest_is_immutable_and_fails_closed_on_one_file_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    protocol = module.load_frozen_protocol()
    manifest_path = module.default_implementation_manifest_path()
    expected_sha256 = (
        "de030722267f30922a5f19b6ac65c4c0ce797bc45ded2d31f692df40254b39a0"
    )
    assert protocol.implementation_manifest_relative_path == (
        "v3_recovery/PHASE2_OUTER_ROTATION_V4_IMPLEMENTATION.sha256"
    )
    payload = manifest_path.read_bytes()
    assert len(payload) == 3_391
    assert hashlib.sha256(payload).hexdigest() == expected_sha256
    rows = tuple(
        (relative, digest)
        for digest, relative in (
            line.split("  ", 1) for line in payload.decode("utf-8").splitlines()
        )
    )
    assert len(rows) == 34
    assert payload == "".join(
        f"{digest}  {relative}\n" for relative, digest in rows
    ).encode("utf-8")
    assert hashlib.sha256(module._canonical_bytes(rows)).hexdigest() == (
        "c4a9a8f31052c047f4796e28d2205678623ae9ac3d74f6b20b899a602c55411d"
    )

    repository = Path(__file__).resolve().parents[2]
    for relative, expected_digest in rows:
        assert hashlib.sha256((repository / relative).read_bytes()).hexdigest() == (
            expected_digest
        )

    recorded_paths = tuple(relative for relative, _ in rows)
    current_paths = protocol.implementation_required_paths
    added_paths = {
        f"v3/{relative}": digest
        for relative, digest in _POST_FREEZE_PACKAGE_SOURCE_SHA256.items()
    }
    assert set(current_paths) - set(recorded_paths) == set(added_paths)
    assert not set(recorded_paths) - set(current_paths)
    assert tuple(path for path in current_paths if path not in added_paths) == recorded_paths
    assert {
        path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
        for path in added_paths
    } == added_paths

    def forbidden(*args, **kwargs):
        raise AssertionError("fit or probe work was reached before source-drift rejection")

    for name in (
        "_build_frozen_trace_corpus",
        "run_outer_rotation",
        "build_balanced_probe_suite",
        "evaluate_probe_suite",
        "evaluate_shortcut_controls",
    ):
        monkeypatch.setattr(module, name, forbidden)
    with pytest.raises(ValueError) as error:
        module.load_implementation_manifest(protocol)
    assert str(error.value) == "implementation manifest path inventory or order changed"
    with pytest.raises(ValueError, match="preregistered path"):
        module.load_implementation_manifest(protocol, Path(__file__))


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
    assert record["claims"][
        "trusted_probe_instrument_constructed_and_sealed_before_v4_model_fit"
    ]
    assert record["claims"][
        "outer_probe_suite_constructed_during_v4_preopen_execution"
    ] is False
    assert record["claims"][
        "outer_probe_answers_read_during_v4_preopen_execution"
    ] is False
    assert record["claims"][
        "v4_model_outer_or_rotated_probe_evaluation_performed"
    ] is False
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


def test_postopen_query_budgets_fail_before_prerequisites_or_first_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()

    def forbidden(*args, **kwargs):
        raise AssertionError("postopen work began before query-budget rejection")

    monkeypatch.setattr(module, "_prerequisites", forbidden)
    monkeypatch.setattr(module, "_build_open_environment_record", forbidden)
    with pytest.raises(SequenceDiscoveryLimitError, match="before probes"):
        module.open_campaign(
            {},
            (),
            max_primary_postopen_model_query_evaluations=95_999,
        )
    with pytest.raises(SequenceDiscoveryLimitError, match="cumulative budget"):
        module.validate_campaign_record(
            {},
            {},
            (),
            max_all_postopen_model_query_evaluations=191_999,
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


def _perfect_evaluation(module, *, rotated: bool = False) -> dict[str, object]:
    protocol = module.load_frozen_protocol()
    factor = 20 if rotated else 1
    query_count = (
        module.EXPECTED_ROTATED_QUERY_COUNT
        if rotated
        else module.EXPECTED_ACTUAL_QUERY_COUNT
    )
    focal_count = (
        module.EXPECTED_ROTATED_FOCAL_QUERY_COUNT
        if rotated
        else module.EXPECTED_ACTUAL_FOCAL_QUERY_COUNT
    )
    case_counts = (1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2)
    families = [
        {
            "family": name,
            "correct_count": factor * family_queries,
            "query_count": factor * family_queries,
            "focal_correct_count": factor,
            "focal_query_count": factor,
            "exact_case_count": factor * family_cases,
            "case_count": factor * family_cases,
        }
        for name, family_queries, family_cases in zip(
            protocol.probe_family_names,
            protocol.probe_family_query_counts,
            case_counts,
            strict=True,
        )
    ]
    pairs = [
        {
            "probe_pair": [key, value],
            "correct_count": module.EXPECTED_ACTUAL_QUERY_COUNT,
            "query_count": module.EXPECTED_ACTUAL_QUERY_COUNT,
            "focal_correct_count": module.EXPECTED_ACTUAL_FOCAL_QUERY_COUNT,
            "focal_query_count": module.EXPECTED_ACTUAL_FOCAL_QUERY_COUNT,
            "exact_case_count": module.EXPECTED_ACTUAL_CASE_COUNT,
            "case_count": module.EXPECTED_ACTUAL_CASE_COUNT,
        }
        for key, value in (
            [(key, value) for key in range(5) for value in range(4)]
            if rotated
            else [(0, 0)]
        )
    ]
    return {
        "query_count": query_count,
        "correct_count": query_count,
        "focal_query_count": focal_count,
        "focal_correct_count": focal_count,
        "path_consistency": 1.0,
        "families": families,
        "pairs": pairs,
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
    rotated = _perfect_evaluation(module, rotated=True)
    actual_suite_sha, rotated_suite_sha = {
        cell: (actual_sha, rotated_sha)
        for cell, actual_sha, rotated_sha in protocol.probe_suite_hashes
    }[(0, 0)]
    postfit = {
        "transition_table": {
            "supported_entry_count": 20,
            "exact_entry_count": 20,
        },
        "probe_instrument": {
            "corrected_after_v3_open": True,
            "nonconfirmatory_corrective_replication": True,
            "added_queries_are_nonfocal_balance_padding": True,
            "family_order": list(protocol.probe_family_names),
            "family_query_counts_per_cell": list(protocol.probe_family_query_counts),
            "family_output_class_counts_per_cell": [
                list(counts) for counts in protocol.probe_family_output_class_counts
            ],
            "actual_queries_per_cell": 96,
            "rotated_queries_per_pair": 96,
            "rotated_pair_count": 20,
            "actual_suite_sha256": actual_suite_sha,
            "rotated_suite_sha256": rotated_suite_sha,
            "actual_balance_certificate": {
                "family_class_counts": [
                    [name, list(counts)]
                    for name, counts in zip(
                        protocol.probe_family_names,
                        protocol.probe_family_output_class_counts,
                        strict=True,
                    )
                ]
            },
            "rotated_balance_certificate": {
                "family_class_counts": [
                    [name, [20 * count for count in counts]]
                    for name, counts in zip(
                        protocol.probe_family_names,
                        protocol.probe_family_output_class_counts,
                        strict=True,
                    )
                ]
            },
        },
        "actual_cell_probe": {
            "outer_cell": [0, 0],
            "case_count": 15,
            "evaluation": actual,
        },
        "exact_probe_family_count": 12,
        "shortcut_controls": [
            {
                "name": module.EXPECTED_SHORTCUT_NAMES[index],
                "evaluation": {
                    "query_count": 96,
                    "correct_count": 95,
                    "focal_query_count": 24,
                    "focal_correct_count": 23,
                },
            }
            for index in range(4)
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
        "query_count": 96,
        "correct_count": 96,
        "focal_query_count": 24,
        "focal_correct_count": 24,
    }
    rejected = module._environment_acceptance(protocol, selection, gate, postfit)
    assert not rejected["all_shortcut_controls_strictly_worse_overall_and_focal"]
    assert not rejected["environment_passed"]


def test_family_pair_balance_and_suite_hash_rows_are_hard_acceptance_gates() -> None:
    module = _load_script()
    protocol = module.load_frozen_protocol()
    selection = _fake_selection(module, protocol, 0)
    gate = module._fold_seen_fit_report(
        protocol,
        selection,
        {"final_train_fit_independently_verified": True},
    )
    # Reuse the production-shaped fixture assembled by the neighbouring test.
    actual = _perfect_evaluation(module)
    rotated = _perfect_evaluation(module, rotated=True)
    actual_sha, rotated_sha = {
        cell: (first, second)
        for cell, first, second in protocol.probe_suite_hashes
    }[(0, 0)]
    base = {
        "transition_table": {"supported_entry_count": 20, "exact_entry_count": 20},
        "probe_instrument": {
            "corrected_after_v3_open": True,
            "nonconfirmatory_corrective_replication": True,
            "added_queries_are_nonfocal_balance_padding": True,
            "family_order": list(protocol.probe_family_names),
            "family_query_counts_per_cell": list(protocol.probe_family_query_counts),
            "family_output_class_counts_per_cell": [
                list(counts) for counts in protocol.probe_family_output_class_counts
            ],
            "actual_queries_per_cell": 96,
            "rotated_queries_per_pair": 96,
            "rotated_pair_count": 20,
            "actual_suite_sha256": actual_sha,
            "rotated_suite_sha256": rotated_sha,
            "actual_balance_certificate": {
                "family_class_counts": [
                    [name, list(counts)]
                    for name, counts in zip(
                        protocol.probe_family_names,
                        protocol.probe_family_output_class_counts,
                        strict=True,
                    )
                ]
            },
            "rotated_balance_certificate": {
                "family_class_counts": [
                    [name, [20 * count for count in counts]]
                    for name, counts in zip(
                        protocol.probe_family_names,
                        protocol.probe_family_output_class_counts,
                        strict=True,
                    )
                ]
            },
        },
        "actual_cell_probe": {
            "outer_cell": [0, 0],
            "case_count": 15,
            "evaluation": actual,
        },
        "exact_probe_family_count": 12,
        "shortcut_controls": [
            {
                "name": module.EXPECTED_SHORTCUT_NAMES[index],
                "evaluation": {
                    "query_count": 96,
                    "correct_count": 95,
                    "focal_query_count": 24,
                    "focal_correct_count": 23,
                },
            }
            for index in range(4)
        ],
        "balanced_rotated_cell_control": {
            "case_count": 300,
            "evaluation": rotated,
        },
    }
    assert module._environment_acceptance(protocol, selection, gate, base)[
        "environment_passed"
    ]
    family_forgery = json.loads(json.dumps(base))
    family_forgery["actual_cell_probe"]["evaluation"]["families"][0][
        "query_count"
    ] = 3
    assert not module._environment_acceptance(
        protocol, selection, gate, family_forgery
    )["all_12_probe_families_exact"]
    pair_forgery = json.loads(json.dumps(base))
    pair_forgery["balanced_rotated_cell_control"]["evaluation"]["pairs"][0][
        "query_count"
    ] = 95
    assert not module._environment_acceptance(
        protocol, selection, gate, pair_forgery
    )["actual_and_rotated_pair_inventories_exact"]
    instrument_forgery = json.loads(json.dumps(base))
    instrument_forgery["probe_instrument"]["actual_suite_sha256"] = _digest(
        "forged-suite"
    )
    assert not module._environment_acceptance(
        protocol, selection, gate, instrument_forgery
    )["corrected_probe_instrument_exact"]


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
    assert '"v4_corrective_replication_is_confirmatory": False' in source
    assert (
        '"trusted_probe_instrument_constructed_and_sealed_before_v4_model_fit": True'
        in source
    )
    assert '"numeric_seed_reuse_is_common_random_number_matching": False' in source
    assert "rotated_control_1800_of_1800" not in source
    assert module.default_protocol_path().name == "outer_rotation_v4.json"
    assert module.default_implementation_manifest_path().name == (
        "PHASE2_OUTER_ROTATION_V4_IMPLEMENTATION.sha256"
    )
    assert module.default_power_control_record_path().name == (
        "PHASE2_ALGEBRA_POWER_CONTROL_V2.json"
    )
    assert module.PREOPEN_ENVIRONMENT_SCHEMA != module.OPEN_ENVIRONMENT_SCHEMA


@pytest.mark.skip(
    reason=(
        "positive V4 replay requires the frozen V4 source snapshot; the current "
        "head intentionally contains post-freeze Phase-III source"
    ),
)
def test_full_v4_two_phase_campaign_opt_in_only(tmp_path: Path) -> None:
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
