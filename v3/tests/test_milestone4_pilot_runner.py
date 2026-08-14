from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "v3" / "scripts" / "run_milestone4_pilot.py"


def load_runner():  # type: ignore[no-untyped-def]
    name = "milestone4_pilot_runner_under_test"
    specification = importlib.util.spec_from_file_location(name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def runner():  # type: ignore[no-untyped-def]
    return load_runner()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def test_inventory_digest_matches_frozen_raw_entry_contract(runner) -> None:  # type: ignore[no-untyped-def]
    entries = (
        runner.BoundFile("z.py", "1" * 40, 2, "b" * 64),
        runner.BoundFile("a.py", "2" * 40, 10, "a" * 64),
    )
    raw = (
        b"domain\0"
        + b"a.py\0"
        + b"10\0"
        + b"a" * 64
        + b"\n"
        + b"z.py\0"
        + b"2\0"
        + b"b" * 64
        + b"\n"
    )
    assert runner._inventory_digest(b"domain\0", entries) == hashlib.sha256(
        raw
    ).hexdigest()


def test_git_provenance_uses_portable_safe_directory(
    runner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    repo = tmp_path / "checkout with space"
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        captured["argv"] = argv
        captured["cwd"] = kwargs["cwd"]
        return SimpleNamespace(returncode=0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner._run_git(repo, "rev-parse", "HEAD") == "ok\n"
    argv = captured["argv"]
    assert argv[2] == f"safe.directory={repo.as_posix()}"
    assert captured["cwd"] == repo


def test_attempt_paths_are_deterministic_and_external(runner) -> None:  # type: ignore[no-untyped-def]
    run_id = "a" * 64
    assert (
        runner._attempt_relative(run_id, 3)
        == f"artifacts/{run_id}/attempt-000003"
    )
    with pytest.raises(runner.PilotRunnerError):
        runner._attempt_relative("../escape", 1)


def test_process_lease_rejects_exact_live_process(
    runner, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
    )
    try:
        created = int(round(psutil.Process(process.pid).create_time() * 1_000_000_000))
        run_id = "a" * 64
        path = tmp_path / "process.json"
        path.write_bytes(
            canonical(
                runner._process_lease_document(
                    run_id=run_id,
                    attempt_number=1,
                    pid=process.pid,
                    process_create_time_ns=created,
                    status="running",
                    returncode=None,
                )
            )
        )
        with pytest.raises(runner.PilotRunnerError, match="still alive"):
            runner._assert_no_live_worker(path, run_id=run_id, attempt_number=1)
        value = json.loads(path.read_bytes())
        value["process_create_time_ns"] += 1
        path.write_bytes(canonical(value))
        runner._assert_no_live_worker(path, run_id=run_id, attempt_number=1)
    finally:
        process.kill()
        process.wait()


def test_strict_worker_json_rejects_duplicates_nonfinite_and_noncanonical(
    runner,  # type: ignore[no-untyped-def]
) -> None:
    for raw in (
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{ "a":1}',
        b'{"a":1}\n',
    ):
        with pytest.raises(runner.PilotRunnerError):
            runner._parse_canonical_json(raw, name="test")


def test_capture_is_bounded_but_hashes_full_stream(runner, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    lease = tmp_path / "process.json"
    raw = b"x" * (runner._MAX_CAPTURE_BYTES + 1000)
    emitter = tmp_path / "emit.py"
    emitter.write_text(
        f"import sys\nsys.stdout.buffer.write(b'x' * {len(raw)})\n",
        encoding="utf-8",
    )
    result = runner._run_worker_process(
        [sys.executable, str(emitter)],
        cwd=tmp_path,
        pythonpath=ROOT / "v3" / "src",
        lease_path=lease,
        run_id="a" * 64,
        attempt_number=1,
        timeout_seconds=30,
    )
    assert result.returncode == 0
    assert result.stdout.truncated
    assert len(result.stdout.text.encode()) == runner._MAX_CAPTURE_BYTES
    assert result.stdout.sha256 == hashlib.sha256(raw).hexdigest()
    lease_value = json.loads(lease.read_bytes())
    assert lease_value["status"] == "exited"
    assert lease_value["pid"] == result.pid


def test_process_failure_and_timeout_are_reported(runner, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    failure = runner._run_worker_process(
        [sys.executable, "-c", "raise SystemExit(7)"],
        cwd=tmp_path,
        pythonpath=ROOT / "v3" / "src",
        lease_path=tmp_path / "failed.json",
        run_id="a" * 64,
        attempt_number=1,
        timeout_seconds=30,
    )
    assert failure.returncode == 7
    assert not failure.timed_out
    timeout = runner._run_worker_process(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        pythonpath=ROOT / "v3" / "src",
        lease_path=tmp_path / "timeout.json",
        run_id="b" * 64,
        attempt_number=2,
        timeout_seconds=0.1,
    )
    assert timeout.timed_out
    assert timeout.returncode != 0


def test_private_reader_rejects_hardlinks_and_detects_tamper(
    runner, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    path = tmp_path / "input.py"
    path.write_bytes(b"one")
    linked = tmp_path / "linked.py"
    try:
        os.link(path, linked)
    except OSError as error:
        pytest.skip(f"hardlinks unavailable: {error}")
    with pytest.raises(runner.PilotRunnerError, match="private"):
        runner._read_private_file(path, maximum_bytes=10, name="input")
    linked.unlink()
    assert runner._read_private_file(path, maximum_bytes=10, name="input") == b"one"
    path.write_bytes(b"changed")
    assert runner._read_private_file(path, maximum_bytes=10, name="input") == b"changed"


def test_private_reader_preserves_windows_control_z_binary_byte(
    runner, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    raw = b"campaign-checkpoint\x1aafter-control-z\x00tail"
    path = tmp_path / "checkpoint.twcp"
    path.write_bytes(raw)
    assert runner._read_private_file(
        path, maximum_bytes=len(raw), name="checkpoint"
    ) == raw


def test_output_root_rejects_checkout_nesting(runner, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    checkout = tmp_path / "checkout"
    output = checkout / "output"
    output.mkdir(parents=True)
    with pytest.raises(runner.PilotRunnerError, match="disjoint"):
        runner._validated_output_root(output.resolve(), checkout.resolve())


def test_worker_argv_is_hermetic_and_binds_resume_and_parent(
    runner, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    provenance = runner.PilotProvenance(
        repo_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        runner_path=tmp_path / "runner.py",
        worker_path=tmp_path / "worker.py",
        code_commit="1" * 40,
        code_tree="2" * 40,
        raw_config_sha256="3" * 64,
        parent_runner_sha256="4" * 64,
        worker_sha256="5" * 64,
        package_tree_sha256="6" * 64,
        executable_bundle_sha256="7" * 64,
        bound_files=(),
    )
    resume = runner.ArtifactReference("artifacts/a/checkpoint.twcp", "8" * 64, 3)
    parent = runner.PilotRunSummary(
        run_id="9" * 64,
        model_id="parent",
        pair_id="pair",
        role="trainable_source",
        attempt_number=1,
        resumed=False,
        result=runner.ArtifactReference("artifacts/p/result.json", "a" * 64, 2),
        output=runner.ArtifactReference("artifacts/p/output.json", "b" * 64, 2),
        final_checkpoint=runner.ArtifactReference(
            "artifacts/p/checkpoint.twcp", "c" * 64, 2
        ),
        compact_artifact=None,
        validation_batch_hashes=(),
        training_batch_hashes=(),
        training_token_counts=(),
        stream_prefix_sha256="d" * 64,
        initial_model_fingerprint="e" * 64,
        final_model_fingerprint="f" * 64,
    )
    class FakeConfig:
        def fingerprint(self) -> str:
            return "e" * 64
    class FakeAuthority:
        plan_sha256 = "f" * 64
        config = FakeConfig()
    class FakeRun:
        run_id = "0" * 64
    argv = runner._build_worker_argv(
        provenance,
        FakeAuthority(),
        FakeRun(),
        2,
        tmp_path,
        tmp_path / "result.json",
        resume,
        parent,
    )
    assert argv[:4] == (sys.executable, "-s", "-B", str(provenance.worker_path))
    assert "--resume-checkpoint" in argv
    assert "--parent-result" in argv
    assert "--parent-checkpoint" in argv


def test_unscheduled_checkpoint_cursor_is_rejected_before_decode(
    runner, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    run = SimpleNamespace(
        role="trainable_source",
        training=SimpleNamespace(optimizer_steps=12, checkpoint_interval=4),
    )
    reference = runner.ArtifactReference(
        "artifacts/a/attempt-000001/checkpoint-step-00000005.twcp",
        "a" * 64,
        1,
    )
    with pytest.raises(runner.PilotRunnerError, match="locked schedule"):
        runner._verify_checkpoint_semantics(
            reference,
            cursor=5,
            expected_streams=None,
            config=None,
            run=run,
            output_root=tmp_path,
        )


def test_immutable_subprocess_envelope_is_idempotent_across_crash_restart(
    runner, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    run = SimpleNamespace(run_id="b" * 64)
    result = runner.ArtifactReference(
        "artifacts/b/attempt-000001/result.json", "c" * 64, 5
    )
    empty = runner._Capture("", hashlib.sha256(b"").hexdigest(), 0, False)
    process = runner._ProcessResult(
        argv=(sys.executable, "worker.py"),
        pid=123,
        process_create_time_ns=456,
        returncode=0,
        timed_out=False,
        stdout=empty,
        stderr=empty,
    )
    document = runner._subprocess_envelope(
        process,
        run=run,
        attempt_number=1,
        result_reference=result,
        validation_status="success",
        failure=None,
    )
    path = tmp_path / "subprocess.json"
    first = runner._atomic_immutable_json(path, document)
    assert runner._atomic_immutable_json(path, document) == first
    assert runner._load_subprocess_envelope(
        path,
        run=run,
        attempt_number=1,
        result_reference=result,
    ) == document
    changed = dict(document)
    changed["pid"] = 999
    with pytest.raises(runner.PilotRunnerError, match="different bytes"):
        runner._atomic_immutable_json(path, changed)


def test_main_reports_failure_without_success_claim(runner, capsys) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit):
        runner.main([])
    captured = capsys.readouterr()
    assert "completed_runs" not in captured.out


def test_exact_three_pair_statistics_are_exhaustive_and_ranked(runner) -> None:  # type: ignore[no-untyped-def]
    values = (Fraction(1, 10), Fraction(0, 1), Fraction(-1, 10))
    result = runner._paired_delta_statistics(values)
    assert result["raw_delta_vector"] == [[1, 10], [0, 1], [-1, 10]]
    assert len(result["ordered_resample_indices"]) == 27
    assert len(result["ordered_empirical_resample_means"]) == 27
    assert result["ordered_resample_indices"][0] == [0, 0, 0]
    assert result["ordered_resample_indices"][-1] == [2, 2, 2]
    assert result["ordered_empirical_resample_means"][0] == [1, 10]
    assert result["ordered_empirical_resample_means"][-1] == [-1, 10]
    assert result["fifth_percentile_nearest_rank"] == 2
    assert result["fifth_percentile"] == [-1, 15]
    assert result["sign_test_minimum_p"] == [1, 8]
    assert result["mean"] == [0, 1]
    assert result["median"] == [0, 1]
    assert result["sample_sd"] == pytest.approx(0.1)
    assert result["standard_error"] == pytest.approx(0.1 / (3**0.5))


def test_parent_strictly_validates_routing_summary_schema_and_identities(runner) -> None:  # type: ignore[no-untyped-def]
    entropy = -(2 / 3) * __import__("math").log(2 / 3) - (1 / 3) * __import__("math").log(1 / 3)
    routing = {
        "route_recovery": {
            "correct": 2,
            "local_event_count": 3,
            "accuracy": 2 / 3,
            "macro_accuracy": 0.75,
            "document_count": 2,
        },
        "route_consistency": {
            "consistent_events": 3,
            "local_event_count": 3,
            "consistency": 1.0,
            "group_count": 2,
            "fully_consistent_groups": 2,
        },
        "router_load": {
            "branch_counts": [2, 1, 0],
            "branch_fractions": [2 / 3, 1 / 3, 0.0],
            "local_event_count": 3,
            "global_event_count": 1,
            "null_event_count": 1,
            "valid_event_count": 5,
            "global_event_fraction": 0.2,
            "null_event_fraction": 0.2,
            "active_branches": 2,
            "collapsed": False,
            "document_count": 2,
            "collapsed_document_count": 0,
            "collapsed_document_fraction": 0.0,
            "mean_active_branches_per_document": 1.5,
            "max_load_fraction": 2 / 3,
            "load_entropy": entropy,
            "normalized_load_entropy": entropy / __import__("math").log(3),
            "mean_assignment_entropy": 0.5,
            "normalized_mean_assignment_entropy": 0.5 / __import__("math").log(3),
            "assignment_entropy_count": 3,
        },
    }
    encoded = runner._validate_routing_metrics(
        routing,
        run=SimpleNamespace(family="routed"),
        config=SimpleNamespace(task=SimpleNamespace(branches=3)),
        episodes=2,
    )
    assert json.loads(encoded) == routing
    tampered = json.loads(json.dumps(routing))
    tampered["router_load"]["branch_fractions"][0] = 0.5
    with pytest.raises(runner.PilotRunnerError, match="exact counts"):
        runner._validate_routing_metrics(
            tampered,
            run=SimpleNamespace(family="routed"),
            config=SimpleNamespace(task=SimpleNamespace(branches=3)),
            episodes=2,
        )
    with pytest.raises(runner.PilotRunnerError, match="must be null"):
        runner._validate_routing_metrics(
            routing,
            run=SimpleNamespace(family="gru"),
            config=SimpleNamespace(task=SimpleNamespace(branches=3)),
            episodes=2,
        )


def test_screen_promotion_selection_gates_and_parameter_invariance(
    runner, monkeypatch  # type: ignore[no-untyped-def]
) -> None:
    config = runner.load_milestone4_campaign_config(
        ROOT / "v3" / "configs" / "milestone4" / "validation_screen_v1.yaml"
    )
    plan = runner.resolve_campaign_plan(
        config,
        "1" * 40,
        "2" * 40,
        "3" * 64,
        "4" * 64,
    )
    scores = {
        "routed-source": 80,
        "routed-oracle": 82,
        "routed-latent": 76,
        "routed-compact-r2": 75,
        "routed-compact-r4": 79,
        "gru-control": 75,
        "transformer-control": 75,
        "ttn-control": 75,
    }
    parameters = {
        "routed-source": 200,
        "routed-oracle": 200,
        "routed-latent": 200,
        "routed-compact-r2": 50,
        "routed-compact-r4": 60,
        "gru-control": 175,
        "transformer-control": 180,
        "ttn-control": 170,
    }

    def metric(run, length: int):  # type: ignore[no-untyped-def]
        correct = scores[run.model_id]
        seen_correct = round(correct * 0.6)
        routing_json = None
        if run.family == "routed":
            routing_json = json.dumps(
                {
                    "route_recovery": {"macro_accuracy": 0.9},
                    "router_load": {
                        "collapsed": False,
                        "collapsed_document_count": 0,
                        "local_event_count": 20,
                        "active_branches": 3,
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        return runner.ValidationLengthSummary(
            length=length,
            query_correct=correct,
            query_count=100,
            seen_correct=seen_correct,
            seen_count=60,
            heldout_correct=correct - seen_correct,
            heldout_count=40,
            structural_values=(("parameter_count", parameters[run.model_id]),),
            routing_json=routing_json,
        )

    summaries = {}
    for run in plan:
        prefix = f"artifacts/{run.run_id}/attempt-000001"
        result = runner.ArtifactReference(f"{prefix}/result.json", "a" * 64, 10)
        output = runner.ArtifactReference(f"{prefix}/subprocess.json", "b" * 64, 10)
        checkpoint = (
            runner.ArtifactReference(f"{prefix}/checkpoint-step-00000512.twcp", "c" * 64, 10)
            if run.role == "trainable_source"
            else None
        )
        compact = (
            runner.ArtifactReference(f"{prefix}/compact-model.tnlm3", "d" * 64, 10)
            if run.role == "derived_compact"
            else None
        )
        summaries[run.run_id] = runner.PilotRunSummary(
            run_id=run.run_id,
            model_id=run.model_id,
            pair_id=run.pair_id,
            role=run.role,
            attempt_number=1,
            resumed=False,
            result=result,
            output=output,
            final_checkpoint=checkpoint,
            compact_artifact=compact,
            validation_batch_hashes=tuple(
                (length, hashlib.sha256(f"{run.pair_id}:{length}".encode()).hexdigest())
                for length in config.data.validation.lengths
            ),
            training_batch_hashes=(),
            training_token_counts=(),
            stream_prefix_sha256=("e" * 64 if checkpoint else None),
            initial_model_fingerprint=("f" * 64 if checkpoint else None),
            final_model_fingerprint="0" * 64,
            validation_metrics=tuple(
                metric(run, length) for length in config.data.validation.lengths
            ),
            parameter_count=parameters[run.model_id],
            checkpoints=(() if checkpoint is None else (checkpoint,)),
            compact_lineage_json=("{}" if compact is not None else None),
        )

    monkeypatch.setattr(runner, "campaign_manifest_sha256", lambda _: "9" * 64)
    document, decision = runner._screen_promotion_document(
        config=config,
        plan=plan,
        authority=SimpleNamespace(plan_sha256="5" * 64),
        provenance=SimpleNamespace(
            raw_config_sha256="3" * 64,
            code_commit="1" * 40,
            code_tree="2" * 40,
            parent_runner_sha256="6" * 64,
            worker_sha256="7" * 64,
            package_tree_sha256="8" * 64,
            executable_bundle_sha256="4" * 64,
        ),
        manifest=SimpleNamespace(generation=73),
        summaries=summaries,
    )
    assert decision == "complete_promote"
    assert document["selection"]["reference"] == {
        "model_id": "routed-source",
        "retained": True,
    }
    assert document["selection"]["oracle_diagnostic"]["promoted"] is False
    assert document["selection"]["compact_winner"]["model_id"] == "routed-compact-r2"
    assert "routed-oracle" not in document["selection"]["promoted_model_ids"]
    assert len(
        document["model_aggregates"]["routed-compact-r2"]["partitions"]["query"]
        ["delta_vs_routed_source"]["ordered_empirical_resample_means"]
    ) == 27
    assert len(canonical(document)) <= runner._MAX_OUTPUT_BYTES

    compact_run = next(
        run
        for run in plan
        if run.model_id == "routed-compact-r2"
        and run.pair_id == config.pairs[0].pair_id
    )
    compact_summary = summaries[compact_run.run_id]
    compact_first = compact_summary.validation_metrics[0]
    compact_zero_local = json.loads(compact_first.routing_json)
    compact_zero_local["router_load"]["local_event_count"] = 0
    compact_zero_local["router_load"]["active_branches"] = 0
    summaries[compact_run.run_id] = replace(
        compact_summary,
        validation_metrics=(
            replace(
                compact_first,
                routing_json=json.dumps(
                    compact_zero_local, sort_keys=True, separators=(",", ":")
                ),
            ),
            *compact_summary.validation_metrics[1:],
        ),
    )
    candidate_document, candidate_decision = runner._screen_promotion_document(
        config=config,
        plan=plan,
        authority=SimpleNamespace(plan_sha256="5" * 64),
        provenance=SimpleNamespace(
            raw_config_sha256="3" * 64,
            code_commit="1" * 40,
            code_tree="2" * 40,
            parent_runner_sha256="6" * 64,
            worker_sha256="7" * 64,
            package_tree_sha256="8" * 64,
            executable_bundle_sha256="4" * 64,
        ),
        manifest=SimpleNamespace(generation=73),
        summaries=summaries,
    )
    assert candidate_decision == "complete_promote"
    assert candidate_document["selection"]["compact_winner"]["model_id"] == (
        "routed-compact-r4"
    )
    global_collapse_gate = next(
        item
        for item in candidate_document["gates"]["results"]
        if item["gate"] == "require_no_route_collapse"
    )
    assert global_collapse_gate["passed"] is True
    rejected_compact = next(
        item
        for item in candidate_document["selection"]["compact_qualifiers"]
        if item["model_id"] == "routed-compact-r2"
    )
    assert rejected_compact["route_evidence_passed"] is False
    summaries[compact_run.run_id] = compact_summary

    latent_run = next(
        run
        for run in plan
        if run.model_id == "routed-latent"
        and run.pair_id == config.pairs[0].pair_id
    )
    latent_summary = summaries[latent_run.run_id]
    latent_first = latent_summary.validation_metrics[0]
    latent_zero_local = json.loads(latent_first.routing_json)
    latent_zero_local["router_load"]["local_event_count"] = 0
    latent_zero_local["router_load"]["active_branches"] = 0
    summaries[latent_run.run_id] = replace(
        latent_summary,
        validation_metrics=(
            replace(
                latent_first,
                routing_json=json.dumps(
                    latent_zero_local, sort_keys=True, separators=(",", ":")
                ),
            ),
            *latent_summary.validation_metrics[1:],
        ),
    )
    latent_document, latent_decision = runner._screen_promotion_document(
        config=config,
        plan=plan,
        authority=SimpleNamespace(plan_sha256="5" * 64),
        provenance=SimpleNamespace(
            raw_config_sha256="3" * 64,
            code_commit="1" * 40,
            code_tree="2" * 40,
            parent_runner_sha256="6" * 64,
            worker_sha256="7" * 64,
            package_tree_sha256="8" * 64,
            executable_bundle_sha256="4" * 64,
        ),
        manifest=SimpleNamespace(generation=73),
        summaries=summaries,
    )
    assert latent_decision == "complete_promote"
    latent_winner = next(
        item
        for item in latent_document["selection"]["standard_winners"]
        if item["stratum"] == "routed_latent"
    )
    assert latent_winner["model_id"] == "routed-latent"
    assert "routed-latent" in latent_document["selection"]["promoted_model_ids"]
    assert latent_document["selection"]["standard_stratum_selection_complete"] == {
        "passed": True,
        "selected": 4,
        "required": 4,
        "diagnostic_only": True,
    }
    latent_candidate = next(
        item
        for item in latent_document["selection"]["standard_candidates"]
        if item["model_id"] == "routed-latent"
    )
    assert latent_candidate["qualified"] is True
    assert latent_candidate["route_evidence_passed"] is False
    latent_global_gate = next(
        item
        for item in latent_document["gates"]["results"]
        if item["gate"] == "require_no_route_collapse"
    )
    assert latent_global_gate["passed"] is True
    summaries[latent_run.run_id] = latent_summary

    oracle_run = next(
        run
        for run in plan
        if run.model_id == "routed-oracle" and run.pair_id == config.pairs[0].pair_id
    )
    oracle_summary = summaries[oracle_run.run_id]
    first_metric = oracle_summary.validation_metrics[0]
    zero_local = json.loads(first_metric.routing_json)
    zero_local["router_load"]["local_event_count"] = 0
    zero_local["router_load"]["active_branches"] = 0
    summaries[oracle_run.run_id] = replace(
        oracle_summary,
        validation_metrics=(
            replace(
                first_metric,
                routing_json=json.dumps(
                    zero_local, sort_keys=True, separators=(",", ":")
                ),
            ),
            *oracle_summary.validation_metrics[1:],
        ),
    )
    rejected_document, rejected_decision = runner._screen_promotion_document(
        config=config,
        plan=plan,
        authority=SimpleNamespace(plan_sha256="5" * 64),
        provenance=SimpleNamespace(
            raw_config_sha256="3" * 64,
            code_commit="1" * 40,
            code_tree="2" * 40,
            parent_runner_sha256="6" * 64,
            worker_sha256="7" * 64,
            package_tree_sha256="8" * 64,
            executable_bundle_sha256="4" * 64,
        ),
        manifest=SimpleNamespace(generation=73),
        summaries=summaries,
    )
    assert rejected_decision == "complete_do_not_promote"
    collapse_gate = next(
        item
        for item in rejected_document["gates"]["results"]
        if item["gate"] == "require_no_route_collapse"
    )
    assert collapse_gate["passed"] is False
    summaries[oracle_run.run_id] = oracle_summary

    changed_run = next(
        run
        for run in plan
        if run.model_id == "routed-source" and run.pair_id == config.pairs[-1].pair_id
    )
    summaries[changed_run.run_id] = replace(
        summaries[changed_run.run_id], parameter_count=201
    )
    with pytest.raises(runner.PilotRunnerError, match="not invariant"):
        runner._screen_model_aggregates(config, summaries, plan)
