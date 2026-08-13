from __future__ import annotations

import hashlib
import importlib.util
import json
import os
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
