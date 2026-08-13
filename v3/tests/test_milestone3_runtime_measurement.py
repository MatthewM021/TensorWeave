from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from tnlm_v3.binding import (
    BindingArchitectureConfig,
    BindingModelConfig,
    RoutedBindingModel,
)
from tnlm_v3.compact_artifact import (
    serialize_compact_binding_model,
)
from tnlm_v3.data import (
    BindingModelInputs,
    BindingTaskConfig,
    collate_binding_episodes,
    generate_binding_episodes,
)
from tnlm_v3.model_export import export_compact_binding_model
from tnlm_v3.routing import CurriculumSchedule, RoutingMode
from tnlm_v3.truncation import (
    model_state_fingerprint,
    select_cp_rank_by_parameter_energy,
)


ROOT = Path(__file__).parents[1]
WORKER_SOURCE = ROOT / "scripts" / "measure_milestone3_runtime.py"


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.resolve().as_posix()}",
            *arguments,
        ],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip().lower()


@pytest.fixture()
def clean_worker_checkout(tmp_path: Path) -> dict[str, object]:
    """Build a small clean Git checkout so the production cleanliness gate runs."""

    checkout = tmp_path / "clean-checkout"
    source = checkout / "v3" / "src"
    scripts = checkout / "v3" / "scripts"
    shutil.copytree(
        ROOT / "src",
        source,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    scripts.mkdir(parents=True)
    worker = scripts / WORKER_SOURCE.name
    shutil.copy2(WORKER_SOURCE, worker)
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=TensorWeave Test",
            "-c",
            "user.email=tensorweave-test@example.invalid",
            "commit",
            "-qm",
            "clean runtime worker fixture",
        ],
        cwd=checkout,
        check=True,
    )
    return {
        "root": checkout,
        "worker": worker,
        "source": source,
        "commit": _git(checkout, "rev-parse", "HEAD"),
        "tree": _git(checkout, "rev-parse", "HEAD^{tree}"),
    }


def _inputs() -> BindingModelInputs:
    task = BindingTaskConfig(
        num_surface_keys=4,
        value_cardinality=3,
        branches=2,
        max_live_bindings=2,
        min_length=10,
        max_length=12,
        heldout_key_value_pairs=((0, 0),),
    )
    return collate_binding_episodes(
        generate_binding_episodes(
            task,
            count=2,
            seed=611,
            split="eval",
            lengths=[10, 12],
        )
    ).inputs


def _fixture(inputs: BindingModelInputs) -> dict[str, object]:
    tensors: dict[str, object] = {}
    for name in (
        "token_ids",
        "event_kinds",
        "primary_key_ids",
        "secondary_key_ids",
        "arguments",
        "valid_mask",
    ):
        value = getattr(inputs, name).detach().cpu()
        tensors[name] = {
            "dtype": "bool" if value.dtype is torch.bool else "int64",
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    return {"schema_version": 1, "inputs": tensors}


def _model() -> RoutedBindingModel:
    torch.manual_seed(781)
    return RoutedBindingModel(
        BindingModelConfig(
            task=BindingArchitectureConfig(4, 3, 2),
            d_model=5,
            cp_rank=6,
            router_hidden_dim=4,
            routing_mode=RoutingMode.CURRICULUM,
            curriculum_schedule=CurriculumSchedule(
                start_step=0,
                end_step=10,
                start_probability=1.0,
                end_probability=0.0,
            ),
            curriculum_seed=17,
            scale_feature_dim=4,
        )
    ).eval()


def _model_config_dict(model: RoutedBindingModel) -> dict[str, object]:
    config = model.config
    schedule = config.curriculum_schedule
    return {
        "task": asdict(config.task),
        "d_model": config.d_model,
        "cp_rank": config.cp_rank,
        "router_hidden_dim": config.router_hidden_dim,
        "routing_mode": config.routing_mode.value,
        "curriculum_schedule": None if schedule is None else schedule.as_dict(),
        "curriculum_seed": config.curriculum_seed,
        "scale_feature_dim": config.scale_feature_dim,
        "straight_through_route_surrogate": config.straight_through_route_surrogate,
    }


def _environment(checkout: dict[str, object]) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(checkout["source"]).resolve())
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _common_command(
    checkout: dict[str, object],
    fixture: Path,
    output: Path,
    model_fingerprint: str,
    *,
    fixture_sha256: str | None = None,
) -> list[str]:
    return [
        sys.executable,
        "-s",
        "-B",
        str(Path(checkout["worker"]).resolve()),
        "--fixture",
        str(fixture.resolve()),
        "--output",
        str(output.resolve()),
        "--expected-fixture-sha256",
        fixture_sha256 or hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "--expected-model-fingerprint",
        model_fingerprint,
        "--expected-code-commit",
        str(checkout["commit"]),
        "--expected-code-tree",
        str(checkout["tree"]),
        "--warmup-iterations",
        "1",
        "--timed-iterations",
        "2",
        "--torch-threads",
        "1",
        "--rss-sample-period-ms",
        "0.5",
    ]


def _run(
    command: list[str], checkout: dict[str, object]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=Path(checkout["root"]) / "v3",
        env=_environment(checkout),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
    )


def _write_fixture(path: Path) -> None:
    path.write_text(
        json.dumps(_fixture(_inputs()), allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )


def test_source_checkpoint_is_measured_in_an_isolated_process(
    tmp_path: Path, clean_worker_checkout: dict[str, object]
) -> None:
    model = _model()
    fixture = tmp_path / "fixture.json"
    checkpoint = tmp_path / "source.pt"
    model_config = tmp_path / "model-config.json"
    output = tmp_path / "runtime.json"
    _write_fixture(fixture)
    torch.save(model.state_dict(), checkpoint)
    model_config.write_text(
        json.dumps(
            _model_config_dict(model),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    command = [
        *_common_command(
            clean_worker_checkout, fixture, output, model_state_fingerprint(model)
        ),
        "--variant",
        "source",
        "--checkpoint",
        str(checkpoint.resolve()),
        "--expected-checkpoint-sha256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "--model-config-json",
        str(model_config.resolve()),
    ]

    completed = _run(command, clean_worker_checkout)

    assert completed.returncode == 0, completed.stderr
    stdout_record = json.loads(completed.stdout)
    record = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_record == record
    assert record["status"] == "passed"
    assert record["scope"] == "isolated_runtime_rss_measurement_not_a_comparative_win_gate"
    assert record["process_id"] != os.getpid()
    assert record["variant"] == "source"
    assert record["implementation"] == "streaming"
    assert record["model"]["model_fingerprint"] == model_state_fingerprint(model)
    assert record["checkout"]["code_commit"] == clean_worker_checkout["commit"]
    measurement = record["measurement"]
    assert measurement["warmup_iterations"] == 1
    assert measurement["timed_iterations"] == 2
    assert measurement["torch_threads"] == 1
    assert len(measurement["elapsed_ns_samples"]) == 2
    assert all(value > 0 for value in measurement["elapsed_ns_samples"])
    assert measurement["valid_events_per_iteration"] == 22
    assert measurement["tensor_positions_per_iteration"] == 24
    assert measurement["loaded_rss_bytes"] > 0
    assert measurement["warmed_rss_bytes"] > 0
    assert measurement["sampled_peak_rss_bytes"] >= measurement["warmed_rss_bytes"]
    assert measurement["incremental_sampled_peak_bytes"] >= 0
    assert measurement["valid_events_per_second"] > 0
    assert measurement["tensor_positions_per_second"] > 0


def test_compact_artifact_is_measured_with_trusted_provenance(
    tmp_path: Path, clean_worker_checkout: dict[str, object]
) -> None:
    source = _model()
    selection = select_cp_rank_by_parameter_energy(source, target_rank=3)
    compact, manifest = export_compact_binding_model(source, selection)
    compact.eval()
    artifact_bytes = serialize_compact_binding_model(compact, manifest, selection)
    artifact = tmp_path / "compact.tnlm3"
    fixture = tmp_path / "fixture.json"
    output = tmp_path / "runtime.json"
    artifact.write_bytes(artifact_bytes)
    _write_fixture(fixture)
    command = [
        *_common_command(
            clean_worker_checkout, fixture, output, model_state_fingerprint(compact)
        ),
        "--variant",
        "compact",
        "--artifact",
        str(artifact.resolve()),
        "--expected-artifact-sha256",
        hashlib.sha256(artifact_bytes).hexdigest(),
        "--expected-source-fingerprint",
        manifest.source_model_fingerprint,
        "--expected-manifest-fingerprint",
        manifest.fingerprint(),
        "--expected-selection-fingerprint",
        selection.fingerprint(),
    ]

    completed = _run(command, clean_worker_checkout)

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["variant"] == "compact"
    assert record["model"]["artifact_sha256"] == hashlib.sha256(artifact_bytes).hexdigest()
    assert record["model"]["source_model_fingerprint"] == manifest.source_model_fingerprint
    assert record["model"]["manifest_fingerprint"] == manifest.fingerprint()
    assert record["model"]["selection_fingerprint"] == selection.fingerprint()


def test_wrong_fixture_checksum_writes_one_strict_failure_record(
    tmp_path: Path, clean_worker_checkout: dict[str, object]
) -> None:
    model = _model()
    fixture = tmp_path / "fixture.json"
    checkpoint = tmp_path / "source.pt"
    model_config = tmp_path / "model-config.json"
    output = tmp_path / "failed.json"
    _write_fixture(fixture)
    torch.save(model.state_dict(), checkpoint)
    model_config.write_text(
        json.dumps(
            _model_config_dict(model),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    command = [
        *_common_command(
            clean_worker_checkout,
            fixture,
            output,
            model_state_fingerprint(model),
            fixture_sha256="0" * 64,
        ),
        "--variant",
        "dense_selected",
        "--checkpoint",
        str(checkpoint.resolve()),
        "--expected-checkpoint-sha256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "--model-config-json",
        str(model_config.resolve()),
    ]

    completed = _run(command, clean_worker_checkout)

    assert completed.returncode == 1
    stdout_record = json.loads(completed.stdout)
    record = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_record == record
    assert record["status"] == "failed"
    assert record["variant"] == "dense_selected"
    assert set(record["error"]) == {"type", "message"}
    assert "fixture SHA-256" in record["error"]["message"]


def test_cli_rejects_mixed_variant_inputs(
    tmp_path: Path, clean_worker_checkout: dict[str, object]
) -> None:
    fixture = tmp_path / "fixture.json"
    output = tmp_path / "out.json"
    _write_fixture(fixture)
    command = [
        *_common_command(clean_worker_checkout, fixture, output, "0" * 64),
        "--variant",
        "compact",
        "--artifact",
        str((tmp_path / "artifact").resolve()),
        "--expected-artifact-sha256",
        "0" * 64,
        "--expected-source-fingerprint",
        "0" * 64,
        "--expected-manifest-fingerprint",
        "0" * 64,
        "--expected-selection-fingerprint",
        "0" * 64,
        "--checkpoint",
        str((tmp_path / "checkpoint").resolve()),
    ]
    completed = _run(command, clean_worker_checkout)
    assert completed.returncode != 0
    assert "variant arguments invalid" in completed.stderr
    assert not output.exists()


def test_dirty_worker_checkout_writes_failure_record(
    tmp_path: Path, clean_worker_checkout: dict[str, object]
) -> None:
    model = _model()
    fixture = tmp_path / "fixture.json"
    checkpoint = tmp_path / "source.pt"
    model_config = tmp_path / "model-config.json"
    output = tmp_path / "dirty.json"
    _write_fixture(fixture)
    torch.save(model.state_dict(), checkpoint)
    model_config.write_text(
        json.dumps(
            _model_config_dict(model),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (Path(clean_worker_checkout["root"]) / "untracked.txt").write_text(
        "dirty", encoding="utf-8"
    )
    command = [
        *_common_command(
            clean_worker_checkout, fixture, output, model_state_fingerprint(model)
        ),
        "--variant",
        "source",
        "--checkpoint",
        str(checkpoint.resolve()),
        "--expected-checkpoint-sha256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "--model-config-json",
        str(model_config.resolve()),
    ]

    completed = _run(command, clean_worker_checkout)

    assert completed.returncode == 1
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["stage"] == "preflight"
    assert "completely clean worktree" in record["error"]["message"]


def test_cli_rejects_output_inside_source_checkout(
    tmp_path: Path, clean_worker_checkout: dict[str, object]
) -> None:
    fixture = tmp_path / "fixture.json"
    _write_fixture(fixture)
    output = Path(clean_worker_checkout["root"]) / "runtime.json"
    command = [
        *_common_command(clean_worker_checkout, fixture, output, "0" * 64),
        "--variant",
        "compact",
        "--artifact",
        str((tmp_path / "artifact").resolve()),
        "--expected-artifact-sha256",
        "0" * 64,
        "--expected-source-fingerprint",
        "0" * 64,
        "--expected-manifest-fingerprint",
        "0" * 64,
        "--expected-selection-fingerprint",
        "0" * 64,
    ]
    completed = _run(command, clean_worker_checkout)
    assert completed.returncode != 0
    assert "outside the source checkout" in completed.stderr
    assert not output.exists()


def test_cli_rejects_hardlinked_inputs(
    tmp_path: Path, clean_worker_checkout: dict[str, object]
) -> None:
    fixture = tmp_path / "fixture.json"
    artifact = tmp_path / "same-storage.tnlm3"
    output = tmp_path / "runtime.json"
    _write_fixture(fixture)
    try:
        os.link(fixture, artifact)
    except OSError as error:
        pytest.skip(f"hard links are unavailable: {error}")
    command = [
        *_common_command(clean_worker_checkout, fixture, output, "0" * 64),
        "--variant",
        "compact",
        "--artifact",
        str(artifact.resolve()),
        "--expected-artifact-sha256",
        hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "--expected-source-fingerprint",
        "0" * 64,
        "--expected-manifest-fingerprint",
        "0" * 64,
        "--expected-selection-fingerprint",
        "0" * 64,
    ]
    completed = _run(command, clean_worker_checkout)
    assert completed.returncode != 0
    assert "hard links to the same file" in completed.stderr
    assert not output.exists()
