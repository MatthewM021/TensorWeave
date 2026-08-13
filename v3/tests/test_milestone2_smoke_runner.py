from __future__ import annotations

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tnlm_v3.factory import load_binding_experiment_config
from tnlm_v3.routing import CurriculumSchedule


ROOT = Path(__file__).parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_milestone2_smoke.py"
CONFIG_PATH = ROOT / "configs" / "milestone2" / "oracle_smoke.yaml"
CURRICULUM_CONFIG_PATH = ROOT / "configs" / "milestone2" / "curriculum_smoke.yaml"


def _runner_module():
    spec = spec_from_file_location("tnlm_v3_milestone2_smoke", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_atomic_progress_record_is_strict_and_replaceable(tmp_path: Path) -> None:
    runner = _runner_module()
    output = tmp_path / "progress.json"
    runner._atomic_json(output, {"status": "in_progress", "conditions": []})
    runner._atomic_json(output, {"status": "passed", "conditions": ["oracle"]})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "conditions": ["oracle"],
        "status": "passed",
    }
    with pytest.raises(ValueError, match="non-finite"):
        runner._atomic_json(output, {"nested": [0.0, float("nan")]})
    with pytest.raises(TypeError, match="unsupported JSON value Tensor"):
        runner._atomic_json(output, {"nested": [torch.tensor(1)]})
    with pytest.raises(TypeError, match="non-string key"):
        runner._atomic_json(output, {1: "not JSON-strict"})


def test_git_commands_bind_safe_directory_to_normalized_checkout() -> None:
    runner = _runner_module()
    repository_root = RUNNER_PATH.resolve().parents[2]
    command = runner._git_command(repository_root, "rev-parse", "HEAD")
    assert command == [
        "git",
        "-c",
        f"safe.directory={repository_root.as_posix()}",
        "rev-parse",
        "HEAD",
    ]


def test_preflight_failure_is_written_after_external_output_is_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_module()
    output = tmp_path / "failed-preflight.json"

    def fail_checkout(_code_commit: str):
        raise RuntimeError("preflight failed deliberately")

    monkeypatch.setattr(runner, "_bind_to_clean_checkout", fail_checkout)
    args = SimpleNamespace(output=str(output), code_commit="0" * 40, config=[])
    with pytest.raises(RuntimeError, match="preflight failed deliberately"):
        runner._execute(args)

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["conditions"] == []
    assert record["requested_code_commit"] == "0" * 40
    assert record["error"] == {
        "type": "RuntimeError",
        "message": "preflight failed deliberately",
    }


def test_failure_record_uses_last_durable_json_snapshot(tmp_path: Path) -> None:
    runner = _runner_module()
    output = tmp_path / "durable-failure.json"
    durable = runner._atomic_json(
        output,
        {"status": "in_progress", "conditions": [{"condition": "oracle"}]},
    )
    contaminated = {
        "status": "in_progress",
        "conditions": [{"condition": "oracle"}, torch.tensor(2)],
    }
    with pytest.raises(TypeError, match="unsupported JSON value Tensor"):
        runner._atomic_json(output, contaminated)

    runner._atomic_json(
        output, runner._failure_record(durable, RuntimeError("condition failed"))
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["conditions"] == [{"condition": "oracle"}]
    assert record["error"]["message"] == "condition failed"


def test_oracle_gaps_are_explicit_for_train_and_heldout_slices() -> None:
    runner = _runner_module()

    def result(train: float, heldout: float, combination: float):
        return {
            "final": {"query": {"accuracy": train}},
            "heldout_evaluation": {
                "query": {"accuracy": heldout},
                "heldout_query": {"accuracy": combination},
            },
        }

    by_mode = {
        "oracle": result(1.0, 0.75, 0.5),
        "curriculum": result(0.9, 0.5, 0.25),
        "latent": result(0.8, 0.25, 0.0),
    }
    runner._attach_oracle_gaps(by_mode)

    assert by_mode["curriculum"]["oracle_gaps"] == {
        "fixed_train_query": {
            "oracle_accuracy": 1.0,
            "autonomous_accuracy": 0.9,
            "gap": pytest.approx(0.1),
        },
        "heldout_all_query": {
            "oracle_accuracy": 0.75,
            "autonomous_accuracy": 0.5,
            "gap": 0.25,
        },
        "heldout_combination_query": {
            "oracle_accuracy": 0.5,
            "autonomous_accuracy": 0.25,
            "gap": 0.25,
        },
    }
    assert by_mode["latent"]["oracle_gaps"]["heldout_all_query"]["gap"] == 0.5


def test_batch_and_initialization_hashes_are_reproducible() -> None:
    runner = _runner_module()
    config = load_binding_experiment_config(CONFIG_PATH)
    first_batch = runner._make_batch(config, "train")
    second_batch = runner._make_batch(config, "train")
    assert runner._batch_sha256(first_batch) == runner._batch_sha256(second_batch)

    torch.manual_seed(config.model_seed)
    first = runner.build_binding_model(config)
    torch.manual_seed(config.model_seed)
    second = runner.build_binding_model(config)
    assert runner._state_dict_sha256(first) == runner._state_dict_sha256(second)


def test_one_step_condition_record_contains_heldout_and_work_evidence() -> None:
    runner = _runner_module()
    config = replace(load_binding_experiment_config(CONFIG_PATH), steps=1)
    result = runner.run_condition(config, config_sha256="0" * 64)
    assert result["status"] == "completed"
    assert result["config_sha256"] == "0" * 64
    assert result["heldout_evaluation"]["heldout_query"]["query_count"] > 0
    assert result["structural_work"]["router_branch_score_work_proxy"] == (
        config.episodes * config.sequence_length * config.task.branches
    )
    load = result["final"]["router_load"]
    assert load["valid_event_count"] == (
        load["local_event_count"]
        + load["global_event_count"]
        + load["null_event_count"]
    )


def test_curriculum_condition_records_declared_schedule_and_realized_trace() -> None:
    runner = _runner_module()
    original = load_binding_experiment_config(CURRICULUM_CONFIG_PATH)
    schedule = CurriculumSchedule(0, 2, 1.0, 0.0)
    model = replace(original.model, curriculum_schedule=schedule)
    config = replace(original, model=model, steps=2)

    result = runner.run_condition(config, config_sha256="1" * 64)

    curriculum = result["curriculum"]
    assert curriculum["declared_schedule"] == schedule.as_dict()
    trace = curriculum["realized_trace"]
    assert len(trace) == 2
    assert [entry["step"] for entry in trace] == [1, 2]
    assert [entry["guidance_probability"] for entry in trace] == [0.5, 0.0]
    assert set(trace[0]) == {
        "step",
        "guidance_probability",
        "guided_events",
        "guided_fraction",
        "route_supervision_count",
    }
    first = trace[0]
    assert first["route_supervision_count"] == first["guided_events"]
    assert first["guided_fraction"] == pytest.approx(
        first["guided_events"] / (config.episodes * config.sequence_length)
    )
    assert trace[1]["guided_events"] == 0
    assert trace[1]["guided_fraction"] == 0.0
    assert trace[1]["route_supervision_count"] == 0
