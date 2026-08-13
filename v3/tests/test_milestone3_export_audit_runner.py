from __future__ import annotations

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tnlm_v3.binding import (
    BindingArchitectureConfig,
    BindingModelConfig,
    RoutedBindingModel,
)
from tnlm_v3.data import (
    BindingTaskConfig,
    collate_binding_episodes,
    generate_binding_episodes,
)
from tnlm_v3.model_export import export_compact_binding_model
from tnlm_v3.factory import load_binding_experiment_config
from tnlm_v3.routing import CurriculumSchedule, RoutingMode
from tnlm_v3.truncation import (
    build_dense_selected_reference,
    select_cp_rank_by_parameter_energy,
)


ROOT = Path(__file__).parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_milestone3_export_audit.py"
SOURCE_CONFIG = ROOT / "configs" / "milestone2" / "curriculum_smoke.yaml"


def _runner_module():
    spec = spec_from_file_location("tnlm_v3_milestone3_export_audit", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _models():
    torch.manual_seed(731)
    source = RoutedBindingModel(
        BindingModelConfig(
            task=BindingArchitectureConfig(4, 3, 2),
            d_model=6,
            cp_rank=7,
            router_hidden_dim=5,
            routing_mode=RoutingMode.LATENT,
            scale_feature_dim=4,
        )
    ).eval()
    selection = select_cp_rank_by_parameter_energy(
        source, target_rank=3, calibration_fingerprint="a" * 64
    )
    dense = build_dense_selected_reference(source, selection).eval()
    compact, manifest = export_compact_binding_model(source, selection)
    return source, dense.eval(), compact.eval(), manifest, selection


def _batch(lengths=(15, 16, 31, 32)):
    task = BindingTaskConfig(
        num_surface_keys=4,
        value_cardinality=3,
        branches=2,
        max_live_bindings=2,
        min_length=10,
        max_length=max(lengths),
        heldout_key_value_pairs=((0, 0),),
        global_distractor_probability=0.5,
    )
    return collate_binding_episodes(
        generate_binding_episodes(
            task,
            count=len(lengths),
            seed=2003,
            split="eval",
            lengths=list(lengths),
        )
    )


def test_generated_paths_are_absolute_distinct_and_external(tmp_path: Path) -> None:
    runner = _runner_module()
    repository_root = RUNNER_PATH.resolve().parents[2]
    external = tmp_path.resolve()
    values = {
        "output": external / "audit.json",
        "artifact": external / "compact.tnlm3",
        "fixture": external / "fixture.json",
        "replay_output": external / "replay.json",
        "runtime_directory": external / "runtime",
    }
    assert runner._strict_external_paths(repository_root, values) == {
        key: value.resolve() for key, value in values.items()
    }
    with pytest.raises(ValueError, match="absolute"):
        runner._strict_external_paths(
            repository_root, {**values, "artifact": Path("relative.bin")}
        )
    with pytest.raises(ValueError, match="source checkout"):
        runner._strict_external_paths(
            repository_root,
            {**values, "artifact": repository_root / "forbidden.bin"},
        )
    with pytest.raises(ValueError, match="distinct"):
        runner._strict_external_paths(
            repository_root, {**values, "fixture": values["artifact"]}
        )


def test_preflight_failure_is_durable_without_requiring_a_clean_git_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_module()
    external = tmp_path.resolve()
    output = external / "audit.json"
    args = SimpleNamespace(
        audit_config=str((ROOT / "configs" / "milestone3" / "export_audit.yaml").resolve()),
        output=str(output),
        artifact=str(external / "compact.tnlm3"),
        fixture=str(external / "fixture.json"),
        replay_output=str(external / "replay.json"),
        runtime_directory=str(external / "runtime"),
        code_commit="0" * 40,
    )

    def fail(_commit: str):
        raise RuntimeError("deliberate clean-checkout failure")

    monkeypatch.setattr(runner, "_bind_to_clean_checkout", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        runner._execute(args)
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["requested_code_commit"] == "0" * 40
    assert record["error"] == {
        "type": "RuntimeError",
        "message": "deliberate clean-checkout failure",
    }


def test_batch_hash_binds_inputs_labels_lengths_and_metadata() -> None:
    runner = _runner_module()
    first = _batch((15, 16))
    second = _batch((15, 16))
    assert runner._batch_sha256(first) == runner._batch_sha256(second)
    changed = _batch((15, 17))
    assert runner._batch_sha256(first) != runner._batch_sha256(changed)
    identity = runner._batch_identity_record(first)
    assert identity["lengths"] == [15, 16]
    assert identity["valid_event_counts"] == [15, 16]
    assert identity["total_valid_events"] == 31
    assert len(identity["document_ids"]) == 2
    assert len(identity["generation_seeds"]) == 2


def test_dense_and_compact_all_output_parity_for_both_implementations() -> None:
    runner = _runner_module()
    _, dense, compact, _, _ = _models()
    batch = _batch((15, 16))
    with torch.no_grad():
        for implementation in ("streaming", "parallel"):
            expected = dense(batch.inputs, implementation=implementation)
            actual = compact(batch.inputs, implementation=implementation)
            comparison = runner._compare_outputs(
                expected, actual, rtol=2e-5, atol=2e-6
            )
            assert comparison["passed"], comparison
            assert comparison["diagnostic_keys_equal"]


def test_tensor_evidence_allows_matching_infinity_sentinels_but_rejects_nan() -> None:
    runner = _runner_module()
    expected = torch.tensor([1.0, float("-inf"), float("inf")])
    same = expected.clone()
    comparison = runner._tensor_difference(expected, same, rtol=0.0, atol=0.0)
    assert comparison["passed"]
    assert comparison["max_absolute_error"] == 0.0
    assert comparison["infinity_sign_patterns_equal"]
    record = runner._tensor_record(expected)
    assert record["positive_infinity_count"] == 1
    assert record["negative_infinity_count"] == 1
    assert record["nan_count"] == 0
    wrong_sign = torch.tensor([1.0, float("inf"), float("inf")])
    assert not runner._tensor_difference(
        expected, wrong_sign, rtol=0.0, atol=0.0
    )["passed"]
    with_nan = torch.tensor([1.0, float("nan"), float("inf")])
    assert not runner._tensor_difference(
        with_nan, with_nan.clone(), rtol=0.0, atol=0.0
    )["passed"]


def test_evaluation_records_queries_and_total_proxy_at_actual_merge_count() -> None:
    runner = _runner_module()
    source, dense, compact, _, _ = _models()
    batch = _batch((15, 16))
    audit = SimpleNamespace(float32_rtol=2e-5, float32_atol=2e-6)
    _, record = runner._evaluate_variants(
        {"source": source, "dense_selected": dense, "compact": compact},
        batch,
        audit,
    )
    for implementation in ("streaming", "parallel"):
        dense_work = record["dense_selected"][implementation]["structural_work"]
        compact_work = record["compact"][implementation]["structural_work"]
        assert dense_work["actual_executed_merge_count"] > 0
        assert compact_work["actual_executed_merge_count"] == dense_work[
            "actual_executed_merge_count"
        ]
        assert compact_work["total_operation_count_proxy"] < dense_work[
            "total_operation_count_proxy"
        ]
        assert record["compact"][implementation]["query_metrics"]["all"]["count"] > 0


def test_artifact_roundtrip_is_deterministic_and_trusted() -> None:
    runner = _runner_module()
    _, _, compact, manifest, selection = _models()
    artifact, record = runner._artifact_roundtrip(compact, manifest, selection)
    assert record["passed"], record
    assert record["serialized_twice_byte_identical"]
    assert record["trusted_roundtrip_byte_identical"]
    assert record["first_sha256"] == runner._sha256(artifact)
    assert record["first_sha256"] == record["second_sha256"]
    assert record["first_sha256"] == record["trusted_roundtrip_sha256"]


def test_cp_axis_record_proves_five_physical_slices_without_extra_rank_state() -> None:
    runner = _runner_module()
    source, _, compact, _, _ = _models()
    record = runner._cp_axis_record(source, compact)
    assert record["passed"], record
    assert set(record["five_cp_tensors"]) == set(runner._CP_STATE_NAMES)
    assert all(
        item["source_rank_extent"] == 7
        and item["compact_rank_extent"] == 3
        and item["physically_sliced"]
        for item in record["five_cp_tensors"].values()
    )
    assert record["compact_extra_state_keys"] == []
    assert record["rank_mask_or_original_rank_state_keys"] == []


def test_training_record_binds_before_after_optimizer_loss_and_guidance() -> None:
    runner = _runner_module()
    original = load_binding_experiment_config(SOURCE_CONFIG)
    schedule = CurriculumSchedule(0, 2, 1.0, 0.0)
    config = replace(
        original,
        model=replace(original.model, curriculum_schedule=schedule),
        steps=2,
    )
    model, record = runner._train_source(config)
    assert record["initial_model_fingerprint"] != record["trained_model_fingerprint"]
    assert record["trained_model_fingerprint"] == runner.model_state_fingerprint(model)
    assert record["optimizer"] == {
        "type": "torch.optim.AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    assert record["loss_config"] == {
        "query_weight": 1.0,
        "route_curriculum_weight": 1.0,
        "router_balance_weight": 0.05,
        "router_entropy_weight": 0.001,
        "route_persistence_weight": 0.05,
    }
    assert set(record["final_loss"]) == {
        "total",
        "query",
        "route_curriculum",
        "router_balance",
        "router_entropy",
        "route_persistence",
        "query_count",
        "route_supervision_count",
        "persistence_pair_count",
    }
    assert record["final_guidance"] == {
        "probability": 0.0,
        "guided_events": 0,
        "guided_fraction": 0.0,
    }


def test_forced_probe_crosses_actual_local_and_global_lane_carries() -> None:
    runner = _runner_module()
    _, dense, compact, _, _ = _models()
    probe = runner._forced_carry_probe(
        dense, compact, rtol=2e-5, atol=2e-6
    )
    assert probe["passed"], probe
    for lane in ("local", "global"):
        boundaries = probe[lane]["boundaries"]
        assert [item["pre_update_lane_count"] for item in boundaries] == [15, 31, 63]
        assert [item["post_update_lane_count"] for item in boundaries] == [16, 32, 64]
        assert [item["expected_merge_depth"] for item in boundaries] == [4, 5, 6]
        assert [item["dense_step_merge_depth"] for item in boundaries] == [4, 5, 6]
        assert [item["compact_step_merge_depth"] for item in boundaries] == [4, 5, 6]
        assert all(item["actual_carry_boundary"] for item in boundaries)
        assert all(item["occupancy_matches_binary_count"] for item in boundaries)
        assert all(
            item["dense_streaming_parallel_prefix_parity"]["passed"]
            and item["compact_streaming_parallel_prefix_parity"]["passed"]
            and item["dense_compact_parallel_prefix_parity"]["passed"]
            for item in boundaries
        )


def test_compact_midstream_forest_codec_resume_matches_uninterrupted_run() -> None:
    runner = _runner_module()
    _, _, compact, _, _ = _models()
    probe = runner._forest_resume_probe(
        compact, _batch((31, 32)), rtol=2e-5, atol=2e-6
    )
    assert probe["passed"], probe
    assert probe["prefix_state_sha256"] == probe["restored_state_sha256"]
    assert probe["forest_state_blob_bytes"] > 0


def test_forced_local_and_global_pre_carry_resume_is_deterministic() -> None:
    runner = _runner_module()
    _, _, compact, _, _ = _models()
    batch = _batch((32, 32))
    with torch.no_grad():
        events, _ = compact.encoder(batch.inputs)
    probe = runner._forced_forest_resume_probe(
        compact,
        events[batch.inputs.valid_mask],
        rtol=2e-5,
        atol=2e-6,
    )
    assert probe["passed"], probe
    for lane in ("local", "global"):
        cuts = probe[lane]["cuts"]
        assert [item["pre_carry_lane_update_count"] for item in cuts] == [15, 31, 63]
        assert [item["next_expected_merge_depth"] for item in cuts] == [4, 5, 6]
        assert all(item["serialization_deterministic"] for item in cuts)
        assert all(item["restored_reserializes_identically"] for item in cuts)
        assert [item["next_actual_merge_depth"] for item in cuts] == [4, 5, 6]
        assert all(item["uninterrupted_resumed_final_state"]["passed"] for item in cuts)


def test_replay_and_runtime_worker_commands_carry_trusted_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_module()
    captured = {}
    replay_output = tmp_path / "replay.json"
    replay_record = {
        "status": "passed",
        "model_fingerprint": "1" * 64,
        "streaming_parallel_hashes_equal": True,
    }
    replay_output.write_text(json.dumps(replay_record), encoding="utf-8")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    record = runner._run_replay_worker(
        ROOT / "scripts" / "replay_compact_artifact.py",
        repository_root=ROOT.parent,
        artifact=tmp_path / "compact.tnlm3",
        fixture=tmp_path / "fixture.json",
        output=replay_output,
        artifact_sha256="a" * 64,
        fixture_sha256="e" * 64,
        source_fingerprint="b" * 64,
        manifest_fingerprint="c" * 64,
        selection_fingerprint="d" * 64,
        code_commit="f" * 40,
        code_tree="0" * 40,
        worker_sha256="1" * 64,
    )
    assert record == replay_record
    command = captured["command"]
    assert "--expected-artifact-sha256" in command
    assert "--expected-source-fingerprint" in command
    assert "--expected-manifest-fingerprint" in command
    assert "--expected-selection-fingerprint" in command
    assert command[command.index("--expected-fixture-sha256") + 1] == "e" * 64
    assert command[command.index("--expected-code-commit") + 1] == "f" * 40
    assert command[command.index("--expected-code-tree") + 1] == "0" * 40
    assert command[command.index("--expected-worker-sha256") + 1] == "1" * 64

    runtime = runner._runtime_worker_command(
        ROOT / "scripts" / "measure_milestone3_runtime.py",
        variant="compact",
        fixture=tmp_path / "fixture.json",
        output=tmp_path / "runtime.json",
        fixture_sha256="a" * 64,
        model_fingerprint="b" * 64,
        code_commit="c" * 40,
        code_tree="d" * 40,
        warmup_iterations=3,
        timed_iterations=20,
        torch_threads=1,
        extra_arguments=("--artifact", str(tmp_path / "compact.tnlm3")),
    )
    assert runtime[runtime.index("--variant") + 1] == "compact"
    assert runtime[runtime.index("--expected-code-commit") + 1] == "c" * 40
    assert runtime[runtime.index("--expected-code-tree") + 1] == "d" * 40
    assert runtime[runtime.index("--torch-threads") + 1] == "1"


def test_runtime_matrix_isolates_every_variant_and_declared_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_module()
    source, dense, compact, manifest, selection = _models()
    evaluation = _batch((15, 16))
    artifact = tmp_path / "compact.tnlm3"
    artifact.write_bytes(b"artifact")
    commands = []

    def fake_worker(command, **kwargs):
        commands.append((command, kwargs))
        return {
            "status": "passed",
            "result_sha256": "a" * 64,
            "process_id": 1000 + len(commands),
            "median_elapsed_ns": 10.0,
            "sampled_peak_rss_bytes": 20,
            "incremental_sampled_peak_bytes": 2,
            "record": {"status": "passed"},
        }

    monkeypatch.setattr(runner, "_run_runtime_worker", fake_worker)
    audit = SimpleNamespace(
        warmup_iterations=1, timed_iterations=2, torch_threads=1
    )
    result = runner._runtime_matrix(
        ROOT / "scripts" / "measure_milestone3_runtime.py",
        repository_root=ROOT.parent,
        runtime_directory=tmp_path / "runtime",
        source=source,
        dense=dense,
        compact=compact,
        artifact=artifact,
        artifact_sha256=runner._sha256(b"artifact"),
        source_fingerprint=manifest.source_model_fingerprint,
        manifest_fingerprint=manifest.fingerprint(),
        selection_fingerprint=selection.fingerprint(),
        evaluation=evaluation,
        lengths=(15, 16),
        code_commit="b" * 40,
        code_tree="c" * 40,
        audit=audit,
    )
    assert result["all_workers_passed"]
    assert result["separate_worker_invocation_per_variant_and_length"]
    assert result["process_count"] == 6
    assert set(result["by_length"]) == {"15", "16"}
    assert all(
        set(entry["variants"]) == {"source", "dense_selected", "compact"}
        for entry in result["by_length"].values()
    )
    assert len(commands) == 6
    assert [command[0][command[0].index("--variant") + 1] for command in commands] == [
        "source",
        "dense_selected",
        "compact",
        "source",
        "dense_selected",
        "compact",
    ]
    assert all(
        set(kwargs["expected_output_evidence"])
        == {"routes_sha256", "value_logits_sha256"}
        and all(len(value) == 64 for value in kwargs["expected_output_evidence"].values())
        and kwargs["expected_worker_path"].name
        == "measure_milestone3_runtime.py"
        and len(kwargs["expected_worker_sha256"]) == 64
        and kwargs["expected_package_path"].name == "__init__.py"
        and len(kwargs["expected_package_sha256"]) == 64
        and kwargs["expected_warmup_iterations"] == 1
        and kwargs["expected_timed_iterations"] == 2
        and kwargs["expected_torch_threads"] == 1
        and kwargs["expected_rss_sample_period_ms"] == 1.0
        for _, kwargs in commands
    )


def test_runtime_parent_hard_gates_worker_protocol_and_clean_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_module()
    output = tmp_path / "runtime.json"
    worker = (ROOT / "scripts" / "measure_milestone3_runtime.py").resolve()
    package = Path(runner.tnlm_v3.__file__).resolve()
    expected_evidence = {
        "routes_sha256": "1" * 64,
        "value_logits_sha256": "2" * 64,
    }
    record = {
        "status": "passed",
        "variant": "compact",
        "process_id": 987654,
        "checkout": {
            "code_commit": "a" * 40,
            "code_tree": "b" * 40,
            "worker_file": str(worker),
            "worker_file_sha256": runner._sha256(worker.read_bytes()),
            "package_file": str(package),
            "package_file_sha256": runner._sha256(package.read_bytes()),
            "worktree_clean": True,
        },
        "fixture": {"sha256": "c" * 64},
        "model": {"model_fingerprint": "d" * 64},
        "measurement": {
            "warmup_iterations": 3,
            "timed_iterations": 2,
            "torch_threads": 1,
            "rss_sample_period_ms": 1.0,
            "loaded_rss_bytes": 100,
            "warmed_rss_bytes": 120,
            "sampled_peak_rss_bytes": 150,
            "incremental_sampled_peak_bytes": 30,
            "elapsed_ns_samples": [10, 12],
        },
        "output_evidence": expected_evidence,
    }

    def fake_run(*_args, **_kwargs):
        encoded = json.dumps(record, sort_keys=True)
        output.write_text(encoded, encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout=encoded, stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    arguments = dict(
        repository_root=ROOT.parent,
        output=output,
        expected_variant="compact",
        expected_fixture_sha256="c" * 64,
        expected_model_fingerprint="d" * 64,
        expected_output_evidence=expected_evidence,
        expected_code_commit="a" * 40,
        expected_code_tree="b" * 40,
        expected_worker_path=worker,
        expected_worker_sha256=runner._sha256(worker.read_bytes()),
        expected_package_path=package,
        expected_package_sha256=runner._sha256(package.read_bytes()),
        expected_warmup_iterations=3,
        expected_timed_iterations=2,
        expected_torch_threads=1,
        expected_rss_sample_period_ms=1.0,
    )
    accepted = runner._run_runtime_worker(["worker"], **arguments)
    assert accepted["status"] == "passed"

    record["checkout"]["worktree_clean"] = False
    with pytest.raises(RuntimeError, match="trusted provenance"):
        runner._run_runtime_worker(["worker"], **arguments)


def test_fixture_is_canonical_and_accepted_by_replay_schema() -> None:
    runner = _runner_module()
    fixture = runner._fixture_bytes(_batch((15, 16)).inputs)
    parsed = json.loads(fixture.decode("utf-8"))
    assert parsed["schema_version"] == 1
    assert set(parsed["inputs"]) == {
        "token_ids",
        "event_kinds",
        "primary_key_ids",
        "secondary_key_ids",
        "arguments",
        "valid_mask",
    }
    assert fixture == json.dumps(
        parsed,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
