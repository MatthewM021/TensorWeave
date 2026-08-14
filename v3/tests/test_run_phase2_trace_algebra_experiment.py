from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

from tnlm_v3.campaign_config import load_milestone4_campaign_config
from tnlm_v3.data import BindingEventKind, generate_binding_episode


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_phase2_trace_algebra_experiment.py"
    )
    spec = importlib.util.spec_from_file_location("phase2_trace_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "milestone4"
        / "validation_screen_v1.yaml"
    )
    return load_milestone4_campaign_config(config_path).task


def test_trusted_attester_binds_visible_semantics_and_outer_firewall() -> None:
    module = _load_script()
    task = _task()
    episode = generate_binding_episode(task, length=64, seed=17, split="train")
    trace = module.attest_episode_for_fold_controller(
        episode,
        task,
        forbidden_outer_cell=(0, 0),
    )
    assert trace.sequence.query_count > 0
    assert len(trace.attestation.pre_event_cells) == episode.length
    assert len(trace.attestation.post_event_cells) == episode.length
    assert len(trace.attestation.query_dependency_cells) == episode.length
    exposed = {
        cell
        for groups in (
            trace.attestation.pre_event_cells,
            trace.attestation.post_event_cells,
            trace.attestation.query_dependency_cells,
        )
        for row in groups
        for cell in row
    }
    assert (0, 0) not in exposed
    assert any(trace.attestation.query_dependency_cells)

    first_bind = next(
        event
        for event in trace.sequence.events
        if event.kind is BindingEventKind.BIND
    )
    with pytest.raises(ValueError, match="outer heldout"):
        module.attest_episode_for_fold_controller(
            episode,
            task,
            forbidden_outer_cell=(first_bind.primary_key, first_bind.argument),
        )


def test_forbidden_evaluation_metadata_does_not_change_trace_attestation() -> None:
    module = _load_script()
    task = _task()
    episode = generate_binding_episode(task, length=64, seed=29, split="train")
    original = module.attest_episode_for_fold_controller(
        episode,
        task,
        forbidden_outer_cell=(0, 0),
    )
    evaluation = episode.evaluation
    poisoned = replace(
        episode,
        evaluation=replace(
            evaluation,
            oracle_routes=torch.flip(evaluation.oracle_routes, dims=(0,)),
            dependency_parents=torch.full_like(evaluation.dependency_parents, 777),
            generation_ids=torch.full_like(evaluation.generation_ids, 888),
            live_binding_counts=torch.zeros_like(evaluation.live_binding_counts),
            heldout_combination_mask=torch.logical_not(
                evaluation.heldout_combination_mask
            ),
        ),
        document_id="poisoned-document-id",
        generation_seed=2**31 - 1,
        config_fingerprint="f" * 64,
    )
    repeated = module.attest_episode_for_fold_controller(
        poisoned,
        task,
        forbidden_outer_cell=(0, 0),
    )
    assert repeated == original


def test_trace_experiment_source_is_claim_scoped_and_self_hashable() -> None:
    module = _load_script()
    v3_root = Path(__file__).resolve().parents[1]
    hashes = module._source_hashes(v3_root)
    assert set(hashes) == {
        "scripts/run_phase2_trace_algebra_experiment.py",
        "src/tnlm_v3/algebra_discovery.py",
        "src/tnlm_v3/algebra_discovery_probes.py",
    }
    for relative, digest in hashes.items():
        assert len(digest) == 64
        assert digest == hashlib.sha256((v3_root / relative).read_bytes()).hexdigest()
    source = (v3_root / "scripts" / "run_phase2_trace_algebra_experiment.py").read_text(
        encoding="utf-8"
    )
    assert "not_representation_discovery" in source
    assert "confirmatory_claim_permitted\": False" in source
    assert "outer_identifier_received_by_coefficient_estimator\": False" in source
    assert "balanced_rotated_cell_control" in source
    assert "metamorphic_control" not in source


@pytest.mark.skipif(
    os.environ.get("TNLM_RUN_SLOW_DISCOVERY") != "1",
    reason="set TNLM_RUN_SLOW_DISCOVERY=1 for the multi-minute frozen run",
)
def test_full_trace_experiment_record_is_canonical_and_nonconfirmatory(
    tmp_path: Path,
) -> None:
    module = _load_script()
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "milestone4"
        / "validation_screen_v1.yaml"
    )
    record = module.build_experiment_record(config_path)
    digest = record.pop("record_sha256")
    assert digest == hashlib.sha256(module._canonical_bytes(record)).hexdigest()
    assert record["claims"]["transition_coefficients_learned_from_traces"]
    assert not record["claims"]["representation_discovery_performed"]
    assert not record["claims"]["confirmatory_claim_permitted"]
    assert record["sealed_actual_cell"]["evaluation"]["accuracy"] == 1.0
    assert record["sealed_actual_cell"]["evaluation"]["focal_accuracy"] == 1.0
    assert json.dumps(record, allow_nan=False)
