from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import torch

from tnlm_v3.binding import (
    BindingArchitectureConfig,
    BindingModelConfig,
    RoutedBindingModel,
)
from tnlm_v3.compact_artifact import serialize_compact_binding_model
from tnlm_v3.data import (
    BindingModelInputs,
    BindingTaskConfig,
    collate_binding_episodes,
    generate_binding_episodes,
)
from tnlm_v3.model_export import export_compact_binding_model
from tnlm_v3.routing import RoutingMode
from tnlm_v3.truncation import (
    model_state_fingerprint,
    select_cp_rank_by_parameter_energy,
)


ROOT = Path(__file__).parents[1]
WORKER = ROOT / "scripts" / "replay_compact_artifact.py"
_HASH_DOMAIN = b"tnlm_v3.compact_replay_tensor.v1\x00"


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    if sys.byteorder != "little" and value.element_size() > 1:
        width = value.element_size()
        raw = b"".join(
            raw[index : index + width][::-1]
            for index in range(0, len(raw), width)
        )
    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    dtype = str(value.dtype).encode("ascii")
    digest.update(struct.pack("<Q", len(dtype)))
    digest.update(dtype)
    digest.update(struct.pack("<Q", value.ndim))
    for dimension in value.shape:
        digest.update(struct.pack("<Q", int(dimension)))
    digest.update(raw)
    return digest.hexdigest()


def _hashes(output) -> dict[str, str]:
    return {
        "routes": _tensor_hash(output.routes),
        "route_logits": _tensor_hash(output.route_logits),
        "route_probabilities": _tensor_hash(output.route_probabilities),
        "value_logits": _tensor_hash(output.value_logits),
        "forest_slots": _tensor_hash(output.forest_state.slots),
        "forest_occupied": _tensor_hash(output.forest_state.occupied),
        "forest_counts": _tensor_hash(output.forest_state.counts),
        "forest_valid_steps": _tensor_hash(output.forest_state.valid_steps),
        "router_prototypes": _tensor_hash(output.router_state.prototypes),
        "router_occupied": _tensor_hash(output.router_state.occupied),
        "router_ages": _tensor_hash(output.router_state.ages),
        "router_loads": _tensor_hash(output.router_state.loads),
        "router_global_state": _tensor_hash(output.router_state.global_state),
        "router_global_occupied": _tensor_hash(
            output.router_state.global_occupied
        ),
        "router_global_load": _tensor_hash(output.router_state.global_load),
        "router_valid_steps": _tensor_hash(output.router_state.valid_steps),
        **{
            f"diagnostic:{name}": _tensor_hash(value)
            for name, value in sorted(output.diagnostics.items())
        },
    }


def _fixture(inputs: BindingModelInputs) -> dict[str, object]:
    encoded: dict[str, object] = {}
    for name in (
        "token_ids",
        "event_kinds",
        "primary_key_ids",
        "secondary_key_ids",
        "arguments",
        "valid_mask",
    ):
        tensor = getattr(inputs, name).detach().cpu()
        encoded[name] = {
            "dtype": "bool" if tensor.dtype is torch.bool else "int64",
            "shape": list(tensor.shape),
            "values": tensor.tolist(),
        }
    return {"schema_version": 1, "inputs": encoded}


def _compact_model():
    torch.manual_seed(531)
    source = RoutedBindingModel(
        BindingModelConfig(
            task=BindingArchitectureConfig(4, 3, 2),
            d_model=6,
            cp_rank=7,
            router_hidden_dim=5,
            routing_mode=RoutingMode.LATENT,
            scale_feature_dim=4,
        )
    ).double().eval()
    selection = select_cp_rank_by_parameter_energy(source, target_rank=3)
    compact, manifest = export_compact_binding_model(source, selection)
    compact.eval()
    return compact, manifest, selection


def _inputs() -> BindingModelInputs:
    task = BindingTaskConfig(
        num_surface_keys=4,
        value_cardinality=3,
        branches=2,
        max_live_bindings=2,
        min_length=10,
        max_length=18,
        heldout_key_value_pairs=((0, 0),),
        global_distractor_probability=0.5,
    )
    return collate_binding_episodes(
        generate_binding_episodes(
            task,
            count=2,
            seed=981,
            split="eval",
            lengths=[17, 18],
        )
    ).inputs


def _run_worker(
    artifact: Path,
    fixture: Path,
    output: Path,
    *,
    artifact_sha256: str,
    source_fingerprint: str,
    manifest_fingerprint: str,
    selection_fingerprint: str,
):
    environment = _worker_environment()
    return subprocess.run(
        [
            sys.executable,
            "-s",
            "-B",
            str(WORKER.resolve()),
            "--artifact",
            str(artifact.resolve()),
            "--fixture",
            str(fixture.resolve()),
            "--output",
            str(output.resolve()),
            "--expected-artifact-sha256",
            artifact_sha256,
            "--expected-source-fingerprint",
            source_fingerprint,
            "--expected-manifest-fingerprint",
            manifest_fingerprint,
            "--expected-selection-fingerprint",
            selection_fingerprint,
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
    )


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    source_root = str((ROOT / "src").resolve())
    environment["PYTHONPATH"] = source_root
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def test_compact_artifact_replays_in_a_fresh_process(tmp_path: Path) -> None:
    compact, manifest, selection = _compact_model()
    inputs = _inputs()
    artifact_bytes = serialize_compact_binding_model(compact, manifest, selection)
    artifact = tmp_path / "compact.tnlm3"
    fixture = tmp_path / "fixture.json"
    output = tmp_path / "replay.json"
    artifact.write_bytes(artifact_bytes)
    fixture.write_text(
        json.dumps(_fixture(inputs), allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )

    with torch.no_grad():
        expected_streaming = _hashes(compact(inputs, implementation="streaming"))
        expected_parallel = _hashes(compact(inputs, implementation="parallel"))
    completed = _run_worker(
        artifact,
        fixture,
        output,
        artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        source_fingerprint=manifest.source_model_fingerprint,
        manifest_fingerprint=manifest.fingerprint(),
        selection_fingerprint=selection.fingerprint(),
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["process_id"] != os.getpid()
    assert record["artifact_sha256"] == hashlib.sha256(artifact_bytes).hexdigest()
    assert record["model_fingerprint"] == model_state_fingerprint(compact)
    assert record["source_model_fingerprint"] == (
        manifest.source_model_fingerprint
    )
    assert record["manifest_fingerprint"] == manifest.fingerprint()
    assert record["selection_fingerprint"] == selection.fingerprint()
    assert record["expected_provenance"] == {
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "source_model_fingerprint": manifest.source_model_fingerprint,
        "manifest_fingerprint": manifest.fingerprint(),
        "selection_fingerprint": selection.fingerprint(),
    }
    assert record["streaming"] == expected_streaming
    assert record["parallel"] == expected_parallel
    assert record["streaming_parallel_hashes_equal"] == (
        expected_streaming == expected_parallel
    )
    assert record["parameter_count"] == sum(
        parameter.numel() for parameter in compact.parameters()
    )
    assert record["raw_tensor_bytes"] == sum(
        tensor.numel() * tensor.element_size()
        for tensor in compact.state_dict().values()
    )
    assert record["environment"]["torch_threads"] == 1


def test_corrupt_artifact_writes_failure_record(tmp_path: Path) -> None:
    compact, manifest, selection = _compact_model()
    artifact_bytes = bytearray(
        serialize_compact_binding_model(compact, manifest, selection)
    )
    expected_artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    artifact_bytes[len(artifact_bytes) // 2] ^= 0x80
    artifact = tmp_path / "corrupt.tnlm3"
    fixture = tmp_path / "fixture.json"
    output = tmp_path / "failed.json"
    artifact.write_bytes(artifact_bytes)
    fixture.write_text(
        json.dumps(_fixture(_inputs()), allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )

    completed = _run_worker(
        artifact,
        fixture,
        output,
        artifact_sha256=expected_artifact_sha256,
        source_fingerprint=manifest.source_model_fingerprint,
        manifest_fingerprint=manifest.fingerprint(),
        selection_fingerprint=selection.fingerprint(),
    )

    assert completed.returncode != 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert set(record["error"]) == {"type", "message"}
    assert record["artifact_sha256"] == hashlib.sha256(artifact_bytes).hexdigest()
    assert record["fixture_sha256"] == hashlib.sha256(
        fixture.read_bytes()
    ).hexdigest()


def test_trusted_manifest_expectation_rejects_self_consistent_artifact(
    tmp_path: Path,
) -> None:
    compact, manifest, selection = _compact_model()
    artifact_bytes = serialize_compact_binding_model(compact, manifest, selection)
    artifact = tmp_path / "compact.tnlm3"
    fixture = tmp_path / "fixture.json"
    output = tmp_path / "failed.json"
    artifact.write_bytes(artifact_bytes)
    fixture.write_text(
        json.dumps(_fixture(_inputs()), allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    completed = _run_worker(
        artifact,
        fixture,
        output,
        artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        source_fingerprint=manifest.source_model_fingerprint,
        manifest_fingerprint="0" * 64,
        selection_fingerprint=selection.fingerprint(),
    )
    assert completed.returncode != 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert "manifest fingerprint" in record["error"]["message"]


def test_oversized_declared_fixture_is_rejected_safely(tmp_path: Path) -> None:
    compact, manifest, selection = _compact_model()
    artifact = tmp_path / "compact.tnlm3"
    fixture = tmp_path / "oversized.json"
    output = tmp_path / "failed.json"
    artifact_bytes = serialize_compact_binding_model(compact, manifest, selection)
    artifact.write_bytes(artifact_bytes)
    malformed = _fixture(_inputs())
    malformed["inputs"]["token_ids"]["shape"] = [1, 4097]
    fixture.write_text(
        json.dumps(malformed, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )

    completed = _run_worker(
        artifact,
        fixture,
        output,
        artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        source_fingerprint=manifest.source_model_fingerprint,
        manifest_fingerprint=manifest.fingerprint(),
        selection_fingerprint=selection.fingerprint(),
    )

    assert completed.returncode != 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert "time dimension" in record["error"]["message"]


def test_oversized_artifact_is_rejected_before_loading(tmp_path: Path) -> None:
    artifact = tmp_path / "oversized.tnlm3"
    fixture = tmp_path / "fixture.json"
    output = tmp_path / "failed.json"
    with artifact.open("wb") as handle:
        handle.truncate(64 * 1024 * 1024 + 1)
    fixture.write_text(
        json.dumps(_fixture(_inputs()), allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )

    completed = _run_worker(
        artifact,
        fixture,
        output,
        artifact_sha256="0" * 64,
        source_fingerprint="0" * 64,
        manifest_fingerprint="0" * 64,
        selection_fingerprint="0" * 64,
    )

    assert completed.returncode != 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert "artifact exceeds" in record["error"]["message"]


def test_worker_rejects_relative_paths() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-s",
            "-B",
            str(WORKER.resolve()),
            "--artifact",
            "relative.bin",
            "--fixture",
            "relative.json",
            "--output",
            "relative-output.json",
            "--expected-artifact-sha256",
            "0" * 64,
            "--expected-source-fingerprint",
            "0" * 64,
            "--expected-manifest-fingerprint",
            "0" * 64,
            "--expected-selection-fingerprint",
            "0" * 64,
        ],
        cwd=ROOT,
        env=_worker_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode != 0
    assert "absolute path" in completed.stderr
