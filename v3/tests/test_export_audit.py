from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
import tnlm_v3.export_audit as export_audit

from tnlm_v3.data import BindingModelInputs
from tnlm_v3.export_audit import (
    ExportAuditConfig,
    atomic_write_json,
    binding_inputs_sha256,
    forest_state_sha256,
    load_export_audit_config,
    tensor_sha256,
    validate_finite_json,
)
from tnlm_v3.forest import ForestState


CONFIG = Path(__file__).parents[1] / "configs" / "milestone3" / "export_audit.yaml"


def test_reference_export_audit_config_is_strict_and_executable() -> None:
    config = load_export_audit_config(CONFIG)
    assert config.target_cp_rank == 4
    assert config.calibration_lengths == (10, 12, 16, 18)
    assert config.evaluation_lengths == (15, 16, 31, 32, 63, 64)
    assert max(config.evaluation_lengths) > max(config.calibration_lengths)
    assert len(config.fingerprint()) == 64
    assert json.loads(config.canonical_json())["source_config"].startswith("v3/")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("schema_version", 1.0, ValueError),
        ("target_cp_rank", True, ValueError),
        ("source_config", "../escape.yaml", ValueError),
        ("source_config", "C:/absolute.yaml", ValueError),
        ("source_config", "C:drive-relative.yaml", ValueError),
        ("selection_method", "soft_gate", ValueError),
        ("calibration_split", "eval", ValueError),
        ("torch_threads", 0, ValueError),
        ("float32_rtol", "0.1", TypeError),
        ("float32_atol", float("nan"), ValueError),
        ("evaluation_lengths", (16, 15), ValueError),
        ("evaluation_lengths", (15, 4097), ValueError),
        ("timed_iterations", 1001, ValueError),
        ("float32_rtol", 0.02, ValueError),
    ),
)
def test_config_dataclass_rejects_malformed_values(field, value, error) -> None:
    config = load_export_audit_config(CONFIG)
    with pytest.raises(error):
        replace(config, **{field: value})


def _fake_repository(tmp_path: Path, monkeypatch) -> tuple[Path, str]:
    module = tmp_path / "v3" / "src" / "tnlm_v3" / "export_audit.py"
    module.parent.mkdir(parents=True)
    module.write_text("# test anchor\n", encoding="utf-8")
    monkeypatch.setattr(export_audit, "__file__", str(module))
    source_config = (
        Path(__file__).parents[1]
        / "configs"
        / "milestone2"
        / "curriculum_smoke.yaml"
    )
    copied_source = (
        tmp_path / "v3" / "configs" / "milestone2" / "curriculum_smoke.yaml"
    )
    copied_source.parent.mkdir(parents=True)
    copied_source.write_bytes(source_config.read_bytes())
    audit_dir = tmp_path / "v3" / "configs" / "milestone3"
    audit_dir.mkdir(parents=True)
    return audit_dir, CONFIG.read_text(encoding="utf-8")


def test_loader_rejects_unknown_float_duplicate_and_alias(
    tmp_path: Path, monkeypatch
) -> None:
    audit_dir, source = _fake_repository(tmp_path, monkeypatch)
    unknown = audit_dir / "unknown.yaml"
    unknown.write_text(source + "unexpected: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_export_audit_config(unknown)

    floating = audit_dir / "floating.yaml"
    floating.write_text(source.replace("schema_version: 1", "schema_version: 1.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="integer 1"):
        load_export_audit_config(floating)

    duplicate = audit_dir / "duplicate.yaml"
    duplicate.write_text(source + "schema_version: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_export_audit_config(duplicate)

    alias = audit_dir / "alias.yaml"
    alias.write_text(source.replace("schema_version: 1", "schema_version: &v 1").replace("target_cp_rank: 4", "target_cp_rank: *v"), encoding="utf-8")
    with pytest.raises(ValueError, match="aliases are forbidden"):
        load_export_audit_config(alias)


def test_loader_rejects_paths_outside_authoritative_repository(
    tmp_path: Path, monkeypatch
) -> None:
    module = tmp_path / "anchor" / "v3" / "src" / "tnlm_v3" / "export_audit.py"
    module.parent.mkdir(parents=True)
    module.write_text("# test anchor\n", encoding="utf-8")
    monkeypatch.setattr(export_audit, "__file__", str(module))
    with pytest.raises(ValueError, match="inside this repository"):
        load_export_audit_config(CONFIG)


def test_strict_finite_atomic_json_is_replaceable(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    first = atomic_write_json(path, {"status": "running", "values": [1, 2.5]})
    second = atomic_write_json(path, {"status": "passed", "values": []})
    assert first["status"] == "running"
    assert json.loads(path.read_text(encoding="utf-8")) == second
    with pytest.raises(ValueError, match="non-finite"):
        validate_finite_json({"bad": float("inf")})
    with pytest.raises(TypeError, match="unsupported"):
        validate_finite_json({"bad": torch.tensor(1)})
    before = path.read_bytes()
    with pytest.raises(TypeError, match="root must be an object"):
        atomic_write_json(path, [])  # type: ignore[arg-type]
    assert path.read_bytes() == before


def test_tensor_and_input_hashes_are_shape_dtype_and_value_sensitive() -> None:
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert tensor_sha256(value) == tensor_sha256(value.clone())
    assert tensor_sha256(value) != tensor_sha256(value.double())
    assert tensor_sha256(value) != tensor_sha256(value.reshape(3, 2))

    inputs = BindingModelInputs(
        token_ids=torch.tensor([[1, 2]], dtype=torch.int64),
        event_kinds=torch.tensor([[1, 2]], dtype=torch.int64),
        primary_key_ids=torch.tensor([[1, 0]], dtype=torch.int64),
        secondary_key_ids=torch.zeros(1, 2, dtype=torch.int64),
        arguments=torch.zeros(1, 2, dtype=torch.int64),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert binding_inputs_sha256(inputs) == binding_inputs_sha256(inputs.to("cpu"))
    changed = BindingModelInputs(
        **{**inputs.__dict__, "token_ids": torch.tensor([[1, 3]])}
    )
    assert binding_inputs_sha256(inputs) != binding_inputs_sha256(changed)


def test_forest_state_hash_is_complete_and_tensor_dtypes_are_bounded() -> None:
    state = ForestState(
        slots=torch.zeros(1, 2, 3, 4),
        occupied=torch.zeros(1, 2, 3, dtype=torch.bool),
        counts=torch.zeros(1, 2, dtype=torch.int64),
        valid_steps=torch.zeros(1, dtype=torch.int64),
    )
    same = ForestState(
        slots=state.slots.clone(),
        occupied=state.occupied.clone(),
        counts=state.counts.clone(),
        valid_steps=state.valid_steps.clone(),
    )
    assert forest_state_sha256(state) == forest_state_sha256(same)
    changed = ForestState(
        slots=state.slots.clone(),
        occupied=state.occupied.clone(),
        counts=state.counts.clone(),
        valid_steps=torch.ones(1, dtype=torch.int64),
    )
    assert forest_state_sha256(state) != forest_state_sha256(changed)
    with pytest.raises(TypeError, match="unsupported"):
        tensor_sha256(torch.ones(1, dtype=torch.complex64))
    with pytest.raises(ValueError, match="finite"):
        tensor_sha256(torch.tensor([float("nan")]))
