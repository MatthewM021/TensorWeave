from __future__ import annotations

from pathlib import Path

import pytest

from tnlm_v3.binding import RoutedBindingModel
from tnlm_v3.factory import (
    build_binding_model,
    build_model,
    load_binding_experiment_config,
    load_forest_config,
)
from tnlm_v3.forest import ForestConfig, RoutedTensorLanguageModel
from tnlm_v3.routing import RoutingMode


CONFIG = Path(__file__).parents[1] / "configs" / "milestone1_cpu.yaml"
MILESTONE2 = Path(__file__).parents[1] / "configs" / "milestone2"


def test_reference_yaml_is_executable_and_length_independent():
    config = load_forest_config(CONFIG)
    assert config == ForestConfig(
        vocab_size=32,
        output_size=7,
        branches=4,
        d_model=8,
        cp_rank=4,
        scale_feature_dim=8,
        pad_token_id=0,
    )
    assert "length" not in config.canonical_json()
    assert load_forest_config(CONFIG).fingerprint() == config.fingerprint()
    model = build_model(CONFIG)
    assert isinstance(model, RoutedTensorLanguageModel)
    assert model.config == config


def test_loader_rejects_unknown_model_field(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """schema_version: 1
model:
  branches: 2
  d_model: 4
  cp_rank: 3
  vocab_size: 8
  max_length: 16
architecture:
  operator: scale_shared_cp_merge
  global_lane: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_length"):
        load_forest_config(path)


def test_config_loaders_reject_float_schema_version(tmp_path: Path):
    sources_and_loaders = (
        (CONFIG.read_text(encoding="utf-8"), load_forest_config),
        (
            (MILESTONE2 / "oracle_smoke.yaml").read_text(encoding="utf-8"),
            load_binding_experiment_config,
        ),
    )
    for index, (source, loader) in enumerate(sources_and_loaders):
        path = tmp_path / f"float-schema-{index}.yaml"
        path.write_text(
            source.replace("schema_version: 1", "schema_version: 1.0", 1),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="integer 1"):
            loader(path)


def test_build_model_rejects_unrelated_object():
    with pytest.raises(TypeError):
        build_model({"branches": 2})


def test_milestone2_modes_are_distinct_executable_paired_configs():
    paths = {
        mode: MILESTONE2 / f"{mode.value}_smoke.yaml" for mode in RoutingMode
    }
    configs = {mode: load_binding_experiment_config(path) for mode, path in paths.items()}
    assert set(configs) == set(RoutingMode)
    assert {config.condition for config in configs.values()} == set(RoutingMode)
    assert len({config.model_seed for config in configs.values()}) == 1
    assert len({config.data_seed for config in configs.values()}) == 1
    assert len({config.model.task.fingerprint() for config in configs.values()}) == 1
    assert configs[RoutingMode.CURRICULUM].model.curriculum_schedule is not None
    assert configs[RoutingMode.ORACLE].model.curriculum_schedule is None
    assert configs[RoutingMode.LATENT].model.curriculum_schedule is None
    for mode, path in paths.items():
        model = build_binding_model(path)
        assert isinstance(model, RoutedBindingModel)
        assert model.config.routing_mode is mode
        assert len(configs[mode].fingerprint()) == 64


def test_binding_config_loader_rejects_mode_schedule_blurring(tmp_path: Path):
    source = (MILESTONE2 / "oracle_smoke.yaml").read_text(encoding="utf-8")
    bad = source.replace("schedule: null", "schedule:\n    start_step: 0\n    end_step: 1")
    path = tmp_path / "blurred.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="non-curriculum"):
        load_binding_experiment_config(path)


def test_binding_experiment_rejects_coerced_numeric_scalars_and_late_schedule():
    curriculum = load_binding_experiment_config(
        MILESTONE2 / "curriculum_smoke.yaml"
    )
    values = dict(curriculum.__dict__)
    for name, value in (
        ("learning_rate", "0.01"),
        ("weight_decay", False),
        ("max_gradient_norm", True),
    ):
        changed = dict(values)
        changed[name] = value
        with pytest.raises(TypeError, match="real number"):
            type(curriculum)(**changed)

    changed = dict(values)
    changed["steps"] = 10
    with pytest.raises(ValueError, match="finish within"):
        type(curriculum)(**changed)
