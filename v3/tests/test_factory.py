from __future__ import annotations

from pathlib import Path

import pytest

from tnlm_v3.factory import build_model, load_forest_config
from tnlm_v3.forest import ForestConfig, RoutedTensorLanguageModel


CONFIG = Path(__file__).parents[1] / "configs" / "milestone1_cpu.yaml"


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


def test_build_model_rejects_unrelated_object():
    with pytest.raises(TypeError):
        build_model({"branches": 2})
