from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import yaml

from tnlm_v3.baselines import (
    BindingBaselineKind,
    CachedCausalTransformerBindingBaseline,
    CachedTransformerBindingBaselineConfig,
    RecurrentBindingBaseline,
    RecurrentBindingBaselineConfig,
)
from tnlm_v3.binding import BindingArchitectureConfig
from tnlm_v3.causal_ttn import (
    CausalCompleteTreeBindingBaseline,
    CausalTreeBindingBaselineConfig,
)
from tnlm_v3.data import collate_binding_episodes, generate_binding_episodes
from tnlm_v3.factory import (
    BindingBaselineExperimentConfig,
    build_binding_baseline,
    load_binding_baseline_experiment_config,
)


CONFIGS = Path(__file__).parents[1] / "configs" / "milestone4"
GRU = CONFIGS / "gru_smoke.yaml"
TRANSFORMER = CONFIGS / "cached_transformer_smoke.yaml"
CAUSAL_TREE = CONFIGS / "causal_tree_smoke.yaml"
BASELINE_CONFIGS = (GRU, TRANSFORMER, CAUSAL_TREE)


def _write_document(tmp_path: Path, document: object, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _document(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    ("path", "kind", "config_type", "model_type"),
    [
        (GRU, BindingBaselineKind.GRU, RecurrentBindingBaselineConfig, RecurrentBindingBaseline),
        (
            TRANSFORMER,
            BindingBaselineKind.CACHED_TRANSFORMER,
            CachedTransformerBindingBaselineConfig,
            CachedCausalTransformerBindingBaseline,
        ),
        (
            CAUSAL_TREE,
            BindingBaselineKind.CAUSAL_TREE,
            CausalTreeBindingBaselineConfig,
            CausalCompleteTreeBindingBaseline,
        ),
    ],
)
def test_milestone4_yaml_is_strict_deterministic_and_executable(
    path: Path,
    kind: BindingBaselineKind,
    config_type: type,
    model_type: type,
) -> None:
    first = load_binding_baseline_experiment_config(path)
    second = load_binding_baseline_experiment_config(path)
    assert first == second
    assert first.kind is kind
    assert isinstance(first.model, config_type)
    assert first.fingerprint() == second.fingerprint()
    assert len(first.fingerprint()) == 64
    torch.manual_seed(first.model_seed)
    model_a = build_binding_baseline(first)
    torch.manual_seed(first.model_seed)
    model_b = build_binding_baseline(path)
    assert isinstance(model_a, model_type)
    assert isinstance(model_b, model_type)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            model_a.state_dict().values(), model_b.state_dict().values(), strict=True
        )
    )
    batch = collate_binding_episodes(
        generate_binding_episodes(
            first.task,
            count=first.episodes,
            seed=first.data_seed,
            split="train",
            lengths=[first.sequence_length] * first.episodes,
        )
    )
    output = model_a(batch.inputs)
    assert output.value_logits.shape == (
        first.episodes,
        first.sequence_length,
        first.task.value_cardinality,
    )
    assert bool(torch.isfinite(output.value_logits).all())


def test_baseline_model_receives_only_sanitized_architecture() -> None:
    for path in BASELINE_CONFIGS:
        config = load_binding_baseline_experiment_config(path)
        assert config.model.task == BindingArchitectureConfig.from_task(config.task)
        assert set(asdict(config.model.task)) == {
            "num_surface_keys",
            "value_cardinality",
            "branches",
        }
        canonical = config.model.canonical_json()
        for forbidden in (
            "heldout",
            "global_distractor_probability",
            "max_length",
            "min_length",
            "route",
            "target",
            "dependency",
            "generation",
        ):
            assert forbidden not in canonical


def test_kind_and_model_settings_are_bound_into_config_fingerprints() -> None:
    gru = load_binding_baseline_experiment_config(GRU)
    transformer = load_binding_baseline_experiment_config(TRANSFORMER)
    causal_tree = load_binding_baseline_experiment_config(CAUSAL_TREE)
    assert '"kind":"gru"' in gru.canonical_json()
    assert '"kind":"cached_transformer"' in transformer.canonical_json()
    assert '"kind":"causal_tree"' in causal_tree.canonical_json()
    assert len(
        {gru.fingerprint(), transformer.fingerprint(), causal_tree.fingerprint()}
    ) == 3
    assert len(
        {
            gru.model.fingerprint(),
            transformer.model.fingerprint(),
            causal_tree.model.fingerprint(),
        }
    ) == 3


@pytest.mark.parametrize("path", BASELINE_CONFIGS)
@pytest.mark.parametrize("version", [True, 1.0, "1", 2])
def test_loader_rejects_noninteger_or_unsupported_schema(
    tmp_path: Path, path: Path, version: object
) -> None:
    document = _document(path)
    document["schema_version"] = version
    with pytest.raises(ValueError, match="integer 1"):
        load_binding_baseline_experiment_config(_write_document(tmp_path, document))


@pytest.mark.parametrize("path", BASELINE_CONFIGS)
def test_loader_rejects_unknown_and_missing_root_fields(
    tmp_path: Path, path: Path
) -> None:
    for change, expected in (("unknown", "surprise"), ("missing", "data_seed")):
        document = _document(path)
        if change == "unknown":
            document["surprise"] = 1
        else:
            del document["data_seed"]
        with pytest.raises(ValueError, match=expected):
            load_binding_baseline_experiment_config(
                _write_document(tmp_path, document, f"{change}.yaml")
            )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            GRU.read_text(encoding="utf-8") + "\nkind: gru\n",
            "duplicate YAML key",
        ),
        (
            "schema_version: 1\nkind: &kind gru\ncopy: *kind\n",
            "aliases are forbidden",
        ),
    ],
)
def test_loader_rejects_duplicate_keys_and_aliases(
    tmp_path: Path, text: str, message: str
) -> None:
    path = tmp_path / "malformed.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_binding_baseline_experiment_config(path)


def test_loader_rejects_oversized_configuration(tmp_path: Path) -> None:
    path = tmp_path / "oversized.yaml"
    path.write_text(
        GRU.read_text(encoding="utf-8") + " " * (64 * 1024),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at most 65536 bytes"):
        load_binding_baseline_experiment_config(path)


@pytest.mark.parametrize(
    ("source", "wrong_field"),
    [
        (GRU, "num_heads"),
        (TRANSFORMER, "hidden_dim"),
        (CAUSAL_TREE, "num_heads"),
    ],
)
def test_loader_enforces_kind_specific_exact_model_fields(
    tmp_path: Path, source: Path, wrong_field: str
) -> None:
    document = _document(source)
    model = document["model"]
    assert isinstance(model, dict)
    model[wrong_field] = 2
    with pytest.raises(ValueError, match=wrong_field):
        load_binding_baseline_experiment_config(_write_document(tmp_path, document))


@pytest.mark.parametrize(
    ("source", "field"),
    [
        (GRU, "hidden_dim"),
        (TRANSFORMER, "num_heads"),
        (CAUSAL_TREE, "cp_rank"),
    ],
)
def test_loader_rejects_missing_model_fields(
    tmp_path: Path, source: Path, field: str
) -> None:
    document = _document(source)
    model = document["model"]
    assert isinstance(model, dict)
    del model[field]
    with pytest.raises(ValueError, match=field):
        load_binding_baseline_experiment_config(_write_document(tmp_path, document))


@pytest.mark.parametrize("source", BASELINE_CONFIGS)
@pytest.mark.parametrize("forbidden", ["route_labels", "heldout_key_value_pairs"])
def test_loader_rejects_routing_and_generator_metadata_in_model(
    tmp_path: Path, source: Path, forbidden: str
) -> None:
    document = _document(source)
    model = document["model"]
    assert isinstance(model, dict)
    model[forbidden] = []
    with pytest.raises(ValueError, match=forbidden):
        load_binding_baseline_experiment_config(_write_document(tmp_path, document))


@pytest.mark.parametrize("source", BASELINE_CONFIGS)
def test_loader_rejects_routing_condition_at_root(
    tmp_path: Path, source: Path
) -> None:
    document = _document(source)
    document["condition"] = "latent"
    with pytest.raises(ValueError, match="condition"):
        load_binding_baseline_experiment_config(_write_document(tmp_path, document))


@pytest.mark.parametrize(
    ("source", "field"),
    [
        (GRU, "num_layers"),
        (TRANSFORMER, "num_heads"),
        (CAUSAL_TREE, "cp_rank"),
    ],
)
def test_loader_rejects_boolean_model_dimensions(
    tmp_path: Path, source: Path, field: str
) -> None:
    document = _document(source)
    model = document["model"]
    assert isinstance(model, dict)
    model[field] = True
    with pytest.raises(ValueError, match="positive integer"):
        load_binding_baseline_experiment_config(_write_document(tmp_path, document))


@pytest.mark.parametrize("path", BASELINE_CONFIGS)
@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("root", "model_seed", True, "integer"),
        ("root", "data_seed", 1.5, "integer"),
        ("training", "episodes", False, "integer"),
        ("training", "sequence_length", 10.0, "integer"),
        ("training", "steps", "20", "integer"),
        ("training", "learning_rate", True, "real number"),
        ("training", "weight_decay", "0.0", "real number"),
        ("training", "max_gradient_norm", False, "real number"),
        ("training", "learning_rate", float("inf"), "finite"),
        ("training", "weight_decay", float("nan"), "finite"),
    ],
)
def test_loader_rejects_coerced_booleans_types_and_nonfinite_values(
    tmp_path: Path,
    path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    document = _document(path)
    target = document if section == "root" else document[section]
    assert isinstance(target, dict)
    target[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        load_binding_baseline_experiment_config(_write_document(tmp_path, document))


def test_experiment_rejects_kind_model_mismatch_and_unsanitized_task() -> None:
    gru = load_binding_baseline_experiment_config(GRU)
    transformer = load_binding_baseline_experiment_config(TRANSFORMER)
    values = dict(gru.__dict__)
    values["model"] = transformer.model
    with pytest.raises(TypeError, match="requires"):
        BindingBaselineExperimentConfig(**values)

    changed_task = type(gru.task)(
        **{**asdict(gru.task), "num_surface_keys": gru.task.num_surface_keys + 1}
    )
    values = dict(gru.__dict__)
    values["task"] = changed_task
    with pytest.raises(ValueError, match="architecture"):
        BindingBaselineExperimentConfig(**values)


def test_build_binding_baseline_rejects_unrelated_objects() -> None:
    with pytest.raises(TypeError, match="baseline"):
        build_binding_baseline(object())
