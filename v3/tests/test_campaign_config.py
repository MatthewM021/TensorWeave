from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import tnlm_v3.campaign_config as campaign_config_module
from tnlm_v3.campaign_config import (
    CampaignStage,
    Milestone4CampaignConfig,
    campaign_plan_sha256,
    load_milestone4_campaign_config,
    resolve_campaign_plan,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
COMMIT = "1" * 40
TREE = "2" * 40


def base_document(stage: str = "pilot") -> dict[str, object]:
    pairs: list[dict[str, object]] = [
        {
            "pair_id": "pair-1",
            "model_seed": 101,
            "train_seed": 201,
            "validation_seed": 301,
            "statistics_seed": 401,
        }
    ]
    if stage == "confirmatory":
        pairs = [
            {
                **pairs[0],
                "pair_id": f"pair-{index}",
                "model_seed": 100 + index,
                "train_seed": 200 + index,
                "validation_seed": 300 + index,
                "statistics_seed": 400 + index,
                "test_seed": 500 + index,
            }
            for index in range(1, 4)
        ]
    document: dict[str, object] = {
        "schema_version": 1,
        "campaign_id": f"m4-{stage}",
        "stage": stage,
        "description": f"Strict {stage} campaign",
        "claim_eligible": stage == "confirmatory",
        "implementation_policy": {
            "require_clean_head": True,
            "require_committed_inputs": True,
            "require_external_output_root": True,
            "deterministic_algorithms": True,
            "intraop_threads": 1,
            "interop_threads": 1,
        },
        "task": {
            "num_surface_keys": 5,
            "value_cardinality": 4,
            "branches": 3,
            "max_live_bindings": 3,
            "min_length": 10,
            "max_length": 2048,
            "heldout_key_value_pairs": [[0, 0]],
            "global_distractor_probability": 0.5,
        },
        "models": [
            {
                "model_id": "routed-source",
                "family": "routed",
                "role": "trainable_source",
                "routing_mode": "curriculum",
                "parent_model_id": None,
                "architecture": {
                    "d_model": 8,
                    "cp_rank": 4,
                    "router_hidden_dim": 8,
                    "scale_feature_dim": 4,
                    "straight_through_route_surrogate": True,
                    "curriculum_seed": 7,
                    "schedule": {
                        "start_step": 0,
                        "end_step": 10,
                        "start_probability": 1.0,
                        "end_probability": 0.0,
                    },
                },
                "export": None,
            },
            {
                "model_id": "routed-compact",
                "family": "routed",
                "role": "derived_compact",
                "routing_mode": "curriculum",
                "parent_model_id": "routed-source",
                "architecture": {
                    "d_model": 8,
                    "cp_rank": 2,
                    "router_hidden_dim": 8,
                    "scale_feature_dim": 4,
                    "straight_through_route_surrogate": True,
                    "curriculum_seed": 7,
                    "schedule": {
                        "start_step": 0,
                        "end_step": 10,
                        "start_probability": 1.0,
                        "end_probability": 0.0,
                    },
                },
                "export": {
                    "selection_method": "parameter_energy_v1",
                    "target_cp_rank": 2,
                },
            },
            {
                "model_id": "gru-control",
                "family": "gru",
                "role": "trainable_source",
                "routing_mode": None,
                "parent_model_id": None,
                "architecture": {
                    "d_model": 8,
                    "hidden_dim": 12,
                    "num_layers": 2,
                },
                "export": None,
            },
            {
                "model_id": "routed-oracle",
                "family": "routed",
                "role": "trainable_source",
                "routing_mode": "oracle",
                "parent_model_id": None,
                "architecture": {
                    "d_model": 8,
                    "cp_rank": 4,
                    "router_hidden_dim": 8,
                    "scale_feature_dim": 4,
                    "straight_through_route_surrogate": True,
                    "curriculum_seed": 0,
                    "schedule": None,
                },
                "export": None,
            },
            {
                "model_id": "routed-latent",
                "family": "routed",
                "role": "trainable_source",
                "routing_mode": "latent",
                "parent_model_id": None,
                "architecture": {
                    "d_model": 8,
                    "cp_rank": 4,
                    "router_hidden_dim": 8,
                    "scale_feature_dim": 4,
                    "straight_through_route_surrogate": True,
                    "curriculum_seed": 0,
                    "schedule": None,
                },
                "export": None,
            },
            {
                "model_id": "transformer-control",
                "family": "cached_transformer",
                "role": "trainable_source",
                "routing_mode": None,
                "parent_model_id": None,
                "architecture": {
                    "d_model": 8,
                    "num_heads": 2,
                    "num_layers": 2,
                    "ff_dim": 16,
                },
                "export": None,
            },
            {
                "model_id": "ttn-control",
                "family": "causal_ttn",
                "role": "trainable_source",
                "routing_mode": None,
                "parent_model_id": None,
                "architecture": {
                    "d_model": 8,
                    "cp_rank": 4,
                    "scale_feature_dim": 4,
                },
                "export": None,
            },
        ],
        "pairs": pairs,
        "data": {
            "generator_version": "binding-v1",
            "train": {
                "batch_size": 2,
                "length_schedule": [10, 12],
                "deterministic_step_stream": True,
            },
            "validation": {"lengths": [16, 32], "episodes_per_length": 8},
        },
        "training": {
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "max_gradient_norm": 1.0,
            "optimizer_steps": 10,
            "train_token_budget": 220,
            "checkpoint_interval": 5,
            "dtype": "float64",
            "device": "cpu",
        },
        "quality": {
            "primary_reference_model_id": "routed-source",
            "metric": "macro_length_query_accuracy",
            "max_absolute_drop": 0.02,
            "operational_rule": "mean_paired_delta_gte_negative_margin",
            "claim_rule": "one_sided_95pct_lower_bound_gt_negative_margin",
        },
        "statistics": {
            "paired_unit": "pair_id",
            "confidence_level": 0.95,
            "method": "paired_percentile_bootstrap_v1",
            "resamples": 2000,
        },
        "runtime": {
            "semantics": "batch1_full_document_streaming",
            "warmups": 2,
            "timed_iterations": 5,
            "process_repetitions": 3,
            "raw_samples_required": True,
            "condition": "operational_quality_gate",
        },
    }
    if stage == "screen":
        document["selection"] = {
            "candidates_by_family": {
                "routed": [
                    "routed-source",
                    "routed-latent",
                    "routed-compact",
                ],
                "gru": ["gru-control"],
                "cached_transformer": ["transformer-control"],
                "causal_ttn": ["ttn-control"],
            },
            "primary_metric": "macro_length_query_accuracy",
            "direction": "maximize",
            "tie_break": ["smaller_parameter_count", "lexical_candidate_id"],
        }
    if stage == "confirmatory":
        data = document["data"]
        assert isinstance(data, dict)
        data["test"] = {"lengths": [64, 128], "episodes_per_length": 8}
        data["scaling"] = {"lengths": [512, 1024, 2048], "episodes_per_length": 2}
        document["promotion"] = {
            "screen_campaign_id": "m4-screen",
            "record_path": "results/m4/screen-promotion.json",
            "record_sha256": SHA_A,
            "screen_manifest_sha256": SHA_B,
            "executable_bundle_sha256": SHA_C,
        }
    return document


def write_document(tmp_path: Path, document: object, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("pilot", CampaignStage.PILOT),
        ("screen", CampaignStage.SCREEN),
        ("confirmatory", CampaignStage.CONFIRMATORY),
    ],
)
def test_all_stage_documents_load_with_exact_presence(
    tmp_path: Path, stage: str, expected: CampaignStage
) -> None:
    config = load_milestone4_campaign_config(
        write_document(tmp_path, base_document(stage))
    )
    assert isinstance(config, Milestone4CampaignConfig)
    assert config.stage is expected
    assert (config.selection is not None) is (stage == "screen")
    assert (config.promotion is not None) is (stage == "confirmatory")
    assert (config.data.test is not None) is (stage == "confirmatory")
    assert all(
        (pair.test_seed is not None) is (stage == "confirmatory")
        for pair in config.pairs
    )
    assert config.fingerprint() == config.fingerprint()
    assert len(config.fingerprint()) == 64


@pytest.mark.parametrize(
    ("stage", "field"),
    [
        ("pilot", "selection"),
        ("pilot", "promotion"),
        ("screen", "promotion"),
        ("confirmatory", "selection"),
    ],
)
def test_stage_forbidden_root_fields_are_rejected_even_when_null(
    tmp_path: Path, stage: str, field: str
) -> None:
    document = base_document(stage)
    document[field] = None
    with pytest.raises(ValueError, match=field):
        load_milestone4_campaign_config(write_document(tmp_path, document))


@pytest.mark.parametrize("stage", ["pilot", "screen"])
@pytest.mark.parametrize("field", ["test", "scaling"])
def test_preconfirm_data_fields_must_be_absent(
    tmp_path: Path, stage: str, field: str
) -> None:
    document = base_document(stage)
    data = document["data"]
    assert isinstance(data, dict)
    data[field] = None
    with pytest.raises(ValueError, match=field):
        load_milestone4_campaign_config(write_document(tmp_path, document))


def test_preconfirm_test_seed_must_be_absent(tmp_path: Path) -> None:
    document = base_document("pilot")
    pairs = document["pairs"]
    assert isinstance(pairs, list) and isinstance(pairs[0], dict)
    pairs[0]["test_seed"] = None
    with pytest.raises(ValueError, match="test_seed"):
        load_milestone4_campaign_config(write_document(tmp_path, document))


def test_confirm_requires_all_confirm_fields_and_three_pairs(tmp_path: Path) -> None:
    document = base_document("confirmatory")
    document["pairs"] = document["pairs"][:2]  # type: ignore[index]
    with pytest.raises(ValueError, match="three pairs"):
        load_milestone4_campaign_config(write_document(tmp_path, document))


def test_literal_policies_and_plain_numeric_types_are_strict(tmp_path: Path) -> None:
    document = base_document()
    policy = document["implementation_policy"]
    assert isinstance(policy, dict)
    policy["require_clean_head"] = False
    with pytest.raises(ValueError, match="literal true"):
        load_milestone4_campaign_config(write_document(tmp_path, document, "false.yaml"))

    document = base_document()
    training = document["training"]
    assert isinstance(training, dict)
    training["optimizer_steps"] = True
    with pytest.raises(TypeError, match="integer"):
        load_milestone4_campaign_config(write_document(tmp_path, document, "bool.yaml"))


def test_exact_train_token_budget_and_mixed_schedule(tmp_path: Path) -> None:
    document = base_document()
    training = document["training"]
    assert isinstance(training, dict)
    training["train_token_budget"] = 219
    with pytest.raises(ValueError, match="token_budget"):
        load_milestone4_campaign_config(write_document(tmp_path, document, "budget.yaml"))

    document = base_document()
    data = document["data"]
    assert isinstance(data, dict) and isinstance(data["train"], dict)
    data["train"]["length_schedule"] = [10, 10]
    with pytest.raises(ValueError, match="mixed"):
        load_milestone4_campaign_config(write_document(tmp_path, document, "mixed.yaml"))


def test_model_family_routing_and_architecture_unions_are_exact(tmp_path: Path) -> None:
    document = base_document()
    models = document["models"]
    assert isinstance(models, list) and isinstance(models[2], dict)
    models[2]["routing_mode"] = "latent"
    with pytest.raises(ValueError, match="non-routed"):
        load_milestone4_campaign_config(write_document(tmp_path, document, "route.yaml"))

    document = base_document()
    models = document["models"]
    assert isinstance(models, list) and isinstance(models[2], dict)
    architecture = models[2]["architecture"]
    assert isinstance(architecture, dict)
    architecture["cp_rank"] = 4
    with pytest.raises(ValueError, match="architecture keys"):
        load_milestone4_campaign_config(write_document(tmp_path, document, "arch.yaml"))


@pytest.mark.parametrize("mutation", ["missing_parent", "wrong_rank", "architecture_drift"])
def test_compact_lineage_is_exact(
    tmp_path: Path, mutation: str
) -> None:
    document = base_document()
    models = document["models"]
    assert isinstance(models, list) and isinstance(models[1], dict)
    compact = models[1]
    if mutation == "missing_parent":
        compact["parent_model_id"] = "absent"
    elif mutation == "wrong_rank":
        export = compact["export"]
        assert isinstance(export, dict)
        export["target_cp_rank"] = 4
    else:
        architecture = compact["architecture"]
        assert isinstance(architecture, dict)
        architecture["d_model"] = 9
    with pytest.raises(ValueError, match="parent|rank|architecture"):
        load_milestone4_campaign_config(
            write_document(tmp_path, document, f"{mutation}.yaml")
        )


@pytest.mark.parametrize(
    ("removed_ids", "message"),
    [
        ({"routed-oracle"}, "oracle, curriculum, and latent"),
        ({"routed-latent"}, "oracle, curriculum, and latent"),
        ({"routed-source", "routed-compact"}, "oracle, curriculum, and latent"),
        ({"routed-compact"}, "compact child"),
        ({"gru-control"}, "GRU, cached Transformer, and causal TTN"),
        ({"transformer-control"}, "GRU, cached Transformer, and causal TTN"),
        ({"ttn-control"}, "GRU, cached Transformer, and causal TTN"),
    ],
)
def test_every_stage_requires_the_complete_control_matrix(
    tmp_path: Path, removed_ids: set[str], message: str
) -> None:
    for stage in ("pilot", "screen", "confirmatory"):
        document = base_document(stage)
        models = document["models"]
        assert isinstance(models, list)
        document["models"] = [
            model
            for model in models
            if isinstance(model, dict) and model["model_id"] not in removed_ids
        ]
        with pytest.raises(ValueError, match=message):
            load_milestone4_campaign_config(
                write_document(tmp_path, document, f"{stage}-matrix.yaml")
            )


@pytest.mark.parametrize(
    "candidate",
    [
        "routed-source",
        "routed-latent",
        "routed-compact",
        "gru-control",
        "transformer-control",
        "ttn-control",
    ],
)
def test_screen_selection_covers_every_non_oracle_source_and_compact_rank(
    tmp_path: Path, candidate: str
) -> None:
    document = base_document("screen")
    selection = document["selection"]
    assert isinstance(selection, dict)
    groups = selection["candidates_by_family"]
    assert isinstance(groups, dict)
    for values in groups.values():
        assert isinstance(values, list)
        if candidate in values:
            values.remove(candidate)
    with pytest.raises(
        ValueError, match="selection family/candidates|every non-oracle source"
    ):
        load_milestone4_campaign_config(
            write_document(tmp_path, document, f"missing-{candidate}.yaml")
        )


def test_screen_selection_excludes_oracle_reference_stratum(tmp_path: Path) -> None:
    document = base_document("screen")
    selection = document["selection"]
    assert isinstance(selection, dict)
    groups = selection["candidates_by_family"]
    assert isinstance(groups, dict) and isinstance(groups["routed"], list)
    groups["routed"].append("routed-oracle")
    with pytest.raises(ValueError, match="every non-oracle source"):
        load_milestone4_campaign_config(write_document(tmp_path, document))


def test_seed_streams_are_distinct_within_pairs_and_globally(tmp_path: Path) -> None:
    document = base_document()
    pairs = document["pairs"]
    assert isinstance(pairs, list) and isinstance(pairs[0], dict)
    pairs[0]["train_seed"] = pairs[0]["model_seed"]
    with pytest.raises(ValueError, match="within each pair"):
        load_milestone4_campaign_config(
            write_document(tmp_path, document, "within-pair.yaml")
        )

    document = base_document("confirmatory")
    pairs = document["pairs"]
    assert isinstance(pairs, list)
    assert isinstance(pairs[0], dict) and isinstance(pairs[1], dict)
    pairs[1]["train_seed"] = pairs[0]["validation_seed"]
    with pytest.raises(ValueError, match="reused"):
        load_milestone4_campaign_config(
            write_document(tmp_path, document, "global-reuse.yaml")
        )


@pytest.mark.parametrize(
    "reference_id", ["gru-control", "routed-compact", "routed-oracle"]
)
def test_quality_reference_is_the_autonomous_curriculum_routed_source(
    tmp_path: Path, reference_id: str
) -> None:
    document = base_document()
    quality = document["quality"]
    assert isinstance(quality, dict)
    quality["primary_reference_model_id"] = reference_id
    with pytest.raises(ValueError, match="routed curriculum trainable source"):
        load_milestone4_campaign_config(
            write_document(tmp_path, document, f"reference-{reference_id}.yaml")
        )


def test_confirm_seed_freshness_is_not_claimed_without_promotion_record(
    tmp_path: Path,
) -> None:
    screen = load_milestone4_campaign_config(
        write_document(tmp_path, base_document("screen"), "screen.yaml")
    )
    confirm = load_milestone4_campaign_config(
        write_document(
            tmp_path, base_document("confirmatory"), "confirmatory.yaml"
        )
    )
    assert screen.pairs[0].model_seed == confirm.pairs[0].model_seed


def test_duplicate_yaml_alias_and_oversize_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("stage: pilot\nstage: pilot\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_milestone4_campaign_config(path)

    path = tmp_path / "alias.yaml"
    path.write_text("stage: &value pilot\ncopy: *value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="aliases"):
        load_milestone4_campaign_config(path)

    path = tmp_path / "oversize.yaml"
    path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="1048576"):
        load_milestone4_campaign_config(path)


def test_yaml_node_and_depth_limits_and_recursion_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "nodes.yaml"
    path.write_text("items:\n" + "- 0\n" * 10_001, encoding="utf-8")
    with pytest.raises(ValueError, match="10000 nodes"):
        load_milestone4_campaign_config(path)

    path = tmp_path / "depth.yaml"
    path.write_text(
        "\n".join("  " * depth + f"level{depth}:" for depth in range(34))
        + "\n"
        + "  " * 34
        + "value: 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="nesting|depth"):
        load_milestone4_campaign_config(path)

    def recurse(*args: object, **kwargs: object) -> object:
        raise RecursionError("synthetic parser recursion")

    monkeypatch.setattr(yaml, "load", recurse)
    with pytest.raises(ValueError, match="depth limit"):
        load_milestone4_campaign_config(path)


def test_resolved_plan_is_paired_content_addressed_and_compact_is_lineage_only(
    tmp_path: Path,
) -> None:
    config = load_milestone4_campaign_config(
        write_document(tmp_path, base_document("confirmatory"))
    )
    plan = resolve_campaign_plan(config, COMMIT, TREE, SHA_A, SHA_C)

    assert len(plan) == len(config.models) * len(config.pairs) == 21
    assert len({run.run_id for run in plan}) == len(plan)
    assert all(len(run.run_id) == 64 for run in plan)
    by_key = {(run.model_id, run.pair_id): run for run in plan}
    for pair in config.pairs:
        source = by_key[("routed-source", pair.pair_id)]
        compact = by_key[("routed-compact", pair.pair_id)]
        assert source.training_required
        assert not compact.training_required
        assert source.training == config.training
        assert compact.training is None
        assert compact.parent_run_id == source.run_id
        assert compact.model_seed == source.model_seed == pair.model_seed
        assert compact.train_seed == source.train_seed == pair.train_seed
        assert compact.test_seed == source.test_seed == pair.test_seed
        assert source.architecture == next(
            model.architecture
            for model in config.models
            if model.model_id == source.model_id
        )
        assert source.task == compact.task == config.task
        assert source.data == compact.data == config.data
        assert source.raw_config_sha256 == compact.raw_config_sha256 == SHA_A
        assert source.semantic_config_sha256 == config.fingerprint()

    repeated = resolve_campaign_plan(config, COMMIT, TREE, SHA_A, SHA_C)
    assert repeated == plan
    assert campaign_plan_sha256(config, plan) == campaign_plan_sha256(
        config, tuple(reversed(plan))
    )
    assert len(campaign_plan_sha256(config, plan)) == 64


def test_run_identity_binds_every_provenance_hash(tmp_path: Path) -> None:
    config = load_milestone4_campaign_config(
        write_document(tmp_path, base_document())
    )
    first = resolve_campaign_plan(config, COMMIT, TREE, SHA_A, SHA_B)
    changed = resolve_campaign_plan(config, COMMIT, TREE, SHA_C, SHA_B)
    assert {run.run_id for run in first}.isdisjoint(run.run_id for run in changed)
    assert all(run.raw_config_sha256 == SHA_A for run in first)
    assert all(run.semantic_config_sha256 == config.fingerprint() for run in first)
    assert config.fingerprint() != SHA_A

    semantic_change = replace(config, description="Semantically changed campaign")
    semantic_runs = resolve_campaign_plan(
        semantic_change, COMMIT, TREE, SHA_A, SHA_B
    )
    assert {run.run_id for run in first}.isdisjoint(
        run.run_id for run in semantic_runs
    )


def test_run_and_plan_revalidate_identity_lineage_and_cardinality(
    tmp_path: Path,
) -> None:
    config = load_milestone4_campaign_config(
        write_document(tmp_path, base_document("confirmatory"))
    )
    plan = resolve_campaign_plan(config, COMMIT, TREE, SHA_A, SHA_C)

    with pytest.raises(ValueError, match="run_id"):
        replace(plan[0], raw_config_sha256=SHA_B)
    with pytest.raises(ValueError, match="complete model/pair product"):
        campaign_plan_sha256(config, plan[:-1])
    with pytest.raises(ValueError, match="run_id values must be unique"):
        campaign_plan_sha256(config, plan + (plan[0],))

    forged = copy.deepcopy(plan)
    compact = next(run for run in forged if run.role == "derived_compact")
    wrong_parent = next(
        run
        for run in forged
        if run.pair_id == compact.pair_id and run.model_id == "routed-oracle"
    )
    object.__setattr__(compact, "parent_run_id", wrong_parent.run_id)
    object.__setattr__(compact, "run_id", campaign_config_module._run_id(compact._identity_payload()))
    with pytest.raises(ValueError, match="parent lineage"):
        campaign_plan_sha256(config, forged)


def test_plan_rejects_whole_pair_and_added_model_axis_deletion(
    tmp_path: Path,
) -> None:
    confirm = load_milestone4_campaign_config(
        write_document(
            tmp_path, base_document("confirmatory"), "confirmatory-axis.yaml"
        )
    )
    confirm_plan = resolve_campaign_plan(confirm, COMMIT, TREE, SHA_A, SHA_C)
    without_pair = tuple(
        run for run in confirm_plan if run.pair_id != "pair-3"
    )
    with pytest.raises(ValueError, match="exact resolved config runs"):
        campaign_plan_sha256(confirm, without_pair)

    document = base_document()
    models = document["models"]
    assert isinstance(models, list) and isinstance(models[2], dict)
    extra = copy.deepcopy(models[2])
    extra["model_id"] = "gru-control-extra"
    models.append(extra)
    pilot = load_milestone4_campaign_config(
        write_document(tmp_path, document, "pilot-extra-model.yaml")
    )
    pilot_plan = resolve_campaign_plan(pilot, COMMIT, TREE, SHA_A, SHA_B)
    without_model = tuple(
        run for run in pilot_plan if run.model_id != "gru-control-extra"
    )
    with pytest.raises(ValueError, match="exact resolved config runs"):
        campaign_plan_sha256(pilot, without_model)


def test_plan_rejects_source_training_divergence_within_a_pair(
    tmp_path: Path,
) -> None:
    config = load_milestone4_campaign_config(
        write_document(tmp_path, base_document())
    )
    forged = copy.deepcopy(resolve_campaign_plan(config, COMMIT, TREE, SHA_A, SHA_B))
    target = next(
        run
        for run in forged
        if run.pair_id == "pair-1" and run.model_id == "gru-control"
    )
    assert target.training is not None
    object.__setattr__(
        target,
        "training",
        replace(target.training, learning_rate=target.training.learning_rate * 2),
    )
    object.__setattr__(
        target,
        "run_id",
        campaign_config_module._run_id(target._identity_payload()),
    )
    with pytest.raises(ValueError, match="exact config training spec"):
        campaign_plan_sha256(config, forged)


@pytest.mark.parametrize(
    "record_path",
    ["/absolute/record.json", "../record.json", "results\\record.json", "C:/record.json"],
)
def test_promotion_path_is_lexical_and_runner_checks_are_explicit(
    tmp_path: Path, record_path: str
) -> None:
    document = base_document("confirmatory")
    promotion = document["promotion"]
    assert isinstance(promotion, dict)
    promotion["record_path"] = record_path
    with pytest.raises(ValueError, match="relative POSIX"):
        load_milestone4_campaign_config(write_document(tmp_path, document))
    module_contract = campaign_config_module.__doc__ or ""
    assert "external output root" in module_contract
    assert "regular file" in module_contract
    assert "seed freshness" in module_contract


def test_confirm_promotion_bundle_must_match_resolved_bundle(tmp_path: Path) -> None:
    config = load_milestone4_campaign_config(
        write_document(tmp_path, base_document("confirmatory"))
    )
    with pytest.raises(ValueError, match="bundle"):
        resolve_campaign_plan(config, COMMIT, TREE, SHA_A, SHA_B)


def test_unknown_or_missing_nested_keys_are_rejected(tmp_path: Path) -> None:
    document = base_document()
    runtime = document["runtime"]
    assert isinstance(runtime, dict)
    runtime["surprise"] = 1
    with pytest.raises(ValueError, match="surprise"):
        load_milestone4_campaign_config(write_document(tmp_path, document, "unknown.yaml"))

    document = base_document()
    training = document["training"]
    assert isinstance(training, dict)
    del training["checkpoint_interval"]
    with pytest.raises(ValueError, match="checkpoint_interval"):
        load_milestone4_campaign_config(write_document(tmp_path, document, "missing.yaml"))
