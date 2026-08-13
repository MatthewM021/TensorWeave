from __future__ import annotations

from pathlib import Path

import yaml

from tnlm_v3.campaign_config import (
    CampaignStage,
    campaign_plan_sha256,
    load_milestone4_campaign_config,
    resolve_campaign_plan,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "v3" / "configs" / "milestone4" / "pilot_smoke.yaml"
COMMIT = "1" * 40
TREE = "2" * 40
RAW_CONFIG_SHA256 = "3" * 64
EXECUTABLE_BUNDLE_SHA256 = "4" * 64


def test_committed_pilot_config_is_strict_and_nonclaiming() -> None:
    config = load_milestone4_campaign_config(CONFIG_PATH)
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config.stage is CampaignStage.PILOT
    assert config.campaign_id == "m4-pilot-smoke"
    assert not config.claim_eligible
    assert config.selection is None
    assert config.promotion is None
    assert set(raw) == {
        "schema_version",
        "campaign_id",
        "stage",
        "description",
        "claim_eligible",
        "implementation_policy",
        "task",
        "models",
        "pairs",
        "data",
        "training",
        "quality",
        "statistics",
        "runtime",
    }
    assert set(raw["data"]) == {"generator_version", "train", "validation"}
    assert "test" not in raw["data"]
    assert "scaling" not in raw["data"]
    assert all("test_seed" not in pair for pair in raw["pairs"])


def test_committed_pilot_covers_the_complete_lineage_once() -> None:
    config = load_milestone4_campaign_config(CONFIG_PATH)
    models = {model.model_id: model for model in config.models}

    assert set(models) == {
        "routed-oracle",
        "routed-source",
        "routed-latent",
        "routed-compact",
        "gru-control",
        "transformer-control",
        "ttn-control",
    }
    assert len(config.pairs) == 1
    assert models["routed-compact"].parent_model_id == "routed-source"
    assert models["routed-compact"].role == "derived_compact"
    assert config.training.train_token_budget == (
        config.training.optimizer_steps * sum(config.data.train.length_schedule)
    )


def test_committed_pilot_resolves_deterministically_with_compact_parent() -> None:
    config = load_milestone4_campaign_config(CONFIG_PATH)
    plan = resolve_campaign_plan(
        config,
        COMMIT,
        TREE,
        RAW_CONFIG_SHA256,
        EXECUTABLE_BUNDLE_SHA256,
    )
    repeated = resolve_campaign_plan(
        load_milestone4_campaign_config(CONFIG_PATH),
        COMMIT,
        TREE,
        RAW_CONFIG_SHA256,
        EXECUTABLE_BUNDLE_SHA256,
    )

    assert plan == repeated
    assert len(plan) == 7
    assert len({run.run_id for run in plan}) == 7
    by_model = {run.model_id: run for run in plan}
    assert not by_model["routed-compact"].training_required
    assert by_model["routed-compact"].training is None
    assert (
        by_model["routed-compact"].parent_run_id
        == by_model["routed-source"].run_id
    )
    assert len(campaign_plan_sha256(config, plan)) == 64
