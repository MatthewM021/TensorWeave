from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import types

import pytest
import torch
import yaml

from tnlm_v3.benchmark import (
    document_local_route_consistency,
    per_document_route_recovery,
    summarize_router_load,
)
from tnlm_v3.campaign_checkpoint import (
    campaign_checkpoint_contract,
    deserialize_campaign_checkpoint,
)
from tnlm_v3.compact_artifact import (
    deserialize_compact_binding_model,
    serialize_compact_binding_model,
)
from tnlm_v3.routing import NULL_ROUTE


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "v3" / "scripts" / "run_milestone4_pilot_worker.py"
RUNNER = ROOT / "v3" / "scripts" / "run_milestone4_pilot.py"
CONFIG = ROOT / "v3" / "configs" / "milestone4" / "pilot_smoke.yaml"
COMMIT = "1" * 40
TREE = "2" * 40
BUNDLE = "3" * 64


def _load_worker():  # type: ignore[no-untyped-def]
    name = "milestone4_pilot_worker_under_test"
    specification = importlib.util.spec_from_file_location(name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def worker():  # type: ignore[no-untyped-def]
    return _load_worker()


@dataclass(frozen=True)
class CampaignFixture:
    config: object
    plan: tuple[object, ...]
    source: object
    compact: object
    source_context: object
    compact_context: object
    raw_config_sha256: str


@pytest.fixture(scope="module")
def campaign(worker) -> CampaignFixture:  # type: ignore[no-untyped-def]
    raw_sha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    config = worker.load_milestone4_campaign_config(CONFIG)
    plan = worker.resolve_campaign_plan(config, COMMIT, TREE, raw_sha, BUNDLE)
    source = next(item for item in plan if item.model_id == "routed-source")
    compact = next(item for item in plan if item.model_id == "routed-compact")
    return CampaignFixture(
        config=config,
        plan=plan,
        source=source,
        compact=compact,
        source_context=worker.CampaignRunContext(config=config, run=source),
        compact_context=worker.CampaignRunContext(config=config, run=compact),
        raw_config_sha256=raw_sha,
    )


def _provenance() -> dict[str, str]:
    return {
        "code_commit": COMMIT,
        "code_tree": TREE,
        "raw_config_sha256": "4" * 64,
        "semantic_config_sha256": "5" * 64,
        "executable_bundle_sha256": BUNDLE,
        "parent_runner_sha256": "6" * 64,
        "worker_sha256": "7" * 64,
        "package_tree_sha256": "8" * 64,
    }


def _real_screen_config(worker, tmp_path: Path):  # type: ignore[no-untyped-def]
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert type(document) is dict
    document["campaign_id"] = "m4-screen-test"
    document["stage"] = "screen"
    document["description"] = "Validation-only screen worker acceptance fixture."
    document["pairs"].extend(
        [
            {
                "pair_id": "development-pair-2",
                "model_seed": 2101,
                "train_seed": 2201,
                "validation_seed": 2301,
                "statistics_seed": 2401,
            },
            {
                "pair_id": "development-pair-3",
                "model_seed": 3101,
                "train_seed": 3201,
                "validation_seed": 3301,
                "statistics_seed": 3401,
            },
        ]
    )
    document["statistics"] = {
        "paired_unit": "pair_id",
        "confidence_level": 0.95,
        "method": "paired_exact_empirical_bootstrap_n3_v1",
        "resamples": 27,
    }
    document["screen_gates"] = {
        "chance": 0.25,
        "oracle_mean_min": 0.80,
        "reference_mean_min": 0.70,
        "reference_pair_min": 0.60,
        "reference_heldout_mean_min": 0.60,
        "reference_longest_length_mean_min": 0.60,
        "chance_corrected_oracle_recovery_min": 0.80,
        "oracle_max_drop_vs_reference": 0.02,
        "candidate_pair_max_drop": 0.10,
        "require_positive_query_partitions": True,
        "require_no_route_collapse": True,
    }
    document["selection"] = {
        "candidates_by_stratum": {
            "routed_latent": ["routed-latent"],
            "routed_compact_rank": ["routed-compact"],
            "gru": ["gru-control"],
            "cached_transformer": ["transformer-control"],
            "causal_ttn": ["ttn-control"],
        },
        "primary_metric": "macro_length_query_accuracy",
        "direction": "maximize",
        "standard_tie_break": [
            "smaller_parameter_count",
            "lexical_candidate_id",
        ],
        "compact_tie_break": [
            "smaller_parameter_count",
            "smaller_target_cp_rank",
            "higher_primary_metric",
            "lexical_candidate_id",
        ],
    }
    path = tmp_path / "screen.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return worker.load_milestone4_campaign_config(path)


def _assert_routing_schema(routing: object) -> None:
    assert type(routing) is dict
    assert set(routing) == {
        "route_recovery",
        "route_consistency",
        "router_load",
    }
    recovery = routing["route_recovery"]
    assert type(recovery) is dict
    assert set(recovery) == {
        "correct",
        "local_event_count",
        "accuracy",
        "macro_accuracy",
        "document_count",
    }
    for name in ("correct", "local_event_count", "document_count"):
        assert type(recovery[name]) is int and recovery[name] >= 0
    for name in ("accuracy", "macro_accuracy"):
        assert type(recovery[name]) is float and math.isfinite(recovery[name])

    consistency = routing["route_consistency"]
    assert type(consistency) is dict
    assert set(consistency) == {
        "consistent_events",
        "local_event_count",
        "consistency",
        "group_count",
        "fully_consistent_groups",
    }
    for name in (
        "consistent_events",
        "local_event_count",
        "group_count",
        "fully_consistent_groups",
    ):
        assert type(consistency[name]) is int and consistency[name] >= 0
    assert type(consistency["consistency"]) is float
    assert math.isfinite(consistency["consistency"])

    load = routing["router_load"]
    assert type(load) is dict
    assert set(load) == {
        "branch_counts",
        "branch_fractions",
        "local_event_count",
        "global_event_count",
        "null_event_count",
        "valid_event_count",
        "global_event_fraction",
        "null_event_fraction",
        "active_branches",
        "collapsed",
        "document_count",
        "collapsed_document_count",
        "collapsed_document_fraction",
        "mean_active_branches_per_document",
        "max_load_fraction",
        "load_entropy",
        "normalized_load_entropy",
        "mean_assignment_entropy",
        "normalized_mean_assignment_entropy",
        "assignment_entropy_count",
    }
    assert type(load["branch_counts"]) is list
    assert all(type(value) is int and value >= 0 for value in load["branch_counts"])
    assert type(load["branch_fractions"]) is list
    assert all(
        type(value) is float and math.isfinite(value)
        for value in load["branch_fractions"]
    )
    for name in (
        "local_event_count",
        "global_event_count",
        "null_event_count",
        "valid_event_count",
        "active_branches",
        "document_count",
        "collapsed_document_count",
        "assignment_entropy_count",
    ):
        assert type(load[name]) is int and load[name] >= 0
    assert type(load["collapsed"]) is bool
    for name in (
        "global_event_fraction",
        "null_event_fraction",
        "collapsed_document_fraction",
        "mean_active_branches_per_document",
        "max_load_fraction",
        "load_entropy",
        "normalized_load_entropy",
        "mean_assignment_entropy",
        "normalized_mean_assignment_entropy",
    ):
        assert type(load[name]) is float and math.isfinite(load[name])
    assert "documents" not in recovery
    assert "groups" not in consistency


@dataclass(frozen=True)
class SourceFixture:
    output_root: Path
    attempt_dir: Path
    artifacts: dict[str, object]
    metrics: dict[str, object]
    stream: dict[str, object]


@pytest.fixture(scope="module")
def fresh_source(
    worker, campaign: CampaignFixture, tmp_path_factory: pytest.TempPathFactory
) -> SourceFixture:  # type: ignore[no-untyped-def]
    output = tmp_path_factory.mktemp("m4-worker-source")
    attempt = output / "artifacts" / campaign.source.run_id / "attempt-000001"
    attempt.mkdir(parents=True)
    artifacts, metrics, stream = worker._source_run(
        campaign.source_context,
        attempt,
        output,
        None,
        None,
        _provenance(),
        1,
    )
    return SourceFixture(output, attempt, artifacts, metrics, stream)


@pytest.fixture(scope="module")
def resumed_source(
    worker,
    campaign: CampaignFixture,
    fresh_source: SourceFixture,
) -> SourceFixture:  # type: ignore[no-untyped-def]
    first = fresh_source.artifacts["checkpoints"][0]  # type: ignore[index]
    checkpoint = fresh_source.output_root / first["path"]
    attempt = (
        fresh_source.output_root
        / "artifacts"
        / campaign.source.run_id
        / "attempt-000002"
    )
    attempt.mkdir(parents=True)
    artifacts, metrics, stream = worker._source_run(
        campaign.source_context,
        attempt,
        fresh_source.output_root,
        checkpoint,
        first["sha256"],
        _provenance(),
        2,
    )
    return SourceFixture(
        fresh_source.output_root, attempt, artifacts, metrics, stream
    )


@dataclass(frozen=True)
class CompactFixture:
    attempt_dir: Path
    parent_result_path: Path
    artifacts: dict[str, object]
    metrics: dict[str, object]
    stream: dict[str, object]


def _successful_parent_document(
    worker, campaign: CampaignFixture, source: SourceFixture
) -> dict[str, object]:  # type: ignore[no-untyped-def]
    plan_sha = worker.campaign_plan_sha256(campaign.config, campaign.plan)
    return {
        "schema_version": 1,
        "status": "success",
        "run_id": campaign.source.run_id,
        "run_sha256": worker._run_hash(campaign.source),
        "model_id": campaign.source.model_id,
        "pair_id": campaign.source.pair_id,
        "family": campaign.source.family,
        "role": campaign.source.role,
        "attempt_number": 1,
        "plan_sha256": plan_sha,
        "provenance": _provenance(),
        "artifacts": source.artifacts,
        "metrics": source.metrics,
        "stream": source.stream,
        "error": None,
    }


@pytest.fixture(scope="module")
def compact_result(
    worker,
    campaign: CampaignFixture,
    fresh_source: SourceFixture,
) -> CompactFixture:  # type: ignore[no-untyped-def]
    parent_document = _successful_parent_document(worker, campaign, fresh_source)
    parent_path = fresh_source.attempt_dir / "result.json"
    worker._write_atomic(
        parent_path, worker._canonical(parent_document), immutable=True
    )
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    final = fresh_source.artifacts["final_checkpoint"]
    checkpoint_path = fresh_source.output_root / final["path"]  # type: ignore[index]
    attempt = (
        fresh_source.output_root
        / "artifacts"
        / campaign.compact.run_id
        / "attempt-000001"
    )
    attempt.mkdir(parents=True)
    artifacts, metrics, stream = worker._compact_run(
        campaign.compact_context,
        campaign.config,
        campaign.plan,
        attempt,
        fresh_source.output_root,
        parent_path,
        parent_sha,
        checkpoint_path,
        final["sha256"],  # type: ignore[index]
    )
    return CompactFixture(attempt, parent_path, artifacts, metrics, stream)


def test_strict_json_rejects_duplicates_nonfinite_and_noncanonical(worker) -> None:  # type: ignore[no-untyped-def]
    valid = worker._canonical({"a": [1, True, None], "z": "value"})
    assert worker._strict_json(valid, "fixture") == {
        "a": [1, True, None],
        "z": "value",
    }
    for raw in (
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{ "a":1}',
        b'{"a":1}\n',
        b'[1,2,3]',
        b'\xff',
    ):
        with pytest.raises(worker.PilotWorkerError):
            worker._strict_json(raw, "fixture")


def test_worker_scope_accepts_real_pilot_and_screen_without_claim_streams(
    worker,
    campaign: CampaignFixture,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    worker._validate_worker_scope(campaign.config)
    screen = _real_screen_config(worker, tmp_path)
    assert screen.stage is worker.CampaignStage.SCREEN
    assert screen.claim_eligible is False
    assert screen.data.test is None
    assert screen.data.scaling is None
    worker._validate_worker_scope(screen)

    unsafe = types.SimpleNamespace(
        stage=worker.CampaignStage.SCREEN,
        claim_eligible=False,
        data=types.SimpleNamespace(test=object(), scaling=None),
    )
    with pytest.raises(worker.PilotWorkerError, match="without test/scaling"):
        worker._validate_worker_scope(unsafe)


def test_sha_prefix_and_checkpoint_schedule_are_exact(
    worker, campaign: CampaignFixture
) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(worker.PilotWorkerError):
        worker._sha("A" * 64, "digest")
    with pytest.raises(worker.PilotWorkerError):
        worker._sha("0" * 63, "digest")
    assert worker._checkpoint_cursors(campaign.source) == (4, 8, 12)
    assert worker._checkpoint_cursors(campaign.compact) == ()
    initial = hashlib.sha256(worker._STREAM_DOMAIN).hexdigest()
    batch_hash = "9" * 64
    expected = hashlib.sha256(
        bytes.fromhex(initial)
        + (0).to_bytes(8, "little", signed=False)
        + bytes.fromhex(batch_hash)
    ).hexdigest()
    assert worker._prefix(initial, 0, batch_hash) == expected


def test_private_paths_reject_links_and_artifacts_stay_external(
    worker, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "external"
    output.mkdir()
    regular = output / "artifact.bin"
    regular.write_bytes(b"artifact")
    reference = worker._artifact(regular, output)
    assert reference == {
        "path": "artifact.bin",
        "size_bytes": 8,
        "sha256": hashlib.sha256(b"artifact").hexdigest(),
    }
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError):
        worker._artifact(outside, output)
    linked = output / "linked.bin"
    try:
        os.link(regular, linked)
    except OSError as error:
        pytest.skip(f"hard links unavailable: {error}")
    with pytest.raises(worker.PilotWorkerError, match="private regular file"):
        worker._read_private(regular, maximum=100, name="artifact")


def test_inventory_binds_sorted_package_and_executable_inputs(
    worker, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    paths = {
        "v3/src/tnlm_v3/z.py": b"z",
        "v3/src/tnlm_v3/a.py": b"alpha",
        CONFIG.relative_to(ROOT).as_posix(): b"config",
        worker._RUNNER_RELATIVE: b"runner",
        worker._WORKER_RELATIVE: b"worker",
    }
    index = {
        path: ("100644", hashlib.sha1(raw).hexdigest())
        for path, raw in paths.items()
    }
    monkeypatch.setattr(worker, "_git_index", lambda _repo: index)

    def fake_git(_repo, *arguments, binary=False):  # type: ignore[no-untyped-def]
        assert arguments[:2] == ("cat-file", "blob")
        blob = arguments[2]
        return next(raw for path, raw in paths.items() if index[path][1] == blob)

    monkeypatch.setattr(worker, "_git", fake_git)
    monkeypatch.setattr(
        worker,
        "_read_private",
        lambda path, **_kwargs: paths[path.relative_to(ROOT).as_posix()],
    )
    inventory = worker._inventory(ROOT, CONFIG, RUNNER)
    assert inventory[worker._WORKER_RELATIVE] == hashlib.sha256(b"worker").hexdigest()
    package_entries = sorted(
        (
            (path, len(raw), hashlib.sha256(raw).hexdigest())
            for path, raw in paths.items()
            if path.startswith(worker._PACKAGE_PREFIX)
        ),
        key=lambda item: item[0].encode("utf-8"),
    )
    digest = hashlib.sha256(worker._PACKAGE_DOMAIN)
    for path, size, raw_sha in package_entries:
        digest.update(path.encode() + b"\0")
        digest.update(str(size).encode() + b"\0")
        digest.update(raw_sha.encode() + b"\n")
    assert inventory["package_tree"] == digest.hexdigest()


def test_import_origin_inventory_rejects_ambient_module(
    worker, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    evil = tmp_path / "evil.py"
    evil.write_text("pass", encoding="utf-8")
    module = types.ModuleType("tnlm_v3.injected_for_worker_test")
    module.__file__ = str(evil)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(
        worker.PilotWorkerError,
        match="outside the checkout|absent from the committed inventory",
    ):
        worker._check_import_origins(ROOT, {})


def test_fresh_source_full_run_has_frozen_schema_and_no_test_stream(
    worker, campaign: CampaignFixture, fresh_source: SourceFixture
) -> None:  # type: ignore[no-untyped-def]
    artifacts, metrics, stream = (
        fresh_source.artifacts,
        fresh_source.metrics,
        fresh_source.stream,
    )
    assert set(artifacts) == {
        "checkpoints",
        "final_checkpoint",
        "compact_artifact",
    }
    checkpoints = artifacts["checkpoints"]
    assert [item["step"] for item in checkpoints] == [4, 8, 12]  # type: ignore[union-attr]
    assert all(
        set(item)
        == {"step", "path", "size_bytes", "sha256", "stream_prefix_sha256"}
        for item in checkpoints  # type: ignore[union-attr]
    )
    assert artifacts["compact_artifact"] is None
    assert artifacts["final_checkpoint"] == {
        key: checkpoints[-1][key]  # type: ignore[index]
        for key in ("path", "size_bytes", "sha256")
    }
    assert set(metrics) == {
        "environment",
        "training",
        "validation_by_length",
        "compact",
    }
    assert metrics["compact"] is None
    environment = metrics["environment"]
    assert set(environment) == {  # type: ignore[arg-type]
        "python_version",
        "torch_version",
        "numpy_version",
        "platform",
        "device",
    }
    assert environment["device"] == "cpu"  # type: ignore[index]
    assert all(type(value) is str and value for value in environment.values())  # type: ignore[union-attr]
    training = metrics["training"]
    assert set(training) == {  # type: ignore[arg-type]
        "initial_model_fingerprint",
        "final_model_fingerprint",
        "optimizer_steps",
        "token_count",
        "steps",
    }
    assert training["optimizer_steps"] == 12  # type: ignore[index]
    assert len(training["steps"]) == 12  # type: ignore[index]
    assert training["token_count"] == sum(  # type: ignore[index]
        item["token_count"] for item in training["steps"]  # type: ignore[index]
    )
    for index, item in enumerate(training["steps"]):  # type: ignore[index]
        assert set(item) == {
            "step",
            "batch_sha256",
            "token_count",
            "loss",
            "counters",
        }
        assert item["step"] == index
        assert set(item["loss"]) == {
            "total",
            "query",
            "route_curriculum",
            "router_balance",
            "router_entropy",
            "route_persistence",
        }
        assert set(item["counters"]) == {
            "query_count",
            "route_supervision_count",
            "persistence_pair_count",
        }
    assert set(stream) == {
        "start_step",
        "resumed_from_step",
        "completed_step",
        "training_batches",
        "stream_prefix_sha256",
        "checkpoint_steps",
    }
    assert stream["start_step"] == stream["resumed_from_step"] == 0
    assert stream["completed_step"] == 12
    assert stream["checkpoint_steps"] == [4, 8, 12]
    assert len(stream["training_batches"]) == 12
    assert [item["step"] for item in stream["training_batches"]] == list(range(12))
    assert all(
        set(item) == {"step", "sha256", "token_count"}
        for item in stream["training_batches"]
    )
    validation = metrics["validation_by_length"]
    assert [item["length"] for item in validation] == [16, 32, 64]  # type: ignore[union-attr]
    for item in validation:  # type: ignore[union-attr]
        assert set(item) == {
            "length",
            "batch_sha256",
            "episodes",
            "query",
            "seen_query",
            "heldout_query",
            "routing",
            "structural",
        }
        for name in ("query", "seen_query", "heldout_query"):
            query = item[name]
            assert set(query) == {"correct", "count", "accuracy", "cross_entropy"}
            assert query["accuracy"] == (
                query["correct"] / query["count"] if query["count"] else 0.0
            )
            assert (query["cross_entropy"] is None) is (query["count"] == 0)
        structural = item["structural"]
        assert structural["fingerprint_sha256"] == hashlib.sha256(
            worker._canonical(structural["values"])
        ).hexdigest()
        _assert_routing_schema(item["routing"])
    assert campaign.config.data.test is None
    assert campaign.config.data.scaling is None


def test_baseline_validation_keeps_routing_explicitly_null(
    worker, campaign: CampaignFixture
) -> None:  # type: ignore[no-untyped-def]
    baseline = next(item for item in campaign.plan if item.model_id == "gru-control")
    context = worker.CampaignRunContext(config=campaign.config, run=baseline)
    torch.manual_seed(baseline.model_seed)
    model = worker.build_campaign_source_model(context)
    validation = worker._evaluate(model, context)
    assert validation
    assert all(item["routing"] is None for item in validation)


def test_routing_serializer_rejects_nonplain_and_nonfinite_aggregates(
    worker, campaign: CampaignFixture
) -> None:  # type: ignore[no-untyped-def]
    torch.manual_seed(campaign.source.model_seed)
    model = worker.build_campaign_source_model(campaign.source_context)
    length = campaign.source.data.validation.lengths[0]
    batch = worker.generate_campaign_evaluation_batch(
        campaign.source_context, stream="validation", length=length
    )
    _, summary = worker.evaluate_binding_model(model, batch)

    nonplain = replace(
        summary,
        route_recovery=replace(summary.route_recovery, accuracy=1),
    )
    with pytest.raises(worker.PilotWorkerError, match="finite float"):
        worker._routing_entry(nonplain)

    nonfinite = replace(
        summary,
        router_load=replace(summary.router_load, load_entropy=float("nan")),
    )
    with pytest.raises(worker.PilotWorkerError, match="finite float"):
        worker._routing_entry(nonfinite)

    nonplain_vector = replace(
        summary,
        router_load=replace(
            summary.router_load,
            branch_counts=list(summary.router_load.branch_counts),
        ),
    )
    with pytest.raises(worker.PilotWorkerError, match="must be a tuple"):
        worker._routing_entry(nonplain_vector)


@pytest.mark.parametrize("route_value", [NULL_ROUTE, 3])
def test_routing_serializer_preserves_complete_no_local_route_evidence(
    worker,
    route_value: int,
) -> None:  # type: ignore[no-untyped-def]
    branches = 3
    routes = torch.full((2, 4), route_value, dtype=torch.int64)
    true_routes = torch.zeros_like(routes)
    valid = torch.ones_like(routes, dtype=torch.bool)
    documents = torch.arange(2, dtype=torch.int64).unsqueeze(1).expand_as(routes)
    fields = torch.zeros_like(routes)
    summary = types.SimpleNamespace(
        route_recovery=per_document_route_recovery(
            routes, true_routes, documents, valid, branches
        ),
        route_consistency=document_local_route_consistency(
            routes, documents, fields, fields, valid, branches
        ),
        router_load=summarize_router_load(
            routes, valid, branches, document_ids=documents
        ),
    )
    assert type(summary.router_load.load_entropy) is int

    routing = worker._routing_entry(summary)

    load = routing["router_load"]
    assert load["local_event_count"] == 0
    assert load["valid_event_count"] == 8
    assert load["load_entropy"] == 0.0
    assert type(load["load_entropy"]) is float
    assert load["collapsed"] is False
    assert load["collapsed_document_count"] == 0


def test_resume_replays_exact_full_record_and_checkpoint_bytes(
    worker,
    fresh_source: SourceFixture,
    resumed_source: SourceFixture,
) -> None:  # type: ignore[no-untyped-def]
    assert resumed_source.stream["resumed_from_step"] == 4
    assert fresh_source.metrics == resumed_source.metrics
    assert {
        key: value
        for key, value in fresh_source.stream.items()
        if key != "resumed_from_step"
    } == {
        key: value
        for key, value in resumed_source.stream.items()
        if key != "resumed_from_step"
    }
    fresh_checkpoints = fresh_source.artifacts["checkpoints"]
    resumed_checkpoints = resumed_source.artifacts["checkpoints"]
    assert [item["step"] for item in resumed_checkpoints] == [4, 8, 12]  # type: ignore[union-attr]
    for fresh, resumed in zip(fresh_checkpoints, resumed_checkpoints, strict=True):  # type: ignore[arg-type]
        assert {
            key: fresh[key]
            for key in ("step", "size_bytes", "sha256", "stream_prefix_sha256")
        } == {
            key: resumed[key]
            for key in ("step", "size_bytes", "sha256", "stream_prefix_sha256")
        }
        assert (
            fresh_source.output_root / fresh["path"]
        ).read_bytes() == (
            resumed_source.output_root / resumed["path"]
        ).read_bytes()
    assert worker._canonical(fresh_source.metrics) == worker._canonical(
        resumed_source.metrics
    )


def test_final_checkpoint_contract_is_bound_to_stream(
    worker, campaign: CampaignFixture, fresh_source: SourceFixture
) -> None:  # type: ignore[no-untyped-def]
    final = fresh_source.artifacts["final_checkpoint"]
    raw = (fresh_source.output_root / final["path"]).read_bytes()  # type: ignore[index]
    fresh_model = worker.build_campaign_source_model(campaign.source_context)
    fresh_optimizer = worker.build_campaign_optimizer(
        campaign.source_context, fresh_model
    )
    model, optimizer, resume = deserialize_campaign_checkpoint(
        raw,
        expected_run_spec_sha256=worker._run_hash(campaign.source),
        expected_stream_prefix_sha256=fresh_source.stream["stream_prefix_sha256"],
        expected_contract=campaign_checkpoint_contract(fresh_model, fresh_optimizer),
        device="cpu",
    )
    assert resume.global_step == resume.data_cursor == 12
    assert worker.campaign_model_fingerprint(model) == fresh_source.metrics["training"][  # type: ignore[index]
        "final_model_fingerprint"
    ]
    assert all(float(state["step"]) == 12 for state in optimizer.state.values())


def test_derived_compact_binds_parent_and_round_trips_canonically(
    worker,
    campaign: CampaignFixture,
    fresh_source: SourceFixture,
    compact_result: CompactFixture,
) -> None:  # type: ignore[no-untyped-def]
    artifacts, metrics, stream = (
        compact_result.artifacts,
        compact_result.metrics,
        compact_result.stream,
    )
    assert artifacts["checkpoints"] == []
    assert artifacts["final_checkpoint"] is None
    assert artifacts["compact_artifact"] is not None
    assert metrics["training"] is None
    assert stream == {
        "start_step": 0,
        "resumed_from_step": 0,
        "completed_step": 0,
        "training_batches": [],
        "stream_prefix_sha256": None,
        "checkpoint_steps": [],
    }
    compact = metrics["compact"]
    assert set(compact) == {  # type: ignore[arg-type]
        "parent_run_id",
        "parent_result",
        "parent_checkpoint",
        "parent_model_fingerprint",
        "selection_fingerprint",
        "manifest_fingerprint",
        "exported_model_fingerprint",
        "compact_artifact_sha256",
    }
    assert compact["parent_run_id"] == campaign.source.run_id  # type: ignore[index]
    assert compact["parent_model_fingerprint"] == fresh_source.metrics["training"][  # type: ignore[index]
        "final_model_fingerprint"
    ]
    reference = artifacts["compact_artifact"]
    raw = (fresh_source.output_root / reference["path"]).read_bytes()  # type: ignore[index]
    assert hashlib.sha256(raw).hexdigest() == compact["compact_artifact_sha256"]  # type: ignore[index]
    model, manifest, selection = deserialize_compact_binding_model(
        raw,
        device="cpu",
        expected_manifest_fingerprint=compact["manifest_fingerprint"],  # type: ignore[index]
        expected_selection_fingerprint=compact["selection_fingerprint"],  # type: ignore[index]
    )
    assert model.config.cp_rank == 2
    assert manifest.exported_model_fingerprint == compact["exported_model_fingerprint"]  # type: ignore[index]
    assert manifest.selection_fingerprint == selection.fingerprint()
    assert serialize_compact_binding_model(model, manifest, selection) == raw
    for item in metrics["validation_by_length"]:  # type: ignore[union-attr]
        _assert_routing_schema(item["routing"])


def test_compact_rejects_tampered_parent_result_hash_and_model_binding(
    worker,
    campaign: CampaignFixture,
    fresh_source: SourceFixture,
    compact_result: CompactFixture,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(worker.PilotWorkerError, match="checksum mismatch"):
        worker._parent_result(compact_result.parent_result_path, "0" * 64)

    parent = _successful_parent_document(worker, campaign, fresh_source)
    parent["metrics"]["training"]["final_model_fingerprint"] = "0" * 64  # type: ignore[index]
    tampered_dir = (
        fresh_source.output_root
        / "artifacts"
        / campaign.source.run_id
        / "attempt-000003"
    )
    tampered_dir.mkdir(parents=True)
    parent["attempt_number"] = 3
    tampered = tampered_dir / "result.json"
    tampered.write_bytes(worker._canonical(parent))
    checkpoint = fresh_source.artifacts["final_checkpoint"]
    checkpoint_source = fresh_source.output_root / checkpoint["path"]  # type: ignore[index]
    checkpoint_copy = tampered_dir / checkpoint_source.name
    checkpoint_copy.write_bytes(checkpoint_source.read_bytes())
    parent["artifacts"]["final_checkpoint"] = {  # type: ignore[index]
        "path": checkpoint_copy.relative_to(fresh_source.output_root).as_posix(),
        "size_bytes": checkpoint_copy.stat().st_size,
        "sha256": checkpoint["sha256"],  # type: ignore[index]
    }
    tampered.write_bytes(worker._canonical(parent))
    attempt = tmp_path / "compact-attempt"
    attempt.mkdir()
    with pytest.raises(worker.PilotWorkerError, match="does not bind"):
        worker._compact_run(
            campaign.compact_context,
            campaign.config,
            campaign.plan,
            attempt,
            fresh_source.output_root,
            tampered,
            hashlib.sha256(tampered.read_bytes()).hexdigest(),
            checkpoint_copy,
            checkpoint["sha256"],  # type: ignore[index]
        )


def test_execute_rejects_result_path_escape_before_run(
    worker,
    campaign: CampaignFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "external"
    output.mkdir()
    attempt = output / "artifacts" / campaign.source.run_id / "attempt-000001"
    attempt.mkdir(parents=True)
    inventory = {
        CONFIG.relative_to(ROOT).as_posix(): campaign.raw_config_sha256,
        worker._RUNNER_RELATIVE: "6" * 64,
        worker._WORKER_RELATIVE: "7" * 64,
        "package_tree": "8" * 64,
        "bundle": BUNDLE,
    }

    def fake_git(_repo, *arguments, **_kwargs):  # type: ignore[no-untyped-def]
        if arguments == ("rev-parse", "HEAD"):
            return COMMIT + "\n"
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return TREE + "\n"
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(worker, "_git", fake_git)
    monkeypatch.setattr(worker, "_inventory", lambda *_args: inventory)
    monkeypatch.setattr(worker, "_check_import_origins", lambda *_args: None)
    monkeypatch.setattr(worker.torch, "set_num_threads", lambda _value: None)
    monkeypatch.setattr(worker.torch, "set_num_interop_threads", lambda _value: None)
    monkeypatch.setattr(worker, "validate_campaign_execution_environment", lambda _context: None)
    plan_sha = worker.campaign_plan_sha256(campaign.config, campaign.plan)
    arguments = types.SimpleNamespace(
        repo_root=str(ROOT),
        parent_runner=str(RUNNER),
        config=str(CONFIG),
        output_root=str(output),
        result_path=str(output / "escaped-result.json"),
        run_id=campaign.source.run_id,
        attempt_number=1,
        plan_sha256=plan_sha,
        code_commit=COMMIT,
        code_tree=TREE,
        raw_config_sha256=campaign.raw_config_sha256,
        semantic_config_sha256=campaign.config.fingerprint(),
        executable_bundle_sha256=BUNDLE,
        parent_runner_sha256="6" * 64,
        worker_sha256="7" * 64,
        package_tree_sha256="8" * 64,
        resume_checkpoint=None,
        resume_checkpoint_sha256=None,
        parent_result=None,
        parent_result_sha256=None,
        parent_checkpoint=None,
        parent_checkpoint_sha256=None,
    )
    with pytest.raises(worker.PilotWorkerError, match="outside the exact attempt"):
        worker._execute(arguments)


def test_execute_refuses_confirmatory_or_test_capable_config(
    worker,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    class Data:
        test = object()
        scaling = object()

    class Confirmatory:
        stage = worker.CampaignStage.CONFIRMATORY
        claim_eligible = True
        data = Data()

    output = tmp_path / "external"
    output.mkdir()
    raw = CONFIG.read_bytes()
    raw_sha = hashlib.sha256(raw).hexdigest()
    inventory = {
        CONFIG.relative_to(ROOT).as_posix(): raw_sha,
        worker._RUNNER_RELATIVE: "6" * 64,
        worker._WORKER_RELATIVE: "7" * 64,
        "package_tree": "8" * 64,
        "bundle": BUNDLE,
    }

    def fake_git(_repo, *arguments, **_kwargs):  # type: ignore[no-untyped-def]
        if arguments == ("rev-parse", "HEAD"):
            return COMMIT + "\n"
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return TREE + "\n"
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(worker, "_git", fake_git)
    monkeypatch.setattr(worker, "_inventory", lambda *_args: inventory)
    monkeypatch.setattr(worker, "_check_import_origins", lambda *_args: None)
    monkeypatch.setattr(worker, "load_milestone4_campaign_config", lambda _path: Confirmatory())
    arguments = types.SimpleNamespace(
        repo_root=str(ROOT),
        parent_runner=str(RUNNER),
        config=str(CONFIG),
        output_root=str(output),
        result_path=str(output / "result.json"),
        run_id="9" * 64,
        attempt_number=1,
        plan_sha256="a" * 64,
        code_commit=COMMIT,
        code_tree=TREE,
        raw_config_sha256=raw_sha,
        semantic_config_sha256="b" * 64,
        executable_bundle_sha256=BUNDLE,
        parent_runner_sha256="6" * 64,
        worker_sha256="7" * 64,
        package_tree_sha256="8" * 64,
        resume_checkpoint=None,
        resume_checkpoint_sha256=None,
        parent_result=None,
        parent_result_sha256=None,
        parent_checkpoint=None,
        parent_checkpoint_sha256=None,
    )
    with pytest.raises(worker.PilotWorkerError, match="non-claiming pilot or screen"):
        worker._execute(arguments)


def test_main_writes_canonical_failure_result(
    worker,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    result_path = tmp_path / "failure.json"
    monkeypatch.setattr(
        worker,
        "_execute",
        lambda _arguments: (_ for _ in ()).throw(
            worker.PilotWorkerError("deliberate failure")
        ),
    )
    values = {
        "repo-root": str(ROOT),
        "parent-runner": str(RUNNER),
        "config": str(CONFIG),
        "output-root": str(tmp_path),
        "result-path": str(result_path),
        "run-id": "9" * 64,
        "plan-sha256": "a" * 64,
        "code-commit": COMMIT,
        "code-tree": TREE,
        "raw-config-sha256": "b" * 64,
        "semantic-config-sha256": "c" * 64,
        "executable-bundle-sha256": BUNDLE,
        "parent-runner-sha256": "d" * 64,
        "worker-sha256": "e" * 64,
        "package-tree-sha256": "f" * 64,
    }
    argv = [item for name, value in values.items() for item in (f"--{name}", value)]
    argv.extend(("--attempt-number", "2"))
    assert worker.main(argv) == 1
    raw = result_path.read_bytes()
    document = worker._strict_json(raw, "failure result")
    assert worker._canonical(document) == raw
    assert document["status"] == "failure"
    assert document["metrics"] is None
    assert document["stream"] is None
    assert document["artifacts"] == {
        "checkpoints": [],
        "final_checkpoint": None,
        "compact_artifact": None,
    }
    assert document["error"] == {
        "type": "PilotWorkerError",
        "message": "deliberate failure",
    }


def test_immutable_write_is_idempotent_and_rejects_changed_bytes(
    worker, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "immutable.json"
    worker._write_atomic(path, b"one", immutable=True)
    worker._write_atomic(path, b"one", immutable=True)
    with pytest.raises(worker.PilotWorkerError, match="already differs"):
        worker._write_atomic(path, b"two", immutable=True)
    assert path.read_bytes() == b"one"
