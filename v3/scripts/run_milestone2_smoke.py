"""Run the deterministic Milestone-2 oracle/curriculum/latent smoke matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import torch
import tnlm_v3

from tnlm_v3.benchmark import (
    EvaluationRunOutputs,
    audit_evaluation_label_independence,
    compute_oracle_gap,
)
from tnlm_v3.data import collate_binding_episodes, generate_binding_episodes
from tnlm_v3.factory import (
    BindingExperimentConfig,
    build_binding_model,
    load_binding_experiment_config,
)
from tnlm_v3.routing import RoutingMode
from tnlm_v3.training import evaluate_binding_model, train_binding_step


def _finite(value: object, path: str = "$") -> None:
    """Reject anything outside the strict, finite JSON value domain."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"smoke record contains a non-finite float at {path}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"smoke record contains a non-string key at {path}")
            _finite(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")
        return
    raise TypeError(
        f"smoke record contains unsupported JSON value {type(value).__name__} at {path}"
    )


def _atomic_json(output: Path, record: dict[str, object]) -> dict[str, object]:
    """Durably replace one strict JSON progress/evidence record."""

    _finite(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    snapshot = json.loads(encoded)
    if not isinstance(snapshot, dict):
        raise TypeError("smoke record root must be a JSON object")
    return snapshot


def _git_command(repository_root: Path, *arguments: str) -> list[str]:
    """Build a Git command explicitly trusting only this normalized checkout."""

    normalized_root = repository_root.resolve().as_posix()
    return ["git", "-c", f"safe.directory={normalized_root}", *arguments]


def _bind_to_clean_checkout(code_commit: str) -> tuple[str, str]:
    """Require the evidence record to name the exact clean source checkout."""

    repository_root = Path(__file__).resolve().parents[2]
    package_root = (repository_root / "v3" / "src" / "tnlm_v3").resolve()
    package_file = Path(tnlm_v3.__file__).resolve()
    if not package_file.is_relative_to(package_root):
        raise RuntimeError(
            f"tnlm_v3 imported from {package_file}, outside checkout {package_root}"
        )

    def git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                _git_command(repository_root, *arguments),
                cwd=repository_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError("smoke evidence requires an accessible Git checkout") from error
        return completed.stdout.strip()

    head = git("rev-parse", "HEAD").lower()
    if head != code_commit.lower():
        raise ValueError(f"--code-commit {code_commit} does not match checkout HEAD {head}")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("smoke evidence requires a completely clean worktree")
    tree = git("rev-parse", "HEAD^{tree}").lower()
    return head, tree


def _require_committed_config(path: Path, repository_root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(repository_root):
        raise ValueError(f"smoke config is outside the checkout: {resolved}")
    relative = resolved.relative_to(repository_root).as_posix()
    try:
        subprocess.run(
            _git_command(
                repository_root, "ls-files", "--error-unmatch", "--", relative
            ),
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"smoke config is not committed: {relative}") from error
    return resolved


def _summary_dict(summary) -> dict[str, object]:
    return {
        "query": asdict(summary.query),
        "seen_query": asdict(summary.seen_query),
        "heldout_query": asdict(summary.heldout_query),
        "route_recovery": {
            key: value
            for key, value in asdict(summary.route_recovery).items()
            if key != "documents"
        },
        "route_consistency": {
            key: value
            for key, value in asdict(summary.route_consistency).items()
            if key != "groups"
        },
        "router_load": asdict(summary.router_load),
    }


def _make_batch(config: BindingExperimentConfig, split: str):
    episodes = generate_binding_episodes(
        config.task,
        count=config.episodes,
        seed=config.data_seed,
        split=split,
        lengths=[config.sequence_length] * config.episodes,
    )
    return collate_binding_episodes(episodes)


def _batch_sha256(batch) -> str:
    digest = hashlib.sha256()
    for owner_name, owner in (
        ("inputs", batch.inputs),
        ("evaluation", batch.evaluation),
    ):
        for field in fields(owner):
            tensor = getattr(owner, field.name).detach().contiguous().cpu()
            digest.update(
                f"{owner_name}.{field.name}|{tensor.dtype}|{tuple(tensor.shape)}|".encode()
            )
            digest.update(tensor.numpy().tobytes())
    lengths = batch.lengths.detach().contiguous().cpu()
    digest.update(f"lengths|{lengths.dtype}|{tuple(lengths.shape)}|".encode())
    digest.update(lengths.numpy().tobytes())
    digest.update(json.dumps(batch.document_ids, separators=(",", ":")).encode())
    digest.update(json.dumps(batch.generation_seeds, separators=(",", ":")).encode())
    return digest.hexdigest()


def _state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().contiguous().cpu()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}|".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run_condition(
    config: BindingExperimentConfig,
    *,
    config_sha256: str,
) -> dict[str, object]:
    torch.manual_seed(config.model_seed)
    model = build_binding_model(config).to(dtype=torch.float32)
    batch = _make_batch(config, "train")
    heldout_batch = _make_batch(config, "eval")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_shapes = {
        name: list(parameter.shape) for name, parameter in model.named_parameters()
    }
    parameter_shapes_sha256 = hashlib.sha256(
        json.dumps(parameter_shapes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    initial_state_dict_sha256 = _state_dict_sha256(model)
    _, initial = evaluate_binding_model(model, batch)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    curriculum_trace: list[dict[str, int | float]] = []
    curriculum_evidence: dict[str, object] | None = None
    if config.condition is RoutingMode.CURRICULUM:
        assert config.model.curriculum_schedule is not None
        curriculum_evidence = {
            "declared_schedule": config.model.curriculum_schedule.as_dict(),
            "realized_trace": curriculum_trace,
        }
    started = time.perf_counter()
    last_loss = None
    last_output = None
    for step in range(1, config.steps + 1):
        last_output, last_loss = train_binding_step(
            model,
            batch,
            optimizer,
            training_step=step,
            loss_config=config.loss,
            max_gradient_norm=config.max_gradient_norm,
        )
        if curriculum_evidence is not None:
            curriculum_trace.append(
                {
                    "step": step,
                    "guidance_probability": float(
                        last_output.diagnostics["guidance_probability"]
                    ),
                    "guided_events": int(last_output.diagnostics["guided_events"]),
                    "guided_fraction": float(
                        last_output.diagnostics["guided_fraction"]
                    ),
                    "route_supervision_count": last_loss.route_supervision_count,
                }
            )
    elapsed = time.perf_counter() - started
    final_output, final = evaluate_binding_model(model, batch)
    _, heldout = evaluate_binding_model(model, heldout_batch)
    assert last_loss is not None and last_output is not None

    label_audit: dict[str, object] | None = None
    if config.condition is RoutingMode.CURRICULUM:
        model.eval()
        reference = model(
            batch.inputs,
            route_labels=batch.evaluation.oracle_routes,
            training_step=0,
        )
        poisoned_labels = torch.full_like(batch.evaluation.oracle_routes, 987654321)
        relabeled = model(
            batch.inputs,
            route_labels=poisoned_labels,
            training_step=0,
        )
        label_audit = asdict(
            audit_evaluation_label_independence(
                EvaluationRunOutputs(reference.value_logits, reference.routes),
                EvaluationRunOutputs(relabeled.value_logits, relabeled.routes),
            )
        )
        route_logits_equal = torch.equal(
            reference.route_logits, relabeled.route_logits
        )
        label_audit["route_logits_equal"] = route_logits_equal
        label_audit["passed"] = bool(label_audit["passed"] and route_logits_equal)

    result = {
        "condition": config.condition.value,
        "status": "completed",
        "config_sha256": config_sha256,
        "config_fingerprint": config.fingerprint(),
        "task_fingerprint": config.task.fingerprint(),
        "model_fingerprint": config.model.fingerprint(),
        "model_seed": config.model_seed,
        "data_seed": config.data_seed,
        "episodes": config.episodes,
        "sequence_length": config.sequence_length,
        "steps": config.steps,
        "parameter_count": parameter_count,
        "parameter_shapes_sha256": parameter_shapes_sha256,
        "initial_state_dict_sha256": initial_state_dict_sha256,
        "train_batch_sha256": _batch_sha256(batch),
        "heldout_batch_sha256": _batch_sha256(heldout_batch),
        "elapsed_training_seconds": elapsed,
        "initial": _summary_dict(initial),
        "final": _summary_dict(final),
        "heldout_evaluation": _summary_dict(heldout),
        "final_loss": {
            "total": float(last_loss.total.detach()),
            "query": float(last_loss.query.detach()),
            "route_curriculum": float(last_loss.route_curriculum.detach()),
            "router_balance": float(last_loss.router_balance.detach()),
            "router_entropy": float(last_loss.router_entropy.detach()),
            "route_persistence": float(last_loss.route_persistence.detach()),
            "query_count": last_loss.query_count,
            "route_supervision_count": last_loss.route_supervision_count,
            "persistence_pair_count": last_loss.persistence_pair_count,
        },
        "final_guidance": {
            "probability": float(last_output.diagnostics["guidance_probability"]),
            "guided_events": int(last_output.diagnostics["guided_events"]),
            "guided_fraction": float(last_output.diagnostics["guided_fraction"]),
        },
        "structural_work": {
            "router_branch_score_work_proxy": int(
                final_output.diagnostics["branch_score_work_proxy"]
            ),
            "forest_merge_count": int(final_output.diagnostics["forest_merge_count"]),
            "forest_active_slots": int(final_output.diagnostics["forest_active_slots"]),
        },
        "curriculum": curriculum_evidence,
        "evaluation_label_independence": label_audit,
    }
    _finite(result)
    return result


def _attach_oracle_gaps(by_mode: dict[str, dict[str, object]]) -> None:
    """Attach explicitly named train and held-out oracle comparisons."""

    oracle = by_mode[RoutingMode.ORACLE.value]
    comparisons = {
        "fixed_train_query": ("final", "query"),
        "heldout_all_query": ("heldout_evaluation", "query"),
        "heldout_combination_query": ("heldout_evaluation", "heldout_query"),
    }
    for mode in (RoutingMode.CURRICULUM.value, RoutingMode.LATENT.value):
        gaps: dict[str, object] = {}
        for name, (evaluation_key, query_key) in comparisons.items():
            oracle_accuracy = oracle[evaluation_key][query_key]["accuracy"]
            autonomous_accuracy = by_mode[mode][evaluation_key][query_key]["accuracy"]
            gaps[name] = asdict(
                compute_oracle_gap(oracle_accuracy, autonomous_accuracy)
            )
        by_mode[mode]["oracle_gaps"] = gaps


def _failure_record(
    durable_record: dict[str, object], error: Exception
) -> dict[str, object]:
    """Build failure evidence from the last record known to be JSON-durable."""

    failure = dict(durable_record)
    failure.update(
        {
            "status": "failed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    )
    _finite(failure)
    return failure


def _execute(args: argparse.Namespace) -> Path:
    """Execute one strict smoke run and persist preflight/runtime failures."""

    torch.set_num_threads(1)
    repository_root = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve()
    if output.is_relative_to(repository_root):
        raise ValueError(
            "--output must be outside the source checkout; copy the completed "
            "record into the repository only after the run"
        )

    record: dict[str, object] = {
        "schema_version": 1,
        "status": "initializing",
        "scope": "deterministic_fixed_batch_smoke_not_scientific_campaign",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_code_commit": args.code_commit,
        "conditions": [],
    }
    durable_record = _atomic_json(output, record)

    try:
        if len(args.code_commit) != 40 or any(
            character not in "0123456789abcdef"
            for character in args.code_commit.lower()
        ):
            raise ValueError("--code-commit must be a full 40-character Git SHA")

        code_commit, code_tree = _bind_to_clean_checkout(args.code_commit)
        config_paths = [
            _require_committed_config(Path(path), repository_root)
            for path in args.config
        ]
        config_snapshots = [path.read_bytes() for path in config_paths]
        config_hashes = [
            hashlib.sha256(content).hexdigest() for content in config_snapshots
        ]
        configs = [load_binding_experiment_config(path) for path in config_paths]
        if any(
            path.read_bytes() != snapshot
            for path, snapshot in zip(config_paths, config_snapshots, strict=True)
        ):
            raise RuntimeError("a smoke config changed while it was being parsed")
        if len(configs) != len(RoutingMode) or {
            config.condition for config in configs
        } != set(RoutingMode):
            raise ValueError("smoke matrix must contain oracle, curriculum, and latent")
        if len({config.model_seed for config in configs}) != 1:
            raise ValueError("smoke conditions must share a paired model seed")
        if len({config.data_seed for config in configs}) != 1:
            raise ValueError("smoke conditions must share a paired data seed")
        if len({config.task.fingerprint() for config in configs}) != 1:
            raise ValueError("smoke conditions must share an identical task")
        if len({config.episodes for config in configs}) != 1 or len(
            {config.sequence_length for config in configs}
        ) != 1:
            raise ValueError("smoke conditions must share one fixed batch shape")
        architecture_fingerprints = {
            config.model.task.fingerprint() for config in configs
        }
        if len(architecture_fingerprints) != 1:
            raise ValueError("smoke conditions must share an identical architecture")
        architecture_signatures = {
            (
                config.model.d_model,
                config.model.cp_rank,
                config.model.router_hidden_dim,
                config.model.scale_feature_dim,
                config.model.straight_through_route_surrogate,
            )
            for config in configs
        }
        if len(architecture_signatures) != 1:
            raise ValueError("smoke conditions must share identical model dimensions")

        train_batch_hashes = {
            _batch_sha256(_make_batch(config, "train")) for config in configs
        }
        heldout_batch_hashes = {
            _batch_sha256(_make_batch(config, "eval")) for config in configs
        }
        if len(train_batch_hashes) != 1 or len(heldout_batch_hashes) != 1:
            raise ValueError("smoke conditions must generate identical paired batches")

        record = {
            "schema_version": 1,
            "status": "in_progress",
            "scope": "deterministic_fixed_batch_smoke_not_scientific_campaign",
            "generated_at_utc": durable_record["generated_at_utc"],
            "code_commit": code_commit,
            "code_tree": code_tree,
            "paired_train_batch_sha256": next(iter(train_batch_hashes)),
            "paired_heldout_batch_sha256": next(iter(heldout_batch_hashes)),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "torch_num_threads": torch.get_num_threads(),
            },
            "conditions": [],
        }
        durable_record = _atomic_json(output, record)

        results: list[dict[str, object]] = []
        for config, config_sha256 in zip(configs, config_hashes, strict=True):
            result = run_condition(config, config_sha256=config_sha256)
            _finite(result)
            candidate_results = [*results, result]
            record["conditions"] = candidate_results
            durable_record = _atomic_json(output, record)
            results = candidate_results

        by_mode = {result["condition"]: result for result in results}
        _attach_oracle_gaps(by_mode)
        _finite(results)
        expected_work = (
            configs[0].episodes
            * configs[0].sequence_length
            * configs[0].task.branches
        )
        gates = {
            "all_conditions_present": set(by_mode)
            == {mode.value for mode in RoutingMode},
            "paired_parameter_count": len(
                {result["parameter_count"] for result in results}
            )
            == 1,
            "paired_parameter_shapes": len(
                {result["parameter_shapes_sha256"] for result in results}
            )
            == 1,
            "paired_initial_weights": len(
                {result["initial_state_dict_sha256"] for result in results}
            )
            == 1,
            "paired_train_batch": len(
                {result["train_batch_sha256"] for result in results}
            )
            == 1,
            "paired_heldout_batch": len(
                {result["heldout_batch_sha256"] for result in results}
            )
            == 1,
            "oracle_fixed_batch_overfit": by_mode["oracle"]["final"]["query"][
                "accuracy"
            ]
            == 1.0,
            "curriculum_fixed_batch_overfit": by_mode["curriculum"]["final"][
                "query"
            ]["accuracy"]
            == 1.0,
            "curriculum_autonomous_label_independence": bool(
                by_mode["curriculum"]["evaluation_label_independence"]["passed"]
            ),
            "curriculum_final_guidance_zero": by_mode["curriculum"][
                "final_guidance"
            ]["guided_events"]
            == 0,
            "curriculum_final_route_supervision_zero": by_mode["curriculum"][
                "final_loss"
            ]["route_supervision_count"]
            == 0,
            "latent_route_supervision_zero": by_mode["latent"]["final_loss"][
                "route_supervision_count"
            ]
            == 0,
            "heldout_queries_present": all(
                result["heldout_evaluation"]["heldout_query"]["query_count"] > 0
                for result in results
            ),
            "router_work_proxy_exact": all(
                result["structural_work"]["router_branch_score_work_proxy"]
                == expected_work
                for result in results
            ),
            "route_loads_complete": all(
                result["final"]["router_load"]["valid_event_count"]
                == result["final"]["router_load"]["local_event_count"]
                + result["final"]["router_load"]["global_event_count"]
                + result["final"]["router_load"]["null_event_count"]
                for result in results
            ),
        }
        if not all(gates.values()):
            raise RuntimeError(f"Milestone-2 smoke gate failed: {gates}")

        if any(
            path.read_bytes() != snapshot
            for path, snapshot in zip(config_paths, config_snapshots, strict=True)
        ):
            raise RuntimeError("a smoke config changed during execution")
        final_commit, final_tree = _bind_to_clean_checkout(code_commit)
        if (final_commit, final_tree) != (code_commit, code_tree):
            raise RuntimeError("the source checkout changed during execution")

        record.update(
            {
                "status": "passed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "gates": gates,
                "conditions": results,
            }
        )
        durable_record = _atomic_json(output, record)
    except Exception as error:
        _atomic_json(output, _failure_record(durable_record, error))
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    output = _execute(args)
    print(f"MILESTONE2_SMOKE_PASSED: {output}")


if __name__ == "__main__":
    main()
