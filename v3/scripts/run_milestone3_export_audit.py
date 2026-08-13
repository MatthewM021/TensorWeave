#!/usr/bin/env python3
"""Run the commit-bound Milestone-3 physical CP-export implementation audit.

This is an implementation audit, not a scientific quality campaign.  It
trains one predeclared fixed-batch curriculum source model, selects CP channels
with the declared structural heuristic, and checks the physically sliced model
against its dense zero-channel reference.  Every generated artifact lives
outside the source checkout until a completed record is reviewed separately.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from typing import Iterable, Mapping

import torch
from torch import Tensor

import tnlm_v3
from tnlm_v3.binding import BindingModelOutput, RoutedBindingModel
from tnlm_v3.compact_artifact import (
    deserialize_compact_binding_model,
    serialize_compact_binding_model,
)
from tnlm_v3.data import (
    BindingBatch,
    BindingEventKind,
    BindingModelInputs,
    collate_binding_episodes,
    generate_binding_episodes,
)
from tnlm_v3.export import deserialize_forest_state, serialize_forest_state
from tnlm_v3.export_audit import (
    ExportAuditConfig,
    atomic_write_json,
    binding_inputs_sha256,
    forest_state_sha256,
    load_export_audit_config,
    tensor_sha256,
    validate_finite_json,
)
from tnlm_v3.factory import (
    BindingExperimentConfig,
    build_binding_model,
    load_binding_experiment_config,
)
from tnlm_v3.forest import ForestState
from tnlm_v3.model_export import export_compact_binding_model
from tnlm_v3.routing import NULL_ROUTE, PersistentRouterState
from tnlm_v3.training import train_binding_step
from tnlm_v3.truncation import (
    build_dense_selected_reference,
    model_state_fingerprint,
    select_cp_rank_by_parameter_energy,
)


_SCHEMA_VERSION = 1
_HEX = frozenset("0123456789abcdef")
_OUTPUT_TENSORS = (
    "routes",
    "route_logits",
    "route_probabilities",
    "value_logits",
)
_FOREST_FIELDS = ("slots", "occupied", "counts", "valid_steps")
_ROUTER_FIELDS = (
    "prototypes",
    "occupied",
    "ages",
    "loads",
    "global_state",
    "global_occupied",
    "global_load",
    "valid_steps",
)
_CP_STATE_NAMES = (
    "forest.merge.left.weight",
    "forest.merge.right.weight",
    "forest.merge.scale_to_rank.weight",
    "forest.merge.global_rank",
    "forest.merge.output.weight",
)
_REPLAY_HASH_DOMAIN = b"tnlm_v3.compact_replay_tensor.v1\x00"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_sha(value: object, name: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in _HEX for character in value.lower())
        or value != value.lower()
    ):
        raise ValueError(f"{name} must be a lowercase {length}-character hex digest")
    return value


def _git_command(repository_root: Path, *arguments: str) -> list[str]:
    return [
        "git",
        "-c",
        f"safe.directory={repository_root.resolve().as_posix()}",
        *arguments,
    ]


def _git(repository_root: Path, *arguments: str) -> str:
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
        raise RuntimeError("Milestone-3 evidence requires an accessible Git checkout") from error
    return completed.stdout.strip()


def _bind_to_clean_checkout(code_commit: str) -> tuple[str, str]:
    """Bind imports, HEAD, tree, and worktree cleanliness to one checkout."""

    repository_root = Path(__file__).resolve().parents[2]
    package_root = (repository_root / "v3" / "src" / "tnlm_v3").resolve()
    package_file = Path(tnlm_v3.__file__).resolve()
    if not package_file.is_relative_to(package_root):
        raise RuntimeError(f"tnlm_v3 imported from outside the checkout: {package_file}")
    if not Path(__file__).resolve().is_relative_to(repository_root):
        raise RuntimeError("audit runner is outside the bound checkout")
    head = _git(repository_root, "rev-parse", "HEAD").lower()
    if head != code_commit.lower():
        raise ValueError(f"--code-commit {code_commit} does not match checkout HEAD {head}")
    if _git(repository_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Milestone-3 evidence requires a completely clean worktree")
    return head, _git(repository_root, "rev-parse", "HEAD^{tree}").lower()


def _require_committed_file(path: Path, repository_root: Path, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(repository_root) or not resolved.is_file():
        raise ValueError(f"{name} must be a file inside the source checkout")
    relative = resolved.relative_to(repository_root).as_posix()
    try:
        subprocess.run(
            _git_command(repository_root, "ls-files", "--error-unmatch", "--", relative),
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"{name} is not committed: {relative}") from error
    return resolved


def _strict_external_paths(
    repository_root: Path, named_paths: Mapping[str, str | Path]
) -> dict[str, Path]:
    """Require distinct absolute generated paths outside the checkout."""

    resolved: dict[str, Path] = {}
    for name, value in named_paths.items():
        candidate = Path(value)
        if not candidate.is_absolute():
            raise ValueError(f"--{name.replace('_', '-')} must be an absolute path")
        candidate = candidate.resolve()
        if candidate.is_relative_to(repository_root):
            raise ValueError(f"--{name.replace('_', '-')} must be outside the source checkout")
        resolved[name] = candidate
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("all generated output paths must be distinct")
    existing = [(name, value) for name, value in resolved.items() if value.exists()]
    for index, (left_name, left) in enumerate(existing):
        for right_name, right in existing[index + 1 :]:
            try:
                same = os.path.samefile(left, right)
            except OSError as error:
                raise ValueError("generated output identity could not be verified") from error
            if same:
                raise ValueError(
                    f"--{left_name.replace('_', '-')} and "
                    f"--{right_name.replace('_', '-')} must not alias"
                )
    files = [value for name, value in resolved.items() if name != "runtime_directory"]
    runtime = resolved.get("runtime_directory")
    if runtime is not None and any(value == runtime or value.is_relative_to(runtime) for value in files):
        raise ValueError("top-level generated files must be outside --runtime-directory")
    return resolved


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _failure_record(durable: dict[str, object], error: Exception) -> dict[str, object]:
    result = dict(durable)
    result.update(
        {
            "status": "failed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    )
    validate_finite_json(result)
    return result


def _make_fixed_train_batch(config: BindingExperimentConfig) -> BindingBatch:
    return collate_binding_episodes(
        generate_binding_episodes(
            config.task,
            count=config.episodes,
            seed=config.data_seed,
            split="train",
            lengths=[config.sequence_length] * config.episodes,
        )
    )


def _make_declared_batch(
    task,
    *,
    split: str,
    seed: int,
    lengths: Iterable[int],
) -> BindingBatch:
    declared = tuple(lengths)
    return collate_binding_episodes(
        generate_binding_episodes(
            task,
            count=len(declared),
            seed=seed,
            split=split,
            lengths=declared,
        )
    )


def _batch_sha256(batch: BindingBatch) -> str:
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(binding_inputs_sha256(batch.inputs)))
    for field in fields(batch.evaluation):
        digest.update(field.name.encode("ascii"))
        digest.update(bytes.fromhex(tensor_sha256(getattr(batch.evaluation, field.name))))
    digest.update(bytes.fromhex(tensor_sha256(batch.lengths)))
    for values in (
        batch.splits,
        batch.document_ids,
        batch.generation_seeds,
    ):
        digest.update(json.dumps(values, separators=(",", ":")).encode("utf-8"))
    digest.update(batch.config_fingerprint.encode("ascii"))
    return digest.hexdigest()


def _train_source(config: BindingExperimentConfig) -> tuple[RoutedBindingModel, dict[str, object]]:
    if config.condition.value != "curriculum":
        raise ValueError("Milestone-3 source condition must be curriculum")
    torch.manual_seed(config.model_seed)
    model = build_binding_model(config).to(dtype=torch.float32)
    initial_fingerprint = model_state_fingerprint(model)
    batch = _make_fixed_train_batch(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    started = time.perf_counter_ns()
    final_loss = None
    final_output = None
    for step in range(1, config.steps + 1):
        final_output, final_loss = train_binding_step(
            model,
            batch,
            optimizer,
            training_step=step,
            loss_config=config.loss,
            max_gradient_norm=config.max_gradient_norm,
        )
    elapsed_ns = time.perf_counter_ns() - started
    assert final_loss is not None and final_output is not None
    trained_fingerprint = model_state_fingerprint(model)
    model.eval()
    return model, {
        "model_seed": config.model_seed,
        "data_seed": config.data_seed,
        "steps": config.steps,
        "initial_model_fingerprint": initial_fingerprint,
        "trained_model_fingerprint": trained_fingerprint,
        "fixed_train_batch_sha256": _batch_sha256(batch),
        "fixed_train_batch": _batch_identity_record(batch),
        "elapsed_training_ns": elapsed_ns,
        "optimizer": {
            "type": "torch.optim.AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
        },
        "loss_config": asdict(config.loss),
        "max_gradient_norm": config.max_gradient_norm,
        "final_loss": {
            "total": float(final_loss.total.detach()),
            "query": float(final_loss.query.detach()),
            "route_curriculum": float(final_loss.route_curriculum.detach()),
            "router_balance": float(final_loss.router_balance.detach()),
            "router_entropy": float(final_loss.router_entropy.detach()),
            "route_persistence": float(final_loss.route_persistence.detach()),
            "query_count": final_loss.query_count,
            "route_supervision_count": final_loss.route_supervision_count,
            "persistence_pair_count": final_loss.persistence_pair_count,
        },
        "final_guidance": {
            "probability": float(final_output.diagnostics["guidance_probability"]),
            "guided_events": int(final_output.diagnostics["guided_events"]),
            "guided_fraction": float(final_output.diagnostics["guided_fraction"]),
        },
    }


def _batch_identity_record(batch: BindingBatch) -> dict[str, object]:
    valid_counts = batch.inputs.valid_mask.sum(dim=1, dtype=torch.int64).tolist()
    return {
        "lengths": [int(value) for value in batch.lengths.tolist()],
        "valid_event_counts": [int(value) for value in valid_counts],
        "total_valid_events": int(batch.inputs.valid_mask.sum().item()),
        "document_ids": list(batch.document_ids),
        "generation_seeds": list(batch.generation_seeds),
        "splits": list(batch.splits),
        "task_fingerprint": batch.config_fingerprint,
    }


def _tensor_record(tensor: Tensor) -> dict[str, object]:
    record = {
        # Route logits deliberately contain -inf at disallowed classes.  Hash
        # their canonical bits rather than rejecting this executable sentinel.
        "sha256": _replay_tensor_hash(tensor),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }
    if tensor.is_floating_point():
        record["positive_infinity_count"] = int(torch.isposinf(tensor).sum().item())
        record["negative_infinity_count"] = int(torch.isneginf(tensor).sum().item())
        record["nan_count"] = int(torch.isnan(tensor).sum().item())
    return record


def _replay_tensor_hash(tensor: Tensor) -> str:
    """Reproduce the independent replay worker's domain-separated tensor hash."""

    if not isinstance(tensor, Tensor) or tensor.layout is not torch.strided:
        raise TypeError("replay-hashed values must be strided tensors")
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    if sys.byteorder != "little" and value.element_size() > 1:
        width = value.element_size()
        raw = b"".join(
            raw[index : index + width][::-1]
            for index in range(0, len(raw), width)
        )
    digest = hashlib.sha256()
    digest.update(_REPLAY_HASH_DOMAIN)
    dtype = str(value.dtype).encode("ascii")
    digest.update(struct.pack("<Q", len(dtype)))
    digest.update(dtype)
    digest.update(struct.pack("<Q", value.ndim))
    for dimension in value.shape:
        digest.update(struct.pack("<Q", int(dimension)))
    digest.update(raw)
    return digest.hexdigest()


def _replay_output_hashes(output: BindingModelOutput) -> dict[str, str]:
    return {
        "routes": _replay_tensor_hash(output.routes),
        "route_logits": _replay_tensor_hash(output.route_logits),
        "route_probabilities": _replay_tensor_hash(output.route_probabilities),
        "value_logits": _replay_tensor_hash(output.value_logits),
        **{
            f"forest_{name}": _replay_tensor_hash(getattr(output.forest_state, name))
            for name in _FOREST_FIELDS
        },
        **{
            f"router_{name}": _replay_tensor_hash(getattr(output.router_state, name))
            for name in _ROUTER_FIELDS
        },
        **{
            f"diagnostic:{name}": _replay_tensor_hash(value)
            for name, value in sorted(output.diagnostics.items())
        },
    }


def _query_metrics(output: BindingModelOutput, batch: BindingBatch) -> dict[str, object]:
    query = batch.inputs.valid_mask & (
        batch.inputs.event_kinds == int(BindingEventKind.QUERY)
    )
    heldout = query & batch.evaluation.heldout_combination_mask
    predicted = output.value_logits.argmax(dim=-1)

    def one(mask: Tensor) -> dict[str, object]:
        count = int(mask.sum().item())
        correct = int((predicted[mask] == batch.evaluation.targets[mask]).sum().item())
        return {"count": count, "correct": correct, "accuracy": correct / count if count else 0.0}

    return {"all": one(query), "seen": one(query & ~heldout), "heldout": one(heldout)}


def _output_record(
    output: BindingModelOutput,
    batch: BindingBatch,
    model: RoutedBindingModel,
) -> dict[str, object]:
    merge_count = int(output.diagnostics["forest_merge_count"].item())
    merge_metrics = model.forest.merge.structural_metrics(merge_count)
    return {
        "query_metrics": _query_metrics(output, batch),
        "structural_work": {
            "actual_executed_merge_count": merge_count,
            "operation_count_proxy_per_merge": merge_metrics[
                "operation_count_proxy_per_merge"
            ],
            "total_operation_count_proxy": merge_metrics["operation_count_proxy"],
        },
        "tensors": {name: _tensor_record(getattr(output, name)) for name in _OUTPUT_TENSORS},
        "forest_state": {
            name: _tensor_record(getattr(output.forest_state, name)) for name in _FOREST_FIELDS
        },
        "router_state": {
            name: _tensor_record(getattr(output.router_state, name)) for name in _ROUTER_FIELDS
        },
        "diagnostics": {
            name: _tensor_record(value) for name, value in sorted(output.diagnostics.items())
        },
    }


def _tensor_difference(
    expected: Tensor, actual: Tensor, *, rtol: float, atol: float
) -> dict[str, object]:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return {
            "passed": False,
            "shape_equal": expected.shape == actual.shape,
            "dtype_equal": expected.dtype == actual.dtype,
            "exact": False,
            "max_absolute_error": 0.0,
        }
    exact = torch.equal(expected, actual)
    if expected.is_floating_point():
        expected_finite = torch.isfinite(expected)
        actual_finite = torch.isfinite(actual)
        finite_mask_equal = torch.equal(expected_finite, actual_finite)
        infinity_equal = (
            torch.equal(torch.isposinf(expected), torch.isposinf(actual))
            and torch.equal(torch.isneginf(expected), torch.isneginf(actual))
        )
        nan_free = not bool(torch.isnan(expected).any()) and not bool(
            torch.isnan(actual).any()
        )
        if finite_mask_equal:
            finite_expected = expected[expected_finite]
            finite_actual = actual[actual_finite]
            difference = (finite_expected - finite_actual).abs()
            maximum = float(difference.max().item()) if difference.numel() else 0.0
            finite_close = bool(
                torch.allclose(finite_expected, finite_actual, rtol=rtol, atol=atol)
            )
        else:
            maximum = 0.0
            finite_close = False
        passed = finite_mask_equal and infinity_equal and nan_free and finite_close
    else:
        maximum = 0.0
        passed = exact
    result = {
        "passed": passed,
        "shape_equal": True,
        "dtype_equal": True,
        "exact": exact,
        "max_absolute_error": maximum,
    }
    if expected.is_floating_point():
        result.update(
            {
                "finite_mask_equal": finite_mask_equal,
                "infinity_sign_patterns_equal": infinity_equal,
                "expected_positive_infinity_count": int(
                    torch.isposinf(expected).sum().item()
                ),
                "actual_positive_infinity_count": int(
                    torch.isposinf(actual).sum().item()
                ),
                "expected_negative_infinity_count": int(
                    torch.isneginf(expected).sum().item()
                ),
                "actual_negative_infinity_count": int(
                    torch.isneginf(actual).sum().item()
                ),
                "expected_nan_count": int(torch.isnan(expected).sum().item()),
                "actual_nan_count": int(torch.isnan(actual).sum().item()),
            }
        )
    return result


def _compare_states(
    expected: object,
    actual: object,
    names: Iterable[str],
    *,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    comparisons = {
        name: _tensor_difference(
            getattr(expected, name), getattr(actual, name), rtol=rtol, atol=atol
        )
        for name in names
    }
    return {
        "passed": all(bool(value["passed"]) for value in comparisons.values()),
        "fields": comparisons,
    }


def _compare_outputs(
    expected: BindingModelOutput,
    actual: BindingModelOutput,
    *,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    tensors = {
        name: _tensor_difference(
            getattr(expected, name), getattr(actual, name), rtol=rtol, atol=atol
        )
        for name in _OUTPUT_TENSORS
    }
    diagnostics = {
        name: _tensor_difference(
            expected.diagnostics[name], actual.diagnostics[name], rtol=rtol, atol=atol
        )
        for name in sorted(expected.diagnostics)
        if name in actual.diagnostics
    }
    diagnostic_keys_equal = set(expected.diagnostics) == set(actual.diagnostics)
    forest = _compare_states(
        expected.forest_state, actual.forest_state, _FOREST_FIELDS, rtol=rtol, atol=atol
    )
    router = _compare_states(
        expected.router_state, actual.router_state, _ROUTER_FIELDS, rtol=rtol, atol=atol
    )
    return {
        "passed": (
            all(bool(value["passed"]) for value in tensors.values())
            and diagnostic_keys_equal
            and all(bool(value["passed"]) for value in diagnostics.values())
            and bool(forest["passed"])
            and bool(router["passed"])
        ),
        "tensors": tensors,
        "diagnostic_keys_equal": diagnostic_keys_equal,
        "diagnostics": diagnostics,
        "forest_state": forest,
        "router_state": router,
    }


@torch.no_grad()
def _evaluate_variants(
    variants: Mapping[str, RoutedBindingModel],
    batch: BindingBatch,
    audit: ExportAuditConfig,
) -> tuple[dict[str, dict[str, BindingModelOutput]], dict[str, object]]:
    outputs: dict[str, dict[str, BindingModelOutput]] = {}
    records: dict[str, object] = {}
    for name, model in variants.items():
        model.eval()
        by_implementation = {
            implementation: model(batch.inputs, implementation=implementation)
            for implementation in ("streaming", "parallel")
        }
        outputs[name] = by_implementation
        records[name] = {
            "model_fingerprint": model_state_fingerprint(model),
            "streaming": _output_record(
                by_implementation["streaming"], batch, model
            ),
            "parallel": _output_record(by_implementation["parallel"], batch, model),
            "streaming_parallel_parity": _compare_outputs(
                by_implementation["streaming"],
                by_implementation["parallel"],
                rtol=audit.float32_rtol,
                atol=audit.float32_atol,
            ),
        }
    records["dense_compact_parity"] = {
        implementation: _compare_outputs(
            outputs["dense_selected"][implementation],
            outputs["compact"][implementation],
            rtol=audit.float32_rtol,
            atol=audit.float32_atol,
        )
        for implementation in ("streaming", "parallel")
    }
    return outputs, records


def _occupancy_matches_count(state: ForestState, row: int, lane: int) -> bool:
    count = int(state.counts[row, lane].item())
    expected = torch.tensor(
        [bool((count >> scale) & 1) for scale in range(state.scales)],
        dtype=torch.bool,
        device=state.occupied.device,
    )
    return torch.equal(state.occupied[row, lane], expected)


def _trim_forest_capacity(state: ForestState, scales: int) -> ForestState:
    """Trim only the all-empty capacity tail used by packed prefix states."""

    if not 1 <= scales <= state.scales:
        raise ValueError("trimmed forest scale count is out of range")
    return ForestState(
        slots=state.slots[:, :, :scales],
        occupied=state.occupied[:, :, :scales],
        counts=state.counts,
        valid_steps=state.valid_steps,
    )


@torch.no_grad()
def _forced_carry_probe(
    dense: RoutedBindingModel,
    compact: RoutedBindingModel,
    *,
    rtol: float,
    atol: float,
    event_vectors: Tensor | None = None,
) -> dict[str, object]:
    """Force actual lane counts over 2^k carry boundaries, local and global.

    When supplied, ``event_vectors`` are real encoder outputs from the bound
    evaluation fixture.  The routes are deliberately forced so the audit does
    not confuse sequence length with the number of updates to one lane.
    """

    result: dict[str, object] = {}
    maximum = 64
    if event_vectors is None:
        base = torch.arange(maximum * compact.config.d_model, dtype=torch.float32)
        events = ((base.reshape(1, maximum, compact.config.d_model) % 37) - 18) / 37
        event_source = "deterministic_synthetic_test_fallback"
    else:
        if (
            not isinstance(event_vectors, Tensor)
            or event_vectors.ndim != 2
            or event_vectors.shape[1] != compact.config.d_model
            or event_vectors.shape[0] < maximum
            or not event_vectors.is_floating_point()
            or not bool(torch.isfinite(event_vectors[:maximum]).all())
        ):
            raise ValueError("event_vectors must contain at least 64 finite [D] vectors")
        events = event_vectors[:maximum].detach().reshape(
            1, maximum, compact.config.d_model
        )
        event_source = "bound_evaluation_fixture_encoder_outputs"
    reference = next(compact.parameters())
    events = events.to(device=reference.device, dtype=reference.dtype)
    result["event_source"] = event_source
    result["event_vectors_sha256"] = tensor_sha256(events)
    for lane_name, lane in (("local", 0), ("global", compact.config.task.branches)):
        routes = torch.full(
            (1, maximum), lane, dtype=torch.int64, device=events.device
        )
        valid = torch.ones(1, maximum, dtype=torch.bool, device=events.device)
        dense_state = dense.forest.initial_state(
            1, device=events.device, dtype=events.dtype
        )
        compact_state = compact.forest.initial_state(
            1, device=events.device, dtype=events.dtype
        )
        dense_prefix_states: dict[int, ForestState] = {}
        compact_prefix_states: dict[int, ForestState] = {}
        boundaries: list[dict[str, object]] = []
        for index in range(maximum):
            before = int(compact_state.counts[0, lane].item())
            dense_run = dense.forest.step(
                dense_state, events[:, index], routes[:, index], valid[:, index]
            )
            compact_run = compact.forest.step(
                compact_state, events[:, index], routes[:, index], valid[:, index]
            )
            dense_state, compact_state = dense_run.state, compact_run.state
            after = int(compact_state.counts[0, lane].item())
            if after in (16, 32, 64):
                dense_prefix_states[after] = dense_state
                compact_prefix_states[after] = compact_state
                expected_depth = after.bit_length() - 1
                parity = _compare_states(
                    dense_state,
                    compact_state,
                    _FOREST_FIELDS,
                    rtol=rtol,
                    atol=atol,
                )
                boundaries.append(
                    {
                        "pre_update_lane_count": before,
                        "post_update_lane_count": after,
                        "expected_pre_update_lane_count": after - 1,
                        "actual_carry_boundary": before == after - 1,
                        "expected_merge_depth": expected_depth,
                        "dense_step_merge_depth": int(dense_run.merge_count.item()),
                        "compact_step_merge_depth": int(compact_run.merge_count.item()),
                        "occupancy_matches_binary_count": (
                            _occupancy_matches_count(dense_state, 0, lane)
                            and _occupancy_matches_count(compact_state, 0, lane)
                        ),
                        "dense_compact_state_parity": parity,
                    }
                )
        dense_parallel = dense.forest.reduce_parallel_prefixes(events, routes, valid)
        compact_parallel = compact.forest.reduce_parallel_prefixes(events, routes, valid)
        for item in boundaries:
            count = int(item["post_update_lane_count"])
            dense_parallel_state = _trim_forest_capacity(
                dense_parallel.states[count - 1], dense_prefix_states[count].scales
            )
            compact_parallel_state = _trim_forest_capacity(
                compact_parallel.states[count - 1], compact_prefix_states[count].scales
            )
            item["dense_streaming_parallel_prefix_parity"] = _compare_states(
                dense_prefix_states[count],
                dense_parallel_state,
                _FOREST_FIELDS,
                rtol=rtol,
                atol=atol,
            )
            item["compact_streaming_parallel_prefix_parity"] = _compare_states(
                compact_prefix_states[count],
                compact_parallel_state,
                _FOREST_FIELDS,
                rtol=rtol,
                atol=atol,
            )
            item["dense_compact_parallel_prefix_parity"] = _compare_states(
                dense_parallel_state,
                compact_parallel_state,
                _FOREST_FIELDS,
                rtol=rtol,
                atol=atol,
            )
        result[lane_name] = {
            "lane": lane,
            "dense_parallel_total_merge_count": int(dense_parallel.merge_count.item()),
            "compact_parallel_total_merge_count": int(compact_parallel.merge_count.item()),
            "boundaries": boundaries,
            "passed": len(boundaries) == 3
            and all(
                bool(item["actual_carry_boundary"])
                and item["dense_step_merge_depth"] == item["expected_merge_depth"]
                and item["compact_step_merge_depth"] == item["expected_merge_depth"]
                and bool(item["occupancy_matches_binary_count"])
                and bool(item["dense_compact_state_parity"]["passed"])
                and bool(item["dense_streaming_parallel_prefix_parity"]["passed"])
                and bool(item["compact_streaming_parallel_prefix_parity"]["passed"])
                and bool(item["dense_compact_parallel_prefix_parity"]["passed"])
                for item in boundaries
            )
            and int(dense_parallel.merge_count.item()) == maximum - 1
            and int(compact_parallel.merge_count.item()) == maximum - 1,
        }
    result["passed"] = all(bool(result[name]["passed"]) for name in ("local", "global"))
    return result


def _slice_inputs(inputs: BindingModelInputs, start: int, end: int) -> BindingModelInputs:
    return BindingModelInputs(
        **{
            field.name: getattr(inputs, field.name)[:, start:end]
            for field in fields(BindingModelInputs)
        }
    )


def _allowed_classes(model: RoutedBindingModel, inputs: BindingModelInputs) -> Tensor:
    local = inputs.valid_mask & (inputs.primary_key_ids > 0)
    allowed = torch.zeros(
        *inputs.valid_mask.shape,
        model.router.class_count,
        dtype=torch.bool,
        device=inputs.valid_mask.device,
    )
    allowed[:, :, : model.config.task.branches] = local.unsqueeze(-1)
    allowed[:, :, model.config.task.branches :] = (
        inputs.valid_mask & ~local
    ).unsqueeze(-1)
    return allowed


def _router_state_comparison(
    expected: PersistentRouterState,
    actual: PersistentRouterState,
    *,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    return _compare_states(expected, actual, _ROUTER_FIELDS, rtol=rtol, atol=atol)


@torch.no_grad()
def _forest_resume_probe(
    model: RoutedBindingModel,
    batch: BindingBatch,
    *,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    """Resume a compact low-level run after codec round-trip at midstream."""

    model.eval()
    inputs = batch.inputs
    width = inputs.valid_mask.shape[1]
    midpoint = width // 2
    prefix_inputs = _slice_inputs(inputs, 0, midpoint)
    suffix_inputs = _slice_inputs(inputs, midpoint, width)

    full_events, full_features = model.encoder(inputs)
    full_router = model.router(
        full_features,
        inputs.valid_mask,
        allowed_classes=_allowed_classes(model, inputs),
    )
    prefix_events, prefix_features = model.encoder(prefix_inputs)
    prefix_router = model.router(
        prefix_features,
        prefix_inputs.valid_mask,
        allowed_classes=_allowed_classes(model, prefix_inputs),
    )
    suffix_events, suffix_features = model.encoder(suffix_inputs)
    suffix_router = model.router(
        suffix_features,
        suffix_inputs.valid_mask,
        allowed_classes=_allowed_classes(model, suffix_inputs),
        initial_state=prefix_router.final_state,
    )
    concatenated_routes = torch.cat((prefix_router.routes, suffix_router.routes), dim=1)
    concatenated_logits = torch.cat((prefix_router.logits, suffix_router.logits), dim=1)
    concatenated_probabilities = torch.cat(
        (prefix_router.probabilities, suffix_router.probabilities), dim=1
    )
    router_trace = {
        "routes": _tensor_difference(full_router.routes, concatenated_routes, rtol=rtol, atol=atol),
        "logits": _tensor_difference(full_router.logits, concatenated_logits, rtol=rtol, atol=atol),
        "probabilities": _tensor_difference(
            full_router.probabilities, concatenated_probabilities, rtol=rtol, atol=atol
        ),
        "final_state": _router_state_comparison(
            full_router.final_state, suffix_router.final_state, rtol=rtol, atol=atol
        ),
    }

    full_strengths = model._route_strengths(full_router)
    prefix_strengths = model._route_strengths(prefix_router)
    suffix_strengths = model._route_strengths(suffix_router)
    full_forest = model.forest.reduce_streaming(
        full_events * full_strengths.unsqueeze(-1),
        full_router.routes,
        inputs.valid_mask,
    )
    prefix_forest = model.forest.reduce_streaming(
        prefix_events * prefix_strengths.unsqueeze(-1),
        prefix_router.routes,
        prefix_inputs.valid_mask,
    )
    state_blob = serialize_forest_state(
        prefix_forest.state, model.config.fingerprint()
    )
    restored = deserialize_forest_state(
        state_blob, expected_config_fingerprint=model.config.fingerprint()
    )
    resumed = model.forest.reduce_streaming(
        suffix_events * suffix_strengths.unsqueeze(-1),
        suffix_router.routes,
        suffix_inputs.valid_mask,
        initial_state=restored,
    )

    state = restored
    suffix_logits: list[Tensor] = []
    for index in range(suffix_events.shape[1]):
        step = model.forest.step(
            state,
            suffix_events[:, index],
            suffix_router.routes[:, index],
            suffix_inputs.valid_mask[:, index],
            route_strength=suffix_strengths[:, index],
        )
        state = step.state
        current = model.readout(state, suffix_events[:, index])
        suffix_logits.append(
            torch.where(
                suffix_inputs.valid_mask[:, index, None],
                current,
                torch.zeros_like(current),
            )
        )
    resumed_logits = torch.stack(suffix_logits, dim=1)
    full_output = model(inputs, implementation="streaming")
    forest_comparison = _compare_states(
        full_forest.state, resumed.state, _FOREST_FIELDS, rtol=rtol, atol=atol
    )
    loop_comparison = _compare_states(
        resumed.state, state, _FOREST_FIELDS, rtol=rtol, atol=atol
    )
    suffix_comparison = _tensor_difference(
        full_output.value_logits[:, midpoint:], resumed_logits, rtol=rtol, atol=atol
    )
    passed = (
        all(bool(value["passed"]) for value in router_trace.values())
        and bool(forest_comparison["passed"])
        and bool(loop_comparison["passed"])
        and bool(suffix_comparison["passed"])
    )
    return {
        "passed": passed,
        "midpoint_tensor_position": midpoint,
        "forest_state_blob_bytes": len(state_blob),
        "forest_state_blob_sha256": _sha256(state_blob),
        "prefix_state_sha256": forest_state_sha256(prefix_forest.state),
        "restored_state_sha256": forest_state_sha256(restored),
        "router_trace_and_final_state": router_trace,
        "uninterrupted_resumed_forest_state": forest_comparison,
        "resumed_reduce_step_loop_state": loop_comparison,
        "suffix_value_logits": suffix_comparison,
    }


@torch.no_grad()
def _forced_forest_resume_probe(
    model: RoutedBindingModel,
    event_vectors: Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    """Round-trip forest state immediately before forced carry boundaries."""

    if (
        not isinstance(event_vectors, Tensor)
        or event_vectors.ndim != 2
        or event_vectors.shape[0] < 64
        or event_vectors.shape[1] != model.config.d_model
    ):
        raise ValueError("forced resume requires at least 64 encoded event vectors")
    reference = next(model.parameters())
    events = event_vectors[:64].detach().reshape(1, 64, model.config.d_model).to(
        device=reference.device, dtype=reference.dtype
    )
    valid = torch.ones(1, 64, dtype=torch.bool, device=events.device)
    result: dict[str, object] = {
        "event_vectors_sha256": tensor_sha256(events),
        "cuts_are_lane_update_counts": True,
    }
    fingerprint = model.config.fingerprint()
    for lane_name, lane in (("local", 0), ("global", model.config.task.branches)):
        routes = torch.full((1, 64), lane, dtype=torch.int64, device=events.device)
        uninterrupted = model.forest.reduce_streaming(events, routes, valid)
        cuts: list[dict[str, object]] = []
        for cut in (15, 31, 63):
            prefix = model.forest.reduce_streaming(
                events[:, :cut], routes[:, :cut], valid[:, :cut]
            )
            if int(prefix.state.counts[0, lane].item()) != cut:
                raise RuntimeError("forced resume prefix did not reach the declared lane count")
            first_blob = serialize_forest_state(prefix.state, fingerprint)
            second_blob = serialize_forest_state(prefix.state, fingerprint)
            restored = deserialize_forest_state(
                first_blob, expected_config_fingerprint=fingerprint
            )
            restored_blob = serialize_forest_state(restored, fingerprint)
            immediate = model.forest.step(
                restored,
                events[:, cut],
                routes[:, cut],
                valid[:, cut],
            )
            expected_depth = (cut + 1).bit_length() - 1
            resumed = model.forest.reduce_streaming(
                events[:, cut:],
                routes[:, cut:],
                valid[:, cut:],
                initial_state=restored,
            )
            final_comparison = _compare_states(
                uninterrupted.state,
                resumed.state,
                _FOREST_FIELDS,
                rtol=rtol,
                atol=atol,
            )
            cuts.append(
                {
                    "pre_carry_lane_update_count": cut,
                    "next_lane_update_count": cut + 1,
                    "next_expected_merge_depth": expected_depth,
                    "next_actual_merge_depth": int(immediate.merge_count.item()),
                    "state_blob_bytes": len(first_blob),
                    "state_blob_sha256": _sha256(first_blob),
                    "serialization_deterministic": first_blob == second_blob,
                    "restored_reserializes_identically": first_blob == restored_blob,
                    "prefix_restored_hash_equal": forest_state_sha256(prefix.state)
                    == forest_state_sha256(restored),
                    "uninterrupted_resumed_final_state": final_comparison,
                    "passed": (
                        first_blob == second_blob
                        and first_blob == restored_blob
                        and forest_state_sha256(prefix.state)
                        == forest_state_sha256(restored)
                        and int(immediate.merge_count.item()) == expected_depth
                        and bool(final_comparison["passed"])
                    ),
                }
            )
        result[lane_name] = {
            "lane": lane,
            "cuts": cuts,
            "passed": all(bool(item["passed"]) for item in cuts),
        }
    result["passed"] = all(bool(result[name]["passed"]) for name in ("local", "global"))
    return result


def _fixture_bytes(inputs: BindingModelInputs) -> bytes:
    encoded: dict[str, object] = {}
    for field in fields(BindingModelInputs):
        tensor = getattr(inputs, field.name).detach().cpu()
        encoded[field.name] = {
            "dtype": "bool" if tensor.dtype is torch.bool else "int64",
            "shape": list(tensor.shape),
            "values": tensor.tolist(),
        }
    value = {"schema_version": 1, "inputs": encoded}
    validate_finite_json(value)
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _worker_environment(repository_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str((repository_root / "v3" / "src").resolve())
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _run_replay_worker(
    worker: Path,
    *,
    repository_root: Path,
    artifact: Path,
    fixture: Path,
    output: Path,
    artifact_sha256: str,
    fixture_sha256: str,
    source_fingerprint: str,
    manifest_fingerprint: str,
    selection_fingerprint: str,
    code_commit: str,
    code_tree: str,
    worker_sha256: str,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-s",
        "-B",
        str(worker),
        "--artifact",
        str(artifact),
        "--fixture",
        str(fixture),
        "--output",
        str(output),
        "--expected-artifact-sha256",
        artifact_sha256,
        "--expected-fixture-sha256",
        fixture_sha256,
        "--expected-source-fingerprint",
        source_fingerprint,
        "--expected-manifest-fingerprint",
        manifest_fingerprint,
        "--expected-selection-fingerprint",
        selection_fingerprint,
        "--expected-code-commit",
        code_commit,
        "--expected-code-tree",
        code_tree,
        "--expected-worker-sha256",
        worker_sha256,
    ]
    completed = subprocess.run(
        command,
        cwd=repository_root / "v3",
        env=_worker_environment(repository_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
        check=False,
    )
    if not output.is_file():
        raise RuntimeError("fresh replay worker did not write its result")
    try:
        record = json.loads(output.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("fresh replay worker wrote invalid JSON") from error
    if completed.returncode != 0 or not isinstance(record, dict) or record.get("status") != "passed":
        raise RuntimeError(
            "fresh replay failed: " + (completed.stderr.strip() or repr(record))
        )
    validate_finite_json(record)
    return record


def _runtime_worker_command(
    worker: Path,
    *,
    variant: str,
    fixture: Path,
    output: Path,
    fixture_sha256: str,
    model_fingerprint: str,
    code_commit: str,
    code_tree: str,
    warmup_iterations: int,
    timed_iterations: int,
    torch_threads: int,
    extra_arguments: Iterable[str],
) -> list[str]:
    """Single integration point for the separately isolated runtime worker."""

    return [
        sys.executable,
        "-s",
        "-B",
        str(worker.resolve()),
        "--variant",
        variant,
        "--fixture",
        str(fixture.resolve()),
        "--output",
        str(output.resolve()),
        "--expected-fixture-sha256",
        fixture_sha256,
        "--expected-model-fingerprint",
        model_fingerprint,
        "--expected-code-commit",
        code_commit,
        "--expected-code-tree",
        code_tree,
        "--warmup-iterations",
        str(warmup_iterations),
        "--timed-iterations",
        str(timed_iterations),
        "--torch-threads",
        str(torch_threads),
        "--rss-sample-period-ms",
        "1.0",
        *extra_arguments,
    ]


def _single_row_inputs(
    inputs: BindingModelInputs, row: int, length: int
) -> BindingModelInputs:
    return BindingModelInputs(
        **{
            field.name: getattr(inputs, field.name)[row : row + 1, :length]
            for field in fields(BindingModelInputs)
        }
    )


def _checkpoint_bytes(model: RoutedBindingModel) -> bytes:
    """Create a weights-only measurement checkpoint (not the release artifact)."""

    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    buffer = io.BytesIO()
    torch.save({"state_dict": state}, buffer)
    return buffer.getvalue()


def _runtime_output_evidence(output: BindingModelOutput) -> dict[str, str]:
    """Match the isolated runtime worker's raw CPU output digest contract."""

    return {
        "routes_sha256": _sha256(
            output.routes.detach().cpu().contiguous().numpy().tobytes()
        ),
        "value_logits_sha256": _sha256(
            output.value_logits.detach().cpu().contiguous().numpy().tobytes()
        ),
    }


def _run_runtime_worker(
    command: list[str],
    *,
    repository_root: Path,
    output: Path,
    expected_variant: str,
    expected_fixture_sha256: str,
    expected_model_fingerprint: str,
    expected_output_evidence: Mapping[str, str],
    expected_code_commit: str,
    expected_code_tree: str,
    expected_worker_path: Path,
    expected_worker_sha256: str,
    expected_package_path: Path,
    expected_package_sha256: str,
    expected_warmup_iterations: int,
    expected_timed_iterations: int,
    expected_torch_threads: int,
    expected_rss_sample_period_ms: float,
) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=repository_root / "v3",
        env=_worker_environment(repository_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
        check=False,
    )
    if not output.is_file():
        raise RuntimeError("isolated runtime worker did not write its result")
    try:
        record = json.loads(output.read_text(encoding="utf-8"))
        stdout_record = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("isolated runtime worker returned invalid JSON") from error
    if (
        completed.returncode != 0
        or not isinstance(record, dict)
        or record != stdout_record
        or record.get("status") != "passed"
    ):
        raise RuntimeError(
            "isolated runtime measurement failed: "
            + (completed.stderr.strip() or repr(record))
        )
    validate_finite_json(record)
    checkout = record.get("checkout")
    fixture = record.get("fixture")
    model = record.get("model")
    measurement = record.get("measurement")
    if not all(isinstance(value, dict) for value in (checkout, fixture, model, measurement)):
        raise RuntimeError("isolated runtime record is missing provenance or measurement")
    assert isinstance(checkout, dict)
    assert isinstance(fixture, dict)
    assert isinstance(model, dict)
    assert isinstance(measurement, dict)
    samples = measurement.get("elapsed_ns_samples")
    rss_values = (
        measurement.get("loaded_rss_bytes"),
        measurement.get("warmed_rss_bytes"),
        measurement.get("sampled_peak_rss_bytes"),
        measurement.get("incremental_sampled_peak_bytes"),
    )
    trusted = (
        record.get("variant") == expected_variant
        and record.get("process_id") != os.getpid()
        and checkout.get("code_commit") == expected_code_commit
        and checkout.get("code_tree") == expected_code_tree
        and checkout.get("worker_file") == str(expected_worker_path.resolve())
        and checkout.get("worker_file_sha256") == expected_worker_sha256
        and checkout.get("package_file") == str(expected_package_path.resolve())
        and checkout.get("package_file_sha256") == expected_package_sha256
        and checkout.get("worktree_clean") is True
        and fixture.get("sha256") == expected_fixture_sha256
        and model.get("model_fingerprint") == expected_model_fingerprint
        and record.get("output_evidence") == dict(expected_output_evidence)
        and measurement.get("warmup_iterations") == expected_warmup_iterations
        and measurement.get("timed_iterations") == expected_timed_iterations
        and measurement.get("torch_threads") == expected_torch_threads
        and measurement.get("rss_sample_period_ms")
        == expected_rss_sample_period_ms
        and isinstance(samples, list)
        and len(samples) == expected_timed_iterations
        and all(type(value) is int and value > 0 for value in samples)
        and all(type(value) is int and value >= 0 for value in rss_values)
        and measurement.get("loaded_rss_bytes", 0) > 0
        and measurement.get("warmed_rss_bytes", 0) > 0
        and measurement.get("sampled_peak_rss_bytes", 0)
        >= max(
            measurement.get("loaded_rss_bytes", 0),
            measurement.get("warmed_rss_bytes", 0),
        )
        and measurement.get("incremental_sampled_peak_bytes")
        == max(
            0,
            measurement.get("sampled_peak_rss_bytes", 0)
            - measurement.get("warmed_rss_bytes", 0),
        )
    )
    if not trusted:
        raise RuntimeError("isolated runtime record failed trusted provenance checks")
    result_bytes = output.read_bytes()
    return {
        "status": "passed",
        "result_sha256": _sha256(result_bytes),
        "process_id": record["process_id"],
        "median_elapsed_ns": float(statistics.median(samples)),
        "sampled_peak_rss_bytes": measurement["sampled_peak_rss_bytes"],
        "incremental_sampled_peak_bytes": measurement[
            "incremental_sampled_peak_bytes"
        ],
        "parent_expected_output_evidence": dict(expected_output_evidence),
        "record": record,
    }


def _runtime_matrix(
    worker: Path,
    *,
    repository_root: Path,
    runtime_directory: Path,
    source: RoutedBindingModel,
    dense: RoutedBindingModel,
    compact: RoutedBindingModel,
    artifact: Path,
    artifact_sha256: str,
    source_fingerprint: str,
    manifest_fingerprint: str,
    selection_fingerprint: str,
    evaluation: BindingBatch,
    lengths: tuple[int, ...],
    code_commit: str,
    code_tree: str,
    audit: ExportAuditConfig,
) -> dict[str, object]:
    """Measure every variant/length in a separate isolated worker process."""

    runtime_directory.mkdir(parents=True, exist_ok=True)
    worker_path = worker.resolve()
    worker_sha256 = _sha256(worker_path.read_bytes())
    package_path = Path(tnlm_v3.__file__).resolve()
    package_sha256 = _sha256(package_path.read_bytes())
    variant_models = {"source": source, "dense_selected": dense, "compact": compact}
    support: dict[str, dict[str, object]] = {}
    extra_by_variant: dict[str, tuple[str, ...]] = {}
    for name in ("source", "dense_selected"):
        model = variant_models[name]
        checkpoint = runtime_directory / f"{name}.checkpoint.pt"
        config_path = runtime_directory / f"{name}.model-config.json"
        checkpoint_data = _checkpoint_bytes(model)
        config_data = model.config.canonical_json().encode("utf-8")
        _atomic_bytes(checkpoint, checkpoint_data)
        _atomic_bytes(config_path, config_data)
        checkpoint_hash = _sha256(checkpoint_data)
        support[name] = {
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_bytes": len(checkpoint_data),
            "model_config_sha256": _sha256(config_data),
            "model_config_bytes": len(config_data),
        }
        extra_by_variant[name] = (
            "--checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            checkpoint_hash,
            "--model-config-json",
            str(config_path),
        )
    extra_by_variant["compact"] = (
        "--artifact",
        str(artifact),
        "--expected-artifact-sha256",
        artifact_sha256,
        "--expected-source-fingerprint",
        source_fingerprint,
        "--expected-manifest-fingerprint",
        manifest_fingerprint,
        "--expected-selection-fingerprint",
        selection_fingerprint,
    )
    support["compact"] = {
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": artifact.stat().st_size,
    }

    by_length: dict[str, object] = {}
    process_ids: list[int] = []
    for row, length in enumerate(lengths):
        single_inputs = _single_row_inputs(evaluation.inputs, row, length)
        fixture_data = _fixture_bytes(single_inputs)
        fixture_hash = _sha256(fixture_data)
        fixture_path = runtime_directory / f"length-{length}.fixture.json"
        _atomic_bytes(fixture_path, fixture_data)
        variants: dict[str, object] = {}
        for name, model in variant_models.items():
            with torch.inference_mode():
                parent_output = model(single_inputs, implementation="streaming")
            expected_output_evidence = _runtime_output_evidence(parent_output)
            measurement_output = runtime_directory / f"length-{length}.{name}.json"
            command = _runtime_worker_command(
                worker,
                variant=name,
                fixture=fixture_path,
                output=measurement_output,
                fixture_sha256=fixture_hash,
                model_fingerprint=model_state_fingerprint(model),
                code_commit=code_commit,
                code_tree=code_tree,
                warmup_iterations=audit.warmup_iterations,
                timed_iterations=audit.timed_iterations,
                torch_threads=audit.torch_threads,
                extra_arguments=extra_by_variant[name],
            )
            measured = _run_runtime_worker(
                command,
                repository_root=repository_root,
                output=measurement_output,
                expected_variant=name,
                expected_fixture_sha256=fixture_hash,
                expected_model_fingerprint=model_state_fingerprint(model),
                expected_output_evidence=expected_output_evidence,
                expected_code_commit=code_commit,
                expected_code_tree=code_tree,
                expected_worker_path=worker_path,
                expected_worker_sha256=worker_sha256,
                expected_package_path=package_path,
                expected_package_sha256=package_sha256,
                expected_warmup_iterations=audit.warmup_iterations,
                expected_timed_iterations=audit.timed_iterations,
                expected_torch_threads=audit.torch_threads,
                expected_rss_sample_period_ms=1.0,
            )
            process_ids.append(int(measured["process_id"]))
            variants[name] = measured
        by_length[str(length)] = {
            "declared_real_length": length,
            "fixture_sha256": fixture_hash,
            "fixture_bytes": len(fixture_data),
            "variants": variants,
        }
    expected_processes = len(lengths) * len(variant_models)
    return {
        "status": "passed",
        "scope": "isolated_cpu_runtime_and_sampled_rss_descriptive_no_quality_claim",
        "implementation": "streaming",
        "separate_worker_invocation_per_variant_and_length": True,
        "runtime_worker_sha256": worker_sha256,
        "package_init_sha256": package_sha256,
        "runtime_win_gate": False,
        "rss_win_gate": False,
        "support_files": support,
        "by_length": by_length,
        "process_count": len(process_ids),
        "process_ids_unique": len(set(process_ids)) == expected_processes,
        "all_workers_passed": len(process_ids) == expected_processes,
    }


def _structural_record(model: RoutedBindingModel) -> dict[str, object]:
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "raw_tensor_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()
        ),
        "cp_rank": model.config.cp_rank,
        "operation_count_proxy_per_merge": model.forest.merge.structural_metrics(1)[
            "operation_count_proxy_per_merge"
        ],
        "state_d_model": model.config.d_model,
        "state_paths": model.forest.paths,
        "state_scalars_per_occupied_slot": model.config.d_model,
    }


def _cp_axis_record(
    source: RoutedBindingModel,
    compact: RoutedBindingModel,
) -> dict[str, object]:
    source_state = source.state_dict()
    compact_state = compact.state_dict()
    tensors: dict[str, object] = {}
    for name in _CP_STATE_NAMES:
        if name not in source_state or name not in compact_state:
            raise RuntimeError(f"missing declared CP-axis tensor {name}")
        source_shape = list(source_state[name].shape)
        compact_shape = list(compact_state[name].shape)
        rank_axis = 1 if name == "forest.merge.output.weight" else 0
        tensors[name] = {
            "rank_axis": rank_axis,
            "source_shape": source_shape,
            "compact_shape": compact_shape,
            "source_rank_extent": source_shape[rank_axis],
            "compact_rank_extent": compact_shape[rank_axis],
            "physically_sliced": (
                source_shape[rank_axis] == source.config.cp_rank
                and compact_shape[rank_axis] == compact.config.cp_rank
                and compact_shape[rank_axis] < source_shape[rank_axis]
            ),
        }
    source_keys = set(source_state)
    compact_keys = set(compact_state)
    extras = sorted(compact_keys - source_keys)
    suspicious = sorted(
        name
        for name in compact_keys
        if any(
            marker in name.lower()
            for marker in ("rank_mask", "retained_indices", "original_rank", "nominal_rank")
        )
    )
    result = {
        "five_cp_tensors": tensors,
        "source_compact_state_key_sets_equal": source_keys == compact_keys,
        "compact_extra_state_keys": extras,
        "rank_mask_or_original_rank_state_keys": suspicious,
    }
    result["passed"] = (
        len(tensors) == 5
        and all(bool(value["physically_sliced"]) for value in tensors.values())
        and source_keys == compact_keys
        and not extras
        and not suspicious
    )
    return result


def _artifact_roundtrip(
    compact: RoutedBindingModel,
    manifest,
    selection,
) -> tuple[bytes, dict[str, object]]:
    """Prove deterministic serialization and trusted load/re-save identity."""

    first = serialize_compact_binding_model(compact, manifest, selection)
    second = serialize_compact_binding_model(compact, manifest, selection)
    loaded, loaded_manifest, loaded_selection = deserialize_compact_binding_model(
        first,
        expected_source_fingerprint=manifest.source_model_fingerprint,
        expected_manifest_fingerprint=manifest.fingerprint(),
        expected_selection_fingerprint=selection.fingerprint(),
        device="cpu",
    )
    third = serialize_compact_binding_model(
        loaded, loaded_manifest, loaded_selection
    )
    model_equal = model_state_fingerprint(loaded) == model_state_fingerprint(compact)
    manifest_equal = loaded_manifest == manifest
    selection_equal = loaded_selection == selection
    record = {
        "first_sha256": _sha256(first),
        "second_sha256": _sha256(second),
        "trusted_roundtrip_sha256": _sha256(third),
        "serialized_twice_byte_identical": first == second,
        "trusted_roundtrip_byte_identical": first == third,
        "loaded_model_fingerprint_equal": model_equal,
        "loaded_manifest_equal": manifest_equal,
        "loaded_selection_equal": selection_equal,
        "passed": first == second
        and first == third
        and model_equal
        and manifest_equal
        and selection_equal,
    }
    return first, record


def _execute(args: argparse.Namespace) -> Path:
    repository_root = Path(__file__).resolve().parents[2]
    paths = _strict_external_paths(
        repository_root,
        {
            "output": args.output,
            "artifact": args.artifact,
            "fixture": args.fixture,
            "replay_output": args.replay_output,
            "runtime_directory": args.runtime_directory,
        },
    )
    output = paths["output"]
    record: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "status": "initializing",
        "scope": "deterministic_milestone3_implementation_audit_not_scientific_campaign",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_code_commit": args.code_commit,
    }
    durable = atomic_write_json(output, record)
    try:
        _strict_sha(args.code_commit, "--code-commit", length=40)
        code_commit, code_tree = _bind_to_clean_checkout(args.code_commit)
        audit_path = _require_committed_file(
            Path(args.audit_config), repository_root, "audit config"
        )
        replay_worker = _require_committed_file(
            repository_root / "v3" / "scripts" / "replay_compact_artifact.py",
            repository_root,
            "fresh replay worker",
        )
        runtime_worker = _require_committed_file(
            repository_root / "v3" / "scripts" / "measure_milestone3_runtime.py",
            repository_root,
            "isolated runtime worker",
        )
        runner_path = Path(__file__).resolve()
        package_file = Path(tnlm_v3.__file__).resolve()
        runner_snapshot = runner_path.read_bytes()
        package_snapshot = package_file.read_bytes()
        audit_snapshot = audit_path.read_bytes()
        replay_worker_snapshot = replay_worker.read_bytes()
        runtime_worker_snapshot = runtime_worker.read_bytes()
        audit = load_export_audit_config(audit_path)
        source_path = _require_committed_file(
            repository_root / audit.source_config, repository_root, "source config"
        )
        source_snapshot = source_path.read_bytes()
        source_config = load_binding_experiment_config(source_path)
        if audit_path.read_bytes() != audit_snapshot or source_path.read_bytes() != source_snapshot:
            raise RuntimeError("a configuration changed while it was parsed")
        if torch.get_num_threads() != audit.torch_threads:
            torch.set_num_threads(audit.torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            if torch.get_num_interop_threads() != 1:
                raise RuntimeError(
                    "Milestone-3 audit requires one Torch interop thread"
                ) from error
        torch.use_deterministic_algorithms(True)

        record.update(
            {
                "status": "in_progress",
                "code_commit": code_commit,
                "code_tree": code_tree,
                "configuration": {
                    "audit_path": audit_path.relative_to(repository_root).as_posix(),
                    "audit_sha256": _sha256(audit_snapshot),
                    "audit_fingerprint": audit.fingerprint(),
                    "source_path": source_path.relative_to(repository_root).as_posix(),
                    "source_sha256": _sha256(source_snapshot),
                    "source_fingerprint": source_config.fingerprint(),
                    "replay_worker_sha256": _sha256(replay_worker_snapshot),
                    "runtime_worker_sha256": _sha256(runtime_worker_snapshot),
                    "runner_sha256": _sha256(runner_snapshot),
                    "package_init_sha256": _sha256(package_snapshot),
                },
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "torch_threads": torch.get_num_threads(),
                    "torch_interop_threads": torch.get_num_interop_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cuda_available": torch.cuda.is_available(),
                },
            }
        )
        durable = atomic_write_json(output, record)

        source, training = _train_source(source_config)
        record.update(
            {
                "phase": "source_training_completed",
                "training": training,
                "source_model_fingerprint": model_state_fingerprint(source),
            }
        )
        durable = atomic_write_json(output, record)
        calibration = _make_declared_batch(
            source_config.task,
            split=audit.calibration_split,
            seed=audit.calibration_seed,
            lengths=audit.calibration_lengths,
        )
        calibration_hash = _batch_sha256(calibration)
        selection = select_cp_rank_by_parameter_energy(
            source,
            target_rank=audit.target_cp_rank,
            calibration_fingerprint=calibration_hash,
        )
        dense = build_dense_selected_reference(source, selection)
        compact, manifest = export_compact_binding_model(source, selection)
        source.eval()
        dense.eval()
        compact.eval()
        record.update(
            {
                "phase": "selection_and_physical_export_completed",
                "calibration_batch_sha256": calibration_hash,
                "selection_fingerprint": selection.fingerprint(),
                "manifest_fingerprint": manifest.fingerprint(),
                "compact_model_fingerprint": model_state_fingerprint(compact),
            }
        )
        durable = atomic_write_json(output, record)

        expanded_task = replace(
            source_config.task, max_length=max(audit.evaluation_lengths)
        )
        evaluation = _make_declared_batch(
            expanded_task,
            split=audit.evaluation_split,
            seed=audit.evaluation_seed,
            lengths=audit.evaluation_lengths,
        )
        evaluation_hash = _batch_sha256(evaluation)
        outputs, evaluation_record = _evaluate_variants(
            {"source": source, "dense_selected": dense, "compact": compact},
            evaluation,
            audit,
        )
        with torch.no_grad():
            encoded_events, _ = compact.encoder(evaluation.inputs)
        real_event_vectors = encoded_events[evaluation.inputs.valid_mask]
        carry = _forced_carry_probe(
            dense,
            compact,
            rtol=audit.float32_rtol,
            atol=audit.float32_atol,
            event_vectors=real_event_vectors,
        )
        autonomous_resume = _forest_resume_probe(
            compact, evaluation, rtol=audit.float32_rtol, atol=audit.float32_atol
        )
        forced_resume = _forced_forest_resume_probe(
            compact,
            real_event_vectors,
            rtol=audit.float32_rtol,
            atol=audit.float32_atol,
        )
        resume = {
            "passed": bool(autonomous_resume["passed"])
            and bool(forced_resume["passed"]),
            "autonomous_fixture_midpoint": autonomous_resume,
            "forced_local_and_global_pre_carry_cuts": forced_resume,
        }
        record.update(
            {
                "phase": "evaluation_and_state_probes_completed",
                "evaluation_batch_sha256": evaluation_hash,
                "evaluation": evaluation_record,
                "forced_real_update_carries": carry,
                "midstream_resume": resume,
            }
        )
        durable = atomic_write_json(output, record)

        artifact_bytes, artifact_roundtrip = _artifact_roundtrip(
            compact, manifest, selection
        )
        artifact_hash = _sha256(artifact_bytes)
        fixture_bytes = _fixture_bytes(evaluation.inputs)
        fixture_hash = _sha256(fixture_bytes)
        _atomic_bytes(paths["artifact"], artifact_bytes)
        _atomic_bytes(paths["fixture"], fixture_bytes)
        replay = _run_replay_worker(
            replay_worker,
            repository_root=repository_root,
            artifact=paths["artifact"],
            fixture=paths["fixture"],
            output=paths["replay_output"],
            artifact_sha256=artifact_hash,
            fixture_sha256=fixture_hash,
            source_fingerprint=manifest.source_model_fingerprint,
            manifest_fingerprint=manifest.fingerprint(),
            selection_fingerprint=selection.fingerprint(),
            code_commit=code_commit,
            code_tree=code_tree,
            worker_sha256=_sha256(replay_worker_snapshot),
        )
        replay_bytes = paths["replay_output"].read_bytes()
        expected_replay_hashes = {
            implementation: _replay_output_hashes(outputs["compact"][implementation])
            for implementation in ("streaming", "parallel")
        }
        replay_engines_match_local = all(
            replay.get(implementation) == expected_replay_hashes[implementation]
            for implementation in ("streaming", "parallel")
        )
        expected_replay_provenance = {
            "artifact_sha256": artifact_hash,
            "fixture_sha256": fixture_hash,
            "source_model_fingerprint": manifest.source_model_fingerprint,
            "manifest_fingerprint": manifest.fingerprint(),
            "selection_fingerprint": selection.fingerprint(),
            "code_commit": code_commit,
            "code_tree": code_tree,
            "worker_sha256": _sha256(replay_worker_snapshot),
        }
        replay_code = replay.get("code_provenance")
        replay_code_matches = isinstance(replay_code, dict) and (
            replay_code.get("code_commit") == code_commit
            and replay_code.get("code_tree") == code_tree
            and replay_code.get("worker_sha256") == _sha256(replay_worker_snapshot)
            and replay_code.get("package_sha256") == _sha256(package_snapshot)
            and replay_code.get("worker_committed") is True
            and replay_code.get("package_committed") is True
            and replay_code.get("worktree_clean") is True
        )
        replay_matches_local = (
            replay.get("model_fingerprint") == model_state_fingerprint(compact)
            and replay.get("source_model_fingerprint") == manifest.source_model_fingerprint
            and replay.get("manifest_fingerprint") == manifest.fingerprint()
            and replay.get("selection_fingerprint") == selection.fingerprint()
            and replay.get("process_id") != os.getpid()
            and replay_engines_match_local
            and replay.get("expected_provenance") == expected_replay_provenance
            and replay.get("fixture_sha256") == fixture_hash
            and replay.get("artifact_sha256") == artifact_hash
            and replay_code_matches
        )
        record.update(
            {
                "phase": "artifact_and_fresh_replay_completed",
                "artifact": {
                    "sha256": artifact_hash,
                    "bytes": len(artifact_bytes),
                    "deterministic_trusted_roundtrip": artifact_roundtrip,
                },
                "fixture": {"sha256": fixture_hash, "bytes": len(fixture_bytes)},
                "fresh_process_replay": {
                    "result_sha256": _sha256(replay_bytes),
                    "trusted_fingerprints_pid_and_per_engine_hashes_match": replay_matches_local,
                    "parent_expected_per_engine_hashes": expected_replay_hashes,
                    "record": replay,
                },
            }
        )
        durable = atomic_write_json(output, record)

        runtime_measurements = _runtime_matrix(
            runtime_worker,
            repository_root=repository_root,
            runtime_directory=paths["runtime_directory"],
            source=source,
            dense=dense,
            compact=compact,
            artifact=paths["artifact"],
            artifact_sha256=artifact_hash,
            source_fingerprint=manifest.source_model_fingerprint,
            manifest_fingerprint=manifest.fingerprint(),
            selection_fingerprint=selection.fingerprint(),
            evaluation=evaluation,
            lengths=audit.evaluation_lengths,
            code_commit=code_commit,
            code_tree=code_tree,
            audit=audit,
        )
        record.update(
            {
                "phase": "isolated_runtime_matrix_completed",
                "runtime_measurements": runtime_measurements,
            }
        )
        durable = atomic_write_json(output, record)

        record.update(
            {
                "training": training,
                "calibration": {
                    "split": audit.calibration_split,
                    "seed": audit.calibration_seed,
                    "lengths": list(audit.calibration_lengths),
                    "batch_sha256": calibration_hash,
                    "inputs_sha256": binding_inputs_sha256(calibration.inputs),
                    "batch_identity": _batch_identity_record(calibration),
                    "selection_uses_batch_values": False,
                    "selection_method_scope": "structural_parameter_energy_heuristic",
                },
                "evaluation_data": {
                    "split": audit.evaluation_split,
                    "seed": audit.evaluation_seed,
                    "lengths": list(audit.evaluation_lengths),
                    "expanded_task": asdict(expanded_task),
                    "expanded_task_fingerprint": expanded_task.fingerprint(),
                    "batch_sha256": evaluation_hash,
                    "inputs_sha256": binding_inputs_sha256(evaluation.inputs),
                    "batch_identity": _batch_identity_record(evaluation),
                },
                "selection": {
                    "fingerprint": selection.fingerprint(),
                    "record": asdict(selection),
                },
                "manifest": {
                    "fingerprint": manifest.fingerprint(),
                    "record": manifest.to_dict(),
                },
                "structural": {
                    "source": _structural_record(source),
                    "dense_selected": _structural_record(dense),
                    "compact": _structural_record(compact),
                    "cp_axis_physical_slicing": _cp_axis_record(source, compact),
                },
                "evaluation": evaluation_record,
                "forced_real_update_carries": carry,
                "midstream_resume": resume,
                "artifact": {
                    "sha256": artifact_hash,
                    "bytes": len(artifact_bytes),
                    "deterministic_trusted_roundtrip": artifact_roundtrip,
                },
                "fixture": {"sha256": fixture_hash, "bytes": len(fixture_bytes)},
                "fresh_process_replay": {
                    "result_sha256": _sha256(replay_bytes),
                    "trusted_fingerprints_pid_and_per_engine_hashes_match": replay_matches_local,
                    "parent_expected_per_engine_hashes": expected_replay_hashes,
                    "record": replay,
                },
                "runtime_measurements": runtime_measurements,
            }
        )
        gates = {
            "calibration_lengths_exact": tuple(calibration.lengths.tolist())
            == audit.calibration_lengths,
            "evaluation_lengths_exact": tuple(evaluation.lengths.tolist())
            == audit.evaluation_lengths,
            "evaluation_extends_beyond_training": max(audit.evaluation_lengths)
            > source_config.task.max_length,
            "selection_calibration_bound": selection.calibration_fingerprint
            == calibration_hash,
            "training_fingerprints_bound": (
                training["initial_model_fingerprint"]
                != training["trained_model_fingerprint"]
                and training["trained_model_fingerprint"]
                == manifest.source_model_fingerprint
            ),
            "curriculum_final_guidance_zero": (
                training["final_guidance"]["probability"] == 0.0
                and training["final_guidance"]["guided_events"] == 0
                and training["final_loss"]["route_supervision_count"] == 0
            ),
            "physical_parameter_reduction": manifest.exported_parameter_count
            < manifest.source_parameter_count,
            "physical_raw_tensor_byte_reduction": manifest.exported_raw_tensor_bytes
            < manifest.source_raw_tensor_bytes,
            "declared_operation_proxy_reduction": (
                manifest.exported_operation_count_proxy_per_merge
                < manifest.source_operation_count_proxy_per_merge
            ),
            "state_interface_honestly_unchanged": manifest.state_interface_unchanged,
            "five_cp_tensors_physically_sliced_without_rank_state": bool(
                record["structural"]["cp_axis_physical_slicing"]["passed"]
            ),
            "all_streaming_parallel_parity": all(
                bool(evaluation_record[name]["streaming_parallel_parity"]["passed"])
                for name in ("source", "dense_selected", "compact")
            ),
            "dense_compact_streaming_parity": bool(
                evaluation_record["dense_compact_parity"]["streaming"]["passed"]
            ),
            "dense_compact_parallel_parity": bool(
                evaluation_record["dense_compact_parity"]["parallel"]["passed"]
            ),
            "query_observations_present": all(
                evaluation_record[name][implementation]["query_metrics"]["all"]["count"]
                > 0
                and evaluation_record[name][implementation]["query_metrics"]["heldout"][
                    "count"
                ]
                > 0
                for name in ("source", "dense_selected", "compact")
                for implementation in ("streaming", "parallel")
            ),
            "compact_total_operation_proxy_reduced_at_actual_merge_count": all(
                evaluation_record["compact"][implementation]["structural_work"][
                    "actual_executed_merge_count"
                ]
                == evaluation_record["dense_selected"][implementation]["structural_work"][
                    "actual_executed_merge_count"
                ]
                and evaluation_record["compact"][implementation]["structural_work"][
                    "total_operation_count_proxy"
                ]
                < evaluation_record["dense_selected"][implementation]["structural_work"][
                    "total_operation_count_proxy"
                ]
                for implementation in ("streaming", "parallel")
            ),
            "forced_local_and_global_carries": bool(carry["passed"]),
            "midstream_state_resume": bool(resume["passed"]),
            "artifact_checksum_rechecked": _sha256(paths["artifact"].read_bytes())
            == artifact_hash,
            "fixture_checksum_rechecked": _sha256(paths["fixture"].read_bytes())
            == fixture_hash,
            "artifact_deterministic_trusted_roundtrip": bool(
                artifact_roundtrip["passed"]
            ),
            "trusted_fresh_process_replay": replay_matches_local,
            "fresh_replay_matches_parent_per_engine": replay_engines_match_local,
            "isolated_runtime_matrix_completed": bool(
                runtime_measurements["all_workers_passed"]
            ),
        }
        if not all(gates.values()):
            raise RuntimeError(f"Milestone-3 export-audit gate failed: {gates}")
        if (
            audit_path.read_bytes() != audit_snapshot
            or source_path.read_bytes() != source_snapshot
            or replay_worker.read_bytes() != replay_worker_snapshot
            or runtime_worker.read_bytes() != runtime_worker_snapshot
            or runner_path.read_bytes() != runner_snapshot
            or package_file.read_bytes() != package_snapshot
        ):
            raise RuntimeError("a bound configuration or worker changed during the audit")
        final_commit, final_tree = _bind_to_clean_checkout(code_commit)
        if (final_commit, final_tree) != (code_commit, code_tree):
            raise RuntimeError("the source checkout changed during the audit")
        record.update(
            {
                "status": "passed",
                "phase": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "gates": gates,
            }
        )
        durable = atomic_write_json(output, record)
    except Exception as error:
        atomic_write_json(output, _failure_record(durable, error))
        raise
    return output


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--replay-output", required=True)
    parser.add_argument("--runtime-directory", required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args(argv)
    if not Path(args.audit_config).is_absolute():
        parser.error("--audit-config must be an absolute path")
    return args


def main() -> None:
    output = _execute(_parse_args())
    print(f"MILESTONE3_EXPORT_AUDIT_PASSED: {output}")


if __name__ == "__main__":
    main()
