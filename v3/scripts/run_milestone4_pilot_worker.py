from __future__ import annotations

"""Execute one commit-bound, non-claiming Milestone-4 pilot run.

The parent owns the campaign manifest.  This worker owns only one immutable
attempt directory and emits a strict result plus model artifacts.  It never
opens confirmatory test/scaling streams and never interprets the pilot as
scientific evidence.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import psutil
import torch
from torch import Tensor
import torch.nn.functional as F

from tnlm_v3.binding import RoutedBindingModel
from tnlm_v3.campaign import evaluate_baseline_model
from tnlm_v3.campaign_checkpoint import (
    CampaignResumeState,
    campaign_checkpoint_contract,
    campaign_model_fingerprint,
    deserialize_campaign_checkpoint,
    serialize_campaign_checkpoint,
)
from tnlm_v3.campaign_config import (
    CampaignStage,
    Milestone4CampaignConfig,
    ResolvedCampaignRun,
    campaign_plan_sha256,
    load_milestone4_campaign_config,
    resolve_campaign_plan,
)
from tnlm_v3.campaign_execution import (
    CampaignRunContext,
    build_campaign_optimizer,
    build_campaign_source_model,
    campaign_batch_sha256,
    derive_campaign_compact_model,
    generate_campaign_evaluation_batch,
    generate_campaign_training_batch,
    run_campaign_training_step,
    validate_campaign_execution_environment,
)
from tnlm_v3.compact_artifact import serialize_compact_binding_model
from tnlm_v3.data import BindingEventKind, BindingBatch
from tnlm_v3.training import evaluate_binding_model


_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_PACKAGE_PREFIX = "v3/src/tnlm_v3/"
_WORKER_RELATIVE = "v3/scripts/run_milestone4_pilot_worker.py"
_RUNNER_RELATIVE = "v3/scripts/run_milestone4_pilot.py"
_PACKAGE_DOMAIN = b"tnlm-v3-package-tree-v1\0"
_BUNDLE_DOMAIN = b"tnlm-v3-pilot-bundle-v1\0"
_STREAM_DOMAIN = b"tnlm-v3-m4-training-stream-prefix-v1\0"
_MAX_JSON = 8 * 1024 * 1024
_MAX_FILE = 64 * 1024 * 1024


class PilotWorkerError(RuntimeError):
    pass


def _sha(value: object, name: str, *, forty: bool = False) -> str:
    pattern = _HEX40 if forty else _HEX64
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise PilotWorkerError(f"{name} must be lowercase hexadecimal")
    return value


def _plain_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PilotWorkerError(f"{name} must be an integer >= {minimum}")
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    ).encode("utf-8")


def _strict_json(raw: bytes, name: str) -> Mapping[str, Any]:
    if len(raw) > _MAX_JSON:
        raise PilotWorkerError(f"{name} is too large")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise PilotWorkerError(f"{name} has duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PilotWorkerError(f"{name} contains {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise PilotWorkerError(f"{name} is not strict JSON") from error
    if type(value) is not dict or _canonical(value) != raw:
        raise PilotWorkerError(f"{name} must be a canonical JSON object")
    return value


def _git(repo: Path, *arguments: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={repo.as_posix()}", *arguments],
        cwd=repo,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise PilotWorkerError(
            "Git provenance command failed: "
            + completed.stderr.decode("utf-8", "replace")[:2048]
        )
    if binary:
        return completed.stdout
    try:
        return completed.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PilotWorkerError("Git output is not UTF-8") from error


def _read_private(path: Path, *, maximum: int, name: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise PilotWorkerError(f"{name} does not exist") from error
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        stat.S_ISLNK(before.st_mode)
        or bool(getattr(before, "st_file_attributes", 0) & reparse)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > maximum
    ):
        raise PilotWorkerError(f"{name} must be a bounded private regular file")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        signature = lambda info: (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_nlink,
        )
        if signature(opened) != signature(before):
            raise PilotWorkerError(f"{name} changed before open")
        chunks: list[bytes] = []
        measured = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - measured))
            if not chunk:
                break
            measured += len(chunk)
            if measured > maximum:
                raise PilotWorkerError(f"{name} is too large")
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if measured != before.st_size or signature(after) != signature(before):
        raise PilotWorkerError(f"{name} changed while read")
    return b"".join(chunks)


def _write_atomic(path: Path, raw: bytes, *, immutable: bool) -> None:
    if len(raw) > _MAX_FILE:
        raise PilotWorkerError("output exceeds the worker byte limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary leaf deliberately short.  Attempt directories contain
    # full SHA-256 run identifiers, so repeating the target filename here can
    # cross the legacy Windows MAX_PATH boundary even when the final path fits.
    temporary = path.with_name(f".w{os.getpid():x}{os.urandom(3).hex()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    if immutable:
        try:
            # A hard-link installation is a create-if-absent CAS: unlike an
            # exists-check followed by replace, it cannot overwrite a file
            # concurrently installed by another worker.
            os.link(temporary, path)
        except FileExistsError:
            current = _read_private(path, maximum=_MAX_FILE, name=path.name)
            if current != raw:
                raise PilotWorkerError(
                    f"immutable output {path.name} already differs"
                )
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    else:
        os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability where the platform exposes it."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        # Windows does not allow ordinary directory handles through os.open;
        # the file itself was flushed before replace/link installation.
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_directory_chain(root: Path, child: Path, name: str) -> None:
    """Reject symlinks, junctions, and non-directories in an output path."""

    root = Path(os.path.abspath(root))
    child = Path(os.path.abspath(child))
    try:
        relative = child.relative_to(root)
    except ValueError as error:
        raise PilotWorkerError(f"{name} is outside output_root") from error
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    current = root
    for part in ((), *relative.parts):
        if part:
            current = current / part
        try:
            information = current.lstat()
        except OSError as error:
            raise PilotWorkerError(f"{name} directory is missing") from error
        if (
            stat.S_ISLNK(information.st_mode)
            or bool(getattr(information, "st_file_attributes", 0) & reparse)
            or not stat.S_ISDIR(information.st_mode)
        ):
            raise PilotWorkerError(f"{name} traverses a linked or non-directory path")


def _artifact(path: Path, output_root: Path) -> dict[str, object]:
    raw = _read_private(path, maximum=_MAX_FILE, name=path.name)
    return {
        "path": path.relative_to(output_root).as_posix(),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _git_index(repo: Path) -> dict[str, tuple[str, str]]:
    raw = _git(repo, "ls-files", "--stage", "-z", binary=True)
    assert isinstance(raw, bytes)
    result: dict[str, tuple[str, str]] = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        metadata, path_raw = entry.split(b"\t", 1)
        mode, blob, stage = metadata.decode("ascii").split(" ")
        path = path_raw.decode("utf-8")
        if stage != "0" or path in result:
            raise PilotWorkerError("Git index is not canonical")
        result[path] = (mode, blob)
    return result


def _inventory(repo: Path, config: Path, runner: Path) -> dict[str, str]:
    index = _git_index(repo)
    worker = Path(__file__).resolve(strict=True)
    expected_worker = (repo / _WORKER_RELATIVE).resolve(strict=True)
    if worker != expected_worker or runner.resolve(strict=True) != (repo / _RUNNER_RELATIVE).resolve(strict=True):
        raise PilotWorkerError("worker or parent runner path is not canonical")
    required_extra = {
        config.relative_to(repo).as_posix(),
        _RUNNER_RELATIVE,
        _WORKER_RELATIVE,
    }
    package = sorted(
        path
        for path, (mode, _) in index.items()
        if path.startswith(_PACKAGE_PREFIX) and mode in {"100644", "100755"}
    )
    required = sorted(set(package) | required_extra, key=lambda item: item.encode("utf-8"))
    entries: list[tuple[str, int, str]] = []
    for relative in required:
        if relative not in index or index[relative][0] not in {"100644", "100755"}:
            raise PilotWorkerError(f"required file {relative!r} is not committed")
        blob = index[relative][1]
        committed = _git(repo, "cat-file", "blob", blob, binary=True)
        assert isinstance(committed, bytes)
        current = _read_private(repo / PurePosixPath(relative), maximum=_MAX_FILE, name=relative)
        if current != committed:
            raise PilotWorkerError(f"required file {relative!r} differs from Git")
        entries.append((relative, len(current), hashlib.sha256(current).hexdigest()))

    def digest(domain: bytes, selected: Sequence[tuple[str, int, str]]) -> str:
        value = hashlib.sha256(domain)
        for path, size, raw_hash in selected:
            value.update(path.encode("utf-8") + b"\0")
            value.update(str(size).encode("ascii") + b"\0")
            value.update(raw_hash.encode("ascii") + b"\n")
        return value.hexdigest()

    package_set = set(package)
    package_entries = tuple(item for item in entries if item[0] in package_set)
    values = {path: raw_hash for path, _, raw_hash in entries}
    values["package_tree"] = digest(_PACKAGE_DOMAIN, package_entries)
    values["bundle"] = digest(_BUNDLE_DOMAIN, entries)
    return values


def _check_import_origins(repo: Path, inventory: Mapping[str, str]) -> None:
    root = (repo / "v3" / "src" / "tnlm_v3").resolve(strict=True)
    for name, module in tuple(sys.modules.items()):
        if name != "tnlm_v3" and not name.startswith("tnlm_v3."):
            continue
        raw = getattr(module, "__file__", None)
        if raw is None:
            continue
        origin = Path(raw).resolve(strict=True)
        try:
            relative_to_package = origin.relative_to(root)
        except ValueError as error:
            raise PilotWorkerError("tnlm_v3 was imported outside the checkout") from error
        relative = (_PACKAGE_PREFIX + relative_to_package.as_posix())
        if relative.endswith(".pyc"):
            relative = relative[:-1]
        if relative not in inventory:
            raise PilotWorkerError("imported module is absent from the committed inventory")


def _run_hash(run: ResolvedCampaignRun) -> str:
    return hashlib.sha256(run.canonical_json().encode("utf-8")).hexdigest()


def _prefix(previous: str, step: int, batch_hash: str) -> str:
    return hashlib.sha256(
        bytes.fromhex(previous)
        + step.to_bytes(8, "little", signed=False)
        + bytes.fromhex(batch_hash)
    ).hexdigest()


def _checkpoint_cursors(run: ResolvedCampaignRun) -> tuple[int, ...]:
    if run.training is None:
        return ()
    values = list(
        range(
            run.training.checkpoint_interval,
            run.training.optimizer_steps + 1,
            run.training.checkpoint_interval,
        )
    )
    if not values or values[-1] != run.training.optimizer_steps:
        values.append(run.training.optimizer_steps)
    return tuple(values)


def _copy_restored(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    restored_model: torch.nn.Module,
    restored_optimizer: torch.optim.AdamW,
) -> None:
    model.load_state_dict(restored_model.state_dict(), strict=True)
    source_modes = dict(restored_model.named_modules())
    for name, module in model.named_modules():
        module.training = source_modes[name].training
    optimizer.load_state_dict(restored_optimizer.state_dict())


def _loss_record(result: object) -> tuple[dict[str, object], dict[str, int]]:
    loss = getattr(result, "loss")

    def number(name: str) -> float:
        tensor = getattr(loss, name)
        value = float(tensor.detach().cpu())
        if not math.isfinite(value) or value < 0:
            raise PilotWorkerError("training loss is not finite and nonnegative")
        return value

    if hasattr(loss, "query"):
        values: dict[str, object] = {
            "total": number("total"),
            "query": number("query"),
            "route_curriculum": number("route_curriculum"),
            "router_balance": number("router_balance"),
            "router_entropy": number("router_entropy"),
            "route_persistence": number("route_persistence"),
        }
        counters = {
            "query_count": int(loss.query_count),
            "route_supervision_count": int(loss.route_supervision_count),
            "persistence_pair_count": int(loss.persistence_pair_count),
        }
    else:
        values = {
            "total": number("total"),
            "query": number("total"),
            "route_curriculum": None,
            "router_balance": None,
            "router_entropy": None,
            "route_persistence": None,
        }
        counters = {
            "query_count": int(loss.query_count),
            "route_supervision_count": 0,
            "persistence_pair_count": 0,
        }
    return values, counters


def _query_entry(logits: Tensor, targets: Tensor, mask: Tensor, accuracy: object) -> dict[str, object]:
    count = int(mask.sum().item())
    correct = int(getattr(accuracy, "correct"))
    observed_count = int(getattr(accuracy, "query_count"))
    if observed_count != count or not 0 <= correct <= count:
        raise PilotWorkerError("query summary counters disagree")
    ce: float | None
    if count:
        ce = float(F.cross_entropy(logits[mask], targets[mask]).detach().cpu())
        if not math.isfinite(ce) or ce < 0:
            raise PilotWorkerError("validation cross entropy is invalid")
    else:
        ce = None
    return {
        "correct": correct,
        "count": count,
        "accuracy": correct / count if count else 0.0,
        "cross_entropy": ce,
    }


def _structural(model: torch.nn.Module, batch: BindingBatch, output: object, summary: object) -> dict[str, object]:
    if type(model) is RoutedBindingModel:
        metrics = model.forest.structural_metrics(
            output.forest_state,
            merge_count=output.diagnostics["forest_merge_count"],
        )
        metrics["parameter_count"] = sum(item.numel() for item in model.parameters())
        metrics["parameter_bytes"] = sum(
            item.numel() * item.element_size() for item in model.parameters()
        )
    else:
        metrics = dict(summary.structural_metrics)
    values: dict[str, int] = {}
    for key, value in metrics.items():
        if type(key) is not str or type(value) is not int or value < 0:
            raise PilotWorkerError("structural metrics must be nonnegative integers")
        values[key] = value
    if not values:
        raise PilotWorkerError("structural metrics cannot be empty")
    fingerprint = hashlib.sha256(_canonical(values)).hexdigest()
    return {"fingerprint_sha256": fingerprint, "values": values}


def _evaluate(model: torch.nn.Module, context: CampaignRunContext) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for length in context.run.data.validation.lengths:
        batch = generate_campaign_evaluation_batch(
            context, stream="validation", length=length
        )
        if type(model) is RoutedBindingModel:
            output, summary = evaluate_binding_model(model, batch)
            query_accuracy = summary.query
            seen_accuracy = summary.seen_query
            heldout_accuracy = summary.heldout_query
        else:
            output, summary = evaluate_baseline_model(model, batch)
            query_accuracy = summary.query.accuracy
            seen_accuracy = summary.seen_query.accuracy
            heldout_accuracy = summary.heldout_query.accuracy
        query_mask = batch.inputs.valid_mask & (
            batch.inputs.event_kinds == int(BindingEventKind.QUERY)
        )
        heldout_mask = query_mask & batch.evaluation.heldout_combination_mask
        seen_mask = query_mask & ~batch.evaluation.heldout_combination_mask
        result.append(
            {
                "length": length,
                "batch_sha256": campaign_batch_sha256(batch),
                "episodes": int(batch.inputs.valid_mask.shape[0]),
                "query": _query_entry(
                    output.value_logits,
                    batch.evaluation.targets,
                    query_mask,
                    query_accuracy,
                ),
                "seen_query": _query_entry(
                    output.value_logits,
                    batch.evaluation.targets,
                    seen_mask,
                    seen_accuracy,
                ),
                "heldout_query": _query_entry(
                    output.value_logits,
                    batch.evaluation.targets,
                    heldout_mask,
                    heldout_accuracy,
                ),
                "structural": _structural(model, batch, output, summary),
            }
        )
    return result


def _progress(
    path: Path,
    *,
    run: ResolvedCampaignRun,
    attempt: int,
    cursor: int,
    prefix: str,
    checkpoint: Mapping[str, object] | None,
    provenance: Mapping[str, str],
) -> None:
    try:
        created = int(round(psutil.Process(os.getpid()).create_time() * 1_000_000_000))
    except psutil.Error as error:
        raise PilotWorkerError("cannot bind worker process creation time") from error
    _write_atomic(
        path,
        _canonical(
            {
                "schema_version": 1,
                "run_id": run.run_id,
                "attempt_number": attempt,
                "pid": os.getpid(),
                "process_create_time_ns": created,
                "cursor": cursor,
                "stream_prefix_sha256": prefix,
                "latest_checkpoint": checkpoint,
                "provenance": dict(provenance),
            }
        ),
        immutable=False,
    )


def _restore_source(
    context: CampaignRunContext,
    checkpoint_path: Path,
    checkpoint_sha: str,
    expected_prefix: str,
    expected_cursor: int,
) -> tuple[torch.nn.Module, torch.optim.AdamW, CampaignResumeState, bytes]:
    raw = _read_private(checkpoint_path, maximum=_MAX_FILE, name="resume checkpoint")
    if hashlib.sha256(raw).hexdigest() != checkpoint_sha:
        raise PilotWorkerError("resume checkpoint checksum mismatch")
    model = build_campaign_source_model(context)
    optimizer = build_campaign_optimizer(context, model)
    contract = campaign_checkpoint_contract(model, optimizer)
    restored_model, restored_optimizer, resume = deserialize_campaign_checkpoint(
        raw,
        expected_run_spec_sha256=_run_hash(context.run),
        expected_stream_prefix_sha256=expected_prefix,
        expected_contract=contract,
        device="cpu",
    )
    if resume.global_step != expected_cursor or resume.data_cursor != expected_cursor:
        raise PilotWorkerError("resume checkpoint cursor mismatch")
    _copy_restored(model, optimizer, restored_model, restored_optimizer)
    if serialize_campaign_checkpoint(model, optimizer, resume) != raw:
        raise PilotWorkerError("bound restored checkpoint is not byte-identical")
    return model, optimizer, resume, raw


def _source_run(
    context: CampaignRunContext,
    attempt_dir: Path,
    output_root: Path,
    resume_path: Path | None,
    resume_sha: str | None,
    provenance: Mapping[str, str],
    attempt: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    run = context.run
    assert run.training is not None
    cursors = _checkpoint_cursors(run)
    cursor = 0
    if resume_path is not None:
        match = re.fullmatch(r"checkpoint-step-([0-9]{8})\.twcp", resume_path.name)
        if match is None:
            raise PilotWorkerError("resume checkpoint filename is invalid")
        cursor = int(match.group(1))
        if cursor not in cursors or resume_sha is None:
            raise PilotWorkerError("resume cursor is not a locked checkpoint")

    # Replay the deterministic prefix so the final record remains complete and
    # all scheduled checkpoints are copied into this attempt directory.
    torch.manual_seed(run.model_seed)
    random.seed(run.statistics_seed)
    model = build_campaign_source_model(context)
    optimizer = build_campaign_optimizer(context, model)
    initial_fingerprint = campaign_model_fingerprint(model)
    prefix = hashlib.sha256(_STREAM_DOMAIN).hexdigest()
    step_records: list[dict[str, object]] = []
    batch_records: list[dict[str, object]] = []
    checkpoint_records: list[dict[str, object]] = []
    resume_raw: bytes | None = None
    if resume_path is not None:
        resume_raw = _read_private(resume_path, maximum=_MAX_FILE, name="resume checkpoint")
        if hashlib.sha256(resume_raw).hexdigest() != resume_sha:
            raise PilotWorkerError("resume checkpoint checksum mismatch")

    for step in range(run.training.optimizer_steps):
        result = run_campaign_training_step(context, model, optimizer, step=step)
        token_count = int(result.batch.lengths.sum().item())
        loss, counters = _loss_record(result)
        prefix = _prefix(prefix, step, result.batch_sha256)
        batch_records.append(
            {"step": step, "sha256": result.batch_sha256, "token_count": token_count}
        )
        step_records.append(
            {
                "step": step,
                "batch_sha256": result.batch_sha256,
                "token_count": token_count,
                "loss": loss,
                "counters": counters,
            }
        )
        current = step + 1
        if current in cursors:
            resume_state = CampaignResumeState(
                global_step=current,
                data_cursor=current,
                run_spec_sha256=_run_hash(run),
                stream_prefix_sha256=prefix,
            )
            raw = serialize_campaign_checkpoint(model, optimizer, resume_state)
            if current == cursor and resume_raw is not None and raw != resume_raw:
                raise PilotWorkerError("replayed prefix differs from resume checkpoint")
            path = attempt_dir / f"checkpoint-step-{current:08d}.twcp"
            _write_atomic(path, raw, immutable=True)
            reference = _artifact(path, output_root)
            item = {
                "step": current,
                **reference,
                "stream_prefix_sha256": prefix,
            }
            checkpoint_records.append(item)
            _progress(
                attempt_dir / "progress.json",
                run=run,
                attempt=attempt,
                cursor=current,
                prefix=prefix,
                checkpoint=item,
                provenance=provenance,
            )
        if current == cursor and resume_path is not None:
            # Exercise the actual codec restore and continue only from the
            # checkpoint-owned state/RNG, not merely the deterministic replay.
            restored, restored_optimizer, _, _ = _restore_source(
                context, resume_path, resume_sha, prefix, cursor
            )
            model = restored
            optimizer = restored_optimizer

    final_fingerprint = campaign_model_fingerprint(model)
    validation = _evaluate(model, context)
    training = {
        "initial_model_fingerprint": initial_fingerprint,
        "final_model_fingerprint": final_fingerprint,
        "optimizer_steps": run.training.optimizer_steps,
        "token_count": sum(item["token_count"] for item in batch_records),
        "steps": step_records,
    }
    metrics = {
        "environment": _environment(run),
        "training": training,
        "validation_by_length": validation,
        "compact": None,
    }
    stream = {
        "start_step": 0,
        "resumed_from_step": cursor,
        "completed_step": run.training.optimizer_steps,
        "training_batches": batch_records,
        "stream_prefix_sha256": prefix,
        "checkpoint_steps": list(cursors),
    }
    artifacts = {
        "checkpoints": checkpoint_records,
        "final_checkpoint": {
            key: checkpoint_records[-1][key]
            for key in ("path", "size_bytes", "sha256")
        },
        "compact_artifact": None,
    }
    return artifacts, metrics, stream


def _environment(run: ResolvedCampaignRun) -> dict[str, str]:
    device = "cpu" if run.training is None else run.training.device
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "platform": platform.platform(),
        "device": device,
    }


def _parent_result(path: Path, expected_sha: str) -> Mapping[str, Any]:
    raw = _read_private(path, maximum=_MAX_JSON, name="parent result")
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise PilotWorkerError("parent result checksum mismatch")
    value = _strict_json(raw, "parent result")
    if value.get("status") != "success":
        raise PilotWorkerError("compact parent result is not successful")
    return value


def _compact_run(
    context: CampaignRunContext,
    config: Milestone4CampaignConfig,
    plan: tuple[ResolvedCampaignRun, ...],
    attempt_dir: Path,
    output_root: Path,
    parent_result_path: Path,
    parent_result_sha: str,
    parent_checkpoint_path: Path,
    parent_checkpoint_sha: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    run = context.run
    parents = tuple(item for item in plan if item.run_id == run.parent_run_id)
    if len(parents) != 1:
        raise PilotWorkerError("compact parent is absent from the exact plan")
    parent_run = parents[0]
    parent_context = CampaignRunContext(config=config, run=parent_run)
    parent_document = _parent_result(parent_result_path, parent_result_sha)
    if (
        parent_document.get("run_id") != parent_run.run_id
        or parent_document.get("run_sha256") != _run_hash(parent_run)
    ):
        raise PilotWorkerError("parent result identity mismatch")
    parent_attempt = _plain_int(
        parent_document.get("attempt_number"), "parent attempt_number", minimum=1
    )
    expected_parent_directory = (
        output_root
        / "artifacts"
        / parent_run.run_id
        / f"attempt-{parent_attempt:06d}"
    )
    if parent_result_path != expected_parent_directory / "result.json":
        raise PilotWorkerError("parent result path is not its deterministic attempt path")
    parent_artifacts = parent_document.get("artifacts")
    if type(parent_artifacts) is not dict:
        raise PilotWorkerError("parent result artifacts are malformed")
    declared_checkpoint = parent_artifacts.get("final_checkpoint")
    if type(declared_checkpoint) is not dict or set(declared_checkpoint) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise PilotWorkerError("parent result has no exact final checkpoint reference")
    expected_checkpoint_path = output_root / PurePosixPath(
        str(declared_checkpoint["path"])
    )
    if (
        parent_checkpoint_path != expected_checkpoint_path
        or parent_checkpoint_path.parent != expected_parent_directory
        or declared_checkpoint["sha256"] != parent_checkpoint_sha
    ):
        raise PilotWorkerError("parent checkpoint path or digest is not result-bound")
    _validate_directory_chain(output_root, expected_parent_directory, "parent attempt")
    expected_prefix = hashlib.sha256(_STREAM_DOMAIN).hexdigest()
    assert parent_run.training is not None
    for step in range(parent_run.training.optimizer_steps):
        batch_hash = campaign_batch_sha256(
            generate_campaign_training_batch(parent_context, step=step)
        )
        expected_prefix = _prefix(expected_prefix, step, batch_hash)
    parent_model, parent_optimizer, resume, _ = _restore_source(
        parent_context,
        parent_checkpoint_path,
        parent_checkpoint_sha,
        expected_prefix,
        parent_run.training.optimizer_steps,
    )
    if type(parent_model) is not RoutedBindingModel:
        raise PilotWorkerError("compact parent is not routed")
    parent_fingerprint = campaign_model_fingerprint(parent_model)
    parent_training = parent_document.get("metrics", {}).get("training")
    if (
        type(parent_training) is not dict
        or parent_training.get("final_model_fingerprint") != parent_fingerprint
    ):
        raise PilotWorkerError("parent result does not bind its checkpoint model")
    derivation = derive_campaign_compact_model(
        parent_context,
        context,
        parent_model,
        parent_optimizer,
    )
    validation = _evaluate(derivation.model, context)
    raw = serialize_compact_binding_model(
        derivation.model, derivation.manifest, derivation.selection
    )
    artifact_path = attempt_dir / "compact-model.tnlm3"
    _write_atomic(artifact_path, raw, immutable=True)
    artifact_ref = _artifact(artifact_path, output_root)
    result_ref = {
        "path": parent_result_path.relative_to(output_root).as_posix(),
        "size_bytes": parent_result_path.stat().st_size,
        "sha256": parent_result_sha,
    }
    checkpoint_ref = {
        "path": parent_checkpoint_path.relative_to(output_root).as_posix(),
        "size_bytes": parent_checkpoint_path.stat().st_size,
        "sha256": parent_checkpoint_sha,
    }
    compact = {
        "parent_run_id": parent_run.run_id,
        "parent_result": result_ref,
        "parent_checkpoint": checkpoint_ref,
        "parent_model_fingerprint": parent_fingerprint,
        "selection_fingerprint": derivation.selection.fingerprint(),
        "manifest_fingerprint": derivation.manifest.fingerprint(),
        "exported_model_fingerprint": derivation.manifest.exported_model_fingerprint,
        "compact_artifact_sha256": artifact_ref["sha256"],
    }
    return (
        {
            "checkpoints": [],
            "final_checkpoint": None,
            "compact_artifact": artifact_ref,
        },
        {
            "environment": _environment(run),
            "training": None,
            "validation_by_length": validation,
            "compact": compact,
        },
        {
            "start_step": 0,
            "resumed_from_step": 0,
            "completed_step": 0,
            "training_batches": [],
            "stream_prefix_sha256": None,
            "checkpoint_steps": [],
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo-root",
        "parent-runner",
        "config",
        "output-root",
        "result-path",
        "run-id",
        "plan-sha256",
        "code-commit",
        "code-tree",
        "raw-config-sha256",
        "semantic-config-sha256",
        "executable-bundle-sha256",
        "parent-runner-sha256",
        "worker-sha256",
        "package-tree-sha256",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--attempt-number", required=True, type=int)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--resume-checkpoint-sha256")
    parser.add_argument("--parent-result")
    parser.add_argument("--parent-result-sha256")
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--parent-checkpoint-sha256")
    return parser


def _execute(arguments: argparse.Namespace) -> dict[str, object]:
    repo = Path(arguments.repo_root).resolve(strict=True)
    output_root = Path(arguments.output_root).resolve(strict=True)
    config_path = Path(arguments.config).resolve(strict=True)
    runner_path = Path(arguments.parent_runner).resolve(strict=True)
    result_path = Path(arguments.result_path).absolute()
    _validate_directory_chain(output_root, output_root, "output_root")
    commit = str(_git(repo, "rev-parse", "HEAD")).strip()
    tree = str(_git(repo, "rev-parse", "HEAD^{tree}")).strip()
    if commit != _sha(arguments.code_commit, "code_commit", forty=True):
        raise PilotWorkerError("worker HEAD differs from requested commit")
    if tree != _sha(arguments.code_tree, "code_tree", forty=True):
        raise PilotWorkerError("worker tree differs from requested tree")
    if str(_git(repo, "status", "--porcelain=v1", "--untracked-files=all")):
        raise PilotWorkerError("worker requires a clean checkout")
    inventory = _inventory(repo, config_path, runner_path)
    expected = {
        "raw_config_sha256": arguments.raw_config_sha256,
        "parent_runner_sha256": arguments.parent_runner_sha256,
        "worker_sha256": arguments.worker_sha256,
        "package_tree_sha256": arguments.package_tree_sha256,
        "executable_bundle_sha256": arguments.executable_bundle_sha256,
    }
    actual = {
        "raw_config_sha256": inventory[config_path.relative_to(repo).as_posix()],
        "parent_runner_sha256": inventory[_RUNNER_RELATIVE],
        "worker_sha256": inventory[_WORKER_RELATIVE],
        "package_tree_sha256": inventory["package_tree"],
        "executable_bundle_sha256": inventory["bundle"],
    }
    for name, value in expected.items():
        if actual[name] != _sha(value, name):
            raise PilotWorkerError(f"{name} differs from committed bytes")
    _check_import_origins(repo, inventory)
    raw_config = _read_private(config_path, maximum=1024 * 1024, name="config")
    config = load_milestone4_campaign_config(config_path)
    if (
        config.stage is not CampaignStage.PILOT
        or config.claim_eligible
        or config.data.test is not None
        or config.data.scaling is not None
    ):
        raise PilotWorkerError("worker accepts only the non-claiming pilot")
    if hashlib.sha256(raw_config).hexdigest() != arguments.raw_config_sha256:
        raise PilotWorkerError("raw config digest mismatch")
    if config.fingerprint() != arguments.semantic_config_sha256:
        raise PilotWorkerError("semantic config digest mismatch")
    plan = resolve_campaign_plan(
        config,
        commit,
        tree,
        arguments.raw_config_sha256,
        arguments.executable_bundle_sha256,
    )
    if campaign_plan_sha256(config, plan) != arguments.plan_sha256:
        raise PilotWorkerError("plan digest mismatch")
    matches = tuple(run for run in plan if run.run_id == arguments.run_id)
    if len(matches) != 1:
        raise PilotWorkerError("run is absent from the exact plan")
    run = matches[0]
    context = CampaignRunContext(config=config, run=run)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(config.implementation_policy.intraop_threads)
    torch.set_num_interop_threads(config.implementation_policy.interop_threads)
    validate_campaign_execution_environment(context)
    torch.manual_seed(run.model_seed)
    random.seed(run.statistics_seed)
    attempt = _plain_int(arguments.attempt_number, "attempt_number", minimum=1)
    attempt_dir = output_root / "artifacts" / run.run_id / f"attempt-{attempt:06d}"
    if result_path != attempt_dir / "result.json":
        raise PilotWorkerError("result path is outside the exact attempt directory")
    _validate_directory_chain(output_root, attempt_dir, "attempt")
    provenance = {
        "code_commit": commit,
        "code_tree": tree,
        "raw_config_sha256": arguments.raw_config_sha256,
        "semantic_config_sha256": arguments.semantic_config_sha256,
        "executable_bundle_sha256": arguments.executable_bundle_sha256,
        "parent_runner_sha256": arguments.parent_runner_sha256,
        "worker_sha256": arguments.worker_sha256,
        "package_tree_sha256": arguments.package_tree_sha256,
    }
    if run.role == "trainable_source":
        if any(
            getattr(arguments, name) is not None
            for name in (
                "parent_result",
                "parent_result_sha256",
                "parent_checkpoint",
                "parent_checkpoint_sha256",
            )
        ):
            raise PilotWorkerError("source run cannot receive compact parent artifacts")
        resume_path = (
            None
            if arguments.resume_checkpoint is None
            else Path(arguments.resume_checkpoint).resolve(strict=True)
        )
        if (resume_path is None) is not (arguments.resume_checkpoint_sha256 is None):
            raise PilotWorkerError("resume checkpoint path/hash must be paired")
        artifacts, metrics, stream = _source_run(
            context,
            attempt_dir,
            output_root,
            resume_path,
            arguments.resume_checkpoint_sha256,
            provenance,
            attempt,
        )
    else:
        if arguments.resume_checkpoint is not None or arguments.resume_checkpoint_sha256 is not None:
            raise PilotWorkerError("compact run cannot resume a training checkpoint")
        parent_values = (
            arguments.parent_result,
            arguments.parent_result_sha256,
            arguments.parent_checkpoint,
            arguments.parent_checkpoint_sha256,
        )
        if any(value is None for value in parent_values):
            raise PilotWorkerError("compact run requires complete parent artifacts")
        raw_parent_result = Path(arguments.parent_result).absolute()
        raw_parent_checkpoint = Path(arguments.parent_checkpoint).absolute()
        _validate_directory_chain(
            output_root, raw_parent_result.parent, "parent result"
        )
        _validate_directory_chain(
            output_root, raw_parent_checkpoint.parent, "parent checkpoint"
        )
        artifacts, metrics, stream = _compact_run(
            context,
            config,
            plan,
            attempt_dir,
            output_root,
            raw_parent_result.resolve(strict=True),
            arguments.parent_result_sha256,
            raw_parent_checkpoint.resolve(strict=True),
            arguments.parent_checkpoint_sha256,
        )
    if _read_private(config_path, maximum=1024 * 1024, name="config") != raw_config:
        raise PilotWorkerError("configuration changed during worker execution")
    repeated_inventory = _inventory(repo, config_path, runner_path)
    for name in ("package_tree", "bundle", _WORKER_RELATIVE, _RUNNER_RELATIVE):
        if repeated_inventory[name] != inventory[name]:
            raise PilotWorkerError("executable provenance changed during worker execution")
    return {
        "schema_version": 1,
        "status": "success",
        "run_id": run.run_id,
        "run_sha256": _run_hash(run),
        "model_id": run.model_id,
        "pair_id": run.pair_id,
        "family": run.family,
        "role": run.role,
        "attempt_number": attempt,
        "plan_sha256": arguments.plan_sha256,
        "provenance": provenance,
        "artifacts": artifacts,
        "metrics": metrics,
        "stream": stream,
        "error": None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    result_path = Path(arguments.result_path).absolute()
    try:
        document = _execute(arguments)
        code = 0
    except Exception as error:
        document = {
            "schema_version": 1,
            "status": "failure",
            "run_id": arguments.run_id,
            "run_sha256": "0" * 64,
            "model_id": "unknown",
            "pair_id": "unknown",
            "family": "unknown",
            "role": "unknown",
            "attempt_number": arguments.attempt_number,
            "plan_sha256": arguments.plan_sha256,
            "provenance": {
                "code_commit": arguments.code_commit,
                "code_tree": arguments.code_tree,
                "raw_config_sha256": arguments.raw_config_sha256,
                "semantic_config_sha256": arguments.semantic_config_sha256,
                "executable_bundle_sha256": arguments.executable_bundle_sha256,
                "parent_runner_sha256": arguments.parent_runner_sha256,
                "worker_sha256": arguments.worker_sha256,
                "package_tree_sha256": arguments.package_tree_sha256,
            },
            "artifacts": {
                "checkpoints": [],
                "final_checkpoint": None,
                "compact_artifact": None,
            },
            "metrics": None,
            "stream": None,
            "error": {"type": type(error).__name__, "message": str(error)[:4096]},
        }
        code = 1
    try:
        _write_atomic(result_path, _canonical(document), immutable=True)
    except Exception as write_error:
        print(f"worker result write failed: {write_error}", file=sys.stderr)
        return 2
    if code:
        print(f"worker failed: {document['error']}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
