#!/usr/bin/env python3
"""Run the non-claiming Milestone-4 pilot through isolated worker processes.

The parent is the sole owner of campaign-manifest transitions.  Workers receive
one immutable resolved run at a time, write only beneath their deterministic
external attempt directory, and return bounded canonical JSON.  This pilot
exercises plumbing; it does not make scientific, quality, runtime, scaling, or
platform claims.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
import os
import platform
from pathlib import Path, PurePosixPath
import random
import re
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import psutil
import numpy as np
import torch

from tnlm_v3.campaign_checkpoint import (
    campaign_model_fingerprint,
    campaign_checkpoint_contract,
    deserialize_campaign_checkpoint,
    serialize_campaign_checkpoint,
)
from tnlm_v3.campaign_execution import (
    CampaignRunContext,
    build_campaign_optimizer,
    build_campaign_source_model,
    campaign_batch_sha256,
    derive_campaign_compact_model,
    generate_campaign_evaluation_batch,
    generate_campaign_training_batch,
)
from tnlm_v3.compact_artifact import (
    deserialize_compact_binding_model,
    serialize_compact_binding_model,
)
from tnlm_v3.data import BindingEventKind

from tnlm_v3.campaign_config import (
    CampaignStage,
    Milestone4CampaignConfig,
    ResolvedCampaignRun,
    campaign_plan_sha256,
    load_milestone4_campaign_config,
    resolve_campaign_plan,
)
from tnlm_v3.campaign_manifest import (
    ArtifactReference,
    CampaignAuthority,
    CampaignManifest,
    CampaignManifestError,
    CampaignRunState,
    InProgressAttemptError,
    ReconciliationRequiredError,
    campaign_manifest_path,
    campaign_manifest_sha256,
    complete_campaign_attempt,
    fail_campaign_attempt,
    heartbeat_campaign_attempt,
    initialize_campaign_manifest,
    load_campaign_attempt_record,
    load_campaign_manifest,
    make_artifact_reference,
    reconcile_campaign_manifest,
    start_campaign_attempt,
)
import tnlm_v3.campaign_config as _campaign_config_module
import tnlm_v3.campaign_checkpoint as _campaign_checkpoint_module
import tnlm_v3.campaign_execution as _campaign_execution_module
import tnlm_v3.campaign_manifest as _campaign_manifest_module


_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_REGULAR_GIT_MODES = {"100644", "100755"}
_PACKAGE_DOMAIN = b"tnlm-v3-package-tree-v1\0"
_BUNDLE_DOMAIN = b"tnlm-v3-pilot-bundle-v1\0"
_MAX_RESULT_BYTES = 4 * 1024 * 1024
_MAX_OUTPUT_BYTES = 1024 * 1024
_MAX_JSON_NODES = 100_000
_MAX_JSON_DEPTH = 48
_MAX_CAPTURE_BYTES = 64 * 1024
_WORKER_TIMEOUT_SECONDS = 3600.0
_WORKER_RELATIVE = "v3/scripts/run_milestone4_pilot_worker.py"
_RUNNER_RELATIVE = "v3/scripts/run_milestone4_pilot.py"
_PACKAGE_PREFIX = "v3/src/tnlm_v3/"
_STREAM_PREFIX_DOMAIN = b"tnlm-v3-m4-training-stream-prefix-v1\0"
_UNSET = object()


class PilotRunnerError(RuntimeError):
    """Raised when pilot provenance, execution, or output is invalid."""


@dataclass(frozen=True)
class BoundFile:
    path: str
    git_blob_sha1: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PilotProvenance:
    repo_root: Path
    config_path: Path
    runner_path: Path
    worker_path: Path
    code_commit: str
    code_tree: str
    raw_config_sha256: str
    parent_runner_sha256: str
    worker_sha256: str
    package_tree_sha256: str
    executable_bundle_sha256: str
    bound_files: tuple[BoundFile, ...]


@dataclass(frozen=True)
class PilotRunSummary:
    run_id: str
    model_id: str
    pair_id: str
    role: str
    attempt_number: int
    resumed: bool
    result: ArtifactReference
    output: ArtifactReference
    final_checkpoint: ArtifactReference | None
    compact_artifact: ArtifactReference | None
    validation_batch_hashes: tuple[tuple[int, str], ...]
    training_batch_hashes: tuple[str, ...]
    training_token_counts: tuple[int, ...]
    stream_prefix_sha256: str | None
    initial_model_fingerprint: str | None
    final_model_fingerprint: str
    validation_metrics: tuple[ValidationLengthSummary, ...] = ()
    parameter_count: int | None = None
    checkpoints: tuple[ArtifactReference, ...] = ()
    compact_lineage_json: str | None = None


@dataclass(frozen=True)
class PilotCampaignSummary:
    campaign_id: str
    plan_sha256: str
    manifest_generation: int
    code_commit: str
    code_tree: str
    raw_config_sha256: str
    semantic_config_sha256: str
    package_tree_sha256: str
    executable_bundle_sha256: str
    runs: tuple[PilotRunSummary, ...]
    claim_eligible: bool = False
    promotion: ArtifactReference | None = None
    promotion_decision: str | None = None


@dataclass(frozen=True)
class ValidationLengthSummary:
    """Trusted, immutable validation facts retained for campaign aggregation."""

    length: int
    query_correct: int
    query_count: int
    seen_correct: int
    seen_count: int
    heldout_correct: int
    heldout_count: int
    structural_values: tuple[tuple[str, int], ...]
    routing_json: str | None


@dataclass(frozen=True)
class _Capture:
    text: str
    sha256: str
    size_bytes: int
    truncated: bool


@dataclass(frozen=True)
class _ProcessResult:
    argv: tuple[str, ...]
    pid: int
    process_create_time_ns: int
    returncode: int
    timed_out: bool
    stdout: _Capture
    stderr: _Capture


@dataclass(frozen=True)
class _WorkerResult:
    raw: bytes
    document: Mapping[str, Any]
    checkpoints: tuple[ArtifactReference, ...]
    final_checkpoint: ArtifactReference | None
    compact_artifact: ArtifactReference | None
    status: str
    error_message: str | None
    validation_batch_hashes: tuple[tuple[int, str], ...]
    training_batch_hashes: tuple[str, ...]
    training_token_counts: tuple[int, ...]
    stream_prefix_sha256: str | None
    completed_step: int
    initial_model_fingerprint: str | None
    final_model_fingerprint: str
    validation_metrics: tuple[ValidationLengthSummary, ...]
    parameter_count: int
    compact_lineage_json: str | None


@dataclass(frozen=True)
class _ExpectedStreams:
    training_hashes: tuple[str, ...]
    training_tokens: tuple[int, ...]
    prefix_by_cursor: tuple[str, ...]
    validation_hashes: tuple[tuple[int, str], ...]
    training_query_counts: tuple[int, ...]
    validation_counts: tuple[tuple[int, int, int, int], ...]


def _sha(value: object, name: str, *, forty: bool = False) -> str:
    pattern = _HEX40 if forty else _HEX64
    if not isinstance(value, str) or not pattern.fullmatch(value):
        width = 40 if forty else 64
        raise PilotRunnerError(f"{name} must be {width} lowercase hexadecimal characters")
    return value


def _plain_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PilotRunnerError(f"{name} must be an integer")
    if value < minimum or value > 2**63 - 1:
        raise PilotRunnerError(f"{name} is outside the signed 64-bit range")
    return value


def _exact(mapping: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        raise PilotRunnerError(
            f"invalid {name} keys; missing={missing}, unknown={unknown}"
        )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_constant(value: str) -> object:
    raise PilotRunnerError(f"non-finite JSON number {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PilotRunnerError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _validate_json_tree(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise PilotRunnerError("worker JSON exceeds its structural limit")
        if item is None or isinstance(item, (str, bool)):
            continue
        if isinstance(item, int) and not isinstance(item, bool):
            _plain_int(item, "JSON integer")
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise PilotRunnerError("worker JSON contains a non-finite number")
            continue
        if isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
            continue
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise PilotRunnerError("worker JSON object keys must be strings")
            pending.extend((child, depth + 1) for child in item.values())
            continue
        raise PilotRunnerError("worker JSON contains a non-plain value")


def _parse_canonical_json(raw: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PilotRunnerError(f"{name} must be UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, RecursionError) as error:
        raise PilotRunnerError(f"{name} must be strict JSON") from error
    _validate_json_tree(value)
    if not isinstance(value, dict):
        raise PilotRunnerError(f"{name} must be a JSON object")
    if _canonical_bytes(value) != raw:
        raise PilotRunnerError(f"{name} must be canonical JSON")
    return value


def _run_git(repo_root: Path, *arguments: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={repo_root}", *arguments],
        cwd=repo_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        stderr = completed.stderr.decode("utf-8", "replace")[:4096]
        raise PilotRunnerError(f"Git provenance command failed: {stderr}")
    if binary:
        return completed.stdout
    try:
        return completed.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PilotRunnerError("Git provenance output is not UTF-8") from error


def _real_directory(path: str | Path, name: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise PilotRunnerError(f"{name} must be absolute")
    try:
        info = candidate.lstat()
    except OSError as error:
        raise PilotRunnerError(f"{name} does not exist") from error
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        stat.S_ISLNK(info.st_mode)
        or bool(getattr(info, "st_file_attributes", 0) & reparse)
        or not stat.S_ISDIR(info.st_mode)
    ):
        raise PilotRunnerError(f"{name} must be a real directory")
    resolved = candidate.resolve(strict=True)
    if resolved != Path(os.path.abspath(candidate)):
        raise PilotRunnerError(f"{name} must not traverse a linked directory")
    return resolved


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _repo_relative(repo_root: Path, path: str | Path, name: str) -> tuple[Path, str]:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise PilotRunnerError(f"{name} must be absolute")
    absolute = Path(os.path.abspath(candidate))
    if not _is_within(absolute, repo_root):
        raise PilotRunnerError(f"{name} must be beneath repo_root")
    relative = absolute.relative_to(repo_root).as_posix()
    if PurePosixPath(relative).as_posix() != relative or "\\" in relative:
        raise PilotRunnerError(f"{name} is not a normalized repository path")
    return absolute, relative


def _git_index(repo_root: Path) -> dict[str, tuple[str, str]]:
    raw = _run_git(repo_root, "ls-files", "--stage", "-z", binary=True)
    assert isinstance(raw, bytes)
    result: dict[str, tuple[str, str]] = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        try:
            metadata, raw_path = entry.split(b"\t", 1)
            mode, blob, stage = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            raise PilotRunnerError("Git index entry is malformed") from error
        if stage != "0" or path in result:
            raise PilotRunnerError("Git index contains an unmerged or duplicate path")
        result[path] = (mode, blob)
    return result


def _read_private_file(path: Path, *, maximum_bytes: int, name: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise PilotRunnerError(f"{name} is missing") from error
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        stat.S_ISLNK(before.st_mode)
        or bool(getattr(before, "st_file_attributes", 0) & reparse)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > maximum_bytes
    ):
        raise PilotRunnerError(f"{name} must be a bounded private regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_nlink,
        )
        identity_opened = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_nlink,
        )
        if identity_opened != identity_before:
            raise PilotRunnerError(f"{name} changed before it was opened")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(maximum_bytes + 1)
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size:
        raise PilotRunnerError(f"{name} was not read in full")
    if len(raw) > maximum_bytes:
        raise PilotRunnerError(f"{name} exceeds its byte limit")
    after = path.lstat()
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_nlink,
    )
    if identity_after != identity_before:
        raise PilotRunnerError(f"{name} changed while it was read")
    return raw


def _bind_file(
    repo_root: Path,
    relative: str,
    index: Mapping[str, tuple[str, str]],
) -> BoundFile:
    entry = index.get(relative)
    if entry is None:
        raise PilotRunnerError(f"required input {relative!r} is not committed")
    mode, blob = entry
    if mode not in _REGULAR_GIT_MODES or not _HEX40.fullmatch(blob):
        raise PilotRunnerError(f"required input {relative!r} is not a regular Git blob")
    path = repo_root / PurePosixPath(relative)
    try:
        info = path.lstat()
    except OSError as error:
        raise PilotRunnerError(f"required input {relative!r} is missing") from error
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        stat.S_ISLNK(info.st_mode)
        or bool(getattr(info, "st_file_attributes", 0) & reparse)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
    ):
        raise PilotRunnerError(f"required input {relative!r} is linked or non-regular")
    committed = _run_git(repo_root, "cat-file", "blob", blob, binary=True)
    assert isinstance(committed, bytes)
    current = _read_private_file(
        path,
        maximum_bytes=max(len(committed), 1),
        name=f"required input {relative!r}",
    )
    if current != committed:
        raise PilotRunnerError(f"required input {relative!r} differs from committed bytes")
    return BoundFile(
        path=relative,
        git_blob_sha1=blob,
        size_bytes=len(committed),
        sha256=hashlib.sha256(committed).hexdigest(),
    )


def _inventory_digest(domain: bytes, entries: Sequence[BoundFile]) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    for entry in sorted(entries, key=lambda item: item.path.encode("utf-8")):
        digest.update(entry.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry.size_bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def collect_pilot_provenance(
    repo_root: str | Path,
    config_path: str | Path,
    worker_path: str | Path,
    *,
    expected_commit: str | None = None,
) -> PilotProvenance:
    """Bind the exact clean checkout, committed inputs, package, and worker."""

    repo = _real_directory(repo_root, "repo_root")
    top = str(_run_git(repo, "rev-parse", "--show-toplevel")).strip()
    if Path(top).resolve(strict=True) != repo:
        raise PilotRunnerError("repo_root is not the exact Git worktree root")
    commit = str(_run_git(repo, "rev-parse", "HEAD")).strip()
    tree = str(_run_git(repo, "rev-parse", "HEAD^{tree}")).strip()
    _sha(commit, "code_commit", forty=True)
    _sha(tree, "code_tree", forty=True)
    if expected_commit is not None and commit != _sha(
        expected_commit, "expected_commit", forty=True
    ):
        raise PilotRunnerError("HEAD does not match expected_commit")
    status = str(
        _run_git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    )
    if status:
        raise PilotRunnerError("pilot requires a completely clean checkout")
    config, config_relative = _repo_relative(repo, config_path, "config_path")
    worker, worker_relative = _repo_relative(repo, worker_path, "worker_path")
    if worker_relative != _WORKER_RELATIVE:
        raise PilotRunnerError(f"worker must be {_WORKER_RELATIVE}")
    runner = Path(__file__).resolve(strict=True)
    expected_runner = (repo / _RUNNER_RELATIVE).resolve(strict=True)
    if runner != expected_runner:
        raise PilotRunnerError("the running parent is not the committed pilot runner")
    index = _git_index(repo)
    package_paths = sorted(
        (
            path
            for path in index
            if path.startswith(_PACKAGE_PREFIX) and index[path][0] in _REGULAR_GIT_MODES
        ),
        key=lambda value: value.encode("utf-8"),
    )
    if not package_paths:
        raise PilotRunnerError("the committed package inventory is empty")
    required = sorted(
        set(package_paths) | {config_relative, _RUNNER_RELATIVE, worker_relative},
        key=lambda value: value.encode("utf-8"),
    )
    bound = tuple(_bind_file(repo, path, index) for path in required)
    by_path = {entry.path: entry for entry in bound}
    package = tuple(by_path[path] for path in package_paths)
    return PilotProvenance(
        repo_root=repo,
        config_path=config,
        runner_path=runner,
        worker_path=worker,
        code_commit=commit,
        code_tree=tree,
        raw_config_sha256=by_path[config_relative].sha256,
        parent_runner_sha256=by_path[_RUNNER_RELATIVE].sha256,
        worker_sha256=by_path[worker_relative].sha256,
        package_tree_sha256=_inventory_digest(_PACKAGE_DOMAIN, package),
        executable_bundle_sha256=_inventory_digest(_BUNDLE_DOMAIN, bound),
        bound_files=bound,
    )


def _verify_provenance_unchanged(provenance: PilotProvenance) -> None:
    repeated = collect_pilot_provenance(
        provenance.repo_root,
        provenance.config_path,
        provenance.worker_path,
        expected_commit=provenance.code_commit,
    )
    if repeated != provenance:
        raise PilotRunnerError("pilot provenance changed during execution")


def _verify_import_origins(provenance: PilotProvenance) -> None:
    package_root = (provenance.repo_root / "v3" / "src" / "tnlm_v3").resolve(
        strict=True
    )
    bound = {entry.path for entry in provenance.bound_files}
    # Check the entire already-imported package surface, not just the modules
    # this file names directly.  An ambient package must not be able to mix a
    # committed campaign module with an unbound sibling module.
    modules = tuple(
        module
        for name, module in sorted(sys.modules.items())
        if (name == "tnlm_v3" or name.startswith("tnlm_v3."))
        and module is not None
        and getattr(module, "__file__", None) is not None
    )
    if not modules:
        raise PilotRunnerError("no imported campaign package modules were found")
    for module in modules:
        filename = getattr(module, "__file__", None)
        if not isinstance(filename, str):
            raise PilotRunnerError("campaign module has no filesystem origin")
        origin = Path(filename).resolve(strict=True)
        if not _is_within(origin, package_root):
            raise PilotRunnerError("campaign module was imported outside the bound package")
        relative = origin.relative_to(provenance.repo_root).as_posix()
        if relative not in bound:
            raise PilotRunnerError("campaign module origin is not in the bound inventory")


def _run_sha256(run: ResolvedCampaignRun) -> str:
    return hashlib.sha256(run.canonical_json().encode("utf-8")).hexdigest()


def _expected_streams(
    config: Milestone4CampaignConfig,
    run: ResolvedCampaignRun,
) -> _ExpectedStreams:
    context = CampaignRunContext(config=config, run=run)
    validation: list[tuple[int, str]] = []
    validation_counts: list[tuple[int, int, int, int]] = []
    for length in run.data.validation.lengths:
        batch = generate_campaign_evaluation_batch(
            context,
            stream="validation",
            length=length,
        )
        query = batch.inputs.valid_mask & (
            batch.inputs.event_kinds == int(BindingEventKind.QUERY)
        )
        heldout = query & batch.evaluation.heldout_combination_mask
        seen = query & ~batch.evaluation.heldout_combination_mask
        validation.append((length, campaign_batch_sha256(batch)))
        validation_counts.append(
            (
                int(batch.lengths.numel()),
                int(query.sum().item()),
                int(seen.sum().item()),
                int(heldout.sum().item()),
            )
        )
    if run.role == "derived_compact":
        return _ExpectedStreams(
            (), (), (), tuple(validation), (), tuple(validation_counts)
        )
    assert run.training is not None
    hashes: list[str] = []
    tokens: list[int] = []
    prefixes = [hashlib.sha256(_STREAM_PREFIX_DOMAIN).hexdigest()]
    query_counts: list[int] = []
    for step in range(run.training.optimizer_steps):
        batch = generate_campaign_training_batch(context, step=step)
        batch_hash = campaign_batch_sha256(batch)
        token_count = int(batch.lengths.sum().item())
        if token_count != sum(run.data.train.length_schedule):
            raise PilotRunnerError("trusted training batch token count changed")
        hashes.append(batch_hash)
        tokens.append(token_count)
        query_counts.append(
            int(
                (
                    batch.inputs.valid_mask
                    & (batch.inputs.event_kinds == int(BindingEventKind.QUERY))
                ).sum().item()
            )
        )
        prefixes.append(
            hashlib.sha256(
                bytes.fromhex(prefixes[-1])
                + step.to_bytes(8, "little", signed=False)
                + bytes.fromhex(batch_hash)
            ).hexdigest()
        )
    return _ExpectedStreams(
        tuple(hashes),
        tuple(tokens),
        tuple(prefixes),
        tuple(validation),
        tuple(query_counts),
        tuple(validation_counts),
    )


def _expected_checkpoint_cursors(run: ResolvedCampaignRun) -> tuple[int, ...]:
    if run.role != "trainable_source":
        return ()
    assert run.training is not None
    cursors = list(
        range(
            run.training.checkpoint_interval,
            run.training.optimizer_steps + 1,
            run.training.checkpoint_interval,
        )
    )
    if not cursors or cursors[-1] != run.training.optimizer_steps:
        cursors.append(run.training.optimizer_steps)
    return tuple(cursors)


def _checkpoint_cursor(reference: ArtifactReference, attempt_prefix: str) -> int:
    expected_parent = PurePosixPath(attempt_prefix)
    path = PurePosixPath(reference.path)
    if path.parent != expected_parent:
        raise PilotRunnerError("checkpoint path is outside its exact attempt directory")
    match = re.fullmatch(r"checkpoint-step-([0-9]{8})\.twcp", path.name)
    if match is None:
        raise PilotRunnerError("checkpoint filename does not bind its cursor")
    return int(match.group(1))


def _verify_checkpoint_semantics(
    reference: ArtifactReference,
    *,
    cursor: int,
    expected_streams: _ExpectedStreams,
    config: Milestone4CampaignConfig,
    run: ResolvedCampaignRun,
    output_root: Path,
) -> str:
    if run.role != "trainable_source" or run.training is None:
        raise PilotRunnerError("only source runs may own campaign checkpoints")
    if cursor not in _expected_checkpoint_cursors(run):
        raise PilotRunnerError("checkpoint cursor is outside the locked schedule")
    path = output_root / PurePosixPath(reference.path)
    raw = _read_private_file(
        path,
        maximum_bytes=max(reference.size_bytes, 1),
        name="campaign checkpoint",
    )
    if (
        len(raw) != reference.size_bytes
        or hashlib.sha256(raw).hexdigest() != reference.sha256
    ):
        raise PilotRunnerError("campaign checkpoint reference does not match its bytes")
    context = CampaignRunContext(config=config, run=run)
    model = build_campaign_source_model(context)
    optimizer = build_campaign_optimizer(context, model)
    contract = campaign_checkpoint_contract(model, optimizer)
    caller_torch = torch.random.get_rng_state()
    caller_python = random.getstate()
    fingerprint: str | None = None
    try:
        restored_model, restored_optimizer, resume = deserialize_campaign_checkpoint(
            raw,
            expected_run_spec_sha256=_run_sha256(run),
            expected_stream_prefix_sha256=expected_streams.prefix_by_cursor[cursor],
            expected_contract=contract,
            device="cpu",
        )
        if resume.global_step != cursor or resume.data_cursor != cursor:
            raise PilotRunnerError("checkpoint resume cursor does not match its filename")
        if serialize_campaign_checkpoint(restored_model, restored_optimizer, resume) != raw:
            raise PilotRunnerError("campaign checkpoint is not canonical on reserialization")
        fingerprint = campaign_model_fingerprint(restored_model)
    finally:
        torch.random.set_rng_state(caller_torch)
        random.setstate(caller_python)
    assert fingerprint is not None
    return fingerprint


def _finite_number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotRunnerError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise PilotRunnerError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise PilotRunnerError(f"{name} is below its minimum")
    if maximum is not None and result > maximum:
        raise PilotRunnerError(f"{name} is above its maximum")
    return result


def _verify_compact_semantics(
    reference: ArtifactReference,
    compact_metrics: Mapping[str, Any],
    *,
    run: ResolvedCampaignRun,
    parent: PilotRunSummary,
    authority: CampaignAuthority,
    output_root: Path,
) -> None:
    if run.role != "derived_compact" or run.parent_run_id != parent.run_id:
        raise PilotRunnerError("compact artifact has the wrong parent run")
    if parent.final_checkpoint is None:
        raise PilotRunnerError("compact parent has no final checkpoint")
    parent_runs = tuple(
        candidate
        for candidate in authority.resolved_plan
        if candidate.run_id == parent.run_id
    )
    if len(parent_runs) != 1:
        raise PilotRunnerError("compact parent is not unique in the exact plan")
    parent_run = parent_runs[0]
    if parent_run.role != "trainable_source" or parent_run.training is None:
        raise PilotRunnerError("compact parent is not a trainable source")
    parent_streams = _expected_streams(authority.config, parent_run)
    parent_cursor = _checkpoint_cursor(
        parent.final_checkpoint,
        _attempt_relative(parent.run_id, parent.attempt_number),
    )
    if parent_cursor != parent_run.training.optimizer_steps:
        raise PilotRunnerError("compact parent checkpoint is not terminal")
    checkpoint_raw = _read_private_file(
        output_root / PurePosixPath(parent.final_checkpoint.path),
        maximum_bytes=max(parent.final_checkpoint.size_bytes, 1),
        name="compact parent checkpoint",
    )
    if (
        len(checkpoint_raw) != parent.final_checkpoint.size_bytes
        or hashlib.sha256(checkpoint_raw).hexdigest()
        != parent.final_checkpoint.sha256
    ):
        raise PilotRunnerError("compact parent checkpoint bytes changed")
    artifact_raw = _read_private_file(
        output_root / PurePosixPath(reference.path),
        maximum_bytes=max(reference.size_bytes, 1),
        name="compact artifact",
    )
    if (
        len(artifact_raw) != reference.size_bytes
        or hashlib.sha256(artifact_raw).hexdigest() != reference.sha256
    ):
        raise PilotRunnerError("compact artifact bytes changed")
    caller_torch = torch.random.get_rng_state()
    caller_python = random.getstate()
    try:
        parent_context = CampaignRunContext(config=authority.config, run=parent_run)
        fresh = build_campaign_source_model(parent_context)
        fresh_optimizer = build_campaign_optimizer(parent_context, fresh)
        contract = campaign_checkpoint_contract(fresh, fresh_optimizer)
        restored, restored_optimizer, resume = deserialize_campaign_checkpoint(
            checkpoint_raw,
            expected_run_spec_sha256=_run_sha256(parent_run),
            expected_stream_prefix_sha256=parent_streams.prefix_by_cursor[parent_cursor],
            expected_contract=contract,
            device="cpu",
        )
        if resume.global_step != parent_cursor or resume.data_cursor != parent_cursor:
            raise PilotRunnerError("compact parent checkpoint cursor changed")
        # The checkpoint codec deliberately reconstructs an unbound model.
        # Copy its exact state into the fresh CampaignRunContext-bound factory
        # model/optimizer before invoking the execution API, then prove that
        # the bound copy still serializes to the exact authoritative bytes.
        fresh.load_state_dict(restored.state_dict(), strict=True)
        restored_modes = dict(restored.named_modules())
        for name, module in fresh.named_modules():
            module.training = restored_modes[name].training
        fresh_optimizer.load_state_dict(restored_optimizer.state_dict())
        if serialize_campaign_checkpoint(fresh, fresh_optimizer, resume) != checkpoint_raw:
            raise PilotRunnerError("bound compact parent differs from its checkpoint")
        if campaign_model_fingerprint(fresh) != parent.final_model_fingerprint:
            raise PilotRunnerError("compact parent model fingerprint changed")
        derivation = derive_campaign_compact_model(
            parent_context,
            CampaignRunContext(config=authority.config, run=run),
            fresh,
            fresh_optimizer,
        )
        # The worker evaluates before serializing, and the artifact format
        # intentionally preserves every nested train/eval flag.  Mirror that
        # declared inference artifact convention before byte comparison.
        derivation.model.eval()
        expected_artifact = serialize_compact_binding_model(
            derivation.model,
            derivation.manifest,
            derivation.selection,
        )
        if expected_artifact != artifact_raw:
            raise PilotRunnerError("compact artifact is not the exact trusted derivation")
        decoded_model, decoded_manifest, decoded_selection = (
            deserialize_compact_binding_model(
                artifact_raw,
                expected_source_fingerprint=derivation.manifest.source_model_fingerprint,
                expected_manifest_fingerprint=derivation.manifest.fingerprint(),
                expected_selection_fingerprint=derivation.selection.fingerprint(),
                device="cpu",
            )
        )
        if serialize_compact_binding_model(
            decoded_model, decoded_manifest, decoded_selection
        ) != artifact_raw:
            raise PilotRunnerError("compact artifact is not canonical on reserialization")
        expected_metrics = {
            "parent_run_id": parent.run_id,
            "parent_result": {
                "path": parent.result.path,
                "size_bytes": parent.result.size_bytes,
                "sha256": parent.result.sha256,
            },
            "parent_checkpoint": {
                "path": parent.final_checkpoint.path,
                "size_bytes": parent.final_checkpoint.size_bytes,
                "sha256": parent.final_checkpoint.sha256,
            },
            "parent_model_fingerprint": parent.final_model_fingerprint,
            "selection_fingerprint": derivation.selection.fingerprint(),
            "manifest_fingerprint": derivation.manifest.fingerprint(),
            "exported_model_fingerprint": derivation.manifest.exported_model_fingerprint,
            "compact_artifact_sha256": reference.sha256,
        }
        if dict(compact_metrics) != expected_metrics:
            raise PilotRunnerError("compact lineage metrics differ from trusted derivation")
    finally:
        torch.random.set_rng_state(caller_torch)
        random.setstate(caller_python)


def _load_authority(
    provenance: PilotProvenance,
) -> tuple[Milestone4CampaignConfig, tuple[ResolvedCampaignRun, ...], CampaignAuthority]:
    raw_before = _read_private_file(
        provenance.config_path,
        maximum_bytes=1024 * 1024,
        name="configuration",
    )
    if hashlib.sha256(raw_before).hexdigest() != provenance.raw_config_sha256:
        raise PilotRunnerError("configuration changed after provenance binding")
    config = load_milestone4_campaign_config(provenance.config_path)
    if config.stage not in {CampaignStage.PILOT, CampaignStage.SCREEN} or config.claim_eligible:
        raise PilotRunnerError(
            "the runner accepts only non-claiming pilot or screen configs"
        )
    if config.data.test is not None or config.data.scaling is not None:
        raise PilotRunnerError("pilot/screen configuration must not expose test or scaling data")
    plan = resolve_campaign_plan(
        config,
        provenance.code_commit,
        provenance.code_tree,
        provenance.raw_config_sha256,
        provenance.executable_bundle_sha256,
    )
    source_count = sum(run.role == "trainable_source" for run in plan)
    compact_count = sum(run.role == "derived_compact" for run in plan)
    if config.stage is CampaignStage.PILOT:
        if len(plan) != 7 or source_count != 6 or compact_count != 1 or len(config.pairs) != 1:
            raise PilotRunnerError("pilot requires exactly six sources and one compact child")
    else:
        if (
            len(config.pairs) != 3
            or compact_count < 3
            or source_count < 18
            or len(plan) != len(config.models) * 3
        ):
            raise PilotRunnerError(
                "screen requires exactly three complete pairs and at least one compact lineage"
            )
        expected_pair_models = {model.model_id for model in config.models}
        for pair in config.pairs:
            observed = {run.model_id for run in plan if run.pair_id == pair.pair_id}
            if observed != expected_pair_models:
                raise PilotRunnerError("screen plan does not contain the exact model matrix per pair")
        references = tuple(
            model
            for model in config.models
            if model.model_id == config.quality.primary_reference_model_id
            and model.family == "routed"
            and model.role == "trainable_source"
            and model.routing_mode == "curriculum"
        )
        oracles = tuple(
            model
            for model in config.models
            if model.family == "routed"
            and model.role == "trainable_source"
            and model.routing_mode == "oracle"
        )
        if len(references) != 1 or len(oracles) != 1:
            raise PilotRunnerError(
                "screen routing authority requires one curriculum reference and one oracle"
            )
        if any(
            model.role == "derived_compact"
            and model.parent_model_id != references[0].model_id
            for model in config.models
        ):
            raise PilotRunnerError("every screen compact must descend from the routed reference")
    plan_sha = campaign_plan_sha256(config, plan)
    authority = CampaignAuthority(
        config=config,
        resolved_plan=plan,
        plan_sha256=plan_sha,
        code_commit=provenance.code_commit,
        code_tree=provenance.code_tree,
        raw_config_sha256=provenance.raw_config_sha256,
        executable_bundle_sha256=provenance.executable_bundle_sha256,
        working_tree_clean=True,
    )
    if _read_private_file(
        provenance.config_path,
        maximum_bytes=1024 * 1024,
        name="configuration",
    ) != raw_before:
        raise PilotRunnerError("configuration changed while it was parsed")
    return config, plan, authority


def _validated_output_root(output_root: str | Path, repo_root: Path) -> Path:
    output = _real_directory(output_root, "output_root")
    if _is_within(output, repo_root) or _is_within(repo_root, output):
        raise PilotRunnerError("output_root and repo_root must be disjoint")
    return output


def _ensure_private_directory(output_root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PilotRunnerError("attempt directory is not normalized")
    current = output_root
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    for part in path.parts:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        info = current.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or bool(getattr(info, "st_file_attributes", 0) & reparse)
            or not stat.S_ISDIR(info.st_mode)
        ):
            raise PilotRunnerError("attempt directory contains a link or non-directory")
    return current


def _attempt_relative(run_id: str, attempt_number: int) -> str:
    _sha(run_id, "run_id")
    _plain_int(attempt_number, "attempt_number", minimum=1)
    return f"artifacts/{run_id}/attempt-{attempt_number:06d}"


def _manifest_state(manifest: CampaignManifest, run_id: str) -> CampaignRunState:
    matches = tuple(state for state in manifest.runs if state.run_id == run_id)
    if len(matches) != 1:
        raise PilotRunnerError("manifest does not contain the exact run once")
    return matches[0]


def _load_or_initialize_manifest(
    authority: CampaignAuthority,
    output_root: Path,
    repo_root: Path,
) -> CampaignManifest:
    options = {"external_root": output_root, "checkout_root": repo_root}
    # campaign_manifest_path deliberately validates every existing ancestor;
    # on first use the campaign namespace does not exist yet.
    path = output_root / "campaigns" / authority.config.campaign_id / "manifest.json"
    if not path.exists():
        return initialize_campaign_manifest(authority, **options)
    try:
        return load_campaign_manifest(authority, **options)
    except ReconciliationRequiredError:
        try:
            return reconcile_campaign_manifest(authority, **options)
        except InProgressAttemptError:
            return load_campaign_manifest(authority, **options)


def _capture_pipe(stream, destination: dict[str, _Capture], name: str) -> None:  # type: ignore[no-untyped-def]
    digest = hashlib.sha256()
    retained = bytearray()
    total = 0
    while True:
        chunk = stream.read(8192)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
        if len(retained) < _MAX_CAPTURE_BYTES:
            retained.extend(chunk[: _MAX_CAPTURE_BYTES - len(retained)])
    destination[name] = _Capture(
        text=bytes(retained).decode("utf-8", "replace"),
        sha256=digest.hexdigest(),
        size_bytes=total,
        truncated=total > len(retained),
    )


def _run_worker_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    pythonpath: Path,
    lease_path: Path,
    run_id: str,
    attempt_number: int,
    timeout_seconds: float = _WORKER_TIMEOUT_SECONDS,
) -> _ProcessResult:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise PilotRunnerError("worker timeout must be finite and positive")
    allow = ("PATH", "SystemRoot", "WINDIR", "TEMP", "TMP", "COMSPEC")
    environment = {name: os.environ[name] for name in allow if name in os.environ}
    environment.update(
        {
            "PYTHONPATH": str(pythonpath),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        create_time_ns = int(
            round(psutil.Process(process.pid).create_time() * 1_000_000_000)
        )
    except (psutil.Error, OSError) as error:
        process.kill()
        process.wait()
        raise PilotRunnerError("cannot bind the worker process creation time") from error
    try:
        _atomic_mutable_json(
            lease_path,
            _process_lease_document(
                run_id=run_id,
                attempt_number=attempt_number,
                pid=process.pid,
                process_create_time_ns=create_time_ns,
                status="running",
                returncode=None,
            ),
        )
    except Exception:
        process.kill()
        process.wait()
        raise
    assert process.stdout is not None and process.stderr is not None
    captured: dict[str, _Capture] = {}
    threads = [
        threading.Thread(
            target=_capture_pipe,
            args=(process.stdout, captured, "stdout"),
            daemon=True,
        ),
        threading.Thread(
            target=_capture_pipe,
            args=(process.stderr, captured, "stderr"),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        returncode = process.wait()
    for thread in threads:
        thread.join(timeout=30)
        if thread.is_alive():
            raise PilotRunnerError("worker output capture did not terminate")
    _atomic_mutable_json(
        lease_path,
        _process_lease_document(
            run_id=run_id,
            attempt_number=attempt_number,
            pid=process.pid,
            process_create_time_ns=create_time_ns,
            status="exited",
            returncode=returncode,
        ),
    )
    empty = _Capture("", hashlib.sha256(b"").hexdigest(), 0, False)
    return _ProcessResult(
        argv=tuple(argv),
        pid=process.pid,
        process_create_time_ns=create_time_ns,
        returncode=returncode,
        timed_out=timed_out,
        stdout=captured.get("stdout", empty),
        stderr=captured.get("stderr", empty),
    )


def _atomic_immutable_json(path: Path, document: Mapping[str, Any]) -> bytes:
    raw = _canonical_bytes(document)
    if len(raw) > _MAX_OUTPUT_BYTES:
        raise PilotRunnerError("parent subprocess envelope exceeds its byte limit")
    temporary = path.with_name(f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _read_private_file(
                path,
                maximum_bytes=_MAX_OUTPUT_BYTES,
                name="parent subprocess envelope",
            ) != raw:
                raise PilotRunnerError("immutable parent output already has different bytes")
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return raw


def _atomic_mutable_json(path: Path, document: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(document)
    if len(raw) > _MAX_OUTPUT_BYTES:
        raise PilotRunnerError("parent process lease exceeds its byte limit")
    temporary = path.with_name(f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise PilotRunnerError("parent process lease is linked or non-regular")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _process_lease_document(
    *,
    run_id: str,
    attempt_number: int,
    pid: int,
    process_create_time_ns: int,
    status: str,
    returncode: int | None,
) -> Mapping[str, Any]:
    if status not in {"running", "exited"}:
        raise PilotRunnerError("process lease status is invalid")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "attempt_number": attempt_number,
        "pid": pid,
        "process_create_time_ns": process_create_time_ns,
        "status": status,
        "returncode": returncode,
    }


def _assert_no_live_worker(
    lease_path: Path,
    *,
    run_id: str,
    attempt_number: int,
) -> None:
    if not lease_path.exists():
        return
    info = lease_path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 65536:
        raise PilotRunnerError("parent process lease is invalid")
    document = _parse_canonical_json(lease_path.read_bytes(), name="process lease")
    _exact(
        document,
        {
            "schema_version",
            "run_id",
            "attempt_number",
            "pid",
            "process_create_time_ns",
            "status",
            "returncode",
        },
        "process lease",
    )
    if (
        document["schema_version"] != 1
        or document["run_id"] != run_id
        or document["attempt_number"] != attempt_number
    ):
        raise PilotRunnerError("process lease belongs to another run or attempt")
    pid = _plain_int(document["pid"], "lease pid", minimum=1)
    created = _plain_int(
        document["process_create_time_ns"], "lease process_create_time_ns", minimum=1
    )
    if document["status"] == "exited":
        if isinstance(document["returncode"], bool) or not isinstance(
            document["returncode"], int
        ):
            raise PilotRunnerError("exited process lease requires a returncode")
        return
    if document["status"] != "running" or document["returncode"] is not None:
        raise PilotRunnerError("process lease state is invalid")
    try:
        process = psutil.Process(pid)
        observed = int(round(process.create_time() * 1_000_000_000))
        alive = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return
    except (psutil.AccessDenied, OSError) as error:
        raise PilotRunnerError("cannot prove whether the recorded worker is alive") from error
    if observed == created and alive:
        raise PilotRunnerError("the exact worker process is still alive; refusing duplicate launch")


def _require_successful_exited_worker(
    lease_path: Path,
    *,
    run_id: str,
    attempt_number: int,
) -> None:
    if not lease_path.exists():
        raise PilotRunnerError("existing result has no parent-published worker lease")
    _assert_no_live_worker(
        lease_path, run_id=run_id, attempt_number=attempt_number
    )
    document = _parse_canonical_json(
        _read_private_file(lease_path, maximum_bytes=65536, name="process lease"),
        name="process lease",
    )
    if document.get("status") != "exited" or document.get("returncode") != 0:
        raise PilotRunnerError("existing result is not bound to a successful worker exit")


def _artifact_from_value(
    value: object,
    *,
    name: str,
    output_root: Path,
    repo_root: Path,
    expected_prefix: str,
) -> ArtifactReference:
    if not isinstance(value, dict):
        raise PilotRunnerError(f"{name} must be an artifact reference object")
    _exact(value, {"path", "size_bytes", "sha256"}, name)
    reference = ArtifactReference(
        path=value["path"],
        size_bytes=value["size_bytes"],
        sha256=value["sha256"],
    )
    if not reference.path.startswith(expected_prefix + "/"):
        raise PilotRunnerError(f"{name} is outside its deterministic attempt directory")
    verified = make_artifact_reference(
        reference.path,
        external_root=output_root,
        checkout_root=repo_root,
    )
    if verified != reference:
        raise PilotRunnerError(f"{name} digest or size does not match its bytes")
    return reference


def _ratio_matches(value: object, numerator: int, denominator: int, name: str) -> float:
    observed = _finite_number(value, name, minimum=0.0, maximum=1.0)
    expected = numerator / denominator if denominator else 0.0
    if observed != expected:
        raise PilotRunnerError(f"{name} disagrees with its exact counts")
    return observed


def _validate_routing_metrics(
    value: object,
    *,
    run: ResolvedCampaignRun,
    config: Milestone4CampaignConfig,
    episodes: int,
) -> str | None:
    """Validate the worker's bounded routing summary and retain canonical bytes."""

    if run.family != "routed":
        if value is not None:
            raise PilotRunnerError("baseline validation routing metrics must be null")
        return None
    if not isinstance(value, dict):
        raise PilotRunnerError("routed validation requires routing metrics")
    _exact(value, {"route_recovery", "route_consistency", "router_load"}, "routing")

    recovery = value["route_recovery"]
    if not isinstance(recovery, dict):
        raise PilotRunnerError("route_recovery must be an object")
    _exact(
        recovery,
        {"correct", "local_event_count", "accuracy", "macro_accuracy", "document_count"},
        "route_recovery",
    )
    recovered = _plain_int(recovery["correct"], "route_recovery.correct")
    recovery_count = _plain_int(
        recovery["local_event_count"], "route_recovery.local_event_count"
    )
    if recovered > recovery_count:
        raise PilotRunnerError("route_recovery.correct exceeds its denominator")
    _ratio_matches(recovery["accuracy"], recovered, recovery_count, "route_recovery.accuracy")
    _finite_number(
        recovery["macro_accuracy"],
        "route_recovery.macro_accuracy",
        minimum=0.0,
        maximum=1.0,
    )
    recovery_documents = _plain_int(
        recovery["document_count"], "route_recovery.document_count"
    )
    if recovery_documents > episodes:
        raise PilotRunnerError("route_recovery.document_count exceeds validation episodes")

    consistency = value["route_consistency"]
    if not isinstance(consistency, dict):
        raise PilotRunnerError("route_consistency must be an object")
    _exact(
        consistency,
        {
            "consistent_events",
            "local_event_count",
            "consistency",
            "group_count",
            "fully_consistent_groups",
        },
        "route_consistency",
    )
    consistent = _plain_int(
        consistency["consistent_events"], "route_consistency.consistent_events"
    )
    consistency_count = _plain_int(
        consistency["local_event_count"], "route_consistency.local_event_count"
    )
    groups = _plain_int(consistency["group_count"], "route_consistency.group_count")
    full_groups = _plain_int(
        consistency["fully_consistent_groups"],
        "route_consistency.fully_consistent_groups",
    )
    if consistent > consistency_count or full_groups > groups:
        raise PilotRunnerError("route_consistency counts are impossible")
    _ratio_matches(
        consistency["consistency"],
        consistent,
        consistency_count,
        "route_consistency.consistency",
    )

    load = value["router_load"]
    if not isinstance(load, dict):
        raise PilotRunnerError("router_load must be an object")
    load_keys = {
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
    _exact(load, load_keys, "router_load")
    raw_counts = load["branch_counts"]
    raw_fractions = load["branch_fractions"]
    branches = config.task.branches
    if (
        not isinstance(raw_counts, list)
        or not isinstance(raw_fractions, list)
        or len(raw_counts) != branches
        or len(raw_fractions) != branches
    ):
        raise PilotRunnerError("router_load branch vectors have the wrong width")
    counts = tuple(
        _plain_int(item, f"router_load.branch_counts[{index}]")
        for index, item in enumerate(raw_counts)
    )
    local_count = _plain_int(load["local_event_count"], "router_load.local_event_count")
    global_count = _plain_int(load["global_event_count"], "router_load.global_event_count")
    null_count = _plain_int(load["null_event_count"], "router_load.null_event_count")
    valid_count = _plain_int(load["valid_event_count"], "router_load.valid_event_count")
    if sum(counts) != local_count or local_count + global_count + null_count != valid_count:
        raise PilotRunnerError("router_load event counts do not partition valid events")
    fractions = tuple(
        _ratio_matches(item, count, local_count, f"router_load.branch_fractions[{index}]")
        for index, (item, count) in enumerate(zip(raw_fractions, counts, strict=True))
    )
    _ratio_matches(
        load["global_event_fraction"], global_count, valid_count, "router_load.global_event_fraction"
    )
    _ratio_matches(
        load["null_event_fraction"], null_count, valid_count, "router_load.null_event_fraction"
    )
    active = _plain_int(load["active_branches"], "router_load.active_branches")
    if active != sum(count > 0 for count in counts):
        raise PilotRunnerError("router_load.active_branches disagrees with branch counts")
    collapsed = load["collapsed"]
    if type(collapsed) is not bool or collapsed is not bool(local_count and branches > 1 and active <= 1):
        raise PilotRunnerError("router_load.collapsed disagrees with branch counts")
    documents = _plain_int(load["document_count"], "router_load.document_count")
    collapsed_documents = _plain_int(
        load["collapsed_document_count"], "router_load.collapsed_document_count"
    )
    if documents != episodes or collapsed_documents > documents:
        raise PilotRunnerError("router_load document counts are invalid")
    _ratio_matches(
        load["collapsed_document_fraction"],
        collapsed_documents,
        documents,
        "router_load.collapsed_document_fraction",
    )
    mean_active = _finite_number(
        load["mean_active_branches_per_document"],
        "router_load.mean_active_branches_per_document",
        minimum=0.0,
        maximum=float(branches),
    )
    if documents == 0 and mean_active != 0.0:
        raise PilotRunnerError("empty router_load documents require zero mean active branches")
    maximum_load = _finite_number(
        load["max_load_fraction"], "router_load.max_load_fraction", minimum=0.0, maximum=1.0
    )
    if maximum_load != max(fractions, default=0.0):
        raise PilotRunnerError("router_load.max_load_fraction disagrees with branch loads")
    normalizer = math.log(branches) if branches > 1 else 0.0
    load_entropy = _finite_number(
        load["load_entropy"], "router_load.load_entropy", minimum=0.0, maximum=normalizer
    )
    normalized_load = _finite_number(
        load["normalized_load_entropy"],
        "router_load.normalized_load_entropy",
        minimum=0.0,
        maximum=1.0,
    )
    expected_entropy = -sum(item * math.log(item) for item in fractions if item > 0)
    if not math.isclose(load_entropy, expected_entropy, rel_tol=0.0, abs_tol=1.0e-12):
        raise PilotRunnerError("router_load.load_entropy disagrees with branch fractions")
    expected_normalized = load_entropy / normalizer if normalizer else 0.0
    if not math.isclose(normalized_load, expected_normalized, rel_tol=0.0, abs_tol=1.0e-12):
        raise PilotRunnerError("router_load.normalized_load_entropy is invalid")
    assignment_entropy = _finite_number(
        load["mean_assignment_entropy"],
        "router_load.mean_assignment_entropy",
        minimum=0.0,
        maximum=normalizer,
    )
    normalized_assignment = _finite_number(
        load["normalized_mean_assignment_entropy"],
        "router_load.normalized_mean_assignment_entropy",
        minimum=0.0,
        maximum=1.0,
    )
    expected_assignment = assignment_entropy / normalizer if normalizer else 0.0
    if not math.isclose(normalized_assignment, expected_assignment, rel_tol=0.0, abs_tol=1.0e-12):
        raise PilotRunnerError("router_load.normalized_mean_assignment_entropy is invalid")
    assignment_count = _plain_int(
        load["assignment_entropy_count"], "router_load.assignment_entropy_count"
    )
    if assignment_count != local_count:
        raise PilotRunnerError("router_load.assignment_entropy_count differs from local events")
    return _canonical_bytes(value).decode("ascii")


def _validate_worker_result(
    path: Path,
    *,
    output_root: Path,
    repo_root: Path,
    attempt_prefix: str,
    run: ResolvedCampaignRun,
    attempt_number: int,
    authority: CampaignAuthority,
    provenance: PilotProvenance,
    parent: PilotRunSummary | None = None,
) -> _WorkerResult:
    expected_streams = _expected_streams(authority.config, run)
    try:
        info = path.lstat()
    except OSError as error:
        raise PilotRunnerError("worker did not create its result JSON") from error
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > _MAX_RESULT_BYTES:
        raise PilotRunnerError("worker result must be a bounded private regular file")
    raw = _read_private_file(path, maximum_bytes=_MAX_RESULT_BYTES, name="worker result")
    if len(raw) != info.st_size:
        raise PilotRunnerError("worker result size changed while it was read")
    document = _parse_canonical_json(raw, name="worker result")
    expected_top = {
        "schema_version",
        "status",
        "run_id",
        "run_sha256",
        "model_id",
        "pair_id",
        "family",
        "role",
        "attempt_number",
        "plan_sha256",
        "provenance",
        "artifacts",
        "metrics",
        "stream",
        "error",
    }
    _exact(document, expected_top, "worker result")
    if document["schema_version"] != 1:
        raise PilotRunnerError("worker result schema_version must be integer 1")
    expected_identity = {
        "run_id": run.run_id,
        "run_sha256": _run_sha256(run),
        "model_id": run.model_id,
        "pair_id": run.pair_id,
        "family": run.family,
        "role": run.role,
        "attempt_number": attempt_number,
        "plan_sha256": authority.plan_sha256,
    }
    for name, expected in expected_identity.items():
        if document[name] != expected:
            raise PilotRunnerError(f"worker result {name} does not match its run")
    provenance_value = document["provenance"]
    if not isinstance(provenance_value, dict):
        raise PilotRunnerError("worker result provenance must be an object")
    expected_provenance = {
        "code_commit": provenance.code_commit,
        "code_tree": provenance.code_tree,
        "raw_config_sha256": provenance.raw_config_sha256,
        "semantic_config_sha256": authority.config.fingerprint(),
        "executable_bundle_sha256": provenance.executable_bundle_sha256,
        "parent_runner_sha256": provenance.parent_runner_sha256,
        "worker_sha256": provenance.worker_sha256,
        "package_tree_sha256": provenance.package_tree_sha256,
    }
    if set(provenance_value) != set(expected_provenance):
        raise PilotRunnerError("worker result provenance keys are not exact")
    for name, expected in expected_provenance.items():
        if provenance_value[name] != expected:
            raise PilotRunnerError(f"worker result provenance {name} changed")
    status = document["status"]
    if status not in {"success", "failure"}:
        raise PilotRunnerError("worker result status is invalid")
    artifacts = document["artifacts"]
    if not isinstance(artifacts, dict):
        raise PilotRunnerError("worker result artifacts must be an object")
    _exact(
        artifacts,
        {"checkpoints", "final_checkpoint", "compact_artifact"},
        "worker artifacts",
    )
    checkpoint_values = artifacts["checkpoints"]
    if not isinstance(checkpoint_values, list):
        raise PilotRunnerError("worker checkpoints must be an array")
    checkpoint_items: list[tuple[int, ArtifactReference, str]] = []
    for index, value in enumerate(checkpoint_values):
        if not isinstance(value, dict):
            raise PilotRunnerError(f"checkpoint[{index}] must be an object")
        _exact(
            value,
            {"step", "path", "size_bytes", "sha256", "stream_prefix_sha256"},
            f"checkpoint[{index}]",
        )
        step = _plain_int(value["step"], f"checkpoint[{index}].step", minimum=1)
        reference = _artifact_from_value(
            {name: value[name] for name in ("path", "size_bytes", "sha256")},
            name=f"checkpoint[{index}]",
            output_root=output_root,
            repo_root=repo_root,
            expected_prefix=attempt_prefix,
        )
        prefix_hash = _sha(
            value["stream_prefix_sha256"],
            f"checkpoint[{index}].stream_prefix_sha256",
        )
        if _checkpoint_cursor(reference, attempt_prefix) != step:
            raise PilotRunnerError("checkpoint step disagrees with its filename")
        checkpoint_items.append((step, reference, prefix_hash))
    checkpoints = tuple(item[1] for item in checkpoint_items)
    if len({reference.path for reference in checkpoints}) != len(checkpoints):
        raise PilotRunnerError("worker checkpoint paths must be unique")
    checkpoint_cursors = tuple(item[0] for item in checkpoint_items)
    if checkpoint_cursors != tuple(sorted(set(checkpoint_cursors))):
        raise PilotRunnerError("worker checkpoints must be strictly cursor ordered")
    allowed_cursors = _expected_checkpoint_cursors(run)
    if any(cursor not in allowed_cursors for cursor in checkpoint_cursors):
        raise PilotRunnerError("worker checkpoint cursor is outside the locked schedule")
    final_value = artifacts["final_checkpoint"]
    final_checkpoint = (
        None
        if final_value is None
        else _artifact_from_value(
            final_value,
            name="final_checkpoint",
            output_root=output_root,
            repo_root=repo_root,
            expected_prefix=attempt_prefix,
        )
    )
    compact_value = artifacts["compact_artifact"]
    compact_artifact = (
        None
        if compact_value is None
        else _artifact_from_value(
            compact_value,
            name="compact_artifact",
            output_root=output_root,
            repo_root=repo_root,
            expected_prefix=attempt_prefix,
        )
    )
    if compact_artifact is not None:
        compact_path = PurePosixPath(compact_artifact.path)
        if compact_path.parent != PurePosixPath(attempt_prefix):
            raise PilotRunnerError("compact artifact path is not deterministic")
    error_value = document["error"]
    error_message: str | None = None
    if status == "success":
        if error_value is not None:
            raise PilotRunnerError("successful worker result cannot contain an error")
        if run.role == "trainable_source":
            if final_checkpoint is None or final_checkpoint not in checkpoints:
                raise PilotRunnerError("successful source requires a listed final checkpoint")
            if compact_artifact is not None:
                raise PilotRunnerError("source result cannot contain a compact artifact")
        elif final_checkpoint is not None or checkpoints or compact_artifact is None:
            raise PilotRunnerError(
                "derived compact result requires only its compact artifact"
            )
    else:
        if not isinstance(error_value, dict):
            raise PilotRunnerError("failed worker result requires an error object")
        _exact(error_value, {"type", "message"}, "worker error")
        if not all(isinstance(error_value[name], str) for name in ("type", "message")):
            raise PilotRunnerError("worker error fields must be strings")
        error_message = f"{error_value['type']}: {error_value['message']}"[:4096]
    metrics = document["metrics"]
    stream = document["stream"]
    if not isinstance(metrics, dict) or not isinstance(stream, dict):
        raise PilotRunnerError("worker metrics and stream must be objects")
    _exact(
        metrics,
        {"environment", "training", "validation_by_length", "compact"},
        "worker metrics",
    )
    environment = metrics["environment"]
    if not isinstance(environment, dict):
        raise PilotRunnerError("worker environment must be an object")
    _exact(
        environment,
        {"python_version", "torch_version", "numpy_version", "platform", "device"},
        "worker environment",
    )
    expected_environment = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "platform": platform.platform(),
        "device": "cpu",
    }
    if environment != expected_environment:
        raise PilotRunnerError("worker environment differs from the parent environment")
    validation_values = metrics["validation_by_length"]
    if not isinstance(validation_values, list):
        raise PilotRunnerError("validation_by_length must be an array")
    expected_lengths = tuple(run.data.validation.lengths)
    validation_hashes: list[tuple[int, str]] = []
    validation_metrics: list[ValidationLengthSummary] = []
    parameter_counts: list[int] = []
    for index, value in enumerate(validation_values):
        if not isinstance(value, dict):
            raise PilotRunnerError("validation entry must be an object")
        required = {
            "length", "batch_sha256", "episodes", "query", "seen_query",
            "heldout_query", "routing", "structural",
        }
        _exact(value, required, f"validation[{index}]")
        length = _plain_int(value["length"], "validation length", minimum=1)
        episodes = _plain_int(value["episodes"], "validation episodes", minimum=1)
        expected_counts = expected_streams.validation_counts[index] if index < len(expected_streams.validation_counts) else None
        if expected_counts is None or episodes != expected_counts[0]:
            raise PilotRunnerError("validation episode count differs from config")
        observed_counts: list[int] = []
        observed_correct: list[int] = []
        for metric_name, expected_count in zip(
            ("query", "seen_query", "heldout_query"),
            expected_counts[1:],
            strict=True,
        ):
            query_metric = value[metric_name]
            if not isinstance(query_metric, dict):
                raise PilotRunnerError(f"{metric_name} must be an object")
            _exact(
                query_metric,
                {"correct", "count", "accuracy", "cross_entropy"},
                metric_name,
            )
            count = _plain_int(query_metric["count"], f"{metric_name}.count")
            correct = _plain_int(query_metric["correct"], f"{metric_name}.correct")
            if count != expected_count or correct > count:
                raise PilotRunnerError(f"{metric_name} count/correct is invalid")
            accuracy = _finite_number(
                query_metric["accuracy"], f"{metric_name}.accuracy", minimum=0, maximum=1
            )
            if accuracy != (correct / count if count else 0.0):
                raise PilotRunnerError(f"{metric_name}.accuracy disagrees with correct/count")
            cross_entropy = query_metric["cross_entropy"]
            if count == 0:
                if cross_entropy is not None:
                    raise PilotRunnerError(f"{metric_name}.cross_entropy must be null when empty")
            else:
                _finite_number(
                    cross_entropy, f"{metric_name}.cross_entropy", minimum=0
                )
            observed_counts.append(count)
            observed_correct.append(correct)
        if observed_counts[0] != observed_counts[1] + observed_counts[2]:
            raise PilotRunnerError("validation query strata do not partition queries")
        routing_json = _validate_routing_metrics(
            value["routing"], run=run, config=authority.config, episodes=episodes
        )
        structural = value["structural"]
        if not isinstance(structural, dict):
            raise PilotRunnerError("validation structural metrics must be an object")
        _exact(structural, {"fingerprint_sha256", "values"}, "validation structural")
        values = structural["values"]
        if not isinstance(values, dict) or not values:
            raise PilotRunnerError("validation structural values must be a nonempty object")
        if any(not isinstance(key, str) or not key for key in values):
            raise PilotRunnerError("validation structural keys must be nonempty strings")
        for key, item in values.items():
            _plain_int(item, f"structural.{key}")
        if "parameter_count" not in values:
            raise PilotRunnerError("validation structural metrics omit parameter_count")
        parameter_count = _plain_int(
            values["parameter_count"], "structural.parameter_count", minimum=1
        )
        parameter_counts.append(parameter_count)
        structural_fingerprint = _sha(
            structural["fingerprint_sha256"], "structural fingerprint"
        )
        if structural_fingerprint != hashlib.sha256(_canonical_bytes(values)).hexdigest():
            raise PilotRunnerError("validation structural fingerprint is invalid")
        batch_hash = _sha(value["batch_sha256"], "validation batch_sha256")
        validation_hashes.append((length, batch_hash))
        validation_metrics.append(
            ValidationLengthSummary(
                length=length,
                query_correct=observed_correct[0],
                query_count=observed_counts[0],
                seen_correct=observed_correct[1],
                seen_count=observed_counts[1],
                heldout_correct=observed_correct[2],
                heldout_count=observed_counts[2],
                structural_values=tuple(sorted(values.items())),
                routing_json=routing_json,
            )
        )
    if tuple(length for length, _ in validation_hashes) != expected_lengths:
        raise PilotRunnerError("validation lengths do not exactly match the config")
    if tuple(validation_hashes) != expected_streams.validation_hashes:
        raise PilotRunnerError("validation batch hashes differ from trusted generation")
    if len(set(parameter_counts)) != 1:
        raise PilotRunnerError("parameter_count changed across validation lengths")
    _exact(
        stream,
        {
            "start_step", "resumed_from_step", "completed_step",
            "training_batches", "stream_prefix_sha256", "checkpoint_steps",
        },
        "worker stream",
    )
    start_step = _plain_int(stream["start_step"], "stream start_step")
    resumed_from_step = _plain_int(stream["resumed_from_step"], "stream resumed_from_step")
    completed_step = _plain_int(stream["completed_step"], "stream completed_step")
    if start_step != 0:
        raise PilotRunnerError("worker stream must logically begin at zero")
    batch_values = stream["training_batches"]
    checkpoint_step_values = stream["checkpoint_steps"]
    if not isinstance(batch_values, list) or not isinstance(checkpoint_step_values, list):
        raise PilotRunnerError("worker stream arrays are invalid")
    stream_hashes: list[str] = []
    stream_tokens: list[int] = []
    for index, item in enumerate(batch_values):
        if not isinstance(item, dict):
            raise PilotRunnerError("training batch entry must be an object")
        _exact(item, {"step", "sha256", "token_count"}, "training batch entry")
        if _plain_int(item["step"], "training batch step") != index:
            raise PilotRunnerError("training batch steps are not complete and ordered")
        stream_hashes.append(_sha(item["sha256"], "training batch sha256"))
        stream_tokens.append(_plain_int(item["token_count"], "training token_count", minimum=1))
    training_hashes = tuple(stream_hashes)
    training_tokens = tuple(stream_tokens)
    reported_checkpoint_steps = tuple(
        _plain_int(value, "stream checkpoint step", minimum=1)
        for value in checkpoint_step_values
    )
    prefix_sha = stream["stream_prefix_sha256"]
    training = metrics["training"]
    compact_metrics = metrics["compact"]
    initial_model_fingerprint: str | None = None
    final_model_fingerprint: str
    if run.role == "trainable_source":
        assert run.training is not None
        if (
            completed_step != run.training.optimizer_steps
            or len(training_hashes) != completed_step
            or len(training_tokens) != completed_step
            or resumed_from_step not in (0, *_expected_checkpoint_cursors(run))
            or resumed_from_step > completed_step
            or reported_checkpoint_steps != _expected_checkpoint_cursors(run)
        ):
            raise PilotRunnerError("source training stream is incomplete")
        prefix_sha = _sha(prefix_sha, "stream_prefix_sha256")
        if (
            training_hashes != expected_streams.training_hashes
            or training_tokens != expected_streams.training_tokens
            or prefix_sha != expected_streams.prefix_by_cursor[-1]
        ):
            raise PilotRunnerError("reported source stream differs from trusted generation")
        if not isinstance(training, dict) or compact_metrics is not None:
            raise PilotRunnerError("source training/compact metric shape is invalid")
        _exact(
            training,
            {
                "initial_model_fingerprint", "final_model_fingerprint",
                "optimizer_steps", "token_count", "steps",
            },
            "training metrics",
        )
        initial_model_fingerprint = _sha(
            training["initial_model_fingerprint"], "initial model fingerprint"
        )
        final_model_fingerprint = _sha(
            training["final_model_fingerprint"], "final model fingerprint"
        )
        if _plain_int(training["optimizer_steps"], "optimizer_steps") != completed_step:
            raise PilotRunnerError("training optimizer_steps disagrees with the plan")
        if _plain_int(training["token_count"], "training token_count") != sum(training_tokens):
            raise PilotRunnerError("training token total is inconsistent")
        fresh_model = build_campaign_source_model(
            CampaignRunContext(config=authority.config, run=run)
        )
        if campaign_model_fingerprint(fresh_model) != initial_model_fingerprint:
            raise PilotRunnerError("initial model fingerprint differs from trusted construction")
        step_values = training["steps"]
        if not isinstance(step_values, list) or len(step_values) != completed_step:
            raise PilotRunnerError("training step metrics are incomplete")
        routed = run.family == "routed"
        for index, step_value in enumerate(step_values):
            if not isinstance(step_value, dict):
                raise PilotRunnerError("training step metric must be an object")
            _exact(
                step_value,
                {"step", "batch_sha256", "token_count", "loss", "counters"},
                "training step metric",
            )
            if (
                _plain_int(step_value["step"], "training step") != index
                or _sha(step_value["batch_sha256"], "step batch sha256") != training_hashes[index]
                or _plain_int(step_value["token_count"], "step token_count", minimum=1) != training_tokens[index]
            ):
                raise PilotRunnerError("training step identity differs from trusted stream")
            loss = step_value["loss"]
            counters = step_value["counters"]
            if not isinstance(loss, dict) or not isinstance(counters, dict):
                raise PilotRunnerError("training loss/counters must be objects")
            loss_fields = {
                "total", "query", "route_curriculum", "router_balance",
                "router_entropy", "route_persistence",
            }
            _exact(loss, loss_fields, "training loss")
            _finite_number(loss["total"], "training loss total", minimum=0)
            _finite_number(loss["query"], "training query loss", minimum=0)
            for field in loss_fields - {"total", "query"}:
                if routed:
                    _finite_number(loss[field], f"training {field}", minimum=0)
                elif loss[field] is not None:
                    raise PilotRunnerError("baseline route loss fields must be null")
            _exact(
                counters,
                {"query_count", "route_supervision_count", "persistence_pair_count"},
                "training counters",
            )
            query_count = _plain_int(counters["query_count"], "training query_count")
            route_count = _plain_int(counters["route_supervision_count"], "route supervision count")
            persistence_count = _plain_int(counters["persistence_pair_count"], "persistence pair count")
            if query_count != expected_streams.training_query_counts[index]:
                raise PilotRunnerError("training query count differs from trusted batch")
            if not routed and (route_count or persistence_count):
                raise PilotRunnerError("baseline route counters must be zero")
    else:
        if (
            completed_step != 0 or resumed_from_step != 0 or training_hashes
            or training_tokens or reported_checkpoint_steps
        ):
            raise PilotRunnerError("derived compact result must not report training")
        if prefix_sha is not None:
            raise PilotRunnerError("derived compact stream prefix must be null")
        if training is not None or not isinstance(compact_metrics, dict):
            raise PilotRunnerError("derived training/compact metric shape is invalid")
        _exact(
            compact_metrics,
            {
                "parent_run_id", "parent_result", "parent_checkpoint",
                "parent_model_fingerprint", "selection_fingerprint",
                "manifest_fingerprint", "exported_model_fingerprint",
                "compact_artifact_sha256",
            },
            "compact metrics",
        )
        for field in (
            "parent_model_fingerprint", "selection_fingerprint",
            "manifest_fingerprint", "exported_model_fingerprint",
            "compact_artifact_sha256",
        ):
            _sha(compact_metrics[field], f"compact {field}")
        if parent is None or compact_artifact is None:
            raise PilotRunnerError("derived result lacks its trusted parent/artifact")
        final_model_fingerprint = _sha(
            compact_metrics["exported_model_fingerprint"], "exported model fingerprint"
        )
    if status == "success" and run.role == "trainable_source":
        if checkpoint_cursors != allowed_cursors:
            raise PilotRunnerError("successful source is missing a locked checkpoint cursor")
        if final_checkpoint != checkpoints[-1]:
            raise PilotRunnerError("final checkpoint is not the locked final cursor")
    checkpoint_fingerprints: list[str] = []
    for cursor_value, reference, reported_prefix in checkpoint_items:
        if reported_prefix != expected_streams.prefix_by_cursor[cursor_value]:
            raise PilotRunnerError("checkpoint stream prefix differs from trusted stream")
        checkpoint_fingerprints.append(_verify_checkpoint_semantics(
            reference,
            cursor=cursor_value,
            expected_streams=expected_streams,
            config=authority.config,
            run=run,
            output_root=output_root,
        ))
    if run.role == "trainable_source":
        if not checkpoint_fingerprints or checkpoint_fingerprints[-1] != final_model_fingerprint:
            raise PilotRunnerError("final model fingerprint disagrees with final checkpoint")
    else:
        assert compact_artifact is not None and isinstance(compact_metrics, dict) and parent is not None
        _verify_compact_semantics(
            compact_artifact,
            compact_metrics,
            run=run,
            parent=parent,
            authority=authority,
            output_root=output_root,
        )
    return _WorkerResult(
        raw=raw,
        document=document,
        checkpoints=checkpoints,
        final_checkpoint=final_checkpoint,
        compact_artifact=compact_artifact,
        status=status,
        error_message=error_message,
        validation_batch_hashes=tuple(validation_hashes),
        training_batch_hashes=training_hashes,
        training_token_counts=training_tokens,
        stream_prefix_sha256=prefix_sha,
        completed_step=completed_step,
        initial_model_fingerprint=initial_model_fingerprint,
        final_model_fingerprint=final_model_fingerprint,
        validation_metrics=tuple(validation_metrics),
        parameter_count=parameter_counts[0],
        compact_lineage_json=(
            _canonical_bytes(compact_metrics).decode("ascii")
            if isinstance(compact_metrics, dict)
            else None
        ),
    )


def _discover_resume_checkpoint(
    attempt_directory: Path,
    *,
    output_root: Path,
    repo_root: Path,
    config: Milestone4CampaignConfig,
    run: ResolvedCampaignRun,
) -> ArtifactReference | None:
    candidates = sorted(attempt_directory.glob("checkpoint-step-*"))
    references: list[tuple[int, ArtifactReference]] = []
    expected_prefix = attempt_directory.relative_to(output_root).as_posix()
    expected_streams = _expected_streams(config, run)
    allowed = _expected_checkpoint_cursors(run)
    for path in candidates:
        if not re.fullmatch(r"checkpoint-step-[0-9]{8}\.twcp", path.name):
            raise PilotRunnerError("attempt contains an unexpected checkpoint filename")
        relative = path.relative_to(output_root).as_posix()
        reference = make_artifact_reference(
            relative,
            external_root=output_root,
            checkout_root=repo_root,
        )
        cursor = _checkpoint_cursor(reference, expected_prefix)
        if cursor not in allowed:
            raise PilotRunnerError("attempt contains an unscheduled checkpoint cursor")
        _verify_checkpoint_semantics(
            reference,
            cursor=cursor,
            expected_streams=expected_streams,
            config=config,
            run=run,
            output_root=output_root,
        )
        references.append((cursor, reference))
    return max(references, key=lambda item: item[0])[1] if references else None


def _build_worker_argv(
    provenance: PilotProvenance,
    authority: CampaignAuthority,
    run: ResolvedCampaignRun,
    attempt_number: int,
    output_root: Path,
    result_path: Path,
    resume: ArtifactReference | None,
    parent: PilotRunSummary | None,
) -> tuple[str, ...]:
    arguments = [
        sys.executable,
        "-s",
        "-B",
        str(provenance.worker_path),
        "--repo-root",
        str(provenance.repo_root),
        "--parent-runner",
        str(provenance.runner_path),
        "--config",
        str(provenance.config_path),
        "--output-root",
        str(output_root),
        "--result-path",
        str(result_path),
        "--run-id",
        run.run_id,
        "--attempt-number",
        str(attempt_number),
        "--plan-sha256",
        authority.plan_sha256,
        "--code-commit",
        provenance.code_commit,
        "--code-tree",
        provenance.code_tree,
        "--raw-config-sha256",
        provenance.raw_config_sha256,
        "--semantic-config-sha256",
        authority.config.fingerprint(),
        "--executable-bundle-sha256",
        provenance.executable_bundle_sha256,
        "--parent-runner-sha256",
        provenance.parent_runner_sha256,
        "--worker-sha256",
        provenance.worker_sha256,
        "--package-tree-sha256",
        provenance.package_tree_sha256,
    ]
    if resume is not None:
        arguments.extend(
            [
                "--resume-checkpoint",
                str(output_root / PurePosixPath(resume.path)),
                "--resume-checkpoint-sha256",
                resume.sha256,
            ]
        )
    if parent is not None:
        if parent.final_checkpoint is None:
            raise PilotRunnerError("compact parent has no source checkpoint")
        arguments.extend(
            [
                "--parent-result",
                str(output_root / PurePosixPath(parent.result.path)),
                "--parent-result-sha256",
                parent.result.sha256,
                "--parent-checkpoint",
                str(output_root / PurePosixPath(parent.final_checkpoint.path)),
                "--parent-checkpoint-sha256",
                parent.final_checkpoint.sha256,
            ]
        )
    return tuple(arguments)


def _subprocess_envelope(
    process: _ProcessResult | None,
    *,
    run: ResolvedCampaignRun,
    attempt_number: int,
    result_reference: ArtifactReference | None,
    validation_status: str,
    failure: str | None,
) -> Mapping[str, Any]:
    if process is None:
        return {
            "schema_version": 1,
            "run_id": run.run_id,
            "attempt_number": attempt_number,
            "argv": [],
            "pid": 0,
            "process_create_time_ns": 0,
            "returncode": 0,
            "timed_out": False,
            "stdout": None,
            "stderr": None,
            "result": None if result_reference is None else {
                "path": result_reference.path,
                "size_bytes": result_reference.size_bytes,
                "sha256": result_reference.sha256,
            },
            "validation_status": validation_status,
            "failure": failure,
        }
    def capture(value: _Capture) -> dict[str, Any]:
        return {
            "text": value.text,
            "sha256": value.sha256,
            "size_bytes": value.size_bytes,
            "truncated": value.truncated,
        }
    return {
        "schema_version": 1,
        "run_id": run.run_id,
        "attempt_number": attempt_number,
        "argv": list(process.argv),
        "pid": process.pid,
        "process_create_time_ns": process.process_create_time_ns,
        "returncode": process.returncode,
        "timed_out": process.timed_out,
        "stdout": capture(process.stdout),
        "stderr": capture(process.stderr),
        "result": None if result_reference is None else {
            "path": result_reference.path,
            "size_bytes": result_reference.size_bytes,
            "sha256": result_reference.sha256,
        },
        "validation_status": validation_status,
        "failure": failure,
    }


def _load_subprocess_envelope(
    path: Path,
    *,
    run: ResolvedCampaignRun,
    attempt_number: int,
    result_reference: ArtifactReference | None | object = _UNSET,
) -> Mapping[str, Any]:
    document = _parse_canonical_json(
        _read_private_file(
            path,
            maximum_bytes=_MAX_OUTPUT_BYTES,
            name="parent subprocess envelope",
        ),
        name="parent subprocess envelope",
    )
    _exact(
        document,
        {
            "schema_version", "run_id", "attempt_number", "argv", "pid",
            "process_create_time_ns", "returncode", "timed_out", "stdout",
            "stderr", "result", "validation_status", "failure",
        },
        "parent subprocess envelope",
    )
    if (
        document["schema_version"] != 1
        or document["run_id"] != run.run_id
        or document["attempt_number"] != attempt_number
    ):
        raise PilotRunnerError("parent subprocess envelope has the wrong identity")
    if (
        not isinstance(document["argv"], list)
        or any(not isinstance(value, str) for value in document["argv"])
        or type(document["timed_out"]) is not bool
    ):
        raise PilotRunnerError("parent subprocess envelope process fields are invalid")
    pid = _plain_int(document["pid"], "subprocess pid")
    created = _plain_int(
        document["process_create_time_ns"], "subprocess creation time"
    )
    returncode = document["returncode"]
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise PilotRunnerError("parent subprocess envelope returncode is invalid")
    if (pid == 0) != (created == 0):
        raise PilotRunnerError("parent subprocess envelope PID binding is incomplete")
    for name in ("stdout", "stderr"):
        capture = document[name]
        if capture is None:
            if pid != 0:
                raise PilotRunnerError("launched subprocess envelope lacks output capture")
            continue
        if not isinstance(capture, dict):
            raise PilotRunnerError("subprocess capture must be an object")
        _exact(capture, {"text", "sha256", "size_bytes", "truncated"}, name)
        if not isinstance(capture["text"], str) or type(capture["truncated"]) is not bool:
            raise PilotRunnerError("subprocess capture fields are invalid")
        _sha(capture["sha256"], f"{name} sha256")
        _plain_int(capture["size_bytes"], f"{name} size")
    if result_reference is not _UNSET:
        expected_result = None if result_reference is None else {
            "path": result_reference.path,
            "size_bytes": result_reference.size_bytes,
            "sha256": result_reference.sha256,
        }
        if document["result"] != expected_result:
            raise PilotRunnerError("parent subprocess envelope binds another result")
    elif document["result"] is not None:
        value = document["result"]
        if not isinstance(value, dict):
            raise PilotRunnerError("subprocess result binding must be an object or null")
        _exact(value, {"path", "size_bytes", "sha256"}, "subprocess result binding")
        ArtifactReference(
            path=value["path"],
            size_bytes=value["size_bytes"],
            sha256=value["sha256"],
        )
    status = document["validation_status"]
    if status == "success":
        if document["failure"] is not None or document["result"] is None or returncode != 0:
            raise PilotRunnerError("successful subprocess envelope is inconsistent")
    elif status == "failure":
        failure = document["failure"]
        if not isinstance(failure, str) or not failure or len(failure) > 4096:
            raise PilotRunnerError("failed subprocess envelope lacks a bounded failure")
    else:
        raise PilotRunnerError("subprocess envelope validation status is invalid")
    return document


def _completed_summary(
    run: ResolvedCampaignRun,
    state: CampaignRunState,
    *,
    authority: CampaignAuthority,
    provenance: PilotProvenance,
    output_root: Path,
    repo_root: Path,
    parent: PilotRunSummary | None = None,
) -> PilotRunSummary:
    latest = state.attempts[-1]
    attempt = latest.attempt_number
    prefix = _attempt_relative(run.run_id, attempt)
    result_path = output_root / PurePosixPath(prefix) / "result.json"
    validated = _validate_worker_result(
        result_path,
        output_root=output_root,
        repo_root=repo_root,
        attempt_prefix=prefix,
        run=run,
        attempt_number=attempt,
        authority=authority,
        provenance=provenance,
        parent=parent,
    )
    if validated.status != "success":
        raise PilotRunnerError("completed manifest run has no successful worker result")
    result_reference = make_artifact_reference(
        f"{prefix}/result.json", external_root=output_root, checkout_root=repo_root
    )
    output_reference = make_artifact_reference(
        f"{prefix}/subprocess.json", external_root=output_root, checkout_root=repo_root
    )
    completed_record = load_campaign_attempt_record(
        authority,
        latest.record,
        external_root=output_root,
        checkout_root=repo_root,
    )
    if (
        completed_record.status != "completed"
        or completed_record.result != result_reference
        or completed_record.output != output_reference
        or completed_record.checkpoint != validated.final_checkpoint
    ):
        raise PilotRunnerError("completed manifest record does not bind expected artifacts")
    return PilotRunSummary(
        run_id=run.run_id,
        model_id=run.model_id,
        pair_id=run.pair_id,
        role=run.role,
        attempt_number=attempt,
        resumed=attempt > 1,
        result=result_reference,
        output=output_reference,
        final_checkpoint=validated.final_checkpoint,
        compact_artifact=validated.compact_artifact,
        validation_batch_hashes=validated.validation_batch_hashes,
        training_batch_hashes=validated.training_batch_hashes,
        training_token_counts=validated.training_token_counts,
        stream_prefix_sha256=validated.stream_prefix_sha256,
        initial_model_fingerprint=validated.initial_model_fingerprint,
        final_model_fingerprint=validated.final_model_fingerprint,
        validation_metrics=validated.validation_metrics,
        parameter_count=validated.parameter_count,
        checkpoints=validated.checkpoints,
        compact_lineage_json=validated.compact_lineage_json,
    )


def _execute_one(
    run: ResolvedCampaignRun,
    manifest: CampaignManifest,
    *,
    authority: CampaignAuthority,
    provenance: PilotProvenance,
    output_root: Path,
    parent: PilotRunSummary | None,
    worker_timeout_seconds: float,
) -> tuple[CampaignManifest, PilotRunSummary]:
    options = {"external_root": output_root, "checkout_root": provenance.repo_root}
    state = _manifest_state(manifest, run.run_id)
    if state.attempts and state.attempts[-1].status == "completed":
        return manifest, _completed_summary(
            run,
            state,
            authority=authority,
            provenance=provenance,
            output_root=output_root,
            repo_root=provenance.repo_root,
            parent=parent,
        )
    resumed_attempt = bool(
        state.attempts and state.attempts[-1].status == "in_progress"
    )
    if not resumed_attempt:
        manifest, _ = start_campaign_attempt(
            authority,
            run.run_id,
            timestamp_ns=time.time_ns(),
            expected_generation=manifest.generation,
            **options,
        )
        state = _manifest_state(manifest, run.run_id)
    attempt_number = state.attempts[-1].attempt_number
    prefix = _attempt_relative(run.run_id, attempt_number)
    directory = _ensure_private_directory(output_root, prefix)
    result_path = directory / "result.json"
    subprocess_path = directory / "subprocess.json"
    lease_path = directory / "process.json"
    process: _ProcessResult | None = None
    validated: _WorkerResult | None = None
    failure: str | None = None
    result_reference: ArtifactReference | None = None
    resume: ArtifactReference | None = None
    durable_checkpoint: ArtifactReference | None = None
    existing_envelope: Mapping[str, Any] | None = None
    trusted = _expected_streams(authority.config, run)
    try:
        current_record = load_campaign_attempt_record(
            authority, state.attempts[-1].record, **options
        )
        if current_record.status != "in_progress":
            raise PilotRunnerError("selected attempt is not in progress")
        durable_checkpoint = current_record.checkpoint
        candidates: list[tuple[int, ArtifactReference]] = []
        if run.role == "trainable_source":
            if durable_checkpoint is not None:
                durable_cursor = _checkpoint_cursor(durable_checkpoint, prefix)
                _verify_checkpoint_semantics(
                    durable_checkpoint,
                    cursor=durable_cursor,
                    expected_streams=trusted,
                    config=authority.config,
                    run=run,
                    output_root=output_root,
                )
                candidates.append((durable_cursor, durable_checkpoint))
            if not resumed_attempt and len(state.attempts) > 1:
                # A failed attempt is allowed to emit no new checkpoint. Walk
                # the immutable history backwards to the newest prior attempt
                # that actually owns one; never rewrite that cross-attempt
                # reference into the current failed record.
                for previous_entry in reversed(state.attempts[:-1]):
                    previous_record = load_campaign_attempt_record(
                        authority, previous_entry.record, **options
                    )
                    if previous_record.status != "failed":
                        raise PilotRunnerError(
                            "new attempt does not follow only failed attempts"
                        )
                    if previous_record.checkpoint is None:
                        continue
                    previous_prefix = _attempt_relative(
                        run.run_id, previous_record.attempt_number
                    )
                    previous_cursor = _checkpoint_cursor(
                        previous_record.checkpoint, previous_prefix
                    )
                    _verify_checkpoint_semantics(
                        previous_record.checkpoint,
                        cursor=previous_cursor,
                        expected_streams=trusted,
                        config=authority.config,
                        run=run,
                        output_root=output_root,
                    )
                    candidates.append((previous_cursor, previous_record.checkpoint))
                    break
            discovered = _discover_resume_checkpoint(
                directory,
                output_root=output_root,
                repo_root=provenance.repo_root,
                config=authority.config,
                run=run,
            )
            if discovered is not None:
                candidates.append((_checkpoint_cursor(discovered, prefix), discovered))
            if candidates:
                resume = max(candidates, key=lambda item: item[0])[1]

        if subprocess_path.exists():
            existing_envelope = _load_subprocess_envelope(
                subprocess_path,
                run=run,
                attempt_number=attempt_number,
            )
            if existing_envelope["validation_status"] == "failure":
                failure = str(existing_envelope["failure"])

        if failure is None and result_path.exists():
            _require_successful_exited_worker(
                lease_path,
                run_id=run.run_id,
                attempt_number=attempt_number,
            )
            validated = _validate_worker_result(
                result_path,
                output_root=output_root,
                repo_root=provenance.repo_root,
                attempt_prefix=prefix,
                run=run,
                attempt_number=attempt_number,
                authority=authority,
                provenance=provenance,
                parent=parent,
            )
            result_reference = make_artifact_reference(
                f"{prefix}/result.json", **options
            )
            if existing_envelope is not None:
                _load_subprocess_envelope(
                    subprocess_path,
                    run=run,
                    attempt_number=attempt_number,
                    result_reference=result_reference,
                )
        elif failure is None:
            if existing_envelope is not None:
                raise PilotRunnerError("successful subprocess envelope has no result")
            if resumed_attempt:
                _assert_no_live_worker(
                    lease_path,
                    run_id=run.run_id,
                    attempt_number=attempt_number,
                )
            argv = _build_worker_argv(
                provenance,
                authority,
                run,
                attempt_number,
                output_root,
                result_path,
                resume,
                parent,
            )
            process = _run_worker_process(
                argv,
                cwd=provenance.repo_root,
                pythonpath=provenance.repo_root / "v3" / "src",
                lease_path=lease_path,
                run_id=run.run_id,
                attempt_number=attempt_number,
                timeout_seconds=worker_timeout_seconds,
            )
            if process.timed_out or process.returncode != 0:
                raise PilotRunnerError(
                    "worker timed out"
                    if process.timed_out
                    else f"worker exited {process.returncode}"
                )
            validated = _validate_worker_result(
                result_path,
                output_root=output_root,
                repo_root=provenance.repo_root,
                attempt_prefix=prefix,
                run=run,
                attempt_number=attempt_number,
                authority=authority,
                provenance=provenance,
                parent=parent,
            )
            result_reference = make_artifact_reference(
                f"{prefix}/result.json", **options
            )
        if failure is None and (
            validated is None or validated.status != "success"
        ):
            raise PilotRunnerError(
                "worker reported failure"
                if validated is None
                else validated.error_message or "worker reported failure"
            )
    except Exception as error:
        if failure is None:
            failure = f"{type(error).__name__}: {error}"[:4096]

    if failure is not None:
        # A terminal record may reference only a checkpoint physically owned
        # by this attempt. ``resume`` may belong to an older attempt and is
        # deliberately excluded so repeated failures remain traversable.
        latest_checkpoint = durable_checkpoint
        if run.role == "trainable_source":
            try:
                discovered = _discover_resume_checkpoint(
                    directory,
                    output_root=output_root,
                    repo_root=provenance.repo_root,
                    config=authority.config,
                    run=run,
                )
                if discovered is not None:
                    discovered_cursor = _checkpoint_cursor(discovered, prefix)
                    latest_cursor = -1
                    if latest_checkpoint is not None:
                        latest_cursor = _checkpoint_cursor(
                            latest_checkpoint,
                            str(PurePosixPath(latest_checkpoint.path).parent),
                        )
                    if discovered_cursor > latest_cursor:
                        latest_checkpoint = discovered
            except Exception as checkpoint_error:
                failure = (
                    failure
                    + f"; invalid emitted checkpoint: {type(checkpoint_error).__name__}: "
                    + str(checkpoint_error)
                )[:4096]
        if existing_envelope is None:
            _atomic_immutable_json(
                subprocess_path,
                _subprocess_envelope(
                    process,
                    run=run,
                    attempt_number=attempt_number,
                    result_reference=result_reference,
                    validation_status="failure",
                    failure=failure,
                ),
            )
        else:
            _load_subprocess_envelope(
                subprocess_path,
                run=run,
                attempt_number=attempt_number,
            )
        manifest, _ = fail_campaign_attempt(
            authority,
            run.run_id,
            attempt_number,
            timestamp_ns=time.time_ns(),
            expected_generation=manifest.generation,
            failure=failure,
            checkpoint=latest_checkpoint,
            **options,
        )
        raise PilotRunnerError(f"run {run.model_id} failed: {failure}")

    assert validated is not None and result_reference is not None
    if existing_envelope is None:
        _atomic_immutable_json(
            subprocess_path,
            _subprocess_envelope(
                process,
                run=run,
                attempt_number=attempt_number,
                result_reference=result_reference,
                validation_status="success",
                failure=None,
            ),
        )
    else:
        _load_subprocess_envelope(
            subprocess_path,
            run=run,
            attempt_number=attempt_number,
            result_reference=result_reference,
        )
    output_reference = make_artifact_reference(f"{prefix}/subprocess.json", **options)
    durable_cursor = -1
    if durable_checkpoint is not None:
        durable_cursor = _checkpoint_cursor(durable_checkpoint, prefix)
    for checkpoint in validated.checkpoints:
        cursor = _checkpoint_cursor(checkpoint, prefix)
        if cursor < durable_cursor:
            continue
        if cursor == durable_cursor:
            if checkpoint != durable_checkpoint:
                raise PilotRunnerError("durable checkpoint cursor has conflicting bytes")
            continue
        manifest, _ = heartbeat_campaign_attempt(
            authority,
            run.run_id,
            attempt_number,
            timestamp_ns=time.time_ns(),
            expected_generation=manifest.generation,
            checkpoint=checkpoint,
            **options,
        )
        durable_checkpoint = checkpoint
        durable_cursor = cursor
    manifest, _ = complete_campaign_attempt(
        authority,
        run.run_id,
        attempt_number,
        timestamp_ns=time.time_ns(),
        expected_generation=manifest.generation,
        output=output_reference,
        result=result_reference,
        checkpoint=validated.final_checkpoint,
        **options,
    )
    return manifest, PilotRunSummary(
        run_id=run.run_id,
        model_id=run.model_id,
        pair_id=run.pair_id,
        role=run.role,
        attempt_number=attempt_number,
        resumed=resumed_attempt or resume is not None,
        result=result_reference,
        output=output_reference,
        final_checkpoint=validated.final_checkpoint,
        compact_artifact=validated.compact_artifact,
        validation_batch_hashes=validated.validation_batch_hashes,
        training_batch_hashes=validated.training_batch_hashes,
        training_token_counts=validated.training_token_counts,
        stream_prefix_sha256=validated.stream_prefix_sha256,
        initial_model_fingerprint=validated.initial_model_fingerprint,
        final_model_fingerprint=validated.final_model_fingerprint,
        validation_metrics=validated.validation_metrics,
        parameter_count=validated.parameter_count,
        checkpoints=validated.checkpoints,
        compact_lineage_json=validated.compact_lineage_json,
    )


def _fraction_json(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _descriptive_statistics(values: Sequence[Fraction]) -> dict[str, object]:
    if not values:
        raise PilotRunnerError("cannot aggregate an empty paired metric")
    ordered = tuple(values)
    mean = sum(ordered, Fraction(0, 1)) / len(ordered)
    sorted_values = sorted(ordered)
    middle = len(sorted_values) // 2
    median = (
        sorted_values[middle]
        if len(sorted_values) % 2
        else (sorted_values[middle - 1] + sorted_values[middle]) / 2
    )
    if len(ordered) == 1:
        sample_sd = 0.0
    else:
        variance = sum((item - mean) ** 2 for item in ordered) / (len(ordered) - 1)
        sample_sd = math.sqrt(float(variance))
    return {
        "pair_values": [_fraction_json(item) for item in ordered],
        "mean": _fraction_json(mean),
        "sample_sd": sample_sd,
        "standard_error": sample_sd / math.sqrt(len(ordered)),
        "minimum": _fraction_json(sorted_values[0]),
        "maximum": _fraction_json(sorted_values[-1]),
        "median": _fraction_json(median),
    }


def _paired_delta_statistics(values: Sequence[Fraction]) -> dict[str, object]:
    if len(values) != 3:
        raise PilotRunnerError("SCREEN paired statistics require exactly three deltas")
    result = _descriptive_statistics(values)
    empirical = tuple(
        sum((values[index] for index in indices), Fraction(0, 1)) / 3
        for indices in product(range(3), repeat=3)
    )
    if len(empirical) != 27:
        raise PilotRunnerError("SCREEN exact bootstrap did not produce 27 resamples")
    percentile_rank = 2
    result.update(
        {
            "raw_delta_vector": [_fraction_json(item) for item in values],
            "ordered_resample_indices": [list(indices) for indices in product(range(3), repeat=3)],
            "ordered_empirical_resample_means": [
                _fraction_json(item) for item in empirical
            ],
            "fifth_percentile_nearest_rank": percentile_rank,
            "fifth_percentile": _fraction_json(sorted(empirical)[percentile_rank - 1]),
            "sign_test_minimum_p": [1, 8],
        }
    )
    return result


def _metric_fraction(metric: ValidationLengthSummary, partition: str) -> Fraction:
    if partition == "query":
        correct, count = metric.query_correct, metric.query_count
    elif partition == "seen_query":
        correct, count = metric.seen_correct, metric.seen_count
    elif partition == "heldout_query":
        correct, count = metric.heldout_correct, metric.heldout_count
    else:  # pragma: no cover - internal programming error
        raise PilotRunnerError(f"unsupported validation partition {partition!r}")
    return Fraction(correct, count) if count else Fraction(0, 1)


def _macro_over_lengths(summary: PilotRunSummary, partition: str) -> Fraction:
    if not summary.validation_metrics:
        raise PilotRunnerError("run summary has no trusted validation metrics")
    return sum(
        (_metric_fraction(metric, partition) for metric in summary.validation_metrics),
        Fraction(0, 1),
    ) / len(summary.validation_metrics)


def _artifact_json(reference: ArtifactReference | None) -> dict[str, object] | None:
    if reference is None:
        return None
    return {
        "path": reference.path,
        "sha256": reference.sha256,
        "size_bytes": reference.size_bytes,
    }


def _validation_metric_json(metric: ValidationLengthSummary) -> dict[str, object]:
    routing = None if metric.routing_json is None else json.loads(metric.routing_json)
    return {
        "length": metric.length,
        "query": {
            "correct": metric.query_correct,
            "count": metric.query_count,
            "accuracy": _fraction_json(_metric_fraction(metric, "query")),
        },
        "seen_query": {
            "correct": metric.seen_correct,
            "count": metric.seen_count,
            "accuracy": _fraction_json(_metric_fraction(metric, "seen_query")),
        },
        "heldout_query": {
            "correct": metric.heldout_correct,
            "count": metric.heldout_count,
            "accuracy": _fraction_json(_metric_fraction(metric, "heldout_query")),
        },
        "structural": dict(metric.structural_values),
        "routing": routing,
    }


def _screen_model_aggregates(
    config: Milestone4CampaignConfig,
    summaries: Mapping[str, PilotRunSummary],
    plan: Sequence[ResolvedCampaignRun],
) -> tuple[dict[str, dict[str, object]], dict[str, tuple[Fraction, ...]]]:
    pair_ids = tuple(pair.pair_id for pair in config.pairs)
    by_model_pair = {
        (run.model_id, run.pair_id): summaries[run.run_id]
        for run in plan
    }
    reference_id = config.quality.primary_reference_model_id
    aggregate_documents: dict[str, dict[str, object]] = {}
    query_vectors: dict[str, tuple[Fraction, ...]] = {}
    for model in config.models:
        model_summaries = tuple(by_model_pair[(model.model_id, pair_id)] for pair_id in pair_ids)
        parameter_counts = {summary.parameter_count for summary in model_summaries}
        if None in parameter_counts or len(parameter_counts) != 1:
            raise PilotRunnerError(
                f"parameter_count is not invariant across pairs for {model.model_id}"
            )
        partitions: dict[str, object] = {}
        for partition in ("query", "seen_query", "heldout_query"):
            values = tuple(_macro_over_lengths(summary, partition) for summary in model_summaries)
            reference_values = tuple(
                _macro_over_lengths(by_model_pair[(reference_id, pair_id)], partition)
                for pair_id in pair_ids
            )
            deltas = tuple(
                value - reference
                for value, reference in zip(values, reference_values, strict=True)
            )
            partitions[partition] = {
                "macro_over_length_by_pair": [
                    {"pair_id": pair_id, "value": _fraction_json(value)}
                    for pair_id, value in zip(pair_ids, values, strict=True)
                ],
                "mean_over_pairs": _fraction_json(
                    sum(values, Fraction(0, 1)) / len(values)
                ),
                "descriptive_statistics": _descriptive_statistics(values),
                "delta_vs_routed_source": _paired_delta_statistics(deltas),
            }
            if partition == "query":
                query_vectors[model.model_id] = values
        aggregate_documents[model.model_id] = {
            "parameter_count": next(iter(parameter_counts)),
            "partitions": partitions,
        }
    return aggregate_documents, query_vectors


def _mean_fraction(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise PilotRunnerError("cannot take the mean of an empty sequence")
    return sum(values, Fraction(0, 1)) / len(values)


def _screen_promotion_document(
    *,
    config: Milestone4CampaignConfig,
    plan: Sequence[ResolvedCampaignRun],
    authority: CampaignAuthority,
    provenance: PilotProvenance,
    manifest: CampaignManifest,
    summaries: Mapping[str, PilotRunSummary],
) -> tuple[dict[str, object], str]:
    if config.stage is not CampaignStage.SCREEN or config.selection is None:
        raise PilotRunnerError("promotion aggregation requires a SCREEN configuration")
    gates = config.screen_gates
    if gates is None:
        raise PilotRunnerError("SCREEN configuration has no validated screen gates")
    aggregate, query_vectors = _screen_model_aggregates(config, summaries, plan)
    pair_ids = tuple(pair.pair_id for pair in config.pairs)
    reference_id = config.quality.primary_reference_model_id
    oracle_models = tuple(
        model.model_id
        for model in config.models
        if model.family == "routed"
        and model.role == "trainable_source"
        and model.routing_mode == "oracle"
    )
    if len(oracle_models) != 1:
        raise PilotRunnerError("SCREEN requires exactly one oracle diagnostic")
    oracle_id = oracle_models[0]
    reference_query = query_vectors[reference_id]
    oracle_query = query_vectors[oracle_id]

    by_model_pair = {
        (run.model_id, run.pair_id): summaries[run.run_id]
        for run in plan
    }
    reference_heldout = tuple(
        _macro_over_lengths(by_model_pair[(reference_id, pair_id)], "heldout_query")
        for pair_id in pair_ids
    )
    reference_longest = tuple(
        _metric_fraction(by_model_pair[(reference_id, pair_id)].validation_metrics[-1], "query")
        for pair_id in pair_ids
    )
    oracle_recovery_values: list[float] = []
    for pair_id in pair_ids:
        run_summary = by_model_pair[(oracle_id, pair_id)]
        per_length: list[float] = []
        for metric in run_summary.validation_metrics:
            if metric.routing_json is None:
                raise PilotRunnerError("oracle validation lacks routing diagnostics")
            routing = json.loads(metric.routing_json)
            per_length.append(float(routing["route_recovery"]["macro_accuracy"]))
        oracle_recovery_values.append(sum(per_length) / len(per_length))
    oracle_recovery = sum(oracle_recovery_values) / len(oracle_recovery_values)
    corrected_recovery = (oracle_recovery - gates.chance) / (1.0 - gates.chance)

    positive_partitions = all(
        metric.query_count > 0 and metric.seen_count > 0 and metric.heldout_count > 0
        for summary in summaries.values()
        for metric in summary.validation_metrics
    )
    def route_evidence_passes(model_id: str) -> bool:
        for pair_id in pair_ids:
            summary = by_model_pair[(model_id, pair_id)]
            for metric in summary.validation_metrics:
                if metric.routing_json is None:
                    return False
                load = json.loads(metric.routing_json)["router_load"]
                if (
                    load["collapsed"]
                    or load["collapsed_document_count"]
                    or load["local_event_count"] == 0
                    or load["active_branches"] <= 1
                ):
                    return False
        return True

    route_evidence_by_model = {
        model.model_id: route_evidence_passes(model.model_id)
        for model in config.models
        if model.family == "routed"
    }
    route_collapse_free = (
        route_evidence_by_model[reference_id]
        and route_evidence_by_model[oracle_id]
    )

    reference_mean = _mean_fraction(reference_query)
    oracle_mean = _mean_fraction(oracle_query)
    gate_results: list[dict[str, object]] = []

    def gate_result(name: str, passed: bool, observed: object, rule: str) -> None:
        gate_results.append(
            {"gate": name, "passed": bool(passed), "observed": observed, "rule": rule}
        )

    gate_result(
        "oracle_mean_min",
        float(oracle_mean) >= gates.oracle_mean_min,
        _fraction_json(oracle_mean),
        f">={gates.oracle_mean_min}",
    )
    gate_result(
        "reference_mean_min",
        float(reference_mean) >= gates.reference_mean_min,
        _fraction_json(reference_mean),
        f">={gates.reference_mean_min}",
    )
    gate_result(
        "reference_pair_min",
        min(map(float, reference_query)) >= gates.reference_pair_min,
        [_fraction_json(item) for item in reference_query],
        f"each_pair>={gates.reference_pair_min}",
    )
    gate_result(
        "reference_heldout_mean_min",
        float(_mean_fraction(reference_heldout)) >= gates.reference_heldout_mean_min,
        _fraction_json(_mean_fraction(reference_heldout)),
        f">={gates.reference_heldout_mean_min}",
    )
    gate_result(
        "reference_longest_length_mean_min",
        float(_mean_fraction(reference_longest)) >= gates.reference_longest_length_mean_min,
        _fraction_json(_mean_fraction(reference_longest)),
        f">={gates.reference_longest_length_mean_min}",
    )
    gate_result(
        "chance_corrected_oracle_recovery_min",
        corrected_recovery >= gates.chance_corrected_oracle_recovery_min,
        corrected_recovery,
        f">={gates.chance_corrected_oracle_recovery_min};chance={gates.chance}",
    )
    oracle_drop = reference_mean - oracle_mean
    gate_result(
        "oracle_max_drop_vs_reference",
        float(oracle_drop) <= gates.oracle_max_drop_vs_reference,
        _fraction_json(oracle_drop),
        f"<={gates.oracle_max_drop_vs_reference}",
    )
    gate_result(
        "require_positive_query_partitions",
        positive_partitions,
        positive_partitions,
        "all_query_seen_heldout_counts_positive",
    )
    gate_result(
        "require_no_route_collapse",
        route_collapse_free,
        route_collapse_free,
        "oracle_and_reference_have_local_multibranch_routes_without_collapsed_documents",
    )

    model_by_id = {model.model_id: model for model in config.models}
    selected_standard: list[dict[str, object]] = []
    standard_candidates: list[dict[str, object]] = []
    compact_qualifiers: list[dict[str, object]] = []
    for stratum, candidate_ids in config.selection.candidates_by_stratum:
        if stratum == "routed_compact_rank":
            for model_id in candidate_ids:
                deltas = tuple(
                    candidate - reference
                    for candidate, reference in zip(
                        query_vectors[model_id], reference_query, strict=True
                    )
                )
                mean_delta = _mean_fraction(deltas)
                mean_pass = float(mean_delta) >= -config.quality.max_absolute_drop
                pair_pass = min(map(float, deltas)) >= -gates.candidate_pair_max_drop
                route_pass = route_evidence_by_model[model_id]
                model = model_by_id[model_id]
                export = model.export_values
                if export is None:
                    raise PilotRunnerError("compact selection candidate has no export contract")
                compact_qualifiers.append(
                    {
                        "model_id": model_id,
                        "qualified": mean_pass and pair_pass and route_pass,
                        "mean_margin_passed": mean_pass,
                        "pair_margin_passed": pair_pass,
                        "route_evidence_passed": route_pass,
                        "mean_delta": _fraction_json(mean_delta),
                        "raw_delta_vector": [_fraction_json(item) for item in deltas],
                        "parameter_count": aggregate[model_id]["parameter_count"],
                        "target_cp_rank": export["target_cp_rank"],
                        "score": _fraction_json(_mean_fraction(query_vectors[model_id])),
                    }
                )
            continue
        eligible: list[str] = []
        for model_id in candidate_ids:
            route_pass = (
                route_evidence_by_model[model_id]
                if stratum == "routed_latent"
                else True
            )
            standard_candidates.append(
                {
                    "stratum": stratum,
                    "model_id": model_id,
                    "qualified": True,
                    "route_evidence_passed": route_pass,
                    "score": _fraction_json(_mean_fraction(query_vectors[model_id])),
                    "parameter_count": aggregate[model_id]["parameter_count"],
                }
            )
            eligible.append(model_id)
        winner = min(
            eligible,
            key=lambda model_id: (
                -float(_mean_fraction(query_vectors[model_id])),
                int(aggregate[model_id]["parameter_count"]),
                model_id,
            ),
        )
        selected_standard.append(
            {
                "stratum": stratum,
                "model_id": winner,
                "score": _fraction_json(_mean_fraction(query_vectors[winner])),
                "parameter_count": aggregate[winner]["parameter_count"],
            }
        )
    compact_qualifiers.sort(
        key=lambda item: (
            not bool(item["qualified"]),
            int(item["parameter_count"]),
            int(item["target_cp_rank"]),
            -float(Fraction(*item["score"])),  # type: ignore[arg-type]
            str(item["model_id"]),
        )
    )
    qualified_compacts = [item for item in compact_qualifiers if item["qualified"]]
    compact_winner = qualified_compacts[0] if qualified_compacts else None
    gate_result(
        "compact_mean_and_pair_margins",
        bool(qualified_compacts),
        compact_qualifiers,
        (
            f"mean_delta>=-{config.quality.max_absolute_drop};"
            f"each_pair_delta>=-{gates.candidate_pair_max_drop};"
            "route_evidence_passed;at_least_one"
        ),
    )
    expected_standard_strata = sum(
        stratum != "routed_compact_rank"
        for stratum, _ in config.selection.candidates_by_stratum
    )
    standard_selection_complete = len(selected_standard) == expected_standard_strata
    all_gates_passed = all(bool(item["passed"]) for item in gate_results)
    decision = "complete_promote" if all_gates_passed else "complete_do_not_promote"
    selected_ids = [reference_id]
    selected_ids.extend(item["model_id"] for item in selected_standard)
    if compact_winner is not None:
        selected_ids.append(compact_winner["model_id"])

    run_documents: list[dict[str, object]] = []
    for run in sorted(plan, key=lambda item: (item.model_id, item.pair_id, item.run_id)):
        summary = summaries[run.run_id]
        run_documents.append(
            {
                "run_id": run.run_id,
                "run_sha256": _run_sha256(run),
                "model_id": run.model_id,
                "pair_id": run.pair_id,
                "family": run.family,
                "role": run.role,
                "routing_mode": run.routing_mode,
                "parent_model_id": run.parent_model_id,
                "parent_run_id": run.parent_run_id,
                "attempt_number": summary.attempt_number,
                "parameter_count": summary.parameter_count,
                "validation_batch_hashes": [
                    {"length": length, "sha256": digest}
                    for length, digest in summary.validation_batch_hashes
                ],
                "validation_by_length": [
                    _validation_metric_json(metric) for metric in summary.validation_metrics
                ],
                "artifacts": {
                    "result": _artifact_json(summary.result),
                    "subprocess": _artifact_json(summary.output),
                    "checkpoints": [
                        _artifact_json(reference) for reference in summary.checkpoints
                    ],
                    "final_checkpoint": _artifact_json(summary.final_checkpoint),
                    "compact_artifact": _artifact_json(summary.compact_artifact),
                },
                "fingerprints": {
                    "initial_model": summary.initial_model_fingerprint,
                    "final_model": summary.final_model_fingerprint,
                    "training_stream_prefix": summary.stream_prefix_sha256,
                },
                "compact_lineage": (
                    None
                    if summary.compact_lineage_json is None
                    else json.loads(summary.compact_lineage_json)
                ),
            }
        )

    document = {
        "schema_version": 1,
        "record_type": "milestone4_screen_promotion_v1",
        "campaign_id": config.campaign_id,
        "stage": "screen",
        "decision": decision,
        "claim_eligible": False,
        "scope": {
            "test_data_used": False,
            "scaling_data_used": False,
            "statement": (
                "Non-claiming SCREEN selection evidence only; no test or scaling "
                "evaluation was exposed or used."
            ),
        },
        "authority": {
            "plan_sha256": authority.plan_sha256,
            "manifest_path": f"campaigns/{config.campaign_id}/manifest.json",
            "manifest_generation": manifest.generation,
            "manifest_sha256": campaign_manifest_sha256(manifest),
            "raw_config_sha256": provenance.raw_config_sha256,
            "semantic_config_sha256": config.fingerprint(),
            "code_commit": provenance.code_commit,
            "code_tree": provenance.code_tree,
            "parent_runner_sha256": provenance.parent_runner_sha256,
            "worker_sha256": provenance.worker_sha256,
            "package_tree_sha256": provenance.package_tree_sha256,
            "executable_bundle_sha256": provenance.executable_bundle_sha256,
        },
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "model_seed": pair.model_seed,
                "train_seed": pair.train_seed,
                "validation_seed": pair.validation_seed,
                "statistics_seed": pair.statistics_seed,
            }
            for pair in config.pairs
        ],
        "statistics": {
            "paired_unit": config.statistics.paired_unit,
            "method": config.statistics.method,
            "resamples": config.statistics.resamples,
            "confidence_level": config.statistics.confidence_level,
            "macro_order": "macro_over_validation_lengths_then_mean_over_three_pairs",
            "fifth_percentile_nearest_rank": 2,
            "sign_test_minimum_p": [1, 8],
        },
        "gates": {
            "specification": {
                name: getattr(gates, name)
                for name in gates.__dataclass_fields__
            },
            "results": gate_results,
            "all_passed": all_gates_passed,
        },
        "selection": {
            "primary_metric": config.selection.primary_metric,
            "reference": {"model_id": reference_id, "retained": True},
            "oracle_diagnostic": {"model_id": oracle_id, "promoted": False},
            "standard_tie_break": list(config.selection.standard_tie_break),
            "compact_tie_break": list(config.selection.compact_tie_break),
            "standard_winners": selected_standard,
            "standard_candidates": standard_candidates,
            "standard_stratum_selection_complete": {
                "passed": standard_selection_complete,
                "selected": len(selected_standard),
                "required": expected_standard_strata,
                "diagnostic_only": True,
            },
            "compact_qualifiers": compact_qualifiers,
            "compact_winner": compact_winner,
            "selected_model_ids": selected_ids,
            "promoted_model_ids": selected_ids if decision == "complete_promote" else [],
        },
        "model_aggregates": aggregate,
        "runs": run_documents,
    }
    return document, decision


def _write_screen_promotion(
    *,
    output_root: Path,
    repo_root: Path,
    config: Milestone4CampaignConfig,
    document: Mapping[str, Any],
) -> ArtifactReference:
    relative = f"artifacts/screen/{config.campaign_id}/promotion.json"
    directory = _ensure_private_directory(output_root, str(PurePosixPath(relative).parent))
    path = directory / "promotion.json"
    _atomic_immutable_json(path, document)
    return make_artifact_reference(
        relative, external_root=output_root, checkout_root=repo_root
    )


def run_milestone4_pilot(
    repo_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    worker_path: str | Path,
    *,
    expected_commit: str | None = None,
    worker_timeout_seconds: float = _WORKER_TIMEOUT_SECONDS,
) -> PilotCampaignSummary:
    """Execute or resume a non-claiming Milestone-4 PILOT or SCREEN campaign."""

    provenance = collect_pilot_provenance(
        repo_root,
        config_path,
        worker_path,
        expected_commit=expected_commit,
    )
    _verify_import_origins(provenance)
    output = _validated_output_root(output_root, provenance.repo_root)
    config, plan, authority = _load_authority(provenance)
    manifest = _load_or_initialize_manifest(authority, output, provenance.repo_root)
    sources = sorted(
        (run for run in plan if run.role == "trainable_source"),
        key=lambda run: (run.model_id, run.pair_id, run.run_id),
    )
    derived = sorted(
        (run for run in plan if run.role == "derived_compact"),
        key=lambda run: (run.model_id, run.pair_id, run.run_id),
    )
    summaries: dict[str, PilotRunSummary] = {}
    for run in sources:
        manifest, summary = _execute_one(
            run,
            manifest,
            authority=authority,
            provenance=provenance,
            output_root=output,
            parent=None,
            worker_timeout_seconds=worker_timeout_seconds,
        )
        summaries[run.run_id] = summary
    for run in derived:
        if run.parent_run_id is None or run.parent_run_id not in summaries:
            raise PilotRunnerError("compact run has no completed curriculum parent")
        parent_state = _manifest_state(manifest, run.parent_run_id)
        if not parent_state.attempts or parent_state.attempts[-1].status != "completed":
            raise PilotRunnerError("compact run parent is not durably completed")
        manifest, summary = _execute_one(
            run,
            manifest,
            authority=authority,
            provenance=provenance,
            output_root=output,
            parent=summaries[run.parent_run_id],
            worker_timeout_seconds=worker_timeout_seconds,
        )
        summaries[run.run_id] = summary
    final = load_campaign_manifest(
        authority,
        external_root=output,
        checkout_root=provenance.repo_root,
    )
    if len(summaries) != len(plan) or any(
        not state.attempts or state.attempts[-1].status != "completed"
        for state in final.runs
    ):
        raise PilotRunnerError("campaign success requires every planned run to complete")
    for pair in config.pairs:
        pair_sources = tuple(
            summaries[run.run_id]
            for run in sources
            if run.pair_id == pair.pair_id
        )
        if not pair_sources:
            raise PilotRunnerError("campaign pair has no source runs")
        expected_validation = pair_sources[0].validation_batch_hashes
        expected_training_hashes = pair_sources[0].training_batch_hashes
        expected_training_tokens = pair_sources[0].training_token_counts
        expected_stream_prefix = pair_sources[0].stream_prefix_sha256
        if any(
            summary.validation_batch_hashes != expected_validation
            or summary.training_batch_hashes != expected_training_hashes
            or summary.training_token_counts != expected_training_tokens
            or summary.stream_prefix_sha256 != expected_stream_prefix
            for summary in pair_sources
        ):
            raise PilotRunnerError("paired source data streams are not identical")
        pair_summaries = tuple(
            summaries[run.run_id] for run in plan if run.pair_id == pair.pair_id
        )
        if any(
            summary.validation_batch_hashes != expected_validation
            for summary in pair_summaries
        ):
            raise PilotRunnerError("validation batch hashes differ within a paired lineage")
    _verify_provenance_unchanged(provenance)
    ordered = tuple(summaries[run.run_id] for run in (*sources, *derived))
    promotion: ArtifactReference | None = None
    promotion_decision: str | None = None
    if config.stage is CampaignStage.SCREEN:
        promotion_document, promotion_decision = _screen_promotion_document(
            config=config,
            plan=plan,
            authority=authority,
            provenance=provenance,
            manifest=final,
            summaries=summaries,
        )
        promotion = _write_screen_promotion(
            output_root=output,
            repo_root=provenance.repo_root,
            config=config,
            document=promotion_document,
        )
        reloaded = load_campaign_manifest(
            authority,
            external_root=output,
            checkout_root=provenance.repo_root,
        )
        if (
            reloaded.generation != final.generation
            or campaign_manifest_sha256(reloaded) != campaign_manifest_sha256(final)
        ):
            raise PilotRunnerError("campaign manifest changed while promotion was published")
        _verify_provenance_unchanged(provenance)
    return PilotCampaignSummary(
        campaign_id=config.campaign_id,
        plan_sha256=authority.plan_sha256,
        manifest_generation=final.generation,
        code_commit=provenance.code_commit,
        code_tree=provenance.code_tree,
        raw_config_sha256=provenance.raw_config_sha256,
        semantic_config_sha256=config.fingerprint(),
        package_tree_sha256=provenance.package_tree_sha256,
        executable_bundle_sha256=provenance.executable_bundle_sha256,
        runs=ordered,
        claim_eligible=False,
        promotion=promotion,
        promotion_decision=promotion_decision,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--worker-timeout-seconds", type=float, default=_WORKER_TIMEOUT_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = run_milestone4_pilot(
            arguments.repo_root,
            arguments.config,
            arguments.output_root,
            arguments.worker,
            expected_commit=arguments.expected_commit,
            worker_timeout_seconds=arguments.worker_timeout_seconds,
        )
    except Exception as error:
        print(f"pilot failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    output_document: dict[str, object] = {
        "campaign_id": summary.campaign_id,
        "plan_sha256": summary.plan_sha256,
        "manifest_generation": summary.manifest_generation,
        "completed_runs": len(summary.runs),
        "claim_eligible": False,
    }
    if summary.promotion is not None:
        output_document["promotion"] = _artifact_json(summary.promotion)
        output_document["promotion_decision"] = summary.promotion_decision
    print(_canonical_bytes(output_document).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
