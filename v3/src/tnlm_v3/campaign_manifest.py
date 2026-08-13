"""Strict, content-addressed Milestone-4 campaign attempt records.

The campaign manifest is only a mutable, atomically replaced index.  Immutable
attempt records are the authority for state transitions.  A record is written
before its manifest pointer, so reconciliation can safely finish an interrupted
update without guessing whether a transition happened.

This module intentionally stores canonical JSON only.  Artifact bytes live
under a caller-declared external root and are referenced by normalized relative
POSIX path, byte count, and SHA-256.  No pickle or other executable format is
accepted here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any, Iterator, Mapping

from .campaign_config import (
    CampaignDataSpec,
    CampaignModelSpec,
    CampaignPairSpec,
    CampaignPromotionSpec,
    CampaignQualitySpec,
    CampaignRuntimeSpec,
    CampaignSelectionSpec,
    CampaignStatisticsSpec,
    CampaignTrainingSpec,
    EvaluationDataSpec,
    ImplementationPolicy,
    Milestone4CampaignConfig,
    ResolvedCampaignRun,
    TrainDataSpec,
    campaign_plan_sha256,
    resolve_campaign_plan,
)


_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_STATUSES = {"in_progress", "completed", "failed"}
_MAX_RECORD_BYTES = 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_JSON_NODES = 50_000
_MAX_JSON_DEPTH = 32
_MAX_FAILURE_CHARS = 4096
_MAX_ARTIFACT_BYTES = 2**63 - 1
_RECORD_FINGERPRINT_DOMAIN = b"tnlm-v3-m4-attempt-record-v1\0"
_MANIFEST_FINGERPRINT_DOMAIN = b"tnlm-v3-m4-campaign-manifest-v1\0"


class CampaignManifestError(ValueError):
    """Base class for invalid or inconsistent campaign state."""


class CampaignManifestLockedError(CampaignManifestError):
    """Raised when another process owns the campaign update lock."""


class StaleManifestGenerationError(CampaignManifestError):
    """Raised when a caller's compare-and-swap generation is stale."""


class ReconciliationRequiredError(CampaignManifestError):
    """Raised when immutable records are newer than the manifest index."""


class InProgressAttemptError(CampaignManifestError):
    """Raised when resume would have to guess whether a worker is still live."""


def _plain_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > 2**63 - 1:
        raise ValueError(f"{name} is outside the signed 64-bit range")
    return value


def _sha(value: object, name: str, *, forty: bool = False) -> str:
    pattern = _HEX40 if forty else _HEX64
    if not isinstance(value, str) or not pattern.fullmatch(value):
        width = 40 if forty else 64
        raise ValueError(f"{name} must be {width} lowercase hexadecimal characters")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase identifier")
    return value


def _relative_posix_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError(f"{name} must be a bounded relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.startswith("//")
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{name} must be a normalized relative POSIX path")
    return value


def _artifact_path(value: object, name: str) -> str:
    path = _relative_posix_path(value, name)
    if not path.startswith("artifacts/") or len(PurePosixPath(path).parts) < 2:
        raise ValueError(f"{name} must be beneath the reserved artifacts/ subtree")
    return path


def _exact(mapping: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        raise CampaignManifestError(
            f"invalid {name} keys; missing={missing}, unknown={unknown}"
        )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_constant(value: str) -> object:
    raise CampaignManifestError(f"non-finite JSON number {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CampaignManifestError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _validate_plain_json(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise CampaignManifestError("JSON contains too many nodes")
        if depth > _MAX_JSON_DEPTH:
            raise CampaignManifestError("JSON nesting is too deep")
        if item is None or isinstance(item, (str, bool)):
            continue
        if isinstance(item, int) and not isinstance(item, bool):
            _plain_int(item, "JSON integer")
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise CampaignManifestError("non-finite JSON numbers are forbidden")
            raise CampaignManifestError("floating-point JSON values are forbidden")
        if isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
            continue
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise CampaignManifestError("JSON object keys must be strings")
            stack.extend((child, depth + 1) for child in item.values())
            continue
        raise CampaignManifestError("JSON contains a non-plain value")


def _parse_canonical_json(raw: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CampaignManifestError(f"{name} must be valid UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, RecursionError) as error:
        raise CampaignManifestError(f"{name} must be bounded strict JSON") from error
    _validate_plain_json(value)
    if not isinstance(value, dict):
        raise CampaignManifestError(f"{name} must be a JSON object")
    if _canonical_json_bytes(value) != raw:
        raise CampaignManifestError(f"{name} is not canonical JSON")
    return value


def _is_link_like(path: Path, information: os.stat_result | None = None) -> bool:
    info = _lstat(path) if information is None else information
    attributes = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & reparse)


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _system_path(path: Path) -> str:
    """Return an extended absolute Windows path for low-level OS calls."""

    raw = str(path)
    if os.name != "nt" or raw.startswith("\\\\?\\"):
        return raw
    absolute = str(path.absolute())
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def _lstat(path: Path) -> os.stat_result:
    return os.stat(_system_path(path), follow_symlinks=False)


def _path_exists(path: Path) -> bool:
    try:
        _lstat(path)
    except FileNotFoundError:
        return False
    return True


def _children(path: Path) -> Iterator[Path]:
    with os.scandir(_system_path(path)) as entries:
        names = [entry.name for entry in entries]
    for name in names:
        yield path / name


@dataclass(frozen=True)
class _Roots:
    external: Path
    checkout: Path


def _validated_roots(
    external_root: str | Path,
    checkout_root: str | Path,
) -> _Roots:
    external_input = Path(external_root)
    checkout_input = Path(checkout_root)
    if not external_input.is_absolute() or not checkout_input.is_absolute():
        raise CampaignManifestError("external_root and checkout_root must be absolute")
    for path, name in ((external_input, "external_root"), (checkout_input, "checkout_root")):
        try:
            info = _lstat(path)
        except OSError as error:
            raise CampaignManifestError(f"{name} must already exist") from error
        if _is_link_like(path, info) or not stat.S_ISDIR(info.st_mode):
            raise CampaignManifestError(f"{name} must be a real directory, not a link")
    external = external_input.resolve(strict=True)
    checkout = checkout_input.resolve(strict=True)
    # Reject roots that arrive through a linked/reparse-point ancestor as well
    # as roots whose final component is itself a link.  The caller must name
    # the authoritative physical directories directly.
    if external != Path(os.path.abspath(external_input)):
        raise CampaignManifestError("external_root must not traverse a link")
    if checkout != Path(os.path.abspath(checkout_input)):
        raise CampaignManifestError("checkout_root must not traverse a link")
    if _within(external, checkout) or _within(checkout, external):
        raise CampaignManifestError(
            "external_root and checkout_root must be disjoint directories"
        )
    return _Roots(external=external, checkout=checkout)


def _safe_child(roots: _Roots, relative_path: str, *, must_exist: bool) -> Path:
    normalized = _relative_posix_path(relative_path, "relative_path")
    current = roots.external
    parts = PurePosixPath(normalized).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            information = _lstat(current)
        except FileNotFoundError:
            if must_exist or index < len(parts) - 1:
                raise CampaignManifestError(f"missing external path {normalized!r}")
            break
        if _is_link_like(current, information):
            raise CampaignManifestError(f"linked external path {normalized!r} is forbidden")
        if index < len(parts) - 1 and not stat.S_ISDIR(information.st_mode):
            raise CampaignManifestError(f"external path parent for {normalized!r} is not a directory")
    if must_exist:
        if not _within(current.absolute(), roots.external):
            raise CampaignManifestError("external path escapes external_root")
    return current


def _ensure_directory(roots: _Roots, relative_path: str) -> Path:
    normalized = _relative_posix_path(relative_path, "directory path")
    current = roots.external
    for part in PurePosixPath(normalized).parts:
        current = current / part
        try:
            os.mkdir(_system_path(current))
        except FileExistsError:
            pass
        info = _lstat(current)
        if _is_link_like(current, info) or not stat.S_ISDIR(info.st_mode):
            raise CampaignManifestError("campaign directory contains a link or non-directory")
    if not _within(current.absolute(), roots.external):
        raise CampaignManifestError("campaign directory escapes external_root")
    return current


def _read_safe_file(
    roots: _Roots,
    relative_path: str,
    *,
    maximum_bytes: int,
) -> bytes:
    path = _safe_child(roots, relative_path, must_exist=True)
    before = _lstat(path)
    if (
        _is_link_like(path, before)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise CampaignManifestError("referenced file must be a non-linked regular file")
    if before.st_size > maximum_bytes:
        raise CampaignManifestError("referenced JSON file exceeds its byte limit")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(_system_path(path), flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        ):
            raise CampaignManifestError("referenced file changed before it was opened")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(maximum_bytes + 1)
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size:
        raise CampaignManifestError("referenced file was not read in full")
    if len(raw) > maximum_bytes:
        raise CampaignManifestError("referenced JSON file exceeds its byte limit")
    after = _lstat(path)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_nlink,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_nlink,
    )
    if identity_before != identity_after or _is_link_like(path, after):
        raise CampaignManifestError("referenced file changed while it was read")
    return raw


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(
            _system_path(path), os.O_RDONLY | getattr(os, "O_BINARY", 0)
        )
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_replace(path: Path, raw: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp-", dir=_system_path(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(_system_path(temporary), _system_path(path))
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(_system_path(temporary))
        except FileNotFoundError:
            pass


def _write_immutable(path: Path, raw: bytes, roots: _Roots) -> None:
    relative = path.relative_to(roots.external).as_posix()
    if _path_exists(path):
        existing = _read_safe_file(roots, relative, maximum_bytes=_MAX_RECORD_BYTES)
        if existing != raw:
            raise CampaignManifestError("content-addressed record path already has other bytes")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp-", dir=_system_path(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(_system_path(temporary), _system_path(path))
        except FileExistsError:
            existing = _read_safe_file(
                roots,
                relative,
                maximum_bytes=_MAX_RECORD_BYTES,
            )
            if existing != raw:
                raise CampaignManifestError(
                    "content-addressed record path already has other bytes"
                )
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(_system_path(temporary))
        except FileNotFoundError:
            pass
    information = _lstat(path)
    if _is_link_like(path, information) or information.st_nlink != 1:
        raise CampaignManifestError("immutable record was not written as a private file")


@dataclass(frozen=True)
class ArtifactReference:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _artifact_path(self.path, "artifact path")
        _sha(self.sha256, "artifact sha256")
        _plain_int(self.size_bytes, "artifact size_bytes")


@dataclass(frozen=True)
class RecordReference:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _relative_posix_path(self.path, "record path")
        _sha(self.sha256, "record sha256")
        _plain_int(self.size_bytes, "record size_bytes", minimum=1)
        if self.size_bytes > _MAX_RECORD_BYTES:
            raise ValueError("record reference exceeds the record byte limit")


@dataclass(frozen=True)
class AttemptRecord:
    schema_version: int
    campaign_id: str
    plan_sha256: str
    run_id: str
    model_id: str
    pair_id: str
    run_sha256: str
    attempt_number: int
    revision: int
    status: str
    base_manifest_generation: int
    target_manifest_generation: int
    started_at_ns: int
    updated_at_ns: int
    finished_at_ns: int | None
    previous_record: RecordReference | None
    checkpoint: ArtifactReference | None
    output: ArtifactReference | None
    result: ArtifactReference | None
    failure: str | None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("attempt record schema_version must be integer 1")
        _identifier(self.campaign_id, "campaign_id")
        _sha(self.plan_sha256, "plan_sha256")
        _sha(self.run_id, "run_id")
        _identifier(self.model_id, "model_id")
        _identifier(self.pair_id, "pair_id")
        _sha(self.run_sha256, "run_sha256")
        _plain_int(self.attempt_number, "attempt_number", minimum=1)
        _plain_int(self.revision, "revision")
        if self.status not in _STATUSES:
            raise ValueError("attempt status is invalid")
        base = _plain_int(self.base_manifest_generation, "base_manifest_generation")
        target = _plain_int(self.target_manifest_generation, "target_manifest_generation")
        if target != base + 1:
            raise ValueError("attempt transition must advance the manifest by one")
        started = _plain_int(self.started_at_ns, "started_at_ns")
        updated = _plain_int(self.updated_at_ns, "updated_at_ns")
        if updated < started:
            raise ValueError("attempt timestamps move backwards")
        if self.revision == 0:
            if self.previous_record is not None or self.status != "in_progress":
                raise ValueError("the first attempt record must start in progress")
            if updated != started:
                raise ValueError("the first attempt record must use one timestamp")
        elif type(self.previous_record) is not RecordReference:
            raise TypeError("later attempt revisions require a RecordReference")
        for name in ("checkpoint", "output", "result"):
            value = getattr(self, name)
            if value is not None and type(value) is not ArtifactReference:
                raise TypeError(f"{name} must be an ArtifactReference")
        if self.status == "in_progress":
            if self.finished_at_ns is not None or self.output is not None or self.result is not None:
                raise ValueError("in-progress records cannot contain terminal fields")
            if self.failure is not None:
                raise ValueError("in-progress records cannot contain a failure")
        elif self.status == "completed":
            if self.finished_at_ns != updated:
                raise ValueError("completed record must finish at its update timestamp")
            if self.output is None or self.result is None or self.failure is not None:
                raise ValueError("completed record requires output and result only")
        else:
            if self.finished_at_ns != updated or self.output is not None or self.result is not None:
                raise ValueError("failed record has invalid terminal fields")
            if (
                not isinstance(self.failure, str)
                or not self.failure.strip()
                or len(self.failure) > _MAX_FAILURE_CHARS
                or "\x00" in self.failure
            ):
                raise ValueError("failed record requires a bounded failure message")


@dataclass(frozen=True)
class AttemptIndexEntry:
    attempt_number: int
    revision: int
    status: str
    record: RecordReference

    def __post_init__(self) -> None:
        _plain_int(self.attempt_number, "attempt_number", minimum=1)
        _plain_int(self.revision, "revision")
        if self.status not in _STATUSES:
            raise ValueError("attempt index status is invalid")
        if type(self.record) is not RecordReference:
            raise TypeError("attempt index record has the wrong type")


@dataclass(frozen=True)
class CampaignRunState:
    run_id: str
    model_id: str
    pair_id: str
    run_sha256: str
    next_attempt: int
    attempts: tuple[AttemptIndexEntry, ...]

    def __post_init__(self) -> None:
        _sha(self.run_id, "run_id")
        _identifier(self.model_id, "model_id")
        _identifier(self.pair_id, "pair_id")
        _sha(self.run_sha256, "run_sha256")
        _plain_int(self.next_attempt, "next_attempt", minimum=1)
        if not isinstance(self.attempts, tuple) or any(
            type(item) is not AttemptIndexEntry for item in self.attempts
        ):
            raise TypeError("attempts must be a tuple of AttemptIndexEntry")
        expected = tuple(range(1, len(self.attempts) + 1))
        if tuple(item.attempt_number for item in self.attempts) != expected:
            raise ValueError("attempt index must be contiguous and unique")
        if self.next_attempt != len(self.attempts) + 1:
            raise ValueError("next_attempt does not match the attempt index")


@dataclass(frozen=True)
class CampaignManifest:
    schema_version: int
    campaign_id: str
    stage: str
    generation: int
    plan_sha256: str
    semantic_config_sha256: str
    raw_config_sha256: str
    executable_bundle_sha256: str
    code_commit: str
    code_tree: str
    model_ids: tuple[str, ...]
    pair_ids: tuple[str, ...]
    runs: tuple[CampaignRunState, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("manifest schema_version must be integer 1")
        _identifier(self.campaign_id, "campaign_id")
        if self.stage not in {"pilot", "screen", "confirmatory"}:
            raise ValueError("manifest stage is invalid")
        _plain_int(self.generation, "generation")
        _sha(self.plan_sha256, "plan_sha256")
        _sha(self.semantic_config_sha256, "semantic_config_sha256")
        _sha(self.raw_config_sha256, "raw_config_sha256")
        _sha(self.executable_bundle_sha256, "executable_bundle_sha256")
        _sha(self.code_commit, "code_commit", forty=True)
        _sha(self.code_tree, "code_tree", forty=True)
        for name, values in (("model_ids", self.model_ids), ("pair_ids", self.pair_ids)):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple")
            for value in values:
                _identifier(value, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if not isinstance(self.runs, tuple) or any(
            type(run) is not CampaignRunState for run in self.runs
        ):
            raise TypeError("manifest runs have the wrong type")
        if tuple(run.run_id for run in self.runs) != tuple(
            sorted(run.run_id for run in self.runs)
        ):
            raise ValueError("manifest runs must be sorted by run_id")


def _rebuild_exact(value: object, expected_type: type) -> object:
    if type(value) is not expected_type:
        raise TypeError(f"campaign member must have exact type {expected_type.__name__}")
    return expected_type(**asdict(value))


def _deep_validate_config(config: Milestone4CampaignConfig) -> None:
    """Rebuild every nested spec so forged frozen objects cannot bypass checks."""

    if type(config) is not Milestone4CampaignConfig:
        raise TypeError("config must be Milestone4CampaignConfig")
    policy = _rebuild_exact(config.implementation_policy, ImplementationPolicy)
    models = tuple(
        _rebuild_exact(model, CampaignModelSpec) for model in config.models
    )
    pairs = tuple(_rebuild_exact(pair, CampaignPairSpec) for pair in config.pairs)
    train = _rebuild_exact(config.data.train, TrainDataSpec)
    validation = _rebuild_exact(config.data.validation, EvaluationDataSpec)
    test = (
        None
        if config.data.test is None
        else _rebuild_exact(config.data.test, EvaluationDataSpec)
    )
    scaling = (
        None
        if config.data.scaling is None
        else _rebuild_exact(config.data.scaling, EvaluationDataSpec)
    )
    data = CampaignDataSpec(
        generator_version=config.data.generator_version,
        train=train,
        validation=validation,
        test=test,
        scaling=scaling,
    )
    training = _rebuild_exact(config.training, CampaignTrainingSpec)
    quality = _rebuild_exact(config.quality, CampaignQualitySpec)
    statistics = _rebuild_exact(config.statistics, CampaignStatisticsSpec)
    runtime = _rebuild_exact(config.runtime, CampaignRuntimeSpec)
    selection = (
        None
        if config.selection is None
        else _rebuild_exact(config.selection, CampaignSelectionSpec)
    )
    promotion = (
        None
        if config.promotion is None
        else _rebuild_exact(config.promotion, CampaignPromotionSpec)
    )
    rebuilt = Milestone4CampaignConfig(
        schema_version=config.schema_version,
        campaign_id=config.campaign_id,
        stage=config.stage,
        description=config.description,
        claim_eligible=config.claim_eligible,
        implementation_policy=policy,
        task=type(config.task)(**asdict(config.task)),
        models=models,
        pairs=pairs,
        data=data,
        training=training,
        quality=quality,
        statistics=statistics,
        runtime=runtime,
        selection=selection,
        promotion=promotion,
    )
    if rebuilt != config:
        raise ValueError("config changes under exact nested reconstruction")


def _deep_validate_run(run: ResolvedCampaignRun) -> None:
    """Rebuild a run and its nested specs before trusting its identity hash."""

    if type(run) is not ResolvedCampaignRun:
        raise TypeError("run must be ResolvedCampaignRun")
    data = CampaignDataSpec(
        generator_version=run.data.generator_version,
        train=_rebuild_exact(run.data.train, TrainDataSpec),
        validation=_rebuild_exact(run.data.validation, EvaluationDataSpec),
        test=(
            None
            if run.data.test is None
            else _rebuild_exact(run.data.test, EvaluationDataSpec)
        ),
        scaling=(
            None
            if run.data.scaling is None
            else _rebuild_exact(run.data.scaling, EvaluationDataSpec)
        ),
    )
    training = (
        None
        if run.training is None
        else _rebuild_exact(run.training, CampaignTrainingSpec)
    )
    values = asdict(run)
    values["task"] = type(run.task)(**asdict(run.task))
    values["data"] = data
    values["training"] = training
    rebuilt = ResolvedCampaignRun(**values)
    if rebuilt != run:
        raise ValueError("run changes under exact nested reconstruction")


@dataclass(frozen=True)
class CampaignAuthority:
    config: Milestone4CampaignConfig
    resolved_plan: tuple[ResolvedCampaignRun, ...]
    plan_sha256: str
    code_commit: str
    code_tree: str
    raw_config_sha256: str
    executable_bundle_sha256: str
    working_tree_clean: bool

    def __post_init__(self) -> None:
        if type(self.config) is not Milestone4CampaignConfig:
            raise TypeError("config must be Milestone4CampaignConfig")
        _deep_validate_config(self.config)
        if not isinstance(self.resolved_plan, tuple) or any(
            type(run) is not ResolvedCampaignRun for run in self.resolved_plan
        ):
            raise TypeError("resolved_plan must be a tuple of ResolvedCampaignRun")
        for run in self.resolved_plan:
            _deep_validate_run(run)
        _sha(self.plan_sha256, "plan_sha256")
        _sha(self.code_commit, "code_commit", forty=True)
        _sha(self.code_tree, "code_tree", forty=True)
        _sha(self.raw_config_sha256, "raw_config_sha256")
        _sha(self.executable_bundle_sha256, "executable_bundle_sha256")
        if self.working_tree_clean is not True:
            raise ValueError("working_tree_clean must be literal true")
        expected = resolve_campaign_plan(
            self.config,
            self.code_commit,
            self.code_tree,
            self.raw_config_sha256,
            self.executable_bundle_sha256,
        )
        if self.resolved_plan != expected:
            raise ValueError("resolved_plan is not the exact ordered config plan")
        expected_sha = campaign_plan_sha256(self.config, self.resolved_plan)
        if self.plan_sha256 != expected_sha:
            raise ValueError("plan_sha256 does not match the exact resolved plan")


def _run_sha256(run: ResolvedCampaignRun) -> str:
    return hashlib.sha256(run.canonical_json().encode("utf-8")).hexdigest()


def _campaign_directory(authority: CampaignAuthority) -> str:
    return f"campaigns/{authority.config.campaign_id}"


def _manifest_relative_path(authority: CampaignAuthority) -> str:
    return f"{_campaign_directory(authority)}/manifest.json"


def campaign_manifest_path(
    authority: CampaignAuthority,
    *,
    external_root: str | Path,
    checkout_root: str | Path,
) -> Path:
    """Return the validated absolute path of this campaign's manifest."""

    if type(authority) is not CampaignAuthority:
        raise TypeError("authority must be CampaignAuthority")
    authority.__post_init__()
    roots = _validated_roots(external_root, checkout_root)
    return _safe_child(roots, _manifest_relative_path(authority), must_exist=False)


def make_artifact_reference(
    relative_path: str,
    *,
    external_root: str | Path,
    checkout_root: str | Path,
) -> ArtifactReference:
    """Hash one private, regular file under the authoritative external root."""

    roots = _validated_roots(external_root, checkout_root)
    normalized = _artifact_path(relative_path, "artifact path")
    path = _safe_child(roots, normalized, must_exist=True)
    before = _lstat(path)
    if (
        _is_link_like(path, before)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise CampaignManifestError("artifact must be a private regular file")
    if before.st_size > _MAX_ARTIFACT_BYTES:
        raise CampaignManifestError("artifact exceeds the supported size range")
    digest = hashlib.sha256()
    measured = 0
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(_system_path(path), flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        ):
            raise CampaignManifestError("artifact changed before it was opened")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                measured += len(chunk)
                if measured > _MAX_ARTIFACT_BYTES:
                    raise CampaignManifestError("artifact exceeds the supported size range")
                digest.update(chunk)
    finally:
        os.close(descriptor)
    after = _lstat(path)
    if measured != before.st_size:
        raise CampaignManifestError("artifact was not hashed in full")
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink)
        or _is_link_like(path, after)
    ):
        raise CampaignManifestError("artifact changed while it was hashed")
    return ArtifactReference(normalized, digest.hexdigest(), measured)


def _verify_artifact(reference: ArtifactReference, roots: _Roots) -> None:
    measured = make_artifact_reference(
        reference.path,
        external_root=roots.external,
        checkout_root=roots.checkout,
    )
    if measured != reference:
        raise CampaignManifestError(f"artifact reference {reference.path!r} no longer matches")


def attempt_record_canonical_bytes(record: AttemptRecord) -> bytes:
    """Return the deterministic canonical bytes of a validated attempt record."""

    if type(record) is not AttemptRecord:
        raise TypeError("record must be AttemptRecord")
    record.__post_init__()
    return _canonical_json_bytes(asdict(record))


def attempt_record_fingerprint(record: AttemptRecord) -> str:
    """Return a domain-separated content fingerprint for an attempt record."""

    return hashlib.sha256(
        _RECORD_FINGERPRINT_DOMAIN + attempt_record_canonical_bytes(record)
    ).hexdigest()


def campaign_manifest_canonical_bytes(manifest: CampaignManifest) -> bytes:
    """Return the deterministic canonical bytes of a validated manifest index."""

    if type(manifest) is not CampaignManifest:
        raise TypeError("manifest must be CampaignManifest")
    manifest.__post_init__()
    value = asdict(manifest)
    return _canonical_json_bytes(value)


def campaign_manifest_fingerprint(manifest: CampaignManifest) -> str:
    """Return a domain-separated fingerprint for the mutable manifest snapshot."""

    return hashlib.sha256(
        _MANIFEST_FINGERPRINT_DOMAIN + campaign_manifest_canonical_bytes(manifest)
    ).hexdigest()


def campaign_manifest_sha256(manifest: CampaignManifest) -> str:
    """Return the plain SHA-256 of the exact canonical manifest file bytes."""

    return hashlib.sha256(campaign_manifest_canonical_bytes(manifest)).hexdigest()


def _artifact_from_json(value: object, name: str) -> ArtifactReference | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CampaignManifestError(f"{name} must be an artifact object or null")
    _exact(value, {"path", "sha256", "size_bytes"}, name)
    return ArtifactReference(**value)


def _record_reference_from_json(value: object, name: str) -> RecordReference | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CampaignManifestError(f"{name} must be a record reference or null")
    _exact(value, {"path", "sha256", "size_bytes"}, name)
    return RecordReference(**value)


def _attempt_record_from_json(value: Mapping[str, Any]) -> AttemptRecord:
    expected = set(AttemptRecord.__dataclass_fields__)
    _exact(value, expected, "attempt record")
    fields = dict(value)
    fields["previous_record"] = _record_reference_from_json(
        fields["previous_record"], "previous_record"
    )
    for name in ("checkpoint", "output", "result"):
        fields[name] = _artifact_from_json(fields[name], name)
    try:
        return AttemptRecord(**fields)
    except (TypeError, ValueError) as error:
        raise CampaignManifestError("invalid attempt record values") from error


def _manifest_from_json(value: Mapping[str, Any]) -> CampaignManifest:
    _exact(value, set(CampaignManifest.__dataclass_fields__), "campaign manifest")
    fields = dict(value)
    model_ids = fields["model_ids"]
    pair_ids = fields["pair_ids"]
    run_values = fields["runs"]
    if not isinstance(model_ids, list) or not isinstance(pair_ids, list):
        raise CampaignManifestError("manifest axes must be JSON arrays")
    if not isinstance(run_values, list):
        raise CampaignManifestError("manifest runs must be a JSON array")
    try:
        runs: list[CampaignRunState] = []
        for run_value in run_values:
            if not isinstance(run_value, dict):
                raise CampaignManifestError("manifest run must be an object")
            _exact(run_value, set(CampaignRunState.__dataclass_fields__), "manifest run")
            run_fields = dict(run_value)
            attempts_value = run_fields["attempts"]
            if not isinstance(attempts_value, list):
                raise CampaignManifestError("manifest attempts must be a JSON array")
            attempts: list[AttemptIndexEntry] = []
            for attempt_value in attempts_value:
                if not isinstance(attempt_value, dict):
                    raise CampaignManifestError("attempt index entry must be an object")
                _exact(
                    attempt_value,
                    set(AttemptIndexEntry.__dataclass_fields__),
                    "attempt index entry",
                )
                attempt_fields = dict(attempt_value)
                reference = _record_reference_from_json(
                    attempt_fields["record"], "attempt index record"
                )
                assert reference is not None
                attempt_fields["record"] = reference
                attempts.append(AttemptIndexEntry(**attempt_fields))
            run_fields["attempts"] = tuple(attempts)
            runs.append(CampaignRunState(**run_fields))
        fields["model_ids"] = tuple(model_ids)
        fields["pair_ids"] = tuple(pair_ids)
        fields["runs"] = tuple(runs)
        return CampaignManifest(**fields)
    except CampaignManifestError:
        raise
    except (TypeError, ValueError) as error:
        raise CampaignManifestError("invalid campaign manifest values") from error


def _initial_run_states(authority: CampaignAuthority) -> tuple[CampaignRunState, ...]:
    return tuple(
        CampaignRunState(
            run_id=run.run_id,
            model_id=run.model_id,
            pair_id=run.pair_id,
            run_sha256=_run_sha256(run),
            next_attempt=1,
            attempts=(),
        )
        for run in sorted(authority.resolved_plan, key=lambda item: item.run_id)
    )


def _initial_manifest(authority: CampaignAuthority) -> CampaignManifest:
    return CampaignManifest(
        schema_version=1,
        campaign_id=authority.config.campaign_id,
        stage=authority.config.stage.value,
        generation=0,
        plan_sha256=authority.plan_sha256,
        semantic_config_sha256=authority.config.fingerprint(),
        raw_config_sha256=authority.raw_config_sha256,
        executable_bundle_sha256=authority.executable_bundle_sha256,
        code_commit=authority.code_commit,
        code_tree=authority.code_tree,
        model_ids=tuple(sorted(model.model_id for model in authority.config.models)),
        pair_ids=tuple(sorted(pair.pair_id for pair in authority.config.pairs)),
        runs=_initial_run_states(authority),
    )


def _validate_manifest_authority(
    manifest: CampaignManifest,
    authority: CampaignAuthority,
) -> None:
    expected = _initial_manifest(authority)
    identity_fields = (
        "schema_version",
        "campaign_id",
        "stage",
        "plan_sha256",
        "semantic_config_sha256",
        "raw_config_sha256",
        "executable_bundle_sha256",
        "code_commit",
        "code_tree",
        "model_ids",
        "pair_ids",
    )
    for name in identity_fields:
        if getattr(manifest, name) != getattr(expected, name):
            raise CampaignManifestError(f"manifest {name} changed from its authority")
    expected_runs = {
        run.run_id: (run.model_id, run.pair_id, run.run_sha256)
        for run in expected.runs
    }
    observed_runs = {
        run.run_id: (run.model_id, run.pair_id, run.run_sha256)
        for run in manifest.runs
    }
    if observed_runs != expected_runs or len(manifest.runs) != len(expected.runs):
        raise CampaignManifestError("manifest is missing or changed a model/pair run axis")


def _load_manifest(authority: CampaignAuthority, roots: _Roots) -> CampaignManifest:
    relative = _manifest_relative_path(authority)
    raw = _read_safe_file(roots, relative, maximum_bytes=_MAX_MANIFEST_BYTES)
    value = _parse_canonical_json(raw, name="campaign manifest")
    manifest = _manifest_from_json(value)
    _validate_manifest_authority(manifest, authority)
    return manifest


def _write_manifest(manifest: CampaignManifest, authority: CampaignAuthority, roots: _Roots) -> None:
    raw = campaign_manifest_canonical_bytes(manifest)
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise CampaignManifestError("campaign manifest exceeds its byte limit")
    path = _safe_child(roots, _manifest_relative_path(authority), must_exist=False)
    if _path_exists(path):
        info = _lstat(path)
        if _is_link_like(path, info) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise CampaignManifestError("manifest target is linked or not a regular file")
    _atomic_replace(path, raw)


@contextmanager
def _campaign_lock(authority: CampaignAuthority, roots: _Roots) -> Iterator[None]:
    campaign_directory = _ensure_directory(roots, _campaign_directory(authority))
    lock_path = campaign_directory / "manifest.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(_system_path(lock_path), flags, 0o600)
    try:
        current = _lstat(lock_path)
        opened = os.fstat(descriptor)
        if (
            _is_link_like(lock_path, current)
            or current.st_nlink != 1
            or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise CampaignManifestError("campaign lock is linked or was replaced")
        if opened.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise CampaignManifestLockedError("campaign manifest is locked") from error
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def initialize_campaign_manifest(
    authority: CampaignAuthority,
    *,
    external_root: str | Path,
    checkout_root: str | Path,
) -> CampaignManifest:
    """Create generation zero for an exact, clean, fully resolved campaign."""

    if type(authority) is not CampaignAuthority:
        raise TypeError("authority must be CampaignAuthority")
    authority.__post_init__()
    roots = _validated_roots(external_root, checkout_root)
    with _campaign_lock(authority, roots):
        manifest_path = _safe_child(
            roots, _manifest_relative_path(authority), must_exist=False
        )
        records_path = roots.external / _campaign_directory(authority) / "records"
        if _path_exists(manifest_path):
            raise FileExistsError("campaign manifest already exists")
        if _path_exists(records_path) and any(_children(records_path)):
            raise CampaignManifestError("campaign record directory is not empty")
        _ensure_directory(roots, f"{_campaign_directory(authority)}/records")
        manifest = _initial_manifest(authority)
        _write_manifest(manifest, authority, roots)
        return manifest


def _record_relative_path(record: AttemptRecord, raw_sha256: str) -> str:
    # The content digest is the authority; run/attempt identity lives in the
    # validated record itself.  The two-level digest layout also stays well
    # below legacy Windows path limits when an external root is already long.
    return (
        f"campaigns/{record.campaign_id}/records/"
        f"{raw_sha256[:2]}/{raw_sha256}.json"
    )


@dataclass(frozen=True)
class _LoadedRecord:
    record: AttemptRecord
    reference: RecordReference


def _scan_record_files(authority: CampaignAuthority, roots: _Roots) -> list[str]:
    records_relative = f"{_campaign_directory(authority)}/records"
    records_root = _safe_child(roots, records_relative, must_exist=True)
    found: list[str] = []
    pending = [records_root]
    while pending:
        directory = pending.pop()
        for child in _children(directory):
            information = _lstat(child)
            if _is_link_like(child, information):
                raise CampaignManifestError("campaign records contain a linked path")
            if stat.S_ISDIR(information.st_mode):
                pending.append(child)
                continue
            if not stat.S_ISREG(information.st_mode) or information.st_nlink != 1:
                raise CampaignManifestError("campaign records contain a non-private file")
            if child.name.startswith(".tmp-"):
                continue
            if child.suffix != ".json":
                raise CampaignManifestError("campaign records contain an unexpected file")
            found.append(child.relative_to(roots.external).as_posix())
    return sorted(found)


def _load_all_records(
    authority: CampaignAuthority,
    roots: _Roots,
) -> tuple[_LoadedRecord, ...]:
    expected_runs = {run.run_id: run for run in authority.resolved_plan}
    loaded: list[_LoadedRecord] = []
    seen_locations: set[tuple[str, int, int]] = set()
    for relative in _scan_record_files(authority, roots):
        raw = _read_safe_file(roots, relative, maximum_bytes=_MAX_RECORD_BYTES)
        value = _parse_canonical_json(raw, name="attempt record")
        record = _attempt_record_from_json(value)
        raw_sha = hashlib.sha256(raw).hexdigest()
        expected_path = _record_relative_path(record, raw_sha)
        if relative != expected_path:
            raise CampaignManifestError("attempt record path is not content-addressed")
        run = expected_runs.get(record.run_id)
        if (
            run is None
            or record.campaign_id != authority.config.campaign_id
            or record.plan_sha256 != authority.plan_sha256
            or record.model_id != run.model_id
            or record.pair_id != run.pair_id
            or record.run_sha256 != _run_sha256(run)
        ):
            raise CampaignManifestError("attempt record does not match its resolved run")
        for artifact in (record.checkpoint, record.output, record.result):
            if artifact is not None:
                _verify_artifact(artifact, roots)
        key = (record.run_id, record.attempt_number, record.revision)
        if key in seen_locations:
            raise CampaignManifestError("duplicate attempt revision is ambiguous")
        seen_locations.add(key)
        loaded.append(
            _LoadedRecord(
                record=record,
                reference=RecordReference(relative, raw_sha, len(raw)),
            )
        )
    return tuple(loaded)


def _pointer(loaded: _LoadedRecord) -> AttemptIndexEntry:
    record = loaded.record
    return AttemptIndexEntry(
        attempt_number=record.attempt_number,
        revision=record.revision,
        status=record.status,
        record=loaded.reference,
    )


def _apply_transition(
    states: dict[str, CampaignRunState],
    loaded: _LoadedRecord,
) -> None:
    record = loaded.record
    state = states[record.run_id]
    pointer = _pointer(loaded)
    if record.revision == 0:
        if record.attempt_number != state.next_attempt:
            raise CampaignManifestError("attempt counter moved backwards or skipped")
        if state.attempts and state.attempts[-1].status != "failed":
            raise CampaignManifestError("new attempt follows a non-failed attempt")
        states[record.run_id] = replace(
            state,
            next_attempt=state.next_attempt + 1,
            attempts=state.attempts + (pointer,),
        )
        return
    if not state.attempts:
        raise CampaignManifestError("later attempt revision has no start record")
    latest = state.attempts[-1]
    if (
        latest.attempt_number != record.attempt_number
        or latest.status != "in_progress"
        or latest.revision + 1 != record.revision
        or record.previous_record != latest.record
    ):
        raise CampaignManifestError("attempt transition is duplicate, stale, or backwards")
    states[record.run_id] = replace(
        state,
        attempts=state.attempts[:-1] + (pointer,),
    )


def _reconstruct(
    manifest: CampaignManifest,
    authority: CampaignAuthority,
    roots: _Roots,
) -> tuple[CampaignManifest, int]:
    loaded = _load_all_records(authority, roots)
    by_reference = {item.reference: item for item in loaded}
    generations: dict[int, _LoadedRecord] = {}
    for item in loaded:
        record = item.record
        if record.base_manifest_generation + 1 != record.target_manifest_generation:
            raise CampaignManifestError("attempt record generation is invalid")
        if record.target_manifest_generation in generations:
            raise CampaignManifestError("multiple records claim one manifest generation")
        generations[record.target_manifest_generation] = item
        if record.previous_record is not None:
            previous = by_reference.get(record.previous_record)
            if previous is None:
                raise CampaignManifestError("attempt record predecessor is missing or changed")
            prior = previous.record
            if (
                prior.run_id != record.run_id
                or prior.attempt_number != record.attempt_number
                or prior.revision + 1 != record.revision
                or prior.status != "in_progress"
                or prior.started_at_ns != record.started_at_ns
                or prior.updated_at_ns > record.updated_at_ns
                or prior.target_manifest_generation >= record.target_manifest_generation
            ):
                raise CampaignManifestError("attempt predecessor chain is invalid")
    maximum_generation = max(generations, default=0)
    if set(generations) != set(range(1, maximum_generation + 1)):
        raise CampaignManifestError("immutable record generations contain a gap")
    if manifest.generation > maximum_generation:
        raise CampaignManifestError("manifest generation has no authoritative record history")

    states = {run.run_id: run for run in _initial_run_states(authority)}
    indexed_state: tuple[CampaignRunState, ...] | None = (
        tuple(sorted(states.values(), key=lambda run: run.run_id))
        if manifest.generation == 0
        else None
    )
    for generation in range(1, maximum_generation + 1):
        item = generations[generation]
        if item.record.base_manifest_generation != generation - 1:
            raise CampaignManifestError("attempt records do not form a serial history")
        _apply_transition(states, item)
        if generation == manifest.generation:
            indexed_state = tuple(sorted(states.values(), key=lambda run: run.run_id))
    if indexed_state != manifest.runs:
        raise CampaignManifestError("manifest index disagrees with immutable record history")
    final_runs = tuple(sorted(states.values(), key=lambda run: run.run_id))
    recovered = maximum_generation - manifest.generation
    return replace(manifest, generation=maximum_generation, runs=final_runs), recovered


def _load_and_reconcile(
    authority: CampaignAuthority,
    roots: _Roots,
    *,
    write_recovery: bool,
) -> tuple[CampaignManifest, int]:
    manifest = _load_manifest(authority, roots)
    reconciled, recovered = _reconstruct(manifest, authority, roots)
    if recovered and write_recovery:
        _write_manifest(reconciled, authority, roots)
    return reconciled, recovered


def load_campaign_manifest(
    authority: CampaignAuthority,
    *,
    external_root: str | Path,
    checkout_root: str | Path,
) -> CampaignManifest:
    """Load a fully validated manifest, refusing an unindexed record tail."""

    if type(authority) is not CampaignAuthority:
        raise TypeError("authority must be CampaignAuthority")
    authority.__post_init__()
    roots = _validated_roots(external_root, checkout_root)
    with _campaign_lock(authority, roots):
        manifest, recovered = _load_and_reconcile(
            authority, roots, write_recovery=False
        )
        if recovered:
            raise ReconciliationRequiredError(
                "immutable records are newer than the manifest; reconcile first"
            )
        return manifest


def reconcile_campaign_manifest(
    authority: CampaignAuthority,
    *,
    external_root: str | Path,
    checkout_root: str | Path,
) -> CampaignManifest:
    """Recover a unique record tail and reject ambiguous live attempts."""

    if type(authority) is not CampaignAuthority:
        raise TypeError("authority must be CampaignAuthority")
    authority.__post_init__()
    roots = _validated_roots(external_root, checkout_root)
    with _campaign_lock(authority, roots):
        manifest, _ = _load_and_reconcile(authority, roots, write_recovery=True)
        in_progress = [
            run.run_id
            for run in manifest.runs
            if run.attempts and run.attempts[-1].status == "in_progress"
        ]
        if in_progress:
            raise InProgressAttemptError(
                "resume cannot classify in-progress attempts as live or stale: "
                + ",".join(in_progress)
            )
        return manifest


def _record_for_reference(reference: RecordReference, roots: _Roots) -> AttemptRecord:
    raw = _read_safe_file(roots, reference.path, maximum_bytes=_MAX_RECORD_BYTES)
    if len(raw) != reference.size_bytes or hashlib.sha256(raw).hexdigest() != reference.sha256:
        raise CampaignManifestError("indexed record reference is stale or tampered")
    return _attempt_record_from_json(_parse_canonical_json(raw, name="attempt record"))


def load_campaign_attempt_record(
    authority: CampaignAuthority,
    reference: RecordReference,
    *,
    external_root: str | Path,
    checkout_root: str | Path,
) -> AttemptRecord:
    """Load one artifact-verified attempt record reachable from the manifest."""

    if type(authority) is not CampaignAuthority:
        raise TypeError("authority must be CampaignAuthority")
    if type(reference) is not RecordReference:
        raise TypeError("reference must be an exact RecordReference")
    authority.__post_init__()
    roots = _validated_roots(external_root, checkout_root)
    with _campaign_lock(authority, roots):
        manifest, recovered = _load_and_reconcile(
            authority, roots, write_recovery=False
        )
        if recovered:
            raise ReconciliationRequiredError(
                "immutable records are newer than the manifest; reconcile first"
            )
        reachable = {
            attempt.record
            for run in manifest.runs
            for attempt in run.attempts
        }
        if reference not in reachable:
            raise CampaignManifestError("attempt record is not reachable from the manifest")
        record = _record_for_reference(reference, roots)
        expected_runs = {run.run_id: run for run in authority.resolved_plan}
        run = expected_runs.get(record.run_id)
        if (
            run is None
            or record.campaign_id != authority.config.campaign_id
            or record.plan_sha256 != authority.plan_sha256
            or record.model_id != run.model_id
            or record.pair_id != run.pair_id
            or record.run_sha256 != _run_sha256(run)
        ):
            raise CampaignManifestError("attempt record does not match its authority")
        for artifact in (record.checkpoint, record.output, record.result):
            if artifact is not None:
                _verify_artifact(artifact, roots)
        return record


def _write_record(record: AttemptRecord, roots: _Roots) -> _LoadedRecord:
    raw = attempt_record_canonical_bytes(record)
    if len(raw) > _MAX_RECORD_BYTES:
        raise CampaignManifestError("attempt record exceeds its byte limit")
    raw_sha = hashlib.sha256(raw).hexdigest()
    relative = _record_relative_path(record, raw_sha)
    parent = str(PurePosixPath(relative).parent)
    _ensure_directory(roots, parent)
    path = _safe_child(roots, relative, must_exist=False)
    _write_immutable(path, raw, roots)
    return _LoadedRecord(record, RecordReference(relative, raw_sha, len(raw)))


def _require_generation(manifest: CampaignManifest, expected_generation: int) -> None:
    _plain_int(expected_generation, "expected_generation")
    if manifest.generation != expected_generation:
        raise StaleManifestGenerationError(
            f"expected manifest generation {expected_generation}, found {manifest.generation}"
        )


def _transition_context(
    authority: CampaignAuthority,
    roots: _Roots,
    run_id: str,
    expected_generation: int,
) -> tuple[CampaignManifest, int, CampaignRunState]:
    _sha(run_id, "run_id")
    manifest, _ = _load_and_reconcile(authority, roots, write_recovery=True)
    _require_generation(manifest, expected_generation)
    for index, state in enumerate(manifest.runs):
        if state.run_id == run_id:
            return manifest, index, state
    raise CampaignManifestError("run_id is not in the exact campaign plan")


def _install_transition(
    manifest: CampaignManifest,
    run_index: int,
    loaded: _LoadedRecord,
    authority: CampaignAuthority,
    roots: _Roots,
) -> CampaignManifest:
    states = {run.run_id: run for run in manifest.runs}
    _apply_transition(states, loaded)
    runs = tuple(sorted(states.values(), key=lambda run: run.run_id))
    updated = replace(manifest, generation=manifest.generation + 1, runs=runs)
    # run_index is intentionally checked even though sorting is authoritative.
    if manifest.runs[run_index].run_id != loaded.record.run_id:
        raise CampaignManifestError("run index changed during transition")
    _write_manifest(updated, authority, roots)
    return updated


def start_campaign_attempt(
    authority: CampaignAuthority,
    run_id: str,
    *,
    timestamp_ns: int,
    expected_generation: int,
    external_root: str | Path,
    checkout_root: str | Path,
) -> tuple[CampaignManifest, AttemptRecord]:
    """Start the next numbered attempt for a pending or failed run."""

    if type(authority) is not CampaignAuthority:
        raise TypeError("authority must be CampaignAuthority")
    authority.__post_init__()
    roots = _validated_roots(external_root, checkout_root)
    timestamp = _plain_int(timestamp_ns, "timestamp_ns")
    with _campaign_lock(authority, roots):
        manifest, run_index, state = _transition_context(
            authority, roots, run_id, expected_generation
        )
        if state.attempts and state.attempts[-1].status != "failed":
            raise CampaignManifestError("run already has a live or completed attempt")
        record = AttemptRecord(
            schema_version=1,
            campaign_id=manifest.campaign_id,
            plan_sha256=manifest.plan_sha256,
            run_id=state.run_id,
            model_id=state.model_id,
            pair_id=state.pair_id,
            run_sha256=state.run_sha256,
            attempt_number=state.next_attempt,
            revision=0,
            status="in_progress",
            base_manifest_generation=manifest.generation,
            target_manifest_generation=manifest.generation + 1,
            started_at_ns=timestamp,
            updated_at_ns=timestamp,
            finished_at_ns=None,
            previous_record=None,
            checkpoint=None,
            output=None,
            result=None,
            failure=None,
        )
        loaded = _write_record(record, roots)
        updated = _install_transition(
            manifest, run_index, loaded, authority, roots
        )
        return updated, record


def _advance_attempt(
    authority: CampaignAuthority,
    run_id: str,
    attempt_number: int,
    *,
    timestamp_ns: int,
    expected_generation: int,
    status: str,
    checkpoint: ArtifactReference | None,
    output: ArtifactReference | None,
    result: ArtifactReference | None,
    failure: str | None,
    external_root: str | Path,
    checkout_root: str | Path,
) -> tuple[CampaignManifest, AttemptRecord]:
    if type(authority) is not CampaignAuthority:
        raise TypeError("authority must be CampaignAuthority")
    authority.__post_init__()
    roots = _validated_roots(external_root, checkout_root)
    timestamp = _plain_int(timestamp_ns, "timestamp_ns")
    _plain_int(attempt_number, "attempt_number", minimum=1)
    for name, reference in (
        ("checkpoint", checkpoint),
        ("output", output),
        ("result", result),
    ):
        if reference is not None and type(reference) is not ArtifactReference:
            raise TypeError(f"{name} must be an ArtifactReference")
        if reference is not None:
            _verify_artifact(reference, roots)
    with _campaign_lock(authority, roots):
        # Recheck after acquiring the serialization lock.  A caller may have
        # created the reference earlier, and artifacts are external files.
        for reference in (checkpoint, output, result):
            if reference is not None:
                _verify_artifact(reference, roots)
        manifest, run_index, state = _transition_context(
            authority, roots, run_id, expected_generation
        )
        if not state.attempts:
            raise CampaignManifestError("run has no attempt to advance")
        latest = state.attempts[-1]
        if latest.attempt_number != attempt_number or latest.status != "in_progress":
            raise CampaignManifestError("attempt is stale or already terminal")
        previous = _record_for_reference(latest.record, roots)
        if timestamp < previous.updated_at_ns:
            raise CampaignManifestError("attempt timestamp moves backwards")
        effective_checkpoint = checkpoint if checkpoint is not None else previous.checkpoint
        record = AttemptRecord(
            schema_version=1,
            campaign_id=manifest.campaign_id,
            plan_sha256=manifest.plan_sha256,
            run_id=state.run_id,
            model_id=state.model_id,
            pair_id=state.pair_id,
            run_sha256=state.run_sha256,
            attempt_number=attempt_number,
            revision=latest.revision + 1,
            status=status,
            base_manifest_generation=manifest.generation,
            target_manifest_generation=manifest.generation + 1,
            started_at_ns=previous.started_at_ns,
            updated_at_ns=timestamp,
            finished_at_ns=timestamp if status != "in_progress" else None,
            previous_record=latest.record,
            checkpoint=effective_checkpoint,
            output=output,
            result=result,
            failure=failure,
        )
        loaded = _write_record(record, roots)
        updated = _install_transition(
            manifest, run_index, loaded, authority, roots
        )
        return updated, record


def heartbeat_campaign_attempt(
    authority: CampaignAuthority,
    run_id: str,
    attempt_number: int,
    *,
    timestamp_ns: int,
    expected_generation: int,
    checkpoint: ArtifactReference | None = None,
    external_root: str | Path,
    checkout_root: str | Path,
) -> tuple[CampaignManifest, AttemptRecord]:
    """Append an immutable heartbeat, optionally advancing its checkpoint."""

    return _advance_attempt(
        authority,
        run_id,
        attempt_number,
        timestamp_ns=timestamp_ns,
        expected_generation=expected_generation,
        status="in_progress",
        checkpoint=checkpoint,
        output=None,
        result=None,
        failure=None,
        external_root=external_root,
        checkout_root=checkout_root,
    )


def complete_campaign_attempt(
    authority: CampaignAuthority,
    run_id: str,
    attempt_number: int,
    *,
    timestamp_ns: int,
    expected_generation: int,
    output: ArtifactReference,
    result: ArtifactReference,
    checkpoint: ArtifactReference | None = None,
    external_root: str | Path,
    checkout_root: str | Path,
) -> tuple[CampaignManifest, AttemptRecord]:
    """Append an immutable successful terminal record."""

    return _advance_attempt(
        authority,
        run_id,
        attempt_number,
        timestamp_ns=timestamp_ns,
        expected_generation=expected_generation,
        status="completed",
        checkpoint=checkpoint,
        output=output,
        result=result,
        failure=None,
        external_root=external_root,
        checkout_root=checkout_root,
    )


def fail_campaign_attempt(
    authority: CampaignAuthority,
    run_id: str,
    attempt_number: int,
    *,
    timestamp_ns: int,
    expected_generation: int,
    failure: str,
    checkpoint: ArtifactReference | None = None,
    external_root: str | Path,
    checkout_root: str | Path,
) -> tuple[CampaignManifest, AttemptRecord]:
    """Durably append a failed terminal record without altering prior records."""

    return _advance_attempt(
        authority,
        run_id,
        attempt_number,
        timestamp_ns=timestamp_ns,
        expected_generation=expected_generation,
        status="failed",
        checkpoint=checkpoint,
        output=None,
        result=None,
        failure=failure,
        external_root=external_root,
        checkout_root=checkout_root,
    )


__all__ = [
    "ArtifactReference",
    "AttemptIndexEntry",
    "AttemptRecord",
    "CampaignAuthority",
    "CampaignManifest",
    "CampaignManifestError",
    "CampaignManifestLockedError",
    "CampaignRunState",
    "InProgressAttemptError",
    "RecordReference",
    "ReconciliationRequiredError",
    "StaleManifestGenerationError",
    "attempt_record_canonical_bytes",
    "attempt_record_fingerprint",
    "campaign_manifest_canonical_bytes",
    "campaign_manifest_fingerprint",
    "campaign_manifest_path",
    "campaign_manifest_sha256",
    "complete_campaign_attempt",
    "fail_campaign_attempt",
    "heartbeat_campaign_attempt",
    "initialize_campaign_manifest",
    "load_campaign_manifest",
    "load_campaign_attempt_record",
    "make_artifact_reference",
    "reconcile_campaign_manifest",
    "start_campaign_attempt",
]
