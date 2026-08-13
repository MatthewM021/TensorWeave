from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

import tnlm_v3.campaign_manifest as manifest_module
from tnlm_v3.campaign_config import (
    campaign_plan_sha256,
    load_milestone4_campaign_config,
    resolve_campaign_plan,
)
from tnlm_v3.campaign_manifest import (
    ArtifactReference,
    CampaignAuthority,
    CampaignManifestError,
    InProgressAttemptError,
    ReconciliationRequiredError,
    StaleManifestGenerationError,
    attempt_record_canonical_bytes,
    attempt_record_fingerprint,
    campaign_manifest_canonical_bytes,
    campaign_manifest_fingerprint,
    campaign_manifest_sha256,
    complete_campaign_attempt,
    fail_campaign_attempt,
    heartbeat_campaign_attempt,
    initialize_campaign_manifest,
    load_campaign_manifest,
    make_artifact_reference,
    reconcile_campaign_manifest,
    start_campaign_attempt,
)
from test_campaign_config import base_document


COMMIT = "1" * 40
TREE = "2" * 40
RAW_CONFIG_SHA = "a" * 64
BUNDLE_SHA = "b" * 64


@pytest.fixture
def campaign(tmp_path: Path) -> tuple[CampaignAuthority, Path, Path]:
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(base_document("pilot")), encoding="utf-8")
    config = load_milestone4_campaign_config(config_path)
    plan = resolve_campaign_plan(
        config,
        COMMIT,
        TREE,
        RAW_CONFIG_SHA,
        BUNDLE_SHA,
    )
    authority = CampaignAuthority(
        config=config,
        resolved_plan=plan,
        plan_sha256=campaign_plan_sha256(config, plan),
        code_commit=COMMIT,
        code_tree=TREE,
        raw_config_sha256=RAW_CONFIG_SHA,
        executable_bundle_sha256=BUNDLE_SHA,
        working_tree_clean=True,
    )
    external = tmp_path / "external"
    checkout = tmp_path / "checkout"
    external.mkdir()
    checkout.mkdir()
    return authority, external.resolve(), checkout.resolve()


def roots(external: Path, checkout: Path) -> dict[str, Path]:
    return {"external_root": external, "checkout_root": checkout}


def write_artifact(external: Path, name: str, raw: bytes) -> str:
    path = external / "artifacts" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(raw)
    return path.relative_to(external).as_posix()


def manifest_path(authority: CampaignAuthority, external: Path) -> Path:
    return external / "campaigns" / authority.config.campaign_id / "manifest.json"


def canonical_write(path: Path, value: object) -> None:
    write_bytes(
        path,
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )


def read_bytes(path: Path) -> bytes:
    with open(manifest_module._system_path(path), "rb") as handle:
        return handle.read()


def write_bytes(path: Path, raw: bytes) -> None:
    with open(manifest_module._system_path(path), "wb") as handle:
        handle.write(raw)


def test_complete_lifecycle_is_canonical_and_content_addressed(
    campaign: tuple[CampaignAuthority, Path, Path],
) -> None:
    authority, external, checkout = campaign
    options = roots(external, checkout)
    manifest = initialize_campaign_manifest(authority, **options)
    assert manifest.generation == 0
    assert len(manifest.runs) == 7
    run_id = manifest.runs[0].run_id

    manifest, start = start_campaign_attempt(
        authority,
        run_id,
        timestamp_ns=10,
        expected_generation=0,
        **options,
    )
    assert manifest.generation == 1
    assert start.status == "in_progress"
    assert attempt_record_fingerprint(start) == attempt_record_fingerprint(start)
    assert json.loads(attempt_record_canonical_bytes(start))["run_id"] == run_id

    checkpoint_path = write_artifact(external, "checkpoint.bin", b"checkpoint")
    checkpoint = make_artifact_reference(checkpoint_path, **options)
    manifest, heartbeat = heartbeat_campaign_attempt(
        authority,
        run_id,
        1,
        timestamp_ns=20,
        expected_generation=1,
        checkpoint=checkpoint,
        **options,
    )
    assert manifest.generation == 2
    assert heartbeat.revision == 1
    assert heartbeat.checkpoint == checkpoint

    output = make_artifact_reference(
        write_artifact(external, "model.bin", b"model"), **options
    )
    result = make_artifact_reference(
        write_artifact(external, "result.json", b'{"accuracy":1}'), **options
    )
    manifest, completed = complete_campaign_attempt(
        authority,
        run_id,
        1,
        timestamp_ns=30,
        expected_generation=2,
        output=output,
        result=result,
        **options,
    )
    assert completed.status == "completed"
    assert completed.checkpoint == checkpoint
    assert load_campaign_manifest(authority, **options) == manifest
    assert reconcile_campaign_manifest(authority, **options) == manifest
    raw = manifest_path(authority, external).read_bytes()
    assert raw == campaign_manifest_canonical_bytes(manifest)
    assert hashlib.sha256(raw).hexdigest() == campaign_manifest_sha256(manifest)
    assert campaign_manifest_fingerprint(manifest) == campaign_manifest_fingerprint(
        manifest
    )
    records = list((external / "campaigns" / "m4-pilot" / "records").rglob("*.json"))
    assert len(records) == 3
    assert len({read_bytes(path) for path in records}) == 3


def test_failure_is_durable_and_retry_counter_is_monotonic(
    campaign: tuple[CampaignAuthority, Path, Path],
) -> None:
    authority, external, checkout = campaign
    options = roots(external, checkout)
    manifest = initialize_campaign_manifest(authority, **options)
    run_id = manifest.runs[0].run_id
    manifest, _ = start_campaign_attempt(
        authority, run_id, timestamp_ns=1, expected_generation=0, **options
    )
    manifest, failed = fail_campaign_attempt(
        authority,
        run_id,
        1,
        timestamp_ns=2,
        expected_generation=1,
        failure="worker exited before evaluation",
        **options,
    )
    assert failed.status == "failed"
    manifest, retried = start_campaign_attempt(
        authority,
        run_id,
        timestamp_ns=3,
        expected_generation=2,
        **options,
    )
    assert retried.attempt_number == 2
    with pytest.raises(CampaignManifestError, match="live or completed"):
        start_campaign_attempt(
            authority,
            run_id,
            timestamp_ns=4,
            expected_generation=3,
            **options,
        )
    manifest, _ = fail_campaign_attempt(
        authority,
        run_id,
        2,
        timestamp_ns=5,
        expected_generation=3,
        failure="explicitly abandoned after recovery",
        **options,
    )
    state = next(run for run in manifest.runs if run.run_id == run_id)
    assert state.next_attempt == 3
    assert [item.status for item in state.attempts] == ["failed", "failed"]


def test_stale_generation_is_compare_and_swap_safe(
    campaign: tuple[CampaignAuthority, Path, Path],
) -> None:
    authority, external, checkout = campaign
    options = roots(external, checkout)
    manifest = initialize_campaign_manifest(authority, **options)
    first, second = manifest.runs[:2]
    start_campaign_attempt(
        authority,
        first.run_id,
        timestamp_ns=1,
        expected_generation=0,
        **options,
    )
    with pytest.raises(StaleManifestGenerationError):
        start_campaign_attempt(
            authority,
            second.run_id,
            timestamp_ns=1,
            expected_generation=0,
            **options,
        )
    loaded = load_campaign_manifest(authority, **options)
    assert loaded.generation == 1
    assert not next(run for run in loaded.runs if run.run_id == second.run_id).attempts


def test_content_addressed_records_work_beyond_legacy_windows_path_limit(
    campaign: tuple[CampaignAuthority, Path, Path],
) -> None:
    authority, external, checkout = campaign
    long_external = external.parent / "long-output-root"
    while len(str(long_external.absolute())) < 215:
        long_external = long_external / "bounded-segment-1234"
    long_external.mkdir(parents=True)
    options = roots(long_external.resolve(), checkout)
    manifest = initialize_campaign_manifest(authority, **options)
    run_id = manifest.runs[0].run_id
    manifest, _ = start_campaign_attempt(
        authority,
        run_id,
        timestamp_ns=1,
        expected_generation=0,
        **options,
    )
    record_path = long_external / manifest.runs[0].attempts[0].record.path
    assert len(str(record_path.absolute())) > 260
    manifest, _ = fail_campaign_attempt(
        authority,
        run_id,
        1,
        timestamp_ns=2,
        expected_generation=1,
        failure="long path exercise",
        **options,
    )
    assert load_campaign_manifest(authority, **options) == manifest


def test_crash_after_record_before_manifest_is_recovered(
    campaign: tuple[CampaignAuthority, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, external, checkout = campaign
    options = roots(external, checkout)
    manifest = initialize_campaign_manifest(authority, **options)
    run_id = manifest.runs[0].run_id
    original_write = manifest_module._write_manifest

    def crash_once(updated, auth, resolved_roots):  # type: ignore[no-untyped-def]
        if updated.generation == 1:
            raise OSError("simulated power loss")
        return original_write(updated, auth, resolved_roots)

    monkeypatch.setattr(manifest_module, "_write_manifest", crash_once)
    with pytest.raises(OSError, match="power loss"):
        start_campaign_attempt(
            authority,
            run_id,
            timestamp_ns=10,
            expected_generation=0,
            **options,
        )
    monkeypatch.setattr(manifest_module, "_write_manifest", original_write)
    assert json.loads(manifest_path(authority, external).read_bytes())["generation"] == 0
    with pytest.raises(ReconciliationRequiredError):
        load_campaign_manifest(authority, **options)
    with pytest.raises(InProgressAttemptError):
        reconcile_campaign_manifest(authority, **options)
    assert json.loads(manifest_path(authority, external).read_bytes())["generation"] == 1
    final, _ = fail_campaign_attempt(
        authority,
        run_id,
        1,
        timestamp_ns=11,
        expected_generation=1,
        failure="recovered orphaned live attempt",
        **options,
    )
    assert final.generation == 2
    assert reconcile_campaign_manifest(authority, **options) == final


def test_tampered_record_and_artifact_are_rejected(
    campaign: tuple[CampaignAuthority, Path, Path],
) -> None:
    authority, external, checkout = campaign
    options = roots(external, checkout)
    manifest = initialize_campaign_manifest(authority, **options)
    run_id = manifest.runs[0].run_id
    manifest, _ = start_campaign_attempt(
        authority, run_id, timestamp_ns=1, expected_generation=0, **options
    )
    artifact_path = write_artifact(external, "checkpoint.bin", b"good")
    reference = make_artifact_reference(artifact_path, **options)
    manifest, _ = fail_campaign_attempt(
        authority,
        run_id,
        1,
        timestamp_ns=2,
        expected_generation=1,
        checkpoint=reference,
        failure="controlled failure",
        **options,
    )
    (external / artifact_path).write_bytes(b"changed")
    with pytest.raises(CampaignManifestError, match="no longer matches"):
        load_campaign_manifest(authority, **options)

    (external / artifact_path).write_bytes(b"good")
    record_path = external / manifest.runs[0].attempts[-1].record.path
    write_bytes(record_path, read_bytes(record_path) + b" ")
    with pytest.raises(CampaignManifestError):
        load_campaign_manifest(authority, **options)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":1,"schema_version":1}',
        b'{"schema_version":NaN}',
        b'{ "schema_version":1}',
        b"not-json",
    ],
)
def test_malformed_duplicate_nonfinite_and_noncanonical_manifest_json_are_rejected(
    campaign: tuple[CampaignAuthority, Path, Path],
    raw: bytes,
) -> None:
    authority, external, checkout = campaign
    options = roots(external, checkout)
    initialize_campaign_manifest(authority, **options)
    manifest_path(authority, external).write_bytes(raw)
    with pytest.raises(CampaignManifestError):
        load_campaign_manifest(authority, **options)


def test_oversize_manifest_is_rejected_before_parsing(
    campaign: tuple[CampaignAuthority, Path, Path],
) -> None:
    authority, external, checkout = campaign
    options = roots(external, checkout)
    initialize_campaign_manifest(authority, **options)
    manifest_path(authority, external).write_bytes(b" " * (4 * 1024 * 1024 + 1))
    with pytest.raises(CampaignManifestError, match="byte limit"):
        load_campaign_manifest(authority, **options)


def test_missing_axis_and_duplicate_attempt_index_are_rejected(
    campaign: tuple[CampaignAuthority, Path, Path],
) -> None:
    authority, external, checkout = campaign
    options = roots(external, checkout)
    manifest = initialize_campaign_manifest(authority, **options)
    path = manifest_path(authority, external)
    value = json.loads(path.read_bytes())
    value["runs"] = value["runs"][1:]
    canonical_write(path, value)
    with pytest.raises(CampaignManifestError, match="axis"):
        load_campaign_manifest(authority, **options)

    path.write_bytes(campaign_manifest_canonical_bytes(manifest))
    run_id = manifest.runs[0].run_id
    manifest, _ = start_campaign_attempt(
        authority, run_id, timestamp_ns=1, expected_generation=0, **options
    )
    value = json.loads(path.read_bytes())
    target = next(item for item in value["runs"] if item["run_id"] == run_id)
    target["attempts"].append(target["attempts"][0])
    target["next_attempt"] = 3
    canonical_write(path, value)
    with pytest.raises(CampaignManifestError, match="manifest values"):
        load_campaign_manifest(authority, **options)


def test_changed_plan_code_tree_config_or_bundle_is_rejected(
    campaign: tuple[CampaignAuthority, Path, Path],
) -> None:
    authority, external, checkout = campaign
    options = roots(external, checkout)
    initialize_campaign_manifest(authority, **options)
    changed_commit = "3" * 40
    changed_tree = "4" * 40
    changed_raw = "c" * 64
    changed_bundle = "d" * 64
    plan = resolve_campaign_plan(
        authority.config,
        changed_commit,
        changed_tree,
        changed_raw,
        changed_bundle,
    )
    changed = CampaignAuthority(
        config=authority.config,
        resolved_plan=plan,
        plan_sha256=campaign_plan_sha256(authority.config, plan),
        code_commit=changed_commit,
        code_tree=changed_tree,
        raw_config_sha256=changed_raw,
        executable_bundle_sha256=changed_bundle,
        working_tree_clean=True,
    )
    with pytest.raises(CampaignManifestError, match="changed"):
        load_campaign_manifest(changed, **options)


@pytest.mark.parametrize(
    "path",
    [
        "../outside.bin",
        "artifacts/../../outside.bin",
        "C:/outside.bin",
        "//server/share/file.bin",
        "\\\\server\\share\\file.bin",
        "/absolute/file.bin",
        "artifacts\\windows.bin",
    ],
)
def test_artifact_path_traversal_windows_and_unc_are_rejected(path: str) -> None:
    with pytest.raises(ValueError):
        ArtifactReference(path, "a" * 64, 1)


def test_external_root_must_be_disjoint_from_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    external = checkout / "outputs"
    checkout.mkdir()
    external.mkdir()
    with pytest.raises(CampaignManifestError, match="disjoint"):
        make_artifact_reference(
            "missing.bin", external_root=external, checkout_root=checkout
        )


@pytest.mark.parametrize(
    "path",
    [
        "campaigns/m4-pilot/manifest.json",
        "campaigns/m4-pilot/manifest.lock",
        "campaigns/m4-pilot/records/aa/record.json",
        "result.json",
    ],
)
def test_artifact_references_cannot_target_campaign_control_namespace(
    path: str,
) -> None:
    with pytest.raises(ValueError, match="artifacts"):
        ArtifactReference(path, "a" * 64, 1)


def test_symlink_and_hardlink_artifacts_are_rejected(
    campaign: tuple[CampaignAuthority, Path, Path],
) -> None:
    _, external, checkout = campaign
    options = roots(external, checkout)
    artifact_directory = external / "artifacts"
    artifact_directory.mkdir()
    target = artifact_directory / "target.bin"
    target.write_bytes(b"bytes")
    hardlink = artifact_directory / "hardlink.bin"
    try:
        os.link(target, hardlink)
    except OSError as error:
        pytest.skip(f"hardlinks are unavailable: {error}")
    with pytest.raises(CampaignManifestError, match="private"):
        make_artifact_reference("artifacts/target.bin", **options)
    hardlink.unlink()
    link = artifact_directory / "symlink.bin"
    try:
        link.symlink_to(target)
    except OSError:
        return
    with pytest.raises(CampaignManifestError):
        make_artifact_reference("artifacts/symlink.bin", **options)


def test_hardlinked_manifest_is_rejected(
    campaign: tuple[CampaignAuthority, Path, Path],
) -> None:
    authority, external, checkout = campaign
    options = roots(external, checkout)
    initialize_campaign_manifest(authority, **options)
    linked = external / "manifest-copy.json"
    try:
        os.link(manifest_path(authority, external), linked)
    except OSError as error:
        pytest.skip(f"hardlinks are unavailable: {error}")
    with pytest.raises(CampaignManifestError, match="non-linked"):
        load_campaign_manifest(authority, **options)


def test_duplicate_or_competing_record_is_ambiguous(
    campaign: tuple[CampaignAuthority, Path, Path],
) -> None:
    authority, external, checkout = campaign
    options = roots(external, checkout)
    manifest = initialize_campaign_manifest(authority, **options)
    run_id = manifest.runs[0].run_id
    manifest, _ = start_campaign_attempt(
        authority, run_id, timestamp_ns=10, expected_generation=0, **options
    )
    original_path = external / manifest.runs[0].attempts[0].record.path
    value = json.loads(read_bytes(original_path))
    value["started_at_ns"] = 11
    value["updated_at_ns"] = 11
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    duplicate_path = original_path.parents[1] / digest[:2] / f"{digest}.json"
    duplicate_path.parent.mkdir(exist_ok=True)
    write_bytes(duplicate_path, raw)
    with pytest.raises(CampaignManifestError, match="duplicate|multiple"):
        load_campaign_manifest(authority, **options)


def test_backwards_timestamp_and_terminal_transition_leave_last_good_state(
    campaign: tuple[CampaignAuthority, Path, Path],
) -> None:
    authority, external, checkout = campaign
    options = roots(external, checkout)
    manifest = initialize_campaign_manifest(authority, **options)
    run_id = manifest.runs[0].run_id
    manifest, _ = start_campaign_attempt(
        authority, run_id, timestamp_ns=10, expected_generation=0, **options
    )
    with pytest.raises(CampaignManifestError, match="backwards"):
        heartbeat_campaign_attempt(
            authority,
            run_id,
            1,
            timestamp_ns=9,
            expected_generation=1,
            **options,
        )
    assert load_campaign_manifest(authority, **options) == manifest
    manifest, _ = fail_campaign_attempt(
        authority,
        run_id,
        1,
        timestamp_ns=11,
        expected_generation=1,
        failure="done",
        **options,
    )
    with pytest.raises(CampaignManifestError, match="terminal"):
        heartbeat_campaign_attempt(
            authority,
            run_id,
            1,
            timestamp_ns=12,
            expected_generation=2,
            **options,
        )
    assert load_campaign_manifest(authority, **options) == manifest


def test_authority_refuses_dirty_attestation_and_plan_reordering(
    campaign: tuple[CampaignAuthority, Path, Path],
) -> None:
    authority, _, _ = campaign
    with pytest.raises(ValueError, match="literal true"):
        replace(authority, working_tree_clean=False)
    with pytest.raises(ValueError, match="exact ordered"):
        replace(authority, resolved_plan=tuple(reversed(authority.resolved_plan)))


def test_authority_rejects_forged_nested_frozen_specs(
    campaign: tuple[CampaignAuthority, Path, Path],
) -> None:
    authority, _, _ = campaign
    object.__setattr__(authority.config.training, "optimizer_steps", 0)
    with pytest.raises(ValueError, match="signed|at least"):
        authority.__post_init__()


@pytest.mark.parametrize("terminal", ["heartbeat", "failed", "completed"])
def test_reconciliation_recovers_crash_tail_after_later_transitions(
    campaign: tuple[CampaignAuthority, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    terminal: str,
) -> None:
    authority, external, checkout = campaign
    options = roots(external, checkout)
    manifest = initialize_campaign_manifest(authority, **options)
    run_id = manifest.runs[0].run_id
    manifest, _ = start_campaign_attempt(
        authority, run_id, timestamp_ns=10, expected_generation=0, **options
    )
    if terminal == "heartbeat":
        manifest, _ = heartbeat_campaign_attempt(
            authority,
            run_id,
            1,
            timestamp_ns=20,
            expected_generation=1,
            **options,
        )
        crash_generation = 3
        transition = lambda: heartbeat_campaign_attempt(  # noqa: E731
            authority,
            run_id,
            1,
            timestamp_ns=30,
            expected_generation=2,
            **options,
        )
    elif terminal == "failed":
        crash_generation = 2
        transition = lambda: fail_campaign_attempt(  # noqa: E731
            authority,
            run_id,
            1,
            timestamp_ns=20,
            expected_generation=1,
            failure="terminal failure",
            **options,
        )
    else:
        crash_generation = 2
        output = make_artifact_reference(
            write_artifact(external, "completed-model.bin", b"model"), **options
        )
        result = make_artifact_reference(
            write_artifact(external, "completed-result.json", b"{}"), **options
        )
        transition = lambda: complete_campaign_attempt(  # noqa: E731
            authority,
            run_id,
            1,
            timestamp_ns=20,
            expected_generation=1,
            output=output,
            result=result,
            **options,
        )
    original_write = manifest_module._write_manifest

    def crash_once(updated, auth, resolved_roots):  # type: ignore[no-untyped-def]
        if updated.generation == crash_generation:
            raise OSError("simulated crash tail")
        return original_write(updated, auth, resolved_roots)

    monkeypatch.setattr(manifest_module, "_write_manifest", crash_once)
    with pytest.raises(OSError, match="crash tail"):
        transition()
    monkeypatch.setattr(manifest_module, "_write_manifest", original_write)
    if terminal == "heartbeat":
        with pytest.raises(InProgressAttemptError):
            reconcile_campaign_manifest(authority, **options)
        recovered = load_campaign_manifest(authority, **options)
        assert recovered.generation == 3
        recovered, _ = fail_campaign_attempt(
            authority,
            run_id,
            1,
            timestamp_ns=31,
            expected_generation=3,
            failure="explicitly abandoned",
            **options,
        )
    else:
        recovered = reconcile_campaign_manifest(authority, **options)
        assert recovered.generation == 2
    assert reconcile_campaign_manifest(authority, **options) == recovered
