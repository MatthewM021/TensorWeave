from __future__ import annotations

from dataclasses import fields, replace
from hashlib import sha256
import json
import os
from pathlib import Path

import pytest

import tnlm_v3.opaque_active_discovery as discovery_module
import tnlm_v3.opaque_active_discovery_protocol as protocol_module
from tnlm_v3.opaque_active_discovery_protocol import (
    MerkleProof,
    Phase3T2Protocol,
    T2ProtocolError,
    build_t2_controller_environment,
    load_phase3_t2_protocol,
    load_t2_campaign_report,
    make_placeholder_protocol,
    make_source_runtime_binding_from_repository,
    open_t2_campaign_postfit_batch,
    reconstruct_t2_campaign,
    require_execution_ready,
    run_t2_preopen_environment,
    write_t2_campaign_report,
)
from tnlm_v3.opaque_partial_operators import EnvironmentKind


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "v3" / "configs" / "phase3" / "opaque_active_discovery_t2.json"
DEVELOPMENT_CAMPAIGN_ENV = "TNLM_RUN_SLOW_PHASE3_T2_PROTOCOL"

# These are already-opened deterministic fixtures, deliberately reused only
# for tests.  They are not prospective entropy and must never enter an
# execution-ready scientific config or official evidence record.
DEVELOPMENT_CONTROLLER_NONCES = (
    "68e1a01aa80a042b82e04f36dfe696a91f4ffd9f82d5f311a027d73adbd2a9a5",
    "a57aebc4843c69b6d5fdff38d49523b3dedbf7cf55ccdd822745f7e14c14304e",
    "323b4597f9e3dfb9e7f954aab5873b8730321dd8476a2992f81e93cc7a44b046",
    "45d4ff9e973a3c9fda9a41ae7ab8c827f9e7299a2ece6f402ab32dc0fd598718",
    "a3d1591df9b8fe82666f78e885f578cfd5677df28dfa6bc2daa5cfa468d87203",
    "0ec845c35cfe807b28f8d012391d9e811211a780e3fd22510ad33a6059dcf5fc",
    "422d894dade5755a90355dea9dec66be8a8943509365c79727046eb895042bb4",
    "be187a3a14f4d575944e93a407f0e1c53142a9db4c27abc0dcb6bbb3d7712df3",
    "219803a3b13f67dbb90f23749b785d71836918b7d5d0cee78d74d85ff0319490",
    "8e9f16318b215a2e117e34bc75cbc649b602dac9d8ce5ffcdace6950c8971ce2",
)


def _development_salts() -> tuple[tuple[str, ...], ...]:
    rows = tuple(
        tuple(
            sha256(
                (
                    "development-only-phase3-t2-salt:"
                    f"omission-{omission_index}:candidate-{candidate_index}"
                ).encode("ascii")
            ).hexdigest()
            for candidate_index in range(23)
        )
        for omission_index in range(8)
    )
    assert len({salt for row in rows for salt in row}) == 8 * 23
    return rows


def _build_development_ready_protocol() -> Phase3T2Protocol:
    """Make a source-bound ready protocol from non-evidence test fixtures."""

    budgets = make_placeholder_protocol().active_budgets
    salts = _development_salts()
    preview_environments = []
    for index, (kind, block, cell) in enumerate(protocol_module._SCHEDULE):
        preview_environments.append(
            build_t2_controller_environment(
                kind=kind,
                relabel_block=block,
                omitted_cell=cell,
                controller_nonce=DEVELOPMENT_CONTROLLER_NONCES[index],
                active_budgets=budgets,
                salts=None if index < 2 else salts[index - 2],
            )
        )
    commitment_roots = tuple(
        environment.commitment.merkle_root_sha256
        for environment in preview_environments[2:]
        if environment.commitment is not None
    )
    long_roots = tuple(
        environment.long_suite_root_sha256
        for environment in preview_environments
    )
    binding = make_source_runtime_binding_from_repository(ROOT)
    payload = {
        "schema": protocol_module.PROTOCOL_SCHEMA,
        "execution_ready": True,
        "schedule_labels": list(protocol_module._SCHEDULE_LABELS),
        "controller_nonces": list(DEVELOPMENT_CONTROLLER_NONCES),
        "omission_salts": [list(row) for row in salts],
        "omission_commitment_roots": list(commitment_roots),
        "long_suite_roots": list(long_roots),
        "source_runtime_binding": binding.payload(),
        "active_budgets": budgets.payload(),
        "min_unopened_candidates": 8,
    }
    return Phase3T2Protocol(
        execution_ready=True,
        schedule_labels=protocol_module._SCHEDULE_LABELS,
        controller_nonces=DEVELOPMENT_CONTROLLER_NONCES,
        omission_salts=salts,
        omission_commitment_roots=commitment_roots,
        long_suite_roots=long_roots,
        source_runtime_binding=binding,
        active_budgets=budgets,
        min_unopened_candidates=8,
        protocol_sha256=protocol_module._digest(payload),
    )


@pytest.fixture(scope="module")
def development_campaign():
    if os.environ.get(DEVELOPMENT_CAMPAIGN_ENV) != "1":
        pytest.skip(
            f"set {DEVELOPMENT_CAMPAIGN_ENV}=1 for the ten-arm development rehearsal"
        )
    protocol = _build_development_ready_protocol()
    captured: dict[str, object] = {}
    real_opener = protocol_module.open_t2_campaign_postfit_batch
    real_postfit = protocol_module._open_t2_postfit_environment
    real_teaching = protocol_module.discover_postfit_teaching_control
    real_shortcut = protocol_module._evaluate_t2_shortcut_evidence
    phase_log: list[str] = []

    def logged_postfit(*args, **kwargs):
        phase_log.append("postfit")
        return real_postfit(*args, **kwargs)

    def logged_teaching(*args, **kwargs):
        assert phase_log.count("postfit") == 8
        phase_log.append("teaching")
        return real_teaching(*args, **kwargs)

    def logged_shortcut(*args, **kwargs):
        assert phase_log.count("postfit") == 8
        phase_log.append("shortcut")
        return real_shortcut(*args, **kwargs)

    def capturing_opener(protocol_arg, environments, preopens, terminal):
        captured["environments"] = tuple(environments)
        captured["preopens"] = tuple(preopens)
        captured["terminal"] = terminal
        return real_opener(protocol_arg, environments, preopens, terminal)

    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        protocol_module, "open_t2_campaign_postfit_batch", capturing_opener
    )
    patcher.setattr(protocol_module, "_open_t2_postfit_environment", logged_postfit)
    patcher.setattr(protocol_module, "discover_postfit_teaching_control", logged_teaching)
    patcher.setattr(protocol_module, "_evaluate_t2_shortcut_evidence", logged_shortcut)
    try:
        report = protocol_module.run_t2_campaign(protocol, ROOT)
    finally:
        patcher.undo()
    assert phase_log[:8] == ["postfit"] * 8
    assert phase_log.count("teaching") == 8
    assert phase_log.count("shortcut") == 10
    return (
        protocol,
        captured["environments"],
        captured["preopens"],
        captured["terminal"],
        report,
    )


def _omission_environment():
    protocol = make_placeholder_protocol()
    return build_t2_controller_environment(
        kind=EnvironmentKind.ROTATED_OMISSION,
        relabel_block=0,
        omitted_cell=(0, 0),
        controller_nonce=DEVELOPMENT_CONTROLLER_NONCES[2],
        active_budgets=protocol.active_budgets,
        salts=_development_salts()[0],
    )


def test_official_config_is_execution_ready_and_source_bound() -> None:
    protocol = load_phase3_t2_protocol(CONFIG)
    assert protocol.execution_ready
    assert protocol.protocol_sha256 == (
        "7c5ee8bcee72e0af5ac2d8404f54b479e1b7d1b1200922ec40caf66483c04292"
    )
    assert sha256(CONFIG.read_bytes()).hexdigest() == (
        "481287f7390ff73a52b9d1c9c9e48a9971eedaad3882aaba41fd3a9ad48b337d"
    )
    assert protocol.source_runtime_binding is not None
    assert protocol.source_runtime_binding.binding_sha256 == (
        "514ebb445d3eb00e456095bf3377bf4f7eb2e15a4282e0765ca307b5203e5e90"
    )
    require_execution_ready(protocol, ROOT)


def test_public_campaign_runner_fails_unready_before_any_controller(
    monkeypatch,
) -> None:
    protocol = make_placeholder_protocol()
    controller_calls: list[str] = []

    def forbidden_controller(*_args, **_kwargs):
        controller_calls.append("built")
        raise AssertionError("unready protocol reached controller construction")

    monkeypatch.setattr(
        protocol_module,
        "build_t2_controller_environment",
        forbidden_controller,
    )
    with pytest.raises(T2ProtocolError, match="not execution-ready"):
        protocol_module.run_t2_campaign(protocol, ROOT)
    assert controller_calls == []


def test_certificate_cap_preflight_fails_before_response_provider() -> None:
    protocol = make_placeholder_protocol()
    tight_budgets = replace(
        protocol.active_budgets,
        max_certificate_bytes=7_999_999,
    )
    environment = build_t2_controller_environment(
        kind=EnvironmentKind.ROTATED_OMISSION,
        relabel_block=0,
        omitted_cell=(0, 0),
        controller_nonce=DEVELOPMENT_CONTROLLER_NONCES[2],
        active_budgets=tight_budgets,
        salts=_development_salts()[0],
    )
    provider_calls: list[str] = []

    def forbidden_provider(choice):
        provider_calls.append(choice.choice_sha256)
        raise AssertionError("certificate-cap preflight reached the answer provider")

    with pytest.raises(
        discovery_module.OpaqueActiveDiscoveryLimitError,
        match="conservative upper bound",
    ):
        discovery_module.run_opaque_active_discovery(
            environment.learner_input,
            forbidden_provider,
        )
    assert provider_calls == []


def test_public_batch_shape_precheck_precedes_every_postfit_sidecar(
    monkeypatch,
) -> None:
    sidecar_calls: list[str] = []

    def forbidden_sidecar(*_args, **_kwargs):
        sidecar_calls.append("opened")
        raise AssertionError("invalid batch opened a sidecar")

    monkeypatch.setattr(
        protocol_module, "_open_t2_postfit_environment", forbidden_sidecar
    )
    with pytest.raises(ValueError, match="all ten scheduled arms"):
        open_t2_campaign_postfit_batch(
            make_placeholder_protocol(),
            (),
            (),
            None,
        )
    assert sidecar_calls == []


def test_commitment_stays_outside_fresh_learner_boundary() -> None:
    environment = _omission_environment()
    assert environment.commitment is not None
    learner_bytes = repr(environment.learner_input.payload())
    assert environment.controller.trusted_controller_nonce not in learner_bytes
    assert environment.commitment.merkle_root_sha256 not in learner_bytes
    assert all(leaf.salt not in learner_bytes for leaf in environment.commitment.leaves)
    request = environment.learner_input.canonical_candidate_requests[0]
    answer = environment.commitment.leaves[0].target_answers
    environment.commitment.verify_opening(request, answer)
    proof = environment.commitment.proof_for(request.request_sha256)
    environment.commitment.verify_proof(environment.commitment.leaves[0].leaf_sha256, proof)
    with pytest.raises(T2ProtocolError, match="inclusion proof"):
        environment.commitment.verify_proof(
            environment.commitment.leaves[0].leaf_sha256,
            MerkleProof(proof.leaf_index, proof.sibling_sha256s[:-1]),
        )
    wrong_answer = next(
        (left, right)
        for left in environment.learner_input.answer_tokens
        for right in environment.learner_input.answer_tokens
        if (left, right) != answer
    )
    with pytest.raises(T2ProtocolError, match="salted commitment leaf"):
        environment.commitment.verify_opening(request, wrong_answer)
    with pytest.raises(T2ProtocolError, match="precomputed commitment root"):
        build_t2_controller_environment(
            kind=EnvironmentKind.ROTATED_OMISSION,
            relabel_block=0,
            omitted_cell=(0, 0),
            controller_nonce=DEVELOPMENT_CONTROLLER_NONCES[2],
            active_budgets=make_placeholder_protocol().active_budgets,
            salts=_development_salts()[0],
            expected_commitment_root_sha256="0" * 64,
        )
    with pytest.raises(T2ProtocolError, match="long-suite root"):
        build_t2_controller_environment(
            kind=EnvironmentKind.ROTATED_OMISSION,
            relabel_block=0,
            omitted_cell=(0, 0),
            controller_nonce=DEVELOPMENT_CONTROLLER_NONCES[2],
            active_budgets=make_placeholder_protocol().active_budgets,
            salts=_development_salts()[0],
            expected_long_suite_root_sha256="0" * 64,
        )


def test_preopen_keeps_exact_acquisition_boundary(monkeypatch) -> None:
    environment = _omission_environment()
    opened_by_sidecar: list[str] = []
    real_single_opener = protocol_module._open_one_committed_candidate_answer

    def logged_single_opener(controller, request):
        opened_by_sidecar.append(request.request_sha256)
        return real_single_opener(controller, request)

    def forbidden_bulk_or_postfit_access(*_args, **_kwargs):
        raise AssertionError("preopen reached a bulk or postfit answer surface")

    monkeypatch.setattr(
        protocol_module,
        "_open_one_committed_candidate_answer",
        logged_single_opener,
    )
    monkeypatch.setattr(
        protocol_module, "_answers_by_candidate", forbidden_bulk_or_postfit_access
    )
    monkeypatch.setattr(
        protocol_module,
        "_open_t2_postfit_environment",
        forbidden_bulk_or_postfit_access,
    )
    monkeypatch.setattr(
        protocol_module,
        "_evaluate_model_against_controller",
        forbidden_bulk_or_postfit_access,
    )
    preopen = run_t2_preopen_environment(environment)
    assert preopen.result.identification_status == "identified"
    assert len(preopen.opened_request_sha256s) == 14
    assert tuple(opened_by_sidecar) == preopen.opened_request_sha256s
    assert preopen.result.active_call_count == 14
    assert preopen.result.structural_inference_count == 1
    inferred = {
        step.inference.request.request_sha256
        for step in preopen.result.final_state.steps
        if hasattr(step, "inference")
    }
    candidate_ids = {
        request.request_sha256
        for request in environment.learner_input.canonical_candidate_requests
    }
    sealed = candidate_ids - set(opened_by_sidecar) - inferred
    assert len(inferred) == 1
    assert len(sealed) == 8
    assert not (set(opened_by_sidecar) & inferred)
    assert not (set(opened_by_sidecar) & sealed)


def test_control_has_no_commitment_or_provider_surface() -> None:
    protocol = make_placeholder_protocol()
    control = build_t2_controller_environment(
        kind=EnvironmentKind.FULL_SUPPORT_CONTROL,
        relabel_block=0,
        omitted_cell=None,
        controller_nonce=DEVELOPMENT_CONTROLLER_NONCES[0],
        active_budgets=protocol.active_budgets,
    )
    assert control.commitment is None
    preopen = run_t2_preopen_environment(control, minimum_unopened_candidates=0)
    assert preopen.result.identification_status == "identified"
    assert preopen.opened_request_sha256s == ()
    evidence = protocol_module._evaluate_t2_shortcut_evidence(
        control,
        preopen,
        (),
    )
    assert not evidence.legal_holdout_applicable
    assert not evidence.t1_first14_applicable
    assert evidence.t1_first14 is None
    assert all(row.heldout_edge_count == 0 for row in evidence.baseline_rows)
    assert all(
        row.undefined_pair_false_accept_count == 46
        for row in evidence.baseline_rows
    )


def _postfit_opening_records(environment, preopen):
    protocol_module._open_t2_postfit_environment(
        environment,
        preopen,
        "1" * 64,
    )
    result = preopen.result
    queried = {
        step.response.request.request_sha256: step
        for step in result.final_state.steps
        if hasattr(step, "response")
    }
    inferred = {
        step.inference.request.request_sha256
        for step in result.final_state.steps
        if hasattr(step, "inference")
    }
    all_requests = {
        request.request_sha256
        for request in environment.learner_input.canonical_candidate_requests
    }
    role_order = (
        list(preopen.opened_request_sha256s)
        + sorted(inferred)
        + sorted(all_requests - set(preopen.opened_request_sha256s) - inferred)
    )
    rows = []
    for ordinal, request_sha256 in enumerate(role_order, 1):
        leaf = next(
            row
            for row in environment.commitment.leaves
            if row.request_sha256 == request_sha256
        )
        proof = environment.commitment.proof_for(request_sha256)
        if request_sha256 in queried:
            step = queried[request_sha256]
            role = "queried"
            choice_sha256 = step.choice.choice_sha256
            response_sha256 = step.response.response_sha256
        elif request_sha256 in inferred:
            role = "inferred"
            choice_sha256 = response_sha256 = "0" * 64
        else:
            role = "sealed"
            choice_sha256 = response_sha256 = "0" * 64
        rows.append(
            protocol_module.T2OpeningRecord(
                role,
                ordinal,
                request_sha256,
                leaf.target_answers,
                leaf.salt,
                proof.leaf_index,
                choice_sha256,
                response_sha256,
                leaf.leaf_sha256,
                proof.sibling_sha256s,
            )
        )
    return tuple(rows)


def test_shortcut_fit_is_visible_only_and_t1_is_actual_nonidentification() -> None:
    environment = _omission_environment()
    preopen = run_t2_preopen_environment(environment)
    opening_records = _postfit_opening_records(environment, preopen)
    evidence = protocol_module._evaluate_t2_shortcut_evidence(
        environment,
        preopen,
        opening_records,
    )
    assert tuple(
        len(row)
        for row in (
            evidence.passive_fit_request_sha256s,
            evidence.primary_visible_fit_request_sha256s,
            evidence.inferred_eval_request_sha256s,
            evidence.sealed_eval_request_sha256s,
        )
    ) == (21, 14, 1, 8)
    assert all(row.heldout_error_count > 0 for row in evidence.baseline_rows)
    assert evidence.t1_first14 is not None
    assert not evidence.t1_first14.identified
    assert sorted(
        count for _, count in evidence.t1_first14.final_event_version_counts
    ) == [1] * 9 + [9]

    candidate_by_sha = {
        request.request_sha256: request
        for request in environment.learner_input.canonical_candidate_requests
    }
    opening_by_sha = {row.request_sha256: row for row in opening_records}
    training_rows = tuple(
        (row.request, row.target_answers)
        for row in environment.learner_input.passive_edge_observations
    ) + tuple(
        (
            candidate_by_sha[digest],
            opening_by_sha[digest].target_answers,
        )
        for digest in preopen.opened_request_sha256s
    )
    heldout_records = tuple(
        row for row in opening_records if row.role in ("inferred", "sealed")
    )
    heldout_rows = tuple(
        (candidate_by_sha[row.request_sha256], row.target_answers)
        for row in heldout_records
    )
    known_sources = dict(preopen.result.model.mask_source_answer_rows)
    original = protocol_module._make_shortcut_baseline_evaluation(
        "constant_mode",
        training_rows,
        heldout_rows,
        known_sources,
        environment.learner_input.canonical_undefined_requests,
    )
    alternate = next(
        target for _, target in training_rows if target != heldout_rows[0][1]
    )
    mutated_heldout = ((heldout_rows[0][0], alternate),) + heldout_rows[1:]
    mutated = protocol_module._make_shortcut_baseline_evaluation(
        "constant_mode",
        training_rows,
        mutated_heldout,
        known_sources,
        environment.learner_input.canonical_undefined_requests,
    )
    assert mutated.fit_model_sha256 == original.fit_model_sha256
    assert mutated.fit_error_count == original.fit_error_count

    with pytest.raises(ValueError, match="must remain unidentified"):
        replace(evidence.t1_first14, identified=True)
    with pytest.raises(ValueError, match="disjointly cover 44"):
        replace(
            evidence,
            sealed_eval_request_sha256s=(
                evidence.inferred_eval_request_sha256s[0],
                *evidence.sealed_eval_request_sha256s[1:],
            ),
        )

    canonical_size = len(
        discovery_module._canonical_bytes(preopen.result._payload(True))
    )
    tight_budgets = replace(
        environment.learner_input.budgets,
        max_certificate_bytes=canonical_size - 1,
    )
    tight_environment = build_t2_controller_environment(
        kind=EnvironmentKind.ROTATED_OMISSION,
        relabel_block=0,
        omitted_cell=(0, 0),
        controller_nonce=DEVELOPMENT_CONTROLLER_NONCES[2],
        active_budgets=tight_budgets,
        salts=_development_salts()[0],
    )
    with pytest.raises(
        discovery_module.OpaqueActiveDiscoveryLimitError,
        match="canonical final-result certificate exceeds byte budget",
    ):
        discovery_module._enforce_result_certificate_budget(
            tight_environment.learner_input,
            preopen.result,
        )


def _rehash_arm(arm, **overrides):
    values = {
        field.name: getattr(arm, field.name)
        for field in fields(protocol_module.T2ArmEvaluation)
        if field.name != "arm_sha256"
    }
    values.update(overrides)
    payload = arm.payload(False)
    for name, value in overrides.items():
        if name in ("opening_records", "defined_rows", "undefined_rows", "long_rows"):
            payload[name] = [row.payload() for row in value]
        elif name in ("opened_request_sha256s", "sealed_request_sha256s"):
            payload[name] = list(value)
        elif name in ("shortcut_evidence", "postfit_teaching_summary"):
            payload[name] = None if value is None else value.payload()
        else:
            payload[name] = value
    return protocol_module.T2ArmEvaluation(
        **values,
        arm_sha256=protocol_module._digest(payload),
    )


def _rehash_report(report, arm_records):
    payload = report.payload(False)
    payload["arm_records"] = [row.payload() for row in arm_records]
    return protocol_module.T2CampaignReport(
        protocol_sha256=report.protocol_sha256,
        terminal_sha256=report.terminal_sha256,
        preopen_sha256s=report.preopen_sha256s,
        total_legal_edges_checked=report.total_legal_edges_checked,
        total_undefined_pairs_checked=report.total_undefined_pairs_checked,
        total_long_paths_checked=report.total_long_paths_checked,
        all_shortcuts_fail=report.all_shortcuts_fail,
        arm_records=tuple(arm_records),
        paired_similarities=report.paired_similarities,
        postfit_teaching_summary=report.postfit_teaching_summary,
        report_sha256=protocol_module._digest(payload),
    )


def test_development_ready_ten_arm_campaign_has_exact_bound_evidence(
    development_campaign,
) -> None:
    protocol, environments, preopens, terminal, report = development_campaign
    assert protocol.execution_ready
    assert protocol.schedule_labels == (
        "control:block0",
        "control:block1",
        "omission:block0:cell00",
        "omission:block0:cell01",
        "omission:block0:cell10",
        "omission:block0:cell11",
        "omission:block1:cell00",
        "omission:block1:cell01",
        "omission:block1:cell10",
        "omission:block1:cell11",
    )
    assert tuple(
        (
            environment.controller.kind,
            environment.controller.relabel_block,
            environment.controller.pseudoheldout_cell,
        )
        for environment in environments
    ) == protocol_module._SCHEDULE
    assert tuple(row.schedule_label for row in preopens) == protocol.schedule_labels
    assert terminal.protocol_sha256 == protocol.protocol_sha256
    assert terminal.scheduled_preopen_sha256s == tuple(
        row.preopen_sha256 for row in preopens
    )
    assert report.protocol_sha256 == protocol.protocol_sha256
    assert report.terminal_sha256 == terminal.terminal_sha256
    assert (
        report.total_legal_edges_checked,
        report.total_undefined_pairs_checked,
        report.total_long_paths_checked,
    ) == (440, 460, 120)
    assert report.all_shortcuts_fail

    assert sum(row.result.active_call_count for row in preopens) == 112
    assert sum(row.result.returned_categorical_token_count for row in preopens) == 224
    assert sum(row.result.structural_inference_count for row in preopens) == 8
    assert tuple(row.opened_request_sha256s for row in preopens[:2]) == ((), ())
    assert all(len(row.opened_request_sha256s) == 14 for row in preopens[2:])

    for index, (environment, preopen, arm) in enumerate(
        zip(environments, preopens, report.arm_records, strict=True)
    ):
        assert arm.schedule_label == protocol.schedule_labels[index]
        assert arm.campaign_terminal_sha256 == terminal.terminal_sha256
        assert arm.controller_sha256 == environment.controller.controller_sha256
        assert arm.learner_input_sha256 == environment.learner_input.input_sha256
        assert arm.preopen_sha256 == preopen.preopen_sha256
        assert arm.result_sha256 == preopen.result.result_sha256
        assert arm.model_sha256 == preopen.result.model.model_sha256
        assert arm.final_state_sha256 == preopen.result.final_state.state_sha256
        assert (
            len(arm.defined_rows),
            len(arm.undefined_rows),
            len(arm.long_rows),
        ) == (44, 46, 12)
        assert all(row.exact for row in arm.defined_rows)
        assert all(row.exact for row in arm.undefined_rows)
        assert all(row.exact for row in arm.long_rows)
        assert all(
            row.expected_answers == row.predicted_answers
            for row in arm.defined_rows + arm.long_rows
        )
        assert all(
            row.expected_answers is None and row.predicted_answers is None
            for row in arm.undefined_rows
        )
        assert preopen.result.model.total_operator is None
        assert all(
            operator.total_operator is None
            for operator in preopen.result.model.operators
        )
        shortcut = arm.shortcut_evidence
        assert type(shortcut) is protocol_module.T2ShortcutEvidence
        assert shortcut.learner_input_sha256 == arm.learner_input_sha256
        assert shortcut.primary_result_sha256 == arm.result_sha256
        assert shortcut.all_shortcuts_fail
        assert tuple(row.baseline_kind for row in shortcut.baseline_rows) == (
            "constant_mode",
            "identity",
            "event_mode",
            "source_mode",
        )
        assert all(row.baseline_failed for row in shortcut.baseline_rows)
        assert all(
            row.undefined_pair_false_accept_count == 46
            for row in shortcut.baseline_rows
        )
        split = (
            shortcut.passive_fit_request_sha256s
            + shortcut.primary_visible_fit_request_sha256s
            + shortcut.inferred_eval_request_sha256s
            + shortcut.sealed_eval_request_sha256s
        )
        assert len(split) == len(set(split)) == 44
        assert set(split) == {
            request.request_sha256
            for request in environment.learner_input.canonical_defined_requests
        }

        if index < 2:
            assert arm.commitment_root_sha256 is None
            assert arm.opening_records == ()
            assert arm.inferred_request_sha256 is None
            assert arm.sealed_request_sha256s == ()
            assert arm.postfit_teaching_summary is None
            assert tuple(
                len(row)
                for row in (
                    shortcut.passive_fit_request_sha256s,
                    shortcut.primary_visible_fit_request_sha256s,
                    shortcut.inferred_eval_request_sha256s,
                    shortcut.sealed_eval_request_sha256s,
                )
            ) == (44, 0, 0, 0)
            assert not shortcut.legal_holdout_applicable
            assert not shortcut.t1_first14_applicable
            assert shortcut.t1_first14 is None
            assert all(row.heldout_edge_count == 0 for row in shortcut.baseline_rows)
            continue

        assert environment.commitment is not None
        teaching = arm.postfit_teaching_summary
        assert type(teaching) is protocol_module.T2PostfitTeachingSummary
        assert teaching.learner_input_sha256 == arm.learner_input_sha256
        assert teaching.primary_reconstruction_result_sha256 == arm.result_sha256
        assert (
            teaching.counterfactual_truth_selected_query_count,
            teaching.answer_free_singleton_inference_count,
            teaching.counterfactual_unqueried_count,
            teaching.closed_restricted_map_count,
            teaching.rank_closed_event_count,
            teaching.counterfactual_returned_categorical_label_count,
            teaching.new_membership_calls_made,
        ) == (13, 2, 8, 10, 10, 26, 0)
        assert teaching.causal_primary_isolated
        assert teaching.truth_specific_noncausal_control
        assert not teaching.selection_eligible
        assert not teaching.confirmatory_claim_eligible
        assert not teaching.global_query_minimality_claimed
        assert not teaching.arbitrary_total_operator_constructed
        assert not (
            preopen.result.posthoc_truth_specific_13_query_teaching_set_used_by_selector
        )
        assert arm.commitment_root_sha256 == (
            environment.commitment.merkle_root_sha256
        )
        assert tuple(
            len(row)
            for row in (
                shortcut.passive_fit_request_sha256s,
                shortcut.primary_visible_fit_request_sha256s,
                shortcut.inferred_eval_request_sha256s,
                shortcut.sealed_eval_request_sha256s,
            )
        ) == (21, 14, 1, 8)
        assert shortcut.primary_visible_fit_request_sha256s == preopen.opened_request_sha256s
        assert shortcut.legal_holdout_applicable
        assert shortcut.t1_first14_applicable
        assert all(row.heldout_error_count > 0 for row in shortcut.baseline_rows)
        t1 = shortcut.t1_first14
        assert type(t1) is protocol_module.T2T1First14Nonidentification
        assert t1.ordered_request_sha256s == tuple(
            request.request_sha256
            for request in environment.controller.learner_input.candidate_edge_requests[:14]
        )
        assert tuple(count for _, count in t1.final_event_version_counts).count(1) == 9
        assert sorted(count for _, count in t1.final_event_version_counts) == [1] * 9 + [9]
        assert sum(observed < legal for _, observed, legal in t1.final_event_rank_rows) == 1
        assert t1.posterior_global_version_mass == 9
        assert not t1.identified
        assert t1.exact_restricted_version_evaluation
        assert t1.controller_supplied_postfit_negative_control
        assert len(arm.opening_records) == 23
        assert tuple(row.access_ordinal for row in arm.opening_records) == tuple(
            range(1, 24)
        )
        assert tuple(row.role for row in arm.opening_records) == (
            ("queried",) * 14 + ("inferred",) + ("sealed",) * 8
        )
        assert tuple(
            row.request_sha256
            for row in arm.opening_records
            if row.role == "queried"
        ) == preopen.opened_request_sha256s
        assert tuple(
            row.request_sha256
            for row in arm.opening_records
            if row.role == "inferred"
        ) == (arm.inferred_request_sha256,)
        assert tuple(
            row.request_sha256
            for row in arm.opening_records
            if row.role == "sealed"
        ) == arm.sealed_request_sha256s
        assert (
            set(preopen.opened_request_sha256s)
            | {arm.inferred_request_sha256}
            | set(arm.sealed_request_sha256s)
        ) == {
            request.request_sha256
            for request in environment.learner_input.canonical_candidate_requests
        }
        for opening in arm.opening_records:
            leaf = environment.commitment.leaves[opening.leaf_index]
            assert (
                leaf.request_sha256,
                leaf.target_answers,
                leaf.salt,
                leaf.leaf_sha256,
            ) == (
                opening.request_sha256,
                opening.target_answers,
                opening.salt,
                opening.leaf_sha256,
            )
            environment.commitment.verify_proof(
                opening.leaf_sha256,
                MerkleProof(opening.leaf_index, opening.proof_siblings),
            )

    expected_pair_indices = ((0, 1), (2, 6), (3, 7), (4, 8), (5, 9))
    assert tuple(
        (similarity.left_arm_sha256, similarity.right_arm_sha256)
        for similarity in report.paired_similarities
    ) == tuple(
        (
            report.arm_records[left].arm_sha256,
            report.arm_records[right].arm_sha256,
        )
        for left, right in expected_pair_indices
    )
    for similarity in report.paired_similarities:
        assert similarity.controller_supplied_postfit_alignment
        assert similarity.all_legal_rows_agree_under_alignment
        assert (
            similarity.anchor_count,
            similarity.tested_state_count,
            similarity.tested_legal_edge_count,
        ) == (5, 9, 44)
        assert len(similarity.change_of_basis) == 5
        assert all(len(row) == 5 for row in similarity.change_of_basis)

    campaign_teaching = report.postfit_teaching_summary
    assert type(campaign_teaching) is protocol_module.T2CampaignTeachingSummary
    assert (
        campaign_teaching.omission_control_count,
        campaign_teaching.total_counterfactual_truth_selected_queries,
        campaign_teaching.total_answer_free_singleton_inferences,
        campaign_teaching.total_counterfactual_unqueried,
        campaign_teaching.total_counterfactual_returned_categorical_labels,
        campaign_teaching.total_closed_restricted_maps,
        campaign_teaching.total_rank_closed_events,
        campaign_teaching.total_new_membership_calls,
    ) == (8, 104, 16, 64, 208, 80, 80, 0)
    assert campaign_teaching.all_causal_primary_isolated
    assert campaign_teaching.all_truth_specific_noncausal
    assert campaign_teaching.all_selection_ineligible
    assert campaign_teaching.no_global_query_minimality_claim
    assert campaign_teaching.no_arbitrary_total_operator


def test_public_batch_precheck_rejects_late_schedule_swap_before_sidecar(
    development_campaign,
    monkeypatch,
) -> None:
    protocol, environments, preopens, terminal, _report = development_campaign
    forged_environments = list(environments)
    forged_environments[-2], forged_environments[-1] = (
        forged_environments[-1],
        forged_environments[-2],
    )
    sidecar_calls: list[str] = []

    def forbidden_sidecar(*_args, **_kwargs):
        sidecar_calls.append("opened")
        raise AssertionError("a postfit sidecar opened before public precheck")

    monkeypatch.setattr(
        protocol_module, "_open_t2_postfit_environment", forbidden_sidecar
    )
    with pytest.raises(T2ProtocolError, match="descriptor differs"):
        open_t2_campaign_postfit_batch(
            protocol,
            tuple(forged_environments),
            preopens,
            terminal,
        )
    assert sidecar_calls == []


def test_suffix_cap_precheck_fails_before_first_postfit_sidecar(
    development_campaign,
    monkeypatch,
) -> None:
    protocol, environments, preopens, _terminal, _report = development_campaign
    tight_budgets = replace(
        protocol.active_budgets,
        max_suffix_events_per_prediction=1,
    )
    protocol_payload = protocol.payload(False)
    protocol_payload["active_budgets"] = tight_budgets.payload()
    tight_protocol = replace(
        protocol,
        active_budgets=tight_budgets,
        protocol_sha256=protocol_module._digest(protocol_payload),
    )
    terminal_payload = {
        "schema": protocol_module.TERMINAL_SCHEMA,
        "protocol_sha256": tight_protocol.protocol_sha256,
        "source_runtime_binding_sha256": tight_protocol.source_runtime_binding.binding_sha256,
        "scheduled_preopen_sha256s": [row.preopen_sha256 for row in preopens],
        "all_ten_models_frozen_before_postfit": True,
    }
    tight_terminal = protocol_module.T2CampaignTerminalPreopen(
        tight_protocol.protocol_sha256,
        tight_protocol.source_runtime_binding.binding_sha256,
        tuple(terminal_payload["scheduled_preopen_sha256s"]),
        protocol_module._digest(terminal_payload),
    )
    sidecar_calls: list[str] = []

    def forbidden_sidecar(*_args, **_kwargs):
        sidecar_calls.append("opened")
        raise AssertionError("suffix-cap precheck opened a postfit sidecar")

    monkeypatch.setattr(
        protocol_module,
        "_open_t2_postfit_environment",
        forbidden_sidecar,
    )
    with pytest.raises(T2ProtocolError, match="exceeds frozen suffix prediction cap"):
        open_t2_campaign_postfit_batch(
            tight_protocol,
            environments,
            preopens,
            tight_terminal,
        )
    assert sidecar_calls == []


def test_campaign_disk_roundtrip_reconstruction_and_nested_reseal_rejection(
    development_campaign,
    tmp_path,
    monkeypatch,
) -> None:
    protocol, _environments, _preopens, _terminal, report = development_campaign
    first_path = tmp_path / "development-report-a.json"
    second_path = tmp_path / "development-report-b.json"
    write_t2_campaign_report(first_path, report)
    first_bytes = first_path.read_bytes()
    write_t2_campaign_report(first_path, report)
    assert first_path.read_bytes() == first_bytes
    loaded = load_t2_campaign_report(first_path)
    write_t2_campaign_report(second_path, loaded)
    assert loaded == report
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_bytes() == protocol_module._canonical(report.payload())

    # This performs one genuine, source-bound deterministic replay of all ten
    # development arms.  The official placeholder config remains untouched.
    reconstruct_t2_campaign(protocol, ROOT, loaded)

    arm = report.arm_records[2]
    defined_rows = list(arm.defined_rows)
    original = defined_rows[0]
    alternate = next(
        row.expected_answers
        for row in defined_rows[1:]
        if row.expected_answers != original.expected_answers
    )
    defined_rows[0] = protocol_module.T2PredictionRow(
        original.row_kind,
        original.item_sha256,
        alternate,
        alternate,
        True,
    )
    opening_rows = list(arm.opening_records)
    opening = opening_rows[-1]
    opening_rows[-1] = protocol_module.T2OpeningRecord(
        opening.role,
        opening.access_ordinal,
        opening.request_sha256,
        opening.target_answers,
        sha256((opening.salt + ":resealed-forgery").encode("ascii")).hexdigest(),
        opening.leaf_index,
        opening.choice_sha256,
        opening.response_sha256,
        opening.leaf_sha256,
        opening.proof_siblings,
    )
    forged_arm = _rehash_arm(
        arm,
        defined_rows=tuple(defined_rows),
        opening_records=tuple(opening_rows),
    )
    forged_arms = list(report.arm_records)
    forged_arms[2] = forged_arm
    forged_report = _rehash_report(report, tuple(forged_arms))
    with pytest.raises(T2ProtocolError, match="existing non-identical"):
        write_t2_campaign_report(first_path, forged_report)
    assert first_path.read_bytes() == first_bytes

    # Avoid a third expensive replay: the preceding public reconstruction
    # already established the trusted deterministic result used here.
    monkeypatch.setattr(
        protocol_module,
        "run_t2_campaign",
        lambda protocol_arg, root_arg: report,
    )
    with pytest.raises(T2ProtocolError, match="authoritative reconstruction"):
        reconstruct_t2_campaign(protocol, ROOT, forged_report)

    raw = json.loads(first_path.read_text(encoding="utf-8"))
    raw["arm_records"][2]["opening_records"][0]["unexpected"] = True
    hostile_path = tmp_path / "development-report-hostile.json"
    hostile_path.write_text(
        json.dumps(raw, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="opening.*schema.*closed"):
        load_t2_campaign_report(hostile_path)

    raw = json.loads(first_path.read_text(encoding="utf-8"))
    raw["arm_records"][2]["shortcut_evidence"]["unexpected"] = True
    hostile_shortcut_path = tmp_path / "development-report-hostile-shortcut.json"
    hostile_shortcut_path.write_text(
        json.dumps(raw, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shortcut evidence.*schema.*closed"):
        load_t2_campaign_report(hostile_shortcut_path)


def test_staged_ten_arm_roundtrip_open_and_fail_before_replay(
    development_campaign,
    tmp_path,
    monkeypatch,
) -> None:
    protocol, _environments, preopens, in_memory_terminal, report = (
        development_campaign
    )
    staged_directory = tmp_path / "preopen"
    records = tuple(
        protocol_module.T2PreopenRecord.from_preopen(preopen)
        for preopen in preopens
    )
    for index, record in enumerate(records):
        path = protocol_module.t2_preopen_record_path(staged_directory, index)
        protocol_module.write_t2_preopen_record(path, record)
        assert protocol_module.load_t2_preopen_record(path) == record
        assert path.read_bytes() == protocol_module._canonical(record.payload())
        payload_text = path.read_text(encoding="utf-8")
        for forbidden in (
            '"target_answers"',
            '"salt"',
            '"proof_siblings"',
            '"inferred_request_sha256"',
            '"sealed_request_sha256s"',
            '"long_rows"',
        ):
            assert forbidden not in payload_text
    first_record_path = protocol_module.t2_preopen_record_path(
        staged_directory, 0
    )
    first_record_bytes = first_record_path.read_bytes()
    protocol_module.write_t2_preopen_record(first_record_path, records[0])
    assert first_record_path.read_bytes() == first_record_bytes
    with pytest.raises(T2ProtocolError, match="existing non-identical"):
        protocol_module.write_t2_preopen_record(first_record_path, records[1])
    assert first_record_path.read_bytes() == first_record_bytes
    directory_destination = tmp_path / "directory-destination"
    directory_destination.mkdir()
    with pytest.raises(T2ProtocolError, match="existing non-identical"):
        protocol_module.write_t2_preopen_record(
            directory_destination, records[0]
        )
    loaded_records = protocol_module.load_t2_preopen_record_set(
        staged_directory
    )
    assert loaded_records == records

    terminal = protocol_module.aggregate_t2_preopen_records(
        protocol,
        ROOT,
        loaded_records,
    )
    assert terminal == in_memory_terminal
    terminal_path = tmp_path / "terminal-preopen.json"
    protocol_module.write_t2_campaign_terminal_preopen(terminal_path, terminal)
    assert (
        protocol_module.load_t2_campaign_terminal_preopen(terminal_path)
        == terminal
    )
    assert terminal_path.read_bytes() == protocol_module._canonical(
        terminal.payload()
    )
    terminal_bytes = terminal_path.read_bytes()
    protocol_module.write_t2_campaign_terminal_preopen(terminal_path, terminal)
    assert terminal_path.read_bytes() == terminal_bytes
    swapped_preopens = list(terminal.scheduled_preopen_sha256s)
    swapped_preopens[-2], swapped_preopens[-1] = (
        swapped_preopens[-1],
        swapped_preopens[-2],
    )
    forged_terminal_payload = terminal.payload(False)
    forged_terminal_payload["scheduled_preopen_sha256s"] = swapped_preopens
    forged_terminal = protocol_module.T2CampaignTerminalPreopen(
        terminal.protocol_sha256,
        terminal.source_runtime_binding_sha256,
        tuple(swapped_preopens),
        protocol_module._digest(forged_terminal_payload),
    )
    with pytest.raises(T2ProtocolError, match="existing non-identical"):
        protocol_module.write_t2_campaign_terminal_preopen(
            terminal_path, forged_terminal
        )
    assert terminal_path.read_bytes() == terminal_bytes

    staged_report = protocol_module.open_t2_staged_campaign(
        protocol,
        ROOT,
        loaded_records,
        terminal,
    )
    assert staged_report == report
    real_staged_opener = protocol_module.open_t2_staged_campaign
    monkeypatch.setattr(
        protocol_module,
        "open_t2_staged_campaign",
        lambda protocol_arg, root_arg, records_arg, terminal_arg: staged_report,
    )
    protocol_module.validate_t2_staged_campaign(
        protocol,
        ROOT,
        loaded_records,
        terminal,
        report,
    )
    monkeypatch.setattr(
        protocol_module,
        "open_t2_staged_campaign",
        real_staged_opener,
    )

    # A fully late record-order attack must fail at the public aggregate
    # precheck, before even one deterministic arm replay or postfit sidecar.
    forged_records = list(loaded_records)
    forged_records[-2], forged_records[-1] = (
        forged_records[-1],
        forged_records[-2],
    )
    replay_calls: list[int] = []
    postfit_calls: list[str] = []

    def forbidden_replay(*_args, **_kwargs):
        replay_calls.append(1)
        raise AssertionError("mismatched staged inventory reached replay")

    def forbidden_postfit(*_args, **_kwargs):
        postfit_calls.append("opened")
        raise AssertionError("mismatched staged inventory opened a sidecar")

    monkeypatch.setattr(
        protocol_module,
        "_reconstruct_t2_preopen",
        forbidden_replay,
    )
    monkeypatch.setattr(
        protocol_module,
        "_open_t2_postfit_environment",
        forbidden_postfit,
    )
    with pytest.raises(T2ProtocolError, match="frozen schedule position"):
        protocol_module.open_t2_staged_campaign(
            protocol,
            ROOT,
            tuple(forged_records),
            terminal,
        )
    assert replay_calls == []
    assert postfit_calls == []

    hostile = json.loads(
        protocol_module.t2_preopen_record_path(
            staged_directory, 2
        ).read_text(encoding="utf-8")
    )
    hostile["unexpected"] = True
    hostile_path = tmp_path / "hostile-preopen.json"
    hostile_path.write_text(
        json.dumps(hostile, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="preopen artifact schema is not closed"):
        protocol_module.load_t2_preopen_record(hostile_path)

    noncanonical_path = tmp_path / "noncanonical-preopen.json"
    noncanonical_path.write_bytes(first_record_bytes + b"\n")
    with pytest.raises(ValueError, match="canonical JSON byte form"):
        protocol_module.load_t2_preopen_record(noncanonical_path)

    duplicate_key_path = tmp_path / "duplicate-key-preopen.json"
    duplicate_key_path.write_text(
        '{"schema":"x","schema":"x"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        protocol_module.load_t2_preopen_record(duplicate_key_path)

    wrong_type = records[2].payload()
    wrong_type["opened_request_sha256s"] = "not-an-array"
    wrong_type_path = tmp_path / "wrong-type-preopen.json"
    wrong_type_path.write_text(
        json.dumps(wrong_type, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="exact JSON array"):
        protocol_module.load_t2_preopen_record(wrong_type_path)

    extra_inventory_path = staged_directory / "preopen-environment-10.json"
    extra_inventory_path.write_bytes(first_record_bytes)
    with pytest.raises(T2ProtocolError, match="exact ten-file inventory"):
        protocol_module.load_t2_preopen_record_set(staged_directory)
