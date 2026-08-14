from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_phase2_algebra_power_control.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase2_algebra_power_control_runner",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def repeated_evidence(tmp_path_factory: pytest.TempPathFactory):
    """Execute exactly two temporary control runs, never a production artifact."""

    module = _load_script()
    directory = tmp_path_factory.mktemp("phase2-power-control")
    first_path = directory / "first.json"
    second_path = directory / "second.json"
    assert module.main(["--output", str(first_path)]) == 0
    assert module.main(
        [
            "--output",
            str(second_path),
            "--verify-against",
            str(first_path),
        ]
    ) == 0
    first_bytes = first_path.read_bytes()
    second_bytes = second_path.read_bytes()
    first = module.load_evidence_record(first_path)
    second = module.load_evidence_record(second_path)
    return module, directory, first_path, second_path, first_bytes, second_bytes, first, second


def _rehashed(module: ModuleType, record: dict[str, object]) -> dict[str, object]:
    material = copy.deepcopy(record)
    material.pop("record_sha256")
    material["record_sha256"] = hashlib.sha256(
        module._canonical_bytes(material)
    ).hexdigest()
    return material


def test_runner_binds_frozen_protocol_config_sources_and_outer_record(
    repeated_evidence,
) -> None:
    module, _, _, _, _, _, record, _ = repeated_evidence
    assert record["schema"] == module.SCHEMA
    assert record["scope"] == (
        "synthetic_observed_transition_address_exception_power_control_only"
    )

    protocol = dict(record["protocol"])
    protocol_digest = protocol.pop("protocol_sha256")
    assert protocol_digest == hashlib.sha256(
        module._canonical_bytes(protocol)
    ).hexdigest()
    assert record["protocol"] == module._protocol_record()

    config = dict(record["config"])
    config_digest = config.pop("config_sha256")
    assert config_digest == hashlib.sha256(
        module._canonical_bytes(config)
    ).hexdigest()
    assert record["config"] == module._config_record()

    v3_root = Path(__file__).resolve().parents[1]
    expected_source_paths = {
        path.relative_to(v3_root).as_posix()
        for path in (v3_root / "src" / "tnlm_v3").rglob("*.py")
    } | {
        "scripts/run_phase2_algebra_power_control.py",
        "pyproject.toml",
    }
    assert set(record["source_file_sha256"]) == expected_source_paths
    assert "src/tnlm_v3/__init__.py" in expected_source_paths
    assert "src/tnlm_v3/routing.py" in expected_source_paths
    for relative, digest in record["source_file_sha256"].items():
        assert digest == hashlib.sha256((v3_root / relative).read_bytes()).hexdigest()

    body = dict(record)
    record_digest = body.pop("record_sha256")
    assert record_digest == hashlib.sha256(module._canonical_bytes(body)).hexdigest()
    assert module.validate_evidence_record(record) == record


def test_two_cli_runs_are_byte_identical_and_rerun_validated(
    repeated_evidence,
) -> None:
    (
        module,
        directory,
        first_path,
        second_path,
        first_bytes,
        second_bytes,
        first,
        second,
    ) = repeated_evidence
    assert first_bytes == second_bytes
    assert hashlib.sha256(first_bytes).hexdigest() == hashlib.sha256(
        second_bytes
    ).hexdigest()
    assert first == second
    assert first_bytes.endswith(b"\n")
    assert first_path.exists() and second_path.exists()
    assert not tuple(directory.glob("*.tmp"))
    assert not tuple(directory.glob(".*.tmp"))
    module.validate_rerun_equivalence(first, second)


def test_record_contains_the_full_immutable_report_tree(
    repeated_evidence,
) -> None:
    _, _, _, _, _, _, record, _ = repeated_evidence
    report = record["report"]
    assert report["schema"] == "tnlm-v3-pair-local-power-report-v1"
    assert len(report["report_sha256"]) == 64
    assert report["design"] == record["protocol"]["design"]
    assert report["budget"] == record["config"]["budget"]

    positive = report["positive"]
    negative = report["negative"]
    for condition in (positive, negative):
        assert len(condition["fold_optimum_certificates"]) == 22
        assert len(condition["direct_penalty_audits"]) == 2
        assert len(condition["corpus"]["trace_manifest"]) == 84
        assert len(condition["corpus"]["trace_corpus"]["traces"]) == 84
        assert condition["selection"]["final_model"]["fit"][
            "training_query_count"
        ] == 225

    positive_audits = {
        row["residual_penalty"]: row
        for row in positive["direct_penalty_audits"]
    }
    assert positive_audits[4]["model"]["local_overrides"] == [
        [
            1,
            {"family": "update", "transform": 0, "source_value": 0},
            2,
        ]
    ]
    assert positive_audits[16]["model"]["local_overrides"] == []


def test_exact_positive_and_null_acceptance_metrics(repeated_evidence) -> None:
    _, _, _, _, _, _, record, _ = repeated_evidence
    acceptance = record["acceptance"]
    assert acceptance["paired"]["matched_visible_programs"]
    assert acceptance["paired"]["balanced_output_classes"]
    assert (
        acceptance["paired"]["positive_program_manifest_sha256"]
        == acceptance["paired"]["negative_program_manifest_sha256"]
    )

    positive = acceptance["positive"]
    assert positive["selected_residual_penalty"] == 4
    assert positive["primary_score_best_penalties"] == [4]
    assert not positive["primary_score_tied"]
    assert positive["selected_sequence_validation_margin"] == 36
    assert positive["separate_full_split_validation_margin"] == 12
    assert positive["crosslink_winning_cells"] == [[2, 0], [2, 1], [2, 2]]
    assert positive["self_pseudoheldout_cells"] == [[1, 0], [1, 1], [1, 2]]
    assert positive["fold_optimum_certificate_count"] == 22
    assert positive["all_fold_candidates_certified_optimal"]
    assert positive["penalty_4"] == {
        "training_mistakes": 0,
        "training_query_count": 225,
        "validation_mistakes": 0,
        "validation_query_count": 231,
        "attained_training_objective": 4,
        "local_override_count": 1,
        "canonical_shared_table_realized": True,
        "semantic_decomposition_gauge_fixed": True,
        "expected_exception_override_realized": True,
    }
    assert positive["penalty_16"] == {
        "training_mistakes": 6,
        "training_query_count": 225,
        "validation_mistakes": 12,
        "validation_query_count": 231,
        "attained_training_objective": 6,
        "local_override_count": 0,
        "canonical_shared_table_realized": True,
        "semantic_decomposition_gauge_fixed": True,
    }

    negative = acceptance["negative"]
    assert negative["selected_residual_penalty"] == 16
    assert negative["primary_score_best_penalties"] == [4, 16]
    assert negative["primary_score_tied"]
    assert negative["selected_sequence_validation_margin"] == 0
    assert negative["separate_full_split_validation_margin"] == 0
    assert negative["fold_optimum_certificate_count"] == 22
    assert negative["all_fold_candidates_certified_optimal"]
    for penalty in ("penalty_4", "penalty_16"):
        assert negative[penalty] == {
            "training_mistakes": 0,
            "validation_mistakes": 0,
            "local_override_count": 0,
            "canonical_shared_table_realized": True,
        }


def test_scope_is_explicitly_synthetic_observed_and_nonconfirmatory(
    repeated_evidence,
) -> None:
    _, _, _, _, _, _, record, _ = repeated_evidence
    claims = record["claims"]
    assert claims["synthetic_control"]
    assert claims["observed_exception_seen_in_direct_train_and_validation"]
    assert claims["supplied_register_transducer_representation"]
    assert claims["transition_coefficients_fitted_from_query_supervision"]
    assert claims["matched_visible_program_negative_control"]
    assert claims["balanced_query_output_classes"]
    assert claims["separate_full_split_audit_reuses_selection_validation_data"]
    assert not claims["independent_holdout_used"]
    assert not claims[
        "outer_test_results_used_for_design_fit_selection_or_acceptance"
    ]
    assert not claims["self_pseudoheldout_exception_fold_identifies_local_exception"]
    assert not claims["unseen_singleton_exception_recovery_demonstrated"]
    assert not claims["representation_discovery_performed"]
    assert not claims["assumption_free_algebra_discovery_performed"]
    assert not claims["confirmatory_claim_permitted"]
    assert not record["protocol"][
        "self_pseudoheldout_exception_fold_claimed_identifying"
    ]
    assert not record["protocol"][
        "separate_full_split_audit_is_independent_holdout"
    ]


def test_valid_outer_digest_cannot_hide_forged_summary_or_scope(
    repeated_evidence,
) -> None:
    module, _, _, _, _, _, record, _ = repeated_evidence

    forged_summary = copy.deepcopy(record)
    forged_summary["acceptance"]["positive"]["penalty_4"][
        "validation_mistakes"
    ] = 1
    forged_summary = _rehashed(module, forged_summary)
    with pytest.raises(ValueError, match="does not reproduce"):
        module.validate_evidence_record(forged_summary)

    forged_scope = copy.deepcopy(record)
    forged_scope["claims"]["independent_holdout_used"] = True
    forged_scope = _rehashed(module, forged_scope)
    with pytest.raises(ValueError, match="claims exceed"):
        module.validate_evidence_record(forged_scope)

    forged_nested_result = copy.deepcopy(record)
    forged_nested_result["report"]["positive"]["selection"][
        "outer_unobserved_labels_used_for_fit_or_selection"
    ] = True
    forged_nested_result = _rehashed(module, forged_nested_result)
    with pytest.raises(ValueError):
        module.validate_evidence_record(forged_nested_result)


def test_loader_is_size_bounded(repeated_evidence) -> None:
    module, _, first_path, _, first_bytes, _, _, _ = repeated_evidence
    with pytest.raises(ValueError, match="exceeds max_record_bytes"):
        module.load_evidence_record(
            first_path,
            max_record_bytes=len(first_bytes) - 1,
        )
