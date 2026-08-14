"""Emit deterministic evidence for the synthetic Phase-II algebra power control.

The run is intentionally narrow.  It demonstrates power to select and realize
an *observed* destination-key/transition-address-local exception inside the
supplied register-transducer representation.  It is not an outer-test run, an
independent-holdout experiment, unseen-singleton recovery, representation
discovery, or a confirmatory result.

Every immutable dataclass in the paired positive/null report is serialized.
The outer record binds the frozen protocol, execution budget, relevant source
files, exact acceptance summary, and the full report with a canonical SHA-256.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping as ABCMapping
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
from typing import (
    Any,
    Mapping,
    Sequence,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from tnlm_v3.algebra_discovery_power import (
    PairLocalExceptionPowerReport,
    default_power_control_budget,
    default_power_control_design,
    run_pair_local_exception_power_control,
)


SCHEMA = "tnlm-v3-phase2-algebra-power-control-evidence-v1"
PROTOCOL_SCHEMA = "tnlm-v3-phase2-algebra-power-control-protocol-v1"
CONFIG_SCHEMA = "tnlm-v3-phase2-algebra-power-control-config-v1"
SCOPE = "synthetic_observed_transition_address_exception_power_control_only"
MAX_RECORD_BYTES = 64 * 1024 * 1024


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _jsonable(value: object) -> object:
    """Losslessly project the immutable evidence tree into JSON values."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError(f"unsupported evidence value type: {type(value).__name__}")


def _v3_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _source_hashes(v3_root: Path | None = None) -> dict[str, str]:
    root = _v3_root() if v3_root is None else v3_root
    if not isinstance(root, Path):
        raise TypeError("v3_root must be pathlib.Path")
    package_paths = tuple(
        sorted((root / "src" / "tnlm_v3").rglob("*.py"))
    )
    paths = package_paths + (
        root / "pyproject.toml",
        Path(__file__).resolve(),
    )
    return {
        path.relative_to(root).as_posix(): _sha256_bytes(path.read_bytes())
        for path in paths
    }


def _protocol_record() -> dict[str, object]:
    design = default_power_control_design()
    body: dict[str, object] = {
        "schema": PROTOCOL_SCHEMA,
        "scope": SCOPE,
        "frozen_before_evidence_execution": True,
        "condition_order": [
            "observed_pair_local_exception",
            "no_exception",
        ],
        "design": _jsonable(design),
        "design_sha256": design.design_sha256,
        "candidate_residual_penalties": [4, 16],
        "expected_positive_selected_penalty": 4,
        "expected_negative_selected_penalty": 16,
        "expected_exception_override": {
            "destination_key": design.exception.destination_key,
            "prototype_address": design.exception.address.label,
            "output": design.exception.exceptional_output,
        },
        "self_pseudoheldout_exception_fold_claimed_identifying": False,
        "separate_full_split_audit_is_independent_holdout": False,
        "unseen_exception_prediction_claimed": False,
        "confirmatory_claim_permitted": False,
    }
    return {
        **body,
        "protocol_sha256": _sha256_bytes(_canonical_bytes(body)),
    }


def _config_record() -> dict[str, object]:
    budget = default_power_control_budget()
    body: dict[str, object] = {
        "schema": CONFIG_SCHEMA,
        "budget": _jsonable(budget),
        "budget_sha256": budget.budget_sha256,
        "serialization": {
            "canonical_hash_sort_keys": True,
            "canonical_hash_separators": [",", ":"],
            "canonical_hash_ensure_ascii": True,
            "allow_nan": False,
            "artifact_sort_keys": True,
            "artifact_indent": 2,
            "artifact_terminal_newline": True,
            "atomic_replace": True,
        },
    }
    return {
        **body,
        "config_sha256": _sha256_bytes(_canonical_bytes(body)),
    }


def _acceptance_summary(report: PairLocalExceptionPowerReport) -> dict[str, object]:
    if type(report) is not PairLocalExceptionPowerReport:
        raise TypeError("report must be exact PairLocalExceptionPowerReport")
    positive = report.positive
    negative = report.negative
    positive_audits = {
        audit.residual_penalty: audit for audit in positive.direct_penalty_audits
    }
    negative_audits = {
        audit.residual_penalty: audit for audit in negative.direct_penalty_audits
    }
    return {
        "paired": {
            "matched_visible_programs": report.matched_visible_programs,
            "balanced_output_classes": report.balanced_output_classes,
            "positive_program_manifest_sha256": (
                positive.corpus.program_manifest_sha256
            ),
            "negative_program_manifest_sha256": (
                negative.corpus.program_manifest_sha256
            ),
        },
        "positive": {
            "selected_residual_penalty": positive.selection.selected_residual_penalty,
            "primary_score_best_penalties": list(
                positive.selection.primary_score_best_penalties
            ),
            "primary_score_tied": positive.selection.primary_score_tied,
            "selected_sequence_validation_margin": (
                positive.selected_sequence_validation_margin
            ),
            "separate_full_split_validation_margin": (
                positive.direct_full_validation_margin
            ),
            "crosslink_winning_cells": [
                list(cell) for cell in positive.crosslink_winning_cells
            ],
            "self_pseudoheldout_cells": [
                list(cell) for cell in positive.self_pseudoheldout_cells
            ],
            "fold_optimum_certificate_count": len(
                positive.fold_optimum_certificates
            ),
            "all_fold_candidates_certified_optimal": all(
                row.global_optimum_certified_for_frozen_control
                for row in positive.fold_optimum_certificates
            ),
            "penalty_4": {
                "training_mistakes": positive_audits[4].training_mistakes,
                "training_query_count": positive_audits[4].training_query_count,
                "validation_mistakes": positive_audits[4].validation_mistakes,
                "validation_query_count": positive_audits[4].validation_query_count,
                "attained_training_objective": (
                    positive_audits[4].attained_training_objective
                ),
                "local_override_count": len(
                    positive_audits[4].model.local_overrides
                ),
                "canonical_shared_table_realized": (
                    positive_audits[4].canonical_shared_table_realized
                ),
                "semantic_decomposition_gauge_fixed": (
                    positive_audits[4].semantic_decomposition_gauge_fixed
                ),
                "expected_exception_override_realized": (
                    positive_audits[4].expected_exception_override_realized
                ),
            },
            "penalty_16": {
                "training_mistakes": positive_audits[16].training_mistakes,
                "training_query_count": positive_audits[16].training_query_count,
                "validation_mistakes": positive_audits[16].validation_mistakes,
                "validation_query_count": positive_audits[16].validation_query_count,
                "attained_training_objective": (
                    positive_audits[16].attained_training_objective
                ),
                "local_override_count": len(
                    positive_audits[16].model.local_overrides
                ),
                "canonical_shared_table_realized": (
                    positive_audits[16].canonical_shared_table_realized
                ),
                "semantic_decomposition_gauge_fixed": (
                    positive_audits[16].semantic_decomposition_gauge_fixed
                ),
            },
        },
        "negative": {
            "selected_residual_penalty": negative.selection.selected_residual_penalty,
            "primary_score_best_penalties": list(
                negative.selection.primary_score_best_penalties
            ),
            "primary_score_tied": negative.selection.primary_score_tied,
            "selected_sequence_validation_margin": (
                negative.selected_sequence_validation_margin
            ),
            "separate_full_split_validation_margin": (
                negative.direct_full_validation_margin
            ),
            "fold_optimum_certificate_count": len(
                negative.fold_optimum_certificates
            ),
            "all_fold_candidates_certified_optimal": all(
                row.global_optimum_certified_for_frozen_control
                for row in negative.fold_optimum_certificates
            ),
            "penalty_4": {
                "training_mistakes": negative_audits[4].training_mistakes,
                "validation_mistakes": negative_audits[4].validation_mistakes,
                "local_override_count": len(
                    negative_audits[4].model.local_overrides
                ),
                "canonical_shared_table_realized": (
                    negative_audits[4].canonical_shared_table_realized
                ),
            },
            "penalty_16": {
                "training_mistakes": negative_audits[16].training_mistakes,
                "validation_mistakes": negative_audits[16].validation_mistakes,
                "local_override_count": len(
                    negative_audits[16].model.local_overrides
                ),
                "canonical_shared_table_realized": (
                    negative_audits[16].canonical_shared_table_realized
                ),
            },
        },
    }


def _claims() -> dict[str, bool]:
    return {
        "synthetic_control": True,
        "observed_exception_seen_in_direct_train_and_validation": True,
        "supplied_register_transducer_representation": True,
        "transition_coefficients_fitted_from_query_supervision": True,
        "candidate_specific_optima_certified_for_frozen_control": True,
        "matched_visible_program_negative_control": True,
        "balanced_query_output_classes": True,
        "separate_full_split_audit_reuses_selection_validation_data": True,
        "independent_holdout_used": False,
        "outer_test_results_used_for_design_fit_selection_or_acceptance": False,
        "self_pseudoheldout_exception_fold_identifies_local_exception": False,
        "unseen_singleton_exception_recovery_demonstrated": False,
        "representation_discovery_performed": False,
        "assumption_free_algebra_discovery_performed": False,
        "confirmatory_claim_permitted": False,
    }


def build_evidence_record() -> dict[str, object]:
    """Run once and return a canonical, source-bound evidence record."""

    report = run_pair_local_exception_power_control()
    protocol = _protocol_record()
    config = _config_record()
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope": SCOPE,
        "protocol": protocol,
        "config": config,
        "source_file_sha256": _source_hashes(),
        "acceptance": _acceptance_summary(report),
        "claims": _claims(),
        "report": _jsonable(report),
    }
    record = {
        **body,
        "record_sha256": _sha256_bytes(_canonical_bytes(body)),
    }
    validate_evidence_record(record)
    return record


def _mapping(name: str, value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys")
    return value


def _validate_protocol(value: object) -> dict[str, object]:
    protocol = _mapping("protocol", value)
    material = dict(protocol)
    digest = material.pop("protocol_sha256", None)
    _require_sha256("protocol_sha256", digest)
    if digest != _sha256_bytes(_canonical_bytes(material)):
        raise ValueError("protocol_sha256 does not bind the protocol")
    if protocol != _protocol_record():
        raise ValueError("record protocol differs from the frozen protocol")
    return protocol


def _validate_config(value: object) -> dict[str, object]:
    config = _mapping("config", value)
    material = dict(config)
    digest = material.pop("config_sha256", None)
    _require_sha256("config_sha256", digest)
    if digest != _sha256_bytes(_canonical_bytes(material)):
        raise ValueError("config_sha256 does not bind the execution config")
    if config != _config_record():
        raise ValueError("record config differs from the frozen execution config")
    return config


def _serialized_acceptance(report: Mapping[str, object]) -> dict[str, object]:
    """Recompute the small acceptance view from the full serialized tree."""

    def condition(name: str) -> tuple[dict[str, object], dict[int, dict[str, object]]]:
        result = _mapping(name, report.get(name))
        audits = result.get("direct_penalty_audits")
        if not isinstance(audits, list) or len(audits) != 2:
            raise ValueError(f"{name} must serialize exactly two direct audits")
        by_penalty: dict[int, dict[str, object]] = {}
        for raw in audits:
            row = _mapping(f"{name} audit", raw)
            penalty = row.get("residual_penalty")
            if type(penalty) is not int or penalty in by_penalty:
                raise ValueError(f"{name} has an invalid audit penalty inventory")
            by_penalty[penalty] = row
        if set(by_penalty) != {4, 16}:
            raise ValueError(f"{name} audit penalties must be exactly 4 and 16")
        return result, by_penalty

    positive, positive_audits = condition("positive")
    negative, negative_audits = condition("negative")

    def model(row: Mapping[str, object]) -> dict[str, object]:
        return _mapping("audit model", row.get("model"))

    def selection(result: Mapping[str, object]) -> dict[str, object]:
        return _mapping("selection", result.get("selection"))

    def corpus(result: Mapping[str, object]) -> dict[str, object]:
        return _mapping("corpus", result.get("corpus"))

    positive_fold_rows = positive.get("fold_optimum_certificates")
    negative_fold_rows = negative.get("fold_optimum_certificates")
    if not isinstance(positive_fold_rows, list) or not isinstance(
        negative_fold_rows, list
    ):
        raise TypeError("fold certificates must serialize as arrays")
    return {
        "paired": {
            "matched_visible_programs": report.get("matched_visible_programs"),
            "balanced_output_classes": report.get("balanced_output_classes"),
            "positive_program_manifest_sha256": corpus(positive).get(
                "program_manifest_sha256"
            ),
            "negative_program_manifest_sha256": corpus(negative).get(
                "program_manifest_sha256"
            ),
        },
        "positive": {
            "selected_residual_penalty": selection(positive).get(
                "selected_residual_penalty"
            ),
            "primary_score_best_penalties": selection(positive).get(
                "primary_score_best_penalties"
            ),
            "primary_score_tied": selection(positive).get("primary_score_tied"),
            "selected_sequence_validation_margin": positive.get(
                "selected_sequence_validation_margin"
            ),
            "separate_full_split_validation_margin": positive.get(
                "direct_full_validation_margin"
            ),
            "crosslink_winning_cells": positive.get("crosslink_winning_cells"),
            "self_pseudoheldout_cells": positive.get("self_pseudoheldout_cells"),
            "fold_optimum_certificate_count": len(positive_fold_rows),
            "all_fold_candidates_certified_optimal": all(
                _mapping("positive fold", row).get(
                    "global_optimum_certified_for_frozen_control"
                )
                is True
                for row in positive_fold_rows
            ),
            "penalty_4": {
                "training_mistakes": positive_audits[4].get("training_mistakes"),
                "training_query_count": positive_audits[4].get(
                    "training_query_count"
                ),
                "validation_mistakes": positive_audits[4].get(
                    "validation_mistakes"
                ),
                "validation_query_count": positive_audits[4].get(
                    "validation_query_count"
                ),
                "attained_training_objective": positive_audits[4].get(
                    "attained_training_objective"
                ),
                "local_override_count": len(
                    model(positive_audits[4]).get("local_overrides", [])
                ),
                "canonical_shared_table_realized": positive_audits[4].get(
                    "canonical_shared_table_realized"
                ),
                "semantic_decomposition_gauge_fixed": positive_audits[4].get(
                    "semantic_decomposition_gauge_fixed"
                ),
                "expected_exception_override_realized": positive_audits[4].get(
                    "expected_exception_override_realized"
                ),
            },
            "penalty_16": {
                "training_mistakes": positive_audits[16].get("training_mistakes"),
                "training_query_count": positive_audits[16].get(
                    "training_query_count"
                ),
                "validation_mistakes": positive_audits[16].get(
                    "validation_mistakes"
                ),
                "validation_query_count": positive_audits[16].get(
                    "validation_query_count"
                ),
                "attained_training_objective": positive_audits[16].get(
                    "attained_training_objective"
                ),
                "local_override_count": len(
                    model(positive_audits[16]).get("local_overrides", [])
                ),
                "canonical_shared_table_realized": positive_audits[16].get(
                    "canonical_shared_table_realized"
                ),
                "semantic_decomposition_gauge_fixed": positive_audits[16].get(
                    "semantic_decomposition_gauge_fixed"
                ),
            },
        },
        "negative": {
            "selected_residual_penalty": selection(negative).get(
                "selected_residual_penalty"
            ),
            "primary_score_best_penalties": selection(negative).get(
                "primary_score_best_penalties"
            ),
            "primary_score_tied": selection(negative).get("primary_score_tied"),
            "selected_sequence_validation_margin": negative.get(
                "selected_sequence_validation_margin"
            ),
            "separate_full_split_validation_margin": negative.get(
                "direct_full_validation_margin"
            ),
            "fold_optimum_certificate_count": len(negative_fold_rows),
            "all_fold_candidates_certified_optimal": all(
                _mapping("negative fold", row).get(
                    "global_optimum_certified_for_frozen_control"
                )
                is True
                for row in negative_fold_rows
            ),
            "penalty_4": {
                "training_mistakes": negative_audits[4].get("training_mistakes"),
                "validation_mistakes": negative_audits[4].get(
                    "validation_mistakes"
                ),
                "local_override_count": len(
                    model(negative_audits[4]).get("local_overrides", [])
                ),
                "canonical_shared_table_realized": negative_audits[4].get(
                    "canonical_shared_table_realized"
                ),
            },
            "penalty_16": {
                "training_mistakes": negative_audits[16].get("training_mistakes"),
                "validation_mistakes": negative_audits[16].get(
                    "validation_mistakes"
                ),
                "local_override_count": len(
                    model(negative_audits[16]).get("local_overrides", [])
                ),
                "canonical_shared_table_realized": negative_audits[16].get(
                    "canonical_shared_table_realized"
                ),
            },
        },
    }


def _decode_typed(expected: object, raw: object, path: str) -> object:
    """Reconstruct the exact immutable scientific tree from canonical JSON."""

    if expected is Any or expected is object:
        return raw
    origin = get_origin(expected)
    arguments = get_args(expected)
    if origin is tuple:
        if type(raw) is not list:
            raise TypeError(f"{path} must be a JSON list encoding a tuple")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                _decode_typed(arguments[0], item, f"{path}[{index}]")
                for index, item in enumerate(raw)
            )
        if len(raw) != len(arguments):
            raise ValueError(f"{path} tuple arity changed")
        return tuple(
            _decode_typed(item_type, item, f"{path}[{index}]")
            for index, (item_type, item) in enumerate(
                zip(arguments, raw, strict=True)
            )
        )
    if origin is list:
        if type(raw) is not list:
            raise TypeError(f"{path} must be an exact list")
        item_type = arguments[0] if arguments else Any
        return [
            _decode_typed(item_type, item, f"{path}[{index}]")
            for index, item in enumerate(raw)
        ]
    if origin in (dict, ABCMapping):
        mapping = _mapping(path, raw)
        key_type, value_type = arguments if arguments else (Any, Any)
        return {
            _decode_typed(key_type, key, f"{path}.key"): _decode_typed(
                value_type,
                value,
                f"{path}[{key!r}]",
            )
            for key, value in mapping.items()
        }
    if origin in (Union, types.UnionType):
        if raw is None and type(None) in arguments:
            return None
        failures: list[Exception] = []
        for option in arguments:
            if option is type(None):
                continue
            try:
                return _decode_typed(option, raw, path)
            except (TypeError, ValueError) as error:
                failures.append(error)
        raise TypeError(f"{path} does not match any declared union member") from (
            failures[-1] if failures else None
        )
    if isinstance(expected, type) and issubclass(expected, Enum):
        try:
            return expected(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path} is not a valid {expected.__name__}") from error
    if isinstance(expected, type) and is_dataclass(expected):
        mapping = _mapping(path, raw)
        expected_names = tuple(field.name for field in fields(expected))
        if set(mapping) != set(expected_names):
            missing = sorted(set(expected_names) - set(mapping))
            extra = sorted(set(mapping) - set(expected_names))
            raise ValueError(
                f"{path} dataclass fields changed: missing={missing}, extra={extra}"
            )
        hints = get_type_hints(expected)
        values = {
            name: _decode_typed(hints[name], mapping[name], f"{path}.{name}")
            for name in expected_names
        }
        return expected(**values)
    if expected is type(None):
        if raw is not None:
            raise TypeError(f"{path} must be null")
        return None
    if expected in (bool, int, float, str):
        if type(raw) is not expected:
            raise TypeError(f"{path} must be exact {expected.__name__}")
        return raw
    raise TypeError(f"{path} uses unsupported annotation {expected!r}")


def _decode_dataclass(expected: type, raw: object, path: str):
    value = _decode_typed(expected, raw, path)
    if type(value) is not expected:
        raise TypeError(f"{path} did not reconstruct exact {expected.__name__}")
    return value


def validate_evidence_record(record: object) -> dict[str, object]:
    """Validate hashes, frozen inputs, full-tree bindings, metrics, and scope."""

    document = _mapping("record", record)
    expected_keys = {
        "schema",
        "scope",
        "protocol",
        "config",
        "source_file_sha256",
        "acceptance",
        "claims",
        "report",
        "record_sha256",
    }
    if set(document) != expected_keys:
        raise ValueError("evidence record keys differ from the closed schema")
    material = dict(document)
    digest = material.pop("record_sha256")
    _require_sha256("record_sha256", digest)
    if digest != _sha256_bytes(_canonical_bytes(material)):
        raise ValueError("record_sha256 does not bind the full evidence record")
    if document["schema"] != SCHEMA or document["scope"] != SCOPE:
        raise ValueError("evidence schema or scope changed")
    protocol = _validate_protocol(document["protocol"])
    config = _validate_config(document["config"])
    sources = _mapping("source_file_sha256", document["source_file_sha256"])
    for name, value in sources.items():
        _require_sha256(f"source hash {name}", value)
    if sources != _source_hashes():
        raise ValueError("source hashes do not match the executing source tree")
    if document["claims"] != _claims():
        raise ValueError("evidence claims exceed or differ from the frozen scope")
    report = _mapping("report", document["report"])
    if report.get("schema") != "tnlm-v3-pair-local-power-report-v1":
        raise ValueError("full report has an unknown schema")
    if report.get("report_sha256") is None:
        raise ValueError("full report is missing its immutable report digest")
    _require_sha256("report_sha256", report["report_sha256"])
    if report.get("design") != protocol.get("design"):
        raise ValueError("full report design differs from the frozen protocol")
    if report.get("budget") != config.get("budget"):
        raise ValueError("full report budget differs from the execution config")
    decoded_report = _decode_dataclass(
        PairLocalExceptionPowerReport,
        report,
        "report",
    )
    if _jsonable(decoded_report) != report:
        raise ValueError("full report is not the canonical immutable dataclass tree")
    acceptance = _mapping("acceptance", document["acceptance"])
    if acceptance != _acceptance_summary(decoded_report):
        raise ValueError("acceptance summary does not reproduce the full report tree")
    expected_exact = {
        "positive_selected": 4,
        "negative_selected": 16,
        "positive_selected_margin": 36,
        "positive_full_split_margin": 12,
        "positive_low_train_mistakes": 0,
        "positive_high_train_mistakes": 6,
        "positive_low_validation_mistakes": 0,
        "positive_high_validation_mistakes": 12,
    }
    actual_exact = {
        "positive_selected": acceptance["positive"]["selected_residual_penalty"],
        "negative_selected": acceptance["negative"]["selected_residual_penalty"],
        "positive_selected_margin": acceptance["positive"][
            "selected_sequence_validation_margin"
        ],
        "positive_full_split_margin": acceptance["positive"][
            "separate_full_split_validation_margin"
        ],
        "positive_low_train_mistakes": acceptance["positive"]["penalty_4"][
            "training_mistakes"
        ],
        "positive_high_train_mistakes": acceptance["positive"]["penalty_16"][
            "training_mistakes"
        ],
        "positive_low_validation_mistakes": acceptance["positive"]["penalty_4"][
            "validation_mistakes"
        ],
        "positive_high_validation_mistakes": acceptance["positive"]["penalty_16"][
            "validation_mistakes"
        ],
    }
    if actual_exact != expected_exact:
        raise ValueError("power-control metrics differ from the frozen acceptance result")
    return document


def validate_rerun_equivalence(first: object, second: object) -> None:
    """Require two independently produced records to be byte-canonically equal."""

    left = validate_evidence_record(first)
    right = validate_evidence_record(second)
    if _canonical_bytes(left) != _canonical_bytes(right):
        raise ValueError("power-control rerun differs from the reference record")


def encode_evidence_record(record: object) -> bytes:
    validated = validate_evidence_record(record)
    return (
        json.dumps(validated, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def load_evidence_record(
    path: Path,
    *,
    max_record_bytes: int = MAX_RECORD_BYTES,
) -> dict[str, object]:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if type(max_record_bytes) is not int or max_record_bytes < 1:
        raise ValueError("max_record_bytes must be a positive exact integer")
    size = path.stat().st_size
    if size > max_record_bytes:
        raise ValueError("evidence record exceeds max_record_bytes")
    try:
        document = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evidence record is not valid UTF-8 JSON") from error
    return validate_evidence_record(document)


def _write_atomic(path: Path, payload: bytes) -> None:
    if not isinstance(path, Path):
        raise TypeError("output path must be pathlib.Path")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be exact bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-against",
        type=Path,
        help=(
            "validate a prior record and require this deterministic rerun to match it"
        ),
    )
    arguments = parser.parse_args(argv)
    record = build_evidence_record()
    if arguments.verify_against is not None:
        reference = load_evidence_record(arguments.verify_against)
        validate_rerun_equivalence(reference, record)
    _write_atomic(arguments.output, encode_evidence_record(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
