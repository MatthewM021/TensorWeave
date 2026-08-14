"""Emit a deterministic Phase-II binding-algebra learnability certificate."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from tnlm_v3.algebra_learning import (
    AlgebraLearnabilityReport,
    SharedPrototypeCoverageSweep,
    analyze_algebra_learnability,
    evaluate_shared_prototype_transducer,
    generate_and_fit_shared_prototype_transducer,
    sweep_shared_prototype_coverage,
)
from tnlm_v3.campaign_config import load_milestone4_campaign_config


SCHEMA = "tnlm-v3-phase2-algebra-learning-analysis-v1"
MAX_COVERAGE_SEEDS = 10_000
MAX_COVERAGE_EVENTS = 10_000_000


def _jsonable(value: object) -> object:
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
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _coverage_summary(sweep: SharedPrototypeCoverageSweep) -> dict[str, object]:
    return {
        "schema": sweep.schema,
        "split": sweep.split,
        "document_count": sweep.document_count,
        "document_length": sweep.document_length,
        "seed_count": len(sweep.seeds),
        "full_rank_count": sweep.full_rank_count,
        "minimum_rank": sweep.minimum_rank,
        "maximum_rank": sweep.maximum_rank,
        "mean_rank": sweep.mean_rank,
        "parameter_count": sweep.results[0].parameter_count,
        "canonical_state_supervision_assumed": True,
        "used_heldout_labels_for_fit": False,
        "used_heldout_labels_for_selection": False,
        "per_seed": [
            {
                "seed": result.seed,
                "design_rank": result.design_rank,
                "nullity": result.nullity,
                "full_rank": result.full_rank,
                "condition_number": result.condition_number,
                "sample_sha256": result.sample_sha256,
            }
            for result in sweep.results
        ],
    }


def build_learning_record(
    config_path: Path,
    *,
    max_states: int,
    max_permutations: int,
    coverage_split: str,
    coverage_seed_count: int,
    coverage_document_count: int,
    coverage_document_length: int,
) -> dict[str, object]:
    """Build a deterministic exact-boundary plus finite-excitation record."""

    if not isinstance(config_path, Path):
        raise TypeError("config_path must be pathlib.Path")
    config_bytes = config_path.read_bytes()
    config = load_milestone4_campaign_config(config_path)
    report: AlgebraLearnabilityReport = analyze_algebra_learnability(
        config.task,
        max_states=max_states,
        max_permutations=max_permutations,
    )
    sweep = sweep_shared_prototype_coverage(
        config.task,
        split=coverage_split,
        seeds=range(coverage_seed_count),
        document_count=coverage_document_count,
        document_length=coverage_document_length,
        max_seeds=MAX_COVERAGE_SEEDS,
        max_total_events=MAX_COVERAGE_EVENTS,
    )
    fitted = generate_and_fit_shared_prototype_transducer(
        config.task,
        split="train",
        seed=0,
        document_count=coverage_document_count,
        document_length=coverage_document_length,
        max_events=MAX_COVERAGE_EVENTS,
    )
    fitted_evaluation = evaluate_shared_prototype_transducer(config.task, fitted)
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope": (
            "canonical_state_binding_algebra_learnability_analysis_"
            "not_sequence_representation_or_language_evidence"
        ),
        "campaign_id": config.campaign_id,
        "campaign_stage": config.stage.value,
        "config_file_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "learnability": _jsonable(report),
        "finite_seen_only_excitation": _coverage_summary(sweep),
        "first_seed_oracle_state_coefficient_control": {
            "fit_split": "train",
            "model": _jsonable(fitted),
            "postfit_sealed_evaluation": _jsonable(fitted_evaluation),
        },
        "claims": {
            "unrestricted_passive_recovery_from_current_support": False,
            "supplied_full_shared_prototypes_conditionally_identified": True,
            "oracle_generated_state_table_control_recovers_exact_tables": (
                fitted_evaluation.all_value_tables_exact
            ),
            "empirical_sequence_only_tables_learned": False,
            "automatic_sharing_law_selection_performed": False,
            "sequence_only_representation_learning_performed": False,
            "hard_equivariance_would_be_an_imposed_positive_control": True,
            "heldout_labels_used_for_fit_or_selection": False,
        },
    }
    return {
        **body,
        "record_sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
    }


def _write_atomic(path: Path, payload: bytes) -> None:
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
    default_config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "milestone4"
        / "validation_screen_v1.yaml"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-states", type=int, default=1_000_000)
    parser.add_argument("--max-permutations", type=int, default=100_000)
    parser.add_argument(
        "--coverage-split", choices=("train", "validation"), default="train"
    )
    parser.add_argument("--coverage-seed-count", type=int, default=100)
    parser.add_argument("--coverage-document-count", type=int, default=8)
    parser.add_argument("--coverage-document-length", type=int, default=64)
    arguments = parser.parse_args(argv)
    for name in (
        "max_states",
        "max_permutations",
        "coverage_seed_count",
        "coverage_document_count",
        "coverage_document_length",
    ):
        if getattr(arguments, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    record = build_learning_record(
        arguments.config,
        max_states=arguments.max_states,
        max_permutations=arguments.max_permutations,
        coverage_split=arguments.coverage_split,
        coverage_seed_count=arguments.coverage_seed_count,
        coverage_document_count=arguments.coverage_document_count,
        coverage_document_length=arguments.coverage_document_length,
    )
    encoded = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        _write_atomic(arguments.output, encoded.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
