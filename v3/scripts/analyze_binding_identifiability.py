"""Emit a deterministic Phase-I held-out identifiability certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from tnlm_v3.algebra_identification import analyze_heldout_identification
from tnlm_v3.campaign_config import load_milestone4_campaign_config


SCHEMA = "tnlm-v3-phase1-heldout-identification-analysis-v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def build_analysis_record(
    config_path: Path,
    *,
    max_states: int,
    max_permutations: int,
) -> dict[str, object]:
    """Build an exact, deterministic certificate from a strict config."""

    config_bytes = config_path.read_bytes()
    config = load_milestone4_campaign_config(config_path)
    report = analyze_heldout_identification(
        config.task,
        max_states=max_states,
        max_permutations=max_permutations,
    )
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope": (
            "exact_binding_v1_identifiability_analysis_not_language_or_model_evidence"
        ),
        "campaign_id": config.campaign_id,
        "campaign_stage": config.stage.value,
        "config_file_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "identification": report.to_dict(),
        "claims": {
            "training_data_alone_identifies_missing_direction": False,
            "cyclic_update_law_alone_identifies_full_transducer": False,
            (
                "declared_universal_key_value_operator_identities_"
                "identify_missing_behavior"
            ): True,
            "exact_behavior_does_not_fix_latent_gauge": True,
            "generic_factorization_alone_guarantees_completion": False,
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
    arguments = parser.parse_args(argv)
    if arguments.max_states < 1:
        parser.error("--max-states must be positive")
    if arguments.max_permutations < 1:
        parser.error("--max-permutations must be positive")
    record = build_analysis_record(
        arguments.config,
        max_states=arguments.max_states,
        max_permutations=arguments.max_permutations,
    )
    encoded = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        _write_atomic(arguments.output, encoded.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
