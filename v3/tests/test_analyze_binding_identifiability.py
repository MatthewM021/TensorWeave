from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def test_identifiability_cli_is_deterministic_and_self_bound(
    tmp_path: Path,
) -> None:
    v3_root = Path(__file__).resolve().parents[1]
    script = v3_root / "scripts" / "analyze_binding_identifiability.py"
    outputs = (tmp_path / "first.json", tmp_path / "second.json")
    environment = os.environ.copy()
    source_root = str(v3_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_root, environment.get("PYTHONPATH", ""))
        if value
    )
    for output in outputs:
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--output",
                str(output),
                "--max-states",
                "821",
            ],
            cwd=v3_root.parent,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == ""

    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    record = json.loads(outputs[0].read_text(encoding="utf-8"))
    claimed_sha256 = record.pop("record_sha256")
    assert claimed_sha256 == hashlib.sha256(_canonical_bytes(record)).hexdigest()
    assert record["schema"] == (
        "tnlm-v3-phase1-heldout-identification-analysis-v1"
    )
    assert record["scope"] == (
        "exact_binding_v1_identifiability_analysis_not_language_or_model_evidence"
    )
    claims = record["claims"]
    assert claims[
        "declared_universal_key_value_operator_identities_identify_missing_behavior"
    ]
    assert not claims["training_data_alone_identifies_missing_direction"]
    identification = record["identification"]
    assert identification["full_feature_rank"] == 21
    assert identification["train_feature_rank"] == 20
    assert identification["feature_nullity"] == 1
    assert identification["data_alone_recovers_heldout_direction"] is False
    system = identification["unrestricted_system"]
    assert system["observed_transitions"] == 16_107
    assert system["total_parameter_count"] == 29_967
    assert system["total_constraint_rank"] == 25_055
    assert system["total_nullity"] == 4_912
    pair = identification["pair_completion"]
    assert pair["arbitrary_pair_table_nullity"] == 1
    assert pair["additive_heldout_behavior_nullity"] == 0
    cyclic = identification["cyclic_update_completion"]
    assert cyclic["unrestricted_function_completions"] == 16
    assert len(cyclic["permutation_completions"]) == 2
    assert len(cyclic["transitive_cyclic_completions"]) == 1
