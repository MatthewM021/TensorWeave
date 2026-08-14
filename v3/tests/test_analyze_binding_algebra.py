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


def test_phase1_analysis_cli_is_deterministic_and_self_bound(tmp_path: Path) -> None:
    v3_root = Path(__file__).resolve().parents[1]
    script = v3_root / "scripts" / "analyze_binding_algebra.py"
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
                "--replay-lengths",
                "16,64",
                "--max-states",
                "822",
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
    assert record["schema"] == "tnlm-v3-phase1-exact-algebra-analysis-v1"
    assert record["scope"] == (
        "exact_binding_v1_task_analysis_not_language_or_model_evidence"
    )
    assert len(record["episode_replays"]) == 4
    task = record["task"]
    assert task["semantic_states"] == 821
    assert task["train_semantic_states"] == 708
    assert task["absence_aware_diagnostic_rank"] == 21
    assert task["train_absence_aware_diagnostic_rank"] == 20
    assert task["promised_query_dimension_upper_bound"] == 16
    assert task["strict_grammar_hankel_rank"] == 192
    assert task["strict_grammar_rank_certificate"]["gf2_lower_bound"] == 192
