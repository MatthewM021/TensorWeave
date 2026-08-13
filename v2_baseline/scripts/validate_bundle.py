#!/usr/bin/env python3
"""Check the internal integrity of the completed V2 evidence bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_TASKS = {
    "interleaved_threads",
    "permuted_hierarchy",
    "predictive_detail",
    "combined_language",
}
REQUIRED_MAIN_MODELS = {
    "mps",
    "fixed_ttn",
    "fixed_mera",
    "routed_ttn_oracle",
    "routed_ttn_learned",
    "routed_mera_oracle",
    "routed_mera_learned",
    "routed_ttn_oracle_adaptive",
    "routed_ttn_oracle_widecore",
    "gru",
    "transformer",
}
REQUIRED_PLOTS = {
    "interleaved_tournament_accuracy.png",
    "hierarchy_topology_accuracy.png",
    "detail_disentangler_accuracy.png",
    "low_rank_unpaired_reference.png",
    "low_rank_paired_seed_control.png",
    "low_rank_paired_seed_mean.png",
    "interleaved_tournament_length.png",
    "detail_disentangler_length.png",
    "interleaved_runtime.png",
    "learned_router_accuracy.png",
    "interleaved_peak_rss.png",
    "interleaved_incremental_memory.png",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"name": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def validate_result_directory(
    root: Path,
    relative: str,
    expected_runs: int,
    expected_rows: int,
    checks: list[dict[str, Any]],
) -> pd.DataFrame:
    directory = root / relative
    metrics = pd.read_csv(directory / "metrics.csv")
    manifest = json.loads((directory / "manifest.json").read_text())
    check(len(metrics) == expected_rows, f"{relative}: expected row count", checks)
    check(
        metrics["run_key"].nunique() == expected_runs,
        f"{relative}: expected run count",
        checks,
    )
    check(
        int(manifest.get("row_count", expected_rows)) == expected_rows,
        f"{relative}: manifest row count",
        checks,
    )
    check(
        int(manifest.get("run_count", expected_runs)) == expected_runs,
        f"{relative}: manifest run count",
        checks,
    )
    for run_key in sorted(str(x) for x in metrics["run_key"].unique()):
        check(
            (directory / "runs" / f"{run_key}.json").exists(),
            f"{relative}: run payload {run_key}",
            checks,
        )
    for checkpoint in sorted(str(x) for x in metrics["checkpoint"].unique()):
        check(
            (directory / checkpoint).exists(),
            f"{relative}: checkpoint {checkpoint}",
            checks,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    checks: list[dict[str, Any]] = []

    main_metrics = validate_result_directory(
        root, "results/reference_cpu", 32, 87, checks
    )
    sweep_metrics = validate_result_directory(
        root, "results/low_rank_seed_sweep", 9, 18, checks
    )
    lucky_metrics = validate_result_directory(
        root, "results/low_rank_reference_seed_control", 3, 6, checks
    )
    validate_result_directory(root, "results/smoke", 13, 13, checks)

    check(
        set(main_metrics["task"].unique()) == REQUIRED_TASKS,
        "main campaign contains all four tasks",
        checks,
    )
    check(
        REQUIRED_MAIN_MODELS.issubset(set(main_metrics["model"].unique())),
        "main campaign contains required model families",
        checks,
    )
    failures = json.loads((root / "results/reference_cpu/failures.json").read_text())
    check(not failures, "main campaign has no recorded failures", checks)

    main_manifest = json.loads(
        (root / "results/reference_cpu/manifest.json").read_text()
    )
    executed = root / "results/reference_cpu/executed_config.yaml"
    check(
        main_manifest["config_sha256"] == sha256(executed),
        "main executed-config hash matches manifest",
        checks,
    )

    check(
        set(sweep_metrics["seed"].unique()) == {4101, 4102, 4103},
        "paired sweep contains all declared seeds",
        checks,
    )
    for seed, group in sweep_metrics[sweep_metrics["eval_length"] == 32].groupby("seed"):
        check(
            group["data_seed_train"].nunique() == 1
            and group["data_seed_validation"].nunique() == 1
            and group["data_seed_test"].nunique() == 1,
            f"paired sweep seed {seed}: shared data splits",
            checks,
        )
        check(
            set(group["variant"]) == {
                "ttn_chi4_rank4",
                "ttn_chi4_rank9",
                "mera_chi4_rank4",
            },
            f"paired sweep seed {seed}: all variants",
            checks,
        )

    lucky32 = lucky_metrics[lucky_metrics["eval_length"] == 32]
    check(
        lucky32["seed"].nunique() == 1
        and int(lucky32["seed"].iloc[0]) == 20290919,
        "reference-initialization control uses seed 20290919",
        checks,
    )
    check(
        lucky32["data_seed_train"].nunique() == 1
        and lucky32["data_seed_validation"].nunique() == 1
        and lucky32["data_seed_test"].nunique() == 1,
        "reference-initialization control uses shared data splits",
        checks,
    )

    for relative, expected in [
        ("results/reference_cpu/replay_verification.json", 3),
        ("results/low_rank_seed_sweep/replay_verification.json", 9),
        ("results/low_rank_reference_seed_control/replay_verification.json", 3),
    ]:
        replay = json.loads((root / relative).read_text())
        check(replay["status"] == "passed", f"{relative}: replay status", checks)
        check(int(replay["run_count"]) == expected, f"{relative}: replay count", checks)

    summary = json.loads((root / "results/reference_cpu/summary.json").read_text())
    check(
        int(summary["total_research_run_count"]) == 44,
        "summary accounts for 44 research runs",
        checks,
    )
    check(
        int(summary["smoke_run_count"]) == 13
        and int(summary["total_included_run_count"]) == 57,
        "summary accounts separately for 13 smoke runs",
        checks,
    )
    check(
        summary["low_rank_disentangler_conclusion"].startswith("not replicated"),
        "summary records corrected low-rank conclusion",
        checks,
    )

    plot_names = {
        p.name for p in (root / "results/reference_cpu/plots").glob("*.png")
    }
    check(REQUIRED_PLOTS.issubset(plot_names), "all required plots exist", checks)
    check(
        (root / "results/reference_cpu/tables/low_rank_paired_controls.csv").exists(),
        "combined paired-control table exists",
        checks,
    )

    required_docs = [
        "README.md",
        "reports/V2_REPORT.md",
        "reports/V2_ARCHITECTURE.md",
        "results/reference_cpu/README.md",
        "results/low_rank_seed_sweep/README.md",
        "results/low_rank_reference_seed_control/README.md",
    ]
    for relative in required_docs:
        check((root / relative).exists(), f"documentation exists: {relative}", checks)
    check(
        (root / "BUILD_VERIFICATION.json").exists(),
        "build verification manifest exists",
        checks,
    )

    payload = {
        "status": "passed",
        "root": root.name,
        "check_count": len(checks),
        "checks": checks,
        "evidence": {
            "main_runs": 32,
            "main_rows": 87,
            "paired_control_runs": 12,
            "paired_control_rows": 24,
            "research_trained_runs": 44,
            "smoke_runs": 13,
            "total_included_runs": 57,
        },
    }
    output = args.output or (root / "VALIDATION.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"validated {len(checks)} checks: {output}")


if __name__ == "__main__":
    main()
