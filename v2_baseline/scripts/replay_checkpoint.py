#!/usr/bin/env python3
"""Replay a stored V2 checkpoint on its exact deterministic test split.

The utility supports both the main benchmark schema and the supplementary
low-rank control schema included in this bundle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from tnlm_v2.data import build_task
from tnlm_v2.factory import create_model
from tnlm_v2.training import evaluate_model


def _row_value(row: pd.Series, name: str, default: Any = None) -> Any:
    if name not in row.index or pd.isna(row[name]):
        return default
    return row[name]


def replay_checkpoint(
    results: Path,
    run_key: str,
    length: int,
    device: str = "cpu",
    batch_size: int = 256,
    test_samples: int | None = None,
    active_branches: int | None = None,
) -> dict[str, Any]:
    """Re-evaluate one checkpoint and compare it with the stored metrics row."""
    results = results.resolve()
    table = pd.read_csv(results / "metrics.csv")
    rows = table[(table["run_key"] == run_key) & (table["eval_length"] == length)]
    if len(rows) != 1:
        raise ValueError(
            f"expected one metrics row for run={run_key!r}, length={length}; "
            f"found {len(rows)}"
        )
    row = rows.iloc[0]
    checkpoint_path = results / str(row["checkpoint"])
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    settings = dict(checkpoint.get("model_settings", {}))
    inferred_branches = active_branches
    if inferred_branches is None:
        inferred_branches = _row_value(row, "active_branches_eval")
    if inferred_branches is None:
        inferred_branches = checkpoint.get("active_branches_eval")
    if inferred_branches is None:
        inferred_branches = settings.get("branches")
    if inferred_branches is None:
        inferred_branches = checkpoint.get("max_branches", 8)
    inferred_branches = int(inferred_branches)

    inferred_samples = test_samples
    if inferred_samples is None:
        inferred_samples = _row_value(row, "test_samples")
    if inferred_samples is None:
        inferred_samples = checkpoint.get("test_samples", 1024)
    inferred_samples = int(inferred_samples)

    task = build_task(str(checkpoint["task"]), inferred_branches)
    model = create_model(
        str(checkpoint["model"]),
        task.spec,
        int(checkpoint["max_length"]),
        settings,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    data_seed_test = int(_row_value(row, "data_seed_test"))
    test = task.generate(
        inferred_samples,
        int(row["eval_length"]),
        data_seed_test,
        inferred_branches,
    )
    replayed = evaluate_model(
        model, test, batch_size=batch_size, device=device
    )

    accuracy_column = "eval_accuracy" if "eval_accuracy" in row.index else "accuracy"
    loss_column = "eval_loss" if "eval_loss" in row.index else "loss"
    recorded_accuracy = float(row[accuracy_column])
    recorded_loss = float(row[loss_column])
    payload = {
        "run_key": run_key,
        "checkpoint": str(Path(str(row["checkpoint"]))),
        "results_directory": results.name,
        "eval_length": int(row["eval_length"]),
        "test_samples": inferred_samples,
        "active_branches": inferred_branches,
        "data_seed_test": data_seed_test,
        "recorded": {
            "accuracy": recorded_accuracy,
            "loss": recorded_loss,
        },
        "replayed": {
            "accuracy": float(replayed["accuracy"]),
            "loss": float(replayed["loss"]),
        },
        "absolute_difference": {
            "accuracy": abs(float(replayed["accuracy"]) - recorded_accuracy),
            "loss": abs(float(replayed["loss"]) - recorded_loss),
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/reference_cpu"))
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--active-branches", type=int)
    args = parser.parse_args()

    try:
        payload = replay_checkpoint(
            results=args.results,
            run_key=args.run_key,
            length=args.length,
            device=args.device,
            batch_size=args.batch_size,
            test_samples=args.test_samples,
            active_branches=args.active_branches,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
