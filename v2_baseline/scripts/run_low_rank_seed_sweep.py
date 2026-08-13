#!/usr/bin/env python3
"""Replicate the critical chi=4 disentangler ablation across paired seeds."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch

from tnlm_v2.data import build_task
from tnlm_v2.factory import create_model
from tnlm_v2.training import (
    TrainConfig,
    evaluate_model,
    set_reproducible_seed,
    train_model,
)


VARIANTS = {
    "ttn_chi4_rank4": ("routed_ttn_oracle", {"tn_dimension": 4, "tn_rank": 4}),
    "ttn_chi4_rank9": (
        "routed_ttn_oracle_widecore",
        {"tn_dimension": 4, "tn_rank": 9},
    ),
    "mera_chi4_rank4": ("routed_mera_oracle", {"tn_dimension": 4, "tn_rank": 4}),
}


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/low_rank_seed_sweep")
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[4101, 4102, 4103])
    parser.add_argument(
        "--variants", nargs="*", choices=tuple(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument("--train-data-seed", type=int)
    parser.add_argument("--validation-data-seed", type=int)
    parser.add_argument("--test-data-seed-32", type=int)
    parser.add_argument("--test-data-seed-64", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    (output / "runs").mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    metrics_path = output / "metrics.csv"
    rows = (
        pd.read_csv(metrics_path).to_dict("records")
        if metrics_path.exists() and not args.no_resume
        else []
    )
    completed = {str(row["run_key"]) for row in rows}
    task = build_task("predictive_detail", 8)

    for seed in args.seeds:
        train_seed = (
            int(args.train_data_seed)
            if args.train_data_seed is not None
            else seed + 100_000
        )
        validation_seed = (
            int(args.validation_data_seed)
            if args.validation_data_seed is not None
            else seed + 200_000
        )
        test_seeds = {
            32: (
                int(args.test_data_seed_32)
                if args.test_data_seed_32 is not None
                else seed + 300_000
            ),
            64: (
                int(args.test_data_seed_64)
                if args.test_data_seed_64 is not None
                else seed + 400_000
            ),
        }
        train = task.generate(2048, 32, train_seed, 8)
        validation = task.generate(512, 32, validation_seed, 8)
        for variant in args.variants:
            model_name, settings = VARIANTS[variant]
            run_key = f"{variant}__seed{seed}"
            if run_key in completed:
                continue
            set_reproducible_seed(seed, 4)
            model = create_model(model_name, task.spec, 128, settings)
            config = TrainConfig(
                epochs=18,
                batch_size=64,
                learning_rate=0.005,
                weight_decay=0.0001,
                gradient_clip=1.0,
                patience=6,
                min_delta=0.0001,
                orthogonality_weight=0.0001,
                rank_weight=0.002,
                router_entropy_weight=0.01,
                router_balance_weight=0.05,
                num_threads=4,
                seed=seed,
            )
            training = train_model(model, train, validation, config, args.device)
            checkpoint_path = output / "checkpoints" / f"{run_key}.pt"
            torch.save(
                {
                    "run_key": run_key,
                    "variant": variant,
                    "model": model_name,
                    "model_settings": settings,
                    "task": task.spec.name,
                    "max_length": 128,
                    "seed": seed,
                    "data_seed_train": train_seed,
                    "data_seed_validation": validation_seed,
                    "active_branches_eval": 8,
                    "test_samples": 1024,
                    "evaluation_data_seeds": {
                        str(k): int(v) for k, v in test_seeds.items()
                    },
                    "state_dict": model.state_dict(),
                    "training": training.as_dict(),
                },
                checkpoint_path,
            )
            run_rows = []
            for length in (32, 64):
                test = task.generate(1024, length, test_seeds[length], 8)
                metrics = evaluate_model(model, test, 256, args.device)
                run_rows.append(
                    {
                        "run_key": run_key,
                        "seed": seed,
                        "variant": variant,
                        "model": model_name,
                        "tn_dimension": settings["tn_dimension"],
                        "tn_rank": settings["tn_rank"],
                        "eval_length": length,
                        "accuracy": metrics["accuracy"],
                        "loss": metrics["loss"],
                        "parameters": model.parameter_count,
                        "training_seconds": training.total_seconds,
                        "best_epoch": training.best_epoch,
                        "best_validation_accuracy": training.best_validation_accuracy,
                        "data_seed_train": train_seed,
                        "data_seed_validation": validation_seed,
                        "data_seed_test": test_seeds[length],
                        "checkpoint": str(checkpoint_path.relative_to(output)),
                    }
                )
            rows.extend(run_rows)
            completed.add(run_key)
            pd.DataFrame(rows).to_csv(metrics_path, index=False)
            write_json(output / "runs" / f"{run_key}.json", {"metrics": run_rows, "training": training.as_dict()})
            print(
                f"{run_key}: L32={run_rows[0]['accuracy']:.4f}, "
                f"L64={run_rows[1]['accuracy']:.4f}"
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(metrics_path, index=False)
    frame.to_json(output / "metrics.json", orient="records", indent=2)
    in_distribution = frame[frame["eval_length"] == 32]
    summary = (
        in_distribution.groupby("variant")
        .agg(
            seeds=("seed", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            accuracy_min=("accuracy", "min"),
            accuracy_max=("accuracy", "max"),
            loss_mean=("loss", "mean"),
            parameters=("parameters", "first"),
        )
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)
    write_json(output / "summary.json", summary.to_dict("records"))
    write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "seeds": sorted(int(x) for x in frame["seed"].unique()),
            "run_count": int(frame["run_key"].nunique()),
            "row_count": int(len(frame)),
            "variants": sorted(str(x) for x in frame["variant"].unique()),
            "paired_design": (
                "Within each seed, all variants share train/validation/test data and "
                "the same PyTorch initialization seed."
            ),
        },
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
