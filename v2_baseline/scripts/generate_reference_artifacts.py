#!/usr/bin/env python3
"""Regenerate V2 summary tables and plots from a completed metrics.csv.

This script never retrains models. It is intentionally deterministic and keeps all
reported conclusions traceable to the raw benchmark rows.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHANCE_BY_TASK = {
    "interleaved_threads": 0.25,
    "permuted_hierarchy": 0.5,
    "predictive_detail": 0.5,
    "combined_language": 0.5,
}


LABELS = {
    "mps": "MPS",
    "fixed_ttn": "Fixed TTN",
    "fixed_mera": "Fixed MERA",
    "routed_ttn_oracle": "Routed TTN (oracle)",
    "routed_ttn_oracle_widecore": "Routed TTN (wide core)",
    "routed_ttn_learned": "Routed TTN (learned)",
    "routed_mera_oracle": "Routed MERA (oracle)",
    "routed_mera_learned": "Routed MERA (learned)",
    "routed_ttn_oracle_adaptive": "Routed TTN (adaptive)",
    "gru": "GRU",
    "transformer": "Transformer",
}


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def _row(df: pd.DataFrame, study: str, model: str, length: int) -> pd.Series:
    hit = df[
        (df["study"] == study)
        & (df["model"] == model)
        & (df["eval_length"] == length)
    ]
    if len(hit) != 1:
        raise ValueError(
            f"expected exactly one row for study={study!r}, model={model!r}, "
            f"length={length}; found {len(hit)}"
        )
    return hit.iloc[0]


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _bar_accuracy(
    df: pd.DataFrame,
    study: str,
    path: Path,
    title: str,
    models: Iterable[str] | None = None,
) -> None:
    data = df[(df["study"] == study) & (df["eval_length"] == df["train_length"])].copy()
    if models is not None:
        order = list(models)
        data = data[data["model"].isin(order)]
        data["_order"] = data["model"].map({name: i for i, name in enumerate(order)})
        data = data.sort_values("_order")
    else:
        data = data.sort_values("eval_accuracy", ascending=False)
    labels = [LABELS.get(x, x) for x in data["model"]]
    plt.figure(figsize=(max(8.0, 0.82 * len(labels)), 5.1))
    plt.bar(np.arange(len(data)), data["eval_accuracy"])
    task_name = str(data.iloc[0]["task"])
    plt.axhline(CHANCE_BY_TASK.get(task_name, 0.5), linestyle="--", label="Chance")
    plt.xticks(np.arange(len(data)), labels, rotation=28, ha="right")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.title(title)
    plt.legend()
    _save_figure(path)


def _length_plot(
    df: pd.DataFrame,
    study: str,
    models: Iterable[str],
    path: Path,
    title: str,
    ylabel: str = "Accuracy",
    value_column: str = "eval_accuracy",
) -> None:
    plt.figure(figsize=(8.0, 5.1))
    for model in models:
        data = df[(df["study"] == study) & (df["model"] == model)].sort_values(
            "eval_length"
        )
        if data.empty:
            continue
        plt.plot(
            data["eval_length"],
            data[value_column],
            marker="o",
            label=LABELS.get(model, model),
        )
    plt.xlabel("Nominal context length")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if value_column == "eval_accuracy":
        plt.ylim(0.0, 1.05)
    _save_figure(path)


def generate(results: Path) -> dict:
    metrics_path = results / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    df = pd.read_csv(metrics_path)
    failures_path = results / "failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
    sweep_path = results.parent / "low_rank_seed_sweep" / "metrics.csv"
    lucky_path = results.parent / "low_rank_reference_seed_control" / "metrics.csv"
    low_rank_sweep = pd.read_csv(sweep_path) if sweep_path.exists() else None
    low_rank_lucky = pd.read_csv(lucky_path) if lucky_path.exists() else None

    tables = results / "tables"
    plots = results / "plots"
    tables.mkdir(exist_ok=True)
    plots.mkdir(exist_ok=True)

    in_distribution = df[df["eval_length"] == df["train_length"]].copy()
    in_distribution = in_distribution[
        [
            "study",
            "model",
            "eval_length",
            "eval_accuracy",
            "eval_loss",
            "parameter_count",
            "training_seconds",
            "runtime_inference_tokens_per_second",
            "eval_route_accuracy_aligned",
            "struct_effective_rank",
        ]
    ].rename(
        columns={
            "eval_length": "length",
            "eval_accuracy": "accuracy",
            "eval_loss": "loss",
            "parameter_count": "parameters",
            "training_seconds": "training_s",
            "runtime_inference_tokens_per_second": "tokens_per_s",
            "eval_route_accuracy_aligned": "route_accuracy",
            "struct_effective_rank": "effective_rank",
        }
    )
    in_distribution = in_distribution.sort_values(
        ["study", "accuracy"], ascending=[True, False]
    )
    in_distribution.to_csv(tables / "in_distribution.csv", index=False)

    if low_rank_sweep is not None and low_rank_lucky is not None:
        ordinary = low_rank_sweep[low_rank_sweep["eval_length"] == 32].copy()
        ordinary["control_set"] = "three_paired_seeds"
        lucky = low_rank_lucky[low_rank_lucky["eval_length"] == 32].copy()
        lucky["control_set"] = "paired_reference_initialization"
        controls = pd.concat([ordinary, lucky], ignore_index=True)
        controls[
            [
                "control_set",
                "seed",
                "variant",
                "accuracy",
                "loss",
                "parameters",
                "data_seed_train",
                "data_seed_validation",
                "data_seed_test",
            ]
        ].to_csv(tables / "low_rank_paired_controls.csv", index=False)

    headline_keys = [
        ("interleaved_tournament", "routed_ttn_oracle", 32),
        ("interleaved_tournament", "transformer", 32),
        ("hierarchy_topology", "routed_ttn_oracle", 32),
        ("hierarchy_topology", "fixed_ttn", 32),
        ("detail_disentangler", "routed_ttn_oracle", 32),
        ("detail_disentangler", "routed_mera_oracle", 32),
        ("combined_stress", "fixed_ttn", 64),
    ]
    headline_rows = []
    for study, model, length in headline_keys:
        r = _row(df, study, model, length)
        headline_rows.append(
            {
                "study": study,
                "model": model,
                "accuracy": float(r["eval_accuracy"]),
                "loss": float(r["eval_loss"]),
                "parameters": int(r["parameter_count"]),
                "tokens_per_s": float(r["runtime_inference_tokens_per_second"]),
            }
        )
    pd.DataFrame(headline_rows).to_csv(tables / "headline.csv", index=False)

    i_ttn_32 = _row(df, "interleaved_tournament", "routed_ttn_oracle", 32)
    i_tr_32 = _row(df, "interleaved_tournament", "transformer", 32)
    i_ttn_128 = _row(df, "interleaved_tournament", "routed_ttn_oracle", 128)
    i_tr_128 = _row(df, "interleaved_tournament", "transformer", 128)
    low_ttn = _row(df, "detail_low_rank", "routed_ttn_oracle", 32)
    low_ttn_wide = _row(df, "detail_low_rank", "routed_ttn_oracle_widecore", 32)
    low_mera = _row(df, "detail_low_rank", "routed_mera_oracle", 32)
    summary = {
        "campaign_run_count": int(df["run_key"].nunique()),
        "campaign_row_count": int(len(df)),
        "campaign_failure_count": int(len(failures)),
        "interleaved_oracle_accuracy_32": float(i_ttn_32["eval_accuracy"]),
        "interleaved_transformer_accuracy_32": float(i_tr_32["eval_accuracy"]),
        "hierarchy_oracle_accuracy_32": float(
            _row(df, "hierarchy_topology", "routed_ttn_oracle", 32)["eval_accuracy"]
        ),
        "hierarchy_fixed_ttn_accuracy_32": float(
            _row(df, "hierarchy_topology", "fixed_ttn", 32)["eval_accuracy"]
        ),
        "detail_oracle_ttn_accuracy_32": float(
            _row(df, "detail_disentangler", "routed_ttn_oracle", 32)["eval_accuracy"]
        ),
        "detail_oracle_mera_accuracy_32": float(
            _row(df, "detail_disentangler", "routed_mera_oracle", 32)["eval_accuracy"]
        ),
        "unpaired_lowrank_ttn_accuracy_32": float(low_ttn["eval_accuracy"]),
        "unpaired_lowrank_ttn_widecore_accuracy_32": float(low_ttn_wide["eval_accuracy"]),
        "unpaired_lowrank_mera_accuracy_32": float(low_mera["eval_accuracy"]),
        "unpaired_lowrank_ttn_parameters": int(low_ttn["parameter_count"]),
        "unpaired_lowrank_ttn_widecore_parameters": int(low_ttn_wide["parameter_count"]),
        "unpaired_lowrank_mera_parameters": int(low_mera["parameter_count"]),
        "unpaired_lowrank_mera_gain_vs_ttn": float(
            low_mera["eval_accuracy"] - low_ttn["eval_accuracy"]
        ),
        "unpaired_lowrank_mera_gain_vs_widecore": float(
            low_mera["eval_accuracy"] - low_ttn_wide["eval_accuracy"]
        ),
        "learned_router_interleaved_route_accuracy": float(
            _row(df, "interleaved_tournament", "routed_ttn_learned", 32)[
                "eval_route_accuracy_aligned"
            ]
        ),
        "learned_router_hierarchy_route_accuracy": float(
            _row(df, "hierarchy_topology", "routed_ttn_learned", 32)[
                "eval_route_accuracy_aligned"
            ]
        ),
        "learned_router_detail_route_accuracy": float(
            _row(df, "detail_disentangler", "routed_ttn_learned", 32)[
                "eval_route_accuracy_aligned"
            ]
        ),
        "adaptive_effective_rank": float(
            _row(df, "adaptive_rank", "routed_ttn_oracle_adaptive", 32)[
                "struct_effective_rank"
            ]
        ),
        "interleaved_routed_ttn_tokens_s_32": float(
            i_ttn_32["runtime_inference_tokens_per_second"]
        ),
        "interleaved_transformer_tokens_s_32": float(
            i_tr_32["runtime_inference_tokens_per_second"]
        ),
        "interleaved_routed_ttn_tokens_s_128": float(
            i_ttn_128["runtime_inference_tokens_per_second"]
        ),
        "interleaved_transformer_tokens_s_128": float(
            i_tr_128["runtime_inference_tokens_per_second"]
        ),
        "runtime_ratio_ttn_to_transformer_32": float(
            i_ttn_32["runtime_inference_tokens_per_second"]
            / i_tr_32["runtime_inference_tokens_per_second"]
        ),
        "runtime_ratio_ttn_to_transformer_128": float(
            i_ttn_128["runtime_inference_tokens_per_second"]
            / i_tr_128["runtime_inference_tokens_per_second"]
        ),
    }
    if low_rank_sweep is not None and low_rank_lucky is not None:
        sweep32 = low_rank_sweep[low_rank_sweep["eval_length"] == 32]
        lucky32 = low_rank_lucky[low_rank_lucky["eval_length"] == 32]
        for variant in ("ttn_chi4_rank4", "ttn_chi4_rank9", "mera_chi4_rank4"):
            values = sweep32[sweep32["variant"] == variant]["accuracy"]
            if len(values) != 3:
                raise ValueError(f"expected three paired-seed rows for {variant}")
            summary[f"paired_seed_sweep_{variant}_accuracy_mean"] = float(values.mean())
            summary[f"paired_seed_sweep_{variant}_accuracy_std"] = float(values.std(ddof=1))
            hit = lucky32[lucky32["variant"] == variant]
            if len(hit) != 1:
                raise ValueError(f"expected one paired reference-seed row for {variant}")
            summary[f"paired_reference_seed_{variant}_accuracy"] = float(hit.iloc[0]["accuracy"])
        summary["paired_seed_sweep_run_count"] = int(low_rank_sweep["run_key"].nunique())
        summary["paired_reference_control_run_count"] = int(low_rank_lucky["run_key"].nunique())
        summary["total_research_run_count"] = (
            summary["campaign_run_count"]
            + summary["paired_seed_sweep_run_count"]
            + summary["paired_reference_control_run_count"]
        )
        smoke_metrics_path = results.parent / "smoke" / "metrics.csv"
        summary["smoke_run_count"] = (
            int(pd.read_csv(smoke_metrics_path)["run_key"].nunique())
            if smoke_metrics_path.exists()
            else 0
        )
        summary["total_included_run_count"] = (
            summary["total_research_run_count"] + summary["smoke_run_count"]
        )
        summary["low_rank_disentangler_conclusion"] = (
            "not replicated: the original unpaired-seed gap was an optimization-seed artifact"
        )

    memory_path = results / "memory_benchmark.csv"
    memory = pd.read_csv(memory_path) if memory_path.exists() else None
    if memory is not None and not memory.empty:
        def memory_value(model: str, length: int, column: str) -> float:
            hit = memory[(memory["model"] == model) & (memory["eval_length"] == length)]
            if len(hit) != 1:
                raise ValueError(
                    f"expected one memory row for model={model!r}, length={length}; "
                    f"found {len(hit)}"
                )
            return float(hit.iloc[0][column])

        summary.update(
            {
                "memory_batch_size": int(memory.iloc[0]["batch_size"]),
                "interleaved_ttn_process_max_rss_mib_32": memory_value(
                    "routed_ttn_oracle", 32, "process_max_rss_mib"
                ),
                "interleaved_transformer_process_max_rss_mib_32": memory_value(
                    "transformer", 32, "process_max_rss_mib"
                ),
                "interleaved_ttn_process_max_rss_mib_128": memory_value(
                    "routed_ttn_oracle", 128, "process_max_rss_mib"
                ),
                "interleaved_transformer_process_max_rss_mib_128": memory_value(
                    "transformer", 128, "process_max_rss_mib"
                ),
                "interleaved_ttn_incremental_peak_mib_128": memory_value(
                    "routed_ttn_oracle", 128, "incremental_measured_peak_mib"
                ),
                "interleaved_transformer_incremental_peak_mib_128": memory_value(
                    "transformer", 128, "incremental_measured_peak_mib"
                ),
            }
        )
        summary["memory_process_rss_reduction_mib_128"] = (
            summary["interleaved_transformer_process_max_rss_mib_128"]
            - summary["interleaved_ttn_process_max_rss_mib_128"]
        )
        summary["memory_incremental_transformer_to_ttn_ratio_128"] = (
            summary["interleaved_transformer_incremental_peak_mib_128"]
            / max(1e-12, summary["interleaved_ttn_incremental_peak_mib_128"])
        )

    _json_write(results / "summary.json", summary)

    _bar_accuracy(
        df,
        "interleaved_tournament",
        plots / "interleaved_tournament_accuracy.png",
        "Interleaved state threads at the training length",
    )
    _bar_accuracy(
        df,
        "hierarchy_topology",
        plots / "hierarchy_topology_accuracy.png",
        "Permuted hierarchy at the training length",
    )
    _bar_accuracy(
        df,
        "detail_disentangler",
        plots / "detail_disentangler_accuracy.png",
        "Predictive detail at the training length",
    )
    stale = plots / "low_rank_disentangler.png"
    if stale.exists():
        stale.unlink()
    if low_rank_sweep is not None and low_rank_lucky is not None:
        variants = ["ttn_chi4_rank4", "ttn_chi4_rank9", "mera_chi4_rank4"]
        variant_labels = {
            "ttn_chi4_rank4": "TTN, chi=4, R=4",
            "ttn_chi4_rank9": "TTN, chi=4, R=9",
            "mera_chi4_rank4": "MERA, chi=4, R=4",
        }
        ordinary = low_rank_sweep[low_rank_sweep["eval_length"] == 32]
        lucky = low_rank_lucky[low_rank_lucky["eval_length"] == 32]
        xlabels = ["seed 4101", "seed 4102", "seed 4103", "reference init"]
        plt.figure(figsize=(9.0, 5.2))
        for variant in variants:
            data = ordinary[ordinary["variant"] == variant].sort_values("seed")
            values = data["accuracy"].tolist()
            values.append(float(lucky[lucky["variant"] == variant].iloc[0]["accuracy"]))
            plt.plot(xlabels, values, marker="o", label=variant_labels[variant])
        plt.axhline(0.5, linestyle="--", label="Chance")
        plt.ylabel("Accuracy at length 32")
        plt.ylim(0.4, 1.02)
        plt.title("Paired seeds remove the apparent low-rank MERA advantage")
        plt.legend()
        _save_figure(plots / "low_rank_paired_seed_control.png")

        means = ordinary.groupby("variant")["accuracy"].mean().reindex(variants)
        stds = ordinary.groupby("variant")["accuracy"].std().reindex(variants)
        plt.figure(figsize=(8.0, 5.0))
        plt.bar(
            [variant_labels[v] for v in variants],
            means.values,
            yerr=stds.values,
            capsize=5,
        )
        plt.axhline(0.5, linestyle="--", label="Chance")
        plt.ylabel("Mean accuracy over three paired seeds")
        plt.ylim(0.4, 0.62)
        plt.title("No replicated disentangler benefit at chi=4")
        plt.xticks(rotation=20, ha="right")
        plt.legend()
        _save_figure(plots / "low_rank_paired_seed_mean.png")

    _length_plot(
        df,
        "interleaved_tournament",
        ["routed_ttn_oracle", "routed_mera_oracle", "transformer", "gru"],
        plots / "interleaved_tournament_length.png",
        "Interleaved threads: context-length extrapolation",
    )
    _length_plot(
        df,
        "detail_disentangler",
        ["routed_ttn_oracle", "routed_mera_oracle", "transformer", "gru"],
        plots / "detail_disentangler_length.png",
        "Predictive detail: context-length extrapolation",
    )
    _bar_accuracy(
        df,
        "detail_low_rank",
        plots / "low_rank_unpaired_reference.png",
        "Unpaired-seed low-rank reference result (not a causal ablation)",
        [
            "routed_ttn_oracle",
            "routed_ttn_oracle_widecore",
            "routed_mera_oracle",
        ],
    )
    _length_plot(
        df,
        "interleaved_tournament",
        ["routed_ttn_oracle", "fixed_ttn", "transformer", "gru"],
        plots / "interleaved_runtime.png",
        "Interleaved threads: measured unfused CPU throughput",
        ylabel="Tokens per second",
        value_column="runtime_inference_tokens_per_second",
    )

    route_rows = []
    for study, model, display in [
        ("interleaved_tournament", "routed_ttn_learned", "Interleaved"),
        ("hierarchy_topology", "routed_ttn_learned", "Hierarchy"),
        ("detail_disentangler", "routed_ttn_learned", "Detail"),
    ]:
        r = _row(df, study, model, 32)
        route_rows.append((display, float(r["eval_route_accuracy_aligned"])))
    plt.figure(figsize=(7.0, 4.8))
    plt.bar([x[0] for x in route_rows], [x[1] for x in route_rows])
    plt.axhline(1.0 / 8.0, linestyle="--", label="Random assignment")
    plt.ylabel("Permutation-aligned route accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("Learned routing remains the dominant bottleneck")
    plt.legend()
    _save_figure(plots / "learned_router_accuracy.png")

    if memory is not None and not memory.empty:
        selected = ["routed_ttn_oracle", "routed_mera_oracle", "transformer", "gru"]
        plt.figure(figsize=(8.0, 5.1))
        for model in selected:
            data = memory[memory["model"] == model].sort_values("eval_length")
            plt.plot(
                data["eval_length"],
                data["process_max_rss_mib"],
                marker="o",
                label=LABELS.get(model, model),
            )
        plt.xlabel("Context length")
        plt.ylabel("Process maximum RSS (MiB)")
        plt.title("Isolated CPU inference memory, batch 64")
        plt.legend()
        _save_figure(plots / "interleaved_peak_rss.png")

        plt.figure(figsize=(8.0, 5.1))
        for model in selected:
            data = memory[memory["model"] == model].sort_values("eval_length")
            plt.plot(
                data["eval_length"],
                data["incremental_measured_peak_mib"],
                marker="o",
                label=LABELS.get(model, model),
            )
        plt.xlabel("Context length")
        plt.ylabel("Incremental measured peak (MiB)")
        plt.title("Forward-workspace RSS above warmed process")
        plt.legend()
        _save_figure(plots / "interleaved_incremental_memory.png")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/reference_cpu"),
        help="Completed benchmark directory containing metrics.csv",
    )
    args = parser.parse_args()
    summary = generate(args.results.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
