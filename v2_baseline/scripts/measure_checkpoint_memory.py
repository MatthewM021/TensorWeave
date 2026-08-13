#!/usr/bin/env python3
"""Measure process-wide CPU inference memory in isolated checkpoint workers.

The parent process launches a fresh Python worker for each model/length pair so
Linux maximum-RSS values are not contaminated by earlier models. The reported
numbers include the Python/PyTorch runtime, model, test batch, and all forward
workspaces; the warmed-RSS and measured incremental peak are also retained.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import psutil
import torch

from tnlm_v2.data import build_task
from tnlm_v2.factory import create_model


def _tensor_bytes(batch) -> int:
    total = 0
    for tensor in [batch.tokens, batch.valid_mask, batch.routes, batch.labels, *batch.metadata.values()]:
        total += tensor.numel() * tensor.element_size()
    return int(total)


def _worker(args: argparse.Namespace) -> None:
    results = args.results.resolve()
    table = pd.read_csv(results / "metrics.csv")
    rows = table[
        (table["run_key"] == args.run_key)
        & (table["eval_length"] == args.length)
    ]
    if len(rows) != 1:
        raise SystemExit(f"expected one row, found {len(rows)}")
    row = rows.iloc[0]
    checkpoint_path = results / str(row["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    max_branches = int(
        checkpoint.get("model_settings", {}).get(
            "branches", max(int(row["active_branches_train"]), int(row["active_branches_eval"]))
        )
    )
    task = build_task(str(checkpoint["task"]), max_branches)
    model = create_model(
        str(checkpoint["model"]),
        task.spec,
        int(checkpoint["max_length"]),
        checkpoint["model_settings"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    torch.set_num_threads(max(1, args.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    data = task.generate(
        max(args.batch_size, 1),
        int(row["eval_length"]),
        int(row["data_seed_test"]),
        int(row["active_branches_eval"]),
    )
    sample = data.slice(torch.arange(min(args.batch_size, len(data))))
    routes = sample.routes if "_oracle" in model.model_name else None
    process = psutil.Process(os.getpid())
    loaded_rss = process.memory_info().rss

    with torch.inference_mode():
        for _ in range(max(1, args.warmup_passes)):
            model(sample.tokens, sample.valid_mask, routes)
    gc.collect()
    warmed_rss = process.memory_info().rss

    peak = [warmed_rss]
    stop = threading.Event()

    def sample_rss() -> None:
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], process.memory_info().rss)
            except psutil.Error:
                break
            time.sleep(0.0005)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max(1, args.measured_passes)):
            model(sample.tokens, sample.valid_mask, routes)
    elapsed = time.perf_counter() - start
    stop.set()
    sampler.join(timeout=1.0)
    measured_peak = max(peak[0], process.memory_info().rss)
    max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)

    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    payload = {
        "run_key": args.run_key,
        "study": str(row["study"]),
        "model": str(row["model"]),
        "runtime_model_name": str(row["runtime_model_name"]),
        "eval_length": int(row["eval_length"]),
        "batch_size": int(len(sample)),
        "passes": int(max(1, args.measured_passes)),
        "loaded_rss_bytes": int(loaded_rss),
        "warmed_rss_bytes": int(warmed_rss),
        "measured_peak_rss_bytes": int(measured_peak),
        "resource_max_rss_bytes": int(max_rss),
        "process_max_rss_bytes": int(max(max_rss, measured_peak, warmed_rss, loaded_rss)),
        "incremental_measured_peak_bytes": int(max(0, measured_peak - warmed_rss)),
        "parameter_bytes": int(parameter_bytes),
        "buffer_bytes": int(buffer_bytes),
        "batch_tensor_bytes": _tensor_bytes(sample),
        "seconds_per_batch": float(elapsed / max(1, args.measured_passes)),
        "tokens_per_second": float(
            len(sample)
            * int(row["eval_length"])
            * max(1, args.measured_passes)
            / elapsed
        ),
    }
    print(json.dumps(payload, sort_keys=True))


def _parent(args: argparse.Namespace) -> None:
    results = args.results.resolve()
    table = pd.read_csv(results / "metrics.csv")
    data = table[table["study"] == args.study].copy()
    if args.models:
        data = data[data["model"].isin(args.models)]
    if args.lengths:
        data = data[data["eval_length"].isin(args.lengths)]
    data = data.sort_values(["eval_length", "model"])
    if data.empty:
        raise SystemExit("no matching benchmark rows")

    records = []
    script = Path(__file__).resolve()
    env = dict(os.environ)
    project_src = str(script.parents[1] / "src")
    env["PYTHONPATH"] = project_src + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    for row in data.itertuples(index=False):
        command = [
            sys.executable,
            str(script),
            "--worker",
            "--results",
            str(results),
            "--run-key",
            str(row.run_key),
            "--length",
            str(int(row.eval_length)),
            "--batch-size",
            str(args.batch_size),
            "--warmup-passes",
            str(args.warmup_passes),
            "--measured-passes",
            str(args.measured_passes),
            "--threads",
            str(args.threads),
        ]
        output = subprocess.check_output(command, text=True, env=env)
        record = json.loads(output.strip().splitlines()[-1])
        records.append(record)
        print(
            f"{record['model']:34s} L={record['eval_length']:3d} "
            f"maxRSS={record['process_max_rss_bytes']/2**20:8.2f} MiB"
        )

    frame = pd.DataFrame(records)
    for source, target in [
        ("loaded_rss_bytes", "loaded_rss_mib"),
        ("warmed_rss_bytes", "warmed_rss_mib"),
        ("measured_peak_rss_bytes", "measured_peak_rss_mib"),
        ("resource_max_rss_bytes", "resource_max_rss_mib"),
        ("process_max_rss_bytes", "process_max_rss_mib"),
        ("incremental_measured_peak_bytes", "incremental_measured_peak_mib"),
        ("parameter_bytes", "parameter_mib"),
        ("buffer_bytes", "buffer_mib"),
        ("batch_tensor_bytes", "batch_tensor_mib"),
    ]:
        frame[target] = frame[source] / 2**20
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    output_path.with_suffix(".json").write_text(
        frame.to_json(orient="records", indent=2) + "\n"
    )
    manifest = {
        "method": (
            "fresh subprocess per checkpoint/length; psutil RSS sampled during "
            "warmed forward passes; process maximum is max(sampled RSS, ru_maxrss)"
        ),
        "system": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "psutil": psutil.__version__,
        },
        "study": args.study,
        "models": list(args.models),
        "lengths": list(args.lengths),
        "batch_size": args.batch_size,
        "warmup_passes": args.warmup_passes,
        "measured_passes": args.measured_passes,
        "threads": args.threads,
        "row_count": len(frame),
    }
    output_path.with_name(output_path.stem + "_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/reference_cpu"))
    parser.add_argument("--study", default="interleaved_tournament")
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--lengths", nargs="*", type=int, default=[])
    parser.add_argument("--output", type=Path, default=Path("results/reference_cpu/memory_benchmark.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-passes", type=int, default=2)
    parser.add_argument("--measured-passes", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-key", help=argparse.SUPPRESS)
    parser.add_argument("--length", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.run_key or args.length is None:
            raise SystemExit("worker requires --run-key and --length")
        _worker(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()
