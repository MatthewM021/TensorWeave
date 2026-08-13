#!/usr/bin/env python3
"""Replay a set of stored V2 checkpoints and write a verification manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from replay_checkpoint import replay_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--run-keys", nargs="*")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    results = args.results.resolve()
    metrics = pd.read_csv(results / "metrics.csv")
    available = metrics[metrics["eval_length"] == args.length]
    if args.run_keys:
        run_keys = list(args.run_keys)
    else:
        run_keys = sorted(str(x) for x in available["run_key"].unique())

    replays = []
    for run_key in run_keys:
        payload = replay_checkpoint(
            results=results,
            run_key=run_key,
            length=args.length,
            device=args.device,
        )
        payload["passed"] = bool(
            payload["absolute_difference"]["accuracy"] <= args.tolerance
            and payload["absolute_difference"]["loss"] <= args.tolerance
        )
        replays.append(payload)
        print(
            f"{run_key}: accuracy_diff="
            f"{payload['absolute_difference']['accuracy']:.3e}, loss_diff="
            f"{payload['absolute_difference']['loss']:.3e}"
        )

    report = {
        "status": "passed" if all(x["passed"] for x in replays) else "failed",
        "results": results.name,
        "eval_length": args.length,
        "tolerance": args.tolerance,
        "run_count": len(replays),
        "replays": replays,
    }
    output = args.output or (results / "replay_verification.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
