from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from .data import build_task
from .factory import MODEL_NAMES, create_model
from .training import (
    TrainConfig,
    benchmark_inference,
    evaluate_model,
    set_reproducible_seed,
    train_model,
)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
        + "\n"
    )
    temp.replace(path)


def _merge(base, override=None):
    result = dict(base)
    if override:
        result.update(override)
    return result


def _train_config(mapping):
    allowed = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(mapping) - allowed
    if unknown:
        raise KeyError(f"unknown train fields: {sorted(unknown)}")
    return TrainConfig(**dict(mapping))


def _active(study, length, default):
    mapping = study.get("active_branches_by_length", {})
    return int(mapping.get(str(length), mapping.get(length, study.get("active_branches", default))))


def _flatten(prefix, mapping):
    return {
        prefix + k: v
        for k, v in mapping.items()
        if isinstance(v, (int, float, bool, str)) or v is None
    }


def run_benchmark(config_path, output_directory, only_studies=None, only_models=None, resume=True):
    config_path = Path(config_path).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "runs").mkdir(exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    config = yaml.safe_load(config_path.read_text())
    (output / "executed_config.yaml").write_text(config_path.read_text())
    base_seed = int(config.get("seed", 17))
    device = config.get("device", "cpu")
    max_length = int(config.get("max_length", 128))
    model_defaults = dict(config.get("model_settings", {}))
    train_defaults = dict(config.get("train", {}))
    inf = dict(config.get("inference", {}))
    allowed_studies = set(only_studies or [])
    allowed_models = set(only_models or [])
    metrics_path = output / "metrics.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if resume and metrics_path.exists() else []
    completed = {str(x["run_key"]) for x in rows if "run_key" in x}
    failures_path = output / "failures.json"
    failures = (
        json.loads(failures_path.read_text())
        if resume and failures_path.exists()
        else []
    )
    manifest = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "system": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "runs": [
            {"run_key": run_key, "status": "completed"}
            for run_key in sorted(completed)
        ],
    }
    _write_json(output / "manifest.json", manifest)

    for si, study in enumerate(config["studies"]):
        sid = str(study["id"])
        if allowed_studies and sid not in allowed_studies:
            continue
        task = build_task(str(study["task"]), int(study.get("max_branches", 8)))
        train_length = int(study["train_length"])
        active_train = _active(study, train_length, task.spec.max_branches)
        train_data_seed = base_seed + 10000 * si + 11
        validation_data_seed = base_seed + 10000 * si + 23
        train_data = task.generate(
            int(study.get("train_samples", 2048)),
            train_length,
            train_data_seed,
            active_train,
        )
        val_data = task.generate(
            int(study.get("validation_samples", 512)),
            train_length,
            validation_data_seed,
            active_train,
        )
        study_model_cfg = _merge(model_defaults, study.get("model_settings"))
        study_train_cfg = _merge(train_defaults, study.get("train"))
        overrides = dict(study.get("model_overrides", {}))
        for mi, model_name in enumerate(study["models"]):
            if model_name not in MODEL_NAMES:
                raise KeyError(model_name)
            if allowed_models and model_name not in allowed_models:
                continue
            seed = base_seed + 10000 * si + 100 * mi + 7
            run_key = f"{sid}__{model_name}__seed{seed}"
            run_file = output / "runs" / f"{run_key}.json"
            if resume and run_key in completed and run_file.exists():
                continue
            override = dict(overrides.get(model_name, {}))
            model_cfg = _merge(study_model_cfg, override.get("model_settings"))
            train_cfg_map = _merge(study_train_cfg, override.get("train"))
            train_cfg_map["seed"] = seed
            train_cfg = _train_config(train_cfg_map)
            set_reproducible_seed(seed, train_cfg.num_threads)
            model = create_model(model_name, task.spec, max_length, model_cfg)
            run_payload = {
                "run_key": run_key,
                "study": sid,
                "task": task.spec.name,
                "model": model_name,
                "seed": seed,
                "data_seed_train": train_data_seed,
                "data_seed_validation": validation_data_seed,
                "status": "running",
                "model_settings": model_cfg,
                "train_settings": dataclasses.asdict(train_cfg),
            }
            _write_json(run_file, run_payload)
            try:
                training = train_model(model, train_data, val_data, train_cfg, device)
                ckpt = output / "checkpoints" / f"{run_key}.pt"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "task": task.spec.name,
                        "model": model_name,
                        "model_settings": model_cfg,
                        "max_length": max_length,
                        "training": training.as_dict(),
                        "run_key": run_key,
                        "seed": seed,
                        "data_seed_train": train_data_seed,
                        "data_seed_validation": validation_data_seed,
                    },
                    ckpt,
                )
                run_rows = []
                for ei, length in enumerate(study.get("eval_lengths", [train_length])):
                    length = int(length)
                    active_eval = _active(study, length, task.spec.max_branches)
                    test_data_seed = (
                        base_seed
                        + 10000 * si
                        + 100 * mi
                        + 1000000 * (ei + 1)
                    )
                    test = task.generate(
                        int(study.get("test_samples", 1024)),
                        length,
                        test_data_seed,
                        active_eval,
                    )
                    ev = evaluate_model(model, test, int(inf.get("evaluation_batch_size", 256)), device)
                    rt = benchmark_inference(
                        model,
                        test,
                        int(inf.get("batch_size", 64)),
                        int(inf.get("warmup_batches", 2)),
                        int(inf.get("measured_batches", 5)),
                        device,
                    )
                    structural = model.structural_metrics(length)
                    last = training.history[-1]
                    row = {
                        "run_key": run_key,
                        "study": sid,
                        "task": task.spec.name,
                        "model": model_name,
                        "runtime_model_name": model.model_name,
                        "seed": seed,
                        "data_seed_train": train_data_seed,
                        "data_seed_validation": validation_data_seed,
                        "data_seed_test": test_data_seed,
                        "train_length": train_length,
                        "eval_length": length,
                        "active_branches_train": active_train,
                        "active_branches_eval": active_eval,
                        "train_samples": len(train_data),
                        "validation_samples": len(val_data),
                        "test_samples": len(test),
                        "valid_tokens_eval_mean": float(test.valid_mask.sum(1).float().mean()),
                        "best_epoch": training.best_epoch,
                        "best_validation_loss": training.best_validation_loss,
                        "best_validation_accuracy": training.best_validation_accuracy,
                        "training_seconds": training.total_seconds,
                        "training_examples_per_second_last_epoch": last.examples_per_second,
                        "training_tokens_per_second_last_epoch": last.tokens_per_second,
                        "parameter_count": model.parameter_count,
                        "checkpoint": str(ckpt.relative_to(output)),
                    }
                    row.update(_flatten("eval_", ev))
                    row.update(_flatten("runtime_", rt))
                    row.update(_flatten("struct_", structural))
                    run_rows.append(row)
                rows.extend(run_rows)
                pd.DataFrame(rows).to_csv(metrics_path, index=False)
                pd.DataFrame(rows).to_json(output / "metrics.json", orient="records", indent=2)
                run_payload.update(
                    {
                        "status": "completed",
                        "training": training.as_dict(),
                        "checkpoint": str(ckpt.relative_to(output)),
                        "metrics": run_rows,
                    }
                )
                _write_json(run_file, run_payload)
                completed.add(run_key)
                manifest["runs"].append({"run_key": run_key, "status": "completed"})
            except Exception as exc:
                failure = {
                    "run_key": run_key,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                failures.append(failure)
                run_payload.update({"status": "failed", "failure": failure})
                _write_json(run_file, run_payload)
                manifest["runs"].append({"run_key": run_key, "status": "failed"})
                if config.get("fail_fast", False):
                    raise
            _write_json(output / "failures.json", failures)
            _write_json(output / "manifest.json", manifest)
    manifest["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["row_count"] = len(rows)
    manifest["run_count"] = len({str(row["run_key"]) for row in rows if "run_key" in row})
    manifest["failure_count"] = len(failures)
    manifest["status"] = "completed" if not failures else "completed_with_failures"
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "failures.json", failures)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--study", action="append", default=[])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    print(
        run_benchmark(
            args.config,
            args.output,
            args.study or None,
            args.model or None,
            not args.no_resume,
        )
    )


if __name__ == "__main__":
    main()
