import json

import pandas as pd
import yaml

from tnlm_v2.benchmark import run_benchmark


def test_resumed_manifest_preserves_completed_runs(tmp_path):
    config = {
        "seed": 41,
        "device": "cpu",
        "max_length": 8,
        "fail_fast": True,
        "model_settings": {"tn_dimension": 4, "tn_rank": 4},
        "train": {
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.003,
            "patience": 1,
            "num_threads": 1,
        },
        "inference": {
            "evaluation_batch_size": 8,
            "batch_size": 8,
            "warmup_batches": 1,
            "measured_batches": 1,
        },
        "studies": [
            {
                "id": "resume_test",
                "task": "interleaved_threads",
                "train_length": 6,
                "active_branches": 2,
                "train_samples": 16,
                "validation_samples": 8,
                "test_samples": 8,
                "eval_lengths": [6],
                "models": ["fixed_ttn"],
            }
        ],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    output = tmp_path / "results"

    run_benchmark(config_path, output)
    run_benchmark(config_path, output)

    metrics = pd.read_csv(output / "metrics.csv")
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(metrics) == 1
    assert metrics["run_key"].nunique() == 1
    assert manifest["row_count"] == 1
    assert manifest["run_count"] == 1
    assert len(manifest["runs"]) == 1
    assert manifest["failure_count"] == 0
    assert manifest["status"] == "completed"
