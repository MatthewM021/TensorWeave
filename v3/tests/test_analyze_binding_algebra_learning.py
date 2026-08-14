from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _cli_environment(v3_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    source_root = str(v3_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_root, environment.get("PYTHONPATH", ""))
        if value
    )
    return environment


def _run_cli(
    script: Path,
    v3_root: Path,
    *arguments: str,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=v3_root.parent,
        env=_cli_environment(v3_root),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_learning_cli_is_deterministic_self_bound_and_writes_output(
    tmp_path: Path,
) -> None:
    v3_root = Path(__file__).resolve().parents[1]
    script = v3_root / "scripts" / "analyze_binding_algebra_learning.py"
    config = v3_root / "configs" / "milestone4" / "validation_screen_v1.yaml"
    outputs = (
        tmp_path / "first" / "record.json",
        tmp_path / "second" / "record.json",
    )
    common_arguments = (
        "--max-states",
        "821",
        "--coverage-seed-count",
        "3",
        "--coverage-document-count",
        "8",
        "--coverage-document-length",
        "64",
    )

    for output in outputs:
        completed = _run_cli(
            script,
            v3_root,
            "--output",
            str(output),
            *common_arguments,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == ""
        assert completed.stderr == ""
        assert output.is_file()
        assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))

    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    record = json.loads(outputs[0].read_text(encoding="utf-8"))
    claimed_sha256 = record.pop("record_sha256")
    assert claimed_sha256 == hashlib.sha256(_canonical_bytes(record)).hexdigest()
    assert record["schema"] == "tnlm-v3-phase2-algebra-learning-analysis-v1"
    assert record["scope"] == (
        "canonical_state_binding_algebra_learnability_analysis_"
        "not_sequence_representation_or_language_evidence"
    )
    assert record["campaign_id"] == "m4-validation-screen-v1"
    assert record["campaign_stage"] == "screen"
    assert record["config_file_sha256"] == hashlib.sha256(
        config.read_bytes()
    ).hexdigest()
    assert record["claims"] == {
        "automatic_sharing_law_selection_performed": False,
        "empirical_sequence_only_tables_learned": False,
        "hard_equivariance_would_be_an_imposed_positive_control": True,
        "heldout_labels_used_for_fit_or_selection": False,
        "oracle_generated_state_table_control_recovers_exact_tables": True,
        "sequence_only_representation_learning_performed": False,
        "supplied_full_shared_prototypes_conditionally_identified": True,
        "unrestricted_passive_recovery_from_current_support": False,
    }

    learnability = record["learnability"]
    assert learnability["schema"] == "tnlm-v3-algebra-learnability-v1"
    assert learnability["heldout_pair"] == [0, 0]
    assert (learnability["full_state_count"], learnability["train_state_count"]) == (
        821,
        708,
    )
    assert (
        learnability["full_feature_rank"],
        learnability["train_feature_rank"],
        learnability["passive_unrestricted_nullity"],
    ) == (21, 20, 4_912)
    assert not learnability["passive_uniform_recovery_possible"]
    assert learnability["zero_support_identical_likelihood_witness_exists"]
    assert learnability["one_separating_membership_probe"] == [
        "BIND(0,0)",
        "QUERY(0)",
    ]
    assert not learnability["automatic_hypothesis_selection_performed"]
    assert not learnability["sequence_only_representation_learning_performed"]

    hypotheses = {
        row["hypothesis"]: (
            row["parameter_count"],
            row["constraint_rank"],
            row["nullity"],
            row["heldout_behavior_identified_conditionally"],
            row["universal_law_supplied"],
        )
        for row in learnability["hypotheses"]
    }
    assert hypotheses == {
        "unrestricted_canonical_operators": (29_967, 25_055, 4_912, False, False),
        "key_local_bind_update_copy_query_blocks": (720, 656, 64, False, False),
        "shared_update_and_copy_prototypes_only": (224, 216, 8, False, True),
        "shared_bind_update_copy_query_prototypes": (96, 96, 0, True, True),
        "per_key_calibrated_cyclic_bind_query_orbits": (40, 40, 0, True, True),
        "shared_calibrated_cyclic_bind_query_orbits": (8, 8, 0, True, True),
    }
    assert all(
        row["canonical_state_supervision_assumed"]
        and not row["sharing_law_selected_from_data"]
        for row in learnability["hypotheses"]
    )

    folds = learnability["pseudoheldout_folds"]
    assert len(folds) == 19
    assert {tuple(row["pseudoheldout_pair"]) for row in folds} == {
        (key, value)
        for key in range(5)
        for value in range(4)
        if (key, value) != (0, 0)
    }
    assert all(
        row["real_heldout_pair"] == [0, 0]
        and row["supplied_shared_parameter_count"] == 96
        and row["supplied_shared_design_rank"] == 96
        and row["supplied_shared_nullity"] == 0
        and row["supplied_shared_identifies_pseudoheldout"]
        and not row["automatic_hypothesis_selection_performed"]
        for row in folds
    )

    coverage = record["finite_seen_only_excitation"]
    coverage_fields = (
        "schema",
        "split",
        "document_count",
        "document_length",
        "seed_count",
        "full_rank_count",
        "minimum_rank",
        "maximum_rank",
        "mean_rank",
        "parameter_count",
        "canonical_state_supervision_assumed",
        "used_heldout_labels_for_fit",
        "used_heldout_labels_for_selection",
    )
    assert {name: coverage[name] for name in coverage_fields} == {
        "schema": "tnlm-v3-shared-prototype-coverage-sweep-v1",
        "split": "train",
        "document_count": 8,
        "document_length": 64,
        "seed_count": 3,
        "full_rank_count": 3,
        "minimum_rank": 96,
        "maximum_rank": 96,
        "mean_rank": 96.0,
        "parameter_count": 96,
        "canonical_state_supervision_assumed": True,
        "used_heldout_labels_for_fit": False,
        "used_heldout_labels_for_selection": False,
    }
    assert [
        (
            row["seed"],
            row["design_rank"],
            row["nullity"],
            row["full_rank"],
            row["sample_sha256"],
        )
        for row in coverage["per_seed"]
    ] == [
        (
            0,
            96,
            0,
            True,
            "c650a969b85ec8d52a5e16d6af5f745c53fcf8de12570ddf75e200fa7ba51542",
        ),
        (
            1,
            96,
            0,
            True,
            "20ca0c73fa8d9d24c76ed9e09e8a3b9170a92ce2c88e16f6922698686e56ab99",
        ),
        (
            2,
            96,
            0,
            True,
            "dd92b101a38df74c4f697d20b196f2b3805d104dbd07b9a67ad7740c275167e4",
        ),
    ]
    assert [row["condition_number"] for row in coverage["per_seed"]] == pytest.approx(
        [3.774917217635375, 3.8078865529319543, 3.027650354097492]
    )

    fitted = record["first_seed_oracle_state_coefficient_control"]
    assert fitted["fit_split"] == "train"
    model = fitted["model"]
    assert model["schema"] == "tnlm-v3-learned-shared-prototype-transducer-v1"
    assert model["supervision_source"] == "sealed_exact_executor_control"
    assert model["universal_key_sharing_supplied"]
    assert not model["cyclic_law_supplied"]
    assert model["canonical_state_supervision_assumed"]
    assert not model["used_heldout_labels_for_fit"]
    assert not model["automatic_hypothesis_selection_performed"]
    assert len(model["cell_outputs"]) == 24
    assert model["coverage"]["observed_cell_count"] == 24
    assert model["coverage"]["total_cell_count"] == 24
    assert model["coverage"]["design_rank"] == 96
    assert model["coverage"]["nullity"] == 0
    assert not model["coverage"]["used_heldout_labels_for_fit"]

    postfit = fitted["postfit_sealed_evaluation"]
    assert postfit == {
        "schema": "tnlm-v3-shared-prototype-fit-evaluation-v1",
        "evaluated_cell_count": 24,
        "exact_cell_count": 24,
        "missing_cell_count": 0,
        "all_value_tables_exact": True,
        "all_valid_programs_exact_given_supplied_state_contract": True,
        "heldout_semantics_used_only_for_postfit_evaluation": True,
    }


def test_learning_cli_keeps_coefficient_control_train_only_when_coverage_is_validation(
    tmp_path: Path,
) -> None:
    v3_root = Path(__file__).resolve().parents[1]
    script = v3_root / "scripts" / "analyze_binding_algebra_learning.py"
    output = tmp_path / "validation-coverage.json"
    completed = _run_cli(
        script,
        v3_root,
        "--output",
        str(output),
        "--max-states",
        "821",
        "--coverage-split",
        "validation",
        "--coverage-seed-count",
        "1",
        "--coverage-document-count",
        "8",
        "--coverage-document-length",
        "64",
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["finite_seen_only_excitation"]["split"] == "validation"
    fitted = record["first_seed_oracle_state_coefficient_control"]
    assert fitted["fit_split"] == "train"
    assert fitted["model"]["coverage"]["split"] == "train"
    assert not fitted["model"]["used_heldout_labels_for_fit"]


@pytest.mark.parametrize(
    "arguments, expected_fragment",
    [
        (("--max-states", "0"), "--max-states must be positive"),
        (("--max-permutations", "0"), "--max-permutations must be positive"),
        (("--coverage-seed-count", "0"), "--coverage-seed-count must be positive"),
        (
            ("--coverage-document-count", "0"),
            "--coverage-document-count must be positive",
        ),
        (
            ("--coverage-document-length", "0"),
            "--coverage-document-length must be positive",
        ),
        (("--coverage-split", "eval"), "invalid choice: 'eval'"),
    ],
)
def test_learning_cli_rejects_invalid_arguments(
    arguments: tuple[str, ...],
    expected_fragment: str,
) -> None:
    v3_root = Path(__file__).resolve().parents[1]
    script = v3_root / "scripts" / "analyze_binding_algebra_learning.py"
    completed = _run_cli(script, v3_root, *arguments, timeout=30.0)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert expected_fragment in completed.stderr
