"""Run the retrospective Phase-II trace-supervised algebra experiment.

The coefficient estimator receives only visible structured events and direct
TRAIN query answers.  This script is the trusted benchmark controller: it
constructs semantic occupancy/dependency attestations for pseudoheldout fold
creation, freezes selection, and only then opens the declared outer-cell probe
suite.  The result is evidence about learning inside a supplied register
transducer family, not representation discovery or a confirmatory claim.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from tnlm_v3.algebra_discovery import (
    SequenceAlgebraSelectionResult,
    TraceSupervisedCorpus,
    make_trace_supervised_corpus,
    make_trace_supervised_sequence,
    select_sequence_algebra,
    visible_sequence_from_episode,
)
from tnlm_v3.algebra_discovery_probes import (
    SealedProbeEvaluation,
    build_balanced_probe_suite,
    cyclic_cell_rotation_inventory,
    evaluate_probe_suite,
    evaluate_shortcut_controls,
)
from tnlm_v3.campaign_config import load_milestone4_campaign_config
from tnlm_v3.data import (
    BindingEpisode,
    BindingEventKind,
    BindingTaskConfig,
    apply_value_transform,
    generate_binding_episodes,
)


SCHEMA = "tnlm-v3-phase2-trace-algebra-experiment-v1"
MAX_DOCUMENTS_PER_SPLIT = 512
MAX_DOCUMENT_LENGTH = 2_048
MAX_TOTAL_EVENTS = 2_000_000
MAX_SELECTION_SCORED_EVENT_WORK = 4_000_000_000


def _plain_int(name: str, value: object, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _state_cells(values: dict[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(values.items()))


def attest_episode_for_fold_controller(
    episode: BindingEpisode,
    task: BindingTaskConfig,
    *,
    forbidden_outer_cell: tuple[int, int],
):
    """Create controller-only state/dependency metadata from visible semantics.

    This is deliberately a trusted semantic audit.  Its outputs are used only
    to construct and score folds; ``select_sequence_algebra`` strips them
    before every estimator call.  Dependency sets retain prior cells across
    UPDATE and COPY so queries after leaving a pseudoheldout cell remain in the
    pseudo-dependent validation score.
    """

    if type(episode) is not BindingEpisode:
        raise TypeError("episode must be exact BindingEpisode")
    if type(task) is not BindingTaskConfig:
        raise TypeError("task must be exact BindingTaskConfig")
    if (
        not isinstance(forbidden_outer_cell, tuple)
        or len(forbidden_outer_cell) != 2
        or any(type(value) is not int for value in forbidden_outer_cell)
    ):
        raise TypeError("forbidden_outer_cell must be an exact integer pair")

    sequence = visible_sequence_from_episode(episode)
    values: dict[int, int] = {}
    dependencies: dict[int, set[tuple[int, int]]] = {}
    pre_event_cells: list[tuple[tuple[int, int], ...]] = []
    post_event_cells: list[tuple[tuple[int, int], ...]] = []
    query_dependency_cells: list[tuple[tuple[int, int], ...]] = []

    for event in sequence.events:
        pre_event_cells.append(_state_cells(values))
        key = event.primary_key
        if event.kind is BindingEventKind.BIND:
            values[key] = event.argument
            dependencies[key] = {(key, event.argument)}
        elif event.kind is BindingEventKind.UPDATE:
            old_value = values[key]
            new_value = apply_value_transform(
                old_value, event.argument, task.value_cardinality
            )
            values[key] = new_value
            dependencies[key] = set(dependencies[key]) | {
                (key, old_value),
                (key, new_value),
            }
        elif event.kind is BindingEventKind.COPY:
            source = event.secondary_key
            copied_value = values[source]
            values[key] = copied_value
            dependencies[key] = set(dependencies[source]) | {(key, copied_value)}
        elif event.kind is BindingEventKind.INVALIDATE:
            del values[key]
            del dependencies[key]

        post_event_cells.append(_state_cells(values))
        if event.kind is BindingEventKind.QUERY:
            query_dependency_cells.append(tuple(sorted(dependencies[key])))
        else:
            query_dependency_cells.append(())

    exposed = {
        cell
        for rows in (pre_event_cells, post_event_cells, query_dependency_cells)
        for row in rows
        for cell in row
    }
    if forbidden_outer_cell in exposed:
        raise ValueError("a direct seen-split episode exposes the outer heldout cell")
    return make_trace_supervised_sequence(
        sequence,
        split=episode.split,
        pre_event_cells=pre_event_cells,
        post_event_cells=post_event_cells,
        query_dependency_cells=query_dependency_cells,
        num_surface_keys=task.num_surface_keys,
        value_cardinality=task.value_cardinality,
    )


def build_trace_corpus(
    task: BindingTaskConfig,
    *,
    outer_cell: tuple[int, int],
    train_seed: int,
    validation_seed: int,
    document_count: int,
    document_length: int,
) -> TraceSupervisedCorpus:
    """Generate direct TRAIN/validation traces and bind their fold audit."""

    _plain_int("train_seed", train_seed, 0)
    _plain_int("validation_seed", validation_seed, 0)
    count = _plain_int("document_count", document_count, 1)
    length = _plain_int("document_length", document_length, task.min_length)
    if count > MAX_DOCUMENTS_PER_SPLIT:
        raise ValueError("document_count exceeds the controller budget")
    if length > min(task.max_length, MAX_DOCUMENT_LENGTH):
        raise ValueError("document_length exceeds the controller budget")
    if 2 * count * length > MAX_TOTAL_EVENTS:
        raise ValueError("requested corpus exceeds the total-event budget")

    traces = []
    for split, seed in (("train", train_seed), ("validation", validation_seed)):
        episodes = generate_binding_episodes(
            task,
            count=count,
            seed=seed,
            split=split,
            lengths=(length,) * count,
        )
        traces.extend(
            attest_episode_for_fold_controller(
                episode,
                task,
                forbidden_outer_cell=outer_cell,
            )
            for episode in episodes
        )
    return make_trace_supervised_corpus(
        task.num_surface_keys,
        task.value_cardinality,
        traces,
    )


def _evaluation_summary(evaluation: SealedProbeEvaluation) -> dict[str, object]:
    return {
        "protocol_status": evaluation.protocol_status.value,
        "suite_sha256": evaluation.suite_sha256,
        "evaluation_sha256": getattr(evaluation, "evaluation_sha256", None),
        "query_count": evaluation.query_count,
        "correct_count": evaluation.correct_count,
        "accuracy": evaluation.accuracy,
        "focal_query_count": evaluation.focal_query_count,
        "focal_correct_count": evaluation.focal_correct_count,
        "focal_accuracy": evaluation.focal_accuracy,
        "macro_family_accuracy": evaluation.macro_family_accuracy,
        "macro_family_focal_accuracy": evaluation.macro_family_focal_accuracy,
        "path_consistency": evaluation.path_consistency,
        "families": _jsonable(evaluation.family_results),
        "pairs": _jsonable(evaluation.pair_results),
        "path_relations": _jsonable(evaluation.equivalence_results),
    }


def _source_hashes(v3_root: Path) -> dict[str, str]:
    paths = (
        v3_root / "src" / "tnlm_v3" / "algebra_discovery.py",
        v3_root / "src" / "tnlm_v3" / "algebra_discovery_probes.py",
        Path(__file__).resolve(),
    )
    return {
        path.relative_to(v3_root).as_posix(): _sha256_bytes(path.read_bytes())
        for path in paths
    }


def build_experiment_record(
    config_path: Path,
    *,
    train_seed: int = 17,
    validation_seed: int = 23,
    document_count: int = 32,
    document_length: int = 64,
    residual_penalties: Sequence[int] = (0, 1, 4, 16),
    optimizer_seed: int = 0,
    restart_count: int = 2,
    max_sweeps: int = 4,
) -> dict[str, object]:
    """Fit/select once, then open actual and balanced rotated-cell probes."""

    if not isinstance(config_path, Path):
        raise TypeError("config_path must be pathlib.Path")
    config_bytes = config_path.read_bytes()
    config = load_milestone4_campaign_config(config_path)
    if len(config.task.heldout_key_value_pairs) != 1:
        raise ValueError("the experiment requires exactly one outer heldout cell")
    outer_cell = config.task.heldout_key_value_pairs[0]
    corpus = build_trace_corpus(
        config.task,
        outer_cell=outer_cell,
        train_seed=train_seed,
        validation_seed=validation_seed,
        document_count=document_count,
        document_length=document_length,
    )
    selection: SequenceAlgebraSelectionResult = select_sequence_algebra(
        corpus,
        residual_penalties=residual_penalties,
        seed=optimizer_seed,
        restart_count=restart_count,
        max_sweeps=max_sweeps,
        max_aggregate_scored_event_work=MAX_SELECTION_SCORED_EVENT_WORK,
    )

    actual_suite = build_balanced_probe_suite(
        config.task.num_surface_keys,
        config.task.value_cardinality,
        (outer_cell,),
    )
    actual_evaluation = evaluate_probe_suite(selection.final_model, actual_suite)
    shortcut_controls = evaluate_shortcut_controls(actual_suite)
    cell_rotations = cyclic_cell_rotation_inventory(
        config.task.num_surface_keys,
        config.task.value_cardinality,
        anchor_key=outer_cell[0],
    )
    rotated_suite = build_balanced_probe_suite(
        config.task.num_surface_keys,
        config.task.value_cardinality,
        (outer_cell,),
        cell_rotations=cell_rotations,
    )
    rotated_evaluation = evaluate_probe_suite(
        selection.final_model, rotated_suite
    )
    outer_pair_result = rotated_evaluation.result_for_pair(outer_cell)

    v3_root = Path(__file__).resolve().parents[1]
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope": (
            "retrospective_trace_supervised_transition_table_selection_in_a_"
            "supplied_register_transducer_not_representation_discovery"
        ),
        "campaign_id": config.campaign_id,
        "config_file_sha256": _sha256_bytes(config_bytes),
        "source_file_sha256": _source_hashes(v3_root),
        "outer_cell": list(outer_cell),
        "outer_probe_labels_opened_only_after_selection": True,
        "outer_identifier_known_to_trusted_controller_before_selection": True,
        "data": {
            "regime": "passive_binding_generator_direct_train_and_validation",
            "train_seed": train_seed,
            "validation_seed": validation_seed,
            "document_count_per_split": document_count,
            "document_length": document_length,
            "trace_corpus_sha256": corpus.corpus_sha256,
        },
        "selection": _jsonable(selection),
        "sealed_actual_cell": {
            "suite_sha256": actual_suite.suite_sha256,
            "case_count": len(actual_suite.cases),
            "evaluation": _evaluation_summary(actual_evaluation),
            "shortcut_controls": [
                {
                    "name": row.name,
                    "evaluation": _evaluation_summary(row.evaluation),
                }
                for row in shortcut_controls
            ],
        },
        "balanced_rotated_cell_control": {
            "suite_sha256": rotated_suite.suite_sha256,
            "case_count": len(rotated_suite.cases),
            "actual_cell_result": _jsonable(outer_pair_result),
            "evaluation": _evaluation_summary(rotated_evaluation),
        },
        "claims": {
            "passive_trace_supervision_used": True,
            "query_labels_only_used_by_coefficient_estimator": True,
            "oracle_semantic_metadata_used_by_fold_controller": True,
            "outer_identifier_used_by_trusted_generation_and_fold_audit": True,
            "outer_identifier_received_by_coefficient_estimator": False,
            "outer_identifier_used_for_candidate_ordering": False,
            "outer_heldout_labels_used_for_fit_or_selection": False,
            "supplied_addressable_register_representation": True,
            "transition_coefficients_learned_from_traces": True,
            "residual_strength_selected_by_seen_only_validation": True,
            "representation_discovery_performed": False,
            "assumption_free_algebra_discovery_performed": False,
            "retrospective_protocol_rehearsal": True,
            "confirmatory_claim_permitted": False,
        },
    }
    return {**body, "record_sha256": _sha256_bytes(_canonical_bytes(body))}


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    v3_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=v3_root / "configs" / "milestone4" / "validation_screen_v1.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-seed", type=int, default=17)
    parser.add_argument("--validation-seed", type=int, default=23)
    parser.add_argument("--document-count", type=int, default=32)
    parser.add_argument("--document-length", type=int, default=64)
    parser.add_argument("--residual-penalties", type=int, nargs="+", default=(0, 1, 4, 16))
    parser.add_argument("--optimizer-seed", type=int, default=0)
    parser.add_argument("--restart-count", type=int, default=2)
    parser.add_argument("--max-sweeps", type=int, default=4)
    arguments = parser.parse_args(argv)
    try:
        record = build_experiment_record(
            arguments.config,
            train_seed=arguments.train_seed,
            validation_seed=arguments.validation_seed,
            document_count=arguments.document_count,
            document_length=arguments.document_length,
            residual_penalties=tuple(arguments.residual_penalties),
            optimizer_seed=arguments.optimizer_seed,
            restart_count=arguments.restart_count,
            max_sweeps=arguments.max_sweeps,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    encoded = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        _write_atomic(arguments.output, encoded.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
