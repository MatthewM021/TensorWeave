"""Emit a deterministic Phase-I exact-algebra analysis record."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from tnlm_v3.campaign_config import load_milestone4_campaign_config
from tnlm_v3.data import BindingEventKind, generate_binding_episode
from tnlm_v3.exact_algebra import (
    canonical_visible_action_count,
    diagnostic_probe_rank,
    full_homogeneous_dimension,
    lane_permutation_quotient_state_count,
    oracle_lane_state_count,
    promised_query_realization_upper_bound,
    replay_episode,
    semantic_state_count,
    strict_grammar_rank_certificate,
    train_semantic_state_count,
)


SCHEMA = "tnlm-v3-phase1-exact-algebra-analysis-v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _parse_lengths(raw: str, *, minimum: int, maximum: int) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("replay lengths must be integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one replay length is required")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("replay lengths must be unique")
    if any(length < minimum or length > maximum for length in values):
        raise argparse.ArgumentTypeError(
            f"replay lengths must be in [{minimum}, {maximum}]"
        )
    return tuple(sorted(values))


def build_analysis_record(
    config_path: Path,
    *,
    replay_lengths: Sequence[int],
    replay_seed: int,
    max_states: int,
) -> dict[str, object]:
    """Build a deterministic record from a strict campaign config."""

    config_bytes = config_path.read_bytes()
    config = load_milestone4_campaign_config(config_path)
    task = config.task
    full_rank = diagnostic_probe_rank(task, max_states=max_states)
    train_rank = diagnostic_probe_rank(
        task, exclude_heldout=True, max_states=max_states
    )
    strict_certificate = strict_grammar_rank_certificate(
        task, max_states=max_states
    )

    replays: list[dict[str, object]] = []
    for split_index, split in enumerate(("train", "eval")):
        for document_index, length in enumerate(replay_lengths):
            episode = generate_binding_episode(
                task,
                length=length,
                seed=replay_seed + split_index,
                split=split,
                document_index=document_index,
            )
            final_state = replay_episode(task, episode)
            replays.append(
                {
                    "document_id": episode.document_id,
                    "event_count": episode.length,
                    "final_state": list(final_state.values),
                    "length": length,
                    "query_count": int(
                        (
                            episode.inputs.event_kinds
                            == int(BindingEventKind.QUERY)
                        ).sum()
                    ),
                    "split": split,
                }
            )

    task_analysis: dict[str, object] = {
        "contract_schema": "tnlm-v3-binding-exact-algebra-v1",
        "task_fingerprint": task.fingerprint(),
        "num_surface_keys": task.num_surface_keys,
        "value_cardinality": task.value_cardinality,
        "branches": task.branches,
        "max_live_bindings": task.max_live_bindings,
        "encoded_vocab_size": task.vocab_size,
        "supported_visible_actions": canonical_visible_action_count(task),
        "full_syntactic_visible_actions": canonical_visible_action_count(
            task, full_syntactic=True
        ),
        "semantic_states": semantic_state_count(task),
        "train_semantic_states": train_semantic_state_count(task),
        "raw_oracle_lane_states": oracle_lane_state_count(task),
        "train_raw_oracle_lane_states": oracle_lane_state_count(
            task, exclude_heldout=True
        ),
        "lane_quotient_states": lane_permutation_quotient_state_count(task),
        "absence_aware_diagnostic_rank": full_rank.rational_rank,
        "train_absence_aware_diagnostic_rank": train_rank.rational_rank,
        "diagnostic_rank_agrees_across_fields": (
            full_rank.agrees_across_fields and train_rank.agrees_across_fields
        ),
        "homogeneous_dimension": full_homogeneous_dimension(task),
        "promised_query_dimension_upper_bound": (
            promised_query_realization_upper_bound(task)
        ),
        "strict_grammar_states_with_sink": semantic_state_count(task) + 1,
        "strict_grammar_hankel_rank": strict_certificate.rank,
        "strict_grammar_rank_certificate": asdict(strict_certificate),
    }

    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope": "exact_binding_v1_task_analysis_not_language_or_model_evidence",
        "campaign_id": config.campaign_id,
        "campaign_stage": config.stage.value,
        "config_file_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "task": task_analysis,
        "episode_replays": replays,
        "claims": {
            "absence_aware_dimension_is_minimal_for_declared_diagnostic": True,
            "natural_language_has_finite_rank": False,
            "promised_query_dimension_is_only_an_upper_bound": True,
            "strict_rank_uses_observable_sink_and_state_diagnostics": True,
        },
    }
    return {
        **body,
        "record_sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
    }


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
    default_config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "milestone4"
        / "validation_screen_v1.yaml"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-lengths", default="16,64,256,2048")
    parser.add_argument("--replay-seed", type=int, default=31001)
    parser.add_argument("--max-states", type=int, default=1_000_000)
    arguments = parser.parse_args(argv)
    config = load_milestone4_campaign_config(arguments.config)
    lengths = _parse_lengths(
        arguments.replay_lengths,
        minimum=config.task.min_length,
        maximum=config.task.max_length,
    )
    if arguments.replay_seed < 0:
        parser.error("--replay-seed must be nonnegative")
    if arguments.max_states < 1:
        parser.error("--max-states must be positive")
    record = build_analysis_record(
        arguments.config,
        replay_lengths=lengths,
        replay_seed=arguments.replay_seed,
        max_states=arguments.max_states,
    )
    encoded = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        _write_atomic(arguments.output, encoded.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
