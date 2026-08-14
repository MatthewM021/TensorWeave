"""Trace-only learning in a supplied compositional register-machine family.

This module is deliberately narrower than representation discovery.  The
learner is given the event grammar, one addressable register per surface key,
an absent/present flag, a ``V``-symbol register alphabet aligned with query
labels, and the facts that invalidation clears a register and distractors do
nothing.  It learns categorical BIND, UPDATE, and COPY transition tables and
selects the strength of destination-key/prototype-cell residuals.

The coefficient estimator consumes only visible structured sequences and
query answers from a declared split.  A separate trusted controller may use
externally attested pre/post occupancy and causal-query annotations to make
closed pseudoheldout folds.  Those annotations are erased before every fit;
they are oracle metadata for split construction and scoring, not learned
state.  No exact executor is imported or called here.

The current benchmark has already exposed its nominal held-out cell during
development.  Certificates consequently label every run a retrospective
protocol rehearsal and forbid a confirmatory claim.  The API is generic over
the cell absent from each corpus so the same frozen protocol can later be run
as an outer rotation over independently omitted environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Iterable, Sequence

from .data import BindingEpisode, BindingEventKind, IGNORE_QUERY_TARGET


Cell = tuple[int, int]
_SEEN_SPLITS = frozenset(("train", "validation"))


class SequenceDiscoveryLimitError(RuntimeError):
    """Raised before or exactly when a declared discovery budget is exhausted."""


class SequenceSelectionMode(str, Enum):
    """How validation scores are ordered.

    ``PSEUDOHELDOUT_PRIMARY`` is the scientific protocol.  The power-control
    mode instead puts all seen-validation queries first; it can demonstrate
    that the selector detects repeated *observed* local exceptions, but it is
    not evidence that an unseen singleton exception can be predicted.
    """

    PSEUDOHELDOUT_PRIMARY = "pseudoheldout_primary"
    OBSERVED_EXCEPTION_POWER_CONTROL = "observed_exception_power_control"


class TraceAttestationSource(str, Enum):
    """Allowed source for fold-firewall metadata."""

    EXTERNAL_SEMANTIC_AUDIT = "external_semantic_audit"


def _plain_int(name: str, value: object, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _outer_omission_commitment(
    num_surface_keys: int,
    value_cardinality: int,
    cells: Iterable[Cell],
) -> str:
    return _sha256(
        {
            "domain": "tnlm-v3-outer-omission-commitment-v1",
            "num_surface_keys": num_surface_keys,
            "value_cardinality": value_cardinality,
            "cells": [list(cell) for cell in sorted(cells)],
        }
    )


@dataclass(frozen=True, order=True)
class VisibleEvent:
    """One model-visible event with raw, zero-based IDs.

    ``-1`` is the absent field sentinel.  Vocabulary-range checks happen when
    an event is placed in a corpus, because the event itself carries no task
    fingerprint or held-out metadata.
    """

    kind: BindingEventKind
    primary_key: int = -1
    secondary_key: int = -1
    argument: int = -1

    def __post_init__(self) -> None:
        if type(self.kind) is not BindingEventKind:
            raise TypeError("kind must be exact BindingEventKind")
        if self.kind is BindingEventKind.PAD:
            raise ValueError("visible sequences cannot contain PAD")
        for name in ("primary_key", "secondary_key", "argument"):
            _plain_int(name, getattr(self, name), -1)


@dataclass(frozen=True)
class VisibleSequence:
    """Visible event sequence with aligned query supervision.

    Query targets are the only supervised values accepted by the estimator.
    Source IDs, task fingerprints, generation seeds, routes, parents,
    generations, held-out masks, and canonical states have no fields here.
    """

    events: tuple[VisibleEvent, ...]
    query_targets: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple) or not self.events:
            raise ValueError("events must be a nonempty exact tuple")
        if any(type(event) is not VisibleEvent for event in self.events):
            raise TypeError("events must contain exact VisibleEvent values")
        if not isinstance(self.query_targets, tuple):
            raise TypeError("query_targets must be an exact tuple")
        if len(self.events) != len(self.query_targets):
            raise ValueError("events and query_targets must be aligned")
        for event, target in zip(self.events, self.query_targets, strict=True):
            if event.kind is BindingEventKind.QUERY:
                if type(target) is not int:
                    raise TypeError("every QUERY requires an exact integer target")
                if target < 0:
                    raise ValueError("query targets must be nonnegative")
            elif target is not None:
                raise ValueError("non-QUERY events cannot carry targets")

    @property
    def query_count(self) -> int:
        return sum(target is not None for target in self.query_targets)


def _event_payload(event: VisibleEvent) -> list[int]:
    return [
        int(event.kind),
        event.primary_key,
        event.secondary_key,
        event.argument,
    ]


def _sequence_payload(sequence: VisibleSequence) -> dict[str, object]:
    return {
        "events": [_event_payload(event) for event in sequence.events],
        "query_targets": list(sequence.query_targets),
    }


def _validate_event_shape(
    event: VisibleEvent, num_surface_keys: int, value_cardinality: int
) -> None:
    key_ok = 0 <= event.primary_key < num_surface_keys
    source_ok = 0 <= event.secondary_key < num_surface_keys
    value_ok = 0 <= event.argument < value_cardinality
    if event.kind in (BindingEventKind.BIND, BindingEventKind.UPDATE):
        valid = key_ok and event.secondary_key == -1 and value_ok
    elif event.kind is BindingEventKind.COPY:
        valid = (
            key_ok
            and source_ok
            and event.primary_key != event.secondary_key
            and event.argument == -1
        )
    elif event.kind in (BindingEventKind.INVALIDATE, BindingEventKind.QUERY):
        valid = key_ok and event.secondary_key == -1 and event.argument == -1
    elif event.kind is BindingEventKind.DISTRACTOR:
        valid = (
            event.primary_key == -1
            and event.secondary_key == -1
            and event.argument in (0, 1)
        )
    else:
        valid = False
    if not valid:
        raise ValueError("event is outside the supplied visible grammar")


def _validate_presence_grammar(
    sequence: VisibleSequence, num_surface_keys: int, value_cardinality: int
) -> None:
    present = [False] * num_surface_keys
    for event, target in zip(sequence.events, sequence.query_targets, strict=True):
        _validate_event_shape(event, num_surface_keys, value_cardinality)
        if (
            event.kind is BindingEventKind.UPDATE
            and event.argument == value_cardinality - 1
        ):
            raise ValueError(
                "the generator-unsupported identity UPDATE is outside the learned family"
            )
        key = event.primary_key
        if event.kind is BindingEventKind.BIND:
            if present[key]:
                raise ValueError("BIND must address an absent register")
            present[key] = True
        elif event.kind is BindingEventKind.COPY:
            if not present[key] or not present[event.secondary_key]:
                raise ValueError("COPY must address two present registers")
        elif event.kind in (BindingEventKind.UPDATE, BindingEventKind.QUERY):
            if not present[key]:
                raise ValueError(f"{event.kind.name} must address a present register")
        elif event.kind is BindingEventKind.INVALIDATE:
            if not present[key]:
                raise ValueError("INVALIDATE must address a present register")
            present[key] = False
        if target is not None and target >= value_cardinality:
            raise ValueError("query target is outside the value vocabulary")


def _corpus_payload(
    num_surface_keys: int,
    value_cardinality: int,
    split: str,
    sequences: tuple[VisibleSequence, ...],
) -> dict[str, object]:
    return {
        "schema": "tnlm-v3-sequence-corpus-v1",
        "num_surface_keys": num_surface_keys,
        "value_cardinality": value_cardinality,
        "split": split,
        "sequences": [_sequence_payload(sequence) for sequence in sequences],
    }


@dataclass(frozen=True)
class SequenceCorpus:
    """Sanitized estimator/evaluator input.

    The class intentionally cannot encode a held-out ID or any canonical trace
    information.  ``split='validation'`` corpora may be scored by the trusted
    selection controller but are rejected by :func:`fit_sequence_algebra`.
    """

    schema: str
    num_surface_keys: int
    value_cardinality: int
    split: str
    sequences: tuple[VisibleSequence, ...]
    sample_sha256: str

    def __post_init__(self) -> None:
        if self.schema != "tnlm-v3-sequence-corpus-v1":
            raise ValueError("unknown sequence corpus schema")
        _plain_int("num_surface_keys", self.num_surface_keys, 2)
        _plain_int("value_cardinality", self.value_cardinality, 2)
        if self.split not in _SEEN_SPLITS:
            raise ValueError("sequence corpus split must be train or validation")
        if not isinstance(self.sequences, tuple) or not self.sequences:
            raise ValueError("sequences must be a nonempty exact tuple")
        if any(type(sequence) is not VisibleSequence for sequence in self.sequences):
            raise TypeError("corpus must contain exact VisibleSequence values")
        if sum(sequence.query_count for sequence in self.sequences) == 0:
            raise ValueError("sequence corpus must contain at least one query target")
        for sequence in self.sequences:
            _validate_presence_grammar(
                sequence, self.num_surface_keys, self.value_cardinality
            )
        _require_sha256("sample_sha256", self.sample_sha256)
        expected = _sha256(
            _corpus_payload(
                self.num_surface_keys,
                self.value_cardinality,
                self.split,
                self.sequences,
            )
        )
        if self.sample_sha256 != expected:
            raise ValueError("sample_sha256 does not bind the sanitized corpus")

    @property
    def event_count(self) -> int:
        return sum(len(sequence.events) for sequence in self.sequences)

    @property
    def query_count(self) -> int:
        return sum(sequence.query_count for sequence in self.sequences)


def make_sequence_corpus(
    num_surface_keys: int,
    value_cardinality: int,
    *,
    split: str,
    sequences: Sequence[VisibleSequence],
) -> SequenceCorpus:
    """Construct and content-bind a sanitized corpus."""

    _plain_int("num_surface_keys", num_surface_keys, 2)
    _plain_int("value_cardinality", value_cardinality, 2)
    if split not in _SEEN_SPLITS:
        raise ValueError("split must be direct train or direct validation")
    if not isinstance(sequences, Sequence) or not sequences:
        raise ValueError("sequences must be a nonempty sequence")
    rows = tuple(sequences)
    if any(type(sequence) is not VisibleSequence for sequence in rows):
        raise TypeError("sequences must contain exact VisibleSequence values")
    payload = _corpus_payload(num_surface_keys, value_cardinality, split, rows)
    return SequenceCorpus(
        schema="tnlm-v3-sequence-corpus-v1",
        num_surface_keys=num_surface_keys,
        value_cardinality=value_cardinality,
        split=split,
        sequences=rows,
        sample_sha256=_sha256(payload),
    )


def visible_sequence_from_episode(episode: BindingEpisode) -> VisibleSequence:
    """Erase all non-visible episode metadata except declared query labels.

    The function trusts the episode's direct split declaration and rejects
    ``eval``/``test``.  It intentionally never reads task fingerprints,
    document IDs, seeds, token IDs, routes, parents, generations, live counts,
    or held-out masks.  Outer-cell cleanliness must be established separately
    by an :class:`ExternalTraceAttestation`.
    """

    if type(episode) is not BindingEpisode:
        raise TypeError("episode must be exact BindingEpisode")
    if episode.split not in _SEEN_SPLITS:
        raise ValueError("only direct train/validation episodes may be sanitized")
    inputs = episode.inputs
    targets = episode.evaluation.targets
    length = episode.length
    if tuple(inputs.event_kinds.shape) != (length,):
        raise ValueError("episode event_kinds must be one-dimensional")
    if tuple(inputs.primary_key_ids.shape) != (length,):
        raise ValueError("episode primary_key_ids must be one-dimensional")
    if tuple(inputs.secondary_key_ids.shape) != (length,):
        raise ValueError("episode secondary_key_ids must be one-dimensional")
    if tuple(inputs.arguments.shape) != (length,):
        raise ValueError("episode arguments must be one-dimensional")
    if tuple(inputs.valid_mask.shape) != (length,) or not bool(inputs.valid_mask.all()):
        raise ValueError("an unpadded episode must have an all-true valid mask")
    if tuple(targets.shape) != (length,):
        raise ValueError("episode targets must be one-dimensional")
    events: list[VisibleEvent] = []
    query_targets: list[int | None] = []
    for index in range(length):
        try:
            kind = BindingEventKind(int(inputs.event_kinds[index]))
        except (TypeError, ValueError) as error:
            raise ValueError("episode contains an unknown event kind") from error
        event = VisibleEvent(
            kind=kind,
            primary_key=int(inputs.primary_key_ids[index]) - 1,
            secondary_key=int(inputs.secondary_key_ids[index]) - 1,
            argument=int(inputs.arguments[index]) - 1,
        )
        raw_target = int(targets[index])
        if kind is BindingEventKind.QUERY:
            if raw_target == IGNORE_QUERY_TARGET or raw_target < 0:
                raise ValueError("QUERY lacks a declared supervised target")
            query_target: int | None = raw_target
        else:
            if raw_target != IGNORE_QUERY_TARGET:
                raise ValueError("non-QUERY event carries a supervised target")
            query_target = None
        events.append(event)
        query_targets.append(query_target)
    return VisibleSequence(tuple(events), tuple(query_targets))


def sequence_corpus_from_episodes(
    num_surface_keys: int,
    value_cardinality: int,
    episodes: Sequence[BindingEpisode],
    *,
    split: str,
    max_events: int = 1_000_000,
) -> SequenceCorpus:
    """Sanitize direct seen-split episodes without forwarding oracle metadata."""

    _plain_int("max_events", max_events, 1)
    if split not in _SEEN_SPLITS:
        raise ValueError("split must be direct train or direct validation")
    if not isinstance(episodes, Sequence) or not episodes:
        raise ValueError("episodes must be a nonempty sequence")
    rows = tuple(episodes)
    if any(type(episode) is not BindingEpisode for episode in rows):
        raise TypeError("episodes must contain exact BindingEpisode values")
    if any(episode.split != split for episode in rows):
        raise ValueError("every episode must use the declared direct split")
    event_count = sum(episode.length for episode in rows)
    if event_count > max_events:
        raise SequenceDiscoveryLimitError("episode sanitization exceeds max_events")
    sequences = tuple(visible_sequence_from_episode(episode) for episode in rows)
    return make_sequence_corpus(
        num_surface_keys,
        value_cardinality,
        split=split,
        sequences=sequences,
    )


def _normalize_cells(
    name: str,
    cells: object,
    num_surface_keys: int,
    value_cardinality: int,
) -> tuple[Cell, ...]:
    if not isinstance(cells, tuple):
        raise TypeError(f"{name} must be an exact tuple")
    normalized: list[Cell] = []
    for cell in cells:
        if not isinstance(cell, tuple) or len(cell) != 2:
            raise TypeError(f"{name} cells must be exact pairs")
        key, value = cell
        _plain_int(f"{name} key", key, 0)
        _plain_int(f"{name} value", value, 0)
        if key >= num_surface_keys or value >= value_cardinality:
            raise ValueError(f"{name} contains an out-of-vocabulary cell")
        normalized.append((key, value))
    if tuple(sorted(set(normalized))) != tuple(normalized):
        raise ValueError(f"{name} cells must be unique and sorted")
    return tuple(normalized)


def _trace_payload(
    sequence: VisibleSequence,
    split: str,
    pre_event_cells: tuple[tuple[Cell, ...], ...],
    post_event_cells: tuple[tuple[Cell, ...], ...],
    query_dependency_cells: tuple[tuple[Cell, ...], ...],
) -> dict[str, object]:
    return {
        "schema": "tnlm-v3-external-trace-attestation-v1",
        "source": TraceAttestationSource.EXTERNAL_SEMANTIC_AUDIT.value,
        "split": split,
        "sequence": _sequence_payload(sequence),
        "pre_event_cells": [[list(cell) for cell in row] for row in pre_event_cells],
        "post_event_cells": [[list(cell) for cell in row] for row in post_event_cells],
        "query_dependency_cells": [
            [list(cell) for cell in row] for row in query_dependency_cells
        ],
    }


@dataclass(frozen=True)
class ExternalTraceAttestation:
    """Trusted, oracle-audited metadata used only by the fold firewall.

    ``pre_event_cells`` and ``post_event_cells`` enumerate occupied
    ``(key,value)`` cells.  ``query_dependency_cells`` identifies every cell
    on which a query answer causally depends, including cells left earlier by
    UPDATE or COPY.  The estimator never receives this object.
    """

    schema: str
    source: TraceAttestationSource
    split: str
    pre_event_cells: tuple[tuple[Cell, ...], ...]
    post_event_cells: tuple[tuple[Cell, ...], ...]
    query_dependency_cells: tuple[tuple[Cell, ...], ...]
    attestation_sha256: str

    def __post_init__(self) -> None:
        if self.schema != "tnlm-v3-external-trace-attestation-v1":
            raise ValueError("unknown trace-attestation schema")
        if type(self.source) is not TraceAttestationSource:
            raise TypeError("source must be exact TraceAttestationSource")
        if self.source is not TraceAttestationSource.EXTERNAL_SEMANTIC_AUDIT:
            raise ValueError("only an external semantic audit may attest traces")
        if self.split not in _SEEN_SPLITS:
            raise ValueError("trace split must be train or validation")
        for name in (
            "pre_event_cells",
            "post_event_cells",
            "query_dependency_cells",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"{name} must be an exact tuple")
        _require_sha256("attestation_sha256", self.attestation_sha256)


@dataclass(frozen=True)
class TraceSupervisedSequence:
    """Visible sequence plus a controller-only external trace attestation."""

    sequence: VisibleSequence
    attestation: ExternalTraceAttestation

    def __post_init__(self) -> None:
        if type(self.sequence) is not VisibleSequence:
            raise TypeError("sequence must be exact VisibleSequence")
        if type(self.attestation) is not ExternalTraceAttestation:
            raise TypeError("attestation must be exact ExternalTraceAttestation")
        length = len(self.sequence.events)
        for name in (
            "pre_event_cells",
            "post_event_cells",
            "query_dependency_cells",
        ):
            if len(getattr(self.attestation, name)) != length:
                raise ValueError(f"{name} must align with the visible sequence")


def make_trace_supervised_sequence(
    sequence: VisibleSequence,
    *,
    split: str,
    pre_event_cells: Sequence[Sequence[Cell]],
    post_event_cells: Sequence[Sequence[Cell]],
    query_dependency_cells: Sequence[Sequence[Cell]],
    num_surface_keys: int,
    value_cardinality: int,
) -> TraceSupervisedSequence:
    """Build and validate an externally attested semantic trace.

    Validation checks state-shape and presence changes, but deliberately does
    not derive UPDATE or COPY values from their known benchmark formulas.
    """

    if type(sequence) is not VisibleSequence:
        raise TypeError("sequence must be exact VisibleSequence")
    if split not in _SEEN_SPLITS:
        raise ValueError("split must be direct train or direct validation")
    _plain_int("num_surface_keys", num_surface_keys, 2)
    _plain_int("value_cardinality", value_cardinality, 2)
    _validate_presence_grammar(sequence, num_surface_keys, value_cardinality)
    raw_groups = (pre_event_cells, post_event_cells, query_dependency_cells)
    if any(not isinstance(group, Sequence) for group in raw_groups):
        raise TypeError("trace fields must be sequences")
    if any(len(group) != len(sequence.events) for group in raw_groups):
        raise ValueError("trace fields must align with visible events")
    pre = tuple(
        _normalize_cells("pre_event_cells", tuple(row), num_surface_keys, value_cardinality)
        for row in pre_event_cells
    )
    post = tuple(
        _normalize_cells("post_event_cells", tuple(row), num_surface_keys, value_cardinality)
        for row in post_event_cells
    )
    dependencies = tuple(
        _normalize_cells(
            "query_dependency_cells", tuple(row), num_surface_keys, value_cardinality
        )
        for row in query_dependency_cells
    )
    previous_post: tuple[Cell, ...] = ()
    expected_active: set[int] = set()
    prefix_state_cells: set[Cell] = set()
    for index, (event, target, before, after, deps) in enumerate(
        zip(sequence.events, sequence.query_targets, pre, post, dependencies, strict=True)
    ):
        if index == 0 and before:
            raise ValueError("an attested document must begin with an empty state")
        if index and before != previous_post:
            raise ValueError("adjacent attested pre/post states must agree")
        before_map = dict(before)
        after_map = dict(after)
        if len(before_map) != len(before) or len(after_map) != len(after):
            raise ValueError("an attested state cannot occupy two values for one key")
        if set(before_map) != expected_active:
            raise ValueError(
                "attested pre-state keys disagree with the visible presence trace"
            )
        key = event.primary_key
        if event.kind is BindingEventKind.BIND:
            valid_change = key not in before_map and key in after_map
            unchanged = {k: v for k, v in before_map.items() if k != key} == {
                k: v for k, v in after_map.items() if k != key
            }
            if not valid_change or not unchanged:
                raise ValueError("BIND trace has an invalid presence transition")
            expected_active.add(key)
        elif event.kind is BindingEventKind.INVALIDATE:
            valid_change = key in before_map and key not in after_map
            unchanged = {k: v for k, v in before_map.items() if k != key} == {
                k: v for k, v in after_map.items() if k != key
            }
            if not valid_change or not unchanged:
                raise ValueError("INVALIDATE trace has an invalid presence transition")
            expected_active.remove(key)
        elif event.kind in (BindingEventKind.UPDATE, BindingEventKind.COPY):
            if set(before_map) != set(after_map):
                raise ValueError("UPDATE/COPY trace must preserve register presence")
            if any(
                before_map[other] != after_map[other]
                for other in before_map
                if other != key
            ):
                raise ValueError("UPDATE/COPY may change only its destination register")
        else:
            if before != after:
                raise ValueError("QUERY/DISTRACTOR trace must preserve state")
        if set(after_map) != expected_active:
            raise ValueError(
                "attested post-state keys disagree with the visible presence trace"
            )
        prefix_state_cells.update(before)
        prefix_state_cells.update(after)
        if event.kind is BindingEventKind.QUERY:
            if target is None:
                raise RuntimeError("validated QUERY unexpectedly lacks a target")
            if before_map.get(key) != target:
                raise ValueError("query target disagrees with the attested query gauge")
            if (key, target) not in deps:
                raise ValueError("query dependencies must include the queried cell")
            if not set(deps).issubset(prefix_state_cells):
                raise ValueError(
                    "query dependencies cannot refer to cells first occupied later"
                )
        elif deps:
            raise ValueError("only QUERY events may carry dependency cells")
        previous_post = after
    payload = _trace_payload(sequence, split, pre, post, dependencies)
    attestation = ExternalTraceAttestation(
        schema="tnlm-v3-external-trace-attestation-v1",
        source=TraceAttestationSource.EXTERNAL_SEMANTIC_AUDIT,
        split=split,
        pre_event_cells=pre,
        post_event_cells=post,
        query_dependency_cells=dependencies,
        attestation_sha256=_sha256(payload),
    )
    return TraceSupervisedSequence(sequence=sequence, attestation=attestation)


def _trace_corpus_payload(
    num_surface_keys: int,
    value_cardinality: int,
    traces: tuple[TraceSupervisedSequence, ...],
) -> dict[str, object]:
    return {
        "schema": "tnlm-v3-trace-supervised-corpus-v1",
        "num_surface_keys": num_surface_keys,
        "value_cardinality": value_cardinality,
        "traces": [
            {
                "split": trace.attestation.split,
                "sequence": _sequence_payload(trace.sequence),
                "attestation_sha256": trace.attestation.attestation_sha256,
            }
            for trace in traces
        ],
    }


@dataclass(frozen=True)
class TraceSupervisedCorpus:
    """Trusted-controller input; never an estimator input."""

    schema: str
    num_surface_keys: int
    value_cardinality: int
    traces: tuple[TraceSupervisedSequence, ...]
    corpus_sha256: str

    def __post_init__(self) -> None:
        if self.schema != "tnlm-v3-trace-supervised-corpus-v1":
            raise ValueError("unknown trace-supervised corpus schema")
        _plain_int("num_surface_keys", self.num_surface_keys, 2)
        _plain_int("value_cardinality", self.value_cardinality, 2)
        if not isinstance(self.traces, tuple) or not self.traces:
            raise ValueError("traces must be a nonempty exact tuple")
        if any(type(trace) is not TraceSupervisedSequence for trace in self.traces):
            raise TypeError("traces must contain exact TraceSupervisedSequence values")
        splits = {trace.attestation.split for trace in self.traces}
        if splits != _SEEN_SPLITS:
            raise ValueError("trace corpus requires direct train and validation traces")
        for trace in self.traces:
            _validate_presence_grammar(
                trace.sequence, self.num_surface_keys, self.value_cardinality
            )
            rebuilt = make_trace_supervised_sequence(
                trace.sequence,
                split=trace.attestation.split,
                pre_event_cells=trace.attestation.pre_event_cells,
                post_event_cells=trace.attestation.post_event_cells,
                query_dependency_cells=trace.attestation.query_dependency_cells,
                num_surface_keys=self.num_surface_keys,
                value_cardinality=self.value_cardinality,
            )
            if rebuilt != trace:
                raise ValueError("trace does not match its fully validated attestation")
            payload = _trace_payload(
                trace.sequence,
                trace.attestation.split,
                trace.attestation.pre_event_cells,
                trace.attestation.post_event_cells,
                trace.attestation.query_dependency_cells,
            )
            if trace.attestation.attestation_sha256 != _sha256(payload):
                raise ValueError("trace attestation digest is inconsistent")
        _require_sha256("corpus_sha256", self.corpus_sha256)
        if self.corpus_sha256 != _sha256(
            _trace_corpus_payload(
                self.num_surface_keys, self.value_cardinality, self.traces
            )
        ):
            raise ValueError("corpus_sha256 does not bind the trace corpus")


def make_trace_supervised_corpus(
    num_surface_keys: int,
    value_cardinality: int,
    traces: Sequence[TraceSupervisedSequence],
) -> TraceSupervisedCorpus:
    """Construct and content-bind trusted fold-controller input."""

    if not isinstance(traces, Sequence) or not traces:
        raise ValueError("traces must be a nonempty sequence")
    rows = tuple(traces)
    if any(type(trace) is not TraceSupervisedSequence for trace in rows):
        raise TypeError("traces must contain exact TraceSupervisedSequence values")
    payload = _trace_corpus_payload(num_surface_keys, value_cardinality, rows)
    return TraceSupervisedCorpus(
        schema="tnlm-v3-trace-supervised-corpus-v1",
        num_surface_keys=num_surface_keys,
        value_cardinality=value_cardinality,
        traces=rows,
        corpus_sha256=_sha256(payload),
    )


@dataclass(frozen=True, order=True)
class PrototypeAddress:
    """One learned categorical transition-table address."""

    family: str
    transform: int | None
    source_value: int

    def __post_init__(self) -> None:
        if type(self.family) is not str or self.family not in {"bind", "update", "copy"}:
            raise ValueError("prototype family must be bind, update, or copy")
        if type(self.source_value) is not int or self.source_value < 0:
            raise ValueError("source_value must be an exact nonnegative integer")
        if self.family == "update":
            if type(self.transform) is not int or self.transform < 0:
                raise ValueError("update addresses require a transform")
        elif self.transform is not None:
            raise ValueError("only update addresses may carry a transform")

    @property
    def label(self) -> str:
        if self.family == "update":
            return f"update:{self.transform}:source:{self.source_value}"
        return f"{self.family}:source:{self.source_value}"


def prototype_inventory(value_cardinality: int) -> tuple[PrototypeAddress, ...]:
    """Return the generator-supported BIND/UPDATE/COPY table inventory.

    UPDATE argument ``V-1`` is the modular identity in the benchmark but is
    never emitted by the generator.  It is omitted rather than silently
    filling an unlearned table or supplying the identity law.
    """

    _plain_int("value_cardinality", value_cardinality, 2)
    values = range(value_cardinality)
    return tuple(
        [PrototypeAddress("bind", None, value) for value in values]
        + [
            PrototypeAddress("update", transform, source)
            for transform in range(value_cardinality - 1)
            for source in values
        ]
        + [PrototypeAddress("copy", None, value) for value in values]
    )


@dataclass(frozen=True)
class SequenceAlgebraFitCertificate:
    schema: str
    training_sample_sha256: str
    seed: int
    restart_count: int
    max_sweeps: int
    max_pairwise_rounds: int
    objective_evaluations: int
    pairwise_uncertain_address_count: int
    pairwise_max_search_address_count: int
    pairwise_objective_evaluations: int
    pairwise_improvement_count: int
    pairwise_search_rule: str
    residual_penalty: int
    training_query_count: int
    training_mistakes: int
    residual_override_count: int
    penalized_objective: int
    override_count_lexicographic_tiebreak: bool
    deterministic_table_lexicographic_tiebreak: bool
    zero_explicit_penalty_still_uses_minimum_override_tiebreak: bool
    optimization_tiebreak: str
    trace_supervised_initializer_used: bool
    trace_supervised_initializer_vote_count: int
    trace_supervised_initializer_covered_address_count: int
    trace_supervised_initializer_conflicting_address_count: int
    trace_supervised_initializer_round_count: int
    trace_supervised_initializer_rule: str
    initializer_received_canonical_state: bool
    initializer_received_exact_executor: bool
    random_restart_count: int
    optimizer_global_optimum_certified: bool
    canonical_state_supervision_used_by_estimator: bool
    exact_executor_used_for_fitting: bool
    supplied_addressable_register_interface: bool
    supplied_presence_and_invalidation_grammar: bool
    supplied_identity_query_gauge: bool
    representation_discovery_performed: bool
    trace_only_transition_table_learning_performed: bool
    direct_split_caller_attested: bool
    heldout_label_absence_independently_certified: bool
    heldout_identifier_received_by_estimator: bool
    evaluation_metadata_received_by_estimator: bool
    certificate_sha256: str

    def __post_init__(self) -> None:
        if self.schema != "tnlm-v3-sequence-algebra-fit-v2":
            raise ValueError("unknown sequence-algebra fit schema")
        for name in ("training_sample_sha256", "certificate_sha256"):
            _require_sha256(name, getattr(self, name))
        for name, minimum in (
            ("seed", 0),
            ("restart_count", 1),
            ("max_sweeps", 1),
            ("max_pairwise_rounds", 0),
            ("objective_evaluations", 1),
            ("pairwise_uncertain_address_count", 0),
            ("pairwise_max_search_address_count", 0),
            ("pairwise_objective_evaluations", 0),
            ("pairwise_improvement_count", 0),
            ("residual_penalty", 0),
            ("training_query_count", 1),
            ("training_mistakes", 0),
            ("residual_override_count", 0),
            ("penalized_objective", 0),
            ("trace_supervised_initializer_vote_count", 0),
            ("trace_supervised_initializer_covered_address_count", 0),
            ("trace_supervised_initializer_conflicting_address_count", 0),
            ("trace_supervised_initializer_round_count", 1),
            ("random_restart_count", 0),
        ):
            _plain_int(name, getattr(self, name), minimum)
        if self.training_mistakes > self.training_query_count:
            raise ValueError("training mistakes cannot exceed query count")
        if self.penalized_objective != (
            self.training_mistakes
            + self.residual_penalty * self.residual_override_count
        ):
            raise ValueError("penalized objective is inconsistent")
        for name in (
            "optimizer_global_optimum_certified",
            "override_count_lexicographic_tiebreak",
            "deterministic_table_lexicographic_tiebreak",
            "zero_explicit_penalty_still_uses_minimum_override_tiebreak",
            "trace_supervised_initializer_used",
            "initializer_received_canonical_state",
            "initializer_received_exact_executor",
            "canonical_state_supervision_used_by_estimator",
            "exact_executor_used_for_fitting",
            "supplied_addressable_register_interface",
            "supplied_presence_and_invalidation_grammar",
            "supplied_identity_query_gauge",
            "representation_discovery_performed",
            "trace_only_transition_table_learning_performed",
            "direct_split_caller_attested",
            "heldout_label_absence_independently_certified",
            "heldout_identifier_received_by_estimator",
            "evaluation_metadata_received_by_estimator",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact boolean")
        expected_tiebreak = (
            "penalized_mistakes_then_mistakes_then_override_count_then_"
            "shared_and_override_tables"
        )
        if self.optimization_tiebreak != expected_tiebreak:
            raise ValueError("optimization_tiebreak does not name the implemented order")
        if self.pairwise_search_rule != (
            "best_lexicographic_two_address_move_over_initializer_"
            "uncertain_or_coordinate_changed_addresses"
        ):
            raise ValueError("pairwise_search_rule is not canonical")
        if self.pairwise_objective_evaluations > self.objective_evaluations:
            raise ValueError("pairwise evaluations cannot exceed all evaluations")
        if (
            self.pairwise_max_search_address_count
            < self.pairwise_uncertain_address_count
        ):
            raise ValueError("pairwise search cannot omit initializer-uncertain addresses")
        if self.pairwise_improvement_count > (
            self.restart_count * self.max_pairwise_rounds
        ):
            raise ValueError("pairwise improvements exceed the configured rounds")
        if self.trace_supervised_initializer_rule != (
            "fixed_point_plurality_of_visible_transition_then_query_targets_"
            "smallest_output_tiebreak_seeded_uncovered_fallback"
        ):
            raise ValueError("trace_supervised_initializer_rule is not canonical")
        if not self.trace_supervised_initializer_used:
            raise ValueError("fit certificate must disclose the trace initializer")
        if self.random_restart_count != self.restart_count - 1:
            raise ValueError("random_restart_count is inconsistent")
        if (
            self.trace_supervised_initializer_conflicting_address_count
            > self.trace_supervised_initializer_covered_address_count
        ):
            raise ValueError("initializer conflicts cannot exceed covered addresses")
        if (
            self.trace_supervised_initializer_covered_address_count
            > self.trace_supervised_initializer_vote_count
        ):
            raise ValueError("initializer coverage cannot exceed its vote count")
        if self.initializer_received_canonical_state:
            raise ValueError("the initializer must not receive canonical states")
        if self.initializer_received_exact_executor:
            raise ValueError("the initializer must not call an exact executor")
        if not all(
            (
                self.override_count_lexicographic_tiebreak,
                self.deterministic_table_lexicographic_tiebreak,
                self.zero_explicit_penalty_still_uses_minimum_override_tiebreak,
            )
        ):
            raise ValueError("fit certificate must disclose every implicit tie-break")
        if self.optimizer_global_optimum_certified:
            raise ValueError("coordinate descent does not certify a global optimum")
        if self.canonical_state_supervision_used_by_estimator:
            raise ValueError("the estimator must not receive canonical states")
        if self.exact_executor_used_for_fitting:
            raise ValueError("the estimator must not call an exact executor")
        if not all(
            (
                self.supplied_addressable_register_interface,
                self.supplied_presence_and_invalidation_grammar,
                self.supplied_identity_query_gauge,
                self.trace_only_transition_table_learning_performed,
                self.direct_split_caller_attested,
            )
        ):
            raise ValueError("the fit certificate must declare its supplied interface")
        if self.representation_discovery_performed:
            raise ValueError("this experiment does not discover a representation")
        if self.heldout_label_absence_independently_certified:
            raise ValueError(
                "the pure fit boundary cannot independently certify label absence"
            )
        if self.heldout_identifier_received_by_estimator:
            raise ValueError("the pure estimator cannot receive a heldout identifier")
        if self.evaluation_metadata_received_by_estimator:
            raise ValueError("the pure estimator cannot receive evaluation metadata")
        if self.certificate_sha256 != _sha256(_fit_certificate_payload(self)):
            raise ValueError("certificate_sha256 does not bind the fit certificate")


def _fit_certificate_payload(certificate: SequenceAlgebraFitCertificate) -> dict[str, object]:
    return {
        name: getattr(certificate, name)
        for name in certificate.__dataclass_fields__
        if name != "certificate_sha256"
    }


@dataclass(frozen=True)
class LearnedSequenceAlgebra:
    """Frozen learned transducer in the supplied register interface."""

    schema: str
    num_surface_keys: int
    value_cardinality: int
    shared_outputs: tuple[tuple[PrototypeAddress, int], ...]
    local_overrides: tuple[tuple[int, PrototypeAddress, int], ...]
    fit: SequenceAlgebraFitCertificate
    model_fingerprint: str

    def __post_init__(self) -> None:
        if self.schema != "tnlm-v3-learned-sequence-algebra-v1":
            raise ValueError("unknown learned sequence algebra schema")
        _plain_int("num_surface_keys", self.num_surface_keys, 2)
        _plain_int("value_cardinality", self.value_cardinality, 2)
        inventory = prototype_inventory(self.value_cardinality)
        if not isinstance(self.shared_outputs, tuple):
            raise TypeError("shared_outputs must be an exact tuple")
        if tuple(address for address, _ in self.shared_outputs) != inventory:
            raise ValueError("shared_outputs must follow the complete canonical inventory")
        shared = dict(self.shared_outputs)
        for address, output in self.shared_outputs:
            if type(output) is not int or not 0 <= output < self.value_cardinality:
                raise ValueError(f"shared output for {address.label} is out of range")
        if not isinstance(self.local_overrides, tuple):
            raise TypeError("local_overrides must be an exact tuple")
        if tuple(sorted(self.local_overrides)) != self.local_overrides:
            raise ValueError("local_overrides must be sorted")
        seen: set[tuple[int, PrototypeAddress]] = set()
        for key, address, output in self.local_overrides:
            if type(key) is not int or not 0 <= key < self.num_surface_keys:
                raise ValueError("override destination key is out of range")
            if address not in shared:
                raise ValueError("override names an unknown prototype address")
            if type(output) is not int or not 0 <= output < self.value_cardinality:
                raise ValueError("override output is out of range")
            pair = (key, address)
            if pair in seen:
                raise ValueError("local override addresses must be unique")
            seen.add(pair)
            if output == shared[address]:
                raise ValueError("residual gauge forbids an override equal to its base")
        if type(self.fit) is not SequenceAlgebraFitCertificate:
            raise TypeError("fit must be exact SequenceAlgebraFitCertificate")
        if self.fit.residual_override_count != len(self.local_overrides):
            raise ValueError("fit certificate override count is inconsistent")
        if self.fit.certificate_sha256 != _sha256(_fit_certificate_payload(self.fit)):
            raise ValueError("fit certificate digest is inconsistent")
        _require_sha256("model_fingerprint", self.model_fingerprint)
        if self.model_fingerprint != _sha256(_model_payload(self)):
            raise ValueError("model_fingerprint does not bind the model")

    def _tables(
        self,
    ) -> tuple[
        dict[PrototypeAddress, int],
        dict[tuple[int, PrototypeAddress], int],
    ]:
        return dict(self.shared_outputs), {
            (key, address): output for key, address, output in self.local_overrides
        }

    def predict(self, program: object) -> tuple[int | None, ...]:
        """Predict aligned query outputs for a sequence or duck-typed probe."""

        events = _coerce_program_events(program)
        _check_program_dimensions(program, self.num_surface_keys, self.value_cardinality)
        shared, overrides = self._tables()
        return _predict_events(
            events,
            self.num_surface_keys,
            self.value_cardinality,
            shared,
            overrides,
        )

    def predict_queries(self, program: object) -> tuple[int, ...]:
        """Return query-only predictions; compatible with balanced probe suites."""

        return tuple(value for value in self.predict(program) if value is not None)


def _model_payload(model: LearnedSequenceAlgebra) -> dict[str, object]:
    return {
        "schema": model.schema,
        "num_surface_keys": model.num_surface_keys,
        "value_cardinality": model.value_cardinality,
        "shared_outputs": [
            [address.label, output] for address, output in model.shared_outputs
        ],
        "local_overrides": [
            [key, address.label, output]
            for key, address, output in model.local_overrides
        ],
        "fit_certificate_sha256": model.fit.certificate_sha256,
    }


def _coerce_program_events(program: object) -> tuple[VisibleEvent, ...]:
    raw_events = getattr(program, "events", None)
    if not isinstance(raw_events, tuple) or not raw_events:
        raise TypeError("program must expose a nonempty exact .events tuple")
    events: list[VisibleEvent] = []
    for raw in raw_events:
        if type(raw) is VisibleEvent:
            events.append(raw)
            continue
        try:
            kind = raw.kind
            if type(kind) is not BindingEventKind:
                kind = BindingEventKind(int(kind))
            events.append(
                VisibleEvent(
                    kind=kind,
                    primary_key=raw.primary_key,
                    secondary_key=raw.secondary_key,
                    argument=raw.argument,
                )
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise TypeError("program events do not match the visible-event protocol") from error
    return tuple(events)


def _check_program_dimensions(program: object, keys: int, values: int) -> None:
    program_keys = getattr(program, "num_surface_keys", keys)
    program_values = getattr(program, "value_cardinality", values)
    if type(program_keys) is not int or program_keys != keys:
        raise ValueError("program num_surface_keys disagrees with the model")
    if type(program_values) is not int or program_values != values:
        raise ValueError("program value_cardinality disagrees with the model")


def _predict_events(
    events: tuple[VisibleEvent, ...],
    num_surface_keys: int,
    value_cardinality: int,
    shared: dict[PrototypeAddress, int],
    overrides: dict[tuple[int, PrototypeAddress], int],
) -> tuple[int | None, ...]:
    states: list[int | None] = [None] * num_surface_keys
    outputs: list[int | None] = []
    for event in events:
        _validate_event_shape(event, num_surface_keys, value_cardinality)
        key = event.primary_key
        if event.kind is BindingEventKind.BIND:
            if states[key] is not None:
                raise ValueError("BIND must address an absent register")
            address = PrototypeAddress("bind", None, event.argument)
            states[key] = overrides.get((key, address), shared[address])
            outputs.append(None)
        elif event.kind is BindingEventKind.UPDATE:
            source = states[key]
            if source is None:
                raise ValueError("UPDATE must address a present register")
            if event.argument == value_cardinality - 1:
                raise ValueError(
                    "identity UPDATE was outside training support and is not predicted"
                )
            address = PrototypeAddress("update", event.argument, source)
            states[key] = overrides.get((key, address), shared[address])
            outputs.append(None)
        elif event.kind is BindingEventKind.COPY:
            source = states[event.secondary_key]
            if states[key] is None or source is None:
                raise ValueError("COPY must address two present registers")
            address = PrototypeAddress("copy", None, source)
            states[key] = overrides.get((key, address), shared[address])
            outputs.append(None)
        elif event.kind is BindingEventKind.INVALIDATE:
            if states[key] is None:
                raise ValueError("INVALIDATE must address a present register")
            states[key] = None
            outputs.append(None)
        elif event.kind is BindingEventKind.QUERY:
            value = states[key]
            if value is None:
                raise ValueError("QUERY must address a present register")
            outputs.append(value)
        else:
            outputs.append(None)
    return tuple(outputs)


def _corpus_mistakes(
    corpus: SequenceCorpus,
    shared: dict[PrototypeAddress, int],
    overrides: dict[tuple[int, PrototypeAddress], int],
    masks: tuple[tuple[bool, ...], ...] | None = None,
) -> tuple[int, int]:
    if masks is not None and len(masks) != len(corpus.sequences):
        raise ValueError("score masks must align with corpus sequences")
    mistakes = 0
    count = 0
    for sequence_index, sequence in enumerate(corpus.sequences):
        predictions = _predict_events(
            sequence.events,
            corpus.num_surface_keys,
            corpus.value_cardinality,
            shared,
            overrides,
        )
        mask = None if masks is None else masks[sequence_index]
        if mask is not None and len(mask) != len(sequence.events):
            raise ValueError("each score mask must align with its sequence")
        for index, (prediction, target) in enumerate(
            zip(predictions, sequence.query_targets, strict=True)
        ):
            if target is None or (mask is not None and not mask[index]):
                continue
            count += 1
            mistakes += prediction != target
    return mistakes, count


def _initial_outputs(
    inventory: tuple[PrototypeAddress, ...], value_cardinality: int, seed: int, restart: int
) -> dict[PrototypeAddress, int]:
    outputs: dict[PrototypeAddress, int] = {}
    for address in inventory:
        digest = hashlib.sha256(
            f"tnlm-v3-sequence-init-v1|{seed}|{restart}|{address.label}".encode("utf-8")
        ).digest()
        outputs[address] = int.from_bytes(digest[:8], "big") % value_cardinality
    return outputs


def _trace_supervised_initial_outputs(
    corpus: SequenceCorpus,
    inventory: tuple[PrototypeAddress, ...],
    seed: int,
) -> tuple[
    dict[PrototypeAddress, int],
    int,
    int,
    int,
    int,
    tuple[PrototypeAddress, ...],
]:
    """Initialize shared tables from model-visible transition/query pairs.

    Each scan maintains only values revealed by earlier QUERY targets or by
    transition entries inferred on an earlier scan.  A BIND, UPDATE, or COPY
    becomes a vote when its destination is queried before that destination is
    modified again and the transition's prototype address is visible or has a
    query-revealed/inferred source value.  Repeating to a fixed point allows a
    query at the end of a longer chain to expose successively earlier unknowns.
    No trace attestation, canonical pre/post state, task fingerprint, held-out
    identifier, or exact executor is available at this boundary.

    Addresses without a vote retain the deterministic seeded restart-zero
    value.  Conflicting votes use plurality and then the smallest output.  This
    is an optimizer initializer, not a law or correctness oracle; coordinate
    descent and all fit certificates remain responsible for the final result.
    """

    inferred: dict[PrototypeAddress, int] = {}
    final_vote_counts: dict[PrototypeAddress, list[int]] = {}
    seen_tables: set[tuple[tuple[PrototypeAddress, int], ...]] = set()
    round_count = 0
    for _ in range(len(inventory) + 1):
        round_count += 1
        vote_counts: dict[PrototypeAddress, list[int]] = {
            address: [0] * corpus.value_cardinality for address in inventory
        }
        for sequence in corpus.sequences:
            known_values: list[int | None] = [None] * corpus.num_surface_keys
            pending: list[PrototypeAddress | None] = [None] * corpus.num_surface_keys
            for event, target in zip(
                sequence.events, sequence.query_targets, strict=True
            ):
                key = event.primary_key
                address: PrototypeAddress | None = None
                if event.kind is BindingEventKind.BIND:
                    address = PrototypeAddress("bind", None, event.argument)
                elif event.kind is BindingEventKind.UPDATE:
                    source_value = known_values[key]
                    if source_value is not None:
                        address = PrototypeAddress(
                            "update", event.argument, source_value
                        )
                elif event.kind is BindingEventKind.COPY:
                    source_value = known_values[event.secondary_key]
                    if source_value is not None:
                        address = PrototypeAddress("copy", None, source_value)

                if event.kind in (
                    BindingEventKind.BIND,
                    BindingEventKind.UPDATE,
                    BindingEventKind.COPY,
                ):
                    pending[key] = address
                    known_values[key] = (
                        None if address is None else inferred.get(address)
                    )
                elif event.kind is BindingEventKind.INVALIDATE:
                    pending[key] = None
                    known_values[key] = None
                elif event.kind is BindingEventKind.QUERY:
                    if target is None:
                        raise RuntimeError(
                            "validated QUERY unexpectedly lacks a target"
                        )
                    address = pending[key]
                    if address is not None:
                        vote_counts[address][target] += 1
                        pending[key] = None
                    known_values[key] = target

        next_inferred: dict[PrototypeAddress, int] = {}
        for address in inventory:
            counts = vote_counts[address]
            if any(counts):
                next_inferred[address] = min(
                    range(corpus.value_cardinality),
                    key=lambda output: (-counts[output], output),
                )
        final_vote_counts = vote_counts
        signature = tuple(sorted(next_inferred.items()))
        if next_inferred == inferred:
            inferred = next_inferred
            break
        if signature in seen_tables:
            inferred = min(
                (inferred, next_inferred),
                key=lambda table: tuple(sorted(table.items())),
            )
            break
        seen_tables.add(signature)
        inferred = next_inferred

    outputs = _initial_outputs(inventory, corpus.value_cardinality, seed, 0)
    outputs.update(inferred)
    vote_count = sum(sum(counts) for counts in final_vote_counts.values())
    covered = sum(any(counts) for counts in final_vote_counts.values())
    conflicting = sum(
        sum(count > 0 for count in counts) > 1
        for counts in final_vote_counts.values()
    )
    uncertain = tuple(
        address
        for address in inventory
        if not any(final_vote_counts[address])
        or sum(count > 0 for count in final_vote_counts[address]) > 1
    )
    return outputs, vote_count, covered, conflicting, round_count, uncertain


def _build_model(
    corpus: SequenceCorpus,
    shared: dict[PrototypeAddress, int],
    overrides: dict[tuple[int, PrototypeAddress], int],
    *,
    seed: int,
    restart_count: int,
    max_sweeps: int,
    evaluations: int,
    residual_penalty: int,
    mistakes: int,
    initializer_vote_count: int,
    initializer_covered_address_count: int,
    initializer_conflicting_address_count: int,
    initializer_round_count: int,
    max_pairwise_rounds: int,
    pairwise_uncertain_address_count: int,
    pairwise_max_search_address_count: int,
    pairwise_objective_evaluations: int,
    pairwise_improvement_count: int,
) -> LearnedSequenceAlgebra:
    certificate_fields: dict[str, object] = {
        "schema": "tnlm-v3-sequence-algebra-fit-v2",
        "training_sample_sha256": corpus.sample_sha256,
        "seed": seed,
        "restart_count": restart_count,
        "max_sweeps": max_sweeps,
        "max_pairwise_rounds": max_pairwise_rounds,
        "objective_evaluations": evaluations,
        "pairwise_uncertain_address_count": pairwise_uncertain_address_count,
        "pairwise_max_search_address_count": pairwise_max_search_address_count,
        "pairwise_objective_evaluations": pairwise_objective_evaluations,
        "pairwise_improvement_count": pairwise_improvement_count,
        "pairwise_search_rule": (
            "best_lexicographic_two_address_move_over_initializer_"
            "uncertain_or_coordinate_changed_addresses"
        ),
        "residual_penalty": residual_penalty,
        "training_query_count": corpus.query_count,
        "training_mistakes": mistakes,
        "residual_override_count": len(overrides),
        "penalized_objective": mistakes + residual_penalty * len(overrides),
        "override_count_lexicographic_tiebreak": True,
        "deterministic_table_lexicographic_tiebreak": True,
        "zero_explicit_penalty_still_uses_minimum_override_tiebreak": True,
        "optimization_tiebreak": (
            "penalized_mistakes_then_mistakes_then_override_count_then_"
            "shared_and_override_tables"
        ),
        "trace_supervised_initializer_used": True,
        "trace_supervised_initializer_vote_count": initializer_vote_count,
        "trace_supervised_initializer_covered_address_count": (
            initializer_covered_address_count
        ),
        "trace_supervised_initializer_conflicting_address_count": (
            initializer_conflicting_address_count
        ),
        "trace_supervised_initializer_round_count": initializer_round_count,
        "trace_supervised_initializer_rule": (
            "fixed_point_plurality_of_visible_transition_then_query_targets_"
            "smallest_output_tiebreak_seeded_uncovered_fallback"
        ),
        "initializer_received_canonical_state": False,
        "initializer_received_exact_executor": False,
        "random_restart_count": restart_count - 1,
        "optimizer_global_optimum_certified": False,
        "canonical_state_supervision_used_by_estimator": False,
        "exact_executor_used_for_fitting": False,
        "supplied_addressable_register_interface": True,
        "supplied_presence_and_invalidation_grammar": True,
        "supplied_identity_query_gauge": True,
        "representation_discovery_performed": False,
        "trace_only_transition_table_learning_performed": True,
        "direct_split_caller_attested": True,
        "heldout_label_absence_independently_certified": False,
        "heldout_identifier_received_by_estimator": False,
        "evaluation_metadata_received_by_estimator": False,
    }
    certificate = SequenceAlgebraFitCertificate(
        **certificate_fields,
        certificate_sha256=_sha256(certificate_fields),
    )
    shared_rows = tuple(
        (address, shared[address])
        for address in prototype_inventory(corpus.value_cardinality)
    )
    override_rows = tuple(
        sorted((key, address, output) for (key, address), output in overrides.items())
    )
    model_fields = {
        "schema": "tnlm-v3-learned-sequence-algebra-v1",
        "num_surface_keys": corpus.num_surface_keys,
        "value_cardinality": corpus.value_cardinality,
        "shared_outputs": shared_rows,
        "local_overrides": override_rows,
        "fit": certificate,
    }
    model_payload = {
        "schema": model_fields["schema"],
        "num_surface_keys": model_fields["num_surface_keys"],
        "value_cardinality": model_fields["value_cardinality"],
        "shared_outputs": [
            [address.label, output] for address, output in shared_rows
        ],
        "local_overrides": [
            [key, address.label, output]
            for key, address, output in override_rows
        ],
        "fit_certificate_sha256": certificate.certificate_sha256,
    }
    return LearnedSequenceAlgebra(
        **model_fields,
        model_fingerprint=_sha256(model_payload),
    )


def fit_sequence_algebra(
    corpus: SequenceCorpus,
    *,
    residual_penalty: int,
    seed: int = 0,
    restart_count: int = 2,
    max_sweeps: int = 4,
    max_pairwise_rounds: int = 2,
    max_events: int = 250_000,
    max_objective_evaluations: int = 100_000,
    max_scored_event_work: int = 50_000_000,
) -> LearnedSequenceAlgebra:
    """Fit transition coefficients from visible TRAIN sequences and answers.

    This is deterministic multi-start categorical coordinate descent.  The
    first start uses a disclosed trace/query-only plurality initializer; any
    remaining starts use deterministic seeded tables.  It is deliberately
    reported as an optimizer without a global-optimum certificate.  A local
    residual is indexed by ``(destination key,
    prototype address)`` and is therefore expressive enough to change one
    otherwise unseen key/value transition while agreeing everywhere else.
    The residual gauge is fixed by deleting overrides equal to the shared base.
    """

    if type(corpus) is not SequenceCorpus:
        raise TypeError("corpus must be exact SequenceCorpus")
    if corpus.split != "train":
        raise ValueError("fit_sequence_algebra accepts direct TRAIN data only")
    _plain_int("residual_penalty", residual_penalty, 0)
    _plain_int("seed", seed, 0)
    _plain_int("restart_count", restart_count, 1)
    _plain_int("max_sweeps", max_sweeps, 1)
    _plain_int("max_pairwise_rounds", max_pairwise_rounds, 0)
    _plain_int("max_events", max_events, 1)
    _plain_int("max_objective_evaluations", max_objective_evaluations, 1)
    _plain_int("max_scored_event_work", max_scored_event_work, 1)
    if corpus.event_count > max_events:
        raise SequenceDiscoveryLimitError("sequence fit exceeds max_events")
    inventory = prototype_inventory(corpus.value_cardinality)
    coordinate_evaluations = (
        max_sweeps
        * len(inventory)
        * corpus.value_cardinality
        * (corpus.num_surface_keys + 1)
    )
    (
        trace_initial_outputs,
        initializer_vote_count,
        initializer_covered_address_count,
        initializer_conflicting_address_count,
        initializer_round_count,
        initializer_uncertain_addresses,
    ) = _trace_supervised_initial_outputs(corpus, inventory, seed)
    pairwise_evaluations = (
        max_pairwise_rounds
        * len(inventory)
        * (len(inventory) - 1)
        // 2
        * corpus.value_cardinality**2
    )
    planned_evaluations = restart_count * (
        1 + coordinate_evaluations + pairwise_evaluations
    )
    if planned_evaluations > max_objective_evaluations:
        raise SequenceDiscoveryLimitError(
            "planned optimization exceeds max_objective_evaluations before work"
        )
    scored_event_work = corpus.event_count * planned_evaluations
    if scored_event_work > max_scored_event_work:
        raise SequenceDiscoveryLimitError(
            "sequence fit exceeds max_scored_event_work before optimization"
        )
    evaluations = 0
    pairwise_objective_evaluations = 0
    pairwise_improvement_count = 0
    pairwise_max_search_address_count = len(initializer_uncertain_addresses)

    def score(
        shared: dict[PrototypeAddress, int],
        overrides: dict[tuple[int, PrototypeAddress], int],
    ) -> tuple[int, int, int, tuple[int, ...], tuple[tuple[int, str, int], ...]]:
        nonlocal evaluations
        if evaluations >= max_objective_evaluations:
            raise SequenceDiscoveryLimitError(
                "sequence fit exceeds max_objective_evaluations"
            )
        evaluations += 1
        mistakes, _ = _corpus_mistakes(corpus, shared, overrides)
        serialized_shared = tuple(shared[address] for address in inventory)
        serialized_overrides = tuple(
            sorted((key, address.label, output) for (key, address), output in overrides.items())
        )
        return (
            mistakes + residual_penalty * len(overrides),
            mistakes,
            len(overrides),
            serialized_shared,
            serialized_overrides,
        )

    best: tuple[
        tuple[int, int, int, tuple[int, ...], tuple[tuple[int, str, int], ...]],
        dict[PrototypeAddress, int],
        dict[tuple[int, PrototypeAddress], int],
    ] | None = None
    for restart in range(restart_count):
        shared = (
            dict(trace_initial_outputs)
            if restart == 0
            else _initial_outputs(inventory, corpus.value_cardinality, seed, restart)
        )
        overrides: dict[tuple[int, PrototypeAddress], int] = {}
        current = score(shared, overrides)
        for _ in range(max_sweeps):
            changed = False
            for address in inventory:
                candidates = []
                for output in range(corpus.value_cardinality):
                    trial_shared = dict(shared)
                    trial_shared[address] = output
                    trial_overrides = {
                        pair: value
                        for pair, value in overrides.items()
                        if pair[1] != address or value != output
                    }
                    candidates.append(
                        (score(trial_shared, trial_overrides), output, trial_overrides)
                    )
                candidate_score, output, candidate_overrides = min(
                    candidates, key=lambda row: row[0]
                )
                if candidate_score < current:
                    shared[address] = output
                    overrides = candidate_overrides
                    current = candidate_score
                    changed = True
            for key in range(corpus.num_surface_keys):
                for address in inventory:
                    pair = (key, address)
                    candidates: list[
                        tuple[
                            tuple[int, int, int, tuple[int, ...], tuple[tuple[int, str, int], ...]],
                            int | None,
                        ]
                    ] = []
                    for output in (None, *range(corpus.value_cardinality)):
                        if output == shared[address]:
                            continue
                        trial = dict(overrides)
                        if output is None:
                            trial.pop(pair, None)
                        else:
                            trial[pair] = output
                        candidates.append((score(shared, trial), output))
                    candidate_score, output = min(candidates, key=lambda row: row[0])
                    if candidate_score < current:
                        if output is None:
                            overrides.pop(pair, None)
                        else:
                            overrides[pair] = output
                        current = candidate_score
                        changed = True
            if not changed:
                break
        pairwise_addresses = tuple(
            address
            for address in inventory
            if address in initializer_uncertain_addresses
            or shared[address] != trace_initial_outputs[address]
        )
        pairwise_max_search_address_count = max(
            pairwise_max_search_address_count, len(pairwise_addresses)
        )
        for _ in range(max_pairwise_rounds):
            if current[1] == 0 and current[2] == 0:
                break
            best_pair: tuple[
                tuple[
                    int,
                    int,
                    int,
                    tuple[int, ...],
                    tuple[tuple[int, str, int], ...],
                ],
                PrototypeAddress | None,
                PrototypeAddress | None,
                int,
                int,
                dict[tuple[int, PrototypeAddress], int],
            ] = (current, None, None, 0, 0, overrides)
            for first_index, first in enumerate(pairwise_addresses):
                for second in pairwise_addresses[first_index + 1 :]:
                    for first_output in range(corpus.value_cardinality):
                        for second_output in range(corpus.value_cardinality):
                            trial_shared = dict(shared)
                            trial_shared[first] = first_output
                            trial_shared[second] = second_output
                            trial_overrides = {
                                pair: value
                                for pair, value in overrides.items()
                                if not (
                                    pair[1] == first and value == first_output
                                )
                                and not (
                                    pair[1] == second and value == second_output
                                )
                            }
                            candidate_score = score(
                                trial_shared, trial_overrides
                            )
                            pairwise_objective_evaluations += 1
                            if candidate_score < best_pair[0]:
                                best_pair = (
                                    candidate_score,
                                    first,
                                    second,
                                    first_output,
                                    second_output,
                                    trial_overrides,
                                )
            if best_pair[1] is None or best_pair[2] is None:
                break
            (
                current,
                first,
                second,
                first_output,
                second_output,
                overrides,
            ) = best_pair
            shared[first] = first_output
            shared[second] = second_output
            pairwise_improvement_count += 1
        candidate = (current, dict(shared), dict(overrides))
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("sequence optimizer produced no candidate")
    best_score, best_shared, best_overrides = best
    return _build_model(
        corpus,
        best_shared,
        best_overrides,
        seed=seed,
        restart_count=restart_count,
        max_sweeps=max_sweeps,
        evaluations=evaluations,
        residual_penalty=residual_penalty,
        mistakes=best_score[1],
        initializer_vote_count=initializer_vote_count,
        initializer_covered_address_count=initializer_covered_address_count,
        initializer_conflicting_address_count=initializer_conflicting_address_count,
        initializer_round_count=initializer_round_count,
        max_pairwise_rounds=max_pairwise_rounds,
        pairwise_uncertain_address_count=len(initializer_uncertain_addresses),
        pairwise_max_search_address_count=pairwise_max_search_address_count,
        pairwise_objective_evaluations=pairwise_objective_evaluations,
        pairwise_improvement_count=pairwise_improvement_count,
    )


def _trace_states(trace: TraceSupervisedSequence) -> set[Cell]:
    result: set[Cell] = set()
    for rows in (
        trace.attestation.pre_event_cells,
        trace.attestation.post_event_cells,
    ):
        for row in rows:
            result.update(row)
    return result


def _trace_dependencies(trace: TraceSupervisedSequence) -> set[Cell]:
    return {
        cell
        for row in trace.attestation.query_dependency_cells
        for cell in row
    }


@dataclass(frozen=True)
class TracePseudoheldoutCandidateScore:
    residual_penalty: int
    model_fingerprint: str
    pseudo_query_count: int
    pseudo_query_mistakes: int
    all_validation_query_count: int
    all_validation_query_mistakes: int
    training_mistakes: int
    residual_override_count: int

    def __post_init__(self) -> None:
        _plain_int("residual_penalty", self.residual_penalty, 0)
        _require_sha256("model_fingerprint", self.model_fingerprint)
        for name, minimum in (
            ("pseudo_query_count", 1),
            ("pseudo_query_mistakes", 0),
            ("all_validation_query_count", 1),
            ("all_validation_query_mistakes", 0),
            ("training_mistakes", 0),
            ("residual_override_count", 0),
        ):
            _plain_int(name, getattr(self, name), minimum)
        if self.pseudo_query_mistakes > self.pseudo_query_count:
            raise ValueError("pseudo query mistakes exceed the query count")
        if self.all_validation_query_mistakes > self.all_validation_query_count:
            raise ValueError("validation mistakes exceed the query count")


@dataclass(frozen=True)
class TracePseudoheldoutFoldCertificate:
    schema: str
    pseudoheldout_cell: Cell
    optimizer_seed: int
    train_sample_sha256: str
    validation_sample_sha256: str
    removed_train_sequence_count: int
    retained_train_sequence_count: int
    scored_validation_sequence_count: int
    outer_unobserved_cell_count: int
    candidates: tuple[TracePseudoheldoutCandidateScore, ...]
    best_penalties_by_primary_score: tuple[int, ...]
    primary_score_tied: bool
    oracle_occupancy_used_only_by_fold_firewall: bool
    oracle_dependencies_used_only_for_score_mask: bool
    estimator_received_oracle_metadata: bool
    whole_sequence_pre_post_censoring: bool
    descendant_query_scoring: bool

    def __post_init__(self) -> None:
        if self.schema != "tnlm-v3-pseudoheldout-sequence-fold-v2":
            raise ValueError("unknown pseudoheldout-fold schema")
        if (
            not isinstance(self.pseudoheldout_cell, tuple)
            or len(self.pseudoheldout_cell) != 2
            or any(type(value) is not int or value < 0 for value in self.pseudoheldout_cell)
        ):
            raise ValueError("pseudoheldout_cell must be a nonnegative integer pair")
        _plain_int("optimizer_seed", self.optimizer_seed, 0)
        for name in ("train_sample_sha256", "validation_sample_sha256"):
            _require_sha256(name, getattr(self, name))
        for name, minimum in (
            ("removed_train_sequence_count", 1),
            ("retained_train_sequence_count", 1),
            ("scored_validation_sequence_count", 1),
            ("outer_unobserved_cell_count", 1),
        ):
            _plain_int(name, getattr(self, name), minimum)
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise ValueError("candidates must be a nonempty exact tuple")
        if any(type(row) is not TracePseudoheldoutCandidateScore for row in self.candidates):
            raise TypeError("candidates must contain exact score rows")
        penalties = tuple(row.residual_penalty for row in self.candidates)
        if penalties != tuple(sorted(set(penalties))):
            raise ValueError("candidate penalties must be unique and sorted")
        if not isinstance(
            self.best_penalties_by_primary_score, tuple
        ) or not self.best_penalties_by_primary_score:
            raise ValueError("best penalty inventory cannot be empty")
        if any(penalty not in penalties for penalty in self.best_penalties_by_primary_score):
            raise ValueError("best penalty inventory names an unknown candidate")
        if self.primary_score_tied != (len(self.best_penalties_by_primary_score) > 1):
            raise ValueError("primary_score_tied is inconsistent")
        for name in (
            "oracle_occupancy_used_only_by_fold_firewall",
            "oracle_dependencies_used_only_for_score_mask",
            "estimator_received_oracle_metadata",
            "whole_sequence_pre_post_censoring",
            "descendant_query_scoring",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact boolean")
        if not self.oracle_occupancy_used_only_by_fold_firewall:
            raise ValueError("folds require trusted occupancy censoring")
        if not self.oracle_dependencies_used_only_for_score_mask:
            raise ValueError("pseudo scores require trusted dependency masks")
        if self.estimator_received_oracle_metadata:
            raise ValueError("estimator must not receive oracle metadata")
        if not self.whole_sequence_pre_post_censoring or not self.descendant_query_scoring:
            raise ValueError("fold closure and descendant scoring are mandatory")


@dataclass(frozen=True)
class AggregatePenaltyScore:
    residual_penalty: int
    pseudo_query_count: int
    pseudo_query_mistakes: int
    all_validation_query_count: int
    all_validation_query_mistakes: int
    total_residual_override_count: int

    def __post_init__(self) -> None:
        for name, minimum in (
            ("residual_penalty", 0),
            ("pseudo_query_count", 1),
            ("pseudo_query_mistakes", 0),
            ("all_validation_query_count", 1),
            ("all_validation_query_mistakes", 0),
            ("total_residual_override_count", 0),
        ):
            _plain_int(name, getattr(self, name), minimum)
        if self.pseudo_query_mistakes > self.pseudo_query_count:
            raise ValueError("aggregate pseudo mistakes exceed the query count")
        if self.all_validation_query_mistakes > self.all_validation_query_count:
            raise ValueError("aggregate validation mistakes exceed the query count")


@dataclass(frozen=True)
class SequenceAlgebraSelectionResult:
    schema: str
    source_corpus_sha256: str
    mode: SequenceSelectionMode
    folds: tuple[TracePseudoheldoutFoldCertificate, ...]
    aggregate_scores: tuple[AggregatePenaltyScore, ...]
    selected_residual_penalty: int
    primary_score_best_penalties: tuple[int, ...]
    primary_score_tied: bool
    selection_basis: str
    final_model: LearnedSequenceAlgebra
    outer_unobserved_cell_count: int
    outer_omission_commitment_sha256: str
    outer_unobserved_identifier_inferred_and_used_by_trusted_firewall: bool
    outer_unobserved_identifier_received_by_estimator: bool
    outer_unobserved_labels_used_for_fit_or_selection: bool
    outer_unobserved_identifier_used_in_candidate_ordering: bool
    trace_supervised_fold_construction: bool
    sequence_only_estimator: bool
    supplied_register_transducer_family: bool
    automatic_residual_strength_selection_performed: bool
    automatic_representation_discovery_performed: bool
    retrospective_protocol_rehearsal: bool
    confirmatory_claim_permitted: bool
    observed_exception_power_control_only: bool
    result_sha256: str

    def __post_init__(self) -> None:
        if self.schema != "tnlm-v3-sequence-algebra-selection-v1":
            raise ValueError("unknown sequence-algebra selection schema")
        for name in ("source_corpus_sha256", "result_sha256"):
            _require_sha256(name, getattr(self, name))
        if type(self.mode) is not SequenceSelectionMode:
            raise TypeError("mode must be exact SequenceSelectionMode")
        if not isinstance(self.folds, tuple) or not self.folds:
            raise ValueError("folds must be a nonempty exact tuple")
        if any(type(fold) is not TracePseudoheldoutFoldCertificate for fold in self.folds):
            raise TypeError("folds must contain exact certificates")
        pseudo_cells = tuple(fold.pseudoheldout_cell for fold in self.folds)
        if pseudo_cells != tuple(sorted(set(pseudo_cells))):
            raise ValueError("pseudoheldout cells must be unique and sorted")
        if not isinstance(self.aggregate_scores, tuple) or not self.aggregate_scores:
            raise ValueError("aggregate_scores must be a nonempty exact tuple")
        penalties = tuple(row.residual_penalty for row in self.aggregate_scores)
        if penalties != tuple(sorted(set(penalties))):
            raise ValueError("aggregate penalty scores must be unique and sorted")
        fold_penalties = tuple(
            row.residual_penalty for row in self.folds[0].candidates
        )
        if penalties != fold_penalties or any(
            tuple(row.residual_penalty for row in fold.candidates) != penalties
            for fold in self.folds
        ):
            raise ValueError("every fold must use the exact aggregate candidate grid")
        expected_aggregates = tuple(
            AggregatePenaltyScore(
                residual_penalty=penalty,
                pseudo_query_count=sum(
                    fold.candidates[index].pseudo_query_count for fold in self.folds
                ),
                pseudo_query_mistakes=sum(
                    fold.candidates[index].pseudo_query_mistakes for fold in self.folds
                ),
                all_validation_query_count=sum(
                    fold.candidates[index].all_validation_query_count
                    for fold in self.folds
                ),
                all_validation_query_mistakes=sum(
                    fold.candidates[index].all_validation_query_mistakes
                    for fold in self.folds
                ),
                total_residual_override_count=sum(
                    fold.candidates[index].residual_override_count
                    for fold in self.folds
                ),
            )
            for index, penalty in enumerate(penalties)
        )
        if self.aggregate_scores != expected_aggregates:
            raise ValueError("aggregate scores do not reproduce the fold rows")
        for fold in self.folds:
            fold_primary = tuple(
                (
                    row.pseudo_query_mistakes
                    if self.mode is SequenceSelectionMode.PSEUDOHELDOUT_PRIMARY
                    else row.all_validation_query_mistakes
                )
                for row in fold.candidates
            )
            fold_best = min(fold_primary)
            expected_fold_winners = tuple(
                row.residual_penalty
                for row, value in zip(fold.candidates, fold_primary, strict=True)
                if value == fold_best
            )
            if fold.best_penalties_by_primary_score != expected_fold_winners:
                raise ValueError("fold primary-score winners are inconsistent")
        if self.selected_residual_penalty not in penalties:
            raise ValueError("selected penalty is outside the candidate grid")
        if not isinstance(
            self.primary_score_best_penalties, tuple
        ) or not self.primary_score_best_penalties:
            raise ValueError("primary score winner inventory cannot be empty")
        if self.primary_score_tied != (len(self.primary_score_best_penalties) > 1):
            raise ValueError("primary_score_tied is inconsistent")
        if self.mode is SequenceSelectionMode.PSEUDOHELDOUT_PRIMARY:
            primary = tuple(row.pseudo_query_mistakes for row in self.aggregate_scores)
            secondary = tuple(
                row.all_validation_query_mistakes for row in self.aggregate_scores
            )
        else:
            primary = tuple(
                row.all_validation_query_mistakes for row in self.aggregate_scores
            )
            secondary = tuple(row.pseudo_query_mistakes for row in self.aggregate_scores)
        best_primary = min(primary)
        expected_primary_winners = tuple(
            row.residual_penalty
            for row, value in zip(self.aggregate_scores, primary, strict=True)
            if value == best_primary
        )
        if self.primary_score_best_penalties != expected_primary_winners:
            raise ValueError("aggregate primary-score winners are inconsistent")
        expected_selected = min(
            zip(self.aggregate_scores, primary, secondary, strict=True),
            key=lambda item: (
                item[1],
                item[2],
                item[0].total_residual_override_count,
                -item[0].residual_penalty,
            ),
        )[0].residual_penalty
        if self.selected_residual_penalty != expected_selected:
            raise ValueError("selected penalty disagrees with the declared tie-break")
        if type(self.selection_basis) is not str or not self.selection_basis:
            raise ValueError("selection_basis must be a nonempty exact string")
        if type(self.final_model) is not LearnedSequenceAlgebra:
            raise TypeError("final_model must be exact LearnedSequenceAlgebra")
        if self.final_model.fit.residual_penalty != self.selected_residual_penalty:
            raise ValueError("final model and selected penalty disagree")
        for fold in self.folds:
            expected_fold_seed = (
                self.final_model.fit.seed
                + fold.pseudoheldout_cell[0] * self.final_model.value_cardinality
                + fold.pseudoheldout_cell[1]
            )
            if fold.optimizer_seed != expected_fold_seed:
                raise ValueError(
                    "fold optimizer seed is not the absolute pseudo-cell seed"
                )
        _plain_int("outer_unobserved_cell_count", self.outer_unobserved_cell_count, 1)
        _require_sha256(
            "outer_omission_commitment_sha256",
            self.outer_omission_commitment_sha256,
        )
        if any(
            fold.outer_unobserved_cell_count != self.outer_unobserved_cell_count
            for fold in self.folds
        ):
            raise ValueError("folds disagree on the outer-unobserved count")
        for name in (
            "outer_unobserved_identifier_inferred_and_used_by_trusted_firewall",
            "outer_unobserved_identifier_received_by_estimator",
            "outer_unobserved_labels_used_for_fit_or_selection",
            "outer_unobserved_identifier_used_in_candidate_ordering",
            "trace_supervised_fold_construction",
            "sequence_only_estimator",
            "supplied_register_transducer_family",
            "automatic_residual_strength_selection_performed",
            "automatic_representation_discovery_performed",
            "retrospective_protocol_rehearsal",
            "confirmatory_claim_permitted",
            "observed_exception_power_control_only",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact boolean")
        if not self.outer_unobserved_identifier_inferred_and_used_by_trusted_firewall:
            raise ValueError("trusted firewall must declare its inferred outer ID")
        if self.outer_unobserved_identifier_received_by_estimator:
            raise ValueError("pure estimator cannot receive an outer identifier")
        if self.outer_unobserved_labels_used_for_fit_or_selection:
            raise ValueError("outer labels are forbidden during fit and selection")
        if self.outer_unobserved_identifier_used_in_candidate_ordering:
            raise ValueError("outer identifiers cannot influence candidate ordering")
        if not all(
            (
                self.trace_supervised_fold_construction,
                self.sequence_only_estimator,
                self.supplied_register_transducer_family,
                self.automatic_residual_strength_selection_performed,
            )
        ):
            raise ValueError("selection result understates required protocol facts")
        if self.automatic_representation_discovery_performed:
            raise ValueError("the supplied register interface is not representation discovery")
        if not self.retrospective_protocol_rehearsal or self.confirmatory_claim_permitted:
            raise ValueError("the current experiment is retrospective and non-confirmatory")
        expected_power = self.mode is SequenceSelectionMode.OBSERVED_EXCEPTION_POWER_CONTROL
        if self.observed_exception_power_control_only != expected_power:
            raise ValueError("power-control flag disagrees with selection mode")
        if self.result_sha256 != _sha256(_selection_payload(self)):
            raise ValueError("result_sha256 does not bind the selection result")


def _candidate_score_payload(row: TracePseudoheldoutCandidateScore) -> dict[str, object]:
    return {
        name: getattr(row, name) for name in row.__dataclass_fields__
    }


def _fold_payload(fold: TracePseudoheldoutFoldCertificate) -> dict[str, object]:
    return {
        "schema": fold.schema,
        "pseudoheldout_cell": list(fold.pseudoheldout_cell),
        "optimizer_seed": fold.optimizer_seed,
        "train_sample_sha256": fold.train_sample_sha256,
        "validation_sample_sha256": fold.validation_sample_sha256,
        "removed_train_sequence_count": fold.removed_train_sequence_count,
        "retained_train_sequence_count": fold.retained_train_sequence_count,
        "scored_validation_sequence_count": fold.scored_validation_sequence_count,
        "outer_unobserved_cell_count": fold.outer_unobserved_cell_count,
        "candidates": [_candidate_score_payload(row) for row in fold.candidates],
        "best_penalties_by_primary_score": list(
            fold.best_penalties_by_primary_score
        ),
        "primary_score_tied": fold.primary_score_tied,
        "oracle_occupancy_used_only_by_fold_firewall": (
            fold.oracle_occupancy_used_only_by_fold_firewall
        ),
        "oracle_dependencies_used_only_for_score_mask": (
            fold.oracle_dependencies_used_only_for_score_mask
        ),
        "estimator_received_oracle_metadata": fold.estimator_received_oracle_metadata,
        "whole_sequence_pre_post_censoring": fold.whole_sequence_pre_post_censoring,
        "descendant_query_scoring": fold.descendant_query_scoring,
    }


def _aggregate_payload(row: AggregatePenaltyScore) -> dict[str, object]:
    return {name: getattr(row, name) for name in row.__dataclass_fields__}


def _selection_payload_fields(fields: dict[str, object]) -> dict[str, object]:
    mode = fields["mode"]
    if type(mode) is not SequenceSelectionMode:
        raise TypeError("selection payload mode must be exact SequenceSelectionMode")
    folds = fields["folds"]
    aggregates = fields["aggregate_scores"]
    model = fields["final_model"]
    if not isinstance(folds, tuple) or not isinstance(aggregates, tuple):
        raise TypeError("selection payload rows must be exact tuples")
    if type(model) is not LearnedSequenceAlgebra:
        raise TypeError("selection payload model must be exact LearnedSequenceAlgebra")
    return {
        "schema": fields["schema"],
        "source_corpus_sha256": fields["source_corpus_sha256"],
        "mode": mode.value,
        "folds": [_fold_payload(fold) for fold in folds],
        "aggregate_scores": [_aggregate_payload(row) for row in aggregates],
        "selected_residual_penalty": fields["selected_residual_penalty"],
        "primary_score_best_penalties": list(
            fields["primary_score_best_penalties"]
        ),
        "primary_score_tied": fields["primary_score_tied"],
        "selection_basis": fields["selection_basis"],
        "final_model_fingerprint": model.model_fingerprint,
        "outer_unobserved_cell_count": fields["outer_unobserved_cell_count"],
        "outer_omission_commitment_sha256": fields[
            "outer_omission_commitment_sha256"
        ],
        "outer_unobserved_identifier_inferred_and_used_by_trusted_firewall": fields[
            "outer_unobserved_identifier_inferred_and_used_by_trusted_firewall"
        ],
        "outer_unobserved_identifier_received_by_estimator": fields[
            "outer_unobserved_identifier_received_by_estimator"
        ],
        "outer_unobserved_labels_used_for_fit_or_selection": fields[
            "outer_unobserved_labels_used_for_fit_or_selection"
        ],
        "outer_unobserved_identifier_used_in_candidate_ordering": fields[
            "outer_unobserved_identifier_used_in_candidate_ordering"
        ],
        "trace_supervised_fold_construction": fields[
            "trace_supervised_fold_construction"
        ],
        "sequence_only_estimator": fields["sequence_only_estimator"],
        "supplied_register_transducer_family": fields[
            "supplied_register_transducer_family"
        ],
        "automatic_residual_strength_selection_performed": fields[
            "automatic_residual_strength_selection_performed"
        ],
        "automatic_representation_discovery_performed": fields[
            "automatic_representation_discovery_performed"
        ],
        "retrospective_protocol_rehearsal": fields[
            "retrospective_protocol_rehearsal"
        ],
        "confirmatory_claim_permitted": fields["confirmatory_claim_permitted"],
        "observed_exception_power_control_only": fields[
            "observed_exception_power_control_only"
        ],
    }


def _selection_payload(result: SequenceAlgebraSelectionResult) -> dict[str, object]:
    return _selection_payload_fields(
        {
            name: getattr(result, name)
            for name in result.__dataclass_fields__
            if name != "result_sha256"
        }
    )


def _score_model(
    model: LearnedSequenceAlgebra,
    corpus: SequenceCorpus,
    masks: tuple[tuple[bool, ...], ...] | None = None,
) -> tuple[int, int]:
    shared, overrides = model._tables()
    return _corpus_mistakes(corpus, shared, overrides, masks)


def select_sequence_algebra(
    corpus: TraceSupervisedCorpus,
    *,
    residual_penalties: Sequence[int] = (0, 1, 4, 16),
    seed: int = 0,
    restart_count: int = 2,
    max_sweeps: int = 4,
    max_pairwise_rounds: int = 2,
    mode: SequenceSelectionMode = SequenceSelectionMode.PSEUDOHELDOUT_PRIMARY,
    required_outer_unobserved_cell_count: int = 1,
    max_folds: int = 256,
    max_fit_calls: int = 2_048,
    max_controller_events: int = 1_000_000,
    max_events_per_fit: int = 250_000,
    max_objective_evaluations_per_fit: int = 100_000,
    max_scored_event_work_per_fit: int = 50_000_000,
    max_aggregate_scored_event_work: int = 500_000_000,
) -> SequenceAlgebraSelectionResult:
    """Select residual strength using closed, rotated pseudoheldout folds.

    Train coefficients are fit only on direct ``train`` traces.  Candidate
    penalties are scored only on direct seen-only ``validation`` traces.
    The trusted controller removes an entire train sequence whenever any
    attested pre/post state enters the pseudo cell.  Pseudo scores use only
    query positions whose external dependency annotation contains that cell,
    including descendant queries after UPDATE/COPY leaves it.

    The naturally absent outer cell is inferred from support only to certify
    that no supplied trace touches it.  Its identifier is neither stored in a
    fold/result nor forwarded to the estimator or score ordering.
    """

    if type(corpus) is not TraceSupervisedCorpus:
        raise TypeError("corpus must be exact TraceSupervisedCorpus")
    if type(mode) is not SequenceSelectionMode:
        raise TypeError("mode must be exact SequenceSelectionMode")
    _plain_int("seed", seed, 0)
    _plain_int("restart_count", restart_count, 1)
    _plain_int("max_sweeps", max_sweeps, 1)
    _plain_int("max_pairwise_rounds", max_pairwise_rounds, 0)
    _plain_int("required_outer_unobserved_cell_count", required_outer_unobserved_cell_count, 1)
    _plain_int("max_folds", max_folds, 1)
    _plain_int("max_fit_calls", max_fit_calls, 1)
    _plain_int("max_controller_events", max_controller_events, 1)
    _plain_int("max_events_per_fit", max_events_per_fit, 1)
    _plain_int(
        "max_objective_evaluations_per_fit",
        max_objective_evaluations_per_fit,
        1,
    )
    _plain_int("max_scored_event_work_per_fit", max_scored_event_work_per_fit, 1)
    _plain_int(
        "max_aggregate_scored_event_work",
        max_aggregate_scored_event_work,
        1,
    )
    if not isinstance(residual_penalties, Sequence) or not residual_penalties:
        raise ValueError("residual_penalties must be a nonempty sequence")
    penalties = tuple(residual_penalties)
    if any(type(penalty) is not int or penalty < 0 for penalty in penalties):
        raise ValueError("residual penalties must be exact nonnegative integers")
    if penalties != tuple(sorted(set(penalties))):
        raise ValueError("residual penalties must be unique and sorted")
    if len(penalties) < 2:
        raise ValueError("automatic selection requires at least two penalties")

    controller_event_count = sum(
        len(trace.sequence.events) for trace in corpus.traces
    )
    if controller_event_count > max_controller_events:
        raise SequenceDiscoveryLimitError("selection exceeds max_controller_events")

    universe = {
        (key, value)
        for key in range(corpus.num_surface_keys)
        for value in range(corpus.value_cardinality)
    }
    support = {cell for trace in corpus.traces for cell in _trace_states(trace)}
    outer_unobserved = universe - support
    if len(outer_unobserved) != required_outer_unobserved_cell_count:
        raise ValueError(
            "trace support does not have the preregistered outer omission count"
        )
    # This is a controller assertion only.  The identifier set is never placed
    # in a SequenceCorpus, fit call, score key, certificate, or result.
    for trace in corpus.traces:
        if _trace_states(trace) & outer_unobserved:
            raise ValueError("a trace reaches an inferred outer-unobserved cell")
        if _trace_dependencies(trace) & outer_unobserved:
            raise ValueError("a query depends on an inferred outer-unobserved cell")

    fold_cells = tuple(sorted(support))
    if len(fold_cells) > max_folds:
        raise SequenceDiscoveryLimitError("rotated folds exceed max_folds")
    required_fit_calls = len(fold_cells) * len(penalties) + 1
    if required_fit_calls > max_fit_calls:
        raise SequenceDiscoveryLimitError("selection exceeds max_fit_calls")
    train_traces = tuple(
        trace for trace in corpus.traces if trace.attestation.split == "train"
    )
    validation_traces = tuple(
        trace for trace in corpus.traces if trace.attestation.split == "validation"
    )
    train_event_count = sum(
        len(trace.sequence.events) for trace in train_traces
    )
    validation_event_count = sum(
        len(trace.sequence.events) for trace in validation_traces
    )
    if train_event_count > max_events_per_fit:
        raise SequenceDiscoveryLimitError(
            "full TRAIN corpus exceeds max_events_per_fit"
        )
    inventory_size = len(prototype_inventory(corpus.value_cardinality))
    planned_evaluations_per_fit = restart_count * (
        1
        + max_sweeps
        * inventory_size
        * corpus.value_cardinality
        * (corpus.num_surface_keys + 1)
        + max_pairwise_rounds
        * inventory_size
        * (inventory_size - 1)
        // 2
        * corpus.value_cardinality**2
    )
    if planned_evaluations_per_fit > max_objective_evaluations_per_fit:
        raise SequenceDiscoveryLimitError(
            "planned fit exceeds max_objective_evaluations_per_fit before fitting"
        )
    conservative_per_fit_work = train_event_count * planned_evaluations_per_fit
    if conservative_per_fit_work > max_scored_event_work_per_fit:
        raise SequenceDiscoveryLimitError(
            "selection fit exceeds max_scored_event_work_per_fit before fitting"
        )
    conservative_validation_work = (
        len(fold_cells)
        * len(penalties)
        * 2
        * validation_event_count
    )
    conservative_aggregate_work = (
        required_fit_calls * conservative_per_fit_work
        + conservative_validation_work
    )
    if conservative_aggregate_work > max_aggregate_scored_event_work:
        raise SequenceDiscoveryLimitError(
            "selection exceeds max_aggregate_scored_event_work before fitting"
        )
    folds: list[TracePseudoheldoutFoldCertificate] = []
    score_rows_by_penalty: dict[int, list[TracePseudoheldoutCandidateScore]] = {
        penalty: [] for penalty in penalties
    }
    for pseudo_cell in fold_cells:
        retained_train = tuple(
            trace
            for trace in train_traces
            if pseudo_cell not in (_trace_states(trace) | _trace_dependencies(trace))
        )
        removed_count = len(train_traces) - len(retained_train)
        if not retained_train or removed_count == 0:
            raise ValueError("each pseudo fold needs retained and removed TRAIN sequences")
        selected_validation = tuple(
            trace
            for trace in validation_traces
            if pseudo_cell in _trace_dependencies(trace)
        )
        if not selected_validation:
            raise ValueError("each pseudo fold needs dependent validation queries")
        fit_corpus = make_sequence_corpus(
            corpus.num_surface_keys,
            corpus.value_cardinality,
            split="train",
            sequences=tuple(trace.sequence for trace in retained_train),
        )
        validation_corpus = make_sequence_corpus(
            corpus.num_surface_keys,
            corpus.value_cardinality,
            split="validation",
            sequences=tuple(trace.sequence for trace in selected_validation),
        )
        masks = tuple(
            tuple(
                event.kind is BindingEventKind.QUERY and pseudo_cell in dependencies
                for event, dependencies in zip(
                    trace.sequence.events,
                    trace.attestation.query_dependency_cells,
                    strict=True,
                )
            )
            for trace in selected_validation
        )
        candidates: list[TracePseudoheldoutCandidateScore] = []
        for penalty in penalties:
            model = fit_sequence_algebra(
                fit_corpus,
                residual_penalty=penalty,
                seed=(
                    seed
                    + pseudo_cell[0] * corpus.value_cardinality
                    + pseudo_cell[1]
                ),
                restart_count=restart_count,
                max_sweeps=max_sweeps,
                max_pairwise_rounds=max_pairwise_rounds,
                max_events=max_events_per_fit,
                max_objective_evaluations=max_objective_evaluations_per_fit,
                max_scored_event_work=max_scored_event_work_per_fit,
            )
            pseudo_mistakes, pseudo_count = _score_model(model, validation_corpus, masks)
            all_mistakes, all_count = _score_model(model, validation_corpus)
            row = TracePseudoheldoutCandidateScore(
                residual_penalty=penalty,
                model_fingerprint=model.model_fingerprint,
                pseudo_query_count=pseudo_count,
                pseudo_query_mistakes=pseudo_mistakes,
                all_validation_query_count=all_count,
                all_validation_query_mistakes=all_mistakes,
                training_mistakes=model.fit.training_mistakes,
                residual_override_count=model.fit.residual_override_count,
            )
            candidates.append(row)
            score_rows_by_penalty[penalty].append(row)
        if mode is SequenceSelectionMode.PSEUDOHELDOUT_PRIMARY:
            primary_values = tuple(row.pseudo_query_mistakes for row in candidates)
        else:
            primary_values = tuple(row.all_validation_query_mistakes for row in candidates)
        best_primary = min(primary_values)
        best_penalties = tuple(
            row.residual_penalty
            for row, value in zip(candidates, primary_values, strict=True)
            if value == best_primary
        )
        folds.append(
            TracePseudoheldoutFoldCertificate(
                schema="tnlm-v3-pseudoheldout-sequence-fold-v2",
                pseudoheldout_cell=pseudo_cell,
                optimizer_seed=(
                    seed
                    + pseudo_cell[0] * corpus.value_cardinality
                    + pseudo_cell[1]
                ),
                train_sample_sha256=fit_corpus.sample_sha256,
                validation_sample_sha256=validation_corpus.sample_sha256,
                removed_train_sequence_count=removed_count,
                retained_train_sequence_count=len(retained_train),
                scored_validation_sequence_count=len(selected_validation),
                outer_unobserved_cell_count=len(outer_unobserved),
                candidates=tuple(candidates),
                best_penalties_by_primary_score=best_penalties,
                primary_score_tied=len(best_penalties) > 1,
                oracle_occupancy_used_only_by_fold_firewall=True,
                oracle_dependencies_used_only_for_score_mask=True,
                estimator_received_oracle_metadata=False,
                whole_sequence_pre_post_censoring=True,
                descendant_query_scoring=True,
            )
        )

    aggregates: list[AggregatePenaltyScore] = []
    for penalty in penalties:
        rows = score_rows_by_penalty[penalty]
        aggregates.append(
            AggregatePenaltyScore(
                residual_penalty=penalty,
                pseudo_query_count=sum(row.pseudo_query_count for row in rows),
                pseudo_query_mistakes=sum(row.pseudo_query_mistakes for row in rows),
                all_validation_query_count=sum(
                    row.all_validation_query_count for row in rows
                ),
                all_validation_query_mistakes=sum(
                    row.all_validation_query_mistakes for row in rows
                ),
                total_residual_override_count=sum(
                    row.residual_override_count for row in rows
                ),
            )
        )
    if mode is SequenceSelectionMode.PSEUDOHELDOUT_PRIMARY:
        primary = tuple(row.pseudo_query_mistakes for row in aggregates)
        secondary = tuple(row.all_validation_query_mistakes for row in aggregates)
    else:
        primary = tuple(row.all_validation_query_mistakes for row in aggregates)
        secondary = tuple(row.pseudo_query_mistakes for row in aggregates)
    best_primary_value = min(primary)
    primary_winners = tuple(
        row.residual_penalty
        for row, value in zip(aggregates, primary, strict=True)
        if value == best_primary_value
    )
    ranked = sorted(
        zip(aggregates, primary, secondary, strict=True),
        key=lambda item: (
            item[1],
            item[2],
            item[0].total_residual_override_count,
            -item[0].residual_penalty,
        ),
    )
    selected_penalty = ranked[0][0].residual_penalty
    full_train = make_sequence_corpus(
        corpus.num_surface_keys,
        corpus.value_cardinality,
        split="train",
        sequences=tuple(trace.sequence for trace in train_traces),
    )
    final_model = fit_sequence_algebra(
        full_train,
        residual_penalty=selected_penalty,
        seed=seed,
        restart_count=restart_count,
        max_sweeps=max_sweeps,
        max_pairwise_rounds=max_pairwise_rounds,
        max_events=max_events_per_fit,
        max_objective_evaluations=max_objective_evaluations_per_fit,
        max_scored_event_work=max_scored_event_work_per_fit,
    )
    basis = (
        "pseudo-dependent validation mistakes; all-validation mistakes; "
        "realized residual description length; stronger-penalty deterministic tie-break"
        if mode is SequenceSelectionMode.PSEUDOHELDOUT_PRIMARY
        else
        "all-validation mistakes; pseudo-dependent mistakes; realized residual "
        "description length; stronger-penalty deterministic tie-break (power control only)"
    )
    fold_rows = tuple(folds)
    aggregate_rows = tuple(aggregates)
    power_control = mode is SequenceSelectionMode.OBSERVED_EXCEPTION_POWER_CONTROL
    selection_fields: dict[str, object] = {
        "schema": "tnlm-v3-sequence-algebra-selection-v1",
        "source_corpus_sha256": corpus.corpus_sha256,
        "mode": mode,
        "folds": fold_rows,
        "aggregate_scores": aggregate_rows,
        "selected_residual_penalty": selected_penalty,
        "primary_score_best_penalties": primary_winners,
        "primary_score_tied": len(primary_winners) > 1,
        "selection_basis": basis,
        "final_model": final_model,
        "outer_unobserved_cell_count": len(outer_unobserved),
        "outer_omission_commitment_sha256": _outer_omission_commitment(
            corpus.num_surface_keys,
            corpus.value_cardinality,
            outer_unobserved,
        ),
        "outer_unobserved_identifier_inferred_and_used_by_trusted_firewall": True,
        "outer_unobserved_identifier_received_by_estimator": False,
        "outer_unobserved_labels_used_for_fit_or_selection": False,
        "outer_unobserved_identifier_used_in_candidate_ordering": False,
        "trace_supervised_fold_construction": True,
        "sequence_only_estimator": True,
        "supplied_register_transducer_family": True,
        "automatic_residual_strength_selection_performed": True,
        "automatic_representation_discovery_performed": False,
        "retrospective_protocol_rehearsal": True,
        "confirmatory_claim_permitted": False,
        "observed_exception_power_control_only": power_control,
    }
    return SequenceAlgebraSelectionResult(
        **selection_fields,
        result_sha256=_sha256(_selection_payload_fields(selection_fields)),
    )


@dataclass(frozen=True)
class OuterRotationResult:
    """Results from independently omitted outer environments, without IDs."""

    schema: str
    num_surface_keys: int
    value_cardinality: int
    results: tuple[SequenceAlgebraSelectionResult, ...]
    omitted_cell_sets: tuple[tuple[Cell, ...], ...]
    environment_count: int
    required_outer_unobserved_cell_count: int
    unique_omission_sets: bool
    complete_single_cell_rotation: bool
    all_sequence_estimators_clean: bool
    all_runs_retrospective_rehearsals: bool
    confirmatory_claim_permitted: bool
    result_sha256: str

    def __post_init__(self) -> None:
        if self.schema != "tnlm-v3-sequence-algebra-outer-rotation-v1":
            raise ValueError("unknown outer-rotation schema")
        _plain_int("num_surface_keys", self.num_surface_keys, 2)
        _plain_int("value_cardinality", self.value_cardinality, 2)
        if not isinstance(self.results, tuple) or not self.results:
            raise ValueError("results must be a nonempty exact tuple")
        if any(type(result) is not SequenceAlgebraSelectionResult for result in self.results):
            raise TypeError("results must contain exact selection results")
        _plain_int("environment_count", self.environment_count, 1)
        if self.environment_count != len(self.results):
            raise ValueError("environment count is inconsistent")
        _plain_int(
            "required_outer_unobserved_cell_count",
            self.required_outer_unobserved_cell_count,
            1,
        )
        if not isinstance(self.omitted_cell_sets, tuple) or len(
            self.omitted_cell_sets
        ) != len(self.results):
            raise ValueError("omitted_cell_sets must align with results")
        normalized_omissions: list[tuple[Cell, ...]] = []
        for omission in self.omitted_cell_sets:
            normalized = _normalize_cells(
                "omitted_cell_sets",
                omission,
                self.num_surface_keys,
                self.value_cardinality,
            )
            if len(normalized) != self.required_outer_unobserved_cell_count:
                raise ValueError("an outer omission has the wrong declared size")
            normalized_omissions.append(normalized)
        result_digests = tuple(result.result_sha256 for result in self.results)
        source_digests = tuple(
            result.source_corpus_sha256 for result in self.results
        )
        if len(result_digests) != len(set(result_digests)):
            raise ValueError("outer rotation requires distinct selection results")
        if len(source_digests) != len(set(source_digests)):
            raise ValueError("outer rotation requires distinct source corpora")
        for result, omission in zip(
            self.results, normalized_omissions, strict=True
        ):
            if (
                result.final_model.num_surface_keys != self.num_surface_keys
                or result.final_model.value_cardinality != self.value_cardinality
            ):
                raise ValueError("selection result vocabulary disagrees with rotation")
            if (
                result.outer_unobserved_cell_count
                != self.required_outer_unobserved_cell_count
            ):
                raise ValueError("selection result has the wrong omission count")
            expected_commitment = _outer_omission_commitment(
                self.num_surface_keys,
                self.value_cardinality,
                omission,
            )
            if result.outer_omission_commitment_sha256 != expected_commitment:
                raise ValueError(
                    "selection result is not committed to its declared omission"
                )
        for name in (
            "unique_omission_sets",
            "complete_single_cell_rotation",
            "all_sequence_estimators_clean",
            "all_runs_retrospective_rehearsals",
            "confirmatory_claim_permitted",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact boolean")
        expected_unique = len(set(normalized_omissions)) == len(normalized_omissions)
        if self.unique_omission_sets != expected_unique or not self.unique_omission_sets:
            raise ValueError("outer rotation requires unique omission sets")
        universe = {
            (key, value)
            for key in range(self.num_surface_keys)
            for value in range(self.value_cardinality)
        }
        expected_complete = (
            self.required_outer_unobserved_cell_count == 1
            and len(normalized_omissions) == len(universe)
            and {omission[0] for omission in normalized_omissions} == universe
        )
        if self.complete_single_cell_rotation != expected_complete:
            raise ValueError("complete rotation flag disagrees with omission coverage")
        if not self.all_sequence_estimators_clean:
            raise ValueError("outer rotation contains a contaminated estimator")
        if not self.all_runs_retrospective_rehearsals or self.confirmatory_claim_permitted:
            raise ValueError("current outer rotation is retrospective and non-confirmatory")
        _require_sha256("result_sha256", self.result_sha256)
        expected = _sha256(
            {
                "schema": self.schema,
                "num_surface_keys": self.num_surface_keys,
                "value_cardinality": self.value_cardinality,
                "result_sha256s": [result.result_sha256 for result in self.results],
                "omitted_cell_sets": [
                    [list(cell) for cell in omission]
                    for omission in self.omitted_cell_sets
                ],
                "environment_count": self.environment_count,
                "required_outer_unobserved_cell_count": (
                    self.required_outer_unobserved_cell_count
                ),
                "unique_omission_sets": self.unique_omission_sets,
                "complete_single_cell_rotation": self.complete_single_cell_rotation,
                "all_sequence_estimators_clean": self.all_sequence_estimators_clean,
                "all_runs_retrospective_rehearsals": self.all_runs_retrospective_rehearsals,
                "confirmatory_claim_permitted": self.confirmatory_claim_permitted,
            }
        )
        if self.result_sha256 != expected:
            raise ValueError("result_sha256 does not bind the outer rotation")


def run_outer_rotation(
    corpora: Sequence[TraceSupervisedCorpus],
    *,
    max_environments: int = 256,
    max_outer_aggregate_scored_event_work: int = 2_000_000_000,
    require_complete_single_cell_rotation: bool = False,
    **selection_kwargs: object,
) -> OuterRotationResult:
    """Run one frozen selector over independently omitted environments."""

    if not isinstance(corpora, Sequence) or not corpora:
        raise ValueError("corpora must be a nonempty sequence")
    _plain_int("max_environments", max_environments, 1)
    _plain_int(
        "max_outer_aggregate_scored_event_work",
        max_outer_aggregate_scored_event_work,
        1,
    )
    if type(require_complete_single_cell_rotation) is not bool:
        raise TypeError("require_complete_single_cell_rotation must be exact bool")
    rows = tuple(corpora)
    if len(rows) > max_environments:
        raise SequenceDiscoveryLimitError("outer rotation exceeds max_environments")
    if any(type(corpus) is not TraceSupervisedCorpus for corpus in rows):
        raise TypeError("corpora must contain exact TraceSupervisedCorpus values")
    dimensions = {
        (corpus.num_surface_keys, corpus.value_cardinality) for corpus in rows
    }
    if len(dimensions) != 1:
        raise ValueError("outer-rotation environments must share one vocabulary")
    keys, values = next(iter(dimensions))
    digests = tuple(corpus.corpus_sha256 for corpus in rows)
    if len(digests) != len(set(digests)):
        raise ValueError("outer-rotation environments must have distinct corpora")
    required_omission_count = _plain_int(
        "required_outer_unobserved_cell_count",
        selection_kwargs.get("required_outer_unobserved_cell_count", 1),
        1,
    )
    universe = {
        (key, value) for key in range(keys) for value in range(values)
    }
    omitted_cell_sets = tuple(
        tuple(
            sorted(
                universe
                - {
                    cell
                    for trace in corpus.traces
                    for cell in _trace_states(trace)
                }
            )
        )
        for corpus in rows
    )
    if any(len(omission) != required_omission_count for omission in omitted_cell_sets):
        raise ValueError("outer environment has the wrong omission count")
    if len(set(omitted_cell_sets)) != len(omitted_cell_sets):
        raise ValueError("outer rotation repeats an omitted environment")
    complete_rotation = (
        required_omission_count == 1
        and len(omitted_cell_sets) == len(universe)
        and {omission[0] for omission in omitted_cell_sets} == universe
    )
    if require_complete_single_cell_rotation and not complete_rotation:
        raise ValueError("outer rotation does not cover every single-cell omission")

    penalties_value = selection_kwargs.get("residual_penalties", (0, 1, 4, 16))
    if not isinstance(penalties_value, Sequence) or isinstance(
        penalties_value, (str, bytes)
    ):
        raise TypeError("residual_penalties must be a finite sequence")
    penalties = tuple(penalties_value)
    if len(penalties) < 2 or any(
        type(penalty) is not int or penalty < 0 for penalty in penalties
    ):
        raise ValueError("outer rotation requires at least two valid penalties")
    if penalties != tuple(sorted(set(penalties))):
        raise ValueError("outer-rotation penalties must be unique and sorted")
    restart_count = _plain_int(
        "restart_count", selection_kwargs.get("restart_count", 2), 1
    )
    max_sweeps = _plain_int(
        "max_sweeps", selection_kwargs.get("max_sweeps", 4), 1
    )
    max_pairwise_rounds = _plain_int(
        "max_pairwise_rounds",
        selection_kwargs.get("max_pairwise_rounds", 2),
        0,
    )
    max_objective = _plain_int(
        "max_objective_evaluations_per_fit",
        selection_kwargs.get("max_objective_evaluations_per_fit", 100_000),
        1,
    )
    per_fit_work_cap = _plain_int(
        "max_scored_event_work_per_fit",
        selection_kwargs.get("max_scored_event_work_per_fit", 50_000_000),
        1,
    )
    max_events_per_fit = _plain_int(
        "max_events_per_fit",
        selection_kwargs.get("max_events_per_fit", 250_000),
        1,
    )
    max_controller_events = _plain_int(
        "max_controller_events",
        selection_kwargs.get("max_controller_events", 1_000_000),
        1,
    )
    max_folds = _plain_int(
        "max_folds", selection_kwargs.get("max_folds", 256), 1
    )
    max_fit_calls = _plain_int(
        "max_fit_calls", selection_kwargs.get("max_fit_calls", 2_048), 1
    )
    per_environment_work_cap = _plain_int(
        "max_aggregate_scored_event_work",
        selection_kwargs.get("max_aggregate_scored_event_work", 500_000_000),
        1,
    )
    inventory_size = len(prototype_inventory(values))
    planned_evaluations = restart_count * (
        1
        + max_sweeps * inventory_size * values * (keys + 1)
        + max_pairwise_rounds
        * inventory_size
        * (inventory_size - 1)
        // 2
        * values**2
    )
    if planned_evaluations > max_objective:
        raise SequenceDiscoveryLimitError(
            "outer rotation planned fits exceed objective budget before work"
        )
    outer_work = 0
    for corpus, omission in zip(rows, omitted_cell_sets, strict=True):
        fold_count = len(universe) - len(omission)
        fit_calls = fold_count * len(penalties) + 1
        controller_events = sum(
            len(trace.sequence.events) for trace in corpus.traces
        )
        if controller_events > max_controller_events:
            raise SequenceDiscoveryLimitError(
                "outer environment exceeds selector controller-event cap"
            )
        if fold_count > max_folds:
            raise SequenceDiscoveryLimitError(
                "outer environment exceeds selector fold cap"
            )
        if fit_calls > max_fit_calls:
            raise SequenceDiscoveryLimitError(
                "outer environment exceeds selector fit-call cap"
            )
        train_events = sum(
            len(trace.sequence.events)
            for trace in corpus.traces
            if trace.attestation.split == "train"
        )
        validation_events = sum(
            len(trace.sequence.events)
            for trace in corpus.traces
            if trace.attestation.split == "validation"
        )
        if train_events > max_events_per_fit:
            raise SequenceDiscoveryLimitError(
                "outer environment exceeds selector per-fit event cap"
            )
        per_fit_work = train_events * planned_evaluations
        if per_fit_work > per_fit_work_cap:
            raise SequenceDiscoveryLimitError(
                "outer rotation per-fit work exceeds selector cap before work"
            )
        environment_work = (
            fit_calls * per_fit_work
            + fold_count * len(penalties) * 2 * validation_events
        )
        if environment_work > per_environment_work_cap:
            raise SequenceDiscoveryLimitError(
                "outer rotation environment exceeds selector aggregate cap before work"
            )
        outer_work += environment_work
    if outer_work > max_outer_aggregate_scored_event_work:
        raise SequenceDiscoveryLimitError(
            "outer rotation exceeds aggregate scored-event work before fitting"
        )
    results = tuple(select_sequence_algebra(corpus, **selection_kwargs) for corpus in rows)
    payload = {
        "schema": "tnlm-v3-sequence-algebra-outer-rotation-v1",
        "num_surface_keys": keys,
        "value_cardinality": values,
        "result_sha256s": [result.result_sha256 for result in results],
        "omitted_cell_sets": [
            [list(cell) for cell in omission] for omission in omitted_cell_sets
        ],
        "environment_count": len(results),
        "required_outer_unobserved_cell_count": required_omission_count,
        "unique_omission_sets": True,
        "complete_single_cell_rotation": complete_rotation,
        "all_sequence_estimators_clean": True,
        "all_runs_retrospective_rehearsals": True,
        "confirmatory_claim_permitted": False,
    }
    return OuterRotationResult(
        schema=payload["schema"],
        num_surface_keys=keys,
        value_cardinality=values,
        results=results,
        omitted_cell_sets=omitted_cell_sets,
        environment_count=len(results),
        required_outer_unobserved_cell_count=required_omission_count,
        unique_omission_sets=True,
        complete_single_cell_rotation=complete_rotation,
        all_sequence_estimators_clean=True,
        all_runs_retrospective_rehearsals=True,
        confirmatory_claim_permitted=False,
        result_sha256=_sha256(payload),
    )


__all__ = [
    "AggregatePenaltyScore",
    "ExternalTraceAttestation",
    "LearnedSequenceAlgebra",
    "OuterRotationResult",
    "PrototypeAddress",
    "TracePseudoheldoutCandidateScore",
    "TracePseudoheldoutFoldCertificate",
    "SequenceAlgebraFitCertificate",
    "SequenceAlgebraSelectionResult",
    "SequenceCorpus",
    "SequenceDiscoveryLimitError",
    "SequenceSelectionMode",
    "TraceAttestationSource",
    "TraceSupervisedCorpus",
    "TraceSupervisedSequence",
    "VisibleEvent",
    "VisibleSequence",
    "fit_sequence_algebra",
    "make_sequence_corpus",
    "make_trace_supervised_corpus",
    "make_trace_supervised_sequence",
    "prototype_inventory",
    "run_outer_rotation",
    "select_sequence_algebra",
    "sequence_corpus_from_episodes",
    "visible_sequence_from_episode",
]
