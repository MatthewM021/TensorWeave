"""Observed pair-local-exception power controls for algebra selection.

This module is a deliberately synthetic *observed-exception* control.  It does
not test whether an exception at an unseen cell can be predicted.  It asks a
smaller question: when train and validation traces repeatedly expose a real
destination-key/prototype exception, can the frozen ``(4, 16)`` residual-
penalty selector prefer the local table and realize the right override?

The register interface, event roles, value gauge, and transition inventory are
all supplied.  A trusted generator creates query labels and controller-only
occupancy/dependency attestations; neither the exception declaration nor the
outer-cell identifier is passed to the coefficient estimator.  No benchmark
outer-test answer is read or used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Sequence

from .algebra_discovery import (
    LearnedSequenceAlgebra,
    PrototypeAddress,
    SequenceAlgebraSelectionResult,
    SequenceCorpus,
    SequenceDiscoveryLimitError,
    SequenceSelectionMode,
    TraceSupervisedCorpus,
    TraceSupervisedSequence,
    VisibleEvent,
    VisibleSequence,
    fit_sequence_algebra,
    make_sequence_corpus,
    make_trace_supervised_corpus,
    make_trace_supervised_sequence,
    prototype_inventory,
    select_sequence_algebra,
)
from .data import BindingEventKind


Cell = tuple[int, int]

_SCHEMA_DESIGN = "tnlm-v3-pair-local-power-design-v1"
_SCHEMA_BUDGET = "tnlm-v3-pair-local-power-budget-v1"
_SCHEMA_CORPUS = "tnlm-v3-pair-local-power-corpus-v1"
_SCHEMA_FOLD = "tnlm-v3-pair-local-power-fold-optimum-v1"
_SCHEMA_AUDIT = "tnlm-v3-pair-local-power-direct-audit-v1"
_SCHEMA_RESULT = "tnlm-v3-pair-local-power-condition-result-v1"
_SCHEMA_REPORT = "tnlm-v3-pair-local-power-report-v1"

_EXPECTED_TRAIN_TRACES = 42
_EXPECTED_VALIDATION_TRACES = 42
_EXPECTED_TRAIN_EVENTS = 341
_EXPECTED_VALIDATION_EVENTS = 355
_EXPECTED_TRAIN_QUERIES = 225
_EXPECTED_VALIDATION_QUERIES = 231

_HARD_MAX_TRACES = 10_000
_HARD_MAX_EVENTS = 100_000
_HARD_MAX_QUERIES = 100_000
_HARD_MAX_FIT_CALLS = 1_000
_HARD_MAX_OBJECTIVE_EVALUATIONS = 1_000_000
_HARD_MAX_SCORED_EVENT_WORK = 1_000_000_000


class PowerControlLimitError(RuntimeError):
    """Raised before a declared power-control work budget is exceeded."""


class PowerControlCondition(str, Enum):
    """The paired synthetic environments in the power control."""

    OBSERVED_PAIR_LOCAL_EXCEPTION = "observed_pair_local_exception"
    NO_EXCEPTION = "no_exception"


class PowerTraceRole(str, Enum):
    """Trusted recipe attribution; never forwarded to the estimator."""

    BIND_WITNESS = "bind_witness"
    UPDATE_ZERO_WITNESS = "update_zero_witness"
    UPDATE_ONE_WITNESS = "update_one_witness"
    COPY_WITNESS = "copy_witness"
    DYNAMIC = "dynamic"
    SUPPORT = "support"
    BALANCE = "balance"


class PowerControlScope(str, Enum):
    """Closed claim boundary for this control."""

    OBSERVED_EXCEPTION_ONLY = "observed_exception_power_only_not_unseen_prediction"


def _plain_int(name: str, value: object, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _normalize_cell(name: str, value: object, keys: int, values: int) -> Cell:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be an exact pair")
    key, symbol = value
    _plain_int(f"{name} key", key, 0)
    _plain_int(f"{name} value", symbol, 0)
    if key >= keys or symbol >= values:
        raise ValueError(f"{name} is outside the declared vocabulary")
    return key, symbol


@dataclass(frozen=True, order=True)
class PairLocalExceptionSpec:
    """One destination-key-local UPDATE-table exception."""

    destination_key: int
    transform: int
    source_value: int
    canonical_output: int
    exceptional_output: int
    spec_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "destination_key",
            "transform",
            "source_value",
            "canonical_output",
            "exceptional_output",
        ):
            _plain_int(name, getattr(self, name), 0)
        if self.exceptional_output == self.canonical_output:
            raise ValueError("the exceptional output must differ from the canonical output")
        _require_sha256("spec_sha256", self.spec_sha256)
        if self.spec_sha256 != _sha256(_exception_payload(self)):
            raise ValueError("spec_sha256 does not bind the exception declaration")

    @property
    def address(self) -> PrototypeAddress:
        return PrototypeAddress("update", self.transform, self.source_value)

    @property
    def source_cell(self) -> Cell:
        return self.destination_key, self.source_value


def _exception_payload(spec: PairLocalExceptionSpec) -> dict[str, object]:
    return {
        "domain": "tnlm-v3-pair-local-exception-spec-v1",
        "destination_key": spec.destination_key,
        "transform": spec.transform,
        "source_value": spec.source_value,
        "canonical_output": spec.canonical_output,
        "exceptional_output": spec.exceptional_output,
    }


@dataclass(frozen=True)
class PowerControlDesign:
    """Frozen small design whose analytic optimum is auditable."""

    schema: str
    num_surface_keys: int
    value_cardinality: int
    outer_cell: Cell
    exception: PairLocalExceptionSpec
    dynamic_repetitions: int
    witness_query_repetitions: int
    residual_penalties: tuple[int, int]
    optimizer_seed: int
    restart_count: int
    max_sweeps: int
    max_pairwise_rounds: int
    scope: PowerControlScope
    design_sha256: str

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA_DESIGN:
            raise ValueError("unknown power-control design schema")
        _plain_int("num_surface_keys", self.num_surface_keys, 2)
        _plain_int("value_cardinality", self.value_cardinality, 2)
        _normalize_cell(
            "outer_cell",
            self.outer_cell,
            self.num_surface_keys,
            self.value_cardinality,
        )
        if type(self.exception) is not PairLocalExceptionSpec:
            raise TypeError("exception must be exact PairLocalExceptionSpec")
        if self.exception.destination_key >= self.num_surface_keys:
            raise ValueError("exception destination key is outside the vocabulary")
        for name in ("source_value", "canonical_output", "exceptional_output"):
            if getattr(self.exception, name) >= self.value_cardinality:
                raise ValueError(f"exception {name} is outside the value vocabulary")
        if self.exception.transform >= self.value_cardinality - 1:
            raise ValueError("exception transform must be generator-supported")
        _plain_int("dynamic_repetitions", self.dynamic_repetitions, 1)
        _plain_int("witness_query_repetitions", self.witness_query_repetitions, 1)
        _plain_int("optimizer_seed", self.optimizer_seed, 0)
        _plain_int("restart_count", self.restart_count, 1)
        _plain_int("max_sweeps", self.max_sweeps, 1)
        _plain_int("max_pairwise_rounds", self.max_pairwise_rounds, 0)
        if not isinstance(self.residual_penalties, tuple):
            raise TypeError("residual_penalties must be an exact tuple")
        if self.residual_penalties != (4, 16):
            raise ValueError("the frozen power-control penalty grid is exactly (4, 16)")
        if type(self.scope) is not PowerControlScope:
            raise TypeError("scope must be exact PowerControlScope")
        if self.scope is not PowerControlScope.OBSERVED_EXCEPTION_ONLY:
            raise ValueError("the control cannot claim unseen-exception prediction")
        expected_fixed = (
            self.num_surface_keys,
            self.value_cardinality,
            self.outer_cell,
            self.exception.destination_key,
            self.exception.transform,
            self.exception.source_value,
            self.exception.canonical_output,
            self.exception.exceptional_output,
            self.dynamic_repetitions,
            self.witness_query_repetitions,
            self.optimizer_seed,
            self.restart_count,
            self.max_sweeps,
            self.max_pairwise_rounds,
        )
        if expected_fixed != (4, 3, (3, 2), 1, 0, 0, 1, 2, 2, 7, 0, 2, 4, 2):
            raise ValueError("this certificate implements only the frozen K4/V3 design")
        _require_sha256("design_sha256", self.design_sha256)
        if self.design_sha256 != _sha256(_design_payload(self)):
            raise ValueError("design_sha256 does not bind the frozen design")


def _design_payload(design: PowerControlDesign) -> dict[str, object]:
    return {
        "schema": design.schema,
        "num_surface_keys": design.num_surface_keys,
        "value_cardinality": design.value_cardinality,
        "outer_cell": list(design.outer_cell),
        "exception_spec_sha256": design.exception.spec_sha256,
        "dynamic_repetitions": design.dynamic_repetitions,
        "witness_query_repetitions": design.witness_query_repetitions,
        "residual_penalties": list(design.residual_penalties),
        "optimizer_seed": design.optimizer_seed,
        "restart_count": design.restart_count,
        "max_sweeps": design.max_sweeps,
        "max_pairwise_rounds": design.max_pairwise_rounds,
        "scope": design.scope.value,
    }


def default_power_control_design() -> PowerControlDesign:
    """Return the content-bound, frozen K4/V3 power-control design."""

    exception_fields = {
        "destination_key": 1,
        "transform": 0,
        "source_value": 0,
        "canonical_output": 1,
        "exceptional_output": 2,
    }
    exception_stub = PairLocalExceptionSpec(
        **exception_fields,
        spec_sha256=_sha256(
            {
                "domain": "tnlm-v3-pair-local-exception-spec-v1",
                **exception_fields,
            }
        ),
    )
    fields: dict[str, object] = {
        "schema": _SCHEMA_DESIGN,
        "num_surface_keys": 4,
        "value_cardinality": 3,
        "outer_cell": (3, 2),
        "exception": exception_stub,
        "dynamic_repetitions": 2,
        "witness_query_repetitions": 7,
        "residual_penalties": (4, 16),
        "optimizer_seed": 0,
        "restart_count": 2,
        "max_sweeps": 4,
        "max_pairwise_rounds": 2,
        "scope": PowerControlScope.OBSERVED_EXCEPTION_ONLY,
    }
    temporary = object.__new__(PowerControlDesign)
    for name, value in fields.items():
        object.__setattr__(temporary, name, value)
    object.__setattr__(temporary, "design_sha256", "0" * 64)
    return PowerControlDesign(
        **fields,
        design_sha256=_sha256(_design_payload(temporary)),
    )


@dataclass(frozen=True)
class PowerControlBudget:
    """Fail-before-work resource limits, themselves content-bound."""

    schema: str
    max_traces: int
    max_events: int
    max_queries: int
    max_fit_calls: int
    max_objective_evaluations_per_fit: int
    max_scored_event_work_per_fit: int
    max_aggregate_scored_event_work: int
    budget_sha256: str

    def __post_init__(self) -> None:
        caps = (
            ("max_traces", self.max_traces, _HARD_MAX_TRACES),
            ("max_events", self.max_events, _HARD_MAX_EVENTS),
            ("max_queries", self.max_queries, _HARD_MAX_QUERIES),
            ("max_fit_calls", self.max_fit_calls, _HARD_MAX_FIT_CALLS),
            (
                "max_objective_evaluations_per_fit",
                self.max_objective_evaluations_per_fit,
                _HARD_MAX_OBJECTIVE_EVALUATIONS,
            ),
            (
                "max_scored_event_work_per_fit",
                self.max_scored_event_work_per_fit,
                _HARD_MAX_SCORED_EVENT_WORK,
            ),
            (
                "max_aggregate_scored_event_work",
                self.max_aggregate_scored_event_work,
                _HARD_MAX_SCORED_EVENT_WORK,
            ),
        )
        if self.schema != _SCHEMA_BUDGET:
            raise ValueError("unknown power-control budget schema")
        for name, value, hard_cap in caps:
            _plain_int(name, value, 1)
            if value > hard_cap:
                raise ValueError(f"{name} exceeds the immutable hard safety cap")
        _require_sha256("budget_sha256", self.budget_sha256)
        if self.budget_sha256 != _sha256(_budget_payload(self)):
            raise ValueError("budget_sha256 does not bind the declared limits")


def _budget_payload(budget: PowerControlBudget) -> dict[str, object]:
    return {
        name: getattr(budget, name)
        for name in budget.__dataclass_fields__
        if name != "budget_sha256"
    }


def default_power_control_budget() -> PowerControlBudget:
    fields = {
        "schema": _SCHEMA_BUDGET,
        "max_traces": 200,
        "max_events": 2_000,
        "max_queries": 1_000,
        "max_fit_calls": 64,
        "max_objective_evaluations_per_fit": 4_000,
        "max_scored_event_work_per_fit": 2_000_000,
        "max_aggregate_scored_event_work": 70_000_000,
    }
    return PowerControlBudget(**fields, budget_sha256=_sha256(fields))


@dataclass(frozen=True)
class PowerTraceManifestRow:
    """Controller-only provenance for one deterministically generated trace."""

    split: str
    trace_index: int
    role: PowerTraceRole
    primary_key: int
    source_value: int
    repetition: int
    isolated_address: str | None
    isolated_output: int | None
    witness_query_count: int
    exception_opportunity_count: int
    exception_applied_count: int
    exception_consequence_query_count: int
    validation_crosslink: bool
    query_count: int
    output_class_counts: tuple[int, ...]
    program_sha256: str
    attestation_sha256: str
    row_sha256: str

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation"}:
            raise ValueError("manifest split must be direct train or validation")
        if type(self.role) is not PowerTraceRole:
            raise TypeError("role must be exact PowerTraceRole")
        for name, minimum in (
            ("trace_index", 0),
            ("primary_key", 0),
            ("source_value", 0),
            ("repetition", 0),
            ("witness_query_count", 0),
            ("exception_opportunity_count", 0),
            ("exception_applied_count", 0),
            ("exception_consequence_query_count", 0),
            ("query_count", 1),
        ):
            _plain_int(name, getattr(self, name), minimum)
        if self.exception_applied_count > self.exception_opportunity_count:
            raise ValueError("applied exceptions cannot exceed opportunities")
        if type(self.validation_crosslink) is not bool:
            raise TypeError("validation_crosslink must be an exact boolean")
        if not isinstance(self.output_class_counts, tuple) or any(
            type(value) is not int or value < 0 for value in self.output_class_counts
        ):
            raise TypeError("output_class_counts must be a tuple of nonnegative integers")
        if sum(self.output_class_counts) != self.query_count:
            raise ValueError("output class counts do not sum to the query count")
        if (self.isolated_address is None) != (self.isolated_output is None):
            raise ValueError("isolated address and output must be present together")
        if self.isolated_address is not None and (
            type(self.isolated_address) is not str or not self.isolated_address
        ):
            raise TypeError("isolated_address must be a nonempty exact string")
        if self.isolated_output is not None:
            _plain_int("isolated_output", self.isolated_output, 0)
        for name in ("program_sha256", "attestation_sha256", "row_sha256"):
            _require_sha256(name, getattr(self, name))
        if self.row_sha256 != _sha256(_manifest_row_payload(self)):
            raise ValueError("row_sha256 does not bind trace provenance")


def _manifest_row_payload(row: PowerTraceManifestRow) -> dict[str, object]:
    return {
        "split": row.split,
        "trace_index": row.trace_index,
        "role": row.role.value,
        "primary_key": row.primary_key,
        "source_value": row.source_value,
        "repetition": row.repetition,
        "isolated_address": row.isolated_address,
        "isolated_output": row.isolated_output,
        "witness_query_count": row.witness_query_count,
        "exception_opportunity_count": row.exception_opportunity_count,
        "exception_applied_count": row.exception_applied_count,
        "exception_consequence_query_count": row.exception_consequence_query_count,
        "validation_crosslink": row.validation_crosslink,
        "query_count": row.query_count,
        "output_class_counts": list(row.output_class_counts),
        "program_sha256": row.program_sha256,
        "attestation_sha256": row.attestation_sha256,
    }


@dataclass(frozen=True)
class _TraceRecipe:
    split: str
    role: PowerTraceRole
    primary_key: int
    source_value: int
    repetition: int
    events: tuple[VisibleEvent, ...]
    isolated_address: str | None = None
    isolated_output: int | None = None
    witness_query_count: int = 0
    exception_consequence_query_count: int = 0
    validation_crosslink: bool = False


def _bind(key: int, value: int) -> VisibleEvent:
    return VisibleEvent(BindingEventKind.BIND, primary_key=key, argument=value)


def _update(key: int, transform: int) -> VisibleEvent:
    return VisibleEvent(BindingEventKind.UPDATE, primary_key=key, argument=transform)


def _copy(destination: int, source: int) -> VisibleEvent:
    return VisibleEvent(
        BindingEventKind.COPY,
        primary_key=destination,
        secondary_key=source,
    )


def _query(key: int) -> VisibleEvent:
    return VisibleEvent(BindingEventKind.QUERY, primary_key=key)


def _event_payload(event: VisibleEvent) -> list[int]:
    return [
        int(event.kind),
        event.primary_key,
        event.secondary_key,
        event.argument,
    ]


def _program_sha256(split: str, events: tuple[VisibleEvent, ...]) -> str:
    return _sha256(
        {
            "domain": "tnlm-v3-pair-local-power-visible-program-v1",
            "split": split,
            "events": [_event_payload(event) for event in events],
        }
    )


def _recipes_for_split(design: PowerControlDesign, split: str) -> list[_TraceRecipe]:
    qn = design.witness_query_repetitions
    recipes: list[_TraceRecipe] = []
    for key in (0, 2):
        for source in range(3):
            events = (_bind(key, source), *(_query(key) for _ in range(qn)))
            address = PrototypeAddress("bind", None, source)
            recipes.append(
                _TraceRecipe(
                    split,
                    PowerTraceRole.BIND_WITNESS,
                    key,
                    source,
                    0,
                    events,
                    isolated_address=address.label,
                    isolated_output=source,
                    witness_query_count=qn,
                )
            )
    for key in (0, 2):
        events = (
            _bind(key, 0),
            _update(key, 0),
            *(_query(key) for _ in range(qn)),
        )
        address = PrototypeAddress("update", 0, 0)
        recipes.append(
            _TraceRecipe(
                split,
                PowerTraceRole.UPDATE_ZERO_WITNESS,
                key,
                0,
                0,
                events,
                isolated_address=address.label,
                isolated_output=1,
                witness_query_count=qn,
            )
        )
    for key in (0, 2):
        for source in range(3):
            output = (source + 2) % 3
            events = (
                _bind(key, source),
                _update(key, 1),
                *(_query(key) for _ in range(qn)),
            )
            address = PrototypeAddress("update", 1, source)
            recipes.append(
                _TraceRecipe(
                    split,
                    PowerTraceRole.UPDATE_ONE_WITNESS,
                    key,
                    source,
                    0,
                    events,
                    isolated_address=address.label,
                    isolated_output=output,
                    witness_query_count=qn,
                )
            )
    for destination in (0, 2):
        source_key = 1
        for source in range(3):
            events = (
                _bind(source_key, source),
                _bind(destination, (source + 1) % 3),
                _copy(destination, source_key),
                *(_query(destination) for _ in range(qn)),
            )
            address = PrototypeAddress("copy", None, source)
            recipes.append(
                _TraceRecipe(
                    split,
                    PowerTraceRole.COPY_WITNESS,
                    destination,
                    source,
                    0,
                    events,
                    isolated_address=address.label,
                    isolated_output=source,
                    witness_query_count=qn,
                )
            )
    for key in (0, 1, 2):
        for source in range(3):
            for repetition in range(design.dynamic_repetitions):
                events: tuple[VisibleEvent, ...] = (
                    _bind(key, source),
                    _update(key, 0),
                    _query(key),
                    _update(key, 1),
                    _query(key),
                    _update(key, 1),
                    _query(key),
                )
                locus = key == 1 and source == 0
                crosslink = split == "validation" and locus
                if crosslink:
                    events = (
                        *events,
                        _bind(2, 1),
                        _copy(2, 1),
                        _query(2),
                        _update(2, 0),
                        _query(2),
                        _update(2, 0),
                        _query(2),
                    )
                recipes.append(
                    _TraceRecipe(
                        split,
                        PowerTraceRole.DYNAMIC,
                        key,
                        source,
                        repetition,
                        events,
                        exception_consequence_query_count=3 if locus else 0,
                        validation_crosslink=crosslink,
                    )
                )
    for value in (0, 1):
        recipes.append(
            _TraceRecipe(
                split,
                PowerTraceRole.SUPPORT,
                3,
                value,
                0,
                (_bind(3, value), _query(3)),
            )
        )
    # The preceding deterministic inventory has deficits (14, 0, 15).  One
    # repeated-query trace per deficient class balances the complete split.
    for value, count in ((0, 14), (2, 15)):
        recipes.append(
            _TraceRecipe(
                split,
                PowerTraceRole.BALANCE,
                0,
                value,
                0,
                (_bind(0, value), *(_query(0) for _ in range(count))),
            )
        )
    return recipes


def _attest_recipe(
    design: PowerControlDesign,
    condition: PowerControlCondition,
    recipe: _TraceRecipe,
) -> tuple[TraceSupervisedSequence, int, int]:
    state: dict[int, int] = {}
    lineage: dict[int, set[Cell]] = {}
    targets: list[int | None] = []
    pre_rows: list[tuple[Cell, ...]] = []
    post_rows: list[tuple[Cell, ...]] = []
    dependency_rows: list[tuple[Cell, ...]] = []
    opportunities = 0
    applied = 0
    spec = design.exception
    for event in recipe.events:
        pre_rows.append(tuple(sorted(state.items())))
        dependencies: tuple[Cell, ...] = ()
        key = event.primary_key
        target: int | None = None
        if event.kind is BindingEventKind.BIND:
            state[key] = event.argument
            lineage[key] = {(key, event.argument)}
        elif event.kind is BindingEventKind.UPDATE:
            source = state[key]
            is_opportunity = (
                key == spec.destination_key
                and event.argument == spec.transform
                and source == spec.source_value
            )
            opportunities += int(is_opportunity)
            if (
                is_opportunity
                and condition is PowerControlCondition.OBSERVED_PAIR_LOCAL_EXCEPTION
            ):
                output = spec.exceptional_output
                applied += 1
            else:
                output = (source + event.argument + 1) % design.value_cardinality
            state[key] = output
            lineage[key] = set(lineage[key]) | {(key, output)}
        elif event.kind is BindingEventKind.COPY:
            output = state[event.secondary_key]
            state[key] = output
            lineage[key] = set(lineage[event.secondary_key]) | {(key, output)}
        elif event.kind is BindingEventKind.QUERY:
            target = state[key]
            dependencies = tuple(sorted(lineage[key] | {(key, target)}))
        else:  # The frozen recipes contain no invalidation or distractor.
            raise RuntimeError("unexpected event kind in frozen power recipe")
        targets.append(target)
        post_rows.append(tuple(sorted(state.items())))
        dependency_rows.append(dependencies)
    sequence = VisibleSequence(tuple(recipe.events), tuple(targets))
    trace = make_trace_supervised_sequence(
        sequence,
        split=recipe.split,
        pre_event_cells=tuple(pre_rows),
        post_event_cells=tuple(post_rows),
        query_dependency_cells=tuple(dependency_rows),
        num_surface_keys=design.num_surface_keys,
        value_cardinality=design.value_cardinality,
    )
    return trace, opportunities, applied


def _materialize_control(
    design: PowerControlDesign,
    condition: PowerControlCondition,
) -> tuple[TraceSupervisedCorpus, tuple[PowerTraceManifestRow, ...]]:
    traces: list[TraceSupervisedSequence] = []
    rows: list[PowerTraceManifestRow] = []
    for split in ("train", "validation"):
        for recipe in _recipes_for_split(design, split):
            trace, opportunities, applied = _attest_recipe(design, condition, recipe)
            counts = tuple(
                sum(target == value for target in trace.sequence.query_targets)
                for value in range(design.value_cardinality)
            )
            fields: dict[str, object] = {
                "split": split,
                "trace_index": len(traces),
                "role": recipe.role,
                "primary_key": recipe.primary_key,
                "source_value": recipe.source_value,
                "repetition": recipe.repetition,
                "isolated_address": recipe.isolated_address,
                "isolated_output": recipe.isolated_output,
                "witness_query_count": recipe.witness_query_count,
                "exception_opportunity_count": opportunities,
                "exception_applied_count": applied,
                "exception_consequence_query_count": (
                    recipe.exception_consequence_query_count
                ),
                "validation_crosslink": recipe.validation_crosslink,
                "query_count": trace.sequence.query_count,
                "output_class_counts": counts,
                "program_sha256": _program_sha256(split, recipe.events),
                "attestation_sha256": trace.attestation.attestation_sha256,
            }
            temporary = object.__new__(PowerTraceManifestRow)
            for name, value in fields.items():
                object.__setattr__(temporary, name, value)
            object.__setattr__(temporary, "row_sha256", "0" * 64)
            rows.append(
                PowerTraceManifestRow(
                    **fields,
                    row_sha256=_sha256(_manifest_row_payload(temporary)),
                )
            )
            traces.append(trace)
    return (
        make_trace_supervised_corpus(
            design.num_surface_keys,
            design.value_cardinality,
            tuple(traces),
        ),
        tuple(rows),
    )


def _trace_cells(trace: TraceSupervisedSequence) -> set[Cell]:
    return {
        cell
        for group in (
            trace.attestation.pre_event_cells,
            trace.attestation.post_event_cells,
            trace.attestation.query_dependency_cells,
        )
        for row in group
        for cell in row
    }


def _program_manifest_sha256(rows: tuple[PowerTraceManifestRow, ...]) -> str:
    return _sha256(
        {
            "domain": "tnlm-v3-pair-local-power-program-manifest-v1",
            "rows": [
                [row.split, row.trace_index, row.role.value, row.program_sha256]
                for row in rows
            ],
        }
    )


def _split_counts(
    corpus: TraceSupervisedCorpus,
    split: str,
) -> tuple[int, int, int, tuple[int, ...]]:
    traces = tuple(
        trace for trace in corpus.traces if trace.attestation.split == split
    )
    return (
        len(traces),
        sum(len(trace.sequence.events) for trace in traces),
        sum(trace.sequence.query_count for trace in traces),
        tuple(
            sum(
                target == value
                for trace in traces
                for target in trace.sequence.query_targets
            )
            for value in range(corpus.value_cardinality)
        ),
    )


@dataclass(frozen=True)
class PowerControlCorpus:
    """Content-bound paired corpus plus controller-only provenance."""

    schema: str
    design: PowerControlDesign
    condition: PowerControlCondition
    trace_corpus: TraceSupervisedCorpus
    trace_manifest: tuple[PowerTraceManifestRow, ...]
    program_manifest_sha256: str
    train_trace_count: int
    validation_trace_count: int
    train_event_count: int
    validation_event_count: int
    train_query_count: int
    validation_query_count: int
    train_output_class_counts: tuple[int, ...]
    validation_output_class_counts: tuple[int, ...]
    observed_cells: tuple[Cell, ...]
    train_exception_opportunity_count: int
    validation_exception_opportunity_count: int
    train_exception_applied_count: int
    validation_exception_applied_count: int
    validation_crosslink_trace_count: int
    outer_firewall_passed: bool
    exception_declaration_received_by_estimator: bool
    outer_identifier_received_by_estimator: bool
    outer_test_results_used: bool
    exact_benchmark_executor_imported: bool
    certificate_sha256: str

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA_CORPUS:
            raise ValueError("unknown power-control corpus schema")
        if type(self.design) is not PowerControlDesign:
            raise TypeError("design must be exact PowerControlDesign")
        if type(self.condition) is not PowerControlCondition:
            raise TypeError("condition must be exact PowerControlCondition")
        if type(self.trace_corpus) is not TraceSupervisedCorpus:
            raise TypeError("trace_corpus must be exact TraceSupervisedCorpus")
        if not isinstance(self.trace_manifest, tuple) or any(
            type(row) is not PowerTraceManifestRow for row in self.trace_manifest
        ):
            raise TypeError("trace_manifest must contain exact immutable rows")
        expected_corpus, expected_manifest = _materialize_control(
            self.design, self.condition
        )
        if self.trace_corpus != expected_corpus or self.trace_manifest != expected_manifest:
            raise ValueError("corpus or provenance differs from the frozen recipe")
        expected = _corpus_summary_fields(
            self.design,
            self.condition,
            self.trace_corpus,
            self.trace_manifest,
        )
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} does not reproduce the frozen corpus")
        for name in (
            "outer_firewall_passed",
            "exception_declaration_received_by_estimator",
            "outer_identifier_received_by_estimator",
            "outer_test_results_used",
            "exact_benchmark_executor_imported",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact boolean")
        if not self.outer_firewall_passed or any(
            (
                self.exception_declaration_received_by_estimator,
                self.outer_identifier_received_by_estimator,
                self.outer_test_results_used,
                self.exact_benchmark_executor_imported,
            )
        ):
            raise ValueError("power corpus violates its leakage/scope boundary")
        _require_sha256("certificate_sha256", self.certificate_sha256)
        if self.certificate_sha256 != _sha256(_corpus_payload(self)):
            raise ValueError("certificate_sha256 does not bind the power corpus")


def _corpus_summary_fields(
    design: PowerControlDesign,
    condition: PowerControlCondition,
    corpus: TraceSupervisedCorpus,
    manifest: tuple[PowerTraceManifestRow, ...],
) -> dict[str, object]:
    train = _split_counts(corpus, "train")
    validation = _split_counts(corpus, "validation")
    observed = tuple(sorted({cell for trace in corpus.traces for cell in _trace_cells(trace)}))
    outer_free = design.outer_cell not in observed
    return {
        "program_manifest_sha256": _program_manifest_sha256(manifest),
        "train_trace_count": train[0],
        "validation_trace_count": validation[0],
        "train_event_count": train[1],
        "validation_event_count": validation[1],
        "train_query_count": train[2],
        "validation_query_count": validation[2],
        "train_output_class_counts": train[3],
        "validation_output_class_counts": validation[3],
        "observed_cells": observed,
        "train_exception_opportunity_count": sum(
            row.exception_opportunity_count for row in manifest if row.split == "train"
        ),
        "validation_exception_opportunity_count": sum(
            row.exception_opportunity_count
            for row in manifest
            if row.split == "validation"
        ),
        "train_exception_applied_count": sum(
            row.exception_applied_count for row in manifest if row.split == "train"
        ),
        "validation_exception_applied_count": sum(
            row.exception_applied_count
            for row in manifest
            if row.split == "validation"
        ),
        "validation_crosslink_trace_count": sum(
            row.validation_crosslink for row in manifest
        ),
        "outer_firewall_passed": outer_free,
        "exception_declaration_received_by_estimator": False,
        "outer_identifier_received_by_estimator": False,
        "outer_test_results_used": False,
        "exact_benchmark_executor_imported": False,
    }


def _corpus_payload(bundle: PowerControlCorpus) -> dict[str, object]:
    return {
        "schema": bundle.schema,
        "design_sha256": bundle.design.design_sha256,
        "condition": bundle.condition.value,
        "trace_corpus_sha256": bundle.trace_corpus.corpus_sha256,
        "trace_manifest_row_sha256": [row.row_sha256 for row in bundle.trace_manifest],
        **{
            name: getattr(bundle, name)
            for name in (
                "program_manifest_sha256",
                "train_trace_count",
                "validation_trace_count",
                "train_event_count",
                "validation_event_count",
                "train_query_count",
                "validation_query_count",
                "train_exception_opportunity_count",
                "validation_exception_opportunity_count",
                "train_exception_applied_count",
                "validation_exception_applied_count",
                "validation_crosslink_trace_count",
                "outer_firewall_passed",
                "exception_declaration_received_by_estimator",
                "outer_identifier_received_by_estimator",
                "outer_test_results_used",
                "exact_benchmark_executor_imported",
            )
        },
        "train_output_class_counts": list(bundle.train_output_class_counts),
        "validation_output_class_counts": list(bundle.validation_output_class_counts),
        "observed_cells": [list(cell) for cell in bundle.observed_cells],
    }


def _check_corpus_budget(budget: PowerControlBudget) -> None:
    requirements = (
        ("max_traces", _EXPECTED_TRAIN_TRACES + _EXPECTED_VALIDATION_TRACES),
        ("max_events", _EXPECTED_TRAIN_EVENTS + _EXPECTED_VALIDATION_EVENTS),
        ("max_queries", _EXPECTED_TRAIN_QUERIES + _EXPECTED_VALIDATION_QUERIES),
    )
    for name, required in requirements:
        if getattr(budget, name) < required:
            raise PowerControlLimitError(f"power corpus exceeds {name} before generation")


def build_power_control_corpus(
    condition: PowerControlCondition,
    *,
    design: PowerControlDesign | None = None,
    budget: PowerControlBudget | None = None,
) -> PowerControlCorpus:
    """Build one condition without consulting any outer-test result."""

    if type(condition) is not PowerControlCondition:
        raise TypeError("condition must be exact PowerControlCondition")
    design = default_power_control_design() if design is None else design
    budget = default_power_control_budget() if budget is None else budget
    if type(design) is not PowerControlDesign:
        raise TypeError("design must be exact PowerControlDesign")
    if type(budget) is not PowerControlBudget:
        raise TypeError("budget must be exact PowerControlBudget")
    _check_corpus_budget(budget)
    corpus, manifest = _materialize_control(design, condition)
    fields: dict[str, object] = {
        "schema": _SCHEMA_CORPUS,
        "design": design,
        "condition": condition,
        "trace_corpus": corpus,
        "trace_manifest": manifest,
        **_corpus_summary_fields(design, condition, corpus, manifest),
    }
    temporary = object.__new__(PowerControlCorpus)
    for name, value in fields.items():
        object.__setattr__(temporary, name, value)
    object.__setattr__(temporary, "certificate_sha256", "0" * 64)
    return PowerControlCorpus(
        **fields,
        certificate_sha256=_sha256(_corpus_payload(temporary)),
    )


def _direct_corpora(bundle: PowerControlCorpus) -> tuple[SequenceCorpus, SequenceCorpus]:
    train = tuple(
        trace.sequence
        for trace in bundle.trace_corpus.traces
        if trace.attestation.split == "train"
    )
    validation = tuple(
        trace.sequence
        for trace in bundle.trace_corpus.traces
        if trace.attestation.split == "validation"
    )
    return (
        make_sequence_corpus(
            bundle.design.num_surface_keys,
            bundle.design.value_cardinality,
            split="train",
            sequences=train,
        ),
        make_sequence_corpus(
            bundle.design.num_surface_keys,
            bundle.design.value_cardinality,
            split="validation",
            sequences=validation,
        ),
    )


def _score(model: LearnedSequenceAlgebra, corpus: SequenceCorpus) -> tuple[int, int]:
    mistakes = 0
    count = 0
    for sequence in corpus.sequences:
        predictions = model.predict(sequence)
        for prediction, target in zip(predictions, sequence.query_targets, strict=True):
            if target is None:
                continue
            count += 1
            mistakes += prediction != target
    return mistakes, count


@dataclass(frozen=True)
class PowerFoldOptimumCertificate:
    schema: str
    pseudoheldout_cell: Cell
    residual_penalty: int
    train_sample_sha256: str
    candidate_model_fingerprint: str
    retained_exception_opportunity_count: int
    exception_consequence_query_count: int
    surviving_isolation_witness_floor: int
    hard_shared_mistake_lower_bound: int
    penalized_objective_lower_bound: int
    attained_training_mistakes: int
    attained_override_count: int
    attained_penalized_objective: int
    global_optimum_certified_for_frozen_control: bool
    self_pseudoheldout_nonidentifying: bool
    certificate_sha256: str

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA_FOLD:
            raise ValueError("unknown fold-optimum certificate schema")
        for name in ("train_sample_sha256", "candidate_model_fingerprint", "certificate_sha256"):
            _require_sha256(name, getattr(self, name))
        for name in (
            "residual_penalty",
            "retained_exception_opportunity_count",
            "exception_consequence_query_count",
            "surviving_isolation_witness_floor",
            "hard_shared_mistake_lower_bound",
            "penalized_objective_lower_bound",
            "attained_training_mistakes",
            "attained_override_count",
            "attained_penalized_objective",
        ):
            _plain_int(name, getattr(self, name), 0)
        for name in (
            "global_optimum_certified_for_frozen_control",
            "self_pseudoheldout_nonidentifying",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact boolean")
        if not self.global_optimum_certified_for_frozen_control:
            raise ValueError("every interpreted power candidate must attain a proven bound")
        if self.attained_penalized_objective != (
            self.attained_training_mistakes
            + self.residual_penalty * self.attained_override_count
        ):
            raise ValueError("attained fold objective is inconsistent")
        if self.attained_penalized_objective != self.penalized_objective_lower_bound:
            raise ValueError("fold candidate does not attain its analytic lower bound")
        if self.certificate_sha256 != _sha256(_fold_payload(self)):
            raise ValueError("certificate_sha256 does not bind the fold optimum")


def _fold_payload(row: PowerFoldOptimumCertificate) -> dict[str, object]:
    return {
        name: (list(value) if name == "pseudoheldout_cell" else value)
        for name, value in (
            (field, getattr(row, field))
            for field in row.__dataclass_fields__
            if field != "certificate_sha256"
        )
    }


def _retained_fold_rows(
    bundle: PowerControlCorpus,
    pseudo_cell: Cell,
) -> tuple[
    tuple[TraceSupervisedSequence, ...],
    tuple[PowerTraceManifestRow, ...],
    tuple[TraceSupervisedSequence, ...],
]:
    paired = tuple(zip(bundle.trace_corpus.traces, bundle.trace_manifest, strict=True))
    retained = tuple(
        (trace, row)
        for trace, row in paired
        if trace.attestation.split == "train" and pseudo_cell not in _trace_cells(trace)
    )
    selected_validation = tuple(
        trace
        for trace, _ in paired
        if trace.attestation.split == "validation"
        and any(
            pseudo_cell in dependencies
            for dependencies in trace.attestation.query_dependency_cells
        )
    )
    return (
        tuple(trace for trace, _ in retained),
        tuple(row for _, row in retained),
        selected_validation,
    )


def _isolation_witness_floor(rows: tuple[PowerTraceManifestRow, ...]) -> int:
    required = (
        "bind:source:0",
        "update:0:source:0",
        "update:1:source:0",
        "update:1:source:1",
        "update:1:source:2",
    )
    counts = tuple(
        sum(
            row.witness_query_count
            for row in rows
            if row.isolated_address == address
        )
        for address in required
    )
    return min(counts)


def _make_fold_certificate(
    *,
    bundle: PowerControlCorpus,
    fold: object,
    candidate: object,
) -> PowerFoldOptimumCertificate:
    pseudo_cell = fold.pseudoheldout_cell
    retained_traces, retained_rows, selected_validation = _retained_fold_rows(
        bundle, pseudo_cell
    )
    fit_corpus = make_sequence_corpus(
        bundle.design.num_surface_keys,
        bundle.design.value_cardinality,
        split="train",
        sequences=tuple(trace.sequence for trace in retained_traces),
    )
    validation_corpus = make_sequence_corpus(
        bundle.design.num_surface_keys,
        bundle.design.value_cardinality,
        split="validation",
        sequences=tuple(trace.sequence for trace in selected_validation),
    )
    if fold.train_sample_sha256 != fit_corpus.sample_sha256:
        raise ValueError("selector fold TRAIN hash disagrees with reconstructed provenance")
    if fold.validation_sample_sha256 != validation_corpus.sample_sha256:
        raise ValueError("selector fold validation hash disagrees with reconstructed provenance")
    train_total = sum(
        trace.attestation.split == "train" for trace in bundle.trace_corpus.traces
    )
    if fold.retained_train_sequence_count != len(retained_traces):
        raise ValueError("selector retained count disagrees with reconstructed firewall")
    if fold.removed_train_sequence_count != train_total - len(retained_traces):
        raise ValueError("selector removed count disagrees with reconstructed firewall")
    if fold.scored_validation_sequence_count != len(selected_validation):
        raise ValueError("selector validation count disagrees with reconstructed provenance")
    opportunities = sum(row.exception_opportunity_count for row in retained_rows)
    consequences = sum(row.exception_consequence_query_count for row in retained_rows)
    if opportunities not in {0, bundle.design.dynamic_repetitions}:
        raise ValueError("partial exception retention violates the frozen fold proof")
    witness_floor = _isolation_witness_floor(retained_rows)
    if witness_floor < bundle.design.witness_query_repetitions:
        raise ValueError("a fold lost the isolation witnesses required by the proof")
    positive = bundle.condition is PowerControlCondition.OBSERVED_PAIR_LOCAL_EXCEPTION
    hard_lower = consequences if positive else 0
    objective_lower = min(hard_lower, candidate.residual_penalty)
    attained = (
        candidate.training_mistakes
        + candidate.residual_penalty * candidate.residual_override_count
    )
    self_fold = pseudo_cell[0] == bundle.design.exception.destination_key
    if self_fold != (opportunities == 0):
        raise ValueError("self-pseudoheldout attribution disagrees with trace censoring")
    fields: dict[str, object] = {
        "schema": _SCHEMA_FOLD,
        "pseudoheldout_cell": pseudo_cell,
        "residual_penalty": candidate.residual_penalty,
        "train_sample_sha256": fit_corpus.sample_sha256,
        "candidate_model_fingerprint": candidate.model_fingerprint,
        "retained_exception_opportunity_count": opportunities,
        "exception_consequence_query_count": consequences,
        "surviving_isolation_witness_floor": witness_floor,
        "hard_shared_mistake_lower_bound": hard_lower,
        "penalized_objective_lower_bound": objective_lower,
        "attained_training_mistakes": candidate.training_mistakes,
        "attained_override_count": candidate.residual_override_count,
        "attained_penalized_objective": attained,
        "global_optimum_certified_for_frozen_control": attained == objective_lower,
        "self_pseudoheldout_nonidentifying": self_fold,
    }
    temporary = object.__new__(PowerFoldOptimumCertificate)
    for name, value in fields.items():
        object.__setattr__(temporary, name, value)
    object.__setattr__(temporary, "certificate_sha256", "0" * 64)
    return PowerFoldOptimumCertificate(
        **fields,
        certificate_sha256=_sha256(_fold_payload(temporary)),
    )


@dataclass(frozen=True)
class PowerPenaltyAudit:
    """Direct full-seen TRAIN/validation audit for one frozen penalty."""

    schema: str
    residual_penalty: int
    model: LearnedSequenceAlgebra
    train_sample_sha256: str
    validation_sample_sha256: str
    training_mistakes: int
    training_query_count: int
    validation_mistakes: int
    validation_query_count: int
    analytic_training_objective_lower_bound: int
    attained_training_objective: int
    global_optimum_certified_for_frozen_control: bool
    canonical_shared_table_realized: bool
    semantic_decomposition_gauge_fixed: bool
    expected_exception_override_realized: bool
    certificate_sha256: str

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA_AUDIT:
            raise ValueError("unknown direct-audit schema")
        if type(self.model) is not LearnedSequenceAlgebra:
            raise TypeError("model must be exact LearnedSequenceAlgebra")
        for name in ("train_sample_sha256", "validation_sample_sha256", "certificate_sha256"):
            _require_sha256(name, getattr(self, name))
        for name in (
            "residual_penalty",
            "training_mistakes",
            "training_query_count",
            "validation_mistakes",
            "validation_query_count",
            "analytic_training_objective_lower_bound",
            "attained_training_objective",
        ):
            _plain_int(name, getattr(self, name), 0)
        if type(self.global_optimum_certified_for_frozen_control) is not bool:
            raise TypeError("global optimum flag must be an exact boolean")
        for name in (
            "canonical_shared_table_realized",
            "semantic_decomposition_gauge_fixed",
            "expected_exception_override_realized",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact boolean")
        if self.model.fit.residual_penalty != self.residual_penalty:
            raise ValueError("audit penalty and model disagree")
        if self.model.fit.training_mistakes != self.training_mistakes:
            raise ValueError("audit and model TRAIN mistakes disagree")
        if self.model.fit.training_query_count != self.training_query_count:
            raise ValueError("audit and model TRAIN query counts disagree")
        if self.model.fit.training_sample_sha256 != self.train_sample_sha256:
            raise ValueError("audit and model TRAIN hashes disagree")
        if self.attained_training_objective != self.model.fit.penalized_objective:
            raise ValueError("audit and model objectives disagree")
        if self.attained_training_objective != self.analytic_training_objective_lower_bound:
            raise ValueError("direct candidate does not attain its analytic lower bound")
        if not self.global_optimum_certified_for_frozen_control:
            raise ValueError("direct candidate lacks the required optimum certificate")
        if not self.canonical_shared_table_realized:
            raise ValueError("direct candidate changed the declared shared semantics")
        if not self.semantic_decomposition_gauge_fixed:
            raise ValueError("direct candidate does not fix the preregistered residual gauge")
        if self.certificate_sha256 != _sha256(_audit_payload(self)):
            raise ValueError("certificate_sha256 does not bind the direct audit")


def _audit_payload(audit: PowerPenaltyAudit) -> dict[str, object]:
    return {
        "schema": audit.schema,
        "residual_penalty": audit.residual_penalty,
        "model_fingerprint": audit.model.model_fingerprint,
        "train_sample_sha256": audit.train_sample_sha256,
        "validation_sample_sha256": audit.validation_sample_sha256,
        "training_mistakes": audit.training_mistakes,
        "training_query_count": audit.training_query_count,
        "validation_mistakes": audit.validation_mistakes,
        "validation_query_count": audit.validation_query_count,
        "analytic_training_objective_lower_bound": audit.analytic_training_objective_lower_bound,
        "attained_training_objective": audit.attained_training_objective,
        "global_optimum_certified_for_frozen_control": audit.global_optimum_certified_for_frozen_control,
        "canonical_shared_table_realized": audit.canonical_shared_table_realized,
        "semantic_decomposition_gauge_fixed": audit.semantic_decomposition_gauge_fixed,
        "expected_exception_override_realized": audit.expected_exception_override_realized,
    }


def _canonical_shared_outputs(
    design: PowerControlDesign,
) -> tuple[tuple[PrototypeAddress, int], ...]:
    rows: list[tuple[PrototypeAddress, int]] = []
    for address in prototype_inventory(design.value_cardinality):
        if address.family in {"bind", "copy"}:
            output = address.source_value
        else:
            if address.transform is None:
                raise RuntimeError("validated UPDATE address unexpectedly lacks a transform")
            output = (
                address.source_value + address.transform + 1
            ) % design.value_cardinality
        rows.append((address, output))
    return tuple(rows)


def _make_direct_audit(
    bundle: PowerControlCorpus,
    model: LearnedSequenceAlgebra,
    train: SequenceCorpus,
    validation: SequenceCorpus,
) -> PowerPenaltyAudit:
    train_mistakes, train_count = _score(model, train)
    validation_mistakes, validation_count = _score(model, validation)
    positive = bundle.condition is PowerControlCondition.OBSERVED_PAIR_LOCAL_EXCEPTION
    lower = min(6 if positive else 0, model.fit.residual_penalty)
    expected_override = (
        (
            bundle.design.exception.destination_key,
            bundle.design.exception.address,
            bundle.design.exception.exceptional_output,
        ),
    )
    realized = model.local_overrides == expected_override
    canonical_shared = model.shared_outputs == _canonical_shared_outputs(bundle.design)
    if positive and model.fit.residual_penalty == 4 and not realized:
        raise ValueError("low-penalty final fit did not realize the declared exception")
    if (not positive or model.fit.residual_penalty == 16) and model.local_overrides:
        raise ValueError("null/high-penalty fit unexpectedly contains local overrides")
    fields: dict[str, object] = {
        "schema": _SCHEMA_AUDIT,
        "residual_penalty": model.fit.residual_penalty,
        "model": model,
        "train_sample_sha256": train.sample_sha256,
        "validation_sample_sha256": validation.sample_sha256,
        "training_mistakes": train_mistakes,
        "training_query_count": train_count,
        "validation_mistakes": validation_mistakes,
        "validation_query_count": validation_count,
        "analytic_training_objective_lower_bound": lower,
        "attained_training_objective": model.fit.penalized_objective,
        "global_optimum_certified_for_frozen_control": (
            model.fit.penalized_objective == lower
        ),
        "canonical_shared_table_realized": canonical_shared,
        # Canonical shared rows plus either the exact declared one-row residual
        # or no residual rule out the misleading reversed gauge in which a
        # human-normal destination carries the override.
        "semantic_decomposition_gauge_fixed": canonical_shared
        and (
            realized
            if positive and model.fit.residual_penalty == 4
            else not model.local_overrides
        ),
        "expected_exception_override_realized": realized,
    }
    temporary = object.__new__(PowerPenaltyAudit)
    for name, value in fields.items():
        object.__setattr__(temporary, name, value)
    object.__setattr__(temporary, "certificate_sha256", "0" * 64)
    return PowerPenaltyAudit(
        **fields,
        certificate_sha256=_sha256(_audit_payload(temporary)),
    )


@dataclass(frozen=True)
class PowerConditionResult:
    schema: str
    corpus: PowerControlCorpus
    selection: SequenceAlgebraSelectionResult
    fold_optimum_certificates: tuple[PowerFoldOptimumCertificate, ...]
    direct_penalty_audits: tuple[PowerPenaltyAudit, ...]
    selected_sequence_validation_margin: int
    direct_full_validation_margin: int
    crosslink_winning_cells: tuple[Cell, ...]
    self_pseudoheldout_cells: tuple[Cell, ...]
    outer_results_used_for_tuning: bool
    unseen_exception_prediction_claimed: bool
    representation_discovery_performed: bool
    supplied_register_interface: bool
    result_sha256: str

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA_RESULT:
            raise ValueError("unknown condition-result schema")
        if type(self.corpus) is not PowerControlCorpus:
            raise TypeError("corpus must be exact PowerControlCorpus")
        if type(self.selection) is not SequenceAlgebraSelectionResult:
            raise TypeError("selection must be exact SequenceAlgebraSelectionResult")
        if self.selection.source_corpus_sha256 != self.corpus.trace_corpus.corpus_sha256:
            raise ValueError("selection is not bound to the declared power corpus")
        if self.selection.mode is not SequenceSelectionMode.OBSERVED_EXCEPTION_POWER_CONTROL:
            raise ValueError("power result must use the observed-exception selector mode")
        expected_folds = _derive_fold_certificates(self.corpus, self.selection)
        if self.fold_optimum_certificates != expected_folds:
            raise ValueError("fold optimum certificates do not reproduce selection/provenance")
        if not isinstance(self.direct_penalty_audits, tuple) or tuple(
            audit.residual_penalty for audit in self.direct_penalty_audits
        ) != self.corpus.design.residual_penalties:
            raise ValueError("direct audits must follow the frozen penalty inventory")
        train, validation = _direct_corpora(self.corpus)
        for audit in self.direct_penalty_audits:
            rebuilt = _make_direct_audit(
                self.corpus,
                audit.model,
                train,
                validation,
            )
            if audit != rebuilt:
                raise ValueError(
                    "direct full-split audit does not reproduce its model/corpus"
                )
        audits = {audit.residual_penalty: audit for audit in self.direct_penalty_audits}
        aggregates = {
            row.residual_penalty: row for row in self.selection.aggregate_scores
        }
        expected_selected_margin = (
            aggregates[16].all_validation_query_mistakes
            - aggregates[4].all_validation_query_mistakes
        )
        expected_direct_margin = audits[16].validation_mistakes - audits[4].validation_mistakes
        if self.selected_sequence_validation_margin != expected_selected_margin:
            raise ValueError("selected-sequence validation margin is inconsistent")
        if self.direct_full_validation_margin != expected_direct_margin:
            raise ValueError("direct full-validation margin is inconsistent")
        expected_crosslinks = tuple(
            fold.pseudoheldout_cell
            for fold in self.selection.folds
            if fold.pseudoheldout_cell[0] == 2
            and fold.candidates[0].all_validation_query_mistakes
            < fold.candidates[1].all_validation_query_mistakes
        )
        if self.crosslink_winning_cells != expected_crosslinks:
            raise ValueError("crosslink-winning cell inventory is inconsistent")
        expected_self = tuple(
            (self.corpus.design.exception.destination_key, value)
            for value in range(self.corpus.design.value_cardinality)
        )
        if self.self_pseudoheldout_cells != expected_self:
            raise ValueError("self-pseudoheldout inventory is inconsistent")
        self_rows = tuple(
            row
            for row in self.fold_optimum_certificates
            if row.self_pseudoheldout_nonidentifying
        )
        if any(row.retained_exception_opportunity_count != 0 for row in self_rows):
            raise ValueError("self folds must censor all matching exception evidence")
        positive = (
            self.corpus.condition
            is PowerControlCondition.OBSERVED_PAIR_LOCAL_EXCEPTION
        )
        expected_penalty = 4 if positive else 16
        if self.selection.selected_residual_penalty != expected_penalty:
            raise ValueError("selector did not exhibit the preregistered control behavior")
        selected_audit = audits[expected_penalty]
        if self.selection.final_model.model_fingerprint != selected_audit.model.model_fingerprint:
            raise ValueError("selected final model and direct audit disagree")
        if positive:
            if self.selected_sequence_validation_margin <= 0:
                raise ValueError("positive control lacks selected-sequence validation power")
            if self.direct_full_validation_margin <= 0:
                raise ValueError("positive control lacks a separate full-split margin")
            if not self.crosslink_winning_cells:
                raise ValueError("positive control lacks a crosslinked identifying fold")
            if not audits[4].expected_exception_override_realized:
                raise ValueError("positive control lacks the exact declared override")
        else:
            if self.selected_sequence_validation_margin != 0:
                raise ValueError("null control should tie on selected validation")
            if self.direct_full_validation_margin != 0:
                raise ValueError("null control should tie on direct validation")
            if self.crosslink_winning_cells:
                raise ValueError("null control cannot have exception-winning anchors")
            if any(audit.model.local_overrides for audit in self.direct_penalty_audits):
                raise ValueError("null control must realize no local overrides")
            if any(audit.training_mistakes for audit in self.direct_penalty_audits):
                raise ValueError("both null candidates must fit TRAIN exactly")
        for name in (
            "outer_results_used_for_tuning",
            "unseen_exception_prediction_claimed",
            "representation_discovery_performed",
            "supplied_register_interface",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact boolean")
        if any(
            (
                self.outer_results_used_for_tuning,
                self.unseen_exception_prediction_claimed,
                self.representation_discovery_performed,
                not self.supplied_register_interface,
            )
        ):
            raise ValueError("condition result exceeds the honest control scope")
        _require_sha256("result_sha256", self.result_sha256)
        if self.result_sha256 != _sha256(_condition_result_payload(self)):
            raise ValueError("result_sha256 does not bind the condition result")


def _derive_fold_certificates(
    bundle: PowerControlCorpus,
    selection: SequenceAlgebraSelectionResult,
) -> tuple[PowerFoldOptimumCertificate, ...]:
    rows: list[PowerFoldOptimumCertificate] = []
    for fold in selection.folds:
        for candidate in fold.candidates:
            rows.append(
                _make_fold_certificate(
                    bundle=bundle,
                    fold=fold,
                    candidate=candidate,
                )
            )
    return tuple(rows)


def _condition_result_payload(result: PowerConditionResult) -> dict[str, object]:
    return {
        "schema": result.schema,
        "corpus_certificate_sha256": result.corpus.certificate_sha256,
        "selection_result_sha256": result.selection.result_sha256,
        "fold_certificate_sha256": [
            row.certificate_sha256 for row in result.fold_optimum_certificates
        ],
        "direct_audit_sha256": [
            row.certificate_sha256 for row in result.direct_penalty_audits
        ],
        "selected_sequence_validation_margin": result.selected_sequence_validation_margin,
        "direct_full_validation_margin": result.direct_full_validation_margin,
        "crosslink_winning_cells": [list(cell) for cell in result.crosslink_winning_cells],
        "self_pseudoheldout_cells": [list(cell) for cell in result.self_pseudoheldout_cells],
        "outer_results_used_for_tuning": result.outer_results_used_for_tuning,
        "unseen_exception_prediction_claimed": result.unseen_exception_prediction_claimed,
        "representation_discovery_performed": result.representation_discovery_performed,
        "supplied_register_interface": result.supplied_register_interface,
    }


def _planned_evaluations(design: PowerControlDesign) -> int:
    inventory_size = len(prototype_inventory(design.value_cardinality))
    # Pairwise search is data-adaptive.  Budget the fail-before-work bound as
    # if every table address were uncertain, even though the frozen witnesses
    # normally make only the declared conflicting address uncertain.
    worst_pairwise = (
        design.max_pairwise_rounds
        * inventory_size
        * (inventory_size - 1)
        // 2
        * design.value_cardinality**2
    )
    return design.restart_count * (
        1
        + design.max_sweeps
        * inventory_size
        * design.value_cardinality
        * (design.num_surface_keys + 1)
        + worst_pairwise
    )


def _check_run_budget(
    design: PowerControlDesign,
    budget: PowerControlBudget,
    *,
    condition_count: int,
) -> None:
    folds = design.num_surface_keys * design.value_cardinality - 1
    fit_calls_per_condition = folds * len(design.residual_penalties) + 2
    fit_calls = condition_count * fit_calls_per_condition
    if fit_calls > budget.max_fit_calls:
        raise PowerControlLimitError("power run exceeds max_fit_calls before fitting")
    evaluations = _planned_evaluations(design)
    if evaluations > budget.max_objective_evaluations_per_fit:
        raise PowerControlLimitError(
            "power run exceeds max_objective_evaluations_per_fit before fitting"
        )
    per_fit_work = _EXPECTED_TRAIN_EVENTS * evaluations
    if per_fit_work > budget.max_scored_event_work_per_fit:
        raise PowerControlLimitError(
            "power run exceeds max_scored_event_work_per_fit before fitting"
        )
    if fit_calls * per_fit_work > budget.max_aggregate_scored_event_work:
        raise PowerControlLimitError(
            "power run exceeds max_aggregate_scored_event_work before fitting"
        )


def run_power_condition(
    corpus: PowerControlCorpus,
    *,
    budget: PowerControlBudget | None = None,
) -> PowerConditionResult:
    """Run the selector and a separate audit of its complete validation split.

    The audit changes the scoring view, not the data: it uses the same direct
    validation corpus already available to fold selection.  It is therefore
    not an independent or untouched holdout.
    """

    if type(corpus) is not PowerControlCorpus:
        raise TypeError("corpus must be exact PowerControlCorpus")
    budget = default_power_control_budget() if budget is None else budget
    if type(budget) is not PowerControlBudget:
        raise TypeError("budget must be exact PowerControlBudget")
    _check_corpus_budget(budget)
    _check_run_budget(corpus.design, budget, condition_count=1)
    design = corpus.design
    folds = design.num_surface_keys * design.value_cardinality - 1
    selection = select_sequence_algebra(
        corpus.trace_corpus,
        residual_penalties=design.residual_penalties,
        seed=design.optimizer_seed,
        restart_count=design.restart_count,
        max_sweeps=design.max_sweeps,
        mode=SequenceSelectionMode.OBSERVED_EXCEPTION_POWER_CONTROL,
        required_outer_unobserved_cell_count=1,
        max_folds=folds,
        max_fit_calls=folds * len(design.residual_penalties) + 1,
        max_controller_events=budget.max_events,
        max_events_per_fit=budget.max_events,
        max_objective_evaluations_per_fit=(
            budget.max_objective_evaluations_per_fit
        ),
        max_scored_event_work_per_fit=budget.max_scored_event_work_per_fit,
        max_aggregate_scored_event_work=budget.max_aggregate_scored_event_work,
    )
    train, validation = _direct_corpora(corpus)
    models: dict[int, LearnedSequenceAlgebra] = {
        selection.selected_residual_penalty: selection.final_model
    }
    for penalty in design.residual_penalties:
        if penalty in models:
            continue
        models[penalty] = fit_sequence_algebra(
            train,
            residual_penalty=penalty,
            seed=design.optimizer_seed,
            restart_count=design.restart_count,
            max_sweeps=design.max_sweeps,
            max_pairwise_rounds=design.max_pairwise_rounds,
            max_events=budget.max_events,
            max_objective_evaluations=budget.max_objective_evaluations_per_fit,
            max_scored_event_work=budget.max_scored_event_work_per_fit,
        )
    audits = tuple(
        _make_direct_audit(corpus, models[penalty], train, validation)
        for penalty in design.residual_penalties
    )
    fold_certificates = _derive_fold_certificates(corpus, selection)
    aggregates = {row.residual_penalty: row for row in selection.aggregate_scores}
    selected_margin = (
        aggregates[16].all_validation_query_mistakes
        - aggregates[4].all_validation_query_mistakes
    )
    audit_map = {row.residual_penalty: row for row in audits}
    direct_margin = audit_map[16].validation_mistakes - audit_map[4].validation_mistakes
    crosslinks = tuple(
        fold.pseudoheldout_cell
        for fold in selection.folds
        if fold.pseudoheldout_cell[0] == 2
        and fold.candidates[0].all_validation_query_mistakes
        < fold.candidates[1].all_validation_query_mistakes
    )
    self_cells = tuple(
        (design.exception.destination_key, value)
        for value in range(design.value_cardinality)
    )
    fields: dict[str, object] = {
        "schema": _SCHEMA_RESULT,
        "corpus": corpus,
        "selection": selection,
        "fold_optimum_certificates": fold_certificates,
        "direct_penalty_audits": audits,
        "selected_sequence_validation_margin": selected_margin,
        "direct_full_validation_margin": direct_margin,
        "crosslink_winning_cells": crosslinks,
        "self_pseudoheldout_cells": self_cells,
        "outer_results_used_for_tuning": False,
        "unseen_exception_prediction_claimed": False,
        "representation_discovery_performed": False,
        "supplied_register_interface": True,
    }
    temporary = object.__new__(PowerConditionResult)
    for name, value in fields.items():
        object.__setattr__(temporary, name, value)
    object.__setattr__(temporary, "result_sha256", "0" * 64)
    return PowerConditionResult(
        **fields,
        result_sha256=_sha256(_condition_result_payload(temporary)),
    )


@dataclass(frozen=True)
class PairLocalExceptionPowerReport:
    """Paired positive/null result with a closed, non-confirmatory scope."""

    schema: str
    design: PowerControlDesign
    budget: PowerControlBudget
    positive: PowerConditionResult
    negative: PowerConditionResult
    matched_visible_programs: bool
    balanced_output_classes: bool
    observed_exception_seen_in_train_and_validation: bool
    self_pseudoheldout_nonidentifying_disclosed: bool
    outer_results_used_for_tuning: bool
    unseen_exception_prediction_claimed: bool
    representation_discovery_performed: bool
    confirmatory_claim_permitted: bool
    report_sha256: str

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA_REPORT:
            raise ValueError("unknown pair-local power-report schema")
        if type(self.design) is not PowerControlDesign:
            raise TypeError("design must be exact PowerControlDesign")
        if type(self.budget) is not PowerControlBudget:
            raise TypeError("budget must be exact PowerControlBudget")
        if type(self.positive) is not PowerConditionResult or type(
            self.negative
        ) is not PowerConditionResult:
            raise TypeError("positive and negative must be exact condition results")
        if self.positive.corpus.design != self.design or self.negative.corpus.design != self.design:
            raise ValueError("paired results do not share the report design")
        if self.positive.corpus.condition is not PowerControlCondition.OBSERVED_PAIR_LOCAL_EXCEPTION:
            raise ValueError("positive result has the wrong condition")
        if self.negative.corpus.condition is not PowerControlCondition.NO_EXCEPTION:
            raise ValueError("negative result has the wrong condition")
        expected_matched = (
            self.positive.corpus.program_manifest_sha256
            == self.negative.corpus.program_manifest_sha256
        )
        expected_balanced = all(
            len(set(counts)) == 1
            for counts in (
                self.positive.corpus.train_output_class_counts,
                self.positive.corpus.validation_output_class_counts,
                self.negative.corpus.train_output_class_counts,
                self.negative.corpus.validation_output_class_counts,
            )
        )
        if self.matched_visible_programs != expected_matched or not expected_matched:
            raise ValueError("positive/null visible programs must match exactly")
        if self.balanced_output_classes != expected_balanced or not expected_balanced:
            raise ValueError("every paired split must be exactly class-balanced")
        expected_seen = all(
            count > 0
            for count in (
                self.positive.corpus.train_exception_applied_count,
                self.positive.corpus.validation_exception_applied_count,
            )
        )
        if self.observed_exception_seen_in_train_and_validation != expected_seen or not expected_seen:
            raise ValueError("the positive exception must be observed in both direct splits")
        for name in (
            "matched_visible_programs",
            "balanced_output_classes",
            "observed_exception_seen_in_train_and_validation",
            "self_pseudoheldout_nonidentifying_disclosed",
            "outer_results_used_for_tuning",
            "unseen_exception_prediction_claimed",
            "representation_discovery_performed",
            "confirmatory_claim_permitted",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact boolean")
        if not self.self_pseudoheldout_nonidentifying_disclosed or any(
            (
                self.outer_results_used_for_tuning,
                self.unseen_exception_prediction_claimed,
                self.representation_discovery_performed,
                self.confirmatory_claim_permitted,
            )
        ):
            raise ValueError("report exceeds the observed-control claim boundary")
        _require_sha256("report_sha256", self.report_sha256)
        if self.report_sha256 != _sha256(_report_payload(self)):
            raise ValueError("report_sha256 does not bind the paired result")


def _report_payload(report: PairLocalExceptionPowerReport) -> dict[str, object]:
    return {
        "schema": report.schema,
        "design_sha256": report.design.design_sha256,
        "budget_sha256": report.budget.budget_sha256,
        "positive_result_sha256": report.positive.result_sha256,
        "negative_result_sha256": report.negative.result_sha256,
        "matched_visible_programs": report.matched_visible_programs,
        "balanced_output_classes": report.balanced_output_classes,
        "observed_exception_seen_in_train_and_validation": report.observed_exception_seen_in_train_and_validation,
        "self_pseudoheldout_nonidentifying_disclosed": report.self_pseudoheldout_nonidentifying_disclosed,
        "outer_results_used_for_tuning": report.outer_results_used_for_tuning,
        "unseen_exception_prediction_claimed": report.unseen_exception_prediction_claimed,
        "representation_discovery_performed": report.representation_discovery_performed,
        "confirmatory_claim_permitted": report.confirmatory_claim_permitted,
    }


def run_pair_local_exception_power_control(
    *,
    design: PowerControlDesign | None = None,
    budget: PowerControlBudget | None = None,
) -> PairLocalExceptionPowerReport:
    """Run the paired positive and no-exception controls under frozen settings."""

    design = default_power_control_design() if design is None else design
    budget = default_power_control_budget() if budget is None else budget
    if type(design) is not PowerControlDesign:
        raise TypeError("design must be exact PowerControlDesign")
    if type(budget) is not PowerControlBudget:
        raise TypeError("budget must be exact PowerControlBudget")
    _check_corpus_budget(budget)
    _check_run_budget(design, budget, condition_count=2)
    positive_corpus = build_power_control_corpus(
        PowerControlCondition.OBSERVED_PAIR_LOCAL_EXCEPTION,
        design=design,
        budget=budget,
    )
    negative_corpus = build_power_control_corpus(
        PowerControlCondition.NO_EXCEPTION,
        design=design,
        budget=budget,
    )
    positive = run_power_condition(positive_corpus, budget=budget)
    negative = run_power_condition(negative_corpus, budget=budget)
    fields: dict[str, object] = {
        "schema": _SCHEMA_REPORT,
        "design": design,
        "budget": budget,
        "positive": positive,
        "negative": negative,
        "matched_visible_programs": True,
        "balanced_output_classes": True,
        "observed_exception_seen_in_train_and_validation": True,
        "self_pseudoheldout_nonidentifying_disclosed": True,
        "outer_results_used_for_tuning": False,
        "unseen_exception_prediction_claimed": False,
        "representation_discovery_performed": False,
        "confirmatory_claim_permitted": False,
    }
    temporary = object.__new__(PairLocalExceptionPowerReport)
    for name, value in fields.items():
        object.__setattr__(temporary, name, value)
    object.__setattr__(temporary, "report_sha256", "0" * 64)
    return PairLocalExceptionPowerReport(
        **fields,
        report_sha256=_sha256(_report_payload(temporary)),
    )


__all__ = [
    "PairLocalExceptionPowerReport",
    "PairLocalExceptionSpec",
    "PowerConditionResult",
    "PowerControlBudget",
    "PowerControlCondition",
    "PowerControlCorpus",
    "PowerControlDesign",
    "PowerControlLimitError",
    "PowerControlScope",
    "PowerFoldOptimumCertificate",
    "PowerPenaltyAudit",
    "PowerTraceManifestRow",
    "PowerTraceRole",
    "build_power_control_corpus",
    "default_power_control_budget",
    "default_power_control_design",
    "run_pair_local_exception_power_control",
    "run_power_condition",
]
