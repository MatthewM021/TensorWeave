"""Run the frozen Phase-II two-phase outer-omission campaign.

This is prospective execution robustness inside known binding semantics, not
secret-law discovery, representation discovery, or a confirmatory experiment.
All forty trace-supervised fit/selection shards must be written and committed
by a terminal preopen aggregate before any outer probe is constructed.  Batch
opening is permitted only when all 1,520 fold candidates and all forty final
fits attain zero training mistakes and zero local residuals.

Repeated numeric seed labels balance their assignment to omitted cells.  They
are not common-random-number matches: the heldout-dependent task fingerprint
changes the generator's derived seed, so the repetitions are independent
environment draws and do not identify a shared RNG-seed effect.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import types
from typing import (
    Any,
    Mapping,
    NamedTuple,
    Sequence,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import torch
import yaml

from tnlm_v3.algebra_discovery import (
    OuterRotationResult,
    SequenceAlgebraSelectionResult,
    SequenceDiscoveryLimitError,
    make_sequence_corpus,
    prototype_inventory,
    run_outer_rotation,
)
from tnlm_v3.algebra_discovery_probes import (
    build_balanced_probe_suite,
    cyclic_cell_rotation_inventory,
    evaluate_probe_suite,
    evaluate_shortcut_controls,
)
from tnlm_v3.campaign_config import load_milestone4_campaign_config
from tnlm_v3.data import apply_value_transform


PROTOCOL_SCHEMA = "tnlm-v3-phase2-outer-rotation-protocol-v3"
PROTOCOL_SHA256 = "8689ab0ad06a1268c96690da27a6f8573b1536ca19c3bc18c403fdc07ffe2649"
PROTOCOL_RELATIVE_PATH = "v3/configs/phase2/outer_rotation_v3.json"
SUPERSEDED_PROTOCOL_SHA256 = (
    "9aef5525f3abf67a9c111b412ad1a5443817ebb1b1afd83a1a0050c0fb6092ce"
)
PREOPEN_ENVIRONMENT_SCHEMA = "tnlm-v3-phase2-trace-algebra-preopen-environment-v3"
PREOPEN_AGGREGATE_SCHEMA = "tnlm-v3-phase2-trace-algebra-terminal-preopen-v3"
OPEN_ENVIRONMENT_SCHEMA = "tnlm-v3-phase2-trace-algebra-open-environment-v3"
CAMPAIGN_SCHEMA = "tnlm-v3-phase2-trace-algebra-open-campaign-v3"
TEST_FIXTURE_ENVIRONMENT_SCHEMA = (
    "tnlm-v3-phase2-trace-algebra-synthetic-preopen-test-fixture-v1"
)
TEST_FIXTURE_CAMPAIGN_SCHEMA = (
    "tnlm-v3-phase2-trace-algebra-synthetic-terminal-test-fixture-v1"
)

DEFAULT_MAX_TOTAL_PRIMARY_GENERATED_EVENTS = 122_880
# Compatibility name: this cap is primary fitting generation only.
DEFAULT_MAX_TOTAL_GENERATED_EVENTS = DEFAULT_MAX_TOTAL_PRIMARY_GENERATED_EVENTS
DEFAULT_MAX_TOTAL_VALIDATION_REPLAY_EVENTS = 491_520
DEFAULT_MAX_TOTAL_SCORED_EVENT_WORK = 19_200_000_000
DEFAULT_MAX_ENVIRONMENT_RECORD_BYTES = 32_000_000
EXPECTED_ACTUAL_CASE_COUNT = 15
EXPECTED_ACTUAL_QUERY_COUNT = 96
EXPECTED_ACTUAL_FOCAL_QUERY_COUNT = 24
EXPECTED_PATH_RELATION_COUNT = 3
EXPECTED_ROTATED_CASE_COUNT = 300
EXPECTED_ROTATED_QUERY_COUNT = 1_800
EXPECTED_ROTATED_FOCAL_QUERY_COUNT = 480
MIN_PSEUDO_DEPENDENT_QUERIES_PER_FOLD = 16


class EnvironmentSpec(NamedTuple):
    environment_index: int
    block_id: str
    outer_cell_index: int
    outer_cell: tuple[int, int]
    seed_pair_index: int
    train_seed: int
    validation_seed: int
    optimizer_seed: int


class FrozenProtocol(NamedTuple):
    protocol_id: str
    protocol_sha256: str
    base_task_relative_path: str
    base_task_sha256: str
    num_surface_keys: int
    value_cardinality: int
    max_live_bindings: int
    documents_per_split: int
    document_length: int
    residual_penalties: tuple[int, ...]
    restart_count: int
    max_sweeps: int
    max_pairwise_rounds: int
    inner_folds_per_environment: int
    minimum_pseudo_dependent_queries_per_fold: int
    required_passing_environments: int
    required_admissible_fold_candidates: int
    exact_supported_transition_entries: int
    exact_probe_families: int
    implementation_manifest_relative_path: str
    implementation_required_paths: tuple[str, ...]
    power_control_relative_path: str
    expected_python_version: str
    expected_torch_version: str
    expected_pyyaml_version: str
    expected_device: str
    planned_objective_evaluations_per_fit: int
    fit_calls_per_environment: int
    max_primary_generated_events_total: int
    max_deterministic_replay_generated_events_total: int
    max_all_generation_work_total: int
    max_scored_event_work_per_fit: int
    conservative_scored_event_work_per_environment: int
    max_scored_event_work_per_environment: int
    conservative_scored_event_work_total: int
    max_scored_event_work_total: int
    schedule: tuple[EnvironmentSpec, ...]
    schedule_sha256: str


class CampaignPreflight(NamedTuple):
    environment_indices: tuple[int, ...]
    environment_count: int
    primary_generated_event_count: int
    prototype_count: int
    planned_objective_evaluations_per_fit: int
    fit_calls_per_environment: int
    fold_candidate_count: int
    scored_event_work_per_environment: int
    total_scored_event_work: int

    @property
    def generated_event_count(self) -> int:
        return self.primary_generated_event_count


class ImplementationManifest(NamedTuple):
    relative_path: str
    manifest_sha256: str
    file_sha256s: tuple[tuple[str, str], ...]


class RuntimeRecord(NamedTuple):
    python: str
    torch: str
    pyyaml: str
    device: str
    platform: str
    machine: str


class PowerControlCommitment(NamedTuple):
    relative_path: str
    file_sha256: str
    record_sha256: str


class _ValidatedPreopen(NamedTuple):
    index: int
    record: Mapping[str, object]
    body: Mapping[str, object]
    selection: SequenceAlgebraSelectionResult
    corpus_sha256: str


def _v3_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repository_root() -> Path:
    return _v3_root().parent


def default_protocol_path() -> Path:
    return _v3_root() / "configs" / "phase2" / "outer_rotation_v3.json"


def default_implementation_manifest_path() -> Path:
    return _repository_root() / "v3_recovery" / (
        "PHASE2_OUTER_ROTATION_V3_IMPLEMENTATION.sha256"
    )


def default_power_control_record_path() -> Path:
    return _repository_root() / "v3_recovery" / (
        "PHASE2_ALGEBRA_POWER_CONTROL_V1.json"
    )


def _plain_int(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{name} keys must be exact strings")
    return value


def _exact_list(name: str, value: object) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an exact list")
    return value


def _require_sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
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
            field.name: _jsonable(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError(f"unsupported JSON evidence type: {type(value).__name__}")


def _decode_typed(expected: object, raw: object, path: str) -> object:
    """Reconstruct exact frozen scientific dataclasses from canonical JSON."""

    if expected is Any or expected is object:
        return raw
    origin = get_origin(expected)
    arguments = get_args(expected)
    if origin is tuple:
        if type(raw) is not list:
            raise TypeError(f"{path} must be a JSON list encoding a tuple")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                _decode_typed(arguments[0], item, f"{path}[{index}]")
                for index, item in enumerate(raw)
            )
        if len(raw) != len(arguments):
            raise ValueError(f"{path} tuple arity changed")
        return tuple(
            _decode_typed(item_type, item, f"{path}[{index}]")
            for index, (item_type, item) in enumerate(
                zip(arguments, raw, strict=True)
            )
        )
    if origin is list:
        if type(raw) is not list:
            raise TypeError(f"{path} must be an exact list")
        item_type = arguments[0] if arguments else Any
        return [
            _decode_typed(item_type, item, f"{path}[{index}]")
            for index, item in enumerate(raw)
        ]
    if origin in (dict, Mapping):
        mapping = _mapping(path, raw)
        key_type, value_type = arguments if arguments else (Any, Any)
        return {
            _decode_typed(key_type, key, f"{path}.key"): _decode_typed(
                value_type, value, f"{path}[{key!r}]"
            )
            for key, value in mapping.items()
        }
    if origin in (Union, types.UnionType):
        if raw is None and type(None) in arguments:
            return None
        failures: list[Exception] = []
        for option in arguments:
            if option is type(None):
                continue
            try:
                return _decode_typed(option, raw, path)
            except (TypeError, ValueError) as error:
                failures.append(error)
        raise TypeError(f"{path} does not match any declared union member") from (
            failures[-1] if failures else None
        )
    if isinstance(expected, type) and issubclass(expected, Enum):
        try:
            return expected(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path} is not a valid {expected.__name__}") from error
    if isinstance(expected, type) and is_dataclass(expected):
        mapping = _mapping(path, raw)
        expected_names = tuple(field.name for field in fields(expected))
        if set(mapping) != set(expected_names):
            missing = sorted(set(expected_names) - set(mapping))
            extra = sorted(set(mapping) - set(expected_names))
            raise ValueError(
                f"{path} dataclass fields changed: missing={missing}, extra={extra}"
            )
        hints = get_type_hints(expected)
        values = {
            name: _decode_typed(hints[name], mapping[name], f"{path}.{name}")
            for name in expected_names
        }
        return expected(**values)
    if expected is type(None):
        if raw is not None:
            raise TypeError(f"{path} must be null")
        return None
    if expected in (bool, int, float, str):
        if type(raw) is not expected:
            raise TypeError(f"{path} must be exact {expected.__name__}")
        return raw
    raise TypeError(f"{path} uses unsupported annotation {expected!r}")


def _decode_dataclass(expected: type, raw: object, path: str):
    value = _decode_typed(expected, raw, path)
    if type(value) is not expected:
        raise TypeError(f"{path} did not reconstruct exact {expected.__name__}")
    return value


def _environment_spec_payload(spec: EnvironmentSpec) -> dict[str, object]:
    return {
        "environment_index": spec.environment_index,
        "block_id": spec.block_id,
        "outer_cell_index": spec.outer_cell_index,
        "outer_cell": list(spec.outer_cell),
        "seed_pair_index": spec.seed_pair_index,
        "train_seed": spec.train_seed,
        "validation_seed": spec.validation_seed,
        "optimizer_seed": spec.optimizer_seed,
    }


def _parse_shift_formula(formula: object, cell_count: int) -> int:
    if formula == "outer_cell_index":
        return 0
    match = re.fullmatch(r"outer_cell_index_plus_(\d+)_mod_(\d+)", str(formula))
    if match is None or int(match.group(2)) != cell_count:
        raise ValueError("unknown crossed-block seed formula")
    return int(match.group(1))


def _incidence_connected(schedule: tuple[EnvironmentSpec, ...], count: int) -> bool:
    graph: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    for row in schedule:
        cell = ("cell", row.outer_cell_index)
        seed = ("seed", row.seed_pair_index)
        graph[cell].add(seed)
        graph[seed].add(cell)
    reached = {next(iter(graph))}
    queue = deque(reached)
    while queue:
        node = queue.popleft()
        for neighbour in graph[node]:
            if neighbour not in reached:
                reached.add(neighbour)
                queue.append(neighbour)
    return len(reached) == 2 * count


def load_frozen_protocol(path: Path | None = None) -> FrozenProtocol:
    """Load and validate the exact frozen v3 preregistration."""

    protocol_path = default_protocol_path() if path is None else path
    if not isinstance(protocol_path, Path):
        raise TypeError("protocol path must be pathlib.Path")
    payload = protocol_path.read_bytes()
    digest = _sha256_bytes(payload)
    if digest == SUPERSEDED_PROTOCOL_SHA256:
        raise ValueError("the v2 protocol was superseded before new environment fitting")
    if digest != PROTOCOL_SHA256:
        raise ValueError("protocol bytes do not match the frozen v3 SHA-256")
    document = _mapping("protocol", json.loads(payload))
    if document.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unknown protocol schema")
    if document.get("frozen_before_new_environment_fitting_or_outer_probe_evaluation") is not True:
        raise ValueError("protocol was not frozen before execution")
    if document.get("claim_eligible") is not False:
        raise ValueError("this protocol cannot be claim eligible")

    base = _mapping("base task", document.get("base_task_config"))
    base_path = base.get("path")
    base_sha = _require_sha256("base task SHA-256", base.get("sha256"))
    if type(base_path) is not str or not base_path:
        raise TypeError("base task path must be a nonempty exact string")
    resolved_base = (_repository_root() / base_path).resolve()
    try:
        resolved_base.relative_to(_repository_root().resolve())
    except ValueError as error:
        raise ValueError("base task path escapes the repository") from error
    if _sha256_bytes(resolved_base.read_bytes()) != base_sha:
        raise ValueError("base task bytes changed")

    task = _mapping("task", document.get("task"))
    keys = _plain_int("num_surface_keys", task.get("num_surface_keys"), 2)
    values = _plain_int("value_cardinality", task.get("value_cardinality"), 2)
    max_live = _plain_int("max_live_bindings", task.get("max_live_bindings"), 1)
    cell_count = keys * values
    if task.get("cell_index") != f"{values}_times_key_plus_value":
        raise ValueError("cell-index formula changed")
    if task.get("supported_learned_transition_entries") != len(
        prototype_inventory(values)
    ):
        raise ValueError("supported transition inventory changed")

    data = _mapping("data", document.get("data"))
    documents = _plain_int("documents_per_split", data.get("documents_per_split"), 1)
    length = _plain_int("document_length", data.get("document_length"), 1)
    if data.get("generator_splits") != ["train", "validation"]:
        raise ValueError("direct split inventory changed")
    if data.get("primary_generated_events_per_environment") != 2 * documents * length:
        raise ValueError("primary generation count changed")
    if data.get("outer_labels_used_for_fit_or_selection") is not False:
        raise ValueError("outer labels cannot be used for fit or selection")

    selector = _mapping("selector", document.get("selector"))
    penalties = tuple(
        _plain_int("residual penalty", item)
        for item in _exact_list("residual penalties", selector.get("residual_penalties"))
    )
    restarts = _plain_int("restart_count", selector.get("restart_count"), 1)
    sweeps = _plain_int("max_sweeps", selector.get("max_sweeps"), 1)
    pairwise = _plain_int("max_pairwise_rounds", selector.get("max_pairwise_rounds"))
    folds = _plain_int("inner folds", selector.get("inner_folds_per_environment"), 1)
    minimum_pseudo = _plain_int(
        "minimum pseudo queries",
        selector.get("minimum_pseudo_dependent_queries_per_fold"),
        1,
    )
    if penalties != (4, 16) or folds != cell_count - 1 or minimum_pseudo != 16:
        raise ValueError("selector grid, fold count, or pseudo-query gate changed")
    if selector.get("fold_optimizer_seed_formula") != (
        "environment_optimizer_seed_plus_4_times_pseudo_key_plus_pseudo_value"
    ):
        raise ValueError("fold seed must use the absolute pseudo-cell index")
    if selector.get("fold_candidate_seen_fit_gate") != (
        "zero_training_mistakes_and_zero_local_overrides"
    ):
        raise ValueError("fold-candidate cleanliness gate changed")

    raw_blocks = _exact_list("crossed blocks", document.get("crossed_blocks"))
    schedule_rows: list[EnvironmentSpec] = []
    for block_number, raw in enumerate(raw_blocks):
        block = _mapping("crossed block", raw)
        block_id = block.get("block_id")
        if type(block_id) is not str or not block_id:
            raise TypeError("block ID must be a nonempty string")
        offset = _plain_int("block offset", block.get("environment_index_offset"))
        if offset != block_number * cell_count:
            raise ValueError("crossed blocks must be contiguous")
        shift = _parse_shift_formula(block.get("seed_pair_index_formula"), cell_count)
        for cell_index in range(cell_count):
            seed_index = (cell_index + shift) % cell_count
            schedule_rows.append(
                EnvironmentSpec(
                    environment_index=offset + cell_index,
                    block_id=block_id,
                    outer_cell_index=cell_index,
                    outer_cell=(cell_index // values, cell_index % values),
                    seed_pair_index=seed_index,
                    train_seed=10_000 + seed_index,
                    validation_seed=20_000 + seed_index,
                    optimizer_seed=30_000 + seed_index,
                )
            )
    schedule = tuple(schedule_rows)
    universe = {(key, value) for key in range(keys) for value in range(values)}
    if len(schedule) != 40 or tuple(row.environment_index for row in schedule) != tuple(range(40)):
        raise ValueError("v3 schedule must contain forty canonical environments")
    if len({(row.outer_cell, row.seed_pair_index) for row in schedule}) != len(schedule):
        raise ValueError("crossed schedule repeats a cell/seed-label assignment")
    if any(
        {row.outer_cell for row in schedule[start : start + cell_count]} != universe
        for start in range(0, len(schedule), cell_count)
    ):
        raise ValueError("every block must cover every cell exactly once")
    expanded = _mapping("expanded design", document.get("expanded_design_invariants"))
    if (
        expanded.get("cell_seed_label_incidence_graph_connected")
        != _incidence_connected(schedule, cell_count)
        or expanded.get("data_realizations_are_independent_not_matched") is not True
        or expanded.get("gold_standard_full_cell_by_seed_cross_not_claimed") is not True
    ):
        raise ValueError("expanded-design claims changed")

    phases = _mapping("execution phases", document.get("execution_phases"))
    phase1 = _mapping("preopen phase", phases.get("phase_1_preopen"))
    phase2 = _mapping(
        "terminal preopen phase", phases.get("phase_2_terminal_preopen_aggregate")
    )
    phase3 = _mapping("batch open phase", phases.get("phase_3_batch_open"))
    if set(_exact_list("preopen forbidden actions", phase1.get("forbidden"))) != {
        "construct_outer_probe_suite",
        "read_outer_probe_answers",
        "evaluate_outer_or_rotated_probes",
    }:
        raise ValueError("preopen probe firewall changed")
    if (
        phase2.get("required_environment_shards") != len(schedule)
        or phase2.get("required_admissible_fold_candidates")
        != len(schedule) * folds * len(penalties)
        or phase2.get("content_binds_every_preopen_shard") is not True
        or phase3.get("permitted_only_after_terminal_preopen_aggregate") is not True
        or phase3.get("evaluates_all_40_frozen_models") is not True
    ):
        raise ValueError("two-phase execution protocol changed")

    implementation = _mapping(
        "implementation commitment", document.get("implementation_commitment")
    )
    manifest_path = implementation.get("manifest_path")
    fixed_paths = _exact_list("implementation paths", implementation.get("required_paths"))
    source_globs = _exact_list(
        "implementation source globs", implementation.get("required_source_globs")
    )
    if type(manifest_path) is not str or not manifest_path:
        raise TypeError("implementation manifest path must be nonempty")
    if source_globs != ["v3/src/tnlm_v3/*.py"]:
        raise ValueError("transitive implementation source glob changed")
    if (
        implementation.get("required_before_any_preopen_environment_fit") is not True
        or implementation.get("manifest_must_include_every_matching_source_file")
        is not True
    ):
        raise ValueError("implementation source closure is not mandatory")
    expanded_sources = {
        path.relative_to(_repository_root()).as_posix()
        for path in (_v3_root() / "src" / "tnlm_v3").glob("*.py")
        if path.is_file()
    }
    if not expanded_sources:
        raise ValueError("transitive implementation source closure is empty")
    if any(type(item) is not str or not item for item in fixed_paths):
        raise TypeError("fixed implementation paths must be nonempty strings")
    implementation_paths = tuple(sorted(set(fixed_paths).union(expanded_sources)))

    runtime = _mapping("runtime reproducibility", document.get("runtime_reproducibility"))
    expected_runtime = _mapping(
        "expected runtime", runtime.get("expected_frozen_environment")
    )
    expected_runtime_values = {
        "python": "3.12.13",
        "torch": "2.13.0+cpu",
        "pyyaml": "6.0.3",
        "device": "cpu",
    }
    if dict(expected_runtime) != expected_runtime_values:
        raise ValueError("frozen runtime versions changed")

    prerequisite = _mapping(
        "prerequisite controls", document.get("prerequisite_controls")
    )
    power_path = prerequisite.get("record_path")
    if type(power_path) is not str or not power_path:
        raise TypeError("power-control record path must be nonempty")
    if (
        prerequisite.get("observed_transition_address_exception_power_control_required")
        is not True
        or prerequisite.get("record_must_validate_before_any_preopen_fit") is not True
        or prerequisite.get("record_sha256_bound_by_every_preopen_and_campaign_artifact")
        is not True
    ):
        raise ValueError("power-control prerequisite changed")

    budget = _mapping("work budget", document.get("work_budget"))
    planned_evaluations = _plain_int(
        "planned objective evaluations",
        budget.get("planned_objective_evaluations_per_fit"),
        1,
    )
    fit_calls = _plain_int("fit calls", budget.get("fit_calls_per_environment"), 1)
    max_per_fit = _plain_int(
        "max scored work per fit", budget.get("max_scored_event_work_per_fit"), 1
    )
    conservative_environment = _plain_int(
        "conservative environment work",
        budget.get("conservative_scored_event_work_per_environment"),
        1,
    )
    max_environment = _plain_int(
        "max environment work", budget.get("max_scored_event_work_per_environment"), 1
    )
    conservative_total = _plain_int(
        "conservative total work", budget.get("conservative_scored_event_work_total"), 1
    )
    max_total = _plain_int(
        "max total work", budget.get("max_scored_event_work_total"), 1
    )
    max_primary = _plain_int(
        "max primary generation", budget.get("max_primary_generated_events_total"), 1
    )
    max_replay = _plain_int(
        "max replay generation",
        budget.get("max_deterministic_replay_generated_events_total"),
        1,
    )
    max_all = _plain_int(
        "max all generation", budget.get("max_all_generation_work_total"), 1
    )
    prototype_count = len(prototype_inventory(values))
    expected_evaluations = restarts * (
        1
        + sweeps * prototype_count * values * (keys + 1)
        + pairwise * prototype_count * (prototype_count - 1) // 2 * values**2
    )
    expected_fit_calls = folds * len(penalties) + 1
    train_events = documents * length
    expected_environment_work = (
        expected_fit_calls * train_events * expected_evaluations
        + folds * len(penalties) * 2 * train_events
    )
    if (
        planned_evaluations != expected_evaluations
        or fit_calls != expected_fit_calls
        or conservative_environment != expected_environment_work
        or conservative_total != len(schedule) * expected_environment_work
        or max_primary != len(schedule) * 2 * train_events
        or max_all != max_primary + max_replay
        or max_per_fit < train_events * expected_evaluations
        or max_environment < conservative_environment
        or max_total < conservative_total
        or budget.get("replay_generation_is_validation_only_and_never_refit_or_reselection")
        is not True
    ):
        raise ValueError("frozen work budget is inconsistent")

    preopen = _mapping("preopen acceptance", document.get("preopen_acceptance"))
    postopen = _mapping("postopen acceptance", document.get("postopen_acceptance"))
    required_passing = _plain_int(
        "required passing environments", preopen.get("required_passing_environments"), 1
    )
    required_candidates = _plain_int(
        "required candidate count", preopen.get("required_admissible_fold_candidates"), 1
    )
    exact_entries = _plain_int(
        "exact transition entries",
        postopen.get("exact_supported_transition_entries_per_environment"),
        1,
    )
    family_rows = _exact_list("exact probe families", postopen.get("exact_probe_families"))
    if (
        required_passing != len(schedule)
        or required_candidates != len(schedule) * folds * len(penalties)
        or preopen.get("minimum_pseudo_dependent_queries_per_fold") != minimum_pseudo
        or postopen.get("actual_cell_cases") != [15, 15]
        or postopen.get("actual_cell_queries") != [96, 96]
        or postopen.get("actual_cell_focal_queries") != [24, 24]
        or family_rows != [12, 12]
        or postopen.get("exact_path_relations") != [3, 3]
        or postopen.get("rotated_control_cases") != [300, 300]
        or postopen.get("rotated_control_queries") != [1800, 1800]
        or postopen.get("rotated_control_focal_queries") != [480, 480]
        or postopen.get("all_shortcut_controls_strictly_below_learned_model")
        is not True
        or postopen.get("no_pooled_average_rescue") is not True
    ):
        raise ValueError("frozen acceptance inventory changed")

    return FrozenProtocol(
        protocol_id=str(document["protocol_id"]),
        protocol_sha256=digest,
        base_task_relative_path=base_path,
        base_task_sha256=base_sha,
        num_surface_keys=keys,
        value_cardinality=values,
        max_live_bindings=max_live,
        documents_per_split=documents,
        document_length=length,
        residual_penalties=penalties,
        restart_count=restarts,
        max_sweeps=sweeps,
        max_pairwise_rounds=pairwise,
        inner_folds_per_environment=folds,
        minimum_pseudo_dependent_queries_per_fold=minimum_pseudo,
        required_passing_environments=required_passing,
        required_admissible_fold_candidates=required_candidates,
        exact_supported_transition_entries=exact_entries,
        exact_probe_families=family_rows[0],
        implementation_manifest_relative_path=manifest_path,
        implementation_required_paths=implementation_paths,
        power_control_relative_path=power_path,
        expected_python_version=expected_runtime_values["python"],
        expected_torch_version=expected_runtime_values["torch"],
        expected_pyyaml_version=expected_runtime_values["pyyaml"],
        expected_device=expected_runtime_values["device"],
        planned_objective_evaluations_per_fit=planned_evaluations,
        fit_calls_per_environment=fit_calls,
        max_primary_generated_events_total=max_primary,
        max_deterministic_replay_generated_events_total=max_replay,
        max_all_generation_work_total=max_all,
        max_scored_event_work_per_fit=max_per_fit,
        conservative_scored_event_work_per_environment=conservative_environment,
        max_scored_event_work_per_environment=max_environment,
        conservative_scored_event_work_total=conservative_total,
        max_scored_event_work_total=max_total,
        schedule=schedule,
        schedule_sha256=_sha256_bytes(
            _canonical_bytes([_environment_spec_payload(row) for row in schedule])
        ),
    )


def _resolve_environment_indices(
    protocol: FrozenProtocol, environment_indices: Sequence[int] | None
) -> tuple[int, ...]:
    if environment_indices is None:
        return tuple(range(len(protocol.schedule)))
    if not isinstance(environment_indices, Sequence) or isinstance(
        environment_indices, (str, bytes)
    ):
        raise TypeError("environment_indices must be a finite sequence")
    rows = tuple(environment_indices)
    if not rows or any(type(value) is not int for value in rows):
        raise TypeError("environment indices must be nonempty exact integers")
    if len(rows) != len(set(rows)):
        raise ValueError("environment indices must be unique")
    if any(value < 0 or value >= len(protocol.schedule) for value in rows):
        raise ValueError("environment index is outside the frozen schedule")
    return tuple(sorted(rows))


def preflight_campaign(
    protocol_path: Path | None = None,
    *,
    environment_indices: Sequence[int] | None = None,
    max_total_primary_generated_events: int = DEFAULT_MAX_TOTAL_PRIMARY_GENERATED_EVENTS,
    max_total_scored_event_work: int = DEFAULT_MAX_TOTAL_SCORED_EVENT_WORK,
) -> tuple[FrozenProtocol, CampaignPreflight]:
    """Fail before generation/fitting when any analytic budget is insufficient."""

    protocol = load_frozen_protocol(protocol_path)
    indices = _resolve_environment_indices(protocol, environment_indices)
    generated_cap = _plain_int(
        "max_total_primary_generated_events", max_total_primary_generated_events, 1
    )
    scored_cap = _plain_int(
        "max_total_scored_event_work", max_total_scored_event_work, 1
    )
    primary_per_environment = 2 * protocol.documents_per_split * protocol.document_length
    primary_total = len(indices) * primary_per_environment
    work_total = len(indices) * protocol.conservative_scored_event_work_per_environment
    if primary_total > generated_cap:
        raise SequenceDiscoveryLimitError(
            "campaign exceeds the primary generated-event budget before generation"
        )
    if work_total > scored_cap:
        raise SequenceDiscoveryLimitError(
            "campaign exceeds the scored-event budget before generation or fitting"
        )
    base_path = (_repository_root() / protocol.base_task_relative_path).resolve()
    task = load_milestone4_campaign_config(base_path).task
    if (
        task.num_surface_keys != protocol.num_surface_keys
        or task.value_cardinality != protocol.value_cardinality
        or task.max_live_bindings != protocol.max_live_bindings
        or not task.min_length <= protocol.document_length <= task.max_length
    ):
        raise ValueError("base task disagrees with the frozen protocol")
    return protocol, CampaignPreflight(
        environment_indices=indices,
        environment_count=len(indices),
        primary_generated_event_count=primary_total,
        prototype_count=len(prototype_inventory(protocol.value_cardinality)),
        planned_objective_evaluations_per_fit=(
            protocol.planned_objective_evaluations_per_fit
        ),
        fit_calls_per_environment=protocol.fit_calls_per_environment,
        fold_candidate_count=(
            len(indices)
            * protocol.inner_folds_per_environment
            * len(protocol.residual_penalties)
        ),
        scored_event_work_per_environment=(
            protocol.conservative_scored_event_work_per_environment
        ),
        total_scored_event_work=work_total,
    )


def load_implementation_manifest(
    protocol: FrozenProtocol,
    path: Path | None = None,
) -> ImplementationManifest:
    """Validate the exact transitive implementation closure before any fit."""

    if type(protocol) is not FrozenProtocol:
        raise TypeError("protocol must be exact FrozenProtocol")
    manifest_path = default_implementation_manifest_path() if path is None else path
    if not isinstance(manifest_path, Path):
        raise TypeError("implementation manifest path must be pathlib.Path")
    expected_path = (_repository_root() / protocol.implementation_manifest_relative_path).resolve()
    if manifest_path.resolve() != expected_path:
        raise ValueError("implementation manifest must use the preregistered path")
    payload = manifest_path.read_bytes()
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\s].*)", line)
        if match is None:
            raise ValueError(f"invalid implementation manifest line {line_number}")
        digest, relative = match.groups()
        if "\\" in relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("implementation manifest paths must be canonical")
        rows.append((relative, digest))
    if tuple(relative for relative, _ in rows) != protocol.implementation_required_paths:
        raise ValueError("implementation manifest path inventory or order changed")
    repository = _repository_root().resolve()
    for relative, expected_digest in rows:
        candidate = (repository / relative).resolve()
        try:
            candidate.relative_to(repository)
        except ValueError as error:
            raise ValueError("implementation path escapes the repository") from error
        if _sha256_bytes(candidate.read_bytes()) != expected_digest:
            raise ValueError(f"implementation file digest changed: {relative}")
    return ImplementationManifest(
        relative_path=protocol.implementation_manifest_relative_path,
        manifest_sha256=_sha256_bytes(payload),
        file_sha256s=tuple(rows),
    )


def validate_runtime(protocol: FrozenProtocol) -> RuntimeRecord:
    """Fail before fitting unless the exact preregistered runtime is active."""

    if type(protocol) is not FrozenProtocol:
        raise TypeError("protocol must be exact FrozenProtocol")
    record = RuntimeRecord(
        python=platform.python_version(),
        torch=str(torch.__version__),
        pyyaml=str(yaml.__version__),
        device="cpu",
        platform=platform.platform(),
        machine=platform.machine(),
    )
    if record[:4] != (
        protocol.expected_python_version,
        protocol.expected_torch_version,
        protocol.expected_pyyaml_version,
        protocol.expected_device,
    ):
        raise ValueError("active runtime differs from the exact frozen environment")
    return record


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_single_runner():
    return _load_script(
        "phase2_trace_algebra_single",
        _v3_root() / "scripts" / "run_phase2_trace_algebra_experiment.py",
    )


def _load_power_runner():
    return _load_script(
        "phase2_algebra_power_control",
        _v3_root() / "scripts" / "run_phase2_algebra_power_control.py",
    )


def load_power_control_commitment(
    protocol: FrozenProtocol,
    path: Path | None = None,
) -> PowerControlCommitment:
    """Validate the observed-exception positive/null control before any fit."""

    if type(protocol) is not FrozenProtocol:
        raise TypeError("protocol must be exact FrozenProtocol")
    record_path = default_power_control_record_path() if path is None else path
    if not isinstance(record_path, Path):
        raise TypeError("power-control path must be pathlib.Path")
    expected_path = (_repository_root() / protocol.power_control_relative_path).resolve()
    if record_path.resolve() != expected_path:
        raise ValueError("power-control record must use the preregistered path")
    payload = record_path.read_bytes()
    document = _load_power_runner().validate_evidence_record(json.loads(payload))
    acceptance = _mapping("power acceptance", document.get("acceptance"))
    positive = _mapping("positive power condition", acceptance.get("positive"))
    negative = _mapping("null power condition", acceptance.get("negative"))
    positive_low = _mapping("positive penalty 4", positive.get("penalty_4"))
    negative_high = _mapping("null penalty 16", negative.get("penalty_16"))
    if (
        positive.get("selected_residual_penalty") != 4
        or positive_low.get("local_override_count") != 1
        or positive_low.get("expected_exception_override_realized") is not True
        or negative.get("selected_residual_penalty") != 16
        or negative.get("primary_score_tied") is not True
        or negative_high.get("local_override_count") != 0
        or negative_high.get("training_mistakes") != 0
        or negative_high.get("validation_mistakes") != 0
    ):
        raise ValueError("power-control prerequisite failed its frozen gates")
    return PowerControlCommitment(
        relative_path=protocol.power_control_relative_path,
        file_sha256=_sha256_bytes(payload),
        record_sha256=_require_sha256(
            "power-control record SHA-256", document.get("record_sha256")
        ),
    )


def _protocol_record(protocol: FrozenProtocol) -> dict[str, object]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": protocol.protocol_id,
        "protocol_file": PROTOCOL_RELATIVE_PATH,
        "protocol_file_sha256": protocol.protocol_sha256,
        "schedule_sha256": protocol.schedule_sha256,
        "superseded_v2_sha256": SUPERSEDED_PROTOCOL_SHA256,
        "max_pairwise_rounds": protocol.max_pairwise_rounds,
    }


def _implementation_record(manifest: ImplementationManifest) -> dict[str, object]:
    return {
        "manifest_path": manifest.relative_path,
        "manifest_sha256": manifest.manifest_sha256,
        "file_sha256s": [
            {"path": path, "sha256": digest}
            for path, digest in manifest.file_sha256s
        ],
    }


def _runtime_record(runtime: RuntimeRecord) -> dict[str, object]:
    return {
        "python": runtime.python,
        "torch": runtime.torch,
        "pyyaml": runtime.pyyaml,
        "device": runtime.device,
        "platform": runtime.platform,
        "machine": runtime.machine,
    }


def _power_record(power: PowerControlCommitment) -> dict[str, object]:
    return {
        "record_path": power.relative_path,
        "file_sha256": power.file_sha256,
        "record_sha256": power.record_sha256,
        "validated_before_any_preopen_fit": True,
        "positive_penalty_4_exact_declared_override": True,
        "null_penalty_16_tied_zero_override": True,
    }


def _base_claims() -> dict[str, bool]:
    return {
        "prospective_execution_schedule_frozen_before_model_fitting": True,
        "prospective_execution_robustness_inside_known_semantics": True,
        "trusted_controller_knows_outer_identifier": True,
        "known_event_semantics_used_by_trusted_attester": True,
        "outer_identifier_received_by_coefficient_estimator": False,
        "selector_seed_derived_from_inferred_outer_identifier": False,
        "fold_seed_uses_absolute_pseudo_cell_index": True,
        "environment_optimizer_seed_is_precommitted_balanced_assignment_factor": True,
        "outer_labels_used_for_fit_or_selection": False,
        "all_fold_candidates_require_exact_seen_fit": True,
        "all_fold_candidates_require_zero_local_overrides": True,
        "numeric_seed_reuse_is_common_random_number_matching": False,
        "shared_rng_seed_effect_identified": False,
        "task_fingerprint_makes_repeated_numeric_seeds_independent_draws": True,
        "schedule_balances_cell_to_seed_index_assignment": True,
        "supplied_addressable_register_representation": True,
        "transition_coefficients_learned_from_visible_traces": True,
        "representation_discovery_performed": False,
        "secret_law_discovery_performed": False,
        "assumption_free_algebra_discovery_performed": False,
        "confirmatory_claim_permitted": False,
        "artifact_is_independently_authenticated_execution_certificate": False,
        "artifact_is_content_bound_trusted_runner_record": True,
        "fold_candidate_training_and_pseudoquery_gate_fields_independently_replayed": False,
        "fold_candidate_training_and_pseudoquery_gate_fields_are_trusted_runner_certificates": True,
        "final_model_direct_train_fit_independently_replayed": True,
    }


def _preopen_claims() -> dict[str, bool]:
    return {
        **_base_claims(),
        "outer_probe_suite_constructed": False,
        "outer_probe_answers_read": False,
        "outer_or_rotated_probe_evaluation_performed": False,
        "terminal_preopen_aggregate_required_before_any_outer_probe": True,
    }


def _open_claims() -> dict[str, bool]:
    return {
        **_base_claims(),
        "terminal_preopen_aggregate_validated_before_any_outer_probe": True,
        "all_forty_frozen_models_opened_in_one_batch_phase": True,
        "failed_environment_replacement_permitted": False,
        "cell_rotation_called_program_conjugacy_or_equivariance": False,
    }


def _seal_body(body: Mapping[str, object]) -> dict[str, object]:
    material = dict(body)
    if "record_sha256" in material:
        raise ValueError("unsealed body cannot contain record_sha256")
    return {**material, "record_sha256": _sha256_bytes(_canonical_bytes(material))}


def _record_body(record: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    material = dict(record)
    digest = material.pop("record_sha256", None)
    _require_sha256("record_sha256", digest)
    if digest != _sha256_bytes(_canonical_bytes(material)):
        raise ValueError("record_sha256 does not bind the full record")
    return material


def _build_frozen_trace_corpus(protocol: FrozenProtocol, spec: EnvironmentSpec):
    base_path = (_repository_root() / protocol.base_task_relative_path).resolve()
    base = load_milestone4_campaign_config(base_path)
    task = replace(base.task, heldout_key_value_pairs=(spec.outer_cell,))
    corpus = _load_single_runner().build_trace_corpus(
        task,
        outer_cell=spec.outer_cell,
        train_seed=spec.train_seed,
        validation_seed=spec.validation_seed,
        document_count=protocol.documents_per_split,
        document_length=protocol.document_length,
    )
    return task, corpus


def _replay_preflight(
    protocol: FrozenProtocol,
    environment_count: int,
    max_validation_replay_generated_events: int,
) -> int:
    cap = _plain_int(
        "max_validation_replay_generated_events",
        max_validation_replay_generated_events,
        1,
    )
    replay = (
        environment_count
        * 2
        * protocol.documents_per_split
        * protocol.document_length
    )
    if replay > cap or replay > protocol.max_deterministic_replay_generated_events_total:
        raise SequenceDiscoveryLimitError(
            "deterministic validation replay exceeds its budget before generation"
        )
    return replay


def _replay_final_model_on_direct_train(
    corpus: object,
    selection: SequenceAlgebraSelectionResult,
) -> dict[str, object]:
    """Independently rescore the serialized final model on direct TRAIN."""

    train_traces = tuple(
        trace
        for trace in corpus.traces
        if trace.attestation.split == "train"
    )
    if not train_traces:
        raise ValueError("final-model replay requires direct TRAIN traces")
    full_train = make_sequence_corpus(
        corpus.num_surface_keys,
        corpus.value_cardinality,
        split="train",
        sequences=tuple(trace.sequence for trace in train_traces),
    )
    mistakes = 0
    query_count = 0
    for trace in train_traces:
        predictions = selection.final_model.predict(trace.sequence)
        for prediction, target in zip(
            predictions, trace.sequence.query_targets, strict=True
        ):
            if target is None:
                if prediction is not None:
                    raise ValueError("final model emitted an answer at a non-query event")
                continue
            query_count += 1
            mistakes += prediction != target
    fit = selection.final_model.fit
    return {
        "replay_uses_serialized_final_model_and_visible_direct_train_only": True,
        "train_sequence_count": len(train_traces),
        "train_event_count": sum(len(trace.sequence.events) for trace in train_traces),
        "training_sample_sha256": full_train.sample_sha256,
        "fit_certificate_training_sample_sha256": fit.training_sample_sha256,
        "training_sample_sha256_matches_fit_certificate": (
            full_train.sample_sha256 == fit.training_sample_sha256
        ),
        "replayed_training_query_count": query_count,
        "fit_certificate_training_query_count": fit.training_query_count,
        "training_query_count_matches_fit_certificate": (
            query_count == fit.training_query_count
        ),
        "replayed_training_mistakes": mistakes,
        "fit_certificate_training_mistakes": fit.training_mistakes,
        "training_mistakes_match_fit_certificate": mistakes == fit.training_mistakes,
        "replayed_final_train_exact": mistakes == 0,
        "final_train_fit_independently_verified": (
            full_train.sample_sha256 == fit.training_sample_sha256
            and query_count == fit.training_query_count
            and mistakes == fit.training_mistakes == 0
        ),
    }


def _fold_seen_fit_report(
    protocol: FrozenProtocol,
    selection: SequenceAlgebraSelectionResult,
    final_train_replay: Mapping[str, object],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for fold in selection.folds:
        absolute_index = (
            fold.pseudoheldout_cell[0] * protocol.value_cardinality
            + fold.pseudoheldout_cell[1]
        )
        expected_seed = selection.final_model.fit.seed + absolute_index
        if fold.optimizer_seed != expected_seed:
            raise ValueError("fold optimizer seed is not based on absolute cell index")
        for candidate in fold.candidates:
            rows.append(
                {
                    "pseudoheldout_cell": list(fold.pseudoheldout_cell),
                    "optimizer_seed": fold.optimizer_seed,
                    "residual_penalty": candidate.residual_penalty,
                    "pseudo_query_count": candidate.pseudo_query_count,
                    "training_mistakes": candidate.training_mistakes,
                    "residual_override_count": candidate.residual_override_count,
                    "model_fingerprint": candidate.model_fingerprint,
                }
            )
    exact = sum(row["training_mistakes"] == 0 for row in rows)
    clean = sum(
        row["training_mistakes"] == 0 and row["residual_override_count"] == 0
        for row in rows
    )
    adequate = sum(
        row["pseudo_query_count"]
        >= protocol.minimum_pseudo_dependent_queries_per_fold
        for row in rows
    )
    final = selection.final_model.fit
    final_replay_verified = (
        final_train_replay.get("final_train_fit_independently_verified") is True
    )
    passed = (
        clean == len(rows)
        and adequate == len(rows)
        and final.training_mistakes == 0
        and final.residual_override_count == 0
        and final_replay_verified
    )
    return {
        "gate": "zero_training_mistakes_and_zero_local_overrides_with_minimum_pseudoquery_support",
        "candidate_count": len(rows),
        "exact_seen_fit_candidate_count": exact,
        "optimization_clean_candidate_count": clean,
        "adequately_scored_candidate_count": adequate,
        "admissible_candidate_count": sum(
            row["training_mistakes"] == 0
            and row["residual_override_count"] == 0
            and row["pseudo_query_count"]
            >= protocol.minimum_pseudo_dependent_queries_per_fold
            for row in rows
        ),
        "every_fold_candidate_exact": exact == len(rows),
        "every_fold_candidate_zero_override": all(
            row["residual_override_count"] == 0 for row in rows
        ),
        "every_fold_candidate_optimization_clean": clean == len(rows),
        "minimum_pseudo_dependent_queries_per_fold_candidate": (
            protocol.minimum_pseudo_dependent_queries_per_fold
        ),
        "every_fold_candidate_has_minimum_pseudo_queries": adequate == len(rows),
        "zero_objective_is_known_global_minimum": True,
        "candidate_rows": rows,
        "final_training_query_count": final.training_query_count,
        "final_training_mistakes": final.training_mistakes,
        "final_local_override_count": final.residual_override_count,
        "final_seen_fit_exact": final.training_mistakes == 0,
        "final_optimization_clean": (
            final.training_mistakes == 0 and final.residual_override_count == 0
        ),
        "final_train_fit_independently_replayed": final_replay_verified,
        "validation_interpretable": clean == len(rows) and adequate == len(rows),
        "environment_preopen_gate_passed": passed,
    }


def _prerequisites(
    protocol: FrozenProtocol,
    implementation_manifest_path: Path | None,
    power_control_record_path: Path | None,
) -> tuple[ImplementationManifest, RuntimeRecord, PowerControlCommitment]:
    # Ordering is deliberate: every item is validated before generation/fitting.
    manifest = load_implementation_manifest(protocol, implementation_manifest_path)
    runtime = validate_runtime(protocol)
    power = load_power_control_commitment(protocol, power_control_record_path)
    return manifest, runtime, power


def build_preopen_environment_record(
    environment_index: int,
    protocol_path: Path | None = None,
    *,
    implementation_manifest_path: Path | None = None,
    power_control_record_path: Path | None = None,
    max_total_primary_generated_events: int = DEFAULT_MAX_TOTAL_PRIMARY_GENERATED_EVENTS,
    max_total_scored_event_work: int = DEFAULT_MAX_TOTAL_SCORED_EVENT_WORK,
) -> dict[str, object]:
    """Fit/select one immutable PREOPEN shard without constructing probes."""

    index = _plain_int("environment_index", environment_index)
    protocol, preflight = preflight_campaign(
        protocol_path,
        environment_indices=(index,),
        max_total_primary_generated_events=max_total_primary_generated_events,
        max_total_scored_event_work=max_total_scored_event_work,
    )
    manifest, runtime, power = _prerequisites(
        protocol, implementation_manifest_path, power_control_record_path
    )
    spec = protocol.schedule[index]
    _, corpus = _build_frozen_trace_corpus(protocol, spec)
    train_events = protocol.documents_per_split * protocol.document_length
    all_events = 2 * train_events
    rotation = run_outer_rotation(
        (corpus,),
        max_environments=1,
        max_outer_aggregate_scored_event_work=(
            protocol.max_scored_event_work_per_environment
        ),
        require_complete_single_cell_rotation=False,
        residual_penalties=protocol.residual_penalties,
        seed=spec.optimizer_seed,
        restart_count=protocol.restart_count,
        max_sweeps=protocol.max_sweeps,
        max_pairwise_rounds=protocol.max_pairwise_rounds,
        required_outer_unobserved_cell_count=1,
        max_events_per_fit=train_events,
        max_controller_events=all_events,
        max_folds=protocol.inner_folds_per_environment,
        max_fit_calls=protocol.fit_calls_per_environment,
        max_objective_evaluations_per_fit=(
            protocol.planned_objective_evaluations_per_fit
        ),
        max_scored_event_work_per_fit=protocol.max_scored_event_work_per_fit,
        max_aggregate_scored_event_work=(
            protocol.max_scored_event_work_per_environment
        ),
    )
    if rotation.omitted_cell_sets != ((spec.outer_cell,),):
        raise ValueError("trusted omission inferred by outer rotation changed")
    selection = rotation.results[0]
    final_train_replay = _replay_final_model_on_direct_train(corpus, selection)
    gate = _fold_seen_fit_report(protocol, selection, final_train_replay)
    status = (
        "preopen_gate_passed_waiting_for_terminal_campaign_aggregate"
        if gate["environment_preopen_gate_passed"]
        else "preopen_gate_failed_outer_open_forbidden"
    )
    rotation_json = _jsonable(rotation)
    if not isinstance(rotation_json, dict):
        raise TypeError("outer rotation must serialize to a mapping")
    rotation_json.pop("results")
    body: dict[str, object] = {
        "schema": PREOPEN_ENVIRONMENT_SCHEMA,
        "scope": (
            "prospective_execution_robustness_inside_known_binding_semantics_"
            "not_secret_law_or_representation_discovery"
        ),
        "protocol": _protocol_record(protocol),
        "implementation": _implementation_record(manifest),
        "runtime": _runtime_record(runtime),
        "power_control": _power_record(power),
        "environment": _environment_spec_payload(spec),
        "base_task_config": {
            "path": protocol.base_task_relative_path,
            "sha256": protocol.base_task_sha256,
        },
        "budget_preflight": _jsonable(preflight),
        "controller_budget_limits": {
            "max_total_primary_generated_events": (
                max_total_primary_generated_events
            ),
            "max_total_scored_event_work": max_total_scored_event_work,
        },
        "generation_work_accounting": {
            "primary_fit_generation_events": all_events,
            "validation_replay_generation_events": 0,
            "all_generation_events_in_this_operation": all_events,
            "replay_is_validation_only_never_refit_or_reselection": True,
        },
        "data": {
            "documents_per_split": protocol.documents_per_split,
            "document_length": protocol.document_length,
            "trace_corpus_sha256": corpus.corpus_sha256,
            "derived_seed_domain_includes_full_task_fingerprint": True,
            "repeated_numeric_seed_pair_is_common_random_number_match": False,
        },
        "execution_status": status,
        "singleton_outer_rotation": rotation_json,
        "selection": _jsonable(selection),
        "final_train_replay": final_train_replay,
        "seen_fit_gate": gate,
        "claims": _preopen_claims(),
    }
    record = _seal_body(body)
    # The in-memory corpus avoids an unbudgeted certificate-regeneration pass.
    _validate_preopen_material(
        record,
        protocol,
        manifest,
        runtime,
        power,
        regenerated_corpus=corpus,
    )
    return record


def _require_environment_match(
    raw: Mapping[str, object], expected: EnvironmentSpec
) -> None:
    if dict(raw) != _environment_spec_payload(expected):
        raise ValueError("environment record does not match the frozen schedule")


def _validate_preopen_material(
    record: Mapping[str, object],
    protocol: FrozenProtocol,
    manifest: ImplementationManifest,
    runtime: RuntimeRecord,
    power: PowerControlCommitment,
    *,
    regenerated_corpus: object,
) -> _ValidatedPreopen:
    body = _record_body(record)
    expected_keys = {
        "schema",
        "scope",
        "protocol",
        "implementation",
        "runtime",
        "power_control",
        "environment",
        "base_task_config",
        "budget_preflight",
        "controller_budget_limits",
        "generation_work_accounting",
        "data",
        "execution_status",
        "singleton_outer_rotation",
        "selection",
        "final_train_replay",
        "seen_fit_gate",
        "claims",
    }
    if set(body) != expected_keys:
        raise ValueError("preopen shard fields differ from the closed schema")
    if body.get("schema") != PREOPEN_ENVIRONMENT_SCHEMA:
        raise ValueError("unknown preopen environment schema")
    if body.get("scope") != (
        "prospective_execution_robustness_inside_known_binding_semantics_"
        "not_secret_law_or_representation_discovery"
    ):
        raise ValueError("preopen scope changed")
    if _mapping("protocol", body.get("protocol")) != _protocol_record(protocol):
        raise ValueError("preopen shard is not bound to the protocol")
    if _mapping("implementation", body.get("implementation")) != _implementation_record(
        manifest
    ):
        raise ValueError("preopen shard implementation binding changed")
    if _mapping("runtime", body.get("runtime")) != _runtime_record(runtime):
        raise ValueError("preopen shard runtime binding changed")
    if _mapping("power control", body.get("power_control")) != _power_record(power):
        raise ValueError("preopen shard power-control binding changed")
    if _mapping("claims", body.get("claims")) != _preopen_claims():
        raise ValueError("preopen shard claims changed")
    environment = _mapping("environment", body.get("environment"))
    index = _plain_int("environment_index", environment.get("environment_index"))
    if index >= len(protocol.schedule):
        raise ValueError("environment index is outside the schedule")
    spec = protocol.schedule[index]
    _require_environment_match(environment, spec)
    if _mapping("base task", body.get("base_task_config")) != {
        "path": protocol.base_task_relative_path,
        "sha256": protocol.base_task_sha256,
    }:
        raise ValueError("base-task binding changed")

    _, expected_preflight = preflight_campaign(
        default_protocol_path(), environment_indices=(index,)
    )
    if body.get("budget_preflight") != _jsonable(expected_preflight):
        raise ValueError("analytic budget preflight changed")
    limits = _mapping("controller limits", body.get("controller_budget_limits"))
    if (
        _plain_int(
            "primary generation limit",
            limits.get("max_total_primary_generated_events"),
            1,
        )
        < expected_preflight.primary_generated_event_count
        or _plain_int(
            "scored work limit", limits.get("max_total_scored_event_work"), 1
        )
        < expected_preflight.total_scored_event_work
    ):
        raise ValueError("recorded controller limits do not admit the planned work")
    all_events = 2 * protocol.documents_per_split * protocol.document_length
    if _mapping("generation accounting", body.get("generation_work_accounting")) != {
        "primary_fit_generation_events": all_events,
        "validation_replay_generation_events": 0,
        "all_generation_events_in_this_operation": all_events,
        "replay_is_validation_only_never_refit_or_reselection": True,
    }:
        raise ValueError("preopen generation accounting changed")

    data = _mapping("data", body.get("data"))
    trace_sha = _require_sha256("trace corpus SHA-256", data.get("trace_corpus_sha256"))
    if (
        data.get("documents_per_split") != protocol.documents_per_split
        or data.get("document_length") != protocol.document_length
        or data.get("derived_seed_domain_includes_full_task_fingerprint") is not True
        or data.get("repeated_numeric_seed_pair_is_common_random_number_match") is not False
    ):
        raise ValueError("preopen data record changed")
    if getattr(regenerated_corpus, "corpus_sha256", None) != trace_sha:
        raise ValueError("trace corpus does not reproduce from the frozen schedule")

    selection_json = _mapping("selection", body.get("selection"))
    selection = _decode_dataclass(
        SequenceAlgebraSelectionResult, selection_json, "selection"
    )
    if _jsonable(selection) != dict(selection_json):
        raise ValueError("selection does not round-trip through its real dataclass")
    if selection.source_corpus_sha256 != trace_sha:
        raise ValueError("selection does not bind the trace corpus")
    fit = selection.final_model.fit
    if (
        fit.seed != spec.optimizer_seed
        or fit.restart_count != protocol.restart_count
        or fit.max_sweeps != protocol.max_sweeps
        or fit.max_pairwise_rounds != protocol.max_pairwise_rounds
        or fit.residual_penalty != selection.selected_residual_penalty
        or selection.selected_residual_penalty not in protocol.residual_penalties
    ):
        raise ValueError("selection optimizer settings changed")
    expected_cells = {
        (key, value)
        for key in range(protocol.num_surface_keys)
        for value in range(protocol.value_cardinality)
    } - {spec.outer_cell}
    if {fold.pseudoheldout_cell for fold in selection.folds} != expected_cells:
        raise ValueError("selection does not contain the exact 19 observed folds")
    if any(
        tuple(row.residual_penalty for row in fold.candidates)
        != protocol.residual_penalties
        for fold in selection.folds
    ):
        raise ValueError("fold candidate penalty grid changed")
    expected_final_train_replay = _replay_final_model_on_direct_train(
        regenerated_corpus, selection
    )
    if _mapping("final TRAIN replay", body.get("final_train_replay")) != (
        expected_final_train_replay
    ):
        raise ValueError("final model direct-TRAIN replay does not reproduce")
    expected_gate = _fold_seen_fit_report(
        protocol, selection, expected_final_train_replay
    )
    if _mapping("seen-fit gate", body.get("seen_fit_gate")) != expected_gate:
        raise ValueError("seen-fit gate does not reproduce the selection")
    expected_status = (
        "preopen_gate_passed_waiting_for_terminal_campaign_aggregate"
        if expected_gate["environment_preopen_gate_passed"]
        else "preopen_gate_failed_outer_open_forbidden"
    )
    if body.get("execution_status") != expected_status:
        raise ValueError("preopen execution status disagrees with its gate")

    rotation_json = _mapping(
        "singleton outer rotation", body.get("singleton_outer_rotation")
    )
    rotation = _decode_dataclass(
        OuterRotationResult,
        {**dict(rotation_json), "results": [dict(selection_json)]},
        "singleton_outer_rotation",
    )
    if rotation.results != (selection,):
        raise ValueError("singleton rotation does not bind the selection")
    if rotation.omitted_cell_sets != ((spec.outer_cell,),):
        raise ValueError("singleton rotation omission changed")
    return _ValidatedPreopen(index, record, body, selection, trace_sha)


def validate_preopen_environment_record(
    record: Mapping[str, object],
    protocol: FrozenProtocol | None = None,
    *,
    implementation_manifest_path: Path | None = None,
    power_control_record_path: Path | None = None,
    max_validation_replay_generated_events: int = (
        DEFAULT_MAX_TOTAL_VALIDATION_REPLAY_EVENTS
    ),
) -> int:
    """Production validation: reconstruct certificates and replay the corpus."""

    protocol = load_frozen_protocol() if protocol is None else protocol
    if type(protocol) is not FrozenProtocol:
        raise TypeError("protocol must be exact FrozenProtocol")
    manifest, runtime, power = _prerequisites(
        protocol, implementation_manifest_path, power_control_record_path
    )
    body = _record_body(record)
    environment = _mapping("environment", body.get("environment"))
    index = _plain_int("environment_index", environment.get("environment_index"))
    if index >= len(protocol.schedule):
        raise ValueError("environment index is outside the schedule")
    _replay_preflight(protocol, 1, max_validation_replay_generated_events)
    _, corpus = _build_frozen_trace_corpus(protocol, protocol.schedule[index])
    return _validate_preopen_material(
        record,
        protocol,
        manifest,
        runtime,
        power,
        regenerated_corpus=corpus,
    ).index


def _terminal_preopen_claims() -> dict[str, bool]:
    return {
        **_preopen_claims(),
        "all_frozen_preopen_shards_content_bound": True,
        "all_preopen_gates_checked_before_outer_open_permission": True,
        "failed_environment_replacement_permitted": False,
    }


def _validate_complete_preopen_records(
    records: Sequence[Mapping[str, object]],
    protocol: FrozenProtocol,
    manifest: ImplementationManifest,
    runtime: RuntimeRecord,
    power: PowerControlCommitment,
    *,
    max_validation_replay_generated_events: int,
) -> tuple[tuple[_ValidatedPreopen, ...], int]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("preopen records must be a finite sequence")
    if len(records) != len(protocol.schedule):
        raise ValueError("terminal preopen aggregation requires all forty shards")
    # Audit index/uniqueness and the entire replay budget before first regeneration.
    indexed: dict[int, Mapping[str, object]] = {}
    for record in records:
        body = _record_body(record)
        if body.get("schema") != PREOPEN_ENVIRONMENT_SCHEMA:
            raise ValueError("production aggregation rejects non-evidence shard schemas")
        environment = _mapping("environment", body.get("environment"))
        index = _plain_int("environment_index", environment.get("environment_index"))
        if index >= len(protocol.schedule):
            raise ValueError("environment index is outside the frozen schedule")
        if index in indexed:
            raise ValueError("terminal preopen aggregation contains a duplicate environment")
        indexed[index] = record
    if set(indexed) != set(range(len(protocol.schedule))):
        raise ValueError("terminal preopen aggregation is missing a frozen environment")
    replay_events = _replay_preflight(
        protocol,
        len(protocol.schedule),
        max_validation_replay_generated_events,
    )
    validated: list[_ValidatedPreopen] = []
    for index in range(len(protocol.schedule)):
        _, corpus = _build_frozen_trace_corpus(protocol, protocol.schedule[index])
        validated.append(
            _validate_preopen_material(
                indexed[index],
                protocol,
                manifest,
                runtime,
                power,
                regenerated_corpus=corpus,
            )
        )
    return tuple(validated), replay_events


def _build_terminal_preopen_from_validated(
    validated: Sequence[_ValidatedPreopen],
    protocol: FrozenProtocol,
    manifest: ImplementationManifest,
    runtime: RuntimeRecord,
    power: PowerControlCommitment,
    *,
    replay_events: int,
    prior_validation_replay_events: int = 0,
) -> dict[str, object]:
    if len(validated) != len(protocol.schedule):
        raise ValueError("terminal preopen requires the complete schedule")
    if tuple(row.index for row in validated) != tuple(range(len(protocol.schedule))):
        raise ValueError("validated preopen rows are not in canonical schedule order")
    record_hashes = tuple(str(row.record["record_sha256"]) for row in validated)
    corpus_hashes = tuple(row.corpus_sha256 for row in validated)
    if len(set(record_hashes)) != len(record_hashes):
        raise ValueError("preopen shard hashes must be unique")
    if len(set(corpus_hashes)) != len(corpus_hashes):
        raise ValueError("every scheduled environment must bind a distinct corpus")

    summaries: list[dict[str, object]] = []
    penalty_counts: Counter[int] = Counter()
    cell_rows: dict[tuple[int, int], list[int]] = defaultdict(list)
    seed_rows: dict[int, list[int]] = defaultdict(list)
    candidate_count = admissible_count = 0
    final_mistakes = final_overrides = 0
    passing = 0
    for row in validated:
        spec = protocol.schedule[row.index]
        gate = _mapping("seen-fit gate", row.body["seen_fit_gate"])
        final_replay = _mapping(
            "final TRAIN replay", row.body["final_train_replay"]
        )
        selection = row.selection
        fit = selection.final_model.fit
        penalty_counts[selection.selected_residual_penalty] += 1
        cell_rows[spec.outer_cell].append(row.index)
        seed_rows[spec.seed_pair_index].append(row.index)
        candidate_count += int(gate["candidate_count"])
        admissible_count += int(gate["admissible_candidate_count"])
        final_mistakes += fit.training_mistakes
        final_overrides += fit.residual_override_count
        passed = gate["environment_preopen_gate_passed"] is True
        passing += passed
        summaries.append(
            {
                **_environment_spec_payload(spec),
                "preopen_record_sha256": row.record["record_sha256"],
                "trace_corpus_sha256": row.corpus_sha256,
                "selection_result_sha256": selection.result_sha256,
                "model_fingerprint": selection.final_model.model_fingerprint,
                "selected_residual_penalty": selection.selected_residual_penalty,
                "fold_candidate_count": gate["candidate_count"],
                "admissible_fold_candidate_count": gate["admissible_candidate_count"],
                "minimum_pseudo_query_count": min(
                    candidate.pseudo_query_count
                    for fold in selection.folds
                    for candidate in fold.candidates
                ),
                "final_training_query_count": fit.training_query_count,
                "final_training_mistakes": fit.training_mistakes,
                "final_local_override_count": fit.residual_override_count,
                "replayed_final_training_query_count": final_replay[
                    "replayed_training_query_count"
                ],
                "replayed_final_training_mistakes": final_replay[
                    "replayed_training_mistakes"
                ],
                "final_train_fit_independently_verified": final_replay[
                    "final_train_fit_independently_verified"
                ],
                "environment_preopen_gate_passed": passed,
            }
        )

    cell_count = protocol.num_surface_keys * protocol.value_cardinality
    block_count = len(protocol.schedule) // cell_count
    universe = {
        (key, value)
        for key in range(protocol.num_surface_keys)
        for value in range(protocol.value_cardinality)
    }
    complete_blocks = all(
        {row.outer_cell for row in protocol.schedule[start : start + cell_count]}
        == universe
        for start in range(0, len(protocol.schedule), cell_count)
    )
    coverage = [
        {
            "outer_cell": list(cell),
            "environment_indices": indices,
            "repetition_count": len(indices),
        }
        for cell, indices in sorted(cell_rows.items())
    ]
    seed_coverage = [
        {
            "seed_pair_index": seed,
            "environment_indices": indices,
            "repetition_count": len(indices),
        }
        for seed, indices in sorted(seed_rows.items())
    ]
    outer_open_permitted = (
        passing == protocol.required_passing_environments
        and candidate_count == protocol.required_admissible_fold_candidates
        and admissible_count == protocol.required_admissible_fold_candidates
        and final_mistakes == 0
        and final_overrides == 0
        and complete_blocks
        and all(item["repetition_count"] == block_count for item in coverage)
        and all(item["repetition_count"] == block_count for item in seed_coverage)
    )
    primary_events = len(protocol.schedule) * 2 * protocol.documents_per_split * protocol.document_length
    prior_replay = _plain_int(
        "prior_validation_replay_events", prior_validation_replay_events
    )
    cumulative_replay = prior_replay + replay_events
    if (
        cumulative_replay > protocol.max_deterministic_replay_generated_events_total
        or primary_events + cumulative_replay > protocol.max_all_generation_work_total
    ):
        raise SequenceDiscoveryLimitError("terminal cumulative generation exceeds protocol")
    body: dict[str, object] = {
        "schema": PREOPEN_AGGREGATE_SCHEMA,
        "scope": (
            "terminal_preopen_commitment_for_prospective_execution_robustness_"
            "inside_known_semantics"
        ),
        "protocol": _protocol_record(protocol),
        "implementation": _implementation_record(manifest),
        "runtime": _runtime_record(runtime),
        "power_control": _power_record(power),
        "preopen_environment_record_sha256s": list(record_hashes),
        "environments": summaries,
        "generation_work_accounting": {
            "primary_fit_generation_events_bound_from_shards": primary_events,
            "prior_resume_validation_replay_events": prior_replay,
            "terminal_aggregate_validation_replay_events": replay_events,
            "cumulative_validation_replay_events_through_terminal_preopen": (
                cumulative_replay
            ),
            "cumulative_generation_events_through_terminal_preopen": (
                primary_events + cumulative_replay
            ),
            "protocol_primary_generation_cap": (
                protocol.max_primary_generated_events_total
            ),
            "protocol_replay_generation_cap": (
                protocol.max_deterministic_replay_generated_events_total
            ),
            "protocol_all_generation_cap": protocol.max_all_generation_work_total,
            "replay_is_validation_only_never_refit_or_reselection": True,
        },
        "aggregate": {
            "environment_count": len(validated),
            "complete_cell_rotation_block_count": block_count,
            "every_block_complete_and_unique": complete_blocks,
            "outer_cell_coverage": coverage,
            "seed_pair_coverage": seed_coverage,
            "selected_penalty_counts": {
                str(penalty): penalty_counts.get(penalty, 0)
                for penalty in protocol.residual_penalties
            },
            "fold_candidate_count": candidate_count,
            "admissible_fold_candidate_count": admissible_count,
            "total_final_training_mistakes": final_mistakes,
            "total_final_local_override_count": final_overrides,
            "passing_preopen_environment_count": passing,
        },
        "acceptance": {
            "required_passing_environments": protocol.required_passing_environments,
            "required_admissible_fold_candidates": (
                protocol.required_admissible_fold_candidates
            ),
            "every_candidate_zero_training_mistakes_and_zero_overrides": (
                admissible_count == candidate_count
                == protocol.required_admissible_fold_candidates
            ),
            "every_final_fit_zero_training_mistakes_and_zero_overrides": (
                final_mistakes == 0 and final_overrides == 0
            ),
            "selected_penalty_not_used_as_gate": True,
            "outer_open_permitted": outer_open_permitted,
        },
        "claims": _terminal_preopen_claims(),
    }
    return _seal_body(body)


def aggregate_preopen_records(
    records: Sequence[Mapping[str, object]],
    protocol_path: Path | None = None,
    *,
    implementation_manifest_path: Path | None = None,
    power_control_record_path: Path | None = None,
    max_validation_replay_generated_events: int = (
        DEFAULT_MAX_TOTAL_VALIDATION_REPLAY_EVENTS
    ),
    prior_validation_replay_generated_events: int = 0,
) -> dict[str, object]:
    """Replay and commit all forty shards; this function never opens probes."""

    protocol, _ = preflight_campaign(protocol_path)
    manifest, runtime, power = _prerequisites(
        protocol, implementation_manifest_path, power_control_record_path
    )
    replay_cap = _plain_int(
        "max_validation_replay_generated_events",
        max_validation_replay_generated_events,
        1,
    )
    prior_replay = _plain_int(
        "prior_validation_replay_generated_events",
        prior_validation_replay_generated_events,
    )
    planned_replay = (
        len(protocol.schedule)
        * 2
        * protocol.documents_per_split
        * protocol.document_length
    )
    if (
        prior_replay + planned_replay > replay_cap
        or prior_replay + planned_replay
        > protocol.max_deterministic_replay_generated_events_total
    ):
        raise SequenceDiscoveryLimitError(
            "terminal cumulative validation replay exceeds its budget before generation"
        )
    validated, replay_events = _validate_complete_preopen_records(
        records,
        protocol,
        manifest,
        runtime,
        power,
        max_validation_replay_generated_events=replay_cap - prior_replay,
    )
    return _build_terminal_preopen_from_validated(
        validated,
        protocol,
        manifest,
        runtime,
        power,
        replay_events=replay_events,
        prior_validation_replay_events=prior_replay,
    )


def validate_terminal_preopen_aggregate(
    record: Mapping[str, object],
    preopen_records: Sequence[Mapping[str, object]],
    protocol_path: Path | None = None,
    *,
    implementation_manifest_path: Path | None = None,
    power_control_record_path: Path | None = None,
    max_validation_replay_generated_events: int = (
        DEFAULT_MAX_TOTAL_VALIDATION_REPLAY_EVENTS
    ),
) -> None:
    body = _record_body(record)
    accounting = _mapping(
        "terminal generation accounting", body.get("generation_work_accounting")
    )
    prior_replay = _plain_int(
        "terminal prior resume replay",
        accounting.get("prior_resume_validation_replay_events"),
    )
    expected = aggregate_preopen_records(
        preopen_records,
        protocol_path,
        implementation_manifest_path=implementation_manifest_path,
        power_control_record_path=power_control_record_path,
        max_validation_replay_generated_events=(
            max_validation_replay_generated_events
        ),
        prior_validation_replay_generated_events=prior_replay,
    )
    if dict(record) != expected:
        raise ValueError("terminal preopen aggregate does not reproduce all forty shards")


def _transition_table_assessment(model: object) -> dict[str, object]:
    learned = dict(model.shared_outputs)
    rows: list[dict[str, object]] = []
    for address in prototype_inventory(model.value_cardinality):
        expected = (
            address.source_value
            if address.family in {"bind", "copy"}
            else apply_value_transform(
                address.source_value, address.transform, model.value_cardinality
            )
        )
        output = learned[address]
        rows.append(
            {
                "prototype": address.label,
                "expected_output": expected,
                "learned_output": output,
                "exact": output == expected,
            }
        )
    return {
        "known_semantic_target_opened_after_terminal_preopen": True,
        "supported_entry_count": len(rows),
        "exact_entry_count": sum(row["exact"] for row in rows),
        "entries": rows,
    }


def _path_relation_satisfied(raw: object) -> bool:
    row = _mapping("path relation", raw)
    predicted = _exact_list(
        "predicted focal answer rows", row.get("predicted_focal_answers")
    )
    normalized = tuple(
        tuple(_plain_int("predicted focal answer", item) for item in _exact_list(
            "predicted focal answers", answer_row
        ))
        for answer_row in predicted
    )
    if not normalized or any(not row_values for row_values in normalized):
        raise ValueError("path relation answer rows must be nonempty")
    if row.get("relation") == "equal":
        return len(set(normalized)) == 1
    if row.get("relation") == "not_equal":
        return len(set(normalized)) != 1
    raise ValueError("unknown path relation")


def _postfit_report(
    protocol: FrozenProtocol,
    spec: EnvironmentSpec,
    selection: SequenceAlgebraSelectionResult,
) -> dict[str, object]:
    """Construct and evaluate probes.  Called only inside batch-open."""

    single = _load_single_runner()
    suite = build_balanced_probe_suite(
        protocol.num_surface_keys,
        protocol.value_cardinality,
        (spec.outer_cell,),
    )
    evaluation = evaluate_probe_suite(selection.final_model, suite)
    shortcuts = evaluate_shortcut_controls(suite)
    rotations = cyclic_cell_rotation_inventory(
        protocol.num_surface_keys,
        protocol.value_cardinality,
        anchor_key=spec.outer_cell[0],
    )
    rotated_suite = build_balanced_probe_suite(
        protocol.num_surface_keys,
        protocol.value_cardinality,
        (spec.outer_cell,),
        cell_rotations=rotations,
    )
    rotated_evaluation = evaluate_probe_suite(selection.final_model, rotated_suite)
    return {
        "outer_labels_opened_after_terminal_campaign_preopen": True,
        "transition_table": _transition_table_assessment(selection.final_model),
        "actual_cell_probe": {
            "outer_cell": list(spec.outer_cell),
            "case_count": len(suite.cases),
            "evaluation": single._evaluation_summary(evaluation),
        },
        "exact_probe_family_count": sum(
            row.exact_case_count == row.case_count
            for row in evaluation.family_results
        ),
        "shortcut_controls": [
            {
                "name": row.name,
                "evaluation": single._evaluation_summary(row.evaluation),
            }
            for row in shortcuts
        ],
        "balanced_rotated_cell_control": {
            "programs_generated_independently_at_each_destination_cell": True,
            "cell_rotation_called_program_conjugacy_or_equivariance": False,
            "suite_sha256": rotated_suite.suite_sha256,
            "case_count": len(rotated_suite.cases),
            "actual_cell_result": _jsonable(
                rotated_evaluation.result_for_pair(spec.outer_cell)
            ),
            "evaluation": single._evaluation_summary(rotated_evaluation),
        },
    }


def _environment_acceptance(
    protocol: FrozenProtocol,
    selection: SequenceAlgebraSelectionResult,
    gate: Mapping[str, object],
    postfit: Mapping[str, object],
) -> dict[str, object]:
    transition = _mapping("transition table", postfit.get("transition_table"))
    actual = _mapping("actual probe", postfit.get("actual_cell_probe"))
    evaluation = _mapping("actual evaluation", actual.get("evaluation"))
    relations = _exact_list("actual path relations", evaluation.get("path_relations"))
    rotated = _mapping(
        "rotated control", postfit.get("balanced_rotated_cell_control")
    )
    rotated_evaluation = _mapping(
        "rotated evaluation", rotated.get("evaluation")
    )
    transition_exact = (
        transition.get("supported_entry_count")
        == transition.get("exact_entry_count")
        == protocol.exact_supported_transition_entries
    )
    query_exact = (
        actual.get("case_count") == EXPECTED_ACTUAL_CASE_COUNT
        and evaluation.get("query_count") == EXPECTED_ACTUAL_QUERY_COUNT
        and evaluation.get("correct_count") == EXPECTED_ACTUAL_QUERY_COUNT
    )
    focal_exact = (
        evaluation.get("focal_query_count") == EXPECTED_ACTUAL_FOCAL_QUERY_COUNT
        and evaluation.get("focal_correct_count") == EXPECTED_ACTUAL_FOCAL_QUERY_COUNT
    )
    families_exact = postfit.get("exact_probe_family_count") == protocol.exact_probe_families
    paths_exact = (
        len(relations) == EXPECTED_PATH_RELATION_COUNT
        and all(_path_relation_satisfied(row) for row in relations)
        and evaluation.get("path_consistency") == 1.0
    )
    rotated_exact = (
        rotated.get("case_count") == EXPECTED_ROTATED_CASE_COUNT
        and rotated_evaluation.get("query_count") == EXPECTED_ROTATED_QUERY_COUNT
        and rotated_evaluation.get("correct_count") == EXPECTED_ROTATED_QUERY_COUNT
        and rotated_evaluation.get("focal_query_count")
        == EXPECTED_ROTATED_FOCAL_QUERY_COUNT
        and rotated_evaluation.get("focal_correct_count")
        == EXPECTED_ROTATED_FOCAL_QUERY_COUNT
    )
    shortcuts = _exact_list("shortcut controls", postfit.get("shortcut_controls"))
    shortcuts_clear = bool(shortcuts) and all(
        _mapping(
            "shortcut evaluation", _mapping("shortcut", row).get("evaluation")
        ).get("correct_count")
        < evaluation.get("correct_count")
        and _mapping(
            "shortcut evaluation", _mapping("shortcut", row).get("evaluation")
        ).get("focal_correct_count")
        < evaluation.get("focal_correct_count")
        for row in shortcuts
    )
    fit = selection.final_model.fit
    fold_clean = gate.get("environment_preopen_gate_passed") is True
    final_clean = fit.training_mistakes == 0 and fit.residual_override_count == 0
    passed = all(
        (
            fold_clean,
            final_clean,
            transition_exact,
            query_exact,
            focal_exact,
            families_exact,
            paths_exact,
            rotated_exact,
            shortcuts_clear,
        )
    )
    return {
        "terminal_preopen_environment_gate_passed": fold_clean,
        "final_seen_fit_zero_mistakes_and_zero_overrides": final_clean,
        "all_supported_transition_entries_exact": transition_exact,
        "actual_cell_96_of_96_queries_correct": query_exact,
        "actual_cell_24_of_24_focal_queries_correct": focal_exact,
        "all_12_probe_families_exact": families_exact,
        "all_3_path_relations_satisfied": paths_exact,
        "rotated_control_1800_of_1800_and_480_focal_exact": rotated_exact,
        "all_shortcut_controls_strictly_worse_overall_and_focal": shortcuts_clear,
        "selected_penalty_is_not_an_acceptance_gate": True,
        "environment_passed": passed,
    }


def _build_open_environment_record(
    protocol: FrozenProtocol,
    manifest: ImplementationManifest,
    runtime: RuntimeRecord,
    power: PowerControlCommitment,
    terminal_preopen: Mapping[str, object],
    validated: _ValidatedPreopen,
) -> dict[str, object]:
    spec = protocol.schedule[validated.index]
    gate = _mapping("seen-fit gate", validated.body["seen_fit_gate"])
    postfit = _postfit_report(protocol, spec, validated.selection)
    acceptance = _environment_acceptance(
        protocol, validated.selection, gate, postfit
    )
    body: dict[str, object] = {
        "schema": OPEN_ENVIRONMENT_SCHEMA,
        "scope": (
            "post_terminal_batch_open_inside_known_binding_semantics_not_"
            "secret_law_or_representation_discovery"
        ),
        "protocol": _protocol_record(protocol),
        "implementation": _implementation_record(manifest),
        "runtime": _runtime_record(runtime),
        "power_control": _power_record(power),
        "terminal_preopen_record_sha256": terminal_preopen["record_sha256"],
        "preopen_environment_record_sha256": validated.record["record_sha256"],
        "environment": _environment_spec_payload(spec),
        "selection_result_sha256": validated.selection.result_sha256,
        "model_fingerprint": validated.selection.final_model.model_fingerprint,
        "selected_residual_penalty": validated.selection.selected_residual_penalty,
        "postfit": postfit,
        "acceptance": acceptance,
        "claims": _open_claims(),
    }
    return _seal_body(body)


def _build_open_campaign_record(
    protocol: FrozenProtocol,
    manifest: ImplementationManifest,
    runtime: RuntimeRecord,
    power: PowerControlCommitment,
    terminal_preopen: Mapping[str, object],
    validated: Sequence[_ValidatedPreopen],
    open_records: Sequence[Mapping[str, object]],
    *,
    preopen_validation_replay_events: int,
) -> dict[str, object]:
    if len(open_records) != len(protocol.schedule):
        raise ValueError("batch open must retain all forty environments")
    open_hashes = tuple(str(record["record_sha256"]) for record in open_records)
    if len(set(open_hashes)) != len(open_hashes):
        raise ValueError("open environment record hashes must be unique")
    penalty_counts: Counter[int] = Counter()
    passing = 0
    actual_queries = actual_correct = actual_focal = actual_focal_correct = 0
    rotated_queries = rotated_correct = rotated_focal = rotated_focal_correct = 0
    summaries: list[dict[str, object]] = []
    for row, record in zip(validated, open_records, strict=True):
        body = _record_body(record)
        if body.get("schema") != OPEN_ENVIRONMENT_SCHEMA:
            raise ValueError("unknown open environment schema")
        if body.get("terminal_preopen_record_sha256") != terminal_preopen["record_sha256"]:
            raise ValueError("open environment does not bind the terminal aggregate")
        if body.get("preopen_environment_record_sha256") != row.record["record_sha256"]:
            raise ValueError("open environment does not bind its preopen shard")
        if _mapping("environment", body.get("environment")) != _environment_spec_payload(
            protocol.schedule[row.index]
        ):
            raise ValueError("open environment schedule row changed")
        penalty = row.selection.selected_residual_penalty
        penalty_counts[penalty] += 1
        acceptance = _mapping("acceptance", body["acceptance"])
        passing += acceptance["environment_passed"] is True
        postfit = _mapping("postfit", body["postfit"])
        actual = _mapping(
            "actual evaluation",
            _mapping("actual probe", postfit["actual_cell_probe"])["evaluation"],
        )
        rotated = _mapping(
            "rotated evaluation",
            _mapping("rotated control", postfit["balanced_rotated_cell_control"])[
                "evaluation"
            ],
        )
        actual_queries += int(actual["query_count"])
        actual_correct += int(actual["correct_count"])
        actual_focal += int(actual["focal_query_count"])
        actual_focal_correct += int(actual["focal_correct_count"])
        rotated_queries += int(rotated["query_count"])
        rotated_correct += int(rotated["correct_count"])
        rotated_focal += int(rotated["focal_query_count"])
        rotated_focal_correct += int(rotated["focal_correct_count"])
        summaries.append(
            {
                **_environment_spec_payload(protocol.schedule[row.index]),
                "preopen_environment_record_sha256": row.record["record_sha256"],
                "open_environment_record_sha256": record["record_sha256"],
                "selection_result_sha256": row.selection.result_sha256,
                "model_fingerprint": row.selection.final_model.model_fingerprint,
                "selected_residual_penalty": penalty,
                "environment_passed": acceptance["environment_passed"],
            }
        )
    campaign_passed = passing == protocol.required_passing_environments
    primary_events = protocol.max_primary_generated_events_total
    terminal_accounting = _mapping(
        "terminal generation accounting",
        terminal_preopen.get("generation_work_accounting"),
    )
    prior_resume_replay = _plain_int(
        "prior resume replay",
        terminal_accounting.get("prior_resume_validation_replay_events"),
    )
    terminal_replay = _plain_int(
        "terminal aggregate replay",
        terminal_accounting.get("terminal_aggregate_validation_replay_events"),
        1,
    )
    cumulative_replay = (
        prior_resume_replay + terminal_replay + preopen_validation_replay_events
    )
    body: dict[str, object] = {
        "schema": CAMPAIGN_SCHEMA,
        "scope": (
            "prospective_execution_robustness_inside_known_binding_semantics_"
            "not_secret_law_or_representation_discovery"
        ),
        "protocol": _protocol_record(protocol),
        "implementation": _implementation_record(manifest),
        "runtime": _runtime_record(runtime),
        "power_control": _power_record(power),
        "terminal_preopen_record_sha256": terminal_preopen["record_sha256"],
        "preopen_environment_record_sha256s": [
            row.record["record_sha256"] for row in validated
        ],
        "open_environment_record_sha256s": list(open_hashes),
        "open_environment_records": list(open_records),
        "environments": summaries,
        "generation_work_accounting": {
            "primary_fit_generation_events_bound_from_preopen": primary_events,
            "prior_resume_validation_replay_events": prior_resume_replay,
            "terminal_preopen_validation_replay_events": terminal_replay,
            "batch_open_prevalidation_replay_events": preopen_validation_replay_events,
            "cumulative_validation_replay_events_through_batch_open": (
                cumulative_replay
            ),
            "cumulative_generation_events_through_batch_open": (
                primary_events + cumulative_replay
            ),
            "protocol_replay_generation_cap": (
                protocol.max_deterministic_replay_generated_events_total
            ),
            "protocol_all_generation_cap": protocol.max_all_generation_work_total,
            "replay_is_validation_only_never_refit_or_reselection": True,
        },
        "aggregate": {
            "environment_count": len(open_records),
            "passing_environment_count": passing,
            "selected_penalty_counts": {
                str(penalty): penalty_counts.get(penalty, 0)
                for penalty in protocol.residual_penalties
            },
            "actual_cell_probe_query_count": actual_queries,
            "actual_cell_probe_correct_count": actual_correct,
            "actual_cell_focal_query_count": actual_focal,
            "actual_cell_focal_correct_count": actual_focal_correct,
            "rotated_control_query_count": rotated_queries,
            "rotated_control_correct_count": rotated_correct,
            "rotated_control_focal_query_count": rotated_focal,
            "rotated_control_focal_correct_count": rotated_focal_correct,
        },
        "acceptance": {
            "required_passing_environments": protocol.required_passing_environments,
            "selected_penalty_not_used_as_gate": True,
            "no_pooled_average_rescue": True,
            "every_environment_must_pass": True,
            "campaign_passed": campaign_passed,
        },
        "claims": _open_claims(),
    }
    return _seal_body(body)


def open_campaign(
    terminal_preopen: Mapping[str, object],
    preopen_records: Sequence[Mapping[str, object]],
    protocol_path: Path | None = None,
    *,
    implementation_manifest_path: Path | None = None,
    power_control_record_path: Path | None = None,
    max_validation_replay_generated_events: int = (
        DEFAULT_MAX_TOTAL_VALIDATION_REPLAY_EVENTS
    ),
) -> dict[str, object]:
    """Validate all preopen evidence, then open all forty models as one batch."""

    protocol, _ = preflight_campaign(protocol_path)
    manifest, runtime, power = _prerequisites(
        protocol, implementation_manifest_path, power_control_record_path
    )
    terminal_body = _record_body(terminal_preopen)
    terminal_accounting = _mapping(
        "terminal generation accounting",
        terminal_body.get("generation_work_accounting"),
    )
    prior_resume_replay = _plain_int(
        "terminal prior resume replay",
        terminal_accounting.get("prior_resume_validation_replay_events"),
    )
    terminal_replay = _plain_int(
        "terminal aggregate replay",
        terminal_accounting.get("terminal_aggregate_validation_replay_events"),
        1,
    )
    cumulative_terminal_replay = _plain_int(
        "terminal cumulative replay",
        terminal_accounting.get(
            "cumulative_validation_replay_events_through_terminal_preopen"
        ),
        1,
    )
    if cumulative_terminal_replay != prior_resume_replay + terminal_replay:
        raise ValueError("terminal replay accounting is inconsistent")
    replay_cap = _plain_int(
        "max_validation_replay_generated_events",
        max_validation_replay_generated_events,
        1,
    )
    planned_open_replay = (
        len(protocol.schedule)
        * 2
        * protocol.documents_per_split
        * protocol.document_length
    )
    if (
        cumulative_terminal_replay + planned_open_replay > replay_cap
        or cumulative_terminal_replay + planned_open_replay
        > protocol.max_deterministic_replay_generated_events_total
    ):
        raise SequenceDiscoveryLimitError(
            "batch-open cumulative validation replay exceeds its budget before generation"
        )
    # This complete validation and its aggregate comparison finish before the
    # first call to _postfit_report/build_balanced_probe_suite.
    validated, replay_events = _validate_complete_preopen_records(
        preopen_records,
        protocol,
        manifest,
        runtime,
        power,
        max_validation_replay_generated_events=(
            replay_cap - cumulative_terminal_replay
        ),
    )
    expected_terminal = _build_terminal_preopen_from_validated(
        validated,
        protocol,
        manifest,
        runtime,
        power,
        replay_events=replay_events,
        prior_validation_replay_events=prior_resume_replay,
    )
    if dict(terminal_preopen) != expected_terminal:
        raise ValueError("terminal preopen aggregate does not reproduce all shards")
    acceptance = _mapping("terminal acceptance", terminal_preopen.get("acceptance"))
    if acceptance.get("outer_open_permitted") is not True:
        raise ValueError("terminal preopen gate forbids every outer probe")
    open_records = tuple(
        _build_open_environment_record(
            protocol, manifest, runtime, power, terminal_preopen, row
        )
        for row in validated
    )
    return _build_open_campaign_record(
        protocol,
        manifest,
        runtime,
        power,
        terminal_preopen,
        validated,
        open_records,
        preopen_validation_replay_events=replay_events,
    )


def validate_campaign_record(
    record: Mapping[str, object],
    terminal_preopen: Mapping[str, object],
    preopen_records: Sequence[Mapping[str, object]],
    protocol_path: Path | None = None,
    *,
    implementation_manifest_path: Path | None = None,
    power_control_record_path: Path | None = None,
    max_validation_replay_generated_events: int = (
        DEFAULT_MAX_TOTAL_VALIDATION_REPLAY_EVENTS
    ),
) -> None:
    """Replay exact preopen artifacts and every postopen probe from frozen models."""

    protocol = load_frozen_protocol(protocol_path)
    terminal_body = _record_body(terminal_preopen)
    terminal_accounting = _mapping(
        "terminal generation accounting",
        terminal_body.get("generation_work_accounting"),
    )
    terminal_cumulative = _plain_int(
        "terminal cumulative replay",
        terminal_accounting.get(
            "cumulative_validation_replay_events_through_terminal_preopen"
        ),
        1,
    )
    one_pass = (
        len(protocol.schedule)
        * 2
        * protocol.documents_per_split
        * protocol.document_length
    )
    cap = _plain_int(
        "max_validation_replay_generated_events",
        max_validation_replay_generated_events,
        1,
    )
    # One pass produced the original batch-open artifact and this call performs
    # one further independent replay.  Meter both before the validating replay.
    if (
        terminal_cumulative + 2 * one_pass > cap
        or terminal_cumulative + 2 * one_pass
        > protocol.max_deterministic_replay_generated_events_total
    ):
        raise SequenceDiscoveryLimitError(
            "postopen validation replay exceeds the cumulative campaign budget"
        )

    expected = open_campaign(
        terminal_preopen,
        preopen_records,
        protocol_path,
        implementation_manifest_path=implementation_manifest_path,
        power_control_record_path=power_control_record_path,
        max_validation_replay_generated_events=(
            max_validation_replay_generated_events
        ),
    )
    if dict(record) != expected:
        raise ValueError("open campaign does not replay from frozen preopen artifacts")


def _aggregate_preopen_records_for_test_only(
    records: Sequence[Mapping[str, object]],
    protocol_path: Path | None = None,
) -> dict[str, object]:
    """Exercise schedule/hash arithmetic; never accepts scientific evidence."""

    protocol = load_frozen_protocol(protocol_path)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("synthetic fixture records must be a finite sequence")
    if len(records) != len(protocol.schedule):
        raise ValueError("synthetic aggregation requires every frozen environment")
    indexed: dict[int, Mapping[str, object]] = {}
    for record in records:
        body = _record_body(record)
        if body.get("schema") != TEST_FIXTURE_ENVIRONMENT_SCHEMA:
            raise ValueError("test-only aggregation rejects evidence-record schemas")
        environment = _mapping("environment", body.get("environment"))
        index = _plain_int("environment_index", environment.get("environment_index"))
        if index >= len(protocol.schedule):
            raise ValueError("synthetic environment index is outside the schedule")
        _require_environment_match(environment, protocol.schedule[index])
        if index in indexed:
            raise ValueError("synthetic aggregation contains a duplicate environment")
        indexed[index] = record
    if set(indexed) != set(range(len(protocol.schedule))):
        raise ValueError("synthetic aggregation is missing an environment")
    ordered = tuple(indexed[index] for index in range(len(protocol.schedule)))
    hashes = tuple(str(record["record_sha256"]) for record in ordered)
    if len(set(hashes)) != len(hashes):
        raise ValueError("synthetic shard hashes must be unique")
    penalty_counts: Counter[int] = Counter()
    candidates = admissible = passing = mistakes = overrides = 0
    summaries: list[dict[str, object]] = []
    for index, record in enumerate(ordered):
        body = _record_body(record)
        penalty = _plain_int("selected penalty", body.get("selected_residual_penalty"))
        if penalty not in protocol.residual_penalties:
            raise ValueError("synthetic selected penalty is outside the frozen grid")
        row_candidates = _plain_int("candidate count", body.get("candidate_count"))
        row_admissible = _plain_int("admissible count", body.get("admissible_candidate_count"))
        row_mistakes = _plain_int("final mistakes", body.get("final_training_mistakes"))
        row_overrides = _plain_int("final overrides", body.get("final_local_override_count"))
        passed = body.get("environment_preopen_gate_passed") is True
        if row_admissible > row_candidates:
            raise ValueError("synthetic admissible count exceeds candidates")
        penalty_counts[penalty] += 1
        candidates += row_candidates
        admissible += row_admissible
        mistakes += row_mistakes
        overrides += row_overrides
        passing += passed
        summaries.append(
            {
                **_environment_spec_payload(protocol.schedule[index]),
                "preopen_record_sha256": record["record_sha256"],
                "selected_residual_penalty": penalty,
                "environment_preopen_gate_passed": passed,
            }
        )
    body = {
        "schema": TEST_FIXTURE_CAMPAIGN_SCHEMA,
        "scope": "synthetic_test_fixture_not_scientific_evidence",
        "protocol": _protocol_record(protocol),
        "preopen_environment_record_sha256s": list(hashes),
        "environments": summaries,
        "aggregate": {
            "environment_count": len(ordered),
            "selected_penalty_counts": {
                str(penalty): penalty_counts.get(penalty, 0)
                for penalty in protocol.residual_penalties
            },
            "fold_candidate_count": candidates,
            "admissible_fold_candidate_count": admissible,
            "total_final_training_mistakes": mistakes,
            "total_final_local_override_count": overrides,
            "passing_preopen_environment_count": passing,
        },
        "acceptance": {
            "outer_open_permitted": False,
            "scientific_evidence_permitted": False,
        },
        "claims": {
            **_terminal_preopen_claims(),
            "synthetic_test_fixture_only": True,
            "scientific_evidence_permitted": False,
        },
    }
    return _seal_body(body)


def _read_json(path: Path, *, max_record_bytes: int) -> dict[str, object]:
    if not isinstance(path, Path):
        raise TypeError("record path must be pathlib.Path")
    limit = _plain_int("max_record_bytes", max_record_bytes, 1)
    size = path.stat().st_size
    if size > limit:
        raise SequenceDiscoveryLimitError("record exceeds the read budget")
    payload = path.read_bytes()
    if len(payload) != size:
        raise ValueError("record changed while it was read")
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise TypeError("record file must contain a JSON object")
    return document


def load_preopen_environment_record(
    path: Path,
    protocol: FrozenProtocol | None = None,
    *,
    implementation_manifest_path: Path | None = None,
    power_control_record_path: Path | None = None,
    max_record_bytes: int = DEFAULT_MAX_ENVIRONMENT_RECORD_BYTES,
    max_validation_replay_generated_events: int = (
        DEFAULT_MAX_TOTAL_VALIDATION_REPLAY_EVENTS
    ),
) -> dict[str, object]:
    record = _read_json(path, max_record_bytes=max_record_bytes)
    validate_preopen_environment_record(
        record,
        protocol,
        implementation_manifest_path=implementation_manifest_path,
        power_control_record_path=power_control_record_path,
        max_validation_replay_generated_events=(
            max_validation_replay_generated_events
        ),
    )
    return record


def aggregate_preopen_record_files(
    paths: Sequence[Path],
    protocol_path: Path | None = None,
    *,
    implementation_manifest_path: Path | None = None,
    power_control_record_path: Path | None = None,
    max_record_bytes: int = DEFAULT_MAX_ENVIRONMENT_RECORD_BYTES,
    max_validation_replay_generated_events: int = (
        DEFAULT_MAX_TOTAL_VALIDATION_REPLAY_EVENTS
    ),
    prior_validation_replay_generated_events: int = 0,
) -> dict[str, object]:
    protocol = load_frozen_protocol(protocol_path)
    if len(paths) != len(protocol.schedule):
        raise ValueError("terminal aggregation requires one file per environment")
    if any(not isinstance(path, Path) for path in paths):
        raise TypeError("record paths must contain pathlib.Path values")
    limit = _plain_int("max_record_bytes", max_record_bytes, 1)
    sizes = tuple(path.stat().st_size for path in paths)
    if any(size > limit for size in sizes):
        raise SequenceDiscoveryLimitError(
            "a preopen record exceeds the read budget before parsing"
        )
    records = tuple(_read_json(path, max_record_bytes=limit) for path in paths)
    return aggregate_preopen_records(
        records,
        protocol_path,
        implementation_manifest_path=implementation_manifest_path,
        power_control_record_path=power_control_record_path,
        max_validation_replay_generated_events=(
            max_validation_replay_generated_events
        ),
        prior_validation_replay_generated_events=(
            prior_validation_replay_generated_events
        ),
    )


def _write_atomic(path: Path, payload: bytes) -> None:
    if not isinstance(path, Path) or type(payload) is not bytes:
        raise TypeError("atomic output requires pathlib.Path and exact bytes")
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


def _encoded(record: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            record,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def run_all_preopen_with_resume(
    shard_directory: Path,
    terminal_output: Path,
    protocol_path: Path | None = None,
    *,
    implementation_manifest_path: Path | None = None,
    power_control_record_path: Path | None = None,
    max_total_primary_generated_events: int = DEFAULT_MAX_TOTAL_PRIMARY_GENERATED_EVENTS,
    max_total_scored_event_work: int = DEFAULT_MAX_TOTAL_SCORED_EVENT_WORK,
    max_validation_replay_generated_events: int = (
        DEFAULT_MAX_TOTAL_VALIDATION_REPLAY_EVENTS
    ),
) -> dict[str, object]:
    """Run/resume PREOPEN only.  It cannot construct or evaluate probes."""

    if not isinstance(shard_directory, Path) or not isinstance(terminal_output, Path):
        raise TypeError("shard directory and terminal output must be pathlib.Path")
    protocol, _ = preflight_campaign(
        protocol_path,
        max_total_primary_generated_events=max_total_primary_generated_events,
        max_total_scored_event_work=max_total_scored_event_work,
    )
    replay_cap = _plain_int(
        "max_validation_replay_generated_events",
        max_validation_replay_generated_events,
        1,
    )
    # Validate global prerequisites before inspecting/running any shard.
    _prerequisites(protocol, implementation_manifest_path, power_control_record_path)
    shard_directory.mkdir(parents=True, exist_ok=True)
    paths = tuple(
        shard_directory / f"preopen-environment-{spec.environment_index:03d}.json"
        for spec in protocol.schedule
    )
    existing_paths = tuple(path for path in paths if path.exists())
    prior_resume_replay = (
        len(existing_paths)
        * 2
        * protocol.documents_per_split
        * protocol.document_length
    )
    terminal_replay = (
        len(protocol.schedule)
        * 2
        * protocol.documents_per_split
        * protocol.document_length
    )
    if (
        prior_resume_replay + terminal_replay
        > replay_cap
        or prior_resume_replay + terminal_replay
        > protocol.max_deterministic_replay_generated_events_total
    ):
        raise SequenceDiscoveryLimitError(
            "resume plus terminal replay exceeds the campaign budget before work"
        )
    if existing_paths:
        # One bounded resume-audit pass before any new fit.
        _replay_preflight(
            protocol,
            len(existing_paths),
            replay_cap,
        )
        for path in existing_paths:
            load_preopen_environment_record(
                path,
                protocol,
                implementation_manifest_path=implementation_manifest_path,
                power_control_record_path=power_control_record_path,
                max_validation_replay_generated_events=(
                    replay_cap
                ),
            )
    for spec, path in zip(protocol.schedule, paths, strict=True):
        if path.exists():
            continue
        record = build_preopen_environment_record(
            spec.environment_index,
            protocol_path,
            implementation_manifest_path=implementation_manifest_path,
            power_control_record_path=power_control_record_path,
            max_total_primary_generated_events=max_total_primary_generated_events,
            max_total_scored_event_work=max_total_scored_event_work,
        )
        _write_atomic(path, _encoded(record))
    terminal = aggregate_preopen_record_files(
        paths,
        protocol_path,
        implementation_manifest_path=implementation_manifest_path,
        power_control_record_path=power_control_record_path,
        max_validation_replay_generated_events=(
            replay_cap
        ),
        prior_validation_replay_generated_events=prior_resume_replay,
    )
    _write_atomic(terminal_output, _encoded(terminal))
    return terminal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=default_protocol_path())
    parser.add_argument(
        "--implementation-manifest",
        type=Path,
        default=default_implementation_manifest_path(),
    )
    parser.add_argument(
        "--power-control-record",
        type=Path,
        default=default_power_control_record_path(),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    shard = commands.add_parser(
        "preopen-environment", help="fit/select one shard without opening probes"
    )
    shard.add_argument("--environment-index", type=int, required=True)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument(
        "--max-total-primary-generated-events",
        type=int,
        default=DEFAULT_MAX_TOTAL_PRIMARY_GENERATED_EVENTS,
    )
    shard.add_argument(
        "--max-total-scored-event-work",
        type=int,
        default=DEFAULT_MAX_TOTAL_SCORED_EVENT_WORK,
    )

    aggregate = commands.add_parser(
        "aggregate-preopen", help="validate all forty shards and write terminal gate"
    )
    aggregate.add_argument("--records", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument(
        "--max-validation-replay-generated-events",
        type=int,
        default=DEFAULT_MAX_TOTAL_VALIDATION_REPLAY_EVENTS,
    )

    opening = commands.add_parser(
        "open", help="batch-open all forty only after terminal preopen validation"
    )
    opening.add_argument("--terminal-preopen", type=Path, required=True)
    opening.add_argument("--records", type=Path, nargs="+", required=True)
    opening.add_argument("--output", type=Path, required=True)
    opening.add_argument(
        "--max-validation-replay-generated-events",
        type=int,
        default=DEFAULT_MAX_TOTAL_VALIDATION_REPLAY_EVENTS,
    )

    all_preopen = commands.add_parser(
        "preopen-all", help="run/resume all preopen shards; never opens probes"
    )
    all_preopen.add_argument("--shard-directory", type=Path, required=True)
    all_preopen.add_argument("--output", type=Path, required=True)
    all_preopen.add_argument(
        "--max-total-primary-generated-events",
        type=int,
        default=DEFAULT_MAX_TOTAL_PRIMARY_GENERATED_EVENTS,
    )
    all_preopen.add_argument(
        "--max-total-scored-event-work",
        type=int,
        default=DEFAULT_MAX_TOTAL_SCORED_EVENT_WORK,
    )
    all_preopen.add_argument(
        "--max-validation-replay-generated-events",
        type=int,
        default=DEFAULT_MAX_TOTAL_VALIDATION_REPLAY_EVENTS,
    )

    arguments = parser.parse_args(argv)
    common = {
        "implementation_manifest_path": arguments.implementation_manifest,
        "power_control_record_path": arguments.power_control_record,
    }
    try:
        if arguments.command == "preopen-environment":
            record = build_preopen_environment_record(
                arguments.environment_index,
                arguments.protocol,
                **common,
                max_total_primary_generated_events=(
                    arguments.max_total_primary_generated_events
                ),
                max_total_scored_event_work=arguments.max_total_scored_event_work,
            )
            _write_atomic(arguments.output, _encoded(record))
        elif arguments.command == "aggregate-preopen":
            record = aggregate_preopen_record_files(
                arguments.records,
                arguments.protocol,
                **common,
                max_validation_replay_generated_events=(
                    arguments.max_validation_replay_generated_events
                ),
            )
            _write_atomic(arguments.output, _encoded(record))
        elif arguments.command == "open":
            terminal = _read_json(
                arguments.terminal_preopen,
                max_record_bytes=DEFAULT_MAX_ENVIRONMENT_RECORD_BYTES,
            )
            preopen_records = tuple(
                _read_json(path, max_record_bytes=DEFAULT_MAX_ENVIRONMENT_RECORD_BYTES)
                for path in arguments.records
            )
            record = open_campaign(
                terminal,
                preopen_records,
                arguments.protocol,
                **common,
                max_validation_replay_generated_events=(
                    arguments.max_validation_replay_generated_events
                ),
            )
            _write_atomic(arguments.output, _encoded(record))
        else:
            run_all_preopen_with_resume(
                arguments.shard_directory,
                arguments.output,
                arguments.protocol,
                **common,
                max_total_primary_generated_events=(
                    arguments.max_total_primary_generated_events
                ),
                max_total_scored_event_work=arguments.max_total_scored_event_work,
                max_validation_replay_generated_events=(
                    arguments.max_validation_replay_generated_events
                ),
            )
    except (OSError, TypeError, ValueError, SequenceDiscoveryLimitError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
