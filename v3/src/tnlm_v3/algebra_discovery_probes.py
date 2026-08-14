"""Balanced post-fit probes for sequence-only algebra discovery.

The learner-facing object in this module is :class:`VisibleProbeProgram`.  It
contains only the same raw event fields available in the binding benchmark;
it never contains a target, probe family, held-out pair, equivalence-group
label, or oracle state.  All of those fields live in a separate sealed case
which is retained by the trusted evaluator.

This is a *retrospective protocol rehearsal*: the repository's nominal
``(0, 0)`` answer has already been inspected during earlier exact-algebra
work.  The closed protocol-status enum deliberately cannot assert that this
historical run was prospectively sealed.  The machinery remains generic over
arbitrary cells so that a genuinely unopened outer rotation can use the same
evaluator later.

Expected answers are produced by a small trusted interpreter in the probe
builder.  The interpreter is not imported by, and is never passed to, the
predictor.  In particular this module does not import ``exact_algebra``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Protocol, Sequence, runtime_checkable

from .data import BindingEventKind


DEFAULT_MAX_CELL_ROTATIONS = 256
DEFAULT_MAX_PROBE_CASES = 10_000
DEFAULT_MAX_PROBE_EVENTS = 1_000_000
DEFAULT_MAX_PROBE_WORK = 2_000_000
_CASES_PER_CELL = 15


class ProbeBudgetExceededError(RuntimeError):
    """Raised before a requested probe construction exceeds its work budget."""


def _plain_int(name: str, value: int, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _pair(name: str, value: tuple[int, int], keys: int, values: int) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a two-integer tuple")
    key, symbol = value
    _plain_int(f"{name} key", key)
    _plain_int(f"{name} value", symbol)
    if key >= keys or symbol >= values:
        raise ValueError(f"{name} is outside the declared vocabulary")


class ProbeProtocolStatus(str, Enum):
    """The only historically honest status for the current repository."""

    RETROSPECTIVE_PROTOCOL_REHEARSAL = "retrospective_protocol_rehearsal"


class ProbeFamily(str, Enum):
    FIRST_ENTRY_BIND = "first_entry_bind"
    FIRST_ENTRY_UPDATE = "first_entry_update"
    FIRST_ENTRY_COPY = "first_entry_copy"
    UPDATE_OUT = "update_out"
    COPY_OUT_NATURALITY = "copy_out_naturality"
    REPEATED_QUERY_IDENTITY = "repeated_query_identity"
    CYCLIC_COMPOSITION = "cyclic_composition"
    INVALIDATE_REBIND = "invalidate_rebind"
    INDEPENDENT_COMMUTATION = "independent_key_commutation"
    DISTRACTOR_INTERLEAVING = "distractor_interleaving"
    LONG_COMPOSITION = "long_composition"
    COPY_ORDER_NONCOMMUTATION = "copy_order_noncommutation"


class ProbeQueryRole(str, Enum):
    """Trusted-only attribution; roles are never visible to a predictor."""

    FOCAL = "focal"
    BALANCE_CONTROL = "balance_control"


class ProbePathRelation(str, Enum):
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"


@dataclass(frozen=True, order=True)
class ProbeCellRotation:
    """Attribution for independently generated balanced destination cells.

    The key permutation and cyclic value map identify which actual probe cell
    corresponds to a declared base cell.  They do not transform the program or
    its outputs.  Consequently this metadata does not certify program-level
    conjugacy, output equivariance, or any metamorphic relation between cases.
    """

    key_permutation: tuple[int, ...]
    value_offset: int
    value_permutation: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key_permutation, tuple) or len(
            self.key_permutation
        ) < 2:
            raise TypeError("key_permutation must contain at least two keys")
        if any(type(value) is not int for value in self.key_permutation):
            raise TypeError("key_permutation entries must be exact integers")
        if tuple(sorted(self.key_permutation)) != tuple(
            range(len(self.key_permutation))
        ):
            raise ValueError("key_permutation must be a bijection")
        _plain_int("value_offset", self.value_offset)
        if not isinstance(self.value_permutation, tuple) or len(
            self.value_permutation
        ) < 2:
            raise TypeError("value_permutation must contain at least two values")
        if any(type(value) is not int for value in self.value_permutation):
            raise TypeError("value_permutation entries must be exact integers")
        cardinality = len(self.value_permutation)
        if self.value_offset >= cardinality:
            raise ValueError("value_offset is outside the value vocabulary")
        expected = tuple(
            (value + self.value_offset) % cardinality
            for value in range(cardinality)
        )
        if self.value_permutation != expected:
            raise ValueError(
                "value_permutation must be the cyclic translation by value_offset"
            )

    @classmethod
    def identity(cls, keys: int, cardinality: int) -> "ProbeCellRotation":
        _plain_int("keys", keys, 2)
        _plain_int("cardinality", cardinality, 2)
        return cls(tuple(range(keys)), 0, tuple(range(cardinality)))

    def apply_pair(self, pair: tuple[int, int]) -> tuple[int, int]:
        key, value = pair
        return self.key_permutation[key], self.value_permutation[value]

    @property
    def label(self) -> str:
        key_text = "-".join(str(value) for value in self.key_permutation)
        return f"keys-{key_text}/value-offset-{self.value_offset}"


def cyclic_cell_rotation_inventory(
    num_surface_keys: int,
    value_cardinality: int,
    *,
    anchor_key: int = 0,
    max_cell_rotations: int = DEFAULT_MAX_CELL_ROTATIONS,
) -> tuple[ProbeCellRotation, ...]:
    """Return KxV cell rotations taking one anchor cell through the grid."""

    keys = _plain_int("num_surface_keys", num_surface_keys, 2)
    cardinality = _plain_int("value_cardinality", value_cardinality, 2)
    anchor = _plain_int("anchor_key", anchor_key)
    limit = _plain_int("max_cell_rotations", max_cell_rotations, 1)
    if anchor >= keys:
        raise ValueError("anchor_key is outside the key vocabulary")
    if keys * cardinality > limit:
        raise ProbeBudgetExceededError(
            "cyclic cell rotation inventory exceeds max_cell_rotations"
        )
    result: list[ProbeCellRotation] = []
    for target_key in range(keys):
        permutation = list(range(keys))
        permutation[anchor], permutation[target_key] = (
            permutation[target_key],
            permutation[anchor],
        )
        for offset in range(cardinality):
            result.append(
                ProbeCellRotation(
                    key_permutation=tuple(permutation),
                    value_offset=offset,
                    value_permutation=tuple(
                        (value + offset) % cardinality
                        for value in range(cardinality)
                    ),
                )
            )
    return tuple(result)


@dataclass(frozen=True, order=True)
class VisibleProbeEvent:
    """One target-free, raw zero-based event visible to a predictor."""

    kind: BindingEventKind
    primary_key: int = -1
    secondary_key: int = -1
    argument: int = -1

    def __post_init__(self) -> None:
        if type(self.kind) is not BindingEventKind:
            raise TypeError("kind must be exact BindingEventKind")
        for name in ("primary_key", "secondary_key", "argument"):
            _plain_int(name, getattr(self, name), -1)
        key = self.primary_key
        source = self.secondary_key
        argument = self.argument
        if self.kind in (BindingEventKind.BIND, BindingEventKind.UPDATE):
            valid = key >= 0 and source == -1 and argument >= 0
        elif self.kind is BindingEventKind.COPY:
            valid = key >= 0 and source >= 0 and key != source and argument == -1
        elif self.kind in (BindingEventKind.INVALIDATE, BindingEventKind.QUERY):
            valid = key >= 0 and source == -1 and argument == -1
        elif self.kind is BindingEventKind.DISTRACTOR:
            valid = key == source == -1 and argument in (0, 1)
        else:
            valid = False
        if not valid:
            raise ValueError("event fields do not match the visible event grammar")

    @classmethod
    def bind(cls, key: int, value: int) -> "VisibleProbeEvent":
        return cls(BindingEventKind.BIND, primary_key=key, argument=value)

    @classmethod
    def update(cls, key: int, transform: int) -> "VisibleProbeEvent":
        return cls(BindingEventKind.UPDATE, primary_key=key, argument=transform)

    @classmethod
    def copy(cls, destination: int, source: int) -> "VisibleProbeEvent":
        return cls(
            BindingEventKind.COPY,
            primary_key=destination,
            secondary_key=source,
        )

    @classmethod
    def invalidate(cls, key: int) -> "VisibleProbeEvent":
        return cls(BindingEventKind.INVALIDATE, primary_key=key)

    @classmethod
    def query(cls, key: int) -> "VisibleProbeEvent":
        return cls(BindingEventKind.QUERY, primary_key=key)

    @classmethod
    def distractor(cls, scope: int) -> "VisibleProbeEvent":
        return cls(BindingEventKind.DISTRACTOR, argument=scope)


@dataclass(frozen=True)
class VisibleProbeProgram:
    """The complete and exclusive object passed to ``predict_queries``."""

    num_surface_keys: int
    value_cardinality: int
    events: tuple[VisibleProbeEvent, ...]
    schema: str = "tnlm-v3-visible-algebra-probe-v1"

    def __post_init__(self) -> None:
        if self.schema != "tnlm-v3-visible-algebra-probe-v1":
            raise ValueError("unknown visible-probe schema")
        _plain_int("num_surface_keys", self.num_surface_keys, 2)
        _plain_int("value_cardinality", self.value_cardinality, 2)
        if not isinstance(self.events, tuple) or not self.events:
            raise TypeError("events must be a nonempty tuple")
        if any(type(event) is not VisibleProbeEvent for event in self.events):
            raise TypeError("events must contain exact VisibleProbeEvent values")
        for event in self.events:
            if event.primary_key >= self.num_surface_keys:
                raise ValueError("primary key is outside the program vocabulary")
            if event.secondary_key >= self.num_surface_keys:
                raise ValueError("secondary key is outside the program vocabulary")
            if event.argument >= self.value_cardinality:
                raise ValueError("argument is outside the program vocabulary")
        if not any(event.kind is BindingEventKind.QUERY for event in self.events):
            raise ValueError("a probe program must contain at least one query")

    @property
    def query_count(self) -> int:
        return sum(event.kind is BindingEventKind.QUERY for event in self.events)

    @property
    def query_targets(self) -> tuple[None, ...]:
        """Compatibility view for sequence learners; it contains no labels."""

        return (None,) * len(self.events)


@dataclass(frozen=True)
class ProbeSupportAudit:
    """Trusted certificate covering every live pre/post-state cell."""

    touched_pairs: tuple[tuple[int, int], ...]
    forbidden_pairs: tuple[tuple[int, int], ...]
    forbidden_intersection: tuple[tuple[int, int], ...]
    every_intermediate_state_audited: bool

    def __post_init__(self) -> None:
        for name in ("touched_pairs", "forbidden_pairs", "forbidden_intersection"):
            rows = getattr(self, name)
            if not isinstance(rows, tuple):
                raise TypeError(f"{name} must be a tuple")
            if tuple(sorted(set(rows))) != rows:
                raise ValueError(f"{name} must be sorted and unique")
            for row in rows:
                if not isinstance(row, tuple) or len(row) != 2:
                    raise TypeError(f"{name} must contain pairs")
                _plain_int(f"{name} key", row[0])
                _plain_int(f"{name} value", row[1])
        expected = tuple(sorted(set(self.touched_pairs) & set(self.forbidden_pairs)))
        if self.forbidden_intersection != expected:
            raise ValueError("forbidden intersection disagrees with support sets")
        if type(self.every_intermediate_state_audited) is not bool:
            raise TypeError("every_intermediate_state_audited must be exact bool")
        if not self.every_intermediate_state_audited:
            raise ValueError("support audit must cover every intermediate state")

    @property
    def passed(self) -> bool:
        return not self.forbidden_intersection


@dataclass(frozen=True)
class SealedProbeCase:
    """Trusted case wrapper which must never be handed to a predictor."""

    case_id: str
    family: ProbeFamily
    base_probe_pair: tuple[int, int]
    probe_pair: tuple[int, int]
    cell_rotation: ProbeCellRotation
    program: VisibleProbeProgram
    expected_answers: tuple[int, ...]
    query_roles: tuple[ProbeQueryRole, ...]
    support_audit: ProbeSupportAudit
    probe_entry_event_index: int
    probe_entry_kind: BindingEventKind
    equivalence_group: str | None = None
    path_relation: ProbePathRelation | None = None

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id:
            raise ValueError("case_id must be a nonempty exact string")
        if type(self.family) is not ProbeFamily:
            raise TypeError("family must be exact ProbeFamily")
        _pair(
            "base_probe_pair",
            self.base_probe_pair,
            self.program.num_surface_keys,
            self.program.value_cardinality,
        )
        _pair(
            "probe_pair",
            self.probe_pair,
            self.program.num_surface_keys,
            self.program.value_cardinality,
        )
        if type(self.cell_rotation) is not ProbeCellRotation:
            raise TypeError("cell_rotation must be exact ProbeCellRotation")
        if len(self.cell_rotation.key_permutation) != self.program.num_surface_keys:
            raise ValueError("cell_rotation key vocabulary disagrees with program")
        if len(self.cell_rotation.value_permutation) != self.program.value_cardinality:
            raise ValueError("cell_rotation value vocabulary disagrees with program")
        if self.cell_rotation.apply_pair(self.base_probe_pair) != self.probe_pair:
            raise ValueError("cell_rotation does not map base pair to probe pair")
        if not isinstance(self.expected_answers, tuple):
            raise TypeError("expected_answers must be a tuple")
        if len(self.expected_answers) != self.program.query_count:
            raise ValueError("expected answers must align with visible queries")
        for answer in self.expected_answers:
            _plain_int("expected answer", answer)
            if answer >= self.program.value_cardinality:
                raise ValueError("expected answer is outside the value vocabulary")
        if not isinstance(self.query_roles, tuple) or len(self.query_roles) != len(
            self.expected_answers
        ):
            raise TypeError("query_roles must align with expected_answers")
        if any(type(role) is not ProbeQueryRole for role in self.query_roles):
            raise TypeError("query roles must be exact ProbeQueryRole values")
        if ProbeQueryRole.FOCAL not in self.query_roles:
            raise ValueError("every probe case must contain a focal query")
        if type(self.support_audit) is not ProbeSupportAudit:
            raise TypeError("support_audit must be exact ProbeSupportAudit")
        if not self.support_audit.passed:
            raise ValueError("a sealed probe case may not touch forbidden support")
        if self.probe_pair not in self.support_audit.touched_pairs:
            raise ValueError("probe program never enters its declared probe cell")
        _plain_int("probe_entry_event_index", self.probe_entry_event_index)
        if self.probe_entry_event_index >= len(self.program.events):
            raise ValueError("probe entry index is outside the visible program")
        if type(self.probe_entry_kind) is not BindingEventKind:
            raise TypeError("probe_entry_kind must be exact BindingEventKind")
        entry_index, entry_kind = _trusted_entry(self.program, self.probe_pair)
        if (self.probe_entry_event_index, self.probe_entry_kind) != (
            entry_index,
            entry_kind,
        ):
            raise ValueError("probe entry attribution disagrees with program trace")
        expected_entry = {
            ProbeFamily.FIRST_ENTRY_BIND: BindingEventKind.BIND,
            ProbeFamily.FIRST_ENTRY_UPDATE: BindingEventKind.UPDATE,
            ProbeFamily.FIRST_ENTRY_COPY: BindingEventKind.COPY,
        }.get(self.family)
        if expected_entry is not None and self.probe_entry_kind is not expected_entry:
            raise ValueError("first-entry family disagrees with causal entry event")
        trusted_answers, trusted_audit, _ = _trusted_execute(
            self.program, self.support_audit.forbidden_pairs
        )
        if self.expected_answers != trusted_answers:
            raise ValueError("expected answers disagree with trusted program execution")
        if self.support_audit != trusted_audit:
            raise ValueError("support audit disagrees with trusted program execution")
        if self.equivalence_group is not None and (
            type(self.equivalence_group) is not str or not self.equivalence_group
        ):
            raise ValueError("equivalence_group must be None or a nonempty string")
        if (self.equivalence_group is None) != (self.path_relation is None):
            raise ValueError(
                "path relation and group must either both be set or absent"
            )
        if self.path_relation is not None and type(
            self.path_relation
        ) is not ProbePathRelation:
            raise TypeError("path_relation must be exact ProbePathRelation or None")


@dataclass(frozen=True)
class ProbeBalanceCertificate:
    """Exact per-family and per-rotation shortcut-control counts."""

    family_class_counts: tuple[tuple[ProbeFamily, tuple[int, ...]], ...]
    family_focal_class_counts: tuple[tuple[ProbeFamily, tuple[int, ...]], ...]
    pair_case_counts: tuple[tuple[tuple[int, int], int], ...]
    pair_focal_query_counts: tuple[tuple[tuple[int, int], int], ...]
    every_family_output_balanced: bool
    every_family_focal_output_balanced: bool
    rotated_cells_equally_weighted: bool

    def __post_init__(self) -> None:
        for name, require_positive in (
            ("family_class_counts", True),
            ("family_focal_class_counts", False),
        ):
            rows = getattr(self, name)
            if not isinstance(rows, tuple) or not rows:
                raise TypeError(f"{name} must be a nonempty tuple")
            families: list[ProbeFamily] = []
            for family, counts in rows:
                if type(family) is not ProbeFamily:
                    raise TypeError("family count keys must be exact ProbeFamily")
                if not isinstance(counts, tuple) or len(counts) < 2:
                    raise TypeError("family class counts need at least two classes")
                minimum = 1 if require_positive else 0
                if any(type(count) is not int or count < minimum for count in counts):
                    raise ValueError("family class counts contain an invalid count")
                if not any(counts):
                    raise ValueError("every family must contain focal queries")
                families.append(family)
            if tuple(families) != tuple(ProbeFamily):
                raise ValueError("family counts must cover ProbeFamily in enum order")
        if type(self.every_family_output_balanced) is not bool or type(
            self.every_family_focal_output_balanced
        ) is not bool:
            raise TypeError("balance flags must be exact booleans")
        computed_balance = all(
            len(set(counts)) == 1 for _, counts in self.family_class_counts
        )
        if self.every_family_output_balanced != computed_balance:
            raise ValueError("family output-balance claim disagrees with counts")
        if not self.every_family_output_balanced:
            raise ValueError("every probe family must be output-balanced")
        computed_focal_balance = all(
            len(set(counts)) == 1 for _, counts in self.family_focal_class_counts
        )
        if self.every_family_focal_output_balanced != computed_focal_balance:
            raise ValueError("family focal-balance claim disagrees with counts")
        for name in ("pair_case_counts", "pair_focal_query_counts"):
            rows = getattr(self, name)
            if not isinstance(rows, tuple) or not rows:
                raise TypeError(f"{name} must be a nonempty tuple")
            pairs = tuple(pair for pair, _ in rows)
            if pairs != tuple(sorted(set(pairs))):
                raise ValueError(f"{name} pair keys must be sorted and unique")
            if any(type(count) is not int or count < 1 for _, count in rows):
                raise ValueError(f"{name} counts must be positive exact integers")
        if tuple(pair for pair, _ in self.pair_case_counts) != tuple(
            pair for pair, _ in self.pair_focal_query_counts
        ):
            raise ValueError("pair count inventories disagree")
        if type(self.rotated_cells_equally_weighted) is not bool:
            raise TypeError("rotated_cells_equally_weighted must be exact bool")
        case_values = tuple(count for _, count in self.pair_case_counts)
        focal_values = tuple(count for _, count in self.pair_focal_query_counts)
        computed_equal = len(set(case_values)) == len(set(focal_values)) == 1
        if self.rotated_cells_equally_weighted != computed_equal:
            raise ValueError("rotated-cell balance claim disagrees with counts")
        if not self.rotated_cells_equally_weighted:
            raise ValueError("rotated probe cells must receive equal weight")


@dataclass(frozen=True)
class SealedProbeSuite:
    protocol_status: ProbeProtocolStatus
    num_surface_keys: int
    value_cardinality: int
    base_probe_pairs: tuple[tuple[int, int], ...]
    cell_rotations: tuple[ProbeCellRotation, ...]
    probe_pairs: tuple[tuple[int, int], ...]
    forbidden_pairs: tuple[tuple[int, int], ...]
    cases: tuple[SealedProbeCase, ...]
    balance: ProbeBalanceCertificate
    scoring_manifest_sha256: str
    suite_sha256: str
    schema: str = "tnlm-v3-balanced-sealed-probes-v1"

    def __post_init__(self) -> None:
        if self.schema != "tnlm-v3-balanced-sealed-probes-v1":
            raise ValueError("unknown sealed-probe suite schema")
        if type(self.protocol_status) is not ProbeProtocolStatus:
            raise TypeError("protocol_status must be exact ProbeProtocolStatus")
        _plain_int("num_surface_keys", self.num_surface_keys, 2)
        _plain_int("value_cardinality", self.value_cardinality, 2)
        for name in ("base_probe_pairs", "probe_pairs", "forbidden_pairs"):
            rows = getattr(self, name)
            if not isinstance(rows, tuple) or tuple(sorted(set(rows))) != rows:
                raise ValueError(f"{name} must be a sorted unique tuple")
            for row in rows:
                _pair(name, row, self.num_surface_keys, self.value_cardinality)
        if not self.probe_pairs:
            raise ValueError("probe_pairs may not be empty")
        if not self.base_probe_pairs:
            raise ValueError("base_probe_pairs may not be empty")
        if not isinstance(self.cell_rotations, tuple) or not self.cell_rotations:
            raise TypeError("cell_rotations must be a nonempty tuple")
        if any(type(row) is not ProbeCellRotation for row in self.cell_rotations):
            raise TypeError(
                "cell_rotations must contain exact ProbeCellRotation values"
            )
        if len(set(self.cell_rotations)) != len(self.cell_rotations):
            raise ValueError("cell_rotations must be unique")
        if any(
            len(row.key_permutation) != self.num_surface_keys
            or len(row.value_permutation) != self.value_cardinality
            for row in self.cell_rotations
        ):
            raise ValueError("cell_rotation vocabularies disagree with suite")
        mapped_pairs = tuple(
            sorted(
                cell_rotation.apply_pair(pair)
                for pair in self.base_probe_pairs
                for cell_rotation in self.cell_rotations
            )
        )
        if len(mapped_pairs) != len(set(mapped_pairs)):
            raise ValueError(
                "base-pair/cell_rotation products must map to unique cells"
            )
        if mapped_pairs != self.probe_pairs:
            raise ValueError("cell_rotation inventory does not produce probe_pairs")
        if set(self.probe_pairs) & set(self.forbidden_pairs):
            raise ValueError("probe and forbidden pairs must be disjoint")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise TypeError("cases must be a nonempty tuple")
        if any(type(case) is not SealedProbeCase for case in self.cases):
            raise TypeError("cases must contain exact SealedProbeCase values")
        ids = tuple(case.case_id for case in self.cases)
        if len(ids) != len(set(ids)):
            raise ValueError("case IDs must be unique")
        if {case.probe_pair for case in self.cases} != set(self.probe_pairs):
            raise ValueError("case probe-pair inventory disagrees with suite")
        allowed_triplets = {
            (base, cell_rotation.apply_pair(base), cell_rotation)
            for base in self.base_probe_pairs
            for cell_rotation in self.cell_rotations
        }
        if any(
            (case.base_probe_pair, case.probe_pair, case.cell_rotation)
            not in allowed_triplets
            for case in self.cases
        ):
            raise ValueError("case cell_rotation attribution disagrees with suite")
        if any(
            case.support_audit.forbidden_pairs != self.forbidden_pairs
            for case in self.cases
        ):
            raise ValueError("case forbidden-pair audit disagrees with suite")
        if any(
            event.kind is BindingEventKind.UPDATE
            and event.argument == self.value_cardinality - 1
            for case in self.cases
            for event in case.program.events
        ):
            raise ValueError(
                "probe suite may not use generator-unsupported identity UPDATE"
            )
        if type(self.balance) is not ProbeBalanceCertificate:
            raise TypeError("balance must be exact ProbeBalanceCertificate")
        for name in ("scoring_manifest_sha256", "suite_sha256"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256")
        expected_manifest = _scoring_manifest_sha256_from_cases(self.cases)
        if self.scoring_manifest_sha256 != expected_manifest:
            raise ValueError("scoring manifest does not match sealed cases")
        recomputed_balance = _balance_certificate(
            self.cases, self.value_cardinality, self.probe_pairs
        )
        if self.balance != recomputed_balance:
            raise ValueError("balance certificate does not match sealed cases")
        payload = _suite_digest_payload(
            protocol_status=self.protocol_status,
            keys=self.num_surface_keys,
            cardinality=self.value_cardinality,
            base_probe_pairs=self.base_probe_pairs,
            cell_rotations=self.cell_rotations,
            probe_pairs=self.probe_pairs,
            forbidden_pairs=self.forbidden_pairs,
            cases=self.cases,
            balance=self.balance,
            scoring_manifest_sha256=self.scoring_manifest_sha256,
        )
        expected_digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if self.suite_sha256 != expected_digest:
            raise ValueError("suite_sha256 does not match the canonical suite payload")


@runtime_checkable
class SequenceProbePredictor(Protocol):
    """Structural post-fit predictor API used by the trusted evaluator."""

    def predict_queries(self, program: VisibleProbeProgram) -> Sequence[int]: ...


@dataclass(frozen=True)
class ProbeCaseResult:
    case_id: str
    family: ProbeFamily
    base_probe_pair: tuple[int, int]
    probe_pair: tuple[int, int]
    cell_rotation: ProbeCellRotation
    value_cardinality: int
    expected_answers: tuple[int, ...]
    predicted_answers: tuple[int, ...]
    query_roles: tuple[ProbeQueryRole, ...]
    equivalence_group: str | None
    path_relation: ProbePathRelation | None

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id:
            raise ValueError("case result ID must be a nonempty exact string")
        if type(self.family) is not ProbeFamily:
            raise TypeError("case result family must be exact ProbeFamily")
        for name in ("base_probe_pair", "probe_pair"):
            pair = getattr(self, name)
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(f"case result {name} must be an integer pair")
            if any(type(value) is not int or value < 0 for value in pair):
                raise ValueError(f"case result {name} must be nonnegative integers")
        if type(self.cell_rotation) is not ProbeCellRotation:
            raise TypeError("case result cell_rotation must be exact ProbeCellRotation")
        cardinality = _plain_int("value_cardinality", self.value_cardinality, 2)
        if len(self.cell_rotation.value_permutation) != cardinality:
            raise ValueError("case result cell_rotation has wrong value vocabulary")
        if (
            self.base_probe_pair[0] >= len(self.cell_rotation.key_permutation)
            or self.base_probe_pair[1] >= cardinality
            or self.probe_pair[0] >= len(self.cell_rotation.key_permutation)
            or self.probe_pair[1] >= cardinality
        ):
            raise ValueError("case result probe pair is outside its vocabulary")
        if self.cell_rotation.apply_pair(self.base_probe_pair) != self.probe_pair:
            raise ValueError("case result cell_rotation does not map its probe pair")
        if not isinstance(self.expected_answers, tuple) or not isinstance(
            self.predicted_answers, tuple
        ):
            raise TypeError("case result answers must be tuples")
        if len(self.expected_answers) != len(self.predicted_answers) or not (
            self.expected_answers
        ):
            raise ValueError("case result answer tuples must be nonempty and aligned")
        if not isinstance(self.query_roles, tuple) or len(self.query_roles) != len(
            self.expected_answers
        ):
            raise ValueError("case result query roles must align with answers")
        if any(
            type(value) is not int or not 0 <= value < cardinality
            for value in self.expected_answers
        ):
            raise ValueError("expected answers must be in the value vocabulary")
        if any(
            type(value) is not int or not 0 <= value < cardinality
            for value in self.predicted_answers
        ):
            raise ValueError("predicted answers must be in the value vocabulary")
        if any(type(role) is not ProbeQueryRole for role in self.query_roles):
            raise TypeError("case result roles must be exact ProbeQueryRole values")
        if ProbeQueryRole.FOCAL not in self.query_roles:
            raise ValueError("case result must contain a focal query")
        if (self.equivalence_group is None) != (self.path_relation is None):
            raise ValueError("case result path group and relation must align")
        if self.equivalence_group is not None and (
            type(self.equivalence_group) is not str or not self.equivalence_group
        ):
            raise ValueError("case result path group must be nonempty or None")
        if self.path_relation is not None and type(
            self.path_relation
        ) is not ProbePathRelation:
            raise TypeError("case result path relation has the wrong type")

    @property
    def correct_count(self) -> int:
        return sum(
            a == b for a, b in zip(self.expected_answers, self.predicted_answers)
        )

    @property
    def focal_correct_count(self) -> int:
        return sum(
            expected == predicted
            for expected, predicted, role in zip(
                self.expected_answers, self.predicted_answers, self.query_roles
            )
            if role is ProbeQueryRole.FOCAL
        )

    @property
    def focal_query_count(self) -> int:
        return self.query_roles.count(ProbeQueryRole.FOCAL)

    @property
    def exact_match(self) -> bool:
        return self.expected_answers == self.predicted_answers


def _cell_rotation_payload(row: ProbeCellRotation) -> dict[str, object]:
    return {
        "key_permutation": row.key_permutation,
        "value_offset": row.value_offset,
        "value_permutation": row.value_permutation,
    }


def _scoring_manifest_row(
    *,
    case_id: str,
    family: ProbeFamily,
    base_probe_pair: tuple[int, int],
    probe_pair: tuple[int, int],
    cell_rotation: ProbeCellRotation,
    value_cardinality: int,
    expected_answers: tuple[int, ...],
    query_roles: tuple[ProbeQueryRole, ...],
    equivalence_group: str | None,
    path_relation: ProbePathRelation | None,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "family": family.value,
        "base_probe_pair": base_probe_pair,
        "probe_pair": probe_pair,
        "cell_rotation": _cell_rotation_payload(cell_rotation),
        "value_cardinality": value_cardinality,
        "expected_answers": expected_answers,
        "query_roles": tuple(role.value for role in query_roles),
        "equivalence_group": equivalence_group,
        "path_relation": None if path_relation is None else path_relation.value,
    }


def _scoring_manifest_sha256(rows: Sequence[dict[str, object]]) -> str:
    payload = {
        "schema": "tnlm-v3-probe-scoring-manifest-v1",
        "cases": tuple(rows),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _scoring_manifest_sha256_from_cases(
    cases: tuple[SealedProbeCase, ...],
) -> str:
    return _scoring_manifest_sha256(
        tuple(
            _scoring_manifest_row(
                case_id=case.case_id,
                family=case.family,
                base_probe_pair=case.base_probe_pair,
                probe_pair=case.probe_pair,
                cell_rotation=case.cell_rotation,
                value_cardinality=case.program.value_cardinality,
                expected_answers=case.expected_answers,
                query_roles=case.query_roles,
                equivalence_group=case.equivalence_group,
                path_relation=case.path_relation,
            )
            for case in cases
        )
    )


def _scoring_manifest_sha256_from_results(
    rows: tuple[ProbeCaseResult, ...],
) -> str:
    return _scoring_manifest_sha256(
        tuple(
            _scoring_manifest_row(
                case_id=row.case_id,
                family=row.family,
                base_probe_pair=row.base_probe_pair,
                probe_pair=row.probe_pair,
                cell_rotation=row.cell_rotation,
                value_cardinality=row.value_cardinality,
                expected_answers=row.expected_answers,
                query_roles=row.query_roles,
                equivalence_group=row.equivalence_group,
                path_relation=row.path_relation,
            )
            for row in rows
        )
    )


@dataclass(frozen=True)
class ProbeFamilyResult:
    family: ProbeFamily
    correct_count: int
    query_count: int
    focal_correct_count: int
    focal_query_count: int
    exact_case_count: int
    case_count: int

    def __post_init__(self) -> None:
        if type(self.family) is not ProbeFamily:
            raise TypeError("family result family must be exact ProbeFamily")
        for name in (
            "correct_count",
            "query_count",
            "focal_correct_count",
            "focal_query_count",
            "exact_case_count",
            "case_count",
        ):
            _plain_int(name, getattr(self, name))
        if self.query_count < 1 or self.focal_query_count < 1 or self.case_count < 1:
            raise ValueError("family result denominators must be positive")
        if self.correct_count > self.query_count:
            raise ValueError("family correct count exceeds query count")
        if self.focal_correct_count > self.focal_query_count:
            raise ValueError("family focal correct count exceeds query count")
        if self.exact_case_count > self.case_count:
            raise ValueError("family exact count exceeds case count")

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.query_count

    @property
    def focal_accuracy(self) -> float:
        return self.focal_correct_count / self.focal_query_count


@dataclass(frozen=True)
class ProbePairResult:
    """Actual-cell score; rotated rotations never dilute this slice."""

    probe_pair: tuple[int, int]
    correct_count: int
    query_count: int
    focal_correct_count: int
    focal_query_count: int
    exact_case_count: int
    case_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.probe_pair, tuple) or len(self.probe_pair) != 2:
            raise TypeError("pair result probe_pair must be a two-integer tuple")
        for value in self.probe_pair:
            _plain_int("pair result coordinate", value)
        for name in (
            "correct_count",
            "query_count",
            "focal_correct_count",
            "focal_query_count",
            "exact_case_count",
            "case_count",
        ):
            _plain_int(name, getattr(self, name))
        if self.query_count < 1 or self.focal_query_count < 1 or self.case_count < 1:
            raise ValueError("pair result denominators must be positive")
        if self.correct_count > self.query_count:
            raise ValueError("pair correct count exceeds query count")
        if self.focal_correct_count > self.focal_query_count:
            raise ValueError("pair focal correct count exceeds focal query count")
        if self.exact_case_count > self.case_count:
            raise ValueError("pair exact count exceeds case count")

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.query_count

    @property
    def focal_accuracy(self) -> float:
        return self.focal_correct_count / self.focal_query_count


@dataclass(frozen=True)
class EquivalenceGroupResult:
    group: str
    family: ProbeFamily
    relation: ProbePathRelation
    member_count: int
    expected_focal_answers: tuple[tuple[int, ...], ...]
    predicted_focal_answers: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.group) is not str or not self.group:
            raise ValueError("equivalence group must be a nonempty exact string")
        if type(self.family) is not ProbeFamily:
            raise TypeError("equivalence family must be exact ProbeFamily")
        if type(self.relation) is not ProbePathRelation:
            raise TypeError("path relation must be exact ProbePathRelation")
        _plain_int("member_count", self.member_count, 2)
        if not isinstance(self.expected_focal_answers, tuple) or not isinstance(
            self.predicted_focal_answers, tuple
        ):
            raise TypeError("equivalence answers must be tuples")
        if len(self.expected_focal_answers) != self.member_count or len(
            self.predicted_focal_answers
        ) != self.member_count:
            raise ValueError("equivalence answer rows must match member_count")
        if any(
            not isinstance(row, tuple) or not row
            for row in self.expected_focal_answers
        ):
            raise ValueError("expected equivalence rows must be nonempty tuples")
        if any(
            not isinstance(row, tuple) or not row
            for row in self.predicted_focal_answers
        ):
            raise ValueError("predicted equivalence rows must be nonempty tuples")
        if any(
            len(expected) != len(predicted)
            for expected, predicted in zip(
                self.expected_focal_answers, self.predicted_focal_answers
            )
        ):
            raise ValueError("equivalence expected/predicted rows must align")
        if not self.expected_relation_satisfied:
            raise ValueError("trusted paths do not satisfy their declared relation")

    @property
    def expected_consistent(self) -> bool:
        return len(set(self.expected_focal_answers)) == 1

    @property
    def predicted_consistent(self) -> bool:
        return len(set(self.predicted_focal_answers)) == 1

    @property
    def expected_relation_satisfied(self) -> bool:
        if self.relation is ProbePathRelation.EQUAL:
            return self.expected_consistent
        return not self.expected_consistent

    @property
    def predicted_relation_satisfied(self) -> bool:
        if self.relation is ProbePathRelation.EQUAL:
            return self.predicted_consistent
        return not self.predicted_consistent


def _equivalence_results_from_cases(
    rows: tuple[ProbeCaseResult, ...],
) -> tuple[EquivalenceGroupResult, ...]:
    grouped: dict[str, list[ProbeCaseResult]] = {}
    family_by_group: dict[str, ProbeFamily] = {}
    relation_by_group: dict[str, ProbePathRelation] = {}
    for row in rows:
        if row.equivalence_group is None:
            continue
        assert row.path_relation is not None
        grouped.setdefault(row.equivalence_group, []).append(row)
        previous_family = family_by_group.setdefault(row.equivalence_group, row.family)
        previous_relation = relation_by_group.setdefault(
            row.equivalence_group, row.path_relation
        )
        if previous_family is not row.family:
            raise ValueError("path group mixes probe families")
        if previous_relation is not row.path_relation:
            raise ValueError("path group mixes equality and inequality")

    def focal(
        values: tuple[int, ...], roles: tuple[ProbeQueryRole, ...]
    ) -> tuple[int, ...]:
        return tuple(
            value
            for value, role in zip(values, roles)
            if role is ProbeQueryRole.FOCAL
        )

    return tuple(
        EquivalenceGroupResult(
            group=group,
            family=family_by_group[group],
            relation=relation_by_group[group],
            member_count=len(grouped[group]),
            expected_focal_answers=tuple(
                focal(row.expected_answers, row.query_roles)
                for row in grouped[group]
            ),
            predicted_focal_answers=tuple(
                focal(row.predicted_answers, row.query_roles)
                for row in grouped[group]
            ),
        )
        for group in sorted(grouped)
    )


@dataclass(frozen=True)
class SealedProbeEvaluation:
    protocol_status: ProbeProtocolStatus
    suite: SealedProbeSuite
    suite_sha256: str
    scoring_manifest_sha256: str
    case_results: tuple[ProbeCaseResult, ...]
    family_results: tuple[ProbeFamilyResult, ...]
    pair_results: tuple[ProbePairResult, ...]
    equivalence_results: tuple[EquivalenceGroupResult, ...]
    path_consistent_group_count: int
    path_group_count: int
    evaluation_sha256: str

    def __post_init__(self) -> None:
        if type(self.protocol_status) is not ProbeProtocolStatus:
            raise TypeError("evaluation status must be exact ProbeProtocolStatus")
        if type(self.suite) is not SealedProbeSuite:
            raise TypeError("evaluation suite must be exact SealedProbeSuite")
        if self.protocol_status is not self.suite.protocol_status:
            raise ValueError("evaluation protocol status disagrees with suite")
        for name in (
            "suite_sha256",
            "scoring_manifest_sha256",
            "evaluation_sha256",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"evaluation {name} must be lowercase SHA-256")
        if not isinstance(self.case_results, tuple) or not self.case_results:
            raise TypeError("evaluation case_results must be a nonempty tuple")
        if any(type(row) is not ProbeCaseResult for row in self.case_results):
            raise TypeError("evaluation cases must be exact ProbeCaseResult values")
        case_ids = tuple(row.case_id for row in self.case_results)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("evaluation case IDs must be unique")
        expected_manifest = _scoring_manifest_sha256_from_results(self.case_results)
        if self.scoring_manifest_sha256 != expected_manifest:
            raise ValueError("evaluation cases do not match the scoring manifest")
        if self.suite_sha256 != self.suite.suite_sha256:
            raise ValueError("evaluation suite hash disagrees with bound suite")
        if self.scoring_manifest_sha256 != self.suite.scoring_manifest_sha256:
            raise ValueError("evaluation scoring manifest disagrees with bound suite")
        if not isinstance(self.family_results, tuple) or tuple(
            row.family for row in self.family_results
        ) != tuple(ProbeFamily):
            raise ValueError("evaluation family results must cover enum order exactly")
        if not isinstance(self.pair_results, tuple) or not self.pair_results:
            raise TypeError("evaluation pair_results must be a nonempty tuple")
        if any(type(row) is not ProbePairResult for row in self.pair_results):
            raise TypeError("evaluation pair rows must be exact ProbePairResult values")
        pair_inventory = tuple(row.probe_pair for row in self.pair_results)
        if pair_inventory != tuple(sorted(set(pair_inventory))):
            raise ValueError("evaluation pair rows must be sorted and unique")
        if not isinstance(self.equivalence_results, tuple) or not (
            self.equivalence_results
        ):
            raise TypeError("evaluation equivalence results must be nonempty tuple")
        if any(
            type(row) is not EquivalenceGroupResult
            for row in self.equivalence_results
        ):
            raise TypeError("equivalence rows must be exact result values")
        expected_equivalence = _equivalence_results_from_cases(self.case_results)
        if self.equivalence_results != expected_equivalence:
            raise ValueError("equivalence results disagree with case results")
        _plain_int("path_consistent_group_count", self.path_consistent_group_count)
        _plain_int("path_group_count", self.path_group_count, 1)
        if self.path_group_count != len(self.equivalence_results):
            raise ValueError("path_group_count disagrees with equivalence results")
        expected_consistent = sum(
            row.predicted_relation_satisfied for row in self.equivalence_results
        )
        if self.path_consistent_group_count != expected_consistent:
            raise ValueError("path consistency count disagrees with predictions")

        for family_result in self.family_results:
            rows = tuple(
                row for row in self.case_results if row.family is family_result.family
            )
            expected = ProbeFamilyResult(
                family=family_result.family,
                correct_count=sum(row.correct_count for row in rows),
                query_count=sum(len(row.expected_answers) for row in rows),
                focal_correct_count=sum(row.focal_correct_count for row in rows),
                focal_query_count=sum(row.focal_query_count for row in rows),
                exact_case_count=sum(row.exact_match for row in rows),
                case_count=len(rows),
            )
            if family_result != expected:
                raise ValueError("family result disagrees with case results")
        for pair_result in self.pair_results:
            rows = tuple(
                row
                for row in self.case_results
                if row.probe_pair == pair_result.probe_pair
            )
            expected = ProbePairResult(
                probe_pair=pair_result.probe_pair,
                correct_count=sum(row.correct_count for row in rows),
                query_count=sum(len(row.expected_answers) for row in rows),
                focal_correct_count=sum(row.focal_correct_count for row in rows),
                focal_query_count=sum(row.focal_query_count for row in rows),
                exact_case_count=sum(row.exact_match for row in rows),
                case_count=len(rows),
            )
            if pair_result != expected:
                raise ValueError("pair result disagrees with case results")
        if set(pair_inventory) != {row.probe_pair for row in self.case_results}:
            raise ValueError("pair result inventory disagrees with case results")
        expected_evaluation_sha256 = _evaluation_sha256(self)
        if self.evaluation_sha256 != expected_evaluation_sha256:
            raise ValueError("evaluation_sha256 does not bind evaluation results")

    @property
    def correct_count(self) -> int:
        return sum(row.correct_count for row in self.case_results)

    @property
    def query_count(self) -> int:
        return sum(len(row.expected_answers) for row in self.case_results)

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.query_count

    @property
    def focal_correct_count(self) -> int:
        return sum(row.focal_correct_count for row in self.case_results)

    @property
    def focal_query_count(self) -> int:
        return sum(row.focal_query_count for row in self.case_results)

    @property
    def focal_accuracy(self) -> float:
        return self.focal_correct_count / self.focal_query_count

    @property
    def macro_family_accuracy(self) -> float:
        return sum(row.accuracy for row in self.family_results) / len(
            self.family_results
        )

    @property
    def macro_family_focal_accuracy(self) -> float:
        return sum(row.focal_accuracy for row in self.family_results) / len(
            self.family_results
        )

    @property
    def path_consistency(self) -> float:
        """Fraction satisfying each predeclared equality *or inequality* law."""

        return self.path_consistent_group_count / self.path_group_count

    @property
    def all_paths_consistent(self) -> bool:
        return self.path_consistent_group_count == self.path_group_count

    @property
    def all_path_relations_satisfied(self) -> bool:
        return self.path_consistent_group_count == self.path_group_count

    @property
    def equality_group_count(self) -> int:
        return sum(
            row.relation is ProbePathRelation.EQUAL
            for row in self.equivalence_results
        )

    @property
    def inequality_group_count(self) -> int:
        return sum(
            row.relation is ProbePathRelation.NOT_EQUAL
            for row in self.equivalence_results
        )

    @property
    def inequality_satisfied_group_count(self) -> int:
        return sum(
            row.relation is ProbePathRelation.NOT_EQUAL
            and row.predicted_relation_satisfied
            for row in self.equivalence_results
        )

    @property
    def all_copy_order_inequalities_satisfied(self) -> bool:
        return self.inequality_satisfied_group_count == self.inequality_group_count

    def result_for_pair(self, pair: tuple[int, int]) -> ProbePairResult:
        for row in self.pair_results:
            if row.probe_pair == pair:
                return row
        raise KeyError(f"probe pair {pair!r} is not present in this evaluation")


def _evaluation_payload(
    *,
    protocol_status: ProbeProtocolStatus,
    suite: SealedProbeSuite,
    suite_sha256: str,
    scoring_manifest_sha256: str,
    case_results: tuple[ProbeCaseResult, ...],
    family_results: tuple[ProbeFamilyResult, ...],
    pair_results: tuple[ProbePairResult, ...],
    equivalence_results: tuple[EquivalenceGroupResult, ...],
    path_consistent_group_count: int,
    path_group_count: int,
) -> dict[str, object]:
    return {
        "schema": "tnlm-v3-sealed-probe-evaluation-v1",
        "protocol_status": protocol_status.value,
        "bound_suite_sha256": suite.suite_sha256,
        "suite_sha256": suite_sha256,
        "scoring_manifest_sha256": scoring_manifest_sha256,
        "case_results": tuple(
            {
                "manifest": _scoring_manifest_row(
                    case_id=row.case_id,
                    family=row.family,
                    base_probe_pair=row.base_probe_pair,
                    probe_pair=row.probe_pair,
                    cell_rotation=row.cell_rotation,
                    value_cardinality=row.value_cardinality,
                    expected_answers=row.expected_answers,
                    query_roles=row.query_roles,
                    equivalence_group=row.equivalence_group,
                    path_relation=row.path_relation,
                ),
                "predicted_answers": row.predicted_answers,
            }
            for row in case_results
        ),
        "family_results": tuple(
            {
                name: (value.value if type(value) is ProbeFamily else value)
                for name, value in (
                    (field, getattr(row, field))
                    for field in row.__dataclass_fields__
                )
            }
            for row in family_results
        ),
        "pair_results": tuple(
            {field: getattr(row, field) for field in row.__dataclass_fields__}
            for row in pair_results
        ),
        "equivalence_results": tuple(
            {
                "group": row.group,
                "family": row.family.value,
                "relation": row.relation.value,
                "member_count": row.member_count,
                "expected_focal_answers": row.expected_focal_answers,
                "predicted_focal_answers": row.predicted_focal_answers,
            }
            for row in equivalence_results
        ),
        "path_consistent_group_count": path_consistent_group_count,
        "path_group_count": path_group_count,
    }


def _evaluation_sha256_from_fields(**fields: object) -> str:
    return hashlib.sha256(
        json.dumps(
            _evaluation_payload(**fields),  # type: ignore[arg-type]
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _evaluation_sha256(evaluation: SealedProbeEvaluation) -> str:
    return _evaluation_sha256_from_fields(
        protocol_status=evaluation.protocol_status,
        suite=evaluation.suite,
        suite_sha256=evaluation.suite_sha256,
        scoring_manifest_sha256=evaluation.scoring_manifest_sha256,
        case_results=evaluation.case_results,
        family_results=evaluation.family_results,
        pair_results=evaluation.pair_results,
        equivalence_results=evaluation.equivalence_results,
        path_consistent_group_count=evaluation.path_consistent_group_count,
        path_group_count=evaluation.path_group_count,
    )


@dataclass(frozen=True)
class ShortcutControlResult:
    name: str
    evaluation: SealedProbeEvaluation


def _trusted_execute(
    program: VisibleProbeProgram,
    forbidden_pairs: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, ...], ProbeSupportAudit, tuple[int, ...]]:
    """Execute a legal program and audit all live cells before and after events."""

    state = [-1] * program.num_surface_keys
    answers: list[int] = []
    query_event_indices: list[int] = []
    touched: set[tuple[int, int]] = set()

    def audit_state() -> None:
        touched.update((key, value) for key, value in enumerate(state) if value >= 0)

    audit_state()
    for index, event in enumerate(program.events):
        audit_state()
        kind = event.kind
        key = event.primary_key
        if kind is BindingEventKind.BIND:
            if state[key] >= 0:
                raise ValueError("probe contains BIND on a live key")
            state[key] = event.argument
        elif kind is BindingEventKind.UPDATE:
            if state[key] < 0:
                raise ValueError("probe contains UPDATE on an absent key")
            state[key] = (
                state[key] + event.argument + 1
            ) % program.value_cardinality
        elif kind is BindingEventKind.COPY:
            if state[key] < 0 or state[event.secondary_key] < 0:
                raise ValueError("probe COPY requires two live keys")
            state[key] = state[event.secondary_key]
        elif kind is BindingEventKind.INVALIDATE:
            if state[key] < 0:
                raise ValueError("probe contains INVALIDATE on an absent key")
            state[key] = -1
        elif kind is BindingEventKind.QUERY:
            if state[key] < 0:
                raise ValueError("probe contains QUERY on an absent key")
            answers.append(state[key])
            query_event_indices.append(index)
        audit_state()
    touched_tuple = tuple(sorted(touched))
    forbidden_tuple = tuple(sorted(set(forbidden_pairs)))
    intersection = tuple(sorted(set(touched_tuple) & set(forbidden_tuple)))
    return (
        tuple(answers),
        ProbeSupportAudit(
            touched_pairs=touched_tuple,
            forbidden_pairs=forbidden_tuple,
            forbidden_intersection=intersection,
            every_intermediate_state_audited=True,
        ),
        tuple(query_event_indices),
    )


def _trusted_entry(
    program: VisibleProbeProgram,
    probe_pair: tuple[int, int],
) -> tuple[int, BindingEventKind]:
    """Return the first transition that enters ``probe_pair``."""

    key, value = probe_pair
    state = [-1] * program.num_surface_keys
    for index, event in enumerate(program.events):
        was_inside = state[key] == value
        if event.kind is BindingEventKind.BIND:
            state[event.primary_key] = event.argument
        elif event.kind is BindingEventKind.UPDATE:
            state[event.primary_key] = (
                state[event.primary_key] + event.argument + 1
            ) % program.value_cardinality
        elif event.kind is BindingEventKind.COPY:
            state[event.primary_key] = state[event.secondary_key]
        elif event.kind is BindingEventKind.INVALIDATE:
            state[event.primary_key] = -1
        if not was_inside and state[key] == value:
            return index, event.kind
    raise ValueError("program never enters its declared probe cell")


def _transform_for(source: int, target: int, cardinality: int) -> int:
    return (target - source - 1) % cardinality


def _allowed_value(
    key: int,
    cardinality: int,
    forbidden: set[tuple[int, int]],
    *,
    exclude: frozenset[int] = frozenset(),
) -> int:
    for value in range(cardinality):
        if value not in exclude and (key, value) not in forbidden:
            return value
    raise ValueError("no firewall-safe value exists for the requested probe route")


def _balance_key(
    keys: int,
    forbidden: set[tuple[int, int]],
    preferred_not: int,
    cardinality: int,
) -> int:
    for key in range(keys):
        if key != preferred_not and all(
            (key, value) not in forbidden for value in range(cardinality)
        ):
            return key
    for key in range(keys):
        if all((key, value) not in forbidden for value in range(cardinality)):
            return key
    raise ValueError(
        "output balancing needs one key whose value orbit is outside forbidden support"
    )


def _append_balance_controls(
    *,
    keys: int,
    cardinality: int,
    events: tuple[VisibleProbeEvent, ...],
    focal_roles: tuple[ProbeQueryRole, ...],
    forbidden_pairs: tuple[tuple[int, int], ...],
    probe_key: int,
) -> tuple[tuple[VisibleProbeEvent, ...], tuple[ProbeQueryRole, ...]]:
    """Pad a focal program to equal target counts without changing its claims."""

    prefix = VisibleProbeProgram(keys, cardinality, events)
    answers, _, _ = _trusted_execute(prefix, forbidden_pairs)
    if len(answers) != len(focal_roles) or any(
        role is not ProbeQueryRole.FOCAL for role in focal_roles
    ):
        raise ValueError("focal roles must align with the unbalanced prefix")
    counts = Counter(answers)
    # Give every class one slot for every focal answer.  This keeps the total
    # probe inventory invariant to whether two focal answers happen to collide:
    # each case has ``cardinality * len(answers)`` total queries.
    target_count = len(answers)
    deficits = tuple(
        target_count - counts.get(value, 0) for value in range(cardinality)
    )
    if not any(deficits):
        return events, focal_roles

    forbidden = set(forbidden_pairs)
    key = _balance_key(keys, forbidden, probe_key, cardinality)
    state = [-1] * keys
    for event in events:
        if event.kind is BindingEventKind.BIND:
            state[event.primary_key] = event.argument
        elif event.kind is BindingEventKind.UPDATE:
            state[event.primary_key] = (
                state[event.primary_key] + event.argument + 1
            ) % cardinality
        elif event.kind is BindingEventKind.COPY:
            state[event.primary_key] = state[event.secondary_key]
        elif event.kind is BindingEventKind.INVALIDATE:
            state[event.primary_key] = -1

    extended = list(events)
    roles = list(focal_roles)
    if state[key] >= 0:
        extended.append(VisibleProbeEvent.invalidate(key))
        state[key] = -1
    for value, deficit in enumerate(deficits):
        for _ in range(deficit):
            extended.append(VisibleProbeEvent.bind(key, value))
            extended.append(VisibleProbeEvent.query(key))
            roles.append(ProbeQueryRole.BALANCE_CONTROL)
            extended.append(VisibleProbeEvent.invalidate(key))
    return tuple(extended), tuple(roles)


def _case(
    *,
    family: ProbeFamily,
    probe_pair: tuple[int, int],
    keys: int,
    cardinality: int,
    events: tuple[VisibleProbeEvent, ...],
    forbidden_pairs: tuple[tuple[int, int], ...],
    variant: str,
    equivalence_group: str | None = None,
    path_relation: ProbePathRelation | None = None,
) -> SealedProbeCase:
    query_count = sum(event.kind is BindingEventKind.QUERY for event in events)
    balanced_events, roles = _append_balance_controls(
        keys=keys,
        cardinality=cardinality,
        events=events,
        focal_roles=(ProbeQueryRole.FOCAL,) * query_count,
        forbidden_pairs=forbidden_pairs,
        probe_key=probe_pair[0],
    )
    program = VisibleProbeProgram(keys, cardinality, balanced_events)
    answers, support, _ = _trusted_execute(program, forbidden_pairs)
    entry_index, entry_kind = _trusted_entry(program, probe_pair)
    return SealedProbeCase(
        case_id=(
            f"{family.value}/key-{probe_pair[0]}/value-{probe_pair[1]}/{variant}"
        ),
        family=family,
        base_probe_pair=probe_pair,
        probe_pair=probe_pair,
        cell_rotation=ProbeCellRotation.identity(keys, cardinality),
        program=program,
        expected_answers=answers,
        query_roles=roles,
        support_audit=support,
        probe_entry_event_index=entry_index,
        probe_entry_kind=entry_kind,
        equivalence_group=equivalence_group,
        path_relation=(
            ProbePathRelation.EQUAL
            if equivalence_group is not None and path_relation is None
            else path_relation
        ),
    )


def _retag_rotated_case(
    case: SealedProbeCase,
    base_probe_pair: tuple[int, int],
    cell_rotation: ProbeCellRotation,
    cell_rotation_index: int,
) -> SealedProbeCase:
    suffix = f"cell_rotation-{cell_rotation_index}/{cell_rotation.label}"
    return SealedProbeCase(
        case_id=f"{case.case_id}/{suffix}",
        family=case.family,
        base_probe_pair=base_probe_pair,
        probe_pair=case.probe_pair,
        cell_rotation=cell_rotation,
        program=case.program,
        expected_answers=case.expected_answers,
        query_roles=case.query_roles,
        support_audit=case.support_audit,
        probe_entry_event_index=case.probe_entry_event_index,
        probe_entry_kind=case.probe_entry_kind,
        equivalence_group=(
            None
            if case.equivalence_group is None
            else f"{case.equivalence_group}/{suffix}"
        ),
        path_relation=case.path_relation,
    )


def _cell_cases(
    keys: int,
    cardinality: int,
    probe_pair: tuple[int, int],
    forbidden_pairs: tuple[tuple[int, int], ...],
    long_neutral_cycles: int,
) -> tuple[SealedProbeCase, ...]:
    """Construct one equally weighted family inventory for a rotated cell."""

    key, value = probe_pair
    forbidden = set(forbidden_pairs)
    auxiliary = _balance_key(keys, forbidden, key, cardinality)
    safe_pre = _allowed_value(
        key,
        cardinality,
        forbidden,
        exclude=frozenset((value,)),
    )
    safe_out = _allowed_value(
        key,
        cardinality,
        forbidden,
        exclude=frozenset((value,)),
    )
    auxiliary_initial = _allowed_value(auxiliary, cardinality, forbidden)
    update_in = _transform_for(safe_pre, value, cardinality)
    update_out = _transform_for(value, safe_out, cardinality)
    cell_tag = f"key-{key}/value-{value}"
    rows: list[SealedProbeCase] = []

    rows.append(
        _case(
            family=ProbeFamily.FIRST_ENTRY_BIND,
            probe_pair=probe_pair,
            keys=keys,
            cardinality=cardinality,
            events=(VisibleProbeEvent.bind(key, value), VisibleProbeEvent.query(key)),
            forbidden_pairs=forbidden_pairs,
            variant="direct",
        )
    )
    rows.append(
        _case(
            family=ProbeFamily.FIRST_ENTRY_UPDATE,
            probe_pair=probe_pair,
            keys=keys,
            cardinality=cardinality,
            events=(
                VisibleProbeEvent.bind(key, safe_pre),
                VisibleProbeEvent.update(key, update_in),
                VisibleProbeEvent.query(key),
            ),
            forbidden_pairs=forbidden_pairs,
            variant="safe-predecessor",
        )
    )
    rows.append(
        _case(
            family=ProbeFamily.FIRST_ENTRY_COPY,
            probe_pair=probe_pair,
            keys=keys,
            cardinality=cardinality,
            events=(
                VisibleProbeEvent.bind(key, safe_pre),
                VisibleProbeEvent.bind(auxiliary, value),
                VisibleProbeEvent.copy(key, auxiliary),
                VisibleProbeEvent.query(key),
            ),
            forbidden_pairs=forbidden_pairs,
            variant="source-entry",
        )
    )
    rows.append(
        _case(
            family=ProbeFamily.UPDATE_OUT,
            probe_pair=probe_pair,
            keys=keys,
            cardinality=cardinality,
            events=(
                VisibleProbeEvent.bind(key, value),
                VisibleProbeEvent.update(key, update_out),
                VisibleProbeEvent.query(key),
            ),
            forbidden_pairs=forbidden_pairs,
            variant="safe-successor",
        )
    )

    naturality_group = f"copy-naturality/{cell_tag}"
    rows.append(
        _case(
            family=ProbeFamily.COPY_OUT_NATURALITY,
            probe_pair=probe_pair,
            keys=keys,
            cardinality=cardinality,
            events=(
                VisibleProbeEvent.bind(key, value),
                VisibleProbeEvent.bind(auxiliary, auxiliary_initial),
                VisibleProbeEvent.update(key, update_out),
                VisibleProbeEvent.copy(auxiliary, key),
                VisibleProbeEvent.query(auxiliary),
            ),
            forbidden_pairs=forbidden_pairs,
            variant="update-then-copy",
            equivalence_group=naturality_group,
        )
    )
    rows.append(
        _case(
            family=ProbeFamily.COPY_OUT_NATURALITY,
            probe_pair=probe_pair,
            keys=keys,
            cardinality=cardinality,
            events=(
                VisibleProbeEvent.bind(key, value),
                VisibleProbeEvent.bind(auxiliary, auxiliary_initial),
                VisibleProbeEvent.copy(auxiliary, key),
                VisibleProbeEvent.update(auxiliary, update_out),
                VisibleProbeEvent.query(auxiliary),
            ),
            forbidden_pairs=forbidden_pairs,
            variant="copy-then-update",
            equivalence_group=naturality_group,
        )
    )

    rows.append(
        _case(
            family=ProbeFamily.REPEATED_QUERY_IDENTITY,
            probe_pair=probe_pair,
            keys=keys,
            cardinality=cardinality,
            events=(
                VisibleProbeEvent.bind(key, value),
                VisibleProbeEvent.query(key),
                VisibleProbeEvent.query(key),
                VisibleProbeEvent.query(key),
            ),
            forbidden_pairs=forbidden_pairs,
            variant="repeat-and-identity-update",
        )
    )

    cycle_events: list[VisibleProbeEvent] = [VisibleProbeEvent.bind(key, value)]
    # A true full orbit is used whenever its intermediate cells are allowed.
    # An inner pseudo-fold with an outer firewall cannot traverse the real cell;
    # there we perform an inverse pair on the probe key and a full orbit on a
    # firewall-safe auxiliary key instead.
    key_orbit_allowed = all(
        (key, symbol) not in forbidden for symbol in range(cardinality)
    )
    if key_orbit_allowed:
        cycle_events.extend(
            VisibleProbeEvent.update(key, 0) for _ in range(cardinality)
        )
    else:
        cycle_events.extend(
            (
                VisibleProbeEvent.update(key, update_out),
                VisibleProbeEvent.update(
                    key, _transform_for(safe_out, value, cardinality)
                ),
                VisibleProbeEvent.bind(auxiliary, 0),
            )
        )
        cycle_events.extend(
            VisibleProbeEvent.update(auxiliary, 0) for _ in range(cardinality)
        )
    cycle_events.append(VisibleProbeEvent.query(key))
    rows.append(
        _case(
            family=ProbeFamily.CYCLIC_COMPOSITION,
            probe_pair=probe_pair,
            keys=keys,
            cardinality=cardinality,
            events=tuple(cycle_events),
            forbidden_pairs=forbidden_pairs,
            variant="full-orbit-identity",
        )
    )

    rows.append(
        _case(
            family=ProbeFamily.INVALIDATE_REBIND,
            probe_pair=probe_pair,
            keys=keys,
            cardinality=cardinality,
            events=(
                VisibleProbeEvent.bind(key, value),
                VisibleProbeEvent.query(key),
                VisibleProbeEvent.invalidate(key),
                VisibleProbeEvent.distractor(0),
                VisibleProbeEvent.bind(key, value),
                VisibleProbeEvent.query(key),
            ),
            forbidden_pairs=forbidden_pairs,
            variant="new-generation",
        )
    )

    commutation_group = f"independent-commutation/{cell_tag}"
    commutation_prefix = (
        VisibleProbeEvent.bind(key, value),
        VisibleProbeEvent.bind(auxiliary, auxiliary_initial),
    )
    key_update = VisibleProbeEvent.update(key, update_out)
    auxiliary_update = VisibleProbeEvent.update(auxiliary, update_out)
    suffix = (VisibleProbeEvent.query(key), VisibleProbeEvent.query(auxiliary))
    for variant, ordered_updates in (
        ("key-then-independent", (key_update, auxiliary_update)),
        ("independent-then-key", (auxiliary_update, key_update)),
    ):
        rows.append(
            _case(
                family=ProbeFamily.INDEPENDENT_COMMUTATION,
                probe_pair=probe_pair,
                keys=keys,
                cardinality=cardinality,
                events=commutation_prefix + ordered_updates + suffix,
                forbidden_pairs=forbidden_pairs,
                variant=variant,
                equivalence_group=commutation_group,
            )
        )

    rows.append(
        _case(
            family=ProbeFamily.DISTRACTOR_INTERLEAVING,
            probe_pair=probe_pair,
            keys=keys,
            cardinality=cardinality,
            events=(
                VisibleProbeEvent.distractor(0),
                VisibleProbeEvent.bind(key, value),
                VisibleProbeEvent.distractor(1),
                VisibleProbeEvent.query(key),
                VisibleProbeEvent.distractor(0),
                VisibleProbeEvent.query(key),
                VisibleProbeEvent.distractor(1),
            ),
            forbidden_pairs=forbidden_pairs,
            variant="both-scopes",
        )
    )

    long_events: list[VisibleProbeEvent] = [
        VisibleProbeEvent.bind(key, value),
        VisibleProbeEvent.bind(auxiliary, auxiliary_initial),
    ]
    for _ in range(long_neutral_cycles):
        long_events.append(VisibleProbeEvent.distractor(0))
        long_events.extend(
            VisibleProbeEvent.update(auxiliary, 0) for _ in range(cardinality)
        )
        long_events.append(VisibleProbeEvent.copy(auxiliary, key))
        long_events.extend(
            VisibleProbeEvent.update(auxiliary, 0) for _ in range(cardinality)
        )
        long_events.append(VisibleProbeEvent.distractor(1))
    long_events.extend(
        (
            VisibleProbeEvent.copy(key, auxiliary),
            VisibleProbeEvent.query(key),
            VisibleProbeEvent.query(auxiliary),
        )
    )
    rows.append(
        _case(
            family=ProbeFamily.LONG_COMPOSITION,
            probe_pair=probe_pair,
            keys=keys,
            cardinality=cardinality,
            events=tuple(long_events),
            forbidden_pairs=forbidden_pairs,
            variant=f"neutral-cycles-{long_neutral_cycles}",
        )
    )

    copy_order_group = f"copy-order-noncommutation/{cell_tag}"
    copy_order_prefix = (
        VisibleProbeEvent.bind(key, value),
        VisibleProbeEvent.bind(auxiliary, safe_pre),
    )
    copy_order_suffix = (
        VisibleProbeEvent.query(key),
        VisibleProbeEvent.query(auxiliary),
    )
    for variant, copies in (
        (
            "destination-first",
            (
                VisibleProbeEvent.copy(auxiliary, key),
                VisibleProbeEvent.copy(key, auxiliary),
            ),
        ),
        (
            "source-first",
            (
                VisibleProbeEvent.copy(key, auxiliary),
                VisibleProbeEvent.copy(auxiliary, key),
            ),
        ),
    ):
        rows.append(
            _case(
                family=ProbeFamily.COPY_ORDER_NONCOMMUTATION,
                probe_pair=probe_pair,
                keys=keys,
                cardinality=cardinality,
                events=copy_order_prefix + copies + copy_order_suffix,
                forbidden_pairs=forbidden_pairs,
                variant=variant,
                equivalence_group=copy_order_group,
                path_relation=ProbePathRelation.NOT_EQUAL,
            )
        )
    result = tuple(rows)
    if len(result) != _CASES_PER_CELL:
        raise AssertionError("probe family inventory changed without budget update")
    return result


def _analytic_probe_event_upper_bound(
    cardinality: int,
    long_neutral_cycles: int,
    mapped_cell_count: int,
) -> int:
    longest_focal_program = 5 + long_neutral_cycles * (2 * cardinality + 3)
    largest_balance_suffix = 1 + 9 * (cardinality - 1)
    return (
        mapped_cell_count
        * _CASES_PER_CELL
        * (longest_focal_program + largest_balance_suffix)
    )


def _balance_certificate(
    cases: tuple[SealedProbeCase, ...],
    cardinality: int,
    probe_pairs: tuple[tuple[int, int], ...],
) -> ProbeBalanceCertificate:
    family_rows: list[tuple[ProbeFamily, tuple[int, ...]]] = []
    family_focal_rows: list[tuple[ProbeFamily, tuple[int, ...]]] = []
    for family in ProbeFamily:
        counts = [0] * cardinality
        focal_counts = [0] * cardinality
        for case in cases:
            if case.family is family:
                for answer, role in zip(case.expected_answers, case.query_roles):
                    counts[answer] += 1
                    if role is ProbeQueryRole.FOCAL:
                        focal_counts[answer] += 1
        family_rows.append((family, tuple(counts)))
        family_focal_rows.append((family, tuple(focal_counts)))
    pair_case_counts = tuple(
        (pair, sum(case.probe_pair == pair for case in cases)) for pair in probe_pairs
    )
    pair_focal_counts = tuple(
        (
            pair,
            sum(
                case.query_roles.count(ProbeQueryRole.FOCAL)
                for case in cases
                if case.probe_pair == pair
            ),
        )
        for pair in probe_pairs
    )
    return ProbeBalanceCertificate(
        family_class_counts=tuple(family_rows),
        family_focal_class_counts=tuple(family_focal_rows),
        pair_case_counts=pair_case_counts,
        pair_focal_query_counts=pair_focal_counts,
        every_family_output_balanced=all(
            len(set(counts)) == 1 for _, counts in family_rows
        ),
        every_family_focal_output_balanced=all(
            len(set(counts)) == 1 for _, counts in family_focal_rows
        ),
        rotated_cells_equally_weighted=(
            len({count for _, count in pair_case_counts}) == 1
            and len({count for _, count in pair_focal_counts}) == 1
        ),
    )


def _suite_digest_payload(
    *,
    protocol_status: ProbeProtocolStatus,
    keys: int,
    cardinality: int,
    base_probe_pairs: tuple[tuple[int, int], ...],
    cell_rotations: tuple[ProbeCellRotation, ...],
    probe_pairs: tuple[tuple[int, int], ...],
    forbidden_pairs: tuple[tuple[int, int], ...],
    cases: tuple[SealedProbeCase, ...],
    balance: ProbeBalanceCertificate,
    scoring_manifest_sha256: str,
) -> dict[str, object]:
    return {
        "schema": "tnlm-v3-balanced-sealed-probes-v1",
        "protocol_status": protocol_status.value,
        "num_surface_keys": keys,
        "value_cardinality": cardinality,
        "base_probe_pairs": base_probe_pairs,
        "cell_rotations": tuple(
            {
                "key_permutation": row.key_permutation,
                "value_offset": row.value_offset,
                "value_permutation": row.value_permutation,
            }
            for row in cell_rotations
        ),
        "probe_pairs": probe_pairs,
        "forbidden_pairs": forbidden_pairs,
        "scoring_manifest_sha256": scoring_manifest_sha256,
        "cases": [
            {
                "case_id": case.case_id,
                "family": case.family.value,
                "base_probe_pair": case.base_probe_pair,
                "probe_pair": case.probe_pair,
                "cell_rotation": {
                    "key_permutation": case.cell_rotation.key_permutation,
                    "value_offset": case.cell_rotation.value_offset,
                    "value_permutation": case.cell_rotation.value_permutation,
                },
                "events": [
                    (
                        int(event.kind),
                        event.primary_key,
                        event.secondary_key,
                        event.argument,
                    )
                    for event in case.program.events
                ],
                "expected_answers": case.expected_answers,
                "query_roles": tuple(role.value for role in case.query_roles),
                "touched_pairs": case.support_audit.touched_pairs,
                "forbidden_intersection": case.support_audit.forbidden_intersection,
                "every_intermediate_state_audited": (
                    case.support_audit.every_intermediate_state_audited
                ),
                "probe_entry_event_index": case.probe_entry_event_index,
                "probe_entry_kind": int(case.probe_entry_kind),
                "equivalence_group": case.equivalence_group,
                "path_relation": (
                    None if case.path_relation is None else case.path_relation.value
                ),
            }
            for case in cases
        ],
        "balance": {
            "family_class_counts": tuple(
                (family.value, counts)
                for family, counts in balance.family_class_counts
            ),
            "family_focal_class_counts": tuple(
                (family.value, counts)
                for family, counts in balance.family_focal_class_counts
            ),
            "pair_case_counts": balance.pair_case_counts,
            "pair_focal_query_counts": balance.pair_focal_query_counts,
            "every_family_output_balanced": (
                balance.every_family_output_balanced
            ),
            "every_family_focal_output_balanced": (
                balance.every_family_focal_output_balanced
            ),
            "rotated_cells_equally_weighted": (
                balance.rotated_cells_equally_weighted
            ),
        },
    }


def build_balanced_probe_suite(
    num_surface_keys: int,
    value_cardinality: int,
    probe_pairs: Sequence[tuple[int, int]],
    *,
    forbidden_pairs: Sequence[tuple[int, int]] = (),
    cell_rotations: Sequence[ProbeCellRotation] | None = None,
    protocol_status: ProbeProtocolStatus = (
        ProbeProtocolStatus.RETROSPECTIVE_PROTOCOL_REHEARSAL
    ),
    long_neutral_cycles: int = 4,
    max_cell_rotations: int = DEFAULT_MAX_CELL_ROTATIONS,
    max_cases: int = DEFAULT_MAX_PROBE_CASES,
    max_probe_events: int = DEFAULT_MAX_PROBE_EVENTS,
    max_probe_work: int = DEFAULT_MAX_PROBE_WORK,
) -> SealedProbeSuite:
    """Build trusted cases after fitting and before calling the evaluator.

    ``forbidden_pairs`` is the outer firewall for pseudo-held-out selection.
    Every live cell in every pre/post intermediate state is audited.  A probe
    pair may not itself be forbidden.  The nominal V3 screen (K=5, V=4) can
    therefore validate all 19 observed rotations without ever entering the
    real ``(0, 0)`` cell.

    Balance controls are queries on a firewall-safe auxiliary key.  They make
    *each family* exactly uniform over output classes while focal metrics stay
    separate, so controls cannot hide failure on the actual held-out route.

    When ``cell_rotations`` is supplied, ``probe_pairs`` are attribution bases.
    Their Cartesian product with the rotation inventory must map injectively
    to actual probe cells.  A fresh balanced program family is generated at
    each destination cell; programs and outputs are not transformations of a
    base template.  These rotations therefore provide balanced cell coverage,
    not program-level conjugacy, output equivariance, or metamorphic evidence.
    The mapped cells, attribution maps, cases, labels, audits, and counts are
    all bound into ``suite_sha256``.  The supported SCREEN protocol is K=5,
    V=4; smaller vocabularies can make a requested first-entry route
    incompatible with an outer forbidden-cell firewall and are rejected
    rather than weakened.
    """

    keys = _plain_int("num_surface_keys", num_surface_keys, 2)
    cardinality = _plain_int("value_cardinality", value_cardinality, 2)
    cycles = _plain_int("long_neutral_cycles", long_neutral_cycles, 1)
    cell_rotation_limit = _plain_int("max_cell_rotations", max_cell_rotations, 1)
    case_limit = _plain_int("max_cases", max_cases, 1)
    event_limit = _plain_int("max_probe_events", max_probe_events, 1)
    work_limit = _plain_int("max_probe_work", max_probe_work, 1)
    if type(protocol_status) is not ProbeProtocolStatus:
        raise TypeError("protocol_status must be exact ProbeProtocolStatus")
    if not isinstance(probe_pairs, Sequence) or isinstance(probe_pairs, (str, bytes)):
        raise TypeError("probe_pairs must be a finite sequence")
    if len(probe_pairs) > case_limit // _CASES_PER_CELL:
        raise ProbeBudgetExceededError(
            "probe_pairs exceed max_cases before deduplication"
        )
    if not isinstance(forbidden_pairs, Sequence) or isinstance(
        forbidden_pairs, (str, bytes)
    ):
        raise TypeError("forbidden_pairs must be a finite sequence")
    if len(forbidden_pairs) > work_limit:
        raise ProbeBudgetExceededError("forbidden_pairs exceed max_probe_work")
    base_probe_pairs = tuple(sorted(set(probe_pairs)))
    normalized_forbidden = tuple(sorted(set(forbidden_pairs)))
    if not base_probe_pairs:
        raise ValueError("probe_pairs may not be empty")
    for name, rows in (
        ("probe_pairs", base_probe_pairs),
        ("forbidden_pairs", normalized_forbidden),
    ):
        for row in rows:
            _pair(name, row, keys, cardinality)
    explicit_cell_rotations = cell_rotations is not None
    if cell_rotations is not None and (
        not isinstance(cell_rotations, Sequence)
        or isinstance(cell_rotations, (str, bytes))
    ):
        raise TypeError("cell_rotations must be a finite sequence or None")
    requested_cell_rotation_count = 1 if cell_rotations is None else len(cell_rotations)
    if requested_cell_rotation_count > cell_rotation_limit:
        raise ProbeBudgetExceededError("cell_rotations exceed max_cell_rotations")
    mapped_cell_count = len(base_probe_pairs) * requested_cell_rotation_count
    requested_case_count = mapped_cell_count * _CASES_PER_CELL
    if requested_case_count > case_limit:
        raise ProbeBudgetExceededError("probe construction exceeds max_cases")
    event_upper_bound = _analytic_probe_event_upper_bound(
        cardinality, cycles, mapped_cell_count
    )
    if event_upper_bound > event_limit:
        raise ProbeBudgetExceededError(
            "probe construction exceeds max_probe_events analytically"
        )
    work_upper_bound = event_upper_bound + mapped_cell_count * keys * cardinality
    if work_upper_bound > work_limit:
        raise ProbeBudgetExceededError(
            "probe construction exceeds max_probe_work analytically"
        )
    normalized_cell_rotations = (
        (ProbeCellRotation.identity(keys, cardinality),)
        if cell_rotations is None
        else tuple(cell_rotations)
    )
    if not normalized_cell_rotations:
        raise ValueError("cell_rotations may not be empty when supplied")
    if any(type(row) is not ProbeCellRotation for row in normalized_cell_rotations):
        raise TypeError("cell_rotations must contain exact ProbeCellRotation values")
    if len(set(normalized_cell_rotations)) != len(normalized_cell_rotations):
        raise ValueError("cell_rotations must be unique")
    if any(
        len(row.key_permutation) != keys
        or len(row.value_permutation) != cardinality
        for row in normalized_cell_rotations
    ):
        raise ValueError("cell_rotation vocabulary disagrees with probe vocabulary")
    mapped = tuple(
        (base, cell_rotation, cell_rotation.apply_pair(base))
        for base in base_probe_pairs
        for cell_rotation in normalized_cell_rotations
    )
    normalized_probes = tuple(sorted(actual for _, _, actual in mapped))
    if len(normalized_probes) != len(set(normalized_probes)):
        raise ValueError("base pairs and cell_rotations must map to unique probe cells")
    if set(normalized_probes) & set(normalized_forbidden):
        raise ValueError("rotated probe cells and forbidden cells must be disjoint")

    case_rows: list[SealedProbeCase] = []
    for base, cell_rotation, actual in mapped:
        generated = _cell_cases(
            keys,
            cardinality,
            actual,
            normalized_forbidden,
            cycles,
        )
        if explicit_cell_rotations:
            index = normalized_cell_rotations.index(cell_rotation)
            generated = tuple(
                _retag_rotated_case(case, base, cell_rotation, index)
                for case in generated
            )
        case_rows.extend(generated)
    cases = tuple(case_rows)
    balance = _balance_certificate(cases, cardinality, normalized_probes)
    scoring_manifest_sha256 = _scoring_manifest_sha256_from_cases(cases)
    payload = _suite_digest_payload(
        protocol_status=protocol_status,
        keys=keys,
        cardinality=cardinality,
        base_probe_pairs=base_probe_pairs,
        cell_rotations=normalized_cell_rotations,
        probe_pairs=normalized_probes,
        forbidden_pairs=normalized_forbidden,
        cases=cases,
        balance=balance,
        scoring_manifest_sha256=scoring_manifest_sha256,
    )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SealedProbeSuite(
        protocol_status=protocol_status,
        num_surface_keys=keys,
        value_cardinality=cardinality,
        base_probe_pairs=base_probe_pairs,
        cell_rotations=normalized_cell_rotations,
        probe_pairs=normalized_probes,
        forbidden_pairs=normalized_forbidden,
        cases=cases,
        balance=balance,
        scoring_manifest_sha256=scoring_manifest_sha256,
        suite_sha256=digest,
    )


def _prediction_tuple(
    raw: Sequence[int], case: SealedProbeCase
) -> tuple[int, ...]:
    if isinstance(raw, (str, bytes)):
        raise TypeError("predict_queries must return a sequence of exact integers")
    try:
        result = tuple(raw)
    except TypeError as error:
        raise TypeError("predict_queries must return a finite sequence") from error
    if len(result) != len(case.expected_answers):
        raise ValueError(
            f"predictor returned {len(result)} answers for "
            f"{len(case.expected_answers)} queries"
        )
    for value in result:
        if type(value) is not int:
            raise TypeError("predicted answers must be exact integers")
        if not 0 <= value < case.program.value_cardinality:
            raise ValueError("predicted answer is outside the value vocabulary")
    return result


def evaluate_probe_suite(
    predictor: SequenceProbePredictor,
    suite: SealedProbeSuite,
) -> SealedProbeEvaluation:
    """Evaluate a frozen predictor without disclosing any trusted case fields.

    The sole callback argument is ``case.program``.  The predictor cannot see
    the case ID, family, probe pair, expected answer count, query roles,
    forbidden support, or equivalence group through this API.
    """

    if not hasattr(predictor, "predict_queries") or not callable(
        predictor.predict_queries
    ):
        raise TypeError("predictor must implement predict_queries(program)")
    if type(suite) is not SealedProbeSuite:
        raise TypeError("suite must be exact SealedProbeSuite")
    results: list[ProbeCaseResult] = []
    for case in suite.cases:
        prediction = _prediction_tuple(
            predictor.predict_queries(case.program),
            case,
        )
        results.append(
            ProbeCaseResult(
                case_id=case.case_id,
                family=case.family,
                base_probe_pair=case.base_probe_pair,
                probe_pair=case.probe_pair,
                cell_rotation=case.cell_rotation,
                value_cardinality=case.program.value_cardinality,
                expected_answers=case.expected_answers,
                predicted_answers=prediction,
                query_roles=case.query_roles,
                equivalence_group=case.equivalence_group,
                path_relation=case.path_relation,
            )
        )
    result_tuple = tuple(results)
    family_results: list[ProbeFamilyResult] = []
    for family in ProbeFamily:
        rows = tuple(row for row in result_tuple if row.family is family)
        family_results.append(
            ProbeFamilyResult(
                family=family,
                correct_count=sum(row.correct_count for row in rows),
                query_count=sum(len(row.expected_answers) for row in rows),
                focal_correct_count=sum(row.focal_correct_count for row in rows),
                focal_query_count=sum(row.focal_query_count for row in rows),
                exact_case_count=sum(row.exact_match for row in rows),
                case_count=len(rows),
            )
        )
    pair_results: list[ProbePairResult] = []
    for pair in suite.probe_pairs:
        rows = tuple(row for row in result_tuple if row.probe_pair == pair)
        pair_results.append(
            ProbePairResult(
                probe_pair=pair,
                correct_count=sum(row.correct_count for row in rows),
                query_count=sum(len(row.expected_answers) for row in rows),
                focal_correct_count=sum(row.focal_correct_count for row in rows),
                focal_query_count=sum(row.focal_query_count for row in rows),
                exact_case_count=sum(row.exact_match for row in rows),
                case_count=len(rows),
            )
        )
    equivalence_results = _equivalence_results_from_cases(result_tuple)
    if any(not row.expected_relation_satisfied for row in equivalence_results):
        raise AssertionError("trusted path group violates its declared relation")
    family_tuple = tuple(family_results)
    pair_tuple = tuple(pair_results)
    consistent_count = sum(
        row.predicted_relation_satisfied for row in equivalence_results
    )
    evaluation_fields = dict(
        protocol_status=suite.protocol_status,
        suite=suite,
        suite_sha256=suite.suite_sha256,
        scoring_manifest_sha256=suite.scoring_manifest_sha256,
        case_results=result_tuple,
        family_results=family_tuple,
        pair_results=pair_tuple,
        equivalence_results=equivalence_results,
        path_consistent_group_count=consistent_count,
        path_group_count=len(equivalence_results),
    )
    return SealedProbeEvaluation(
        **evaluation_fields,
        evaluation_sha256=_evaluation_sha256_from_fields(**evaluation_fields),
    )


class _ConstantPredictor:
    def __init__(self, value: int) -> None:
        self.value = value

    def predict_queries(self, program: VisibleProbeProgram) -> tuple[int, ...]:
        return (self.value,) * program.query_count


class _LastVisibleArgumentPredictor:
    def predict_queries(self, program: VisibleProbeProgram) -> tuple[int, ...]:
        last = 0
        result: list[int] = []
        for event in program.events:
            if event.argument >= 0:
                last = event.argument % program.value_cardinality
            if event.kind is BindingEventKind.QUERY:
                result.append(last)
        return tuple(result)


class _LatestBindEchoPredictor:
    def predict_queries(self, program: VisibleProbeProgram) -> tuple[int, ...]:
        values = [0] * program.num_surface_keys
        result: list[int] = []
        for event in program.events:
            if event.kind is BindingEventKind.BIND:
                values[event.primary_key] = event.argument
            elif event.kind is BindingEventKind.QUERY:
                result.append(values[event.primary_key])
        return tuple(result)


class _SourceBindEchoPredictor:
    def predict_queries(self, program: VisibleProbeProgram) -> tuple[int, ...]:
        values = [0] * program.num_surface_keys
        result: list[int] = []
        for event in program.events:
            if event.kind is BindingEventKind.BIND:
                values[event.primary_key] = event.argument
            elif event.kind is BindingEventKind.COPY:
                values[event.primary_key] = values[event.secondary_key]
            elif event.kind is BindingEventKind.QUERY:
                result.append(values[event.primary_key])
        return tuple(result)


def evaluate_shortcut_controls(
    suite: SealedProbeSuite,
) -> tuple[ShortcutControlResult, ...]:
    """Score declared non-compositional shortcuts on the identical cases."""

    controls: tuple[tuple[str, SequenceProbePredictor], ...] = (
        ("constant_class_0", _ConstantPredictor(0)),
        ("last_visible_argument", _LastVisibleArgumentPredictor()),
        ("latest_bind_argument_for_query_key", _LatestBindEchoPredictor()),
        ("source_key_bind_echo", _SourceBindEchoPredictor()),
    )
    results = tuple(
        ShortcutControlResult(name, evaluate_probe_suite(predictor, suite))
        for name, predictor in controls
    )
    constant = results[0].evaluation
    chance = 1.0 / suite.value_cardinality
    if not math.isclose(constant.accuracy, chance, rel_tol=0.0, abs_tol=0.0):
        raise AssertionError("constant-class control is not at exact balanced chance")
    if any(
        not math.isclose(row.accuracy, chance, rel_tol=0.0, abs_tol=0.0)
        for row in constant.family_results
    ):
        raise AssertionError("a probe family is not independently output-balanced")
    if any(result.evaluation.focal_accuracy == 1.0 for result in results):
        raise AssertionError("a declared trivial shortcut passes every focal probe")
    return results


def build_and_evaluate_balanced_probes(
    predictor: SequenceProbePredictor,
    num_surface_keys: int,
    value_cardinality: int,
    probe_pairs: Sequence[tuple[int, int]],
    *,
    forbidden_pairs: Sequence[tuple[int, int]] = (),
    cell_rotations: Sequence[ProbeCellRotation] | None = None,
    long_neutral_cycles: int = 4,
) -> tuple[SealedProbeSuite, SealedProbeEvaluation]:
    """Post-fit convenience API: disclose cells only in this evaluation call."""

    suite = build_balanced_probe_suite(
        num_surface_keys,
        value_cardinality,
        probe_pairs,
        forbidden_pairs=forbidden_pairs,
        cell_rotations=cell_rotations,
        protocol_status=ProbeProtocolStatus.RETROSPECTIVE_PROTOCOL_REHEARSAL,
        long_neutral_cycles=long_neutral_cycles,
    )
    return suite, evaluate_probe_suite(predictor, suite)


__all__ = [
    "DEFAULT_MAX_CELL_ROTATIONS",
    "DEFAULT_MAX_PROBE_CASES",
    "DEFAULT_MAX_PROBE_EVENTS",
    "DEFAULT_MAX_PROBE_WORK",
    "EquivalenceGroupResult",
    "ProbeBalanceCertificate",
    "ProbeCaseResult",
    "ProbeCellRotation",
    "ProbeFamily",
    "ProbeFamilyResult",
    "ProbeProtocolStatus",
    "ProbePathRelation",
    "ProbePairResult",
    "ProbeBudgetExceededError",
    "ProbeQueryRole",
    "ProbeSupportAudit",
    "SealedProbeCase",
    "SealedProbeEvaluation",
    "SealedProbeSuite",
    "SequenceProbePredictor",
    "ShortcutControlResult",
    "VisibleProbeEvent",
    "VisibleProbeProgram",
    "build_and_evaluate_balanced_probes",
    "build_balanced_probe_suite",
    "cyclic_cell_rotation_inventory",
    "evaluate_probe_suite",
    "evaluate_shortcut_controls",
]
