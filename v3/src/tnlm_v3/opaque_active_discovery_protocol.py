"""Controller-side evidence protocol for Phase III-T2.

This module is intentionally outside :mod:`opaque_active_discovery`'s pure
learner boundary.  It owns fresh controller entropy, salted answer
commitments, staged opening, source/runtime binding, and authoritative replay.
Neither a commitment root nor a leaf, salt, proof, nonce, semantic role, or
T1 ordering is passed to the learner constructor or learner functions.

Execution remains fail-closed unless a separately frozen, execution-ready
configuration supplies fresh inventory and exact source hashes.  Synthetic
local entropy may be used only for development tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Callable, Mapping, Sequence

from .opaque_active_discovery import (
    AutonomousPartialOperatorResult,
    OpaqueActiveDiscoveryBudgets,
    OpaqueActiveLearnerInput,
    OpaqueActiveMembershipResponse,
    OpaqueActiveNotIdentifiedResult,
    OpaqueActiveChoiceCertificate,
    make_opaque_active_input_from_rows,
    predict_defined_suffix,
    _rank_profile,
    _KnownEdge,
    _filtered_versions,
    _known_material,
    _observed_event_rows,
    _rank,
    run_opaque_active_discovery,
)
from .opaque_active_teaching_control import (
    PostfitTeachingControlResult,
    discover_postfit_teaching_control,
    validate_postfit_teaching_control,
)
from .opaque_partial_operators import (
    EnvironmentKind,
    OpaqueEdgeRequest,
    OpaqueSealedProgram,
    PartialOperatorBudgets,
    ToyPartialControllerEnvironment,
    build_full_support_controller_environment,
    build_omission_controller_environment,
)


PROTOCOL_SCHEMA = "tnlm-v3-phase3-t2-opaque-active-protocol-v1"
COMMITMENT_SCHEMA = "tnlm-v3-phase3-t2-salted-answer-commitment-v1"
PREOPEN_SCHEMA = "tnlm-v3-phase3-t2-preopen-environment-v1"
TERMINAL_SCHEMA = "tnlm-v3-phase3-t2-terminal-preopen-v1"
OPEN_SCHEMA = "tnlm-v3-phase3-t2-atomic-postfit-open-v1"
TEACHING_SUMMARY_SCHEMA = "tnlm-v3-phase3-t2-postfit-teaching-summary-v1"
TEACHING_CAMPAIGN_SCHEMA = "tnlm-v3-phase3-t2-postfit-teaching-campaign-v1"
SHORTCUT_BASELINE_SCHEMA = "tnlm-v3-phase3-t2-shortcut-baseline-v1"
SHORTCUT_T1_SCHEMA = "tnlm-v3-phase3-t2-t1-first14-nonidentification-v1"
SHORTCUT_EVIDENCE_SCHEMA = "tnlm-v3-phase3-t2-shortcut-evidence-v1"

_HEX = frozenset("0123456789abcdef")
_OMISSION_COUNT = 8
_CONTROL_COUNT = 2
_CANDIDATE_COUNT = 23
_OPENED_COUNT = 14
_INFERRED_COUNT = 1
_SEALED_COUNT = 8
_SCHEDULE = (
    (EnvironmentKind.FULL_SUPPORT_CONTROL, 0, None),
    (EnvironmentKind.FULL_SUPPORT_CONTROL, 1, None),
) + tuple((EnvironmentKind.ROTATED_OMISSION, block, cell) for block in (0, 1) for cell in ((0, 0), (0, 1), (1, 0), (1, 1)))
_SCHEDULE_LABELS = ("control:block0", "control:block1", "omission:block0:cell00", "omission:block0:cell01", "omission:block0:cell10", "omission:block0:cell11", "omission:block1:cell00", "omission:block1:cell01", "omission:block1:cell10", "omission:block1:cell11")


class T2ProtocolError(RuntimeError):
    """Raised when a controller-side evidence firewall is violated."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _require_digest(name: str, value: object) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in _HEX for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build one JSON object while rejecting aliases through duplicate keys."""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _load_closed_canonical_json(path: Path) -> object:
    """Load an artifact only when its bytes are its unique canonical encoding."""

    encoded = path.read_bytes()
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("artifact is not strict UTF-8 JSON") from error
    value = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonfinite_json,
    )
    if encoded != _canonical(value):
        raise ValueError("artifact is not in canonical JSON byte form")
    return value


def _write_closed_canonical_json(path: Path, value: object) -> None:
    """Write once atomically, allowing only byte-identical idempotent resume."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical(value)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != encoded:
            raise T2ProtocolError("refusing to replace an existing non-identical evidence artifact")
        return
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # Recheck after the temporary write so a concurrent differing artifact
        # cannot be silently replaced during an official resume.
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != encoded:
                raise T2ProtocolError("refusing to replace a concurrently created evidence artifact")
            Path(temporary_name).unlink()
            temporary_name = None
            return
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    if destination.read_bytes() != encoded:
        raise T2ProtocolError("atomic artifact write did not preserve canonical bytes")


def _require_closed_object(value: object, expected: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{name} schema is not closed")
    return value


def _require_exact_list(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an exact JSON array")
    return value


def _require_exact_string(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact JSON string")
    return value


def _require_exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact JSON boolean")
    return value


def _require_exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact JSON integer")
    return value


def _request_payload(request: OpaqueEdgeRequest) -> dict[str, object]:
    return {
        "schema": request.schema,
        "source_word": list(request.source_word),
        "event_token": request.event_token,
        "program": list(request.program),
        "request_sha256": request.request_sha256,
    }


def _leaf_payload(request: OpaqueEdgeRequest, answers: tuple[str, ...], salt: str) -> dict[str, object]:
    return {
        "schema": COMMITMENT_SCHEMA,
        "request": _request_payload(request),
        "target_answers": list(answers),
        "salt": salt,
    }


def _tree_parent(left: str, right: str) -> str:
    return _digest({"schema": COMMITMENT_SCHEMA, "left": left, "right": right})


@dataclass(frozen=True)
class T2SourceRuntimeBinding:
    """Immutable source and runtime inventory required before execution."""

    required_file_sha256s: tuple[tuple[str, str], ...]
    python_version: str
    platform: str
    torch_version: str
    pyyaml_version: str
    device: str
    binding_sha256: str

    def __post_init__(self) -> None:
        if not self.required_file_sha256s:
            raise ValueError("source binding requires at least one source path")
        if tuple(sorted(self.required_file_sha256s)) != self.required_file_sha256s:
            raise ValueError("source binding paths must have canonical order")
        for path, digest in self.required_file_sha256s:
            if type(path) is not str or not path or Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("source binding path must be a safe relative path")
            _require_digest("source file digest", digest)
        if type(self.python_version) is not str or type(self.platform) is not str or type(self.torch_version) is not str or type(self.pyyaml_version) is not str or self.device != "cpu":
            raise TypeError("runtime binding fields must be strings")
        expected = _digest(self.payload(False))
        if _require_digest("binding_sha256", self.binding_sha256) != expected:
            raise ValueError("source/runtime binding digest mismatch")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "required_file_sha256s": [list(row) for row in self.required_file_sha256s],
            "python_version": self.python_version,
            "platform": self.platform,
            "torch_version": self.torch_version,
            "pyyaml_version": self.pyyaml_version,
            "device": self.device,
        }
        if include_digest:
            result["binding_sha256"] = self.binding_sha256
        return result

    def verify(self, repository_root: Path) -> None:
        root = repository_root.resolve()
        actual_paths = {str(path.relative_to(root)).replace("\\", "/") for path in (root / "v3" / "src" / "tnlm_v3").glob("*.py")}
        actual_paths.update({"v3/pyproject.toml", "v3/scripts/run_phase3_t2_opaque_active_discovery.py"})
        if {path for path, _ in self.required_file_sha256s} != actual_paths:
            raise T2ProtocolError("frozen source closure no longer matches exact package inventory")
        for relative, expected in self.required_file_sha256s:
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as error:
                raise T2ProtocolError("source-binding path escapes repository") from error
            if not candidate.is_file() or sha256(candidate.read_bytes()).hexdigest() != expected:
                raise T2ProtocolError(f"source binding changed: {relative}")
        import torch, yaml
        if self.python_version != platform.python_version() or self.platform != platform.platform() or self.torch_version != str(torch.__version__) or self.pyyaml_version != str(yaml.__version__):
            raise T2ProtocolError("runtime inventory differs from frozen protocol")


def make_source_runtime_binding(required_file_sha256s: Mapping[str, str]) -> T2SourceRuntimeBinding:
    import torch, yaml
    rows = tuple(sorted((str(path), _require_digest("source digest", digest)) for path, digest in required_file_sha256s.items()))
    payload = {
        "required_file_sha256s": [list(row) for row in rows],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "pyyaml_version": str(yaml.__version__),
        "device": "cpu",
    }
    return T2SourceRuntimeBinding(rows, payload["python_version"], payload["platform"], payload["torch_version"], payload["pyyaml_version"], payload["device"], _digest(payload))


def make_source_runtime_binding_from_repository(repository_root: Path) -> T2SourceRuntimeBinding:
    """Freeze the complete local package closure used by the T2 runner."""
    root = repository_root.resolve()
    paths = sorted((root / "v3" / "src" / "tnlm_v3").glob("*.py"))
    paths.extend((root / relative for relative in ("v3/pyproject.toml", "v3/scripts/run_phase3_t2_opaque_active_discovery.py")))
    if any(not path.is_file() for path in paths):
        raise T2ProtocolError("cannot freeze an incomplete T2 source closure")
    return make_source_runtime_binding({str(path.relative_to(root)).replace("\\", "/"): sha256(path.read_bytes()).hexdigest() for path in paths})


@dataclass(frozen=True)
class SaltedAnswerLeaf:
    request_sha256: str
    target_answers: tuple[str, ...]
    salt: str
    leaf_sha256: str

    def __post_init__(self) -> None:
        _require_digest("request_sha256", self.request_sha256)
        if not isinstance(self.target_answers, tuple) or len(self.target_answers) != 2:
            raise ValueError("a committed answer must contain two categorical labels")
        if type(self.salt) is not str or len(self.salt) < 32:
            raise ValueError("answer salt must contain at least 128 bits of encoded entropy")
        _require_digest("leaf_sha256", self.leaf_sha256)


@dataclass(frozen=True)
class MerkleProof:
    """Controller-private inclusion proof for one salted answer leaf."""

    leaf_index: int
    sibling_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.leaf_index) is not int or self.leaf_index < 0:
            raise ValueError("Merkle proof index must be nonnegative")
        for sibling in self.sibling_sha256s:
            _require_digest("Merkle proof sibling", sibling)


@dataclass(frozen=True)
class SaltedAnswerCommitment:
    """Request-SHA ordered hiding commitment for one omission environment."""

    leaves: tuple[SaltedAnswerLeaf, ...]
    merkle_root_sha256: str
    commitment_sha256: str

    def __post_init__(self) -> None:
        if len(self.leaves) != _CANDIDATE_COUNT:
            raise ValueError("each omission commitment must bind exactly 23 answers")
        if tuple(row.request_sha256 for row in self.leaves) != tuple(sorted(row.request_sha256 for row in self.leaves)):
            raise ValueError("commitment leaves must use request-SHA order")
        if len({row.request_sha256 for row in self.leaves}) != _CANDIDATE_COUNT:
            raise ValueError("commitment request inventory is not unique")
        if len({row.salt for row in self.leaves}) != _CANDIDATE_COUNT:
            raise ValueError("each committed answer needs a distinct salt")
        root = _merkle_root(tuple(row.leaf_sha256 for row in self.leaves))
        if _require_digest("merkle_root_sha256", self.merkle_root_sha256) != root:
            raise ValueError("salted-answer Merkle root mismatch")
        expected = _digest(self.payload(False))
        if _require_digest("commitment_sha256", self.commitment_sha256) != expected:
            raise ValueError("salted-answer commitment digest mismatch")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": COMMITMENT_SCHEMA,
            "leaves": [
                {
                    "request_sha256": row.request_sha256,
                    "target_answers": list(row.target_answers),
                    "salt": row.salt,
                    "leaf_sha256": row.leaf_sha256,
                }
                for row in self.leaves
            ],
            "merkle_root_sha256": self.merkle_root_sha256,
        }
        if include_digest:
            result["commitment_sha256"] = self.commitment_sha256
        return result

    def verify_opening(self, request: OpaqueEdgeRequest, answers: tuple[str, ...]) -> SaltedAnswerLeaf:
        matches = [row for row in self.leaves if row.request_sha256 == request.request_sha256]
        if len(matches) != 1:
            raise T2ProtocolError("opened request is absent from the precommitted pool")
        leaf = matches[0]
        expected_leaf = _digest(_leaf_payload(request, answers, leaf.salt))
        if leaf.target_answers != answers or leaf.leaf_sha256 != expected_leaf:
            raise T2ProtocolError("opened answer fails its salted commitment leaf")
        self.verify_proof(leaf.leaf_sha256, self.proof_for(request.request_sha256))
        return leaf

    def proof_for(self, request_sha256: str) -> MerkleProof:
        _require_digest("request_sha256", request_sha256)
        index = next((i for i, leaf in enumerate(self.leaves) if leaf.request_sha256 == request_sha256), None)
        if index is None:
            raise T2ProtocolError("cannot prove a request absent from the commitment")
        position = index
        level = [row.leaf_sha256 for row in self.leaves]
        siblings: list[str] = []
        while len(level) > 1:
            if len(level) % 2:
                level.append(level[-1])
            siblings.append(level[position ^ 1])
            level = [_tree_parent(level[i], level[i + 1]) for i in range(0, len(level), 2)]
            position //= 2
        return MerkleProof(index, tuple(siblings))

    def verify_proof(self, leaf_sha256: str, proof: MerkleProof) -> None:
        _require_digest("Merkle leaf", leaf_sha256)
        if type(proof) is not MerkleProof or proof.leaf_index >= len(self.leaves):
            raise T2ProtocolError("invalid controller-side Merkle proof")
        value = leaf_sha256
        position = proof.leaf_index
        for sibling in proof.sibling_sha256s:
            value = _tree_parent(value, sibling) if position % 2 == 0 else _tree_parent(sibling, value)
            position //= 2
        if value != self.merkle_root_sha256:
            raise T2ProtocolError("opened answer fails commitment inclusion proof")


def _merkle_root(leaves: tuple[str, ...]) -> str:
    if not leaves:
        raise ValueError("Merkle commitment requires at least one leaf")
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [_tree_parent(level[index], level[index + 1]) for index in range(0, len(level), 2)]
    return level[0]


def make_salted_answer_commitment(
    requests: Sequence[OpaqueEdgeRequest],
    answers_by_request: Mapping[str, tuple[str, ...]],
    salts_by_request: Mapping[str, str],
) -> SaltedAnswerCommitment:
    rows = tuple(sorted(tuple(requests), key=lambda request: request.request_sha256))
    if len(rows) != _CANDIDATE_COUNT or len({row.request_sha256 for row in rows}) != _CANDIDATE_COUNT:
        raise ValueError("commitment requires exactly the 23 distinct candidate requests")
    leaves: list[SaltedAnswerLeaf] = []
    for request in rows:
        answers = answers_by_request.get(request.request_sha256)
        salt = salts_by_request.get(request.request_sha256)
        if not isinstance(answers, tuple) or type(salt) is not str:
            raise ValueError("every candidate needs an answer and a salt")
        leaf = _digest(_leaf_payload(request, answers, salt))
        leaves.append(SaltedAnswerLeaf(request.request_sha256, answers, salt, leaf))
    tuple_leaves = tuple(leaves)
    root = _merkle_root(tuple(row.leaf_sha256 for row in tuple_leaves))
    payload = {
        "schema": COMMITMENT_SCHEMA,
        "leaves": [
            {"request_sha256": row.request_sha256, "target_answers": list(row.target_answers), "salt": row.salt, "leaf_sha256": row.leaf_sha256}
            for row in tuple_leaves
        ],
        "merkle_root_sha256": root,
    }
    return SaltedAnswerCommitment(tuple_leaves, root, _digest(payload))


@dataclass(frozen=True)
class Phase3T2Protocol:
    execution_ready: bool
    schedule_labels: tuple[str, ...]
    controller_nonces: tuple[str, ...]
    omission_salts: tuple[tuple[str, ...], ...]
    omission_commitment_roots: tuple[str, ...]
    long_suite_roots: tuple[str, ...]
    source_runtime_binding: T2SourceRuntimeBinding | None
    active_budgets: OpaqueActiveDiscoveryBudgets
    min_unopened_candidates: int
    protocol_sha256: str
    schema: str = PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROTOCOL_SCHEMA:
            raise ValueError("unknown Phase III-T2 protocol schema")
        if type(self.execution_ready) is not bool:
            raise TypeError("execution_ready must be exact bool")
        if self.schedule_labels != _SCHEDULE_LABELS:
            raise ValueError("T2 protocol schedule must use the frozen ten-arm canonical order")
        if self.execution_ready:
            if len(self.controller_nonces) != _CONTROL_COUNT + _OMISSION_COUNT or len(set(self.controller_nonces)) != 10:
                raise ValueError("ready protocol requires ten distinct controller nonces")
            for nonce in self.controller_nonces:
                _require_digest("controller nonce", nonce)
            if len(self.omission_salts) != _OMISSION_COUNT or any(len(row) != _CANDIDATE_COUNT for row in self.omission_salts):
                raise ValueError("ready protocol requires 8x23 controller salts")
            flat_salts = tuple(salt for row in self.omission_salts for salt in row)
            if len(set(flat_salts)) != _OMISSION_COUNT * _CANDIDATE_COUNT:
                raise ValueError("ready protocol requires 184 distinct controller salts")
            for salt in flat_salts:
                _require_digest("controller salt", salt)
            if len(self.omission_commitment_roots) != _OMISSION_COUNT:
                raise ValueError("ready protocol requires eight precomputed commitment roots")
            for root in self.omission_commitment_roots:
                _require_digest("precomputed commitment root", root)
            if len(self.long_suite_roots) != 10:
                raise ValueError("ready protocol requires ten precomputed long-suite roots")
            for root in self.long_suite_roots:
                _require_digest("precomputed long-suite root", root)
            if self.source_runtime_binding is None:
                raise ValueError("ready protocol requires a source/runtime binding")
        else:
            if self.controller_nonces or self.omission_salts or self.omission_commitment_roots or self.long_suite_roots or self.source_runtime_binding is not None:
                raise ValueError("unready protocol must contain placeholders only, not entropy or bindings")
        if type(self.active_budgets) is not OpaqueActiveDiscoveryBudgets:
            raise TypeError("protocol requires exact active budgets")
        if (self.active_budgets.max_active_calls, self.active_budgets.max_structural_inferences, self.active_budgets.max_returned_categorical_tokens) != (14, 1, 28):
            raise ValueError("official T2 budgets are exactly 14 calls, one inference, and 28 labels")
        if self.min_unopened_candidates != _SEALED_COUNT:
            raise ValueError("official T2 protocol must retain eight sealed candidates")
        expected = _digest(self.payload(False))
        if _require_digest("protocol_sha256", self.protocol_sha256) != expected:
            raise ValueError("T2 protocol digest mismatch")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": self.schema,
            "execution_ready": self.execution_ready,
            "schedule_labels": list(self.schedule_labels),
            "controller_nonces": list(self.controller_nonces),
            "omission_salts": [list(row) for row in self.omission_salts],
            "omission_commitment_roots": list(self.omission_commitment_roots),
            "long_suite_roots": list(self.long_suite_roots),
            "source_runtime_binding": None if self.source_runtime_binding is None else self.source_runtime_binding.payload(),
            "active_budgets": self.active_budgets.payload(),
            "min_unopened_candidates": self.min_unopened_candidates,
        }
        if include_digest:
            value["protocol_sha256"] = self.protocol_sha256
        return value


def make_placeholder_protocol() -> Phase3T2Protocol:
    budgets = OpaqueActiveDiscoveryBudgets(max_active_calls=14, max_structural_inferences=1, max_returned_categorical_tokens=28)
    payload = {"schema": PROTOCOL_SCHEMA, "execution_ready": False, "schedule_labels": list(_SCHEDULE_LABELS), "controller_nonces": [], "omission_salts": [], "omission_commitment_roots": [], "long_suite_roots": [], "source_runtime_binding": None, "active_budgets": budgets.payload(), "min_unopened_candidates": 8}
    return Phase3T2Protocol(False, _SCHEDULE_LABELS, (), (), (), (), None, budgets, 8, _digest(payload))


def _fresh_active_input(controller: ToyPartialControllerEnvironment, budgets: OpaqueActiveDiscoveryBudgets) -> OpaqueActiveLearnerInput:
    source = controller.learner_input
    # Pass only primitive opaque rows.  The controller object, nonce, semantic
    # correspondence, old input digest and candidate order never cross this line.
    return make_opaque_active_input_from_rows(
        event_tokens=source.event_tokens,
        query_tokens=source.query_tokens,
        answer_tokens=source.answer_tokens,
        passive_state_observations=source.passive_state_observations,
        passive_edge_observations=source.passive_edge_observations,
        candidate_requests=source.candidate_edge_requests,
        defined_requests=source.defined_edge_requests,
        undefined_requests=source.undefined_edge_requests,
        budgets=budgets,
    )


def _answers_by_candidate(controller: ToyPartialControllerEnvironment) -> dict[str, tuple[str, ...]]:
    result = {row.request.request_sha256: row.target_answers for row in controller.active_responses}
    by_program = {row.program: row.expected_answers for row in controller.sealed_edge_programs}
    for request in controller.learner_input.candidate_edge_requests:
        if request.request_sha256 not in result and request.program in by_program:
            result[request.request_sha256] = by_program[request.program]
    if len(result) != _CANDIDATE_COUNT:
        raise T2ProtocolError("controller cannot supply exactly 23 candidate answers")
    return result


def _open_one_committed_candidate_answer(controller: ToyPartialControllerEnvironment, request: OpaqueEdgeRequest) -> tuple[str, ...]:
    """Read exactly one controller answer after its pure choice is committed."""
    for row in controller.active_responses:
        if row.request.request_sha256 == request.request_sha256:
            return row.target_answers
    for row in controller.sealed_edge_programs:
        if row.program == request.program:
            return row.expected_answers
    raise T2ProtocolError("controller has no answer for the chosen candidate request")


@dataclass(frozen=True)
class T2ControllerEnvironment:
    """Private execution wrapper; never a pure-learner argument."""

    controller: ToyPartialControllerEnvironment
    learner_input: OpaqueActiveLearnerInput
    commitment: SaltedAnswerCommitment | None
    long_path_programs: tuple[OpaqueSealedProgram, ...]
    long_suite_root_sha256: str

    def __post_init__(self) -> None:
        omission = self.controller.kind is EnvironmentKind.ROTATED_OMISSION
        if omission != (self.commitment is not None):
            raise ValueError("only omission arms own a candidate-answer commitment")
        if omission and len(self.learner_input.canonical_candidate_requests) != _CANDIDATE_COUNT:
            raise ValueError("omission learner input must retain all 23 candidates")
        if not omission and self.learner_input.canonical_candidate_requests:
            raise ValueError("full-support control must have no candidate pool")
        if self.long_path_programs != self.controller.long_path_programs:
            raise ValueError("controller long/path inventory mismatch")
        if self.long_suite_root_sha256 != _long_suite_root(self.long_path_programs):
            raise ValueError("long suite root mismatch")


def _long_suite_root(rows: Sequence[OpaqueSealedProgram]) -> str:
    if len(rows) != 12:
        raise ValueError("long suite must contain twelve programs")
    return _digest({"schema": "tnlm-v3-phase3-t2-long-suite-v1", "rows": [row.payload() for row in rows]})


def build_t2_controller_environment(
    *,
    kind: EnvironmentKind,
    relabel_block: int,
    omitted_cell: tuple[int, int] | None,
    controller_nonce: str,
    active_budgets: OpaqueActiveDiscoveryBudgets,
    salts: Sequence[str] | None = None,
    expected_commitment_root_sha256: str | None = None,
    expected_long_suite_root_sha256: str | None = None,
) -> T2ControllerEnvironment:
    if kind is EnvironmentKind.FULL_SUPPORT_CONTROL:
        if omitted_cell is not None or salts not in (None, (), []):
            raise ValueError("controls cannot carry omitted cells or answer salts")
        controller = build_full_support_controller_environment(relabel_block, controller_nonce=controller_nonce, budgets=PartialOperatorBudgets())
        if expected_commitment_root_sha256 is not None:
            raise ValueError("controls cannot bind a candidate commitment root")
    elif kind is EnvironmentKind.ROTATED_OMISSION:
        if omitted_cell not in ((0, 0), (0, 1), (1, 0), (1, 1)) or salts is None or len(salts) != _CANDIDATE_COUNT:
            raise ValueError("omission arms require one cell and 23 salts")
        controller = build_omission_controller_environment(omitted_cell, relabel_block, controller_nonce=controller_nonce, budgets=PartialOperatorBudgets())
    else:
        raise ValueError("unsupported T2 arm kind")
    learner_input = _fresh_active_input(controller, active_budgets)
    long_root = _long_suite_root(controller.long_path_programs)
    if expected_long_suite_root_sha256 is not None and long_root != _require_digest("expected long suite root", expected_long_suite_root_sha256):
        raise T2ProtocolError("precomputed long-suite root does not bind this controller")
    commitment: SaltedAnswerCommitment | None = None
    if kind is EnvironmentKind.ROTATED_OMISSION:
        requests = learner_input.canonical_candidate_requests
        answer_map = _answers_by_candidate(controller)
        salt_map = {request.request_sha256: salt for request, salt in zip(requests, salts, strict=True)}
        commitment = make_salted_answer_commitment(requests, answer_map, salt_map)
        if expected_commitment_root_sha256 is not None and commitment.merkle_root_sha256 != _require_digest("expected commitment root", expected_commitment_root_sha256):
            raise T2ProtocolError("precomputed commitment root does not bind this controller before learner invocation")
    return T2ControllerEnvironment(controller, learner_input, commitment, controller.long_path_programs, long_root)


@dataclass(frozen=True)
class T2PreopenEnvironment:
    schedule_label: str
    controller_sha256: str
    learner_input_sha256: str
    result_sha256: str
    model_sha256: str | None
    long_suite_root_sha256: str
    commitment_root_sha256: str | None
    result: AutonomousPartialOperatorResult | OpaqueActiveNotIdentifiedResult
    opened_request_sha256s: tuple[str, ...]
    preopen_sha256: str

    def __post_init__(self) -> None:
        if self.schedule_label not in _SCHEDULE_LABELS:
            raise ValueError("unknown preopen schedule label")
        for value in (self.controller_sha256, self.learner_input_sha256, self.result_sha256, self.long_suite_root_sha256):
            _require_digest("preopen binding", value)
        _require_digest("learner_input_sha256", self.learner_input_sha256)
        if self.commitment_root_sha256 is not None:
            _require_digest("commitment_root_sha256", self.commitment_root_sha256)
        if self.result.learner_input_sha256 != self.learner_input_sha256:
            raise ValueError("preopen result/input mismatch")
        if self.result.result_sha256 != self.result_sha256:
            raise ValueError("preopen result digest does not bind result")
        if isinstance(self.result, AutonomousPartialOperatorResult):
            if self.model_sha256 != self.result.model.model_sha256:
                raise ValueError("preopen model digest does not bind result")
            opened = tuple(step.response.request.request_sha256 for step in self.result.final_state.steps if hasattr(step, "response"))
            if opened != self.opened_request_sha256s:
                raise ValueError("preopen opening order does not bind transcript")
        elif self.model_sha256 is not None:
            raise ValueError("unidentified preopen cannot bind a model")
        if tuple(sorted(self.opened_request_sha256s)) != tuple(sorted(set(self.opened_request_sha256s))):
            raise ValueError("preopen opened-request inventory contains duplicates")
        expected = _digest(self.payload(False))
        if _require_digest("preopen_sha256", self.preopen_sha256) != expected:
            raise ValueError("preopen record digest mismatch")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        result_digest = self.result.result_sha256
        payload: dict[str, object] = {"schema": PREOPEN_SCHEMA, "schedule_label": self.schedule_label, "controller_sha256": self.controller_sha256, "learner_input_sha256": self.learner_input_sha256, "result_sha256": self.result_sha256, "model_sha256": self.model_sha256, "long_suite_root_sha256": self.long_suite_root_sha256, "commitment_root_sha256": self.commitment_root_sha256, "identification_status": self.result.identification_status, "opened_request_sha256s": list(self.opened_request_sha256s)}
        if include_digest:
            payload["preopen_sha256"] = self.preopen_sha256
        return payload


@dataclass(frozen=True)
class T2PreopenRecord:
    """Closed disk-safe summary of a preopen run.

    The record deliberately contains no learner object and no candidate answer,
    inferred answer, sealed answer, proof, salt, or long-program expectation.
    Its digest is the same preopen digest produced in memory, so the sole
    postfit opener can require exact equality after deterministic replay.
    """

    schedule_label: str
    controller_sha256: str
    learner_input_sha256: str
    result_sha256: str
    model_sha256: str
    long_suite_root_sha256: str
    commitment_root_sha256: str | None
    identification_status: str
    opened_request_sha256s: tuple[str, ...]
    preopen_sha256: str

    def __post_init__(self) -> None:
        if type(self.schedule_label) is not str or self.schedule_label not in _SCHEDULE_LABELS:
            raise ValueError("unknown preopen record schedule label")
        for name, value in (
            ("controller_sha256", self.controller_sha256),
            ("learner_input_sha256", self.learner_input_sha256),
            ("result_sha256", self.result_sha256),
            ("model_sha256", self.model_sha256),
            ("long_suite_root_sha256", self.long_suite_root_sha256),
        ):
            _require_digest(name, value)
        if self.commitment_root_sha256 is not None:
            _require_digest("commitment_root_sha256", self.commitment_root_sha256)
        if type(self.identification_status) is not str or self.identification_status != "identified":
            raise ValueError("a staged preopen record must bind an identified model")
        if type(self.opened_request_sha256s) is not tuple:
            raise TypeError("opened_request_sha256s must be an exact tuple")
        for value in self.opened_request_sha256s:
            _require_digest("opened request", value)
        if len(set(self.opened_request_sha256s)) != len(self.opened_request_sha256s):
            raise ValueError("preopen record opened-request inventory has duplicates")
        if self.schedule_label.startswith("control:"):
            if self.commitment_root_sha256 is not None or self.opened_request_sha256s:
                raise ValueError("control preopen record cannot expose candidate inventory")
        elif self.commitment_root_sha256 is None or len(self.opened_request_sha256s) != _OPENED_COUNT:
            raise ValueError("omission preopen record requires one root and fourteen openings")
        if _require_digest("preopen_sha256", self.preopen_sha256) != _digest(self.payload(False)):
            raise ValueError("preopen record digest mismatch")

    @classmethod
    def from_preopen(cls, preopen: T2PreopenEnvironment) -> "T2PreopenRecord":
        if type(preopen) is not T2PreopenEnvironment:
            raise TypeError("preopen must be exact T2PreopenEnvironment")
        if not isinstance(preopen.result, AutonomousPartialOperatorResult):
            raise T2ProtocolError("cannot persist an unidentified preopen result")
        assert preopen.model_sha256 is not None
        return cls(
            schedule_label=preopen.schedule_label,
            controller_sha256=preopen.controller_sha256,
            learner_input_sha256=preopen.learner_input_sha256,
            result_sha256=preopen.result_sha256,
            model_sha256=preopen.model_sha256,
            long_suite_root_sha256=preopen.long_suite_root_sha256,
            commitment_root_sha256=preopen.commitment_root_sha256,
            identification_status=preopen.result.identification_status,
            opened_request_sha256s=preopen.opened_request_sha256s,
            preopen_sha256=preopen.preopen_sha256,
        )

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": PREOPEN_SCHEMA,
            "schedule_label": self.schedule_label,
            "controller_sha256": self.controller_sha256,
            "learner_input_sha256": self.learner_input_sha256,
            "result_sha256": self.result_sha256,
            "model_sha256": self.model_sha256,
            "long_suite_root_sha256": self.long_suite_root_sha256,
            "commitment_root_sha256": self.commitment_root_sha256,
            "identification_status": self.identification_status,
            "opened_request_sha256s": list(self.opened_request_sha256s),
        }
        if include_digest:
            value["preopen_sha256"] = self.preopen_sha256
        return value


def run_t2_preopen_environment(environment: T2ControllerEnvironment, *, minimum_unopened_candidates: int = 8) -> T2PreopenEnvironment:
    """Run acquisition only; no inferred/sealed/long answer is opened here."""

    if environment.controller.kind is EnvironmentKind.FULL_SUPPORT_CONTROL:
        result = run_opaque_active_discovery(environment.learner_input, lambda choice: (_ for _ in ()).throw(T2ProtocolError("controls must not request a membership answer")), minimum_unopened_candidate_count=0)
        root: str | None = None
        opened: tuple[str, ...] = ()
    else:
        commitment = environment.commitment
        assert commitment is not None
        opened_rows: list[str] = []
        def provider(choice: OpaqueActiveChoiceCertificate) -> tuple[str, ...]:
            answer = _open_one_committed_candidate_answer(environment.controller, choice.chosen_request)
            commitment.verify_opening(choice.chosen_request, answer)
            opened_rows.append(choice.chosen_request.request_sha256)
            return answer
        result = run_opaque_active_discovery(environment.learner_input, provider, minimum_unopened_candidate_count=minimum_unopened_candidates)
        root = commitment.merkle_root_sha256
        opened = tuple(opened_rows)
    if environment.controller.kind is EnvironmentKind.FULL_SUPPORT_CONTROL:
        label = f"control:block{environment.controller.relabel_block}"
    else:
        cell = environment.controller.pseudoheldout_cell
        assert cell is not None
        label = f"omission:block{environment.controller.relabel_block}:cell{cell[0]}{cell[1]}"
    model_sha = result.model.model_sha256 if isinstance(result, AutonomousPartialOperatorResult) else None
    payload = {"schema": PREOPEN_SCHEMA, "schedule_label": label, "controller_sha256": environment.controller.controller_sha256, "learner_input_sha256": environment.learner_input.input_sha256, "result_sha256": result.result_sha256, "model_sha256": model_sha, "long_suite_root_sha256": environment.long_suite_root_sha256, "commitment_root_sha256": root, "identification_status": result.identification_status, "opened_request_sha256s": list(opened)}
    return T2PreopenEnvironment(label, environment.controller.controller_sha256, environment.learner_input.input_sha256, result.result_sha256, model_sha, environment.long_suite_root_sha256, root, result, opened, _digest(payload))


@dataclass(frozen=True)
class _OmissionTerminalPreopen:
    omission_preopen_sha256s: tuple[str, ...]
    all_models_frozen_before_postfit: bool
    terminal_sha256: str

    def __post_init__(self) -> None:
        if len(self.omission_preopen_sha256s) != _OMISSION_COUNT:
            raise ValueError("terminal preopen must bind eight omissions")
        if not self.all_models_frozen_before_postfit:
            raise ValueError("terminal record must precede every postfit opening")
        expected = _digest(self.payload(False))
        if _require_digest("terminal_sha256", self.terminal_sha256) != expected:
            raise ValueError("terminal preopen digest mismatch")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {"schema": TERMINAL_SCHEMA, "omission_preopen_sha256s": list(self.omission_preopen_sha256s), "all_models_frozen_before_postfit": self.all_models_frozen_before_postfit}
        if include_digest:
            value["terminal_sha256"] = self.terminal_sha256
        return value


def _freeze_omission_terminal_preopen(preopens: Sequence[T2PreopenEnvironment]) -> _OmissionTerminalPreopen:
    if len(preopens) != _OMISSION_COUNT:
        raise ValueError("exactly eight omission preopens are required")
    if any(row.result.identification_status != "identified" for row in preopens):
        raise T2ProtocolError("cannot freeze terminal record from an unidentified omission")
    if any(len(row.opened_request_sha256s) != _OPENED_COUNT for row in preopens):
        raise T2ProtocolError("terminal record requires exactly fourteen prefit openings per omission")
    rows = tuple(row.preopen_sha256 for row in preopens)
    payload = {"schema": TERMINAL_SCHEMA, "omission_preopen_sha256s": list(rows), "all_models_frozen_before_postfit": True}
    return _OmissionTerminalPreopen(rows, True, _digest(payload))


@dataclass(frozen=True)
class T2AtomicPostfitOpen:
    terminal_preopen_sha256: str
    verified_candidate_leaf_count: int
    inferred_answer_count: int
    sealed_edge_count: int
    long_path_count: int
    open_sha256: str

    def __post_init__(self) -> None:
        _require_digest("terminal_preopen_sha256", self.terminal_preopen_sha256)
        if (self.verified_candidate_leaf_count, self.inferred_answer_count, self.sealed_edge_count, self.long_path_count) != (_CANDIDATE_COUNT, _INFERRED_COUNT, _SEALED_COUNT, 12):
            raise ValueError("atomic postfit opening inventory mismatch")
        expected = _digest(self.payload(False))
        if _require_digest("open_sha256", self.open_sha256) != expected:
            raise ValueError("postfit opening digest mismatch")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {"schema": OPEN_SCHEMA, "terminal_preopen_sha256": self.terminal_preopen_sha256, "verified_candidate_leaf_count": self.verified_candidate_leaf_count, "inferred_answer_count": self.inferred_answer_count, "sealed_edge_count": self.sealed_edge_count, "long_path_count": self.long_path_count}
        if include_digest:
            value["open_sha256"] = self.open_sha256
        return value


def _open_t2_postfit_environment(environment: T2ControllerEnvironment, preopen: T2PreopenEnvironment, campaign_terminal_sha256: str) -> T2AtomicPostfitOpen:
    """Atomically verify all candidate leaves after terminal model freeze."""

    if environment.controller.kind is not EnvironmentKind.ROTATED_OMISSION or environment.commitment is None:
        raise ValueError("postfit opening applies only to committed omission environments")
    _require_digest("campaign terminal", campaign_terminal_sha256)
    if preopen.result.identification_status != "identified" or len(preopen.opened_request_sha256s) != _OPENED_COUNT:
        raise T2ProtocolError("postfit batch requires a complete fourteen-answer preopen model")
    result = preopen.result
    assert isinstance(result, AutonomousPartialOperatorResult)
    answers = _answers_by_candidate(environment.controller)
    requests = environment.learner_input.canonical_candidate_requests
    for request in requests:
        environment.commitment.verify_opening(request, answers[request.request_sha256])
    inferred = [row for row in result.final_state.steps if hasattr(row, "inference")]
    if len(inferred) != _INFERRED_COUNT:
        raise T2ProtocolError("postfit batch requires exactly one answer-sidecar-free inference")
    inferred_row = inferred[0].inference
    if answers[inferred_row.request.request_sha256] != inferred_row.inferred_target_answers:
        raise T2ProtocolError("postfit inferred answer disagrees with its committed sidecar")
    unopened = set(request.request_sha256 for request in requests) - set(preopen.opened_request_sha256s) - {inferred_row.request.request_sha256}
    if len(unopened) != _SEALED_COUNT:
        raise T2ProtocolError("postfit independent sealed-edge count is not eight")
    payload = {"schema": OPEN_SCHEMA, "terminal_preopen_sha256": campaign_terminal_sha256, "verified_candidate_leaf_count": 23, "inferred_answer_count": 1, "sealed_edge_count": 8, "long_path_count": len(environment.long_path_programs)}
    return T2AtomicPostfitOpen(campaign_terminal_sha256, 23, 1, 8, len(environment.long_path_programs), _digest(payload))


def _controller_legal_answers(environment: T2ControllerEnvironment) -> dict[str, tuple[str, ...]]:
    rows = {row.request.request_sha256: row.target_answers for row in environment.controller.learner_input.passive_edge_observations}
    if environment.controller.kind is EnvironmentKind.ROTATED_OMISSION:
        rows.update(_answers_by_candidate(environment.controller))
    if len(rows) != 44:
        raise T2ProtocolError("controller does not close the complete legal graph")
    return rows


def _evaluate_model_against_controller(environment: T2ControllerEnvironment, result: AutonomousPartialOperatorResult, max_suffix_events: int) -> tuple[int, int, int]:
    """Controller-only exact guarded graph plus long-program evaluation."""
    expected = _controller_legal_answers(environment)
    known = dict(result.model.mask_source_answer_rows)
    legal_correct = 0
    for request in environment.learner_input.canonical_defined_requests:
        predicted = predict_defined_suffix(result, known[request.source_word], (request.event_token,), max_events=max_suffix_events)
        legal_correct += int(predicted == expected[request.request_sha256])
    guarded = result.guarded_language
    undefined_correct = 46 if guarded.all_undefined_pairs_rejected and guarded.undefined_pair_count == 46 else 0
    initial = environment.controller.initial_answers
    long_correct = sum(int(predict_defined_suffix(result, initial, row.program, max_events=max_suffix_events) == row.expected_answers) for row in environment.long_path_programs)
    return legal_correct, undefined_correct, long_correct


@dataclass(frozen=True)
class T2CampaignTerminalPreopen:
    protocol_sha256: str
    source_runtime_binding_sha256: str
    scheduled_preopen_sha256s: tuple[str, ...]
    terminal_sha256: str

    def __post_init__(self) -> None:
        if len(self.scheduled_preopen_sha256s) != 10 or len(set(self.scheduled_preopen_sha256s)) != 10:
            raise ValueError("campaign terminal must bind ten unique scheduled preopens")
        _require_digest("protocol_sha256", self.protocol_sha256)
        _require_digest("source_runtime_binding_sha256", self.source_runtime_binding_sha256)
        if _require_digest("terminal_sha256", self.terminal_sha256) != _digest(self.payload(False)):
            raise ValueError("campaign terminal digest mismatch")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {"schema": TERMINAL_SCHEMA, "protocol_sha256": self.protocol_sha256, "source_runtime_binding_sha256": self.source_runtime_binding_sha256, "scheduled_preopen_sha256s": list(self.scheduled_preopen_sha256s), "all_ten_models_frozen_before_postfit": True}
        if include_digest: value["terminal_sha256"] = self.terminal_sha256
        return value


def freeze_t2_campaign_terminal(protocol: Phase3T2Protocol, preopens: Sequence[T2PreopenEnvironment]) -> T2CampaignTerminalPreopen:
    if not protocol.execution_ready or protocol.source_runtime_binding is None:
        raise T2ProtocolError("cannot freeze a campaign terminal from an unready protocol")
    if len(preopens) != 10 or len({row.preopen_sha256 for row in preopens}) != 10:
        raise T2ProtocolError("campaign terminal requires ten distinct scheduled preopens")
    if tuple(row.schedule_label for row in preopens) != protocol.schedule_labels:
        raise T2ProtocolError("preopen descriptors do not match frozen schedule order")
    if any(row.result.identification_status != "identified" for row in preopens):
        raise T2ProtocolError("all controls and omissions must identify before postfit")
    if any(len(row.opened_request_sha256s) != 14 for row in preopens[2:]) or any(row.opened_request_sha256s for row in preopens[:2]):
        raise T2ProtocolError("preopen schedule violates control/omission opening policy")
    payload = {"schema": TERMINAL_SCHEMA, "protocol_sha256": protocol.protocol_sha256, "source_runtime_binding_sha256": protocol.source_runtime_binding.binding_sha256, "scheduled_preopen_sha256s": [row.preopen_sha256 for row in preopens], "all_ten_models_frozen_before_postfit": True}
    return T2CampaignTerminalPreopen(protocol.protocol_sha256, protocol.source_runtime_binding.binding_sha256, tuple(payload["scheduled_preopen_sha256s"]), _digest(payload))


def _require_schedule_index(index: object) -> int:
    if type(index) is not int or not 0 <= index < len(_SCHEDULE):
        raise ValueError("T2 schedule index must be an exact integer from 0 through 9")
    return index


def _build_t2_scheduled_environment(
    protocol: Phase3T2Protocol,
    index: int,
) -> T2ControllerEnvironment:
    """Build one already-source-validated arm from the frozen schedule."""

    _require_schedule_index(index)
    kind, block, cell = _SCHEDULE[index]
    omission_index = index - _CONTROL_COUNT
    return build_t2_controller_environment(
        kind=kind,
        relabel_block=block,
        omitted_cell=cell,
        controller_nonce=protocol.controller_nonces[index],
        active_budgets=protocol.active_budgets,
        salts=None if index < _CONTROL_COUNT else protocol.omission_salts[omission_index],
        expected_commitment_root_sha256=(
            None if index < _CONTROL_COUNT else protocol.omission_commitment_roots[omission_index]
        ),
        expected_long_suite_root_sha256=protocol.long_suite_roots[index],
    )


def build_t2_scheduled_environment(
    protocol: Phase3T2Protocol,
    repository_root: Path,
    index: int,
) -> T2ControllerEnvironment:
    """Source-validate and deterministically rebuild one scheduled controller arm."""

    if type(protocol) is not Phase3T2Protocol:
        raise TypeError("protocol must be exact Phase3T2Protocol")
    require_execution_ready(protocol, repository_root)
    return _build_t2_scheduled_environment(protocol, _require_schedule_index(index))


def _reconstruct_t2_preopen(
    protocol: Phase3T2Protocol,
    index: int,
) -> tuple[T2ControllerEnvironment, T2PreopenEnvironment, T2PreopenRecord]:
    environment = _build_t2_scheduled_environment(protocol, index)
    preopen = run_t2_preopen_environment(
        environment,
        minimum_unopened_candidates=(0 if index < _CONTROL_COUNT else protocol.min_unopened_candidates),
    )
    return environment, preopen, T2PreopenRecord.from_preopen(preopen)


def reconstruct_t2_preopen_record(
    protocol: Phase3T2Protocol,
    repository_root: Path,
    index: int,
) -> T2PreopenRecord:
    """Re-run one ready arm and return its answer-free canonical summary."""

    if type(protocol) is not Phase3T2Protocol:
        raise TypeError("protocol must be exact Phase3T2Protocol")
    require_execution_ready(protocol, repository_root)
    return _reconstruct_t2_preopen(protocol, _require_schedule_index(index))[2]


def _validate_t2_preopen_record_binding(
    protocol: Phase3T2Protocol,
    index: int,
    record: T2PreopenRecord,
) -> None:
    if type(record) is not T2PreopenRecord:
        raise TypeError("preopen record must be exact T2PreopenRecord")
    if record.schedule_label != protocol.schedule_labels[index]:
        raise T2ProtocolError("preopen record is not in the frozen schedule position")
    if record.long_suite_root_sha256 != protocol.long_suite_roots[index]:
        raise T2ProtocolError("preopen record long-suite root differs from protocol")
    expected_commitment = (
        None if index < _CONTROL_COUNT else protocol.omission_commitment_roots[index - _CONTROL_COUNT]
    )
    if record.commitment_root_sha256 != expected_commitment:
        raise T2ProtocolError("preopen record commitment root differs from protocol")
    expected_opened = 0 if index < _CONTROL_COUNT else _OPENED_COUNT
    if len(record.opened_request_sha256s) != expected_opened:
        raise T2ProtocolError("preopen record opening inventory differs from schedule")


def aggregate_t2_preopen_records(
    protocol: Phase3T2Protocol,
    repository_root: Path,
    records: Sequence[T2PreopenRecord],
) -> T2CampaignTerminalPreopen:
    """Freeze the exact ten-record inventory without opening postfit sidecars."""

    if type(protocol) is not Phase3T2Protocol:
        raise TypeError("protocol must be exact Phase3T2Protocol")
    require_execution_ready(protocol, repository_root)
    if type(records) not in (tuple, list) or len(records) != len(_SCHEDULE):
        raise T2ProtocolError("aggregate-preopen requires exactly ten staged records")
    tuple_records = tuple(records)
    for index, record in enumerate(tuple_records):
        _validate_t2_preopen_record_binding(protocol, index, record)
    preopen_sha256s = tuple(record.preopen_sha256 for record in tuple_records)
    if len(set(preopen_sha256s)) != len(_SCHEDULE):
        raise T2ProtocolError("aggregate-preopen requires ten distinct preopen digests")
    assert protocol.source_runtime_binding is not None
    payload = {
        "schema": TERMINAL_SCHEMA,
        "protocol_sha256": protocol.protocol_sha256,
        "source_runtime_binding_sha256": protocol.source_runtime_binding.binding_sha256,
        "scheduled_preopen_sha256s": list(preopen_sha256s),
        "all_ten_models_frozen_before_postfit": True,
    }
    return T2CampaignTerminalPreopen(
        protocol.protocol_sha256,
        protocol.source_runtime_binding.binding_sha256,
        preopen_sha256s,
        _digest(payload),
    )


def write_t2_preopen_record(path: Path, record: T2PreopenRecord) -> None:
    if type(record) is not T2PreopenRecord:
        raise TypeError("record must be exact T2PreopenRecord")
    _write_closed_canonical_json(path, record.payload())


def load_t2_preopen_record(path: Path) -> T2PreopenRecord:
    raw = _require_closed_object(
        _load_closed_canonical_json(path),
        frozenset(
            {
                "schema",
                "schedule_label",
                "controller_sha256",
                "learner_input_sha256",
                "result_sha256",
                "model_sha256",
                "long_suite_root_sha256",
                "commitment_root_sha256",
                "identification_status",
                "opened_request_sha256s",
                "preopen_sha256",
            }
        ),
        "preopen artifact",
    )
    if raw["schema"] != PREOPEN_SCHEMA:
        raise ValueError("unknown preopen artifact schema")
    opened_raw = _require_exact_list(raw["opened_request_sha256s"], "opened_request_sha256s")
    opened = tuple(_require_exact_string(value, "opened request") for value in opened_raw)
    commitment = raw["commitment_root_sha256"]
    if commitment is not None:
        commitment = _require_exact_string(commitment, "commitment_root_sha256")
    return T2PreopenRecord(
        _require_exact_string(raw["schedule_label"], "schedule_label"),
        _require_exact_string(raw["controller_sha256"], "controller_sha256"),
        _require_exact_string(raw["learner_input_sha256"], "learner_input_sha256"),
        _require_exact_string(raw["result_sha256"], "result_sha256"),
        _require_exact_string(raw["model_sha256"], "model_sha256"),
        _require_exact_string(raw["long_suite_root_sha256"], "long_suite_root_sha256"),
        commitment,
        _require_exact_string(raw["identification_status"], "identification_status"),
        opened,
        _require_exact_string(raw["preopen_sha256"], "preopen_sha256"),
    )


def write_t2_campaign_terminal_preopen(path: Path, terminal: T2CampaignTerminalPreopen) -> None:
    if type(terminal) is not T2CampaignTerminalPreopen:
        raise TypeError("terminal must be exact T2CampaignTerminalPreopen")
    _write_closed_canonical_json(path, terminal.payload())


def load_t2_campaign_terminal_preopen(path: Path) -> T2CampaignTerminalPreopen:
    raw = _require_closed_object(
        _load_closed_canonical_json(path),
        frozenset(
            {
                "schema",
                "protocol_sha256",
                "source_runtime_binding_sha256",
                "scheduled_preopen_sha256s",
                "all_ten_models_frozen_before_postfit",
                "terminal_sha256",
            }
        ),
        "terminal preopen artifact",
    )
    if raw["schema"] != TERMINAL_SCHEMA:
        raise ValueError("unknown terminal preopen artifact schema")
    if not _require_exact_bool(
        raw["all_ten_models_frozen_before_postfit"],
        "all_ten_models_frozen_before_postfit",
    ):
        raise ValueError("terminal artifact must freeze all ten models")
    scheduled_raw = _require_exact_list(
        raw["scheduled_preopen_sha256s"],
        "scheduled_preopen_sha256s",
    )
    return T2CampaignTerminalPreopen(
        _require_exact_string(raw["protocol_sha256"], "protocol_sha256"),
        _require_exact_string(
            raw["source_runtime_binding_sha256"],
            "source_runtime_binding_sha256",
        ),
        tuple(_require_exact_string(value, "scheduled preopen digest") for value in scheduled_raw),
        _require_exact_string(raw["terminal_sha256"], "terminal_sha256"),
    )


# Short aliases retain the same exact closed-record implementation.
write_t2_campaign_terminal = write_t2_campaign_terminal_preopen
load_t2_campaign_terminal = load_t2_campaign_terminal_preopen


def t2_preopen_record_path(directory: Path, index: int) -> Path:
    return Path(directory) / f"preopen-environment-{_require_schedule_index(index):02d}.json"


def load_t2_preopen_record_set(directory: Path) -> tuple[T2PreopenRecord, ...]:
    root = Path(directory)
    expected = tuple(t2_preopen_record_path(root, index) for index in range(len(_SCHEDULE)))
    actual = tuple(sorted(root.glob("preopen-environment-*.json"))) if root.is_dir() else ()
    if actual != expected:
        raise T2ProtocolError("preopen directory does not contain the exact ten-file inventory")
    return tuple(load_t2_preopen_record(path) for path in expected)


@dataclass(frozen=True)
class T2PredictionRow:
    row_kind: str
    item_sha256: str
    expected_answers: tuple[str, ...] | None
    predicted_answers: tuple[str, ...] | None
    exact: bool

    def __post_init__(self) -> None:
        if self.row_kind not in ("defined_edge", "undefined_pair", "long_path"): raise ValueError("unknown prediction row kind")
        _require_digest("prediction item", self.item_sha256)
        if self.row_kind == "undefined_pair":
            if self.expected_answers is not None or self.predicted_answers is not None or not self.exact: raise ValueError("undefined row must certify rejection")
        elif not isinstance(self.expected_answers, tuple) or not isinstance(self.predicted_answers, tuple) or self.exact != (self.expected_answers == self.predicted_answers):
            raise ValueError("prediction row mismatch")

    def payload(self) -> dict[str, object]:
        return {"row_kind": self.row_kind, "item_sha256": self.item_sha256, "expected_answers": None if self.expected_answers is None else list(self.expected_answers), "predicted_answers": None if self.predicted_answers is None else list(self.predicted_answers), "exact": self.exact}


@dataclass(frozen=True)
class T2OpeningRecord:
    role: str
    access_ordinal: int
    request_sha256: str
    target_answers: tuple[str, ...]
    salt: str
    leaf_index: int
    choice_sha256: str
    response_sha256: str
    leaf_sha256: str
    proof_siblings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.role not in ("queried", "inferred", "sealed") or type(self.access_ordinal) is not int or self.access_ordinal < 1: raise ValueError("opening role/ordinal invalid")
        for value in (self.request_sha256, self.choice_sha256, self.response_sha256, self.leaf_sha256): _require_digest("opening transcript", value)
        if not isinstance(self.target_answers, tuple) or len(self.target_answers) != 2 or type(self.salt) is not str or self.leaf_index < 0: raise ValueError("opening leaf material invalid")
        for value in self.proof_siblings: _require_digest("opening proof", value)

    def payload(self) -> dict[str, object]:
        return {"role": self.role, "access_ordinal": self.access_ordinal, "request_sha256": self.request_sha256, "target_answers": list(self.target_answers), "salt": self.salt, "leaf_index": self.leaf_index, "choice_sha256": self.choice_sha256, "response_sha256": self.response_sha256, "leaf_sha256": self.leaf_sha256, "proof_siblings": list(self.proof_siblings)}


@dataclass(frozen=True)
class T2PostfitTeachingSummary:
    """Typed omission-arm summary of the truth-aware teaching control."""

    learner_input_sha256: str
    teaching_result_sha256: str
    complete_candidate_answer_map_sha256: str
    primary_reconstruction_result_sha256: str
    counterfactual_truth_selected_query_count: int
    answer_free_singleton_inference_count: int
    counterfactual_unqueried_count: int
    closed_restricted_map_count: int
    rank_closed_event_count: int
    counterfactual_returned_categorical_label_count: int
    new_membership_calls_made: int
    causal_primary_isolated: bool
    truth_specific_noncausal_control: bool
    selection_eligible: bool
    confirmatory_claim_eligible: bool
    global_query_minimality_claimed: bool
    arbitrary_total_operator_constructed: bool
    summary_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("learner_input_sha256", self.learner_input_sha256),
            ("teaching_result_sha256", self.teaching_result_sha256),
            ("complete_candidate_answer_map_sha256", self.complete_candidate_answer_map_sha256),
            ("primary_reconstruction_result_sha256", self.primary_reconstruction_result_sha256),
        ):
            _require_digest(name, value)
        if (
            self.counterfactual_truth_selected_query_count,
            self.answer_free_singleton_inference_count,
            self.counterfactual_unqueried_count,
            self.closed_restricted_map_count,
            self.rank_closed_event_count,
            self.counterfactual_returned_categorical_label_count,
            self.new_membership_calls_made,
        ) != (13, 2, 8, 10, 10, 26, 0):
            raise ValueError("postfit teaching summary must bind exact 13Q+2I+8 accounting")
        for name in (
            "counterfactual_truth_selected_query_count",
            "answer_free_singleton_inference_count",
            "counterfactual_unqueried_count",
            "closed_restricted_map_count",
            "rank_closed_event_count",
            "counterfactual_returned_categorical_label_count",
            "new_membership_calls_made",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer")
        for name, value, required in (
            ("causal_primary_isolated", self.causal_primary_isolated, True),
            ("truth_specific_noncausal_control", self.truth_specific_noncausal_control, True),
            ("selection_eligible", self.selection_eligible, False),
            ("confirmatory_claim_eligible", self.confirmatory_claim_eligible, False),
            ("global_query_minimality_claimed", self.global_query_minimality_claimed, False),
            ("arbitrary_total_operator_constructed", self.arbitrary_total_operator_constructed, False),
        ):
            if type(value) is not bool or value is not required:
                raise ValueError(f"{name} must be {required}")
        if _require_digest("teaching summary", self.summary_sha256) != _digest(self.payload(False)):
            raise ValueError("postfit teaching summary digest mismatch")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": TEACHING_SUMMARY_SCHEMA,
            "learner_input_sha256": self.learner_input_sha256,
            "teaching_result_sha256": self.teaching_result_sha256,
            "complete_candidate_answer_map_sha256": self.complete_candidate_answer_map_sha256,
            "primary_reconstruction_result_sha256": self.primary_reconstruction_result_sha256,
            "counterfactual_truth_selected_query_count": self.counterfactual_truth_selected_query_count,
            "answer_free_singleton_inference_count": self.answer_free_singleton_inference_count,
            "counterfactual_unqueried_count": self.counterfactual_unqueried_count,
            "closed_restricted_map_count": self.closed_restricted_map_count,
            "rank_closed_event_count": self.rank_closed_event_count,
            "counterfactual_returned_categorical_label_count": self.counterfactual_returned_categorical_label_count,
            "new_membership_calls_made": self.new_membership_calls_made,
            "causal_primary_isolated": self.causal_primary_isolated,
            "truth_specific_noncausal_control": self.truth_specific_noncausal_control,
            "selection_eligible": self.selection_eligible,
            "confirmatory_claim_eligible": self.confirmatory_claim_eligible,
            "global_query_minimality_claimed": self.global_query_minimality_claimed,
            "arbitrary_total_operator_constructed": self.arbitrary_total_operator_constructed,
        }
        if include_digest:
            value["summary_sha256"] = self.summary_sha256
        return value


def _summarize_postfit_teaching_control(
    result: PostfitTeachingControlResult,
    primary_result_sha256: str,
) -> T2PostfitTeachingSummary:
    if type(result) is not PostfitTeachingControlResult:
        raise TypeError("teaching result must be exact PostfitTeachingControlResult")
    if result.primary_reconstruction_result_sha256 != primary_result_sha256:
        raise T2ProtocolError("teaching control reconstructed a different primary result")
    queried = tuple(row for row in result.teaching_decisions if row.acquisition_kind == "queried")
    inferred = tuple(row for row in result.teaching_decisions if row.acquisition_kind == "singleton_inferred")
    truth_selected = sum(row.complete_truth_map_used_to_select_request for row in queried)
    zero_label_inferences = sum(row.returned_categorical_label_count == 0 for row in inferred)
    singleton_events = sum(count == 1 for _, count, _ in result.final_event_version_rows)
    rank_closed_events = sum(observed == legal for _, observed, legal in result.final_event_rank_rows)
    if result.total_operator is not None:
        raise T2ProtocolError("postfit teaching control unexpectedly constructed a total operator")
    payload = {
        "schema": TEACHING_SUMMARY_SCHEMA,
        "learner_input_sha256": result.learner_input_sha256,
        "teaching_result_sha256": result.result_sha256,
        "complete_candidate_answer_map_sha256": result.complete_candidate_answer_map_sha256,
        "primary_reconstruction_result_sha256": result.primary_reconstruction_result_sha256,
        "counterfactual_truth_selected_query_count": truth_selected,
        "answer_free_singleton_inference_count": zero_label_inferences,
        "counterfactual_unqueried_count": result.unopened_count,
        "closed_restricted_map_count": singleton_events,
        "rank_closed_event_count": rank_closed_events,
        "counterfactual_returned_categorical_label_count": result.counterfactual_returned_categorical_label_count,
        "new_membership_calls_made": result.new_membership_calls_made,
        "causal_primary_isolated": result.primary_selector_received_teaching_choices is False and result.primary_result_unchanged_by_control and result.primary_posthoc_selector_flag_remains_false,
        "truth_specific_noncausal_control": result.truth_specific_noncausal_control,
        "selection_eligible": result.selection_eligible,
        "confirmatory_claim_eligible": result.confirmatory_claim_eligible,
        "global_query_minimality_claimed": result.global_query_minimality_claimed,
        "arbitrary_total_operator_constructed": result.arbitrary_total_operator_constructed,
    }
    return T2PostfitTeachingSummary(
        **{key: value for key, value in payload.items() if key != "schema"},
        summary_sha256=_digest(payload),
    )


@dataclass(frozen=True)
class T2CampaignTeachingSummary:
    arm_summary_sha256s: tuple[str, ...]
    omission_control_count: int
    total_counterfactual_truth_selected_queries: int
    total_answer_free_singleton_inferences: int
    total_counterfactual_unqueried: int
    total_counterfactual_returned_categorical_labels: int
    total_closed_restricted_maps: int
    total_rank_closed_events: int
    total_new_membership_calls: int
    all_causal_primary_isolated: bool
    all_truth_specific_noncausal: bool
    all_selection_ineligible: bool
    no_global_query_minimality_claim: bool
    no_arbitrary_total_operator: bool
    summary_sha256: str

    def __post_init__(self) -> None:
        if len(self.arm_summary_sha256s) != _OMISSION_COUNT:
            raise ValueError("campaign teaching summary must bind eight omission summaries")
        for value in self.arm_summary_sha256s:
            _require_digest("arm teaching summary", value)
        if (
            self.omission_control_count,
            self.total_counterfactual_truth_selected_queries,
            self.total_answer_free_singleton_inferences,
            self.total_counterfactual_unqueried,
            self.total_counterfactual_returned_categorical_labels,
            self.total_closed_restricted_maps,
            self.total_rank_closed_events,
            self.total_new_membership_calls,
        ) != (8, 104, 16, 64, 208, 80, 80, 0):
            raise ValueError("campaign teaching-control aggregate accounting mismatch")
        for name in (
            "omission_control_count",
            "total_counterfactual_truth_selected_queries",
            "total_answer_free_singleton_inferences",
            "total_counterfactual_unqueried",
            "total_counterfactual_returned_categorical_labels",
            "total_closed_restricted_maps",
            "total_rank_closed_events",
            "total_new_membership_calls",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer")
        for name in (
            "all_causal_primary_isolated",
            "all_truth_specific_noncausal",
            "all_selection_ineligible",
            "no_global_query_minimality_claim",
            "no_arbitrary_total_operator",
        ):
            if type(getattr(self, name)) is not bool or not getattr(self, name):
                raise ValueError(f"{name} must be true")
        if _require_digest("campaign teaching summary", self.summary_sha256) != _digest(self.payload(False)):
            raise ValueError("campaign teaching summary digest mismatch")

    @classmethod
    def from_arms(cls, rows: Sequence[T2PostfitTeachingSummary]) -> "T2CampaignTeachingSummary":
        if len(rows) != _OMISSION_COUNT or any(type(row) is not T2PostfitTeachingSummary for row in rows):
            raise TypeError("campaign teaching summary requires eight exact arm summaries")
        payload = {
            "schema": TEACHING_CAMPAIGN_SCHEMA,
            "arm_summary_sha256s": [row.summary_sha256 for row in rows],
            "omission_control_count": len(rows),
            "total_counterfactual_truth_selected_queries": sum(row.counterfactual_truth_selected_query_count for row in rows),
            "total_answer_free_singleton_inferences": sum(row.answer_free_singleton_inference_count for row in rows),
            "total_counterfactual_unqueried": sum(row.counterfactual_unqueried_count for row in rows),
            "total_counterfactual_returned_categorical_labels": sum(row.counterfactual_returned_categorical_label_count for row in rows),
            "total_closed_restricted_maps": sum(row.closed_restricted_map_count for row in rows),
            "total_rank_closed_events": sum(row.rank_closed_event_count for row in rows),
            "total_new_membership_calls": sum(row.new_membership_calls_made for row in rows),
            "all_causal_primary_isolated": all(row.causal_primary_isolated for row in rows),
            "all_truth_specific_noncausal": all(row.truth_specific_noncausal_control for row in rows),
            "all_selection_ineligible": all(not row.selection_eligible and not row.confirmatory_claim_eligible for row in rows),
            "no_global_query_minimality_claim": all(not row.global_query_minimality_claimed for row in rows),
            "no_arbitrary_total_operator": all(not row.arbitrary_total_operator_constructed for row in rows),
        }
        return cls(
            tuple(payload["arm_summary_sha256s"]),
            payload["omission_control_count"],
            payload["total_counterfactual_truth_selected_queries"],
            payload["total_answer_free_singleton_inferences"],
            payload["total_counterfactual_unqueried"],
            payload["total_counterfactual_returned_categorical_labels"],
            payload["total_closed_restricted_maps"],
            payload["total_rank_closed_events"],
            payload["total_new_membership_calls"],
            payload["all_causal_primary_isolated"],
            payload["all_truth_specific_noncausal"],
            payload["all_selection_ineligible"],
            payload["no_global_query_minimality_claim"],
            payload["no_arbitrary_total_operator"],
            _digest(payload),
        )

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": TEACHING_CAMPAIGN_SCHEMA,
            "arm_summary_sha256s": list(self.arm_summary_sha256s),
            "omission_control_count": self.omission_control_count,
            "total_counterfactual_truth_selected_queries": self.total_counterfactual_truth_selected_queries,
            "total_answer_free_singleton_inferences": self.total_answer_free_singleton_inferences,
            "total_counterfactual_unqueried": self.total_counterfactual_unqueried,
            "total_counterfactual_returned_categorical_labels": self.total_counterfactual_returned_categorical_labels,
            "total_closed_restricted_maps": self.total_closed_restricted_maps,
            "total_rank_closed_events": self.total_rank_closed_events,
            "total_new_membership_calls": self.total_new_membership_calls,
            "all_causal_primary_isolated": self.all_causal_primary_isolated,
            "all_truth_specific_noncausal": self.all_truth_specific_noncausal,
            "all_selection_ineligible": self.all_selection_ineligible,
            "no_global_query_minimality_claim": self.no_global_query_minimality_claim,
            "no_arbitrary_total_operator": self.no_arbitrary_total_operator,
        }
        if include_digest:
            value["summary_sha256"] = self.summary_sha256
        return value


@dataclass(frozen=True)
class T2ShortcutBaselineEvaluation:
    baseline_kind: str
    deterministic_fit_rule: str
    fit_model_sha256: str
    fit_edge_count: int
    fit_error_count: int
    heldout_edge_count: int
    heldout_error_count: int
    undefined_pair_false_accept_count: int
    baseline_failed: bool
    evaluation_sha256: str

    def __post_init__(self) -> None:
        rules = {
            "constant_mode": "global_categorical_mode_canonical_tie_break",
            "identity": "source_diagnostic_identity_no_target_fit",
            "event_mode": "per_event_categorical_mode_with_global_fallback",
            "source_mode": "per_source_categorical_mode_with_global_fallback",
        }
        if self.baseline_kind not in rules or self.deterministic_fit_rule != rules[self.baseline_kind]:
            raise ValueError("unknown shortcut baseline kind")
        _require_digest("shortcut fit model", self.fit_model_sha256)
        for name in ("fit_edge_count", "fit_error_count", "heldout_edge_count", "heldout_error_count", "undefined_pair_false_accept_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{name} must be an exact nonnegative integer")
        if self.fit_edge_count not in (35, 44) or not 0 <= self.fit_error_count <= self.fit_edge_count:
            raise ValueError("shortcut fit accounting mismatch")
        if self.heldout_edge_count not in (0, 9) or not 0 <= self.heldout_error_count <= self.heldout_edge_count:
            raise ValueError("shortcut heldout accounting mismatch")
        if self.heldout_edge_count == 9 and self.heldout_error_count < 1:
            raise ValueError("omission shortcut must fail on independently heldout legal edges")
        if self.undefined_pair_false_accept_count != 46:
            raise ValueError("total shortcut must falsely accept all 46 guarded-undefined pairs")
        failed = self.fit_error_count > 0 or self.heldout_error_count > 0 or self.undefined_pair_false_accept_count > 0
        if type(self.baseline_failed) is not bool or self.baseline_failed is not failed or not failed:
            raise ValueError("shortcut failure flag does not match evaluation counts")
        if _require_digest("shortcut evaluation", self.evaluation_sha256) != _digest(self.payload(False)):
            raise ValueError("shortcut evaluation digest mismatch")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": SHORTCUT_BASELINE_SCHEMA,
            "baseline_kind": self.baseline_kind,
            "deterministic_fit_rule": self.deterministic_fit_rule,
            "fit_model_sha256": self.fit_model_sha256,
            "fit_edge_count": self.fit_edge_count,
            "fit_error_count": self.fit_error_count,
            "heldout_edge_count": self.heldout_edge_count,
            "heldout_error_count": self.heldout_error_count,
            "undefined_pair_false_accept_count": self.undefined_pair_false_accept_count,
            "baseline_failed": self.baseline_failed,
        }
        if include_digest:
            value["evaluation_sha256"] = self.evaluation_sha256
        return value


@dataclass(frozen=True)
class T2T1First14Nonidentification:
    ordered_request_sha256s: tuple[str, ...]
    observed_source_missing_count: int
    final_event_version_counts: tuple[tuple[str, int], ...]
    final_event_rank_rows: tuple[tuple[str, int, int], ...]
    singleton_event_count: int
    non_singleton_event_count: int
    posterior_global_version_mass: int
    identified: bool
    exact_restricted_version_evaluation: bool
    controller_supplied_postfit_negative_control: bool
    baseline_sha256: str

    def __post_init__(self) -> None:
        if len(self.ordered_request_sha256s) != 14 or len(set(self.ordered_request_sha256s)) != 14:
            raise ValueError("T1-first14 baseline must bind fourteen distinct ordered requests")
        for value in self.ordered_request_sha256s:
            _require_digest("T1 request", value)
        if type(self.observed_source_missing_count) is not int or self.observed_source_missing_count != 0:
            raise ValueError("T1-first14 baseline must observe every requested source")
        if len(self.final_event_version_counts) != 10 or tuple(sorted(self.final_event_version_counts)) != self.final_event_version_counts:
            raise ValueError("T1-first14 version counts must bind ten canonical events")
        for token, count in self.final_event_version_counts:
            if type(token) is not str or type(count) is not int or count < 1:
                raise TypeError("T1-first14 version row has invalid types")
        counts = tuple(count for _, count in self.final_event_version_counts)
        if counts.count(1) != 9 or sorted(counts) != [1] * 9 + [9]:
            raise ValueError("T1-first14 must leave exactly one nine-version event")
        if (self.singleton_event_count, self.non_singleton_event_count, self.posterior_global_version_mass) != (9, 1, 9):
            raise ValueError("T1-first14 nonidentification accounting mismatch")
        if len(self.final_event_rank_rows) != 10 or tuple(token for token, _, _ in self.final_event_rank_rows) != tuple(token for token, _ in self.final_event_version_counts):
            raise ValueError("T1-first14 rank rows must align with version rows")
        deficient = 0
        for _, observed, legal in self.final_event_rank_rows:
            if type(observed) is not int or type(legal) is not int or observed < 1 or legal < 1 or observed > legal:
                raise TypeError("T1-first14 rank row is invalid")
            deficient += observed < legal
        if deficient != 1:
            raise ValueError("T1-first14 must leave exactly one rank-deficient event")
        if type(self.identified) is not bool or self.identified:
            raise ValueError("T1-first14 baseline must remain unidentified")
        if type(self.exact_restricted_version_evaluation) is not bool or not self.exact_restricted_version_evaluation:
            raise ValueError("T1-first14 baseline must use exact restricted versions")
        if type(self.controller_supplied_postfit_negative_control) is not bool or not self.controller_supplied_postfit_negative_control:
            raise ValueError("T1-first14 is controller-supplied postfit negative-control evidence")
        if _require_digest("T1-first14 baseline", self.baseline_sha256) != _digest(self.payload(False)):
            raise ValueError("T1-first14 baseline digest mismatch")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": SHORTCUT_T1_SCHEMA,
            "ordered_request_sha256s": list(self.ordered_request_sha256s),
            "observed_source_missing_count": self.observed_source_missing_count,
            "final_event_version_counts": [list(row) for row in self.final_event_version_counts],
            "final_event_rank_rows": [list(row) for row in self.final_event_rank_rows],
            "singleton_event_count": self.singleton_event_count,
            "non_singleton_event_count": self.non_singleton_event_count,
            "posterior_global_version_mass": self.posterior_global_version_mass,
            "identified": self.identified,
            "exact_restricted_version_evaluation": self.exact_restricted_version_evaluation,
            "controller_supplied_postfit_negative_control": self.controller_supplied_postfit_negative_control,
        }
        if include_digest:
            value["baseline_sha256"] = self.baseline_sha256
        return value


@dataclass(frozen=True)
class T2ShortcutEvidence:
    learner_input_sha256: str
    primary_result_sha256: str
    passive_fit_request_sha256s: tuple[str, ...]
    primary_visible_fit_request_sha256s: tuple[str, ...]
    inferred_eval_request_sha256s: tuple[str, ...]
    sealed_eval_request_sha256s: tuple[str, ...]
    baseline_rows: tuple[T2ShortcutBaselineEvaluation, ...]
    t1_first14: T2T1First14Nonidentification | None
    legal_holdout_applicable: bool
    t1_first14_applicable: bool
    all_shortcuts_fail: bool
    evidence_sha256: str

    def __post_init__(self) -> None:
        _require_digest("shortcut learner input", self.learner_input_sha256)
        _require_digest("shortcut primary result", self.primary_result_sha256)
        inventories = (
            self.passive_fit_request_sha256s,
            self.primary_visible_fit_request_sha256s,
            self.inferred_eval_request_sha256s,
            self.sealed_eval_request_sha256s,
        )
        if any(type(row) is not tuple for row in inventories):
            raise TypeError("shortcut request inventories must be exact tuples")
        for value in (item for row in inventories for item in row):
            _require_digest("shortcut request", value)
        combined = tuple(item for row in inventories for item in row)
        if len(set(combined)) != len(combined) or len(combined) != 44:
            raise ValueError("shortcut fit/evaluation partitions must disjointly cover 44 legal edges")
        omission = bool(self.primary_visible_fit_request_sha256s)
        if type(self.legal_holdout_applicable) is not bool or self.legal_holdout_applicable is not omission:
            raise ValueError("shortcut legal-holdout applicability mismatch")
        if type(self.t1_first14_applicable) is not bool or self.t1_first14_applicable is not omission:
            raise ValueError("shortcut T1 applicability mismatch")
        expected_counts = (21, 14, 1, 8) if omission else (44, 0, 0, 0)
        if tuple(len(row) for row in inventories) != expected_counts:
            raise ValueError("shortcut fit/evaluation inventory counts mismatch")
        if tuple(row.baseline_kind for row in self.baseline_rows) != ("constant_mode", "identity", "event_mode", "source_mode"):
            raise ValueError("shortcut baseline rows must use frozen canonical order")
        if any(type(row) is not T2ShortcutBaselineEvaluation for row in self.baseline_rows):
            raise TypeError("shortcut baseline rows must be exact")
        if any(row.fit_edge_count != sum(expected_counts[:2]) or row.heldout_edge_count != sum(expected_counts[2:]) for row in self.baseline_rows):
            raise ValueError("shortcut baseline rows do not bind shared split counts")
        if omission != (type(self.t1_first14) is T2T1First14Nonidentification):
            raise ValueError("only omission arms require the actual T1-first14 baseline")
        failed = all(row.baseline_failed for row in self.baseline_rows) and (not omission or not self.t1_first14.identified)
        if type(self.all_shortcuts_fail) is not bool or self.all_shortcuts_fail is not failed or not failed:
            raise ValueError("shortcut aggregate failure flag mismatch")
        if _require_digest("shortcut evidence", self.evidence_sha256) != _digest(self.payload(False)):
            raise ValueError("shortcut evidence digest mismatch")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": SHORTCUT_EVIDENCE_SCHEMA,
            "learner_input_sha256": self.learner_input_sha256,
            "primary_result_sha256": self.primary_result_sha256,
            "passive_fit_request_sha256s": list(self.passive_fit_request_sha256s),
            "primary_visible_fit_request_sha256s": list(self.primary_visible_fit_request_sha256s),
            "inferred_eval_request_sha256s": list(self.inferred_eval_request_sha256s),
            "sealed_eval_request_sha256s": list(self.sealed_eval_request_sha256s),
            "baseline_rows": [row.payload() for row in self.baseline_rows],
            "t1_first14": None if self.t1_first14 is None else self.t1_first14.payload(),
            "legal_holdout_applicable": self.legal_holdout_applicable,
            "t1_first14_applicable": self.t1_first14_applicable,
            "all_shortcuts_fail": self.all_shortcuts_fail,
        }
        if include_digest:
            value["evidence_sha256"] = self.evidence_sha256
        return value


def _categorical_mode(rows: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
    counts: dict[tuple[str, ...], int] = {}
    for row in rows:
        counts[row] = counts.get(row, 0) + 1
    if not counts:
        raise ValueError("cannot fit a categorical mode from no rows")
    return min(counts, key=lambda row: (-counts[row], row))


def _make_shortcut_baseline_evaluation(
    kind: str,
    training_rows: Sequence[tuple[OpaqueEdgeRequest, tuple[str, ...]]],
    heldout_rows: Sequence[tuple[OpaqueEdgeRequest, tuple[str, ...]]],
    known_source_answers: Mapping[tuple[str, ...], tuple[str, ...]],
    undefined_requests: Sequence[OpaqueEdgeRequest],
) -> T2ShortcutBaselineEvaluation:
    fit_rules = {
        "constant_mode": "global_categorical_mode_canonical_tie_break",
        "identity": "source_diagnostic_identity_no_target_fit",
        "event_mode": "per_event_categorical_mode_with_global_fallback",
        "source_mode": "per_source_categorical_mode_with_global_fallback",
    }
    fallback = _categorical_mode(tuple(target for _, target in training_rows))
    if kind == "constant_mode":
        parameters: object = {"fallback": list(fallback)}
        predict = lambda request: fallback
    elif kind == "identity":
        source_rows = tuple(sorted((word, known_source_answers[word]) for word in {request.source_word for request, _ in training_rows + tuple(heldout_rows)}))
        parameters = {"source_answer_rows": [[list(word), list(answers)] for word, answers in source_rows]}
        predict = lambda request: known_source_answers[request.source_word]
    elif kind == "event_mode":
        event_rows = tuple(sorted((token, _categorical_mode(tuple(target for request, target in training_rows if request.event_token == token))) for token in {request.event_token for request, _ in training_rows}))
        event_map = dict(event_rows)
        parameters = {"fallback": list(fallback), "event_rows": [[token, list(answers)] for token, answers in event_rows]}
        predict = lambda request: event_map.get(request.event_token, fallback)
    elif kind == "source_mode":
        source_rows = tuple(sorted((word, _categorical_mode(tuple(target for request, target in training_rows if request.source_word == word))) for word in {request.source_word for request, _ in training_rows}))
        source_map = dict(source_rows)
        parameters = {"fallback": list(fallback), "source_rows": [[list(word), list(answers)] for word, answers in source_rows]}
        predict = lambda request: source_map.get(request.source_word, fallback)
    else:
        raise ValueError("unknown shortcut baseline kind")
    fit_model_sha256 = _digest({"schema": SHORTCUT_BASELINE_SCHEMA, "baseline_kind": kind, "parameters": parameters})
    fit_errors = sum(predict(request) != target for request, target in training_rows)
    heldout_errors = sum(predict(request) != target for request, target in heldout_rows)
    undefined_false_accepts = sum(predict(request) is not None for request in undefined_requests)
    payload = {
        "schema": SHORTCUT_BASELINE_SCHEMA,
        "baseline_kind": kind,
        "deterministic_fit_rule": fit_rules[kind],
        "fit_model_sha256": fit_model_sha256,
        "fit_edge_count": len(training_rows),
        "fit_error_count": fit_errors,
        "heldout_edge_count": len(heldout_rows),
        "heldout_error_count": heldout_errors,
        "undefined_pair_false_accept_count": undefined_false_accepts,
        "baseline_failed": bool(fit_errors or heldout_errors or undefined_false_accepts),
    }
    return T2ShortcutBaselineEvaluation(
        **{key: value for key, value in payload.items() if key != "schema"},
        evaluation_sha256=_digest(payload),
    )


def _evaluate_t1_first14_nonidentification(
    environment: T2ControllerEnvironment,
    answer_map: Mapping[str, tuple[str, ...]],
) -> T2T1First14Nonidentification:
    learner_input = environment.learner_input
    ordered_requests = tuple(environment.controller.learner_input.candidate_edge_requests[:14])
    if len(ordered_requests) != 14:
        raise T2ProtocolError("controller does not expose the frozen T1-first14 trace")
    known, passive_edges = _known_material(learner_input, ())
    edges = list(passive_edges)
    missing = 0
    for request in ordered_requests:
        source_answers = known.get(request.source_word)
        if source_answers is None:
            missing += 1
            continue
        target = answer_map[request.request_sha256]
        edges.append(_KnownEdge(request, source_answers, target))
        previous = known.setdefault(request.program, target)
        if previous != target:
            raise T2ProtocolError("T1-first14 trace contradicts a known diagnostic")
    versions = _filtered_versions(learner_input, known, tuple(edges))
    counts = tuple((token, len(rows)) for token, rows in sorted(versions.items()))
    rank_rows = tuple(
        (
            token,
            _rank(_observed_event_rows(learner_input, tuple(edges), token)[0]),
            len(rows[0].domain_basis_image_rows),
        )
        for token, rows in sorted(versions.items())
    )
    posterior_mass = 1
    for _, count in counts:
        posterior_mass *= count
    payload = {
        "schema": SHORTCUT_T1_SCHEMA,
        "ordered_request_sha256s": [request.request_sha256 for request in ordered_requests],
        "observed_source_missing_count": missing,
        "final_event_version_counts": [list(row) for row in counts],
        "final_event_rank_rows": [list(row) for row in rank_rows],
        "singleton_event_count": sum(count == 1 for _, count in counts),
        "non_singleton_event_count": sum(count != 1 for _, count in counts),
        "posterior_global_version_mass": posterior_mass,
        "identified": missing == 0 and all(count == 1 for _, count in counts),
        "exact_restricted_version_evaluation": True,
        "controller_supplied_postfit_negative_control": True,
    }
    return T2T1First14Nonidentification(
        tuple(payload["ordered_request_sha256s"]),
        payload["observed_source_missing_count"],
        counts,
        rank_rows,
        payload["singleton_event_count"],
        payload["non_singleton_event_count"],
        payload["posterior_global_version_mass"],
        payload["identified"],
        payload["exact_restricted_version_evaluation"],
        payload["controller_supplied_postfit_negative_control"],
        _digest(payload),
    )


def _evaluate_t2_shortcut_evidence(
    environment: T2ControllerEnvironment,
    preopen: T2PreopenEnvironment,
    opening_records: Sequence[T2OpeningRecord],
) -> T2ShortcutEvidence:
    if not isinstance(preopen.result, AutonomousPartialOperatorResult):
        raise T2ProtocolError("shortcut evidence requires an identified primary result")
    learner_input = environment.learner_input
    passive_rows = tuple((row.request, row.target_answers) for row in learner_input.passive_edge_observations)
    candidate_by_sha = {request.request_sha256: request for request in learner_input.canonical_candidate_requests}
    opening_by_sha = {row.request_sha256: row for row in opening_records}
    queried_rows = tuple((candidate_by_sha[digest], opening_by_sha[digest].target_answers) for digest in preopen.opened_request_sha256s)
    inferred_records = tuple(row for row in opening_records if row.role == "inferred")
    sealed_records = tuple(row for row in opening_records if row.role == "sealed")
    heldout_rows = tuple((candidate_by_sha[row.request_sha256], row.target_answers) for row in inferred_records + sealed_records)
    training_rows = passive_rows + queried_rows
    # Source diagnostics are features, not fitted targets.  The complete
    # diagnostic codebook/definedness mask is supplied at the learner boundary;
    # only transition targets are split between visible fit and postfit holdout.
    known = dict(preopen.result.model.mask_source_answer_rows)
    required_sources = {request.source_word for request, _ in training_rows + heldout_rows}
    if any(word not in known for word in required_sources):
        raise T2ProtocolError("shortcut evaluation source was not learner-visible after fourteen queries")
    known_sources = {word: known[word] for word in required_sources}
    baseline_rows = tuple(
        _make_shortcut_baseline_evaluation(
            kind,
            training_rows,
            heldout_rows,
            known_sources,
            learner_input.canonical_undefined_requests,
        )
        for kind in ("constant_mode", "identity", "event_mode", "source_mode")
    )
    answer_map = {row.request_sha256: row.target_answers for row in opening_records}
    t1 = None if environment.controller.kind is EnvironmentKind.FULL_SUPPORT_CONTROL else _evaluate_t1_first14_nonidentification(environment, answer_map)
    payload = {
        "schema": SHORTCUT_EVIDENCE_SCHEMA,
        "learner_input_sha256": learner_input.input_sha256,
        "primary_result_sha256": preopen.result.result_sha256,
        "passive_fit_request_sha256s": [request.request_sha256 for request, _ in passive_rows],
        "primary_visible_fit_request_sha256s": [request.request_sha256 for request, _ in queried_rows],
        "inferred_eval_request_sha256s": [row.request_sha256 for row in inferred_records],
        "sealed_eval_request_sha256s": [row.request_sha256 for row in sealed_records],
        "baseline_rows": [row.payload() for row in baseline_rows],
        "t1_first14": None if t1 is None else t1.payload(),
        "legal_holdout_applicable": t1 is not None,
        "t1_first14_applicable": t1 is not None,
        "all_shortcuts_fail": all(row.baseline_failed for row in baseline_rows) and (t1 is None or not t1.identified),
    }
    return T2ShortcutEvidence(
        learner_input.input_sha256,
        preopen.result.result_sha256,
        tuple(payload["passive_fit_request_sha256s"]),
        tuple(payload["primary_visible_fit_request_sha256s"]),
        tuple(payload["inferred_eval_request_sha256s"]),
        tuple(payload["sealed_eval_request_sha256s"]),
        baseline_rows,
        t1,
        payload["legal_holdout_applicable"],
        payload["t1_first14_applicable"],
        payload["all_shortcuts_fail"],
        _digest(payload),
    )


@dataclass(frozen=True)
class T2ArmEvaluation:
    schedule_label: str
    campaign_terminal_sha256: str
    controller_sha256: str
    learner_input_sha256: str
    result_sha256: str
    model_sha256: str
    guarded_language_certificate_sha256: str
    final_state_sha256: str
    preopen_sha256: str
    commitment_root_sha256: str | None
    long_suite_root_sha256: str
    opened_request_sha256s: tuple[str, ...]
    inferred_request_sha256: str | None
    sealed_request_sha256s: tuple[str, ...]
    legal_edge_count: int
    undefined_pair_count: int
    long_path_count: int
    guarded_bisimulation: bool
    shortcut_controls_fail: bool
    shortcut_evidence: T2ShortcutEvidence
    postfit_teaching_summary: T2PostfitTeachingSummary | None
    opening_records: tuple[T2OpeningRecord, ...]
    defined_rows: tuple[T2PredictionRow, ...]
    undefined_rows: tuple[T2PredictionRow, ...]
    long_rows: tuple[T2PredictionRow, ...]
    arm_sha256: str

    def __post_init__(self) -> None:
        if self.schedule_label not in _SCHEDULE_LABELS:
            raise ValueError("unknown frozen arm label")
        for value in (self.campaign_terminal_sha256, self.controller_sha256, self.learner_input_sha256, self.result_sha256, self.model_sha256, self.guarded_language_certificate_sha256, self.final_state_sha256, self.preopen_sha256, self.long_suite_root_sha256, self.arm_sha256): _require_digest("arm digest", value)
        if self.commitment_root_sha256 is not None: _require_digest("candidate root", self.commitment_root_sha256)
        if type(self.shortcut_evidence) is not T2ShortcutEvidence or self.shortcut_evidence.learner_input_sha256 != self.learner_input_sha256 or self.shortcut_evidence.primary_result_sha256 != self.result_sha256:
            raise ValueError("arm shortcut evidence does not bind this learner/result")
        if type(self.shortcut_controls_fail) is not bool or self.shortcut_controls_fail is not self.shortcut_evidence.all_shortcuts_fail:
            raise ValueError("arm shortcut aggregate flag differs from typed evidence")
        if self.schedule_label.startswith("control"):
            if self.commitment_root_sha256 is not None or self.opened_request_sha256s or self.inferred_request_sha256 is not None or self.sealed_request_sha256s or self.postfit_teaching_summary is not None:
                raise ValueError("control arm cannot contain candidate sidecars")
        else:
            if len(self.opened_request_sha256s) != 14 or self.inferred_request_sha256 is None or len(self.sealed_request_sha256s) != 8 or type(self.postfit_teaching_summary) is not T2PostfitTeachingSummary:
                raise ValueError("omission arm does not bind 14 opened, one inferred, and eight sealed requests")
            if self.postfit_teaching_summary.primary_reconstruction_result_sha256 != self.result_sha256:
                raise ValueError("omission teaching summary does not bind this primary result")
            if self.postfit_teaching_summary.learner_input_sha256 != self.learner_input_sha256:
                raise ValueError("omission teaching summary does not bind this learner input")
        if (self.legal_edge_count, self.undefined_pair_count, self.long_path_count) != (44, 46, 12) or not self.guarded_bisimulation or not self.shortcut_controls_fail:
            raise ValueError("arm evaluation is incomplete")
        if len(self.defined_rows) != 44 or len(self.undefined_rows) != 46 or len(self.long_rows) != 12 or not all(row.exact for row in self.defined_rows + self.long_rows):
            raise ValueError("arm lacks exact per-row postfit evidence")
        if self.schedule_label.startswith("control"):
            if self.opening_records: raise ValueError("control opening transcript must be empty")
        elif len(self.opening_records) != 23 or tuple(row.access_ordinal for row in self.opening_records) != tuple(range(1, 24)) or tuple(row.role for row in self.opening_records) != (("queried",) * 14 + ("inferred",) + ("sealed",) * 8):
            raise ValueError("omission must bind complete ordered 23-role opening ledger")
        else:
            request_sha256s = tuple(row.request_sha256 for row in self.opening_records)
            if len(set(request_sha256s)) != 23:
                raise ValueError("omission opening ledger contains duplicate requests")
            if request_sha256s[:14] != self.opened_request_sha256s or request_sha256s[14] != self.inferred_request_sha256 or request_sha256s[15:] != self.sealed_request_sha256s:
                raise ValueError("omission opening ledger role partitions differ from arm summary")
            answer_rows = tuple(sorted((row.request_sha256, row.target_answers) for row in self.opening_records))
            assert self.postfit_teaching_summary is not None
            if _digest(answer_rows) != self.postfit_teaching_summary.complete_candidate_answer_map_sha256:
                raise ValueError("omission teaching answer map differs from verified opening ledger")
        if self.arm_sha256 != _digest(self.payload(False)):
            raise ValueError("arm evaluation digest mismatch")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {"schedule_label": self.schedule_label, "campaign_terminal_sha256": self.campaign_terminal_sha256, "controller_sha256": self.controller_sha256, "learner_input_sha256": self.learner_input_sha256, "result_sha256": self.result_sha256, "model_sha256": self.model_sha256, "guarded_language_certificate_sha256": self.guarded_language_certificate_sha256, "final_state_sha256": self.final_state_sha256, "preopen_sha256": self.preopen_sha256, "commitment_root_sha256": self.commitment_root_sha256, "long_suite_root_sha256": self.long_suite_root_sha256, "opened_request_sha256s": list(self.opened_request_sha256s), "inferred_request_sha256": self.inferred_request_sha256, "sealed_request_sha256s": list(self.sealed_request_sha256s), "legal_edge_count": self.legal_edge_count, "undefined_pair_count": self.undefined_pair_count, "long_path_count": self.long_path_count, "guarded_bisimulation": self.guarded_bisimulation, "shortcut_controls_fail": self.shortcut_controls_fail, "shortcut_evidence": self.shortcut_evidence.payload(), "postfit_teaching_summary": None if self.postfit_teaching_summary is None else self.postfit_teaching_summary.payload(), "opening_records": [row.payload() for row in self.opening_records], "defined_rows": [row.payload() for row in self.defined_rows], "undefined_rows": [row.payload() for row in self.undefined_rows], "long_rows": [row.payload() for row in self.long_rows]}
        if include_digest: value["arm_sha256"] = self.arm_sha256
        return value


@dataclass(frozen=True)
class T2PairedSimilarity:
    left_arm_sha256: str
    right_arm_sha256: str
    controller_supplied_postfit_alignment: bool
    all_legal_rows_agree_under_alignment: bool
    change_of_basis: tuple[tuple[Fraction, ...], ...]
    anchor_count: int
    tested_state_count: int
    tested_legal_edge_count: int
    similarity_sha256: str

    def __post_init__(self) -> None:
        for value in (self.left_arm_sha256, self.right_arm_sha256, self.similarity_sha256): _require_digest("similarity digest", value)
        if not self.controller_supplied_postfit_alignment or not self.all_legal_rows_agree_under_alignment or self.anchor_count != 5 or self.tested_state_count != 9 or self.tested_legal_edge_count != 44:
            raise ValueError("paired similarity must be a supplied postfit exact alignment")
        if len(self.change_of_basis) != 5 or any(len(row) != 5 for row in self.change_of_basis): raise ValueError("similarity must bind a 5x5 rational gauge")
        if self.similarity_sha256 != _digest(self.payload(False)): raise ValueError("similarity digest mismatch")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {"left_arm_sha256": self.left_arm_sha256, "right_arm_sha256": self.right_arm_sha256, "controller_supplied_postfit_alignment": True, "all_legal_rows_agree_under_alignment": True, "change_of_basis": [[[item.numerator, item.denominator] for item in row] for row in self.change_of_basis], "anchor_count": 5, "tested_state_count": 9, "tested_legal_edge_count": 44}
        if include_digest: value["similarity_sha256"] = self.similarity_sha256
        return value


@dataclass(frozen=True)
class T2CampaignReport:
    protocol_sha256: str
    terminal_sha256: str
    preopen_sha256s: tuple[str, ...]
    total_legal_edges_checked: int
    total_undefined_pairs_checked: int
    total_long_paths_checked: int
    all_shortcuts_fail: bool
    arm_records: tuple["T2ArmEvaluation", ...]
    paired_similarities: tuple["T2PairedSimilarity", ...]
    postfit_teaching_summary: T2CampaignTeachingSummary
    report_sha256: str

    def __post_init__(self) -> None:
        if (self.total_legal_edges_checked, self.total_undefined_pairs_checked, self.total_long_paths_checked) != (440, 460, 120):
            raise ValueError("campaign aggregate postfit checks mismatch")
        if not self.all_shortcuts_fail or len(self.preopen_sha256s) != 10 or len(self.arm_records) != 10 or len(self.paired_similarities) != 5:
            raise ValueError("campaign report requires all scheduled checks")
        if tuple(row.schedule_label for row in self.arm_records) != _SCHEDULE_LABELS:
            raise ValueError("campaign arm evidence is not in frozen schedule order")
        if type(self.postfit_teaching_summary) is not T2CampaignTeachingSummary:
            raise TypeError("campaign report requires an exact teaching summary")
        teaching_rows = tuple(row.postfit_teaching_summary for row in self.arm_records[2:])
        if any(type(row) is not T2PostfitTeachingSummary for row in teaching_rows) or self.postfit_teaching_summary.arm_summary_sha256s != tuple(row.summary_sha256 for row in teaching_rows):
            raise ValueError("campaign teaching summary does not bind its eight omission arms")
        if _require_digest("report_sha256", self.report_sha256) != _digest(self.payload(False)):
            raise ValueError("campaign report digest mismatch")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {"schema": OPEN_SCHEMA, "protocol_sha256": self.protocol_sha256, "terminal_sha256": self.terminal_sha256, "preopen_sha256s": list(self.preopen_sha256s), "total_legal_edges_checked": self.total_legal_edges_checked, "total_undefined_pairs_checked": self.total_undefined_pairs_checked, "total_long_paths_checked": self.total_long_paths_checked, "all_shortcuts_fail": self.all_shortcuts_fail, "arm_records": [row.payload() for row in self.arm_records], "paired_similarities": [row.payload() for row in self.paired_similarities], "postfit_teaching_summary": self.postfit_teaching_summary.payload()}
        if include_digest: value["report_sha256"] = self.report_sha256
        return value


def _invert(matrix: Sequence[Sequence[object]]) -> tuple[tuple[Fraction, ...], ...]:
    work = [[Fraction(getattr(value, "numerator"), getattr(value, "denominator")) for value in row] for row in matrix]
    size = len(work)
    if size != 5 or any(len(row) != size for row in work): raise ValueError("gauge anchors must be 5x5")
    aug = [row + [Fraction(int(i == j)) for j in range(size)] for i, row in enumerate(work)]
    for col in range(size):
        pivot = next((row for row in range(col, size) if aug[row][col]), None)
        if pivot is None: raise T2ProtocolError("deterministic gauge anchors are singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]; aug[col] = [value / scale for value in aug[col]]
        for row in range(size):
            if row != col and aug[row][col]:
                factor = aug[row][col]; aug[row] = [value - factor * pivot_value for value, pivot_value in zip(aug[row], aug[col], strict=True)]
    return tuple(tuple(row[size:]) for row in aug)


def _row_times(row: Sequence[object], matrix: Sequence[Sequence[Fraction]]) -> tuple[Fraction, ...]:
    left = [Fraction(getattr(value, "numerator"), getattr(value, "denominator")) for value in row]
    return tuple(sum(left[index] * matrix[index][column] for index in range(5)) for column in range(5))


def _paired_similarity(left_env: T2ControllerEnvironment, left: T2ArmEvaluation, left_result: AutonomousPartialOperatorResult, right_env: T2ControllerEnvironment, right: T2ArmEvaluation, right_result: AutonomousPartialOperatorResult, max_suffix_events: int) -> T2PairedSimilarity:
    left_rows = dict(left_result.model.diagnostic_codebook); right_rows = dict(right_result.model.diagnostic_codebook)
    left_states = dict(left_env.controller.canonical_state_answers); right_states = dict(right_env.controller.canonical_state_answers)
    states = sorted(left_states, key=repr)
    rank, pivot_indices, _ = _rank_profile([left_rows[left_states[state]] for state in states])
    if rank != 5 or len(pivot_indices) != 5: raise T2ProtocolError("paired diagnostic state table is not rank five")
    anchors = [states[index] for index in pivot_indices]
    inverse = _invert([left_rows[left_states[state]] for state in anchors])
    right_anchor_rows = [right_rows[right_states[state]] for state in anchors]
    gauge = tuple(tuple(sum(inverse[row][k] * Fraction(getattr(right_anchor_rows[k][column], "numerator"), getattr(right_anchor_rows[k][column], "denominator")) for k in range(5)) for column in range(5)) for row in range(5))
    all_states = all(_row_times(left_rows[left_states[state]], gauge) == tuple(Fraction(getattr(value, "numerator"), getattr(value, "denominator")) for value in right_rows[right_states[state]]) for state in states)
    # Controller supplied state/action correspondence is postfit-only.  Check
    # every left legal prediction transforms into the corresponding right one.
    left_tokens = dict(left_env.controller.canonical_event_tokens); right_tokens = dict(right_env.controller.canonical_event_tokens)
    left_known = dict(left_result.model.mask_source_answer_rows); right_known = dict(right_result.model.mask_source_answer_rows)
    edge_ok = 0
    for action, left_token in left_tokens.items():
        right_token = right_tokens[action]
        for state in states:
            left_answer = left_states[state]; right_answer = right_states[state]
            try:
                lpred = predict_defined_suffix(left_result, left_answer, (left_token,), max_events=max_suffix_events); rpred = predict_defined_suffix(right_result, right_answer, (right_token,), max_events=max_suffix_events)
            except ValueError:
                continue
            if _row_times(left_rows[lpred], gauge) == tuple(Fraction(getattr(value, "numerator"), getattr(value, "denominator")) for value in right_rows[rpred]): edge_ok += 1
    if edge_ok != 44 or not all_states: raise T2ProtocolError("paired controller alignment fails exact state/edge gauge test")
    payload = {"left_arm_sha256": left.arm_sha256, "right_arm_sha256": right.arm_sha256, "controller_supplied_postfit_alignment": True, "all_legal_rows_agree_under_alignment": True, "change_of_basis": [[[item.numerator, item.denominator] for item in row] for row in gauge], "anchor_count": 5, "tested_state_count": 9, "tested_legal_edge_count": 44}
    return T2PairedSimilarity(left.arm_sha256, right.arm_sha256, True, True, gauge, 5, 9, 44, _digest(payload))


def open_t2_campaign_postfit_batch(protocol: Phase3T2Protocol, environments: Sequence[T2ControllerEnvironment], preopens: Sequence[T2PreopenEnvironment], terminal: T2CampaignTerminalPreopen) -> T2CampaignReport:
    """The sole public postfit opener: validate every arm before any sidecar."""
    if len(environments) != 10 or len(preopens) != 10:
        raise ValueError("postfit batch requires all ten scheduled arms")
    if tuple(row.schedule_label for row in preopens) != protocol.schedule_labels:
        raise T2ProtocolError("public postfit batch schedule differs from frozen protocol")
    if protocol.source_runtime_binding is None or terminal.protocol_sha256 != protocol.protocol_sha256 or terminal.source_runtime_binding_sha256 != protocol.source_runtime_binding.binding_sha256:
        raise T2ProtocolError("terminal/protocol/source binding mismatch")
    if tuple(row.preopen_sha256 for row in preopens) != terminal.scheduled_preopen_sha256s:
        raise T2ProtocolError("terminal no longer binds this exact scheduled preopen order")
    suffix_cap = protocol.active_budgets.max_suffix_events_per_prediction
    # No hidden answer is read above this line.  Validate every frozen learner
    # result and every controller commitment root first.
    for index, (environment, preopen) in enumerate(zip(environments, preopens, strict=True)):
        kind, block, cell = _SCHEDULE[index]
        if environment.controller.kind is not kind or environment.controller.relabel_block != block or environment.controller.pseudoheldout_cell != cell:
            raise T2ProtocolError("public postfit environment descriptor differs from frozen schedule")
        if environment.long_suite_root_sha256 != protocol.long_suite_roots[index] or preopen.long_suite_root_sha256 != protocol.long_suite_roots[index]:
            raise T2ProtocolError("public postfit long-suite root differs from precommitment")
        if any(len(row.program) > suffix_cap for row in environment.long_path_programs):
            raise T2ProtocolError("precommitted long program exceeds frozen suffix prediction cap")
        if environment.learner_input.input_sha256 != preopen.learner_input_sha256:
            raise T2ProtocolError("preopen no longer matches its fresh learner input")
        if index >= 2:
            assert environment.commitment is not None
            if environment.commitment.merkle_root_sha256 != protocol.omission_commitment_roots[index - 2]:
                raise T2ProtocolError("scheduled precomputed commitment root mismatch")
    # Cross the postfit barrier for all eight omissions before constructing
    # any truth-aware teaching control.  Each call verifies all 23 committed
    # leaves and the primary inferred/sealed partition against the terminal.
    for environment, preopen in zip(environments[2:], preopens[2:], strict=True):
        _open_t2_postfit_environment(environment, preopen, terminal.terminal_sha256)
    teaching_summaries: list[T2PostfitTeachingSummary] = []
    for environment, preopen in zip(environments[2:], preopens[2:], strict=True):
        assert environment.commitment is not None
        assert isinstance(preopen.result, AutonomousPartialOperatorResult)
        verified_answer_map = {
            leaf.request_sha256: leaf.target_answers
            for leaf in environment.commitment.leaves
        }
        teaching_result = discover_postfit_teaching_control(
            environment.learner_input,
            verified_answer_map,
        )
        validate_postfit_teaching_control(
            environment.learner_input,
            verified_answer_map,
            teaching_result,
        )
        teaching_summaries.append(
            _summarize_postfit_teaching_control(
                teaching_result,
                preopen.result.result_sha256,
            )
        )
    legal = undefined = long = 0
    shortcuts = True
    arm_records: list[T2ArmEvaluation] = []
    for index, (environment, preopen) in enumerate(zip(environments, preopens, strict=True)):
        result = preopen.result
        assert isinstance(result, AutonomousPartialOperatorResult)
        teaching_summary = None if index < 2 else teaching_summaries[index - 2]
        checked_legal, checked_undefined, checked_long = _evaluate_model_against_controller(environment, result, suffix_cap)
        legal += checked_legal; undefined += checked_undefined; long += checked_long
        inferred = [row.inference.request.request_sha256 for row in result.final_state.steps if hasattr(row, "inference")]
        sealed = () if index < 2 else tuple(sorted(set(request.request_sha256 for request in environment.learner_input.canonical_candidate_requests) - set(preopen.opened_request_sha256s) - set(inferred)))
        expected = _controller_legal_answers(environment); known = dict(result.model.mask_source_answer_rows)
        defined_rows = tuple(T2PredictionRow("defined_edge", request.request_sha256, expected[request.request_sha256], predict_defined_suffix(result, known[request.source_word], (request.event_token,), max_events=suffix_cap), predict_defined_suffix(result, known[request.source_word], (request.event_token,), max_events=suffix_cap) == expected[request.request_sha256]) for request in environment.learner_input.canonical_defined_requests)
        def undefined_rejected(request: OpaqueEdgeRequest) -> bool:
            try:
                predict_defined_suffix(result, known[request.source_word], (request.event_token,), max_events=suffix_cap)
            except ValueError:
                return True
            return False
        undefined_rows = tuple(T2PredictionRow("undefined_pair", request.request_sha256, None, None, undefined_rejected(request)) for request in environment.learner_input.canonical_undefined_requests)
        long_rows = tuple(T2PredictionRow("long_path", row.program_sha256, row.expected_answers, predict_defined_suffix(result, environment.controller.initial_answers, row.program, max_events=suffix_cap), predict_defined_suffix(result, environment.controller.initial_answers, row.program, max_events=suffix_cap) == row.expected_answers) for row in environment.long_path_programs)
        opening_records: tuple[T2OpeningRecord, ...] = ()
        if environment.commitment is not None:
            queried = {step.response.request.request_sha256: step for step in result.final_state.steps if hasattr(step, "response")}
            inferred_requests = {step.inference.request.request_sha256 for step in result.final_state.steps if hasattr(step, "inference")}
            ledger: list[T2OpeningRecord] = []
            role_order = list(preopen.opened_request_sha256s) + sorted(inferred_requests) + sorted(set(request.request_sha256 for request in environment.learner_input.canonical_candidate_requests) - set(preopen.opened_request_sha256s) - inferred_requests)
            for ordinal, request_sha in enumerate(role_order, 1):
                leaf = next(item for item in environment.commitment.leaves if item.request_sha256 == request_sha)
                request = next(row for row in environment.learner_input.canonical_candidate_requests if row.request_sha256 == leaf.request_sha256)
                proof = environment.commitment.proof_for(leaf.request_sha256)
                if leaf.request_sha256 in queried:
                    step = queried[leaf.request_sha256]; role, choice_sha, response_sha = "queried", step.choice.choice_sha256, step.response.response_sha256
                elif leaf.request_sha256 in inferred_requests:
                    role, choice_sha, response_sha = "inferred", "0" * 64, "0" * 64
                else:
                    role, choice_sha, response_sha = "sealed", "0" * 64, "0" * 64
                environment.commitment.verify_opening(request, leaf.target_answers)
                ledger.append(T2OpeningRecord(role, ordinal, leaf.request_sha256, leaf.target_answers, leaf.salt, proof.leaf_index, choice_sha, response_sha, leaf.leaf_sha256, proof.sibling_sha256s))
            opening_records = tuple(ledger)
        shortcut_evidence = _evaluate_t2_shortcut_evidence(environment, preopen, opening_records)
        shortcut = shortcut_evidence.all_shortcuts_fail
        shortcuts = shortcuts and shortcut
        arm_payload = {"schedule_label": protocol.schedule_labels[index], "campaign_terminal_sha256": terminal.terminal_sha256, "controller_sha256": environment.controller.controller_sha256, "learner_input_sha256": environment.learner_input.input_sha256, "result_sha256": result.result_sha256, "model_sha256": result.model.model_sha256, "guarded_language_certificate_sha256": result.guarded_language.certificate_sha256, "final_state_sha256": result.final_state.state_sha256, "preopen_sha256": preopen.preopen_sha256, "commitment_root_sha256": None if environment.commitment is None else environment.commitment.merkle_root_sha256, "long_suite_root_sha256": environment.long_suite_root_sha256, "opened_request_sha256s": tuple(preopen.opened_request_sha256s), "inferred_request_sha256": None if not inferred else inferred[0], "sealed_request_sha256s": sealed, "legal_edge_count": checked_legal, "undefined_pair_count": checked_undefined, "long_path_count": checked_long, "guarded_bisimulation": result.guarded_language.arbitrary_length_legal_suffix_induction, "shortcut_controls_fail": shortcut, "shortcut_evidence": shortcut_evidence, "postfit_teaching_summary": teaching_summary, "opening_records": opening_records, "defined_rows": defined_rows, "undefined_rows": undefined_rows, "long_rows": long_rows}
        arm_digest_payload = {**arm_payload, "shortcut_evidence": shortcut_evidence.payload(), "postfit_teaching_summary": None if teaching_summary is None else teaching_summary.payload(), "opening_records": [row.payload() for row in opening_records], "defined_rows": [row.payload() for row in defined_rows], "undefined_rows": [row.payload() for row in undefined_rows], "long_rows": [row.payload() for row in long_rows]}
        arm_records.append(T2ArmEvaluation(**arm_payload, arm_sha256=_digest(arm_digest_payload)))
    pairs = ((0, 1), (2, 6), (3, 7), (4, 8), (5, 9))
    similarities = tuple(_paired_similarity(environments[a], arm_records[a], preopens[a].result, environments[b], arm_records[b], preopens[b].result, suffix_cap) for a, b in pairs)
    teaching_campaign_summary = T2CampaignTeachingSummary.from_arms(teaching_summaries)
    payload = {"schema": OPEN_SCHEMA, "protocol_sha256": protocol.protocol_sha256, "terminal_sha256": terminal.terminal_sha256, "preopen_sha256s": [row.preopen_sha256 for row in preopens], "total_legal_edges_checked": legal, "total_undefined_pairs_checked": undefined, "total_long_paths_checked": long, "all_shortcuts_fail": shortcuts, "arm_records": [row.payload() for row in arm_records], "paired_similarities": [row.payload() for row in similarities], "postfit_teaching_summary": teaching_campaign_summary.payload()}
    return T2CampaignReport(protocol.protocol_sha256, terminal.terminal_sha256, tuple(payload["preopen_sha256s"]), legal, undefined, long, shortcuts, tuple(arm_records), similarities, teaching_campaign_summary, _digest(payload))


def open_t2_staged_campaign(
    protocol: Phase3T2Protocol,
    repository_root: Path,
    records: Sequence[T2PreopenRecord],
    terminal: T2CampaignTerminalPreopen,
) -> T2CampaignReport:
    """Sole disk-staged opener, with all ten summaries checked first.

    The terminal and complete staged inventory are validated before replay.
    Replay then reconstructs all ten preopen results and requires exact summary
    identity.  Only after every equality holds is the public atomic postfit
    batch called, so no inferred, sealed, or long-program sidecar can be
    selectively opened from a partial or mismatched record set.
    """

    if type(protocol) is not Phase3T2Protocol:
        raise TypeError("protocol must be exact Phase3T2Protocol")
    if type(terminal) is not T2CampaignTerminalPreopen:
        raise TypeError("terminal must be exact T2CampaignTerminalPreopen")
    tuple_records = tuple(records)
    expected_terminal = aggregate_t2_preopen_records(
        protocol,
        repository_root,
        tuple_records,
    )
    if terminal != expected_terminal:
        raise T2ProtocolError("terminal artifact is not the exact aggregate of staged preopens")

    environments: list[T2ControllerEnvironment] = []
    preopens: list[T2PreopenEnvironment] = []
    reconstructed_records: list[T2PreopenRecord] = []
    for index in range(len(_SCHEDULE)):
        environment, preopen, record = _reconstruct_t2_preopen(protocol, index)
        environments.append(environment)
        preopens.append(preopen)
        reconstructed_records.append(record)
    if tuple(reconstructed_records) != tuple_records:
        raise T2ProtocolError("staged preopen record fails deterministic exact reconstruction")
    reconstructed_terminal = freeze_t2_campaign_terminal(protocol, tuple(preopens))
    if reconstructed_terminal != terminal:
        raise T2ProtocolError("terminal artifact fails deterministic exact reconstruction")

    # This call is intentionally the first postfit sidecar surface.
    return open_t2_campaign_postfit_batch(
        protocol,
        tuple(environments),
        tuple(preopens),
        terminal,
    )


def validate_t2_staged_campaign(
    protocol: Phase3T2Protocol,
    repository_root: Path,
    records: Sequence[T2PreopenRecord],
    terminal: T2CampaignTerminalPreopen,
    report: T2CampaignReport,
) -> None:
    """Reconstruct staged evidence and require exact report byte semantics."""

    if type(report) is not T2CampaignReport:
        raise TypeError("report must be exact T2CampaignReport")
    tuple_records = tuple(records)
    if report.protocol_sha256 != protocol.protocol_sha256:
        raise T2ProtocolError("campaign report protocol digest differs from ready protocol")
    if report.terminal_sha256 != terminal.terminal_sha256:
        raise T2ProtocolError("campaign report terminal digest differs from staged terminal")
    if report.preopen_sha256s != tuple(record.preopen_sha256 for record in tuple_records):
        raise T2ProtocolError("campaign report preopen inventory differs from staged records")
    reconstructed = open_t2_staged_campaign(
        protocol,
        repository_root,
        tuple_records,
        terminal,
    )
    if reconstructed != report:
        raise T2ProtocolError("staged campaign report fails deterministic authoritative reconstruction")


def run_t2_campaign(protocol: Phase3T2Protocol, repository_root: Path) -> T2CampaignReport:
    """Authoritative ready-only campaign execution in the frozen schedule."""
    require_execution_ready(protocol, repository_root)
    schedule = _SCHEDULE
    environments: list[T2ControllerEnvironment] = []
    for index, (kind, block, cell) in enumerate(schedule):
        omission_index = index - 2
        environments.append(build_t2_controller_environment(
            kind=kind, relabel_block=block, omitted_cell=cell,
            controller_nonce=protocol.controller_nonces[index], active_budgets=protocol.active_budgets,
            salts=None if kind is EnvironmentKind.FULL_SUPPORT_CONTROL else protocol.omission_salts[omission_index],
            expected_commitment_root_sha256=None if kind is EnvironmentKind.FULL_SUPPORT_CONTROL else protocol.omission_commitment_roots[omission_index],
            expected_long_suite_root_sha256=protocol.long_suite_roots[index],
        ))
    preopens = tuple(run_t2_preopen_environment(row, minimum_unopened_candidates=protocol.min_unopened_candidates) for row in environments)
    terminal = freeze_t2_campaign_terminal(protocol, preopens)
    return open_t2_campaign_postfit_batch(protocol, tuple(environments), preopens, terminal)


def reconstruct_t2_campaign(protocol: Phase3T2Protocol, repository_root: Path, report: T2CampaignReport) -> None:
    """Replay every source-bound arm and require exact report identity."""
    if type(report) is not T2CampaignReport:
        raise TypeError("report must be exact T2CampaignReport")
    rebuilt = run_t2_campaign(protocol, repository_root)
    if rebuilt != report:
        raise T2ProtocolError("campaign report fails deterministic authoritative reconstruction")


def write_t2_campaign_report(path: Path, report: T2CampaignReport) -> None:
    if type(report) is not T2CampaignReport: raise TypeError("report must be exact")
    _write_closed_canonical_json(path, report.payload())


def load_t2_campaign_report(path: Path) -> T2CampaignReport:
    expected = {"schema", "protocol_sha256", "terminal_sha256", "preopen_sha256s", "total_legal_edges_checked", "total_undefined_pairs_checked", "total_long_paths_checked", "all_shortcuts_fail", "arm_records", "paired_similarities", "postfit_teaching_summary", "report_sha256"}
    raw = _require_closed_object(_load_closed_canonical_json(path), frozenset(expected), "campaign artifact")
    if raw["schema"] != OPEN_SCHEMA: raise ValueError("campaign artifact schema is not closed")
    arm_expected = {"schedule_label", "campaign_terminal_sha256", "controller_sha256", "learner_input_sha256", "result_sha256", "model_sha256", "guarded_language_certificate_sha256", "final_state_sha256", "preopen_sha256", "commitment_root_sha256", "long_suite_root_sha256", "opened_request_sha256s", "inferred_request_sha256", "sealed_request_sha256s", "legal_edge_count", "undefined_pair_count", "long_path_count", "guarded_bisimulation", "shortcut_controls_fail", "shortcut_evidence", "postfit_teaching_summary", "opening_records", "defined_rows", "undefined_rows", "long_rows", "arm_sha256"}
    opening_expected = {"role", "access_ordinal", "request_sha256", "target_answers", "salt", "leaf_index", "choice_sha256", "response_sha256", "leaf_sha256", "proof_siblings"}
    prediction_expected = {"row_kind", "item_sha256", "expected_answers", "predicted_answers", "exact"}
    shortcut_baseline_expected = {"schema", "baseline_kind", "deterministic_fit_rule", "fit_model_sha256", "fit_edge_count", "fit_error_count", "heldout_edge_count", "heldout_error_count", "undefined_pair_false_accept_count", "baseline_failed", "evaluation_sha256"}
    shortcut_t1_expected = {"schema", "ordered_request_sha256s", "observed_source_missing_count", "final_event_version_counts", "final_event_rank_rows", "singleton_event_count", "non_singleton_event_count", "posterior_global_version_mass", "identified", "exact_restricted_version_evaluation", "controller_supplied_postfit_negative_control", "baseline_sha256"}
    shortcut_evidence_expected = {"schema", "learner_input_sha256", "primary_result_sha256", "passive_fit_request_sha256s", "primary_visible_fit_request_sha256s", "inferred_eval_request_sha256s", "sealed_eval_request_sha256s", "baseline_rows", "t1_first14", "legal_holdout_applicable", "t1_first14_applicable", "all_shortcuts_fail", "evidence_sha256"}
    def decode_shortcut_baseline(item: object) -> T2ShortcutBaselineEvaluation:
        row = _require_closed_object(item, frozenset(shortcut_baseline_expected), "shortcut baseline")
        if row["schema"] != SHORTCUT_BASELINE_SCHEMA:
            raise ValueError("unknown shortcut baseline schema")
        return T2ShortcutBaselineEvaluation(
            _require_exact_string(row["baseline_kind"], "baseline_kind"),
            _require_exact_string(row["deterministic_fit_rule"], "deterministic_fit_rule"),
            _require_exact_string(row["fit_model_sha256"], "fit_model_sha256"),
            _require_exact_int(row["fit_edge_count"], "fit_edge_count"),
            _require_exact_int(row["fit_error_count"], "fit_error_count"),
            _require_exact_int(row["heldout_edge_count"], "heldout_edge_count"),
            _require_exact_int(row["heldout_error_count"], "heldout_error_count"),
            _require_exact_int(row["undefined_pair_false_accept_count"], "undefined_pair_false_accept_count"),
            _require_exact_bool(row["baseline_failed"], "baseline_failed"),
            _require_exact_string(row["evaluation_sha256"], "evaluation_sha256"),
        )
    def decode_t1_first14(item: object) -> T2T1First14Nonidentification:
        row = _require_closed_object(item, frozenset(shortcut_t1_expected), "T1-first14 baseline")
        if row["schema"] != SHORTCUT_T1_SCHEMA:
            raise ValueError("unknown T1-first14 baseline schema")
        version_rows_raw = _require_exact_list(row["final_event_version_counts"], "final_event_version_counts")
        version_rows: list[tuple[str, int]] = []
        for item_row in version_rows_raw:
            pair = _require_exact_list(item_row, "event version row")
            if len(pair) != 2:
                raise ValueError("event version row must have token and count")
            version_rows.append((_require_exact_string(pair[0], "event token"), _require_exact_int(pair[1], "event version count")))
        rank_rows_raw = _require_exact_list(row["final_event_rank_rows"], "final_event_rank_rows")
        rank_rows: list[tuple[str, int, int]] = []
        for item_row in rank_rows_raw:
            triple = _require_exact_list(item_row, "event rank row")
            if len(triple) != 3:
                raise ValueError("event rank row must have token and two ranks")
            rank_rows.append((_require_exact_string(triple[0], "event token"), _require_exact_int(triple[1], "observed rank"), _require_exact_int(triple[2], "legal rank")))
        return T2T1First14Nonidentification(
            tuple(_require_exact_string(value, "T1 request") for value in _require_exact_list(row["ordered_request_sha256s"], "ordered_request_sha256s")),
            _require_exact_int(row["observed_source_missing_count"], "observed_source_missing_count"),
            tuple(version_rows),
            tuple(rank_rows),
            _require_exact_int(row["singleton_event_count"], "singleton_event_count"),
            _require_exact_int(row["non_singleton_event_count"], "non_singleton_event_count"),
            _require_exact_int(row["posterior_global_version_mass"], "posterior_global_version_mass"),
            _require_exact_bool(row["identified"], "identified"),
            _require_exact_bool(row["exact_restricted_version_evaluation"], "exact_restricted_version_evaluation"),
            _require_exact_bool(row["controller_supplied_postfit_negative_control"], "controller_supplied_postfit_negative_control"),
            _require_exact_string(row["baseline_sha256"], "baseline_sha256"),
        )
    def decode_shortcut_evidence(item: object) -> T2ShortcutEvidence:
        row = _require_closed_object(item, frozenset(shortcut_evidence_expected), "shortcut evidence")
        if row["schema"] != SHORTCUT_EVIDENCE_SCHEMA:
            raise ValueError("unknown shortcut evidence schema")
        def digest_tuple(name: str) -> tuple[str, ...]:
            return tuple(_require_exact_string(value, name) for value in _require_exact_list(row[name], name))
        t1_raw = row["t1_first14"]
        return T2ShortcutEvidence(
            _require_exact_string(row["learner_input_sha256"], "learner_input_sha256"),
            _require_exact_string(row["primary_result_sha256"], "primary_result_sha256"),
            digest_tuple("passive_fit_request_sha256s"),
            digest_tuple("primary_visible_fit_request_sha256s"),
            digest_tuple("inferred_eval_request_sha256s"),
            digest_tuple("sealed_eval_request_sha256s"),
            tuple(decode_shortcut_baseline(value) for value in _require_exact_list(row["baseline_rows"], "baseline_rows")),
            None if t1_raw is None else decode_t1_first14(t1_raw),
            _require_exact_bool(row["legal_holdout_applicable"], "legal_holdout_applicable"),
            _require_exact_bool(row["t1_first14_applicable"], "t1_first14_applicable"),
            _require_exact_bool(row["all_shortcuts_fail"], "all_shortcuts_fail"),
            _require_exact_string(row["evidence_sha256"], "evidence_sha256"),
        )
    teaching_expected = {"schema", "learner_input_sha256", "teaching_result_sha256", "complete_candidate_answer_map_sha256", "primary_reconstruction_result_sha256", "counterfactual_truth_selected_query_count", "answer_free_singleton_inference_count", "counterfactual_unqueried_count", "closed_restricted_map_count", "rank_closed_event_count", "counterfactual_returned_categorical_label_count", "new_membership_calls_made", "causal_primary_isolated", "truth_specific_noncausal_control", "selection_eligible", "confirmatory_claim_eligible", "global_query_minimality_claimed", "arbitrary_total_operator_constructed", "summary_sha256"}
    def decode_teaching_summary(item: object) -> T2PostfitTeachingSummary:
        row = _require_closed_object(item, frozenset(teaching_expected), "postfit teaching summary")
        if row["schema"] != TEACHING_SUMMARY_SCHEMA:
            raise ValueError("unknown postfit teaching summary schema")
        return T2PostfitTeachingSummary(
            _require_exact_string(row["learner_input_sha256"], "learner_input_sha256"),
            _require_exact_string(row["teaching_result_sha256"], "teaching_result_sha256"),
            _require_exact_string(row["complete_candidate_answer_map_sha256"], "complete_candidate_answer_map_sha256"),
            _require_exact_string(row["primary_reconstruction_result_sha256"], "primary_reconstruction_result_sha256"),
            _require_exact_int(row["counterfactual_truth_selected_query_count"], "counterfactual_truth_selected_query_count"),
            _require_exact_int(row["answer_free_singleton_inference_count"], "answer_free_singleton_inference_count"),
            _require_exact_int(row["counterfactual_unqueried_count"], "counterfactual_unqueried_count"),
            _require_exact_int(row["closed_restricted_map_count"], "closed_restricted_map_count"),
            _require_exact_int(row["rank_closed_event_count"], "rank_closed_event_count"),
            _require_exact_int(row["counterfactual_returned_categorical_label_count"], "counterfactual_returned_categorical_label_count"),
            _require_exact_int(row["new_membership_calls_made"], "new_membership_calls_made"),
            _require_exact_bool(row["causal_primary_isolated"], "causal_primary_isolated"),
            _require_exact_bool(row["truth_specific_noncausal_control"], "truth_specific_noncausal_control"),
            _require_exact_bool(row["selection_eligible"], "selection_eligible"),
            _require_exact_bool(row["confirmatory_claim_eligible"], "confirmatory_claim_eligible"),
            _require_exact_bool(row["global_query_minimality_claimed"], "global_query_minimality_claimed"),
            _require_exact_bool(row["arbitrary_total_operator_constructed"], "arbitrary_total_operator_constructed"),
            _require_exact_string(row["summary_sha256"], "summary_sha256"),
        )
    def decode_opening(item: object) -> T2OpeningRecord:
        row = _require_closed_object(item, frozenset(opening_expected), "opening record")
        targets = tuple(_require_exact_string(value, "target answer") for value in _require_exact_list(row["target_answers"], "target_answers"))
        siblings = tuple(_require_exact_string(value, "proof sibling") for value in _require_exact_list(row["proof_siblings"], "proof_siblings"))
        return T2OpeningRecord(
            _require_exact_string(row["role"], "opening role"),
            _require_exact_int(row["access_ordinal"], "access_ordinal"),
            _require_exact_string(row["request_sha256"], "request_sha256"),
            targets,
            _require_exact_string(row["salt"], "salt"),
            _require_exact_int(row["leaf_index"], "leaf_index"),
            _require_exact_string(row["choice_sha256"], "choice_sha256"),
            _require_exact_string(row["response_sha256"], "response_sha256"),
            _require_exact_string(row["leaf_sha256"], "leaf_sha256"),
            siblings,
        )
    def decode_prediction(item: object) -> T2PredictionRow:
        row = _require_closed_object(item, frozenset(prediction_expected), "prediction row")
        expected_answers = row["expected_answers"]
        predicted_answers = row["predicted_answers"]
        expected_tuple = None if expected_answers is None else tuple(
            _require_exact_string(value, "expected answer")
            for value in _require_exact_list(expected_answers, "expected_answers")
        )
        predicted_tuple = None if predicted_answers is None else tuple(
            _require_exact_string(value, "predicted answer")
            for value in _require_exact_list(predicted_answers, "predicted_answers")
        )
        return T2PredictionRow(
            _require_exact_string(row["row_kind"], "row_kind"),
            _require_exact_string(row["item_sha256"], "item_sha256"),
            expected_tuple,
            predicted_tuple,
            _require_exact_bool(row["exact"], "prediction exact"),
        )
    def decode_arm(row: object) -> T2ArmEvaluation:
        closed = _require_closed_object(row, frozenset(arm_expected), "arm artifact")
        copied = dict(closed)
        copied["opened_request_sha256s"] = tuple(_require_exact_string(value, "opened request") for value in _require_exact_list(copied["opened_request_sha256s"], "opened_request_sha256s"))
        copied["sealed_request_sha256s"] = tuple(_require_exact_string(value, "sealed request") for value in _require_exact_list(copied["sealed_request_sha256s"], "sealed_request_sha256s"))
        copied["opening_records"] = tuple(decode_opening(item) for item in _require_exact_list(copied["opening_records"], "opening_records"))
        copied["defined_rows"] = tuple(decode_prediction(item) for item in _require_exact_list(copied["defined_rows"], "defined_rows"))
        copied["undefined_rows"] = tuple(decode_prediction(item) for item in _require_exact_list(copied["undefined_rows"], "undefined_rows"))
        copied["long_rows"] = tuple(decode_prediction(item) for item in _require_exact_list(copied["long_rows"], "long_rows"))
        for name in ("legal_edge_count", "undefined_pair_count", "long_path_count"):
            copied[name] = _require_exact_int(copied[name], name)
        copied["guarded_bisimulation"] = _require_exact_bool(copied["guarded_bisimulation"], "guarded_bisimulation")
        copied["shortcut_controls_fail"] = _require_exact_bool(copied["shortcut_controls_fail"], "shortcut_controls_fail")
        copied["shortcut_evidence"] = decode_shortcut_evidence(copied["shortcut_evidence"])
        copied["postfit_teaching_summary"] = None if copied["postfit_teaching_summary"] is None else decode_teaching_summary(copied["postfit_teaching_summary"])
        return T2ArmEvaluation(**copied)
    arms = tuple(decode_arm(row) for row in _require_exact_list(raw["arm_records"], "arm_records"))
    sim_expected = {"left_arm_sha256", "right_arm_sha256", "controller_supplied_postfit_alignment", "all_legal_rows_agree_under_alignment", "change_of_basis", "anchor_count", "tested_state_count", "tested_legal_edge_count", "similarity_sha256"}
    def decode_similarity(row: object) -> T2PairedSimilarity:
        closed = _require_closed_object(row, frozenset(sim_expected), "similarity")
        copied = dict(closed)
        matrix = _require_exact_list(copied["change_of_basis"], "change_of_basis")
        rational_rows: list[tuple[Fraction, ...]] = []
        for basis in matrix:
            basis_items = _require_exact_list(basis, "change-of-basis row")
            fractions: list[Fraction] = []
            for item in basis_items:
                pair = _require_exact_list(item, "rational pair")
                if len(pair) != 2:
                    raise ValueError("rational pair must contain numerator and denominator")
                fractions.append(Fraction(_require_exact_int(pair[0], "numerator"), _require_exact_int(pair[1], "denominator")))
            rational_rows.append(tuple(fractions))
        copied["change_of_basis"] = tuple(rational_rows)
        copied["controller_supplied_postfit_alignment"] = _require_exact_bool(copied["controller_supplied_postfit_alignment"], "controller_supplied_postfit_alignment")
        copied["all_legal_rows_agree_under_alignment"] = _require_exact_bool(copied["all_legal_rows_agree_under_alignment"], "all_legal_rows_agree_under_alignment")
        for name in ("anchor_count", "tested_state_count", "tested_legal_edge_count"):
            copied[name] = _require_exact_int(copied[name], name)
        return T2PairedSimilarity(**copied)
    similarities = tuple(decode_similarity(row) for row in _require_exact_list(raw["paired_similarities"], "paired_similarities"))
    campaign_teaching_expected = {"schema", "arm_summary_sha256s", "omission_control_count", "total_counterfactual_truth_selected_queries", "total_answer_free_singleton_inferences", "total_counterfactual_unqueried", "total_counterfactual_returned_categorical_labels", "total_closed_restricted_maps", "total_rank_closed_events", "total_new_membership_calls", "all_causal_primary_isolated", "all_truth_specific_noncausal", "all_selection_ineligible", "no_global_query_minimality_claim", "no_arbitrary_total_operator", "summary_sha256"}
    campaign_teaching_raw = _require_closed_object(raw["postfit_teaching_summary"], frozenset(campaign_teaching_expected), "campaign teaching summary")
    if campaign_teaching_raw["schema"] != TEACHING_CAMPAIGN_SCHEMA:
        raise ValueError("unknown campaign teaching summary schema")
    campaign_teaching = T2CampaignTeachingSummary(
        tuple(_require_exact_string(value, "arm teaching summary") for value in _require_exact_list(campaign_teaching_raw["arm_summary_sha256s"], "arm_summary_sha256s")),
        _require_exact_int(campaign_teaching_raw["omission_control_count"], "omission_control_count"),
        _require_exact_int(campaign_teaching_raw["total_counterfactual_truth_selected_queries"], "total_counterfactual_truth_selected_queries"),
        _require_exact_int(campaign_teaching_raw["total_answer_free_singleton_inferences"], "total_answer_free_singleton_inferences"),
        _require_exact_int(campaign_teaching_raw["total_counterfactual_unqueried"], "total_counterfactual_unqueried"),
        _require_exact_int(campaign_teaching_raw["total_counterfactual_returned_categorical_labels"], "total_counterfactual_returned_categorical_labels"),
        _require_exact_int(campaign_teaching_raw["total_closed_restricted_maps"], "total_closed_restricted_maps"),
        _require_exact_int(campaign_teaching_raw["total_rank_closed_events"], "total_rank_closed_events"),
        _require_exact_int(campaign_teaching_raw["total_new_membership_calls"], "total_new_membership_calls"),
        _require_exact_bool(campaign_teaching_raw["all_causal_primary_isolated"], "all_causal_primary_isolated"),
        _require_exact_bool(campaign_teaching_raw["all_truth_specific_noncausal"], "all_truth_specific_noncausal"),
        _require_exact_bool(campaign_teaching_raw["all_selection_ineligible"], "all_selection_ineligible"),
        _require_exact_bool(campaign_teaching_raw["no_global_query_minimality_claim"], "no_global_query_minimality_claim"),
        _require_exact_bool(campaign_teaching_raw["no_arbitrary_total_operator"], "no_arbitrary_total_operator"),
        _require_exact_string(campaign_teaching_raw["summary_sha256"], "summary_sha256"),
    )
    preopen_sha256s = tuple(_require_exact_string(value, "preopen digest") for value in _require_exact_list(raw["preopen_sha256s"], "preopen_sha256s"))
    return T2CampaignReport(
        _require_exact_string(raw["protocol_sha256"], "protocol_sha256"),
        _require_exact_string(raw["terminal_sha256"], "terminal_sha256"),
        preopen_sha256s,
        _require_exact_int(raw["total_legal_edges_checked"], "total_legal_edges_checked"),
        _require_exact_int(raw["total_undefined_pairs_checked"], "total_undefined_pairs_checked"),
        _require_exact_int(raw["total_long_paths_checked"], "total_long_paths_checked"),
        _require_exact_bool(raw["all_shortcuts_fail"], "all_shortcuts_fail"),
        arms,
        similarities,
        campaign_teaching,
        _require_exact_string(raw["report_sha256"], "report_sha256"),
    )


def load_phase3_t2_protocol(path: Path) -> Phase3T2Protocol:
    raw_value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonfinite_json,
    )
    protocol_keys = frozenset(
        {
            "schema",
            "execution_ready",
            "schedule_labels",
            "controller_nonces",
            "omission_salts",
            "omission_commitment_roots",
            "long_suite_roots",
            "source_runtime_binding",
            "active_budgets",
            "min_unopened_candidates",
            "protocol_sha256",
        }
    )
    raw = _require_closed_object(raw_value, protocol_keys, "T2 protocol")
    budgets_expected = frozenset(
        {
            "max_word_length",
            "max_candidate_requests",
            "max_active_calls",
            "max_structural_inferences",
            "max_returned_categorical_tokens",
            "max_outcome_branches_per_choice",
            "max_candidate_score_rows",
            "max_exact_rank_evaluations",
            "max_conditional_assignment_blocks_per_choice",
            "max_basis_image_candidates_per_choice",
            "max_materialized_versions_per_assignment",
            "max_validation_replay_decisions",
            "max_suffix_events_per_prediction",
            "max_certificate_bytes",
        }
    )
    budgets = _require_closed_object(raw["active_budgets"], budgets_expected, "T2 active budgets")
    exact_budgets = {name: _require_exact_int(value, name) for name, value in budgets.items()}
    binding_raw = raw["source_runtime_binding"]
    binding = None
    if binding_raw is not None:
        binding = _require_closed_object(
            binding_raw,
            frozenset({"required_file_sha256s", "python_version", "platform", "torch_version", "pyyaml_version", "device", "binding_sha256"}),
            "source/runtime binding",
        )
        rows_raw = _require_exact_list(binding["required_file_sha256s"], "required_file_sha256s")
        rows: list[tuple[str, str]] = []
        for raw_row in rows_raw:
            row = _require_exact_list(raw_row, "source digest row")
            if len(row) != 2:
                raise ValueError("source digest row must contain path and digest")
            rows.append((_require_exact_string(row[0], "source path"), _require_exact_string(row[1], "source digest")))
        binding = T2SourceRuntimeBinding(
            tuple(rows),
            _require_exact_string(binding["python_version"], "python_version"),
            _require_exact_string(binding["platform"], "platform"),
            _require_exact_string(binding["torch_version"], "torch_version"),
            _require_exact_string(binding["pyyaml_version"], "pyyaml_version"),
            _require_exact_string(binding["device"], "device"),
            _require_exact_string(binding["binding_sha256"], "binding_sha256"),
        )
    schedule = tuple(_require_exact_string(value, "schedule label") for value in _require_exact_list(raw["schedule_labels"], "schedule_labels"))
    nonces = tuple(_require_exact_string(value, "controller nonce") for value in _require_exact_list(raw["controller_nonces"], "controller_nonces"))
    salts = tuple(
        tuple(_require_exact_string(value, "controller salt") for value in _require_exact_list(row, "omission salt row"))
        for row in _require_exact_list(raw["omission_salts"], "omission_salts")
    )
    commitment_roots = tuple(_require_exact_string(value, "commitment root") for value in _require_exact_list(raw["omission_commitment_roots"], "omission_commitment_roots"))
    long_roots = tuple(_require_exact_string(value, "long-suite root") for value in _require_exact_list(raw["long_suite_roots"], "long_suite_roots"))
    return Phase3T2Protocol(
        _require_exact_bool(raw["execution_ready"], "execution_ready"),
        schedule,
        nonces,
        salts,
        commitment_roots,
        long_roots,
        binding,
        OpaqueActiveDiscoveryBudgets(**exact_budgets),
        _require_exact_int(raw["min_unopened_candidates"], "min_unopened_candidates"),
        _require_exact_string(raw["protocol_sha256"], "protocol_sha256"),
        _require_exact_string(raw["schema"], "schema"),
    )


def require_execution_ready(protocol: Phase3T2Protocol, repository_root: Path) -> None:
    if not protocol.execution_ready or protocol.source_runtime_binding is None:
        raise T2ProtocolError("T2 protocol is deliberately not execution-ready")
    protocol.source_runtime_binding.verify(repository_root)


__all__ = (
    "COMMITMENT_SCHEMA", "OPEN_SCHEMA", "PREOPEN_SCHEMA", "PROTOCOL_SCHEMA", "TERMINAL_SCHEMA", "TEACHING_SUMMARY_SCHEMA", "TEACHING_CAMPAIGN_SCHEMA", "SHORTCUT_BASELINE_SCHEMA", "SHORTCUT_T1_SCHEMA", "SHORTCUT_EVIDENCE_SCHEMA",
    "MerkleProof", "Phase3T2Protocol", "SaltedAnswerCommitment", "SaltedAnswerLeaf",
    "T2ControllerEnvironment", "T2PreopenEnvironment", "T2PreopenRecord", "T2ProtocolError", "T2SourceRuntimeBinding",
    "T2CampaignTerminalPreopen", "T2CampaignReport", "T2PostfitTeachingSummary", "T2CampaignTeachingSummary", "T2ShortcutBaselineEvaluation", "T2T1First14Nonidentification", "T2ShortcutEvidence", "build_t2_controller_environment", "freeze_t2_campaign_terminal",
    "aggregate_t2_preopen_records", "build_t2_scheduled_environment",
    "load_phase3_t2_protocol", "make_placeholder_protocol", "make_salted_answer_commitment",
    "load_t2_campaign_report", "load_t2_campaign_terminal", "load_t2_campaign_terminal_preopen",
    "load_t2_preopen_record", "load_t2_preopen_record_set",
    "make_source_runtime_binding", "make_source_runtime_binding_from_repository",
    "open_t2_campaign_postfit_batch", "open_t2_staged_campaign", "reconstruct_t2_campaign",
    "reconstruct_t2_preopen_record", "require_execution_ready", "t2_preopen_record_path",
    "validate_t2_staged_campaign", "write_t2_campaign_report", "write_t2_campaign_terminal",
    "write_t2_campaign_terminal_preopen", "write_t2_preopen_record",
    "run_t2_campaign", "run_t2_preopen_environment",
)
