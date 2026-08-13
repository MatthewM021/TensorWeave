"""Causal document-local dynamic-binding data for TNLM V3.

The public data objects deliberately separate model-visible event fields from
evaluation-only oracle routes, query targets, and dependency annotations.  A
surface key therefore cannot reveal its document-local lane through the model
input API.

Event semantics
---------------

``BIND`` introduces an inactive surface key with a mutable value. ``UPDATE``
applies a visible modular transform. ``COPY`` copies the current value of a
secondary live key into a primary live key. ``INVALIDATE`` expires a binding,
``QUERY`` asks for its current value, and ``DISTRACTOR`` touches no local
binding and may use either the global or null route.  Rebinding an expired key
creates a new generation.

Dependency parent zero is the primary binding's latest mutation (or the prior
invalidation for a rebind). Parent one is used only by ``COPY`` for the source
binding's latest mutation. All parents are strict causal indices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
from enum import IntEnum
import json
import math
import random
from typing import Iterable, Literal, Sequence

import torch
from torch import Tensor

from .routing import NULL_ROUTE


PAD_TOKEN_ID = 0
IGNORE_QUERY_TARGET = -100
NO_PARENT = -1
NO_GENERATION = -1

BindingSplit = Literal["train", "validation", "eval", "test"]
_SPLITS = frozenset(("train", "validation", "eval", "test"))
_HELDOUT_EXCLUDED_SPLITS = frozenset(("train", "validation"))


class BindingEventKind(IntEnum):
    """Model-visible event type IDs; zero is reserved for batch padding."""

    PAD = 0
    BIND = 1
    UPDATE = 2
    COPY = 3
    INVALIDATE = 4
    QUERY = 5
    DISTRACTOR = 6


_ENTITY_KINDS = frozenset(
    (
        BindingEventKind.BIND,
        BindingEventKind.UPDATE,
        BindingEventKind.COPY,
        BindingEventKind.INVALIDATE,
        BindingEventKind.QUERY,
    )
)


def _require_plain_int(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


@dataclass(frozen=True)
class BindingTaskConfig:
    """Length and vocabulary contract for dynamic-binding episodes.

    ``heldout_key_value_pairs`` use zero-based raw surface-key and value IDs.
    Train/validation generation excludes these pairs from every live state;
    eval/test generation forces at least one held-out bind and query.
    """

    num_surface_keys: int = 8
    value_cardinality: int = 8
    branches: int = 4
    max_live_bindings: int = 4
    min_length: int = 10
    max_length: int = 64
    heldout_key_value_pairs: tuple[tuple[int, int], ...] = ((0, 0),)
    global_distractor_probability: float = 0.5

    def __post_init__(self) -> None:
        for name, minimum in (
            ("num_surface_keys", 2),
            ("value_cardinality", 2),
            ("branches", 2),
            ("max_live_bindings", 2),
            ("min_length", 10),
            ("max_length", 10),
        ):
            _require_plain_int(name, getattr(self, name), minimum)
        if self.max_live_bindings > min(self.num_surface_keys, self.branches):
            raise ValueError(
                "max_live_bindings cannot exceed surface keys or local branches"
            )
        if self.min_length > self.max_length:
            raise ValueError("min_length cannot exceed max_length")
        probability = self.global_distractor_probability
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise TypeError("global_distractor_probability must be a real number")
        probability = float(probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("global_distractor_probability must be in [0, 1]")
        object.__setattr__(self, "global_distractor_probability", probability)

        normalized: list[tuple[int, int]] = []
        for pair in self.heldout_key_value_pairs:
            if len(pair) != 2:
                raise ValueError("held-out combinations must be (key, value) pairs")
            key, value = pair
            _require_plain_int("held-out key", key, 0)
            _require_plain_int("held-out value", value, 0)
            if key >= self.num_surface_keys or value >= self.value_cardinality:
                raise ValueError("held-out combination is outside the task vocabulary")
            normalized.append((key, value))
        if len(set(normalized)) != len(normalized):
            raise ValueError("held-out combinations must be unique")
        object.__setattr__(self, "heldout_key_value_pairs", tuple(normalized))

        if not _mandatory_candidates(self, exclude_heldout=True):
            raise ValueError(
                "held-out pairs leave no train-time bind/update/copy fixture"
            )

    @property
    def paths(self) -> int:
        """Local branches plus the dedicated global lane."""

        return self.branches + 1

    @property
    def vocab_size(self) -> int:
        """Size of the injective composite-event token vocabulary."""

        key_fields = self.num_surface_keys + 1
        value_fields = self.value_cardinality + 1
        return 1 + 6 * key_fields * key_fields * value_fields

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BindingModelInputs:
    """Only fields that a model may observe.

    Key and argument fields reserve zero for absent/padded values; real raw IDs
    are shifted by one. ``token_ids`` injectively combines all four event
    fields and also reserves zero exclusively for padding.
    """

    token_ids: Tensor
    event_kinds: Tensor
    primary_key_ids: Tensor
    secondary_key_ids: Tensor
    arguments: Tensor
    valid_mask: Tensor

    def to(self, device: torch.device | str) -> "BindingModelInputs":
        return BindingModelInputs(
            **{field.name: getattr(self, field.name).to(device) for field in fields(self)}
        )


@dataclass(frozen=True)
class BindingEvaluation:
    """Evaluation/training-label fields that must not enter autonomous routing."""

    oracle_routes: Tensor
    targets: Tensor
    dependency_parents: Tensor
    generation_ids: Tensor
    live_binding_counts: Tensor
    heldout_combination_mask: Tensor

    def to(self, device: torch.device | str) -> "BindingEvaluation":
        return BindingEvaluation(
            **{field.name: getattr(self, field.name).to(device) for field in fields(self)}
        )


@dataclass(frozen=True)
class BindingEpisode:
    """One unpadded document with structurally separated inputs and labels."""

    inputs: BindingModelInputs
    evaluation: BindingEvaluation
    split: str
    document_id: str
    generation_seed: int
    config_fingerprint: str

    @property
    def length(self) -> int:
        return int(self.inputs.token_ids.shape[0])

    def to(self, device: torch.device | str) -> "BindingEpisode":
        return BindingEpisode(
            inputs=self.inputs.to(device),
            evaluation=self.evaluation.to(device),
            split=self.split,
            document_id=self.document_id,
            generation_seed=self.generation_seed,
            config_fingerprint=self.config_fingerprint,
        )


@dataclass(frozen=True)
class BindingBatch:
    """A padded collection of variable-length binding documents."""

    inputs: BindingModelInputs
    evaluation: BindingEvaluation
    lengths: Tensor
    splits: tuple[str, ...]
    document_ids: tuple[str, ...]
    generation_seeds: tuple[int, ...]
    config_fingerprint: str

    @property
    def batch_size(self) -> int:
        return int(self.inputs.token_ids.shape[0])

    @property
    def padded_length(self) -> int:
        return int(self.inputs.token_ids.shape[1])

    def to(self, device: torch.device | str) -> "BindingBatch":
        return BindingBatch(
            inputs=self.inputs.to(device),
            evaluation=self.evaluation.to(device),
            lengths=self.lengths.to(device),
            splits=self.splits,
            document_ids=self.document_ids,
            generation_seeds=self.generation_seeds,
            config_fingerprint=self.config_fingerprint,
        )


def apply_value_transform(value: int, transform: int, cardinality: int) -> int:
    """Apply the visible update transform ``value + transform + 1 (mod V)``."""

    _require_plain_int("value", value, 0)
    _require_plain_int("transform", transform, 0)
    _require_plain_int("cardinality", cardinality, 2)
    if value >= cardinality or transform >= cardinality:
        raise ValueError("value and transform must be inside the value vocabulary")
    return (value + transform + 1) % cardinality


def encode_binding_token(
    config: BindingTaskConfig,
    kind: BindingEventKind | int,
    primary_key: int = -1,
    secondary_key: int = -1,
    argument: int = -1,
) -> int:
    """Encode one unpadded, model-visible composite event as a nonzero token."""

    try:
        event_kind = BindingEventKind(int(kind))
    except (TypeError, ValueError) as error:
        raise ValueError("unknown binding event kind") from error
    if event_kind is BindingEventKind.PAD:
        if (primary_key, secondary_key, argument) != (-1, -1, -1):
            raise ValueError("padding cannot carry event fields")
        return PAD_TOKEN_ID
    if not -1 <= primary_key < config.num_surface_keys:
        raise ValueError("primary_key outside the surface vocabulary")
    if not -1 <= secondary_key < config.num_surface_keys:
        raise ValueError("secondary_key outside the surface vocabulary")
    if not -1 <= argument < config.value_cardinality:
        raise ValueError("argument outside the value vocabulary")

    key_fields = config.num_surface_keys + 1
    value_fields = config.value_cardinality + 1
    encoded = int(event_kind) - 1
    encoded = encoded * key_fields + primary_key + 1
    encoded = encoded * key_fields + secondary_key + 1
    encoded = encoded * value_fields + argument + 1
    return encoded + 1


def _allowed_values(
    config: BindingTaskConfig, key: int, *, exclude_heldout: bool
) -> tuple[int, ...]:
    heldout = set(config.heldout_key_value_pairs) if exclude_heldout else set()
    return tuple(
        value
        for value in range(config.value_cardinality)
        if (key, value) not in heldout
    )


def _mandatory_candidates(
    config: BindingTaskConfig, *, exclude_heldout: bool
) -> list[tuple[int, int, int, int, int]]:
    """Return (A, initial-A, updated-A, B, initial-B) fixture choices."""

    result: list[tuple[int, int, int, int, int]] = []
    for a_key in range(config.num_surface_keys):
        a_values = _allowed_values(config, a_key, exclude_heldout=exclude_heldout)
        for a_value in a_values:
            for updated in a_values:
                if updated == a_value:
                    continue
                for b_key in range(config.num_surface_keys):
                    if b_key == a_key:
                        continue
                    b_values = _allowed_values(
                        config, b_key, exclude_heldout=exclude_heldout
                    )
                    if updated not in b_values:
                        continue
                    for b_value in b_values:
                        result.append((a_key, a_value, updated, b_key, b_value))
    return result


def _normalise_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized not in _SPLITS:
        raise ValueError(f"split must be one of {sorted(_SPLITS)}")
    return normalized


def _derived_seed(
    config: BindingTaskConfig, seed: int, split: str, document_index: int
) -> int:
    _require_plain_int("seed", seed, 0)
    _require_plain_int("document_index", document_index, 0)
    material = (
        f"tnlm-v3-binding|{config.fingerprint()}|{seed}|{split}|{document_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


@dataclass
class _LiveBinding:
    lane: int
    value: int
    generation: int
    last_mutation: int


class _EpisodeBuilder:
    def __init__(
        self, config: BindingTaskConfig, split: str, rng: random.Random
    ) -> None:
        self.config = config
        self.split = split
        self.rng = rng
        self.live: dict[int, _LiveBinding] = {}
        self.last_invalidation: dict[int, int] = {}
        self.next_generation = 0
        self.token_ids: list[int] = []
        self.event_kinds: list[int] = []
        self.primary_keys: list[int] = []
        self.secondary_keys: list[int] = []
        self.arguments: list[int] = []
        self.routes: list[int] = []
        self.targets: list[int] = []
        self.parents: list[tuple[int, int]] = []
        self.generations: list[int] = []
        self.live_counts: list[int] = []
        self.heldout_mask: list[bool] = []

    @property
    def exclude_heldout(self) -> bool:
        return self.split in _HELDOUT_EXCLUDED_SPLITS

    @property
    def free_lanes(self) -> list[int]:
        used = {binding.lane for binding in self.live.values()}
        return [lane for lane in range(self.config.branches) if lane not in used]

    def _emit(
        self,
        kind: BindingEventKind,
        *,
        primary_key: int = -1,
        secondary_key: int = -1,
        argument: int = -1,
        route: int,
        target: int = IGNORE_QUERY_TARGET,
        primary_parent: int = NO_PARENT,
        secondary_parent: int = NO_PARENT,
        generation: int = NO_GENERATION,
        combination_value: int | None = None,
        live_count: int,
    ) -> int:
        index = len(self.token_ids)
        self.token_ids.append(
            encode_binding_token(
                self.config, kind, primary_key, secondary_key, argument
            )
        )
        self.event_kinds.append(int(kind))
        self.primary_keys.append(primary_key + 1)
        self.secondary_keys.append(secondary_key + 1)
        self.arguments.append(argument + 1)
        self.routes.append(route)
        self.targets.append(target)
        self.parents.append((primary_parent, secondary_parent))
        self.generations.append(generation)
        self.live_counts.append(live_count)
        self.heldout_mask.append(
            combination_value is not None
            and (primary_key, combination_value)
            in set(self.config.heldout_key_value_pairs)
        )
        return index

    def bind(self, key: int, value: int, lane: int) -> None:
        if key in self.live or lane not in self.free_lanes:
            raise RuntimeError("attempted invalid bind")
        generation = self.next_generation
        self.next_generation += 1
        parent = self.last_invalidation.get(key, NO_PARENT)
        index = self._emit(
            BindingEventKind.BIND,
            primary_key=key,
            argument=value,
            route=lane,
            primary_parent=parent,
            generation=generation,
            combination_value=value,
            live_count=len(self.live) + 1,
        )
        self.live[key] = _LiveBinding(lane, value, generation, index)

    def update_to(self, key: int, new_value: int) -> None:
        binding = self.live[key]
        transform = (
            new_value - binding.value - 1
        ) % self.config.value_cardinality
        index = self._emit(
            BindingEventKind.UPDATE,
            primary_key=key,
            argument=transform,
            route=binding.lane,
            primary_parent=binding.last_mutation,
            generation=binding.generation,
            combination_value=new_value,
            live_count=len(self.live),
        )
        binding.value = new_value
        binding.last_mutation = index

    def copy(self, destination: int, source: int) -> None:
        dest = self.live[destination]
        src = self.live[source]
        index = self._emit(
            BindingEventKind.COPY,
            primary_key=destination,
            secondary_key=source,
            route=dest.lane,
            primary_parent=dest.last_mutation,
            secondary_parent=src.last_mutation,
            generation=dest.generation,
            combination_value=src.value,
            live_count=len(self.live),
        )
        dest.value = src.value
        dest.last_mutation = index

    def invalidate(self, key: int) -> int:
        binding = self.live[key]
        index = self._emit(
            BindingEventKind.INVALIDATE,
            primary_key=key,
            route=binding.lane,
            primary_parent=binding.last_mutation,
            generation=binding.generation,
            combination_value=binding.value,
            live_count=len(self.live) - 1,
        )
        del self.live[key]
        self.last_invalidation[key] = index
        return binding.lane

    def query(self, key: int) -> None:
        binding = self.live[key]
        self._emit(
            BindingEventKind.QUERY,
            primary_key=key,
            route=binding.lane,
            target=binding.value,
            primary_parent=binding.last_mutation,
            generation=binding.generation,
            combination_value=binding.value,
            live_count=len(self.live),
        )

    def distractor(self) -> None:
        is_global = self.rng.random() < self.config.global_distractor_probability
        route = self.config.branches if is_global else NULL_ROUTE
        self._emit(
            BindingEventKind.DISTRACTOR,
            # A causal scope bit makes the non-permutation-symmetric global
            # versus null state effect model-visible rather than hidden RNG.
            argument=0 if is_global else 1,
            route=route,
            live_count=len(self.live),
        )

    def random_event(self) -> None:
        allowed_by_key = {
            key: _allowed_values(
                self.config, key, exclude_heldout=self.exclude_heldout
            )
            for key in range(self.config.num_surface_keys)
        }
        inactive = [
            key
            for key in range(self.config.num_surface_keys)
            if key not in self.live and allowed_by_key[key]
        ]
        bindable = bool(
            inactive
            and self.free_lanes
            and len(self.live) < self.config.max_live_bindings
        )
        updates = [
            (key, value)
            for key, binding in self.live.items()
            for value in allowed_by_key[key]
            if value != binding.value
        ]
        copies = [
            (destination, source)
            for destination in self.live
            for source, source_binding in self.live.items()
            if destination != source
            and source_binding.value in allowed_by_key[destination]
        ]

        choices: list[str] = ["distractor", "distractor"]
        if self.live:
            choices.extend(("query", "query", "query", "query", "invalidate"))
        if updates:
            choices.extend(("update", "update", "update"))
        if copies:
            choices.extend(("copy", "copy"))
        if bindable:
            choices.extend(("bind", "bind"))
        choice = self.rng.choice(choices)

        if choice == "query":
            self.query(self.rng.choice(sorted(self.live)))
        elif choice == "update":
            key, value = self.rng.choice(updates)
            self.update_to(key, value)
        elif choice == "copy":
            destination, source = self.rng.choice(copies)
            self.copy(destination, source)
        elif choice == "invalidate":
            self.invalidate(self.rng.choice(sorted(self.live)))
        elif choice == "bind":
            rebound = [key for key in inactive if key in self.last_invalidation]
            key = self.rng.choice(rebound or inactive)
            value = self.rng.choice(allowed_by_key[key])
            self.bind(key, value, self.rng.choice(self.free_lanes))
        else:
            self.distractor()

    def finish(
        self,
        *,
        split: str,
        document_id: str,
        generation_seed: int,
        config_fingerprint: str,
    ) -> BindingEpisode:
        length = len(self.token_ids)
        return BindingEpisode(
            inputs=BindingModelInputs(
                token_ids=torch.tensor(self.token_ids, dtype=torch.int64),
                event_kinds=torch.tensor(self.event_kinds, dtype=torch.int64),
                primary_key_ids=torch.tensor(self.primary_keys, dtype=torch.int64),
                secondary_key_ids=torch.tensor(self.secondary_keys, dtype=torch.int64),
                arguments=torch.tensor(self.arguments, dtype=torch.int64),
                valid_mask=torch.ones(length, dtype=torch.bool),
            ),
            evaluation=BindingEvaluation(
                oracle_routes=torch.tensor(self.routes, dtype=torch.int64),
                targets=torch.tensor(self.targets, dtype=torch.int64),
                dependency_parents=torch.tensor(self.parents, dtype=torch.int64),
                generation_ids=torch.tensor(self.generations, dtype=torch.int64),
                live_binding_counts=torch.tensor(self.live_counts, dtype=torch.int64),
                heldout_combination_mask=torch.tensor(
                    self.heldout_mask, dtype=torch.bool
                ),
            ),
            split=split,
            document_id=document_id,
            generation_seed=generation_seed,
            config_fingerprint=config_fingerprint,
        )


def generate_binding_episode(
    config: BindingTaskConfig,
    *,
    length: int,
    seed: int,
    split: BindingSplit = "train",
    document_index: int = 0,
) -> BindingEpisode:
    """Generate one deterministic causal document.

    The first ten events deliberately exercise every event family, an exact
    query, a two-parent copy, invalidation, and rebinding. Remaining events are
    sampled subject to live-binding and held-out-combination constraints.
    """

    _require_plain_int("length", length, config.min_length)
    if length > config.max_length:
        raise ValueError("length exceeds config.max_length")
    normalized_split = _normalise_split(split)
    derived_seed = _derived_seed(config, seed, normalized_split, document_index)
    rng = random.Random(derived_seed)
    builder = _EpisodeBuilder(config, normalized_split, rng)

    exclude = normalized_split in _HELDOUT_EXCLUDED_SPLITS
    candidates = _mandatory_candidates(config, exclude_heldout=exclude)
    if not exclude and config.heldout_key_value_pairs:
        a_key, a_value = config.heldout_key_value_pairs[
            document_index % len(config.heldout_key_value_pairs)
        ]
        updated_values = [
            value for value in range(config.value_cardinality) if value != a_value
        ]
        a_updated = rng.choice(updated_values)
        b_keys = [key for key in range(config.num_surface_keys) if key != a_key]
        b_key = rng.choice(b_keys)
        b_value = rng.randrange(config.value_cardinality)
    else:
        candidate_keys = sorted({candidate[0] for candidate in candidates})
        a_key = candidate_keys[document_index % len(candidate_keys)]
        choices = [candidate for candidate in candidates if candidate[0] == a_key]
        a_key, a_value, a_updated, b_key, b_value = rng.choice(choices)

    # Each document receives its own seeded branch permutation.  The route is
    # not a global function of the surface key or the fixture position.
    lane_order = list(range(config.branches))
    rng.shuffle(lane_order)
    lane_a, lane_b = lane_order[:2]

    builder.bind(a_key, a_value, lane_a)
    builder.bind(b_key, b_value, lane_b)
    builder.query(a_key)
    builder.update_to(a_key, a_updated)
    builder.copy(b_key, a_key)
    builder.query(b_key)
    old_lane = builder.invalidate(a_key)

    rebind_values = _allowed_values(
        config, a_key, exclude_heldout=builder.exclude_heldout
    )
    rebind_value = rng.choice(rebind_values)
    preferred_lanes = [lane for lane in builder.free_lanes if lane != old_lane]
    builder.bind(
        a_key,
        rebind_value,
        rng.choice(preferred_lanes or builder.free_lanes),
    )
    builder.distractor()
    builder.query(a_key)

    while len(builder.token_ids) < length:
        builder.random_event()

    document_id = f"{normalized_split}:{document_index}:{derived_seed:016x}"
    episode = builder.finish(
        split=normalized_split,
        document_id=document_id,
        generation_seed=derived_seed,
        config_fingerprint=config.fingerprint(),
    )
    validate_binding_episode(episode, config)
    return episode


def generate_binding_episodes(
    config: BindingTaskConfig,
    *,
    count: int,
    seed: int,
    split: BindingSplit = "train",
    lengths: Sequence[int] | None = None,
) -> tuple[BindingEpisode, ...]:
    """Generate a deterministic set, choosing varied lengths when omitted."""

    _require_plain_int("count", count, 1)
    normalized_split = _normalise_split(split)
    if lengths is not None:
        if len(lengths) != count:
            raise ValueError("lengths must contain one entry per episode")
        chosen_lengths = list(lengths)
    else:
        rng = random.Random(_derived_seed(config, seed, normalized_split, 2**31))
        chosen_lengths = [
            rng.randint(config.min_length, config.max_length) for _ in range(count)
        ]
        if count >= 2 and config.min_length < config.max_length:
            chosen_lengths[0] = config.min_length
            chosen_lengths[1] = config.max_length
    return tuple(
        generate_binding_episode(
            config,
            length=int(chosen_lengths[index]),
            seed=seed,
            split=normalized_split,
            document_index=index,
        )
        for index in range(count)
    )


def collate_binding_episodes(
    episodes: Sequence[BindingEpisode], *, pad_to_length: int | None = None
) -> BindingBatch:
    """Pad episodes with zero-valued model fields and ignored evaluation labels."""

    if not episodes:
        raise ValueError("cannot collate an empty episode collection")
    fingerprint = episodes[0].config_fingerprint
    if any(episode.config_fingerprint != fingerprint for episode in episodes):
        raise ValueError("all episodes must use the same task configuration")
    device = episodes[0].inputs.token_ids.device
    if any(episode.inputs.token_ids.device != device for episode in episodes):
        raise ValueError("all episodes must share one device")
    lengths = [episode.length for episode in episodes]
    width = max(lengths)
    if pad_to_length is not None:
        _require_plain_int("pad_to_length", pad_to_length, width)
        width = pad_to_length
    batch = len(episodes)

    def full(fill: int | bool, *, dtype: torch.dtype, tail: tuple[int, ...] = ()) -> Tensor:
        return torch.full((batch, width, *tail), fill, dtype=dtype, device=device)

    token_ids = full(PAD_TOKEN_ID, dtype=torch.int64)
    event_kinds = full(int(BindingEventKind.PAD), dtype=torch.int64)
    primary = full(0, dtype=torch.int64)
    secondary = full(0, dtype=torch.int64)
    arguments = full(0, dtype=torch.int64)
    valid = full(False, dtype=torch.bool)
    routes = full(NULL_ROUTE, dtype=torch.int64)
    targets = full(IGNORE_QUERY_TARGET, dtype=torch.int64)
    parents = full(NO_PARENT, dtype=torch.int64, tail=(2,))
    generations = full(NO_GENERATION, dtype=torch.int64)
    live_counts = full(0, dtype=torch.int64)
    heldout = full(False, dtype=torch.bool)

    for row, episode in enumerate(episodes):
        length = episode.length
        source_inputs = episode.inputs
        source_eval = episode.evaluation
        token_ids[row, :length] = source_inputs.token_ids
        event_kinds[row, :length] = source_inputs.event_kinds
        primary[row, :length] = source_inputs.primary_key_ids
        secondary[row, :length] = source_inputs.secondary_key_ids
        arguments[row, :length] = source_inputs.arguments
        valid[row, :length] = source_inputs.valid_mask
        routes[row, :length] = source_eval.oracle_routes
        targets[row, :length] = source_eval.targets
        parents[row, :length] = source_eval.dependency_parents
        generations[row, :length] = source_eval.generation_ids
        live_counts[row, :length] = source_eval.live_binding_counts
        heldout[row, :length] = source_eval.heldout_combination_mask

    result = BindingBatch(
        inputs=BindingModelInputs(
            token_ids, event_kinds, primary, secondary, arguments, valid
        ),
        evaluation=BindingEvaluation(
            routes, targets, parents, generations, live_counts, heldout
        ),
        lengths=torch.tensor(lengths, dtype=torch.int64, device=device),
        splits=tuple(episode.split for episode in episodes),
        document_ids=tuple(episode.document_id for episode in episodes),
        generation_seeds=tuple(episode.generation_seed for episode in episodes),
        config_fingerprint=fingerprint,
    )
    return result


def _check_tensor(
    tensor: Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    name: str,
) -> None:
    if not isinstance(tensor, Tensor) or tensor.shape != shape or tensor.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")


def validate_binding_episode(
    episode: BindingEpisode, config: BindingTaskConfig
) -> None:
    """Validate shapes and replay all causal route/target/dependency semantics."""

    if episode.config_fingerprint != config.fingerprint():
        raise ValueError("episode configuration fingerprint mismatch")
    split = _normalise_split(episode.split)
    length = episode.length
    if not config.min_length <= length <= config.max_length:
        raise ValueError("episode length outside configured range")
    inputs, evaluation = episode.inputs, episode.evaluation
    for name in (
        "token_ids",
        "event_kinds",
        "primary_key_ids",
        "secondary_key_ids",
        "arguments",
    ):
        _check_tensor(getattr(inputs, name), (length,), torch.int64, name)
    _check_tensor(inputs.valid_mask, (length,), torch.bool, "valid_mask")
    for name in (
        "oracle_routes",
        "targets",
        "generation_ids",
        "live_binding_counts",
    ):
        _check_tensor(getattr(evaluation, name), (length,), torch.int64, name)
    _check_tensor(
        evaluation.dependency_parents,
        (length, 2),
        torch.int64,
        "dependency_parents",
    )
    _check_tensor(
        evaluation.heldout_combination_mask,
        (length,),
        torch.bool,
        "heldout_combination_mask",
    )
    tensors = [getattr(inputs, field.name) for field in fields(inputs)] + [
        getattr(evaluation, field.name) for field in fields(evaluation)
    ]
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all episode tensors must share one device")
    if not bool(inputs.valid_mask.all()):
        raise ValueError("an episode is unpadded; every position must be valid")

    live: dict[int, _LiveBinding] = {}
    last_invalidation: dict[int, int] = {}
    seen_generations: set[int] = set()
    observed_kinds: set[BindingEventKind] = set()
    heldout = set(config.heldout_key_value_pairs)

    for index in range(length):
        try:
            kind = BindingEventKind(int(inputs.event_kinds[index]))
        except ValueError as error:
            raise ValueError("unknown event kind") from error
        if kind is BindingEventKind.PAD:
            raise ValueError("padding cannot occur inside an episode")
        observed_kinds.add(kind)
        primary = int(inputs.primary_key_ids[index]) - 1
        secondary = int(inputs.secondary_key_ids[index]) - 1
        argument = int(inputs.arguments[index]) - 1
        route = int(evaluation.oracle_routes[index])
        target = int(evaluation.targets[index])
        parent0, parent1 = map(
            int, evaluation.dependency_parents[index].tolist()
        )
        generation = int(evaluation.generation_ids[index])
        if parent0 >= index or parent1 >= index or parent0 < -1 or parent1 < -1:
            raise ValueError("dependency parents must be strict causal indices")
        expected_token = encode_binding_token(
            config, kind, primary, secondary, argument
        )
        if int(inputs.token_ids[index]) != expected_token:
            raise ValueError("composite token does not match visible event fields")

        combination: tuple[int, int] | None = None
        if kind is BindingEventKind.BIND:
            if not 0 <= primary < config.num_surface_keys or secondary != -1:
                raise ValueError("bind fields are invalid")
            if not 0 <= argument < config.value_cardinality or primary in live:
                raise ValueError("bind requires an inactive key and valid value")
            if not 0 <= route < config.branches:
                raise ValueError("entity events require a local oracle lane")
            if route in {binding.lane for binding in live.values()}:
                raise ValueError("simultaneously live bindings cannot share a lane")
            if parent0 != last_invalidation.get(primary, NO_PARENT) or parent1 != -1:
                raise ValueError("bind dependency parent is incorrect")
            if generation < 0 or generation in seen_generations:
                raise ValueError("a bind must introduce a unique generation")
            seen_generations.add(generation)
            live[primary] = _LiveBinding(route, argument, generation, index)
            combination = (primary, argument)
        elif kind is BindingEventKind.UPDATE:
            if primary not in live or secondary != -1:
                raise ValueError("update requires one live primary key")
            binding = live[primary]
            if not 0 <= argument < config.value_cardinality:
                raise ValueError("update transform is invalid")
            if (route, generation, parent0, parent1) != (
                binding.lane,
                binding.generation,
                binding.last_mutation,
                -1,
            ):
                raise ValueError("update route, generation, or parent is incorrect")
            binding.value = apply_value_transform(
                binding.value, argument, config.value_cardinality
            )
            binding.last_mutation = index
            combination = (primary, binding.value)
        elif kind is BindingEventKind.COPY:
            if primary not in live or secondary not in live or primary == secondary:
                raise ValueError("copy requires distinct live primary and secondary keys")
            if argument != -1:
                raise ValueError("copy has no argument")
            dest, source = live[primary], live[secondary]
            if (route, generation, parent0, parent1) != (
                dest.lane,
                dest.generation,
                dest.last_mutation,
                source.last_mutation,
            ):
                raise ValueError("copy route, generation, or parents are incorrect")
            dest.value = source.value
            dest.last_mutation = index
            combination = (primary, dest.value)
        elif kind is BindingEventKind.INVALIDATE:
            if primary not in live or secondary != -1 or argument != -1:
                raise ValueError("invalidate requires one live primary key")
            binding = live[primary]
            if (route, generation, parent0, parent1) != (
                binding.lane,
                binding.generation,
                binding.last_mutation,
                -1,
            ):
                raise ValueError("invalidate route, generation, or parent is incorrect")
            combination = (primary, binding.value)
            del live[primary]
            last_invalidation[primary] = index
        elif kind is BindingEventKind.QUERY:
            if primary not in live or secondary != -1 or argument != -1:
                raise ValueError("query requires one live primary key")
            binding = live[primary]
            if (route, generation, parent0, parent1) != (
                binding.lane,
                binding.generation,
                binding.last_mutation,
                -1,
            ):
                raise ValueError("query route, generation, or parent is incorrect")
            if target != binding.value:
                raise ValueError("query target does not equal the causal binding value")
            combination = (primary, binding.value)
        else:
            if primary != -1 or secondary != -1 or argument not in (0, 1):
                raise ValueError("distractors require one visible scope bit")
            if route not in (NULL_ROUTE, config.branches):
                raise ValueError("distractor route must be null or global")
            expected_route = config.branches if argument == 0 else NULL_ROUTE
            if route != expected_route:
                raise ValueError("distractor scope bit does not match its route")
            if (parent0, parent1, generation) != (-1, -1, -1):
                raise ValueError("distractors cannot have dependency labels")

        if kind is not BindingEventKind.QUERY and target != IGNORE_QUERY_TARGET:
            raise ValueError("query targets must be ignored outside query events")
        expected_heldout = combination in heldout if combination is not None else False
        if bool(evaluation.heldout_combination_mask[index]) != expected_heldout:
            raise ValueError("held-out combination annotation is incorrect")
        if int(evaluation.live_binding_counts[index]) != len(live):
            raise ValueError("live-binding count annotation is incorrect")
        if len(live) > config.max_live_bindings:
            raise ValueError("simultaneous live-binding cap exceeded")

    required = set(BindingEventKind) - {BindingEventKind.PAD}
    if not required.issubset(observed_kinds):
        raise ValueError("episode does not cover every required event family")
    heldout_mask = evaluation.heldout_combination_mask
    if split in _HELDOUT_EXCLUDED_SPLITS and bool(heldout_mask.any()):
        raise ValueError("train/validation episode contains held-out combinations")
    heldout_query = heldout_mask & (
        inputs.event_kinds == int(BindingEventKind.QUERY)
    )
    if split not in _HELDOUT_EXCLUDED_SPLITS and heldout and not bool(
        heldout_query.any()
    ):
        raise ValueError("eval/test episode must query a held-out combination")


def validate_binding_batch(batch: BindingBatch, config: BindingTaskConfig) -> None:
    """Validate padding and replay each unpadded row as an episode."""

    if batch.config_fingerprint != config.fingerprint():
        raise ValueError("batch configuration fingerprint mismatch")
    n, width = batch.inputs.token_ids.shape
    if n <= 0 or width <= 0 or batch.lengths.shape != (n,):
        raise ValueError("batch and lengths have incompatible shapes")
    if len(batch.splits) != n or len(batch.document_ids) != n:
        raise ValueError("batch metadata must contain one entry per row")
    if len(batch.generation_seeds) != n:
        raise ValueError("batch generation seeds must contain one entry per row")
    for name in (
        "event_kinds",
        "primary_key_ids",
        "secondary_key_ids",
        "arguments",
        "valid_mask",
    ):
        if getattr(batch.inputs, name).shape != (n, width):
            raise ValueError("batch model-input shapes are inconsistent")
    for name in (
        "oracle_routes",
        "targets",
        "generation_ids",
        "live_binding_counts",
        "heldout_combination_mask",
    ):
        if getattr(batch.evaluation, name).shape != (n, width):
            raise ValueError("batch evaluation shapes are inconsistent")
    if batch.evaluation.dependency_parents.shape != (n, width, 2):
        raise ValueError("batch dependency-parent shape is inconsistent")

    for row in range(n):
        length = int(batch.lengths[row])
        if not config.min_length <= length <= width:
            raise ValueError("batch length is invalid")
        expected_valid = torch.arange(width, device=batch.lengths.device) < length
        if not torch.equal(batch.inputs.valid_mask[row], expected_valid):
            raise ValueError("batch valid mask must be a contiguous prefix")
        padded = slice(length, width)
        for name in (
            "token_ids",
            "event_kinds",
            "primary_key_ids",
            "secondary_key_ids",
            "arguments",
        ):
            if bool((getattr(batch.inputs, name)[row, padded] != 0).any()):
                raise ValueError("model-visible padding fields must be zero")
        if bool((batch.evaluation.oracle_routes[row, padded] != NULL_ROUTE).any()):
            raise ValueError("padded oracle routes must use NULL_ROUTE")
        if bool(
            (batch.evaluation.targets[row, padded] != IGNORE_QUERY_TARGET).any()
        ):
            raise ValueError("padded query targets must be ignored")
        if bool((batch.evaluation.dependency_parents[row, padded] != -1).any()):
            raise ValueError("padded dependency parents must be absent")
        if bool((batch.evaluation.generation_ids[row, padded] != -1).any()):
            raise ValueError("padded generation IDs must be absent")
        if bool((batch.evaluation.live_binding_counts[row, padded] != 0).any()):
            raise ValueError("padded live-binding counts must be zero")
        if bool(batch.evaluation.heldout_combination_mask[row, padded].any()):
            raise ValueError("padding cannot be a held-out combination")

        inputs = BindingModelInputs(
            **{
                field.name: getattr(batch.inputs, field.name)[row, :length]
                for field in fields(BindingModelInputs)
            }
        )
        evaluation = BindingEvaluation(
            **{
                field.name: getattr(batch.evaluation, field.name)[row, :length]
                for field in fields(BindingEvaluation)
            }
        )
        validate_binding_episode(
            BindingEpisode(
                inputs=inputs,
                evaluation=evaluation,
                split=batch.splits[row],
                document_id=batch.document_ids[row],
                generation_seed=batch.generation_seeds[row],
                config_fingerprint=batch.config_fingerprint,
            ),
            config,
        )


__all__ = [
    "BindingBatch",
    "BindingEpisode",
    "BindingEvaluation",
    "BindingEventKind",
    "BindingModelInputs",
    "BindingTaskConfig",
    "IGNORE_QUERY_TARGET",
    "NO_GENERATION",
    "NO_PARENT",
    "PAD_TOKEN_ID",
    "apply_value_transform",
    "collate_binding_episodes",
    "encode_binding_token",
    "generate_binding_episode",
    "generate_binding_episodes",
    "validate_binding_batch",
    "validate_binding_episode",
]
