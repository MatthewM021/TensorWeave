"""Scale-shared routed binary-counter forest for TNLM V3.

The state is a collection of dyadic blocks.  For every routing lane, bit ``s``
of ``counts`` is represented by one occupied slot at scale ``s``.  Streaming
updates and packed parallel reduction use the same chronological merge DAG:
the older block is always the left operand and the newer block the right.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .operators import ScaleSharedCPMerge, analytic_scale_features
from .routing import NULL_ROUTE, route_counts, routed_lane_mask, validate_routes


@dataclass(frozen=True)
class ForestConfig:
    """Length-independent construction settings for a routed forest model."""

    branches: int
    d_model: int
    cp_rank: int
    vocab_size: int = 256
    output_size: int | None = None
    scale_feature_dim: int = 8
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        for name in ("branches", "d_model", "cp_rank", "vocab_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.output_size is not None and self.output_size <= 0:
            raise ValueError("output_size must be positive when supplied")
        if self.scale_feature_dim <= 0:
            raise ValueError("scale_feature_dim must be positive")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be inside the vocabulary")

    @property
    def paths(self) -> int:
        """Local branches plus one dedicated global lane."""

        return self.branches + 1

    @property
    def resolved_output_size(self) -> int:
        return self.vocab_size if self.output_size is None else self.output_size

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass
class ForestState:
    """Batched binary-counter state.

    Shapes are ``slots[N,P,S,D]``, ``occupied[N,P,S]``, ``counts[N,P]``,
    and ``valid_steps[N]``.  ``P`` includes the dedicated global lane.
    """

    slots: Tensor
    occupied: Tensor
    counts: Tensor
    valid_steps: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.slots.shape[0])

    @property
    def paths(self) -> int:
        return int(self.slots.shape[1])

    @property
    def scales(self) -> int:
        return int(self.slots.shape[2])

    @property
    def d_model(self) -> int:
        return int(self.slots.shape[3])

    def detach(self) -> "ForestState":
        return ForestState(
            slots=self.slots.detach(),
            occupied=self.occupied.detach(),
            counts=self.counts.detach(),
            valid_steps=self.valid_steps.detach(),
        )

    def to(self, *args: object, **kwargs: object) -> "ForestState":
        slots = self.slots.to(*args, **kwargs)
        return ForestState(
            slots=slots,
            occupied=self.occupied.to(device=slots.device),
            counts=self.counts.to(device=slots.device),
            valid_steps=self.valid_steps.to(device=slots.device),
        )


@dataclass
class ForestRun:
    state: ForestState
    merge_count: Tensor


@dataclass
class ForestPrefixRun:
    states: tuple[ForestState, ...]
    merge_count: Tensor


@dataclass
class ForestModelOutput:
    logits: Tensor
    state: ForestState
    merge_count: Tensor


def _required_scales(counts: Tensor) -> int:
    maximum = int(counts.max().item()) if counts.numel() else 0
    return max(1, maximum.bit_length())


def _routed_counts_fit_valid_steps(counts: Tensor, valid_steps: Tensor) -> bool:
    """Check row sums with Python integers so int64 accumulation cannot wrap."""

    rows = counts.detach().cpu().tolist()
    clocks = valid_steps.detach().cpu().tolist()
    return all(sum(int(value) for value in row) <= int(clock) for row, clock in zip(rows, clocks, strict=True))


def _validate_state(
    state: ForestState,
    *,
    batch_size: int,
    paths: int,
    d_model: int,
) -> None:
    if state.slots.ndim != 4:
        raise ValueError("state.slots must have shape [N,P,S,D]")
    n, p, s, d = state.slots.shape
    if (n, p, d) != (batch_size, paths, d_model) or not 1 <= s <= 63:
        raise ValueError("state dimensions do not match the forest")
    if state.occupied.shape != (n, p, s) or state.occupied.dtype != torch.bool:
        raise ValueError("state.occupied must be boolean with shape [N,P,S]")
    if state.counts.shape != (n, p) or state.counts.dtype != torch.int64:
        raise ValueError("state.counts must be int64 with shape [N,P]")
    if state.valid_steps.shape != (n,) or state.valid_steps.dtype != torch.int64:
        raise ValueError("state.valid_steps must be int64 with shape [N]")
    if state.slots.device != state.occupied.device or state.slots.device != state.counts.device:
        raise ValueError("all state tensors must share a device")
    if state.slots.device != state.valid_steps.device:
        raise ValueError("all state tensors must share a device")
    if bool((state.counts < 0).any()) or bool((state.valid_steps < 0).any()):
        raise ValueError("state counters cannot be negative")
    if not _routed_counts_fit_valid_steps(state.counts, state.valid_steps):
        raise ValueError("routed counts cannot exceed valid steps")
    scale_ids = torch.arange(s, device=state.counts.device, dtype=torch.int64)
    expected = ((state.counts.unsqueeze(-1) >> scale_ids) & 1).bool()
    if not torch.equal(expected, state.occupied):
        raise ValueError("state occupancy does not equal the binary count pattern")
    if s < 63 and bool((state.counts >= (1 << s)).any()):
        raise ValueError("state lacks capacity for its binary counts")
    inactive_values = state.slots.masked_select(~state.occupied.unsqueeze(-1))
    if bool((inactive_values != 0).any()):
        raise ValueError("unoccupied state slots must be zero")


class ScaleSharedBinaryForest(nn.Module):
    """Routed forest with one merge DAG and declared-tolerance numeric parity."""

    def __init__(
        self,
        *,
        d_model: int,
        branches: int,
        cp_rank: int,
        scale_feature_dim: int = 8,
        merge: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if min(d_model, branches, cp_rank, scale_feature_dim) <= 0:
            raise ValueError("forest dimensions must be positive")
        self.d_model = d_model
        self.branches = branches
        self.paths = branches + 1
        self.cp_rank = cp_rank
        self.scale_feature_dim = scale_feature_dim
        self.merge = merge or ScaleSharedCPMerge(
            d_model=d_model,
            cp_rank=cp_rank,
            scale_feature_dim=scale_feature_dim,
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ForestState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        reference = next(self.parameters(), None)
        if device is None and reference is not None:
            device = reference.device
        if dtype is None:
            dtype = reference.dtype if reference is not None else torch.get_default_dtype()
        return ForestState(
            slots=torch.zeros(
                batch_size, self.paths, 1, self.d_model, device=device, dtype=dtype
            ),
            occupied=torch.zeros(batch_size, self.paths, 1, device=device, dtype=torch.bool),
            counts=torch.zeros(batch_size, self.paths, device=device, dtype=torch.int64),
            valid_steps=torch.zeros(batch_size, device=device, dtype=torch.int64),
        )

    def step(
        self,
        state: ForestState,
        event: Tensor,
        route: Tensor,
        valid: Tensor,
    ) -> ForestRun:
        """Apply one event without mutating ``state``.

        Invalid events are total no-ops.  A valid ``NULL_ROUTE`` advances the
        valid-event clock but does not modify a lane.  Routed events update
        their local branch or the dedicated global lane before readout.
        """

        if event.ndim != 2 or event.shape[1] != self.d_model:
            raise ValueError("event must have shape [N,D]")
        n = int(event.shape[0])
        if route.shape != (n,) or valid.shape != (n,):
            raise ValueError("route and valid must have shape [N]")
        if valid.dtype != torch.bool:
            raise TypeError("valid must be boolean")
        if event.device != route.device or event.device != valid.device:
            raise ValueError("event, route, and valid must share a device")
        if event.device != state.slots.device or event.dtype != state.slots.dtype:
            raise ValueError("event must match the state device and dtype")
        _validate_state(
            state, batch_size=n, paths=self.paths, d_model=self.d_model
        )
        validate_routes(route.unsqueeze(1), valid.unsqueeze(1), self.branches)

        safe_route = route.to(torch.int64).clamp(min=0, max=self.paths - 1)
        routed = valid & (route != NULL_ROUTE)
        lane_mask = F.one_hot(safe_route, num_classes=self.paths).bool()
        lane_mask = lane_mask & routed.unsqueeze(-1)

        counts = state.counts + lane_mask.to(torch.int64)
        valid_steps = state.valid_steps + valid.to(torch.int64)
        scales = max(state.scales, _required_scales(counts))
        if scales > state.scales:
            extra = scales - state.scales
            slots = torch.cat(
                (
                    state.slots,
                    state.slots.new_zeros(n, self.paths, extra, self.d_model),
                ),
                dim=2,
            )
            occupied = torch.cat(
                (
                    state.occupied,
                    torch.zeros(
                        n, self.paths, extra, device=event.device, dtype=torch.bool
                    ),
                ),
                dim=2,
            )
        else:
            slots = state.slots
            occupied = state.occupied

        carry = event.unsqueeze(1).expand(-1, self.paths, -1)
        has_carry = lane_mask
        global_path = (
            torch.arange(self.paths, device=event.device) == self.branches
        ).unsqueeze(0)
        global_path = global_path.expand(n, -1)
        slot_slices: list[Tensor] = []
        occupied_slices: list[Tensor] = []
        merge_count = torch.zeros((), device=event.device, dtype=torch.int64)

        for scale in range(scales):
            previous = slots[:, :, scale, :]
            was_occupied = occupied[:, :, scale]
            place = has_carry & ~was_occupied
            collide = has_carry & was_occupied
            next_carry = torch.zeros_like(carry)
            if bool(collide.any()):
                merged = self.merge(
                    previous[collide],
                    carry[collide],
                    scale=scale,
                    global_path=global_path[collide],
                )
                indices = collide.nonzero(as_tuple=True)
                next_carry = next_carry.index_put(indices, merged)
            slot_slices.append(
                torch.where(
                    place.unsqueeze(-1),
                    carry,
                    torch.where(collide.unsqueeze(-1), torch.zeros_like(previous), previous),
                )
            )
            occupied_slices.append(
                torch.where(
                    place,
                    torch.ones_like(was_occupied),
                    torch.where(collide, torch.zeros_like(was_occupied), was_occupied),
                )
            )
            carry = next_carry
            has_carry = collide
            merge_count = merge_count + collide.sum(dtype=torch.int64)

        if bool(has_carry.any()):
            raise RuntimeError("internal error: scale capacity did not absorb carry")
        result = ForestState(
            slots=torch.stack(slot_slices, dim=2),
            occupied=torch.stack(occupied_slices, dim=2),
            counts=counts,
            valid_steps=valid_steps,
        )
        return ForestRun(state=result, merge_count=merge_count)

    def reduce_streaming(
        self,
        events: Tensor,
        routes: Tensor,
        valid_mask: Tensor,
        *,
        initial_state: ForestState | None = None,
    ) -> ForestRun:
        if events.ndim != 3 or events.shape[2] != self.d_model:
            raise ValueError("events must have shape [N,T,D]")
        n, t, _ = events.shape
        if n <= 0:
            raise ValueError("events must have a positive batch dimension")
        if routes.shape != (n, t) or valid_mask.shape != (n, t):
            raise ValueError("routes and valid_mask must have shape [N,T]")
        if not events.is_floating_point():
            raise TypeError("events must have a floating-point dtype")
        if events.device != routes.device or events.device != valid_mask.device:
            raise ValueError("events, routes, and valid_mask must share a device")
        validate_routes(routes, valid_mask, self.branches)
        state = initial_state or self.initial_state(
            n, device=events.device, dtype=events.dtype
        )
        _validate_state(
            state, batch_size=n, paths=self.paths, d_model=self.d_model
        )
        if state.slots.device != events.device or state.slots.dtype != events.dtype:
            raise ValueError("initial_state must match events device and dtype")
        total = torch.zeros((), device=events.device, dtype=torch.int64)
        for index in range(t):
            result = self.step(
                state, events[:, index], routes[:, index], valid_mask[:, index]
            )
            state = result.state
            total = total + result.merge_count
        return ForestRun(state=state, merge_count=total)

    def reduce_parallel(
        self,
        events: Tensor,
        routes: Tensor,
        valid_mask: Tensor,
    ) -> ForestRun:
        """Reduce packed lane subsequences through the canonical dyadic DAG.

        Unmatched nodes are not promoted.  Slot ``s`` is extracted from level
        ``s`` at index ``(count >> s) - 1`` when bit ``s`` is active, exactly
        matching a binary-counter carry forest without requiring associativity.
        """

        if events.ndim != 3 or events.shape[2] != self.d_model:
            raise ValueError("events must have shape [N,T,D]")
        n, t, _ = events.shape
        if n <= 0:
            raise ValueError("events must have a positive batch dimension")
        if routes.shape != (n, t) or valid_mask.shape != (n, t):
            raise ValueError("routes and valid_mask must have shape [N,T]")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean")
        if not events.is_floating_point():
            raise TypeError("events must have a floating-point dtype")
        if events.device != routes.device or events.device != valid_mask.device:
            raise ValueError("events, routes, and valid_mask must share a device")
        validate_routes(routes, valid_mask, self.branches)
        counts = route_counts(routes, valid_mask, self.branches).to(events.device)
        scales = _required_scales(counts)
        batch_slots: list[Tensor] = []
        batch_occupied: list[Tensor] = []
        merge_count = torch.zeros((), device=events.device, dtype=torch.int64)

        for batch_index in range(n):
            path_slots: list[Tensor] = []
            path_occupied: list[Tensor] = []
            for path in range(self.paths):
                selected = valid_mask[batch_index] & (routes[batch_index] == path)
                sequence = events[batch_index, selected]
                count = int(sequence.shape[0])
                levels: list[Tensor] = [sequence]
                level = sequence
                scale = 0
                while level.shape[0] >= 2:
                    pairs = int(level.shape[0] // 2)
                    left = level[: 2 * pairs : 2]
                    right = level[1 : 2 * pairs : 2]
                    level = self.merge(
                        left,
                        right,
                        scale=scale,
                        global_path=(path == self.branches),
                    )
                    levels.append(level)
                    merge_count = merge_count + pairs
                    scale += 1

                scale_values: list[Tensor] = []
                scale_flags: list[Tensor] = []
                for scale in range(scales):
                    active = bool((count >> scale) & 1)
                    if active:
                        node_index = (count >> scale) - 1
                        scale_values.append(levels[scale][node_index])
                    else:
                        scale_values.append(events.new_zeros(self.d_model))
                    scale_flags.append(
                        torch.tensor(active, device=events.device, dtype=torch.bool)
                    )
                path_slots.append(torch.stack(scale_values, dim=0))
                path_occupied.append(torch.stack(scale_flags, dim=0))
            batch_slots.append(torch.stack(path_slots, dim=0))
            batch_occupied.append(torch.stack(path_occupied, dim=0))

        state = ForestState(
            slots=torch.stack(batch_slots, dim=0),
            occupied=torch.stack(batch_occupied, dim=0),
            counts=counts,
            valid_steps=valid_mask.sum(dim=1, dtype=torch.int64),
        )
        return ForestRun(state=state, merge_count=merge_count)

    def reduce_parallel_prefixes(
        self,
        events: Tensor,
        routes: Tensor,
        valid_mask: Tensor,
    ) -> ForestPrefixRun:
        """Build every causal prefix state from one packed dyadic reduction.

        Each routed lane's merge pyramid is constructed exactly once.  Prefix
        states gather the binary-counter blocks selected by their prefix
        counts, avoiding an accidental quadratic sequence of full reductions.
        The work is ``O(N*T*P*log(T))`` for state materialization and only
        ``O(N*T)`` learned merge applications; no token history is exposed to
        the readout.
        """

        if events.ndim != 3 or events.shape[2] != self.d_model:
            raise ValueError("events must have shape [N,T,D]")
        n, t, _ = events.shape
        if n <= 0:
            raise ValueError("events must have a positive batch dimension")
        if routes.shape != (n, t) or valid_mask.shape != (n, t):
            raise ValueError("routes and valid_mask must have shape [N,T]")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean")
        if not events.is_floating_point():
            raise TypeError("events must have a floating-point dtype")
        if events.device != routes.device or events.device != valid_mask.device:
            raise ValueError("events, routes, and valid_mask must share a device")
        validate_routes(routes, valid_mask, self.branches)
        if t == 0:
            return ForestPrefixRun(
                states=(),
                merge_count=torch.zeros((), device=events.device, dtype=torch.int64),
            )

        lane_updates = routed_lane_mask(routes, valid_mask, self.branches)
        prefix_counts = lane_updates.cumsum(dim=1, dtype=torch.int64)
        final_counts = prefix_counts[:, -1]
        scales = _required_scales(final_counts)
        pyramids: list[list[list[Tensor]]] = []
        merge_count = torch.zeros((), device=events.device, dtype=torch.int64)

        for batch_index in range(n):
            batch_pyramids: list[list[Tensor]] = []
            for path in range(self.paths):
                selected = lane_updates[batch_index, :, path]
                level = events[batch_index, selected]
                levels = [level]
                scale = 0
                while level.shape[0] >= 2:
                    pairs = int(level.shape[0] // 2)
                    level = self.merge(
                        level[: 2 * pairs : 2],
                        level[1 : 2 * pairs : 2],
                        scale=scale,
                        global_path=(path == self.branches),
                    )
                    levels.append(level)
                    merge_count = merge_count + pairs
                    scale += 1
                batch_pyramids.append(levels)
            pyramids.append(batch_pyramids)

        prefix_valid_steps = valid_mask.cumsum(dim=1, dtype=torch.int64)
        states: list[ForestState] = []
        for time_index in range(t):
            batch_slots: list[Tensor] = []
            batch_occupied: list[Tensor] = []
            for batch_index in range(n):
                path_slots: list[Tensor] = []
                path_occupied: list[Tensor] = []
                for path in range(self.paths):
                    count = int(prefix_counts[batch_index, time_index, path])
                    scale_values: list[Tensor] = []
                    scale_flags: list[Tensor] = []
                    for scale in range(scales):
                        active = bool((count >> scale) & 1)
                        if active:
                            node_index = (count >> scale) - 1
                            scale_values.append(
                                pyramids[batch_index][path][scale][node_index]
                            )
                        else:
                            scale_values.append(events.new_zeros(self.d_model))
                        scale_flags.append(
                            torch.tensor(
                                active, device=events.device, dtype=torch.bool
                            )
                        )
                    path_slots.append(torch.stack(scale_values, dim=0))
                    path_occupied.append(torch.stack(scale_flags, dim=0))
                batch_slots.append(torch.stack(path_slots, dim=0))
                batch_occupied.append(torch.stack(path_occupied, dim=0))
            states.append(
                ForestState(
                    slots=torch.stack(batch_slots, dim=0),
                    occupied=torch.stack(batch_occupied, dim=0),
                    counts=prefix_counts[:, time_index],
                    valid_steps=prefix_valid_steps[:, time_index],
                )
            )
        return ForestPrefixRun(states=tuple(states), merge_count=merge_count)

    def structural_metrics(
        self,
        state: ForestState,
        *,
        merge_count: int | Tensor = 0,
    ) -> dict[str, int]:
        """Return length-transparent structural and operation diagnostics."""

        _validate_state(
            state,
            batch_size=state.batch_size,
            paths=self.paths,
            d_model=self.d_model,
        )
        merges = int(merge_count.item()) if isinstance(merge_count, Tensor) else int(merge_count)
        if merges < 0:
            raise ValueError("merge_count cannot be negative")
        merge_metrics = self.merge.structural_metrics(merges) if hasattr(
            self.merge, "structural_metrics"
        ) else {
            "nominal_rank": self.cp_rank,
            "effective_rank": self.cp_rank,
            "exported_rank": self.cp_rank,
            "merge_parameter_count": sum(p.numel() for p in self.merge.parameters()),
            "operation_count_proxy": merges,
        }
        return {
            **merge_metrics,
            "forest_parameter_count": sum(p.numel() for p in self.parameters()),
            "executed_merge_count": merges,
            "active_slots": int(state.occupied.sum().item()),
            "allocated_slots": state.occupied.numel(),
            "active_state_elements": int(state.occupied.sum().item()) * self.d_model,
            "allocated_state_elements": state.slots.numel(),
        }

    def state_for_batch(self, state: ForestState, index: int) -> ForestState:
        """Extract one sample and trim capacity to its logical highest scale."""

        _validate_state(
            state,
            batch_size=state.batch_size,
            paths=self.paths,
            d_model=self.d_model,
        )
        if not -state.batch_size <= index < state.batch_size:
            raise IndexError("batch index is out of range")
        index %= state.batch_size
        scales = _required_scales(state.counts[index : index + 1])
        return ForestState(
            slots=state.slots[index : index + 1, :, :scales],
            occupied=state.occupied[index : index + 1, :, :scales],
            counts=state.counts[index : index + 1],
            valid_steps=state.valid_steps[index : index + 1],
        )


class ForestReadout(nn.Module):
    """Query-aware, local-branch-permutation-invariant active-slot readout."""

    def __init__(
        self,
        *,
        d_model: int,
        branches: int,
        output_size: int,
        scale_feature_dim: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.branches = branches
        self.paths = branches + 1
        self.scale_feature_dim = scale_feature_dim
        self.scale_projection = nn.Linear(scale_feature_dim, d_model, bias=False)
        self.global_type = nn.Parameter(torch.zeros(d_model))
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, output_size)

    def forward(self, state: ForestState, query: Tensor) -> Tensor:
        if query.shape != (state.batch_size, self.d_model):
            raise ValueError("query must have shape [N,D]")
        scales = torch.arange(
            state.scales, device=query.device, dtype=query.dtype
        )
        features = analytic_scale_features(scales, self.scale_feature_dim)
        scale_signal = self.scale_projection(features).view(
            1, 1, state.scales, self.d_model
        )
        global_mask = torch.zeros(
            1, self.paths, 1, 1, device=query.device, dtype=query.dtype
        )
        global_mask[:, self.branches] = 1
        typed_slots = state.slots + scale_signal + global_mask * self.global_type.view(
            1, 1, 1, -1
        )
        keys = self.key_projection(typed_slots)
        values = self.value_projection(state.slots)
        projected_query = self.query_projection(query)
        scores = torch.einsum("nd,npsd->nps", projected_query, keys)
        scores = scores / math.sqrt(self.d_model)
        floor = -torch.finfo(scores.dtype).max
        scores = scores.masked_fill(~state.occupied, floor)
        weights = torch.softmax(scores.flatten(1), dim=-1).view_as(scores)
        weights = weights * state.occupied.to(weights.dtype)
        weights = weights / weights.sum(dim=(1, 2), keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        context = torch.einsum("nps,npsd->nd", weights, values)
        return self.output(self.output_norm(query + context))


class RoutedTensorLanguageModel(nn.Module):
    """Minimal Milestone-1 model around the routed tensor-network forest.

    Routes are supplied externally in this milestone.  The constrained causal
    router and dynamic binding benchmark are introduced in Milestone 2.
    """

    def __init__(self, config: ForestConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.event_norm = nn.LayerNorm(config.d_model)
        self.forest = ScaleSharedBinaryForest(
            d_model=config.d_model,
            branches=config.branches,
            cp_rank=config.cp_rank,
            scale_feature_dim=config.scale_feature_dim,
        )
        self.readout = ForestReadout(
            d_model=config.d_model,
            branches=config.branches,
            output_size=config.resolved_output_size,
            scale_feature_dim=config.scale_feature_dim,
        )

    def forward(
        self,
        token_ids: Tensor,
        routes: Tensor,
        valid_mask: Tensor,
        *,
        implementation: Literal["streaming", "parallel"] = "streaming",
        initial_state: ForestState | None = None,
    ) -> ForestModelOutput:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [N,T]")
        if routes.shape != token_ids.shape or valid_mask.shape != token_ids.shape:
            raise ValueError("routes and valid_mask must match token_ids")
        if implementation not in ("streaming", "parallel"):
            raise ValueError("implementation must be 'streaming' or 'parallel'")
        if implementation == "parallel" and initial_state is not None:
            raise ValueError("parallel reduction currently requires an empty state")
        validate_routes(routes, valid_mask, self.config.branches)

        safe_token_ids = torch.where(
            valid_mask,
            token_ids,
            torch.full_like(token_ids, self.config.pad_token_id),
        )
        queries = self.token_embedding(safe_token_ids)
        events = self.event_norm(queries)
        n, t = token_ids.shape
        logits: list[Tensor] = []
        total = torch.zeros((), device=token_ids.device, dtype=torch.int64)

        if implementation == "streaming":
            state = initial_state or self.forest.initial_state(
                n, device=events.device, dtype=events.dtype
            )
            for index in range(t):
                result = self.forest.step(
                    state, events[:, index], routes[:, index], valid_mask[:, index]
                )
                state = result.state
                total = total + result.merge_count
                current = self.readout(state, queries[:, index])
                logits.append(
                    torch.where(valid_mask[:, index, None], current, torch.zeros_like(current))
                )
        else:
            prefix_run = self.forest.reduce_parallel_prefixes(
                events, routes, valid_mask
            )
            state = self.forest.initial_state(
                n, device=events.device, dtype=events.dtype
            )
            for index, prefix_state in enumerate(prefix_run.states):
                state = prefix_state
                current = self.readout(state, queries[:, index])
                logits.append(
                    torch.where(valid_mask[:, index, None], current, torch.zeros_like(current))
                )
            total = prefix_run.merge_count

        if logits:
            stacked = torch.stack(logits, dim=1)
        else:
            stacked = events.new_zeros(n, 0, self.config.resolved_output_size)
        return ForestModelOutput(logits=stacked, state=state, merge_count=total)
