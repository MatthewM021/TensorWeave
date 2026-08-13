"""Route validation and lane accounting for the V3 streaming model.

Route semantics are intentionally distinct from padding semantics:

* ``NULL_ROUTE`` (``-1``) at a valid position is a read-only/query event. It
  advances the valid-event clock but updates no routed lane.
* Routes ``0`` through ``branches - 1`` select local branch lanes.
* Route ``branches`` selects the dedicated global lane.
* A position whose ``valid_mask`` is false is padding and is a total no-op,
  including no clock advance. Its route value is ignored and may be garbage.

The helpers in this module only validate and account for routed lanes. The
caller owns the valid-event clock, including advancement for valid null events.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import operator
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


NULL_ROUTE = -1

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _validate_branches(branches: int) -> int:
    if isinstance(branches, bool):
        raise TypeError("branches must be a positive integer")
    try:
        value = operator.index(branches)
    except TypeError as exc:
        raise TypeError("branches must be a positive integer") from exc
    if value < 1:
        raise ValueError("branches must be positive")
    return value


def validate_routes(
    routes: torch.Tensor,
    valid_mask: torch.Tensor,
    branches: int,
) -> None:
    """Validate a batch of route IDs without inspecting padded route values.

    Args:
        routes: Integer tensor shaped ``[batch, time]``.
        valid_mask: Boolean tensor with the same shape as ``routes``.
        branches: Number of local branches. The route equal to this value is
            reserved for the dedicated global lane.

    Raises:
        TypeError: If inputs have unsupported types or dtypes.
        ValueError: If shapes, ranks, devices, branch count, or valid route IDs
            violate the routing contract.
    """

    branch_count = _validate_branches(branches)
    if not isinstance(routes, torch.Tensor):
        raise TypeError("routes must be a torch.Tensor")
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be a torch.Tensor")
    if routes.ndim != 2:
        raise ValueError("routes and valid_mask must have shape [batch, time]")
    if routes.shape != valid_mask.shape:
        raise ValueError("routes and valid_mask must have the same shape")
    if routes.device != valid_mask.device:
        raise ValueError("routes and valid_mask must be on the same device")
    if routes.dtype not in _INTEGER_DTYPES:
        raise TypeError("routes must use an integer dtype")
    if valid_mask.dtype is not torch.bool:
        raise TypeError("valid_mask must use torch.bool")

    valid_routes = routes[valid_mask]
    if valid_routes.numel() == 0:
        return
    in_range = (valid_routes >= NULL_ROUTE) & (valid_routes <= branch_count)
    if not bool(in_range.all()):
        invalid = valid_routes[~in_range]
        sample = invalid[:8].detach().cpu().tolist()
        raise ValueError(
            f"valid route IDs must be in [{NULL_ROUTE}, {branch_count}]; "
            f"found {sample}"
        )


def routed_lane_mask(
    routes: torch.Tensor,
    valid_mask: torch.Tensor,
    branches: int,
) -> torch.Tensor:
    """Return a boolean ``[batch, time, branches + 1]`` routed-lane mask.

    Valid null events and padded positions produce all-false rows. The final
    lane is the dedicated global lane.
    """

    validate_routes(routes, valid_mask, branches)
    branch_count = _validate_branches(branches)
    lane_ids = torch.arange(
        branch_count + 1,
        dtype=routes.dtype,
        device=routes.device,
    )
    return (routes.unsqueeze(-1) == lane_ids) & valid_mask.unsqueeze(-1)


def route_counts(
    routes: torch.Tensor,
    valid_mask: torch.Tensor,
    branches: int,
) -> torch.Tensor:
    """Count routed updates per sample, including the dedicated global lane."""

    return routed_lane_mask(routes, valid_mask, branches).sum(
        dim=1,
        dtype=torch.int64,
    )


class RoutingMode(str, Enum):
    """Scientifically distinct routing conditions."""

    ORACLE = "oracle"
    CURRICULUM = "curriculum"
    LATENT = "latent"


@dataclass(frozen=True)
class CurriculumSchedule:
    """A declared linear teacher-guidance schedule with recorded endpoints."""

    start_step: int
    end_step: int
    start_probability: float = 1.0
    end_probability: float = 0.0

    def __post_init__(self) -> None:
        start = _nonnegative_index(self.start_step, "start_step")
        end = _nonnegative_index(self.end_step, "end_step")
        if end <= start:
            raise ValueError("end_step must be greater than start_step")
        for name in ("start_probability", "end_probability"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"{name} must be a finite real number")
            if not math.isfinite(float(raw)):
                raise ValueError(f"{name} must be finite")
        start_probability = float(self.start_probability)
        end_probability = float(self.end_probability)
        if not 0.0 <= start_probability <= 1.0:
            raise ValueError("start_probability must lie in [0,1]")
        if not 0.0 <= end_probability <= 1.0:
            raise ValueError("end_probability must lie in [0,1]")
        if end_probability > start_probability:
            raise ValueError("curriculum guidance probability must not increase")
        object.__setattr__(self, "start_step", start)
        object.__setattr__(self, "end_step", end)
        object.__setattr__(self, "start_probability", start_probability)
        object.__setattr__(self, "end_probability", end_probability)

    def probability(self, step: int) -> float:
        step = _nonnegative_index(step, "step")
        if step <= self.start_step:
            return self.start_probability
        if step >= self.end_step:
            return self.end_probability
        fraction = (step - self.start_step) / (self.end_step - self.start_step)
        return self.start_probability + fraction * (
            self.end_probability - self.start_probability
        )

    def guidance_probability(self, step: int) -> float:
        """Alias that makes the scheduled quantity explicit at call sites."""

        return self.probability(step)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "start_step": self.start_step,
            "end_step": self.end_step,
            "start_probability": self.start_probability,
            "end_probability": self.end_probability,
        }

    @property
    def endpoints(self) -> dict[str, int | float]:
        return self.as_dict()


def _nonnegative_index(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a nonnegative integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a nonnegative integer") from error
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


_GUIDANCE_MODULUS = 2_147_483_647
_GUIDANCE_COEFFICIENTS = (73_856_093, 19_349_663, 83_492_791, 26_544_357)


def deterministic_guidance_mask(
    valid_mask: Tensor,
    probability: float,
    *,
    training_step: int,
    seed: int = 0,
    position_offsets: Tensor | None = None,
) -> Tensor:
    """Return a reproducible teacher-guidance mask independent of labels/data.

    The integer mixer is keyed only by the declared seed, training step, batch
    index, and time index.  It does not consume global RNG state and therefore
    reproduces after checkpoint resume on either CPU or accelerator devices.
    """

    if not isinstance(valid_mask, Tensor) or valid_mask.ndim != 2:
        raise ValueError("valid_mask must be a tensor shaped [N,T]")
    if valid_mask.dtype is not torch.bool:
        raise TypeError("valid_mask must use torch.bool")
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    step = _nonnegative_index(training_step, "training_step")
    seed = operator.index(seed)
    if probability == 0.0:
        return torch.zeros_like(valid_mask)
    if probability == 1.0:
        return valid_mask.clone()

    n, t = valid_mask.shape
    batch = torch.arange(n, device=valid_mask.device, dtype=torch.int64).unsqueeze(1)
    if position_offsets is None:
        offsets = torch.zeros(n, device=valid_mask.device, dtype=torch.int64)
    else:
        if (
            not isinstance(position_offsets, Tensor)
            or position_offsets.shape != (n,)
            or position_offsets.dtype != torch.int64
            or position_offsets.device != valid_mask.device
        ):
            raise ValueError("position_offsets must be int64 with shape [N]")
        if bool((position_offsets < 0).any()):
            raise ValueError("position_offsets must be nonnegative")
        offsets = position_offsets
    # Hash valid-event ordinals rather than padded tensor positions so guidance
    # is invariant to chunking and interspersed padding.
    ordinal = valid_mask.to(torch.int64).cumsum(dim=1) - 1
    time = offsets.unsqueeze(1) + ordinal.clamp_min(0)
    key = torch.zeros((n, t), device=valid_mask.device, dtype=torch.int64)
    for value, coefficient in zip(
        (batch, time, step % _GUIDANCE_MODULUS, seed % _GUIDANCE_MODULUS),
        _GUIDANCE_COEFFICIENTS,
    ):
        key = (key + value * coefficient) % _GUIDANCE_MODULUS
    # Two bounded nonlinear avalanche rounds avoid the long contiguous runs
    # produced by an affine modular threshold while staying bit-reproducible
    # across CPU and accelerator int64 kernels (all products fit int64).
    key = (key * key + 1_103_515_245 * key + 12_345) % _GUIDANCE_MODULUS
    key = key ^ (key >> 16)
    key = (key * key + 1_664_525 * key + 1_013_904_223) % _GUIDANCE_MODULUS
    threshold = int(probability * _GUIDANCE_MODULUS)
    return (key < threshold) & valid_mask


def _validate_permutation(permutation: Tensor | list[int] | tuple[int, ...], branches: int, device: torch.device) -> Tensor:
    permutation = torch.as_tensor(permutation, device=device)
    if permutation.ndim != 1 or permutation.numel() != branches:
        raise ValueError("branch permutation must have shape [B]")
    if permutation.dtype not in _INTEGER_DTYPES:
        raise TypeError("branch permutation must use an integer dtype")
    permutation = permutation.to(torch.int64)
    expected = torch.arange(branches, device=device, dtype=torch.int64)
    if not torch.equal(torch.sort(permutation).values, expected):
        raise ValueError("branch permutation must contain each local branch exactly once")
    return permutation


@dataclass
class PersistentRouterState:
    """Prefix-only state for the persistent causal router.

    ``prototypes``, ``occupied``, ``ages``, and ``loads`` have a local branch
    axis.  ``global_state`` is a bounded exponential summary of non-null routed
    events; valid null events remain read-only but still advance time/ages.
    """

    prototypes: Tensor
    occupied: Tensor
    ages: Tensor
    loads: Tensor
    global_state: Tensor
    global_occupied: Tensor
    global_load: Tensor
    valid_steps: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.prototypes.shape[0])

    @property
    def branches(self) -> int:
        return int(self.prototypes.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.prototypes.shape[2])

    def detach(self) -> "PersistentRouterState":
        return PersistentRouterState(
            prototypes=self.prototypes.detach(),
            occupied=self.occupied.detach(),
            ages=self.ages.detach(),
            loads=self.loads.detach(),
            global_state=self.global_state.detach(),
            global_occupied=self.global_occupied.detach(),
            global_load=self.global_load.detach(),
            valid_steps=self.valid_steps.detach(),
        )

    def to(self, *args: Any, **kwargs: Any) -> "PersistentRouterState":
        prototypes = self.prototypes.to(*args, **kwargs)
        device = prototypes.device
        return PersistentRouterState(
            prototypes=prototypes,
            occupied=self.occupied.to(device=device),
            ages=self.ages.to(device=device),
            loads=self.loads.to(device=device),
            global_state=self.global_state.to(device=device, dtype=prototypes.dtype),
            global_occupied=self.global_occupied.to(device=device),
            global_load=self.global_load.to(device=device),
            valid_steps=self.valid_steps.to(device=device),
        )

    def permute_branches(
        self, permutation: Tensor | list[int] | tuple[int, ...]
    ) -> "PersistentRouterState":
        """Return ``new[:,j] = old[:,permutation[j]]`` for local state only."""

        permutation = _validate_permutation(
            permutation, self.branches, self.prototypes.device
        )
        return PersistentRouterState(
            prototypes=self.prototypes.index_select(1, permutation),
            occupied=self.occupied.index_select(1, permutation),
            ages=self.ages.index_select(1, permutation),
            loads=self.loads.index_select(1, permutation),
            global_state=self.global_state,
            global_occupied=self.global_occupied,
            global_load=self.global_load,
            valid_steps=self.valid_steps,
        )


@dataclass
class PersistentRouterOutput:
    """Complete routing trace and causal state after the final input event."""

    logits: Tensor
    probabilities: Tensor
    routes: Tensor
    final_state: PersistentRouterState
    diagnostics: dict[str, Tensor]

    @property
    def state(self) -> PersistentRouterState:
        """Convenience alias for stateful streaming call sites."""

        return self.final_state


def permute_local_routes(
    routes: Tensor,
    permutation: Tensor | list[int] | tuple[int, ...],
    branches: int,
) -> Tensor:
    """Map route IDs to match ``PersistentRouterState.permute_branches``.

    With ``new[j] = old[permutation[j]]``, an old local route ``r`` becomes
    ``inverse_permutation[r]``.  Global, null, and padding sentinel values are
    returned unchanged.
    """

    branches = _validate_branches(branches)
    permutation = _validate_permutation(permutation, branches, routes.device)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(branches, device=routes.device)
    local = (routes >= 0) & (routes < branches)
    safe = routes.to(torch.int64).clamp(0, branches - 1)
    mapped = inverse[safe]
    return torch.where(local, mapped.to(routes.dtype), routes)


def _local_loads_fit_global_load(loads: Tensor, global_load: Tensor) -> bool:
    """Compare row totals using Python integers so int64 sums cannot wrap."""

    rows = loads.detach().cpu().tolist()
    totals = global_load.detach().cpu().tolist()
    return all(
        sum(int(value) for value in row) <= int(total)
        for row, total in zip(rows, totals, strict=True)
    )


class PersistentCausalRouter(nn.Module):
    """O(B) persistent router over prefix-only branch summaries.

    Probability/logit classes are ordered as local branches ``0..B-1``, then
    the optional global route ``B``, then the optional null route ``-1``.
    Every local branch uses exactly the same scorer: there are no learned
    branch IDs, token lookup tables, maximum-length parameters, or stored raw
    feature histories.
    """

    def __init__(
        self,
        feature_dim: int,
        branches: int,
        hidden_dim: int | None = None,
        *,
        mode: RoutingMode | str = RoutingMode.LATENT,
        include_global: bool = True,
        include_null: bool = True,
        curriculum_schedule: CurriculumSchedule | None = None,
        curriculum_seed: int = 0,
        prototype_update_rate: float = 0.25,
        global_update_rate: float = 0.10,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_dim = _validate_branches(feature_dim)
        self.branches = _validate_branches(branches)
        self.hidden_dim = self.feature_dim if hidden_dim is None else _validate_branches(hidden_dim)
        self.mode = RoutingMode(mode)
        if not isinstance(include_global, bool) or not isinstance(include_null, bool):
            raise TypeError("include_global and include_null must be booleans")
        self.include_global = include_global
        self.include_null = include_null
        if self.mode is RoutingMode.CURRICULUM:
            if not isinstance(curriculum_schedule, CurriculumSchedule):
                raise ValueError("curriculum mode requires an explicit CurriculumSchedule")
        elif curriculum_schedule is not None:
            raise ValueError("a curriculum schedule is valid only in curriculum mode")
        self.curriculum_schedule = curriculum_schedule
        self.curriculum_seed = operator.index(curriculum_seed)
        self.prototype_update_rate = _unit_interval(
            prototype_update_rate, "prototype_update_rate", positive=True
        )
        self.global_update_rate = _unit_interval(
            global_update_rate, "global_update_rate", positive=True
        )
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or not self.temperature > 0:
            raise ValueError("temperature must be finite and positive")

        h = self.hidden_dim
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.feature_projection = nn.Linear(self.feature_dim, h)
        self.prototype_projection = nn.Linear(self.feature_dim, h, bias=False)
        self.global_projection = nn.Linear(self.feature_dim, h, bias=False)
        self.branch_scorer = nn.Sequential(
            nn.Linear(4 * h + 3, h), nn.GELU(), nn.Linear(h, 1)
        )
        self.global_scorer = (
            nn.Sequential(nn.Linear(3 * h + 2, h), nn.GELU(), nn.Linear(h, 1))
            if include_global
            else None
        )
        self.null_scorer = (
            nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, 1))
            if include_null
            else None
        )

    @property
    def class_count(self) -> int:
        return self.branches + int(self.include_global) + int(self.include_null)

    @property
    def class_route_ids(self) -> tuple[int, ...]:
        routes = list(range(self.branches))
        if self.include_global:
            routes.append(self.branches)
        if self.include_null:
            routes.append(NULL_ROUTE)
        return tuple(routes)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> PersistentRouterState:
        batch_size = _validate_branches(batch_size)
        reference = next(self.parameters())
        device = reference.device if device is None else torch.device(device)
        dtype = reference.dtype if dtype is None else dtype
        if not dtype.is_floating_point:
            raise TypeError("router state requires a floating dtype")
        return PersistentRouterState(
            prototypes=torch.zeros(
                batch_size, self.branches, self.feature_dim, device=device, dtype=dtype
            ),
            occupied=torch.zeros(
                batch_size, self.branches, device=device, dtype=torch.bool
            ),
            ages=torch.zeros(
                batch_size, self.branches, device=device, dtype=torch.int64
            ),
            loads=torch.zeros(
                batch_size, self.branches, device=device, dtype=torch.int64
            ),
            global_state=torch.zeros(
                batch_size, self.feature_dim, device=device, dtype=dtype
            ),
            global_occupied=torch.zeros(batch_size, device=device, dtype=torch.bool),
            global_load=torch.zeros(batch_size, device=device, dtype=torch.int64),
            valid_steps=torch.zeros(batch_size, device=device, dtype=torch.int64),
        )

    def permute_state(
        self,
        state: PersistentRouterState,
        permutation: Tensor | list[int] | tuple[int, ...],
    ) -> PersistentRouterState:
        self._validate_state(state, state.batch_size, state.prototypes.device, state.prototypes.dtype)
        return state.permute_branches(permutation)

    def _validate_state(
        self,
        state: PersistentRouterState,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not isinstance(state, PersistentRouterState):
            raise TypeError("initial_state must be a PersistentRouterState")
        n, b, d = state.prototypes.shape
        if (n, b, d) != (batch_size, self.branches, self.feature_dim):
            raise ValueError("router state dimensions do not match the router")
        expected_branch = (batch_size, self.branches)
        if state.occupied.shape != expected_branch or state.occupied.dtype is not torch.bool:
            raise ValueError("state.occupied must be bool [N,B]")
        if state.ages.shape != expected_branch or state.ages.dtype is not torch.int64:
            raise ValueError("state.ages must be int64 [N,B]")
        if state.loads.shape != expected_branch or state.loads.dtype is not torch.int64:
            raise ValueError("state.loads must be int64 [N,B]")
        if state.global_state.shape != (batch_size, self.feature_dim):
            raise ValueError("state.global_state must have shape [N,D]")
        if state.global_occupied.shape != (batch_size,) or state.global_occupied.dtype is not torch.bool:
            raise ValueError("state.global_occupied must be bool [N]")
        if state.global_load.shape != (batch_size,) or state.global_load.dtype is not torch.int64:
            raise ValueError("state.global_load must be int64 [N]")
        if state.valid_steps.shape != (batch_size,) or state.valid_steps.dtype is not torch.int64:
            raise ValueError("state.valid_steps must be int64 [N]")
        tensors = (
            state.prototypes,
            state.occupied,
            state.ages,
            state.loads,
            state.global_state,
            state.global_occupied,
            state.global_load,
            state.valid_steps,
        )
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("all router state tensors must share the input device")
        if state.prototypes.dtype != dtype or state.global_state.dtype != dtype:
            raise ValueError("router prototype/global dtypes must match route_features")
        if bool((state.ages < 0).any()) or bool((state.loads < 0).any()):
            raise ValueError("router ages and loads must be nonnegative")
        if bool((state.global_load < 0).any()) or bool((state.valid_steps < 0).any()):
            raise ValueError("router counters must be nonnegative")
        if not bool(torch.isfinite(state.prototypes).all()) or not bool(
            torch.isfinite(state.global_state).all()
        ):
            raise ValueError("router prototype and global states must be finite")
        if not torch.equal(state.occupied, state.loads > 0):
            raise ValueError("occupied branches must be exactly those with positive load")
        if not torch.equal(state.global_occupied, state.global_load > 0):
            raise ValueError("global occupancy must equal positive global load")
        if not _local_loads_fit_global_load(state.loads, state.global_load):
            raise ValueError("local loads cannot exceed the routed global load")
        if bool((state.global_load > state.valid_steps).any()):
            raise ValueError("routed global load cannot exceed valid steps")
        if bool(
            (
                state.occupied
                & (state.ages >= state.valid_steps.unsqueeze(-1))
            ).any()
        ):
            raise ValueError("occupied branch ages must be below valid steps")
        if bool((state.prototypes.masked_select(~state.occupied.unsqueeze(-1)) != 0).any()):
            raise ValueError("unoccupied branch prototypes must be zero")
        if bool((state.ages.masked_select(~state.occupied) != 0).any()):
            raise ValueError("unoccupied branch ages must be zero")
        if bool((state.global_state.masked_select(~state.global_occupied.unsqueeze(-1)) != 0).any()):
            raise ValueError("unoccupied global state must be zero")

    def _validate_labels(self, route_labels: Tensor, valid_mask: Tensor) -> None:
        validate_routes(route_labels, valid_mask, self.branches)
        labels = route_labels[valid_mask]
        if not self.include_global and bool((labels == self.branches).any()):
            raise ValueError("global labels are disabled for this router")
        if not self.include_null and bool((labels == NULL_ROUTE).any()):
            raise ValueError("null labels are disabled for this router")

    def _score(self, feature: Tensor, state: PersistentRouterState) -> Tensor:
        q = self.feature_projection(self.feature_norm(feature))
        prototypes = self.prototype_projection(self.feature_norm(state.prototypes))
        global_hidden = self.global_projection(self.feature_norm(state.global_state))
        q_branch = q.unsqueeze(1).expand(-1, self.branches, -1)
        g_branch = global_hidden.unsqueeze(1).expand(-1, self.branches, -1)
        denominator = state.valid_steps.clamp_min(1).to(feature.dtype).unsqueeze(-1)
        age = (state.ages.to(feature.dtype) / denominator).clamp(max=1).unsqueeze(-1)
        load = (state.loads.to(feature.dtype) / denominator).clamp(max=1).unsqueeze(-1)
        occupied = state.occupied.to(feature.dtype).unsqueeze(-1)
        branch_input = torch.cat(
            (q_branch, prototypes, q_branch * prototypes, g_branch, occupied, age, load),
            dim=-1,
        )
        scores = [self.branch_scorer(branch_input).squeeze(-1)]
        common = (q, global_hidden, q * global_hidden)
        if self.global_scorer is not None:
            global_load = (
                state.global_load.to(feature.dtype) / denominator.squeeze(-1)
            ).clamp(max=1)
            global_input = torch.cat(
                common
                + (
                    state.global_occupied.to(feature.dtype).unsqueeze(-1),
                    global_load.unsqueeze(-1),
                ),
                dim=-1,
            )
            scores.append(self.global_scorer(global_input))
        if self.null_scorer is not None:
            scores.append(self.null_scorer(torch.cat(common, dim=-1)))
        return torch.cat(scores, dim=-1) / self.temperature

    def _classes_to_routes(self, classes: Tensor) -> Tensor:
        routes = classes.to(torch.int64)
        offset = self.branches
        if self.include_global:
            routes = torch.where(classes == offset, self.branches, routes)
            offset += 1
        if self.include_null:
            routes = torch.where(classes == offset, NULL_ROUTE, routes)
        return routes

    def _routes_to_classes(self, routes: Tensor) -> Tensor:
        classes = routes.to(torch.int64)
        if self.include_null:
            null_class = self.branches + int(self.include_global)
            classes = torch.where(
                routes == NULL_ROUTE,
                torch.full_like(classes, null_class),
                classes,
            )
        return classes

    def _update_state(
        self,
        state: PersistentRouterState,
        feature: Tensor,
        route: Tensor,
        valid: Tensor,
    ) -> PersistentRouterState:
        local = valid & (route >= 0) & (route < self.branches)
        safe_route = route.clamp(0, self.branches - 1)
        selected = F.one_hot(safe_route, self.branches).bool() & local.unsqueeze(-1)
        maximum = torch.iinfo(torch.int64).max
        routed = valid & (route != NULL_ROUTE)
        ages_to_increment = valid.unsqueeze(-1) & state.occupied & ~selected
        if (
            bool((selected & (state.loads == maximum)).any())
            or bool((ages_to_increment & (state.ages == maximum)).any())
            or bool((routed & (state.global_load == maximum)).any())
            or bool((valid & (state.valid_steps == maximum)).any())
        ):
            raise OverflowError("router state counters would overflow int64")
        candidate = torch.where(
            state.occupied.unsqueeze(-1),
            (1.0 - self.prototype_update_rate) * state.prototypes
            + self.prototype_update_rate * feature.unsqueeze(1),
            feature.unsqueeze(1),
        )
        prototypes = torch.where(selected.unsqueeze(-1), candidate, state.prototypes)
        occupied = state.occupied | selected
        incremented_age = torch.where(
            valid.unsqueeze(-1) & state.occupied, state.ages + 1, state.ages
        )
        ages = torch.where(selected, torch.zeros_like(incremented_age), incremented_age)
        loads = state.loads + selected.to(torch.int64)

        global_candidate = torch.where(
            state.global_occupied.unsqueeze(-1),
            (1.0 - self.global_update_rate) * state.global_state
            + self.global_update_rate * feature,
            feature,
        )
        global_state = torch.where(
            routed.unsqueeze(-1), global_candidate, state.global_state
        )
        global_occupied = state.global_occupied | routed
        global_load = state.global_load + routed.to(torch.int64)
        valid_steps = state.valid_steps + valid.to(torch.int64)
        return PersistentRouterState(
            prototypes=prototypes,
            occupied=occupied,
            ages=ages,
            loads=loads,
            global_state=global_state,
            global_occupied=global_occupied,
            global_load=global_load,
            valid_steps=valid_steps,
        )

    def forward(
        self,
        route_features: Tensor,
        valid_mask: Tensor,
        *,
        route_labels: Tensor | None = None,
        allowed_classes: Tensor | None = None,
        training_step: int = 0,
        initial_state: PersistentRouterState | None = None,
    ) -> PersistentRouterOutput:
        if not isinstance(route_features, Tensor) or route_features.ndim != 3:
            raise ValueError("route_features must have shape [N,T,D]")
        n, t, d = route_features.shape
        if d != self.feature_dim:
            raise ValueError("route_features has the wrong final dimension")
        if not route_features.is_floating_point():
            raise TypeError("route_features must use a floating dtype")
        if not isinstance(valid_mask, Tensor) or valid_mask.shape != (n, t):
            raise ValueError("valid_mask must have shape [N,T]")
        if valid_mask.dtype is not torch.bool:
            raise TypeError("valid_mask must use torch.bool")
        if valid_mask.device != route_features.device:
            raise ValueError("valid_mask and route_features must share a device")
        if not bool(torch.isfinite(route_features[valid_mask]).all()):
            raise ValueError("valid route features must be finite")
        if allowed_classes is not None:
            expected = (n, t, self.class_count)
            if (
                not isinstance(allowed_classes, Tensor)
                or allowed_classes.shape != expected
                or allowed_classes.dtype != torch.bool
            ):
                raise ValueError(
                    f"allowed_classes must be boolean with shape {expected}"
                )
            if allowed_classes.device != route_features.device:
                raise ValueError("allowed_classes must share the input device")
            if bool((~allowed_classes.any(dim=-1) & valid_mask).any()):
                raise ValueError("every valid event must allow at least one route class")
        step = _nonnegative_index(training_step, "training_step")

        if self.mode is RoutingMode.LATENT and route_labels is not None:
            raise ValueError("latent routing must not receive route labels")
        uses_labels = self.mode is RoutingMode.ORACLE or (
            self.mode is RoutingMode.CURRICULUM and self.training
        )
        if uses_labels:
            if route_labels is None:
                raise ValueError(f"{self.mode.value} routing requires route labels")
            if route_labels.shape != (n, t) or route_labels.device != route_features.device:
                raise ValueError("route_labels must have shape [N,T] on the input device")
            self._validate_labels(route_labels, valid_mask)
            if allowed_classes is not None:
                label_classes = self._routes_to_classes(route_labels)
                safe_label_classes = torch.where(
                    valid_mask, label_classes, torch.zeros_like(label_classes)
                )
                label_allowed = allowed_classes.gather(
                    -1, safe_label_classes.unsqueeze(-1)
                ).squeeze(-1)
                if bool((valid_mask & ~label_allowed).any()):
                    raise ValueError("a route label is disallowed by allowed_classes")
        # In curriculum evaluation route_labels is deliberately never read,
        # validated, or copied: evaluation is fully autonomous.

        state = initial_state or self.initial_state(
            n, device=route_features.device, dtype=route_features.dtype
        )
        self._validate_state(state, n, route_features.device, route_features.dtype)

        if self.mode is RoutingMode.ORACLE:
            guidance_probability = 1.0
            guidance_mask = valid_mask.clone()
        elif self.mode is RoutingMode.CURRICULUM and self.training:
            assert self.curriculum_schedule is not None
            guidance_probability = self.curriculum_schedule.probability(step)
            guidance_mask = deterministic_guidance_mask(
                valid_mask,
                guidance_probability,
                training_step=step,
                seed=self.curriculum_seed,
                position_offsets=state.valid_steps,
            )
        else:
            guidance_probability = 0.0
            guidance_mask = torch.zeros_like(valid_mask)

        logits_trace: list[Tensor] = []
        probability_trace: list[Tensor] = []
        route_trace: list[Tensor] = []
        autonomous_trace: list[Tensor] = []
        for time_index in range(t):
            valid = valid_mask[:, time_index]
            feature = torch.where(
                valid.unsqueeze(-1),
                route_features[:, time_index],
                torch.zeros_like(route_features[:, time_index]),
            )
            logits = self._score(feature, state)
            if allowed_classes is not None:
                allowed = allowed_classes[:, time_index].clone()
                # Padding is a total no-op and may have no allowed classes.
                # Give its softmax a harmless finite support point; all padded
                # outputs and state effects are masked immediately afterwards.
                allowed[:, 0] |= ~valid_mask[:, time_index]
                logits = logits.masked_fill(
                    ~allowed,
                    -torch.inf,
                )
            probabilities = torch.softmax(logits, dim=-1)
            autonomous = self._classes_to_routes(probabilities.argmax(dim=-1))
            if self.mode is RoutingMode.ORACLE:
                assert route_labels is not None
                chosen = route_labels[:, time_index].to(torch.int64)
            elif self.mode is RoutingMode.CURRICULUM and self.training:
                assert route_labels is not None
                chosen = torch.where(
                    guidance_mask[:, time_index],
                    route_labels[:, time_index].to(torch.int64),
                    autonomous,
                )
            else:
                chosen = autonomous
            chosen = torch.where(valid, chosen, torch.full_like(chosen, NULL_ROUTE))
            state = self._update_state(
                state, feature, chosen, valid
            )
            logits_trace.append(
                torch.where(valid.unsqueeze(-1), logits, torch.zeros_like(logits))
            )
            probability_trace.append(
                torch.where(
                    valid.unsqueeze(-1), probabilities, torch.zeros_like(probabilities)
                )
            )
            route_trace.append(chosen)
            autonomous_trace.append(
                torch.where(valid, autonomous, torch.full_like(autonomous, NULL_ROUTE))
            )

        if t:
            logits_out = torch.stack(logits_trace, dim=1)
            probabilities_out = torch.stack(probability_trace, dim=1)
            routes_out = torch.stack(route_trace, dim=1)
            autonomous_out = torch.stack(autonomous_trace, dim=1)
        else:
            logits_out = route_features.new_zeros(n, 0, self.class_count)
            probabilities_out = route_features.new_zeros(n, 0, self.class_count)
            routes_out = torch.empty(n, 0, device=route_features.device, dtype=torch.int64)
            autonomous_out = routes_out.clone()

        valid_count = valid_mask.sum(dtype=torch.int64)
        guided_count = guidance_mask.sum(dtype=torch.int64)
        entropy_terms = -(
            probabilities_out.clamp_min(torch.finfo(probabilities_out.dtype).tiny).log()
            * probabilities_out
        ).sum(dim=-1)
        mean_entropy = (
            (entropy_terms * valid_mask.to(entropy_terms.dtype)).sum()
            / valid_count.clamp_min(1).to(entropy_terms.dtype)
        )
        work = torch.tensor(
            n * t * self.branches, device=route_features.device, dtype=torch.int64
        )
        diagnostics: dict[str, Tensor] = {
            "autonomous_routes": autonomous_out,
            "branch_score_work": work,
            "branch_score_work_proxy": work,
            "final_active_branches": state.occupied.sum(dtype=torch.int64),
            "guidance_mask": guidance_mask,
            "guidance_probability": route_features.new_tensor(guidance_probability),
            "guided_events": guided_count,
            "guided_fraction": guided_count.to(route_features.dtype)
            / valid_count.clamp_min(1).to(route_features.dtype),
            "mean_entropy": mean_entropy,
            "valid_events": valid_count,
        }
        if self.curriculum_schedule is not None:
            diagnostics.update(
                {
                    "schedule_start_step": torch.tensor(
                        self.curriculum_schedule.start_step,
                        device=route_features.device,
                        dtype=torch.int64,
                    ),
                    "schedule_end_step": torch.tensor(
                        self.curriculum_schedule.end_step,
                        device=route_features.device,
                        dtype=torch.int64,
                    ),
                    "schedule_start_probability": route_features.new_tensor(
                        self.curriculum_schedule.start_probability
                    ),
                    "schedule_end_probability": route_features.new_tensor(
                        self.curriculum_schedule.end_probability
                    ),
                }
            )
        return PersistentRouterOutput(
            logits=logits_out,
            probabilities=probabilities_out,
            routes=routes_out,
            final_state=state,
            diagnostics=diagnostics,
        )


def _unit_interval(value: float, name: str, *, positive: bool) -> float:
    value = float(value)
    lower_ok = value > 0.0 if positive else value >= 0.0
    if not lower_ok or value > 1.0:
        boundary = "(0,1]" if positive else "[0,1]"
        raise ValueError(f"{name} must lie in {boundary}")
    return value


# Short aliases are useful to callers without weakening the explicit public
# names used in reports and serialized configuration.
RouterState = PersistentRouterState
RouterOutput = PersistentRouterOutput


__all__ = [
    "NULL_ROUTE",
    "CurriculumSchedule",
    "PersistentCausalRouter",
    "PersistentRouterOutput",
    "PersistentRouterState",
    "RouterOutput",
    "RouterState",
    "RoutingMode",
    "deterministic_guidance_mask",
    "permute_local_routes",
    "route_counts",
    "routed_lane_mask",
    "validate_routes",
]
