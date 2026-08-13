"""Corrected causal complete-tree TTN control for dynamic binding.

The persistent state is the canonical binary decomposition of each document
prefix.  Slot ``s`` contains a chronological block of ``2**s`` real events
exactly when bit ``s`` of ``valid_steps`` is set.  Appending an event performs
binary-counter carries with a single scale-shared CP merge.  The causal prefix
root is the mask-aware complete-tree reduction obtained by right-padding the
prefix to the next power of two and treating empty children as pass-throughs.

This construction has no learned maximum-length table.  Its parameters are
independent of context length, while persistent state grows logarithmically in
the number of real (non-padding) events.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .baselines import BaselineBindingOutput
from .binding import BindingArchitectureConfig
from .data import BindingEventKind, BindingModelInputs, BindingTaskConfig
from .operators import ScaleSharedCPMerge


def _architecture(
    task: BindingArchitectureConfig | BindingTaskConfig,
) -> BindingArchitectureConfig:
    if isinstance(task, BindingTaskConfig):
        return BindingArchitectureConfig.from_task(task)
    if not isinstance(task, BindingArchitectureConfig):
        raise TypeError("task must be a BindingArchitectureConfig or BindingTaskConfig")
    return task


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _required_scales(valid_steps: Tensor) -> int:
    maximum = int(valid_steps.max().item()) if valid_steps.numel() else 0
    return max(1, maximum.bit_length())


def _positions(steps: Tensor, width: int, dtype: torch.dtype) -> Tensor:
    """Return fixed sinusoidal features for zero-based real-event positions."""

    work = steps.to(torch.float64).unsqueeze(-1)
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(0, width, 2, device=steps.device, dtype=torch.float64)
        / max(width, 1)
    )
    angles = work * frequencies
    result = torch.zeros(*steps.shape, width, device=steps.device, dtype=torch.float64)
    result[..., 0::2] = angles.sin()
    if width > 1:
        result[..., 1::2] = angles[..., : result[..., 1::2].shape[-1]].cos()
    return result.to(dtype)


@dataclass(frozen=True)
class CausalTreeBindingBaselineConfig:
    """Length-independent construction settings for the causal TTN control."""

    task: BindingArchitectureConfig | BindingTaskConfig
    d_model: int = 32
    cp_rank: int = 16
    scale_feature_dim: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", _architecture(self.task))
        for name in ("d_model", "cp_rank", "scale_feature_dim"):
            _positive_int(getattr(self, name), name)

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass
class CausalTreeBindingState:
    """Canonical binary-prefix state.

    Shapes are ``slots[N,S,D]``, ``occupied[N,S]``, and
    ``valid_steps[N]``.  Capacity is always the minimum number of scales needed
    by the largest row in the batch (with one empty scale at prefix length
    zero).
    """

    slots: Tensor
    occupied: Tensor
    valid_steps: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.slots.shape[0])

    @property
    def scales(self) -> int:
        return int(self.slots.shape[1])

    @property
    def d_model(self) -> int:
        return int(self.slots.shape[2])

    def detach(self) -> "CausalTreeBindingState":
        return CausalTreeBindingState(
            slots=self.slots.detach(),
            occupied=self.occupied.detach(),
            valid_steps=self.valid_steps.detach(),
        )

    def to(self, *args: object, **kwargs: object) -> "CausalTreeBindingState":
        slots = self.slots.to(*args, **kwargs)
        return CausalTreeBindingState(
            slots=slots,
            occupied=self.occupied.to(device=slots.device),
            valid_steps=self.valid_steps.to(device=slots.device),
        )


def _validate_inputs(
    inputs: BindingModelInputs, task: BindingArchitectureConfig
) -> None:
    if not isinstance(inputs, BindingModelInputs):
        raise TypeError("inputs must be BindingModelInputs")
    if inputs.token_ids.ndim != 2 or inputs.token_ids.shape[0] <= 0:
        raise ValueError("binding input tensors must have shape [N,T] with N > 0")
    shape = inputs.token_ids.shape
    integer = (
        inputs.token_ids,
        inputs.event_kinds,
        inputs.primary_key_ids,
        inputs.secondary_key_ids,
        inputs.arguments,
    )
    if any(item.shape != shape or item.dtype != torch.int64 for item in integer):
        raise ValueError("all binding integer fields must be int64 with shape [N,T]")
    if inputs.valid_mask.shape != shape or inputs.valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with shape [N,T]")
    if any(item.device != inputs.token_ids.device for item in (*integer, inputs.valid_mask)):
        raise ValueError("all binding model inputs must share a device")

    valid = inputs.valid_mask
    checks = (
        (inputs.token_ids, 1, task.vocab_size, "token_ids"),
        (inputs.event_kinds, 1, len(BindingEventKind), "event_kinds"),
        (inputs.primary_key_ids, 0, task.num_surface_keys + 1, "primary_key_ids"),
        (
            inputs.secondary_key_ids,
            0,
            task.num_surface_keys + 1,
            "secondary_key_ids",
        ),
        (inputs.arguments, 0, task.value_cardinality + 1, "arguments"),
    )
    for tensor, lower, upper, name in checks:
        selected = tensor[valid]
        if bool(((selected < lower) | (selected >= upper)).any()):
            raise ValueError(f"valid {name} values are outside their vocabulary")


class _VisibleBindingEncoder(nn.Module):
    def __init__(self, task: BindingArchitectureConfig, d_model: int) -> None:
        super().__init__()
        self.task = task
        self.kind = nn.Embedding(len(BindingEventKind), d_model, padding_idx=0)
        self.primary = nn.Embedding(
            task.num_surface_keys + 1, d_model, padding_idx=0
        )
        self.secondary = nn.Embedding(
            task.num_surface_keys + 1, d_model, padding_idx=0
        )
        self.argument = nn.Embedding(
            task.value_cardinality + 1, d_model, padding_idx=0
        )
        self.projection = nn.Linear(4 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs: BindingModelInputs) -> Tensor:
        _validate_inputs(inputs, self.task)
        valid = inputs.valid_mask
        zero = torch.zeros_like(inputs.event_kinds)
        kind = self.kind(torch.where(valid, inputs.event_kinds, zero))
        primary = self.primary(torch.where(valid, inputs.primary_key_ids, zero))
        secondary = self.secondary(
            torch.where(valid, inputs.secondary_key_ids, zero)
        )
        argument = self.argument(torch.where(valid, inputs.arguments, zero))
        encoded = self.norm(
            self.projection(torch.cat((kind, primary, secondary, argument), dim=-1))
        )
        return encoded * valid.unsqueeze(-1).to(encoded.dtype)


class CausalCompleteTreeBindingBaseline(nn.Module):
    """Causal fixed-geometry TTN with a canonical logarithmic prefix state."""

    def __init__(self, config: CausalTreeBindingBaselineConfig) -> None:
        super().__init__()
        if not isinstance(config, CausalTreeBindingBaselineConfig):
            raise TypeError("config must be CausalTreeBindingBaselineConfig")
        self.config = config
        self.encoder = _VisibleBindingEncoder(config.task, config.d_model)
        self.merge = ScaleSharedCPMerge(
            d_model=config.d_model,
            cp_rank=config.cp_rank,
            scale_feature_dim=config.scale_feature_dim,
        )
        self.query_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.readout_norm = nn.LayerNorm(config.d_model)
        self.readout = nn.Linear(config.d_model, config.task.value_cardinality)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> CausalTreeBindingState:
        _positive_int(batch_size, "batch_size")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a floating-point torch dtype")
        try:
            floating = torch.empty((), dtype=dtype).is_floating_point()
        except (TypeError, RuntimeError) as error:
            raise TypeError("dtype must be a floating-point torch dtype") from error
        if not floating:
            raise TypeError("dtype must be a floating-point torch dtype")
        return CausalTreeBindingState(
            slots=torch.zeros(
                batch_size, 1, self.config.d_model, device=device, dtype=dtype
            ),
            occupied=torch.zeros(batch_size, 1, device=device, dtype=torch.bool),
            valid_steps=torch.zeros(batch_size, device=device, dtype=torch.int64),
        )

    def _validate_state(
        self,
        state: CausalTreeBindingState,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not isinstance(state, CausalTreeBindingState):
            raise TypeError("state must be CausalTreeBindingState")
        if state.slots.ndim != 3:
            raise ValueError("TTN slots must have shape [N,S,D]")
        n, scales, width = state.slots.shape
        if (n, width) != (batch, self.config.d_model) or not 1 <= scales <= 63:
            raise ValueError("TTN slot dimensions do not match the model")
        if state.occupied.shape != (batch, scales) or state.occupied.dtype != torch.bool:
            raise ValueError("TTN occupancy must be boolean with shape [N,S]")
        if state.valid_steps.shape != (batch,) or state.valid_steps.dtype != torch.int64:
            raise ValueError("TTN valid_steps must be int64 with shape [N]")
        if (
            state.slots.device != device
            or state.occupied.device != device
            or state.valid_steps.device != device
            or state.slots.dtype != dtype
        ):
            raise ValueError("TTN state device or dtype does not match inputs")
        if not state.slots.is_floating_point() or not bool(
            torch.isfinite(state.slots).all()
        ):
            raise ValueError("TTN slots must be finite floating-point values")
        if bool((state.valid_steps < 0).any()):
            raise ValueError("TTN valid_steps must be nonnegative")
        if scales != _required_scales(state.valid_steps):
            raise ValueError("TTN state capacity is not the canonical prefix capacity")

        scale_ids = torch.arange(scales, device=device, dtype=torch.int64)
        expected = ((state.valid_steps.unsqueeze(1) >> scale_ids) & 1).bool()
        if not torch.equal(state.occupied, expected):
            raise ValueError("TTN occupancy does not match the binary prefix")
        unused = state.slots.masked_select(~state.occupied.unsqueeze(-1))
        if bool((unused != 0).any()):
            raise ValueError("unoccupied TTN slots must be zero")

    def _append(
        self,
        state: CausalTreeBindingState,
        event: Tensor,
        valid: Tensor,
    ) -> tuple[CausalTreeBindingState, Tensor]:
        next_steps = state.valid_steps + valid.to(torch.int64)
        scales = _required_scales(next_steps)
        slots = F.pad(state.slots, (0, 0, 0, scales - state.scales))
        occupied = F.pad(state.occupied, (0, scales - state.scales))
        carry = event
        has_carry = valid
        slot_slices: list[Tensor] = []
        merge_count = torch.zeros((), device=event.device, dtype=torch.int64)

        for scale in range(scales):
            previous = slots[:, scale]
            was_occupied = occupied[:, scale]
            place = has_carry & ~was_occupied
            collide = has_carry & was_occupied
            next_carry = torch.zeros_like(carry)
            if bool(collide.any()):
                merged = self.merge(
                    previous[collide],
                    carry[collide],
                    scale=scale,
                    global_path=True,
                )
                next_carry = next_carry.index_put((collide,), merged)
            slot_slices.append(
                torch.where(
                    place.unsqueeze(-1),
                    carry,
                    torch.where(
                        collide.unsqueeze(-1), torch.zeros_like(previous), previous
                    ),
                )
            )
            carry = next_carry
            has_carry = collide
            merge_count = merge_count + collide.sum(dtype=torch.int64)

        if bool(has_carry.any()):
            raise RuntimeError("internal error: TTN state did not absorb a carry")
        canonical_occupied = (
            (
                next_steps.unsqueeze(1)
                >> torch.arange(scales, device=event.device, dtype=torch.int64)
            )
            & 1
        ).bool()
        return (
            CausalTreeBindingState(
                slots=torch.stack(slot_slices, dim=1),
                occupied=canonical_occupied,
                valid_steps=next_steps,
            ),
            merge_count,
        )

    def _prefix_root(
        self, state: CausalTreeBindingState, active: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Reduce occupied blocks as a right-padded complete prefix tree.

        Low-to-high traversal first builds the newest partial subtree.  Each
        higher occupied slot is an older complete left child and contracts at
        its own scale with that newer right subtree.
        """

        if active.shape != (state.batch_size,) or active.dtype != torch.bool:
            raise ValueError("active must be boolean with shape [N]")
        if active.device != state.slots.device:
            raise ValueError("active must share the state device")
        root = state.slots.new_zeros(state.batch_size, self.config.d_model)
        has_root = torch.zeros(
            state.batch_size, device=state.slots.device, dtype=torch.bool
        )
        merge_count = torch.zeros((), device=state.slots.device, dtype=torch.int64)
        for scale in range(state.scales):
            present = state.occupied[:, scale] & active
            first = present & ~has_root
            combine = present & has_root
            candidate = torch.zeros_like(root)
            if bool(combine.any()):
                merged = self.merge(
                    state.slots[combine, scale],
                    root[combine],
                    scale=scale,
                    global_path=True,
                )
                candidate = candidate.index_put((combine,), merged)
            root = torch.where(
                combine.unsqueeze(-1),
                candidate,
                torch.where(first.unsqueeze(-1), state.slots[:, scale], root),
            )
            has_root = has_root | present
            merge_count = merge_count + combine.sum(dtype=torch.int64)
        return root, merge_count

    def step(
        self,
        inputs: BindingModelInputs,
        state: CausalTreeBindingState,
    ) -> BaselineBindingOutput[CausalTreeBindingState]:
        encoded = self.encoder(inputs)
        if encoded.shape[1] != 1:
            raise ValueError("step requires exactly one time position")
        batch = encoded.shape[0]
        self._validate_state(state, batch, encoded.device, encoded.dtype)
        valid = inputs.valid_mask[:, 0]
        if bool(
            (valid & (state.valid_steps == torch.iinfo(torch.int64).max)).any()
        ):
            raise ValueError("TTN valid_steps cannot advance without overflow")

        if not bool(valid.any()):
            dependency = next(self.parameters()).reshape(-1)[0] * 0.0
            logits = encoded.new_zeros(
                batch, 1, self.config.task.value_cardinality
            ) + dependency
            zero = torch.zeros((), device=encoded.device, dtype=torch.int64)
            return BaselineBindingOutput(
                value_logits=logits,
                final_state=CausalTreeBindingState(
                    slots=state.slots.clone(),
                    occupied=state.occupied.clone(),
                    valid_steps=state.valid_steps.clone(),
                ),
                diagnostics={
                    "valid_events": zero,
                    "update_merge_count": zero,
                    "readout_merge_count": zero,
                    "active_slots": state.occupied.sum(dtype=torch.int64),
                    "allocated_scales": torch.tensor(
                        state.scales, device=encoded.device, dtype=torch.int64
                    ),
                },
            )

        position_steps = torch.where(valid, state.valid_steps, 0)
        position = _positions(
            position_steps, self.config.d_model, encoded.dtype
        )
        leaf = (encoded[:, 0] + position) * valid.unsqueeze(-1).to(encoded.dtype)
        final_state, state_merges = self._append(state, leaf, valid)
        root, readout_merges = self._prefix_root(final_state, valid)
        representation = self.readout_norm(
            root + self.query_projection(leaf)
        )
        logits = self.readout(representation)
        logits = torch.where(valid[:, None], logits, torch.zeros_like(logits))
        return BaselineBindingOutput(
            value_logits=logits[:, None],
            final_state=final_state,
            diagnostics={
                "valid_events": valid.sum(dtype=torch.int64),
                "update_merge_count": state_merges,
                "readout_merge_count": readout_merges,
                "active_slots": final_state.occupied.sum(dtype=torch.int64),
                "allocated_scales": torch.tensor(
                    final_state.scales, device=encoded.device, dtype=torch.int64
                ),
            },
        )

    def forward(
        self,
        inputs: BindingModelInputs,
        initial_state: CausalTreeBindingState | None = None,
    ) -> BaselineBindingOutput[CausalTreeBindingState]:
        _validate_inputs(inputs, self.config.task)
        batch, time = inputs.token_ids.shape
        parameter = next(self.parameters())
        state = (
            self.initial_state(
                batch, device=parameter.device, dtype=parameter.dtype
            )
            if initial_state is None
            else initial_state
        )
        self._validate_state(state, batch, parameter.device, parameter.dtype)

        outputs: list[Tensor] = []
        state_merges = torch.zeros((), device=parameter.device, dtype=torch.int64)
        readout_merges = torch.zeros(
            (), device=parameter.device, dtype=torch.int64
        )
        for index in range(time):
            one = BindingModelInputs(
                token_ids=inputs.token_ids[:, index : index + 1],
                event_kinds=inputs.event_kinds[:, index : index + 1],
                primary_key_ids=inputs.primary_key_ids[:, index : index + 1],
                secondary_key_ids=inputs.secondary_key_ids[:, index : index + 1],
                arguments=inputs.arguments[:, index : index + 1],
                valid_mask=inputs.valid_mask[:, index : index + 1],
            )
            result = self.step(one, state)
            state = result.final_state
            outputs.append(result.value_logits[:, 0])
            state_merges = state_merges + result.diagnostics["update_merge_count"]
            readout_merges = (
                readout_merges + result.diagnostics["readout_merge_count"]
            )

        if outputs:
            logits = torch.stack(outputs, dim=1)
        else:
            logits = parameter.new_zeros(
                batch, 0, self.config.task.value_cardinality
            ) + parameter.reshape(-1)[0] * 0.0
            state = CausalTreeBindingState(
                slots=state.slots.clone(),
                occupied=state.occupied.clone(),
                valid_steps=state.valid_steps.clone(),
            )
        return BaselineBindingOutput(
            value_logits=logits,
            final_state=state,
            diagnostics={
                "valid_events": inputs.valid_mask.sum(dtype=torch.int64),
                "update_merge_count": state_merges,
                "readout_merge_count": readout_merges,
                "active_slots": state.occupied.sum(dtype=torch.int64),
                "allocated_scales": torch.tensor(
                    state.scales, device=parameter.device, dtype=torch.int64
                ),
            },
        )

    def structural_metrics(
        self,
        state: CausalTreeBindingState | None = None,
        *,
        merge_count: int | Tensor = 0,
    ) -> dict[str, int]:
        if isinstance(merge_count, Tensor):
            if merge_count.numel() != 1:
                raise ValueError("merge_count tensor must contain one value")
            if (
                merge_count.dtype == torch.bool
                or merge_count.is_floating_point()
                or merge_count.is_complex()
            ):
                raise TypeError("merge_count tensor must have an integer dtype")
            merges = int(merge_count.item())
        elif isinstance(merge_count, bool) or not isinstance(merge_count, int):
            raise TypeError("merge_count must be an integer")
        else:
            merges = merge_count
        if merges < 0:
            raise ValueError("merge_count cannot be negative")

        merge_metrics = self.merge.structural_metrics(merges)
        metrics = {
            **merge_metrics,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "state_scalars_per_occupied_slot": self.config.d_model,
            "occupancy_scalars_per_allocated_slot": 1,
            "state_counter_scalars_per_batch_row": 1,
            "tree_lanes": 1,
        }
        if state is not None:
            self._validate_state(
                state,
                state.batch_size,
                state.slots.device,
                state.slots.dtype,
            )
            active_slots = int(state.occupied.sum().item())
            allocated_slots = state.occupied.numel()
            batch_size = state.valid_steps.numel()
            active_float_elements = active_slots * self.config.d_model
            allocated_float_elements = state.slots.numel()
            active_logical_scalars = active_float_elements + active_slots + batch_size
            allocated_logical_scalars = (
                allocated_float_elements + allocated_slots + batch_size
            )
            metrics.update(
                {
                    "active_slots": active_slots,
                    "allocated_slots": allocated_slots,
                    "active_state_elements": active_float_elements,
                    "allocated_state_elements": allocated_float_elements,
                    "active_state_logical_scalars": active_logical_scalars,
                    "allocated_state_logical_scalars": allocated_logical_scalars,
                    "active_state_bytes": (
                        active_float_elements * state.slots.element_size()
                        + active_slots * state.occupied.element_size()
                        + state.valid_steps.numel()
                        * state.valid_steps.element_size()
                    ),
                    "allocated_state_bytes": (
                        state.slots.numel() * state.slots.element_size()
                        + state.occupied.numel() * state.occupied.element_size()
                        + state.valid_steps.numel()
                        * state.valid_steps.element_size()
                    ),
                }
            )
        return metrics


__all__ = [
    "CausalCompleteTreeBindingBaseline",
    "CausalTreeBindingBaselineConfig",
    "CausalTreeBindingState",
]
