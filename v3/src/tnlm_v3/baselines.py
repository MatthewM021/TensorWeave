"""Strong causal recurrent and cached-Transformer binding controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .binding import BindingArchitectureConfig
from .data import BindingEventKind, BindingModelInputs, BindingTaskConfig


class BindingBaselineKind(str, Enum):
    GRU = "gru"
    CACHED_TRANSFORMER = "cached_transformer"


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


@dataclass(frozen=True)
class RecurrentBindingBaselineConfig:
    task: BindingArchitectureConfig | BindingTaskConfig
    d_model: int = 32
    hidden_dim: int = 64
    num_layers: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", _architecture(self.task))
        for name in ("d_model", "hidden_dim", "num_layers"):
            _positive_int(getattr(self, name), name)

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


@dataclass(frozen=True)
class CachedTransformerBindingBaselineConfig:
    task: BindingArchitectureConfig | BindingTaskConfig
    d_model: int = 32
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", _architecture(self.task))
        for name in ("d_model", "num_heads", "num_layers", "ff_dim"):
            _positive_int(getattr(self, name), name)
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


@dataclass
class RecurrentBindingState:
    hidden: Tensor
    valid_steps: Tensor


@dataclass
class CachedTransformerBindingState:
    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]
    occupied: Tensor
    valid_steps: Tensor


@dataclass
class BaselineBindingOutput:
    value_logits: Tensor
    final_state: RecurrentBindingState | CachedTransformerBindingState
    diagnostics: dict[str, Tensor]

    @property
    def logits(self) -> Tensor:
        return self.value_logits


def _validate_inputs(inputs: BindingModelInputs, task: BindingArchitectureConfig) -> None:
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
        (inputs.secondary_key_ids, 0, task.num_surface_keys + 1, "secondary_key_ids"),
        (inputs.arguments, 0, task.value_cardinality + 1, "arguments"),
    )
    for tensor, low, high, name in checks:
        chosen = tensor[valid]
        if bool(((chosen < low) | (chosen >= high)).any()):
            raise ValueError(f"valid {name} values are outside their vocabulary")


class _VisibleBindingEncoder(nn.Module):
    def __init__(self, task: BindingArchitectureConfig, d_model: int) -> None:
        super().__init__()
        self.task = task
        self.kind = nn.Embedding(len(BindingEventKind), d_model, padding_idx=0)
        self.primary = nn.Embedding(task.num_surface_keys + 1, d_model, padding_idx=0)
        self.secondary = nn.Embedding(task.num_surface_keys + 1, d_model, padding_idx=0)
        self.argument = nn.Embedding(task.value_cardinality + 1, d_model, padding_idx=0)
        self.projection = nn.Linear(4 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs: BindingModelInputs) -> Tensor:
        _validate_inputs(inputs, self.task)
        valid = inputs.valid_mask
        zero = torch.zeros_like(inputs.event_kinds)
        kind = self.kind(torch.where(valid, inputs.event_kinds, zero))
        primary = self.primary(torch.where(valid, inputs.primary_key_ids, zero))
        secondary = self.secondary(torch.where(valid, inputs.secondary_key_ids, zero))
        argument = self.argument(torch.where(valid, inputs.arguments, zero))
        encoded = self.norm(self.projection(torch.cat((kind, primary, secondary, argument), -1)))
        return encoded * valid.unsqueeze(-1).to(encoded.dtype)


def _positions(steps: Tensor, width: int, dtype: torch.dtype) -> Tensor:
    if width <= 0:
        raise ValueError("position width must be positive")
    work = steps.to(torch.float64).unsqueeze(-1)
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(0, width, 2, device=steps.device, dtype=torch.float64)
        / max(width, 1)
    )
    angles = work * frequencies
    output = torch.zeros(*steps.shape, width, device=steps.device, dtype=torch.float64)
    output[..., 0::2] = angles.sin()
    if width > 1:
        output[..., 1::2] = angles[..., : output[..., 1::2].shape[-1]].cos()
    return output.to(dtype)


class RecurrentBindingBaseline(nn.Module):
    def __init__(self, config: RecurrentBindingBaselineConfig) -> None:
        super().__init__()
        if not isinstance(config, RecurrentBindingBaselineConfig):
            raise TypeError("config must be RecurrentBindingBaselineConfig")
        self.config = config
        self.encoder = _VisibleBindingEncoder(config.task, config.d_model)
        self.cells = nn.ModuleList(
            [
                nn.GRUCell(config.d_model if layer == 0 else config.hidden_dim, config.hidden_dim)
                for layer in range(config.num_layers)
            ]
        )
        self.readout = nn.Linear(config.hidden_dim, config.task.value_cardinality)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> RecurrentBindingState:
        _positive_int(batch_size, "batch_size")
        return RecurrentBindingState(
            hidden=torch.zeros(
                self.config.num_layers,
                batch_size,
                self.config.hidden_dim,
                device=device,
                dtype=dtype,
            ),
            valid_steps=torch.zeros(batch_size, device=device, dtype=torch.int64),
        )

    def _validate_state(
        self,
        state: RecurrentBindingState,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not isinstance(state, RecurrentBindingState):
            raise TypeError("state must be RecurrentBindingState")
        if state.hidden.shape != (self.config.num_layers, batch, self.config.hidden_dim):
            raise ValueError("recurrent hidden state has an invalid shape")
        if state.valid_steps.shape != (batch,) or state.valid_steps.dtype != torch.int64:
            raise ValueError("recurrent valid_steps has an invalid shape or dtype")
        if (
            state.hidden.device != device
            or state.valid_steps.device != device
            or state.hidden.dtype != dtype
        ):
            raise ValueError("recurrent state device or dtype does not match inputs")
        if not bool(torch.isfinite(state.hidden).all()) or bool((state.valid_steps < 0).any()):
            raise ValueError("recurrent state is invalid")

    def step(
        self, inputs: BindingModelInputs, state: RecurrentBindingState
    ) -> BaselineBindingOutput:
        encoded = self.encoder(inputs)
        if encoded.shape[1] != 1:
            raise ValueError("step requires exactly one time position")
        batch = encoded.shape[0]
        self._validate_state(state, batch, encoded.device, encoded.dtype)
        valid = inputs.valid_mask[:, 0]
        if bool(
            (valid & (state.valid_steps == torch.iinfo(torch.int64).max)).any()
        ):
            raise ValueError("recurrent valid_steps cannot advance without overflow")
        current = encoded[:, 0]
        next_hidden: list[Tensor] = []
        for layer, cell in enumerate(self.cells):
            candidate = cell(current, state.hidden[layer])
            current = torch.where(valid[:, None], candidate, state.hidden[layer])
            next_hidden.append(current)
        hidden = torch.stack(next_hidden)
        logits = self.readout(hidden[-1])
        logits = torch.where(valid[:, None], logits, torch.zeros_like(logits))
        final = RecurrentBindingState(hidden, state.valid_steps + valid.to(torch.int64))
        return BaselineBindingOutput(
            logits[:, None],
            final,
            {"valid_events": valid.sum(dtype=torch.int64)},
        )

    def forward(
        self,
        inputs: BindingModelInputs,
        initial_state: RecurrentBindingState | None = None,
    ) -> BaselineBindingOutput:
        _validate_inputs(inputs, self.config.task)
        batch, time = inputs.token_ids.shape
        parameter = next(self.parameters())
        state = initial_state or self.initial_state(
            batch, device=parameter.device, dtype=parameter.dtype
        )
        self._validate_state(state, batch, parameter.device, parameter.dtype)
        outputs: list[Tensor] = []
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
        if outputs:
            logits = torch.stack(outputs, 1)
        else:
            logits = parameter.new_zeros(
                batch, 0, self.config.task.value_cardinality
            ) + parameter.reshape(-1)[0] * 0.0
            state = RecurrentBindingState(
                hidden=state.hidden.clone(),
                valid_steps=state.valid_steps.clone(),
            )
        return BaselineBindingOutput(
            logits,
            state,
            {"valid_events": inputs.valid_mask.sum(dtype=torch.int64)},
        )

    def structural_metrics(self) -> dict[str, int]:
        return {
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "state_scalars_per_batch_row": self.config.num_layers * self.config.hidden_dim + 1,
            "cache_layers": 0,
        }


class _CachedTransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, ff_dim)
        self.ff2 = nn.Linear(ff_dim, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def project(self, tensor: Tensor, layer: nn.Linear) -> Tensor:
        batch = tensor.shape[0]
        return layer(tensor).view(batch, self.num_heads, self.head_dim)


class CachedCausalTransformerBindingBaseline(nn.Module):
    def __init__(self, config: CachedTransformerBindingBaselineConfig) -> None:
        super().__init__()
        if not isinstance(config, CachedTransformerBindingBaselineConfig):
            raise TypeError("config must be CachedTransformerBindingBaselineConfig")
        self.config = config
        self.encoder = _VisibleBindingEncoder(config.task, config.d_model)
        self.layers = nn.ModuleList(
            [
                _CachedTransformerLayer(
                    config.d_model, config.num_heads, config.ff_dim
                )
                for _ in range(config.num_layers)
            ]
        )
        self.readout = nn.Linear(config.d_model, config.task.value_cardinality)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> CachedTransformerBindingState:
        _positive_int(batch_size, "batch_size")
        shape = (batch_size, self.config.num_heads, 0, self.config.d_model // self.config.num_heads)
        empty = tuple(torch.zeros(shape, device=device, dtype=dtype) for _ in self.layers)
        return CachedTransformerBindingState(
            keys=empty,
            values=tuple(item.clone() for item in empty),
            occupied=torch.zeros(batch_size, 0, device=device, dtype=torch.bool),
            valid_steps=torch.zeros(batch_size, device=device, dtype=torch.int64),
        )

    def _validate_state(
        self,
        state: CachedTransformerBindingState,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not isinstance(state, CachedTransformerBindingState):
            raise TypeError("state must be CachedTransformerBindingState")
        if len(state.keys) != len(self.layers) or len(state.values) != len(self.layers):
            raise ValueError("cache layer count does not match model")
        if (
            state.occupied.ndim != 2
            or state.occupied.shape[0] != batch
            or state.occupied.dtype != torch.bool
        ):
            raise ValueError("cache occupied mask has an invalid shape or dtype")
        length = state.occupied.shape[1]
        expected = (
            batch,
            self.config.num_heads,
            length,
            self.config.d_model // self.config.num_heads,
        )
        if state.valid_steps.shape != (batch,) or state.valid_steps.dtype != torch.int64:
            raise ValueError("cache valid_steps has an invalid shape or dtype")
        for item in (*state.keys, *state.values):
            if (
                item.shape != expected
                or item.device != device
                or item.dtype != dtype
                or not bool(torch.isfinite(item).all())
            ):
                raise ValueError("cached key/value tensor is invalid")
        if state.occupied.device != device or state.valid_steps.device != device:
            raise ValueError("cache device does not match inputs")
        counts = state.occupied.sum(1, dtype=torch.int64)
        if bool((state.valid_steps < 0).any()):
            raise ValueError("cache valid_steps must be nonnegative")
        expected_occupied = (
            torch.arange(length, device=device).unsqueeze(0)
            < state.valid_steps.unsqueeze(1)
        )
        if not torch.equal(state.occupied, expected_occupied):
            raise ValueError("cache occupancy must be a packed prefix")
        if not torch.equal(counts, state.valid_steps):
            raise ValueError("cache occupancy and valid_steps disagree")
        expected_capacity = int(state.valid_steps.max().item()) if batch else 0
        if length != expected_capacity:
            raise ValueError("cache capacity must equal the maximum valid_steps")
        for item in (*state.keys, *state.values):
            unused = item.masked_select(~state.occupied[:, None, :, None])
            if bool((unused != 0).any()):
                raise ValueError("unoccupied cache entries must be zero")

    def step(
        self,
        inputs: BindingModelInputs,
        state: CachedTransformerBindingState,
    ) -> BaselineBindingOutput:
        encoded = self.encoder(inputs)
        if encoded.shape[1] != 1:
            raise ValueError("step requires exactly one time position")
        batch = encoded.shape[0]
        valid = inputs.valid_mask[:, 0]
        if bool(
            isinstance(state, CachedTransformerBindingState)
            and state.valid_steps.shape == (batch,)
            and state.valid_steps.dtype == torch.int64
            and (
                valid.to(state.valid_steps.device)
                & (state.valid_steps == torch.iinfo(torch.int64).max)
            ).any()
        ):
            raise ValueError("cache valid_steps cannot advance without overflow")
        self._validate_state(state, batch, encoded.device, encoded.dtype)
        old_capacity = state.occupied.shape[1]
        new_capacity = max(
            old_capacity,
            int((state.valid_steps + valid.to(torch.int64)).max().item()),
        )
        occupied = F.pad(state.occupied, (0, new_capacity - old_capacity))
        rows = torch.arange(batch, device=encoded.device)
        slots = state.valid_steps
        occupied = occupied.clone()
        occupied[rows[valid], slots[valid]] = True
        current = encoded[:, 0] + _positions(state.valid_steps, self.config.d_model, encoded.dtype)
        keys: list[Tensor] = []
        values: list[Tensor] = []
        for index, layer in enumerate(self.layers):
            old_k = F.pad(state.keys[index], (0, 0, 0, new_capacity - old_capacity))
            old_v = F.pad(state.values[index], (0, 0, 0, new_capacity - old_capacity))
            k = layer.project(current, layer.k)
            v = layer.project(current, layer.v)
            next_k = old_k.clone()
            next_v = old_v.clone()
            next_k[rows[valid], :, slots[valid], :] = k[valid]
            next_v[rows[valid], :, slots[valid], :] = v[valid]
            q = layer.project(current, layer.q)
            scores = torch.einsum("nhd,nhsd->nhs", q, next_k) / math.sqrt(layer.head_dim)
            scores = scores.masked_fill(~occupied[:, None, :], float("-inf"))
            safe_scores = torch.where(valid[:, None, None], scores, torch.zeros_like(scores))
            weights = torch.softmax(safe_scores, -1)
            weights = torch.where(valid[:, None, None], weights, torch.zeros_like(weights))
            context = torch.einsum("nhs,nhsd->nhd", weights, next_v).reshape(
                batch, self.config.d_model
            )
            attended = layer.norm1(current + layer.o(context))
            current = layer.norm2(attended + layer.ff2(F.gelu(layer.ff1(attended))))
            current = torch.where(valid[:, None], current, torch.zeros_like(current))
            keys.append(next_k)
            values.append(next_v)
        logits = self.readout(current)
        logits = torch.where(valid[:, None], logits, torch.zeros_like(logits))
        final = CachedTransformerBindingState(
            tuple(keys),
            tuple(values),
            occupied,
            state.valid_steps + valid.to(torch.int64),
        )
        return BaselineBindingOutput(
            logits[:, None],
            final,
            {
                "valid_events": valid.sum(dtype=torch.int64),
                "cache_capacity": torch.tensor(
                    new_capacity, device=encoded.device
                ),
            },
        )

    def forward(
        self,
        inputs: BindingModelInputs,
        initial_state: CachedTransformerBindingState | None = None,
    ) -> BaselineBindingOutput:
        _validate_inputs(inputs, self.config.task)
        batch, time = inputs.token_ids.shape
        parameter = next(self.parameters())
        state = initial_state or self.initial_state(
            batch, device=parameter.device, dtype=parameter.dtype
        )
        self._validate_state(state, batch, parameter.device, parameter.dtype)
        outputs: list[Tensor] = []
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
        if outputs:
            logits = torch.stack(outputs, 1)
        else:
            logits = parameter.new_zeros(
                batch, 0, self.config.task.value_cardinality
            ) + parameter.reshape(-1)[0] * 0.0
            state = CachedTransformerBindingState(
                keys=tuple(item.clone() for item in state.keys),
                values=tuple(item.clone() for item in state.values),
                occupied=state.occupied.clone(),
                valid_steps=state.valid_steps.clone(),
            )
        return BaselineBindingOutput(
            logits,
            state,
            {
                "valid_events": inputs.valid_mask.sum(dtype=torch.int64),
                "cache_capacity": torch.tensor(
                    state.occupied.shape[1], device=parameter.device
                ),
            },
        )

    def structural_metrics(self) -> dict[str, int]:
        return {
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "state_scalars_per_cached_event": 2 * self.config.num_layers * self.config.d_model + 1,
            "state_scalars_per_batch_row": 1,
            "cache_layers": self.config.num_layers,
        }


__all__ = [
    "BaselineBindingOutput",
    "BindingBaselineKind",
    "CachedCausalTransformerBindingBaseline",
    "CachedTransformerBindingBaselineConfig",
    "CachedTransformerBindingState",
    "RecurrentBindingBaseline",
    "RecurrentBindingBaselineConfig",
    "RecurrentBindingState",
]
