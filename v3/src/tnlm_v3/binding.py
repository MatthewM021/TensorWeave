"""Integrated dynamic-binding model for V3 Milestone 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal

import torch
from torch import Tensor, nn

from .data import BindingEventKind, BindingModelInputs, BindingTaskConfig
from .forest import ForestReadout, ForestState, ScaleSharedBinaryForest
from .routing import (
    NULL_ROUTE,
    CurriculumSchedule,
    PersistentCausalRouter,
    PersistentRouterOutput,
    PersistentRouterState,
    RoutingMode,
)


@dataclass(frozen=True)
class BindingArchitectureConfig:
    """Only task structure that the model may know at construction time."""

    num_surface_keys: int
    value_cardinality: int
    branches: int

    def __post_init__(self) -> None:
        for name in ("num_surface_keys", "value_cardinality", "branches"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")

    @property
    def vocab_size(self) -> int:
        key_fields = self.num_surface_keys + 1
        value_fields = self.value_cardinality + 1
        return 1 + 6 * key_fields * key_fields * value_fields

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_task(cls, task: BindingTaskConfig) -> "BindingArchitectureConfig":
        return cls(
            num_surface_keys=task.num_surface_keys,
            value_cardinality=task.value_cardinality,
            branches=task.branches,
        )


@dataclass(frozen=True)
class BindingModelConfig:
    """Construction settings for the routed dynamic-binding model."""

    task: BindingArchitectureConfig | BindingTaskConfig
    d_model: int = 24
    cp_rank: int = 12
    router_hidden_dim: int = 24
    routing_mode: RoutingMode | str = RoutingMode.ORACLE
    curriculum_schedule: CurriculumSchedule | None = None
    curriculum_seed: int = 0
    scale_feature_dim: int = 8
    straight_through_route_surrogate: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.task, BindingTaskConfig):
            object.__setattr__(self, "task", BindingArchitectureConfig.from_task(self.task))
        elif not isinstance(self.task, BindingArchitectureConfig):
            raise TypeError("task must be a BindingTaskConfig or BindingArchitectureConfig")
        for name in ("d_model", "cp_rank", "router_hidden_dim", "scale_feature_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        mode = RoutingMode(self.routing_mode)
        object.__setattr__(self, "routing_mode", mode)
        if mode is RoutingMode.CURRICULUM:
            if not isinstance(self.curriculum_schedule, CurriculumSchedule):
                raise ValueError("curriculum mode requires a CurriculumSchedule")
        elif self.curriculum_schedule is not None:
            raise ValueError("only curriculum mode may define a curriculum schedule")
        if not isinstance(self.curriculum_seed, int) or isinstance(
            self.curriculum_seed, bool
        ):
            raise TypeError("curriculum_seed must be an integer")
        if not isinstance(self.straight_through_route_surrogate, bool):
            raise TypeError("straight_through_route_surrogate must be boolean")

    def canonical_json(self) -> str:
        value = asdict(self)
        value["routing_mode"] = self.routing_mode.value
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass
class BindingModelOutput:
    value_logits: Tensor
    routes: Tensor
    route_logits: Tensor
    route_probabilities: Tensor
    forest_state: ForestState
    router_state: PersistentRouterState
    diagnostics: dict[str, Tensor]

    @property
    def logits(self) -> Tensor:
        return self.value_logits


class BindingEventEncoder(nn.Module):
    """Encode only the explicitly model-visible structured event fields."""

    def __init__(self, task: BindingArchitectureConfig, d_model: int) -> None:
        super().__init__()
        self.task = task
        self.d_model = d_model
        self.kind_embedding = nn.Embedding(
            len(BindingEventKind), d_model, padding_idx=int(BindingEventKind.PAD)
        )
        self.primary_embedding = nn.Embedding(
            task.num_surface_keys + 1, d_model, padding_idx=0
        )
        self.secondary_embedding = nn.Embedding(
            task.num_surface_keys + 1, d_model, padding_idx=0
        )
        self.argument_embedding = nn.Embedding(
            task.value_cardinality + 1, d_model, padding_idx=0
        )
        self.event_projection = nn.Linear(4 * d_model, d_model)
        self.event_norm = nn.LayerNorm(d_model)
        self.route_projection = nn.Linear(3 * d_model, d_model)
        self.route_norm = nn.LayerNorm(d_model)

    def _validate(self, inputs: BindingModelInputs) -> tuple[int, int]:
        if not isinstance(inputs, BindingModelInputs):
            raise TypeError("inputs must be BindingModelInputs")
        if inputs.token_ids.ndim != 2:
            raise ValueError("binding input tensors must have shape [N,T]")
        shape = inputs.token_ids.shape
        tensors = (
            inputs.token_ids,
            inputs.event_kinds,
            inputs.primary_key_ids,
            inputs.secondary_key_ids,
            inputs.arguments,
        )
        if any(tensor.shape != shape for tensor in tensors):
            raise ValueError("all binding event fields must have shape [N,T]")
        if any(tensor.dtype != torch.int64 for tensor in tensors):
            raise TypeError("binding event fields must use int64")
        if inputs.valid_mask.shape != shape or inputs.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean with shape [N,T]")
        device = inputs.token_ids.device
        if any(tensor.device != device for tensor in (*tensors, inputs.valid_mask)):
            raise ValueError("all binding model inputs must share a device")
        if shape[0] <= 0:
            raise ValueError("binding inputs require a positive batch dimension")

        valid = inputs.valid_mask
        checks = (
            (inputs.token_ids, 1, self.task.vocab_size, "token_ids"),
            (inputs.event_kinds, 1, len(BindingEventKind), "event_kinds"),
            (
                inputs.primary_key_ids,
                0,
                self.task.num_surface_keys + 1,
                "primary_key_ids",
            ),
            (
                inputs.secondary_key_ids,
                0,
                self.task.num_surface_keys + 1,
                "secondary_key_ids",
            ),
            (inputs.arguments, 0, self.task.value_cardinality + 1, "arguments"),
        )
        for tensor, lower, upper, name in checks:
            selected = tensor[valid]
            if bool(((selected < lower) | (selected >= upper)).any()):
                raise ValueError(f"valid {name} values are outside their vocabulary")
        return int(shape[0]), int(shape[1])

    def forward(self, inputs: BindingModelInputs) -> tuple[Tensor, Tensor]:
        self._validate(inputs)
        valid = inputs.valid_mask
        zeros = torch.zeros_like(inputs.token_ids)
        kind_ids = torch.where(valid, inputs.event_kinds, zeros)
        primary_ids = torch.where(valid, inputs.primary_key_ids, zeros)
        secondary_ids = torch.where(valid, inputs.secondary_key_ids, zeros)
        argument_ids = torch.where(valid, inputs.arguments, zeros)
        kind = self.kind_embedding(kind_ids)
        primary = self.primary_embedding(primary_ids)
        secondary = self.secondary_embedding(secondary_ids)
        argument = self.argument_embedding(argument_ids)
        event = self.event_norm(
            self.event_projection(torch.cat((kind, primary, secondary, argument), dim=-1))
        )
        route_feature = self.route_norm(
            self.route_projection(torch.cat((kind, primary, argument), dim=-1))
        )
        mask = valid.unsqueeze(-1).to(event.dtype)
        return event * mask, route_feature * mask


class RoutedBindingModel(nn.Module):
    """Persistent router plus scale-shared tensor forest for binding queries."""

    def __init__(self, config: BindingModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = BindingEventEncoder(config.task, config.d_model)
        self.router = PersistentCausalRouter(
            feature_dim=config.d_model,
            branches=config.task.branches,
            hidden_dim=config.router_hidden_dim,
            mode=config.routing_mode,
            include_global=True,
            include_null=True,
            curriculum_schedule=config.curriculum_schedule,
            curriculum_seed=config.curriculum_seed,
        )
        self.forest = ScaleSharedBinaryForest(
            d_model=config.d_model,
            branches=config.task.branches,
            cp_rank=config.cp_rank,
            scale_feature_dim=config.scale_feature_dim,
        )
        self.readout = ForestReadout(
            d_model=config.d_model,
            branches=config.task.branches,
            output_size=config.task.value_cardinality,
            scale_feature_dim=config.scale_feature_dim,
        )

    def _route_classes(self, routes: Tensor) -> Tensor:
        branches = self.config.task.branches
        null_class = branches + 1
        return torch.where(
            routes == NULL_ROUTE,
            torch.full_like(routes, null_class),
            routes,
        )

    def _route_strengths(self, output: PersistentRouterOutput) -> Tensor:
        classes = self._route_classes(output.routes)
        selected = output.probabilities.gather(-1, classes.unsqueeze(-1)).squeeze(-1)
        if self.config.straight_through_route_surrogate:
            return torch.ones_like(selected) - selected.detach() + selected
        return torch.ones_like(selected)

    def forward(
        self,
        inputs: BindingModelInputs,
        *,
        route_labels: Tensor | None = None,
        training_step: int = 0,
        implementation: Literal["streaming", "parallel"] = "streaming",
    ) -> BindingModelOutput:
        if implementation not in ("streaming", "parallel"):
            raise ValueError("implementation must be 'streaming' or 'parallel'")
        events, route_features = self.encoder(inputs)
        local_event = inputs.valid_mask & (inputs.primary_key_ids > 0)
        allowed_classes = torch.zeros(
            *inputs.valid_mask.shape,
            self.router.class_count,
            device=events.device,
            dtype=torch.bool,
        )
        allowed_classes[:, :, : self.config.task.branches] = local_event.unsqueeze(-1)
        allowed_classes[:, :, self.config.task.branches :] = (
            inputs.valid_mask & ~local_event
        ).unsqueeze(-1)
        router_output = self.router(
            route_features,
            inputs.valid_mask,
            route_labels=route_labels,
            allowed_classes=allowed_classes,
            training_step=training_step,
        )
        n, t, _ = events.shape
        value_logits: list[Tensor] = []
        strengths = self._route_strengths(router_output)

        if implementation == "streaming":
            state = self.forest.initial_state(n, device=events.device, dtype=events.dtype)
            merge_count = torch.zeros((), device=events.device, dtype=torch.int64)
            for index in range(t):
                result = self.forest.step(
                    state,
                    events[:, index],
                    router_output.routes[:, index],
                    inputs.valid_mask[:, index],
                    route_strength=strengths[:, index],
                )
                state = result.state
                merge_count = merge_count + result.merge_count
                current = self.readout(state, events[:, index])
                value_logits.append(
                    torch.where(
                        inputs.valid_mask[:, index, None],
                        current,
                        torch.zeros_like(current),
                    )
                )
        else:
            prefix = self.forest.reduce_parallel_prefixes(
                events * strengths.unsqueeze(-1),
                router_output.routes,
                inputs.valid_mask,
            )
            state = self.forest.initial_state(n, device=events.device, dtype=events.dtype)
            for index, state in enumerate(prefix.states):
                current = self.readout(state, events[:, index])
                value_logits.append(
                    torch.where(
                        inputs.valid_mask[:, index, None],
                        current,
                        torch.zeros_like(current),
                    )
                )
            merge_count = prefix.merge_count

        logits = (
            torch.stack(value_logits, dim=1)
            if value_logits
            else events.new_zeros(n, 0, self.config.task.value_cardinality)
        )
        diagnostics = dict(router_output.diagnostics)
        diagnostics.update(
            {
                "forest_merge_count": merge_count,
                "forest_active_slots": state.occupied.sum(dtype=torch.int64),
                "straight_through_route_surrogate": torch.tensor(
                    int(self.config.straight_through_route_surrogate),
                    device=events.device,
                    dtype=torch.int64,
                ),
            }
        )
        return BindingModelOutput(
            value_logits=logits,
            routes=router_output.routes,
            route_logits=router_output.logits,
            route_probabilities=router_output.probabilities,
            forest_state=state,
            router_state=router_output.final_state,
            diagnostics=diagnostics,
        )


__all__ = [
    "BindingArchitectureConfig",
    "BindingEventEncoder",
    "BindingModelConfig",
    "BindingModelOutput",
    "RoutedBindingModel",
]
