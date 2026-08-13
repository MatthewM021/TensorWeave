"""Shared autonomous training and evaluation for binding baselines.

This module deliberately supports only the three non-routing Milestone-4
controls: the GRU, cached causal Transformer, and causal complete-tree TTN.
Every model invocation receives exactly :class:`BindingModelInputs`.  Oracle
routes and all other evaluation-only annotations remain outside the forward
path; predictive query targets enter only the masked supervised objective, and
the held-out-combination mask enters only post-forward evaluation summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import math
from typing import TypeAlias

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .baselines import (
    BaselineBindingOutput,
    CachedCausalTransformerBindingBaseline,
    CachedTransformerBindingState,
    RecurrentBindingBaseline,
    RecurrentBindingState,
)
from .benchmark import QueryAccuracy, exact_query_accuracy
from .causal_ttn import (
    CausalCompleteTreeBindingBaseline,
    CausalTreeBindingState,
)
from .data import BindingBatch, BindingEventKind, BindingModelInputs


BaselineCampaignModel: TypeAlias = (
    RecurrentBindingBaseline
    | CachedCausalTransformerBindingBaseline
    | CausalCompleteTreeBindingBaseline
)

_MODEL_TYPES = (
    RecurrentBindingBaseline,
    CachedCausalTransformerBindingBaseline,
    CausalCompleteTreeBindingBaseline,
)


@dataclass
class BaselineQueryLoss:
    """Mean cross-entropy over valid query events."""

    total: Tensor
    query_count: int


@dataclass(frozen=True)
class BaselineQueryMetrics:
    """Finite query cross-entropy and exact class accuracy for one stratum."""

    cross_entropy: float
    accuracy: QueryAccuracy

    @property
    def correct(self) -> int:
        return self.accuracy.correct

    @property
    def query_count(self) -> int:
        return self.accuracy.query_count


@dataclass(frozen=True)
class BaselineCampaignEvaluationSummary:
    """Overall, seen, and held-out predictive metrics plus model structure."""

    query: BaselineQueryMetrics
    seen_query: BaselineQueryMetrics
    heldout_query: BaselineQueryMetrics
    structural_metrics: dict[str, int]


def _validate_model(model: nn.Module) -> BaselineCampaignModel:
    if type(model) not in _MODEL_TYPES:
        raise TypeError(
            "model must be a recurrent, cached-Transformer, or causal-tree "
            "binding baseline"
        )
    return model


def _validate_training_options(
    optimizer: torch.optim.Optimizer,
    model: BaselineCampaignModel,
    max_gradient_norm: float | None,
) -> float | None:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if max_gradient_norm is not None:
        if isinstance(max_gradient_norm, bool) or not isinstance(
            max_gradient_norm, (int, float)
        ):
            raise TypeError("max_gradient_norm must be positive and finite or None")
        max_gradient_norm = float(max_gradient_norm)
        if not math.isfinite(max_gradient_norm) or max_gradient_norm <= 0:
            raise ValueError("max_gradient_norm must be positive and finite or None")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimized = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    trainable_ids = [id(parameter) for parameter in trainable]
    optimized_ids = [id(parameter) for parameter in optimized]
    if len(optimized_ids) != len(set(optimized_ids)):
        raise ValueError("optimizer contains a model parameter more than once")
    if set(optimized_ids) != set(trainable_ids):
        raise ValueError("optimizer parameters must exactly match trainable model parameters")
    return max_gradient_norm


def _validate_model_values(model: BaselineCampaignModel) -> None:
    parameters = tuple(model.parameters())
    if not parameters:
        raise ValueError("baseline model must expose parameters")
    for parameter in parameters:
        if not parameter.is_floating_point():
            raise TypeError("baseline parameters must use floating-point dtypes")
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError("baseline parameters must be finite")
    for buffer in model.buffers():
        if buffer.is_floating_point() and not bool(torch.isfinite(buffer).all()):
            raise ValueError("baseline floating-point buffers must be finite")


def _validate_output(
    model: BaselineCampaignModel,
    inputs: BindingModelInputs,
    output: object,
) -> BaselineBindingOutput[object]:
    if not isinstance(output, BaselineBindingOutput):
        raise TypeError("baseline forward must return BaselineBindingOutput")
    logits = output.value_logits
    expected = (
        inputs.valid_mask.shape[0],
        inputs.valid_mask.shape[1],
        model.config.task.value_cardinality,
    )
    if not isinstance(logits, Tensor) or logits.shape != expected:
        raise ValueError(f"value_logits must have shape {expected}")
    if not logits.is_floating_point():
        raise TypeError("value_logits must use a floating-point dtype")
    if logits.device != inputs.valid_mask.device:
        raise ValueError("value_logits and inputs must share a device")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("value_logits must be finite")
    padded = logits.masked_select(~inputs.valid_mask.unsqueeze(-1))
    if bool((padded != 0).any()):
        raise ValueError("padded value_logits must be exactly zero")
    if not isinstance(output.diagnostics, dict) or any(
        not isinstance(key, str) or not isinstance(value, Tensor)
        for key, value in output.diagnostics.items()
    ):
        raise TypeError("baseline diagnostics must map strings to tensors")
    for value in output.diagnostics.values():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("baseline diagnostics must be finite")
    _validate_final_state(model, inputs, output)
    return output


def _forward_baseline(
    model: BaselineCampaignModel,
    inputs: BindingModelInputs,
) -> BaselineBindingOutput[object]:
    # Keep this call intentionally positional and single-argument. In
    # particular, no evaluation object or routing label can enter the model.
    return _validate_output(model, inputs, model(inputs))


def _query_mask(inputs: BindingModelInputs) -> Tensor:
    return inputs.valid_mask & (
        inputs.event_kinds == int(BindingEventKind.QUERY)
    )


def _validate_targets(logits: Tensor, targets: Tensor) -> None:
    expected = logits.shape[:2]
    if not isinstance(targets, Tensor) or targets.shape != expected:
        raise ValueError(f"targets must have shape {tuple(expected)}")
    if targets.dtype != torch.int64:
        raise TypeError("targets must use torch.int64")
    if targets.device != logits.device:
        raise ValueError("targets and value_logits must share a device")


def _masked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> tuple[Tensor, int]:
    if mask.shape != logits.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("query mask must be boolean with shape [N,T]")
    if mask.device != logits.device:
        raise ValueError("query mask and value_logits must share a device")
    count = int(mask.sum().item())
    if count:
        selected_targets = targets[mask]
        if not bool(
            (
                (selected_targets >= 0)
                & (selected_targets < logits.shape[-1])
            ).all()
        ):
            raise ValueError("query targets must index the value-logit classes")
        loss = F.cross_entropy(logits[mask], selected_targets)
    else:
        # This remains a well-formed zero for direct callers. The train helper
        # explicitly skips backward and optimizer.step for an empty selection.
        loss = logits.sum() * 0.0
    if not bool(torch.isfinite(loss)):
        raise ValueError("query cross-entropy must be finite")
    return loss, count


def compute_baseline_query_loss(
    output: BaselineBindingOutput[object],
    inputs: BindingModelInputs,
    targets: Tensor,
) -> BaselineQueryLoss:
    """Compute predictive CE over valid query positions only.

    The API accepts the target tensor directly so route and held-out metadata
    cannot accidentally enter the training objective.
    """

    if not isinstance(output, BaselineBindingOutput):
        raise TypeError("output must be BaselineBindingOutput")
    if not isinstance(inputs, BindingModelInputs):
        raise TypeError("inputs must be BindingModelInputs")
    logits = output.value_logits
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise ValueError("value_logits must have shape [N,T,C]")
    if logits.shape[:2] != inputs.valid_mask.shape or logits.shape[-1] <= 0:
        raise ValueError("value_logits and inputs have incompatible shapes")
    if not logits.is_floating_point():
        raise TypeError("value_logits must use a floating-point dtype")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("value_logits must be finite")
    _validate_targets(logits, targets)
    total, count = _masked_cross_entropy(
        logits, targets, _query_mask(inputs)
    )
    return BaselineQueryLoss(total=total, query_count=count)


def _query_metrics(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> BaselineQueryMetrics:
    loss, count = _masked_cross_entropy(logits, targets, mask)
    accuracy = exact_query_accuracy(logits, targets, mask)
    if accuracy.query_count != count:
        raise RuntimeError("query metric counts disagree")
    value = float(loss.detach().cpu())
    if not math.isfinite(value):
        raise ValueError("reported query cross-entropy must be finite")
    return BaselineQueryMetrics(cross_entropy=value, accuracy=accuracy)


def _state_tensors(value: object, seen: set[int]) -> list[Tensor]:
    if isinstance(value, Tensor):
        identity = id(value)
        if identity in seen:
            return []
        seen.add(identity)
        return [value]
    if is_dataclass(value) and not isinstance(value, type):
        result: list[Tensor] = []
        for field in fields(value):
            result.extend(_state_tensors(getattr(value, field.name), seen))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_state_tensors(item, seen))
        return result
    if isinstance(value, dict):
        result = []
        for key in sorted(value):
            result.extend(_state_tensors(value[key], seen))
        return result
    if value is None:
        return []
    raise TypeError("baseline state contains an unsupported non-tensor field")


def _nonnegative_metric(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"structural metric {name!r} must be an integer")
    if value < 0:
        raise ValueError(f"structural metric {name!r} cannot be negative")
    return value


def _diagnostic_count(output: BaselineBindingOutput[object], name: str) -> int:
    value = output.diagnostics.get(name)
    if (
        not isinstance(value, Tensor)
        or value.numel() != 1
        or value.dtype == torch.bool
        or value.is_floating_point()
        or value.is_complex()
    ):
        raise ValueError(f"diagnostic {name!r} must be one integer tensor")
    result = int(value.item())
    if result < 0:
        raise ValueError(f"diagnostic {name!r} cannot be negative")
    return result


def _expected_tree_merge_counts(valid_counts: Tensor) -> tuple[int, int]:
    update_merges = 0
    readout_merges = 0
    for raw_count in valid_counts.detach().cpu().tolist():
        count = int(raw_count)
        update_merges += count - count.bit_count()
        readout_merges += sum(step.bit_count() - 1 for step in range(1, count + 1))
    return update_merges, readout_merges


def _validate_final_state(
    model: BaselineCampaignModel,
    inputs: BindingModelInputs,
    output: BaselineBindingOutput[object],
) -> None:
    parameter = next(model.parameters())
    batch = inputs.valid_mask.shape[0]
    valid_counts = inputs.valid_mask.sum(dim=1, dtype=torch.int64)
    expected_valid_events = int(valid_counts.sum().item())
    if _diagnostic_count(output, "valid_events") != expected_valid_events:
        raise ValueError("valid-event diagnostic does not match the model inputs")

    state = output.final_state
    if type(model) is RecurrentBindingBaseline:
        if type(state) is not RecurrentBindingState:
            raise TypeError("recurrent output has the wrong final state type")
        model._validate_state(state, batch, parameter.device, parameter.dtype)
    elif type(model) is CachedCausalTransformerBindingBaseline:
        if type(state) is not CachedTransformerBindingState:
            raise TypeError("cached-Transformer output has the wrong final state type")
        model._validate_state(state, batch, parameter.device, parameter.dtype)
        if _diagnostic_count(output, "cache_capacity") != state.occupied.shape[1]:
            raise ValueError("cache-capacity diagnostic does not match final state")
    else:
        if type(state) is not CausalTreeBindingState:
            raise TypeError("causal-tree output has the wrong final state type")
        model._validate_state(state, batch, parameter.device, parameter.dtype)
        update_merges, readout_merges = _expected_tree_merge_counts(valid_counts)
        if _diagnostic_count(output, "update_merge_count") != update_merges:
            raise ValueError("tree update-merge diagnostic does not match inputs")
        if _diagnostic_count(output, "readout_merge_count") != readout_merges:
            raise ValueError("tree readout-merge diagnostic does not match inputs")
        if _diagnostic_count(output, "active_slots") != int(state.occupied.sum().item()):
            raise ValueError("active-slot diagnostic does not match final state")
        if _diagnostic_count(output, "allocated_scales") != state.scales:
            raise ValueError("allocated-scale diagnostic does not match final state")

    if not torch.equal(state.valid_steps, valid_counts.to(state.valid_steps.device)):
        raise ValueError("final valid_steps do not match the model inputs")


def baseline_structural_metrics(
    model: BaselineCampaignModel,
    inputs: BindingModelInputs,
    output: BaselineBindingOutput[object],
) -> dict[str, int]:
    """Return native architecture metrics plus exact persistent-state size."""

    model = _validate_model(model)
    output = _validate_output(model, inputs, output)

    if type(model) is RecurrentBindingBaseline:
        native = model.structural_metrics()
    elif type(model) is CachedCausalTransformerBindingBaseline:
        native = model.structural_metrics()
    else:
        merge_count = _diagnostic_count(output, "update_merge_count")
        merge_count += _diagnostic_count(output, "readout_merge_count")
        native = model.structural_metrics(
            output.final_state, merge_count=merge_count
        )

    metrics = {
        name: _nonnegative_metric(name, value)
        for name, value in native.items()
    }
    parameters = tuple(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if "parameter_count" in metrics and metrics["parameter_count"] != parameter_count:
        raise ValueError("native and observed parameter counts disagree")
    metrics["parameter_count"] = parameter_count
    metrics["trainable_parameter_count"] = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    metrics["parameter_bytes"] = sum(
        parameter.numel() * parameter.element_size() for parameter in parameters
    )
    metrics["trainable_parameter_bytes"] = sum(
        parameter.numel() * parameter.element_size()
        for parameter in parameters
        if parameter.requires_grad
    )

    tensors = _state_tensors(output.final_state, set())
    for tensor in tensors:
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("persistent floating-point state must be finite")
    floating = sum(tensor.numel() for tensor in tensors if tensor.is_floating_point())
    nonfloating = sum(
        tensor.numel() for tensor in tensors if not tensor.is_floating_point()
    )
    metrics.update(
        {
            "persistent_state_tensor_count": len(tensors),
            "persistent_state_floating_elements": floating,
            "persistent_state_nonfloating_elements": nonfloating,
            "persistent_state_tensor_elements": floating + nonfloating,
            "persistent_state_bytes": sum(
                tensor.numel() * tensor.element_size() for tensor in tensors
            ),
        }
    )
    return metrics


def train_baseline_step(
    model: BaselineCampaignModel,
    batch: BindingBatch,
    optimizer: torch.optim.Optimizer,
    *,
    max_gradient_norm: float | None = 1.0,
) -> tuple[BaselineBindingOutput[object], BaselineQueryLoss]:
    """Run one deterministic autonomous query-training step.

    A batch containing no valid queries is an explicit optimization no-op:
    backward and ``optimizer.step`` are both skipped, so decoupled weight decay
    and optimizer counters cannot silently change model or optimizer state.
    """

    model = _validate_model(model)
    if not isinstance(batch, BindingBatch):
        raise TypeError("batch must be BindingBatch")
    max_gradient_norm = _validate_training_options(
        optimizer, model, max_gradient_norm
    )
    _validate_model_values(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    output = _forward_baseline(model, batch.inputs)
    # Predictive targets are read only after the autonomous forward has ended.
    loss = compute_baseline_query_loss(
        output, batch.inputs, batch.evaluation.targets
    )
    if loss.query_count == 0:
        return output, loss
    if not loss.total.requires_grad:
        raise ValueError("nonempty query loss must retain a gradient path")

    loss.total.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise ValueError("nonempty query loss produced no parameter gradients")
    if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise ValueError("baseline gradients must be finite")
    if max_gradient_norm is not None:
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_gradient_norm,
            error_if_nonfinite=True,
        )
        if not bool(torch.isfinite(norm)):
            raise ValueError("baseline gradient norm must be finite")
    optimizer.step()
    _validate_model_values(model)
    return output, loss


@torch.no_grad()
def evaluate_baseline_model(
    model: BaselineCampaignModel,
    batch: BindingBatch,
) -> tuple[
    BaselineBindingOutput[object], BaselineCampaignEvaluationSummary
]:
    """Evaluate one baseline without exposing evaluation fields to forward."""

    model = _validate_model(model)
    if not isinstance(batch, BindingBatch):
        raise TypeError("batch must be BindingBatch")
    _validate_model_values(model)
    model.eval()
    output = _forward_baseline(model, batch.inputs)

    # These are intentionally accessed after forward. Targets are predictive
    # labels; the held-out mask is used only to stratify reported metrics.
    targets = batch.evaluation.targets
    heldout_mask = batch.evaluation.heldout_combination_mask
    _validate_targets(output.value_logits, targets)
    if (
        not isinstance(heldout_mask, Tensor)
        or heldout_mask.shape != batch.inputs.valid_mask.shape
    ):
        raise ValueError("heldout_combination_mask must have shape [N,T]")
    if heldout_mask.dtype != torch.bool:
        raise TypeError("heldout_combination_mask must use torch.bool")
    if heldout_mask.device != output.value_logits.device:
        raise ValueError("heldout_combination_mask and value_logits must share a device")

    query_mask = _query_mask(batch.inputs)
    seen_mask = query_mask & ~heldout_mask
    heldout_query_mask = query_mask & heldout_mask
    summary = BaselineCampaignEvaluationSummary(
        query=_query_metrics(output.value_logits, targets, query_mask),
        seen_query=_query_metrics(output.value_logits, targets, seen_mask),
        heldout_query=_query_metrics(
            output.value_logits, targets, heldout_query_mask
        ),
        structural_metrics=baseline_structural_metrics(model, batch.inputs, output),
    )
    if (
        summary.query.query_count
        != summary.seen_query.query_count + summary.heldout_query.query_count
    ):
        raise RuntimeError("seen and held-out query strata do not partition queries")
    return output, summary


__all__ = [
    "BaselineCampaignEvaluationSummary",
    "BaselineCampaignModel",
    "BaselineQueryLoss",
    "BaselineQueryMetrics",
    "baseline_structural_metrics",
    "compute_baseline_query_loss",
    "evaluate_baseline_model",
    "train_baseline_step",
]
