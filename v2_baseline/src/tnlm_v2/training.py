from __future__ import annotations

import copy
import dataclasses
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from .data import TaskBatch
from .models.components import ModelOutput, PredictiveModel


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.004
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    patience: int = 6
    min_delta: float = 0.0001
    seed: int = 7
    orthogonality_weight: float = 0.0001
    rank_weight: float = 0.002
    router_entropy_weight: float = 0.01
    router_balance_weight: float = 0.05
    router_temperature_start: float = 0.8
    router_temperature_end: float = 0.25
    router_hard_fraction: float = 0.35
    label_smoothing: float = 0.0
    num_threads: int = 4


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_primary_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    learning_rate: float
    seconds: float
    examples_per_second: float
    tokens_per_second: float
    aux_means: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingResult:
    history: List[EpochRecord]
    best_epoch: int
    best_validation_loss: float
    best_validation_accuracy: float
    total_seconds: float
    parameter_count: int
    trainable_parameter_count: int

    def as_dict(self):
        return {
            "history": [dataclasses.asdict(x) for x in self.history],
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "best_validation_accuracy": self.best_validation_accuracy,
            "total_seconds": self.total_seconds,
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
        }


def set_reproducible_seed(seed: int, num_threads: int = 4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(num_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _iter_minibatches(batch, batch_size, shuffle, generator):
    indices = (
        torch.randperm(len(batch), generator=generator)
        if shuffle
        else torch.arange(len(batch))
    )
    for start in range(0, len(batch), batch_size):
        yield batch.slice(indices[start : start + batch_size])


def _auxiliary_loss(output, config):
    weights = {
        "orthogonality": config.orthogonality_weight,
        "rank_regularizer": config.rank_weight,
        "router_entropy": config.router_entropy_weight,
        "router_balance": config.router_balance_weight,
    }
    total = torch.zeros((), device=output.logits.device)
    detached = {}
    for name, value in output.aux_losses.items():
        weight = weights.get(name, 0.0)
        if weight:
            total = total + weight * value
            detached[name] = value.detach()
    return total, detached


def _configure_router(model, progress, config):
    setter = getattr(model, "set_router_temperature", None)
    if setter is None:
        return
    start, end = config.router_temperature_start, config.router_temperature_end
    temperature = start * ((end / start) ** min(1.0, max(0.0, progress)))
    setter(temperature, hard=progress >= config.router_hard_fraction)


def permutation_aligned_route_metrics(predicted, truth):
    valid = truth >= 0
    if not bool(valid.any()):
        return {
            "route_accuracy_aligned": float("nan"),
            "route_purity": float("nan"),
            "route_used_branches": 0.0,
        }
    p = predicted[valid].long().numpy()
    t = truth[valid].long().numpy()
    size = max(int(p.max()) + 1, int(t.max()) + 1)
    confusion = np.zeros((size, size), np.int64)
    np.add.at(confusion, (p, t), 1)
    rows, cols = linear_sum_assignment(-confusion)
    return {
        "route_accuracy_aligned": float(confusion[rows, cols].sum() / len(p)),
        "route_purity": float(confusion.max(axis=1).sum() / len(p)),
        "route_used_branches": float(np.unique(p).size),
    }


def evaluate_model(model, data, batch_size=256, device="cpu"):
    model.eval()
    device = torch.device(device)
    generator = torch.Generator().manual_seed(0)
    loss_sum = 0.0
    correct = 0
    count = 0
    route_pred, route_true = [], []
    diag_sum, diag_count = {}, {}
    with torch.inference_mode():
        for batch in _iter_minibatches(data, batch_size, False, generator):
            batch = batch.to(device)
            routes = batch.routes if "_oracle" in model.model_name else None
            output = model(batch.tokens, batch.valid_mask, routes)
            loss_sum += float(
                F.cross_entropy(output.logits, batch.labels, reduction="sum").cpu()
            )
            correct += int((output.logits.argmax(-1) == batch.labels).sum().cpu())
            count += len(batch)
            assignments = output.diagnostics.get("assignments")
            if assignments is not None and "_learned" in model.model_name:
                route_pred.append(assignments.argmax(-1).cpu())
                route_true.append(batch.routes.cpu())
            for key, value in output.diagnostics.items():
                if value.numel() == 1:
                    diag_sum[key] = diag_sum.get(key, 0.0) + float(value.cpu())
                    diag_count[key] = diag_count.get(key, 0) + 1
    result = {
        "loss": loss_sum / max(1, count),
        "accuracy": correct / max(1, count),
        "examples": float(count),
    }
    for key in diag_sum:
        result[key] = diag_sum[key] / diag_count[key]
    if route_pred:
        result.update(
            permutation_aligned_route_metrics(
                torch.cat(route_pred, 0), torch.cat(route_true, 0)
            )
        )
    return result


def train_model(model, train_data, validation_data, config, device="cpu"):
    set_reproducible_seed(config.seed, config.num_threads)
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs), eta_min=config.learning_rate * 0.08
    )
    generator = torch.Generator().manual_seed(config.seed + 101)
    best_state = copy.deepcopy(model.state_dict())
    best_loss, best_accuracy, best_epoch = float("inf"), 0.0, -1
    stale = 0
    history = []
    start_all = time.perf_counter()
    for epoch in range(config.epochs):
        start = time.perf_counter()
        progress = epoch / max(1, config.epochs - 1)
        _configure_router(model, progress, config)
        model.train()
        total, primary_total, correct, count = 0.0, 0.0, 0, 0
        aux_sum, aux_count = {}, {}
        for batch in _iter_minibatches(
            train_data, config.batch_size, True, generator
        ):
            batch = batch.to(device)
            routes = batch.routes if "_oracle" in model.model_name else None
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.tokens, batch.valid_mask, routes)
            primary = F.cross_entropy(
                output.logits,
                batch.labels,
                label_smoothing=config.label_smoothing,
            )
            aux, detached = _auxiliary_loss(output, config)
            loss = primary + aux
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            if config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            n = len(batch)
            total += float(loss.detach().cpu()) * n
            primary_total += float(primary.detach().cpu()) * n
            correct += int((output.logits.argmax(-1) == batch.labels).sum().cpu())
            count += n
            for k, v in detached.items():
                aux_sum[k] = aux_sum.get(k, 0.0) + float(v.cpu())
                aux_count[k] = aux_count.get(k, 0) + 1
        scheduler.step()
        _configure_router(model, 1.0, config)
        val = evaluate_model(model, validation_data, max(config.batch_size, 128), device)
        seconds = time.perf_counter() - start
        history.append(
            EpochRecord(
                epoch,
                total / count,
                primary_total / count,
                correct / count,
                val["loss"],
                val["accuracy"],
                optimizer.param_groups[0]["lr"],
                seconds,
                count / seconds,
                count * train_data.sequence_length / seconds,
                {k: aux_sum[k] / aux_count[k] for k in aux_sum},
            )
        )
        if val["loss"] < best_loss - config.min_delta:
            best_loss = val["loss"]
            best_accuracy = val["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    _configure_router(model, 1.0, config)
    return TrainingResult(
        history,
        best_epoch,
        best_loss,
        best_accuracy,
        time.perf_counter() - start_all,
        model.parameter_count,
        model.trainable_parameter_count,
    )


def benchmark_inference(
    model,
    data,
    batch_size=64,
    warmup_batches=2,
    measured_batches=5,
    device="cpu",
):
    model.eval()
    sample = data.slice(torch.arange(min(len(data), batch_size))).to(device)
    routes = sample.routes if "_oracle" in model.model_name else None
    with torch.inference_mode():
        for _ in range(max(1, warmup_batches)):
            model(sample.tokens, sample.valid_mask, routes)
        start = time.perf_counter()
        for _ in range(max(1, measured_batches)):
            model(sample.tokens, sample.valid_mask, routes)
        elapsed = time.perf_counter() - start
    passes = max(1, measured_batches)
    examples = len(sample) * passes
    return {
        "inference_seconds_per_batch": elapsed / passes,
        "inference_examples_per_second": examples / elapsed,
        "inference_tokens_per_second": examples * data.sequence_length / elapsed,
        "inference_batch_size": float(len(sample)),
    }
