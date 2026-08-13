from __future__ import annotations

import math
from typing import Union

import torch
from torch import Tensor, nn


ScaleLike = Union[int, float, Tensor]
PathLike = Union[bool, Tensor]


def analytic_scale_features(scales: Tensor, feature_dim: int) -> Tensor:
    """Return bounded, table-free features for non-negative binary scales.

    ``scales`` may have any shape. The result has the same shape followed by a
    feature axis of length ``feature_dim`` and preserves the input tensor's
    floating-point dtype and device. Using ``log1p`` before the sinusoidal map
    keeps features finite for very large, finite scale values.
    """

    if not isinstance(scales, Tensor):
        raise TypeError("scales must be a torch.Tensor")
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    if not scales.is_floating_point():
        raise TypeError("scales must have a floating-point dtype")
    if not bool(torch.isfinite(scales).all()):
        raise ValueError("scales must be finite")
    if bool((scales < 0).any()):
        raise ValueError("scales must be non-negative")

    log_scales = torch.log1p(scales)
    frequency_count = (feature_dim + 1) // 2
    frequency_index = torch.arange(
        frequency_count, dtype=scales.dtype, device=scales.device
    )
    if frequency_count == 1:
        frequencies = torch.ones_like(frequency_index)
    else:
        frequencies = torch.exp(
            frequency_index * (-math.log(10_000.0) / (frequency_count - 1))
        )
    angles = log_scales.unsqueeze(-1) * frequencies
    features = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
    return features.flatten(start_dim=-2)[..., :feature_dim]


def _broadcast_condition(
    value: ScaleLike | PathLike,
    target_shape: torch.Size,
    *,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> Tensor:
    condition = torch.as_tensor(value, dtype=dtype, device=device)
    try:
        return torch.broadcast_to(condition, target_shape)
    except RuntimeError as error:
        raise ValueError(
            f"{name} with shape {tuple(condition.shape)} cannot broadcast to "
            f"input shape {tuple(target_shape)}"
        ) from error


class ScaleSharedCPMerge(nn.Module):
    """Chronological CP merge shared by every binary-counter scale.

    The only scale dependence comes from fixed-size analytic features. There
    is no level-indexed parameter table, maximum length, or stochastic layer.
    ``left`` is the older dyadic block and ``right`` is the newer block, so the
    separately parameterized projections intentionally make the operation
    non-commutative.
    """

    def __init__(
        self,
        d_model: int,
        cp_rank: int,
        scale_feature_dim: int = 8,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if cp_rank <= 0:
            raise ValueError("cp_rank must be positive")
        if scale_feature_dim <= 0:
            raise ValueError("scale_feature_dim must be positive")

        self.d_model = int(d_model)
        self.cp_rank = int(cp_rank)
        self.scale_feature_dim = int(scale_feature_dim)

        self.left = nn.Linear(self.d_model, self.cp_rank, bias=False)
        self.right = nn.Linear(self.d_model, self.cp_rank, bias=False)
        self.scale_to_rank = nn.Linear(
            self.scale_feature_dim, self.cp_rank, bias=False
        )
        self.global_rank = nn.Parameter(torch.zeros(self.cp_rank))
        self.output = nn.Linear(self.cp_rank, self.d_model, bias=False)

        self.left_residual = nn.Linear(self.d_model, self.d_model, bias=False)
        self.right_residual = nn.Linear(self.d_model, self.d_model, bias=False)
        self.scale_to_state = nn.Linear(
            self.scale_feature_dim, self.d_model, bias=False
        )
        self.global_state = nn.Parameter(torch.zeros(self.d_model))
        self.gate = nn.Linear(self.d_model, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)

        nn.init.xavier_uniform_(self.left.weight)
        nn.init.xavier_uniform_(self.right.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.normal_(self.scale_to_rank.weight, std=0.02)
        nn.init.normal_(self.scale_to_state.weight, std=0.02)
        nn.init.eye_(self.left_residual.weight)
        nn.init.eye_(self.right_residual.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -0.5)

    def forward(
        self,
        left: Tensor,
        right: Tensor,
        scale: ScaleLike,
        global_path: PathLike = False,
    ) -> Tensor:
        if left.shape != right.shape:
            raise ValueError("left and right must have identical shapes")
        if left.ndim == 0 or left.shape[-1] != self.d_model:
            raise ValueError(
                f"left and right must end in d_model={self.d_model}; "
                f"got {tuple(left.shape)}"
            )
        if left.dtype != right.dtype or left.device != right.device:
            raise ValueError("left and right must share dtype and device")
        if not left.is_floating_point():
            raise TypeError("left and right must have floating-point dtype")

        leading_shape = left.shape[:-1]
        scales = _broadcast_condition(
            scale,
            leading_shape,
            dtype=left.dtype,
            device=left.device,
            name="scale",
        )
        if not bool(torch.isfinite(scales).all()) or bool((scales < 0).any()):
            raise ValueError("scale must contain finite, non-negative values")
        global_indicator = _broadcast_condition(
            global_path,
            leading_shape,
            dtype=left.dtype,
            device=left.device,
            name="global_path",
        )
        if not bool(torch.isfinite(global_indicator).all()):
            raise ValueError("global_path must be finite")

        scale_features = analytic_scale_features(scales, self.scale_feature_dim)
        global_indicator = global_indicator.unsqueeze(-1)

        rank_condition = 1.0 + torch.tanh(
            self.scale_to_rank(scale_features)
            + global_indicator * self.global_rank
        )
        interaction = (
            torch.tanh(self.left(left))
            * torch.tanh(self.right(right))
            * rank_condition
        )
        interaction = self.output(interaction)

        residual = 0.5 * (
            self.left_residual(left) + self.right_residual(right)
        )
        gate_context = (
            0.5 * (left + right)
            + self.scale_to_state(scale_features)
            + global_indicator * self.global_state
        )
        gate = torch.sigmoid(self.gate(gate_context))
        return self.norm(residual + gate * interaction)

    def structural_metrics(self, merge_count: int = 0) -> dict[str, int]:
        """Expose rank, size, and a declared scalar-operation proxy.

        Milestone 1 has no pruned channels, so nominal, effective, and
        checkpoint-exported ranks are identical.  The operation proxy counts
        dense scalar multiply/add/nonlinear elements in the learned merge; it
        is a stable architecture comparison metric rather than wall-clock FLOPs.
        """

        if isinstance(merge_count, bool) or int(merge_count) != merge_count:
            raise TypeError("merge_count must be an integer")
        merge_count = int(merge_count)
        if merge_count < 0:
            raise ValueError("merge_count cannot be negative")
        d = self.d_model
        rank = self.cp_rank
        features = self.scale_feature_dim
        per_merge = (
            3 * d * rank
            + features * rank
            + 3 * d * d
            + features * d
            + 8 * d
            + 6 * rank
        )
        return {
            "nominal_rank": rank,
            "effective_rank": rank,
            "exported_rank": rank,
            "merge_parameter_count": sum(p.numel() for p in self.parameters()),
            "operation_count_proxy_per_merge": per_merge,
            "operation_count_proxy": per_merge * merge_count,
        }
