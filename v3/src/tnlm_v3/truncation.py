"""Deterministic CP-rank selection and dense selected-model references.

Milestone 3 separates *selection* from physical compact export.  This module
scores the rank axis of :class:`~tnlm_v3.operators.ScaleSharedCPMerge`, records
an immutable selection, and builds a dense reference in which discarded CP
channels are exactly zero.  The dense reference is the parity oracle for a
later physically sliced model; it is not itself a compact export.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import struct
import sys

import torch
from torch import Tensor

from .binding import RoutedBindingModel
from .operators import ScaleSharedCPMerge


_SELECTION_SCHEMA_VERSION = 1
_PARAMETER_ENERGY_METHOD = "parameter_energy_v1"
_FINGERPRINT_DOMAIN = b"tnlm_v3.routed_binding_model_state.v1\x00"
_ROUTER_RUNTIME_DEFAULTS = {
    "prototype_update_rate": 0.25,
    "global_update_rate": 0.1,
    "temperature": 1.0,
}
_HEX_DIGITS = frozenset("0123456789abcdef")
_CP_CHANNEL_TENSORS = (
    "left.weight",
    "right.weight",
    "scale_to_rank.weight",
    "global_rank",
    "output.weight",
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class CPRankSelection:
    """Strict, immutable record of one CP-channel selection.

    ``channel_scores`` are stored in original channel-index order, while
    ``retained_indices`` are stored in ascending order.  Version 1 records a
    genuine truncation, so the exported rank is positive and strictly below
    the nominal rank.
    """

    schema_version: int
    source_model_fingerprint: str
    method: str
    nominal_rank: int
    exported_rank: int
    retained_indices: tuple[int, ...]
    channel_scores: tuple[float, ...]
    calibration_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != _SELECTION_SCHEMA_VERSION
        ):
            raise ValueError("schema_version must be integer 1")
        if not _is_sha256(self.source_model_fingerprint):
            raise ValueError("source_model_fingerprint must be lowercase SHA-256")
        if self.method != _PARAMETER_ENERGY_METHOD:
            raise ValueError(f"method must be {_PARAMETER_ENERGY_METHOD!r}")
        for name in ("nominal_rank", "exported_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.nominal_rank <= 1:
            raise ValueError("nominal_rank must exceed one")
        if not 1 <= self.exported_rank < self.nominal_rank:
            raise ValueError("exported_rank must satisfy 1 <= K < nominal_rank")

        if not isinstance(self.retained_indices, tuple):
            raise TypeError("retained_indices must be a tuple")
        if len(self.retained_indices) != self.exported_rank:
            raise ValueError("retained_indices length must equal exported_rank")
        for index in self.retained_indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("retained indices must be integers")
        if tuple(sorted(set(self.retained_indices))) != self.retained_indices:
            raise ValueError("retained_indices must be sorted and unique")
        if any(not 0 <= index < self.nominal_rank for index in self.retained_indices):
            raise ValueError("retained index is outside the nominal rank")

        if not isinstance(self.channel_scores, tuple):
            raise TypeError("channel_scores must be a tuple")
        if len(self.channel_scores) != self.nominal_rank:
            raise ValueError("channel_scores length must equal nominal_rank")
        normalized_scores: list[float] = []
        for score in self.channel_scores:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError("channel scores must be real numbers")
            value = float(score)
            if not math.isfinite(value) or value < 0:
                raise ValueError("channel scores must be finite and nonnegative")
            normalized_scores.append(value)
        object.__setattr__(self, "channel_scores", tuple(normalized_scores))

        if self.calibration_fingerprint is not None and not _is_sha256(
            self.calibration_fingerprint
        ):
            raise ValueError("calibration_fingerprint must be lowercase SHA-256")

    def canonical_json(self) -> str:
        """Return the strict canonical JSON representation."""

        return _canonical_json(asdict(self))

    def fingerprint(self) -> str:
        """Return a deterministic digest of this selection record."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _length_prefixed(value: bytes) -> bytes:
    return struct.pack("<Q", len(value)) + value


def _canonical_tensor_bytes(tensor: Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    # Flatten first because PyTorch cannot reinterpret a rank-zero tensor as
    # bytes directly.  Named state may legitimately contain scalar buffers.
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    if sys.byteorder == "little" or value.element_size() == 1:
        return raw
    width = value.element_size()
    return b"".join(
        raw[index : index + width][::-1]
        for index in range(0, len(raw), width)
    )


def model_state_fingerprint(model: RoutedBindingModel) -> str:
    """Hash executable configuration plus all named parameters and buffers.

    State entries are ordered lexicographically and include name, dtype, shape,
    and canonical raw bytes.  The digest is therefore independent of device
    and state-dict insertion order while remaining sensitive to every bit of
    model state and to configuration-only behavior.
    """

    if not isinstance(model, RoutedBindingModel):
        raise TypeError("model must be a RoutedBindingModel")
    _validate_model_architecture(model)
    digest = hashlib.sha256()
    digest.update(_FINGERPRINT_DOMAIN)
    config = model.config.canonical_json().encode("utf-8")
    digest.update(_length_prefixed(config))
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, Tensor) or tensor.layout is not torch.strided:
            raise TypeError("model state must contain only strided tensors")
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        shape = tuple(int(dimension) for dimension in tensor.shape)
        raw = _canonical_tensor_bytes(tensor)
        digest.update(_length_prefixed(name_bytes))
        digest.update(_length_prefixed(dtype_bytes))
        digest.update(struct.pack("<Q", len(shape)))
        for dimension in shape:
            digest.update(struct.pack("<Q", dimension))
        digest.update(_length_prefixed(raw))
    return digest.hexdigest()


def _checked_merge(model: RoutedBindingModel) -> ScaleSharedCPMerge:
    if not isinstance(model, RoutedBindingModel):
        raise TypeError("model must be a RoutedBindingModel")
    merge = model.forest.merge
    if type(merge) is not ScaleSharedCPMerge:
        raise TypeError("model forest must use exactly ScaleSharedCPMerge")
    rank = model.config.cp_rank
    expected_shapes = {
        "left.weight": (rank, model.config.d_model),
        "right.weight": (rank, model.config.d_model),
        "scale_to_rank.weight": (rank, model.config.scale_feature_dim),
        "global_rank": (rank,),
        "output.weight": (model.config.d_model, rank),
    }
    state = merge.state_dict()
    if set(expected_shapes) - set(state):
        raise ValueError("merge is missing CP-channel tensors")
    for name, shape in expected_shapes.items():
        if tuple(state[name].shape) != shape:
            raise ValueError(f"unexpected shape for CP tensor {name}")
        if not bool(torch.isfinite(state[name]).all()):
            raise ValueError(f"CP tensor {name} must be finite")
    return merge


def _validate_model_architecture(model: RoutedBindingModel) -> None:
    """Reject live-module metadata that disagrees with the hashed config."""

    config = model.config
    task = config.task
    if (
        model.encoder.task != task
        or model.encoder.d_model != config.d_model
        or model.router.feature_dim != config.d_model
        or model.router.branches != task.branches
        or model.router.hidden_dim != config.router_hidden_dim
        or model.router.mode is not config.routing_mode
        or model.router.curriculum_schedule != config.curriculum_schedule
        or model.router.curriculum_seed != config.curriculum_seed
        or not model.router.include_global
        or not model.router.include_null
        or any(
            getattr(model.router, name) != expected
            for name, expected in _ROUTER_RUNTIME_DEFAULTS.items()
        )
        or model.forest.d_model != config.d_model
        or model.forest.branches != task.branches
        or model.forest.paths != task.branches + 1
        or model.forest.cp_rank != config.cp_rank
        or model.forest.scale_feature_dim != config.scale_feature_dim
        or model.readout.d_model != config.d_model
        or model.readout.branches != task.branches
        or model.readout.paths != task.branches + 1
        or model.readout.scale_feature_dim != config.scale_feature_dim
    ):
        raise ValueError("live model architecture does not match its configuration")
    merge = _checked_merge(model)
    if (
        merge.d_model != config.d_model
        or merge.cp_rank != config.cp_rank
        or merge.scale_feature_dim != config.scale_feature_dim
    ):
        raise ValueError("live merge architecture does not match its configuration")


def _parameter_energy_scores(merge: ScaleSharedCPMerge) -> tuple[float, ...]:
    """Return an augmented parameter-energy heuristic for each CP channel.

    For channel ``r`` the score is the squared Frobenius norm of the augmented
    pre-nonlinearity CP outer product::

        output[:,r] ⊗ left[r,:] ⊗ right[r,:]
            ⊗ [1, scale_to_rank[r,:], global_rank[r]]

    The score is the product of the four factors' squared L2 norms. Because
    the executable merge applies nonlinearities and bounded additive
    conditioning, this is only a deterministic structural ranking heuristic;
    it is neither exact functional energy nor data-dependent attribution.
    """

    left = merge.left.weight.detach().to(device="cpu", dtype=torch.float64)
    right = merge.right.weight.detach().to(device="cpu", dtype=torch.float64)
    scale = merge.scale_to_rank.weight.detach().to(
        device="cpu", dtype=torch.float64
    )
    global_rank = merge.global_rank.detach().to(device="cpu", dtype=torch.float64)
    output = merge.output.weight.detach().to(device="cpu", dtype=torch.float64)
    scores = (
        left.square().sum(dim=1)
        * right.square().sum(dim=1)
        * output.square().sum(dim=0)
        * (1.0 + scale.square().sum(dim=1) + global_rank.square())
    )
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("CP parameter-energy scores must be finite")
    return tuple(float(value) for value in scores.tolist())


def select_cp_rank_by_parameter_energy(
    model: RoutedBindingModel,
    *,
    target_rank: int,
    calibration_fingerprint: str | None = None,
) -> CPRankSelection:
    """Select the highest-energy CP channels with a stable index tie-break.

    Channels are ranked by decreasing augmented parameter energy; equal scores
    prefer the lower original channel index.  The returned retained indices
    are then sorted into original order for deterministic tensor slicing.
    ``target_rank`` must request a real reduction, ``1 <= K < R``.
    """

    merge = _checked_merge(model)
    nominal_rank = merge.cp_rank
    if isinstance(target_rank, bool) or not isinstance(target_rank, int):
        raise TypeError("target_rank must be an integer")
    if not 1 <= target_rank < nominal_rank:
        raise ValueError("target_rank must satisfy 1 <= K < nominal rank")
    if calibration_fingerprint is not None and not _is_sha256(
        calibration_fingerprint
    ):
        raise ValueError("calibration_fingerprint must be lowercase SHA-256")

    scores = _parameter_energy_scores(merge)
    ranked = sorted(range(nominal_rank), key=lambda index: (-scores[index], index))
    retained = tuple(sorted(ranked[:target_rank]))
    return CPRankSelection(
        schema_version=_SELECTION_SCHEMA_VERSION,
        source_model_fingerprint=model_state_fingerprint(model),
        method=_PARAMETER_ENERGY_METHOD,
        nominal_rank=nominal_rank,
        exported_rank=target_rank,
        retained_indices=retained,
        channel_scores=scores,
        calibration_fingerprint=calibration_fingerprint,
    )


def build_dense_selected_reference(
    model: RoutedBindingModel,
    selection: CPRankSelection,
) -> RoutedBindingModel:
    """Deep-copy ``model`` and zero exactly its discarded CP-rank channels.

    This never mutates the source.  Only the five tensors carrying the CP rank
    axis are changed: left/right/scale-conditioning rows, ``global_rank``
    entries, and output columns.  All router, encoder, residual, gate,
    normalization, readout, and non-rank scale tensors remain byte-identical.
    """

    merge = _checked_merge(model)
    if not isinstance(selection, CPRankSelection):
        raise TypeError("selection must be a CPRankSelection")
    fingerprint = model_state_fingerprint(model)
    if fingerprint != selection.source_model_fingerprint:
        raise ValueError("selection source model fingerprint mismatch")
    if selection.nominal_rank != merge.cp_rank:
        raise ValueError("selection nominal rank does not match model")
    expected_scores = _parameter_energy_scores(merge)
    if selection.channel_scores != expected_scores:
        raise ValueError("selection channel scores do not match the source model")
    ranked = sorted(
        range(merge.cp_rank),
        key=lambda index: (-expected_scores[index], index),
    )
    expected_indices = tuple(sorted(ranked[: selection.exported_rank]))
    if selection.retained_indices != expected_indices:
        raise ValueError("selection retained indices do not match its declared method")

    reference = copy.deepcopy(model)
    reference_merge = _checked_merge(reference)
    retained = set(selection.retained_indices)
    discarded = [index for index in range(merge.cp_rank) if index not in retained]
    indices = torch.tensor(
        discarded, device=reference_merge.global_rank.device, dtype=torch.int64
    )
    with torch.no_grad():
        reference_merge.left.weight.index_fill_(0, indices, 0)
        reference_merge.right.weight.index_fill_(0, indices, 0)
        reference_merge.scale_to_rank.weight.index_fill_(0, indices, 0)
        reference_merge.global_rank.index_fill_(0, indices, 0)
        reference_merge.output.weight.index_fill_(1, indices, 0)
    return reference


__all__ = [
    "CPRankSelection",
    "build_dense_selected_reference",
    "model_state_fingerprint",
    "select_cp_rank_by_parameter_energy",
]
