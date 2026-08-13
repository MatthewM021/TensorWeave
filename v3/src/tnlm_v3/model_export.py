"""Physical compact-model export for CP-rank-selected binding models.

This module deliberately treats CP-rank compaction separately from state-width
compaction.  Exporting fewer CP channels reduces learned merge tensors and the
declared merge-work proxy, but it does not reduce ``d_model``, the number of
routing paths, or the number of scalars in one occupied forest slot.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, replace
import hashlib
import json
import re
from typing import Mapping

import torch
from torch import nn

from .binding import BindingModelConfig, RoutedBindingModel
from .operators import ScaleSharedCPMerge, slice_cp_merge
from .truncation import CPRankSelection, model_state_fingerprint


_MANIFEST_SCHEMA_VERSION = 1
_MANIFEST_FINGERPRINT_DOMAIN = b"tnlm_v3.compact_export_manifest.v1\x00"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CP_STATE_SUFFIXES = frozenset(
    {
        "left.weight",
        "right.weight",
        "scale_to_rank.weight",
        "global_rank",
        "output.weight",
    }
)


def _strict_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _strict_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class CompactExportManifest:
    """Strict, JSON-safe proof of one physical CP-rank export.

    ``state_interface_unchanged`` is intentionally required to be true.  This
    exporter compacts only the CP interaction axis; a future state-channel
    exporter needs a distinct schema rather than overloading these fields.
    """

    schema_version: int
    source_model_fingerprint: str
    exported_model_fingerprint: str
    source_config_fingerprint: str
    exported_config_fingerprint: str
    selection_fingerprint: str
    nominal_cp_rank: int
    exported_cp_rank: int
    retained_indices: tuple[int, ...]
    source_parameter_count: int
    exported_parameter_count: int
    source_raw_tensor_bytes: int
    exported_raw_tensor_bytes: int
    source_state_d_model: int
    exported_state_d_model: int
    source_state_paths: int
    exported_state_paths: int
    source_state_scalars_per_slot: int
    exported_state_scalars_per_slot: int
    state_interface_unchanged: bool
    source_operation_count_proxy_per_merge: int
    exported_operation_count_proxy_per_merge: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != _MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError(f"schema_version must equal {_MANIFEST_SCHEMA_VERSION}")
        for name in (
            "source_model_fingerprint",
            "exported_model_fingerprint",
            "source_config_fingerprint",
            "exported_config_fingerprint",
            "selection_fingerprint",
        ):
            _strict_sha256(getattr(self, name), name)

        nominal = _strict_positive_int(self.nominal_cp_rank, "nominal_cp_rank")
        exported = _strict_positive_int(self.exported_cp_rank, "exported_cp_rank")
        if exported >= nominal:
            raise ValueError("exported_cp_rank must be smaller than nominal_cp_rank")
        if type(self.retained_indices) is not tuple:
            raise TypeError("retained_indices must be a tuple")
        if len(self.retained_indices) != exported:
            raise ValueError("retained_indices length must equal exported_cp_rank")
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in self.retained_indices
        ):
            raise TypeError("retained_indices must contain only integers")
        if tuple(sorted(set(self.retained_indices))) != self.retained_indices:
            raise ValueError("retained_indices must be sorted and unique")
        if self.retained_indices[0] < 0 or self.retained_indices[-1] >= nominal:
            raise ValueError("retained_indices are outside nominal_cp_rank")

        decreasing_pairs = (
            (
                "parameter_count",
                self.source_parameter_count,
                self.exported_parameter_count,
            ),
            (
                "raw_tensor_bytes",
                self.source_raw_tensor_bytes,
                self.exported_raw_tensor_bytes,
            ),
            (
                "operation_count_proxy_per_merge",
                self.source_operation_count_proxy_per_merge,
                self.exported_operation_count_proxy_per_merge,
            ),
        )
        for name, source, compact in decreasing_pairs:
            _strict_positive_int(source, f"source_{name}")
            _strict_positive_int(compact, f"exported_{name}")
            if compact >= source:
                raise ValueError(f"exported_{name} must be smaller than source_{name}")

        state_fields = (
            "source_state_d_model",
            "exported_state_d_model",
            "source_state_paths",
            "exported_state_paths",
            "source_state_scalars_per_slot",
            "exported_state_scalars_per_slot",
        )
        for name in state_fields:
            _strict_positive_int(getattr(self, name), name)
        if type(self.state_interface_unchanged) is not bool:
            raise TypeError("state_interface_unchanged must be boolean")
        interface_equal = (
            self.source_state_d_model == self.exported_state_d_model
            and self.source_state_paths == self.exported_state_paths
            and self.source_state_scalars_per_slot
            == self.exported_state_scalars_per_slot
            and self.source_state_scalars_per_slot == self.source_state_d_model
            and self.exported_state_scalars_per_slot == self.exported_state_d_model
        )
        if not self.state_interface_unchanged or not interface_equal:
            raise ValueError(
                "CP-rank export must record an unchanged forest-state interface"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-domain representation for schema version 1."""

        return {
            field.name: list(value) if field.name == "retained_indices" else value
            for field in fields(self)
            for value in (getattr(self, field.name),)
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CompactExportManifest":
        """Parse a manifest while rejecting missing, extra, or mistyped fields."""

        if not isinstance(value, Mapping):
            raise TypeError("manifest must be a mapping")
        expected = {field.name for field in fields(cls)}
        if set(value) != expected or any(not isinstance(key, str) for key in value):
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected, key=str)
            raise ValueError(
                f"invalid compact-export manifest fields; missing={missing}, extra={extra}"
            )
        indices = value["retained_indices"]
        if type(indices) is not list:
            raise TypeError("retained_indices must be a JSON array")
        payload = dict(value)
        payload["retained_indices"] = tuple(indices)
        return cls(**payload)  # type: ignore[arg-type]

    def canonical_json(self) -> str:
        """Return stable, finite, compact JSON suitable for artifact codecs."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def fingerprint(self) -> str:
        """Return a domain-separated SHA-256 of the canonical manifest JSON."""

        digest = hashlib.sha256()
        digest.update(_MANIFEST_FINGERPRINT_DOMAIN)
        digest.update(self.canonical_json().encode("utf-8"))
        return digest.hexdigest()


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _raw_tensor_bytes(model: nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
    )


def _operation_proxy_per_merge(merge: ScaleSharedCPMerge) -> int:
    metrics = merge.structural_metrics(merge_count=1)
    value = metrics["operation_count_proxy_per_merge"]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError("merge returned an invalid operation-count proxy")
    return value


def _validate_selection(
    source: RoutedBindingModel,
    selection: CPRankSelection,
    source_fingerprint: str,
) -> tuple[int, ...]:
    if not isinstance(selection, CPRankSelection):
        raise TypeError("selection must be a CPRankSelection")
    if selection.source_model_fingerprint != source_fingerprint:
        raise ValueError("selection source_model_fingerprint does not match source")
    nominal = source.config.cp_rank
    merge = source.forest.merge
    if type(merge) is not ScaleSharedCPMerge:
        raise TypeError("source forest merge must be exactly ScaleSharedCPMerge")
    if source.forest.cp_rank != nominal or merge.cp_rank != nominal:
        raise ValueError("source config, forest, and merge CP ranks disagree")
    if selection.nominal_rank != nominal:
        raise ValueError("selection nominal_rank does not match source CP rank")
    if selection.exported_rank != len(selection.retained_indices):
        raise ValueError("selection exported_rank does not match retained_indices")
    return tuple(selection.retained_indices)


def _assert_non_merge_state_copy(
    source: RoutedBindingModel,
    compact: RoutedBindingModel,
) -> None:
    source_state = source.state_dict()
    compact_state = compact.state_dict()
    if set(source_state) != set(compact_state):
        raise RuntimeError("compact model changed the model-state key set")
    prefix = "forest.merge."
    for name, source_tensor in source_state.items():
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            if suffix not in _CP_STATE_SUFFIXES:
                if not torch.equal(source_tensor, compact_state[name]):
                    raise RuntimeError(f"non-CP merge tensor changed during export: {name}")
            continue
        if not torch.equal(source_tensor, compact_state[name]):
            raise RuntimeError(f"non-merge state tensor changed during export: {name}")


def export_compact_binding_model(
    source: RoutedBindingModel,
    selection: CPRankSelection,
) -> tuple[RoutedBindingModel, CompactExportManifest]:
    """Build an independent binding model with physically sliced CP tensors.

    The source is fingerprinted before and after export.  Every state tensor
    outside the five CP-axis tensors is copied exactly; no mask, original-rank
    buffer, or padded dense CP tensor is attached to the result.
    """

    if not isinstance(source, RoutedBindingModel):
        raise TypeError("source must be a RoutedBindingModel")
    if not isinstance(source.config, BindingModelConfig):
        raise TypeError("source must carry a BindingModelConfig")

    source_fingerprint = model_state_fingerprint(source)
    retained_indices = _validate_selection(source, selection, source_fingerprint)
    compact_merge = slice_cp_merge(source.forest.merge, retained_indices)

    # Deep-copying preserves mixed dtype/device, requires-grad, and per-module
    # train/eval flags exactly. ``slice_cp_merge`` preserves RNG state while
    # constructing its physically smaller module.
    compact = copy.deepcopy(source)
    compact_config = replace(
        copy.deepcopy(source.config), cp_rank=selection.exported_rank
    )
    compact.config = compact_config
    compact.encoder.task = compact_config.task
    compact.forest.cp_rank = selection.exported_rank
    compact.forest.merge = compact_merge

    _assert_non_merge_state_copy(source, compact)
    if model_state_fingerprint(source) != source_fingerprint:
        raise RuntimeError("source model changed during compact export")

    source_parameters = _parameter_count(source)
    compact_parameters = _parameter_count(compact)
    source_bytes = _raw_tensor_bytes(source)
    compact_bytes = _raw_tensor_bytes(compact)
    source_operation_proxy = _operation_proxy_per_merge(source.forest.merge)
    compact_operation_proxy = _operation_proxy_per_merge(compact.forest.merge)

    source_d_model = source.config.d_model
    compact_d_model = compact.config.d_model
    source_paths = source.forest.paths
    compact_paths = compact.forest.paths
    state_interface_unchanged = (
        source_d_model == compact_d_model and source_paths == compact_paths
    )

    manifest = CompactExportManifest(
        schema_version=_MANIFEST_SCHEMA_VERSION,
        source_model_fingerprint=source_fingerprint,
        exported_model_fingerprint=model_state_fingerprint(compact),
        source_config_fingerprint=source.config.fingerprint(),
        exported_config_fingerprint=compact.config.fingerprint(),
        selection_fingerprint=selection.fingerprint(),
        nominal_cp_rank=source.config.cp_rank,
        exported_cp_rank=compact.config.cp_rank,
        retained_indices=retained_indices,
        source_parameter_count=source_parameters,
        exported_parameter_count=compact_parameters,
        source_raw_tensor_bytes=source_bytes,
        exported_raw_tensor_bytes=compact_bytes,
        source_state_d_model=source_d_model,
        exported_state_d_model=compact_d_model,
        source_state_paths=source_paths,
        exported_state_paths=compact_paths,
        source_state_scalars_per_slot=source_d_model,
        exported_state_scalars_per_slot=compact_d_model,
        state_interface_unchanged=state_interface_unchanged,
        source_operation_count_proxy_per_merge=source_operation_proxy,
        exported_operation_count_proxy_per_merge=compact_operation_proxy,
    )
    return compact, manifest


__all__ = ["CompactExportManifest", "export_compact_binding_model"]
