"""Deterministic model and fixture construction for Milestone-4 campaigns.

This module translates an already validated :class:`ResolvedCampaignRun` into
the exact executable model, optimizer, and data streams used by a campaign
worker.  It does not perform filesystem or Git provenance checks; those remain
the runner's responsibility.

The selection set deliberately uses the binding generator's ``eval`` split.
That split forces the held-out combinations needed for validation-only model
selection without exposing the later confirmatory ``test`` stream.  Train,
validation, test, and scaling streams use separate domain-separated seed
derivations even when they originate from one paired seed record.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, fields, is_dataclass
from hashlib import sha256
import json
from typing import TypeAlias
import weakref

import torch
from torch import nn

from .baselines import (
    CachedCausalTransformerBindingBaseline,
    CachedTransformerBindingBaselineConfig,
    RecurrentBindingBaseline,
    RecurrentBindingBaselineConfig,
)
from .binding import BindingModelConfig, RoutedBindingModel
from .campaign_config import (
    CampaignDataSpec,
    CampaignModelSpec,
    CampaignPairSpec,
    CampaignPromotionSpec,
    CampaignQualitySpec,
    CampaignRuntimeSpec,
    CampaignSelectionSpec,
    CampaignStatisticsSpec,
    CampaignTrainingSpec,
    EvaluationDataSpec,
    ImplementationPolicy,
    Milestone4CampaignConfig,
    ResolvedCampaignRun,
    TrainDataSpec,
    resolve_campaign_plan,
)
from .causal_ttn import (
    CausalCompleteTreeBindingBaseline,
    CausalTreeBindingBaselineConfig,
)
from .data import (
    BindingBatch,
    BindingEvaluation,
    BindingModelInputs,
    BindingTaskConfig,
    collate_binding_episodes,
    generate_binding_episodes,
)
from .model_export import CompactExportManifest, export_compact_binding_model
from .routing import CurriculumSchedule, RoutingMode
from .truncation import CPRankSelection, select_cp_rank_by_parameter_energy
from .campaign import train_baseline_step
from .training import train_binding_step


CampaignModel: TypeAlias = (
    RoutedBindingModel
    | RecurrentBindingBaseline
    | CachedCausalTransformerBindingBaseline
    | CausalCompleteTreeBindingBaseline
)

_SOURCE_TYPES = {
    "routed": RoutedBindingModel,
    "gru": RecurrentBindingBaseline,
    "cached_transformer": CachedCausalTransformerBindingBaseline,
    "causal_ttn": CausalCompleteTreeBindingBaseline,
}
_SEED_DOMAIN = "tnlm-v3-milestone4-data-stream-v1"
_ADAMW_STEP_CORE = getattr(torch.optim.AdamW.step, "__wrapped__", torch.optim.AdamW.step)
_OPTIMIZER_ZERO_GRAD = torch.optim.Optimizer.zero_grad
_MODEL_BINDINGS: weakref.WeakKeyDictionary[nn.Module, tuple[str, int]] = (
    weakref.WeakKeyDictionary()
)


def _is_exact_adamw_step(value: object) -> bool:
    """Accept only PyTorch's core AdamW step plus its lazy profiling wrapper."""

    if value is _ADAMW_STEP_CORE:
        return True
    inner = getattr(value, "__wrapped__", None)
    if inner is _ADAMW_STEP_CORE:
        return True
    return getattr(inner, "__wrapped__", None) is _ADAMW_STEP_CORE


def _rebuild_exact(value: object, expected_type: type) -> object:
    if type(value) is not expected_type:
        raise TypeError(f"campaign member must have exact type {expected_type.__name__}")
    return expected_type(**asdict(value))


def _deep_rebuild_config(config: Milestone4CampaignConfig) -> Milestone4CampaignConfig:
    """Reconstruct every nested value to defeat forged frozen/subclass objects."""

    if type(config) is not Milestone4CampaignConfig:
        raise TypeError("config must be an exact Milestone4CampaignConfig")
    if type(config.data) is not CampaignDataSpec:
        raise TypeError("data must be an exact CampaignDataSpec")
    if type(config.task) is not BindingTaskConfig:
        raise TypeError("task must be an exact BindingTaskConfig")
    data = CampaignDataSpec(
        generator_version=config.data.generator_version,
        train=_rebuild_exact(config.data.train, TrainDataSpec),
        validation=_rebuild_exact(config.data.validation, EvaluationDataSpec),
        test=(
            None
            if config.data.test is None
            else _rebuild_exact(config.data.test, EvaluationDataSpec)
        ),
        scaling=(
            None
            if config.data.scaling is None
            else _rebuild_exact(config.data.scaling, EvaluationDataSpec)
        ),
    )
    rebuilt = Milestone4CampaignConfig(
        schema_version=config.schema_version,
        campaign_id=config.campaign_id,
        stage=config.stage,
        description=config.description,
        claim_eligible=config.claim_eligible,
        implementation_policy=_rebuild_exact(
            config.implementation_policy, ImplementationPolicy
        ),
        task=BindingTaskConfig(**asdict(config.task)),
        models=tuple(_rebuild_exact(value, CampaignModelSpec) for value in config.models),
        pairs=tuple(_rebuild_exact(value, CampaignPairSpec) for value in config.pairs),
        data=data,
        training=_rebuild_exact(config.training, CampaignTrainingSpec),
        quality=_rebuild_exact(config.quality, CampaignQualitySpec),
        statistics=_rebuild_exact(config.statistics, CampaignStatisticsSpec),
        runtime=_rebuild_exact(config.runtime, CampaignRuntimeSpec),
        selection=(
            None
            if config.selection is None
            else _rebuild_exact(config.selection, CampaignSelectionSpec)
        ),
        promotion=(
            None
            if config.promotion is None
            else _rebuild_exact(config.promotion, CampaignPromotionSpec)
        ),
    )
    if rebuilt != config:
        raise ValueError("config changes under exact nested reconstruction")
    return rebuilt


def _deep_rebuild_run(run: ResolvedCampaignRun) -> ResolvedCampaignRun:
    if type(run) is not ResolvedCampaignRun:
        raise TypeError("run must be an exact ResolvedCampaignRun")
    if type(run.data) is not CampaignDataSpec:
        raise TypeError("run data must be an exact CampaignDataSpec")
    if type(run.task) is not BindingTaskConfig:
        raise TypeError("run task must be an exact BindingTaskConfig")
    data = CampaignDataSpec(
        generator_version=run.data.generator_version,
        train=_rebuild_exact(run.data.train, TrainDataSpec),
        validation=_rebuild_exact(run.data.validation, EvaluationDataSpec),
        test=(
            None
            if run.data.test is None
            else _rebuild_exact(run.data.test, EvaluationDataSpec)
        ),
        scaling=(
            None
            if run.data.scaling is None
            else _rebuild_exact(run.data.scaling, EvaluationDataSpec)
        ),
    )
    values = {field.name: getattr(run, field.name) for field in fields(run)}
    values["task"] = BindingTaskConfig(**asdict(run.task))
    values["data"] = data
    values["training"] = (
        None
        if run.training is None
        else _rebuild_exact(run.training, CampaignTrainingSpec)
    )
    rebuilt = ResolvedCampaignRun(**values)
    if rebuilt != run:
        raise ValueError("run changes under exact nested reconstruction")
    return rebuilt


@dataclass(frozen=True)
class CompactCampaignDerivation:
    """One physically compact child and its exact source lineage metadata."""

    model: RoutedBindingModel
    selection: CPRankSelection
    manifest: CompactExportManifest
    parent_run_id: str
    compact_run_id: str

    def __post_init__(self) -> None:
        if type(self.model) is not RoutedBindingModel:
            raise TypeError("model must be an exact RoutedBindingModel")
        if type(self.selection) is not CPRankSelection:
            raise TypeError("selection must be CPRankSelection")
        if type(self.manifest) is not CompactExportManifest:
            raise TypeError("manifest must be CompactExportManifest")
        for name in ("parent_run_id", "compact_run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 run identifier")


@dataclass(frozen=True)
class CampaignTrainingStepResult:
    """One exact generated batch and the model-specific optimization result."""

    step: int
    batch: BindingBatch
    batch_sha256: str
    output: object
    loss: object

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a nonnegative integer")
        if type(self.batch) is not BindingBatch:
            raise TypeError("batch must be an exact BindingBatch")
        if not isinstance(self.batch_sha256, str) or len(self.batch_sha256) != 64:
            raise ValueError("batch_sha256 must be a SHA-256 digest")


@dataclass(frozen=True)
class CampaignRunContext:
    """A run proven to be one exact member of its complete campaign config."""

    config: Milestone4CampaignConfig
    run: ResolvedCampaignRun

    def __post_init__(self) -> None:
        config = _deep_rebuild_config(self.config)
        run = _deep_rebuild_run(self.run)
        expected = resolve_campaign_plan(
            config,
            run.code_commit,
            run.code_tree,
            run.raw_config_sha256,
            run.executable_bundle_sha256,
        )
        matches = tuple(item for item in expected if item.run_id == run.run_id)
        if matches != (run,):
            raise ValueError("run is not an exact member of the complete campaign plan")

    def __getattr__(self, name: str) -> object:
        """Expose immutable run fields without weakening context validation."""

        try:
            run = object.__getattribute__(self, "run")
        except AttributeError as error:
            raise AttributeError(name) from error
        return getattr(run, name)


def _validated_run(context: CampaignRunContext) -> ResolvedCampaignRun:
    if type(context) is not CampaignRunContext:
        raise TypeError("context must be an exact CampaignRunContext")
    context.__post_init__()
    return _deep_rebuild_run(context.run)


def validate_campaign_execution_environment(context: CampaignRunContext) -> None:
    """Verify the isolated worker already installed the declared Torch policy.

    This function intentionally does not mutate process-global state.  A fresh
    campaign worker must configure threads and deterministic algorithms before
    importing/constructing executable models, then call this guard.
    """

    _validated_run(context)
    policy = context.config.implementation_policy
    if torch.are_deterministic_algorithms_enabled() is not policy.deterministic_algorithms:
        raise RuntimeError("Torch deterministic-algorithm policy is not installed")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError("Torch deterministic algorithms cannot use warn-only mode")
    if torch.get_num_threads() != policy.intraop_threads:
        raise RuntimeError("Torch intra-op thread policy is not installed")
    if torch.get_num_interop_threads() != policy.interop_threads:
        raise RuntimeError("Torch inter-op thread policy is not installed")


def _model_spec(run: ResolvedCampaignRun) -> CampaignModelSpec:
    return CampaignModelSpec(
        model_id=run.model_id,
        family=run.family,
        role=run.role,
        routing_mode=run.routing_mode,
        parent_model_id=run.parent_model_id,
        architecture=run.architecture,
        export=run.export,
    )


def _source_config(run: ResolvedCampaignRun) -> object:
    if run.role != "trainable_source" or not run.training_required:
        raise ValueError("only a trainable source run can construct a source model")
    values = _model_spec(run).architecture_values
    if run.family == "routed":
        raw_schedule = values.pop("schedule")
        schedule = (
            None
            if raw_schedule is None
            else CurriculumSchedule(**raw_schedule)
        )
        return BindingModelConfig(
            task=run.task,
            d_model=values["d_model"],
            cp_rank=values["cp_rank"],
            router_hidden_dim=values["router_hidden_dim"],
            routing_mode=RoutingMode(run.routing_mode),
            curriculum_schedule=schedule,
            curriculum_seed=values["curriculum_seed"],
            scale_feature_dim=values["scale_feature_dim"],
            straight_through_route_surrogate=values[
                "straight_through_route_surrogate"
            ],
        )
    if run.family == "gru":
        return RecurrentBindingBaselineConfig(task=run.task, **values)
    if run.family == "cached_transformer":
        return CachedTransformerBindingBaselineConfig(task=run.task, **values)
    if run.family == "causal_ttn":
        return CausalTreeBindingBaselineConfig(task=run.task, **values)
    raise ValueError(f"unsupported campaign model family {run.family!r}")


def _construct(config: object) -> CampaignModel:
    if type(config) is BindingModelConfig:
        return RoutedBindingModel(config)
    if type(config) is RecurrentBindingBaselineConfig:
        return RecurrentBindingBaseline(config)
    if type(config) is CachedTransformerBindingBaselineConfig:
        return CachedCausalTransformerBindingBaseline(config)
    if type(config) is CausalTreeBindingBaselineConfig:
        return CausalCompleteTreeBindingBaseline(config)
    raise TypeError("unsupported campaign model config")


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("campaign dtype must be float32 or float64")


def _reject_global_hooks() -> None:
    import torch.nn.modules.module as module_hooks
    import torch.optim.optimizer as optimizer_hooks

    registries = [
        value
        for owner in (module_hooks, optimizer_hooks)
        for name, value in vars(owner).items()
        if name.startswith("_global_") and name.endswith("hooks")
    ]
    if any(bool(registry) for registry in registries):
        raise ValueError("global PyTorch module/optimizer hooks are unsupported")


_HOOK_FIELDS = (
    "_backward_hooks",
    "_backward_pre_hooks",
    "_forward_hooks",
    "_forward_hooks_always_called",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_load_state_dict_post_hooks",
    "_load_state_dict_pre_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
)


def _simple_metadata(module: nn.Module) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in module.__dict__.items():
        if name == "training" or name.startswith("_forward") or name in _HOOK_FIELDS:
            continue
        if value is None or isinstance(value, (bool, int, float, str, torch.dtype)):
            result[name] = value
        elif isinstance(value, tuple) and all(
            item is None or isinstance(item, (bool, int, float, str)) for item in value
        ):
            result[name] = value
        elif is_dataclass(value) and not isinstance(value, type):
            result[name] = value
    return result


def _tensor_storage_interval(tensor: torch.Tensor) -> tuple[torch.device, int, int, int]:
    if tensor.layout is not torch.strided or not tensor.is_contiguous():
        raise ValueError("campaign tensors must be contiguous strided tensors")
    storage = tensor.untyped_storage()
    start = int(tensor.storage_offset()) * tensor.element_size()
    end = start + tensor.numel() * tensor.element_size()
    if start != 0 or storage.nbytes() != end:
        raise ValueError("campaign tensors cannot be views or oversized storage")
    return tensor.device, int(storage.data_ptr()), start, end


def _validate_disjoint_named_tensors(model: nn.Module) -> None:
    intervals: list[tuple[str, torch.device, int, int, int]] = []
    named_parameters = list(model.named_parameters())
    named_buffers = list(model.named_buffers())
    if any(type(tensor) is not nn.Parameter for _, tensor in named_parameters):
        raise TypeError("model parameters must be exact torch.nn.Parameter objects")
    if any(type(tensor) is not torch.Tensor for _, tensor in named_buffers):
        raise TypeError("model buffers must be exact torch.Tensor objects")
    for name, tensor in named_parameters + named_buffers:
        device, pointer, start, end = _tensor_storage_interval(tensor)
        if tensor.numel() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"model tensor {name!r} must be finite")
        for prior_name, prior_device, prior_pointer, prior_start, prior_end in intervals:
            if (
                tensor.numel()
                and device == prior_device
                and pointer == prior_pointer
                and max(start, prior_start) < min(end, prior_end)
            ):
                raise ValueError(
                    f"model tensors {prior_name!r} and {name!r} share storage"
                )
        intervals.append((name, device, pointer, start, end))


def _construct_schema(config: object) -> CampaignModel:
    with torch.random.fork_rng(devices=[], enabled=True):
        with torch.device("meta"):
            return _construct(config)


def _validate_live_model(model: CampaignModel, expected_config: object) -> None:
    _reject_global_hooks()
    schema = _construct_schema(expected_config)
    actual_modules = list(model.named_modules(remove_duplicate=False))
    schema_modules = list(schema.named_modules(remove_duplicate=False))
    if [name for name, _ in actual_modules] != [name for name, _ in schema_modules]:
        raise ValueError("model module topology does not match its resolved config")
    for (name, actual), (_, expected) in zip(
        actual_modules, schema_modules, strict=True
    ):
        if type(actual) is not type(expected):
            raise ValueError(f"model module type changed at {name!r}")
        if _simple_metadata(actual) != _simple_metadata(expected):
            raise ValueError(f"model executable metadata changed at {name!r}")
        if any(bool(getattr(actual, field, None)) for field in _HOOK_FIELDS):
            raise ValueError(f"model hooks are unsupported at {name!r}")
        if any(callable(value) for value in actual.__dict__.values()):
            raise ValueError(f"instance callables are unsupported at {name!r}")
    actual_parameters = dict(model.named_parameters())
    schema_parameters = dict(schema.named_parameters())
    if set(actual_parameters) != set(schema_parameters):
        raise ValueError("model parameter keys changed from the resolved config")
    if any(
        tuple(actual_parameters[name].shape) != tuple(schema_parameters[name].shape)
        or actual_parameters[name].requires_grad
        != schema_parameters[name].requires_grad
        for name in actual_parameters
    ):
        raise ValueError("model parameter shape or trainability changed")
    actual_state = model.state_dict()
    schema_state = schema.state_dict()
    if set(actual_state) != set(schema_state) or any(
        tuple(actual_state[name].shape) != tuple(schema_state[name].shape)
        for name in actual_state
    ):
        raise ValueError("model state topology changed from the resolved config")
    for name, parameter in actual_parameters.items():
        if bool(getattr(parameter, "_backward_hooks", None)) or bool(
            getattr(parameter, "_post_accumulate_grad_hooks", None)
        ):
            raise ValueError(f"parameter hooks are unsupported at {name!r}")
        if parameter.grad is not None:
            raise ValueError("campaign model must be at an optimizer step boundary")
    _validate_disjoint_named_tensors(model)


def build_campaign_source_model(context: CampaignRunContext) -> CampaignModel:
    """Construct one exact seeded source model without changing caller RNG."""

    run = _validated_run(context)
    _reject_global_hooks()
    config = _source_config(run)
    assert run.training is not None
    dtype = _dtype(run.training.dtype)
    prior_dtype = torch.get_default_dtype()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.set_default_dtype(dtype)
            torch.manual_seed(run.model_seed)
            with torch.device(run.training.device):
                model = _construct(config)
    finally:
        torch.set_default_dtype(prior_dtype)
    if type(model) is not _SOURCE_TYPES[run.family]:
        raise RuntimeError("campaign model factory returned the wrong exact type")
    _validate_live_model(model, config)
    _MODEL_BINDINGS[model] = (run.run_id, run.model_seed)
    return model


def _validate_model_for_run(model: CampaignModel, context: CampaignRunContext) -> None:
    run = _validated_run(context)
    expected_type = _SOURCE_TYPES.get(run.family)
    if expected_type is None or type(model) is not expected_type:
        raise TypeError("model exact type does not match the resolved source run")
    if _MODEL_BINDINGS.get(model) != (run.run_id, run.model_seed):
        raise ValueError("model construction identity does not match the resolved run")
    expected_config = _source_config(run)
    if getattr(model, "config", None) != expected_config:
        raise ValueError("model config does not match the resolved source run")
    _validate_live_model(model, expected_config)
    parameters = tuple(model.parameters())
    if not parameters:
        raise ValueError("campaign model has no parameters")
    assert run.training is not None
    expected_dtype = _dtype(run.training.dtype)
    expected_device = torch.device(run.training.device)
    if any(parameter.dtype != expected_dtype for parameter in parameters):
        raise ValueError("model parameter dtype does not match the resolved run")
    if any(parameter.device != expected_device for parameter in parameters):
        raise ValueError("model parameter device does not match the resolved run")


def build_campaign_optimizer(
    context: CampaignRunContext,
    model: CampaignModel,
) -> torch.optim.AdamW:
    """Build the exact declared AdamW optimizer for a matching source model."""

    run = _validated_run(context)
    _validate_model_for_run(model, context)
    assert run.training is not None
    if run.training.optimizer != "adamw":
        raise ValueError("campaign optimizer must be adamw")
    return torch.optim.AdamW(
        list(model.parameters()),
        lr=run.training.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=run.training.weight_decay,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


def _derived_seed(base_seed: int, stream: str, index: int) -> int:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("stream index must be a nonnegative integer")
    if stream not in {"train", "validation", "test", "scaling"}:
        raise ValueError("unknown campaign data stream")
    material = f"{_SEED_DOMAIN}|{base_seed}|{stream}|{index}".encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], "little") & (2**63 - 1)


def generate_campaign_training_batch(
    context: CampaignRunContext,
    *,
    step: int,
) -> BindingBatch:
    """Generate the paired mixed-length training batch for one optimizer step."""

    run = _validated_run(context)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("training step must be a nonnegative integer")
    if not run.training_required:
        raise ValueError("derived compact runs do not own a training stream")
    if run.training is None or step >= run.training.optimizer_steps:
        raise ValueError("training step is outside the resolved run")
    lengths = run.data.train.length_schedule
    episodes = generate_binding_episodes(
        run.task,
        count=run.data.train.batch_size,
        seed=_derived_seed(run.train_seed, "train", step),
        split="train",
        lengths=lengths,
    )
    return collate_binding_episodes(episodes)


def generate_campaign_evaluation_batch(
    context: CampaignRunContext,
    *,
    stream: str,
    length: int,
) -> BindingBatch:
    """Generate one immutable validation, test, or scaling fixture batch.

    ``validation`` maps to the generator's held-out-capable ``eval`` split.
    Confirmatory ``test`` and ``scaling`` both use the generator's ``test``
    semantics but remain disjoint through their seed domains.
    """

    run = _validated_run(context)
    if stream == "validation":
        spec = run.data.validation
        base_seed = run.validation_seed
        generator_split = "eval"
    elif stream == "test":
        spec = run.data.test
        base_seed = run.test_seed
        generator_split = "test"
    elif stream == "scaling":
        spec = run.data.scaling
        base_seed = run.test_seed
        generator_split = "test"
    else:
        raise ValueError("stream must be validation, test, or scaling")
    if spec is None or base_seed is None:
        raise ValueError(f"{stream} stream is unavailable for this campaign stage")
    if isinstance(length, bool) or not isinstance(length, int) or length not in spec.lengths:
        raise ValueError(f"length is not declared for the {stream} stream")
    length_index = spec.lengths.index(length)
    episodes = generate_binding_episodes(
        run.task,
        count=spec.episodes_per_length,
        seed=_derived_seed(base_seed, stream, length_index),
        split=generator_split,
        lengths=[length] * spec.episodes_per_length,
    )
    return collate_binding_episodes(episodes)


def campaign_batch_sha256(batch: BindingBatch) -> str:
    """Hash all model-visible, evaluation, and document-provenance fields."""

    if type(batch) is not BindingBatch:
        raise TypeError("batch must be an exact BindingBatch")
    if type(batch.inputs) is not BindingModelInputs or type(batch.evaluation) is not BindingEvaluation:
        raise TypeError("batch must contain exact input and evaluation dataclasses")
    shape = batch.inputs.valid_mask.shape
    if len(shape) != 2 or shape[0] <= 0:
        raise ValueError("campaign batch tensors must have shape [N,T] with N > 0")
    input_names = tuple(batch.inputs.__dataclass_fields__)
    evaluation_names = tuple(batch.evaluation.__dataclass_fields__)
    tensors = [getattr(batch.inputs, name) for name in input_names] + [
        getattr(batch.evaluation, name) for name in evaluation_names
    ]
    if any(type(tensor) is not torch.Tensor for tensor in tensors):
        raise TypeError("campaign batch values must be exact tensors")
    expected_shapes = {
        **{name: shape for name in input_names},
        **{name: shape for name in evaluation_names},
        "dependency_parents": (*shape, 2),
    }
    for owner in (batch.inputs, batch.evaluation):
        for name in owner.__dataclass_fields__:
            if getattr(owner, name).shape != expected_shapes[name]:
                raise ValueError("campaign batch tensor shapes are inconsistent")
    integer_names = (set(input_names) | set(evaluation_names)) - {
        "valid_mask",
        "heldout_combination_mask",
    }
    for owner in (batch.inputs, batch.evaluation):
        for name in owner.__dataclass_fields__:
            tensor = getattr(owner, name)
            expected = torch.int64 if name in integer_names else torch.bool
            if tensor.dtype is not expected:
                raise TypeError(f"campaign batch tensor {name!r} has the wrong dtype")
    if batch.lengths.shape != (shape[0],) or batch.lengths.dtype is not torch.int64:
        raise ValueError("campaign batch lengths must be int64 [N]")
    if len(batch.splits) != shape[0] or len(batch.document_ids) != shape[0] or len(batch.generation_seeds) != shape[0]:
        raise ValueError("campaign batch provenance cardinality is inconsistent")
    if any(split not in {"train", "validation", "eval", "test"} for split in batch.splits):
        raise ValueError("campaign batch split metadata is invalid")
    if any(not isinstance(value, str) or not value for value in batch.document_ids):
        raise ValueError("campaign document IDs must be nonempty strings")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64 for seed in batch.generation_seeds):
        raise ValueError("campaign generation seeds must be nonnegative integers")
    if (
        not isinstance(batch.config_fingerprint, str)
        or len(batch.config_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in batch.config_fingerprint)
    ):
        raise ValueError("campaign batch config fingerprint must be lowercase SHA-256")
    valid_counts = batch.inputs.valid_mask.to(torch.int64).sum(dim=1)
    if not torch.equal(batch.lengths.to(valid_counts.device), valid_counts):
        raise ValueError("campaign batch lengths do not match its valid-event mask")
    digest = sha256()
    for owner_name, owner in (("inputs", batch.inputs), ("evaluation", batch.evaluation)):
        for name in owner.__dataclass_fields__:
            tensor = getattr(owner, name).detach().contiguous().cpu()
            if tensor.dtype is torch.int64:
                raw = tensor.numpy().astype("<i8", copy=False).tobytes(order="C")
                dtype_name = "int64-le"
            else:
                raw = tensor.numpy().astype("u1", copy=False).tobytes(order="C")
                dtype_name = "bool-u8"
            digest.update(
                f"{owner_name}.{name}|{dtype_name}|{tuple(tensor.shape)}|".encode()
            )
            digest.update(raw)
    lengths = batch.lengths.detach().contiguous().cpu()
    digest.update(f"lengths|int64-le|{tuple(lengths.shape)}|".encode())
    digest.update(lengths.numpy().astype("<i8", copy=False).tobytes(order="C"))
    for value in (
        batch.splits,
        batch.document_ids,
        batch.generation_seeds,
        batch.config_fingerprint,
    ):
        digest.update(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    return digest.hexdigest()


def campaign_validation_fingerprint(context: CampaignRunContext) -> str:
    """Bind every declared validation length and its complete held-out fixture."""

    run = _validated_run(context)
    digest = sha256(b"tnlm-v3-m4-validation-fixtures-v1\0")
    for length in run.data.validation.lengths:
        batch_hash = campaign_batch_sha256(
            generate_campaign_evaluation_batch(
                context, stream="validation", length=length
            )
        )
        digest.update(length.to_bytes(8, "little", signed=False))
        digest.update(bytes.fromhex(batch_hash))
    return digest.hexdigest()


def _validate_optimizer(context: CampaignRunContext, model: CampaignModel, optimizer: torch.optim.Optimizer) -> None:
    if (
        type(optimizer) is not torch.optim.AdamW
        or type(optimizer.param_groups) is not list
        or len(optimizer.param_groups) != 1
    ):
        raise TypeError("campaign optimizer must be one exact AdamW parameter group")
    run = _validated_run(context)
    assert run.training is not None
    group = optimizer.param_groups[0]
    if type(group) is not dict or type(group.get("params")) is not list:
        raise TypeError("optimizer parameter group must use canonical dict/list storage")
    if [id(value) for value in group["params"]] != [
        id(value) for value in model.parameters()
    ]:
        raise ValueError("optimizer parameter order does not match the model")
    expected = {
        "lr": run.training.learning_rate,
        "weight_decay": run.training.weight_decay,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "decoupled_weight_decay": True,
    }
    def exact_value(actual: object, wanted: object) -> bool:
        if type(actual) is not type(wanted):
            return False
        if type(wanted) is tuple:
            return len(actual) == len(wanted) and all(
                type(left) is type(right) and left == right
                for left, right in zip(actual, wanted, strict=True)
            )
        return actual == wanted

    if set(group) != {"params", *expected} or any(
        not exact_value(group.get(name), value) for name, value in expected.items()
    ):
        raise ValueError("optimizer hyperparameters changed from the campaign contract")
    if type(optimizer.defaults) is not dict:
        raise TypeError("optimizer defaults must use canonical dict storage")
    if set(optimizer.defaults) != set(expected) or any(
        not exact_value(optimizer.defaults.get(name), value)
        for name, value in expected.items()
    ):
        raise ValueError("optimizer defaults changed from the campaign contract")
    if any(
        "hook" in name and bool(value)
        for name, value in optimizer.__dict__.items()
    ):
        raise ValueError("optimizer instance hooks are unsupported")
    if any(callable(value) for value in optimizer.__dict__.values()):
        raise ValueError("optimizer instance-level callables are unsupported")
    if "step" in optimizer.__dict__ or "zero_grad" in optimizer.__dict__:
        raise ValueError("optimizer executable methods cannot be shadowed")
    if not _is_exact_adamw_step(getattr(type(optimizer), "step")):
        raise ValueError("optimizer step implementation is not exact AdamW")
    if getattr(type(optimizer), "zero_grad") is not _OPTIMIZER_ZERO_GRAD:
        raise ValueError("optimizer zero_grad implementation is not exact")
    if type(optimizer.state) is not defaultdict or optimizer.state.default_factory is not dict:
        raise TypeError("optimizer state must be the canonical defaultdict(dict)")
    model_parameters = set(model.parameters())
    if not set(optimizer.state).issubset(model_parameters):
        raise ValueError("optimizer state contains a foreign parameter")
    intervals: list[tuple[str, torch.device, int, int, int]] = []
    for parameter_index, parameter in enumerate(model.parameters()):
        device, pointer, start, end = _tensor_storage_interval(parameter)
        intervals.append((f"parameter.{parameter_index}", device, pointer, start, end))
    for parameter_index, (parameter, state) in enumerate(optimizer.state.items()):
        if type(state) is not dict or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("AdamW optimizer state schema is invalid")
        for name, tensor in state.items():
            if type(tensor) is not torch.Tensor:
                raise TypeError("AdamW optimizer state values must be tensors")
            expected_shape = () if name == "step" else tuple(parameter.shape)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError("AdamW optimizer state shape is invalid")
            if name != "step" and (
                tensor.dtype != parameter.dtype or tensor.device != parameter.device
            ):
                raise ValueError("AdamW moment dtype/device does not match its parameter")
            if name == "step" and (
                tensor.dtype is not torch.float32 or tensor.device.type != "cpu"
            ):
                raise ValueError("AdamW step must be the canonical CPU float32 scalar")
            if tensor.numel() and not bool(torch.isfinite(tensor).all()):
                raise ValueError("AdamW optimizer state must be finite")
            device, pointer, start, end = _tensor_storage_interval(tensor)
            for prior_name, prior_device, prior_pointer, prior_start, prior_end in intervals:
                if (
                    tensor.numel()
                    and device == prior_device
                    and pointer == prior_pointer
                    and max(start, prior_start) < min(end, prior_end)
                ):
                    raise ValueError(
                        f"optimizer tensor {parameter_index}.{name} shares storage with {prior_name}"
                    )
            intervals.append(
                (f"optimizer.{parameter_index}.{name}", device, pointer, start, end)
            )


def _validate_finite_routed_result(
    model: RoutedBindingModel,
    output: object,
    loss: object,
) -> None:
    """Apply the same numerical step-boundary checks used by baseline controls."""

    value_logits = getattr(output, "value_logits", None)
    route_logits = getattr(output, "route_logits", None)
    if type(value_logits) is not torch.Tensor or not bool(torch.isfinite(value_logits).all()):
        raise ValueError("routed value logits must be finite")
    if type(route_logits) is not torch.Tensor:
        raise TypeError("routed route logits must be tensors")
    if bool(torch.isnan(route_logits).any()) or bool(torch.isposinf(route_logits).any()):
        raise ValueError("routed route logits contain invalid nonfinite values")
    total = getattr(loss, "total", None)
    if type(total) is not torch.Tensor or total.numel() != 1 or not bool(torch.isfinite(total)):
        raise ValueError("routed training loss must be a finite scalar")
    for parameter in model.parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError("routed parameters must remain finite")


def _validate_optimizer_cursor(
    model: CampaignModel,
    optimizer: torch.optim.Optimizer,
    *,
    expected_step: int,
    require_complete: bool = True,
) -> None:
    parameters = tuple(model.parameters())
    initialized = [parameter for parameter in parameters if optimizer.state.get(parameter)]
    if initialized and require_complete and len(initialized) != len(parameters):
        raise ValueError("optimizer state is partially initialized")
    if not initialized:
        if expected_step != 0:
            raise ValueError("a fresh optimizer must begin at training step zero")
        return
    counters: list[int] = []
    for parameter in initialized:
        counter = optimizer.state[parameter]["step"]
        if (
            type(counter) is not torch.Tensor
            or counter.shape != ()
            or counter.dtype is not torch.float32
            or counter.device.type != "cpu"
        ):
            raise ValueError("optimizer step counter is invalid")
        value = float(counter.item())
        if not value.is_integer() or value < 0:
            raise ValueError("optimizer step counter must be a nonnegative integer")
        counters.append(int(value))
    if any(value != expected_step for value in counters):
        raise ValueError("requested training step does not match optimizer state")


def run_campaign_training_step(
    context: CampaignRunContext,
    model: CampaignModel,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
) -> CampaignTrainingStepResult:
    """Generate and execute exactly one declared optimizer step."""

    run = _validated_run(context)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("training step must be a nonnegative integer")
    _validate_model_for_run(model, context)
    _validate_optimizer(context, model, optimizer)
    _validate_optimizer_cursor(model, optimizer, expected_step=step)
    batch = generate_campaign_training_batch(context, step=step)
    assert run.training is not None
    if type(model) is RoutedBindingModel:
        output, loss = train_binding_step(
            model,
            batch,
            optimizer,
            training_step=step,
            max_gradient_norm=run.training.max_gradient_norm,
        )
        _validate_finite_routed_result(model, output, loss)
    else:
        output, loss = train_baseline_step(
            model,
            batch,
            optimizer,
            max_gradient_norm=run.training.max_gradient_norm,
        )
    _validate_optimizer(context, model, optimizer)
    optimizer.zero_grad(set_to_none=True)
    return CampaignTrainingStepResult(
        step=step,
        batch=batch,
        batch_sha256=campaign_batch_sha256(batch),
        output=output,
        loss=loss,
    )


def derive_campaign_compact_model(
    parent_context: CampaignRunContext,
    compact_context: CampaignRunContext,
    parent_model: RoutedBindingModel,
    parent_optimizer: torch.optim.Optimizer,
) -> CompactCampaignDerivation:
    """Derive a compact child only at its parent's declared final train cursor.

    The campaign runner remains responsible for binding this in-memory parent
    to a completed immutable attempt record and checkpoint artifact.
    """

    parent_run = _validated_run(parent_context)
    compact_run = _validated_run(compact_context)
    _validate_model_for_run(parent_model, parent_context)
    _validate_optimizer(parent_context, parent_model, parent_optimizer)
    assert parent_run.training is not None
    _validate_optimizer_cursor(
        parent_model,
        parent_optimizer,
        expected_step=parent_run.training.optimizer_steps,
    )
    if (
        parent_run.family != "routed"
        or parent_run.routing_mode != "curriculum"
        or compact_run.role != "derived_compact"
        or compact_run.parent_run_id != parent_run.run_id
        or compact_run.parent_model_id != parent_run.model_id
    ):
        raise ValueError("compact run is not the exact child of this curriculum source")
    export = _model_spec(compact_run).export_values
    assert export is not None
    calibration_fingerprint = campaign_validation_fingerprint(parent_context)
    selection = select_cp_rank_by_parameter_energy(
        parent_model,
        target_rank=export["target_cp_rank"],
        calibration_fingerprint=calibration_fingerprint,
    )
    compact, manifest = export_compact_binding_model(parent_model, selection)
    compact_values = _model_spec(compact_run).architecture_values
    if compact.config.cp_rank != compact_values["cp_rank"]:
        raise RuntimeError("physical compact rank does not match the resolved child")
    if compact.config.task != parent_model.config.task:
        raise RuntimeError("compact export changed the model-visible task architecture")
    return CompactCampaignDerivation(
        model=compact,
        selection=selection,
        manifest=manifest,
        parent_run_id=parent_run.run_id,
        compact_run_id=compact_run.run_id,
    )


__all__ = [
    "CampaignModel",
    "CampaignRunContext",
    "CampaignTrainingStepResult",
    "CompactCampaignDerivation",
    "build_campaign_optimizer",
    "build_campaign_source_model",
    "campaign_batch_sha256",
    "campaign_validation_fingerprint",
    "derive_campaign_compact_model",
    "generate_campaign_evaluation_batch",
    "generate_campaign_training_batch",
    "run_campaign_training_step",
    "validate_campaign_execution_environment",
]
