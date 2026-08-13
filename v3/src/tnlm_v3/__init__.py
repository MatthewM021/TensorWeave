"""TensorWeave TNLM V3 research implementation."""

from .binding import (
    BindingArchitectureConfig,
    BindingEventEncoder,
    BindingModelConfig,
    BindingModelOutput,
    RoutedBindingModel,
)
from .data import (
    BindingBatch,
    BindingEpisode,
    BindingEvaluation,
    BindingEventKind,
    BindingModelInputs,
    BindingTaskConfig,
    collate_binding_episodes,
    generate_binding_episode,
    generate_binding_episodes,
)
from .forest import (
    ForestConfig,
    ForestModelOutput,
    ForestPrefixRun,
    ForestReadout,
    ForestRun,
    ForestState,
    RoutedTensorLanguageModel,
    ScaleSharedBinaryForest,
)
from .model_export import CompactExportManifest, export_compact_binding_model
from .operators import ScaleSharedCPMerge, analytic_scale_features, slice_cp_merge
from .routing import (
    NULL_ROUTE,
    CurriculumSchedule,
    PersistentCausalRouter,
    RoutingMode,
)
from .training import (
    BindingEvaluationSummary,
    BindingLoss,
    BindingLossConfig,
    compute_binding_loss,
    evaluate_binding_model,
    train_binding_step,
)
from .truncation import (
    CPRankSelection,
    build_dense_selected_reference,
    model_state_fingerprint,
    select_cp_rank_by_parameter_energy,
)
from .factory import (
    BindingExperimentConfig,
    build_binding_model,
    build_model,
    load_binding_experiment_config,
    load_forest_config,
)

__all__ = [
    "BindingBatch",
    "BindingArchitectureConfig",
    "BindingEpisode",
    "BindingEvaluation",
    "BindingEvaluationSummary",
    "BindingEventEncoder",
    "BindingEventKind",
    "BindingExperimentConfig",
    "BindingLoss",
    "BindingLossConfig",
    "BindingModelConfig",
    "BindingModelInputs",
    "BindingModelOutput",
    "BindingTaskConfig",
    "CPRankSelection",
    "CompactExportManifest",
    "CurriculumSchedule",
    "ForestConfig",
    "ForestModelOutput",
    "ForestPrefixRun",
    "ForestReadout",
    "ForestRun",
    "ForestState",
    "NULL_ROUTE",
    "PersistentCausalRouter",
    "RoutedBindingModel",
    "RoutedTensorLanguageModel",
    "RoutingMode",
    "ScaleSharedBinaryForest",
    "ScaleSharedCPMerge",
    "analytic_scale_features",
    "build_dense_selected_reference",
    "build_binding_model",
    "build_model",
    "collate_binding_episodes",
    "compute_binding_loss",
    "evaluate_binding_model",
    "export_compact_binding_model",
    "generate_binding_episode",
    "generate_binding_episodes",
    "load_binding_experiment_config",
    "load_forest_config",
    "model_state_fingerprint",
    "select_cp_rank_by_parameter_energy",
    "slice_cp_merge",
    "train_binding_step",
]
