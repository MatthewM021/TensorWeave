from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from tnlm_v3.baselines import (
    CachedCausalTransformerBindingBaseline,
    RecurrentBindingBaseline,
)
from tnlm_v3.binding import RoutedBindingModel
from tnlm_v3.campaign_config import (
    load_milestone4_campaign_config,
    resolve_campaign_plan,
)
from tnlm_v3.campaign_execution import (
    CampaignRunContext,
    build_campaign_optimizer,
    build_campaign_source_model,
    campaign_batch_sha256,
    campaign_validation_fingerprint,
    derive_campaign_compact_model,
    generate_campaign_evaluation_batch,
    generate_campaign_training_batch,
    run_campaign_training_step,
    validate_campaign_execution_environment,
)
from tnlm_v3.causal_ttn import CausalCompleteTreeBindingBaseline


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "v3" / "configs" / "milestone4" / "pilot_smoke.yaml"


@pytest.fixture(scope="module")
def plan():
    config = load_milestone4_campaign_config(CONFIG_PATH)
    runs = resolve_campaign_plan(config, "1" * 40, "2" * 40, "3" * 64, "4" * 64)
    return tuple(CampaignRunContext(config, run) for run in runs)


def state_dict_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_every_source_constructs_deterministically_without_rng_side_effect(plan) -> None:
    expected_types = {
        "routed": RoutedBindingModel,
        "gru": RecurrentBindingBaseline,
        "cached_transformer": CachedCausalTransformerBindingBaseline,
        "causal_ttn": CausalCompleteTreeBindingBaseline,
    }
    for run in plan:
        if not run.training_required:
            continue
        torch.manual_seed(991)
        before = torch.random.get_rng_state().clone()
        first = build_campaign_source_model(run)
        after = torch.random.get_rng_state()
        second = build_campaign_source_model(run)
        assert torch.equal(before, after)
        assert type(first) is expected_types[run.family]
        assert type(second) is expected_types[run.family]
        assert first.config.task.num_surface_keys == run.task.num_surface_keys
        assert not hasattr(first.config.task, "heldout_key_value_pairs")
        assert state_dict_copy(first).keys() == state_dict_copy(second).keys()
        assert all(
            torch.equal(value, state_dict_copy(second)[name])
            for name, value in state_dict_copy(first).items()
        )
        assert all(parameter.dtype == torch.float64 for parameter in first.parameters())


def test_worker_environment_guard_requires_declared_torch_policy(plan, monkeypatch) -> None:
    context = next(item for item in plan if item.model_id == "gru-control")
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False
    )
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    validate_campaign_execution_environment(context)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    with pytest.raises(RuntimeError, match="intra-op"):
        validate_campaign_execution_environment(context)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", lambda: True
    )
    with pytest.raises(RuntimeError, match="warn-only"):
        validate_campaign_execution_environment(context)


def test_source_optimizer_is_exact_and_wrong_model_is_rejected(plan) -> None:
    sources = [run for run in plan if run.training_required]
    model = build_campaign_source_model(sources[0])
    optimizer = build_campaign_optimizer(sources[0], model)
    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.param_groups[0]["lr"] == sources[0].training.learning_rate
    assert optimizer.param_groups[0]["weight_decay"] == sources[0].training.weight_decay

    other = next(run for run in sources if run.family != sources[0].family)
    with pytest.raises(TypeError, match="exact type"):
        build_campaign_optimizer(other, model)
    compact = next(run for run in plan if not run.training_required)
    with pytest.raises(ValueError, match="trainable source"):
        build_campaign_source_model(compact)


def test_paired_training_stream_is_identical_across_models_and_changes_by_step(plan) -> None:
    sources = [run for run in plan if run.training_required]
    first_hashes = {
        campaign_batch_sha256(generate_campaign_training_batch(run, step=0))
        for run in sources
    }
    second_hashes = {
        campaign_batch_sha256(generate_campaign_training_batch(run, step=1))
        for run in sources
    }
    assert len(first_hashes) == len(second_hashes) == 1
    assert first_hashes != second_hashes
    batch = generate_campaign_training_batch(sources[0], step=0)
    assert batch.lengths.tolist() == [10, 12, 16]
    assert batch.splits == ("train", "train", "train")
    with pytest.raises(ValueError, match="outside"):
        generate_campaign_training_batch(
            sources[0], step=sources[0].training.optimizer_steps
        )


def test_validation_is_heldout_capable_and_seed_domains_are_distinct(plan) -> None:
    run = next(item for item in plan if item.training_required)
    batches = [
        generate_campaign_evaluation_batch(run, stream="validation", length=length)
        for length in run.data.validation.lengths
    ]
    assert all(batch.splits == ("eval", "eval") for batch in batches)
    assert all(
        bool(batch.evaluation.heldout_combination_mask.any()) for batch in batches
    )
    assert len({campaign_batch_sha256(batch) for batch in batches}) == len(batches)
    with pytest.raises(ValueError, match="unavailable"):
        generate_campaign_evaluation_batch(run, stream="test", length=64)


def test_compact_is_physically_derived_from_exact_curriculum_parent(plan) -> None:
    parent_run = next(run for run in plan if run.model_id == "routed-source")
    compact_run = next(run for run in plan if run.model_id == "routed-compact")
    parent = build_campaign_source_model(parent_run)
    optimizer = build_campaign_optimizer(parent_run, parent)
    for step in range(parent_run.training.optimizer_steps):
        run_campaign_training_step(parent_run, parent, optimizer, step=step)
    before = state_dict_copy(parent)
    calibration = campaign_validation_fingerprint(parent_run)
    derived = derive_campaign_compact_model(
        parent_run,
        compact_run,
        parent,
        optimizer,
    )
    assert derived.selection.calibration_fingerprint == calibration
    assert derived.model.config.cp_rank == 2
    assert derived.manifest.nominal_cp_rank == 4
    assert derived.manifest.exported_cp_rank == 2
    assert derived.manifest.exported_parameter_count < derived.manifest.source_parameter_count
    assert all(torch.equal(value, parent.state_dict()[name]) for name, value in before.items())

    wrong_parent = copy.deepcopy(parent_run)
    object.__setattr__(wrong_parent.run, "run_id", "f" * 64)
    with pytest.raises(ValueError):
        derive_campaign_compact_model(
            wrong_parent,
            compact_run,
            parent,
            optimizer,
        )


def test_compact_derivation_requires_completed_parent_cursor(plan) -> None:
    parent_run = next(run for run in plan if run.model_id == "routed-source")
    compact_run = next(run for run in plan if run.model_id == "routed-compact")
    parent = build_campaign_source_model(parent_run)
    optimizer = build_campaign_optimizer(parent_run, parent)
    with pytest.raises(ValueError, match="fresh optimizer"):
        derive_campaign_compact_model(
            parent_run, compact_run, parent, optimizer
        )


@pytest.mark.parametrize("model_id", ["routed-source", "gru-control", "transformer-control", "ttn-control"])
def test_shared_training_executor_binds_step_batch_and_optimizer(plan, model_id: str) -> None:
    run = next(item for item in plan if item.model_id == model_id)
    model = build_campaign_source_model(run)
    optimizer = build_campaign_optimizer(run, model)
    result = run_campaign_training_step(run, model, optimizer, step=0)
    assert result.step == 0
    assert result.batch_sha256 == campaign_batch_sha256(result.batch)
    assert result.loss is not None
    optimizer.param_groups[0]["eps"] = 1e-7
    with pytest.raises(ValueError, match="hyperparameters"):
        run_campaign_training_step(run, model, optimizer, step=1)


def test_execution_context_rejects_a_self_hashed_run_from_another_config(plan) -> None:
    context = next(item for item in plan if item.model_id == "gru-control")
    forged = copy.deepcopy(context.run)
    object.__setattr__(forged, "model_seed", forged.model_seed + 10_000)
    from tnlm_v3 import campaign_config as config_module

    object.__setattr__(forged, "run_id", config_module._run_id(forged._identity_payload()))
    with pytest.raises(ValueError, match="complete campaign plan"):
        CampaignRunContext(context.config, forged)


def test_model_construction_is_bound_to_exact_pair_identity(plan) -> None:
    source = next(item for item in plan if item.model_id == "gru-control")
    from dataclasses import replace
    from tnlm_v3.campaign_config import CampaignPairSpec, resolve_campaign_plan

    second_pair = CampaignPairSpec(
        pair_id="pair-2",
        model_seed=source.model_seed + 10_000,
        train_seed=source.train_seed + 20_000,
        validation_seed=source.validation_seed + 30_000,
        statistics_seed=source.statistics_seed + 40_000,
    )
    config = replace(source.config, pairs=(source.config.pairs[0], second_pair))
    contexts = tuple(
        CampaignRunContext(config, run)
        for run in resolve_campaign_plan(config, "1" * 40, "2" * 40, "3" * 64, "4" * 64)
    )
    first_pair_id = source.config.pairs[0].pair_id
    first = next(item for item in contexts if item.model_id == "gru-control" and item.pair_id == first_pair_id)
    second = next(item for item in contexts if item.model_id == "gru-control" and item.pair_id == "pair-2")
    model = build_campaign_source_model(second)
    with pytest.raises(ValueError, match="construction identity"):
        build_campaign_optimizer(first, model)


def test_mutated_model_and_optimizer_state_are_rejected(plan) -> None:
    context = next(item for item in plan if item.model_id == "gru-control")
    model = build_campaign_source_model(context)
    optimizer = build_campaign_optimizer(context, model)
    run_campaign_training_step(context, model, optimizer, step=0)
    first_parameter = next(iter(model.parameters()))
    optimizer.state[first_parameter]["exp_avg"] = optimizer.state[first_parameter][
        "exp_avg_sq"
    ]
    with pytest.raises(ValueError, match="shares storage"):
        run_campaign_training_step(context, model, optimizer, step=1)

    model = build_campaign_source_model(context)
    model.encoder.norm.eps = 1.0
    with pytest.raises(ValueError, match="metadata"):
        build_campaign_optimizer(context, model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_source_construction_preserves_all_cuda_rng_states(plan) -> None:
    context = next(item for item in plan if item.model_id == "gru-control")
    before = [value.clone() for value in torch.cuda.get_rng_state_all()]
    build_campaign_source_model(context)
    after = torch.cuda.get_rng_state_all()
    assert len(before) == len(after)
    assert all(torch.equal(left, right) for left, right in zip(before, after, strict=True))


def test_training_executor_rejects_skip_and_replay_cursors(plan) -> None:
    context = next(item for item in plan if item.model_id == "gru-control")
    model = build_campaign_source_model(context)
    optimizer = build_campaign_optimizer(context, model)
    with pytest.raises(ValueError, match="begin at training step zero"):
        run_campaign_training_step(context, model, optimizer, step=5)
    run_campaign_training_step(context, model, optimizer, step=0)
    with pytest.raises(ValueError, match="does not match optimizer state"):
        run_campaign_training_step(context, model, optimizer, step=0)
    with pytest.raises(ValueError, match="does not match optimizer state"):
        run_campaign_training_step(context, model, optimizer, step=2)


def test_training_executor_rejects_shadowed_optimizer_methods(plan) -> None:
    context = next(item for item in plan if item.model_id == "gru-control")
    model = build_campaign_source_model(context)
    optimizer = build_campaign_optimizer(context, model)
    optimizer.step = lambda closure=None: None  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="callables|shadowed"):
        run_campaign_training_step(context, model, optimizer, step=0)


def test_training_executor_rejects_optimizer_schema_and_class_mutations(
    plan, monkeypatch
) -> None:
    context = next(item for item in plan if item.model_id == "gru-control")
    model = build_campaign_source_model(context)

    optimizer = build_campaign_optimizer(context, model)
    optimizer.defaults["lr"] = 999.0
    with pytest.raises(ValueError, match="defaults"):
        run_campaign_training_step(context, model, optimizer, step=0)

    optimizer = build_campaign_optimizer(context, model)
    optimizer.param_groups[0]["decoupled_weight_decay"] = False
    with pytest.raises(ValueError, match="hyperparameters"):
        run_campaign_training_step(context, model, optimizer, step=0)

    optimizer = build_campaign_optimizer(context, model)
    optimizer.state.default_factory = lambda: {}  # type: ignore[assignment]
    with pytest.raises(TypeError, match="defaultdict"):
        run_campaign_training_step(context, model, optimizer, step=0)

    optimizer = build_campaign_optimizer(context, model)
    optimizer.state = {}  # type: ignore[assignment]
    with pytest.raises(TypeError, match="defaultdict"):
        run_campaign_training_step(context, model, optimizer, step=0)

    class SneakyFloat(float):
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

    optimizer = build_campaign_optimizer(context, model)
    optimizer.param_groups[0]["lr"] = SneakyFloat(1000.0)
    optimizer.defaults["lr"] = SneakyFloat(1000.0)
    with pytest.raises(ValueError, match="hyperparameters"):
        run_campaign_training_step(context, model, optimizer, step=0)

    optimizer = build_campaign_optimizer(context, model)
    monkeypatch.setattr(torch.optim.AdamW, "zero_grad", lambda self, *a, **k: None)
    with pytest.raises(ValueError, match="zero_grad"):
        run_campaign_training_step(context, model, optimizer, step=0)


def test_execution_context_rejects_nested_subclasses_and_mutable_architecture(plan) -> None:
    context = next(item for item in plan if item.model_id == "gru-control")
    from tnlm_v3.campaign_config import TrainDataSpec

    class EvilTrain(TrainDataSpec):
        pass

    forged = copy.deepcopy(context.config)
    object.__setattr__(
        forged.data,
        "train",
        EvilTrain(**vars(forged.data.train)),
    )
    with pytest.raises(TypeError, match="exact type"):
        CampaignRunContext(forged, context.run)

    forged = copy.deepcopy(context.config)
    object.__setattr__(forged.models[0], "architecture", list(forged.models[0].architecture))
    with pytest.raises(TypeError, match="tuple"):
        CampaignRunContext(forged, context.run)


def test_model_parameter_subclass_is_rejected(plan) -> None:
    context = next(item for item in plan if item.model_id == "gru-control")
    model = build_campaign_source_model(context)

    class EvilParameter(torch.nn.Parameter):
        pass

    model.readout.weight = EvilParameter(model.readout.weight.detach().clone())
    with pytest.raises(TypeError, match="exact torch.nn.Parameter"):
        build_campaign_optimizer(context, model)


def test_optimizer_state_dict_subclass_is_rejected(plan) -> None:
    context = next(item for item in plan if item.model_id == "gru-control")
    model = build_campaign_source_model(context)
    optimizer = build_campaign_optimizer(context, model)
    run_campaign_training_step(context, model, optimizer, step=0)
    parameter = next(iter(model.parameters()))

    class EvilState(dict):
        pass

    optimizer.state[parameter] = EvilState(optimizer.state[parameter])
    with pytest.raises(ValueError, match="state schema"):
        run_campaign_training_step(context, model, optimizer, step=1)
