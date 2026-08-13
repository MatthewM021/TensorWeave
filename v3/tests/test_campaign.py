from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass, replace
import math

import pytest
import torch
from torch import Tensor
import torch.nn.functional as F

from tnlm_v3.baselines import (
    BaselineBindingOutput,
    CachedCausalTransformerBindingBaseline,
    CachedTransformerBindingBaselineConfig,
    CachedTransformerBindingState,
    RecurrentBindingBaseline,
    RecurrentBindingBaselineConfig,
    RecurrentBindingState,
)
from tnlm_v3.binding import (
    BindingArchitectureConfig,
    BindingModelConfig,
    RoutedBindingModel,
)
from tnlm_v3.campaign import (
    BaselineCampaignEvaluationSummary,
    BaselineQueryLoss,
    baseline_structural_metrics,
    compute_baseline_query_loss,
    evaluate_baseline_model,
    train_baseline_step,
)
from tnlm_v3.causal_ttn import (
    CausalCompleteTreeBindingBaseline,
    CausalTreeBindingBaselineConfig,
    CausalTreeBindingState,
)
from tnlm_v3.data import (
    BindingBatch,
    BindingEvaluation,
    BindingEventKind,
    BindingModelInputs,
    BindingTaskConfig,
    collate_binding_episodes,
    generate_binding_episodes,
)
from tnlm_v3.routing import RoutingMode


def task() -> BindingTaskConfig:
    return BindingTaskConfig(
        num_surface_keys=5,
        value_cardinality=4,
        branches=3,
        max_live_bindings=3,
        min_length=10,
        max_length=14,
        heldout_key_value_pairs=((0, 0),),
        global_distractor_probability=0.5,
    )


def binding_batch(*, split: str = "eval") -> BindingBatch:
    return collate_binding_episodes(
        generate_binding_episodes(
            task(),
            count=2,
            seed=41_771,
            split=split,
            lengths=(14, 11),
        )
    )


def make_model(kind: str) -> torch.nn.Module:
    architecture = BindingArchitectureConfig.from_task(task())
    torch.manual_seed(713)
    if kind == "gru":
        return RecurrentBindingBaseline(
            RecurrentBindingBaselineConfig(
                architecture, d_model=8, hidden_dim=11, num_layers=2
            )
        ).double()
    if kind == "transformer":
        return CachedCausalTransformerBindingBaseline(
            CachedTransformerBindingBaselineConfig(
                architecture,
                d_model=8,
                num_heads=2,
                num_layers=2,
                ff_dim=13,
            )
        ).double()
    if kind == "causal_tree":
        return CausalCompleteTreeBindingBaseline(
            CausalTreeBindingBaselineConfig(
                architecture,
                d_model=8,
                cp_rank=5,
                scale_feature_dim=4,
            )
        ).double()
    raise AssertionError(f"unknown fixture kind {kind}")


MODEL_KINDS = ("gru", "transformer", "causal_tree")


def empty_batch() -> BindingBatch:
    shape = (2, 0)
    inputs = BindingModelInputs(
        token_ids=torch.empty(shape, dtype=torch.int64),
        event_kinds=torch.empty(shape, dtype=torch.int64),
        primary_key_ids=torch.empty(shape, dtype=torch.int64),
        secondary_key_ids=torch.empty(shape, dtype=torch.int64),
        arguments=torch.empty(shape, dtype=torch.int64),
        valid_mask=torch.empty(shape, dtype=torch.bool),
    )
    evaluation = BindingEvaluation(
        oracle_routes=torch.empty(shape, dtype=torch.int64),
        targets=torch.empty(shape, dtype=torch.int64),
        dependency_parents=torch.empty((2, 0, 2), dtype=torch.int64),
        generation_ids=torch.empty(shape, dtype=torch.int64),
        live_binding_counts=torch.empty(shape, dtype=torch.int64),
        heldout_combination_mask=torch.empty(shape, dtype=torch.bool),
    )
    return BindingBatch(
        inputs=inputs,
        evaluation=evaluation,
        lengths=torch.zeros(2, dtype=torch.int64),
        splits=("eval", "eval"),
        document_ids=("empty-0", "empty-1"),
        generation_seeds=(0, 1),
        config_fingerprint=task().fingerprint(),
    )


def assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, Tensor):
        assert isinstance(right, Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def state_tensors(value: object, seen: set[int] | None = None) -> list[Tensor]:
    seen = set() if seen is None else seen
    if isinstance(value, Tensor):
        if id(value) in seen:
            return []
        seen.add(id(value))
        return [value]
    if is_dataclass(value) and not isinstance(value, type):
        result: list[Tensor] = []
        for field in fields(value):
            result.extend(state_tensors(getattr(value, field.name), seen))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(state_tensors(item, seen))
        return result
    raise AssertionError(f"unexpected state field {type(value)!r}")


def test_query_loss_masks_nonqueries_and_padding() -> None:
    inputs = BindingModelInputs(
        token_ids=torch.ones(1, 4, dtype=torch.int64),
        event_kinds=torch.tensor(
            [[
                int(BindingEventKind.QUERY),
                int(BindingEventKind.UPDATE),
                int(BindingEventKind.QUERY),
                int(BindingEventKind.QUERY),
            ]]
        ),
        primary_key_ids=torch.ones(1, 4, dtype=torch.int64),
        secondary_key_ids=torch.zeros(1, 4, dtype=torch.int64),
        arguments=torch.zeros(1, 4, dtype=torch.int64),
        valid_mask=torch.tensor([[True, True, False, True]]),
    )
    logits = torch.tensor(
        [[[0.2, 1.2], [99.0, -99.0], [50.0, -50.0], [1.7, 0.1]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor([[1, -100, -100, 0]])
    output = BaselineBindingOutput(logits, None, {})

    loss = compute_baseline_query_loss(output, inputs, targets)

    expected = F.cross_entropy(logits[:, (0, 3)].reshape(2, 2), torch.tensor([1, 0]))
    assert isinstance(loss, BaselineQueryLoss)
    assert loss.query_count == 2
    torch.testing.assert_close(loss.total, expected, rtol=0, atol=0)


def test_query_loss_rejects_nonfinite_logits_and_invalid_selected_targets() -> None:
    batch = binding_batch()
    model = make_model("gru")
    output = model(batch.inputs)
    query = batch.inputs.valid_mask & (
        batch.inputs.event_kinds == int(BindingEventKind.QUERY)
    )
    row, column = query.nonzero()[0].tolist()

    nonfinite = output.value_logits.clone()
    nonfinite[row, column, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compute_baseline_query_loss(
            replace(output, value_logits=nonfinite),
            batch.inputs,
            batch.evaluation.targets,
        )

    targets = batch.evaluation.targets.clone()
    targets[row, column] = output.value_logits.shape[-1]
    with pytest.raises(ValueError, match="index"):
        compute_baseline_query_loss(output, batch.inputs, targets)


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_evaluation_reports_exact_overall_seen_and_heldout_metrics(kind: str) -> None:
    batch = binding_batch()
    model = make_model(kind)

    output, summary = evaluate_baseline_model(model, batch)

    assert isinstance(output, BaselineBindingOutput)
    assert isinstance(summary, BaselineCampaignEvaluationSummary)
    query = batch.inputs.valid_mask & (
        batch.inputs.event_kinds == int(BindingEventKind.QUERY)
    )
    masks = (
        query,
        query & ~batch.evaluation.heldout_combination_mask,
        query & batch.evaluation.heldout_combination_mask,
    )
    reported = (summary.query, summary.seen_query, summary.heldout_query)
    for mask, metrics in zip(masks, reported, strict=True):
        count = int(mask.sum())
        expected_loss = (
            float(
                F.cross_entropy(
                    output.value_logits[mask], batch.evaluation.targets[mask]
                )
            )
            if count
            else 0.0
        )
        expected_correct = int(
            (
                output.value_logits.argmax(-1)[mask]
                == batch.evaluation.targets[mask]
            ).sum()
        )
        assert metrics.query_count == count
        assert metrics.correct == expected_correct
        assert metrics.accuracy.accuracy == (
            expected_correct / count if count else 0.0
        )
        assert metrics.cross_entropy == pytest.approx(expected_loss)
        assert math.isfinite(metrics.cross_entropy)
    assert summary.query.query_count == (
        summary.seen_query.query_count + summary.heldout_query.query_count
    )
    assert not model.training


class OracleRouteAccessSpy:
    def __init__(self, evaluation: BindingEvaluation, forward_seen: dict[str, bool]):
        self._evaluation = evaluation
        self._forward_seen = forward_seen
        self.oracle_reads = 0
        self.target_reads = 0
        self.heldout_reads = 0

    @property
    def oracle_routes(self) -> Tensor:
        self.oracle_reads += 1
        raise AssertionError("baseline campaign accessed oracle routes")

    @property
    def targets(self) -> Tensor:
        assert self._forward_seen["value"]
        self.target_reads += 1
        return self._evaluation.targets

    @property
    def heldout_combination_mask(self) -> Tensor:
        assert self._forward_seen["value"]
        self.heldout_reads += 1
        return self._evaluation.heldout_combination_mask

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"baseline campaign accessed evaluation field {name}")


@pytest.mark.parametrize("operation", ("train", "evaluate"))
def test_forward_receives_only_inputs_and_never_reads_oracle_routes(
    operation: str,
) -> None:
    batch = binding_batch(split="train")
    model = make_model("gru")
    forward_seen = {"value": False}
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original_forward = model.forward

    def recording_forward(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        assert len(args) == 1 and args[0] is batch.inputs
        assert kwargs == {}
        forward_seen["value"] = True
        return original_forward(*args, **kwargs)

    model.forward = recording_forward  # type: ignore[method-assign]
    spy = OracleRouteAccessSpy(batch.evaluation, forward_seen)
    guarded = replace(batch, evaluation=spy)
    if operation == "train":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        train_baseline_step(model, guarded, optimizer)
        assert spy.heldout_reads == 0
    else:
        evaluate_baseline_model(model, guarded)
        assert spy.heldout_reads == 1
    assert calls and spy.target_reads == 1
    assert spy.oracle_reads == 0


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_empty_sequence_evaluation_is_finite_and_explicit(kind: str) -> None:
    model = make_model(kind)
    output, summary = evaluate_baseline_model(model, empty_batch())
    assert output.value_logits.shape == (2, 0, task().value_cardinality)
    for metrics in (summary.query, summary.seen_query, summary.heldout_query):
        assert metrics.query_count == 0
        assert metrics.correct == 0
        assert metrics.accuracy.accuracy == 0.0
        assert metrics.cross_entropy == 0.0
    assert summary.structural_metrics["persistent_state_tensor_elements"] >= 2
    assert all(value >= 0 for value in summary.structural_metrics.values())


def test_empty_query_training_skips_backward_step_and_weight_decay() -> None:
    batch = binding_batch(split="train")
    model = make_model("gru")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-3, weight_decay=0.5
    )
    # Populate Adam's counters and moments before auditing the no-query path.
    train_baseline_step(model, batch, optimizer)
    no_queries = replace(
        batch,
        inputs=replace(
            batch.inputs,
            event_kinds=torch.full_like(
                batch.inputs.event_kinds, int(BindingEventKind.DISTRACTOR)
            ),
        ),
    )
    parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    _, loss = train_baseline_step(model, no_queries, optimizer)

    assert loss.query_count == 0
    assert loss.total.item() == 0.0 and torch.isfinite(loss.total)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, parameters[name])
    assert_nested_equal(optimizer.state_dict(), optimizer_state)


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_training_step_is_deterministic_for_all_baselines(kind: str) -> None:
    batch = binding_batch(split="train")
    left = make_model(kind)
    right = copy.deepcopy(left)
    left_optimizer = torch.optim.AdamW(left.parameters(), lr=7.0e-4)
    right_optimizer = torch.optim.AdamW(right.parameters(), lr=7.0e-4)

    left_output, left_loss = train_baseline_step(
        left, batch, left_optimizer, max_gradient_norm=1.5
    )
    right_output, right_loss = train_baseline_step(
        right, batch, right_optimizer, max_gradient_norm=1.5
    )

    assert torch.equal(left_output.value_logits, right_output.value_logits)
    assert torch.equal(left_loss.total, right_loss.total)
    assert left_loss.query_count == right_loss.query_count > 0
    for left_parameter, right_parameter in zip(
        left.parameters(), right.parameters(), strict=True
    ):
        assert torch.equal(left_parameter, right_parameter)
    assert_nested_equal(left_optimizer.state_dict(), right_optimizer.state_dict())


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_structural_metrics_count_exact_persistent_state_storage(kind: str) -> None:
    batch = binding_batch()
    model = make_model(kind)
    output = model(batch.inputs)

    metrics = baseline_structural_metrics(model, batch.inputs, output)
    tensors = state_tensors(output.final_state)
    expected_elements = sum(tensor.numel() for tensor in tensors)
    expected_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in tensors
    )

    assert metrics["parameter_count"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert metrics["trainable_parameter_count"] == metrics["parameter_count"]
    assert metrics["persistent_state_tensor_count"] == len(tensors)
    assert metrics["persistent_state_tensor_elements"] == expected_elements
    assert metrics["persistent_state_bytes"] == expected_bytes
    assert metrics["persistent_state_floating_elements"] + metrics[
        "persistent_state_nonfloating_elements"
    ] == expected_elements


def test_output_contract_rejects_nonfinite_and_nonzero_padding() -> None:
    batch = binding_batch()
    model = make_model("gru")
    original_forward = model.forward

    def nonfinite_forward(inputs: BindingModelInputs) -> object:
        output = original_forward(inputs)
        logits = output.value_logits.clone()
        logits[0, 0, 0] = float("inf")
        return replace(output, value_logits=logits)

    model.forward = nonfinite_forward  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="finite"):
        evaluate_baseline_model(model, batch)

    def padded_forward(inputs: BindingModelInputs) -> object:
        output = original_forward(inputs)
        logits = output.value_logits.clone()
        row, column = (~inputs.valid_mask).nonzero()[0].tolist()
        logits[row, column, 0] = 1.0
        return replace(output, value_logits=logits)

    model.forward = padded_forward  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="Padded|padded"):
        evaluate_baseline_model(model, batch)


def test_nonfinite_gradient_is_rejected_before_optimizer_step() -> None:
    batch = binding_batch(split="train")
    model = make_model("gru")
    parameter = next(model.parameters())
    before = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    handle = parameter.register_hook(
        lambda gradient: torch.full_like(gradient, float("nan"))
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    try:
        with pytest.raises(ValueError, match="gradients must be finite"):
            train_baseline_step(model, batch, optimizer)
    finally:
        handle.remove()
    for name, value in model.named_parameters():
        assert torch.equal(value, before[name])
    assert optimizer.state == {}


def test_routed_forest_is_kept_on_the_existing_routing_aware_path() -> None:
    batch = binding_batch()
    routed = RoutedBindingModel(
        BindingModelConfig(
            task=task(),
            d_model=8,
            cp_rank=4,
            router_hidden_dim=7,
            routing_mode=RoutingMode.LATENT,
        )
    )
    with pytest.raises(TypeError, match="baseline"):
        evaluate_baseline_model(routed, batch)


def test_subclass_cannot_bypass_the_audited_model_boundary() -> None:
    class CapturingBaseline(RecurrentBindingBaseline):
        pass

    architecture = BindingArchitectureConfig.from_task(task())
    model = CapturingBaseline(
        RecurrentBindingBaselineConfig(
            architecture, d_model=8, hidden_dim=11, num_layers=2
        )
    ).double()
    calls = 0
    original_forward = model.forward

    def recording_forward(inputs: BindingModelInputs) -> object:
        nonlocal calls
        calls += 1
        return original_forward(inputs)

    model.forward = recording_forward  # type: ignore[method-assign]
    with pytest.raises(TypeError, match="baseline"):
        evaluate_baseline_model(model, binding_batch())
    assert calls == 0


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_structural_metrics_reject_malformed_native_state(kind: str) -> None:
    batch = binding_batch()
    model = make_model(kind)
    output = model(batch.inputs)
    state = output.final_state
    if kind == "gru":
        malformed = RecurrentBindingState(
            hidden=state.hidden[:, :, :-1],
            valid_steps=state.valid_steps,
        )
    elif kind == "transformer":
        malformed = CachedTransformerBindingState(
            keys=state.keys,
            values=state.values,
            occupied=state.occupied,
            valid_steps=state.valid_steps - 1,
        )
    else:
        malformed = CausalTreeBindingState(
            slots=state.slots,
            occupied=~state.occupied,
            valid_steps=state.valid_steps,
        )
    with pytest.raises((TypeError, ValueError)):
        baseline_structural_metrics(
            model, batch.inputs, replace(output, final_state=malformed)
        )


def test_tree_structural_work_rejects_forged_diagnostics() -> None:
    batch = binding_batch()
    model = make_model("causal_tree")
    output = model(batch.inputs)
    diagnostics = dict(output.diagnostics)
    diagnostics["update_merge_count"] = torch.zeros_like(
        diagnostics["update_merge_count"]
    )
    diagnostics["readout_merge_count"] = torch.zeros_like(
        diagnostics["readout_merge_count"]
    )
    with pytest.raises(ValueError, match="merge diagnostic"):
        baseline_structural_metrics(
            model, batch.inputs, replace(output, diagnostics=diagnostics)
        )


@pytest.mark.parametrize("invalid", (True, 0.0, -1.0, float("inf"), float("nan")))
def test_training_options_are_strict_and_fail_before_forward(invalid: object) -> None:
    batch = binding_batch(split="train")
    model = make_model("gru")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    calls = 0
    original_forward = model.forward

    def recording_forward(inputs: BindingModelInputs) -> object:
        nonlocal calls
        calls += 1
        return original_forward(inputs)

    model.forward = recording_forward  # type: ignore[method-assign]
    with pytest.raises((TypeError, ValueError)):
        train_baseline_step(
            model,
            batch,
            optimizer,
            max_gradient_norm=invalid,  # type: ignore[arg-type]
        )
    assert calls == 0
