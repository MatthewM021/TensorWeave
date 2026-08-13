from __future__ import annotations

import pytest
import torch

from tnlm_v3.routing import (
    NULL_ROUTE,
    CurriculumSchedule,
    PersistentCausalRouter,
    RoutingMode,
    deterministic_guidance_mask,
    permute_local_routes,
)


def make_features(length: int = 7, dim: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(83017)
    features = torch.randn(2, length, dim, generator=generator)
    valid = torch.ones(2, length, dtype=torch.bool)
    return features, valid


def assert_state_equal(left, right) -> None:
    for name in (
        "prototypes",
        "occupied",
        "ages",
        "loads",
        "global_state",
        "global_occupied",
        "global_load",
        "valid_steps",
    ):
        assert torch.equal(getattr(left, name), getattr(right, name)), name


def test_schedule_records_and_reaches_declared_endpoints() -> None:
    schedule = CurriculumSchedule(10, 30, 0.9, 0.1)
    assert schedule.probability(0) == pytest.approx(0.9)
    assert schedule.probability(10) == pytest.approx(0.9)
    assert schedule.probability(20) == pytest.approx(0.5)
    assert schedule.probability(30) == pytest.approx(0.1)
    assert schedule.probability(100) == pytest.approx(0.1)
    assert schedule.endpoints == {
        "start_step": 10,
        "end_step": 30,
        "start_probability": 0.9,
        "end_probability": 0.1,
    }
    with pytest.raises(ValueError, match="must not increase"):
        CurriculumSchedule(0, 10, 0.0, 1.0)


@pytest.mark.parametrize(
    ("name", "value"),
    (("start_probability", True), ("end_probability", "0.0")),
)
def test_schedule_rejects_coerced_probability_scalars(name, value) -> None:
    values = {
        "start_step": 0,
        "end_step": 10,
        "start_probability": 1.0,
        "end_probability": 0.0,
    }
    values[name] = value
    with pytest.raises(TypeError, match="finite real number"):
        CurriculumSchedule(**values)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_schedule_rejects_nonfinite_probabilities(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        CurriculumSchedule(0, 10, value, 0.0)


def test_guidance_mask_is_deterministic_and_independent_of_data_or_labels() -> None:
    valid = torch.tensor([[True, True, False, True], [True, True, True, True]])
    first = deterministic_guidance_mask(valid, 0.4, training_step=8, seed=91)
    second = deterministic_guidance_mask(valid, 0.4, training_step=8, seed=91)
    assert torch.equal(first, second)
    assert not bool(first[~valid].any())
    assert deterministic_guidance_mask(valid, 0, training_step=8).sum() == 0
    assert torch.equal(
        deterministic_guidance_mask(valid, 1, training_step=8), valid
    )


def test_guidance_hash_has_event_level_mix_without_long_affine_runs() -> None:
    valid = torch.ones(32, 64, dtype=torch.bool)
    mask = deterministic_guidance_mask(
        valid, 0.5, training_step=7, seed=55
    )
    fraction = float(mask.to(torch.float32).mean())
    adjacent_agreement = float((mask[:, 1:] == mask[:, :-1]).to(torch.float32).mean())
    assert 0.4 < fraction < 0.6
    assert adjacent_agreement < 0.7


def test_oracle_requires_labels_and_routes_exactly() -> None:
    router = PersistentCausalRouter(5, 3, mode=RoutingMode.ORACLE)
    features, valid = make_features()
    valid[0, -1] = False
    labels = torch.tensor(
        [[0, 1, 3, NULL_ROUTE, 2, 0, 999], [2, 1, 0, 3, NULL_ROUTE, 1, 2]]
    )
    with pytest.raises(ValueError, match="requires route labels"):
        router(features, valid)
    output = router(features, valid, route_labels=labels)
    expected = torch.where(valid, labels, torch.full_like(labels, NULL_ROUTE))
    assert torch.equal(output.routes, expected)
    assert torch.equal(output.diagnostics["guidance_mask"], valid)


def test_latent_mode_rejects_labels() -> None:
    router = PersistentCausalRouter(5, 3, mode="latent")
    features, valid = make_features()
    with pytest.raises(ValueError, match="must not receive"):
        router(features, valid, route_labels=torch.zeros_like(valid, dtype=torch.int64))


def test_curriculum_training_is_reproducible_and_honors_endpoints() -> None:
    schedule = CurriculumSchedule(0, 10, 1.0, 0.0)
    router = PersistentCausalRouter(
        5,
        3,
        mode="curriculum",
        curriculum_schedule=schedule,
        curriculum_seed=713,
    )
    router.train()
    features, valid = make_features()
    labels = torch.tensor([[0, 1, 2, 3, -1, 0, 1], [2, 1, 0, 3, -1, 2, 0]])
    guided = router(features, valid, route_labels=labels, training_step=0)
    repeated = router(features, valid, route_labels=labels, training_step=0)
    assert torch.equal(guided.routes, labels)
    assert torch.equal(guided.routes, repeated.routes)
    assert_state_equal(guided.final_state, repeated.final_state)

    autonomous = router(features, valid, route_labels=labels, training_step=10)
    assert torch.equal(
        autonomous.routes, autonomous.diagnostics["autonomous_routes"]
    )
    assert autonomous.diagnostics["guided_events"].item() == 0


def test_curriculum_guidance_and_state_are_chunk_resume_equivalent() -> None:
    router = PersistentCausalRouter(
        5,
        3,
        mode="curriculum",
        curriculum_schedule=CurriculumSchedule(0, 10, 0.4, 0.4),
        curriculum_seed=713,
    )
    router.train()
    features, valid = make_features(length=9)
    valid[0, 2] = False
    labels = torch.tensor(
        [[0, 1, 999, 2, 3, -1, 0, 1, 2], [2, 1, 0, 3, -1, 2, 0, 1, 2]]
    )
    full = router(features, valid, route_labels=labels, training_step=4)
    first = router(
        features[:, :4], valid[:, :4], route_labels=labels[:, :4], training_step=4
    )
    second = router(
        features[:, 4:],
        valid[:, 4:],
        route_labels=labels[:, 4:],
        training_step=4,
        initial_state=first.final_state,
    )
    assert torch.equal(
        full.diagnostics["guidance_mask"],
        torch.cat(
            (first.diagnostics["guidance_mask"], second.diagnostics["guidance_mask"]),
            dim=1,
        ),
    )
    assert torch.equal(full.routes, torch.cat((first.routes, second.routes), dim=1))
    assert_state_equal(full.final_state, second.final_state)


def test_curriculum_evaluation_ignores_labels_completely() -> None:
    router = PersistentCausalRouter(
        5,
        3,
        mode="curriculum",
        curriculum_schedule=CurriculumSchedule(0, 20),
    )
    router.eval()
    features, valid = make_features()
    labels_a = torch.zeros_like(valid, dtype=torch.int64)
    labels_b = torch.full_like(labels_a, 999_999)  # intentionally invalid
    first = router(features, valid, route_labels=labels_a, training_step=0)
    second = router(features, valid, route_labels=labels_b, training_step=0)
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.probabilities, second.probabilities)
    assert torch.equal(first.routes, second.routes)
    assert_state_equal(first.final_state, second.final_state)


def test_future_changes_cannot_change_prefix_logits_routes_or_state() -> None:
    router = PersistentCausalRouter(5, 3, mode="latent")
    router.eval()
    features, valid = make_features(length=8)
    changed = features.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 100
    first = router(features, valid)
    second = router(changed, valid)
    assert torch.equal(first.logits[:, :4], second.logits[:, :4])
    assert torch.equal(first.routes[:, :4], second.routes[:, :4])

    prefix = router(features[:, :4], valid[:, :4])
    assert_state_equal(prefix.final_state, router(changed[:, :4], valid[:, :4]).final_state)


def test_padding_is_total_noop_even_with_nonfinite_features() -> None:
    router = PersistentCausalRouter(5, 3, mode="latent")
    initial = router.initial_state(2)
    features = torch.full((2, 4, 5), float("nan"), requires_grad=True)
    valid = torch.zeros(2, 4, dtype=torch.bool)
    output = router(features, valid, initial_state=initial)
    assert_state_equal(output.final_state, initial)
    assert torch.equal(output.logits, torch.zeros_like(output.logits))
    assert torch.equal(output.probabilities, torch.zeros_like(output.probabilities))
    assert torch.equal(output.routes, torch.full_like(output.routes, NULL_ROUTE))
    (output.logits.sum() + output.probabilities.sum()).backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in router.parameters()
    )


def test_nonfinite_valid_route_features_are_rejected() -> None:
    router = PersistentCausalRouter(5, 3, mode="latent")
    features = torch.zeros(1, 2, 5)
    features[0, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        router(features, torch.ones(1, 2, dtype=torch.bool))


def test_decision_precedes_prototype_update() -> None:
    router = PersistentCausalRouter(4, 2, mode="latent")
    router.eval()
    feature = torch.randn(1, 1, 4)
    initial = router.initial_state(1)
    first = router(feature, torch.ones(1, 1, dtype=torch.bool), initial_state=initial)
    with torch.no_grad():
        expected_logits = router._score(feature[:, 0], initial)
    assert torch.equal(first.logits[:, 0], expected_logits)


def test_branch_permutation_equivariance_and_route_mapping() -> None:
    torch.manual_seed(29)
    router = PersistentCausalRouter(
        4, 3, mode="oracle", include_global=False, include_null=False
    )
    router.eval()
    seed_features = torch.randn(1, 3, 4)
    seed_labels = torch.tensor([[0, 1, 2]])
    state = router(
        seed_features, torch.ones(1, 3, dtype=torch.bool), route_labels=seed_labels
    ).final_state

    latent = PersistentCausalRouter(
        4, 3, mode="latent", include_global=False, include_null=False
    )
    latent.load_state_dict(router.state_dict())
    latent.eval()
    permutation = torch.tensor([2, 0, 1])
    feature = torch.randn(1, 1, 4)
    valid = torch.ones(1, 1, dtype=torch.bool)
    original = latent(feature, valid, initial_state=state)
    permuted = latent(feature, valid, initial_state=state.permute_branches(permutation))
    torch.testing.assert_close(
        permuted.logits[..., :3], original.logits[..., :3].index_select(-1, permutation)
    )
    expected_route = permute_local_routes(original.routes, permutation, branches=3)
    assert torch.equal(permuted.routes, expected_route)
    expected_state = original.final_state.permute_branches(permutation)
    assert_state_equal(permuted.final_state, expected_state)


def test_parameters_are_length_and_branch_count_independent() -> None:
    short = PersistentCausalRouter(6, 2, hidden_dim=7)
    wide = PersistentCausalRouter(6, 9, hidden_dim=7)
    short_shapes = [(name, tuple(value.shape)) for name, value in short.named_parameters()]
    wide_shapes = [(name, tuple(value.shape)) for name, value in wide.named_parameters()]
    assert short_shapes == wide_shapes
    assert all("branch_embedding" not in name for name, _ in short_shapes)

    parameter_count = sum(parameter.numel() for parameter in short.parameters())
    features = torch.randn(2, 19, 6)
    short(features, torch.ones(2, 19, dtype=torch.bool))
    assert sum(parameter.numel() for parameter in short.parameters()) == parameter_count

    with pytest.raises(ValueError, match="finite"):
        PersistentCausalRouter(6, 2, temperature=float("inf"))


def test_symmetric_hard_tie_uses_canonical_lowest_local_index() -> None:
    router = PersistentCausalRouter(
        4, 2, mode="latent", include_global=False, include_null=False
    ).eval()
    with torch.no_grad():
        for parameter in router.parameters():
            parameter.zero_()
    output = router(torch.zeros(1, 1, 4), torch.ones(1, 1, dtype=torch.bool))
    assert torch.equal(output.logits[..., 0], output.logits[..., 1])
    assert output.routes.item() == 0


def test_branch_score_work_proxy_is_exactly_ntb() -> None:
    router = PersistentCausalRouter(4, 5, mode="latent")
    features = torch.randn(3, 11, 4)
    valid = torch.rand(3, 11) > 0.4
    output = router(features, valid)
    assert output.diagnostics["branch_score_work"].item() == 3 * 11 * 5


def test_only_selected_branch_updates_and_null_is_read_only() -> None:
    router = PersistentCausalRouter(4, 3, mode="oracle")
    features = torch.randn(1, 3, 4)
    labels = torch.tensor([[2, NULL_ROUTE, 3]])
    output = router(
        features, torch.ones(1, 3, dtype=torch.bool), route_labels=labels
    )
    assert output.final_state.loads.tolist() == [[0, 0, 1]]
    assert output.final_state.valid_steps.tolist() == [3]
    assert output.final_state.global_load.tolist() == [2]
    assert torch.equal(output.final_state.prototypes[0, 2], features[0, 0])


def test_router_has_no_history_or_token_lookup_state() -> None:
    router = PersistentCausalRouter(8, 4)
    assert not hasattr(router, "history")
    assert not hasattr(router, "token_embedding")
    state = router.initial_state(2)
    tensor_scalars = sum(value.numel() for value in state.__dict__.values())
    assert tensor_scalars == 2 * (4 * 8 + 4 + 4 + 4 + 8 + 1 + 1 + 1)


def test_state_load_validation_cannot_overflow_int64() -> None:
    router = PersistentCausalRouter(4, 2, mode="latent")
    state = router.initial_state(1)
    maximum = torch.iinfo(torch.int64).max
    state.prototypes.fill_(1)
    state.occupied.fill_(True)
    state.loads.fill_(maximum)
    state.global_state.fill_(1)
    state.global_occupied.fill_(True)
    state.global_load.fill_(maximum)
    state.valid_steps.fill_(maximum)

    with pytest.raises(ValueError, match="local loads"):
        router(
            torch.empty(1, 0, 4),
            torch.empty(1, 0, dtype=torch.bool),
            initial_state=state,
        )


def test_valid_state_update_cannot_overflow_int64_counters() -> None:
    router = PersistentCausalRouter(4, 1, mode="oracle")
    state = router.initial_state(1)
    maximum = torch.iinfo(torch.int64).max
    state.prototypes.fill_(1)
    state.occupied.fill_(True)
    state.loads.fill_(maximum)
    state.global_state.fill_(1)
    state.global_occupied.fill_(True)
    state.global_load.fill_(maximum)
    state.valid_steps.fill_(maximum)

    with pytest.raises(OverflowError, match="overflow"):
        router(
            torch.ones(1, 1, 4),
            torch.ones(1, 1, dtype=torch.bool),
            route_labels=torch.zeros(1, 1, dtype=torch.int64),
            initial_state=state,
        )


def test_state_validation_rejects_nonfinite_summaries() -> None:
    router = PersistentCausalRouter(4, 2, mode="latent")
    state = router.initial_state(1)
    state.global_state[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        router(
            torch.empty(1, 0, 4),
            torch.empty(1, 0, dtype=torch.bool),
            initial_state=state,
        )


def test_state_validation_rejects_impossible_branch_age() -> None:
    router = PersistentCausalRouter(4, 2, mode="latent")
    state = router.initial_state(1)
    state.prototypes[0, 0].fill_(1)
    state.occupied[0, 0] = True
    state.loads[0, 0] = 1
    state.ages[0, 0] = 1
    state.global_state.fill_(1)
    state.global_occupied.fill_(True)
    state.global_load.fill_(1)
    state.valid_steps.fill_(1)
    with pytest.raises(ValueError, match="ages"):
        router(
            torch.empty(1, 0, 4),
            torch.empty(1, 0, dtype=torch.bool),
            initial_state=state,
        )


def test_model_visible_class_constraints_mask_routes_and_probabilities() -> None:
    torch.manual_seed(14)
    router = PersistentCausalRouter(4, 3, mode="latent")
    features = torch.randn(2, 5, 4)
    valid = torch.ones(2, 5, dtype=torch.bool)
    allowed = torch.zeros(2, 5, router.class_count, dtype=torch.bool)
    allowed[:, :, :3] = True
    output = router(features, valid, allowed_classes=allowed)
    assert torch.all((output.routes >= 0) & (output.routes < 3))
    assert torch.equal(
        output.probabilities[:, :, 3:],
        torch.zeros_like(output.probabilities[:, :, 3:]),
    )


def test_class_constraint_is_absolute_at_extreme_finite_logits() -> None:
    router = PersistentCausalRouter(4, 2, mode="latent")
    with torch.no_grad():
        for parameter in router.parameters():
            parameter.zero_()
        router.branch_scorer[-1].bias.fill_(-torch.finfo(torch.float32).max)
        assert router.global_scorer is not None and router.null_scorer is not None
        router.global_scorer[-1].bias.fill_(-torch.finfo(torch.float32).max)
        router.null_scorer[-1].bias.fill_(-torch.finfo(torch.float32).max)
    allowed = torch.tensor([[[False, False, True, True]]])
    output = router(torch.zeros(1, 1, 4), torch.ones(1, 1, dtype=torch.bool), allowed_classes=allowed)
    selected_class = output.probabilities.argmax(dim=-1)
    assert bool(allowed.gather(-1, selected_class.unsqueeze(-1)).all())
    assert output.routes.item() in (router.branches, NULL_ROUTE)


def test_oracle_label_must_obey_model_visible_class_constraint() -> None:
    router = PersistentCausalRouter(4, 3, mode="oracle")
    features = torch.randn(1, 2, 4)
    valid = torch.ones(1, 2, dtype=torch.bool)
    allowed = torch.zeros(1, 2, router.class_count, dtype=torch.bool)
    allowed[:, :, :3] = True
    with pytest.raises(ValueError, match="disallowed"):
        router(
            features,
            valid,
            route_labels=torch.tensor([[0, NULL_ROUTE]]),
            allowed_classes=allowed,
        )


def test_class_constraint_ignores_out_of_range_padding_labels() -> None:
    router = PersistentCausalRouter(4, 3, mode="oracle")
    features = torch.randn(1, 2, 4)
    valid = torch.tensor([[True, False]])
    allowed = torch.zeros(1, 2, router.class_count, dtype=torch.bool)
    allowed[0, 0, :3] = True
    output = router(
        features,
        valid,
        route_labels=torch.tensor([[0, 987_654_321]]),
        allowed_classes=allowed,
    )
    assert output.routes.tolist() == [[0, NULL_ROUTE]]
