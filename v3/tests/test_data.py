from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from tnlm_v3.data import (
    BindingEvaluation,
    BindingEventKind,
    BindingModelInputs,
    BindingTaskConfig,
    IGNORE_QUERY_TARGET,
    NO_GENERATION,
    NO_PARENT,
    PAD_TOKEN_ID,
    collate_binding_episodes,
    generate_binding_episode,
    generate_binding_episodes,
    validate_binding_batch,
    validate_binding_episode,
)
from tnlm_v3.routing import NULL_ROUTE


def make_config(**overrides) -> BindingTaskConfig:
    settings = dict(
        num_surface_keys=6,
        value_cardinality=5,
        branches=4,
        max_live_bindings=3,
        min_length=10,
        max_length=31,
        heldout_key_value_pairs=((0, 0), (1, 2)),
    )
    settings.update(overrides)
    return BindingTaskConfig(**settings)


def test_task_config_rejects_coerced_probability_scalars():
    with pytest.raises(TypeError, match="real number"):
        make_config(global_distractor_probability="0.5")
    with pytest.raises(TypeError, match="real number"):
        make_config(global_distractor_probability=True)


def assert_episode_equal(left, right):
    assert left.split == right.split
    assert left.document_id == right.document_id
    assert left.generation_seed == right.generation_seed
    assert left.config_fingerprint == right.config_fingerprint
    for field in fields(BindingModelInputs):
        assert torch.equal(
            getattr(left.inputs, field.name), getattr(right.inputs, field.name)
        )
    for field in fields(BindingEvaluation):
        assert torch.equal(
            getattr(left.evaluation, field.name),
            getattr(right.evaluation, field.name),
        )


def replay_episode_independently(episode, config):
    """Independent test oracle for values, lanes, generations, and parents."""

    live = {}
    last_invalidation = {}
    generations = set()
    max_live = 0
    for index in range(episode.length):
        kind = BindingEventKind(int(episode.inputs.event_kinds[index]))
        key = int(episode.inputs.primary_key_ids[index]) - 1
        secondary = int(episode.inputs.secondary_key_ids[index]) - 1
        argument = int(episode.inputs.arguments[index]) - 1
        route = int(episode.evaluation.oracle_routes[index])
        target = int(episode.evaluation.targets[index])
        parent0, parent1 = episode.evaluation.dependency_parents[index].tolist()
        generation = int(episode.evaluation.generation_ids[index])

        if kind is BindingEventKind.BIND:
            assert key not in live
            assert 0 <= route < config.branches
            assert route not in {entry[0] for entry in live.values()}
            assert parent0 == last_invalidation.get(key, NO_PARENT)
            assert parent1 == NO_PARENT
            assert generation not in generations
            generations.add(generation)
            live[key] = [route, argument, generation, index]
        elif kind is BindingEventKind.UPDATE:
            lane, value, old_generation, last = live[key]
            assert (route, generation, parent0, parent1) == (
                lane,
                old_generation,
                last,
                NO_PARENT,
            )
            value = (value + argument + 1) % config.value_cardinality
            live[key] = [lane, value, generation, index]
        elif kind is BindingEventKind.COPY:
            lane, _, old_generation, last = live[key]
            source_lane, source_value, _, source_last = live[secondary]
            assert source_lane != lane
            assert (route, generation, parent0, parent1) == (
                lane,
                old_generation,
                last,
                source_last,
            )
            live[key] = [lane, source_value, generation, index]
        elif kind is BindingEventKind.INVALIDATE:
            lane, _, old_generation, last = live[key]
            assert (route, generation, parent0, parent1) == (
                lane,
                old_generation,
                last,
                NO_PARENT,
            )
            del live[key]
            last_invalidation[key] = index
        elif kind is BindingEventKind.QUERY:
            lane, value, old_generation, last = live[key]
            assert (route, generation, parent0, parent1) == (
                lane,
                old_generation,
                last,
                NO_PARENT,
            )
            assert target == value
        else:
            assert (key, secondary) == (-1, -1)
            assert argument in (0, 1)
            assert route == (config.branches if argument == 0 else NULL_ROUTE)
            assert (parent0, parent1, generation) == (
                NO_PARENT,
                NO_PARENT,
                NO_GENERATION,
            )

        if kind is not BindingEventKind.QUERY:
            assert target == IGNORE_QUERY_TARGET
        assert int(episode.evaluation.live_binding_counts[index]) == len(live)
        max_live = max(max_live, len(live))
    return max_live


def test_generation_is_reproducible_and_split_seeded():
    config = make_config()
    first = generate_binding_episode(
        config, length=27, seed=411, split="train", document_index=3
    )
    second = generate_binding_episode(
        config, length=27, seed=411, split="train", document_index=3
    )
    assert_episode_equal(first, second)

    changed = generate_binding_episode(
        config, length=27, seed=411, split="validation", document_index=3
    )
    assert changed.generation_seed != first.generation_seed
    assert not torch.equal(changed.inputs.token_ids, first.inputs.token_ids)


def test_every_event_family_and_exact_causal_targets_replay():
    config = make_config()
    episode = generate_binding_episode(config, length=31, seed=29, split="train")
    observed = set(map(BindingEventKind, episode.inputs.event_kinds.tolist()))
    assert observed == set(BindingEventKind) - {BindingEventKind.PAD}
    assert replay_episode_independently(episode, config) <= config.max_live_bindings


def test_nonlocal_distractor_route_is_visible_from_scope_bit():
    config = make_config(max_length=24)
    episodes = generate_binding_episodes(
        config, count=20, seed=991, split="train", lengths=[24] * 20
    )
    observed = set()
    for episode in episodes:
        mask = episode.inputs.event_kinds == int(BindingEventKind.DISTRACTOR)
        arguments = episode.inputs.arguments[mask] - 1
        routes = episode.evaluation.oracle_routes[mask]
        assert torch.equal(
            routes,
            torch.where(
                arguments == 0,
                torch.full_like(routes, config.branches),
                torch.full_like(routes, NULL_ROUTE),
            ),
        )
        observed.update(int(value) for value in arguments.tolist())
    assert observed == {0, 1}
    validate_binding_episode(episode, config)


def test_document_local_remapping_prevents_global_key_to_lane_lookup():
    config = make_config(max_length=16)
    episodes = generate_binding_episodes(
        config, count=24, seed=9001, split="train", lengths=[16] * 24
    )
    key_lanes = {key: set() for key in range(config.num_surface_keys)}
    query_token_lanes = {}
    for episode in episodes:
        for index in range(episode.length):
            kind = BindingEventKind(int(episode.inputs.event_kinds[index]))
            if kind in (
                BindingEventKind.BIND,
                BindingEventKind.UPDATE,
                BindingEventKind.COPY,
                BindingEventKind.INVALIDATE,
                BindingEventKind.QUERY,
            ):
                key = int(episode.inputs.primary_key_ids[index]) - 1
                lane = int(episode.evaluation.oracle_routes[index])
                key_lanes[key].add(lane)
                if kind is BindingEventKind.QUERY:
                    token = int(episode.inputs.token_ids[index])
                    query_token_lanes.setdefault(token, set()).add(lane)

    assert any(len(lanes) > 1 for lanes in key_lanes.values())
    assert any(len(lanes) > 1 for lanes in query_token_lanes.values())


def test_invalidation_allows_rebinding_with_new_generation_and_dependency():
    config = make_config()
    episode = generate_binding_episode(config, length=10, seed=71, split="train")
    kinds = episode.inputs.event_kinds
    bind_indices = torch.nonzero(
        kinds == int(BindingEventKind.BIND), as_tuple=False
    ).flatten()
    invalidation = int(
        torch.nonzero(
            kinds == int(BindingEventKind.INVALIDATE), as_tuple=False
        )[0]
    )
    invalidated_key = int(episode.inputs.primary_key_ids[invalidation])
    rebound = [
        int(index)
        for index in bind_indices
        if int(index) > invalidation
        and int(episode.inputs.primary_key_ids[index]) == invalidated_key
    ]
    assert len(rebound) == 1
    rebound_index = rebound[0]
    assert int(episode.evaluation.dependency_parents[rebound_index, 0]) == invalidation
    assert int(episode.evaluation.generation_ids[rebound_index]) != int(
        episode.evaluation.generation_ids[invalidation]
    )
    # Four branches leave a genuinely different free lane for the new generation.
    assert int(episode.evaluation.oracle_routes[rebound_index]) != int(
        episode.evaluation.oracle_routes[invalidation]
    )


def test_simultaneous_live_binding_cap_holds_under_long_random_continuation():
    config = make_config(max_length=128, max_live_bindings=2)
    for document_index in range(12):
        episode = generate_binding_episode(
            config,
            length=128,
            seed=82,
            split="train",
            document_index=document_index,
        )
        assert int(episode.evaluation.live_binding_counts.max()) <= 2
        assert replay_episode_independently(episode, config) <= 2


def test_variable_lengths_collate_with_strict_padding_contract():
    config = make_config(min_length=10, max_length=23)
    episodes = generate_binding_episodes(config, count=7, seed=55, split="train")
    assert len({episode.length for episode in episodes}) > 1
    batch = collate_binding_episodes(episodes, pad_to_length=25)
    assert batch.inputs.token_ids.shape == (7, 25)

    positions = torch.arange(25).unsqueeze(0)
    padding = positions >= batch.lengths.unsqueeze(1)
    assert torch.all(batch.inputs.token_ids[padding] == PAD_TOKEN_ID)
    assert torch.all(batch.inputs.event_kinds[padding] == int(BindingEventKind.PAD))
    assert torch.all(batch.inputs.primary_key_ids[padding] == 0)
    assert torch.all(batch.inputs.secondary_key_ids[padding] == 0)
    assert torch.all(batch.inputs.arguments[padding] == 0)
    assert torch.all(batch.evaluation.oracle_routes[padding] == NULL_ROUTE)
    assert torch.all(batch.evaluation.targets[padding] == IGNORE_QUERY_TARGET)
    assert torch.all(batch.evaluation.dependency_parents[padding] == NO_PARENT)
    assert torch.all(batch.evaluation.generation_ids[padding] == NO_GENERATION)
    assert torch.all(batch.evaluation.live_binding_counts[padding] == 0)
    assert not bool(batch.evaluation.heldout_combination_mask[padding].any())
    validate_binding_batch(batch, config)

    batch.evaluation.live_binding_counts[0, int(batch.lengths[0])] = 999
    with pytest.raises(ValueError, match="live-binding counts"):
        validate_binding_batch(batch, config)


def test_small_smoke_fixture_uses_document_local_branch_permutations():
    config = make_config(
        num_surface_keys=5,
        value_cardinality=4,
        branches=3,
        max_live_bindings=3,
        min_length=10,
        max_length=18,
    )
    episodes = generate_binding_episodes(
        config, count=4, seed=777, split="train", lengths=[10] * 4
    )
    first_binding_routes = {
        int(episode.evaluation.oracle_routes[0]) for episode in episodes
    }
    assert len(first_binding_routes) > 1


def test_dependency_graph_has_causal_primary_and_copy_secondary_parents():
    config = make_config()
    episode = generate_binding_episode(config, length=30, seed=101, split="train")
    parents = episode.evaluation.dependency_parents
    indices = torch.arange(episode.length).unsqueeze(1)
    assert torch.all((parents == NO_PARENT) | ((parents >= 0) & (parents < indices)))

    copy_mask = episode.inputs.event_kinds == int(BindingEventKind.COPY)
    assert bool(copy_mask.any())
    assert torch.all(parents[copy_mask, 0] >= 0)
    assert torch.all(parents[copy_mask, 1] >= 0)
    noncopy = ~copy_mask
    assert torch.all(parents[noncopy, 1] == NO_PARENT)


def test_train_excludes_and_eval_forces_heldout_symbol_value_combinations():
    config = make_config(max_length=20)
    train = generate_binding_episodes(
        config, count=10, seed=404, split="train", lengths=[20] * 10
    )
    evaluation = generate_binding_episodes(
        config, count=10, seed=404, split="eval", lengths=[20] * 10
    )
    assert not any(
        bool(episode.evaluation.heldout_combination_mask.any()) for episode in train
    )
    for episode in evaluation:
        mask = episode.evaluation.heldout_combination_mask
        assert bool(mask.any())
        assert bool(
            (
                mask
                & (episode.inputs.event_kinds == int(BindingEventKind.BIND))
            ).any()
        )
        assert bool(
            (
                mask
                & (episode.inputs.event_kinds == int(BindingEventKind.QUERY))
            ).any()
        )


def test_train_generation_skips_key_with_every_value_held_out():
    heldout = tuple((0, value) for value in range(4))
    config = make_config(
        num_surface_keys=5,
        value_cardinality=4,
        max_length=96,
        heldout_key_value_pairs=heldout,
    )
    episode = generate_binding_episode(
        config, length=96, seed=818, split="train", document_index=7
    )
    entity_mask = episode.inputs.primary_key_ids > 0
    assert not bool((episode.inputs.primary_key_ids[entity_mask] == 1).any())
    validate_binding_episode(episode, config)


def test_model_inputs_do_not_contain_oracle_routes_targets_or_dependencies():
    input_names = {field.name for field in fields(BindingModelInputs)}
    evaluation_names = {field.name for field in fields(BindingEvaluation)}
    protected = {"oracle_routes", "targets", "dependency_parents", "generation_ids"}
    assert input_names.isdisjoint(protected)
    assert protected.issubset(evaluation_names)

    config = make_config()
    episode = generate_binding_episode(config, length=18, seed=3, split="train")
    before = tuple(tensor.clone() for tensor in (
        episode.inputs.token_ids,
        episode.inputs.event_kinds,
        episode.inputs.primary_key_ids,
        episode.inputs.secondary_key_ids,
        episode.inputs.arguments,
        episode.inputs.valid_mask,
    ))
    episode.evaluation.oracle_routes.fill_(0)
    after = (
        episode.inputs.token_ids,
        episode.inputs.event_kinds,
        episode.inputs.primary_key_ids,
        episode.inputs.secondary_key_ids,
        episode.inputs.arguments,
        episode.inputs.valid_mask,
    )
    assert all(torch.equal(left, right) for left, right in zip(before, after))


def test_validation_rejects_wrong_query_target_and_noncausal_parent():
    config = make_config()
    episode = generate_binding_episode(config, length=16, seed=75, split="train")
    query = int(
        torch.nonzero(
            episode.inputs.event_kinds == int(BindingEventKind.QUERY),
            as_tuple=False,
        )[0]
    )
    bad_targets = episode.evaluation.targets.clone()
    bad_targets[query] = (bad_targets[query] + 1) % config.value_cardinality
    with pytest.raises(ValueError, match="query target"):
        validate_binding_episode(
            replace(
                episode,
                evaluation=replace(episode.evaluation, targets=bad_targets),
            ),
            config,
        )

    bad_parents = episode.evaluation.dependency_parents.clone()
    bad_parents[query, 0] = query
    with pytest.raises(ValueError, match="strict causal"):
        validate_binding_episode(
            replace(
                episode,
                evaluation=replace(
                    episode.evaluation, dependency_parents=bad_parents
                ),
            ),
            config,
        )


def test_composite_tokens_are_nonzero_and_match_visible_field_changes():
    config = make_config()
    episode = generate_binding_episode(config, length=24, seed=502, split="train")
    assert int(episode.inputs.token_ids.min()) > PAD_TOKEN_ID
    assert int(episode.inputs.token_ids.max()) < config.vocab_size
    visible = torch.stack(
        (
            episode.inputs.event_kinds,
            episode.inputs.primary_key_ids,
            episode.inputs.secondary_key_ids,
            episode.inputs.arguments,
        ),
        dim=1,
    )
    for left in range(episode.length):
        for right in range(left):
            if torch.equal(visible[left], visible[right]):
                assert episode.inputs.token_ids[left] == episode.inputs.token_ids[right]
            else:
                assert episode.inputs.token_ids[left] != episode.inputs.token_ids[right]
