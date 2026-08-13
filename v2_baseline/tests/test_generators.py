import torch

from tnlm_v2.data import available_tasks, build_task


def test_all_generators_are_deterministic_and_well_formed():
    for name in available_tasks():
        task = build_task(name)
        length = max(64 if name == "combined_language" else 32, task.minimum_length)
        a = task.generate(16, length, seed=123)
        b = task.generate(16, length, seed=123)
        assert torch.equal(a.tokens, b.tokens)
        assert torch.equal(a.routes, b.routes)
        assert torch.equal(a.labels, b.labels)
        assert a.tokens.shape == (16, length)
        assert a.routes.shape == (16, length)
        assert torch.all((a.labels >= 0) & (a.labels < task.spec.num_classes))
        assert torch.all(a.routes[a.valid_mask] < task.spec.max_branches)
        assert torch.all(a.routes[~a.valid_mask] == -1)


def test_oracle_route_labels_cover_live_threads():
    task = build_task("interleaved_threads")
    batch = task.generate(64, 32, seed=9, active_branches=8)
    for sample in range(len(batch)):
        used = set(batch.routes[sample][batch.routes[sample] >= 0].tolist())
        assert used == set(range(8))


def test_hierarchy_nominal_length_adds_only_padding():
    task = build_task("permuted_hierarchy")
    batch = task.generate(8, 128, seed=4)
    assert torch.all(batch.valid_mask.sum(dim=1) == 17)


def _valid_names(task, batch, sample):
    ids = batch.tokens[sample][batch.valid_mask[sample]].tolist()
    return task.decode(ids)


def test_generated_labels_recompute_from_serialized_programs():
    # Interleaved last-write retrieval.
    task = build_task("interleaved_threads")
    batch = task.generate(32, 64, seed=71)
    for i in range(len(batch)):
        names = _valid_names(task, batch, i)
        states = {}
        cursor = 0
        while names[cursor] != "<QUERY>":
            branch = int(names[cursor].split("_")[1])
            value = int(names[cursor + 1].split("_")[1])
            states[branch] = value
            cursor += 2
        query = int(names[cursor + 1].split("_")[1])
        assert batch.labels[i].item() == states[query]

    # Permuted leaves followed by the known branch reduction.
    task = build_task("permuted_hierarchy")
    batch = task.generate(32, 32, seed=72)
    for i in range(len(batch)):
        names = _valid_names(task, batch, i)
        states = [0] * task.spec.max_branches
        for cursor in range(0, len(names) - 1, 2):
            branch = int(names[cursor].split("_")[1])
            states[branch] = int(names[cursor + 1].split("_")[1])
        assert batch.labels[i].item() == task.reduce_branch_values(states)

    # Mixed value minus nuisance reconstructs the predictive signal.
    task = build_task("predictive_detail")
    batch = task.generate(32, 64, seed=73)
    for i in range(len(batch)):
        names = _valid_names(task, batch, i)
        states = {}
        cursor = 0
        while names[cursor] != "<QUERY>":
            branch = int(names[cursor].split("_")[1])
            mixed = int(names[cursor + 1].split("_")[1])
            nuisance = int(names[cursor + 2].split("_")[1])
            states[branch] = (mixed - nuisance) % task.value_cardinality
            cursor += 3
        query = int(names[cursor + 1].split("_")[1])
        assert batch.labels[i].item() == states[query]

    # Combined SET/ADD updates followed by the branch tree.
    task = build_task("combined_language")
    batch = task.generate(32, 64, seed=74)
    for i in range(len(batch)):
        names = _valid_names(task, batch, i)
        states = [0] * task.spec.max_branches
        cursor = 0
        while names[cursor] != "<ROOT_QUERY>":
            branch = int(names[cursor].split("_")[1])
            op, mixed_text = names[cursor + 1].rsplit("_", 1)
            mixed = int(mixed_text)
            nuisance = int(names[cursor + 2].split("_")[1])
            signal = (mixed - nuisance) % task.value_cardinality
            if op == "MIXSET":
                states[branch] = signal
            else:
                assert op == "MIXADD"
                states[branch] = (states[branch] + signal) % task.value_cardinality
            cursor += 3
        assert batch.labels[i].item() == task.reduce_branch_values(states)
