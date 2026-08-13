from tnlm_v2.data import build_task
from tnlm_v2.factory import create_model
from tnlm_v2.training import TrainConfig, evaluate_model, train_model


def test_short_training_run_completes():
    task = build_task("interleaved_threads")
    train = task.generate(64, 32, seed=1)
    validation = task.generate(32, 32, seed=2)
    model = create_model(
        "routed_ttn_oracle",
        task.spec,
        64,
        {"tn_dimension": 8, "tn_rank": 8},
    )
    result = train_model(
        model,
        train,
        validation,
        TrainConfig(epochs=2, batch_size=16, patience=2, num_threads=2),
    )
    metrics = evaluate_model(model, validation, batch_size=32)
    assert len(result.history) >= 1
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] > 0.0


def test_route_metric_is_permutation_invariant():
    import torch

    from tnlm_v2.training import permutation_aligned_route_metrics

    truth = torch.tensor([[0, 0, 1, 1, 2, 2, -1]])
    predicted = torch.tensor([[2, 2, 0, 0, 1, 1, 0]])
    metrics = permutation_aligned_route_metrics(predicted, truth)
    assert metrics["route_accuracy_aligned"] == 1.0
    assert metrics["route_purity"] == 1.0
    assert metrics["route_used_branches"] == 3.0
