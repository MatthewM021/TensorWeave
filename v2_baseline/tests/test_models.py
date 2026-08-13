import pytest
import torch

from tnlm_v2.data import build_task
from tnlm_v2.factory import create_model


@pytest.mark.parametrize(
    "name",
    [
        "mps",
        "fixed_ttn",
        "fixed_mera",
        "routed_ttn_oracle",
        "routed_ttn_oracle_widecore",
        "routed_ttn_learned",
        "routed_mera_oracle",
        "routed_mera_learned",
        "routed_ttn_oracle_adaptive",
        "gru",
        "transformer",
    ],
)
def test_model_forward_backward(name):
    task = build_task("interleaved_threads")
    batch = task.generate(4, 32, seed=7)
    torch.manual_seed(5)
    model = create_model(
        name,
        task.spec,
        max_length=64,
        settings={
            "tn_dimension": 8,
            "tn_rank": 8,
            "mps_bond_dimension": 8,
            "baseline_dimension": 16,
            "transformer_layers": 1,
            "transformer_heads": 4,
        },
    )
    routes = batch.routes if "_oracle" in model.model_name else None
    output = model(batch.tokens, batch.valid_mask, routes)
    assert output.logits.shape == (4, task.spec.num_classes)
    loss = torch.nn.functional.cross_entropy(output.logits, batch.labels)
    loss = loss + sum(output.aux_losses.values(), torch.zeros(())) * 1e-4
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_oracle_model_rejects_missing_routes():
    task = build_task("interleaved_threads")
    batch = task.generate(2, 32, seed=3)
    model = create_model("routed_ttn_oracle", task.spec, 64, {"tn_dimension": 8})
    with pytest.raises(ValueError):
        model(batch.tokens, batch.valid_mask)
