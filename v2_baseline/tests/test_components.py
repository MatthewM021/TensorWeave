import torch

from tnlm_v2.models.components import PairDisentangler, RankGate, TreeTensorReducer


def test_disentangler_initializes_as_identity():
    module = PairDisentangler(6)
    left = torch.randn(5, 6)
    right = torch.randn(5, 6)
    new_left, new_right = module(left, right)
    assert torch.allclose(new_left, left)
    assert torch.allclose(new_right, right)
    assert module.orthogonality_error().item() == 0.0


def test_masked_tree_passes_single_leaf_without_nan():
    reducer = TreeTensorReducer(8, 8, 16, True, False)
    states = torch.randn(3, 16, 8)
    masks = torch.zeros(3, 16, 1)
    masks[:, 5, 0] = 1.0
    root, root_mask, diagnostics = reducer(states, masks)
    assert root.shape == (3, 8)
    assert torch.all(root_mask == 1)
    assert torch.isfinite(root).all()
    assert diagnostics["levels_used"].item() == 4


def test_rank_gate_reports_full_initial_participation_rank():
    gate = RankGate(12)
    assert 11.9 < gate.effective_rank().item() <= 12.0001
