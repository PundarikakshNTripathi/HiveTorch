import torch
from src.fedavg_core.server.secure_aggregator import add_state_dicts, scale_state_dict

def test_scale_state_dict():
    state = {'weight': torch.tensor([1.0, 2.0])}
    scaled = scale_state_dict(state, 0.5)
    assert torch.allclose(scaled['weight'], torch.tensor([0.5, 1.0]))

def test_add_state_dicts():
    state_a = {'weight': torch.tensor([1.0, 2.0])}
    state_b = {'weight': torch.tensor([3.0, 4.0])}
    result = add_state_dicts(state_a, state_b)
    assert torch.allclose(result['weight'], torch.tensor([4.0, 6.0]))
