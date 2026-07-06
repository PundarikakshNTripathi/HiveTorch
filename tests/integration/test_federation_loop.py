import torch
from src.fedavg_core.server.coordinator import initialize_global_state

def test_global_state_initialization():
    state = initialize_global_state(input_size=10, hidden_size=20, num_classes=5, seed=42)
    # Ensure it's a valid state dict with the right layer dimensions
    assert 'layer1.weight' in state
    assert state['layer1.weight'].shape == (20, 10)
    assert state['layer2.weight'].shape == (5, 20)
