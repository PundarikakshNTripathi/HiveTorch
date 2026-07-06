import torch

from src.fedavg_core.models.architectures import build_mlp_classifier
from src.fedavg_core.utils.tracking import clone_model_state

def initialize_global_state(input_size: int, hidden_size: int, num_classes: int, seed: int) -> dict:
    """
    Creates the deterministic Round 0 global state dictionary for FedAvg.
    
    Args:
        input_size: The number of input features.
        hidden_size: The width of the hidden layer.
        num_classes: The number of output classes.
        seed: The integer seed to lock the PRNG state.
        
    Returns:
        An OrderedDict containing freshly initialized, completely detached 
        model parameters ready for network transmission.
    """
    # 1. Lock the global PRNG state BEFORE instantiation
    torch.manual_seed(seed)
    
    # 2. Build the temporary scaffolding model. 
    # Because the seed is locked, its initial weights are strictly deterministic.
    temp_model = build_mlp_classifier(
        input_size=input_size, 
        hidden_size=hidden_size, 
        num_classes=num_classes
    )
    
    # 3. Extract a safe, immutable copy of the state
    global_state = clone_model_state(temp_model)
    
    # temp_model is now garbage collected, leaving only the raw data payload.
    return global_state