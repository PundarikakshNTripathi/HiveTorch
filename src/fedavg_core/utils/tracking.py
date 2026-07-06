import torch.nn as nn

def clone_model_state(model: nn.Module) -> dict:
    """
    Creates a deep copy of a model's state dictionary, ensuring complete 
    memory independence and severing all autograd graph connections.
    
    Args:
        model: The PyTorch neural network module to snapshot.
        
    Returns:
        A new dictionary mapping parameter/buffer names to fresh, detached tensors.
    """
    # 1. Grab the dictionary of memory references
    live_state = model.state_dict()
    
    # 2. Build a completely independent snapshot
    snapshot = {}
    for name, tensor in live_state.items():
        # Detach removes it from the graph; clone allocates new memory
        snapshot[name] = tensor.detach().clone()
        
    return snapshot