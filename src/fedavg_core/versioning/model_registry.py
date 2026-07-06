import torch
import torch.nn as nn

def load_model_state(model: nn.Module, state_dict: dict) -> nn.Module:
    """
    Safely injects a parameter state dictionary into a model in-place.
    
    Args:
        model: The PyTorch neural network module to be updated.
        state_dict: An OrderedDict mapping parameter/buffer names to tensors, 
                    matching the model's exact architecture.
        
    Returns:
        The identical model instance, mutated in-place, for method chaining.
    """
    # PyTorch's native method safely updates the internal Parameter registry
    # without breaking autograd tracking or optimizer memory references.
    model.load_state_dict(state_dict)
    
    return model