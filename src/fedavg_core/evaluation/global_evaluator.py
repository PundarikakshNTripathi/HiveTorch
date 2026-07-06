import torch
import torch.nn as nn

def evaluate_accuracy(model: nn.Module, test_features: torch.Tensor, test_labels: torch.Tensor) -> float:
    """
    Evaluates the model's classification accuracy on a held-out test set.
    
    Args:
        model: The PyTorch neural network module to evaluate.
        test_features: Float tensor of shape (T, input_size).
        test_labels: Int64 tensor of shape (T,) containing true class indices.
        
    Returns:
        A scalar float between 0.0 and 1.0 representing the fraction of 
        correct predictions.
    """
    # 1. Toggle the model to evaluation mode (disables Dropout/BatchNorm side-effects)
    model.eval()
    
    # 2. Disable the autograd engine to save VRAM and execution time
    with torch.no_grad():
        
        # 3. Forward pass to get raw unnormalized logits
        # Shape: (T, num_classes)
        logits = model(test_features)
        
        # 4. Extract the predicted class by finding the index of the highest logit
        # We reduce along dimension 1 (the class axis)
        # Shape: (T,)
        predictions = torch.argmax(logits, dim=1)
        
        # 5. Calculate the fraction of correct predictions via vectorized math
        # (predictions == test_labels) creates a boolean tensor [True, False, True...]
        # .float() converts it to [1.0, 0.0, 1.0...]
        # .mean() calculates the exact ratio of 1s (the accuracy)
        accuracy_tensor = (predictions == test_labels).float().mean()
        
    # 6. Extract the primitive Python float
    return accuracy_tensor.item()