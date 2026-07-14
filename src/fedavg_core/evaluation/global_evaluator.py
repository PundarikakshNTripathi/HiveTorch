import torch
import torch.nn as nn
from src.fedavg_core.models.architectures import build_mlp_classifier
from src.fedavg_core.data.processing_pipeline import iterate_client_batches
from src.fedavg_core.optimization.weight_alignment import local_sgd_step

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
    # Toggle the model to evaluation mode (disables Dropout/BatchNorm side-effects)
    model.eval()
    
    # Disable the autograd engine to save VRAM and execution time
    with torch.no_grad():
        
        # Forward pass to get raw unnormalized logits
        # Shape: (T, num_classes)
        logits = model(test_features)
        
        # Extract the predicted class by finding the index of the highest logit
        # We reduce along dimension 1 (the class axis)
        # Shape: (T,)
        predictions = torch.argmax(logits, dim=1)
        
        # Calculate the fraction of correct predictions via vectorized math
        # (predictions == test_labels) creates a boolean tensor [True, False, True...]
        # .float() converts it to [1.0, 0.0, 1.0...]
        # .mean() calculates the exact ratio of 1s (the accuracy)
        accuracy_tensor = (predictions == test_labels).float().mean()
        
    # Extract the primitive Python float
    return accuracy_tensor.item()



def train_centralized_baseline(
    train_features: torch.Tensor, 
    train_labels: torch.Tensor, 
    test_features: torch.Tensor, 
    test_labels: torch.Tensor, 
    model_config: dict, 
    num_epochs: int, 
    batch_size: int, 
    learning_rate: float, 
    seed: int
) -> float:
    """
    Trains a centralized monolithic model on the entire pooled dataset to 
    establish a baseline accuracy yardstick.
    
    Args:
        train_features: Float tensor of all pooled training features (M, input_size).
        train_labels: Int64 tensor of all pooled training labels (M,).
        test_features: Float tensor of all held-out test features.
        test_labels: Int64 tensor of all held-out test labels.
        model_config: Dictionary of architecture dimensions.
        num_epochs: Total number of full passes over the pooled dataset.
        batch_size: Mini-batch size for SGD.
        learning_rate: Optimizer step size.
        seed: Base random seed for reproducible initialization and batching.
        
    Returns:
        The model's classification accuracy on the test set as a float [0, 1].
    """
    # Lock the seed to guarantee the monolithic model starts with the 
    # exact same random weights as the federated global model.
    torch.manual_seed(seed)
    
    # Instantiate the monolithic model and its optimizer
    model = build_mlp_classifier(**model_config)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Standard training loop over the pooled data
    model.train()
    for epoch in range(num_epochs):
        
        # Reshuffle the batches predictably but uniquely per epoch
        epoch_seed = seed + epoch
        
        # We reuse the client batching logic, treating the centralized 
        # dataset as one massive client shard.
        batches = iterate_client_batches(train_features, train_labels, batch_size, epoch_seed)
        
        for batch_features, batch_labels in batches:
            local_sgd_step(model, optimizer, batch_features, batch_labels)
            
    # Evaluate and return the final theoretical upper bound
    accuracy = evaluate_accuracy(model, test_features, test_labels)
    
    return accuracy