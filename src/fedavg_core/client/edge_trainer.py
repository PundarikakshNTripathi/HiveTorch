import torch
import torch.nn.functional as F

from src.fedavg_core.data.processing_pipeline import iterate_client_batches
from src.fedavg_core.optimization.weight_alignment import local_sgd_step

def compute_batch_loss(model: torch.nn.Module, batch_features: torch.Tensor, batch_labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross-entropy loss for a single mini-batch.
    
    Args:
        model: The PyTorch neural network module (must be in train mode).
        batch_features: Float tensor of shape (B, input_size).
        batch_labels: Int64 tensor of shape (B,) containing the true class indices.
        
    Returns:
        A scalar (0-dimensional) float tensor representing the mean batch loss,
        attached to the computation graph (requires_grad=True).
    """
    # 1. Forward Pass: Run the features through the model to get raw, unnormalized logits.
    # Shape of logits: (B, num_classes)
    logits = model(batch_features)
    
    # 2. Compute Loss: F.cross_entropy expects raw logits and integer class labels.
    # It internally handles the numerically stable fused Log-Softmax + NLLLoss.
    # By default, this returns the mean loss across the batch size (B).
    loss = F.cross_entropy(logits, batch_labels)
    
    return loss

def train_client_local(
    model: torch.nn.Module, 
    client_features: torch.Tensor, 
    client_labels: torch.Tensor, 
    local_epochs: int, 
    batch_size: int, 
    learning_rate: float, 
    seed: int
) -> dict:
    """
    Executes the full local training loop for a single federated client.
    
    Args:
        model: The PyTorch neural network module (seeded with global weights).
        client_features: Float tensor of shape (N_k, input_size).
        client_labels: Int64 tensor of shape (N_k,).
        local_epochs: The number of full passes over the client's dataset.
        batch_size: The number of samples per mini-batch.
        learning_rate: The step size for the SGD optimizer.
        seed: Base integer seed for reproducibility.
        
    Returns:
        An OrderedDict containing the updated model parameters (the state_dict).
    """
    # 1. Instantiate the optimizer once, binding it to the model's parameters
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Ensure the model is in training mode (enables dropout/batchnorm if added later)
    model.train()
    
    # 2. Iterate through the requested number of local epochs
    for epoch in range(local_epochs):
        
        # 3. Reshuffle and batch the data predictably but uniquely per epoch
        epoch_seed = seed + epoch
        batches = iterate_client_batches(client_features, client_labels, batch_size, epoch_seed)
        
        # 4. Process every batch using our atomic SGD step
        for batch_features, batch_labels in batches:
            local_sgd_step(model, optimizer, batch_features, batch_labels)
            
    # 5. Extract and return the raw weights payload for network transmission
    return model.state_dict()