import torch

def local_sgd_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch_features: torch.Tensor, batch_labels: torch.Tensor) -> float:
    """
    Performs a single in-place SGD update on the model for a given mini-batch.
    
    Args:
        model: The PyTorch neural network module to be updated.
        optimizer: The PyTorch optimizer holding references to the model parameters.
        batch_features: Float tensor of shape (B, input_size).
        batch_labels: Int64 tensor of shape (B,).
        
    Returns:
        The scalar loss for this batch as a standard Python float.
    """
    # 1. Clear the gradient buffers from the previous step
    optimizer.zero_grad()
    
    # Local import to avoid circular dependency with edge_trainer.py
    from src.fedavg_core.client.edge_trainer import compute_batch_loss

    # 2. Forward pass: compute the loss and build the computation graph
    # We reuse the compute_batch_loss function built in Step 8
    loss = compute_batch_loss(model, batch_features, batch_labels)
    
    # 3. Backward pass: traverse the graph and populate .grad attributes
    loss.backward()
    
    # 4. Optimizer step: mutate the model parameters in-place down the gradient
    optimizer.step()
    
    # 5. Extract the raw float to prevent memory leaks and allow garbage collection of the DAG
    return loss.item()