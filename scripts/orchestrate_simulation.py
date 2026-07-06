import torch

from src.fedavg_core.data.non_iid_sharder import partition_data_iid, partition_data_non_iid
from src.fedavg_core.server.coordinator import run_fedavg

def run_fedavg_iid(
    train_features: torch.Tensor, 
    train_labels: torch.Tensor, 
    test_features: torch.Tensor, 
    test_labels: torch.Tensor, 
    model_config: dict, 
    num_clients: int, 
    num_rounds: int, 
    client_fraction: float, 
    local_epochs: int, 
    batch_size: int, 
    learning_rate: float, 
    seed: int
) -> list:
    """
    Executes a complete Federated Averaging experiment under an ideal IID data distribution.
    
    Args:
        train_features: Float tensor of pooled training data (M, input_size).
        train_labels: Int64 tensor of pooled training labels (M,).
        test_features: Float tensor of held-out test data.
        test_labels: Int64 tensor of held-out test labels.
        model_config: Architecture dimensions for the underlying neural network.
        num_clients: Total number of simulated federated clients.
        num_rounds: Number of global communication rounds to execute.
        client_fraction: Proportion of clients selected per round.
        local_epochs: Number of local training passes per client.
        batch_size: Mini-batch size for local SGD.
        learning_rate: Optimizer step size.
        seed: Base random seed for partitioning and training.
        
    Returns:
        A list of floats representing the global model's test accuracy after each round.
    """
    # 1. Prepare the ideal baseline data distribution
    iid_partitions = partition_data_iid(
        train_features=train_features, 
        train_labels=train_labels, 
        num_clients=num_clients, 
        seed=seed
    )
    
    # 2. Drive the federated loop, capturing both the final model and the telemetry
    final_model, round_accuracies = run_fedavg(
        client_partitions=iid_partitions,
        test_features=test_features,
        test_labels=test_labels,
        model_config=model_config,
        num_rounds=num_rounds,
        client_fraction=client_fraction,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed
    )
    
    # 3. Discard the model object to free VRAM and return only the learning curve
    return round_accuracies


def run_fedavg_non_iid(
    train_features: torch.Tensor, 
    train_labels: torch.Tensor, 
    test_features: torch.Tensor, 
    test_labels: torch.Tensor, 
    model_config: dict, 
    num_clients: int, 
    shards_per_client: int,
    num_rounds: int, 
    client_fraction: float, 
    local_epochs: int, 
    batch_size: int, 
    learning_rate: float, 
    seed: int
) -> tuple:
    """
    Executes a complete Federated Averaging experiment under a skewed, non-IID data distribution.
    
    Args:
        train_features: Float tensor of pooled training data (M, input_size).
        train_labels: Int64 tensor of pooled training labels (M,).
        test_features: Float tensor of held-out test data.
        test_labels: Int64 tensor of held-out test labels.
        model_config: Architecture dimensions for the underlying neural network.
        num_clients: Total number of simulated federated clients.
        shards_per_client: Number of distinct label shards assigned to each client.
        num_rounds: Number of global communication rounds to execute.
        client_fraction: Proportion of clients selected per round.
        local_epochs: Number of local training passes per client.
        batch_size: Mini-batch size for local SGD.
        learning_rate: Optimizer step size.
        seed: Base random seed for partitioning and training.
        
    Returns:
        A tuple of (final_global_model, list_of_round_accuracies).
    """
    # 1. Prepare the skewed, real-world data distribution
    non_iid_partitions = partition_data_non_iid(
        train_features=train_features, 
        train_labels=train_labels, 
        num_clients=num_clients, 
        shards_per_client=shards_per_client,
        seed=seed
    )
    
    # 2. Drive the federated loop using the skewed data shards
    final_model, round_accuracies = run_fedavg(
        client_partitions=non_iid_partitions,
        test_features=test_features,
        test_labels=test_labels,
        model_config=model_config,
        num_rounds=num_rounds,
        client_fraction=client_fraction,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed
    )
    
    # 3. Return both the forensic model artifact and the telemetry trace
    return final_model, round_accuracies