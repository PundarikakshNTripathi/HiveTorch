import torch
import numpy as np

from src.fedavg_core.models.architectures import build_mlp_classifier
from src.fedavg_core.utils.tracking import clone_model_state
from src.fedavg_core.versioning.model_registry import load_model_state
from src.fedavg_core.client.edge_trainer import train_client_local
from src.fedavg_core.server.secure_aggregator import aggregate_weighted_average
from src.fedavg_core.evaluation.global_evaluator import evaluate_accuracy
from src.fedavg_core.monitoring.drift_detector import ConceptDriftDetector

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
    # Lock the global PRNG state BEFORE instantiation
    torch.manual_seed(seed)
    
    # Build the temporary scaffolding model. 
    # Because the seed is locked, its initial weights are strictly deterministic.
    temp_model = build_mlp_classifier(
        input_size=input_size, 
        hidden_size=hidden_size, 
        num_classes=num_classes
    )
    
    # Extract a safe, immutable copy of the state
    global_state = clone_model_state(temp_model)
    
    # temp_model is now garbage collected, leaving only the raw data payload.
    return global_state

def select_round_clients(num_clients: int, client_fraction: float, seed: int) -> list:
    """
    Selects a deterministic random subset of clients to participate in a communication round.
    
    Args:
        num_clients: The total number of available clients (K).
        client_fraction: The proportion of clients to select (C).
        seed: The integer seed to lock the PRNG state for reproducibility.
        
    Returns:
        A sorted list of distinct integer indices representing the chosen clients.
    """
    # Calculate the target number of clients, guaranteeing at least 1
    target_count = max(1, round(client_fraction * num_clients))
    
    # Lock the NumPy pseudo-random number generator state
    rng = np.random.default_rng(seed)
    
    # Draw distinct indices without replacement
    selected_indices = rng.choice(num_clients, size=target_count, replace=False)
    
    # Convert to a Python list and sort to guarantee deterministic downstream aggregation
    return sorted(selected_indices.tolist())

def run_communication_round(
    global_state: dict, 
    client_partitions: list, 
    selected_clients: list, 
    model_config: dict, 
    local_epochs: int, 
    batch_size: int, 
    learning_rate: float, 
    seed: int
) -> dict:
    """
    Executes a single Federated Averaging communication round across a subset of clients.
    
    Args:
        global_state: The starting model weights broadcasted by the server.
        client_partitions: The full list of (features, labels) for all clients.
        selected_clients: A list of integer indices dictating which clients participate this round.
        model_config: Dictionary containing architecture dimensions (input_size, hidden_size, num_classes).
        local_epochs: Number of local training passes per client.
        batch_size: Mini-batch size for local SGD.
        learning_rate: Optimizer step size.
        seed: Base integer seed for deterministic execution.
        
    Returns:
        The newly aggregated global state dictionary.
    """
    client_states = []
    client_sample_counts = []
    
    # Dispatch the workload to the selected worker nodes
    for client_idx in selected_clients:
        
        # Isolate Memory: Build a fresh, sterile model to simulate a physically separate client
        client_model = build_mlp_classifier(**model_config)
        
        # Synchronize State: Inject the server's broadcasted weights
        load_model_state(client_model, global_state)
        
        # Extract the specific dataset residing on this client
        client_features, client_labels = client_partitions[client_idx]
        
        # Create a unique but perfectly reproducible random state for this client's batching
        client_seed = seed + client_idx
        
        # Execute Local Training
        trained_state = train_client_local(
            model=client_model,
            client_features=client_features,
            client_labels=client_labels,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=client_seed
        )
        
        # Collect Payloads
        client_states.append(trained_state)
        
        # Track the number of local samples (N_k) for the downstream weighted average
        num_samples = client_features.shape[0]
        client_sample_counts.append(num_samples)
        
    # Server-Side Aggregation
    # Fuse the collected states based on their data volume
    new_global_state = aggregate_weighted_average(client_states, client_sample_counts)
    
    return new_global_state

def run_fedavg(
    client_partitions: list, 
    test_features: torch.Tensor, 
    test_labels: torch.Tensor, 
    model_config: dict, 
    num_rounds: int, 
    client_fraction: float, 
    local_epochs: int, 
    batch_size: int, 
    learning_rate: float, 
    seed: int
) -> tuple:
    """
    Drives the complete Federated Averaging lifecycle.
    
    Args:
        client_partitions: List of (features, labels) for all clients.
        test_features: Held-out global test features.
        test_labels: Held-out global test labels.
        model_config: Dictionary of architecture dimensions.
        num_rounds: Total communication rounds to simulate.
        client_fraction: Fraction of clients to sample each round.
        local_epochs: Local training passes per client.
        batch_size: Local mini-batch size.
        learning_rate: Local optimizer step size.
        seed: Base random seed for the entire execution.
        
    Returns:
        A tuple of (final_global_model, list_of_round_accuracies).
    """
    # Initialize the pristine starting weights
    global_state = initialize_global_state(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        num_classes=model_config['num_classes'],
        seed=seed
    )
    
    round_accuracies = []
    num_clients = len(client_partitions)
    
    # Execute the communication loop
    for round_idx in range(num_rounds):
        
        # Advance the seed to guarantee fresh entropy per round
        round_seed = seed + round_idx
        
        # Select the active cohort
        selected_clients = select_round_clients(num_clients, client_fraction, round_seed)
        
        # Dispatch workloads and aggregate the updated state
        global_state = run_communication_round(
            global_state=global_state,
            client_partitions=client_partitions,
            selected_clients=selected_clients,
            model_config=model_config,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=round_seed
        )
        
        # Telemetry: Build an ephemeral model to evaluate the new global state
        eval_model = build_mlp_classifier(**model_config)
        load_model_state(eval_model, global_state)
        
        acc = evaluate_accuracy(eval_model, test_features, test_labels)
        round_accuracies.append(acc)
        
        # Monitor for Concept Drift against the best accuracy so far
        if len(round_accuracies) > 1:
            best_acc = max(round_accuracies[:-1])
            detector = ConceptDriftDetector(threshold=0.10)
            if detector.check_accuracy_drift(best_acc, acc):
                print(f"WARNING: Concept Drift Detected at Round {round_idx}! Accuracy dropped from {best_acc:.4f} to {acc:.4f}")
        
    # Finalize and package the results
    final_model = build_mlp_classifier(**model_config)
    load_model_state(final_model, global_state)
    
    return final_model, round_accuracies