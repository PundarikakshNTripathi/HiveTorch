import torch
from src.fedavg_core.server.coordinator import run_fedavg

def compute_non_iid_gap(iid_accuracies: list, non_iid_accuracies: list) -> dict:
    """
    Quantifies the accuracy degradation caused by non-IID data distributions.
    
    Args:
        iid_accuracies: List of test accuracies from the IID control run.
        non_iid_accuracies: List of test accuracies from the skewed non-IID run.
        
    Returns:
        A dictionary containing the final IID accuracy, the final non-IID accuracy,
        and the mathematical gap between them.
    """
    # Extract the converged performance from the final communication round.
    # Casting to float() defensively ensures type safety for downstream logging.
    iid_final = float(iid_accuracies[-1])
    non_iid_final = float(non_iid_accuracies[-1])
    
    # Calculate the penalty (Positive gap = IID performed better)
    gap = iid_final - non_iid_final
    
    return {
        'iid_final': iid_final,
        'non_iid_final': non_iid_final,
        'gap': gap
    }

def rounds_to_target_vs_local_epochs(
    client_partitions: list, 
    test_features: torch.Tensor, 
    test_labels: torch.Tensor, 
    model_config: dict, 
    local_epochs_list: list, 
    target_accuracy: float, 
    num_rounds: int, 
    client_fraction: float, 
    batch_size: int, 
    learning_rate: float, 
    seed: int
) -> dict:
    """
    Sweeps over different local epoch counts to determine how quickly the 
    global model reaches a target accuracy.
    
    Args:
        client_partitions: List of client data shards.
        test_features: Global test features.
        test_labels: Global test labels.
        model_config: Neural network architecture dimensions.
        local_epochs_list: A list of integers representing different E values to test.
        target_accuracy: The float threshold to reach.
        num_rounds: Maximum number of rounds to simulate per run.
        client_fraction: Fraction of clients selected per round.
        batch_size: Local mini-batch size.
        learning_rate: Local optimizer step size.
        seed: Fixed random seed for strict isolation.
        
    Returns:
        A dictionary mapping each local_epoch value to the first round index 
        where accuracy >= target_accuracy, or None if the target was never met.
    """
    results = {}
    
    for E in local_epochs_list:
        # Execute the full federated simulation for this specific E
        _, round_accuracies = run_fedavg(
            client_partitions=client_partitions,
            test_features=test_features,
            test_labels=test_labels,
            model_config=model_config,
            num_rounds=num_rounds,
            client_fraction=client_fraction,
            local_epochs=E,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed
        )
        
        # Scan the telemetry trace to find the threshold crossing
        target_index = None
        for idx, acc in enumerate(round_accuracies):
            if acc >= target_accuracy:
                target_index = idx
                break  # Exit the scan immediately upon finding the first success
                
        # Record the index (or None) to the results dictionary
        results[E] = target_index
        
    return results

def accuracy_vs_client_fraction(
    client_partitions: list, 
    test_features: torch.Tensor, 
    test_labels: torch.Tensor, 
    model_config: dict, 
    client_fraction_list: list, 
    num_rounds: int, 
    local_epochs: int, 
    batch_size: int, 
    learning_rate: float, 
    seed: int
) -> dict:
    """
    Sweeps over different client participation fractions to measure their impact 
    on the final converged accuracy of the global model.
    
    Args:
        client_partitions: List of client data shards.
        test_features: Global test features.
        test_labels: Global test labels.
        model_config: Neural network architecture dimensions.
        client_fraction_list: A list of floats representing different C values to test.
        num_rounds: Total communication rounds to simulate per run.
        local_epochs: Local training passes per client.
        batch_size: Local mini-batch size.
        learning_rate: Local optimizer step size.
        seed: Fixed random seed for strict isolation across runs.
        
    Returns:
        A dictionary mapping each client fraction to its final converged test accuracy.
    """
    results = {}
    
    for client_fraction in client_fraction_list:
        
        # Execute the full federated simulation for this specific fraction.
        # CRITICAL: We pass the exact same base 'seed' to every run.
        _, round_accuracies = run_fedavg(
            client_partitions=client_partitions,
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
        
        # Extract the terminal accuracy of the converged model
        final_accuracy = float(round_accuracies[-1])
        
        # Map the fraction to its performance outcome
        results[client_fraction] = final_accuracy
        
    return results