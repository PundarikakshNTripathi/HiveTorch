import torch

def train_test_split_dataset(features: torch.Tensor, labels: torch.Tensor, test_fraction: float, seed: int):
    """
    Deterministically splits features and labels into training and testing sets.
    
    Args:
        features: Float tensor of shape (N, input_size).
        labels: Int64 tensor of shape (N,).
        test_fraction: Float between 0.0 and 1.0 representing the proportion of data for testing.
        seed: Integer seed to guarantee exact reproducibility of the shuffle.
        
    Returns:
        A tuple containing four tensors:
        (train_features, train_labels, test_features, test_labels)
    """
    # 1. Lock the random state
    gen = torch.Generator().manual_seed(seed)
    
    # 2. Determine exact boundaries
    num_samples = features.shape[0]
    num_test = int(num_samples * test_fraction)
    
    # 3. Generate a single deterministic permutation of indices from 0 to N-1
    # Shape: (num_samples,)
    shuffled_indices = torch.randperm(num_samples, generator=gen)
    
    # 4. Slice the indices into test and train sets. 
    # The prompt requests the *first* int(N * test_fraction) for the test set.
    test_indices = shuffled_indices[:num_test]
    train_indices = shuffled_indices[num_test:]
    
    # 5. Apply the shared indices to both tensors to maintain paired supervision
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    
    return train_features, train_labels, test_features, test_labels

def count_client_samples(client_partitions: list) -> list:
    """
    Calculates the number of data samples held by each federated client.
    
    Args:
        client_partitions: A list of tuples, where each tuple is 
                           (client_features, client_labels).
        
    Returns:
        A list of integers representing the sample count for each client, 
        in the exact order they were provided.
    """
    # Using a list comprehension for execution speed.
    # We unpack the tuple, ignoring the labels with '_', and 
    # instantly read the 0th dimension of the features tensor.
    sample_counts = [features.shape[0] for features, _ in client_partitions]
    
    return sample_counts