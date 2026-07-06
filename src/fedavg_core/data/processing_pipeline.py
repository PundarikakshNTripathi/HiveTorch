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

def iterate_client_batches(client_features: torch.Tensor, client_labels: torch.Tensor, batch_size: int, seed: int) -> list:
    """
    Shuffles a client's dataset and slices it into mini-batches for local training.
    
    Args:
        client_features: Float tensor of shape (n, input_size).
        client_labels: Int64 tensor of shape (n,).
        batch_size: The desired number of samples per batch (B).
        seed: Integer seed to guarantee exact reproducibility of the shuffle.
        
    Returns:
        A list of tuples, where each tuple is (batch_features, batch_labels).
        The final tuple may contain fewer than B samples if n is not divisible by B.
    """
    gen = torch.Generator().manual_seed(seed)
    num_samples = client_features.shape[0]
    
    # 1. Generate a single permutation of indices to maintain feature-label pairing
    shuffled_indices = torch.randperm(num_samples, generator=gen)
    
    # 2. Shuffle the entire client dataset once using the indices
    shuffled_features = client_features[shuffled_indices]
    shuffled_labels = client_labels[shuffled_indices]
    
    batches = []
    
    # 3. Step through the dataset in increments of batch_size
    for start_idx in range(0, num_samples, batch_size):
        
        # Slicing in PyTorch automatically handles out-of-bounds ends, 
        # seamlessly returning the smaller remainder batch.
        end_idx = start_idx + batch_size
        
        batch_features = shuffled_features[start_idx:end_idx]
        batch_labels = shuffled_labels[start_idx:end_idx]
        
        batches.append((batch_features, batch_labels))
        
    return batches