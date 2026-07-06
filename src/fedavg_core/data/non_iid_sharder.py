import torch

def partition_data_iid(train_features: torch.Tensor, train_labels: torch.Tensor, num_clients: int, seed: int):
    """
    Distributes training data across clients in an Independent and Identically Distributed (IID) manner.
    
    Args:
        train_features: Float tensor of shape (M, input_size).
        train_labels: Int64 tensor of shape (M,).
        num_clients: The number of simulated federated clients (K).
        seed: Integer seed to guarantee exact reproducibility of the shuffle.
        
    Returns:
        A list of length `num_clients`, where each element is a tuple of 
        (client_features, client_labels).
    """
    # 1. Lock the random state for deterministic shuffling
    gen = torch.Generator().manual_seed(seed)
    num_samples = train_features.shape[0]
    
    # 2. Generate a single random permutation of indices from 0 to M-1
    shuffled_indices = torch.randperm(num_samples, generator=gen)
    
    # 3. Calculate base chunk size and the remainder of rows
    base_size = num_samples // num_clients
    remainder = num_samples % num_clients
    
    partitions = []
    current_idx = 0
    
    # 4. Distribute the indices and slice the data
    for i in range(num_clients):
        # Distribute the remainder by giving the first 'remainder' clients one extra row
        client_size = base_size + (1 if i < remainder else 0)
        
        # Grab the specific indices for this client
        client_indices = shuffled_indices[current_idx : current_idx + client_size]
        
        # Slice the features and labels using the shared indices
        client_features = train_features[client_indices]
        client_labels = train_labels[client_indices]
        
        partitions.append((client_features, client_labels))
        
        # Advance the index pointer for the next client
        current_idx += client_size
        
    return partitions

def partition_data_non_iid(train_features: torch.Tensor, train_labels: torch.Tensor, num_clients: int, shards_per_client: int, seed: int):
    """
    Distributes training data across clients with deliberately skewed label distributions.
    
    Args:
        train_features: Float tensor of shape (M, input_size).
        train_labels: Int64 tensor of shape (M,).
        num_clients: The total number of simulated clients.
        shards_per_client: The number of distinct data shards assigned to each client.
        seed: Integer seed to guarantee exact reproducibility of the shard assignment.
        
    Returns:
        A list of length `num_clients`, where each element is a tuple of 
        (client_features, client_labels).
    """
    gen = torch.Generator().manual_seed(seed)
    num_samples = train_features.shape[0]
    
    # 1. Sort the dataset by label to cluster the classes together.
    # We only sort the indices to avoid moving heavy feature data in memory.
    sorted_indices = torch.argsort(train_labels)
    
    # 2. Determine the size and count of the shards.
    total_shards = num_clients * shards_per_client
    base_shard_size = num_samples // total_shards
    remainder = num_samples % total_shards
    
    # 3. Cut the sorted indices into discrete shards.
    shards = []
    current_idx = 0
    for i in range(total_shards):
        # Distribute the remainder evenly among the first few shards
        size = base_shard_size + (1 if i < remainder else 0)
        shards.append(sorted_indices[current_idx : current_idx + size])
        current_idx += size
        
    # 4. Shuffle the order of the shards (crucial: we do NOT shuffle the data inside them)
    shard_order = torch.randperm(total_shards, generator=gen).tolist()
    
    partitions = []
    
    # 5. Distribute the shards to the clients
    for i in range(num_clients):
        client_indices = []
        
        for j in range(shards_per_client):
            # Calculate which shard to give to this client based on the shuffled order
            shard_idx = shard_order[i * shards_per_client + j]
            client_indices.append(shards[shard_idx])
            
        # Concatenate the selected shard indices into a single 1D tensor
        client_indices = torch.cat(client_indices)
        
        # Slice the original data using the skewed, concatenated indices
        client_features = train_features[client_indices]
        client_labels = train_labels[client_indices]
        
        partitions.append((client_features, client_labels))
        
    return partitions