import numpy as np
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
    # Lock the random state for deterministic shuffling
    gen = torch.Generator().manual_seed(seed)
    num_samples = train_features.shape[0]
    
    # Generate a single random permutation of indices from 0 to M-1
    shuffled_indices = torch.randperm(num_samples, generator=gen)
    
    # Calculate base chunk size and the remainder of rows
    base_size = num_samples // num_clients
    remainder = num_samples % num_clients
    
    partitions = []
    current_idx = 0
    
    # Distribute the indices and slice the data
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
    
    # Sort the dataset by label to cluster the classes together.
    # We only sort the indices to avoid moving heavy feature data in memory.
    sorted_indices = torch.argsort(train_labels)
    
    # Determine the size and count of the shards.
    total_shards = num_clients * shards_per_client
    base_shard_size = num_samples // total_shards
    remainder = num_samples % total_shards
    
    # Cut the sorted indices into discrete shards.
    shards = []
    current_idx = 0
    for i in range(total_shards):
        # Distribute the remainder evenly among the first few shards
        size = base_shard_size + (1 if i < remainder else 0)
        shards.append(sorted_indices[current_idx : current_idx + size])
        current_idx += size
        
    # Shuffle the order of the shards (crucial: we do NOT shuffle the data inside them)
    shard_order = torch.randperm(total_shards, generator=gen).tolist()
    
    partitions = []
    
    # Distribute the shards to the clients
    for i in range(num_clients):
        client_indices = []
        
        for j in range(shards_per_client):
            # Calculate which shard to give to this client based on the shuffled order
            shard_idx = shard_order[i * shards_per_client + j]
            client_indices.append(shards[shard_idx])
            
        # Concatenate the selected shard indices into a single 1D tensor
        client_idx = torch.cat(client_indices)
        
        # Slice the original data using the skewed, concatenated indices
        client_features = train_features[client_idx]
        client_labels = train_labels[client_idx]
        
        partitions.append((client_features, client_labels))
        
    return partitions

def partition_data_dirichlet(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    num_clients: int,
    alpha: float,
    seed: int,
    min_client_size: int = 10,
    max_attempts: int = 100,
):
    """
    Label-skewed partition by Dirichlet allocation (Hsu et al., 2019): for every class c, draw
    p_c ~ Dirichlet(alpha * 1_K) and give client k a fraction p_c[k] of that class's samples.

    Small alpha (e.g. 0.1) makes each client hold only a few classes; large alpha (e.g. 100) approaches IID.
    To keep clients trainable, a client that already holds more than its fair share N / K is skipped when
    allocating the remaining classes (NIID-Bench), and the draw is repeated until every client has at least
    `min_client_size` samples.

    Args:
        train_features: Float tensor of shape (M, input_size).
        train_labels: Int64 tensor of shape (M,).
        num_clients: Number of simulated clients (K).
        alpha: Dirichlet concentration (> 0).
        seed: Seed for the NumPy generator (full reproducibility).
        min_client_size: Minimum samples per client.
        max_attempts: Redraws before giving up.

    Returns:
        List of (client_features, client_labels) tuples, one per client.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    labels_np = train_labels.cpu().numpy()
    num_samples = len(labels_np)
    classes = sorted(set(labels_np.tolist()))
    rng = np.random.default_rng(seed)

    for _ in range(max_attempts):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = np.flatnonzero(labels_np == c)
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            # Do not give more data to clients that are already above the fair share
            proportions = np.array([p if len(ci) < num_samples / num_clients else 0.0 for p, ci in zip(proportions, client_indices)])
            if proportions.sum() == 0:
                proportions = np.full(num_clients, 1.0 / num_clients)
            proportions = proportions / proportions.sum()
            cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            for ci, part in zip(client_indices, np.split(idx_c, cuts)):
                ci.extend(part.tolist())
        if min(len(ci) for ci in client_indices) >= min_client_size:
            break
    else:
        raise RuntimeError(f"could not draw a Dirichlet partition with >= {min_client_size} samples per client in {max_attempts} attempts")

    partitions = []
    for ci in client_indices:
        sel = torch.as_tensor(sorted(ci), dtype=torch.int64)
        partitions.append((train_features[sel], train_labels[sel]))
    return partitions
