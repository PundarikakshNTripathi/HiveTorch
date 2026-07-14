import torch

def build_synthetic_dataset(num_samples: int, input_size: int, num_classes: int, seed: int):
    """
    Builds a deterministic synthetic classification dataset using clustered centroids.
    
    Args:
        num_samples: Total number of data points to generate.
        input_size: The number of features (dimensions) per data point.
        num_classes: The number of unique categories.
        seed: Integer seed to guarantee exact reproducibility.
        
    Returns:
        A tuple of (features, labels):
            features: Float tensor of shape (num_samples, input_size)
            labels: Int64 tensor of shape (num_samples,)
    """
    # Isolate the random state so it doesn't affect global randomness
    gen = torch.Generator().manual_seed(seed)
    
    # Generate labels: random integers from 0 up to (but not including) num_classes.
    # We strictly enforce dtype=torch.int64 because loss functions require it for indexing.
    labels = torch.randint(
        low=0, 
        high=num_classes, 
        size=(num_samples,), 
        generator=gen, 
        dtype=torch.int64
    )
    
    # Generate class centers (centroids) in the high-dimensional feature space.
    # Shape: (num_classes, input_size)
    centers = torch.randn(num_classes, input_size, generator=gen)
    
    # Map each sample to its corresponding class center using advanced indexing.
    # If labels is [0, 2, 1], sample_centers grabs the 0th, 2nd, and 1st row of centers.
    # Shape: (num_samples, input_size)
    sample_centers = centers[labels]
    
    # Generate random noise to scatter the points around their centers.
    # Shape: (num_samples, input_size)
    noise = torch.randn(num_samples, input_size, generator=gen)
    
    # Combine the centers and the noise to create the final float32 features.
    features = sample_centers + noise
    
    return features, labels