import torch
from src.fedavg_core.data.non_iid_sharder import partition_data_iid

def test_partition_data_iid():
    features = torch.randn(100, 10)
    labels = torch.randint(0, 5, (100,))
    partitions = partition_data_iid(features, labels, num_clients=5, seed=42)
    
    assert len(partitions) == 5
    for p_features, p_labels in partitions:
        assert p_features.shape[0] == 20
        assert p_labels.shape[0] == 20
