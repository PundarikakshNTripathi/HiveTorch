import numpy as np
import pytest
import torch

from src.fedavg_core.data.non_iid_sharder import partition_data_dirichlet


def _data(n=3000, classes=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, classes, (n,), generator=g)
    feats = torch.randn(n, 4, generator=g)
    return feats, labels


def _mean_entropy(parts, classes=10):
    ents = []
    for _, y in parts:
        p = np.bincount(y.numpy(), minlength=classes) / len(y)
        p = p[p > 0]
        ents.append(float(-(p * np.log(p)).sum()))
    return float(np.mean(ents))


def test_partition_is_a_disjoint_cover():
    x, y = _data()
    parts = partition_data_dirichlet(x, y, num_clients=8, alpha=0.3, seed=1)
    assert len(parts) == 8
    assert sum(len(p[1]) for p in parts) == len(y)
    # same multiset of labels, no sample duplicated or lost
    assert torch.equal(torch.cat([p[1] for p in parts]).sort().values, y.sort().values)
    assert min(len(p[1]) for p in parts) >= 10


def test_deterministic_given_seed():
    x, y = _data()
    a = partition_data_dirichlet(x, y, 6, 0.5, seed=3)
    b = partition_data_dirichlet(x, y, 6, 0.5, seed=3)
    c = partition_data_dirichlet(x, y, 6, 0.5, seed=4)
    assert all(torch.equal(p[1], q[1]) for p, q in zip(a, b))
    assert not all(torch.equal(p[1], q[1]) for p, q in zip(a, c))


def test_smaller_alpha_means_more_label_skew():
    x, y = _data()
    ent = {a: _mean_entropy(partition_data_dirichlet(x, y, 10, a, seed=0)) for a in (0.05, 0.5, 100.0)}
    assert ent[0.05] < ent[0.5] < ent[100.0]
    assert ent[100.0] > 0.9 * np.log(10)       # alpha -> infinity approaches the IID label distribution


def test_invalid_alpha():
    x, y = _data(100)
    with pytest.raises(ValueError):
        partition_data_dirichlet(x, y, 4, 0.0, seed=0)
