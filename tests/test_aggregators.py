import torch

from src.federated.aggregators.fedavg import FedAvgAggregator
from src.federated.aggregators.fedprox import (
    FedProxAggregator,
    fedprox_proximal_term,
)


def test_fedavg_uniform_weights_average():
    agg = FedAvgAggregator()
    s1 = {"w": torch.tensor([1.0, 1.0])}
    s2 = {"w": torch.tensor([3.0, 3.0])}
    out = agg.aggregate([s1, s2], [10, 10])
    assert torch.allclose(out["w"], torch.tensor([2.0, 2.0]))


def test_fedavg_volume_weighted():
    agg = FedAvgAggregator()
    s1 = {"w": torch.tensor([0.0])}
    s2 = {"w": torch.tensor([4.0])}
    out = agg.aggregate([s1, s2], [10, 30])
    assert torch.allclose(out["w"], torch.tensor([3.0]))


def test_fedprox_aggregation_matches_fedavg():
    s1 = {"w": torch.tensor([1.0])}
    s2 = {"w": torch.tensor([3.0])}
    fedavg = FedAvgAggregator().aggregate([s1, s2], [1, 1])
    fedprox = FedProxAggregator().aggregate([s1, s2], [1, 1])
    assert torch.allclose(fedavg["w"], fedprox["w"])


def test_fedprox_proximal_zero_at_global():
    p = [torch.zeros(3)]
    g = [torch.zeros(3)]
    assert fedprox_proximal_term(p, g, mu=1.0).item() == 0.0


def test_fedprox_proximal_quadratic():
    p = [torch.tensor([1.0, 0.0, 0.0])]
    g = [torch.zeros(3)]
    # 0.5 * mu * ||p - g||^2 = 0.5 * 2 * 1 = 1.0
    assert torch.allclose(fedprox_proximal_term(p, g, mu=2.0), torch.tensor(1.0))
