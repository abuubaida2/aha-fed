"""Unit tests for AHA-Fed weight computation and aggregation."""
import math

import torch

from src.federated.aggregators.adaptive import AdaptiveAggregator


def test_aha_fed_reduces_to_fedavg_when_beta_gamma_zero():
    agg = AdaptiveAggregator(alpha=1.0, beta=0.0, gamma=0.0)
    sizes = [10, 30]
    distances = [0.0, 0.5]
    quality = [1.0, 0.1]
    weights = agg.compute_weights(sizes, distances, quality)
    expected = [10 / 40, 30 / 40]
    assert math.isclose(weights[0], expected[0], rel_tol=1e-6)
    assert math.isclose(weights[1], expected[1], rel_tol=1e-6)


def test_aha_fed_downweights_distant_client():
    agg = AdaptiveAggregator(alpha=1.0, beta=4.0, gamma=0.0)
    sizes = [100, 100]
    quality = [1.0, 1.0]
    distances_close = [0.0, 0.0]
    distances_far = [0.0, 1.0]
    w_close = agg.compute_weights(sizes, distances_close, quality)
    w_far = agg.compute_weights(sizes, distances_far, quality)
    assert math.isclose(w_close[0], 0.5, rel_tol=1e-6)
    assert w_far[1] < 0.5  # the distant client must lose weight


def test_aha_fed_downweights_low_quality_client():
    agg = AdaptiveAggregator(alpha=1.0, beta=0.0, gamma=4.0)
    sizes = [100, 100]
    distances = [0.0, 0.0]
    quality = [1.0, 0.1]
    w = agg.compute_weights(sizes, distances, quality)
    assert w[0] > w[1]
    assert math.isclose(sum(w), 1.0, rel_tol=1e-6)


def test_aha_fed_aggregate_respects_weights():
    agg = AdaptiveAggregator(alpha=1.0, beta=0.0, gamma=4.0)
    s1 = {"w": torch.tensor([0.0])}
    s2 = {"w": torch.tensor([10.0])}
    out = agg.aggregate(
        [s1, s2],
        client_sizes=[100, 100],
        client_distances=[0.0, 0.0],
        client_quality=[1.0, 0.01],
    )
    # Client 1 should dominate because client 2 has very low quality
    assert out["w"].item() < 1.0


def test_aha_fed_falls_back_to_fedavg_without_signals():
    agg = AdaptiveAggregator()
    s1 = {"w": torch.tensor([0.0])}
    s2 = {"w": torch.tensor([4.0])}
    out = agg.aggregate([s1, s2], client_sizes=[10, 30])
    assert torch.allclose(out["w"], torch.tensor([3.0]))
