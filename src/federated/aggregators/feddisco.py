"""FedDisco: discrepancy-aware aggregation (Ye et al., ICML 2023).

Per-round client weights are proportional to:

    log w_k ∝ log n_k − a · D_k

where D_k is the total-variation distance between the client's empirical
label prior and the across-client consensus. This is the natural "no
quality term" baseline against AHA-Fed: it shares the heterogeneity
penalty but lacks the loss-derived quality reward, so it cannot identify
a noisy client whose prior is unremarkable.

Reduces to FedAvg when a = 0.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

from .fedavg import FedAvgAggregator


class FedDiscoAggregator(FedAvgAggregator):
    def __init__(self, a: float = 1.0):
        self.a = a

    def compute_weights(
        self,
        client_sizes: Sequence[int],
        client_distances: Sequence[float],
    ) -> list[float]:
        log_scores = [
            math.log(max(client_sizes[i], 1)) - self.a * client_distances[i]
            for i in range(len(client_sizes))
        ]
        m = max(log_scores)
        exps = [math.exp(s - m) for s in log_scores]
        z = sum(exps)
        return [e / z for e in exps]

    def aggregate(
        self,
        client_states: Sequence[dict[str, torch.Tensor]],
        client_sizes: Sequence[int],
        client_distances: Sequence[float] | None = None,
        client_quality: Sequence[float] | None = None,  # ignored
    ) -> dict[str, torch.Tensor]:
        if client_distances is None:
            return super().aggregate(client_states, client_sizes)
        weights = self.compute_weights(client_sizes, client_distances)
        out: dict[str, torch.Tensor] = {}
        for key in client_states[0].keys():
            stacked = torch.stack(
                [w * s[key].float() for w, s in zip(weights, client_states)],
                dim=0,
            )
            out[key] = stacked.sum(dim=0)
        return out
