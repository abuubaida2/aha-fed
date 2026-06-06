"""AHA-Fed: Adaptive Heterogeneity-Aware Aggregation.

Per-round client weights are computed in log-space:

    log w_k ∝ α · log(n_k) − β · D_k + γ · log(q_k + ε)

where
    n_k : client k's training-set size
    D_k : distributional distance from client k's label distribution to the
          across-client consensus (default: total-variation distance)
    q_k : client k's local quality signal in [0, 1] (e.g., 1 / (1 + train loss))
    ε   : small constant for numerical stability of the log

The vector ``w`` is normalized via numerically-stable softmax, so it always
defines a valid convex combination over clients.

Reduces to FedAvg when β = 0, γ = 0, α = 1: weights become softmax over
``log(n_k)``, which is identical (up to a constant) to ``n_k / Σ n_j``.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

from .fedavg import FedAvgAggregator


class AdaptiveAggregator(FedAvgAggregator):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        log_eps: float = 1e-6,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.log_eps = log_eps

    def compute_weights(
        self,
        client_sizes: Sequence[int],
        client_distances: Sequence[float],
        client_quality: Sequence[float],
    ) -> list[float]:
        n = len(client_sizes)
        if not (len(client_distances) == len(client_quality) == n):
            raise ValueError(
                "client_sizes, client_distances, client_quality must align"
            )

        log_scores = [
            self.alpha * math.log(max(client_sizes[i], 1))
            - self.beta * client_distances[i]
            + self.gamma * math.log(max(client_quality[i], self.log_eps))
            for i in range(n)
        ]
        max_score = max(log_scores)
        exps = [math.exp(s - max_score) for s in log_scores]
        z = sum(exps)
        return [e / z for e in exps]

    def aggregate(
        self,
        client_states: Sequence[dict[str, torch.Tensor]],
        client_sizes: Sequence[int],
        client_distances: Sequence[float] | None = None,
        client_quality: Sequence[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        if client_distances is None or client_quality is None:
            return super().aggregate(client_states, client_sizes)

        weights = self.compute_weights(
            client_sizes, client_distances, client_quality
        )
        aggregated: dict[str, torch.Tensor] = {}
        for key in client_states[0].keys():
            stacked = torch.stack(
                [
                    w * state[key].float()
                    for w, state in zip(weights, client_states)
                ],
                dim=0,
            )
            aggregated[key] = stacked.sum(dim=0)
        return aggregated
