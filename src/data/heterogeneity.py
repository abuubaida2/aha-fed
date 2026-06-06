"""Statistical heterogeneity metrics for federated medical imaging clients.

These metrics support:
    * Empirical characterization of inter-client distributional shift (RQ1)
    * Per-client weights for adaptive aggregation (C2 / RQ2)
    * Predicted personalization gain (C4 / RQ4)
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def label_distribution_skew(client_label_dists: Iterable[np.ndarray]) -> float:
    """Average pairwise total-variation distance between client label distributions.

    Each entry of ``client_label_dists`` is a 1-D probability vector summing to 1.
    Returns a value in [0, 1]; 0 = identical label distributions across clients.
    """
    dists = [np.asarray(d, dtype=np.float64) for d in client_label_dists]
    n = len(dists)
    if n < 2:
        return 0.0

    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 0.5 * np.abs(dists[i] - dists[j]).sum()
            pairs += 1
    return total / pairs


def feature_mmd(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Squared Maximum Mean Discrepancy with an RBF kernel.

    Use for distributional distance between two clients' feature embeddings
    (e.g. activations from a pretrained backbone).
    """
    def rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a2 = (a ** 2).sum(axis=1, keepdims=True)
        b2 = (b ** 2).sum(axis=1, keepdims=True)
        d2 = a2 + b2.T - 2.0 * a @ b.T
        return np.exp(-d2 / (2.0 * sigma ** 2))

    return float(rbf(x, x).mean() + rbf(y, y).mean() - 2.0 * rbf(x, y).mean())
