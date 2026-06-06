"""Federated partitioning of single-source datasets and client construction."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .datasets import SyntheticMedicalImaging


def make_synthetic_clients(
    num_clients: int,
    samples_per_client: int,
    num_classes: int,
    image_size: int = 64,
    heterogeneity: float = 0.5,
    base_seed: int = 0,
    label_noise: float | Sequence[float] = 0.0,
) -> list[SyntheticMedicalImaging]:
    """Construct heterogeneous clients with Dirichlet-mixed class priors.

    Args:
        heterogeneity: in [0, 1]. 0.0 → near-uniform priors across clients
            (IID). 1.0 → highly skewed priors (peaked Dirichlet samples).
        label_noise: scalar applied to all clients, or a per-client sequence
            of noise rates with length == num_clients. Used to simulate
            labelling-quality variation across centers.
    """
    if isinstance(label_noise, (int, float)):
        noise_rates: list[float] = [float(label_noise)] * num_clients
    else:
        noise_rates = [float(r) for r in label_noise]
        if len(noise_rates) != num_clients:
            raise ValueError(
                f"label_noise length {len(noise_rates)} != num_clients {num_clients}"
            )

    alpha = 10.0 * (1.0 - heterogeneity) + 0.1 * heterogeneity
    rng = np.random.default_rng(base_seed)
    priors_np = rng.dirichlet([alpha] * num_classes, size=num_clients)
    priors = torch.from_numpy(priors_np).float()

    return [
        SyntheticMedicalImaging(
            num_samples=samples_per_client,
            num_classes=num_classes,
            image_size=image_size,
            center_id=i,
            class_prior=priors[i],
            seed=base_seed + i,
            label_noise=noise_rates[i],
        )
        for i in range(num_clients)
    ]
