"""MedMNIST loader with Dirichlet-based federation across centers.

Provides:
- ``load_pathmnist`` and ``load_pneumoniamnist`` returning torch Dataset objects
  for train / val / test splits.
- ``federate_dirichlet`` partitioning a dataset into K clients with a
  Dirichlet-controlled label-prior heterogeneity, optionally injecting
  per-client label noise.

PathMNIST: 9-class colon-pathology multi-class (89,996 train / 7,180 test).
PneumoniaMNIST: binary chest-X-ray pneumonia (4,708 train / 624 test).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

try:
    from medmnist import PathMNIST, PneumoniaMNIST
except ImportError as e:
    raise RuntimeError(
        "medmnist is not installed. Run: pip install medmnist"
    ) from e


class MedMNISTWrapper(Dataset):
    """Tensor wrapper around a MedMNIST split.

    Returns (image_tensor [3, H, H], int_label) tuples — uniform with the
    rest of the project (image is always 3-channel float).
    """
    def __init__(self, base, image_size: int = 28):
        self.base = base
        self.size = image_size
        # cache as tensors
        x = base.imgs            # uint8, [N, H, W] or [N, H, W, 3]
        y = base.labels          # [N, 1] for multi-class; [N, C] for multi-label
        if x.ndim == 3:
            x = np.repeat(x[:, None, :, :], 3, axis=1)  # [N, 3, H, W]
        else:
            x = x.transpose(0, 3, 1, 2)                 # [N, 3, H, W]
        x = x.astype(np.float32) / 255.0
        # multi-class with shape [N, 1] -> [N]
        y = y.squeeze(-1) if (y.ndim == 2 and y.shape[1] == 1) else y
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def load_pathmnist(split: str = "train", image_size: int = 28) -> MedMNISTWrapper:
    base = PathMNIST(split=split, download=True, size=image_size)
    return MedMNISTWrapper(base, image_size=image_size)


def load_pneumoniamnist(split: str = "train", image_size: int = 28) -> MedMNISTWrapper:
    base = PneumoniaMNIST(split=split, download=True, size=image_size)
    return MedMNISTWrapper(base, image_size=image_size)


def federate_dirichlet(
    dataset: MedMNISTWrapper,
    num_clients: int,
    num_classes: int,
    *,
    heterogeneity: float = 0.5,
    base_seed: int = 42,
    label_noise: Sequence[float] | None = None,
    samples_per_client: int | None = None,
) -> list[Subset]:
    """Partition into K clients with Dirichlet(α) label-prior heterogeneity.

    Args:
        heterogeneity: 0.0 → IID across clients; 1.0 → highly skewed priors.
            Internally translates to Dirichlet concentration α = (1 - h) * 5.
        label_noise: per-client probability of replacing a label with a uniform
            random class. Applied in-place to a copy of dataset.y.
        samples_per_client: cap each client's sample count for fast experiments.

    Returns:
        List of K ``Subset`` objects. Each subset's underlying dataset is a
        clone with possibly noised labels (so different clients can have
        different noise rates without crosstalk).
    """
    rng = np.random.default_rng(base_seed)
    n = len(dataset)
    y = dataset.y.numpy()

    # Heterogeneity → concentration. h=0 → very large alpha (uniform);
    # h=1 → very small alpha (one-hot).
    alpha = max(0.05, (1.0 - heterogeneity) * 5.0)

    # Per-class index buckets
    class_idx = [np.where(y == c)[0].tolist() for c in range(num_classes)]
    for c in class_idx:
        rng.shuffle(c)

    # Sample Dirichlet proportions per class across clients
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        proportions = rng.dirichlet([alpha] * num_clients)
        # number of items of class c going to each client
        counts = (proportions * len(class_idx[c])).astype(int)
        # fix rounding
        counts[-1] = len(class_idx[c]) - counts[:-1].sum()
        offset = 0
        for k in range(num_clients):
            client_indices[k].extend(
                class_idx[c][offset: offset + counts[k]]
            )
            offset += counts[k]

    # Optional per-client cap
    if samples_per_client is not None:
        for k in range(num_clients):
            rng.shuffle(client_indices[k])
            client_indices[k] = client_indices[k][:samples_per_client]

    # Inject per-client label noise into a per-client clone of labels
    rates = list(label_noise) if label_noise else [0.0] * num_clients
    if len(rates) != num_clients:
        raise ValueError("label_noise length must equal num_clients")

    # Build per-client subsets, possibly with noised labels.
    clients: list[Subset] = []
    for k in range(num_clients):
        idx = np.array(client_indices[k], dtype=np.int64)
        # We'll create a per-client wrapper that shares the X tensor but owns
        # a noised copy of y, so different clients can have different noise.
        if rates[k] > 0:
            local = _NoisedView(dataset, idx, rates[k], num_classes,
                                seed=base_seed + 7919 * k + 1)
            clients.append(local)
        else:
            local = _NoisedView(dataset, idx, 0.0, num_classes,
                                seed=base_seed + 7919 * k + 1)
            clients.append(local)
    return clients


class _NoisedView(Dataset):
    """Subset with optional uniform-random label noise."""
    def __init__(self, base: MedMNISTWrapper, indices: np.ndarray,
                 noise_rate: float, num_classes: int, seed: int):
        self.base = base
        self.indices = indices
        rng = np.random.default_rng(seed)
        y = base.y[indices].clone().numpy()
        if noise_rate > 0:
            mask = rng.random(len(y)) < noise_rate
            random_labels = rng.integers(0, num_classes, size=len(y))
            y = np.where(mask, random_labels, y)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x = self.base.x[self.indices[idx]]
        return x, self.y[idx]
