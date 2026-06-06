"""Differential-Privacy SGD wrapper using Opacus.

Used to enforce (epsilon, delta)-DP on per-client local training, supporting
contribution C3 (utility-preserving DP-FL).
"""
from __future__ import annotations

import torch.nn as nn
from torch.utils.data import DataLoader


def attach_dp(
    model: nn.Module,
    optimizer,
    data_loader: DataLoader,
    target_epsilon: float,
    target_delta: float,
    epochs: int,
    max_grad_norm: float = 1.0,
):
    """Attach an Opacus PrivacyEngine to a model + optimizer + dataloader.

    Returns the (private_model, private_optimizer, private_loader) triple.
    """
    try:
        from opacus import PrivacyEngine
    except ImportError as e:
        raise ImportError(
            "DP training requires `opacus`. Install with `pip install opacus`."
        ) from e

    engine = PrivacyEngine()
    return engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
    )
