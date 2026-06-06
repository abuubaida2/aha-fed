"""FedProx: server-side aggregation matches FedAvg.

The proximal regularizer ``mu/2 * ||w - w_global||^2`` is applied client-side
during local training (see ``src/federated/client.py::FederatedClient.local_train``).

Reference: Li et al., "Federated Optimization in Heterogeneous Networks" (MLSys 2020).
"""
from __future__ import annotations

from typing import Sequence

import torch

from .fedavg import FedAvgAggregator


class FedProxAggregator(FedAvgAggregator):
    pass


def fedprox_proximal_term(
    local_params: Sequence[torch.Tensor],
    global_params: Sequence[torch.Tensor],
    mu: float,
) -> torch.Tensor:
    term = torch.zeros((), device=local_params[0].device)
    for lp, gp in zip(local_params, global_params):
        term = term + ((lp - gp) ** 2).sum()
    return 0.5 * mu * term
