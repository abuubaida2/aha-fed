"""FedAvg: weighted average of client model updates by data volume.

Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data" (AISTATS 2017).
"""
from __future__ import annotations

from typing import Sequence

import torch


class FedAvgAggregator:
    def aggregate(
        self,
        client_states: Sequence[dict[str, torch.Tensor]],
        client_sizes: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        if len(client_states) != len(client_sizes):
            raise ValueError("client_states and client_sizes must align")
        total = float(sum(client_sizes))
        if total <= 0:
            raise ValueError("client_sizes must sum to a positive value")

        weights = [n / total for n in client_sizes]
        aggregated: dict[str, torch.Tensor] = {}
        for key in client_states[0].keys():
            stacked = torch.stack(
                [w * state[key].float() for w, state in zip(weights, client_states)],
                dim=0,
            )
            aggregated[key] = stacked.sum(dim=0)
        return aggregated
