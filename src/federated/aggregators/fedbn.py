"""FedBN: Federated Learning on Non-IID Features via Local Batch Normalization.

Aggregates all parameters EXCEPT BatchNorm parameters and running statistics,
which are kept local per client. This is critical for medical-imaging FL
where each scanner produces a different feature-statistic distribution.

Reference: Li et al., "FedBN: Federated Learning on Non-IID Features via
Local Batch Normalization" (ICLR 2021).
"""
from __future__ import annotations

from typing import Sequence

import torch

from .fedavg import FedAvgAggregator


class FedBNAggregator(FedAvgAggregator):
    BN_KEY_HINTS: tuple[str, ...] = (
        ".bn",
        "batchnorm",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    )

    def __init__(self, bn_key_hints: Sequence[str] | None = None):
        self.bn_key_hints = (
            tuple(bn_key_hints)
            if bn_key_hints is not None
            else self.BN_KEY_HINTS
        )

    def _is_bn_key(self, key: str) -> bool:
        lower = key.lower()
        return any(hint in lower for hint in self.bn_key_hints)

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
            if self._is_bn_key(key):
                aggregated[key] = client_states[0][key].clone()
            else:
                stacked = torch.stack(
                    [
                        w * state[key].float()
                        for w, state in zip(weights, client_states)
                    ],
                    dim=0,
                )
                aggregated[key] = stacked.sum(dim=0)
        return aggregated
