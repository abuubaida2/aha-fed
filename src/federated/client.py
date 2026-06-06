"""Federated client: holds local data, performs local training, returns
the updated state plus a local quality signal used by AHA-Fed.

The quality signal is q_k = 1 / (1 + L_k), where L_k is the average
training-set cross-entropy after local epochs. Heavily noisy or mis-labeled
clients cannot drive their loss arbitrarily low, so q_k stays small for
them — providing a downweighting signal to AHA-Fed.

Optional DP-SGD training is supported via ``dp_config``. When enabled, the
client uses Opacus to enforce per-client (epsilon, delta)-DP on its local
training round.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .aggregators.fedprox import fedprox_proximal_term


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.device = device

    def __len__(self) -> int:
        return len(self.dataset)

    def local_train(
        self,
        model: nn.Module,
        epochs: int,
        lr: float,
        proximal_mu: float = 0.0,
        dp_config: dict | None = None,
    ) -> dict:
        if dp_config is None:
            return self._train_standard(model, epochs, lr, proximal_mu)
        if proximal_mu > 0:
            raise ValueError(
                "proximal_mu (FedProx) is not supported with DP-SGD; "
                "set proximal_mu=0 when dp_config is provided."
            )
        return self._train_dp(model, epochs, lr, dp_config)

    def _train_standard(
        self,
        model: nn.Module,
        epochs: int,
        lr: float,
        proximal_mu: float,
    ) -> dict:
        model = model.to(self.device).train()
        global_params = (
            [p.detach().clone() for p in model.parameters()]
            if proximal_mu > 0
            else None
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                if proximal_mu > 0 and global_params is not None:
                    loss = loss + fedprox_proximal_term(
                        list(model.parameters()), global_params, proximal_mu
                    )
                loss.backward()
                optimizer.step()

        train_loss = self._train_loss(model, criterion)
        quality = 1.0 / (1.0 + train_loss)
        return {
            "state": {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            },
            "quality": quality,
            "train_loss": train_loss,
        }

    def _train_dp(
        self,
        model: nn.Module,
        epochs: int,
        lr: float,
        dp_config: dict,
    ) -> dict:
        from .privacy.dp_optimizer import attach_dp

        model = model.to(self.device).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        private_model, private_opt, private_loader = attach_dp(
            model=model,
            optimizer=optimizer,
            data_loader=self.loader,
            target_epsilon=float(dp_config["epsilon"]),
            target_delta=float(dp_config["delta"]),
            epochs=epochs,
            max_grad_norm=float(dp_config.get("max_grad_norm", 1.0)),
        )
        criterion = nn.CrossEntropyLoss()
        private_model.train()
        for _ in range(epochs):
            for x, y in private_loader:
                x, y = x.to(self.device), y.to(self.device)
                private_opt.zero_grad()
                loss = criterion(private_model(x), y)
                loss.backward()
                private_opt.step()

        inner = (
            private_model._module
            if hasattr(private_model, "_module")
            else private_model
        )
        train_loss = self._train_loss(inner, criterion)
        quality = 1.0 / (1.0 + train_loss)
        return {
            "state": {
                k: v.detach().cpu().clone()
                for k, v in inner.state_dict().items()
            },
            "quality": quality,
            "train_loss": train_loss,
            "dp_epsilon_used": float(dp_config["epsilon"]),
        }

    @torch.no_grad()
    def _train_loss(self, model: nn.Module, criterion: nn.Module) -> float:
        was_training = model.training
        model.eval()
        total_loss = 0.0
        total = 0
        for x, y in self.loader:
            x, y = x.to(self.device), y.to(self.device)
            total_loss += criterion(model(x), y).item() * y.shape[0]
            total += y.shape[0]
        if was_training:
            model.train()
        return total_loss / max(total, 1)
