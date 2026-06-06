"""Federated server: orchestrates rounds, evaluation, and aggregation.

Computes per-client label distributions once at construction, then provides
the across-client distances and per-round quality signals required by
AHA-Fed. Standard aggregators (FedAvg, FedProx, FedBN) ignore the extra
signals and behave exactly as before.

Evaluation reports loss, accuracy, AUROC (one-vs-rest macro), AUPRC, and
ECE on top-1 confidence. Optionally writes a per-run JSON summary.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import (
    expected_calibration_error,
    multiclass_auprc,
    multiclass_auroc,
)

from .aggregators.adaptive import AdaptiveAggregator
from .aggregators.fedavg import FedAvgAggregator
from .aggregators.fedbn import FedBNAggregator
from .aggregators.feddisco import FedDiscoAggregator
from .aggregators.fedprox import FedProxAggregator
from .client import FederatedClient


def build_aggregator(name: str, **kwargs):
    if name == "fedavg":
        return FedAvgAggregator()
    if name == "fedprox":
        return FedProxAggregator()
    if name == "fedbn":
        return FedBNAggregator()
    if name == "feddisco":
        return FedDiscoAggregator(**kwargs)
    if name == "adaptive":
        return AdaptiveAggregator(**kwargs)
    raise ValueError(f"unknown aggregator: {name}")


class FederatedServer:
    def __init__(
        self,
        global_model: nn.Module,
        clients: list[FederatedClient],
        aggregator,
        eval_loader: DataLoader,
        num_classes: int,
        device: str = "cpu",
    ):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.aggregator = aggregator
        self.eval_loader = eval_loader
        self.num_classes = num_classes
        self.device = device

        self.client_priors = self._empirical_priors()
        self.mean_prior = torch.stack(self.client_priors).mean(dim=0)

    def _empirical_priors(self) -> list[torch.Tensor]:
        priors: list[torch.Tensor] = []
        for c in self.clients:
            counts = torch.zeros(self.num_classes)
            for _, y in c.loader:
                for v in y.view(-1):
                    counts[int(v.item())] += 1.0
            total = counts.sum()
            prior = (
                counts / total
                if total > 0
                else torch.ones(self.num_classes) / self.num_classes
            )
            priors.append(prior)
        return priors

    def _client_distances_to_consensus(self) -> list[float]:
        return [
            0.5 * (p - self.mean_prior).abs().sum().item()
            for p in self.client_priors
        ]

    def run(
        self,
        num_rounds: int,
        local_epochs: int,
        local_lr: float,
        proximal_mu: float = 0.0,
        eval_every: int = 5,
        dp_config: dict | None = None,
        output_path: str | Path | None = None,
        run_config: dict | None = None,
        round_callback=None,
    ) -> list[dict]:
        history: list[dict] = []
        is_adaptive = isinstance(self.aggregator, AdaptiveAggregator)
        is_feddisco = isinstance(self.aggregator, FedDiscoAggregator)
        needs_distances = is_adaptive or is_feddisco
        distances = (
            self._client_distances_to_consensus() if needs_distances else None
        )

        for r in range(1, num_rounds + 1):
            client_states: list[dict] = []
            client_sizes: list[int] = []
            client_quality: list[float] = []
            for c in self.clients:
                local_model = copy.deepcopy(self.global_model)
                res = c.local_train(
                    local_model,
                    epochs=local_epochs,
                    lr=local_lr,
                    proximal_mu=proximal_mu,
                    dp_config=dp_config,
                )
                client_states.append(res["state"])
                client_sizes.append(len(c))
                client_quality.append(res["quality"])

            if is_adaptive:
                agg_state = self.aggregator.aggregate(
                    client_states,
                    client_sizes,
                    client_distances=distances,
                    client_quality=client_quality,
                )
            elif is_feddisco:
                agg_state = self.aggregator.aggregate(
                    client_states,
                    client_sizes,
                    client_distances=distances,
                )
            else:
                agg_state = self.aggregator.aggregate(
                    client_states, client_sizes
                )
            self.global_model.load_state_dict(agg_state, strict=False)

            if r % eval_every == 0 or r == num_rounds:
                metrics = self.evaluate()
                metrics["round"] = r
                if is_adaptive:
                    metrics["client_quality"] = [
                        round(q, 3) for q in client_quality
                    ]
                history.append(metrics)
                self._print_round(metrics)
                if round_callback is not None:
                    round_callback(metrics)

        if output_path is not None:
            self._write_json(output_path, history, run_config, dp_config)
        return history

    @staticmethod
    def _print_round(metrics: dict) -> None:
        parts = [
            f"[round {metrics['round']:>3d}]",
            f"loss={metrics['loss']:.4f}",
            f"acc={metrics['accuracy']:.4f}",
        ]
        if metrics.get("auroc") is not None:
            parts.append(f"auroc={metrics['auroc']:.4f}")
        if metrics.get("ece") is not None:
            parts.append(f"ece={metrics['ece']:.4f}")
        print(" ".join(parts))

    @staticmethod
    def _write_json(
        path: str | Path,
        history: list[dict],
        run_config: dict | None,
        dp_config: dict | None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": run_config or {},
            "dp_config": dp_config,
            "history": history,
            "final": history[-1] if history else None,
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"Wrote results: {path}")

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum")
        total = 0
        correct = 0
        loss_sum = 0.0
        all_probs: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        for x, y in self.eval_loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.global_model(x)
            loss_sum += criterion(out, y).item()
            probs = torch.softmax(out, dim=1)
            correct += (out.argmax(1) == y).sum().item()
            total += y.shape[0]
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())
        self.global_model.train()

        probs_np = torch.cat(all_probs).numpy()
        labels_np = torch.cat(all_labels).numpy()
        preds = probs_np.argmax(axis=1)
        top_conf = probs_np.max(axis=1)
        correct_mask = (preds == labels_np).astype(np.float32)

        return {
            "loss": loss_sum / total,
            "accuracy": correct / total,
            "auroc": multiclass_auroc(probs_np, labels_np),
            "auprc": multiclass_auprc(probs_np, labels_np),
            "ece": expected_calibration_error(top_conf, correct_mask),
        }
