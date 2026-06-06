"""Side-by-side comparison of FedAvg vs AHA-Fed on a noisy-client scenario.

Setup
-----
4 clients with IID label priors and 200 samples each. Client #3 has
``--noisy-rate`` (default 0.5) of its labels replaced uniformly at random,
simulating a single hospital with severe annotation noise. Other clients
have clean labels.

Expected
--------
FedAvg weights all clients by data volume → contaminated global model.
AHA-Fed observes that client #3's training-loss-derived quality signal
is much lower → downweights it and converges to a cleaner solution.

Usage
-----
    python scripts/demo_aha_fed.py
    python scripts/demo_aha_fed.py --noisy-rate 0.7 --num-rounds 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Emit UTF-8 so Unicode in the summary (e.g. Δ) survives non-UTF-8 consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import DataLoader

from src.data.datasets import SyntheticMedicalImaging
from src.data.partitioning import make_synthetic_clients
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer, build_aggregator
from src.models.backbones import build_model
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-clients", type=int, default=4)
    p.add_argument("--noisy-client", type=int, default=3)
    p.add_argument("--noisy-rate", type=float, default=0.5)
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--samples-per-client", type=int, default=200)
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_one(args, aggregator_name: str, agg_kwargs: dict | None = None) -> dict:
    set_seed(args.seed)
    rates = [0.0] * args.num_clients
    rates[args.noisy_client] = args.noisy_rate

    client_datasets = make_synthetic_clients(
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        num_classes=args.num_classes,
        image_size=args.image_size,
        heterogeneity=0.0,
        base_seed=args.seed,
        label_noise=rates,
    )
    clients = [
        FederatedClient(i, ds, batch_size=32, device="cpu")
        for i, ds in enumerate(client_datasets)
    ]
    eval_dataset = SyntheticMedicalImaging(
        num_samples=400,
        num_classes=args.num_classes,
        image_size=args.image_size,
        center_id=0,
        center_offset=0.0,
        seed=args.seed + 9999,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=32)
    model = build_model("small_cnn", num_classes=args.num_classes, pretrained=False)
    aggregator = build_aggregator(aggregator_name, **(agg_kwargs or {}))
    server = FederatedServer(
        global_model=model,
        clients=clients,
        aggregator=aggregator,
        eval_loader=eval_loader,
        num_classes=args.num_classes,
        device="cpu",
    )
    history = server.run(
        num_rounds=args.num_rounds,
        local_epochs=1,
        local_lr=0.01,
        eval_every=max(1, args.num_rounds // 5),
    )
    return history[-1]


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(
        f"Demo: {args.num_clients} clients, client #{args.noisy_client} "
        f"has {int(args.noisy_rate * 100)}% label noise"
    )
    print("=" * 60)

    print("\n--- FedAvg ---")
    fedavg_final = run_one(args, "fedavg")

    print("\n--- AHA-Fed (alpha=1, beta=2, gamma=4) ---")
    aha_final = run_one(
        args, "adaptive", {"alpha": 1.0, "beta": 2.0, "gamma": 4.0}
    )

    print("\n" + "=" * 60)
    print("FINAL COMPARISON (test-set accuracy on clean held-out)")
    print("=" * 60)
    print(f"  FedAvg   : acc = {fedavg_final['accuracy']:.4f}")
    print(f"  AHA-Fed  : acc = {aha_final['accuracy']:.4f}")
    delta = aha_final["accuracy"] - fedavg_final["accuracy"]
    sign = "+" if delta >= 0 else ""
    print(f"  Δ (AHA-Fed - FedAvg) = {sign}{delta:.4f}")
    if "client_quality" in aha_final:
        print(f"  AHA-Fed final-round client quality: {aha_final['client_quality']}")


if __name__ == "__main__":
    main()
