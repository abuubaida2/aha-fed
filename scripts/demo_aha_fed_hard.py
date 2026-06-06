"""Harder AHA-Fed vs FedAvg demo: 6 clients, 3 with severe label noise."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Emit UTF-8 so Unicode in the summary (e.g. Δ) survives non-UTF-8 consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from torch.utils.data import DataLoader

from src.data.datasets import SyntheticMedicalImaging
from src.data.partitioning import make_synthetic_clients
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer, build_aggregator
from src.models.backbones import build_model
from src.utils.seed import set_seed

NUM_CLIENTS = 6
NUM_CLASSES = 6
SAMPLES_PER = 80
IMAGE_SIZE = 16
NUM_ROUNDS = 30
SEED = 7
NOISE_RATES = [0.0, 0.0, 0.0, 0.7, 0.7, 0.7]


def run(name: str, kwargs: dict | None = None) -> dict:
    set_seed(SEED)
    ds = make_synthetic_clients(
        num_clients=NUM_CLIENTS,
        samples_per_client=SAMPLES_PER,
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        heterogeneity=0.0,
        base_seed=SEED,
        label_noise=NOISE_RATES,
    )
    clients = [FederatedClient(i, d, 32, "cpu") for i, d in enumerate(ds)]
    eval_ds = SyntheticMedicalImaging(
        num_samples=400,
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        center_id=0,
        center_offset=0.0,
        seed=SEED + 9999,
    )
    eval_loader = DataLoader(eval_ds, batch_size=32)
    model = build_model("small_cnn", NUM_CLASSES, False)
    agg = build_aggregator(name, **(kwargs or {}))
    server = FederatedServer(
        model, clients, agg, eval_loader, NUM_CLASSES, "cpu"
    )
    history = server.run(NUM_ROUNDS, 1, 0.01, 0.0, max(1, NUM_ROUNDS // 5))
    return history[-1]


def main() -> None:
    print(f"Setup: {NUM_CLIENTS} clients, noise rates = {NOISE_RATES}")
    print("=" * 60)
    print("--- FedAvg ---")
    fa = run("fedavg")
    print("--- AHA-Fed (alpha=1, beta=2, gamma=4) ---")
    ah = run("adaptive", {"alpha": 1.0, "beta": 2.0, "gamma": 4.0})

    print()
    print("=" * 60)
    print(f"FedAvg   : acc = {fa['accuracy']:.4f}   loss = {fa['loss']:.4f}")
    print(f"AHA-Fed  : acc = {ah['accuracy']:.4f}   loss = {ah['loss']:.4f}")
    print(f"Δ acc    : {ah['accuracy'] - fa['accuracy']:+.4f}")
    print(f"Δ loss   : {fa['loss'] - ah['loss']:+.4f}")
    if "client_quality" in ah:
        print(f"AHA-Fed final quality: {ah['client_quality']}")


if __name__ == "__main__":
    main()
