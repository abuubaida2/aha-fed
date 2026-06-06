"""End-to-end FL experiment runner.

Runs FedAvg, FedProx, FedBN, or AHA-Fed (adaptive) on synthetic, multi-center
medical-imaging data. Real datasets plug in by replacing
``make_synthetic_clients`` with a per-dataset loader that returns one
``Dataset`` per client/center.

Quickstart:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --aggregator fedprox --proximal-mu 0.01
    python scripts/run_experiment.py --aggregator fedbn
    python scripts/run_experiment.py --aggregator adaptive --beta 4 --gamma 4
    python scripts/run_experiment.py --noisy-client 3 --noisy-rate 0.5
    python scripts/run_experiment.py --dp-epsilon 1.0 --dp-delta 1e-5
    python scripts/run_experiment.py --output runs/exp1.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
    p.add_argument(
        "--aggregator",
        choices=["fedavg", "fedprox", "fedbn", "adaptive"],
        default="fedavg",
    )
    p.add_argument("--proximal-mu", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=1.0, help="AHA-Fed: log-volume coefficient")
    p.add_argument("--beta", type=float, default=2.0, help="AHA-Fed: distance penalty")
    p.add_argument("--gamma", type=float, default=2.0, help="AHA-Fed: quality reward")
    p.add_argument("--num-clients", type=int, default=4)
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--local-lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--samples-per-client", type=int, default=200)
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--heterogeneity", type=float, default=0.7)
    p.add_argument("--noisy-client", type=int, default=-1)
    p.add_argument("--noisy-rate", type=float, default=0.0)
    p.add_argument("--dp-epsilon", type=float, default=None, help="enable DP-SGD with target epsilon")
    p.add_argument("--dp-delta", type=float, default=1e-5)
    p.add_argument("--dp-max-grad-norm", type=float, default=1.0)
    p.add_argument("--arch", default="small_cnn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None, help="JSON path for results")
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return p.parse_args()


def build_label_noise(args) -> list[float]:
    rates = [0.0] * args.num_clients
    if 0 <= args.noisy_client < args.num_clients:
        rates[args.noisy_client] = args.noisy_rate
    return rates


def build_aggregator_from_args(args):
    if args.aggregator == "adaptive":
        return build_aggregator(
            "adaptive", alpha=args.alpha, beta=args.beta, gamma=args.gamma
        )
    return build_aggregator(args.aggregator)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    label_noise = build_label_noise(args)
    print(
        f"Building {args.num_clients} synthetic clients "
        f"(heterogeneity={args.heterogeneity}, noise rates={label_noise})..."
    )
    client_datasets = make_synthetic_clients(
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        num_classes=args.num_classes,
        image_size=args.image_size,
        heterogeneity=args.heterogeneity,
        base_seed=args.seed,
        label_noise=label_noise,
    )
    clients = [
        FederatedClient(i, ds, batch_size=args.batch_size, device=args.device)
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
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

    model = build_model(
        args.arch, num_classes=args.num_classes, pretrained=False
    )
    server = FederatedServer(
        global_model=model,
        clients=clients,
        aggregator=build_aggregator_from_args(args),
        eval_loader=eval_loader,
        num_classes=args.num_classes,
        device=args.device,
    )

    dp_config = None
    if args.dp_epsilon is not None:
        dp_config = {
            "epsilon": args.dp_epsilon,
            "delta": args.dp_delta,
            "max_grad_norm": args.dp_max_grad_norm,
        }
        print(f"DP-SGD enabled: epsilon={args.dp_epsilon}, delta={args.dp_delta}")

    print(
        f"Starting FL training: {args.aggregator} for {args.num_rounds} rounds"
    )
    history = server.run(
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        local_lr=args.local_lr,
        proximal_mu=args.proximal_mu,
        eval_every=max(1, args.num_rounds // 10),
        dp_config=dp_config,
        output_path=args.output,
        run_config=vars(args),
    )
    print(f"Final metrics: {history[-1]}")


if __name__ == "__main__":
    main()
