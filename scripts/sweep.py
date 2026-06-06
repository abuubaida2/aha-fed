"""Multi-seed sweep runner.

Executes a single configuration across N seeds, then reports
mean ± std for each tracked metric. Writes a JSON summary to
``runs/sweeps/{tag}.json``.

Example:
    python scripts/sweep.py --aggregator adaptive --beta 2 --gamma 4 --seeds 5
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Emit UTF-8 so Unicode in the summary (e.g. ±) survives non-UTF-8 consoles.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--aggregator", default="fedavg")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--proximal-mu", type=float, default=0.0)
    p.add_argument("--num-clients", type=int, default=4)
    p.add_argument("--num-rounds", type=int, default=15)
    p.add_argument("--samples-per-client", type=int, default=200)
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--heterogeneity", type=float, default=0.7)
    p.add_argument("--noisy-client", type=int, default=-1)
    p.add_argument("--noisy-rate", type=float, default=0.0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--tag", default="sweep")
    return p.parse_args()


def run_single(args, seed: int) -> dict:
    set_seed(seed)
    rates = [0.0] * args.num_clients
    if 0 <= args.noisy_client < args.num_clients:
        rates[args.noisy_client] = args.noisy_rate

    client_datasets = make_synthetic_clients(
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        num_classes=args.num_classes,
        image_size=args.image_size,
        heterogeneity=args.heterogeneity,
        base_seed=seed,
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
        seed=seed + 9999,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=32)

    if args.aggregator == "adaptive":
        agg = build_aggregator(
            "adaptive", alpha=args.alpha, beta=args.beta, gamma=args.gamma
        )
    else:
        agg = build_aggregator(args.aggregator)

    model = build_model("small_cnn", num_classes=args.num_classes, pretrained=False)
    server = FederatedServer(
        global_model=model,
        clients=clients,
        aggregator=agg,
        eval_loader=eval_loader,
        num_classes=args.num_classes,
        device="cpu",
    )
    history = server.run(
        num_rounds=args.num_rounds,
        local_epochs=1,
        local_lr=0.01,
        proximal_mu=args.proximal_mu,
        eval_every=args.num_rounds,
    )
    final = history[-1]
    return {"seed": seed, "accuracy": final["accuracy"], "loss": final["loss"]}


def summarize(runs: list[dict]) -> dict:
    accs = [r["accuracy"] for r in runs]
    losses = [r["loss"] for r in runs]

    def mean_std(xs: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.mean(xs),
            "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs),
            "max": max(xs),
        }

    return {
        "accuracy": mean_std(accs),
        "loss": mean_std(losses),
        "n_seeds": len(runs),
    }


def main() -> None:
    args = parse_args()
    print(f"Running {args.aggregator} for {args.seeds} seeds...")
    runs = []
    for s in range(args.seeds):
        result = run_single(args, seed=42 + s)
        print(
            f"  seed {result['seed']:>3d}: "
            f"acc={result['accuracy']:.4f} loss={result['loss']:.4f}"
        )
        runs.append(result)

    summary = summarize(runs)
    out = {"args": vars(args), "runs": runs, "summary": summary}

    out_dir = ROOT / "runs" / "sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n--- Summary ({args.aggregator}, n={args.seeds}) ---")
    print(
        f"accuracy: {summary['accuracy']['mean']:.4f} "
        f"± {summary['accuracy']['std']:.4f}  "
        f"[{summary['accuracy']['min']:.4f}, {summary['accuracy']['max']:.4f}]"
    )
    print(
        f"loss    : {summary['loss']['mean']:.4f} "
        f"± {summary['loss']['std']:.4f}"
    )
    print(f"saved   : {out_path}")


if __name__ == "__main__":
    main()
