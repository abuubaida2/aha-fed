"""Aggregate all saved runs/ JSON results into readable summary tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, pstdev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "runs" / "paper"


def load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def final_of(p: Path) -> dict:
    d = load(p) or {}
    return d.get("final", {})


def fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "  --  "


def row(name: str, f: dict, extra: str = "") -> str:
    return (
        f"  {name:<14} acc={fmt(f.get('accuracy'))}  "
        f"auroc={fmt(f.get('auroc'))}  "
        f"auprc={fmt(f.get('auprc'))}  "
        f"ece={fmt(f.get('ece'))}  "
        f"loss={fmt(f.get('loss'))}{extra}"
    )


def header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# 1. Headline aggregator comparison (synthetic, 30 rounds, noisy client #3)
header("1. AGGREGATOR COMPARISON  (4 clients, client#3 = 50% label noise)")
for name, fn in [
    ("FedAvg", "fedavg.json"),
    ("FedProx", "fedprox.json"),
    ("FedBN", "fedbn.json"),
    ("AHA-Fed", "aha_fed.json"),
]:
    f = final_of(PAPER / fn)
    if f:
        print(row(name, f))
aha = load(PAPER / "aha_fed.json")
if aha:
    print(f"\n  AHA-Fed final client quality: {aha['final'].get('client_quality')}")
    print("  -> noisy client #3 driven to ~0.42 while clean clients ~0.99")

# 2. Noise robustness sweep
header("2. NOISE ROBUSTNESS  (final accuracy / loss vs noise rate on client#3)")
print(f"  {'noise':<8}{'FedAvg acc':<14}{'AHA acc':<14}{'FedAvg loss':<14}{'AHA loss':<12}")
for nr in ["0.0", "0.3", "0.5", "0.7"]:
    fa = final_of(PAPER / f"fedavg_n{nr}.json")
    ah = final_of(PAPER / f"aha_n{nr}.json")
    if fa and ah:
        print(
            f"  {nr:<8}{fa['accuracy']:<14.4f}{ah['accuracy']:<14.4f}"
            f"{fa['loss']:<14.4f}{ah['loss']:<12.4f}"
        )

# 3. Heterogeneity sweep
header("3. HETEROGENEITY SWEEP  (final accuracy / loss vs Dirichlet skew)")
print(f"  {'het':<8}{'FedAvg acc':<14}{'AHA acc':<14}{'FedAvg loss':<14}{'AHA loss':<12}")
for h in ["0.0", "0.3", "0.5", "0.7", "0.9"]:
    fa = final_of(PAPER / f"het_fedavg_h{h}.json")
    ah = final_of(PAPER / f"het_aha_h{h}.json")
    if fa and ah:
        print(
            f"  {h:<8}{fa['accuracy']:<14.4f}{ah['accuracy']:<14.4f}"
            f"{fa['loss']:<14.4f}{ah['loss']:<12.4f}"
        )

# 4. Scale sweep
header("4. SCALE SWEEP  (final accuracy / loss vs number of clients K)")
print(f"  {'K':<8}{'FedAvg acc':<14}{'AHA acc':<14}{'FedAvg loss':<14}{'AHA loss':<12}")
for k in ["4", "8", "12"]:
    fa = final_of(PAPER / f"scale_fedavg_k{k}.json")
    ah = final_of(PAPER / f"scale_aha_k{k}.json")
    if fa and ah:
        print(
            f"  {k:<8}{fa['accuracy']:<14.4f}{ah['accuracy']:<14.4f}"
            f"{fa['loss']:<14.4f}{ah['loss']:<12.4f}"
        )

# 5. Multi-seed robustness (mean +/- std over seeds 42/123/456)
header("5. MULTI-SEED  (mean +/- std over seeds 42 / 123 / 456)")
seeds = ["42", "123", "456"]
for agg in ["fedavg", "fedprox", "fedbn", "feddisco", "aha_fed"]:
    accs, losses = [], []
    for s in seeds:
        f = final_of(PAPER / f"seed_{agg}_s{s}.json")
        if f:
            accs.append(f["accuracy"])
            losses.append(f["loss"])
    if accs:
        sd_a = pstdev(accs) if len(accs) > 1 else 0.0
        sd_l = pstdev(losses) if len(losses) > 1 else 0.0
        print(
            f"  {agg:<12} acc={mean(accs):.4f} +/- {sd_a:.4f}    "
            f"loss={mean(losses):.4f} +/- {sd_l:.4f}   (n={len(accs)})"
        )

# 6. Real dataset: PneumoniaMNIST
header("6. REAL DATA: PneumoniaMNIST")
real = PAPER / "real"
for name, fn in [
    ("FedAvg", "pneu_fedavg.json"),
    ("FedProx", "pneu_fedprox.json"),
    ("FedBN", "pneu_fedbn.json"),
    ("FedDisco", "pneu_feddisco.json"),
    ("AHA-Fed", "pneu_aha_fed.json"),
]:
    f = final_of(real / fn)
    if f:
        print(row(name, f))

# 7. Differential privacy trade-off
header("7. DIFFERENTIAL PRIVACY  (accuracy vs privacy budget epsilon)")
print(f"  {'epsilon':<10}{'FedAvg acc':<14}{'AHA acc':<14}")
for eps in ["0.5", "1.0", "5.0"]:
    fa = final_of(real / f"path_fedavg_dp_eps{eps}.json")
    ah = final_of(real / f"path_aha_dp_eps{eps}.json")
    if fa or ah:
        fav = f"{fa['accuracy']:.4f}" if fa else "--"
        ahv = f"{ah['accuracy']:.4f}" if ah else "--"
        print(f"  {eps:<10}{fav:<14}{ahv:<14}")

# 8. AHA-Fed hyperparameter grid (beta x gamma) on real data
header("8. AHA-FED HYPERPARAMETER GRID  (final accuracy, beta rows x gamma cols)")
betas = ["0", "1", "2", "4", "8"]
gammas = ["0", "1", "2", "4", "8"]
print("  beta\\gamma " + "".join(f"{('g=' + g):<9}" for g in gammas))
for b in betas:
    cells = []
    for g in gammas:
        f = final_of(real / f"hp_b{b}_g{g}.json")
        cells.append(f"{f['accuracy']:.3f}" if f else "  -- ")
    print(f"  b={b:<8} " + "".join(f"{c:<9}" for c in cells))

print()
