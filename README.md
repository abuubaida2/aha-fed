# Federated Learning for Medical Imaging

Heterogeneity-aware and privacy-preserving federated learning across multi-institutional medical imaging datasets.

## Features

- **Aggregators:** FedAvg, FedProx, FedBN, AHA-Fed (adaptive heterogeneity-aware)
- **Differential Privacy:** opt-in DP-SGD per client via Opacus, with end-to-end (ε, δ) tracking
- **Real-dataset loaders:** CheXpert (multi-label, view-based federation) and NIH ChestX-ray14 (patient-disjoint federation)
- **Synthetic playground:** controllable label noise, per-center distribution shift, Dirichlet-mixed class priors
- **Clinical-grade metrics:** AUROC, AUPRC, ECE (calibration), worst-center / worst-group gap, communication cost
- **Reproducibility:** seeded runs, JSON result persistence, multi-seed sweeper with mean ± std summaries
- **Tests:** unit tests for FedAvg, FedProx, and AHA-Fed weight computation

## Structure

```
configs/        Hydra configs for experiments
src/
  data/         Dataset loaders (synthetic, CheXpert, NIH), partitioning, heterogeneity metrics
  models/       Backbones (SmallCNN, DenseNet)
  federated/    Server, client (DP-aware), aggregators, privacy hooks
  evaluation/   AUROC, AUPRC, ECE, fairness, communication-cost metrics
  utils/        Reproducibility helpers
experiments/    Per-experiment scripts
notebooks/      Exploratory analysis
scripts/        run_experiment, debug_train, demo_aha_fed, demo_aha_fed_hard, sweep
tests/          Unit tests
```

## Setup

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
# FedAvg baseline
python scripts/run_experiment.py --aggregator fedavg

# FedProx with proximal mu=0.01
python scripts/run_experiment.py --aggregator fedprox --proximal-mu 0.01

# FedBN (BN params kept local — important for medical imaging)
python scripts/run_experiment.py --aggregator fedbn

# AHA-Fed (adaptive aggregation)
python scripts/run_experiment.py --aggregator adaptive --beta 4 --gamma 4

# AHA-Fed vs FedAvg head-to-head on a noisy-client scenario
python scripts/demo_aha_fed.py

# DP-SGD (epsilon = 1.0, delta = 1e-5)
python scripts/run_experiment.py --dp-epsilon 1.0 --dp-delta 1e-5

# Save run results to JSON
python scripts/run_experiment.py --output runs/exp1.json

# Multi-seed sweep (mean ± std)
python scripts/sweep.py --aggregator adaptive --beta 2 --gamma 4 --seeds 5 --tag aha_default
```

## AHA-Fed in one line

Per-round client weights are softmax over `α·log(n_k) − β·D_k + γ·log(q_k + ε)`, where `n_k` is data volume, `D_k` is distance from the across-client consensus, and `q_k = 1/(1 + train_loss)` is a local quality signal. Reduces to FedAvg at `β=γ=0`.

## Real datasets

Loaders are present for CheXpert (`src/data/chexpert.py`) and NIH ChestX-ray14 (`src/data/nih_chestxray.py`); both expect data downloaded under their standard layouts and raise `FileNotFoundError` with a path hint otherwise. Federation helpers:

- `federate_chexpert_by_view(...)` — split by Frontal/Lateral as a quick 2-center proxy
- `federate_nih_by_patient(...)` — patient-disjoint partition into N centers (image-level splits leak features)
