"""Evaluation metrics beyond raw accuracy.

Clinical FL must report calibration, fairness, and discriminative performance
under class imbalance — not just argmax accuracy.
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """ECE between predicted confidences and observed accuracies.

    For binary tasks, ``probs`` is the predicted positive-class probability
    and ``labels`` is the binary outcome.

    For multi-class tasks, pass ``probs`` = top-1 confidence per sample and
    ``labels`` = 1 if argmax was correct else 0.
    """
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (probs > lo) & (probs <= hi)
        if not in_bin.any():
            continue
        avg_conf = probs[in_bin].mean()
        avg_acc = labels[in_bin].mean()
        ece += in_bin.mean() * abs(avg_conf - avg_acc)
    return float(ece)


def multiclass_auroc(probs: np.ndarray, labels: np.ndarray) -> float | None:
    """Macro-averaged one-vs-rest AUROC. Returns None if sklearn missing or ill-defined."""
    try:
        from sklearn.metrics import roc_auc_score

        probs = np.asarray(probs)
        labels = np.asarray(labels)
        if probs.ndim != 2 or probs.shape[1] < 2:
            return None
        if len(np.unique(labels)) < 2:
            return None
        return float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )
    except (ImportError, ValueError):
        return None


def multiclass_auprc(probs: np.ndarray, labels: np.ndarray) -> float | None:
    """Macro-averaged one-vs-rest AUPRC."""
    try:
        from sklearn.metrics import average_precision_score
        from sklearn.preprocessing import label_binarize

        probs = np.asarray(probs)
        labels = np.asarray(labels)
        if probs.ndim != 2 or probs.shape[1] < 2:
            return None
        if len(np.unique(labels)) < 2:
            return None
        bin_labels = label_binarize(labels, classes=list(range(probs.shape[1])))
        return float(average_precision_score(bin_labels, probs, average="macro"))
    except (ImportError, ValueError):
        return None


def worst_center_gap(per_center_metric: dict[str, float]) -> float:
    """Spread between best- and worst-performing client (fairness proxy)."""
    values = list(per_center_metric.values())
    return float(max(values) - min(values))


def worst_group_metric(per_group_metric: dict[str, float]) -> float:
    """Worst-case-group value (use for under-represented sub-population eval)."""
    return float(min(per_group_metric.values()))


def communication_cost(
    rounds: int, params_per_round: int, bytes_per_param: int = 4
) -> int:
    """Total upstream communication in bytes per client."""
    return int(rounds * params_per_round * bytes_per_param)
