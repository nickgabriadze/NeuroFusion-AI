"""Metrics and participant-level bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def brier_multiclass(y, proba) -> float:
    onehot = np.eye(proba.shape[1])[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def ece(y, proba, n_bins: int = 10) -> float:
    """Expected calibration error on the top-class confidence."""
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)


def macro_f1(y, proba) -> float:
    return float(f1_score(y, proba.argmax(axis=1), average="macro"))


def auroc_ovr(y, proba) -> float:
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))


def evaluate(y, proba) -> dict:
    return {
        "macro_f1": macro_f1(y, proba),
        "auroc_ovr": auroc_ovr(y, proba),
        "brier": brier_multiclass(y, proba),
        "ece": ece(y, proba),
    }


def bootstrap_ci(y, proba, groups, metric=macro_f1, n_boot=1000, seed=42, alpha=0.05):
    """Resample PARTICIPANTS (not rows) with replacement; return (lo, hi)."""
    rng = np.random.default_rng(seed)
    y, proba, groups = np.asarray(y), np.asarray(proba), np.asarray(groups)
    ids = np.unique(groups)
    idx_by_id = {g: np.where(groups == g)[0] for g in ids}
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([idx_by_id[g] for g in sample])
        try:
            stats.append(metric(y[idx], proba[idx]))
        except ValueError:      # a resample missing a class
            continue
    return tuple(np.quantile(stats, [alpha / 2, 1 - alpha / 2]))
