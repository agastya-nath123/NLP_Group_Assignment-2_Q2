"""Evaluation metrics. Labels are 0..4 internally (= 1..5 stars), so |pred - true| is
directly the error in stars."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

LABELS = [0, 1, 2, 3, 4]
STAR_NAMES = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    diff = np.abs(y_pred - y_true)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    row = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0)
    errors = diff > 0
    n_err = int(errors.sum())
    pair = lambda a, b: int(cm[a, b] + cm[b, a])  # noqa: E731
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0).tolist(),
        "mae": float(diff.mean()),
        # ordinal error profile
        "error_distance_dist": [float((diff == d).mean()) for d in range(5)],
        "off_by_one_rate": float((diff == 1).mean()),
        "off_by_two_plus_rate": float((diff >= 2).mean()),
        "adjacent_share_of_errors": float((diff[errors] == 1).mean()) if n_err else 0.0,
        "distant_share_of_errors": float((diff[errors] >= 2).mean()) if n_err else 0.0,
        "confusions_4v5": pair(3, 4),
        "confusions_1v2": pair(0, 1),
        "confusions_1v5": pair(0, 4),
        "rate_4v5_within_4_and_5": pair(3, 4) / max(int(cm[3].sum() + cm[4].sum()), 1),
        "rate_1v5_within_1_and_5": pair(0, 4) / max(int(cm[0].sum() + cm[4].sum()), 1),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


def aggregate(metric_dicts: list[dict]) -> dict:
    """Mean and sample standard deviation (ddof=1) over seeds for every numeric entry."""
    out = {}
    for key, val in metric_dicts[0].items():
        if key in ("n", "confusion_matrix", "confusion_matrix_normalized"):
            continue
        arr = np.asarray([m[key] for m in metric_dicts], dtype=float)
        std = arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros_like(arr[0])
        out[key] = {"mean": arr.mean(axis=0).tolist(), "std": np.asarray(std).tolist()}
    pooled_cm = np.sum([np.asarray(m["confusion_matrix"]) for m in metric_dicts], axis=0)
    out["pooled_confusion_matrix"] = pooled_cm.tolist()
    out["pooled_confusion_matrix_normalized"] = (pooled_cm / pooled_cm.sum(1, keepdims=True)).tolist()
    out["n_seeds"] = len(metric_dicts)
    return out


def fmt(stat: dict, pct: bool = True, digits: int = 2) -> str:
    """{'mean': .., 'std': ..} -> '58.31 ± 0.22'."""
    k = 100.0 if pct else 1.0
    return f"{stat['mean'] * k:.{digits}f} ± {stat['std'] * k:.{digits}f}"
