"""
evaluate.py — Metrics, confusion matrix, ROC, and error analysis for MURA.

Checkpoint 2 delivers the metric-computation helpers used by baselines.
Full evaluation on the held-out test set (confusion matrix heatmap,
ROC/AUC, per-class metrics, and error-analysis gallery) will be
completed in Checkpoint 4 after the final model is chosen.

Authors: Team — Evaluation & Deployment Engineer work package.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute the full set of classification metrics.

    Parameters
    ----------
    y_true : array-like   ground-truth labels (0 or 1)
    y_pred : array-like   hard predictions (0 or 1)
    y_prob : array-like, optional   predicted probabilities for class 1

    Returns
    -------
    dict   metric_name → value
    """
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }

    if y_prob is not None:
        metrics["auc"] = roc_auc_score(y_true, y_prob)

    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Print and return a sklearn classification report."""
    target_names = ["Normal (0)", "Abnormal (1)"]
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    print(report)
    return report


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return the 2×2 confusion matrix."""
    return confusion_matrix(y_true, y_pred)
