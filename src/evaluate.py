"""
evaluate.py — Metrics, confusion matrix, study-level aggregation, and ROC helpers.

Checkpoint 2 delivers full classification metric computation (including per-class
recall via sensitivity and specificity, macro-F1, Cohen's kappa, ROC-AUC, and
study-level aggregation).  Test-set ROC plotting and failure-case inspection
will be extended in Checkpoint 4 after neural models are trained.

Authors: Team — Evaluation & Deployment Engineer work package.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
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
    """Compute the full set of binary classification triage metrics.

    Includes both per-class recalls:
      - ``recall`` (Abnormal / class-1 sensitivity)
      - ``normal_recall`` (Normal / class-0 specificity)

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0 = normal, 1 = abnormal).
    y_pred : array-like
        Predicted binary labels (0 or 1).
    y_prob : array-like, optional
        Predicted probability for the positive (abnormal) class.

    Returns
    -------
    dict
        Dictionary mapping metric names to scalar float values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "normal_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))

    return metrics


def evaluate_study_level(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Aggregate image-level probabilities to study-level predictions via mean pooling."""
    study_frame = pd.DataFrame(
        {
            "study_id": df["study_id"].values,
            "label": df["label"].values,
            "prob": np.asarray(y_prob),
        }
    )
    agg = study_frame.groupby("study_id", sort=False).agg(
        label=("label", "first"),
        prob=("prob", "mean"),
    )
    y_true_study = agg["label"].values
    y_prob_study = agg["prob"].values
    y_pred_study = (y_prob_study >= threshold).astype(int)
    return compute_metrics(y_true_study, y_pred_study, y_prob=y_prob_study)


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Print and return a scikit-learn classification report."""
    target_names = ["Normal (0)", "Abnormal (1)"]
    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    print(report)
    return report


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return the 2x2 confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
