"""
models.py — Baseline models and neural network interfaces for MURA triage.

Checkpoint 2 delivers the two mandatory benchmark baselines before any deep
neural network:
  1. ``MajorityBaseline`` — trivial majority-class predictor (predicts Normal = 0).
  2. ``LogisticBaseline`` — leak-free ``StandardScaler`` + L2-regularized
     logistic regression trained on flattened 64x64 grayscale pixels, evaluated
     both unweighted and with ``class_weight='balanced'`` to quantify the
     effect of imbalance handling on per-class recall.

Neural network architectures (Model A: Custom CNN from scratch; Model B:
Frozen ResNet-50; Model C: Fine-tuned ResNet-50) will be added in Checkpoint 3.

Authors: Team — Baseline & Analysis Engineer + Model Architect work packages.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import compute_metrics


# ════════════════════════════════════════════════════════════
#  BASELINES  (Checkpoint 2)
# ════════════════════════════════════════════════════════════


class MajorityBaseline:
    """Trivial baseline: always predicts the training majority class (Normal = 0).

    Establishes the chance-level performance floor (kappa = 0, abnormal recall = 0)
    and illustrates why raw accuracy (~59.9% on validation) is misleading on
    imbalanced musculoskeletal screening data.
    """

    def __init__(self) -> None:
        self.clf = DummyClassifier(strategy="most_frequent")
        self.name = "MajorityBaseline"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityBaseline":
        """Fit the trivial baseline on training labels."""
        self.clf.fit(self._flatten(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the majority class for every input sample."""
        return self.clf.predict(self._flatten(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates."""
        return self.clf.predict_proba(self._flatten(X))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute triage metrics on an evaluation split."""
        X_flat = self._flatten(X)
        preds = self.clf.predict(X_flat)
        probs = self.clf.predict_proba(X_flat)[:, 1]
        metrics = compute_metrics(y, preds, y_prob=probs)
        metrics["model"] = self.name
        return metrics

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1) if X.ndim > 2 else X


class LogisticBaseline:
    """Simple learned baseline: StandardScaler + Logistic Regression on 64x64 pixels.

    Uses a scikit-learn ``Pipeline`` so pixel standardization is fitted strictly
    on the training split and applied unchanged to validation/test (no leakage).

    Parameters
    ----------
    C : float
        Inverse L2 regularization strength (default 0.01 for 4,096 correlated pixels).
    max_iter : int
        Maximum L-BFGS iterations.
    class_weight : str or dict or None
        Passed to ``LogisticRegression``; set to ``"balanced"`` to handle
        class imbalance or ``None`` for the unweighted comparison.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        C: float = 0.01,
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
        seed: int = 42,
    ) -> None:
        self.class_weight = class_weight
        suffix = "balanced" if class_weight == "balanced" else "unweighted"
        self.name = f"LogisticBaseline_{suffix}"
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        solver="lbfgs",
                        class_weight=class_weight,
                        random_state=seed,
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        """Fit scaler + logistic regression strictly on training features."""
        self.pipeline.fit(self._flatten(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels (0 = normal, 1 = abnormal)."""
        return self.pipeline.predict(self._flatten(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities ``[P(normal), P(abnormal)]``."""
        return self.pipeline.predict_proba(self._flatten(X))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute triage metrics on an evaluation split."""
        X_flat = self._flatten(X)
        preds = self.pipeline.predict(X_flat)
        probs = self.pipeline.predict_proba(X_flat)[:, 1]
        metrics = compute_metrics(y, preds, y_prob=probs)
        metrics["model"] = self.name
        return metrics

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1) if X.ndim > 2 else X
