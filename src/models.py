"""
models.py — Baseline models and neural network architectures for MURA.

This module provides:
  - Trivial baseline  : majority-class predictor (always predicts "normal").
  - Learned baseline   : logistic regression on flattened pixel features.
  - (Future) Model A   : custom CNN from scratch.
  - (Future) Model B   : ResNet-50 with frozen base + new classifier head.
  - (Future) Model C   : Fine-tuned ResNet-50 (last block unfrozen).

Checkpoint 2 delivers the two baselines.  Models A–C are built in Checkpoint 3.

Authors: Team — Baseline & Analysis Engineer + Model Architect work packages.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)


# ════════════════════════════════════════════════════════════
#  BASELINES  (Checkpoint 2)
# ════════════════════════════════════════════════════════════

class MajorityBaseline:
    """Trivial baseline: always predicts the majority class (normal = 0).

    This provides the absolute floor that any learned model must beat.
    On the MURA training split the majority class is label 0 (normal),
    comprising ~61 % of studies.
    """

    def __init__(self):
        self.clf = DummyClassifier(strategy="most_frequent")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityBaseline":
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        return _classification_report(y, preds, name="MajorityBaseline")


class LogisticBaseline:
    """Simple learned baseline: logistic regression on flattened pixels.

    Images are resized to 64×64 grayscale (= 4 096 features) to keep
    memory and training time manageable.  This deliberately simple model
    provides the second reference point required by the specification.

    Justification for hyper-parameters:
      - max_iter=1000 : convergence on high-dimensional pixel data
      - solver='saga' : efficient for large-n, supports L2 by default
      - class_weight='balanced' : compensates for class imbalance
    """

    def __init__(self, max_iter: int = 1000):
        self.clf = LogisticRegression(
            max_iter=max_iter,
            solver="saga",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        """Fit on flattened image arrays.  X shape: (n_samples, H*W) or (n, H, W, C)."""
        X_flat = self._flatten(X)
        self.clf.fit(X_flat, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(self._flatten(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self._flatten(X))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        return _classification_report(y, preds, name="LogisticBaseline")

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        if X.ndim > 2:
            return X.reshape(X.shape[0], -1)
        return X


# ════════════════════════════════════════════════════════════
#  BASELINE HELPERS
# ════════════════════════════════════════════════════════════

def sample_from_generator(gen, n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Draw a fixed number of images + labels from a Keras DataGenerator.

    Used to feed sklearn baselines which expect numpy arrays.  Images are
    converted to grayscale and down-sampled to 64×64 to keep memory low.

    Parameters
    ----------
    gen : keras ImageDataGenerator flow
    n_samples : int
        Maximum number of samples to draw.

    Returns
    -------
    X : np.ndarray  shape (n, 4096)   flattened grayscale 64×64
    y : np.ndarray  shape (n,)
    """
    from PIL import Image

    X_list, y_list = [], []
    seen = 0
    gen.reset()

    for X_batch, y_batch in gen:
        for i in range(X_batch.shape[0]):
            if seen >= n_samples:
                break
            img = X_batch[i]
            # Convert to grayscale and resize to 64×64
            pil = Image.fromarray((img * 255).astype(np.uint8))
            pil = pil.convert("L").resize((64, 64))
            X_list.append(np.array(pil).flatten() / 255.0)
            y_list.append(int(y_batch[i]))
            seen += 1
        if seen >= n_samples:
            break

    return np.array(X_list), np.array(y_list)


def _classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "model",
) -> Dict[str, float]:
    """Compute standard classification metrics and return as a dict."""
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }
