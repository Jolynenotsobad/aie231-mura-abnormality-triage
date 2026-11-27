"""
train.py — Training loop, callbacks, and logging for MURA models.

Checkpoint 2 provides the baseline training/evaluation entry point.
Full neural-network training (Checkpoint 3+) will extend this module
with Keras model.fit(), callbacks (EarlyStopping, ModelCheckpoint),
and TensorBoard integration.

Usage (baselines — Checkpoint 2):
    python -m src.train --config configs/base.yaml --baseline

Authors: Team — Tech Lead work package.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np

from src.data import load_config, prepare_data, set_global_seed
from src.models import (
    LogisticBaseline,
    MajorityBaseline,
    sample_from_generator,
)


def run_baselines(cfg: dict) -> list[dict]:
    """Run both baselines and return their evaluation dicts.

    Steps:
      1. Prepare data with the full pipeline (leak-free splits).
      2. Sample images from train and val generators into numpy arrays.
      3. Fit and evaluate MajorityBaseline and LogisticBaseline.
      4. Log results to experiments/experiment_log.csv.
    """
    data = prepare_data(cfg=cfg)
    train_gen = data["train_gen"]
    val_gen = data["val_gen"]

    print("Sampling images for sklearn baselines …")
    X_train, y_train = sample_from_generator(train_gen, n_samples=3000)
    X_val, y_val = sample_from_generator(val_gen, n_samples=1000)

    results = []

    # ── 1. Majority-class baseline ───────────────────────────
    print("\n══ Majority-class baseline ══")
    t0 = time.time()
    majority = MajorityBaseline().fit(X_train, y_train)
    metrics_maj = majority.evaluate(X_val, y_val)
    metrics_maj["train_time_s"] = round(time.time() - t0, 2)
    _print_metrics(metrics_maj)
    results.append(metrics_maj)

    # ── 2. Logistic regression baseline ──────────────────────
    print("\n══ Logistic Regression baseline ══")
    t0 = time.time()
    logistic = LogisticBaseline().fit(X_train, y_train)
    metrics_lr = logistic.evaluate(X_val, y_val)
    metrics_lr["train_time_s"] = round(time.time() - t0, 2)
    _print_metrics(metrics_lr)
    results.append(metrics_lr)

    # ── Log to CSV ───────────────────────────────────────────
    _log_experiments(results, cfg)

    return results


def _print_metrics(m: dict) -> None:
    """Pretty-print a metrics dict."""
    print(f"  Model     : {m['model']}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  Macro-F1  : {m['macro_f1']:.4f}")
    print(f"  Cohen κ   : {m['kappa']:.4f}")
    if "train_time_s" in m:
        print(f"  Train time: {m['train_time_s']}s")


def _log_experiments(results: list[dict], cfg: dict) -> None:
    """Append experiment results to experiments/experiment_log.csv."""
    log_path = "experiments/experiment_log.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fieldnames = [
        "run_id", "date", "author", "model", "seed",
        "accuracy", "precision", "recall", "f1", "macro_f1", "kappa",
        "train_time_s", "notes",
    ]

    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i, m in enumerate(results, start=1):
            writer.writerow({
                "run_id": f"baseline_{i}",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "author": "team",
                "model": m["model"],
                "seed": cfg.get("seed", 42),
                "accuracy": round(m["accuracy"], 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "macro_f1": round(m["macro_f1"], 4),
                "kappa": round(m["kappa"], 4),
                "train_time_s": m.get("train_time_s", ""),
                "notes": "Checkpoint 2 baseline",
            })

    print(f"\n✓ Results appended to {log_path}")


# ────────────────────────────────────────────────────────────
#  CLI entry point
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train MURA models")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Run baseline models only (Checkpoint 2)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.get("seed", 42))

    if args.baseline:
        run_baselines(cfg)
    else:
        print("Neural network training will be implemented in Checkpoint 3.")
        print("Use --baseline to run baseline evaluation.")


if __name__ == "__main__":
    main()
