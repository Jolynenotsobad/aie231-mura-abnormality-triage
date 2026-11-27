"""
train.py — Training entry point and experiment logger for MURA triage.

Checkpoint 2 implements full-split baseline evaluation (``--baseline``):
  - ``MajorityBaseline`` (trivial floor)
  - ``LogisticBaseline`` unweighted vs ``class_weight='balanced'``
    (demonstrating the effect of imbalance handling on per-class recall)

All metrics are computed strictly on the patient-grouped validation split
(never touching the held-out test split) and logged to
``experiments/experiment_log.csv``.

Usage:
    python -m src.train --config configs/exp_baseline.yaml --baseline
    python -m src.train --config configs/exp_baseline.yaml --baseline --data-root /path/to/MURA-v1.1

Authors: Team — Tech Lead work package.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Optional

from src.data import load_config, load_flattened_images, prepare_data, set_global_seed
from src.evaluate import evaluate_study_level, print_classification_report
from src.models import LogisticBaseline, MajorityBaseline

LOG_PATH = "experiments/experiment_log.csv"
LOG_FIELDS = [
    "run_id",
    "date",
    "author",
    "model",
    "hyperparameters",
    "seed",
    "train_time_s",
    "parameters",
    "train_accuracy",
    "train_kappa",
    "val_accuracy",
    "val_normal_recall",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_macro_f1",
    "val_kappa",
    "val_auc",
    "notes",
]


def run_baselines(cfg: dict, data_root: Optional[str] = None) -> list[dict]:
    """Run all Checkpoint 2 baselines on the full patient-grouped validation split."""
    data = prepare_data(cfg=cfg, data_root=data_root, build_tf=False)
    train_df = data["train_df"]
    val_df = data["val_df"]
    test_df = data["test_df"]
    seed = int(cfg.get("seed", 42))

    print("Data preparation summary:")
    print(f"  Data root        : {data['data_root']}")
    print(f"  Cleaning report  : {data['cleaning_report']}")
    print(
        f"  Split images     : train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,} (held out)"
    )
    print(f"  Train class wts  : { {k: round(v, 4) for k, v in data['class_weights'].items()} }")

    print("\nLoading 64x64 grayscale pixels for full train & validation splits ...")
    t_load = time.time()
    X_train, y_train = load_flattened_images(train_df, size=(64, 64))
    X_val, y_val = load_flattened_images(val_df, size=(64, 64))
    print(f"  Loaded X_train {X_train.shape}, X_val {X_val.shape} in {time.time() - t_load:.2f}s")

    runs = [
        (
            "baseline_majority",
            MajorityBaseline(),
            {"strategy": "most_frequent"},
            0,
            "Trivial majority-class floor (Checkpoint 2)",
        ),
        (
            "baseline_logreg_unweighted",
            LogisticBaseline(C=0.01, max_iter=1000, class_weight=None, seed=seed),
            {"input": "64x64_gray", "scaler": "StandardScaler", "C": 0.01, "class_weight": None},
            X_train.shape[1] + 1,
            "Learned linear baseline without imbalance handling (Checkpoint 2)",
        ),
        (
            "baseline_logreg_balanced",
            LogisticBaseline(C=0.01, max_iter=1000, class_weight="balanced", seed=seed),
            {"input": "64x64_gray", "scaler": "StandardScaler", "C": 0.01, "class_weight": "balanced"},
            X_train.shape[1] + 1,
            "Learned linear baseline with balanced class weights (Checkpoint 2)",
        ),
    ]

    logged_rows: list[dict] = []
    for run_id, model, hparams, n_params, notes in runs:
        print(f"\n══ {model.name} ({run_id}) ══")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - t0, 2)

        train_metrics = model.evaluate(X_train, y_train)
        val_metrics = model.evaluate(X_val, y_val)
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = model.predict(X_val)
        study_metrics = evaluate_study_level(val_df, val_probs)

        _print_metrics(val_metrics, train_time=train_time, study_metrics=study_metrics)
        print_classification_report(y_val, val_preds)

        row = {
            "run_id": run_id,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "author": "team",
            "model": model.name,
            "hyperparameters": json.dumps(hparams),
            "seed": seed,
            "train_time_s": train_time,
            "parameters": n_params,
            "train_accuracy": round(train_metrics["accuracy"], 4),
            "train_kappa": round(train_metrics["kappa"], 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_normal_recall": round(val_metrics["normal_recall"], 4),
            "val_precision": round(val_metrics["precision"], 4),
            "val_recall": round(val_metrics["recall"], 4),
            "val_f1": round(val_metrics["f1"], 4),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
            "val_kappa": round(val_metrics["kappa"], 4),
            "val_auc": round(val_metrics.get("auc", 0.5), 4),
            "notes": notes,
        }
        logged_rows.append(row)

    _upsert_experiment_log(logged_rows)
    return logged_rows


def _print_metrics(
    m: dict,
    train_time: float,
    study_metrics: Optional[dict] = None,
) -> None:
    """Pretty-print image-level and optional study-level triage metrics."""
    print(f"  Accuracy        : {m['accuracy']:.4f}")
    print(f"  Normal Recall   : {m['normal_recall']:.4f}")
    print(f"  Abnormal Recall : {m['recall']:.4f}")
    print(f"  Precision       : {m['precision']:.4f}")
    print(f"  F1 (Abnormal)   : {m['f1']:.4f}")
    print(f"  Macro-F1        : {m['macro_f1']:.4f}")
    print(f"  Cohen kappa (k) : {m['kappa']:.4f}")
    print(f"  ROC-AUC         : {m.get('auc', 0.5):.4f}")
    print(f"  Train time      : {train_time:.2f}s")
    if study_metrics is not None:
        print(
            f"  [Study-level]   : Acc={study_metrics['accuracy']:.4f}, "
            f"Rec={study_metrics['recall']:.4f}, "
            f"MacroF1={study_metrics['macro_f1']:.4f}, "
            f"Kappa={study_metrics['kappa']:.4f}, "
            f"AUC={study_metrics.get('auc', 0.5):.4f}"
        )


def _upsert_experiment_log(rows: list[dict], log_path: str = LOG_PATH) -> None:
    """Write or update experiment rows in ``experiments/experiment_log.csv``."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    existing: list[dict] = []
    if os.path.isfile(log_path):
        with open(log_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames == LOG_FIELDS:
                incoming_ids = {r["run_id"] for r in rows}
                existing = [r for r in reader if r.get("run_id") not in incoming_ids]

    all_rows = existing + rows
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[OK] Logged {len(rows)} baseline runs to {log_path}")


def main() -> None:
    """CLI entry point for baseline and neural network training."""
    parser = argparse.ArgumentParser(description="Train MURA triage models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline models only (Checkpoint 2)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional override path to MURA-v1.1 directory (or set MURA_DATA_DIR)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("seed", 42)))

    if args.baseline:
        run_baselines(cfg, data_root=args.data_root)
    else:
        print("Neural network training (Models A-C) will be implemented in Checkpoint 3.")
        print("Use --baseline to run Checkpoint 2 baseline evaluation.")


if __name__ == "__main__":
    main()
