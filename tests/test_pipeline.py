"""
test_pipeline.py — Unit tests for Checkpoint 1 & 2 MURA pipeline components.

Verifies:
  - MURA metadata extraction and study_id uniqueness across body parts
  - Data cleaning (path deduplication, missing file removal, MD5 content deduplication)
  - Patient-level stratified splitting with strict zero-leakage assertions
  - Training-only balanced class weight calculation
  - MajorityBaseline and LogisticBaseline fit / predict / evaluate
  - Evaluation metrics (including per-class recall, kappa, AUC, study aggregation)
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

from src.data import (
    _extract_meta,
    clean_dataframe,
    compute_weights,
    load_config,
    load_flattened_images,
    patient_stratified_split,
)
from src.evaluate import compute_metrics, evaluate_study_level
from src.models import LogisticBaseline, MajorityBaseline


def test_extract_meta_unique_study_across_body_parts():
    """Patients scanned on multiple body parts must yield distinct study IDs."""
    m1 = _extract_meta("MURA-v1.1/train/XR_WRIST/patient00010/study1_positive/image1.png")
    m2 = _extract_meta("MURA-v1.1/train/XR_ELBOW/patient00010/study1_negative/image1.png")

    assert m1["patient_id"] == m2["patient_id"] == "patient00010"
    assert m1["study_id"] != m2["study_id"]
    assert m1["label"] == 1
    assert m2["label"] == 0


def test_clean_dataframe_deduplicates_paths_and_content():
    """Cleaning drops path duplicates, missing files, and byte-identical images."""
    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "a.png")
        p2 = os.path.join(tmp, "b.png")
        p3 = os.path.join(tmp, "c.png")
        missing = os.path.join(tmp, "missing.png")

        # Create identical PNGs for p1 and p2, distinct for p3
        img_a = Image.new("L", (16, 16), color=100)
        img_a.save(p1)
        img_a.save(p2)
        Image.new("RGB", (16, 16), color=(200, 10, 10)).save(p3)

        df = pd.DataFrame(
            {
                "image_path": ["p/a.png", "p/a.png", "p/b.png", "p/c.png", "p/missing.png"],
                "abs_path": [p1, p1, p2, p3, missing],
                "label": [0, 0, 0, 1, 1],
            }
        )

        cleaned, report = clean_dataframe(df, check_files=True, deduplicate_content=True)
        assert report["duplicates_removed"] == 1
        assert report["missing_files_removed"] == 1
        assert report["content_duplicates_removed"] == 1
        assert len(cleaned) == 2


def test_patient_stratified_split_zero_leakage():
    """Patients must never appear in more than one split; proportions preserved."""
    rows = []
    for i in range(100):
        pid = f"patient{i:03d}"
        bp = "XR_WRIST" if i % 2 == 0 else "XR_SHOULDER"
        label = 1 if i < 40 else 0
        for v in range(1, 4):
            rows.append(
                {
                    "image_path": f"MURA-v1.1/train/{bp}/{pid}/study1/image{v}.png",
                    "body_part": bp,
                    "patient_id": pid,
                    "study_id": f"{bp}/{pid}/study1",
                    "label": label,
                }
            )
    df = pd.DataFrame(rows)
    tr, va, te = patient_stratified_split(df, val_frac=0.15, test_frac=0.15, seed=42)

    p_tr = set(tr["patient_id"])
    p_va = set(va["patient_id"])
    p_te = set(te["patient_id"])

    assert p_tr.isdisjoint(p_va)
    assert p_tr.isdisjoint(p_te)
    assert p_va.isdisjoint(p_te)
    assert len(tr) + len(va) + len(te) == len(df)
    assert abs(tr["label"].mean() - 0.40) < 0.05
    assert abs(va["label"].mean() - 0.40) < 0.10


def test_compute_weights_balances_imbalance():
    """Minority class receives strictly larger weight than majority class."""
    train_df = pd.DataFrame({"label": [0] * 60 + [1] * 40})
    weights = compute_weights(train_df)
    assert weights[1] > weights[0]
    assert np.isclose(weights[0] * 60 + weights[1] * 40, 100.0)


def test_baselines_and_metrics_end_to_end():
    """Baselines train cleanly on mixed L/RGB images and return full metrics."""
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(42)
        records = []
        for idx in range(40):
            label = 1 if idx < 16 else 0
            base_val = 200 if label == 1 else 50
            arr = np.clip(rng.normal(base_val, 15, size=(64, 64)), 0, 255).astype(np.uint8)
            mode = "L" if idx % 2 == 0 else "RGB"
            img = Image.fromarray(arr, mode="L").convert(mode)
            path = os.path.join(tmp, f"img_{idx}.png")
            img.save(path)
            records.append(
                {
                    "abs_path": path,
                    "study_id": f"XR_WRIST/p{idx // 2}/study1",
                    "label": label,
                }
            )
        df = pd.DataFrame(records)
        X, y = load_flattened_images(df, size=(32, 32), max_workers=2)
        assert X.shape == (40, 1024)
        assert X.min() >= 0.0 and X.max() <= 1.0

        maj = MajorityBaseline().fit(X, y)
        m_maj = maj.evaluate(X, y)
        assert m_maj["kappa"] == 0.0
        assert m_maj["recall"] == 0.0
        assert m_maj["normal_recall"] == 1.0

        lr = LogisticBaseline(C=0.1, max_iter=200, class_weight="balanced", seed=42).fit(X, y)
        m_lr = lr.evaluate(X, y)
        assert m_lr["kappa"] > 0.5
        assert m_lr["auc"] > 0.8

        study_metrics = evaluate_study_level(df, lr.predict_proba(X)[:, 1])
        assert "kappa" in study_metrics and "auc" in study_metrics


def test_configs_valid_yaml():
    """All config files load without error and expose expected top-level keys."""
    for name in (
        "configs/base.yaml",
        "configs/exp_baseline.yaml",
        "configs/exp_custom_cnn.yaml",
        "configs/exp_resnet50_frozen.yaml",
        "configs/exp_resnet50_finetune.yaml",
    ):
        cfg = load_config(name)
        assert "data" in cfg and "split" in cfg and "seed" in cfg
