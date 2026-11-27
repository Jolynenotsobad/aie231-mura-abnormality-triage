"""
data.py — Data loading, cleaning, splitting, and preprocessing for the MURA dataset.

This module implements a strictly leak-free data engineering pipeline:
  1. Parses the MURA v1.1 CSV manifests to construct an image-level dataframe
     with body part, patient ID, globally unique study ID, and binary label.
  2. Cleans the manifest by removing path duplicates, verifying file presence,
     and deduplicating byte-identical radiograph files (via file-size + MD5 hash)
     so identical scans cannot leak across patient splits.
  3. Splits official training data into Train / Validation / Test (70 / 15 / 15)
     at the **patient** level, stratified by both primary body part and label.
     The official MURA ``valid`` set is kept separate and untouched as an
     external multi-radiologist benchmark.
  4. Computes class weights strictly from the training split to mitigate class
     imbalance (59.6% normal vs 40.4% abnormal at image level).
  5. Provides fast grayscale array loading for sklearn baselines and modern
     ``tf.data`` + Keras preprocessing layers (TF 2.15 through Keras 3) with
     data augmentation restricted strictly to the training split.

Authors: Team — Data Engineer work package.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ───────────────────── configuration & seeds ────────────────


def load_config(path: str = "configs/base.yaml") -> dict:
    """Load a YAML configuration file and return it as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_root(cfg: dict, cli_root: Optional[str] = None) -> str:
    """Resolve the MURA dataset root without hard-coding user paths.

    Resolution priority:
      1. Explicit CLI argument (``cli_root``)
      2. Environment variable ``MURA_DATA_DIR``
      3. ``cfg["data"]["root_dir"]`` from the YAML config
    """
    if cli_root:
        return cli_root
    env_root = os.environ.get("MURA_DATA_DIR")
    if env_root:
        return env_root
    return cfg["data"]["root_dir"]


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and TensorFlow (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if importlib.util.find_spec("tensorflow") is not None:
        import tensorflow as tf

        tf.random.set_seed(seed)


# ───────────────────── CSV parsing ──────────────────────────


def _extract_meta(path: str) -> dict:
    """Extract body-part, patient id, study id, and label from a MURA path.

    Example path:
        ``MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/image1.png``

    Note: ``study_id`` includes ``body_part`` because ~1,467 MURA patients
    appear under multiple body parts with overlapping ``study1_*`` folders.
    """
    normalized = path.replace("\\", "/")
    parts = pathlib.PurePosixPath(normalized).parts
    if len(parts) < 6:
        raise ValueError(f"Unexpected MURA path format: {path!r}")

    body_part = parts[2]
    patient_id = parts[3]
    study_dir = parts[4]
    study_id = f"{body_part}/{patient_id}/{study_dir}"
    study_path = "/".join(parts[:5]) + "/"
    label = 1 if "positive" in study_dir else 0

    return {
        "image_path": normalized,
        "body_part": body_part,
        "patient_id": patient_id,
        "study_id": study_id,
        "study_path": study_path,
        "label": label,
    }


def build_dataframe(
    root_dir: str,
    image_paths_csv: str,
    labeled_studies_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Build an image-level DataFrame from a MURA image-paths CSV manifest.

    Parameters
    ----------
    root_dir : str
        Filesystem path to the extracted ``MURA-v1.1`` directory.
    image_paths_csv : str
        Filename of the image-paths CSV (e.g. ``"train_image_paths.csv"``).
    labeled_studies_csv : str, optional
        Optional study-level labels CSV to cross-verify directory labels.

    Returns
    -------
    pd.DataFrame
        Columns: ``image_path``, ``body_part``, ``patient_id``, ``study_id``,
        ``study_path``, ``label``, ``abs_path``.
    """
    csv_path = os.path.join(root_dir, image_paths_csv)
    raw = pd.read_csv(csv_path, header=None, names=["image_path"])

    records = raw["image_path"].apply(_extract_meta).tolist()
    df = pd.DataFrame(records)

    base = os.path.dirname(os.path.abspath(root_dir))
    df["abs_path"] = df["image_path"].apply(lambda p: os.path.join(base, *p.split("/")))

    if labeled_studies_csv is not None:
        labels_path = os.path.join(root_dir, labeled_studies_csv)
        if os.path.isfile(labels_path):
            studies = pd.read_csv(
                labels_path, header=None, names=["study_path", "csv_label"]
            )
            df = df.merge(studies, on="study_path", how="left")
            if df["csv_label"].notna().all():
                df["label"] = df["csv_label"].astype(int)
            df = df.drop(columns=["csv_label"])

    return df


# ───────────────────── cleaning ─────────────────────────────


def _md5_file(path: str) -> str:
    """Compute MD5 hex digest for a file on disk."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_dataframe(
    df: pd.DataFrame,
    check_files: bool = True,
    deduplicate_content: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Clean the image dataframe and return cleaning diagnostics.

    Steps:
      1. Drop exact duplicate ``image_path`` rows.
      2. Optionally verify all referenced files exist on disk and drop missing entries.
      3. Optionally detect byte-identical image files (fast file-size prefilter
         followed by MD5 comparison on collisions) so duplicate radiographs
         across patients cannot leak across splits.
      4. Record missing-value counts per column.
    """
    report: dict = {"initial_rows": len(df)}

    # 1. Exact path duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    report["duplicates_removed"] = n_before - len(df)

    # 2. Missing values audit
    report["missing_values"] = {k: int(v) for k, v in df.isnull().sum().items()}

    # 3. Full file-existence check
    missing_files = 0
    if check_files and "abs_path" in df.columns and len(df) > 0:
        exists_mask = df["abs_path"].apply(os.path.isfile)
        missing_files = int((~exists_mask).sum())
        if missing_files > 0:
            df = df[exists_mask].reset_index(drop=True)
    report["missing_files_removed"] = missing_files

    # 4. Exact byte-content deduplication via st_size prefilter + MD5
    content_dups = 0
    if deduplicate_content and check_files and "abs_path" in df.columns and len(df) > 1:
        sizes = df["abs_path"].apply(os.path.getsize)
        size_collisions = sizes.duplicated(keep=False)
        if size_collisions.any():
            candidate_idx = df.index[size_collisions].tolist()
            with ThreadPoolExecutor(max_workers=8) as ex:
                hashes = list(ex.map(_md5_file, df.loc[candidate_idx, "abs_path"]))
            hash_series = pd.Series(hashes, index=candidate_idx)
            dup_hash_idx = hash_series[hash_series.duplicated(keep="first")].index
            content_dups = len(dup_hash_idx)
            if content_dups > 0:
                df = df.drop(index=dup_hash_idx).reset_index(drop=True)
    report["content_duplicates_removed"] = content_dups
    report["final_rows"] = len(df)

    return df, report


# ───────────────────── splitting ────────────────────────────


def patient_stratified_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into Train / Validation / Test strictly at the patient level.

    Every patient appears in exactly one split (zero patient leakage). Splits are
    stratified at the patient level by ``(primary_body_part, modal_label)`` when
    all strata contain at least 2 patients, falling back cleanly to ``modal_label``
    on tiny subsets.
    """
    if not (0.0 < val_frac < 1.0 and 0.0 < test_frac < 1.0 and val_frac + test_frac < 1.0):
        raise ValueError(f"Invalid split fractions: val={val_frac}, test={test_frac}")

    agg_spec = {"patient_label": ("label", lambda s: int(s.mode().iloc[0]))}
    if "body_part" in df.columns:
        agg_spec["primary_body_part"] = ("body_part", lambda s: str(s.mode().iloc[0]))

    patient_df = df.groupby("patient_id", sort=True).agg(**agg_spec).reset_index()

    if "primary_body_part" in patient_df.columns:
        joint_strat = patient_df["primary_body_part"] + "_" + patient_df["patient_label"].astype(str)
        if joint_strat.value_counts().min() >= 3:
            strat_col = joint_strat
        elif patient_df["patient_label"].value_counts().min() >= 3:
            strat_col = patient_df["patient_label"]
        else:
            strat_col = None
    elif patient_df["patient_label"].value_counts().min() >= 3:
        strat_col = patient_df["patient_label"]
    else:
        strat_col = None

    patients_rest, patients_test = train_test_split(
        patient_df,
        test_size=test_frac,
        random_state=seed,
        stratify=strat_col,
    )

    relative_val = val_frac / (1.0 - test_frac)
    if strat_col is not None:
        rest_strat = strat_col.loc[patients_rest.index]
        if rest_strat.value_counts().min() < 2:
            rest_strat = patients_rest["patient_label"]
            if rest_strat.value_counts().min() < 2:
                rest_strat = None
    else:
        rest_strat = None

    patients_train, patients_val = train_test_split(
        patients_rest,
        test_size=relative_val,
        random_state=seed,
        stratify=rest_strat,
    )

    train_ids = set(patients_train["patient_id"])
    val_ids = set(patients_val["patient_id"])
    test_ids = set(patients_test["patient_id"])

    assert train_ids.isdisjoint(val_ids), "Patient leakage detected: train ∩ val"
    assert train_ids.isdisjoint(test_ids), "Patient leakage detected: train ∩ test"
    assert val_ids.isdisjoint(test_ids), "Patient leakage detected: val ∩ test"

    train_df = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df


# ───────────────────── class weights ────────────────────────


def compute_weights(train_df: pd.DataFrame) -> Dict[int, float]:
    """Compute balanced class weights strictly from the training split."""
    labels = train_df["label"].values
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


# ───────────────────── baseline array loader ────────────────


def load_flattened_images(
    df: pd.DataFrame,
    size: Tuple[int, int] = (64, 64),
    max_workers: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load images into flattened grayscale float32 arrays in ``[0, 1]``.

    Converts mixed ``L`` (grayscale) and ``RGB`` PNG radiographs consistently
    to single-channel grayscale and resizes to ``size`` for fast baseline models.
    """
    paths = df["abs_path"].tolist()
    labels = df["label"].values.astype(np.int64)

    def _load_one(path: str) -> np.ndarray:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L").resize(size), dtype=np.float32)
        return arr.ravel() / 255.0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        rows = list(ex.map(_load_one, paths))

    return np.vstack(rows), labels


# ───────────────────── TensorFlow dataset pipeline ──────────


def make_tf_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
) -> Tuple:
    """Create leak-free ``tf.data.Dataset`` pipelines for Keras training & evaluation.

    Compatible with both TensorFlow 2.15 (Keras 2) and TensorFlow 2.16+ (Keras 3).
    All images (grayscale or RGB) are decoded consistently to 3-channel RGB,
    resized to ``cfg["image"]["target_size"]``, and rescaled to ``[0, 1]``.
    Data augmentation is applied **only** to ``train_ds``.
    """
    import tensorflow as tf

    img_h, img_w = cfg["image"]["target_size"]
    batch_size = cfg["training"]["batch_size"]
    rescale_factor = float(cfg["image"].get("rescale", 1.0 / 255.0))
    seed = int(cfg.get("seed", 42))
    aug_cfg = cfg.get("augmentation", {})

    aug_layers = None
    if aug_cfg.get("enabled", False):
        layers = []
        if aug_cfg.get("horizontal_flip", True):
            layers.append(tf.keras.layers.RandomFlip("horizontal", seed=seed))
        rot_deg = float(aug_cfg.get("rotation_range", 20))
        if rot_deg > 0:
            layers.append(tf.keras.layers.RandomRotation(rot_deg / 360.0, seed=seed))
        w_shift = float(aug_cfg.get("width_shift_range", 0.1))
        h_shift = float(aug_cfg.get("height_shift_range", 0.1))
        if w_shift > 0 or h_shift > 0:
            layers.append(
                tf.keras.layers.RandomTranslation(
                    height_factor=h_shift, width_factor=w_shift, seed=seed
                )
            )
        zoom = float(aug_cfg.get("zoom_range", 0.1))
        if zoom > 0:
            layers.append(tf.keras.layers.RandomZoom(zoom, seed=seed))
        if layers:
            aug_layers = tf.keras.Sequential(layers, name="train_augmentation")

    def _decode_and_resize(path, label):
        raw_bytes = tf.io.read_file(path)
        img = tf.io.decode_png(raw_bytes, channels=3)
        img = tf.image.resize(img, [img_h, img_w])
        img = tf.cast(img, tf.float32) * rescale_factor
        return img, tf.cast(label, tf.float32)

    def _build_ds(df: pd.DataFrame, is_train: bool):
        ds = tf.data.Dataset.from_tensor_slices(
            (df["abs_path"].values, df["label"].values)
        )
        if is_train:
            ds = ds.shuffle(buffer_size=min(len(df), 8192), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(_decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        if is_train and aug_layers is not None:
            ds = ds.map(
                lambda x, y: (aug_layers(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = _build_ds(train_df, is_train=True)
    val_ds = _build_ds(val_df, is_train=False)
    test_ds = _build_ds(test_df, is_train=False)
    return train_ds, val_ds, test_ds


def make_generators(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
) -> Tuple:
    """Alias for :func:`make_tf_datasets` retained for backwards compatibility."""
    return make_tf_datasets(train_df, val_df, test_df, cfg)


# ───────────────────── convenience entry-point ──────────────


def prepare_data(
    cfg: Optional[dict] = None,
    config_path: str = "configs/base.yaml",
    data_root: Optional[str] = None,
    build_tf: bool = False,
) -> dict:
    """Execute the full leak-free data preparation pipeline.

    Parameters
    ----------
    cfg : dict, optional
        Pre-loaded YAML config dict. Loaded from ``config_path`` if omitted.
    config_path : str
        Fallback config path when ``cfg`` is None.
    data_root : str, optional
        Optional dataset root override (CLI / env var).
    build_tf : bool
        If True, also construct ``tf.data.Dataset`` pipelines (requires TensorFlow).
    """
    if cfg is None:
        cfg = load_config(config_path)

    set_global_seed(int(cfg.get("seed", 42)))

    root = resolve_data_root(cfg, data_root)

    df = build_dataframe(
        root,
        cfg["data"]["train_image_paths"],
        labeled_studies_csv=cfg["data"].get("train_labels"),
    )

    df, cleaning_report = clean_dataframe(df, check_files=True, deduplicate_content=True)

    split_cfg = cfg["split"]
    train_df, val_df, test_df = patient_stratified_split(
        df,
        val_frac=float(split_cfg["val_fraction"]),
        test_frac=float(split_cfg["test_fraction"]),
        seed=int(split_cfg["random_seed"]),
    )

    class_weights = compute_weights(train_df)

    official_valid_df = None
    valid_csv = cfg["data"].get("valid_image_paths")
    if valid_csv and os.path.isfile(os.path.join(root, valid_csv)):
        official_valid_df = build_dataframe(
            root,
            valid_csv,
            labeled_studies_csv=cfg["data"].get("valid_labels"),
        )

    result = {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "official_valid_df": official_valid_df,
        "class_weights": class_weights,
        "cleaning_report": cleaning_report,
        "data_root": root,
        "cfg": cfg,
    }

    if build_tf:
        train_ds, val_ds, test_ds = make_tf_datasets(train_df, val_df, test_df, cfg)
        result.update(
            {
                "train_gen": train_ds,
                "val_gen": val_ds,
                "test_gen": test_ds,
                "train_ds": train_ds,
                "val_ds": val_ds,
                "test_ds": test_ds,
            }
        )

    return result
