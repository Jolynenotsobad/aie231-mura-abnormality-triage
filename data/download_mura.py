"""
download_mura.py — Download and verify the MURA v1.1 dataset locally.

Supports:
  1. Verifying an already-extracted MURA-v1.1 folder against expected CSV counts.
  2. Downloading from Kaggle via the ``kaggle`` CLI if credentials are configured.

Usage:
    python data/download_mura.py --verify --data-root /path/to/MURA-v1.1
    python data/download_mura.py --kaggle --dest data/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile


EXPECTED_FILES = [
    "train_image_paths.csv",
    "train_labeled_studies.csv",
    "valid_image_paths.csv",
    "valid_labeled_studies.csv",
]


def verify_dataset(data_root: str) -> bool:
    """Verify that ``data_root`` contains all expected MURA v1.1 manifests and folders."""
    if not os.path.isdir(data_root):
        print(f"[FAIL] Directory does not exist: {data_root}")
        return False

    ok = True
    for fname in EXPECTED_FILES:
        fpath = os.path.join(data_root, fname)
        if not os.path.isfile(fpath):
            print(f"[FAIL] Missing manifest: {fpath}")
            ok = False
        else:
            with open(fpath, "r", encoding="utf-8") as f:
                n_lines = sum(1 for _ in f)
            print(f"[OK]   {fname}: {n_lines:,} rows")

    for split in ("train", "valid"):
        sdir = os.path.join(data_root, split)
        if not os.path.isdir(sdir):
            print(f"[FAIL] Missing split folder: {sdir}")
            ok = False
        else:
            parts = sorted(os.listdir(sdir))
            print(f"[OK]   {split}/ contains {len(parts)} body-part folders: {parts}")

    return ok


def download_from_kaggle(dest_dir: str = "data") -> None:
    """Download and unzip MURA v1.1 using the Kaggle CLI."""
    os.makedirs(dest_dir, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", "cjinny/mura-v11", "-p", dest_dir]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    zip_path = os.path.join(dest_dir, "mura-v11.zip")
    if os.path.isfile(zip_path):
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        print("Extraction complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or verify MURA v1.1 dataset")
    parser.add_argument("--verify", action="store_true", help="Verify local MURA directory")
    parser.add_argument("--kaggle", action="store_true", help="Download MURA via Kaggle CLI")
    parser.add_argument("--data-root", type=str, default="data/MURA-v1.1")
    parser.add_argument("--dest", type=str, default="data")
    args = parser.parse_args()

    if args.kaggle:
        download_from_kaggle(args.dest)
    if args.verify or not args.kaggle:
        success = verify_dataset(args.data_root)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
