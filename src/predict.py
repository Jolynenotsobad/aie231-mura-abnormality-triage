"""
predict.py — Inference pipeline for MURA abnormality detection.

Provides a single entry point that takes a raw X-ray image file path,
applies the same preprocessing as training, and returns a prediction
with its confidence score.

Full implementation will be completed after the final model is saved
(Checkpoint 4).  The interface is defined here for Checkpoint 2.

Usage:
    python -m src.predict --image path/to/xray.png

Authors: Team — Evaluation & Deployment Engineer work package.
"""

from __future__ import annotations

import argparse


def predict_single(image_path: str, model_path: str = "models/final_model.keras") -> dict:
    """Run inference on a single X-ray image.

    Parameters
    ----------
    image_path : str
        Path to a PNG/JPEG X-ray image.
    model_path : str
        Path to the saved Keras model.

    Returns
    -------
    dict  with keys: label, confidence, class_name
    """
    # Will be fully implemented in Checkpoint 4 after the model is trained.
    raise NotImplementedError(
        "predict_single() requires a trained model. "
        "Run training first (Checkpoint 3) and save the model to models/."
    )


def main():
    parser = argparse.ArgumentParser(description="MURA single-image inference")
    parser.add_argument("--image", type=str, required=True, help="Path to X-ray image")
    parser.add_argument("--model", type=str, default="models/final_model.keras")
    args = parser.parse_args()

    result = predict_single(args.image, args.model)
    print(f"Prediction : {result['class_name']}")
    print(f"Confidence : {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
