"""Inference helpers for the final Nomex/Teflon scalogram classifiers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ("Nomex", "Teflon")
IMAGE_SIZE = 224

MODEL_SPECS: Mapping[str, dict] = {
    "VGG16": {
        "path": PROJECT_ROOT / "runs" / "primary_png_20261104_01" / "vgg16_final" / "best.keras",
        "sha256": "900b8714de627a9d7a01d4548fc7f97ca490862a704fa238ba7e16f4bd3c47b1",
        "preprocessing": "unit_scale",
    },
    "ResNet50": {
        "path": PROJECT_ROOT / "runs" / "resnet50_corrected_20261105_01" / "resnet50_finetune_final" / "best.keras",
        "sha256": "733c050765c8ba4e619eace93ded7f1bbcd581016128d7175502e3cb95ebb6d0",
        "preprocessing": "resnet50_imagenet",
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(model_name: str) -> Path:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model_name}")
    spec = MODEL_SPECS[model_name]
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{model_name} checkpoint is unavailable. Expected: {path}")
    if file_sha256(path) != spec["sha256"]:
        raise RuntimeError(f"{model_name} checkpoint hash mismatch; refusing unverified inference.")
    return path


def load_final_model(model_name: str):
    """Load and validate one immutable final-fit checkpoint."""
    import tensorflow as tf

    model = tf.keras.models.load_model(validate_checkpoint(model_name), compile=False)
    if tuple(model.input_shape[1:]) != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise RuntimeError(f"Unexpected {model_name} input shape: {model.input_shape}")
    if model.output_shape[-1] != 1 or "p_teflon" not in model.output_names:
        raise RuntimeError(f"Unexpected {model_name} output contract")
    return model


def decode_uploaded_image(uploaded_file) -> tuple[np.ndarray, dict]:
    """Decode an uploaded scalogram as RGB and resize exactly as evaluation did."""
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a readable PNG/JPEG image.") from exc
    original_size, original_mode = image.size, image.mode
    resized = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.NEAREST
    )
    array = np.asarray(resized, dtype=np.uint8)
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected decoded image shape: {array.shape}")
    return array, {
        "original_width": original_size[0],
        "original_height": original_size[1],
        "original_mode": original_mode,
        "inference_width": IMAGE_SIZE,
        "inference_height": IMAGE_SIZE,
        "resize_interpolation": "nearest",
    }


def preprocess_image(image: np.ndarray, model_name: str) -> np.ndarray:
    """Apply the architecture-specific preprocessing saved with training."""
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model_name}")
    image = np.asarray(image, dtype=np.float32)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or not np.isfinite(image).all():
        raise ValueError("Expected one finite 224x224 RGB scalogram")
    profile = MODEL_SPECS[model_name]["preprocessing"]
    if profile == "unit_scale":
        transformed = image / 255.0
    elif profile == "resnet50_imagenet":
        from tensorflow.keras.applications.resnet50 import preprocess_input

        transformed = preprocess_input(image.copy())
    else:
        raise RuntimeError(f"Unsupported preprocessing profile: {profile}")
    return transformed[None, ...]


def predict_scalogram(image: np.ndarray, model, model_name: str) -> dict:
    score = float(model.predict(preprocess_image(image, model_name), verbose=0).reshape(-1)[0])
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError(f"Invalid {model_name} prediction: {score}")
    return {
        "model": model_name,
        "predicted_class": "Teflon" if score >= 0.5 else "Nomex",
        "p_teflon": score,
        "p_nomex": 1.0 - score,
        "decision_threshold": 0.5,
        "checkpoint_sha256": MODEL_SPECS[model_name]["sha256"],
        "preprocessing": MODEL_SPECS[model_name]["preprocessing"],
    }


def agreement_label(predictions: list[dict]) -> str:
    labels = {item["predicted_class"] for item in predictions}
    return labels.pop() if len(labels) == 1 else "Disagreement"
