"""Load trained VGG16 insulation classifier and run inference on scalogram images."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "VGG16_insulation" / "fold_2" / "best_weights.h5"
CLASS_NAMES = ["Nomex", "Teflon"]

HIGH_CONFIDENCE = 0.85
MODERATE_CONFIDENCE = 0.70


def build_vgg16_model() -> Model:
    base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation="sigmoid")(x)
    return Model(inputs=base_model.input, outputs=predictions)


def load_insulation_model(weights_path: Optional[Union[Path, str]] = None) -> Model:
    weights_path = Path(weights_path or DEFAULT_WEIGHTS)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = build_vgg16_model()
    model.load_weights(str(weights_path))
    return model


def predict_from_image(img_array: np.ndarray, model: Model) -> dict:
    """Predict insulation class from a 224x224 RGB scalogram array."""
    x = img_array.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    score = float(model.predict(x, verbose=0)[0][0])

    if score >= 0.5:
        predicted_class = "Teflon"
        confidence = score
    else:
        predicted_class = "Nomex"
        confidence = 1.0 - score

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "teflon_probability": score,
        "nomex_probability": 1.0 - score,
        "class_names": CLASS_NAMES,
    }


def interpret_prediction(prediction: dict) -> dict:
    """
    Turn model probabilities into user-facing explanations with confidence tiers.
    """
    predicted = prediction["predicted_class"]
    confidence = prediction["confidence"]
    other = "Teflon" if predicted == "Nomex" else "Nomex"
    other_prob = prediction["teflon_probability"] if predicted == "Nomex" else prediction["nomex_probability"]
    nomex_pct = prediction["nomex_probability"]
    teflon_pct = prediction["teflon_probability"]

    if confidence >= HIGH_CONFIDENCE:
        level = "high"
        headline = f"High confidence — classified as {predicted}"
        summary = (
            f"The UHF partial discharge signal is **classified as {predicted}** "
            f"with **{confidence:.1%}** model confidence. "
            f"The generated scalogram closely matches the {predicted} patterns seen during training."
        )
        detail = (
            f"**{predicted}:** {confidence:.1%} · **{other}:** {other_prob:.1%}. "
            f"The model is strongly leaning toward {predicted}."
        )
        recommendation = "This result is reliable enough for reporting."
    elif confidence >= MODERATE_CONFIDENCE:
        level = "moderate"
        headline = f"Moderate confidence — likely {predicted}"
        summary = (
            f"The material **appears to be {predicted}** ({confidence:.1%} confidence). "
            f"It looks more like {predicted} than {other}, but some signal features overlap "
            f"with both insulation types."
        )
        detail = (
            f"**{predicted}:** {confidence:.1%} · **{other}:** {other_prob:.1%}. "
            f"The signal resembles {predicted}, but not decisively."
        )
        recommendation = (
            "Suitable for initial screening. If this result is critical, "
            "upload another sample or verify manually."
        )
    else:
        level = "low"
        headline = f"Low confidence — uncertain, leans toward {predicted}"
        summary = (
            f"The material **looks like {predicted}** ({confidence:.1%} confidence), "
            f"but the model is **not sure**. "
            f"Nomex probability: **{nomex_pct:.1%}**, Teflon probability: **{teflon_pct:.1%}**. "
            f"The signal may be noisy, mixed, or different from typical training samples."
        )
        detail = (
            f"Because confidence is below {MODERATE_CONFIDENCE:.0%}, treat this as a weak indication "
            f"toward {predicted}, not a final decision."
        )
        recommendation = (
            "Manual review is recommended. Check CSV format, signal quality, "
            "and try another measurement if possible."
        )

    return {
        "level": level,
        "headline": headline,
        "summary": summary,
        "detail": detail,
        "recommendation": recommendation,
        "predicted_class": predicted,
        "confidence": confidence,
        "nomex_probability": nomex_pct,
        "teflon_probability": teflon_pct,
    }
