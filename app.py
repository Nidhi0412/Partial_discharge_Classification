"""Streamlit UI for UHF Partial Discharge insulation classification."""

import os
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from inference import (  # noqa: E402
    DEFAULT_WEIGHTS,
    HIGH_CONFIDENCE,
    MODERATE_CONFIDENCE,
    interpret_prediction,
    load_insulation_model,
    predict_from_image,
)
from scalogram_utils import ScalogramProcessor  # noqa: E402

PROJECT_TITLE = "UHF Partial Discharge Insulation Classification"
PROJECT_SUBTITLE = (
    "Upload a UHF PD signal CSV to classify insulation type (Nomex vs Teflon) "
    "using a trained VGG16 model on wavelet scalograms."
)


@st.cache_resource(show_spinner="Loading trained model...")
def get_model():
    return load_insulation_model(DEFAULT_WEIGHTS)


@st.cache_resource
def get_scalogram_processor():
    return ScalogramProcessor(target_size=(224, 224))


def render_header():
    st.set_page_config(
        page_title=PROJECT_TITLE,
        page_icon="⚡",
        layout="centered",
    )
    st.title(PROJECT_TITLE)
    st.caption(PROJECT_SUBTITLE)
    st.divider()


def render_result(prediction: dict, scalogram_bytes: BytesIO):
    interpretation = interpret_prediction(prediction)
    predicted = prediction["predicted_class"]
    confidence = prediction["confidence"]

    st.subheader("Classification Result")

    if interpretation["level"] == "high":
        st.success(f"**{interpretation['headline']}**")
    elif interpretation["level"] == "moderate":
        st.warning(f"**{interpretation['headline']}**")
    else:
        st.error(f"**{interpretation['headline']}**")

    st.markdown("#### What this means")
    st.markdown(interpretation["summary"])
    st.caption(interpretation["detail"])

    st.markdown("#### Recommendation")
    st.info(interpretation["recommendation"])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted insulation", predicted)
    with col2:
        st.metric("Model confidence", f"{confidence:.1%}")

    st.progress(confidence, text=f"Confidence: {confidence:.1%}")

    prob_col1, prob_col2 = st.columns(2)
    with prob_col1:
        st.metric("Nomex probability", f"{prediction['nomex_probability']:.1%}")
    with prob_col2:
        st.metric("Teflon probability", f"{prediction['teflon_probability']:.1%}")

    with st.expander("How to read confidence levels"):
        st.markdown(
            f"""
            - **High (≥ {HIGH_CONFIDENCE:.0%}):** The signal strongly matches one insulation type.
            - **Moderate ({MODERATE_CONFIDENCE:.0%}–{HIGH_CONFIDENCE:.0%}):** The material *appears* to be
              the predicted class, but some overlap exists with the other type.
            - **Low (< {MODERATE_CONFIDENCE:.0%}):** The material *looks like* the predicted class,
              but the result is uncertain — manual review is advised.
            """
        )

    st.image(scalogram_bytes, caption="Generated scalogram (224×224)", use_container_width=True)


def run_pipeline(uploaded_file):
    processor = get_scalogram_processor()
    model = get_model()

    with st.status("Running analysis pipeline...", expanded=True) as status:
        st.write("**Step 1/3 — Preprocessing CSV**")
        st.caption("Reading signal, validating columns, and computing sampling frequency.")
        signal_values, sampling_freq = processor.csv_to_signal(uploaded_file)
        st.success(
            f"Loaded {len(signal_values):,} samples at {sampling_freq:,.0f} Hz sampling frequency."
        )

        st.write("**Step 2/3 — Converting to scalogram**")
        st.caption("Applying DWT (db3) and CWT (Morlet) to build a 224×224 image.")
        img_array, scalogram_buf = processor.generate_scalogram(
            signal_values, sampling_freq, level=1
        )
        st.success("Scalogram generated successfully.")

        st.write("**Step 3/3 — Classifying insulation type**")
        st.caption("Running VGG16 (fold 2) on the generated scalogram.")
        prediction = predict_from_image(img_array, model)
        st.success(
            f"Classification complete: **{prediction['predicted_class']}** "
            f"({prediction['confidence']:.1%} confidence)."
        )

        status.update(label="Analysis complete", state="complete", expanded=False)

    return prediction, scalogram_buf


def main():
    render_header()

    st.markdown("### Upload signal")
    st.info(
        "CSV format: skip first 21 rows; column 1 = time, column 2 = UHF signal. "
        "Expected ~10,000 data points per file."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload an unseen UHF partial discharge signal CSV.",
    )

    run_clicked = st.button(
        "Run Classification",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
    )

    if uploaded_file is None:
        st.warning("Upload a CSV file to start classification.")
        return

    if not run_clicked:
        st.caption(f"Selected file: **{uploaded_file.name}** — click **Run Classification** to begin.")
        return

    try:
        uploaded_file.seek(0)
        prediction, scalogram_buf = run_pipeline(uploaded_file)
        st.divider()
        render_result(prediction, scalogram_buf)
    except Exception as exc:
        st.error(f"Processing failed: {exc}")
        st.exception(exc)


if __name__ == "__main__":
    main()
