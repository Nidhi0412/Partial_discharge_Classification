"""Streamlit UI for final Nomex/Teflon scalogram classification models."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from inference import (  # noqa: E402
    MODEL_SPECS,
    agreement_label,
    decode_uploaded_image,
    load_final_model,
    predict_scalogram,
)

RESULTS_DIR = PROJECT_ROOT / "analysis" / "manuscript_results_package_20261112_corrected_resnet_final"
GRADCAM_DIR = PROJECT_ROOT / "analysis" / "gradcam_final_20261112" / "manuscript_package"

st.set_page_config(
    page_title="UHF PD Scalogram Material Classification",
    page_icon="⚡",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_model(model_name: str):
    return load_final_model(model_name)


def probability_card(prediction: dict) -> None:
    st.subheader(prediction["model"])
    st.success(f"Predicted material: **{prediction['predicted_class']}**")
    left, right = st.columns(2)
    left.metric("Nomex output probability", f"{prediction['p_nomex']:.2%}")
    right.metric("Teflon output probability", f"{prediction['p_teflon']:.2%}")
    st.caption(
        f"Decision threshold: p(Teflon) ≥ {prediction['decision_threshold']:.2f} · "
        f"Preprocessing: {prediction['preprocessing']}"
    )


def classification_tab() -> None:
    st.header("Classify a scalogram")
    st.info(
        "Upload a PNG or JPEG scalogram generated consistently with the study dataset. "
        "This version does not generate a scalogram from a raw waveform CSV."
    )
    uploaded = st.file_uploader("Scalogram image", type=["png", "jpg", "jpeg"])
    selection = st.radio(
        "Model",
        ["Compare both", "VGG16", "ResNet50"],
        horizontal=True,
        help="The comparison reports both models independently; it is not an ensemble.",
    )
    if uploaded is None:
        st.caption("Choose a scalogram image to enable classification.")
        return

    try:
        image, image_metadata = decode_uploaded_image(uploaded)
    except Exception as exc:
        st.error(str(exc))
        return

    preview, details = st.columns([2, 1])
    with preview:
        st.image(image, caption="Inference image (224×224 RGB)", width=448)
    with details:
        st.markdown("#### Input processing")
        st.json(image_metadata)
        st.caption("Nearest-neighbour resizing matches the corrected evaluation pipeline.")

    if not st.button("Run classification", type="primary", width="stretch"):
        return

    names = list(MODEL_SPECS) if selection == "Compare both" else [selection]
    predictions = []
    try:
        progress = st.status("Running final fitted model(s)…", expanded=True)
        for name in names:
            progress.write(f"Loading and verifying {name} checkpoint…")
            model = get_model(name)
            progress.write(f"Applying {name}-specific preprocessing and inference…")
            predictions.append(predict_scalogram(image, model, name))
        progress.update(label="Classification complete", state="complete", expanded=False)
    except Exception as exc:
        st.error(f"Classification could not be completed: {exc}")
        st.exception(exc)
        return

    st.markdown("### Model output")
    columns = st.columns(len(predictions))
    for column, prediction in zip(columns, predictions):
        with column:
            probability_card(prediction)

    if len(predictions) == 2:
        agreement = agreement_label(predictions)
        if agreement == "Disagreement":
            st.warning(
                "The two final models disagree. Report both outputs and review the image; "
                "no unvalidated probability averaging has been applied."
            )
        else:
            st.success(f"Both models classify this scalogram as **{agreement}**.")

    st.warning(
        "These are model output probabilities, not externally validated diagnostic certainty. "
        "The models were evaluated internally on the supplied Nomex/Teflon scalogram dataset."
    )
    uploaded.seek(0)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "uploaded_filename": uploaded.name,
        "uploaded_sha256": hashlib.sha256(uploaded.read()).hexdigest(),
        "image_processing": image_metadata,
        "predictions": predictions,
        "scope": "Internal Nomex/Teflon scalogram material classification",
    }
    st.download_button(
        "Download prediction record",
        json.dumps(record, indent=2),
        file_name="scalogram_prediction.json",
        mime="application/json",
        width="stretch",
    )


def evidence_tab() -> None:
    st.header("Corrected experimental evidence")
    st.caption("Development results use common grouped five-fold out-of-fold predictions.")
    st.dataframe(
        [
            {"Model": "Corrected ResNet50", "Development accuracy": "99.45%", "Grouped 95% CI": "99.01–99.84%", "ROC-AUC": "0.99998", "Locked holdout": "99.38%"},
            {"Model": "VGG16", "Development accuracy": "99.06%", "Grouped 95% CI": "98.44–99.60%", "ROC-AUC": "0.99952", "Locked holdout": "100.00%"},
            {"Model": "MobileNetV3-Small fine-tuned", "Development accuracy": "96.72%", "Grouped 95% CI": "95.60–97.79%", "ROC-AUC": "0.9947", "Locked holdout": "Not evaluated"},
            {"Model": "MobileNetV3-Small frozen", "Development accuracy": "95.31%", "Grouped 95% CI": "93.99–96.59%", "ROC-AUC": "0.9919", "Locked holdout": "Not evaluated"},
            {"Model": "Small CNN", "Development accuracy": "78.28%", "Grouped 95% CI": "75.81–80.70%", "ROC-AUC": "0.8914", "Locked holdout": "Not evaluated"},
        ],
        hide_index=True,
        width="stretch",
    )
    st.info(
        "Corrected ResNet50 exceeded VGG16 by 0.39 percentage points in development, "
        "but the paired grouped 95% interval (−0.23 to 1.03 points) included zero. "
        "The evidence supports comparable accuracy, not established superiority."
    )
    figures = [
        ("Grouped development accuracy", RESULTS_DIR / "figures" / "figure_1_cv_accuracy_with_grouped_ci.png"),
        ("Out-of-fold ROC curves", RESULTS_DIR / "figures" / "figure_2_oof_roc_curves.png"),
        ("Out-of-fold confusion matrices", RESULTS_DIR / "figures" / "figure_3_oof_confusion_matrices.png"),
        ("Calibration reliability", RESULTS_DIR / "figures" / "figure_6_calibration_reliability.png"),
        ("Fold learning curves", RESULTS_DIR / "figures" / "figure_7_fold_learning_curves.png"),
    ]
    for index in range(0, len(figures), 2):
        cols = st.columns(2)
        for col, (caption, path) in zip(cols, figures[index:index + 2]):
            with col:
                if path.is_file():
                    st.image(str(path), caption=caption, width="stretch")


def research_comparison_tab() -> None:
    st.header("Research comparison and experimental reasoning")
    st.markdown(
        "The comparison was designed to answer more than which network obtained the largest "
        "accuracy. Every model used the same 1,280-image development cohort, identical grouped "
        "five-fold membership and training-only augmentation, so architectural and transfer-"
        "learning differences could be interpreted under one evaluation protocol."
    )

    st.subheader("Why these five configurations were included")
    model_cards = [
        (
            "Small CNN — representation baseline",
            "Trained without an ImageNet-pretrained backbone to test whether the supplied "
            "scalograms automatically make the task easy for any CNN.",
            "78.28% accuracy and 0.8914 ROC-AUC. The large gap from the transfer-learning "
            "models shows that model architecture and pretrained representation materially matter.",
        ),
        (
            "MobileNetV3-Small frozen — lightweight transfer baseline",
            "Used a compact pretrained backbone with only the task head trained. This tested "
            "whether an efficient feature extractor could provide a practical alternative.",
            "95.31% accuracy and 0.9919 ROC-AUC. Compact pretrained features were already useful, "
            "but remained below the larger final models.",
        ),
        (
            "MobileNetV3-Small fine-tuned — controlled adaptation",
            "Started from the saved frozen-head checkpoint, unfroze six late weighted layers and "
            "kept Batch Normalization frozen. This isolated the contribution of limited adaptation.",
            "96.72% accuracy and 0.9947 ROC-AUC. Fine-tuning improved accuracy by 1.41 percentage "
            "points over the frozen MobileNet configuration.",
        ),
        (
            "VGG16 — principal transfer-learning model",
            "Retained the established VGG16 experiment under the corrected grouped protocol and "
            "training-only scalogram augmentation.",
            "99.06% development accuracy and 100.00% internal-holdout accuracy. It remained a "
            "strong final classifier but was not declared statistically superior to ResNet50.",
        ),
        (
            "Corrected ResNet50 — preprocessing and fine-tuning correction",
            "Used native ResNet50 preprocessing, a lower regularization burden, late-layer "
            "fine-tuning and frozen Batch Normalization to correct the weak earlier configuration.",
            "99.45% development accuracy and 99.38% internal-holdout accuracy. Its 0.39-point "
            "development advantage over VGG16 was not statistically established.",
        ),
    ]
    for title, purpose, finding in model_cards:
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.markdown(f"**Experimental purpose:** {purpose}")
            st.markdown(f"**Finding:** {finding}")

    st.subheader("How the comparison supports the study")
    st.markdown(
        """
        1. **Leakage-controlled evaluation:** all configurations used the same group-aware folds, preventing an unfair comparison based on different data membership.
        2. **A genuine lower-capacity baseline:** the small CNN result demonstrates that near-perfect performance was not produced by every image classifier.
        3. **Efficiency evidence:** MobileNetV3-Small achieved strong performance with a compact architecture, providing a useful accuracy–complexity comparison.
        4. **Fine-tuning evidence:** the controlled MobileNet comparison shows that limited backbone adaptation improved performance over frozen feature extraction.
        5. **Architecture-specific preprocessing matters:** correcting the ResNet50 input transform and training strategy removed the earlier artificial weakness.
        6. **No exaggerated winner claim:** the paired grouped interval for ResNet50 minus VGG16 accuracy was −0.23 to 1.03 percentage points, so the two final models are described as comparable.
        """
    )

    comparison_figure = RESULTS_DIR / "figures" / "figure_4_accuracy_f1_auc_comparison.png"
    ablation_figure = RESULTS_DIR / "figures" / "figure_5_mobilenet_finetuning_ablation.png"
    cols = st.columns(2)
    if comparison_figure.is_file():
        cols[0].image(str(comparison_figure), caption="Five-configuration performance comparison", width="stretch")
    if ablation_figure.is_file():
        cols[1].image(str(ablation_figure), caption="MobileNet frozen versus fine-tuned ablation", width="stretch")

    st.subheader("What should and should not be concluded")
    supported, unsupported = st.columns(2)
    with supported:
        st.success("Supported by the current evidence")
        st.markdown(
            """
            - Strong internal Nomex/Teflon scalogram discrimination.
            - Clear benefit from pretrained representations over the small CNN baseline.
            - A measurable benefit from controlled MobileNet fine-tuning.
            - Comparable final VGG16 and corrected ResNet50 accuracy.
            """
        )
    with unsupported:
        st.warning("Not established by the current evidence")
        st.markdown(
            """
            - Universal performance on new laboratories or acquisition sessions.
            - ResNet50 statistical superiority over VGG16.
            - Physical causality inferred solely from Grad-CAM.
            - Optimality of db3 or the selected wavelet decomposition level.
            - Validated physical waveform-noise robustness.
            """
        )


def explainability_tab() -> None:
    st.header("Explainability evidence")
    st.markdown(
        "Grad-CAM was evaluated across the complete 320-image locked holdout for both final "
        "models. The maps indicate image regions influencing a prediction; they do not prove "
        "a physical discharge mechanism or calibrated RF frequency band."
    )
    for caption, path in [
        ("Representative final-model Grad-CAM overlays", GRADCAM_DIR / "figures" / "figure_10_gradcam_representative_overlays.png"),
        ("Grad-CAM quantitative diagnostics", GRADCAM_DIR / "figures" / "figure_12_gradcam_quantitative_diagnostics.png"),
    ]:
        if path.is_file():
            st.image(str(path), caption=caption, width="stretch")
    st.warning(
        "A large fraction of Teflon maps were zero or spatially flat (86.88% for VGG16 and "
        "78.13% for corrected ResNet50). Such maps do not support reliable localization claims."
    )


def limitations_tab() -> None:
    st.header("Scope and limitations")
    st.markdown(
        """
        - This application classifies **Nomex versus Teflon scalogram images**; it does not detect PD absence/presence, severity or fault location.
        - Validation is internal to the supplied image dataset and recorded grouping scheme.
        - Generalization to new laboratories, acquisition sessions, voltages, sensors or electrode configurations is not established.
        - The uploaded image should follow the same scalogram-generation convention as the study dataset.
        - The current application does not convert raw waveform CSV files.
        - Controlled image-space corruption testing and acquisition-independent validation remain future experiments.
        """
    )


def main() -> None:
    st.title("UHF Partial-Discharge Scalogram Material Classification")
    st.caption("Final VGG16 and corrected ResNet50 · Nomex versus Teflon")
    classify, evidence, comparison, explainability, limitations = st.tabs(
        ["Classification", "Experimental evidence", "Research comparison", "Explainability", "Limitations"]
    )
    with classify:
        classification_tab()
    with evidence:
        evidence_tab()
    with comparison:
        research_comparison_tab()
    with explainability:
        explainability_tab()
    with limitations:
        limitations_tab()


if __name__ == "__main__":
    main()
