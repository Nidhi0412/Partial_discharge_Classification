"""Create a manuscript package from frozen full-cohort Grad-CAM diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


MODELS = {
    "VGG16": "vgg16",
    "ResNet50 (corrected)": "resnet50_corrected",
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_model_data(root, folder):
    base = root / folder
    selection = json.loads((base / "selection.json").read_text(encoding="utf-8"))
    metadata = {row["image_id"]: row for row in selection["rows"]}
    diagnostics = read_csv(base / "image_diagnostics.csv")
    for row in diagnostics:
        row.update(metadata[row["image_id"]])
        for key in ["faithfulness_margin", "normalized_entropy", "map_is_flat",
                    "stability_cosine", "p_teflon"]:
            if row.get(key, "") != "":
                row[key] = float(row[key])
    return diagnostics


def representative(rows, material):
    candidates = [r for r in rows if r["material"] == material and not r["map_is_flat"]]
    if not candidates:
        raise ValueError(f"No non-flat maps for {material}")
    faith = np.array([r["faithfulness_margin"] for r in candidates])
    entropy = np.array([r["normalized_entropy"] for r in candidates])
    fscale = max(float(np.ptp(faith)), 1e-12)
    escale = max(float(np.ptp(entropy)), 1e-12)
    fm, em = float(np.median(faith)), float(np.median(entropy))
    return min(candidates, key=lambda r: (abs(r["faithfulness_margin"]-fm)/fscale +
                                          abs(r["normalized_entropy"]-em)/escale,
                                          r["image_id"]))


def image_and_heat(row):
    image = np.asarray(Image.open(row["image_path"]).convert("RGB"), dtype=float) / 255.0
    heat = np.load(row["map_path"], allow_pickle=False)
    return image, heat


def add_triplet(axes, row, title):
    image, heat = image_and_heat(row)
    axes[0].imshow(image); axes[0].set_title(title + "\nOriginal")
    axes[1].imshow(heat, cmap="jet", vmin=0, vmax=1); axes[1].set_title("Grad-CAM")
    axes[2].imshow(image); axes[2].imshow(heat, cmap="jet", alpha=.42,
                                         vmin=0, vmax=1,
                                         extent=(0, image.shape[1], image.shape[0], 0))
    axes[2].set_title("Overlay")
    for ax in axes: ax.axis("off")


def save(fig, output, stem):
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def metric_lookup(path):
    rows = read_csv(path)
    return {(r["material"], r["metric"]): r for r in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root, out = Path(args.analysis_dir).resolve(), Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    figures = out / "figures"; figures.mkdir()

    data = {name: load_model_data(root, folder) for name, folder in MODELS.items()}
    stats = {name: metric_lookup(root / folder / "explainability_statistics.csv")
             for name, folder in MODELS.items()}

    # Deterministic representative: closest jointly to within-stratum median
    # faithfulness and entropy among non-flat maps.
    chosen = []
    fig, axes = plt.subplots(4, 3, figsize=(11, 12))
    row_index = 0
    for model, rows in data.items():
        for material in ["Nomex", "Teflon"]:
            row = representative(rows, material)
            chosen.append({"model": model, "material": material, "image_id": row["image_id"],
                           "selection_rule": "Closest to stratum medians of faithfulness margin and normalized entropy among non-flat maps"})
            add_triplet(axes[row_index], row, f"{model}; {material}")
            row_index += 1
    fig.suptitle("Representative true-class Grad-CAM maps from the locked holdout", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .97))
    save(fig, figures, "figure_10_gradcam_representative_overlays")
    write_csv(out / "representative_selection.csv", chosen)

    # Include all ResNet errors; no error examples exist for VGG16.
    errors = [r for r in data["ResNet50 (corrected)"]
              if (r["p_teflon"] >= .5) != (r["material"] == "Teflon")]
    if errors:
        fig, axes = plt.subplots(len(errors), 3, figsize=(11, 3.3 * len(errors)), squeeze=False)
        for index, row in enumerate(sorted(errors, key=lambda r: r["image_id"])):
            add_triplet(axes[index], row,
                        f"ResNet50 error; true {row['material']}; p(Teflon)={row['p_teflon']:.3f}")
        fig.suptitle("All corrected ResNet50 locked-holdout errors", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, .95))
        save(fig, figures, "figure_11_resnet50_all_misclassified_gradcam")

    summary = []
    for model in MODELS:
        for material in ["Nomex", "Teflon"]:
            lookup = stats[model]
            row = {"Model": model, "Material": material}
            for label, metric in [("Flat-map rate", "map_is_flat"),
                                  ("Stability cosine", "stability_cosine"),
                                  ("Faithfulness margin", "faithfulness_margin"),
                                  ("80%-mass area fraction", "mass_area_fraction"),
                                  ("Left-third activation mass", "region_left_time_third"),
                                  ("Middle-third activation mass", "region_middle_time_third"),
                                  ("Right-third activation mass", "region_right_time_third"),
                                  ("Upper-half activation mass", "region_upper_image_half")]:
                item = lookup[(material, metric)]
                row[label] = item["mean"]
                row[label + " 95% CI"] = f"{item['ci95_lower']} to {item['ci95_upper']}"
                row[label + " valid images"] = item["images_valid"]
            summary.append(row)
    write_csv(out / "table_5_gradcam_quantitative_summary.csv", summary)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    labels = [f"{m.split()[0]}\n{c}" for m in MODELS for c in ["Nomex", "Teflon"]]
    combinations = [(m, c) for m in MODELS for c in ["Nomex", "Teflon"]]
    for ax, metric, title in [
        (axes[0], "map_is_flat", "Zero/flat-map fraction"),
        (axes[1], "stability_cosine", "Translation stability cosine"),
        (axes[2], "faithfulness_margin", "Attended-minus-random mask drop")]:
        values = [float(stats[m][(c, metric)]["mean"]) for m, c in combinations]
        ax.bar(np.arange(4), values, color=["#355C7D", "#6C8EAD", "#B56576", "#D59AA5"])
        ax.set_xticks(np.arange(4), labels, fontsize=8)
        ax.set_title(title); ax.grid(axis="y", alpha=.25)
        if metric == "stability_cosine": ax.set_ylim(.95, 1.0)
        if metric == "map_is_flat": ax.set_ylim(0, 1.0)
        ax.axhline(0, color="black", linewidth=.7)
    fig.suptitle("Full-cohort Grad-CAM diagnostics on the locked holdout", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, figures, "figure_12_gradcam_quantitative_diagnostics")

    v_nomex = stats["VGG16"][("Nomex", "faithfulness_margin")]
    r_nomex = stats["ResNet50 (corrected)"][("Nomex", "faithfulness_margin")]
    text = f"""# Grad-CAM explainability results

## Methods

True-class Grad-CAM was calculated from the final convolutional layer (`block5_conv3` for VGG16 and `conv5_block3_out` for corrected ResNet50) for every image in the 320-image locked internal holdout. The analysis specification was frozen before map inspection. It included 1,000 `group_id`-clustered bootstrap resamples, horizontal translation stability at -2 and +2 pixels, and a faithfulness diagnostic that masked the most activated 10% of pixels and compared the true-class score reduction with ten equal-area random masks. All summaries include the complete cohort rather than selected correct predictions.

## Results

Both models produced highly translation-stable non-flat activation maps. Mean stability cosine was {float(stats['VGG16'][('Nomex','stability_cosine')]['mean']):.3f} for VGG16-Nomex and {float(stats['ResNet50 (corrected)'][('Nomex','stability_cosine')]['mean']):.3f} for ResNet50-Nomex. Prediction-change rates under the small translations were zero for VGG16 and ResNet50-Nomex and 0.003 for ResNet50-Teflon.

For Nomex, attended-region masking reduced the true-class score more than equal-area random masking. The mean faithfulness margin was {float(v_nomex['mean']):.4f} (95% grouped interval {float(v_nomex['ci95_lower']):.4f} to {float(v_nomex['ci95_upper']):.4f}) for VGG16 and {float(r_nomex['mean']):.4f} ({float(r_nomex['ci95_lower']):.4f} to {float(r_nomex['ci95_upper']):.4f}) for ResNet50.

True-class Teflon Grad-CAM was frequently zero/flat at the selected final layer: {100*float(stats['VGG16'][('Teflon','map_is_flat')]['mean']):.1f}% of VGG16 maps and {100*float(stats['ResNet50 (corrected)'][('Teflon','map_is_flat')]['mean']):.1f}% of ResNet50 maps. Faithfulness and stability summaries for this stratum therefore use only valid non-flat maps and must not be generalized to all Teflon images. This behavior is compatible with saturated binary outputs and the positive-gradient/ReLU definition of Grad-CAM, but the present analysis cannot identify a unique cause.

## Interpretation and limitations

The maps show where final-layer activations support each model's true-class score in image coordinates. They do not demonstrate a physical discharge mechanism, calibrated frequency band or causal material signature. The PNGs lack complete calibrated time/frequency axes, and masking or translations can create out-of-distribution images. Grad-CAM results should therefore be described as model-attention diagnostics. The high zero-map rate for Teflon is an explicit method limitation; it must not be hidden by showing only attractive non-flat examples.

The representative panels use a deterministic rule: the non-flat map closest to the within-model/material medians of faithfulness margin and normalized entropy. The ResNet50 error panel contains every misclassified holdout image. Quantitative conclusions come from the complete cohort, not the displayed examples.
"""
    (out / "GRADCAM_MANUSCRIPT_WRITEUP.md").write_text(text, encoding="utf-8")
    (out / "README.md").write_text(
        "# Final Grad-CAM package\n\nFull locked-holdout true-class maps and diagnostics are in the sibling model folders. This package contains manuscript figures, the deterministic example-selection manifest, the quantitative summary table, and cautious manuscript wording.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
