"""Paired, group-aware comparison of saved development out-of-fold predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, rankdata


MODEL_FILES = {
    "VGG16": "runs/primary_png_20261104_01/vgg16_cv/oof_predictions.csv",
    "ResNet50": "runs/primary_png_20261104_01/resnet50_cv/oof_predictions.csv",
    "Small CNN": "runs/baselines_20261105_01/small_cnn_cv/oof_predictions.csv",
    "MobileNetV3-Small (frozen)": "runs/baselines_20261105_01/mobilenet_v3_small_head_cv/oof_predictions.csv",
    "MobileNetV3-Small (fine-tuned)": "runs/baselines_20261105_01/mobilenet_v3_small_finetune_cv/oof_predictions.csv",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: row["image_id"])
    return rows


def auc_score(y, p):
    positive, negative = y == 1, y == 0
    n_positive, n_negative = int(positive.sum()), int(negative.sum())
    if not n_positive or not n_negative:
        return float("nan")
    ranks = rankdata(p, method="average")
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) /
                 (n_positive * n_negative))


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn = int(np.sum((y == 0) & (pred == 0))); fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0))); tp = int(np.sum((y == 1) & (pred == 1)))
    matrix = np.array([[tn, fp], [fn, tp]])
    precision_0 = tn / (tn + fn) if tn + fn else 0.0
    precision_1 = tp / (tp + fp) if tp + fp else 0.0
    recall_0 = tn / (tn + fp) if tn + fp else 0.0
    recall_1 = tp / (tp + fn) if tp + fn else 0.0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if precision_0 + recall_0 else 0.0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if precision_1 + recall_1 else 0.0
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "accuracy": float(np.mean(y == pred)),
        "auc": auc_score(y, p),
        "macro_f1": (f1_0 + f1_1) / 2,
        "macro_precision": (precision_0 + precision_1) / 2,
        "macro_recall": (recall_0 + recall_1) / 2,
        "sensitivity_teflon": tp / (tp + fn),
        "specificity_nomex": tn / (tn + fp),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "confusion_matrix": matrix.tolist(),
    }


def ece(y, p, bins=10):
    pred = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1 - p)
    edges = np.linspace(0.5, 1.0, bins + 1)
    value = 0.0
    points = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence <= high if high == 1 else confidence < high)
        if not mask.any():
            continue
        accuracy = float(np.mean(pred[mask] == y[mask]))
        mean_confidence = float(np.mean(confidence[mask]))
        value += mask.mean() * abs(accuracy - mean_confidence)
        points.append({"lower": float(low), "upper": float(high), "count": int(mask.sum()),
                       "accuracy": accuracy, "confidence": mean_confidence})
    return float(value), points


def cluster_samples(groups, draws, seed):
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        yield np.concatenate([members[group] for group in chosen])


def interval(values):
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return [float(np.percentile(clean, 2.5)), float(np.percentile(clean, 97.5))]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    root, out = Path(args.root).resolve(), Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)

    paths = {name: root / relative for name, relative in MODEL_FILES.items()}
    tables = {name: load(path) for name, path in paths.items()}
    reference = tables["VGG16"]
    identity_fields = ["image_id", "y_true", "fold", "group_id", "split_sha256"]
    for name, rows in tables.items():
        if len(rows) != len(reference) or len({row["image_id"] for row in rows}) != len(rows):
            raise ValueError(f"{name}: incomplete or repeated OOF membership")
        for left, right in zip(reference, rows):
            if any(left[field] != right[field] for field in identity_fields):
                raise ValueError(f"{name}: OOF membership differs from VGG16")

    y = np.array([int(row["y_true"]) for row in reference])
    groups = np.array([row["group_id"] for row in reference])
    probabilities = {name: np.array([float(row["p_teflon"]) for row in rows])
                     for name, rows in tables.items()}
    samples = list(cluster_samples(groups, args.bootstrap, args.seed))

    model_rows, calibration = [], {}
    for name, p in probabilities.items():
        point = metrics(y, p)
        boot = {key: [] for key in ["accuracy", "auc", "macro_f1"]}
        for index in samples:
            result = metrics(y[index], p[index])
            for key in boot: boot[key].append(result[key])
        calibration_error, bins = ece(y, p)
        calibration[name] = bins
        row = {"model": name, "samples": len(y), "groups": len(np.unique(groups)),
               **{key: point[key] for key in point if key != "confusion_matrix"},
               "ece_10": calibration_error,
               "accuracy_ci_low": interval(boot["accuracy"])[0],
               "accuracy_ci_high": interval(boot["accuracy"])[1],
               "auc_ci_low": interval(boot["auc"])[0], "auc_ci_high": interval(boot["auc"])[1],
               "macro_f1_ci_low": interval(boot["macro_f1"])[0],
               "macro_f1_ci_high": interval(boot["macro_f1"])[1],
               "confusion_matrix": json.dumps(point["confusion_matrix"])}
        model_rows.append(row)

    pair_rows = []
    names = list(probabilities)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1:]:
            pa, pb = probabilities[first], probabilities[second]
            ca, cb = (pa >= .5) == y, (pb >= .5) == y
            first_only, second_only = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
            discordant = first_only + second_only
            p_value = float(binomtest(first_only, discordant, .5).pvalue) if discordant else 1.0
            acc_diff, auc_diff = [], []
            for index in samples:
                acc_diff.append(np.mean(y[index] == (pa[index] >= .5)) -
                                np.mean(y[index] == (pb[index] >= .5)))
                auc_diff.append(auc_score(y[index], pa[index]) - auc_score(y[index], pb[index]))
            pair_rows.append({"model_a": first, "model_b": second,
                              "accuracy_difference_a_minus_b": np.mean(y == (pa >= .5))-np.mean(y == (pb >= .5)),
                              "accuracy_difference_ci_low": interval(acc_diff)[0],
                              "accuracy_difference_ci_high": interval(acc_diff)[1],
                              "auc_difference_a_minus_b": auc_score(y, pa)-auc_score(y, pb),
                              "auc_difference_ci_low": interval(auc_diff)[0],
                              "auc_difference_ci_high": interval(auc_diff)[1],
                              "a_correct_b_wrong": first_only, "a_wrong_b_correct": second_only,
                              "mcnemar_exact_p": p_value,
                              "mcnemar_note": "Image-level exact test; dependence may remain within group_id clusters."})

    write_csv(out / "model_metrics.csv", model_rows)
    write_csv(out / "paired_comparisons.csv", pair_rows)
    (out / "comparison.json").write_text(json.dumps({"scope":"development OOF only",
        "bootstrap":{"draws":args.bootstrap,"seed":args.seed,"cluster":"group_id"},
        "models":model_rows,"paired":pair_rows,"calibration_bins":calibration},indent=2),encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps({name:{"path":str(path),"sha256":digest(path)}
                                                       for name,path in paths.items()},indent=2),encoding="utf-8")

    ordered = sorted(model_rows, key=lambda row: row["accuracy"])
    fig, ax = plt.subplots(figsize=(9, 5))
    values = np.array([row["accuracy"] for row in ordered])
    low = values - np.array([row["accuracy_ci_low"] for row in ordered])
    high = np.array([row["accuracy_ci_high"] for row in ordered]) - values
    ax.errorbar(values, range(len(ordered)), xerr=[low, high], fmt="o", capsize=4)
    ax.set_yticks(range(len(ordered)), [row["model"] for row in ordered])
    ax.set_xlabel("Development out-of-fold accuracy")
    ax.set_title("Grouped-bootstrap 95% intervals for model accuracy")
    ax.grid(axis="x", alpha=.25); fig.tight_layout(); fig.savefig(out/"accuracy_intervals.png",dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, bins in calibration.items():
        ax.plot([item["confidence"] for item in bins],[item["accuracy"] for item in bins],marker="o",label=name)
    ax.plot([.5,1],[.5,1],linestyle="--",color="black",label="Perfect calibration")
    ax.set(xlabel="Mean predicted-class confidence",ylabel="Observed accuracy",
           title="Development out-of-fold reliability diagram",xlim=(.5,1),ylim=(.5,1))
    ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/"calibration.png",dpi=180); plt.close(fig)

    lines = ["# Development OOF model comparison", "",
             "Scope: paired out-of-fold predictions on the common 1,280-image development set. "
             "Intervals use group_id-clustered bootstrap resampling. These are development results, not locked-holdout estimates.", "",
             "## Files", "", "- `model_metrics.csv`: point estimates, calibration, and grouped 95% intervals.",
             "- `paired_comparisons.csv`: paired differences and exact McNemar calculations.",
             "- `comparison.json`: machine-readable complete output.",
             "- `provenance.json`: input paths and SHA-256 hashes.",
             "- `accuracy_intervals.png` and `calibration.png`: manuscript-ready diagnostic figures.", "",
             "McNemar p-values use image-level discordances and can retain within-group dependence; clustered bootstrap differences are the primary uncertainty summary."]
    (out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
