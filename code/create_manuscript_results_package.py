"""Create manuscript-ready tables, figures, captions, and results prose."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


DISPLAY = {
    "VGG16": "VGG16",
    "ResNet50 (corrected fine-tuned)": "ResNet50\n(corrected fine-tuned)",
    "Small CNN": "Small CNN",
    "MobileNetV3-Small (frozen)": "MobileNetV3-Small\n(frozen)",
    "MobileNetV3-Small (fine-tuned)": "MobileNetV3-Small\n(fine-tuned)",
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save_figure(fig, folder, stem):
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(folder / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def roc_points(y, p):
    order = np.argsort(-p, kind="mergesort")
    y = y[order]; p = p[order]
    positives, negatives = max(1, int((y == 1).sum())), max(1, int((y == 0).sum()))
    tp, fp = np.cumsum(y == 1), np.cumsum(y == 0)
    distinct = np.r_[np.where(np.diff(p))[0], len(p) - 1]
    return np.r_[0, fp[distinct] / negatives], np.r_[0, tp[distinct] / positives]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--comparison-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root, comparison, out = map(lambda value: Path(value).resolve(),
                                [args.root, args.comparison_dir, args.output_dir])
    out.mkdir(parents=True, exist_ok=False)
    figures = out / "figures"; tables = out / "tables"
    figures.mkdir(); tables.mkdir()

    metrics = read_csv(comparison / "model_metrics.csv")
    paired = read_csv(comparison / "paired_comparisons.csv")
    by_model = {row["model"]: row for row in metrics}
    order = ["ResNet50 (corrected fine-tuned)", "VGG16",
             "MobileNetV3-Small (fine-tuned)", "MobileNetV3-Small (frozen)", "Small CNN"]
    prediction_paths = {
        "VGG16": root / "runs/primary_png_20261104_01/vgg16_cv/oof_predictions.csv",
        "ResNet50 (corrected fine-tuned)": root / "runs/resnet50_corrected_20261105_01/resnet50_finetune_cv/oof_predictions.csv",
        "Small CNN": root / "runs/baselines_20261105_01/small_cnn_cv/oof_predictions.csv",
        "MobileNetV3-Small (frozen)": root / "runs/baselines_20261105_01/mobilenet_v3_small_head_cv/oof_predictions.csv",
        "MobileNetV3-Small (fine-tuned)": root / "runs/baselines_20261105_01/mobilenet_v3_small_finetune_cv/oof_predictions.csv",
    }

    cv_rows = []
    for name in order:
        row = by_model[name]
        cv_rows.append({
            "Model": name,
            "Accuracy (%)": f"{100*float(row['accuracy']):.2f}",
            "Grouped 95% CI (%)": f"{100*float(row['accuracy_ci_low']):.2f}-{100*float(row['accuracy_ci_high']):.2f}",
            "ROC-AUC": f"{float(row['auc']):.4f}",
            "Macro precision (%)": f"{100*float(row['macro_precision']):.2f}",
            "Macro recall (%)": f"{100*float(row['macro_recall']):.2f}",
            "Macro F1 (%)": f"{100*float(row['macro_f1']):.2f}",
            "Nomex specificity (%)": f"{100*float(row['specificity_nomex']):.2f}",
            "Teflon sensitivity (%)": f"{100*float(row['sensitivity_teflon']):.2f}",
            "Brier score": f"{float(row['brier']):.4f}",
            "ECE (10 bins)": f"{float(row['ece_10']):.4f}",
        })
    write_csv(tables / "table_1_development_cv.csv", cv_rows)

    holdout_summary = json.loads((root / "runs/primary_png_20261104_01/both_models_holdout_summary.json").read_text())
    corrected_resnet_holdout = json.loads((root / "runs/resnet50_corrected_20261105_01/resnet50_finetune_final/holdout_evaluation/metrics.json").read_text())
    holdout_rows = []
    for name, source in [("VGG16", holdout_summary["vgg16"]),
                         ("ResNet50 (corrected fine-tuned)", corrected_resnet_holdout)]:
        item = source["image"]
        interval = source["recording_accuracy_cluster_bootstrap_95_percentile"]
        holdout_rows.append({"Model": name, "Samples": item["samples"],
            "Accuracy (%)": f"{100*item['accuracy']:.2f}",
            "Grouped 95% CI (%)": f"{100*interval['lower']:.2f}-{100*interval['upper']:.2f}",
            "ROC-AUC": f"{item['auc']:.4f}", "Macro precision (%)": f"{100*item['macro']['precision']:.2f}",
            "Macro recall (%)": f"{100*item['macro']['recall']:.2f}",
            "Macro F1 (%)": f"{100*item['macro']['f1']:.2f}",
            "Confusion matrix": json.dumps(item["confusion_matrix"])})
    write_csv(tables / "table_2_locked_holdout_completed_models.csv", holdout_rows)

    historical_rows = [
        {"Model":"VGG16","Historical reported accuracy (%)":"98.20","Corrected CV accuracy (%)":"99.06",
         "Corrected holdout accuracy (%)":"100.00","Comparability":"Historical predictions and true fold membership unavailable; descriptive only"},
        {"Model":"ResNet50","Historical reported accuracy (%)":"97.03","Corrected CV accuracy (%)":"99.45",
         "Corrected holdout accuracy (%)":"99.38","Comparability":"Historical prediction-level evidence unavailable; descriptive only"},
    ]
    write_csv(tables / "table_3_historical_vs_corrected_descriptive.csv", historical_rows)

    mobile_pair = next(row for row in paired if row["model_a"] == "VGG16" and
                       row["model_b"] == "MobileNetV3-Small (fine-tuned)")
    selected_pairs = []
    for row in paired:
        if row["model_a"] == "VGG16" or (row["model_a"] == "MobileNetV3-Small (frozen)" and
                                           row["model_b"] == "MobileNetV3-Small (fine-tuned)"):
            selected_pairs.append(row)
    write_csv(tables / "table_4_paired_comparisons.csv", selected_pairs)

    # Figure 1: accuracy with grouped intervals.
    fig, ax = plt.subplots(figsize=(10, 5.8))
    values = np.array([100*float(by_model[name]["accuracy"]) for name in order])
    lows = np.array([100*float(by_model[name]["accuracy_ci_low"]) for name in order])
    highs = np.array([100*float(by_model[name]["accuracy_ci_high"]) for name in order])
    yloc = np.arange(len(order))
    ax.barh(yloc, values, color=["#355C7D", "#4C956C", "#8AB17D", "#B56576", "#D4A373"])
    ax.errorbar(values, yloc, xerr=[values-lows, highs-values], fmt="none", ecolor="black", capsize=4)
    ax.set_yticks(yloc, [DISPLAY[name] for name in order]); ax.invert_yaxis()
    ax.set_xlabel("Development out-of-fold accuracy (%)"); ax.set_xlim(70, 101)
    ax.set_title("Model accuracy under identical grouped five-fold membership")
    ax.grid(axis="x", alpha=.25)
    for i, value in enumerate(values): ax.text(value+.25, i, f"{value:.2f}", va="center", fontsize=9)
    save_figure(fig, figures, "figure_1_cv_accuracy_with_grouped_ci")

    # Figure 2: ROC curves.
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for name in order:
        rows = sorted(read_csv(prediction_paths[name]), key=lambda row: row["image_id"])
        y = np.array([int(row["y_true"]) for row in rows]); p = np.array([float(row["p_teflon"]) for row in rows])
        fpr, tpr = roc_points(y, p)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC {float(by_model[name]['auc']):.4f})")
    ax.plot([0,1],[0,1],"k--",linewidth=1,label="Chance")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate",
           title="Development out-of-fold ROC curves", xlim=(0,1), ylim=(0,1.01))
    ax.grid(alpha=.2); ax.legend(fontsize=8, loc="lower right")
    save_figure(fig, figures, "figure_2_oof_roc_curves")

    # Figure 3: aggregate confusion matrices.
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4)); axes = axes.ravel()
    for ax, name in zip(axes, order):
        matrix = np.array(json.loads(by_model[name]["confusion_matrix"]))
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=640)
        for (i,j), value in np.ndenumerate(matrix):
            ax.text(j,i,str(value),ha="center",va="center",fontsize=12,
                    color="white" if value>350 else "black")
        ax.set_xticks([0,1],["Nomex","Teflon"]); ax.set_yticks([0,1],["Nomex","Teflon"])
        ax.set_xlabel("Predicted class", fontsize=9); ax.set_ylabel("True class", fontsize=9); ax.set_title(DISPLAY[name], fontsize=11)
        ax.tick_params(labelsize=9)
    axes[-1].axis("off")
    fig.suptitle("Development out-of-fold confusion matrices",fontsize=14, y=.98)
    fig.subplots_adjust(left=.08, right=.97, bottom=.08, top=.89, wspace=.42, hspace=.48)
    save_figure(fig, figures, "figure_3_oof_confusion_matrices")

    # Figure 4: metric comparison.
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(order)); width=.24
    for offset, key, label in [(-width,"accuracy","Accuracy"),(0,"macro_f1","Macro F1"),(width,"auc","ROC-AUC")]:
        ax.bar(x+offset,[100*float(by_model[name][key]) for name in order],width,label=label)
    ax.set_xticks(x,[DISPLAY[name] for name in order],rotation=12,ha="right")
    ax.set_ylabel("Performance (%)"); ax.set_ylim(70,101); ax.set_title("Accuracy, macro F1 and ROC-AUC comparison")
    ax.legend(); ax.grid(axis="y",alpha=.25)
    save_figure(fig, figures, "figure_4_accuracy_f1_auc_comparison")

    # Figure 5: MobileNet ablation.
    fig, ax = plt.subplots(figsize=(7, 5))
    mobile_names=["MobileNetV3-Small (frozen)","MobileNetV3-Small (fine-tuned)"]
    mobile_values=[100*float(by_model[name]["accuracy"]) for name in mobile_names]
    bars=ax.bar(["Frozen backbone","Late-layer fine-tuning"],mobile_values,color=["#8AB17D","#4C956C"])
    ax.set_ylim(90,100);ax.set_ylabel("Development OOF accuracy (%)")
    ax.set_title("Effect of MobileNetV3-Small fine-tuning")
    for bar,value in zip(bars,mobile_values):ax.text(bar.get_x()+bar.get_width()/2,value+.12,f"{value:.2f}%",ha="center")
    ax.text(.5,91,f"Improvement: {mobile_values[1]-mobile_values[0]:.2f} percentage points",ha="center")
    ax.grid(axis="y",alpha=.25)
    save_figure(fig, figures, "figure_5_mobilenet_finetuning_ablation")

    # Figure 6: reliability diagram.
    fig, ax = plt.subplots(figsize=(7.2,6.2)); edges=np.linspace(.5,1,11)
    for name in order:
        rows=sorted(read_csv(prediction_paths[name]),key=lambda row:row["image_id"])
        y=np.array([int(row["y_true"]) for row in rows]);p=np.array([float(row["p_teflon"]) for row in rows])
        pred=p>=.5;confidence=np.maximum(p,1-p);xs=[];ys=[]
        for low,high in zip(edges[:-1],edges[1:]):
            mask=(confidence>=low)&(confidence<high if high<1 else confidence<=high)
            if mask.any():xs.append(confidence[mask].mean());ys.append((pred[mask]==y[mask]).mean())
        ax.plot(xs,ys,marker="o",label=f"{name} (ECE {float(by_model[name]['ece_10']):.3f})")
    ax.plot([.5,1],[.5,1],"k--",label="Perfect calibration")
    ax.set(xlabel="Mean predicted-class confidence",ylabel="Observed accuracy",
           title="Development out-of-fold reliability diagram",xlim=(.5,1),ylim=(.5,1))
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    save_figure(fig, figures, "figure_6_calibration_reliability")

    # Figure 7: fold learning curves, validation accuracy.
    run_dirs={
        "VGG16":root/"runs/primary_png_20261104_01/vgg16_cv",
        "ResNet50 corrected":root/"runs/resnet50_corrected_20261105_01/resnet50_finetune_cv",
        "Small CNN":root/"runs/baselines_20261105_01/small_cnn_cv",
        "MobileNet frozen":root/"runs/baselines_20261105_01/mobilenet_v3_small_head_cv",
        "MobileNet fine-tuned":root/"runs/baselines_20261105_01/mobilenet_v3_small_finetune_cv",
    }
    fig,axes=plt.subplots(2,3,figsize=(13,9.2));axes=axes.ravel()
    for ax,(name,directory) in zip(axes,run_dirs.items()):
        for fold in range(1,6):
            rows=read_csv(directory/f"fold_{fold}/training_log.csv")
            ax.plot(np.arange(1,len(rows)+1),[float(row["val_accuracy"]) for row in rows],label=f"Fold {fold}")
        ax.set(title=name,xlabel="Epoch",ylabel="Validation accuracy",ylim=(.45,1.01));ax.grid(alpha=.2)
        ax.title.set_fontsize(11); ax.xaxis.label.set_fontsize(9); ax.yaxis.label.set_fontsize(9); ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7,ncol=2);axes[-1].axis("off")
    fig.suptitle("Fold-specific validation-accuracy learning curves",fontsize=14, y=.98)
    fig.subplots_adjust(left=.07, right=.98, bottom=.08, top=.90, wspace=.28, hspace=.42)
    save_figure(fig,figures,"figure_7_fold_learning_curves")

    # Figure 8: corrected evaluation workflow.
    fig,ax=plt.subplots(figsize=(13,4.8));ax.set_xlim(0,13.8);ax.set_ylim(0,5);ax.axis("off")
    nodes=[(0.3,2.0,1.8,1.1,"1,600 supplied\nscalogram PNGs"),(2.6,2.0,1.8,1.1,"Grouped split\nfrom fixed mapping"),
           (4.9,3.15,2.0,1.1,"Development set\n1,280 images"),(4.9,.85,2.0,1.1,"Locked holdout\n320 images"),
           (7.4,3.15,2.0,1.1,"Five-fold CV\ntraining-only augmentation"),(9.9,3.15,2.0,1.1,"Configuration selection\nfrom development\nevidence"),
           (9.9,.85,2.0,1.1,"Selected model final fit\nwith internal validation"),(12.15,.85,1.35,1.1,"One-time locked\nholdout evaluation")]
    boxes=[]
    for x,y,w,h,text in nodes:
        patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04",facecolor="#EAF2F8",edgecolor="#355C7D")
        ax.add_patch(patch);ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=8);boxes.append((x,y,w,h))
    def arrow(a,b):
        x,y,w,h=boxes[a];x2,y2,w2,h2=boxes[b]
        ax.annotate("",xy=(x2,y2+h2/2),xytext=(x+w,y+h/2),arrowprops=dict(arrowstyle="->",lw=1.5,color="#333333"))
    arrow(0,1);arrow(1,2);arrow(1,3);arrow(2,4);arrow(4,5)
    ax.annotate("",xy=(10.9,1.95),xytext=(10.9,3.15),arrowprops=dict(arrowstyle="->",lw=1.5,color="#333333"))
    arrow(6,7)
    ax.text(6.9,4.55,"Corrected leakage-controlled model-development and evaluation workflow",ha="center",fontsize=14)
    ax.text(6.9,.15,"Augmentation is applied only to each fold's training membership; validation and holdout images remain unaugmented.",ha="center",fontsize=9)
    save_figure(fig,figures,"figure_8_corrected_experimental_workflow")

    # Figure 9: historical comparison, explicitly descriptive.
    fig,ax=plt.subplots(figsize=(8,5.2));x=np.arange(2);width=.24
    old=np.array([98.20,97.03]);cv=np.array([99.0625,99.453125]);hold=np.array([100,99.375])
    ax.bar(x-width,old,width,label="Historical reported",hatch="//",color="#BDBDBD")
    ax.bar(x,cv,width,label="Corrected development CV",color="#355C7D")
    ax.bar(x+width,hold,width,label="Corrected locked holdout",color="#4C956C")
    ax.set_xticks(x,["VGG16","ResNet50"]);ax.set_ylabel("Accuracy (%)");ax.set_ylim(70,102)
    ax.set_title("Historical and corrected results (descriptive; protocols differ)")
    ax.legend();ax.grid(axis="y",alpha=.2)
    save_figure(fig,figures,"figure_9_historical_vs_corrected_descriptive")

    # Markdown tables.
    def md_table(rows):
        keys=list(rows[0]); lines=["| "+" | ".join(keys)+" |","|"+"|".join(["---"]*len(keys))+"|"]
        lines += ["| "+" | ".join(str(row[key]) for key in keys)+" |" for row in rows]
        return "\n".join(lines)
    (out/"MANUSCRIPT_TABLES.md").write_text("# Manuscript tables\n\n## Table 1. Common development-CV comparison\n\n"+
        md_table(cv_rows)+"\n\n## Table 2. Completed locked-holdout evaluations\n\n"+md_table(holdout_rows)+
        "\n\n## Table 3. Historical versus corrected results\n\n"+md_table(historical_rows)+"\n",encoding="utf-8")

    captions = """# Figure captions

**Figure 1.** Development out-of-fold accuracy for five CNN configurations evaluated using identical grouped five-fold membership. Error bars represent 95% percentile intervals from 5,000 bootstrap draws clustered by `group_id`.

**Figure 2.** Receiver operating characteristic curves computed from the complete out-of-fold prediction probabilities of the common 1,280-image development set.

**Figure 3.** Aggregate out-of-fold confusion matrices. Rows denote true material class and columns denote predicted material class.

**Figure 4.** Comparison of accuracy, macro-averaged F1-score and ROC-AUC under the common development protocol.

**Figure 5.** MobileNetV3-Small ablation comparing a frozen ImageNet backbone with controlled fine-tuning of six late weighted layers while Batch Normalization layers remained frozen.

**Figure 6.** Reliability diagram comparing predicted-class confidence with observed accuracy. Expected calibration error (ECE) was calculated using 10 equal-width confidence bins.

**Figure 7.** Fold-specific validation-accuracy learning curves for all evaluated configurations. Different stopping epochs reflect validation-loss early stopping or the prespecified maximum epoch count.

**Figure 8.** Corrected experimental workflow. Grouped development folds were separated before augmentation; augmentation was restricted to training membership, and the locked holdout was reserved for final evaluation.

**Figure 9.** Descriptive comparison of historical reported accuracies and corrected results. Historical and corrected values were obtained using different validation procedures and must not be interpreted as a controlled before-after experiment.
"""
    (out/"FIGURE_CAPTIONS.md").write_text(captions,encoding="utf-8")

    writeup = f"""# Draft results write-up

## Comparative classification performance

VGG16, corrected fine-tuned ResNet50, a small CNN baseline and two MobileNetV3-Small configurations were compared under a common grouped five-fold development protocol. All models used the same 1,280 development images and identical fold membership. Augmentation was restricted to each fold's training membership, while validation images remained unaugmented. Complete out-of-fold probabilities were retained for classification, ROC-AUC, calibration and paired analyses.

Corrected fine-tuned ResNet50 produced the highest development point estimate: 99.45% accuracy (grouped-bootstrap 95% CI: 99.01-99.84%), ROC-AUC 1.0000 and macro F1 99.45%. VGG16 achieved 99.06% accuracy (95% CI: 98.44-99.60%), ROC-AUC 0.9995 and macro F1 99.06%. Fine-tuned and frozen MobileNetV3-Small achieved 96.72% and 95.31% accuracy, respectively. The small CNN achieved 78.28% accuracy and ROC-AUC 0.8914, showing that near-perfect classification was not obtained automatically from the supplied scalogram images.

Corrected ResNet50 correctly classified all 640 Nomex images and 633 of 640 Teflon images. VGG16 correctly classified 637 Nomex and 631 Teflon images. Fine-tuned MobileNetV3-Small correctly classified 620 Nomex and 618 Teflon images.

## Paired comparison and uncertainty

Uncertainty intervals used 5,000 bootstrap resamples clustered by `group_id`. Corrected ResNet50 exceeded VGG16 by 0.39 percentage points, but the grouped 95% accuracy-difference interval included zero (-0.23 to 1.03 percentage points for ResNet50 minus VGG16). The secondary exact image-level McNemar test gave p=0.302. The models are therefore described as having comparable accuracy, not statistically established accuracy superiority. VGG16 exceeded fine-tuned MobileNetV3-Small by {100*float(mobile_pair['accuracy_difference_a_minus_b']):.2f} percentage points (grouped 95% interval: {100*float(mobile_pair['accuracy_difference_ci_low']):.2f} to {100*float(mobile_pair['accuracy_difference_ci_high']):.2f}).

## Locked-holdout evaluation

VGG16 and corrected fine-tuned ResNet50 were fitted using the development data with an internal grouped validation subset and evaluated once on the locked 320-image holdout. VGG16 classified all 320 images correctly (100% accuracy; ROC-AUC 1.0000). Corrected ResNet50 classified 318 of 320 images correctly (99.38% accuracy; ROC-AUC 1.0000). These internal-holdout results do not establish equivalent performance under a new acquisition session, sensor configuration, voltage condition or laboratory environment.

## Interpretation relative to the historical manuscript

Historical accuracies of 98.20% for VGG16 and 97.03% for ResNet50 are reported separately because the original prediction probabilities, exact fold membership and recording-independent evaluation cannot be completely reconstructed. The corrected experiments used explicit grouped membership, training-only augmentation, saved OOF probabilities and a locked holdout. Historical-versus-corrected comparisons are descriptive rather than controlled estimates of improvement.

## Reporting limitation

Separation is verified according to the supplied image mapping and `group_id` values. Source CSV paths, acquisition-session identifiers, calibrated time/frequency axes and complete wavelet-generation provenance are unavailable for every PNG. Performance should therefore be described as Nomex-versus-Teflon scalogram classification within the supplied dataset. Broader claims require acquisition-independent measurements.
"""
    (out/"MANUSCRIPT_RESULTS_WRITEUP.md").write_text(writeup,encoding="utf-8")
    index = """# Manuscript results package

The `figures` folder contains PNG (300 dpi), PDF and SVG versions of every figure. The `tables` folder contains editable CSV tables. `MANUSCRIPT_RESULTS_WRITEUP.md` provides draft results text, and `FIGURE_CAPTIONS.md` provides captions.

Development-CV results and locked-holdout results are intentionally separated. MobileNetV3-Small and small-CNN holdout evaluations are pending and are not fabricated in this package.
"""
    (out/"README.md").write_text(index,encoding="utf-8")


if __name__ == "__main__":
    main()
