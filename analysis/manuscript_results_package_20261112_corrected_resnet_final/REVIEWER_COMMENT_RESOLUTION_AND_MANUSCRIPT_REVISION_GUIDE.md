# Reviewer Comment Resolution and Manuscript Revision Guide

## Purpose

This guide maps the technical reviewer comments to the corrected experiments and identifies the manuscript sections that must be rewritten. It distinguishes between issues that are **RESOLVED**, **PARTIALLY RESOLVED**, and **UNRESOLVED**. A resolved status means that saved code and experimental artifacts directly support a response; it does not mean that every broader scientific concern has been eliminated.

The present corrected evidence consists of a fixed image mapping, grouped development/holdout separation, identical grouped five-fold membership across models, training-only augmentation, saved out-of-fold (OOF) probabilities, corrected transfer-learning experiments, grouped-bootstrap uncertainty intervals and a one-time locked internal-holdout evaluation.

## Reviewer-to-evidence resolution matrix

| Reviewer concern | Status | What the new work establishes | Required manuscript action |
|---|---|---|---|
| Reviewer 5-IV: Fold 1 missing; four-fold/five-fold and 80/20 protocols are inconsistent | **RESOLVED** | All 1,280 development images have one OOF prediction under a common grouped five-fold protocol. A separate 320-image locked holdout was evaluated after development. | Replace every description of four-fold CV, missing Fold 1 and the old mixed split procedure with one consistent protocol. Describe the 80/20 division only as development/locked-holdout separation, followed by five-fold CV within development. |
| Reviewer 5-V: confidence intervals based on four dependent folds are unreliable | **RESOLVED for the new analysis** | Accuracy, AUC and macro-F1 intervals were calculated from 5,000 bootstrap resamples clustered by `group_id`; paired differences use the same clustered resamples. Exact McNemar tests are secondary because image-level dependence may remain. | Remove fold-mean t intervals and the claim that narrow four-fold intervals prove reliability. Report grouped-bootstrap intervals and explicitly state their conditional/internal-dataset scope. |
| Reviewer 1 and Reviewer 5-VIII: no simpler or modern model baselines | **RESOLVED** | Small CNN, frozen MobileNetV3-Small and fine-tuned MobileNetV3-Small were evaluated using identical folds. Corrected fine-tuned ResNet50 and VGG16 are also compared under the same protocol. | Expand Methods with baseline definitions. Replace the two-model-only results with the five-configuration comparison table and Figures 1-7 from this package. |
| Reviewer 1: deep networks may overfit the limited dataset | **PARTIALLY RESOLVED** | Training-only augmentation, early stopping, regularization, frozen Batch Normalization during controlled fine-tuning, grouped CV, learning curves and a locked holdout reduce and expose overfitting risk. The small-CNN baseline performs much worse, while VGG16 and corrected ResNet generalize consistently within the supplied dataset. | Report safeguards and learning curves, but do not claim that overfitting has been eliminated. Retain limited sample size and lack of external acquisition validation as limitations. |
| Reviewer 2-5: augmentation techniques are unclear | **RESOLVED for implementation/reporting** | The corrected pipeline records the actual augmentations and applies them only to training membership after splitting. Validation and holdout data are not augmented. Current PNG-space augmentation includes realistic intensity/contrast and controlled time/frequency masking or time-axis shifting as configured; horizontal flips and rotations are excluded from the corrected protocol. | Delete the manuscript statements claiming horizontal flip and +/-10-degree rotation. Describe only augmentations actually enabled in the corrected run, their parameter ranges from the saved configuration, and their training-only placement. Do not describe PNG operations as waveform-domain augmentation. |
| Reviewer 5-IX: horizontal flipping and rotation may be physically invalid | **RESOLVED in the corrected training protocol** | These transformations are not part of the corrected experiment. | State that time reversal and geometric rotation were excluded because they alter the physical interpretation of time-frequency axes. |
| Reviewer 5-II: class-dependent coefficient selection may cause label leakage | **RESOLVED BY FIXED PREPROCESSING DESCRIPTION, SUBJECT TO FINAL COEFFICIENT VERIFICATION** | The author confirms that, after the earlier exploratory COMPEL study, the classification scalograms were generated using a fixed db3 configuration. The new manuscript will not use the old class-dependent maximum-energy narrative. The same wavelet and detail-coefficient construction rule must be documented for Nomex and Teflon. | State that db3 was fixed before classification and applied independently of the material label. Verify and report the exact common detail level or common multi-level construction used for the final PNGs. Do not describe a rule that chooses a level after observing the class. |
| Reviewer 5-III: 400 signals, four/eight scalograms and 1,600-image counts conflict | **PARTIALLY RESOLVED** | The corrected classifier uses 1,600 supplied PNGs: 1,280 development and 320 holdout. Each image has a fixed mapping and grouped identity used for partitioning. The historical raw-CSV-to-PNG generation chain is not fully recoverable from the supplied evidence. | Replace contradictory four/eight-scalogram derivations with a dataset accounting table based only on verified files. Separate “400 historically described acquisitions” from “1,600 supplied images” unless raw-source linkage is proven. Do not invent a one-to-four mapping. |
| Reviewer 1, Reviewer 2-2/3/4 and Reviewer 5-X: wavelet/entropy and decomposition-level selection are not demonstrated | **RESOLVED BY NARROWING THE CLAIM; OPTIMALITY IS NOT CLAIMED** | The earlier COMPEL study supplies a physical signal-processing rationale, sampling frequency, nominal dyadic bands and exploratory db2/sym2 energy observations. The final classification scalograms were subsequently generated using fixed db3 preprocessing. Neither entropy nor maximum coefficient energy is treated as proof of classification optimality. | Retain a concise 2500-MHz sampling/Nyquist and nominal-band calculation to justify transient-focused detail coefficients. Present db3 and the decomposition rule as predetermined, label-independent preprocessing. Remove entropy/maximum-energy optimization claims. A wavelet/level ablation is needed only if the paper wishes to claim best or optimal preprocessing. |
| Reviewer 2-6: performance on noise-corrupted scalograms | **UNRESOLVED** | Training augmentation is not a substitute for a separate robustness test. | Do not claim noise robustness. Add a controlled noise/masking/intensity perturbation test or list noise robustness as future work. |
| Reviewer 2-8: explainability section | **RESOLVED WITH EXPLICIT LIMITATIONS** | True-class Grad-CAM was calculated for the complete 320-image locked holdout for both final models. The analysis includes deterministic overlays, translation stability, attended-versus-random masking, clustered intervals and every ResNet50 error. High Teflon zero/flat-map rates are reported explicitly. | Add the explainability Methods, Results, figures and limitations from `analysis/gradcam_final_20261112/manuscript_package`. Describe the maps as image-space model-attention diagnostics, not physical causality or calibrated RF-band evidence. |
| Reviewer 5-VII: acquisition/electrode/voltage/session confounding | **UNRESOLVED** | Grouped splitting prevents overlap according to `group_id`, but available metadata do not establish independence across acquisition session, voltage, sensor position or electrode geometry. | Add this explicitly to Limitations. Avoid claiming material-intrinsic or cross-laboratory generalization. A metadata-stratified or leave-one-condition/session-out experiment is required to resolve it. |
| Reviewer 5-VI: task is overstated as PD detection/defect/severity diagnosis | **RESOLVED by claim correction, not by a new experiment** | The task is binary Nomex-versus-Teflon classification of supplied PD scalogram images. | Replace “PD detection,” “fault identification,” “severity” and broad diagnostic claims with “insulation-material classification from supplied UHF PD scalograms.” |
| Reviewers 1, 4 and 5-I: novelty is overstated | **RESOLVED by positioning, not experimental proof of uniqueness** | The new contribution is a leakage-controlled comparative evaluation with reproducible predictions and uncertainty analysis, not a unique wavelet-CNN combination. | Remove “unique,” “novel combination” and “first.” Position the work as an experimentally validated comparison for Nomex/Teflon scalograms under a corrected evaluation protocol. |
| Reviewer 1: classical methods may solve an easy problem | **PARTIALLY RESOLVED** | A small CNN baseline is included, but no hand-crafted-feature SVM, random forest or logistic-regression baseline has been tested. | Treat the small CNN as a neural baseline. If responding fully, add at least one classical baseline built without leakage; otherwise acknowledge that classical feature-based comparison remains future work. |
| Reviewer 2-7: UHF preprocessing is unclear | **PARTIALLY RESOLVED** | The classifier-side resizing, normalization/preprocessing and augmentation are recorded. Raw waveform preprocessing cannot be completely reconstructed for every PNG. | Separate waveform-to-scalogram preprocessing from model-input preprocessing. Report only verified steps and identify unavailable raw-generation provenance as a limitation. |

## New model comparison to add

Use the following development-OOF results as the principal model-comparison table. All models used the same 1,280 images and identical grouped five-fold membership.

| Model | Accuracy | Grouped 95% CI | ROC-AUC | Macro F1 |
|---|---:|---:|---:|---:|
| ResNet50, corrected fine-tuned | 99.45% | 99.01-99.84% | 0.99998 | 99.45% |
| VGG16 | 99.06% | 98.44-99.60% | 0.99952 | 99.06% |
| MobileNetV3-Small, fine-tuned | 96.72% | 95.60-97.79% | 0.99470 | 96.72% |
| MobileNetV3-Small, frozen | 95.31% | 93.99-96.59% | 0.99188 | 95.31% |
| Small CNN | 78.28% | 75.81-80.70% | 0.89144 | 77.94% |

The corrected ResNet50-versus-VGG16 accuracy difference is small. ResNet50 exceeded VGG16 by 0.39 percentage points, but the grouped 95% difference interval included zero (-0.23 to 1.03 percentage points for ResNet50 minus VGG16), and the secondary exact McNemar test gave p=0.302. The defensible conclusion is **comparable accuracy**, not statistically established ResNet50 superiority. The AUC point estimate favored ResNet50 by a very small absolute amount.

The locked internal holdout contained 320 images. VGG16 obtained 100.00% accuracy with confusion matrix `[[160, 0], [0, 160]]`; corrected fine-tuned ResNet50 obtained 99.38% accuracy with confusion matrix `[[160, 0], [2, 158]]`. These are internal-dataset holdout results, not external or cross-session validation.

## Comparison with the present manuscript

The current manuscript must not be corrected by changing only the reported percentages. Its experimental narrative reflects a different and internally inconsistent protocol.

### Statements that must be removed or replaced

1. **Abstract:** remove “unique approach,” “most informative coefficients,” historical 98.20%/97.03% results, fold-mean confidence intervals and claims that the results confirm broad generalization.
2. **Introduction and contributions:** remove claims of novelty based solely on combining wavelets with VGG16/ResNet50. Add reproducible grouped evaluation, modern/lightweight baselines, paired statistical comparison and locked-holdout evaluation as the actual methodological contributions.
3. **Dataset section:** the manuscript currently states both four scalograms per signal and later eight scalograms per signal while still totaling 1,600. Replace this with verified supplied-image counts and a transparent provenance limitation.
4. **Wavelet Sections 3.1-3.2:** replace the long entropy/maximum-energy optimization narrative with the fixed db3 and nominal-frequency-band subsection provided below. Retain the physical reason for using transient-focused detail coefficients, but do not call db3 or any level optimal, most informative or minimum-loss. The earlier db2/sym2 analysis may be mentioned as preliminary signal characterization only.
5. **Evaluation protocol:** delete statements that Fold 1 was unavailable, that only Folds 2-5 were analyzed, and that a four-fold CV was used. Delete the fold-mean t-interval formula with `n=4`.
6. **Augmentation:** delete horizontal flip and geometric rotation. State that all augmentation occurred after fold assignment and only for training membership. List exact enabled PNG-space transformations and parameters from the corrected run configuration.
7. **Training configuration:** replace the old batch size, regularization, learning rate and trainable-layer counts with the actual saved configuration for each reported model. Do not reuse one generic setting across architectures.
8. **Results:** replace old fold tables, confusion matrices, ROC plots, learning curves and historical conclusions with the files in this package.
9. **Discussion:** remove claims that VGG16 is superior. Discuss ResNet50 and VGG16 as statistically comparable in accuracy, MobileNet as a strong lower-complexity alternative and the small CNN as evidence that architecture/transfer learning materially affects performance.
10. **Conclusion:** remove “robust,” “reliable for real-time monitoring,” “fault identification,” and broad generalization claims unless bounded to the supplied dataset. State the unresolved external-acquisition and noise-robustness limitations, and report the completed Grad-CAM analysis with its high Teflon flat-map limitation.

## Recommended rewritten manuscript structure

### Abstract

Use five elements: task and dataset scope; verified grouped evaluation protocol; compared models; corrected principal results; limitations. A suitable result sentence is:

> Under identical recording-grouped five-fold development splits, fine-tuned ResNet50 and VGG16 achieved OOF accuracies of 99.45% and 99.06%, respectively. The grouped-bootstrap interval for their paired accuracy difference included zero, indicating comparable accuracy. On a locked internal holdout, the corresponding accuracies were 99.38% and 100.00%.

End the Abstract by restricting the conclusion to the supplied Nomex/Teflon scalogram dataset.

### Introduction

- Expand related work on PD scalograms, transfer learning, leakage risks in derived-image datasets and uncertainty-aware evaluation.
- State that wavelet-CNN combinations already exist.
- Define the narrower gap: reliable comparison under source-grouped splitting with saved OOF predictions and lightweight baselines.
- Rewrite contributions as evaluation and evidence contributions rather than algorithmic uniqueness.

### Dataset and preprocessing

- Report 1,600 supplied scalogram PNGs, class balance, fixed mapping and `group_id` construction.
- Explain 1,280-image development and 320-image locked-holdout membership.
- State that all images sharing a source group remain in the same partition/fold.
- Describe waveform/scalogram provenance only where supported; explicitly label missing raw-source linkage.
- Avoid claiming that four wavelets created independent recordings.
- State that the COMPEL db2/sym2 results were preliminary exploratory signal analysis; the final classification scalograms were subsequently generated using fixed db3 preprocessing.
- Retain the 2500-MHz sampling rate only if it applies to the final recordings. State the corresponding Nyquist limit of 1250 MHz.
- Report nominal dyadic detail bands as approximate bands because db3 filters have finite transition regions rather than ideal cut-offs.
- Confirm and report the exact common detail level or common five-level construction used to produce every final PNG.
- State explicitly that the same transformation was applied before the material label was known.

### Final wavelet/scalogram positioning for the new manuscript

The wavelet stage is necessary preprocessing, not the principal optimization contribution. The new manuscript should retain the physical justification for detail coefficients and the concise frequency-band calculation, while omitting claims that Shannon entropy or maximum energy proves classification suitability.

Recommended wording:

> UHF partial-discharge signals are characterized by short-duration impulsive transients containing substantial high-frequency content. A fixed Daubechies-3 (db3) wavelet configuration was therefore used to separate the signals into approximation and detail components. The detail coefficients emphasize rapid temporal variations, whereas the approximation coefficients represent lower-frequency signal components. Detail-coefficient scalograms were consequently used as transient-focused input representations. This operation is described as band-selective preprocessing rather than complete denoising because high-frequency measurement noise or electromagnetic interference may also remain in the detail coefficients.
>
> The signals were sampled at 2500 MHz, giving a Nyquist frequency of 1250 MHz. For a dyadic decomposition, the nominal detail-coefficient interval at level j was estimated as D_j = [f_s/2^(j+1), f_s/2^j]. The corresponding nominal intervals for D1-D5 were 625-1250, 312.5-625, 156.25-312.5, 78.125-156.25 and 39.063-78.125 MHz, respectively. These intervals are approximate because the db3 analysis filters have finite transition bands.
>
> The db3 configuration and the detail-coefficient/scalogram-construction rule were fixed before classifier evaluation and applied identically to every recording, independently of its material label. The frequency-band calculation provides a physical rationale for using detail coefficients; it is not interpreted as proof that db3 or a particular level is classification-optimal.

After confirming the exact PNG-generation rule, add one of the following sentences:

- **Single common level:** “The same D_j detail level was used for every recording, irrespective of material and electrode configuration.” Replace `D_j` with the verified level.
- **Common multi-level construction:** “The scalogram was constructed from the fixed five-level db3 detail representation using an identical procedure for every recording.”

Do not include the following claims unless a controlled grouped classification ablation is performed:

- maximum Shannon entropy identifies the classification-optimal wavelet;
- the maximum-energy coefficient is necessarily the most informative classification feature;
- approximation coefficients are universally noise;
- detail coefficients contain only PD activity;
- different coefficients can be selected using the known Nomex/Teflon class at inference.

The COMPEL analysis remains useful as preliminary physical and exploratory context: it documents the laboratory setup, the 2500-MHz sampling assumption, nominal band calculations, energy distributions, coefficient statistics and representative scalograms. Its db2/sym2 observations should not be described as the preprocessing used for the final db3 classification dataset.

### Experimental protocol

- Present the five-fold grouped CV workflow before model details.
- State that every development image receives exactly one OOF prediction.
- State that augmentation is training-only.
- Describe VGG16, corrected fine-tuned ResNet50, frozen/fine-tuned MobileNetV3-Small and the small CNN separately.
- Explain early stopping, checkpoint selection, preprocessing and Batch Normalization handling using actual saved configurations.
- Describe the locked holdout as used once after development/configuration selection.

### Statistical analysis

- Report point estimates from concatenated OOF predictions.
- State that 95% intervals used 5,000 `group_id`-clustered bootstrap resamples.
- Report paired accuracy/AUC difference intervals.
- Present McNemar as a secondary image-level analysis with its dependence limitation.
- Do not calculate significance from fold means or claim that overlapping/non-overlapping marginal CIs alone prove model differences.

### Results

Recommended order:

1. Principal OOF performance table.
2. Accuracy with grouped confidence intervals.
3. OOF ROC curves.
4. OOF confusion matrices.
5. Accuracy/F1/AUC comparison.
6. MobileNet frozen-versus-fine-tuned ablation.
7. Learning curves and calibration.
8. Paired ResNet50-versus-VGG16 comparison.
9. Locked-holdout results in a separate subsection.
10. Historical values only in a clearly labelled descriptive subsection or appendix.

### Discussion

- Interpret ResNet50 and VGG16 as comparable in accuracy.
- Explain that corrected native ResNet preprocessing and controlled fine-tuning materially changed ResNet performance, without claiming that preprocessing alone caused the full difference.
- Discuss MobileNetV3-Small as a compact alternative whose performance is below the two strongest models but substantially above the small CNN.
- Use the small CNN to show that the task is not solved equally by every image classifier; do not present it as a classical machine-learning baseline.
- Separate statistical discrimination from physical interpretation.
- Avoid inferring material-specific physics until explainability and acquisition-confound analyses are completed.

### Limitations and conclusion

Include a dedicated limitations subsection covering:

- internal rather than external/cross-session holdout;
- unavailable complete raw CSV-to-PNG provenance;
- fixed db3 preprocessing is physically motivated but not claimed to be wavelet/level optimal;
- possible voltage/electrode/session/sensor confounding;
- absence of a dedicated corruption/noise test;
- high Teflon zero/flat-map rates limit class-wide Grad-CAM interpretation;
- binary material classification rather than PD presence, defect type or severity diagnosis.

The conclusion should state that the corrected workflow provides strong **within-dataset** discrimination and reproducible comparative evidence, while external physical generalization remains to be tested.

## Files to use during revision

- `tables/table_1_development_cv.csv`: principal common-protocol comparison.
- `tables/table_2_locked_holdout_completed_models.csv`: VGG16 and corrected ResNet internal-holdout results.
- `tables/table_3_historical_vs_corrected_descriptive.csv`: historical values, descriptive only.
- `tables/table_4_paired_comparisons.csv`: paired differences and secondary McNemar calculations.
- `figures/figure_1_cv_accuracy_with_grouped_ci.*`: accuracy and grouped intervals.
- `figures/figure_2_oof_roc_curves.*`: common OOF ROC curves.
- `figures/figure_3_oof_confusion_matrices.*`: common OOF confusion matrices.
- `figures/figure_4_accuracy_f1_auc_comparison.*`: metric comparison.
- `figures/figure_5_mobilenet_finetuning_ablation.*`: MobileNet ablation.
- `figures/figure_6_calibration_reliability.*`: calibration analysis.
- `figures/figure_7_fold_learning_curves.*`: fold-specific learning curves.
- `figures/figure_8_corrected_experimental_workflow.*`: corrected split/training workflow.
- `figures/figure_9_historical_vs_corrected_descriptive.*`: historical comparison with protocol warning.
- `MANUSCRIPT_RESULTS_WRITEUP.md`: draft corrected Results language.

## Recommended next actions before final submission

1. Insert the completed Grad-CAM Methods, Results, captions and limitations from `analysis/gradcam_final_20261112/manuscript_package`.
2. Run a separate controlled corruption/noise robustness analysis if responding experimentally to Reviewer 2-6.
3. Verify the exact common db3 detail level or multi-level rule used to generate the final PNGs. A raw-signal wavelet/level ablation is optional unless an optimality claim is desired.
4. Add a classical feature-based baseline if a complete response to Reviewer 1 is required.
5. Rewrite the manuscript in the order above, then perform a final consistency audit so the Abstract, Methods, Results, tables, figures and Conclusion all use the same protocol and numbers.

## Bottom-line readiness assessment

The manuscript is ready for a substantial technical rewrite using the corrected classification and explainability results. The evaluation-protocol, model-comparison, statistical-reliability and model-attention evidence are now much stronger than in the submitted version. The wavelet concern can be addressed without a new optimization experiment by reporting db3 as fixed, label-independent, physically motivated preprocessing and retaining only the concise sampling-frequency/nominal-band calculation. The manuscript is **not yet ready for final submission** if it claims wavelet optimality, external generalization, noise robustness, mechanistic physical explainability or acquisition-independent material recognition. Those claims must either be supported by additional experiments or explicitly narrowed/removed.
