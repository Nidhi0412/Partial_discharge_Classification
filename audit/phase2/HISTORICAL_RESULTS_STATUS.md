# Historical results status and disposition

Date: 2026-11-02. This is a recovery assessment, not a replacement set of experimental results. Exact path aliases below are defined in EVIDENCE_RECOVERY_REPORT.md; all machine-readable evidence files are in this same Phase 2 directory.

## VGG16, reported 98.20%

**CONFIRMED — architecture of the recovered associated checkpoint set:** VGG16. All eight June archive checkpoints contain its actual serialized convolution/pooling graph. REPORT_LINKS.json shows that their four co-located detailed reports are byte-identical to `MIN\results\VGG16\VGG16\fold_N_report.txt`. HISTORY_LINKS.json independently matches the June archive Fold 4/5 histories to `PAPER\VGG16_insulation`. The ResNet50-named archive subdirectory is misleading.

**CONFIRMED — saved-summary arithmetic:** four report values .9844/.9812/.9812/.9812 average to .9820. `PAPER\VGG16_insulation\crossvalidation_avg_classification_report.xlsx`, sheet1 H9:H13, records exactly these and their mean. I13 records precision 1.0, J13 recall .964075 and K13 F1 .9817, based on rounded report inputs. This corroborates the headline and contradicts copied VGG precision/F1 entries in the minimal README. It does not reproduce predictions.

**CONFIRMED — recovered Fold 5 sources agree:** `PAPER\VGG16_insulation\fold_5\detailed_classification_report.txt`, the same archive member and workbook H12 report .9812; `PAPER\VGG16_insulation\fold_5\confusion_matrix.jpg` visibly shows TN=160, FP=0, FN=6, TP=154, implying 314/320=.98125. No material Fold 5 discrepancy was found in recovered artifacts. Any different manuscript value must be traced to its own source. Figure transcription was inspected directly; it was not model inference.

**NOT VERIFIABLE — independent accuracy/AUC reproduction:** no original per-image probability/label files or evaluator record linking a checkpoint hash to predictions were recovered. The archive recovers usable candidate checkpoints, but existence plus report co-location is not proof of score provenance. The current .9981 AUC headline can be checked against saved summary arithmetic only.

**NOT VERIFIABLE — recording-independent generalization and five distinct effective folds:** original recording IDs and historical memberships are missing. The recovered dataset copy has exact Train/Test image duplication, but its identity with the actual historical input is unproven. The current source repeats fixed directory subsets; its exact historical use is unproven and cannot be assumed because the actual VGG graph differs from the current CLI dispatch.

**Disposition:** retain VGG checkpoints/reports as historical artifacts and document the now-confirmed architecture. Do not retain 98.20% as a validated recording-independent estimate. Candidate checkpoints permit a future, approved independent inference reevaluation if a traceable independent test set is available. Proper grouped-CV claims require a new grouped training/evaluation experiment unless historical membership evidence establishing such a protocol is recovered.

## ResNet50, reported 97.03%

**CONFIRMED — historical named config and report arithmetic:** `MIN\results\ResNet50\experiment_config.json` names `CV_Aug_20250529_210046`; `PAPER\ResNet50_insulation\crossval_summary.txt` agrees with `MIN\results\ResNet50\crossval_metrics.csv`. The four accuracies .971875/.965625/.975/.96875 average to .9703125, which rounds to 97.03%.

**CONFIRMED — supporting residual-model text exists:** `PAPER\fwdresnet50_insulation\fine_tuning_report.txt` lists 178 layers and ResNet conv5 residual-block layer names. `PAPER\fwdresnet50_insulation.zip!crossval_summary.txt` carries the matching reported scores. This supports a ResNet association, but the text does not contain a checkpoint hash uniquely tying the listed graph to the May run.

**NOT VERIFIABLE — direct checkpoint backbone verification, score reproduction and recording independence:** no May `model.h5`/`best_weights.h5`, per-image scores/labels, original commands, dated source revision or recording/fold manifest was recovered. The June VGG checkpoints must not be used as the missing May ResNet checkpoints. Available histories and logs corroborate training activity but cannot identify the exact model weights or input membership.

**Disposition:** retain the saved summaries as historical claims, not independently verified performance. Recover the actual ResNet checkpoint/predictions first. Without them the historical numerical result cannot be reproduced from this workspace; new training would be needed to produce a new model result, which must be explicitly distinguished from recovery of the old result.

## Dataset and validation claims

**CONFIRMED:** `DATA\Insulation_classifier` has 1600 files, 1280 Train and 320 Test; the four-class tree has the same complete content multiset. **CONTRADICTED:** these recovered trees are image-disjoint. Insulation has 105/320 Test files with identical bytes in Train. **NOT VERIFIABLE:** their exact historical use, 400 original acquisition records, wavelet/level lineage and recording-independent splits.

**CONFIRMED:** `PAPER\paper with hybrid results_2025.docx` paragraph 29 claims four db1–db4 images per recording. **CONTRADICTED:** paragraph 66's eight images per recording and total 1600 are consistent with 400 recordings; they would give 3200. `PAPER\vgg16insulation_results\energy_analysis.txt` recovers class-specific energy-selection recommendations, but **NOT VERIFIABLE** whether these or any Shannon-entropy rule built the historical dataset.

**NOT VERIFIABLE:** historical grouped-CV validity. The supplied current code is defective, recovered logs lack memberships, Fold 1 is absent, and evaluation scripts reuse a fixed Test directory. Repeated scoring of one Test set must not be reported as independent test folds or multiplied sample size.

## Statistical claims

**CONFIRMED:** Phase 1 reproduced summary/CI formula arithmetic from `MIN\results\statistical_analysis_results.json` and saved AUC CSV, but Phase 2 recovered no AUT probability arrays or the exact statistics-generating inputs. **NOT VERIFIABLE:** the paired numerical comparison from original predictions or any recording-level confidence interval. T-test significance of four dependent run scores cannot establish population-level superiority; the saved Wilcoxon p=.125 does not establish significance either. New valid uncertainty estimates need aligned predictions and independent recording/session units.

## Retain, reevaluate, or retrain?

1. **Retain as historical evidence:** original archives, checkpoint files, epoch logs, reports and saved arithmetic. VGG16 backbone identity is now directly confirmed for the recovered June set. Preserve the distinction between source defects and actual serialized models.
2. **Independent reevaluation is needed before using the old models for a generalization claim:** use hash-identified checkpoints and a traceable, genuinely independent test set, with acquisition-level grouping and prespecified aggregation. This is possible for the recovered VGG candidates once approved and equipped; it is blocked for the May ResNet run until checkpoints are recovered. Reevaluating the same contaminated copy may reproduce old scores but cannot validate independence.
3. **Complete fresh training/evaluation is required if the recovered leaky partitions were the original inputs, or if valid grouped-CV results are needed and no effective historical group split can be established.** Grouping must follow recovered acquisition lineage; a filename/hash-only deduplication cannot certify independence. Complete retraining of every existing model is not proven inevitable solely by this incomplete recovery. An existing VGG model could still be tested on a new independent acquisition set, which would yield a new external-evaluation result rather than validate the old CV protocol.

**Final decision:** neither 98.20% nor 97.03% currently qualifies as independently verified, recording-independent AUT performance. The evidence justifies retention of the historical artifacts, mandatory independent reevaluation for defensible claims, and conditional fresh grouped training once provenance and the intended validation protocol are established. No original result was silently corrected or replaced.
