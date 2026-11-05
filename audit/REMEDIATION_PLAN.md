# Prioritized remediation plan

No remediation below was applied to original research code or data. New processing, model inference and training remain outside this read-only stage.

## 1. Immediate evidence recovery — P0

Recover the exact source revision, console logs, complete run trees and input/output artifacts listed in CODE_AUDIT_REPORT.md. Establish how five db3 positional scalograms became the final 1600 images. Join raw acquisition metadata to each original recording before assigning derived images. Qualify duplicate bare filenames with session and source folder. If lineage cannot be recovered, report that historical leakage cannot be excluded rather than inventing IDs.

Populate DATASET_PROVENANCE.csv with actual rows; replace the missing-evidence marker. Save source and image SHA256 hashes. Count original recordings, derived variants, class/configuration groups and partition membership separately. Detect exact hash duplicates across partitions, then inspect perceptual/near-duplicate candidates with their source records; similarity alone is not proof of common acquisition. Investigate voltage/electrode/session associations with material labels.

## 2. Code/documentation-only work — no new result claims

After review, correct coefficient nomenclature, magnitude/energy descriptions, actual augmentation descriptions, binary-positive versus macro metrics, four-available-run language and duplicated VGG metric summaries. Remove unsupported statistical superiority/physical-mechanism claims. Keep a record of every historical number and its source rather than silently replacing published results.

Repair syntax and missing imports/dependency declarations, remove duplicate generator definitions, fix launcher task/model paths and lowercase/uppercase model dispatch, and make failed folds explicit. Recover actual backbone identity from checkpoints: the supplied VGG CLI takes the ResNet branch. These fixes make code inspectable/executable; they do not retroactively validate saved experiments. Pin an appropriate historical environment when evidence exists. Preserve the unmodified audited source hashes.

## 3. Reprocessing existing raw data — approval required

If necessary, regenerate provenance and coefficients into a new versioned directory, never overwrite originals. Validate DWT ordering with a synthetic multiband signal and PyWavelets conventions. Check sample spacing, valid levels, signal length, nonfinite values, aliasing, time/frequency coordinate ordering, boundary effects, and magnitude versus energy definition. Separate a label-only frequency-axis fix from changed scales/coefficients/normalization that alter model inputs. Define a training-derived common normalization if absolute amplitude comparisons are scientifically intended.

All recordings must use a fixed label-independent pipeline at inference; a band choice tuned with labels belongs inside development fitting only. Preserve raw numeric coefficient/CWT data as well as rendered images. Recover acquisition bandwidth before claiming physical RF localization.

## 4. Reproducible evaluation design — before retraining

First decide the deployment unit: one original recording, a burst, or an acquisition session. Reserve a locked independent holdout of recording/session groups stratified by material where feasible, also checking electrode configuration balance. Save a seed and explicit sorted membership; the manifest, not the seed, is authoritative. If the historical test has repeatedly guided choices, reserve a new holdout if enough independent recordings exist and disclose the distinction.

Within development groups, use StratifiedGroupKFold with five outer folds if group/class counts support it. For each outer iteration, reserve an inner grouped training/validation split for early stopping and any wavelet/augmentation/model selection; keep the outer fold untouched until evaluation. Every derived image from a recording stays together, and every augmentation is generated only from that iteration's training groups. An alternative simpler design is grouped development CV for tuning, followed by one refit using an internal validation split and a single locked-holdout evaluation; explain which performance estimate is being reported.

The included SPLIT_MANIFEST_TEMPLATE.csv is a schema, not a fabricated historical split. For fixed holdout plus development CV, each row records original ID, class, electrode, session, image path/hash, outer_fold and train/validation/test membership. If an inner early-stop split is used, add an explicit inner_partition field rather than silently conflating it with the outer validation assignment. Add a manifest schema version, source dataset hash, split seed, generation code revision and grouping policy in companion metadata. A populated generated manifest can be checked with:

```
python audit/check_split_manifest.py path/to/populated_manifest.csv
```

The checker enforces pairwise recording/image/hash separation inside folds, a global development/holdout boundary, distinct disjoint outer validation image sets and unique recording validation-fold assignment. It does not infer provenance, prove acquisition independence, detect perceptual duplicates, or currently generate/check an inner split. Original membership is unavailable, so real-data overlap assertions are blocked, not passed. Add complete-development coverage and per-class/group-balance checks to the finalized split generator once the actual inventory and missing-data policy are known.

## 5. Model retraining — approval required

Train fresh VGG16/ResNet50 under the verified grouped design, identical material/configuration groups and matched comparison settings. Save actual trainable-layer lists, model graph, applied regularization, preprocessing, environment, seeds, membership hashes, completed epochs, failure logs and exact checkpoint selection. Change one scientific representation at a time. Validate generator tensor shapes/ranges and labels on small synthetic batches before committing compute.

Start with no augmentation and standard controls. Ablate justified raw-signal shifts/gain/noise and specialized transforms separately, using the same grouped partitions and repeated matched seeds. Re-enable a specialized transform only after checking it is actually consumed by fit and preserves physical meaning and labels. Record failed runs; do not silently exclude Fold 1 or select runs for favorable test accuracy.

Export one row per prediction including recording ID, image filename/hash, true material, probability of Teflon, run/fold/checkpoint ID and partition. Recalculate confusion matrices, per-class/macro metrics, specificity and ROC using these aligned rows. For a recording-level endpoint predefine aggregation across that recording's variants. Keep OOF predictions separate from repeated fixed-test predictions.

## 6. Inference/statistical work with existing checkpoints — approval required

If historical checkpoints and image membership are recovered, prediction-only verification can establish exact old metrics without retraining, but cannot repair leakage or ineffective CV. Recover both final model and val_accuracy checkpoint because they may differ. Compare exact probabilities/labels, including VGG Fold 5 and class-index order. For an ensemble, align samples and average probabilities before calculating metrics.

Recalculate paired uncertainty by bootstrapping independent recordings (or sessions) with all derived images retained inside each resampled cluster. Use identical resampled units for model comparisons and distinguish score/seed variation from sampling uncertainty. Do not treat four overlapping runs or 1280 repeated predictions as independent. Recover statistics-generation code before interpreting the old p-values.

Recompute explicit positive/negative binary Grad-CAM with training-matched preprocessing. Save numeric maps, targets, logits/probabilities, raw input identifiers and selection rules. Use prespecified concentration statistics only as descriptive attention evidence, never standalone proof of a discharge mechanism.

## 7. New experiments or acquisition information — conditional

If voltage, electrode or session are confounded with material, a new acquisition campaign or independent external evaluation may be needed; code changes alone cannot separate those effects. If independent original recordings or lineage are unavailable, a new clean campaign may be necessary to support generalization. Obtain sensor/oscilloscope bandwidth and trigger/sampling records for physical frequency claims.

Recover baseline and ablation artifacts before deciding whether new SVM/RF/CNN hybrid, wavelet-selection or noise tests are necessary. Any new baseline must share the verified groups and fit preprocessing only inside training. Do not expand the manuscript with incomparable experiments.

## Acceptance gates

Before manuscript resubmission: (1) traceable recording inventory and image selection, (2) documented overlap checks on actual data, (3) effective grouped membership consumed by training, (4) all prescribed runs accounted for, (5) prediction-level metrics with consistent class order and checkpoint identity, (6) recording-aware uncertainty, and (7) conclusions/figures limited to supported scientific interpretations. Present unresolved historical evidence explicitly if recovery remains impossible.
