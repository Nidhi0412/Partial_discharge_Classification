# Implemented pipeline fixes

Date: 2026-11-03. Changes apply to the local classification repository only. No full training, inference on research images, wavelet regeneration or replacement experimental results were produced.

## What changed

The 17 overlapping legacy research entrypoints now delegate to five shared implementations, rather than maintaining incompatible training methods:

- `code/dataset_manifest.py`: trusted image-to-recording mapping, recording/session grouping, deterministic material/electrode balancing, fixed independent holdout, effective development folds, membership/coverage/content assertions and hashed split configuration. Original images are never moved or deleted. Unknown recording identities are rejected, not inferred from filenames.
- `code/training_pipeline.py`: explicit VGG16/ResNet50 factories, fresh models per fold, shared manifest-driven loaders, RGB/255 preprocessing, optional training-only horizontal translation, applicable L2/dropout settings, consistent best-val-loss checkpoint selection, explicit failed-run status, per-fold actual configuration, architecture and membership exports, traced prediction output, recording-level aggregation and conditional cluster-bootstrap holdout accuracy intervals.
- `code/evaluation_tools.py`: shared evaluation, prediction validation, disjoint-development prediction summaries and ROC plotting. Duplicate image predictions cannot be pooled as independent observations. Per-image scores are linked to the recording, model, fold, split and checkpoint hashes. Reports distinguish development validation from the final locked holdout.
- `code/explain_pipeline.py`: actual checkpoint loading, training-matched preprocessing, explicit p(Teflon) or 1-p(Teflon) targets, checkpoint/selection verification and numerical Grad-CAM outputs. No selection by best holdout score.
- `code/wavelet_pipeline.py`: optional version-2 new-data generation with actual DWT level indexing, decimated-coefficient sampling interval, explicit representation/normalization, source-recording lineage, per-variant coefficients/coordinates and image/CSV/config hashes. No source images were regenerated. This representation is not silently substituted for historical transforms.

`scalogram_utils.py` now requires an explicit version-2 configuration for new raw-signal inference. Generation/evaluation/Grad-CAM wrappers share the new tools. `main.py` and `main_clean_scalograms.py` no longer delete directories or run their independent four-class filename splitters. Launchers default to the intended backbone and require explicit manifests and output destinations rather than targeting electrode datasets.

Historical CLI signatures and callable interfaces were replaced deliberately. Use each file's `--help` and the rewritten README for the supported commands. The old implementations remain available in the backup; no claim is made that the corrected pipeline reproduces the old experiment methodology.

## Evaluation protocol actually supported

Development grouped CV is used for settings selection and early stopping. Its reported validation scores are explicitly labeled development estimates; they are not claimed to be unbiased nested outer-CV performance. If unbiased outer-CV performance is required, a distinct inner grouped validation/tuning stage remains a separate protocol extension.

`final-fit` uses grouped training/validation inside development. It saves the selected checkpoint without consuming holdout images. A separate `evaluate` command verifies the manifest and checkpoint and creates one primary locked-holdout evaluation output. No repeated test predictions are counted as additional independent observations.

Augmentation is off by default. `--augment` enables bounded horizontal image translation only during fitting. Rotation, frequency-axis transforms, special masking and Gaussian noise were not automatically enabled merely because they existed in the old source. Physical label invariance and benefits require a scientifically justified, matched ablation. The optional image transform is named accurately and not called waveform phase shifting.

The splitter rejects cross-partition identical content for review of the proposed mapping; it does not delete samples, invent original recording IDs or make claims about laboratory acquisition history. Session grouping is available when that is the appropriate independent acquisition unit. The grouped balancing algorithm is heuristic; actual ratios and class/group counts must be inspected for the real mapped campaign.

## Validation performed

`tests/test_pipeline.py` runs on temporary synthetic data. Latest result: **26 discovered tests, 25 passed, 1 runtime test skipped**. Covered behavior includes:

- Actual grouped assignment/coverage across all development folds and a fixed holdout.
- Sessions containing multiple material labels; deterministic splitting.
- Rejection of missing/conflicting recording identities, duplicate/unsafe image IDs, changed images/manifests and invalid cross-partition memberships/content.
- Source-file preservation and refusal to overwrite split outputs.
- VGG/ResNet constructor dispatch using mocked dependencies; this is a factory contract check, not actual TensorFlow model construction.
- Sigmoid threshold, confusion counts, specificity, AUC ties, recording aggregation and recording/session-bootstrap grouping.
- Rejection of repeated recording validation assignments, correct binary Grad-CAM score targets and ROC tie handling.
- CLI help working without importing TensorFlow.

All **22 research Python files parse successfully**. `git diff --check` passes; Git reports only repository line-ending conversion notices. Detailed preflight output is in FIX_VALIDATION.json.

The old quick checker was updated to check the corrected shared pipeline. It uses read-only static/dependency checks and runs contract tests with actual training disabled. It returns a blocked status if runtime dependencies are unavailable, even when static/unit checks pass.

## What was not verified

The installed Python 3.14.6 interpreter and the two discovered virtual environments lack TensorFlow. The checked interpreter also lacks Pillow, Matplotlib and PyWavelets. No packages or Python versions were installed. An explicit attempt to invoke `smoke-test` stopped at the missing TensorFlow dependency before constructing a model or creating synthetic training outputs.

Therefore actual TensorFlow fitting, callback execution, model persistence/reload and wavelet/Pillow rendering were **not runtime verified**. Static and mocked tests do not prove they will run without errors in a configured environment.

The included explicit smoke test uses both real backbones, frozen random-initialized weights, synthetic 32×32 inputs, one epoch, training augmentation, checkpoint save/reload and aligned predictions. It downloads no ImageNet weights and does not touch the research dataset. Run it after environment setup:

```
python code/model_training_vgg16.py smoke-test
```

Do not start full training until it passes. The README documents installation into an isolated environment, contract checks, grouped mapping/splitting, development training, final fitting and evaluation. Requirements are shared by the supported code; no separately incompatible standalone Keras pin is introduced.

## Preservation and concrete next input

All **19 original source/documentation files** are preserved with SHA256 hashes in `audit/before_pipeline_fixes_20261103/`. Hash comparisons verified the backups, all **42 historical result files** and all **1600 original dataset images**. Original datasets and results were not modified.

`audit/training_pipeline_review/IMAGE_MAPPING_TEMPLATE.csv` contains 1600 genuine existing image paths/hashes and observed material-folder labels. The recording_id, electrode/session and raw-generation provenance fields remain blank where evidence is missing. It is a mapping template, not a validated acquisition manifest. Fill it from acquisition records or an authoritative original-to-image mapping; do not assign independent recording IDs merely to satisfy the software.

Next required inputs are: **trusted recording mapping** and **a configured runtime in which the tiny smoke test passes**. Until both are available, the corrected code is implemented and statically/unit checked, but real-data training is not claimed ready or executed.
