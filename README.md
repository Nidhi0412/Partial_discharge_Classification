# UHF partial-discharge insulation classification

**2026-11-04 correction:** the unattended no-augmentation run has been stopped
at the user's request. Its files are preserved under `runs/overnight_20261103_01`;
do not resume that queue. The final primary profile is `scalogram-image`, with
augmentation and fine-tuning ON, and rotation/flipping/zoom/vertical shifts/stretch
OFF. This follows the actual reviewer comment and the user's agreed change.
Current authoritative settings and checks:
`audit/training_pipeline_review/FINAL_TRAINING_CONFIGURATION.md`.
Earlier restoration/combined-profile reports are historical snapshots, not current
default settings. Full training remains stopped; engineering checks are separate.

The supported pipeline uses a trusted recording-to-image manifest, a locked holdout,
and shared grouped development folds for VGG16 and ResNet50. Historical code and
README are preserved in `audit/before_pipeline_fixes_20261103/`. Existing images,
checkpoints and historical results have not been regenerated or replaced.

## Environment and quick checks

Create an isolated environment (Python 3.11 is a conservative baseline). Run commands
from this repository root using that environment's Python. Activation is optional;
you can replace `python` with the full path to `.venv\Scripts\python.exe`.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python audit/training_pipeline_review/quick_training_check.py
```

The unit tests use temporary synthetic inputs. They do not train or change the
laboratory dataset. A separate explicit test exercises both actual backbones,
training augmentation, checkpoint persistence and prediction alignment using tiny
synthetic 32x32 images, frozen backbones, one epoch and no ImageNet downloads:

```powershell
python code/model_training_vgg16.py smoke-test
```

Do not start full training until that test passes in the intended environment.
Static checks cannot guarantee training will finish without runtime errors. The project .venv now has Python 3.11 and the required packages. Imports, pip check
and the current contract tests passed. The tiny actual-training smoke test also passed
for both VGG16 and ResNet50 on synthetic inputs; this does not validate research
accuracy, acquisition mapping or full-size fine-tuned training.
Use .venv/Scripts/python.exe rather than the unrelated global Python interpreter.
To opt into the smoke test through unittest instead, set PD_RUN_TF_SMOKE=1.

## 1. Create a mapping template

```powershell
python code/split_dataset.py inventory --data-dir Insulation_classifier --output image_mapping.csv
```

The command reads images and creates a NEW CSV. It never moves or deletes images.
Fill recording_id from acquisition evidence; material is a folder-derived label
that you must verify. Also supply electrode and session_id where known. Leave
original filenames alone. Different images from one raw recording share one
recording_id. IDs must be unique across sessions/folders, not bare numbering.

A template with blank recording IDs cannot be trained. Image hashes check content
consistency; they cannot identify original laboratory acquisitions. If content is
identical across proposed partitions, the splitter stops for mapping review rather
than deleting files or inventing a recording ID.

If independently acquired images cannot be associated with raw CSVs, document the
actual acquisition unit accurately. Do not assert recording-independent evaluation
merely by giving each image a unique ID.

## 2. Save grouped splits once

```powershell
python code/split_dataset.py split --manifest image_mapping.csv --output-dir splits/run01 --folds 5 --test-fraction 0.2 --seed 42
```

Use `--group-by session_id` when acquisition sessions are the independent units.
All images of a group remain together. The splitter uses deterministic greedy
balancing of recording counts by material/electrode; exact proportions are not
guaranteed under group constraints. It rejects missing classes or invalid overlap.
The same splits.csv and split_config.json must be used by both backbones.

For the current user-confirmed one-recording-per-image dataset, `splits/run01`
has now been generated with `--keep-identical-together`. That option retains all
distinct recording IDs but co-locates files with identical bytes as a conservative
partitioning constraint. It does not equate their acquisition identities. The
effective `group_id` is used consistently for CV, final-fit validation and
uncertainty clustering. The validated run01 split has 1024 training, 256 validation
and 320 fixed holdout images in every development iteration, balanced by material.

Train/Test assignments in an old folder are not automatically treated as verified
recording-level assignments. This command makes a NEW, explicitly documented split
from the complete trusted mapping. Nothing is moved. The test partition stays fixed
through development folds, and every development image has one validation fold.

## 3. Development cross-validation

### Local machine limits and interruption recovery

Training now defaults to batch size 2, four TensorFlow compute threads, one
inter-op thread, one data-loading thread and one prefetched batch. Folds run
sequentially, and a project-wide OS lock rejects concurrent training jobs.
Training shuffles file paths before decoding instead of buffering all decoded
images. These controls bound concurrency; they are not a hard process-RAM cap.

Before training and before training/validation batches, resource checks require
at least 1.5 GiB available physical RAM and 5 GiB free disk. A failed check stops
with an explicit error. Close other applications before resuming. Flags are
`--cpu-threads`, `--data-threads`, `--prefetch-batches`, `--min-free-ram-gb`, and
`--min-free-disk-gb`. Do not run both model commands simultaneously.

Each completed epoch saves a full `.keras` checkpoint including optimizer state,
then atomically commits `recovery_state.json` with epoch, best loss, early-stopping
patience, learning-rate counters and history. The latest and best epoch snapshots
are retained. The JSON history and CSV log are exported after every committed
epoch. `best.keras` is exported from the selected snapshot when fitting completes.
Early stopping still uses val_loss with patience 10 by default; the learning rate
uses the historical factor 0.1 after a plateau, with a floor of 1e-7. Nonfinite metrics stop training without committing the
invalid epoch.

After Ctrl+C, a resource stop or a process/power interruption:

```powershell
.\.venv\Scripts\python.exe code/model_training_vgg16.py resume --run-dir runs/vgg16_cv01
```

The same resume command works for either architecture and for final-fit runs;
settings and model identity come from the run configuration. Completed folds are
hash-checked and skipped. Source/split changes are rejected. Failed run records
remain visible; status becomes COMPLETE only after successful recovery.
An unfinished epoch is repeated from the previous completed epoch. If interruption
occurs before epoch 1 is saved, that fold restarts from initialization. Shuffle,
dropout and augmentation RNG streams are not restored bit-for-bit after a restart;
this is documented epoch recovery, not an exact replay. An OS kill during a batch
cannot be intercepted, and a check cannot guarantee that memory remains available
throughout the next operation. The previous committed snapshot is retained.

The explicit engineering capacity check uses two development training images and
two validation images, full 224x224 inputs, augmentation, batch size 2, normal
fine-tuning layers, and cached official ImageNet weights when requested below.
It deliberately interrupts and resumes, checking optimizer continuity and actual
backbone weight updates while frozen weights remain unchanged. It does not use holdout
tensors or generate manuscript metrics:

```powershell
.\.venv\Scripts\python.exe audit/training_pipeline_review/capacity_recovery_check.py --model vgg16 --primary-png --weights imagenet --output-dir audit/training_pipeline_review/capacity_checks/vgg16_new_check
.\.venv\Scripts\python.exe audit/training_pipeline_review/capacity_recovery_check.py --model resnet50 --primary-png --weights imagenet --output-dir audit/training_pipeline_review/capacity_checks/resnet50_new_check
```

ImageNet weights remain the default for actual training, and both final primary
checks use the cached official weights. Tiny checks
do not establish sustained thermals, full-run time or eventual accuracy.

```powershell
python code/model_training_vgg16.py cv --model vgg16 --splits splits/run01/splits.csv --output-dir runs/vgg16_cv01
python code/model_training.py cv --model resnet50 --splits splits/run01/splits.csv --output-dir runs/resnet50_cv01
```

These are commands for an approved full experiment; they have not been run here.
Each fold builds fresh ImageNet-initialized weights, reads its actual saved file
lists and saves effective train/validation membership, actual architecture,
trainable layers, settings, histories, checkpoint hashes and predictions.

Historical-profile preprocessing is explicitly RGB, nearest resize and float32/255
for BOTH backbones, matching the old generator's interpolation default.
Changing to standard ImageNet preprocessing is a separate experiment, not a hidden
inference change. Binary outputs always represent p(Teflon); Nomex=0/Teflon=1.
L2 and dropout apply according to the saved settings. ResNet fine-tuning keeps batch
normalization frozen. Failed folds stop the experiment and produce failure.json;
they are not silently excluded from a reported mean.

Augmentation is ON by default (`--augmentation-profile scalogram-image`). Only
training images pass through these four composed stages:
non-circular horizontal image shift up to 3% with nearest borders, one row band
mask up to 15%, one column band mask up to 15%, and scaled RGB pixel noise
(stddev .05 times current absolute intensity). Rotation, flipping, zoom, vertical
shift/stretch and the old geometric/brightness generator are not called.
No stage silently discards earlier outputs. These are PNG perturbations, not
physical phase/MHz changes or waveform AWGN; benefit still needs evaluation.
Historical/combined profiles remain available only through separate explicit
entrypoint runs. The unattended primary launcher rejects those profiles,
disabled augmentation, feature-only training and nonzero vertical stretching.
The 50-epoch maximum, patience 10, learning rate 1e-5 and dropout .5 follow saved
historical configurations. Batch size remains 2 for this machine. ResNet's historical
kernel/bias L2 and gradient clipping are restored; VGG's old binary head had no L2
despite the old configuration containing an unused l2_reg value. Frozen ResNet BN
remains a documented small-batch safeguard.

Original/augmented PNG examples, exact generated float32 NPY examples and their
parent recording/group IDs are saved per fold under `augmented_samples`, using
five examples per class. Examples come from the actual training input pipeline;
intermediate PNG/NPY stages and preview sheets are exported too. Every generated
specialized input also gets a parent-ID/seed/recipe entry under `augmentation_ledger`;
prefetched examples are not claimed to be consumed optimizer batches. Training
curves, TensorBoard events, confusion matrices, ROC, classification reports, CSV
histories, predictions and both primary/secondary checkpoints are also exported.
`model.h5` contains the selected best-val-loss model; `best_weights.h5` contains
the secondary best-val-accuracy model and is not used to select holdout results.
No augmentation is applied to validation or holdout inputs. Use `--no-augment`
only for an explicitly named separate baseline. The former bounded translation
option is available explicitly as `--augmentation-profile time-translation`.
The repaired specialized PNG implementation is new; old dormant frequency/phase
code is not retroactively described as executed. Image-space augmentations require
scientific justification; their restoration alone does not establish invariance.

This CV is development validation used for early stopping/settings selection.
It must NOT be described as unbiased nested outer-CV performance. For unbiased
outer-CV estimates while tuning, implement a distinct inner grouped validation
stage. The supported generalization endpoint below is the locked independent
holdout. Never use holdout scores to choose a fold/checkpoint or hyperparameters.

## 4. Final fit, then one primary holdout evaluation

After settings are selected using development information:

```powershell
python code/model_training_vgg16.py final-fit --model vgg16 --splits splits/run01/splits.csv --output-dir runs/vgg16_final01
python code/evaluation.py evaluate --run-dir runs/vgg16_final01 --splits splits/run01/splits.csv
```

Final-fit uses a new grouped 80/20 split inside development for training/early
stopping. It saves the best val_loss checkpoint; it does not consume holdout
images. Use the selected development settings explicitly, including augmentation.
Evaluation requires a completed final-fit run and the same hashed splits. It
writes one primary holdout_evaluation directory exclusively; existing outputs are
not overwritten. A failed attempt remains visible and is not a successful test.

Metrics distinguish image-level and recording-level endpoints. Recording-level
probability is the mean across that recording's images, with threshold >=0.5.
Holdout uncertainty uses a recording/session cluster bootstrap conditional on the
fitted model. It is not an independent-fold t-test or a measure of training/model
selection variability. No superiority p-value is manufactured.

## 5. ROC and Grad-CAM

```powershell
python code/make_foldwise_roc.py roc --predictions runs/vgg16_cv01/oof_predictions.csv --output-dir runs/vgg16_cv01/roc01
python code/crossval_aggregate_evaluation.py summarize --predictions runs/vgg16_cv01/oof_predictions.csv --output-dir runs/vgg16_cv01/summary01
python code/run_gradcam.py --run-dir runs/vgg16_cv01 --checkpoint runs/vgg16_cv01/fold_1/best.keras --predictions runs/vgg16_cv01/fold_1/predictions.csv --output-dir runs/vgg16_cv01/gradcam01 --target true
```

OOF plots are explicitly development-validation plots. The summarizer rejects
repeated image IDs; it does not concatenate repeated holdout scores as independent
observations. Grad-CAM loads the actual checkpoint, verifies prediction/manifest
identity and uses saved preprocessing. It explains p for Teflon and 1-p for Nomex,
and exports numerical heatmaps and a deterministic selection manifest. Attention
maps alone do not prove a physical discharge mechanism.

## Quantitative explainability after training

Quantitative post-training explainability is available with a frozen analysis plan.
See `audit/training_pipeline_review/EXPLAINABILITY_ANALYSIS_METHODS.md` for recording/
session-bootstrap intervals, concentration/region statistics, aligned translation
stability and masking-versus-random diagnostics, including their limitations and
commands. Freeze the specification before inspecting final maps; no research
explainability analysis has been run. These diagnostics do not prove physical
mechanisms or supply calibrated RF-band labels.

## Optional: approved raw-data reprocessing

Existing image-based training does not invoke scalogram generation. If raw inputs
must be regenerated, first create raw_mapping.csv with recording_id, csv_path,
material, electrode and session_id. Both generation entrypoints use the same new
versioned implementation:

```powershell
python code/generate_clean_scalograms.py --raw-manifest raw_mapping.csv --output-dir generated/run01 --wavelets db3 --levels 1 --representation magnitude
```

This is an EXAMPLE configuration, not a claim that db3/D1 is scientifically optimal
or matches historical images. Set the actual approved wavelet/level combination
explicitly. The command has not been run on the research data.

Version 2 uses actual cD1/cD2/etc. indexing, decimated-coefficient sample spacing,
explicit magnitude/energy and normalization, and unique per-recording outputs.
Each image has coefficient and coordinate sidecars and a source CSV/image hash in
images.csv. Rows follow CWT scale index, not uniform MHz; frequencies describe the
decimated coefficient sequence, not automatically the original RF band centers.
Per-image normalization removes absolute color-amplitude comparability; a fixed
normalization requires explicit bounds chosen without test leakage. No Shannon
entropy selection or class-dependent level rule is fabricated. This renderer is
not implicitly compatible with old positional dc labels or historical checkpoints.

## Entry points and original artifacts

After both corrected models are trained/evaluated, an old-versus-new comparison is
required. See `audit/training_pipeline_review/REVIEWER_COVERAGE_AND_COMPARISON_PLAN.md`
for reviewer-specific evidence requirements, metrics/artifact comparisons and the
limits on comparing historical scores with the new grouped protocol. A successful
training run alone does not resolve novelty, wavelet-selection, baseline, noise or
acquisition-confound concerns.

Both model_training*.py files, both run_parallel_training*.py files and main*.py
now route to one training implementation; they no longer auto-delete data or launch
an electrode task. split_dataset.py routes to the shared manifest tool. Evaluation,
ROC, Grad-CAM and generation wrappers expose the shared tools' --help interfaces.
Legacy command-line signatures/functions were intentionally replaced rather than
preserving conflicting APIs. Historical numerical results remain historical
artifacts: this code change does not retrospectively validate their provenance.
