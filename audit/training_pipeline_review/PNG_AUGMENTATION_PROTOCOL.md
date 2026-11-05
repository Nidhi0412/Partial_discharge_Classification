# Current PNG augmentation and fine-tuning protocol

2026-11-04, following the user's choice: **Use current PNG scalograms**.
Full research training is still stopped. This document supersedes earlier notes
that specialized image augmentation had not yet been implemented.

## Implemented profile

`code/scalogram_augmentation.py`, `code/legacy_augmentation.py` and
`code/training_pipeline.py` now provide a composed `combined-image` profile.
The launcher and CLI explicitly default to augmentation ON with this profile.

Each training PNG is decoded/resized/normalized, then passes through:

1. Historical image augmentation: rotation up to 10 degrees, width/height shifts
   up to 10%, zoom up to 10%, horizontal flip, brightness 0.9–1.1, nearest borders.
2. Vertical image stretching: factor uniformly sampled between 0.85 and 1.15;
   centered interpolation keeps output dimensions fixed, with nearest borders.
3. Horizontal image shift: nonzero integer shift up to 15% of image width,
   without circular wrapping, using nearest borders.
4. Row band masking: one zero-RGB band up to 15% of image height.
5. Column band masking: one zero-RGB band up to 15% of image width.
6. Scaled pixel noise: Gaussian pixel perturbation with stddev .05 times the
   current absolute RGB intensity; final values stay in [0,1].

Every stage receives the previous stage's output. Input/output tensor hashes
check the composition chain. Failures raise an error; there is no catch-and-return-
the-original fallback. Defaults echo the magnitudes in the old custom declarations
but are a repaired, explicitly specified image experiment, not a claim of exact
historical execution or optimal/physically invariant augmentation.

The operations are PNG perturbations. Row/column bands do not establish calibrated
frequency/time units. Horizontal image translation is not a waveform phase shift;
vertical stretching is not automatically physical frequency scaling; pixel noise
is not laboratory waveform AWGN. Historical rotation/zoom also alter axis
interpretation, so the combined profile must be described as image-space
augmentation rather than physically calibrated time-frequency modification.

For a controlled ablation, `historical-image` keeps only the historical operators;
`scalogram-image` keeps only stages 2–6. Individual specialized magnitudes can be
set to zero explicitly. `--no-augment` names a separate no-augmentation experiment.
Use development evidence for augmentation selection; never choose settings using
locked-holdout accuracy. No robustness benefit is asserted before those results.

## Leakage boundaries and mapping

The supported order is original image/recording mapping, fixed grouped split,
then on-the-fly augmentation of that fitted model's TRAIN rows only. The same
original recording can train in some CV folds and validate in another; it cannot
occupy both partitions within one fold/model. VGG16 and ResNet50 share saved splits.

`load_splits` verifies original file and split hashes, group/recording/image/path/
content disjointness and fixed holdout membership. `validate_fit_membership`
independently checks train/validation partition labels and identity overlap.
`make_dataset(training=True)` and the specialized transformer reject non-training
parents. Validation/test use deterministic decoding/resizing/normalization and
never enter an augmentation branch or produce an augmentation ledger.

Every generated specialized input gets an `augmentation_ledger/*.jsonl` record:
parent image ID/path/hash, original recording ID, group ID, material, fold,
partition, seed, profile, parameters and ordered stage tensor hashes. A recipe ID
identifies the source/seed/profile; a unique occurrence ID and UTC generation time
identify each generated instance. Ledger records explicitly include possible
prefetch and are not presented as proof that a particular optimizer batch consumed
every generated tensor. Each resumed input pipeline writes a fresh ledger file;
a crash-truncated final line in an older file must not be silently treated as a
complete record. The model inputs and saved exact examples are compared in tests.

Up to five examples per class/fold are saved under `augmented_samples` with
original/final PNG, exact float32 NPY, intermediate PNG/NPY stages, source IDs,
seed, sampled parameters and hashes. Stage preview sheets are exported as
`augmentation_preview_Nomex.png` and `augmentation_preview_Teflon.png`.
These files stay inside the run directory and never enter the source inventory
or splits. Generated variants keep their parent's recording/group/label; they
are not additional independent experimental recordings. Original data are unchanged.

These safeguards verify software boundaries for the supplied mapping. The mapping
still reflects the user's confirmed distinct acquisitions; missing raw/session
evidence is not fabricated and no universal acquisition-independence certificate
is claimed. Raw CSV links are not needed for the selected PNG-only operators.

## Fine-tuning and outputs

VGG16 block 5 and the weighted non-BN layers in ResNet50's last-15-layer window
are trainable; the head is also trained. Earlier layers and ResNet BN stay frozen.
Actual weight-change checks in `vgg16_finetuning_weight_check.json` and
`resnet50_finetuning_weight_check.json` confirm the backbone updates and unchanged
frozen weights. This is not head-only feature-extractor training.

The 50-epoch maximum, early stopping, LR plateau reduction, batch size 2, bounded
threads/input buffering and resource reserves remain configured. Full Keras
optimizer snapshots, H5 compatibility models, histories, TensorBoard events,
classification reports, ROC/confusion/training figures, checkpoint hashes and
traced predictions remain saved. No no-augmentation checkpoint is reused for
the new augmented experiment.

## Verification and evidence

- Full tests: 64 tests, 63 passed, one original optional training smoke test
  skipped. Tests prove deterministic recipe replay, all-stage composition,
  bounded masks, non-wrapping shifts, valid shapes/ranges, changed training inputs,
  saved exact tensors/stage hashes, unchanged parent IDs, and clean validation.
- Actual VGG16 check: `capacity_checks/vgg16_png_combined_20261104/check_result.json`.
- Actual ResNet50 check: `capacity_checks/resnet50_png_combined_20261104/check_result.json`.
- Both used 224x224, batch size 2, fine-tuning, the six-stage combined profile,
  two tiny development-training images and two validation images. Each deliberately
  interrupted after epoch 1, resumed optimizer steps from 1 to 2, reloaded H5 with
  matching predictions and exported plots/examples/parent-only ledgers.
- These used random initialization for engineering checks, not replacement
  scientific results. Real full runs still default to verified ImageNet weights.
- Preview sheets were generated from the actual check tensors; the Nomex sheet
  was visually inspected to verify each successive stage is visible.
- `PNG_SPLIT_GUARD_VERIFICATION.json` records the actual saved split checks and
  verifies that engineering-check ledger parents come only from training.
- `CORRECTED_TRAINING_PLAN.json` freezes the proposed new sequential queue,
  including explicit augmentation enablement, magnitudes, order and source hashes.
  It has not been launched. The stopped old run remains preserved.

Paths in this section are relative to `audit/training_pipeline_review/`.
