# Fine-tuning, specialized augmentation and leakage-boundary recheck

2026-11-04. Full training remains stopped. No new research results were generated.
Paths below are relative to the repository root.

## Actual fine-tuning: CONFIRMED

`code/training_pipeline.py:build_model` unfreezes VGG16 block5_conv1/2/3 and the
weighted non-BN layers in ResNet50's last-15-layer window. Earlier layers stay
frozen. The binary head is trained too. A new runtime check rejects a requested
fine-tuning configuration if no backbone weights are trainable.

Actual synthetic 32x32 one-step checks, using separate processes and random
initial weights, prove that unfrozen convolution kernels change while the first
frozen convolution remains bitwise unchanged. These are engineering checks,
not scientific performance measurements:

- `vgg16_finetuning_weight_check.json`: block5_conv3 changed by up to about
  9.85e-6; block1_conv1 unchanged; head changed.
- `resnet50_finetuning_weight_check.json`: conv5_block3_3_conv changed by up to
  9.94e-6; conv1_conv unchanged; head changed; all BN layers frozen.
- Verification script: `audit/training_pipeline_review/verify_finetuning_step.py`.

## Extra augmentation code found: CONFIRMED, execution is a separate question

`audit/before_pipeline_fixes_20261103/code/model_training_vgg16.py:84–263`
declares frequency-row masking, time-column masking, image circular shifting,
vertical resizing/cropping/padding and magnitude-scaled Gaussian image noise,
in addition to ordinary image transforms. Similar declarations occur in the
archived `model_training.py`.

The specialized generator's `apply_scalogram_augmentations` at lines 315–350
computes separate results from the same x, then returns only `freq_stretched`.
The frequency-masked, time-masked and shifted values are not composed into that
returned tensor. `layers[-4:]` at 307–310 also does not name all five declared
custom layers; the model ends with FrequencyMasking, TimeMasking, PhaseShift,
FrequencyStretch and ScaledGaussianNoise. Index-based names are therefore
unreliable. The function catches failures and returns the original input, which
can silently turn requested augmentation into a no-op.

The main historical CV loop separately uses `get_data_generator(augment=True)`
at 952, whose standard ImageDataGenerator is at 814. A function's presence or a
saved `use_specialized_augmentation` flag alone does not establish that fit
consumed it. The corrected historical-image profile restores that standard path.
The specialized transforms are NOT currently enabled in the corrected launcher;
they need an explicitly chosen, tested path instead of unverified old wiring.

## Where augmentation belongs

1. Establish parent recording IDs and any session grouping before any split.
   All images/views from a parent share its ID; IDs are never regenerated for
   augmented copies.
2. Lock the test groups and development folds once, using the same memberships
   for both backbones. Validate original files, metadata and content hashes.
3. For each fold, only its TRAIN groups enter stochastic augmentation. The
   fold's validation groups receive deterministic preprocessing. A development
   recording may train in some folds and validate in another; it must not occupy
   both partitions within one fitted model.
4. For raw-waveform shifts, gain, AWGN or filtering, operate on that fold's
   training CSVs before producing their scalograms. Keep the original parent ID,
   sampling/time axes, source hash, seed and fixed generation configuration.
   Any data-dependent filter/normalization/entropy selection is fitted/selected
   using training/development information only.
5. For PNG operations, decode the training image, apply named image-space
   transforms, save source/parent IDs and exact parameters, then feed the result
   to the model. Image row masking is not calibrated MHz-band masking without
   an axis mapping; an image roll is not a waveform phase shift; vertical image
   resizing is not automatically a valid physical frequency perturbation.
6. Export augmented examples under the run directory. Never inventory them back
   into the original dataset or split them as independent recordings. Keep
   validation and test clean; conduct noise robustness testing as a separately
   declared endpoint rather than contaminating the primary holdout.

For frequency-aware image operations, rotation/vertical shifts/zoom should not
be blindly chained ahead of row masking: they alter the interpretation of the
image axes. A practical specialized profile should use explicitly bounded
operations with verified axes/labels and development-only ablations. Save every
enabled operation, its order and parameters; fail on errors, never silently
return unaugmented input while claiming augmentation succeeded.

## Code-level leakage checks: strengthened and verified

`code/dataset_manifest.py:validate_split/load_splits` checks within-fold disjoint
recording/group/image/path/content identities, fixed holdout membership, unique
validation assignment, metadata consistency, split hash and original image hashes.

`code/training_pipeline.py:validate_fit_membership` now independently rejects
incorrect train/validation partition labels, missing identities and any overlap
in image ID, path, recording ID, group ID or content hash immediately before fit.
`make_dataset(training=True)` also rejects validation/test members, preventing
their entry into the training/augmentation branch. Fine-tuning checks fail loudly.
The old synthetic smoke fixture was updated to label training members correctly.

Tests in `tests/test_historical_augmentation.py` cover wrong-partition rejection,
each identity-overlap rejection, actual transformed training tensors and their
saved source mapping, historical transform equality, and unchanged deterministic
validation/inference. Full test discovery: 59 tests, 58 passed, 1 opt-in original
training smoke test skipped. The real backbone-update checks above did run.

These guards establish the checked software boundaries, not unknown acquisition
history. `image_mapping.csv` has 1600 mapped images based on the user's confirmed
distinct-recording identities; all source_csv_path fields are currently blank.
Raw waveform augmentation cannot be wired to those images without real CSV links.
Session independence also cannot be inferred from invented bookkeeping IDs.
No conclusion of confirmed recording leakage is made here.

## Pending input before enabling specialized transforms

Choose current PNG scalogram augmentation, explicitly labelled as image-space
masking/perturbation, or provide the original waveform CSV location and genuine
CSV-to-image links for waveform augmentation. Neither path may invent missing
physical frequency axes, calibration, acquisition IDs or label-preserving ranges.
The existing no-augmentation run and original data remain preserved and stopped.
