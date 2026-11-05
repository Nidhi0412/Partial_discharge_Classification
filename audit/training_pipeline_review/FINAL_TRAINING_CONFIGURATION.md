# Final primary training configuration — 2026-11-04

This is the current configuration after reading the actual reviewer comments and
the user's agreement to skip rotation/flipping. Earlier combined/historical-profile
reports describe previous code states. Full research training has not restarted.

## Enabled training behavior

- Current original PNG scalograms and existing saved recording/group splits.
- Actual ImageNet-initialized VGG16 and ResNet50, trained sequentially.
- Fine-tuning ON: VGG block 5 and weighted non-BN layers in ResNet's last-15-layer
  window update, along with the binary classification head. Earlier weights and
  ResNet batch-normalization remain frozen. Not feature-extractor-only training.
- Primary augmentation ON: `scalogram-image`. Four sequential training-only stages:
  horizontal image shift up to 3% (nearest borders, no wrap), row mask up to 15%,
  column mask up to 15%, scaled pixel noise with stddev .05 times current absolute
  RGB intensity. Each stage consumes the preceding stage's output.
- Rotation, flips, zoom, vertical shifting, vertical stretching and the old
  geometric/brightness generator are OFF in the primary profile. The old profiles
  remain explicit standalone ablation options, not unattended primary defaults.
- Same profile and saved splits for both backbones. No stochastic augmentation
  in validation/holdout. Variants keep parent recording/group/label and stay out of
  the original source inventory and split construction.

## Fixed model/runtime parameters

50-epoch maximum per fold/fit; early stopping on val_loss with patience 10;
learning rate 1e-5; LR plateau patience 5, reduction factor .1 and floor 1e-7;
dropout .5; batch size 2; 224x224 RGB, nearest resizing and division by 255.
The historical model-head profile preserves VGG's actual unregularized binary head
and ResNet kernel/bias L2 .01 plus Adam clipnorm 1.0. The name `historical` here
identifies model-head settings, not the enabled augmentation profile.

Four compute threads, one inter-op/data thread, one prefetched batch, path-first
shuffling, sequential jobs, available-RAM reserve 1.5 GiB and disk reserve 5 GiB.
The queue waits for 3 GiB free RAM before launching a model process. No safety
reserve was lowered to make checks pass.

## Execution and protection against unintended settings

`code/training_pipeline.py` and `code/train_overnight.py` share the primary defaults.
The unattended launcher rejects augmentation OFF, fine-tuning OFF, another profile
or nonzero vertical stretching. Resuming/skipping existing training outputs also
requires their stored model/purpose/settings to match this plan. The training
pipeline checks implementation/split hashes when resuming. The stopped old queue
is explicitly blocked; a new output directory is required.

The intended queue is both five-fold development CVs, both final fits, then one
locked-holdout evaluation per model. This is development CV for early stopping/
settings selection plus an independent holdout, not nested outer CV. Baselines,
separate corruption studies, acquisition-confound tests and post-training Grad-CAM
interpretation are separate work and are not falsely reported as completed here.

## Saved evidence and outputs

Original/effective membership, architecture, trainable layers, run/fold settings,
all generated-input recipe ledgers, bounded per-class original/augmented examples,
exact float32 tensors, ordered intermediate stages, preview sheets, histories,
TensorBoard events, model files, checkpoint hashes, predictions, classification
reports, ROC/confusion/training plots and recording/group metrics remain saved.
Stage files now use short names such as `<image_id>_s00.png`; operation names stay
in metadata. This fixes a Windows path-length issue caught in the full-size check.

Completed-epoch model/optimizer snapshots and early-stopping/LR counters allow
recovery. An unfinished epoch repeats. Random streams across process restarts are
not claimed bit-identical. Generated/prefetched records are not claimed to be
independent acquisitions or proof of consumption by a specific optimizer batch.

PNG masks/noise are image-space perturbations. No calibrated waveform phase/MHz/
laboratory-AWGN claim is made. The pipeline uses the supplied, user-confirmed
recording mapping; original acquisition history and manuscript counts remain
human/source-evidence questions, not invented by code.

## Verification

Final unit/contract discovery: 67 tests, 66 passed and one older optional smoke
test skipped. Tests cover strict launch defaults, changed-settings resume refusal,
actual transform composition and mapping, clean validation/inference, overlap/
partition rejection, recovery, nonfinite/resource stops, and deep Windows paths.

Both actual primary-profile engineering checks used cached official ImageNet weights,
224x224, batch size 2 and fine-tuning. They used only two development-training and
two validation images for two epochs, deliberately interrupted after epoch 1 and
resumed optimizer steps from 1 to 2. These are engineering checks, not manuscript
performance results. Each verified a real backbone kernel update, unchanged frozen
kernel, four actual augmentation stages, source/group-only ledgers, H5 reload with
matching predictions, and complete figure/report/example exports.

- VGG16: `capacity_checks/vgg16_primary_imagenet_20261104_r2/check_result.json`.
- ResNet50: `capacity_checks/resnet50_primary_imagenet_20261104/check_result.json`.
- The first VGG check failed during example export on a long Windows filename.
  That failed-check folder is retained; shortened stage names and a regression test
  fixed it before either full experiment was restarted.
- `FINAL_PRIMARY_VERIFICATION.json`: split/parent/stage/source checks and actual
  model-check summaries for the current implementation.
- `CORRECTED_TRAINING_PLAN.json`: the reviewed settings and source fingerprints
  for a fresh queue under `runs/primary_png_20261104_01`. Not launched by this change.

Paths in this section are relative to `audit/training_pipeline_review/`.
Existing research images, split memberships, old results and manuscript files
were not overwritten. Tiny checks cannot guarantee every future OS/data/hardware
operation will succeed; they verify the tested training path and recovery behavior.
