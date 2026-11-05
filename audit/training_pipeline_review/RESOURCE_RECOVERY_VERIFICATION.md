# Local training resource and recovery verification

Verified 2026-11-03 using the project Python 3.11 environment and TensorFlow 2.21.0.
No full research experiment or locked-holdout evaluation was run.

## Implemented controls

- `code/training_safety.py`: available-RAM/free-disk checks, CPU/data/prefetch
  limits, OS-released exclusive training lock, completed-epoch recovery callback,
  atomic recovery/history exports, saved early-stopping and LR-plateau counters.
- `code/training_pipeline.py`: default batch size 2; four CPU compute threads,
  one inter-op thread, one data thread, one prefetched batch; path shuffling before
  decoding; full optimizer/model reload; `resume --run-dir`; completed-fold artifact
  verification/skipping and OOF aggregation; explicit interruption/failure records.
- `tests/test_training_safety.py`: RAM/disk refusal, lock contention/release,
  snapshot path validation, nonfinite rejection, best/latest retention,
  patience/LR/history recovery, and CLI resume/skip/aggregation behavior.
- `README.md`: supported settings, recovery command and limitations.

Memory reserve: 1.5 GiB available physical RAM. Disk reserve: 5 GiB. These are
polled before model construction and before training/validation batches; they
are not hard allocation limits or guarantees against an OS kill. Folds and model
checks run sequentially. A hardware/process crash can lose the current unfinished
epoch; recovery uses the last committed completed epoch. A first-epoch interruption
without a committed snapshot restarts the fold.

## Runtime evidence

`capacity_recovery_check.py` selects one Nomex and one Teflon image from each of
fold 1's training and validation memberships. Both architectures use 224x224,
batch size 2, augmentation and normal fine-tuning layers. Weights are randomly
initialized to avoid downloads. Each check deliberately raises KeyboardInterrupt
before epoch 2, reloads epoch 1 from disk including its optimizer, and completes
epoch 2. Optimizer iterations advance from 1 to 2. Predictions are checked for
membership/count/probability validity, with no scientific accuracy interpretation.

- VGG16: **PASS**, 8.20 seconds for the tiny check, available RAM 3.25 GiB before
  and 2.63 GiB after. Evidence:
  `capacity_checks/vgg16_224_run02/check_result.json`,
  `recovery_state.json`, `interruption.json`, and `training_log.csv` in that folder.
- ResNet50: **PASS**, 22.75 seconds for the tiny check, available RAM 2.53 GiB before
  and 2.20 GiB after. Evidence:
  `capacity_checks/resnet50_224_run01/check_result.json`,
  `recovery_state.json`, `interruption.json`, and `training_log.csv` in that folder.
- First VGG check `capacity_checks/vgg16_224_run01` also passed; use run02 for the
  final corrected check script and its more precise holdout-access description.
- Final test discovery: **50 tests, 49 passed, 1 opt-in smoke test skipped**.
  The separate actual 224px training/recovery checks above did run.
- `quick_training_check.py`: all source syntax, dependency availability and
  contract checks passed, including the new safety module.
- `resume --help` succeeds without starting training.

Paths in this report are relative to this report's directory unless prefixed
`code/`, `tests/` or `README.md`, which are relative to the repository root.

## Practical limits and next run

The machine exposes 12 logical CPU threads and approximately 15.87 GiB physical
RAM. The installed TensorFlow build exposes no GPU. Batch size 2 passed these
checks; batch size 4 was not tested. Keep one model process active and close
memory-heavy applications. Larger samples/full epochs can still encounter memory,
thermal, disk, input or OS failures; no full-run duration is inferred from these
tiny timings. ImageNet download/cache availability was not tested.

No holdout image was trained on, decoded for model input, or predicted by these
checks. Loading the split manifest does read all image files to verify their
recorded hashes, including holdout file hashes. Random shuffle/augmentation/dropout
streams are not replayed bit-for-bit across process restarts. Save the interruption
record when documenting the final experiment.

For an interrupted future experiment, from the repository root:

```powershell
.\.venv\Scripts\python.exe code/model_training_vgg16.py resume --run-dir runs/vgg16_cv01
```

This loads the saved architecture/settings, verifies the current training/safety
source hashes and split hash, skips verified completed folds, and continues the
incomplete fold. This recovery protocol applies to newly created runs using this
implementation, not arbitrary historical checkpoints.
