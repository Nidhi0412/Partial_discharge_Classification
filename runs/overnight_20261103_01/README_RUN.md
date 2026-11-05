# Unattended VGG16 + ResNet50 experiment

**STOPPED BY USER on 2026-11-04 for augmentation/protocol correction. Do not resume
this queue. All files below describe the preserved no-augmentation run.**

Started 2026-11-03. Read `queue_status.json` for current state and the current log
path. `supervisor_stdout.log` records stage starts/completions; the per-stage
`*_attempt_*.log` files contain actual TensorFlow output/errors. Fold CSV logs
update after completed epochs. This directory contains new research runs, not the
earlier tiny capacity checks or historical results.

The frozen `plan.json` orders VGG16 five-fold development CV, ResNet50 five-fold
development CV, VGG16 final-fit, ResNet50 final-fit, then one locked-holdout
evaluation for each. Each fold has a 40-epoch maximum with validation-loss early
stopping (patience 10). Epoch count can differ across folds. Checkpoints contain
the actual requested architecture and originate from verified official ImageNet
weights. No random-weight fallback is enabled.

Both architectures use the same saved split, seed 42, 224x224 inputs, batch size 2,
learning rate 1e-5, dropout .5, L2 .01, four compute threads and bounded input
loading. Existing default preprocessing RGB/255 is preserved. This initial
fixed-settings run has augmentation OFF, as in the documented default baseline.
No automatic hyperparameter/model selection or holdout-based selection occurs.
The standalone time-translation augmentation option is not implicitly enabled.

The launcher runs in the background without requiring the Codex conversation or
a visible terminal to remain open. Keep the PC powered, signed in (locking is
fine) and awake; do not shut down, restart or sign out. Its Windows awake request
allows the display to turn off and lasts while this queue process is alive; the
plugged-in sleep setting was also verified as Never at launch. Power loss or an
OS restart cannot be prevented by training code. There is no reboot auto-start.

The launcher retries recognized memory/resource failures at most six times per
training stage, using the last completed epoch and preserving safety reserves.
It waits for at least 3 GiB available RAM before starting a TensorFlow process.
Long waits are visible as WAITING_FOR_RAM, not reported as active training.
Unexpected code errors stop the queue with FAILED and a log path. A failed holdout
evaluation is not automatically rerun. The launcher does not promise that every
possible fault can be resolved without human intervention.

If Windows restarts or the launcher is stopped, restart the same queue from the
repository root (only when no prior queue/worker remains active):

```powershell
.\.venv\Scripts\python.exe -u code/train_overnight.py --batch-dir runs/overnight_20261103_01
```

The queue skips completed stages and resumes interrupted training stages from
their saved settings/state. It refuses changed training sources. An incomplete
epoch repeats; stochastic streams are not replayed bit-for-bit. At successful
completion, `both_models_holdout_summary.json` contains the two new final results.
Comparison with the historical manuscript and scientific interpretation remain
separate analyses; the launcher does not fabricate old/new comparability.

To view live status in another PowerShell window:

```powershell
powershell -ExecutionPolicy Bypass -File runs/overnight_20261103_01/Monitor.ps1
```

Closing that monitor does not stop training.
