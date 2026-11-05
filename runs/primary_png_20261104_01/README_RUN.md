# Corrected primary PNG experiment

Started 2026-11-04 from fresh official ImageNet initialization. The previous
no-augmentation experiment is preserved separately and is not resumed here.

`plan.json` holds the verified settings and source fingerprints. `queue_status.json`
shows current stage, process, log path and RAM. `supervisor_stdout.log` records
stage transitions; each `*_attempt_*.log` contains TensorFlow output. Fold
`training_log.csv` files update after completed epochs.

Fine-tuning ON, augmentation ON, batch 2, 224x224, learning rate 1e-5, maximum 50
epochs with patience 10. The primary profile has horizontal image shifts up to
3%, row/column masks up to 15%, and scaled pixel noise .05. No rotation, flipping,
zoom or vertical stretching is applied. All variants keep their parent IDs and
are created only from current training membership. Validation/holdout stay clean.

The automatic sequence is VGG16 CV, ResNet50 CV, VGG16 final fit, ResNet50 final
fit, VGG16 holdout evaluation and ResNet50 holdout evaluation. Logs, checkpoints,
model files, augmented examples, intermediate stages, mappings, figures and
predictions are saved inside this new directory.

The supervisor and Windows desktop alert monitor run in the background. Keep the
PC powered, signed in and awake; locking the screen and turning off the display
are fine. There is no automatic restart after a Windows reboot. Unexpected faults
stop with visible errors; recognized memory stops are retried with saved progress.

From the project root, view status in another PowerShell terminal:

```powershell
powershell -ExecutionPolicy Bypass -File runs/primary_png_20261104_01/Monitor.ps1
```

Closing this read-only monitor does not stop training. If an OS restart stops the
queue, restart the same queue only once no previous queue/worker remains active:

```powershell
.\.venv\Scripts\python.exe -u code/train_overnight.py --batch-dir runs/primary_png_20261104_01
```

The queue verifies source/settings consistency and resumes completed-epoch
snapshots. Final new results appear in `both_models_holdout_summary.json` only
after all stages complete. They must not be inferred from intermediate validation
scores. Raw wavelet provenance, historical counts and deferred reviewer studies
are not fabricated or declared resolved by this training run.
