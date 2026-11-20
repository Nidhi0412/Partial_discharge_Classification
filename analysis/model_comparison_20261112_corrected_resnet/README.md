# Development OOF model comparison

Scope: paired out-of-fold predictions on the common 1,280-image development set. Intervals use group_id-clustered bootstrap resampling. These are development results, not locked-holdout estimates.

## Files

- `model_metrics.csv`: point estimates, calibration, and grouped 95% intervals.
- `paired_comparisons.csv`: paired differences and exact McNemar calculations.
- `comparison.json`: machine-readable complete output.
- `provenance.json`: input paths and SHA-256 hashes.
- `accuracy_intervals.png` and `calibration.png`: manuscript-ready diagnostic figures.

McNemar p-values use image-level discordances and can retain within-group dependence; clustered bootstrap differences are the primary uncertainty summary.
