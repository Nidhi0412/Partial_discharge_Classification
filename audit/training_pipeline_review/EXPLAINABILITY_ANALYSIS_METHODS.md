# Post-training quantitative explainability analysis

Implemented 2026-11-03. No laboratory images, trained research checkpoints or final-test maps were analyzed. The supplied config is a proposed descriptive analysis specification: review its domain assumptions and freeze it before inspecting final maps. A local timestamp/hash is not external preregistration and cannot prove that a researcher has never inspected the test data.

## What is calculated

Grad-CAM remains a coarse, class-targeted localization method, following [Selvaraju et al.](https://arxiv.org/abs/1610.02391). Numerical diagnostics supplement pictures; visual appeal alone is not validation, as discussed in [Adebayo et al., Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292). The implementation below does not claim to implement their model/data randomization tests.

1. **Localization and concentration.** Nonnegative maps are normalized to unit total activation mass. Report normalized entropy, fraction of map cells needed to reach 80% activation mass, normalized centroid and activation mass in predefined image rectangles. Lower normalized entropy/mass-area means more concentrated attention, not necessarily a better or more physically correct model. No raw intensity is compared between networks or called physical energy.

   This entropy describes a normalized activation map. It is not the manuscript's wavelet-selection entropy and does not validate choosing the mother wavelet with highest Shannon entropy.
2. **Predefined regions.** Default rectangles are left/middle/right image thirds and upper/lower image halves, using pixel centers and fractional coordinates. They are image-coordinate summaries, not calibrated MHz bands or DWT subband labels. Time-axis and scale/frequency interpretation requires verified rendering provenance; upper/lower halves must not automatically be named DC1/DC4/DC5. Regions are evaluated on the native coarse activation map; resolution limits remain visible.
3. **Stability diagnostic.** Shift the rendered input horizontally by -2/+2 pixels with per-image channel-mean fill. Recompute the same fixed-target map; resize maps to input resolution and undo the shift on their common valid support before cosine comparison. Also report the prediction-change rate. This is an image-space translation diagnostic, not a physically validated waveform perturbation or proof of robustness to laboratory noise. Use acquisition-justified perturbations for scientific robustness claims.
4. **Masking diagnostic.** Mask the top 10% of resized attention cells using the image's channel-mean baseline; compare the same target score drop with 10 equal-budget random masks. The prespecified primary diagnostic is `faithfulness_margin = attended_score_drop - mean(random_score_drop)`. Larger positive values suggest this particular attended mask affects this model more than these random controls. Negative differences are retained. Random-control SD is reported as Monte Carlo variability, not an independent sample size.

Masking can create out-of-distribution inputs, and scattered random masks can differ in shape from attended masks. Thus this is a simple diagnostic sensitive to the baseline/budget, not a validated causal measure or proof of physical relevance. An effect cannot be interpreted without these limitations. A shape-matched control or alternative baseline is a separate prespecified sensitivity analysis, not an automatic post-hoc adjustment to make the result favorable.

## Independent units and intervals

- Average defined image-variant diagnostics inside each original recording first. Each recording receives equal weight in the stratum mean, regardless of how many images it produced.
- Bootstrap recording IDs, or session IDs when the trained split uses session grouping. A resampled session carries all its recording-level values. Default: 1000 draws, fixed seed 42, 2.5/97.5 percentile intervals.
- Stratify summaries by model, true material and explanation target. No material-difference p-values or independent-fold t-tests are generated. Do not interpret two overlapping/non-overlapping bootstrap intervals as a paired model comparison.
- Defaults require at least 10 valid independent clusters for an interval. This is a conservative reporting rule, not a universal sufficient-sample theorem; assess uncertainty adequacy for the actual acquisition design. With fewer clusters, retain the descriptive mean/counts but leave the interval undefined.
- Intervals are conditional on the fitted model and declared diagnostic. They exclude training/selection variability and do not correct confounding or multiple exploratory comparisons. `faithfulness_margin` is marked primary; other diagnostic intervals are descriptive, not multiple confirmatory significance tests.
- Zero maps have no activation-mass localization: localization statistics are undefined, not zero. Flat maps cannot distinguish an attended region: masking and map-stability cosine are undefined. Report zero/flat fractions and valid/excluded image and recording counts. Recording-level means for a metric average only its defined variants; report this conditioning rather than concealing exclusions.
- Repeated image IDs and recording observations from different folds/checkpoints in the same estimate are rejected. A future paired VGG/ResNet comparison must align recording IDs/targets, resample the same recording/session clusters and report paired differences. This module does not fabricate that inference from two independently summarized model reports.

## Cohort and specification controls

Quantitative analysis requires the complete pipeline-produced prediction endpoint for one checkpoint, with original split membership rechecked. It refuses `--count` truncation, a substitute hand-selected CSV, changed split/image identities, incomplete runs, mismatched checkpoint hashes and a target changed from the frozen plan. It includes incorrect predictions; it does not select only attractive or correct examples. Recomputed model probabilities must agree with stored predictions.

Use final-fit holdout predictions for the main post-training analysis. Development-fold maps may be analyzed descriptively but do not become independent holdout evidence. No statistics are automatically run during training or the accuracy holdout evaluation. Masking/translation calls are additional diagnostic inference after endpoint selection; do not use the final diagnostics to retune models and still call the same holdout untouched.

The frozen specification hashes the numerical-statistics module and explanation pipeline and records a UTC freeze time. The output records the prediction-file hash, checkpoint identity, target, grouping unit, full selection and the plan. Keep the plan version fixed between VGG16 and ResNet50 when comparing them.

## Commands after training and mapping verification

From the repository root, using the project virtual environment:

```powershell
.\.venv\Scripts\python.exe code/explain_statistics.py freeze --config config/explainability_plan.json --output explainability_plan_frozen.json

.\.venv\Scripts\python.exe code/run_gradcam.py --run-dir runs/vgg16_final01 --checkpoint runs/vgg16_final01/best.keras --predictions runs/vgg16_final01/holdout_evaluation/predictions.csv --analysis-plan explainability_plan_frozen.json --output-dir runs/vgg16_final01/explainability01
```

The run paths are examples, not existing completed experiments. Use the same frozen plan and the corresponding ResNet final checkpoint/prediction paths for its analysis. A fresh output directory is required; existing outputs are never overwritten.

Outputs: numerical `.npy` maps; `selection.json`; `analysis_plan.json`; `analysis_context.json`; `activation_summary.csv`; `image_diagnostics.csv`; `recording_diagnostics.csv`; `explainability_statistics.csv`. The statistics file includes metric means, descriptive intervals, valid/excluded counts and independent-cluster counts. This supplements the mandatory old/new comparison; unavailable historical numeric maps cannot be recovered from colormapped composites or assigned invented statistics.

## Synthetic validation

Tests cover concentrated/uniform/zero maps, normalization invariance, coordinate regions, shift alignment, mask-budget equality/determinism, paired random-control differences, recording aggregation before bootstrapping, inadequate-cluster handling, exclusion counts, repeated-fold/image rejection and frozen-plan edits/overwrite protection. A small synthetic TensorFlow CNN exercises actual positive/negative-target gradients, stability inference, masking inference and undefined zero-map behavior. It trains no model and uses no research data.
