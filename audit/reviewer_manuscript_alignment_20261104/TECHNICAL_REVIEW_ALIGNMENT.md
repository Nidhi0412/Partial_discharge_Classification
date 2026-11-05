# Technical alignment: manuscript, reviewer comments and corrected pipeline

Reviewed 2026-11-04. The manuscript and reviewer DOCX files were read without
modification. No training was started and no pipeline code was changed by this
document review. The stopped run is not a completed corrected experiment.

Root: `D:\Nidhiiiiii\Partial_discharge_Classification_minimal\Partial_discharge_Classification_minimal`.

Sources: `Manuscript_EngineeringIAI_elsever.docx` and `Reviewers_comments.docx`.
Their SHA256 hashes and ordered extracted text are preserved in the same directory
as this report. Manuscript references Pxxxx below identify XML paragraph order;
review references P0001.Lxxx identify lines in the reviewer document's first
paragraph. These are not invented Word page numbers. Tables/equation text were
extracted, but no manuscript layout or figure-image visual approval is claimed.

## Main finding

The intended study is fine-tuned VGG16/ResNet50 classification of Nomex versus
Teflon from laboratory UHF-PD scalogram PNGs. Correcting training and evaluation
supports this study, but does not by itself validate wavelet selection, source
counts, physical augmentation, noise robustness, baseline superiority, confound
independence, novelty or deployment claims.

An immediate conflict was found: Reviewer 5 IX explicitly challenges horizontal
flipping and rotation (review L075), while the current `combined-image` launcher
profile restores them. Implementing these operations correctly does not answer
that objection. A reviewer-aligned primary profile should omit them or supply
development-only evidence/justification. A concrete proposal is saved in
`REVIEWER_ALIGNED_PNG_PROFILE_PROPOSAL.json`, but is NOT applied or launched.
It retains train-only row/column masking, bounded horizontal shift and pixel noise,
and disables the historical geometry stage and vertical stretching. Physical
validity and benefit still require appropriate evidence.

## Reviewer-by-reviewer technical status

### Evaluation protocol — R5 IV; manuscript P0132, P0138, P0144

**IMPLEMENTED AND TESTED; final experimental evidence pending.**

The manuscript alternates among five folds, a missing Fold 1, four-fold analysis,
and an 80/20 split. `code/dataset_manifest.py:validate_split/load_splits` and
`code/training_pipeline.py:run_training` now consume real saved membership and
fail on within-model recording/group/image/path/content overlap. `splits/run01`
has five development iterations, each with 1024 training, 256 validation and 320
fixed holdout PNGs. Every development image validates once; both backbones share
the split. New failures are not silently excluded from the average.

This implementation is DEVELOPMENT CV used for early stopping/settings selection,
followed by a final fit and one locked-holdout evaluation. It is not unbiased
nested outer CV. Rewrite the manuscript to describe that exact design. Do not
claim all five final research folds are complete because synthetic tests passed.
Evidence: `audit/training_pipeline_review/PNG_SPLIT_GUARD_VERIFICATION.json` and
`tests/test_pipeline.py`, `tests/test_training_safety.py`.

### Fine-tuning and overfitting — R1 bullets 4/9; manuscript P0131, P0139, P0148

**FINE-TUNING CONFIRMED; overfitting/generalization conclusions pending.**

`code/training_pipeline.py:build_model` unfreezes VGG block 5 and weighted non-BN
layers in ResNet50's last-15-layer window. Actual one-step checks prove unfrozen
backbone kernels and the head change, while the first frozen convolution stays
unchanged. This is not head-only feature extraction. Evidence:
`audit/training_pipeline_review/vgg16_finetuning_weight_check.json` and
`resnet50_finetuning_weight_check.json`.

Dropout, early stopping, smaller batches, frozen ResNet BN and checkpoint recovery
are implemented. They do not prove that the dataset is large enough, that no
overfitting occurs, or that features are material-specific. Use completed learning
curves and independent evaluation, with trainable parameter/layer counts from the
actual new architecture export. Avoid copying the manuscript's old 19/178 and
8/22 layer descriptions without recalculating them.

### Augmentation execution/meaning — R2 #5; R5 IX; manuscript P0142–P0147

**EXECUTION FIXED; scientific objection NOT FULLY ADDRESSED.**

The manuscript describes ordinary image augmentation followed by masking and
phase-aware circular shifts. The old custom generator had inconsistent positional
layer indexing and returned only the last separately computed variant. New
`code/scalogram_augmentation.py` composes actual stage outputs, fails on errors,
and saves seeds, parameters, intermediate tensors and parent-ID ledgers. Both
architectures passed 224x224 engineering training/recovery/export checks.

However, the current `combined-image` profile still includes rotation and
horizontal flipping, explicitly criticized in review L075. Vertical stretching
also needs justification if described physically. Restoring the old recipe and
addressing that reviewer concern are distinct tasks. The concrete profile proposal
omits those operations from the primary reviewer-aligned experiment and retains
their implementation as optional ablations. No changes were silently applied.

For the selected PNG-only route, call these image-space row/column occlusion,
horizontal image translation and pixel noise. Do not call them physical frequency
filtering, waveform phase shifts or calibrated signal AWGN. The revised shift is
non-circular and is not an exact reproduction of the old circular operation.
Augmentation must follow grouped splitting and use TRAIN members only; derived
examples retain the parent recording/group/label and are not independent samples.

### Label-dependent wavelet/level selection — R5 II; R1 bullets 2/7/8; R2 #2–4

**NOT RESOLVED BY CURRENT PNG RETRAINING; previously deferred.**

Manuscript P0091–P0103 explicitly describes class-specific detail-level selection.
Review L068 asks how an unknown incoming material could determine its own
preprocessing rule. A classifier split guard cannot undo label-dependent choices
already baked into reused PNGs.

The corrected `code/wavelet_pipeline.py` supports explicit generation settings,
but the current PNG training route does not invoke it. `image_mapping.csv` has no
filled source_csv_path, wavelet or true_dwt_level fields. Therefore current reuse
does not establish which wavelet/level generated each PNG or that label-dependent
generation was removed. This is not a declaration that current images necessarily
have that defect; their original generation remains unverified here.

The claim that maximum Shannon entropy or energy produces the most informative or
best-classifying features needs development-only comparative evidence. P0082's
db1–db4/symlet/entropy account, P0090's five-level statement, and the precise
DWT-to-CWT strategy must match real generation. Removing optimality language is
possible; replacing it with a fixed-db3 claim is truthful only if actual images
were generated that way. Do not invent a generation history to answer the review.

### Original-recording versus image counts — R1 bullet 6; R5 III

**UNRESOLVED MANUSCRIPT/PROVENANCE ACCOUNT.**

P0067 says 400 originals become four scalograms each, yielding 1600. P0099 says
eight scalograms per original and still reports 1600; those accounts cannot both
describe the same four-configuration, 100-original-per-configuration expansion.
The present mapping contains 1600 PNGs, 800 per material, with recording IDs based
on the user's confirmed distinct acquisitions. Those bookkeeping IDs do not
independently establish the historical 400-to-1600 derivation or its wavelet links.

Choose the truthful description for the actual dataset used in the new study,
consistent with source acquisition evidence and the user's confirmed recording
account. Do not reuse the old 400-recording expansion story by default, or count
online augmentation as new laboratory acquisitions. Missing raw links alone are
not evidence of confirmed recording leakage.

### Acquisition confounds — R1 bullet 5; R5 VII; manuscript P0067

**NOT IMPLEMENTED/NOT VERIFIED; previously deferred.**

`CURRENT_MAPPING_COVERAGE.json` records 1600 populated recording IDs but zero
populated electrode, session, source CSV, wavelet or true-level fields. Grouping
currently protects the stored recording/content identities; it does not establish
independence of unrecorded sessions or control voltage/sensor/electrode effects.
Simply merging electrode configurations into a material class, as in P0067, does
not prove geometry-independent recognition. Additional metadata and matched or
held-out-condition tests are needed for that claim; otherwise narrow the claim
to the tested acquisition conditions and disclose limitations.

### Noise-corrupted scalograms — R2 #6

**NOT ANSWERED BY TRAINING AUGMENTATION; evaluation pending/deferred.**

Training-time scaled RGB noise is a regularizer, not a measured robustness result.
No separate corruption benchmark currently runs in `code/train_overnight.py`.
For the chosen PNG route, a separately declared corruption test can evaluate
image-level robustness with aligned parent IDs and frozen models. It must not be
reported as waveform-SNR/laboratory-noise robustness without the raw signals,
noise definitions and sampling evidence. Never tune on corrupted holdout outcomes.

### Simpler and contemporary baselines — R1 bullet 10; R5 VIII

**NOT IMPLEMENTED; previously deferred.**

VGG16 versus ResNet50 alone does not answer this request. Additional comparators
must use the same independent-unit split and comparable development tuning. The
separate Machine Gnostics CSV study is not automatically a matched baseline for
this PNG experiment. Manuscript comparisons must not claim broad superiority
without those experiments or accurately state the restricted scope.

### Explainability — R2 #8

**TOOLS IMPLEMENTED/TESTED; scientific analysis NOT YET RUN.**

`code/explain_pipeline.py` and `code/explain_statistics.py` support checkpoint-
verified Grad-CAM, recording/group summaries, concentration/region statistics,
translation stability and attended-mask versus random-mask diagnostics. These
tools do not themselves constitute completed explanations or physical validation.
Freeze the analysis specification before inspecting final maps, run it after
training, interpret representative successes/failures and report limitations.
The current overnight queue has CV/final-fit/evaluate stages; it does not yet
automatically execute these explanation analyses.

### Confidence intervals and superiority — R5 V; manuscript P0133–P0135, P0355–P0357

**PARTIALLY ADDRESSED; old inference must be rewritten.**

The old t interval over four dependent CV scores does not justify statements that
future performance is guaranteed or that narrower bands prove generalization.
The new holdout evaluator supports recording/group-cluster bootstrap intervals
for accuracy, conditional on the fitted model. Those do not include model-
selection or training variability. AUC intervals and paired inferential model
comparisons are not automatically supplied by that accuracy bootstrap. Report
which uncertainties were actually computed and avoid unsupported significance.

### Intended task and practical claims — R5 VI; manuscript P0032, P0258, P0357, P0360

**MANUSCRIPT REWRITE REQUIRED.**

The labels are Nomex versus Teflon under PD acquisition conditions. They do not
encode healthy versus PD, defect category, PD severity, localization or remaining
life. Do not call these material-label predictions PD presence detection, healthy-
class separation, defect diagnosis or field deployment validation. Describe the
measured classification target and tested conditions precisely.

### Raw UHF preprocessing — R2 #7

**PNG PREPROCESSING CONFIRMED; RAW PREPROCESSING HISTORY UNVERIFIED.**

The current route decodes RGB, resizes to 224x224 and rescales by 1/255; the main
historical-compatible preset uses nearest resizing. It does not process raw UHF
CSVs. Explain signal preprocessing and scalogram generation only from genuine
source evidence. New generation code existing in the repository does not show
that old training PNGs were produced with those new settings.

### Wavelet/CNN equations and uncertainty foundations — R4; R1 bullet 3; R5 X

**TEXT/SOURCE REVIEW REQUIRED.**

Define variables, units and boundaries of the decomposition/entropy/frequency-
band/CI equations (P0074–P0102, P0134), and distinguish DWT details, CWT scalograms,
magnitude versus energy, sampling rate and mother-wavelet choices. A maximum
decomposition formula must account for the selected transform/filter constraints;
do not claim that existing image training validates these equations. Clarify the
handcrafted representation versus learned CNN features explanation. Relevant
uncertainty literature may support methods/limitations, but citations alone are
not an uncertainty experiment or evidence for unrelated claims.

### Novelty, contributions and conclusions — R1 bullet 1; R2 #1/#9; R4; R5 I

**NOT RESOLVED BY RETRAINING.**

P0006 calls the WT+CNN combination unique, and P0033–P0035 make broad superiority/
real-world claims. The reviewers challenge novelty explicitly. Frame contributions
around the actual, defensible study and comparison with closest relevant work.
Do not equate a corrected split, extra augmentation or high validation accuracy
with methodological novelty. Rewrite conclusions only after final evidence is
available. Nontechnical language/reference presentation issues are outside the
code correction itself.

## Specific manuscript corrections exposed by this review

- P0148 specifies ResNet LR 5e-5 and VGG LR 1e-4, while available June experiment
  configurations and the current launcher use 1e-5. The revised paper must use
  the actual new run configuration, not whichever historical number reads better.
- P0141 claims L2 for both models; the historical VGG binary head did not apply
  that parameter. The current historical model profile preserves that behavior.
  State per-model regularization accurately, or declare a deliberate new experiment.
- P0144's batch 16 differs from the machine-safe new batch 2. Document that
  intentional difference instead of claiming the new protocol is identical.
- P0258 says ResNet accuracy 97.27%; arithmetic from Table III gives 97.03125%.
  It also makes a no-false-positives/no-false-negatives statement inconsistent
  with the nonzero off-diagonal entries. VGG Fold 5's Table III accuracy is
  314/320=98.125%, not its printed 98.143%. Table IV ResNet Fold 4 also differs
  from Table III's 312/320=97.5%. `OLD_TABLE_ARITHMETIC.json` records these checks.
  These are arithmetic checks of supplied counts, not reproduction of historical
  prediction probabilities, architectures or evaluation independence.
- Distinguish SD from confidence-interval half-width everywhere ± notation is
  used; regenerate all revised tables/figures from the same saved predictions.

## Practical scope for the next iteration

Proceed with the unchanged original PNGs only under an honest description of
their provenance and generation limitations. Keep the fine-tuning and reviewed
split/output safeguards. Apply a reviewer-aligned augmentation choice after
reviewing the explicit profile proposal; do not automatically relaunch the old
combined profile. Resolve or explicitly defer wavelet provenance/counts, baselines,
corruption testing and acquisition confounds. Run explainability after training.

Existing code is substantial progress on reliable execution. It is not yet a
complete experimental response to all technical reviewer comments. The original
DOCX files and stopped research outputs remain untouched.
