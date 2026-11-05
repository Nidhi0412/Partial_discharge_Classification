# Reviewer coverage and required old/new results comparison

Recorded 2026-11-03. This requirement remains part of the task after full training: compare the historical and corrected experiments, preserving differences in provenance and evaluation protocol. No new research training/results exist yet.

Quantitative explainability tooling is now implemented: see `EXPLAINABILITY_ANALYSIS_METHODS.md` and `config/explainability_plan.json`. It includes recording/session-aware descriptive intervals, concentration/region shares, aligned stability and a masking-versus-random diagnostic. Its proposal must be frozen before final-map inspection; no research maps/statistics have been produced. It does not resolve the missing wavelet-selection ablation or prove physical validity.

Source reviewed directly: `D:\Nidhiiiiii\Paper_04_Saurabh_wavelet\Paper_04_Saurabh_wavelet\Engineering of AI_elsver\Reviewers_comments.docx`, paragraph 1 (contains all scientific reviewer comments). Reviewer identifiers below refer to its original numbering/bullet order. Code changes and smoke-test success are implementation evidence, not experimental resolution of a reviewer concern.

## What the corrected workflow can address after actual execution

- **R5-II: class-dependent coefficient selection/label leakage.** The new optional raw generator uses the same explicitly supplied wavelet/level variants for every recording, not material-dependent selection. This addresses the new generator's rule; the existing PNGs' historical selection remains unknown. Retraining on existing PNGs alone does not establish label-independent image construction.
- **R5-III: 400 originals versus four/eight variants and 1600 images.** The new manifest can count actual unique recordings and variants once trustworthy acquisition IDs are supplied. The current template does not resolve the historical arithmetic or establish 400 independent recordings.
- **R5-IV: five/four folds, missing Fold 1, fixed 80/20 split.** The corrected code consumes actual grouped development-fold membership, accounts for every prescribed fold and fails explicitly on errors. The supported protocol is development CV for settings/early stopping, then final fit and a locked independent holdout. Do not call development scores unbiased nested outer-CV test performance.
- **R5-V and R4 uncertainty concern.** The new evaluator provides recording/session cluster-bootstrap holdout accuracy uncertainty conditional on the fitted model. This replaces naive independent-fold interpretation for that endpoint. It does not quantify training/model-selection variability or retrospectively validate old intervals.
- **R1 limited data/overfitting and R5-VII confounding.** Grouped acquisition-level splitting and actual learning curves improve evaluation. They do not prove absence of overfitting or eliminate electrode/voltage/session/sensor confounds. Acquisition metadata, factor balance and appropriate cross-condition tests remain necessary.
- **R2-5 and R5-IX augmentation.** The effective training transform is explicit and training-only. Rotation/time reversal/frequency stretching are not automatically enabled. Optional bounded horizontal translation must still be scientifically justified and compared with a no-augmentation baseline on identical splits; smoke testing is not such an ablation.
- **R2-7 preprocessing; R2-4/R5-X wavelet conventions.** New raw generation records actual detail levels, coefficient spacing, representation and normalization. Existing images are not automatically regenerated or reconciled by code changes. Scientific band interpretations and manuscript equations still require review.
- **R2-8 explainability.** The new Grad-CAM path uses explicit binary targets and matching model preprocessing and saves numerical maps/selection metadata. Actual maps and a carefully limited explanation section must still be produced after model evaluation. Maps do not prove a physical mechanism.

## Reviewer requests that retraining two CNNs does not by itself resolve

### Reviewer 1 (original bullet order)

1. Novelty of combining wavelets and deep networks: revise claims against closest literature; code fixes do not establish novelty.
2. Proof that selected coefficients are most informative: requires a controlled wavelet/level ablation using development groups only.
3. Meaning of the manual feature-extraction statement: clarify manuscript terminology.
4. Justification for deep networks with limited independent data: compare simple baselines and report generalization/overfitting evidence.
5. Variation among the 100 laboratory recordings and apparently separable classes: recover acquisition conditions, show representative/raw signal variability and compare classical methods.
6. Multiple wavelets from one sample treated as more data: keep parent recordings together; distinguish representation variants from independent acquisitions. Test whether multiple variants actually help.
7. Maximum Shannon entropy as a classification criterion: not justified by code or smoke tests. Either provide a training-only tested criterion/ablation or remove the unsupported optimality claim.
8. Energy versus entropy level-selection contradiction: resolve the actual scientific procedure before preprocessing is claimed correct.
9. Trainable parameter counts/overfitting: report actual trained layers/parameters, curves and independent evaluation; regularization alone is not proof.
10. Comparisons with other methods/models: new comparable baselines still needed unless existing matched evidence can be recovered.
11. Missing transfer-learning/wavelet references: manuscript/literature work.
12. Language/presentation: manuscript editing.

### Reviewer 2

1. Introduction, state of the art and contributions: manuscript/literature work.
2. Two scalogram-generation strategies: identify the actual intended strategies and compare them if both are claimed. The new shared generator does not supply that experimental comparison.
3. Wavelet modulation and higher-order Daubechies choice: requires signal-based justification/tests and appropriate wavelet ablation; not resolved by correcting coefficient indexing.
4. Number of decomposition levels: explicit in new configuration; reconcile with final input data/manuscript.
5. Actual augmentation techniques: implementation now explicit; document only applied transforms and verify their benefits separately.
6. Noise-corrupted scalograms: robustness testing is not yet implemented/executed for this corrected experiment. Define physically meaningful noise levels/where noise is injected, then evaluate matched held-out recordings without tuning on the test set.
7. Required UHF preprocessing: specify validated parsing, acquisition intervals, normalization and any actual denoising; do not invent a denoising step.
8. Explainability: corrected tooling is available; post-training maps and interpretation remain to do.
9. Conclusion: revise only after verified results exist.

### Reviewer 4

Introduction/accessibility/state of the art/novelty, limitations of existing work and justification, uncertainty foundations, equation-variable definitions, conclusions and reference quality/relevance all require manuscript work. Include only scientifically relevant literature; requested citations are not automatically methodological evidence. Recording-aware uncertainty in the new code addresses one computational component, not the full theoretical discussion.

### Reviewer 5

I. Novelty: manuscript/literature evidence, unresolved by retraining.
II. Class-specific coefficients: new generator avoids using the unknown class to choose a band, but existing-image provenance still must be resolved.
III. Counts/provenance: requires trusted mapping, not just 1600 files.
IV. Validation protocol: implemented explicitly; verify with completed manifests/results after training.
V. Statistical independence: recording/session-aware holdout uncertainty implemented; actual independent units and assumptions still need evidence.
VI. Practical scope: state insulation-material classification, not PD presence detection, defect-type identification or severity estimation.
VII. Electrode/voltage/session/magnitude/sensor shortcuts: recover metadata and run appropriate controlled/transfer diagnostics; grouping alone cannot separate confounded factors.
VIII. Simpler and recent baselines: require comparable experiments with identical recording groups and training-only feature fitting.
IX. Physical augmentation validity: constrain and justify transforms; run matched ablations.
X. Entropy optimality and inconsistent formulas/metrics/confusion descriptions: require wavelet-selection evidence and numerical/manuscript reconciliation, beyond the software fixes.

## Mandatory post-training deliverables

After completed VGG16 and ResNet50 final runs, produce `OLD_VS_NEW_COMPARISON.md` and `OLD_VS_NEW_COMPARISON.csv` in a new results directory. Include:

1. Historical reported/saved values, corrected values, original artifact paths and verification status. Starting historical claims: VGG16 accuracy 98.20%, AUC .9981; ResNet50 accuracy 97.03%, AUC .9966. Historical precision/F1, intervals and Fold 5 values must be sourced from original reports rather than copied from the old README.
2. Actual architecture/checkpoint IDs; independent recording counts; derived-image counts; class/configuration/session balance; manifest hashes; preprocessing/wavelet provenance; grouped validation/holdout protocol; augmentation; seeds; trainable parameters; selected epochs; checkpoint-selection policy.
3. Image- and recording-level accuracy, per-class precision/recall/F1, specificity, macro metrics, confusion matrices, ROC/AUC and documented recording/session uncertainty. Report denominators and avoid treating variants/repeated test predictions as new independent observations.
4. Old/new learning curves where histories exist, clearly labeled by differing datasets/protocols. Do not present their numerical differences as a controlled ablation when multiple factors changed.
5. Signed changes in percentage points for accuracy and absolute AUC differences, with explicit comparability labels. Historical-summary comparison is descriptive; it is not evidence of improvement or degradation caused by the fixes.
6. A paired comparison of the NEW VGG16 and ResNet50 on identical locked holdout recordings, with recording/session-aware paired uncertainty if scientifically justified. Do not create an independent-fold significance test from the development runs.
7. An updated reviewer response map: which claims are now supported by numerical evidence, which require additional baselines/wavelet/noise/confound experiments, and which remain limitations.

## Comparability rules

- A lower corrected accuracy must be reported honestly; no target to reproduce or exceed 98.20%/97.03% is imposed.
- Historical and corrected scores can be placed side by side even when protocols differ, but must be labeled as different experimental estimates rather than a fair paired contest.
- Rerunning an old checkpoint on the new holdout is only a valid independent comparison if that model did not train on those recordings or their variants. Historical memberships remain incomplete, so do not assume a newly designated holdout is independent of the old model.
- The recovered June VGG16 checkpoint set can support diagnostic prediction reproduction if separately authorized; the exact old report-producing checkpoint still needs linkage. The May ResNet checkpoint has not been recovered. Missing artifacts must remain NOT VERIFIABLE rather than be replaced with new model outputs.
- Full old/new comparison remains pending actual corrected training, authoritative recording mapping and approved evaluation. Synthetic smoke-test accuracies are not research results and must never enter this comparison.
