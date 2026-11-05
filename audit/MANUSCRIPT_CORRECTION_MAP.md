# AUT manuscript correction map

Scope: section names supplied in the audit request; actual manuscript text, tables, figure numbers and reviewer files were not available. Findings refer to CODE_AUDIT_REPORT.md. No manuscript was edited.

## Section 2 — Data acquisition

F01/F02: describe 400 recordings and 1600 images as historical claims pending acquisition and lineage verification. Separate recordings, derived images and training-time augmented views. Recover original recording IDs, electrode mappings (NN/TN versus NP/TP terminology must be verified), session, applied voltage, instrument sample interval, sensor bandwidth and recording exclusions. The source loops produce five db3 detail-position images per accepted recording, so the missing final-selection step must explain 1600. Do not call 1600 images independent experimental observations.

F02/F05: replace any unsupported independent partition claim with the actual evidence limitations. Document recording/session grouping and electrode/voltage balance only after manifests exist. A saved 1024/256/320 report does not validate unique raw-source separation.

## Section 3 — Wavelet decomposition and scalograms

F06–F10: distinguish fixed db3, five-level DWT and magnitude Morlet CWT in supplied scripts from the claimed db1–db4, energy/entropy selection pipeline. Provide actual selection code before asserting class-independent feature selection. Correct positional labels: dc1=cD5, dc2=cD4, dc3=cD3, dc4=cD2, dc5=cD1 for the five-level files. Explain downsampling and calibrated axes; do not attach RF-band interpretations to the current decimated-coefficient frequency labels. Review scalogram figures, coefficient tables, colorbar labels, energy formulas and any DC-band physical conclusions.

Document per-image autoscaling, jet rendering, clean versus annotated generation and image resizing. Correcting terminology alone can be documented; changing the scientific representation or actual input bands requires reprocessing and fresh evaluation. Clean coefficient CSVs were overwritten by level iterations and cannot validate all bands.

## Section 4 — CNN methodology

F03–F05: supplied VGG16 fold loop repeatedly uses the same directory split; do not describe it as effective five-fold stratified CV. Four available saved runs do not establish five completed folds. Present a new grouped protocol only as planned until executed. Separate development CV, validation for early stopping, and a locked independent holdout.

F11–F13: list only standard ImageDataGenerator transformations supported by the active path. Remove unverified claims that specialized masking/stretching/noise reached training. State that graph/shape/label preservation and physical justification require validation before enabling them.

F14–F17/F22: describe source architecture precisely: VGG block5 fine-tuning, GAP/dropout/binary sigmoid, no binary VGG L2 in supplied implementation; last 15 Keras ResNet layers rather than 15 convolution layers; declared versus historically confirmed hyperparameters. The supplied lowercase VGG CLI dispatches to ResNet50 due to an uppercase comparison: model identity must be recovered from historical checkpoints before claiming the VGG constructor produced the saved VGG results. Disclose absent checkpoints/logs and launcher/task mismatch. State RGB/255 preprocessing for the active training path; standard ImageNet preprocessing was not used there. Identify checkpoint selection criteria (val_accuracy checkpoint versus val_loss restoration).

F21: define class-specific Grad-CAM scores for binary sigmoid, matching preprocessing and chosen layer/model, sample-selection rule and heatmap normalization. Current positive-output maps for both materials do not support negative-class attention claims.

## Section 5 — Results and statistical analysis

F18: headline 98.20%/97.03% and .9981/.9966 match saved-summary arithmetic after rounding, but are not prediction-verified, recording-independent estimates. Preserve this distinction in text and table captions. The exact report-implied VGG mean is 98.203125%; do not treat rounding differences as a substantive model change. The available Fold 5 report implies 314/320 correct; another reported value requires its source before resolution.

Fix the VGG precision/F1 entries copied from ResNet-like values, subject to original-prediction confirmation. Clearly label Teflon-positive metrics separately from macro metrics. Report four available runs, and distinguish conditional reconstructed confusion counts from original observed predictions. Do not publish the audit's reconstructed matrices as newly measured results.

F05/F19: pooled repeated test predictions do not increase independent sample size. Existing t-based intervals/p-values do not establish recording-level uncertainty or model superiority. Wilcoxon JSON p=.125 does not support a significance claim. Use paired grouped holdout uncertainty after provenance recovery. Define SD convention and distinguish SD bands from confidence intervals; don't describe padded learning curves as confidence intervals.

F20/F21: baseline comparisons and noise/wavelet ablations remain unverified. Identify ResNet versus VGG Grad-CAM evidence correctly; remove mechanistic proof and unsupported named-band claims. Review Grad-CAM captions in light of preprocessing mismatch, constant positive target and ten selected examples. Raw heatmap statistics cannot be reproduced from rendered composites.

## Section 6 — Conclusion

F01–F05/F18–F21: limit conclusions to what current artifacts support: saved image-level summaries show high reported discrimination in an incompletely documented fixed-split experiment. Recording-independent generalization, statistical superiority, robustness and physical-mechanism learning are not established. Do not claim that the audit invalidates all historical outputs; it identifies source defects, arithmetic consistency and unresolved experimental evidence separately. Update abstract/result/conclusion claims consistently only after the recovery and approved validation stages.
