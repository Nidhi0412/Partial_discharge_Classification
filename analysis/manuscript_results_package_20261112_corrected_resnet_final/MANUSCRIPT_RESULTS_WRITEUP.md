# Draft results write-up

## Comparative classification performance

VGG16, corrected fine-tuned ResNet50, a small CNN baseline and two MobileNetV3-Small configurations were compared under a common grouped five-fold development protocol. All models used the same 1,280 development images and identical fold membership. Augmentation was restricted to each fold's training membership, while validation images remained unaugmented. Complete out-of-fold probabilities were retained for classification, ROC-AUC, calibration and paired analyses.

Corrected fine-tuned ResNet50 produced the highest development point estimate: 99.45% accuracy (grouped-bootstrap 95% CI: 99.01-99.84%), ROC-AUC 1.0000 and macro F1 99.45%. VGG16 achieved 99.06% accuracy (95% CI: 98.44-99.60%), ROC-AUC 0.9995 and macro F1 99.06%. Fine-tuned and frozen MobileNetV3-Small achieved 96.72% and 95.31% accuracy, respectively. The small CNN achieved 78.28% accuracy and ROC-AUC 0.8914, showing that near-perfect classification was not obtained automatically from the supplied scalogram images.

Corrected ResNet50 correctly classified all 640 Nomex images and 633 of 640 Teflon images. VGG16 correctly classified 637 Nomex and 631 Teflon images. Fine-tuned MobileNetV3-Small correctly classified 620 Nomex and 618 Teflon images.

## Paired comparison and uncertainty

Uncertainty intervals used 5,000 bootstrap resamples clustered by `group_id`. Corrected ResNet50 exceeded VGG16 by 0.39 percentage points, but the grouped 95% accuracy-difference interval included zero (-0.23 to 1.03 percentage points for ResNet50 minus VGG16). The secondary exact image-level McNemar test gave p=0.302. The models are therefore described as having comparable accuracy, not statistically established accuracy superiority. VGG16 exceeded fine-tuned MobileNetV3-Small by 2.34 percentage points (grouped 95% interval: 1.33 to 3.41).

## Locked-holdout evaluation

VGG16 and corrected fine-tuned ResNet50 were fitted using the development data with an internal grouped validation subset and evaluated once on the locked 320-image holdout. VGG16 classified all 320 images correctly (100% accuracy; ROC-AUC 1.0000). Corrected ResNet50 classified 318 of 320 images correctly (99.38% accuracy; ROC-AUC 1.0000). These internal-holdout results do not establish equivalent performance under a new acquisition session, sensor configuration, voltage condition or laboratory environment.

## Interpretation relative to the historical manuscript

Historical accuracies of 98.20% for VGG16 and 97.03% for ResNet50 are reported separately because the original prediction probabilities, exact fold membership and recording-independent evaluation cannot be completely reconstructed. The corrected experiments used explicit grouped membership, training-only augmentation, saved OOF probabilities and a locked holdout. Historical-versus-corrected comparisons are descriptive rather than controlled estimates of improvement.

## Reporting limitation

Separation is verified according to the supplied image mapping and `group_id` values. Source CSV paths, acquisition-session identifiers, calibrated time/frequency axes and complete wavelet-generation provenance are unavailable for every PNG. Performance should therefore be described as Nomex-versus-Teflon scalogram classification within the supplied dataset. Broader claims require acquisition-independent measurements.
