# Draft results write-up

## Comparative classification performance

This section compares the classification performance of VGG16, ResNet50, a small CNN baseline, and MobileNetV3-Small under a common grouped five-fold development protocol. The comparison was performed using the same 1,280 development images and identical fold membership for every model. Data augmentation was restricted to the training membership of each fold, whereas validation images were evaluated without augmentation. Complete out-of-fold probabilities were retained to calculate the classification measures, ROC-AUC values, calibration measures, and paired model comparisons.

As listed in Table 1, VGG16 produced the highest development accuracy of 99.06% (grouped-bootstrap 95% CI: 98.45–99.60%), with a ROC-AUC of 0.9995 and a macro F1-score of 99.06%. The fine-tuned MobileNetV3-Small model achieved the second-highest accuracy of 96.72% (95% CI: 95.60–97.77%), together with a ROC-AUC of 0.9947 and a macro F1-score of 96.72%. When the MobileNetV3-Small backbone remained frozen, the model achieved 95.31% accuracy. Fine-tuning six late weighted layers therefore improved accuracy by 1.41 percentage points while all Batch Normalization layers remained frozen.

The small CNN baseline achieved 78.28% accuracy and a ROC-AUC of 0.8914. The current ResNet50 configuration obtained a similar accuracy of 78.75%, although its ROC-AUC was lower at 0.8633. These baseline findings show that strong classification performance was not obtained automatically from the scalogram PNG inputs. In particular, the substantial separation between VGG16 and the small CNN indicates that model architecture and transfer-learning behaviour materially affected classification performance.

The aggregate confusion matrices further demonstrate the class-wise behaviour of the models. VGG16 correctly classified 637 of 640 Nomex images and 631 of 640 Teflon images. Fine-tuned MobileNetV3-Small correctly classified 620 Nomex and 618 Teflon images, producing balanced sensitivity and specificity. In contrast, ResNet50 and the small CNN showed lower Teflon sensitivity, with a greater number of Teflon images assigned to the Nomex class.

## Paired comparison and uncertainty

Because all models generated predictions for the same development images, performance differences were evaluated using paired predictions. Uncertainty intervals were calculated using 2,000 bootstrap resamples clustered by `group_id`. VGG16 exceeded fine-tuned MobileNetV3-Small by 2.34 percentage points, with a grouped 95% interval of 1.33–3.38 percentage points. VGG16 correctly classified 35 images that were misclassified by fine-tuned MobileNetV3-Small, whereas MobileNetV3-Small correctly classified five images that were misclassified by VGG16. The image-level exact McNemar result was p=1.38e-06; however, the grouped bootstrap interval is treated as the primary uncertainty summary because observations within a `group_id` cluster may remain dependent.

## Locked-holdout evaluation

Following development, the selected VGG16 and ResNet50 configurations were fitted using the development data with an internal grouped validation subset. Each resulting checkpoint was then evaluated once on the locked 320-image holdout set. VGG16 correctly classified all 320 holdout images, resulting in 100% accuracy and a ROC-AUC of 1.0000. ResNet50 correctly classified 249 of 320 images and achieved 77.81% accuracy with a ROC-AUC of 0.8146. The perfect VGG16 result is restricted to the present holdout and does not establish perfect performance under a new acquisition session, sensor configuration, voltage condition, or laboratory environment.

## Interpretation relative to the historical manuscript

The historical manuscript reported accuracies of 98.20% for VGG16 and 97.03% for ResNet50. These values should be presented separately from the corrected experiments because the original prediction probabilities, exact fold membership and recording-independent evaluation cannot be reconstructed completely. The corrected results use explicit grouped membership, training-only augmentation, saved out-of-fold probabilities and a locked final holdout. Consequently, the historical and corrected values provide a descriptive comparison rather than a controlled estimate of improvement or deterioration.

## Reporting limitation

The present analysis verifies separation according to the supplied image mapping and recorded `group_id` values. Source CSV paths, acquisition-session identifiers, calibrated time/frequency axes and complete wavelet-generation provenance are not available for every PNG. Accordingly, model performance should be described as classification of Nomex and Teflon scalogram images within the supplied dataset. Broader claims concerning new laboratory sessions or specific physical discharge mechanisms require additional acquisition-independent measurements.
