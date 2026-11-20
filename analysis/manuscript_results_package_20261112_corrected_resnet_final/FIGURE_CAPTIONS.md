# Figure captions

**Figure 1.** Development out-of-fold accuracy for five CNN configurations evaluated using identical grouped five-fold membership. Error bars represent 95% percentile intervals from 5,000 bootstrap draws clustered by `group_id`.

**Figure 2.** Receiver operating characteristic curves computed from the complete out-of-fold prediction probabilities of the common 1,280-image development set.

**Figure 3.** Aggregate out-of-fold confusion matrices. Rows denote true material class and columns denote predicted material class.

**Figure 4.** Comparison of accuracy, macro-averaged F1-score and ROC-AUC under the common development protocol.

**Figure 5.** MobileNetV3-Small ablation comparing a frozen ImageNet backbone with controlled fine-tuning of six late weighted layers while Batch Normalization layers remained frozen.

**Figure 6.** Reliability diagram comparing predicted-class confidence with observed accuracy. Expected calibration error (ECE) was calculated using 10 equal-width confidence bins.

**Figure 7.** Fold-specific validation-accuracy learning curves for all evaluated configurations. Different stopping epochs reflect validation-loss early stopping or the prespecified maximum epoch count.

**Figure 8.** Corrected experimental workflow. Grouped development folds were separated before augmentation; augmentation was restricted to training membership, and the locked holdout was reserved for final evaluation.

**Figure 9.** Descriptive comparison of historical reported accuracies and corrected results. Historical and corrected values were obtained using different validation procedures and must not be interpreted as a controlled before-after experiment.
