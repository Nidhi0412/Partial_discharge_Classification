# Grad-CAM figure captions

**Figure 10.** Representative true-class Grad-CAM maps from the locked internal holdout. One non-flat map per model and material was selected deterministically as the observation closest to the stratum medians of faithfulness margin and normalized entropy. The displayed examples are illustrative; quantitative conclusions use all 320 holdout images per model.

**Figure 11.** True-class Grad-CAM maps for all corrected ResNet50 errors on the locked internal holdout. Both errors were Teflon images classified as Nomex. Grad-CAM identifies image regions that locally support the true-class score at the final convolutional layer; it does not establish a physical discharge mechanism.

**Figure 12.** Full-cohort Grad-CAM diagnostics for VGG16 and corrected ResNet50. Translation stability was evaluated after horizontal shifts of -2 and +2 pixels. Faithfulness margin is the true-class score reduction after masking the most activated 10% of pixels minus the mean reduction from ten equal-area random masks. Stability and faithfulness are calculated only for non-flat maps. True-class Teflon maps were frequently zero/flat, which is reported as a method limitation rather than excluded from the cohort accounting.
