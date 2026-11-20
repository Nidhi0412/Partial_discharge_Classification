# Grad-CAM explainability results

## Methods

True-class Grad-CAM was calculated from the final convolutional layer (`block5_conv3` for VGG16 and `conv5_block3_out` for corrected ResNet50) for every image in the 320-image locked internal holdout. The analysis specification was frozen before map inspection. It included 1,000 `group_id`-clustered bootstrap resamples, horizontal translation stability at -2 and +2 pixels, and a faithfulness diagnostic that masked the most activated 10% of pixels and compared the true-class score reduction with ten equal-area random masks. All summaries include the complete cohort rather than selected correct predictions.

## Results

Both models produced highly translation-stable non-flat activation maps. Mean stability cosine was 0.997 for VGG16-Nomex and 0.991 for ResNet50-Nomex. Prediction-change rates under the small translations were zero for VGG16 and ResNet50-Nomex and 0.003 for ResNet50-Teflon.

For Nomex, attended-region masking reduced the true-class score more than equal-area random masking. The mean faithfulness margin was 0.0106 (95% grouped interval 0.0061 to 0.0156) for VGG16 and 0.0006 (0.0000 to 0.0018) for ResNet50.

True-class Teflon Grad-CAM was frequently zero/flat at the selected final layer: 86.9% of VGG16 maps and 78.1% of ResNet50 maps. Faithfulness and stability summaries for this stratum therefore use only valid non-flat maps and must not be generalized to all Teflon images. This behavior is compatible with saturated binary outputs and the positive-gradient/ReLU definition of Grad-CAM, but the present analysis cannot identify a unique cause.

## Interpretation and limitations

The maps show where final-layer activations support each model's true-class score in image coordinates. They do not demonstrate a physical discharge mechanism, calibrated frequency band or causal material signature. The PNGs lack complete calibrated time/frequency axes, and masking or translations can create out-of-distribution images. Grad-CAM results should therefore be described as model-attention diagnostics. The high zero-map rate for Teflon is an explicit method limitation; it must not be hidden by showing only attractive non-flat examples.

The representative panels use a deterministic rule: the non-flat map closest to the within-model/material medians of faithfulness margin and normalized entropy. The ResNet50 error panel contains every misclassified holdout image. Quantitative conclusions come from the complete cohort, not the displayed examples.
