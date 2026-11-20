# Statistical interpretation for manuscript correction

## Analysis scope

This analysis compares saved out-of-fold (OOF) predictions for the common 1,280-image development set. Every model was evaluated with identical five-fold membership. The 95% confidence intervals were calculated using 5,000 bootstrap resamples clustered by `group_id`. Exact McNemar tests are reported as secondary image-level paired tests because residual dependence may remain among images belonging to the same group.

## Principal corrected results

| Model | OOF accuracy | Grouped-bootstrap 95% CI | ROC-AUC | Macro F1 |
|---|---:|---:|---:|---:|
| ResNet50, corrected fine-tuned | 99.45% | 99.01–99.84% | 0.99998 | 99.45% |
| VGG16 | 99.06% | 98.44–99.60% | 0.99952 | 99.06% |
| ResNet50, corrected frozen head | 97.03% | 95.96–98.00% | 0.99603 | 97.03% |
| MobileNetV3-Small, fine-tuned | 96.72% | 95.60–97.79% | 0.99470 | 96.72% |
| MobileNetV3-Small, frozen | 95.31% | 93.99–96.59% | 0.99188 | 95.31% |
| ResNet50, legacy preprocessing | 78.75% | 76.26–81.13% | 0.86328 | 78.33% |
| Small CNN | 78.28% | 75.81–80.70% | 0.89144 | 77.94% |

## VGG16 versus corrected fine-tuned ResNet50

Corrected fine-tuned ResNet50 exceeded VGG16 by 0.39 percentage points in OOF accuracy. Expressed as VGG16 minus ResNet50, the paired difference was -0.39 percentage points, with a grouped-bootstrap 95% interval from -1.03 to +0.23 percentage points. The interval includes zero. VGG16 alone was correct for five images, whereas corrected ResNet50 alone was correct for ten images; the image-level exact McNemar test gave p = 0.302.

The corresponding AUC difference (VGG16 minus corrected ResNet50) was -0.000464, with grouped-bootstrap 95% limits of -0.000979 to -0.000117. Although this interval favors corrected ResNet50, the absolute difference is very small and should not be described as a large practical improvement.

## Defensible manuscript claim

Under the common grouped development protocol, both VGG16 and corrected fine-tuned ResNet50 achieved very high discrimination. ResNet50 had the highest point estimates for accuracy and ROC-AUC. However, its accuracy advantage over VGG16 was not statistically established by the grouped-bootstrap accuracy-difference interval or the secondary McNemar test. The two architectures should therefore be reported as having comparable classification accuracy, while noting ResNet50's marginally higher AUC point estimate.

## Claims to avoid

- Do not claim that corrected ResNet50 is statistically superior to VGG16 in accuracy.
- Do not compare the legacy ResNet50 result with corrected ResNet50 as if preprocessing were the only changed factor unless every other training setting is explicitly controlled.
- Do not combine historical manuscript percentages with the corrected OOF results in an inferential statistical test; the historical prediction-level evidence and exact fold membership are unavailable.
- Do not describe OOF results as external validation. The locked holdout is an internal holdout from the supplied dataset, not a new acquisition session.

## Provenance

Machine-readable point estimates, intervals, paired comparisons, calibration data and SHA-256 hashes of every input prediction file are stored beside this document in `model_metrics.csv`, `paired_comparisons.csv`, `comparison.json`, and `provenance.json`.
