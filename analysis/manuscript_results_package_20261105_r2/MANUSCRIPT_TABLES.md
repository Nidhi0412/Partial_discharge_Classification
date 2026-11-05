# Manuscript tables

## Table 1. Common development-CV comparison

| Model | Accuracy (%) | Grouped 95% CI (%) | ROC-AUC | Macro precision (%) | Macro recall (%) | Macro F1 (%) | Nomex specificity (%) | Teflon sensitivity (%) | Brier score | ECE (10 bins) |
|---|---|---|---|---|---|---|---|---|---|---|
| VGG16 | 99.06 | 98.45–99.60 | 0.9995 | 99.07 | 99.06 | 99.06 | 99.53 | 98.59 | 0.0077 | 0.0064 |
| MobileNetV3-Small (fine-tuned) | 96.72 | 95.60–97.77 | 0.9947 | 96.72 | 96.72 | 96.72 | 96.88 | 96.56 | 0.0256 | 0.0202 |
| MobileNetV3-Small (frozen) | 95.31 | 94.02–96.61 | 0.9919 | 95.31 | 95.31 | 95.31 | 95.47 | 95.16 | 0.0391 | 0.0470 |
| ResNet50 | 78.75 | 76.22–81.09 | 0.8633 | 81.16 | 78.75 | 78.33 | 92.66 | 64.84 | 0.1523 | 0.0648 |
| Small CNN | 78.28 | 75.83–80.67 | 0.8914 | 80.17 | 78.28 | 77.94 | 90.78 | 65.78 | 0.1372 | 0.0493 |

## Table 2. Completed locked-holdout evaluations

| Model | Samples | Accuracy (%) | Grouped 95% CI (%) | ROC-AUC | Macro precision (%) | Macro recall (%) | Macro F1 (%) | Confusion matrix |
|---|---|---|---|---|---|---|---|---|
| VGG16 | 320 | 100.00 | 100.00–100.00 | 1.0000 | 100.00 | 100.00 | 100.00 | [[160, 0], [0, 160]] |
| ResNet50 | 320 | 77.81 | 72.09–82.58 | 0.8146 | 79.05 | 77.81 | 77.57 | [[141, 19], [52, 108]] |

## Table 3. Historical versus corrected results

| Model | Historical reported accuracy (%) | Corrected CV accuracy (%) | Corrected holdout accuracy (%) | Comparability |
|---|---|---|---|---|
| VGG16 | 98.20 | 99.06 | 100.00 | Historical predictions and true fold membership unavailable; descriptive only |
| ResNet50 | 97.03 | 78.75 | 77.81 | Historical fixed-split/repeated-fold concern; descriptive only |
