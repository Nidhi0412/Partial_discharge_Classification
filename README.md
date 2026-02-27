# UHF Partial Discharge Insulation Classification

**Wavelet-scalogram-based image classification using pretrained VGG16 and ResNet50, with 5-fold cross-validation and Grad-CAM interpretability.**

Repository: [github.com/Nidhi0412/Partial_discharge_Classification](https://github.com/Nidhi0412/Partial_discharge_Classification)

---

## 1. Overview

This project classifies **insulation type** (Nomex vs Teflon) from UHF partial discharge (PD) signals by:

1. Converting raw PD signals (CSV) into **wavelet scalograms** (time–frequency images).
2. Training **VGG16** and **ResNet50** (ImageNet pretrained) on scalogram images with **5-fold stratified cross-validation**.
3. Evaluating with **ROC-AUC** and **Grad-CAM** for interpretability.

**Best model: VGG16** — 98.20% accuracy, 0.9981 ROC-AUC.

---

## 2. Dataset

| Item | Description |
|------|-------------|
| **Classes** | 2: Nomex, Teflon |
| **Total samples** | 1600 (e.g. 800 per class) |
| **Split** | 5-fold stratified CV (80% train/val, 20% test per fold) |
| **Input** | CSV per signal: time (column 1), signal (column 2); first 20 rows skipped |
| **Augmentation** | Rotation, zoom, flip, brightness; scalogram-specific (frequency/time masking, phase shift, etc.) |

Data layout for training:

- `Data for Classification/Nomex/` and `Teflon/` (or TN/TF/NN/NF folders) with CSV files.
- After scalogram generation: `Scalograms/Nomex/`, `Scalograms/Teflon/` with PNG images.

Full dataset description and availability can be linked here (e.g. Zenodo/figshare or “on request”).

---

## 3. Method

### 3.1 Pipeline

1. **Scalograms:** Continuous wavelet transform (e.g. PyWavelets) → save as PNG (e.g. 224×224).
2. **Training:** VGG16 or ResNet50 (ImageNet weights), fine-tuning last layers, dropout, L2; 5-fold CV; EarlyStopping, ReduceLROnPlateau.
3. **Evaluation:** Accuracy, precision, recall, F1; ROC curves and AUC per fold.
4. **Interpretability:** Grad-CAM heatmaps on last conv layer to show which frequency/time regions the model uses.

### 3.2 Key scripts (in `code/`)

| Script | Purpose |
|--------|---------|
| `generate_scalograms.py` | Generate scalograms from CSV (skip 20 rows, time + signal columns). |
| `generate_clean_scalograms.py` | Clean scalogram pipeline. |
| `main.py` | Prepare data, train/val/test split. |
| `model_training_vgg16.py` | VGG16 model build and training (insulation). |
| `run_parallel_training_vgg16.py` | Run 5-fold VGG16 training. |
| `model_training.py` | ResNet50 training (same pipeline). |
| `run_parallel_training.py` | Run 5-fold ResNet50 training. |
| `eval_roc_auc.py` | Compute ROC-AUC and save curves. |
| `make_foldwise_roc.py` | Fold-wise ROC plots. |
| `run_gradcam.py` | Generate Grad-CAM heatmaps. |
| `gradcam_visualization.py` | Plot and save Grad-CAM. |

---

## 4. Results

### 4.1 Main results (insulation classification)

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|-------|----------|---------|-----------|--------|-----|
| **VGG16** | **98.20%** | **0.9981** | 97.67% | 96.41% | 97.01% |
| ResNet50 | 97.03% | 0.9966 | 97.67% | 96.41% | 97.01% |

- **VGG16** is consistently better (accuracy +1.17%, AUC +0.0015).
- Both show very high AUC (near-perfect discrimination).

### 4.2 Where to find results in this repo

| Content | Location |
|---------|----------|
| One-page summary | `results/KEY_FINDINGS_ONE_PAGE.md` |
| VGG16 fold-wise reports, learning curves | `results/VGG16/` (reports and figures only; no model weights in repo) |
| ResNet50 Grad-CAM heatmaps, crossval metrics | `results/ResNet50/`, `results/ResNet50/gradcam_results/` |
| ROC curves (VGG16, ResNet50, comparison) | `results/roc_foldwise_VGG16.pdf`, `roc_foldwise_ResNet50.pdf`, `roc_comparison_vgg16_resnet50.pdf` |
| AUC summary | `results/auc_foldwise_summary.csv` |
| Statistical comparison (VGG16 vs ResNet50) | `results/statistical_analysis_results.json`, `results/statistical_comparison.png` |

Trained model weights (`.h5`) are not included to keep the repo small; they can be shared via Zenodo or on request.

### 4.3 Grad-CAM findings

- **Nomex:** Model focuses on DC1 and DC4 frequency bands.
- **Teflon:** Model focuses on DC4 and DC5 frequency bands.
- Attention aligns with wavelet energy; supports that the model uses material-specific features.

---

## 5. How to run

### 5.1 Environment

```bash
pip install -r requirements.txt
```

Requirements: TensorFlow ≥2.15, Keras, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, Pillow (see `requirements.txt`).

### 5.2 Generate scalograms

1. Place CSV files in `Data for Classification/Nomex/` and `Teflon/` (or your class folders).
2. Run:

```bash
python code/generate_scalograms.py
```

Output: `Scalograms/<class>/` with PNG images.

### 5.3 Train VGG16 (5-fold)

Configure paths and parameters in the training script if needed, then:

```bash
python code/run_parallel_training_vgg16.py
```

Models and logs will be saved per fold (e.g. in `results/VGG16/` or a path you set).

### 5.4 Evaluate ROC-AUC

After training:

```bash
python code/eval_roc_auc.py
python code/make_foldwise_roc.py
```

### 5.5 Generate Grad-CAM

```bash
python code/run_gradcam.py
```

Heatmaps will be saved in the path configured in the script (e.g. `results/ResNet50/gradcam_results/`).

---

## 6. Repository structure

```
.
├── README.md                 ← This file
├── requirements.txt
├── .gitignore
├── code/                     ← Insulation classification + VGG16/ResNet50 scripts
│   ├── generate_scalograms.py
│   ├── generate_clean_scalograms.py
│   ├── scalogram_utils.py
│   ├── main.py
│   ├── main_clean_scalograms.py
│   ├── split_dataset.py
│   ├── model_training_vgg16.py
│   ├── run_parallel_training_vgg16.py
│   ├── model_training.py
│   ├── run_parallel_training.py
│   ├── eval_roc_auc.py
│   ├── make_foldwise_roc.py
│   ├── run_gradcam.py
│   ├── gradcam_visualization.py
│   ├── generate_vgg_gradcam_for_dashboard.py
│   ├── evaluation.py
│   └── crossval_aggregate_evaluation.py
└── results/                  ← VGG16/ResNet50 results, ROC, Grad-CAM, summaries
    ├── KEY_FINDINGS_ONE_PAGE.md
    ├── README_START_HERE.md
    ├── auc_foldwise_summary.csv
    ├── statistical_analysis_results.json
    ├── roc_foldwise_VGG16.pdf
    ├── roc_foldwise_ResNet50.pdf
    ├── roc_comparison_vgg16_resnet50.pdf
    ├── statistical_comparison.png
    ├── VGG16/
    └── ResNet50/
        ├── gradcam_results/
        ├── crossval_metrics.csv
        ├── crossval_summary.txt
        └── experiment_config.json
```

---

## 7. Citation and contact

If you use this code or results, please cite the paper (add title, journal, year, DOI when published).  

Dataset/code: [github.com/Nidhi0412/Partial_discharge_Classification](https://github.com/Nidhi0412/Partial_discharge_Classification).
