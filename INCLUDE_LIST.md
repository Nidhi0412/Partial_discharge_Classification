# What to Include — Minimal Repo (Insulation Classification + VGG16 + Results + README)

Use this list to copy **only** these into your GitHub repo folder. No electrode code, no ablation/wavelet extras unless you add them later.

---

## 1. Code (insulation classification + VGG16)

Copy these Python files into a `code/` folder:

| File | Purpose |
|------|--------|
| `generate_scalograms.py` | Scalogram generation from UHF PD CSV |
| `generate_clean_scalograms.py` | Clean scalogram pipeline |
| `scalogram_utils.py` | Wavelet/scalogram utilities |
| `main.py` | Data prep + train/val/test split |
| `main_clean_scalograms.py` | Clean scalogram pipeline entry |
| `split_dataset.py` | Dataset split for cross-validation |
| `model_training_vgg16.py` | **VGG16** training (insulation) |
| `run_parallel_training_vgg16.py` | Run VGG16 5-fold training |
| `model_training.py` | ResNet50 training (comparison) |
| `run_parallel_training.py` | Run ResNet50 5-fold training |
| `eval_roc_auc.py` | ROC-AUC evaluation (insulation) |
| `make_foldwise_roc.py` | Fold-wise ROC curves |
| `run_gradcam.py` | Grad-CAM generation |
| `gradcam_visualization.py` | Grad-CAM plotting |
| `generate_vgg_gradcam_for_dashboard.py` | VGG16 Grad-CAM for dashboard |
| `evaluation.py` | Classification evaluation |
| `crossval_aggregate_evaluation.py` | Cross-validation aggregation |

**Root:** `requirements.txt` (copy from My_Project).

---

## 2. Results (VGG16 + key outputs)

Copy into a `results/` folder:

### From `Result_insulation/`

Copy **only** these (do **not** copy `.h5` model files or large `.tiff` — disk/repo stay small):

| Item | Description |
|------|-------------|
| `KEY_FINDINGS_ONE_PAGE.md` | One-page results summary |
| `README_START_HERE.md` | Results/manuscript guide |
| `auc_foldwise_summary.csv` | AUC summary |
| `roc_foldwise_VGG16.pdf`, `roc_foldwise_ResNet50.pdf`, `roc_comparison_vgg16_resnet50.pdf` | ROC curves |
| `roc_foldwise_VGG16.png`, `roc_foldwise_ResNet50.png`, `roc_comparison_vgg16_resnet50.png` | ROC (web) |
| `statistical_analysis_results.json` | Stats (CI, p-values) |
| `statistical_comparison.png` | VGG16 vs ResNet50 figure |
| `VGG16/` | Only: `dataset_size_report.txt`, `learning_curves_all_folds.png`, and per-fold `classification_report.txt` / `detailed_classification_report.txt`. **Exclude:** `*.h5`, `training_history.json`, `confusion_matrix.tiff` |
| `ResNet50/` | Only: `gradcam_results/` (all PNG/JPG), `crossval_metrics.csv`, `crossval_summary.txt`, `experiment_config.json`, and per-fold `classification_report.txt`. **Exclude:** `*.h5`, `training_history.json` |

### Optional (if you want one archived run)

| Item | Description |
|------|-------------|
| `outputs_roc_gradcam_insulation-20260120T073725Z-1-001/outputs_roc_gradcam_insulation/` | VGG16/ and ResNet50/ subfolders with metrics_foldN.json, gradcam_selection.csv, auc_summary.json |

---

## 3. README (one detailed file)

Use the single **README.md** in this folder (`Partial_discharge_Classification_minimal/README.md`). It contains:

- Project title and one-line description  
- Dataset (size, classes, split)  
- Method (scalograms → VGG16/ResNet50, 5-fold CV, Grad-CAM)  
- **Results:** VGG16 98.20%, 0.9981 AUC; ResNet50 97.03%, 0.9966 AUC; table and figure locations  
- How to run (install, generate scalograms, train, evaluate, ROC, Grad-CAM)  
- Repo structure (code/, results/)  
- Link to your GitHub repo  

---

## 4. Do NOT include (keep repo minimal)

- Electrode classifier code/results (`train_electrode_classifier_fixed.py`, `eval_electrode_roc_auc.py`, `generate_electrode_gradcam.py`, Electrode_Analysis)
- Traditional ML classifier (`traditional_ml_classifier.py`, `run_traditional_ml_classifier.py`)
- Ablation experiments, Wavelet_Justification (unless you add later)
- Extra manuscript docs (READY_TO_USE_TEXT, VISUAL_COMPARISON, etc.) unless you want them
- Raw data (Classifications_data/, Scalograms/), large models (.h5), big .tiff figures
- codes/, Extra/, app.py (dashboard) — include only if you want the web app

---

## 5. Suggested repo layout

```
Partial_discharge_Classification/
├── README.md              ← One detailed README (in this folder)
├── requirements.txt
├── .gitignore
├── INCLUDE_LIST.md        ← This file (optional to commit)
├── code/                  ← ✅ Already populated (17 scripts)
│   ├── generate_scalograms.py
│   └── ...
└── results/               ← Copy only lightweight files (see list above)
    ├── KEY_FINDINGS_ONE_PAGE.md
    ├── README_START_HERE.md
    ├── auc_foldwise_summary.csv
    ├── roc_foldwise_VGG16.pdf
    ├── roc_foldwise_ResNet50.pdf
    ├── roc_comparison_vgg16_resnet50.pdf
    ├── statistical_analysis_results.json
    ├── statistical_comparison.png
    ├── VGG16/   (reports and learning curves only, no .h5)
    └── ResNet50/
        ├── gradcam_results/
        ├── crossval_metrics.csv
        ├── crossval_summary.txt
        └── experiment_config.json
```

---

## 6. How to finish and push

1. **Copy results** from `My_Project/Result_insulation/` into `results/` using the table in §2 (only lightweight files; no `.h5`, no large `.tiff`).
2. From this folder run:
   ```bash
   cd Partial_discharge_Classification_minimal
   git init
   git add .
   git commit -m "Insulation classification: code, VGG16/ResNet50 results, README"
   git remote add origin https://github.com/Nidhi0412/Partial_discharge_Classification.git
   git push -u origin main --force
   ```
