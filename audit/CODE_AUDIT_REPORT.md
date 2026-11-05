# UHF PD research code audit

Audit date: 2026-11-02 (Asia/Calcutta). Scope: the local `Partial_discharge_Classification_minimal/Partial_discharge_Classification_minimal` repository, 17 Python scripts and supplied results. This is a read-only inspection of original research artifacts; only this audit directory was created. No training, inference, preprocessing, package installation, or original evaluation scripts were executed. Python 3.14.6 `ast.parse` checks do not import TensorFlow or write bytecode. Source hashes and file sizes are in SOURCE_INVENTORY.csv. An existing README or summary asserting verification is not experimental evidence.

## Main assessment

The supplied VGG16-named module repeats a fixed directory split instead of applying the calculated cross-validation indices. Its CLI also dispatches lowercase `vgg16` to the ResNet50 constructor because the factory compares with uppercase `VGG16` (F22). The evaluation scripts repeatedly score one fixed test directory. These artifacts do not establish five-fold generalization or recording-independent performance. Reported headline accuracy/AUC values are broadly consistent with saved summaries after rounding, but this is arithmetic agreement, not reproduction from predictions. Recording-level leakage is a serious unresolved risk, not an observed overlap: original classified images, source recordings, and their mapping are absent from this reduced repository.

The ResNet50 configuration explicitly names `CV_Aug_20250529_210046`, 2 classes ordered Nomex/Teflon, 40 epochs and 5 configured splits. The evaluation script's historical absolute path also names that experiment (`crossval_aggregate_evaluation.py:13`). This connects the supplied summaries to a claimed run, but does not identify the exact source revision, training membership or checkpoint. The current ResNet50 source cannot parse. Its historical execution must have used different text or a different script; neither possibility is proven.

## P0 findings: provenance and validation first

### F01 — Missing recording-to-image lineage [P0, missing evidence]

Evidence: `generate_scalograms.py:156–188` and `generate_clean_scalograms.py:129–160` generate five images per accepted input using db3 and five coefficient-array positions. They do not generate four db1–db4 variants. Output stem preserves the input stem and directories carry level/class, but no acquisition manifest is saved. The clean pipeline outputs `dc{position}/{input_folder}/{stem}.png`; the other pipeline uses the same pattern. The standalone `generate_scalogram` function generates one image but is not the folder loop's chosen path (`generate_scalograms.py:109`).

Impact: if exactly 400 recordings passed these five-level loops, they would produce 2000 images, not 1600. This is a conditional count, not evidence that either number was actually generated. A subsequent level-selection/copying step could explain 1600, but is absent. Nothing establishes 100 recordings per configuration, 800 images per material, or how four source folders were merged into two classes. `results/VGG16/VGG16/dataset_size_report.txt` records 1024/256/320 images; those are saved claims, not present image counts. DATASET_PROVENANCE.csv deliberately contains one missing-evidence marker rather than invented sample IDs.

Correction: recover original CSVs, original image trees before and after splitting, the construction/copying script and acquisition manifest. Use a stable recording ID qualified by acquisition session and original folder, not a bare numeric filename. Then enumerate actual output variants. New training may be required after lineage is established.

### F02 — Image-level splitting does not enforce recording independence [P0, confirmed design omission; actual overlap unverified]

Evidence: `split_dataset.py:40–64` shuffles PNG filenames and moves them individually; no recording-group variable exists. `main.py:36–47` uses individual filenames with 70/15/15 splitting, a different protocol from the claimed 80/20 then 80/20 development split. `split_dataset.py:8–9` creates `class/train` and `class/test`, whereas training expects `Train/class` and `Test/class` (`model_training_vgg16.py:845–846`). The supplied pipeline pieces are not a single verified end-to-end recipe. `os.listdir` is unsorted before shuffling, so a seed alone does not guarantee the same partition on another filesystem.

Impact: variants of one recording can enter different partitions. Exact/near duplicate overlap cannot be measured without classified input images. Electrode type, voltage, acquisition session, sensor settings and folder-specific rendering may be shortcuts; no acquisition metadata or stratified diagnostic evaluation is supplied to test them. No actual shortcut is proven.

Correction: split recording/session groups before rendering or augmenting; save explicit sorted membership and source hashes. See REMEDIATION_PLAN.md. Rerun experiments if overlap or confounding is confirmed, or if historical independence cannot be demonstrated.

### F03 — Fold indices are unused [P0, confirmed in supplied VGG16 code]

Evidence: `model_training_vgg16.py:935–939,991` constructs StratifiedKFold and obtains `train_idx,val_idx`. `1009–1029` then constructs every generator from the same `train_dir`, `subset='training'/'validation'`, fixed validation_split=0.2 and seed=42. No indexed path subset reaches these generators or `fit` at 1056. The later `create_dataset_from_paths` helper is not used by this fold loop. Model initialization is fresh per iteration (1003), so this is repeated fresh training on fixed membership, not independent folds. The analogous text in `model_training.py:980–1014` has the same omission, but that module cannot run as supplied.

Impact: the README's five-fold stratified CV description is unsupported by this code. The fixed test generator is passed into the VGG16 loop but not used there. Report the observed procedure as repeated fixed-split training/evaluation, subject to historical-version uncertainty. Valid CV requires new indexed/grouped training, not relabeling these runs as proper folds.

### F04 — Missing Fold 1 and silent omission of failures [P0, confirmed control paths; historical cause unknown]

Evidence: `model_training_vgg16.py:1033–1046` performs a special first-iteration augmentation verification and skips that iteration if verification fails. `1084–1086` catches all fold exceptions and continues. `eval_roc_auc.py:27–34` explicitly evaluates only folds 2–5. Saved reports and CSVs contain only those four folds. `crossval_aggregate_evaluation.py:88–94` also skips missing models.

Impact: these control paths plausibly explain omission but do not prove why historical Fold 1 is missing. Obtain original stdout/stderr, first-fold directory, training logs and source revision. Report four available runs and missing Fold 1; do not impute its results. A complete valid five-fold experiment requires all prescribed folds or a documented rerun policy.

### F05 — Same fixed test set and pooled repeated predictions [P0, confirmed evaluation procedure]

Evidence: `crossval_aggregate_evaluation.py:14,21–32,86–101` creates one test generator and reuses it for each model. `eval_roc_auc.py:16,68–77,105,237–245` creates generators from one fixed test directory. `282–284` concatenates all those labels and scores for a pooled curve.

Impact: four runs of 320 test images are not 1280 independent test observations. A pooled AUC also mixes scores from different models and is not the same estimand as a mean model AUC. Exact historical membership is not independently established by a filename manifest. Correct ROC captions and sample counts; retain 320 as the claimed unique-image count, subject to provenance. For an ensemble, average scores for each aligned sample before a single evaluation and explicitly identify the ensemble.

### F06 — Class-dependent wavelet selection not found [P0 concern, unconfirmed]

Evidence: `generate_clean_scalograms.py:68–94` accepts a numeric coefficient position, hard-codes db3 and processes every level for every label at 130–149. `scalogram_utils.py:41–54` similarly accepts a caller-provided position. No supplied selection routine chooses a coefficient using the true material label. Class folders are used for organization, which alone is not label leakage.

Impact: the concerning label-dependent selection may reside in missing dataset-construction or inference code. It must not be asserted as a confirmed bug here. Obtain the rule that selected the final images; any supervised choice must be fitted using training groups only and applied without the unknown test label.

## Wavelet and scalogram findings

### F07 — Coefficient positions mislabeled as detail levels [P1, confirmed]

Evidence: `generate_scalograms.py:66–74,158–184`, `generate_clean_scalograms.py:83–92`, `scalogram_utils.py:45–54`. At five-level decomposition the returned order is `[cA5,cD5,cD4,cD3,cD2,cD1]`; hence stored dc1 is cD5, dc2 is cD4, dc3 is cD3, dc4 is cD2, dc5 is cD1. This order is documented by [PyWavelets wavedec](https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html).

Impact: all corresponding coefficient CSV headers, scalogram titles and DC-band interpretations, including README/KEY_FINDINGS Grad-CAM band claims, require review. A label-only correction need not change training inputs, but changing which coefficient feeds the model requires reprocessing and retraining. Preserve the historical convention explicitly.

### F08 — Wrong sample interval and misleading frequency rendering [P1, confirmed]

Evidence: `generate_clean_scalograms.py:92,154`, `generate_scalograms.py:180–183`, `scalogram_utils.py:54`. They supply original dt to a CWT of downsampled coefficients. For actual detail level j, coefficient spacing is approximately 2**j times original dt (boundary extension requires care); reported coefficient-sequence frequencies are consequently inflated by 2**j and shown duration compressed by that factor. A CWT of a decimated subband is not automatically a direct map of original RF frequencies. Moreover, linearly spaced scales produce nonlinear frequencies: `imshow` with only endpoint extents treats rows as uniformly spaced physical frequencies; `origin='lower'` puts the first, highest-frequency scale at the bottom despite ascending extent endpoints.

The [PyWavelets CWT documentation](https://pywavelets.readthedocs.io/en/latest/ref/cwt.html) specifies frequency calibration by sampling_period and that the coefficient values themselves do not depend on that parameter. Therefore correcting dt alone may correct labels without changing clean image pixels. Revising scales, row ordering or representation can change the input and requires retraining. Use reconstructed subbands at original sampling if the intended interpretation requires it; validate using synthetic known-frequency signals before drawing physical conclusions.

Impact: MHz/time axes and claimed discharge-frequency localization are unreliable. No raw timestamps, oscilloscope sampling specification, analog bandwidth or sensor response are available in this repository; effective acquisition bandwidth cannot be established. Scales beginning at 1 also warrant aliasing checks against wavelet support and Nyquist.

### F09 — Magnitude rather than energy; fixed decomposition [P1, confirmed]

Evidence: `generate_clean_scalograms.py:59,83–85` and `generate_scalograms.py:158–160,183` use `abs(W)`, db3, fixed level 5. No db1–db4 comparison, energy-driven level selection or Shannon entropy implementation exists in these supplied scalogram scripts. The maximum useful level is not checked. For 10000 samples/db3 (filter length 6), the documented maximum-level formula yields floor(log2(10000/5))=10, so level 5 itself is not excessive for that length. Other lengths must be checked independently.

Correction: describe the actual magnitude representation and fixed parameters unless missing experiment evidence establishes another pipeline. Changing magnitude to squared magnitude can change normalized colormap pixels and requires evaluating/retraining affected models. There is no entropy formula in this code to validate.

### F10 — Independent image normalization and overwritten coefficient evidence [P1, confirmed]

Evidence: `generate_clean_scalograms.py:57–64` uses imshow autoscaling with no shared vmin/vmax, a jet colormap and axis-free raster export; `generate_scalograms.py:181–188` instead includes title, axes and colorbar. Autoscaling removes absolute color-amplitude comparability. In clean generation, `159–160` writes the same `{stem}_detailed_coeffs.csv` inside every level iteration and passes a one-element list; `save_coefficients:105–108` always names it dc1. Only the final iteration survives, containing cD1 mislabeled dc1.

Impact: saved clean coefficient files cannot recover all original bands; image artifacts differ between pipelines. Which rendering reached the historical CNN is unknown. Preserve originals; recover raw data and document normalization and rendering before regeneration. Input dimensions are nominally 224 square in clean export and resized to 224 by generators; actual classified image dimensions cannot be counted here.

## Augmentation and model implementation

### F11 — Specialized augmentation is bypassed on the active VGG16 path [P1, confirmed]

Evidence: `model_training_vgg16.py` defines `get_data_generator` twice (401 and 814). Python uses the later definition, which returns a plain ImageDataGenerator. The loop at 952–953 uses it. Thus `fit` receives rotation 10 degrees, width/height shifts 0.1, zoom 0.1, horizontal flip, brightness [0.9,1.1], rescaling 1/255 and nearest fill; specialized masking, phase shift, stretching and Gaussian noise are not attached to this training path. `save_to_dir` stores transformed views; it does not increase independent sample count. Validation in the active loop uses augment=False. The separate load_dataset helper instead creates validation from the augmented train_datagen at 878–887, so that helper's validation previews can be stochastic.

Correction: report only transformations supported by the active path; saved visualizations do not establish training usage. New specialized-augmentation claims require instrumented runtime checks and ablations on identical grouped partitions.

### F12 — Custom augmentation implementation is incorrect even if activated [P1, confirmed dormant defects; runtime effects untested]

Evidence: custom model's final five layers at 255–261 are FrequencyMasking, TimeMasking, PhaseShift, FrequencyStretch, ScaledGaussianNoise. Negative indices -4/-3/-2/-1 at 307–310 therefore bind to TimeMasking/PhaseShift/FrequencyStretch/ScaledGaussianNoise. `apply_scalogram_augmentations:328–340` applies every operation to original x separately and returns only the last tensor at 346, discarding the earlier outputs. On exceptions it prints a warning and returns x (348–350). Thus its `freq_stretched` return is actually Gaussian noise given this layer ordering. Noise addition is not explicitly range-clipped. Tensor shape/type correctness under graph execution is not established by parsing; custom iterator __next__ behavior alone is not proof of invocation by Keras' indexed adapters.

Correction: use named layers and a tested sequential training-only chain, validate finite float tensors, [batch,224,224,3] shape and unchanged label alignment, and surface failures. Do not describe these dormant functions as demonstrated historical augmentation.

### F13 — Physical meaning of augmentations requires validation [P1, methodological concern]

Evidence: image rotation/vertical translation mix or shift time/frequency coordinates; horizontal flips reverse time. Brightness on a colormapped magnitude image is not a calibrated gain change. Pixel-axis rolling in PhaseShift (`model_training_vgg16.py:186`) is a time-image shift, not the waveform phase operation its name suggests. Frequency stretching and masking may alter discriminative spectral content.

Recommendation: start with no augmentation, then bounded raw-signal time shifts, measured noise and gain perturbations justified by acquisition variability; fit normalization on training recordings. Compare each addition separately and in combination using the same grouped splits and seeds. Do not assume any transform preserves the label without acquisition evidence. These ablations need approval to execute.

### F14 — Syntax failure blocks supplied ResNet50 script [P1, confirmed]

Python ast.parse fails `model_training.py:675`: expected an indented block after if at 673. All 16 other code scripts parse. STATIC_SYNTAX_CHECKS.csv records the first syntax error in each file; fixing it could expose further errors. Do not infer that a historical run used this broken file. Original dated source or commit and logs are required. No original code was repaired in this audit.

### F15 — Architecture/configuration claims exceed available evidence [P1, confirmed source behavior; historical weights missing]

`model_training_vgg16.py:create_vgg16_model:638–658` initializes ImageNet VGG16 without its top; enables block5_conv1 onward (three convolution layers plus the block pool), then GlobalAveragePooling2D, configured dropout, and a one-unit sigmoid for binary classification. The binary VGG16 head has no intermediate dense 512 and no L2 regularizer despite a declared l2_reg setting. Adam uses configured learning rate. The stale docstring says no fine-tuning but actual code unfreezes block5.

`create_resnet50_model:665–756` initializes ImageNet ResNet50; when enabled, last 15 Keras layers are trainable (not 15 convolution layers). The binary head uses global pooling/dropout and a regularized sigmoid; additional 512/256 dense layers are for multiclass. L2 is hard-coded 0.01, not read from l2_reg. An optional previous-weight branch exists, but the fold factory at 1341–1344 supplies no load_weights, and a new model is built per iteration. This describes the syntactically valid alternate implementation, not verified saved weights.

The supplied ResNet experiment JSON declares batch 16, epochs 40, learning rate 5e-5, early-stop patience 10, reduce-LR patience 5, splits 5, dropout .5, L2 .01, fine-tuning true, sigmoid/BCE. Without histories, checkpoints and dated invocation logs, declared maxima are not completed epochs or proof of applied settings. No corresponding complete VGG historical configuration is supplied. `get_callbacks:771–793` chooses best val_accuracy for checkpoint but restores best val_loss for final model.h5, so those two files can encode different epochs. Which checkpoint a result used must be recorded.

### F16 — Launchers target different tasks/models [P1, confirmed]

`run_parallel_training.py:87–91` targets Electrode_classifier. `run_parallel_training_vgg16.py:70` sets model_type='resnet50' despite its name and VGG prompts; 102–104 names a VGG output directory but selects `2A_Electrode_Classifier_Nomex`. Neither current launcher is a verified entrypoint for the claimed binary insulation run. Recover historical command lines before using defaults. `main.py:4–6` also imports absent preprocessing/model_training_resnet50 modules, so syntax success is not end-to-end executability.

### F17 — Preprocessing inconsistent for one Grad-CAM path; environment incomplete [P1, confirmed]

Active training and evaluation normalize RGB by 1/255. `scalogram_utils.py:preprocess_for_model` and the two newer Grad-CAM helpers do the same. `gradcam_visualization.py:73–80` instead uses ResNet50 preprocess_input (ImageNet channel/mean convention), inconsistent with supplied training. This can change predictions and attention; using an ImageNet-initialized backbone does not prove its standard preprocessing was used during training. Retraining with a different preprocessing convention would be a new experiment, not an audit correction.

requirements.txt omits direct script dependencies PyWavelets, scipy, tqdm and psutil; versions are lower bounds, no original environment lock is supplied. Recreating historical TensorFlow/Keras behavior from this file alone is unsupported. No dependency installation or TensorFlow compatibility claim was tested on this audit's Python 3.14 interpreter.

## Results and statistics

### F18 — Headline arithmetic agrees after rounding; detailed metrics have contradictions [P0/P1]

RESULT_VERIFICATION.csv separates summary arithmetic, conditional count reconstruction and prediction-level verification. CONDITIONAL_CONFUSION_MATRICES.csv is explicitly conditional evidence, never a replacement for original predictions. Rows are true Nomex/Teflon, columns predicted Nomex/Teflon.

VGG16 supports of 160 each and rounded recalls uniquely support [[160,0],[5,155]] for Fold 2 and [[160,0],[6,154]] for Folds 3–5. Their exact implied accuracies are 0.984375 and 0.98125, yielding mean 0.98203125 (98.203125%). Fold 5 report says 0.9812, consistent with six errors/320 at four-decimal reporting; a materially different Fold 5 result is not supplied. The mean of rounded text accuracies is 0.9820. The statistics JSON mean .982 and SD .0016 reflect rounded inputs rather than exact counts (exact sample SD .0015625). This is rounding-level agreement, not a new accuracy result.

ResNet50 saved accuracy CSV yields mean .9703125 (97.03125%), rounding to 97.03%. JSON .970325 reflects rounded inputs. With balanced supports assumed, reconstructed matrices for folds 2–5 are [[155,5],[4,156]], [[152,8],[3,157]], [[158,2],[6,154]], [[159,1],[9,151]]. The CSV precision/recall values are internally consistent with those counts, but supports and identities require original evidence.

VGG16 reports give positive-class precision 1.0 in all four runs, mean recall .9640625 and mean F1 about .98170. README/KEY_FINDINGS instead copy the ResNet-like precision 97.67% and F1 97.01% into VGG16. Replace copied summaries only after checking original predictions. Binary precision/recall/F1 are Teflon metrics, not macro averages; per-class and macro calculations are in the verification CSV. Specificity of Teflon equals Nomex recall.

The AUC CSV means are VGG16 .998095703125 and ResNet50 .996591796875, rounding to .9981/.9966. Statistics JSON means .998075/.996575 differ at the fifth decimal, consistent with rounded fold AUCs; its exact input arrays and generator script are missing. Raw probabilities/labels are absent, so no ROC or AUC has been independently recomputed. Do not digitize raster curves as original predictions.

### F19 — Confidence intervals and superiority claims are not established [P1, confirmed formula; inferential validity unsupported]

`statistical_analysis_results.json` uses n=4, t critical 3.182446305284263, and margin=t*SD/sqrt(4). This arithmetic is reproduced. It reflects variability of four run summaries under assumptions, not uncertainty over independent recordings. The generating statistical script and exact arrays are absent. `crossval_aggregate_evaluation.py:108–109` uses pandas sample SD; `eval_roc_auc.py:278` uses numpy population SD. `model_training_vgg16.py:1089–1095` aggregates entire per-epoch arrays, not one final metric per fold; unequal lengths can also fail aggregation. `evaluation.py` learning curves pad stopped runs with their last value and shade mean +/- SD, which is not a confidence interval.

JSON paired t-test p-values are approximately .0096 for accuracy and .00945 for AUC, while Wilcoxon p=.125 for both. With four nonzero same-sign paired differences, an exact two-sided signed-rank p=.125 is consistent, but this does not validate the input pairing. Do not claim a statistically supported VGG16 advantage from these four dependent fixed-test scores. Use recording-cluster bootstrap of paired per-recording predictions on a locked holdout; specify whether score aggregation represents a recording or an image. If the deployed target is one recording, predefine recording-level aggregation before testing. Resample sessions rather than individual recordings if sessions are the independent acquisition units. Separate run/seed variation from holdout population uncertainty. For proper nested grouped CV, report outer held-out predictions and account for shared training data; avoid a naive independent-fold t-test.

## Baselines and explainability

### F20 — Baseline and ablation comparability cannot be established [P2, missing evidence]

No SVM, RF, CNN-SVM/RF, db1–db4 ablation, or noise-robustness scripts/results are included. INCLUDE_LIST.md explicitly excludes traditional ML and ablations. Other workspace projects have prediction files but no demonstrated linkage to this experiment's image membership; they were not substituted. Obtain original baseline source, feature-fitting procedure, split manifests and predictions, then compare on exactly the same recording groups without fitting transformations on the test set.

### F21 — Binary Grad-CAM always targets the positive output [P1, confirmed]

`gradcam_visualization.py:26–35` and `generate_vgg_gradcam_for_dashboard.py:20–25` take argmax of a one-unit sigmoid and select output 0, which is the Teflon probability regardless of the true class. `run_gradcam.py:54–56` explicitly selects that sigmoid for every binary image, ignoring passed class_index. For a Nomex explanation the appropriate target is 1-p (or negative logit); positive evidence maps for the two classes differ after ReLU. The first helper targets ResNet50 conv5_block3_out; the dashboard helper targets VGG16 block5_conv3; run_gradcam supports both. These are different implementations, not interchangeable evidence.

Saved analysis_summary.txt says ResNet fold_4, ten selected images (five each), mean Nomex confidence .07%, Teflon 80.02%, heatmap intensities .1025/.0893. The very low Nomex confidence is inconsistent with a general successful Nomex interpretation and is compatible with preprocessing mismatch, but causation is unproven. Only rendered PNG/JPG composites and a rendered mean-map figure are available; raw numerical heatmaps/selection metadata are absent. Their activation statistics cannot be independently recovered faithfully from colormapped annotated figures.

Correction: align preprocessing and explicit binary targets, export numerical heatmaps, predictions and selection manifests, state normalization/threshold definitions. Recompute with historical checkpoints if recovered; this needs inference approval beyond the present static stage. Heatmaps do not directly establish DC1/DC4/DC5 attention from a single coefficient image, nor prove a physical mechanism. Resolve localized/diffuse descriptions against identical targets and quantitative, prespecified concentration measures; otherwise remove both interpretations.

### F22 — Lowercase VGG16 CLI option dispatches to ResNet50 [P0, confirmed]

Evidence: `model_training_vgg16.py:1407` accepts lowercase choices `vgg16,resnet50`; `1468–1470` forwards the lowercase argument unchanged. `train_model_parallel:1344–1347` compares model_type with uppercase `VGG16`; every accepted CLI value therefore takes the ResNet50 branch. The loop still places outputs under a hard-coded VGG16 directory at 995. A direct Python caller using uppercase VGG16 could select the VGG constructor, but that is not the supplied CLI path.

Impact: neither filename, directory name nor a reported VGG label proves which backbone was actually trained. The current CLI cannot reproduce a true VGG16 run as written. This is distinct from the launcher that explicitly selects ResNet50. Recover original saved model configuration and historical invocation/source before attributing the saved VGG16 results to this implementation. Repair dispatch only in a later approved code-fix stage; new training is needed if no correctly identified historical VGG model exists. Architecture statements in F15 describe the constructors, not the model proven to have produced the VGG-labeled results.

## Evidence limits and required recovery

The full local reduced repository was inventoried; no raw source recordings, classified image partition directories, model checkpoints, per-image predictions, numerical heatmaps, training histories, or recording manifests are included. A broad workspace filename search did not locate this experiment's checkpoint/prediction artifacts; it did locate unrelated multitask/Gnostics artifacts, which were not treated as evidence for this run. Google Drive folders and the linked GitHub repository were not fetched; this report does not claim an audit of their contents. The manuscript and reviewer documents themselves were not supplied, so correction mapping uses the section names and historical numbers in the request.

Required: complete `CV_Aug_20250529_210046` run tree (all folds including failed/missing Fold 1); original `model.h5`, `best_weights.h5`, training_history.json, training_log.csv and console logs; dated VGG16 run/config; exact source revision and environment lock; original `Classifications_data/Insulation_classifier/Train` and `Test` trees; pre-split scalograms; raw four-configuration CSVs and acquisition/session/voltage metadata; image-selection/merging code; `outputs/{VGG16,ResNet50}/fold_N/y_true_foldN.npy`, `y_prob_foldN.npy`, per-image CSV/filenames, metrics and gradcam_selection.csv; statistics-generating script and input arrays; baseline and ablation artifacts. Original IDs must come from acquisition evidence, never inferred from numbering alone.

## Validation and deliverables

Run `python audit/verify_audit.py` from the repository root to repeat the source inventory, static parsing and summary checks. It writes only audit outputs. `check_split_manifest.py` provides overlap assertions for a future populated fixed-holdout/development-CV manifest; the template is empty because actual memberships are unknown. It checks supplied hashes rather than hashing original images, so trusted hash generation and near-duplicate inspection are separate recovery steps. No historical split manifest or valid replacement training results have been fabricated. The detailed remediation and manuscript maps distinguish documentation corrections from work requiring new processing/training.
