# Files to change before corrected training

**Historical pre-fix review:** the fixes described below have now been implemented in shared modules. These line references describe the preserved pre-fix code, not the current thin entrypoints. See `FIX_IMPLEMENTATION_REPORT.md` and the repository README for current behavior and validation limits.

Reviewed: 2026-11-03, Asia/Calcutta. Scope: every one of the 17 research Python files under `D:\Nidhiiiiii\Partial_discharge_Classification_minimal\Partial_discharge_Classification_minimal\code`, plus requirements.txt. Audit utilities are evidence tools, not research training entrypoints. All original research files remain unchanged. No original dataset was modified, no modules were imported/executed, and no training or inference ran. SOURCE_REVIEW_INVENTORY.json records source hashes, line counts, syntax status, imports and function locations.

This review identifies implementation changes for the newly agreed pipeline. It does not infer that the currently supplied source produced the historical experiments, and it does not infer laboratory recording identities from image names or hashes. Recovered June checkpoints really contain VGG16; the present dispatch defect is a separate current-source problem.

## Effective current routes

`run_parallel_training_vgg16.py:127–139` starts `model_training_vgg16.py`. Its main calls `train_model_parallel`, which calls `load_dataset`, creates a model factory and calls `train_model_with_cross_validation`. That loop constructs the generators actually supplied to fit. Consequently, changing only split_dataset.py or the unused create_dataset_from_paths helper would NOT fix the effective training route.

`run_parallel_training.py:163–180` starts `model_training.py`. This module currently fails parsing and cannot train as supplied. Its textual fold loop also ignores indices, and has a model-factory argument mismatch in addition to malformed indentation.

The two main*.py files are alternate legacy orchestrators, not callers of the above launchers. They import preprocessing/training modules absent from this repository, use four configuration classes and a different splitting protocol. The remaining evaluation/scalogram/Grad-CAM files are separate executable tools, rather than one integrated training pipeline. Their configuration and sample membership must be brought into agreement with the chosen route.

## Five core files: change or redirect before training

### 1. code/split_dataset.py — replace splitting behavior [required]

Evidence: `split_dataset:17–69` shuffles individual PNG filenames, moves original files (57,62), and saves no recording/fold manifest. Directory layout is class/train and class/test at 8–9, whereas trainers expect Train/class and Test/class. `process_all_datasets:74–78` targets other classifier folders, not the newly supplied Insulation_classifier directory. Lists are not sorted before seeded shuffling.

Required: accept an authoritative image-to-recording manifest, group by recording/session where appropriate, stratify material and check electrode balance, reserve the holdout, and generate explicit development-fold membership. Save membership without moving or deleting original images. Refuse unknown group IDs rather than treating each manually named image as an independent recording. Validate group separation, coverage, class mapping and actual loader membership. Hash comparisons can flag copy consistency but cannot manufacture acquisition identities.

### 2. code/model_training_vgg16.py — primary training refactor [required]

This is the most consequential file to change:

- `get_data_generator:401` is overwritten by the second definition at 814. The active later definition returns ordinary ImageDataGenerator; the custom scalogram layers are bypassed. Keep one clearly configured training-only augmentation path. Do not enable every dormant transform automatically: first justify which preserve PD signal meaning.
- `load_dataset:878–885` builds its validation generator from the augmented training generator. The effective fold loop later uses a separate unaugmented validation generator, so distinguish these paths; remove the inconsistent alternate validation behavior.
- `load_dataset:910–918` lists paths/labels but no recording groups; iteration order is unsorted. Load paths, labels, recording IDs and partitions from the shared manifest instead.
- `train_model_with_cross_validation:991–1029` computes train_idx/val_idx but rebuilds the same fixed directory subset each time. Build datasets from the actual indexed/grouped lists. Use identical saved folds for both backbones; a fixed seed alone does not establish shared membership.
- `create_dataset_from_paths:1107–1122` is currently unused by the loop and its augmentation branch is `pass`. If retained as the effective loader, implement training-only augmentation there, static shape/RGB/range validation, training shuffling, and deterministic held-out iteration while preserving ID order for prediction export.
- `ScalogramDataGenerator:307–310` binds negative layer indices incorrectly because Gaussian noise is the last layer. `apply_scalogram_augmentations:328–346` applies every operation to the original tensor separately and returns only the last result. Exceptions return the input (348–350). Correct or retire this dormant implementation. Also replace `classes=generator.classes` at 387: those are numeric sample labels, not the ordered class-directory names that this argument expects.
- `save_augmented_samples:530–614` loops until preview quotas are met and continues on repeated errors without a bounded attempt budget. Previews must not hang training, remap labels independently, lose source IDs or determine whether a fold runs. `1033–1046` can skip Fold 1 when preview verification fails; remove that coupling. Save original/augmented examples with parent IDs and transform settings.
- `train_model_parallel:1344–1347` tests uppercase VGG16 while CLI choices are lowercase at 1407. Normalize model identity once, reject unknown values and verify the actual built backbone. `995` hard-codes a VGG16 output folder even for the ResNet factory. Record actual architecture and checkpoint identity, not a folder-derived assumption.
- `create_vgg16_model:638–658` fine-tunes block5 despite a stale frozen-backbone docstring; binary head has no L2 despite the CLI option. `create_resnet50_model:690` hard-codes L2 .01. Apply only intended, explicitly recorded regularization/fine-tuning parameters; match logs to the actual layer configuration.
- `get_callbacks:771–783` selects checkpoint by val_accuracy but restores final weights by val_loss. Choose/document the evaluation checkpoint policy and save its hash. Fresh model creation per fold already exists at 1003; preserve it and add controlled seeds.
- `1084–1086` silently continues after failed folds. Record failures explicitly and prevent an incomplete run from being reported as a complete CV experiment.
- `1089–1095` aggregates per-epoch history arrays, not one evaluated metric per held-out fold. Summarize explicit outer held-out predictions/metrics; keep learning curves separate. Unequal early-stopped history lengths can also break array aggregation.
- `train_model:1141–1147` calls the fold function with incompatible positional meanings and an unsupported n_splits keyword. Retire or redirect this old wrapper rather than maintaining a second training API.
- `evaluate_model:1194–1212` uses categorical labels and argmax even for a one-unit binary sigmoid output; argmax on that output always gives zero. It also tries to access class_names after mapping the tf.data dataset at 1197–1200, which does not preserve arbitrary loader attributes by contract. Use a shared binary-aware evaluator with class metadata stored separately.
- `plot_training_history:1281` saves a fixed filename into a directory that its caller passes as cross_validation at 1381 without creating it there. Ensure new output directories exist and give each original fold ID its own filename; do not renumber successful folds after failures.
- Save actual invocation, environment, manifests/hashes, class mapping, per-layer trainability, preprocessing, run/fold IDs, checkpoint hashes and predictions with image/recording IDs. The test_generator parameter is not evaluated inside this fold loop; preserve an explicit separation between development CV and final holdout evaluation rather than implying the current loop already tests each fold.

### 3. code/model_training.py — consolidate or repair comprehensively [required if retained]

Evidence: syntax fails at 675 after if at 673; additional visibly malformed indentation exists around fit/except at 1019–1049. Static parsing currently reports only the first failure.

The intended loop at 980–1011 repeats fixed directory subsets. At 991 it calls `model_fn(len(class_names), params)`, but the factories at 1311/1313 accept only one argument. Fixing indentation alone will not make this work. The loop also hard-codes ResNet50 output paths (987), even if another backbone is selected. It duplicates the generator and augmentation issues in the VGG-named file. Its VGG constructor at 640–642 freezes the whole backbone, unlike the other file's block5 fine-tuning; these cannot both serve as an undocumented definition of the same experiment.

Recommendation: use one shared corrected engine for both architectures, and redirect this entrypoint to it or retire it from the supported workflow. Do not maintain two drifting implementations. If retained independently, it needs the same manifest, effective grouped folds, augmentation, prediction and reporting fixes as the primary trainer plus a complete post-repair syntax/interface check.

### 4. code/run_parallel_training_vgg16.py — fix launch configuration [required]

Evidence: 70 chooses resnet50, while prompts and output labels say VGG16; 104 points to `2A_Electrode_Classifier_Nomex`. Its relative script name at 129 depends on the current terminal folder. get_training_config is not used by main, leaving competing parameter definitions.

Required: explicit model/data/manifest/config arguments, correct insulation task, absolute script resolution, one parameter source, one shared fold manifest and versioned new output directory. Record the resolved command before execution. The resource check's result is not itself enforced by the launcher; the name parallel does not prove concurrent training occurs.

### 5. code/run_parallel_training.py — fix task and training destination [required]

Evidence: 87–91 points to Electrode_classifier; 165 launches the currently broken model_training.py. Required: point to the shared corrected engine, pass insulation data and the exact same split manifest as VGG16, record the selected ResNet model/config and invoke sequentially initially unless resource-safe parallelism is actually verified. Keep originals/results separate from new run outputs.

## Four evaluation files: align before reporting new results

### 6. code/evaluation.py — shared manifest-aware evaluation [required for new reporting]

Evidence: fixed Linux model/Test paths at 13–17, hard-coded binary class list at 20, and load_best_model actually loads model.h5 at 24–30 despite its name. Its sigmoid threshold at 87 and shuffle=False at 74 are appropriate binary behaviors to preserve. However inspect_fine_tuning at 38–60 concludes fine-tuning from any trainable layer, including classifier-head/input flags; inspect backbone weight-bearing layers separately. `print_dataset_sizes:299–349` counts a reconstructed directory subset, not an authoritative grouped manifest. Learning-curve padding at 231–245 and later confidence-interval captions misrepresent SD bands as confidence intervals.

Required: explicit checkpoint/config, manifest-ordered OOF or holdout sample list, class mapping assertion, shared preprocessing, per-image and per-recording score export, documented metrics, and recording/session-level uncertainty. Keep epoch histories and evaluated fold metrics distinct. Write new reports into new run directories rather than historical fold folders.

### 7. code/crossval_aggregate_evaluation.py — distinguish OOF and repeated holdout [required]

Evidence: 14/78/88 reuse one Test generator for every fold model; 84–86 skips missing checkpoints; 95–107 summarizes those fixed-test scores as crossval. Sigmoid thresholding itself is appropriate (39–43).

Required: aggregate actual disjoint outer held-out prediction rows for CV, verify unique OOF assignment by recording and retain fold/checkpoint IDs. Keep repeated holdout model evaluations separately labeled; they do not add independent observations. Missing prescribed folds should prevent a complete CV claim. Summarize matched VGG/ResNet predictions and record denominators, class metrics, macro metrics and chosen SD convention.

### 8. code/eval_roc_auc.py — use traced predictions, not pooled repeated tests [required]

Evidence: 27–34 restricts folds to 2–5; 68–77/105 read a fixed Test tree; 282–288 concatenates repeated test predictions. It already exports probabilities, labels and filenames (136–149): preserve this useful capability and extend it with recording ID, manifest/fold/checkpoint hashes and preprocessing identity. Generated text at 189–225 makes fixed favorable interpretations rather than conditional conclusions.

Required: read the actual run manifest/checkpoint registry, include all prescribed folds, align score/label IDs, and compute ROC for a declared endpoint: OOF, independent final holdout, or aligned-score ensemble. Do not pool repeated predictions as independent samples. Generate conclusions only from verified metrics and grouping evidence, using the stored class order.

### 9. code/make_foldwise_roc.py — update plot inputs and captions [required for new ROC figures]

Evidence: 23 defaults to folds 2–5; 130–134 hard-codes their count and Fold-1 exclusion; 271–273 assumes that numbering in colors. The file plots existing arrays and does not train.

Required: discover folds from completed-run metadata; verify prediction IDs, endpoint and class order; derive captions/sample counts dynamically. Distinguish mean per-model AUC, pooled disjoint OOF AUC and ensemble AUC. State whether shading is SD or an actual recording-aware interval. Preserve historical figures and write new ones into the new experiment directory.

## Three scalogram files: change if regenerating inputs; centralize the scientific definition

### 10. code/generate_clean_scalograms.py — provenance-producing clean generator [required if raw reprocessing is chosen]

Evidence: process_signal 83–92 fixes db3/level5, calls coeffs[level] as though it were the actual level, and passes original dt to decimated coefficients. process_dataset 130–160 produces five positions per CSV and overwrites the same coefficient CSV at each iteration, with only a one-element list always labeled dc1. save_clean_scalogram uses magnitude and per-image plot autoscaling at 59. FileHandler at 15 creates an output at import time.

Required: explicit validated wavelet/true-level/representation configuration; sampling checks; a clearly documented coordinate convention; one source-recording ID on every output; no coefficient overwrites; raw/output hashes and generation parameters in a provenance manifest; unique versioned destinations and explicit logging setup. The four-versus-eight-image methodology must be resolved before choosing variants. A coefficient-name correction must not silently change the physical band fed into the network. Do not regenerate existing images in this review.

### 11. code/generate_scalograms.py — unify or retain as illustration-only [conditional]

Evidence: 67–79 and 158–188 repeat the level/spacing/magnitude issues and save annotated plots with axes/colorbar. `process_folder:109` invokes the five-detail exporter, not the single-scalogram helper. Paths at 194–195 point outside this project. The skip_rows argument at 11 is ignored in favor of fixed skiprows=21 at 21.

Required if used: share the corrected transform/provenance implementation with the clean generator, validate inputs, and explicitly distinguish publication figures from CNN input images. Prefer keeping this as an illustration exporter so annotated and clean renderers cannot silently create different training datasets.

### 12. code/scalogram_utils.py — align inference/preprocessing [conditional on raw/inference use]

Evidence: 11–14 repeats hard-coded transform parameters, 47–55 repeats coefficient-position and dt behavior, 60 uses per-image magnitude rendering and 80–86 assumes RGB/255 normalization. The module is not imported by the current trainers; it is a separate inference utility.

Required: obtain transform/preprocessing from the trained run configuration, share coefficient/normalization logic, validate timestamps and signal shapes, and save source/transform identifiers when generating predictions. Preserve existing image inputs unless the approved scientific pipeline requires regeneration; grouped training on verified existing images does not automatically require changing this renderer first.

## Three Grad-CAM files: downstream fixes; not prerequisites for fitting

### 13. code/gradcam_visualization.py — align model, target and preprocessing [required before interpreting new maps]

Evidence: 33–35 uses argmax on the single sigmoid and always explains its positive output. 80 applies ResNet preprocess_input while supplied training uses RGB/255. get_best_fold 95–101 selects a model using evaluated crossval_metrics, which the current aggregator derives from Test scores. The apparent best-model selection therefore must not be used as an independent holdout protocol.

Required: load the exact selected checkpoint/config, select models on development results only, use explicit Nomex=1-p/Teflon=p targets (or corresponding logits), and reuse training preprocessing. Save numerical maps and IDs. Also maintain consistent RGB/BGR handling for overlays (61–70) so display artifacts are not interpreted as scientific evidence.

### 14. code/generate_vgg_gradcam_for_dashboard.py — redirect to shared explanation implementation [conditional]

Evidence: 23–25 always explains the positive sigmoid; 78–86 rebuilds a hard-coded VGG head and loads a fixed fold's best_weights; 92 takes first unsorted images. Required: explicit class target, checkpoint/config-based loading, deterministic documented selection, recording IDs and numerical map export. Its RGB/255 path already matches supplied training; preserve consistency rather than replacing it merely because the backbone was ImageNet-initialized.

### 15. code/run_gradcam.py — correct sigmoid target and model reconstruction [conditional]

Evidence: 54–56 ignores class_index for binary models even though 296–301 passes one. Model construction at 16–30 hard-codes dropout .3, unlike the trainer's configured value; dropout is inactive during inference but reconstruction still drifts from actual saved configuration. Sampling at 285 is unseeded, and 305 infers class order from available input folders rather than the trained label mapping. Loading failures return instead of producing a clear failed-run status (254–256).

Required: shared binary Grad-CAM target logic, exact run configuration, stored class map, deterministic sampling/selection manifest, numerical maps and explicit error states. The fallback layer search at 36–39 should find a real convolution layer compatible with the selected Keras version, not merely any four-dimensional output.

## Two alternate orchestrators: retire from the supported training route

### 16. code/main.py — legacy incompatible route [redirect or mark unsupported]

Evidence: imports at 4–6 refer to unavailable preprocessing and My_Project.model_training_resnet50; 16 defines four acquisition classes; 37–38 splits filenames 70/15/15; 22 recursively deletes output directories; 47 moves images. Calls at 63–79 do not match the available training/evaluation function interfaces.

Recommendation: preserve as historical source and remove it from recommended execution instructions, or turn it into a thin caller of the one supported manifest-driven pipeline. Do not repair its independent splitter and create another competing evaluation protocol.

### 17. code/main_clean_scalograms.py — same legacy issues [redirect or mark unsupported]

Evidence: absent preprocessing_clean/My_Project imports at 4–6, independent file splitter at 25–47, destructive setup at 18–23, and old training interfaces at 63–79. Recommendation: same treatment as main.py; if clean generation is needed, call the shared generator explicitly and keep generation separate from splitting/training.

## Supporting files/components

requirements.txt has TensorFlow/Keras/NumPy/Pandas/Matplotlib/Seaborn/scikit-learn/Pillow lower bounds but omits direct dependencies used by scripts: PyWavelets, SciPy, tqdm, psutil and OpenCV (if the older Grad-CAM tool is retained). Pin a compatible environment after deciding the supported tools; no dependency install or runtime compatibility check was performed. README execution instructions must point to the chosen route and the actual insulation data/manifest.

A small shared manifest/data component is missing: it must load trusted mapping, construct/validate grouped memberships, expose ordered datasets and share preprocessing/label metadata with all consumers. This can be a new module (for example dataset_manifest.py/data_pipeline.py), rather than inserting separate grouping logic into every old script. A mapping spreadsheet or CSV is useful only if its recording IDs accurately represent acquisition lineage; the present numbering cannot supply that automatically.

## Correct implementation sequence

1. Establish trusted recording/image mapping and the chosen scientific image representation. Keep original names; add IDs in a separate manifest. Unknown lineage is a data issue that code alone cannot resolve.
2. Build one non-destructive grouped splitter and save one locked holdout plus five development folds shared by VGG16 and ResNet50. Assert recording/session separation and manifest coverage; inspect potential content-copy inconsistencies without inferring acquisition provenance from hashes.
3. Correct/consolidate the trainer and launchers. Use the actual fold file lists; make training-only augmentation explicit; save architecture, seeds, settings and checkpoint identity. Fail on missing groups or incomplete folds.
4. Use development CV for model/settings selection. If outer-fold performance is to be an unbiased reported estimate while tuning/early stopping occurs, add an inner grouped validation split; do not call the same tuning fold an untouched outer test. Otherwise clearly describe CV as development validation and reserve generalization claims for the locked holdout.
5. Correct evaluation and prediction exports. Select the final training/refit and checkpoint policy using development information; evaluate the locked independent holdout according to the declared protocol. Report image and recording endpoints distinctly.
6. Align ROC/uncertainty and Grad-CAM downstream. Change scalogram rendering only if the approved scientific method calls for reprocessing, and version those new inputs separately.

## Verification required after implementation, before expensive runs

Meaningful lightweight checks should prove: effective loaded IDs equal manifest IDs; no recording/session group crosses partitions; folds differ and cover prescribed development groups; VGG/ResNet use identical membership; each factory builds the intended backbone; binary evaluation thresholds a sigmoid; augmentation is exercised only on training with finite tensors and unchanged labels; failed folds cannot disappear; predictions retain exact IDs/checkpoint hashes; and saving/reloading one small smoke-test checkpoint preserves evaluation behavior. These checks have not been run against corrected code because no code has been changed yet.

Static validation now: all 17 files read; 16 parse successfully, model_training.py fails at 675. Parsing does not establish TensorFlow runtime compatibility. The source inventory confirms the research code was unchanged during this review.
