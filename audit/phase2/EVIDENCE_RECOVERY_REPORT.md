# Phase 2: Historical evidence recovery

Date: 2026-11-02, Asia/Calcutta. Original code, datasets, archives and checkpoints were read only. No model was instantiated, no inference or training ran, and no dependencies were installed. All recovery outputs are in this new `audit/phase2` directory; Phase 1 files remain unchanged.

Status meanings: **CONFIRMED** means the specified artifact directly supports the precisely scoped claim; **CONTRADICTED** means an artifact directly conflicts with that claim; **NOT VERIFIABLE** means the required evidence was not recovered. A folder label or manuscript statement alone does not prove acquisition conditions, recording identity, model identity or runtime membership.

## Findings that change Phase 1's evidence position

1. **CONFIRMED — A genuine VGG16 checkpoint set was recovered.** Eight HDF5 checkpoints in the June insulation archive serialize VGG16's actual layer graph, despite being stored below a `ResNet50` directory. All four archived fold reports are byte-identical to the minimal repository's VGG16 reports. Two archived training histories are byte-identical to the separate VGG exports. The current source dispatch bug therefore does not establish that the historical VGG results were ResNet50. See MODEL_IDENTITY_VERIFICATION.md and CHECKPOINT_METADATA.json.
2. **CONFIRMED — A 1600-image insulation dataset was recovered, with exact Train/Test duplication.** It contains 1280 Train and 320 Test PNGs, but only 1284 distinct whole-file SHA256 hashes. There are 105 Test images with a byte-identical Train image (32.8125% of Test). Its connection to the exact AUT input tree is **NOT VERIFIABLE** without an original input manifest or recording map. See SPLIT_OVERLAP_REPORT.md.
3. **CONFIRMED — Earlier wavelet-selection narrative exists, but the generating implementation remains missing.** A class-specific energy note discusses db2 and db4, and a historical manuscript claims db1–db4. The same manuscript contradicts itself about four versus eight images per recording. These documents cannot substitute for source code or provenance.
4. **NOT VERIFIABLE — Recording-independent historical evaluation and prediction-level reproduction remain unestablished.** No original recording IDs, complete 400-recording campaign, effective historical fold memberships, exact training commands, or AUT per-image probability/label files were recovered.

## Exact evidence locations

The following aliases are defined solely to make long paths readable; append the specified relative path to obtain the exact source location. ZIP members use `/` separators and are not extracted over original folders.

- **MIN** = `D:\Nidhiiiiii\Partial_discharge_Classification_minimal\Partial_discharge_Classification_minimal`
- **PAPER** = `D:\Nidhiiiiii\Paper_04_Saurabh_wavelet\Paper_04_Saurabh_wavelet\with new results`
- **DATA** = `D:\Nidhiiiiii\Mutlitask_PD_Classification\Mutlitask_PD_Classification\Classifications_data\Classifications_data`
- **VGG_ARCHIVE** = `D:\Nidhiiiiii\Paper_04_Saurabh_wavelet\Paper_04_Saurabh_wavelet\with new results\Insulation_classifier_VGG16_CV_Aug_20250604_205456-20250614T122102Z-1-001.zip`
- **VGG_MEMBER_BASE** = `Insulation_classifier_VGG16_CV_Aug_20250604_205456/VGG16_CV_Aug_20250604_205459/VGG16/CV_Aug_20250604_205459/ResNet50/`

## Search coverage and preservation

**CONFIRMED:** WORKSPACE_FILE_INVENTORY.csv inventories 4241 files across the five research project directories plus root archives in `D:\Nidhiiiiii`, excluding `.git`, virtual environments, dependency/cache directories and audit outputs. Eight accessible ZIP files were indexed, including the large VGG archive inside the Paper project. ARCHIVE_MEMBER_INVENTORY.csv gives every member name, uncompressed size and ZIP CRC; ARCHIVE_HASHES.csv gives full SHA256 hashes of the archive files. The outer Paper ZIP and its separately available inner ZIP were indexed; the accessible inner ZIP was inspected directly rather than redundantly decompressing the outer copy. All 3200 recovered dataset PNGs were also checked against their source multitask ZIP and match byte-for-byte (DATASET_ARCHIVE_COPY_VERIFICATION.json).

The search included local source text, experiment configurations, historical result exports, CSV headers, checkpoints, training histories/logs, archive contents, available Git branch histories, and relevant OOXML manuscript/workbook text. GIT_HISTORY_SCOPE.json lists the three accessible Git repositories and their available commits. The minimal repository's earliest available commit is `13ea2d0f589e8f8735afc954bba66e2e081d3d76`, dated 2026-02-27, after the 2025 experiment dates. No exact 2025 source revision or command line was recovered from that history. Git was queried with read-only commands and command-scoped ownership exceptions; no persistent Git configuration was changed.

**CONFIRMED:** all 63 original files in Phase 1's inventory still have their original hashes (PRESERVATION_CHECK.json). Checkpoint hashes and serialized configuration snapshots were saved as evidence; checkpoint files themselves were not extracted or rewritten. TensorBoard TFRecord envelopes were inspected without TensorFlow, recording their hashes and record counts. No explicit Nomex/Teflon/input-path or fold-index strings were found in those streams; the limited byte scan is not a full TensorBoard semantic decoding and does not prove that no recoverable information exists in every possible embedded representation.

**NOT VERIFIABLE outside scope:** other disks, unavailable historical Linux/Windows paths, unmounted storage, cloud folder contents, remote Git branches not present locally and acquisition PCs. No cloud folders were fetched. A missing artifact here means not recovered from this accessible workspace, not nonexistent everywhere.

## Priority 1: Model identity

**CONFIRMED:** every `VGG_ARCHIVE!VGG_MEMBER_BASE/fold_{2,3,4,5}/{model.h5,best_weights.h5}` stores a Functional model with 22 layers, 13 Conv2D layers named block1_conv1 through block5_conv3, five MaxPooling2D layers, GlobalAveragePooling2D, Dropout and a one-unit sigmoid Dense output. The three block5 convolutions are trainable. This is a VGG16 backbone, not a residual architecture. `CHECKPOINT_METADATA.json` preserves exact member names, full checkpoint hashes, JSON offsets and configurations.

**CONFIRMED:** archived `detailed_classification_report.txt` files for folds 2–5 match `MIN\results\VGG16\VGG16\fold_N_report.txt` byte-for-byte (REPORT_LINKS.json). Archived fold 4/5 histories match `PAPER\VGG16_insulation\fold_N\training_history.json` (HISTORY_LINKS.json). This establishes a strong documentary link between this genuine VGG checkpoint set and the VGG result exports. **NOT VERIFIABLE:** that a particular checkpoint actually generated a report, because no evaluation manifest or per-image predictions tie its hash to the metrics.

**NOT VERIFIABLE:** no checkpoint from `CV_Aug_20250529_210046` was recovered. `MIN\results\ResNet50\experiment_config.json` identifies that run but contains no model graph. `PAPER\fwdresnet50_insulation\fine_tuning_report.txt` records a 178-layer residual model with conv5_block2/conv5_block3 names and matches the task's expected architecture family; without its checkpoint hash or an authenticated evaluation linkage, this is corroborating text rather than direct identity verification of the 97.03% run. See MODEL_IDENTITY_VERIFICATION.md.

## Priority 2: Raw data and provenance

**CONFIRMED:** `DATA\Insulation_classifier` has 1600 observed 224×224 RGBA PNG files, 800 per material folder. `DATA\Full_4Class_Classifier` separately has 1600 files, 400 per configuration folder. Both have exactly the same SHA256 content multiset, including duplicate multiplicities. They are two arrangements of the same image content, not 3200 independent images. Every insulation image hash matches one unique configuration label in the four-class tree, although often multiple files share that hash. DATASET_PROVENANCE.csv records all 3200 actual file observations, full paths, hashes, folder labels, partitions and exact-content matches. No raw recording ID is inferred.

**CONTRADICTED:** these recovered Train/Test trees are byte-image-disjoint. See duplicate evidence and exact examples in SPLIT_OVERLAP_REPORT.md. **NOT VERIFIABLE:** recording-level separation and whether the historical AUT run used exactly this tree. The two representations even assign different Train/Test multiplicities to 442 content-hash groups (CROSS_DATASET_PARTITION_DIFFERENCES.json), so the four-class partition cannot stand in for the insulation partition.

**CONFIRMED:** `D:\Nidhiiiiii\Multi-class-Machine-gnotics-codes\Multi-class-Machine-gnotics-codes\streamlit_cloud\sample_uploads` contains eight Tektronix-format sample CSVs but just four distinct whole-file hashes. Header evidence includes Model MDO3024, Record Length 10000 and Sample Interval 4e-10 s for the inspected samples; RAW_CSV_CANDIDATES.json records each complete header. **NOT VERIFIABLE:** these renamed demo samples' original acquisition IDs, their relationship to AUT images, or completeness of the claimed 400 recordings. They cannot populate missing recording-to-image mappings or establish campaign-wide acquisition bandwidth.

## Priority 3: Historical validation procedure

**CONFIRMED:** June VGG training histories/logs exist for folds 2–5 with 43, 50, 43 and 38 completed epochs. Four TensorBoard train event files and many saved augmented PNGs also exist in the archive. The configs request five splits and identify Nomex/Teflon. Epoch logs record metrics and learning rates, not training/validation filenames or recording groups. Fold 1 artifacts were not recovered in that archive. Snapshots: `archive_fold_N_training_history.json`, `archive_fold_N_training_log.csv`, ARCHIVE_TEXT_EVIDENCE.json, TENSORBOARD_RECORD_INVENTORY.json.

**CONFIRMED:** `PAPER\ResNet50_insulation\fold_{3,4,5}\training_history.json` records 19, 31 and 30 epochs; fold 2/5 CSV logs also exist. Its saved mean accuracy matches the minimal export. **NOT VERIFIABLE:** exact originating checkpoint, source revision, commands and effective fold membership.

**CONFIRMED:** the currently supplied source ignores fold indices and evaluation code reuses a fixed Test directory (`MIN\code\model_training_vgg16.py:991–1029`, `crossval_aggregate_evaluation.py:14,86–101`, `eval_roc_auc.py:16,105,237–244`). **NOT VERIFIABLE:** that the recovered historical run executed precisely those versions. Its VGG checkpoint graph conflicts with the current CLI dispatch behavior, demonstrating that direct historical extrapolation is unsafe. No recovered evidence establishes distinct grouped folds; current documentation's five-fold claim remains unsupported. Neither a completed-epoch history nor fresh random weights proves changed fold membership.

## Priority 4: Results and reproduction

**CONFIRMED:** the VGG archive reports reproduce the exported summary arithmetic: .9844, .9812, .9812, .9812 average to .9820. `PAPER\VGG16_insulation\crossvalidation_avg_classification_report.xlsx`, sheet1 cells G9:H13, contains exactly those run labels, accuracies and average. Its I13/K13 give precision 1.0 and F1 .9817. The saved VGG Fold 5 confusion figure shows [[160,0],[6,154]], consistent with its text report; there is no contradictory Fold 5 value in these recovered artifacts. These are existing report/figure checks, not generated replacement results.

**CONFIRMED:** `PAPER\ResNet50_insulation\crossval_summary.txt` repeats the four minimal-export scores and mean .970313. **NOT VERIFIABLE:** per-image accuracy, ROC/AUC or paired statistical inference from original predictions; no AUT y_true/y_prob, prediction CSV, or checkpoint-identified evaluation manifest was found. Unrelated 1D CNN/Gnostics predictions, electrode feature NPZ and 2026 multitask checkpoint were excluded from AUT verification.

## Priority 5: Earlier wavelets and selection

**CONFIRMED:** `PAPER\vgg16insulation_results\energy_analysis.txt`, lines 1–27, records highest-mean-energy coefficient recommendations by NN/NF/TF/TN class; headings explicitly mention db2 and db4, and other headings omit wavelet identity. The final db4 block recommends NN DC1, NF DC4, TF DC5, TN DC4. It recommends selecting levels by known acquisition class. It does not supply the energy formula, entropy computation, recording list, coefficient convention or proof that those rules built the final dataset. Do not assign db1/db3 to its unlabeled blocks.

**CONFIRMED:** `PAPER\paper with hybrid results_2025.docx`, paragraph 29, claims four db1–db4 images per original and 1600 images. Paragraph 65 refers to db2. Paragraph 66 says two levels × four wavelets = eight images per original but still calls the total 1600. **CONTRADICTED:** that arithmetic is internally consistent: 400 originals × eight gives 3200. MANUSCRIPT_TEXT_EVIDENCE.json preserves exact paragraphs and the document hash. This is a historical draft, not a verified final AUT manuscript.

**NOT VERIFIABLE:** no executed db1–db4 generator, Shannon entropy selection implementation, final filtering/copying/renaming recipe, wavelet-specific input tree or original-to-output manifest was recovered in local source/archives/Git history. The existing db3 five-position scripts remain a different documented implementation. A 400-images-per-configuration folder cannot establish that four wavelets were used, and filename numbering cannot recover original recording IDs.

## Disposition

**Existing results cannot currently be retained as validated recording-independent performance.** Retain the artifacts as historical evidence. Genuine VGG model identity is substantially recovered; historical ResNet identity and both runs' recording independence are not. A checkpoint-identified independent reevaluation is the next numerical verification stage, but it needs approval, a compatible inference environment and a genuinely independent, traceable test set. If these recovered leaky partitions were used for training/evaluation, or historical group separation cannot be established and the intended claim is grouped generalization, perform fresh grouped training/evaluation after lineage recovery. Complete retraining was not performed, and this recovery cannot prove that every historical model must be discarded.
