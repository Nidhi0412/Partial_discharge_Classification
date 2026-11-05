# Model identity verification

Date: 2026-11-02. No inference, TensorFlow import, Keras deserialization or dependency installation occurred. Status definitions and absolute path aliases are in EVIDENCE_RECOVERY_REPORT.md.

## June VGG-labeled insulation run

**CONFIRMED — all eight recovered checkpoints have a VGG16 backbone.**

Exact archive:
`D:\Nidhiiiiii\Paper_04_Saurabh_wavelet\Paper_04_Saurabh_wavelet\with new results\Insulation_classifier_VGG16_CV_Aug_20250604_205456-20250614T122102Z-1-001.zip`

Exact member prefix:
`Insulation_classifier_VGG16_CV_Aug_20250604_205456/VGG16_CV_Aug_20250604_205459/VGG16/CV_Aug_20250604_205459/ResNet50/`

Under that prefix, `fold_2`, `fold_3`, `fold_4`, `fold_5` each contain `model.h5` and `best_weights.h5`. Every file has HDF5 magic and a serialized model_config. The inspected Functional graph contains:

- 13 Conv2D layers in VGG's five blocks: block1 has 2, block2 has 2, blocks3–5 have 3 each.
- Five MaxPooling2D layers named block1_pool through block5_pool.
- GlobalAveragePooling2D, Dropout and Dense(units=1, activation=sigmoid).
- No residual Add layers and no ResNet convN_blockM branch graph.
- Three trainable backbone convolutions: block5_conv1, block5_conv2, block5_conv3.
- No kernel regularizer on the binary output head; serialized training configuration is Adam/binary_crossentropy.

This conclusion uses graph structure, not the VGG archive name or misleading ResNet50 member directory. Config snapshots for each file are `checkpoint_fold_N_{model,best_weights}_model_config.json`; CHECKPOINT_METADATA.json lists full source/member paths, checkpoint hashes, config byte offsets, layer counts and training configs. Root config JSON was recovered from the first MiB of each HDF5 byte stream using the Python standard library; it was parsed as data, not executed. This verifies the serialized graph, not tensor integrity, numerical behavior, ImageNet weight provenance or optimizer reproducibility. No h5py was needed for this limited metadata inspection.

For example, Fold 4 `model.h5` has SHA256 `16ac1d143d363276e47e653ae7045bf8aa5875b3e98b61d21e339ca9c3aa7b16`; Fold 5 `model.h5` has `8d7a130d194e262227f8f4f57387ad917ac56598fcbf13088f320fb47a8cc869`. The similarly named best checkpoints have different full-file hashes. That proves different serialized files, not necessarily different trainable tensors: optimizer state and metadata can also differ.

## Connection to 98.20% results

**CONFIRMED — documentary association is strong.** All four archived `fold_N/detailed_classification_report.txt` files have identical SHA256 hashes to:
`D:\Nidhiiiiii\Partial_discharge_Classification_minimal\Partial_discharge_Classification_minimal\results\VGG16\VGG16\fold_N_report.txt`

REPORT_LINKS.json records both exact paths and hashes. These are the reports whose rounded accuracies average to 98.20%. Archived Fold 4/5 training histories also exactly match:
`D:\Nidhiiiiii\Paper_04_Saurabh_wavelet\Paper_04_Saurabh_wavelet\with new results\VGG16_insulation\fold_N\training_history.json`

Their epoch counts are 43 and 38. The archive also recovers 43-epoch Fold 2 and 50-epoch Fold 3 histories, absent from those separate export folders. HISTORY_LINKS.json records these joins.

**NOT VERIFIABLE — exact report-producing checkpoint.** Reports name folds, not checkpoint hashes or input manifests. Co-location and matching exports cannot prove that a specific final or best checkpoint produced those probabilities. Original predictions, filenames and an evaluation record are required to close that link. Current evidence supports “genuine VGG16 checkpoints recovered alongside the exact historical VGG reports,” not “98.20% independently reproduced.”

## Reassessment of Phase 1 F22

**CONTRADICTED — the inference that the recovered June VGG checkpoint set is actually ResNet50.** Its actual serialized architecture is VGG16.

**CONFIRMED — the supplied current source still has the dispatch bug.** At `D:\Nidhiiiiii\Partial_discharge_Classification_minimal\Partial_discharge_Classification_minimal\code\model_training_vgg16.py:1407`, accepted CLI choices are lowercase; at 1344 the factory compares uppercase `VGG16`. This is a current-code defect, not proof of the historical model family. The exact command and source version that created the archived VGG files are **NOT VERIFIABLE**. A corrected older version, direct Python invocation or another entrypoint is possible, but none was recovered. No original code was fixed in Phase 2.

## May ResNet-labeled run, CV_Aug_20250529_210046

**CONFIRMED — named historical config and matching metric exports exist.**

- `D:\Nidhiiiiii\Partial_discharge_Classification_minimal\Partial_discharge_Classification_minimal\results\ResNet50\experiment_config.json` names `CV_Aug_20250529_210046`, classes Nomex/Teflon and the expected hyperparameters. It contains no backbone layer graph.
- `D:\Nidhiiiiii\Paper_04_Saurabh_wavelet\Paper_04_Saurabh_wavelet\with new results\ResNet50_insulation\crossval_summary.txt` matches the exported fold scores and 97.03% rounded mean.
- `D:\Nidhiiiiii\Paper_04_Saurabh_wavelet\Paper_04_Saurabh_wavelet\with new results\fwdresnet50_insulation\fine_tuning_report.txt` lists 178 layers including conv5_block2/conv5_block3 names, consistent with a ResNet50-family model. The ZIP `fwdresnet50_insulation.zip!fine_tuning_report.txt` provides a similar report, plus the matching crossval summary.

**NOT VERIFIABLE — direct checkpoint identity for the 97.03% run.** No May-run model.h5, best_weights.h5 or serialized model graph was recovered. The fine-tuning text lacks a checkpoint hash and experiment ID tying its graph listing uniquely to the claimed run. It is corroboration, not a substitute for checkpoint inspection. The June VGG files below a ResNet50 directory must not be repurposed as the missing May ResNet model.

## Excluded checkpoints

**CONFIRMED — a separate multitask checkpoint exists** at `D:\Nidhiiiiii\Mutlitask_PD_Classification\Mutlitask_PD_Classification\models\Multitask_VGG16\Multitask_VGG16_20260720_152331\fold_1\best_weights.h5`. It belongs to a later, two-head material/electrode workflow described by that project's source and log. It has no recovered linkage to AUT fold predictions and was excluded. PyTorch .pt files in the Gnostics/1D CNN project likewise cannot identify either AUT backbone.

## What an additional verification stage would require

Metadata recovery is complete without additional dependencies. Full HDF5 attribute/tensor validation would require an available compatible HDF5 reader such as h5py; it was not installed. Numerical reproduction would additionally require compatible TensorFlow/Keras, image decoding, preprocessing and the original evaluation data/labels, plus explicit approval for inference. No such inference or installation was started. Even reproducing old scores on a leaky test copy would not establish recording-independent generalization.
