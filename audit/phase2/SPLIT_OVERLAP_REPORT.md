# Split overlap report

Date: 2026-11-02. This report measures existing image files only. It does not invent original recording IDs or equate byte hashes with acquisition identities.

## Exact dataset locations and counting units

Insulation tree:
`D:\Nidhiiiiii\Mutlitask_PD_Classification\Mutlitask_PD_Classification\Classifications_data\Classifications_data\Insulation_classifier`

Four-configuration tree:
`D:\Nidhiiiiii\Mutlitask_PD_Classification\Mutlitask_PD_Classification\Classifications_data\Classifications_data\Full_4Class_Classifier`

**CONFIRMED:** each tree has 1600 existing PNG files. All are 224×224, PNG bit depth 8/color type 6 (RGBA). Insulation has 640 Train + 160 Test per material; four-class has 320 Train + 80 Test per configuration. DATASET_PROVENANCE.csv records 3200 file observations across these two arrangements, not 3200 independent examples.

**CONFIRMED:** both trees have exactly the same multiset of 1600 SHA256 file hashes and multiplicities. There are 1284 distinct byte contents: 1016 occur once, 228 twice, 32 three times and 8 four times in each tree. Identical whole-file bytes necessarily imply identical decoded image content; these measurements do not depend on image-library behavior. The true number of distinct decoded pixel arrays could be lower if other PNGs differ only in encoding/metadata. No perceptual-near-duplicate count was estimated.

**CONFIRMED:** every insulation image content matches exactly one four-configuration folder label, although that content may appear in several files. This recovers an observed content-to-folder-label association, not the raw recording ID, acquisition conditions, mother wavelet or coefficient position. Full matching candidate paths are in the provenance CSV; numbering was not interpreted as provenance.

## Exact Train/Test image overlap

**CONTRADICTED — recovered insulation Train/Test images are disjoint.** There are 96 distinct content hashes present in both Train and Test. They cover 217 files in total: 112 Train and 105 Test. Thus 105/320 Test files (32.8125%) have an exact byte-identical Train counterpart. Of these Test files, 103 have the Nomex folder label and 2 Teflon. This is a confirmed dataset-copy overlap; a claim of historical AUT contamination still requires input-tree linkage.

Exact insulation examples:

- `...\Insulation_classifier\Train\Nomex\Nomex_01.png` and `...\Insulation_classifier\Test\Nomex\Nomex_301.png`
- `...\Insulation_classifier\Train\Nomex\Nomex_04.png` and `...\Insulation_classifier\Test\Nomex\Nomex_304.png`
- `...\Insulation_classifier\Train\Nomex\Nomex_09.png` and `...\Insulation_classifier\Test\Nomex\Nomex_309.png`

The `...` in these examples is the exact shared prefix `D:\Nidhiiiiii\Mutlitask_PD_Classification\Mutlitask_PD_Classification\Classifications_data\Classifications_data`. Full absolute paths and SHA256 hashes for every overlap file are in EXACT_DUPLICATE_OVERLAP.csv; no examples rely on inferred numeric relationships.

**CONTRADICTED — recovered four-class Train/Test images are disjoint.** There are 89 distinct cross-partition hashes covering 206 files, including 96 Test files (30% of 320; 95 Nomex and 1 Teflon). Examples include:

- `...\Full_4Class_Classifier\Train\Nomex_Flat\Nomex_Flat_12.png` and `...\Full_4Class_Classifier\Test\Nomex_Flat\Nomex_Flat_312.png`
- `...\Full_4Class_Classifier\Train\Nomex_Flat\Nomex_Flat_303.png` and `...\Full_4Class_Classifier\Test\Nomex_Flat\Nomex_Flat_03.png`

The two representations have different partition counts for 442 content-hash groups. Therefore copying four-class Train/Test assignments into an insulation manifest would be incorrect even though the overall content multisets match. CROSS_DATASET_PARTITION_DIFFERENCES.json records each such group.

## Provenance boundaries

**NOT VERIFIABLE — original recording overlap.** No acquisition ID or original-CSV-to-image manifest was recovered. Exact image duplication establishes that these images are not independent input observations; it does not identify the original recording or distinguish repeated copying from duplicate raw acquisitions or repeated rendering. Raw recordings could also generate nonidentical wavelet variants across partitions, so removing only exact duplicate hashes would not prove recording separation.

**NOT VERIFIABLE — how 400 originals became 1600 images.** The image counts match the manuscript's total, but the numbered filenames encode neither wavelet nor source ID. A historical draft says four db1–db4 variants, while another paragraph says eight variants. The observed 1284 byte-distinct PNGs do not establish 1284 independent recordings or 400 originals. DATASET_PROVENANCE.csv leaves raw IDs, source CSV paths, wavelet and DWT fields blank rather than reconstructing them from filename ranges.

**NOT VERIFIABLE — exact AUT model input membership.** The trees reside in the later multitask project. Their 1600/1280/320 counts and Nomex/Teflon organization match historical reports, and all files match the October 2026 multitask export ZIP, but no historical input manifest proves that these are unchanged copies of May/June 2025 training/test folders. Present evidence does not allow the recovered overlap rate to be assigned numerically to either reported model.

## Historical cross-validation status

**CONFIRMED — current supplied source uses repeated fixed directory membership.** `D:\Nidhiiiiii\Partial_discharge_Classification_minimal\Partial_discharge_Classification_minimal\code\model_training_vgg16.py:991` obtains fold indices; lines 1009–1029 create generators from the same train_dir and fixed .2 subset/seed. The exact historical source/commands remain **NOT VERIFIABLE**, so this source defect cannot alone prove what the recovered June run did.

**NOT VERIFIABLE — historical distinct/grouped folds.** Recovered histories/epoch logs identify four available runs but no train_idx/val_idx arrays, image/recording membership, or split hashes. Four logs with different learning curves are evidence of training runs, not independently partitioned folds. Fold 1 and its exclusion reason are missing. Historical evaluation scripts use the same fixed Test directory for each model; whether the historical inputs were unchanged requires the missing manifests. No existing evidence establishes recording-independent five-fold CV.

## Consequence for scientific claims

These recovered partitions cannot support a claim of disjoint-image evaluation as they stand. Do not repair the historical result by deleting duplicates only from Test or retraining on an arbitrarily deduplicated numbering scheme. Recover acquisition lineage, group all variants by recording/session, then use a locked independent test and effective grouped development splits. If the recovered leaky copy was the actual training/evaluation input, old metrics require replacement under a sound design. If it was only a later altered copy, recover the actual historical manifest before judging that experiment's overlap.
