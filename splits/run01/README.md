# Validated new split

All 1600 user-confirmed recording IDs are preserved. One CSV per manually renamed
image is a user-provided acquisition statement; IDs are bookkeeping identifiers,
not recovered original CSV filenames. No electrode/session metadata was invented.

The new split uses the complete mapped image set, not the historical Train/Test
folder assignments. It reads originals in place without moving or editing files.

- Fixed holdout: 320 images (160 Nomex, 160 Teflon).
- Development: 1280 images, split into five distinct validation folds.
- Each iteration: 1024 train (512/class), 256 validation (128/class).
- Every development image is held out for validation exactly once.
- All pairwise image/recording/effective-group/content overlap checks pass.

Files with identical whole-file SHA256 values are conservatively kept in the same
effective split group, without renaming or equating their distinct recording IDs.
There are 1284 such effective groups across 1600 declared recording IDs. Grouping
does not prove or refute independent laboratory acquisition. No images were removed.
Effective groups also control final-fit inner validation and uncertainty clustering.

`split_config.json` records the source mapping hash, seed 42, original acquisition
group policy, content-co-location option, effective grouping and split file hash.
`VALIDATION_REPORT.json` records actual fold/class counts and validation results.
No full model training was started while generating this split.
