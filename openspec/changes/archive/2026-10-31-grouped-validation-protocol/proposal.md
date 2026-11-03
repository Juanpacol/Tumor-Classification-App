# Proposal

## Why

Block B9 of the agreed preprocessing pipeline (`docs/ml-engine/preprocessing-pipeline.md`, Azure #112) makes fold assignment its own step: one `StratifiedGroupKFold` assignment per Training image, made once on the purged Training split and shared by every later step. Today the training code has no such step. Classifier selection, the block ablation and the calibration each rebuild the folds by calling `group_folds().split(...)` on the feature matrix, after the quality gate has dropped some images. The folds are the same only because the matrix is the same, and two things break that:

- **Augmentation (#167)** adds rows. Re-splitting a larger matrix yields different folds, and a group that lands in validation takes its augmented copies with it, which breaks guarantee G3 (no augmented copy in a validation fold).
- **Quality thresholds (#165)** decide which rows exist, so changing a threshold reshuffles every fold.

B2 (#165) is merged, so the purged Training split now exists and B9 can be fixed on top of it. The story also asks for balanced accuracy, which the evaluation does not compute yet.

## What Changes

- New fold assignment step. Every image of the purged Training split gets one `fold_id` from `StratifiedGroupKFold` (5 folds, group = `group_id`, stratified by label, fixed seed). The assignment runs before feature extraction and the quality gate. An image that the gate rejects keeps its fold but contributes no row.
- Classifier selection, block ablation and `CalibratedClassifierCV` all take the same precomputed splits instead of recomputing them from `groups`. `group_folds` and the `groups` / `n_splits` arguments of those functions are replaced by the splits.
- Cross-validation reports balanced accuracy (mean and std) next to macro-F1 for every candidate. Selection still uses mean CV macro-F1.
- The Testing evaluation adds balanced accuracy. Testing is still scored exactly once, by the calibrated model (G7).
- Each run writes `folds.csv` to its report folder (one row per Training image: path, label, `group_id`, `fold`, `used`). The report gives the image count per fold and class. #167 and #168 will use this file.
- Calibration keeps `ensemble=True`: the saved classifier is one calibrated pipeline per fold, averaged. This becomes an explicit, documented decision.
- A run fails with a clear error if the gate leaves any fold without rows.
- Docs: new decision in `docs/ml-engine/decisions.md`, and the B9 row and G7 status in `docs/ml-engine/preprocessing-pipeline.md` are updated.

## Capabilities

### New Capabilities
- `model-validation`: how the classifier is validated during training: the grouped fold assignment and its record, the shared use of the folds for selection, ablation and calibration, the cross-validated and held-out metrics, and the single scoring of the Testing split.

### Modified Capabilities
<!-- None. training-data-split (B2) supplies the purged Training split as its input but its requirements do not change; quality-threshold-calibration is unaffected. -->

## Impact

- **Code** (`app/ml_engine/src/training/`): new `folds.py`; `model.py` (`group_folds` removed, selection takes splits, CV scores gain balanced accuracy), `ablation.py`, `calibration.py`, `evaluation.py` (balanced accuracy), `report.py` (fold counts, balanced accuracy, `folds.csv`), `run.py` (assign folds before features, map them onto the rows that survive the gate).
- **Tests**: `test_training_model.py`, `test_training_calibration.py`, `test_training_ablation.py`, `test_training_evaluation.py`, `test_training_report.py`, `test_training_run.py`, and a new `test_training_folds.py`.
- **Artifacts**: the report JSON and markdown gain fields, and the report folder gains `folds.csv`. The model artifact's file layout does not change. Its manifest metrics gain balanced accuracy. Serving (#113) reads none of the changed fields.
- **Dependencies**: none new.
- **Out of scope**: augmentation (#167), the shortcut audit (#168), the run on the real dataset and its report, and isotonic vs. sigmoid calibration.
