# Tasks

Run tests from `app/ml_engine` with `pytest src/ -v`. Work on branch `AB-112-grouped-validation-protocol` from `develop`.

## 1. Fold assignment

- [x] 1.1 Add `training/folds.py` with `assign_folds(samples, n_splits, seed) -> dict[Path, int]` (StratifiedGroupKFold, shuffled, on labels and `group_id` of the samples) and `fold_splits(fold_of_row, n_splits)` returning `(fit_idx, val_idx)` pairs and raising `ValueError` naming an empty fold; verify with new `tests/test_training_folds.py`: every sample gets one fold in 0..n-1, no `group_id` in two folds, same seed gives an identical assignment, a different seed changes it, and an empty fold raises with its number
- [x] 1.2 Add a test that the assignment of a sample does not depend on which other samples later get rows (assign once, then drop rows as the gate would, and check `fold_splits` uses the original folds); verify it passes

## 2. Shared splits in selection, ablation and calibration

- [x] 2.1 In `model.py`, replace `cross_validate_macro_f1` with `cross_validate_scores(classifier, matrix, y, n_radiomic, splits, seed, blocks)` using `cross_validate` with `f1_macro` and `balanced_accuracy`; extend `CandidateScore` with mean, std and per-fold balanced accuracy; make `compare_classifiers` take `splits` instead of `groups`/`n_splits`; delete `group_folds` and `N_SPLITS` if unused; verify `tests/test_training_model.py` (moved fold tests now live in 1.1) and a new test that selection still picks the highest mean macro-F1 when another candidate has a higher balanced accuracy
- [x] 2.2 Make `ablate_blocks` take `splits` instead of `groups`/`n_splits` and keep returning macro-F1 per variant; verify `tests/test_training_ablation.py`
- [x] 2.3 Make `calibrate(model, matrix, y, splits)` pass the given splits and `ensemble=True` explicitly to `CalibratedClassifierCV`, with a docstring stating the per-fold ensemble; verify `tests/test_training_calibration.py`, updated so the "never splits a group" test checks the passed splits are used as-is and that the result holds one calibrated model per fold
- [x] 2.4 Add a test that selection, ablation and calibration receive identical validation indices for every fold (record the splits each one gets during a synthetic run); verify it passes

## 3. Metrics

- [x] 3.1 Add `balanced_accuracy` to `EvaluationMetrics` from the same predictions (one `predict` call); verify `tests/test_training_evaluation.py` gains the spec example (3 glioma right, 1 notumor wrong gives accuracy 0.75 and balanced accuracy 0.5)

## 4. Run, report and fold record

- [x] 4.1 In `run.train`, call `assign_folds` on `split.train` right after `split_samples`, map surviving rows to their folds, build the splits once with `fold_splits` and pass them to selection, ablation and calibration; verify `tests/test_training_run.py` still passes, including "Test split is scored exactly once" and "same seed reproduces the same metrics"
- [x] 4.2 Build the fold record (relative path, label, `group_id`, fold, used) and have `write_report` write it as `folds.csv`; add `validation` (`method`, `n_splits`, `seed`, `used_per_fold_and_class`) to `report.json`, and to `report.md` add a per-fold/class table, CV balanced-accuracy columns and test balanced accuracy; verify `tests/test_training_report.py` and a run test that `folds.csv` has one row per purged Training image and that its used rows per fold equal the validation rows of the splits
- [x] 4.3 Add a run test where the gate rejects an image: it is in `folds.csv` with `used = 0` and in no split; and a test that a run whose gate empties a fold stops with an error naming the fold before fitting; verify both pass
- [x] 4.4 Add test balanced accuracy and CV balanced accuracy to the artifact manifest metrics (they come through `asdict`); verify `tests/test_training_artifact.py` and that `tests/test_pipeline_*` still load the artifact

## 5. Documentation

- [x] 5.1 Add decision D16 to `docs/ml-engine/decisions.md`: folds assigned before the gate and shared by all CV steps, `folds.csv`, balanced accuracy as a reported (not deciding) metric, and `ensemble=True` with its alternative; verify that the entry names its status (Verified on synthetic data / not run on real data)
- [x] 5.2 Update `docs/ml-engine/preprocessing-pipeline.md` (B9 traceability row and its state, G7 noted as tested) and the training section of `app/ml_engine/README.md` (mention `folds.csv` and balanced accuracy); verify the commands in the README run as written on the synthetic dataset

## 6. Integration check

- [x] 6.1 Run the full `pytest src/ -v` and `black --check src` in `app/ml_engine`, then run the training CLI end to end on the synthetic dataset and confirm the report folder contains `report.json`, `report.md` and `folds.csv` with the new fields

## Workflow follow-up

- Run the training on the real dataset and attach its report to Azure #112 (out of scope here).
- Commit with `AB-112:` messages, open the PR to `develop`, then archive the change after review.
