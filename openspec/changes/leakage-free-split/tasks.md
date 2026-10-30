# Tasks

All commands run from `app/ml_engine`.

## 1. Leakage-free split (B2)

- [x] 1.1 In `src/training/dataset.py`, add the frozen `Split` dataclass (`train`, `test`, `removed_per_class` with every `TumorLabel`, zeros included) and make `split_samples` drop each Train sample whose `group_id` is in the set of Test `group_id`s. Verify with `pytest src/tests/test_training_dataset.py`.
- [x] 1.2 In `test_training_dataset.py`, replace the folder-only split test with synthetic-`Sample` tests: a Train image sharing a group with Test is removed; a Train image with no Test group member is kept; Test is returned unchanged; `removed_per_class` lists all four labels; no `group_id` is in both splits. Verify the tests pass.
- [x] 1.3 Add `src/tests/test_training_split_reference.py`. It loads the versioned manifest `src/data/processed/dedup_20261028T234450903626Z/manifest.csv` (path resolved from the test file, no `skipif`) and asserts removed counts glioma 142, meningioma 93, notumor 15, pituitary 245, Train 1,830 and Test 1,109. Verify it passes, and that it fails when the path is pointed at a missing file.
- [x] 1.4 Update `test_training_integration.py` for the `Split` return type. Verify with `pytest src/tests/test_training_integration.py`.

## 2. Purge counts in the training run

- [x] 2.1 In `src/training/run.py`, consume `Split` in `_feature_tables` and add `split_removed_per_class` to `_dataset_summary`, so it reaches the report and the artifact `manifest.json`. Print the counts in `src/training/__main__.py`. Verify that a test in `test_training_run.py` asserts the key appears in the report and in the manifest `dataset` section.

## 3. Threshold calibration on purged Training only (B4)

- [x] 3.1 Create `src/training/thresholds.py`: move `calibrate(paths, percentile)` from `src/preprocessing/calibrate.py` unchanged, and add a CLI with required `--data-dir` and `--manifest` and `--percentile` (default 1.0). The CLI calibrates on `split_samples(load_samples(...)).train` and prints `min_snr`, `min_entropy`, `percentile` and `images_used` as JSON. Delete `src/preprocessing/calibrate.py`. Verify `python -m src.training.thresholds --help` works and `python -m src.preprocessing.calibrate` no longer exists.
- [x] 3.2 Replace `src/tests/test_calibrate.py` with `src/tests/test_training_thresholds.py`. Keep the three `calibrate` tests (0th percentile passes the gate, unreadable skipped, nothing readable raises). Add CLI tests on a synthetic dataset plus manifest: `images_used` equals the purged Train count; Testing files and `keep = 0` files are never passed to the loader (patch `load_mri_image` and assert no `Test/` path); the CLI exits with an error without `--manifest`. Verify with `pytest src/tests/test_training_thresholds.py`.

## 4. Calibrate on the versioned dataset and record the values

- [x] 4.1 Run `python -m src.training.thresholds --data-dir src/data --manifest src/data/processed/dedup_20261028T234450903626Z/manifest.csv` and keep the printed JSON. Then, with the new thresholds, count per class how many purged Train images and how many Test images fail `check_quality` (a one-off measurement script; do not commit it). Verify that `images_used` is 1,830 or fewer (minus unreadable images), and that the Train rejection rate is at most about 2% (two thresholds, each at the 1st percentile). If Test rejections concentrate in one class, stop and raise it before task 4.2 (design R2).
- [x] 4.2 Keep the `QualityThresholds` defaults at 4.0 / 2.0 (design D5) and change the docstring to point to D15 (calibrated values recorded, not applied, and why) instead of saying the thresholds were never calibrated. Verify with `pytest src/tests/test_quality.py src/tests/test_prepare.py`.
- [x] 4.3 In `docs/ml-engine/decisions.md`, add D15 (AB-165): the leakage-free split rule, the kept-Testing-groups definition (495 vs 504), the per-class removals, the calibrated `min_snr` and `min_entropy`, the date, the percentile, the image count, the 4.1 per-class rejection counts for purged Train and Test, the visual-review finding, and why the defaults stay unchanged. Update D7 to point to D15, and replace the "Calibrate the quality thresholds" open verification with one for redefining the metrics, using the new command and the follow-up ticket. Verify by reading the entry against the 4.1 output.
- [x] 4.4 Replace `python -m src.preprocessing.calibrate` with the new command in `app/ml_engine/README.md`, rewrite its "provisional" note to point to D15 (calibrated, not applied), and update the B4 row and section of `docs/ml-engine/preprocessing-pipeline.md` (`training/thresholds.py`, status). Verify that `grep -rn "preprocessing.calibrate" app docs` returns nothing.

## 5. Integration check

- [x] 5.1 Run the full ml_engine suite locally with `pytest src/ -v` and check it passes, including the reference test. The CI does not enforce it until AB-174 lands.

## Workflow follow-up

- AB-174 (child of #103) tracks making `ml-validation.yml` select and enforce all ml_engine tests.
- Archive the change after the PR to `develop` is approved and merged.
