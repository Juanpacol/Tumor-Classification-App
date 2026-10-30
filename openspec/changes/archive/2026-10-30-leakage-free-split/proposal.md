# Proposal

## Why

After deduplication (#150), 263 near-duplicate groups still have kept images in both Training and Testing: 495 Training images have a near-twin in the held-out set, so any test metric partly measures memorization (AB-165; [R1] found 28.8% of this dataset's official test split has a twin in training). Separately, the quality-gate thresholds are still provisional guesses (D7), and the existing calibration CLI reads every image under `Train/` and `Test/`, so calibrating it as-is would fit thresholds on test data.

## What Changes

- **BREAKING** `split_samples` in `src/training/dataset.py` becomes leakage-free: the official Testing folder stays the held-out set, and every Training sample whose `group_id` also belongs to a kept Testing sample is removed. It returns the train and test lists plus the removed count per class.
- The training run reports the purged counts per class in the CLI output, the training report and the artifact `manifest.json` (`dataset` section).
- **BREAKING** Threshold calibration moves from `src/preprocessing/calibrate.py` to `src/training/thresholds.py` (CLI `python -m src.training.thresholds --data-dir ... --manifest ...`). It reads the cleaning manifest, takes only the kept images of the purged Training split, and never opens a Testing image. The old module and CLI are removed.
- Run the calibration on the versioned dataset and record the values and their per-class rejections. The `QualityThresholds` defaults (`min_snr = 4.0`, `min_entropy = 2.0`) stay in place: a visual review showed that the calibrated values mostly reject usable scans and that the rejections concentrate in notumor. Fixing the gate's metrics is tracked in AB-175.
- Record the calibrated values, the date, the number of images used and the per-class rejections in `docs/ml-engine/decisions.md` (D15), and replace the D7 open verification with one for redefining the metrics (AB-175); update the CLI path in the README, the decision log and the pipeline report.
- Add tests: a synthetic test that no `group_id` is in both splits, and a test on the versioned manifest that the purge removes glioma 142, meningioma 93, notumor 15, pituitary 245 (Training 2,325 to 1,830; Testing 1,109 unchanged).

## Capabilities

### New Capabilities
- `training-data-split`: how the cleaned dataset is divided into the training set and the held-out test set, the guarantee that no near-duplicate group crosses that boundary, and how the purge is reported.
- `quality-threshold-calibration`: how the quality-gate thresholds are derived from training data only, and how the calibrated values are recorded.

### Modified Capabilities
<!-- None: no specs exist yet. -->

## Impact

- **Code**: `app/ml_engine/src/training/dataset.py`, `training/run.py` (consumes the new split result and reports purged counts), new `training/thresholds.py`, `preprocessing/quality.py` (docstring only, defaults unchanged), removal of `preprocessing/calibrate.py`.
- **Tests**: `test_training_dataset.py`, `test_training_integration.py`, `test_training_run.py`; `test_calibrate.py` replaced by a test for `training/thresholds.py`; new test reading the versioned manifest.
- **Metrics**: every model trained after this change sees 1,830 instead of 2,325 Training images (pituitary 519 to 274). Cross-validated and test metrics are expected to drop; earlier numbers are not comparable.
- **Feature cache**: unaffected, because the threshold defaults do not change.
- **Docs**: `docs/ml-engine/decisions.md`, `docs/ml-engine/preprocessing-pipeline.md`, `app/ml_engine/README.md`.
- **Out of scope**: the CI workflow `ml-validation.yml` runs ml_engine tests with a `-k` filter and `|| echo`, so it neither selects nor enforces these tests. Fixing that affects every ml_engine test and is tracked in AB-174.
