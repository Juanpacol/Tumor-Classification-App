# Tasks

Run tests from `app/ml_engine` with `pytest src/ -v`. Work on branch `AB-167-train-only-augmentation` from `develop`.

## 1. Augmentation transforms

- [x] 1.1 Add `training/augmentation.py` with `AugmentationConfig` (copies 1, rotation 10 degrees, flip probability 0.5, scale, brightness and contrast ranges 0.1), `draw(config, rng)` and `augment(image, params)` (flip, one `cv2.warpAffine` for rotation and scale with black border, then contrast around the mean and multiplicative brightness, clipped to [0, 1], `float32`); verify with new `src/tests/test_training_augmentation.py`: neutral parameters return the input, output keeps the 224x224 shape and stays in [0, 1], and drawn parameters stay in range over many seeds
- [x] 1.2 Add `copy_rng(seed, sha256, index)` built on `np.random.default_rng([seed, int(sha256[:16], 16), index])`; verify tests that the same seed, content and index give identical copies, and that a different index or seed gives a different draw

## 2. Copy features and cache

- [x] 2.1 In `training/features.py`, add an augmentation fingerprint (hash of `asdict(AugmentationConfig)` and the run seed) and a copy cache under `<fingerprint>/aug-<augmentation fingerprint>/<sha256[:2]>/<sha256>.npz` holding only the `(k, C)` `cnn` of the copies; on a miss, re-run load and `prepare_image` on the original, augment `cnn_image` and extract CNN features; verify `src/tests/test_training_features.py`: a repeated run recomputes no copy, and changing a range, k or the seed recomputes every copy without touching original entries
- [x] 2.2 Build the copy rows only for usable Training originals, with the original's radiomic row and label; verify tests that a gate-rejected image has no copy, that copy radiomics equal the original's, and that no copy is built for Testing samples

## 3. Splits with copies

- [x] 3.1 Change `fold_splits(fold_of_row, is_copy, n_splits)` so validation rows of fold k are originals of fold k and fit rows are every row of the other folds, still raising on a fold with no original rows; verify `src/tests/test_training_folds.py`: no copy in any validation set, every copy of a fold-k image is a fit row of every other fold and of no fold-k split, and validation index arrays are identical with and without the copy rows appended
- [x] 3.2 Update the existing callers of `fold_splits` for the new argument (all-false mask when there are no copies); verify `src/tests/test_training_run_folds.py` passes unchanged in behavior

## 4. Run: augmentation decision and step order

- [x] 4.1 In `run.train`, append copy rows after the originals with inherited folds, keep selection on the originals, cross-validate the selected classifier with copies, and set `augmentation_on = mean_with >= mean_without + std_without`; run the ablation and calibration on the chosen matrix and splits, reusing the step 1 or step 2 score as the hybrid variant; verify `src/tests/test_training_run.py`, including "Test split is scored exactly once" and "same seed reproduces the same metrics"
- [x] 4.2 Add run tests with a stubbed CV score for the two spec scenarios (0.80 ± 0.02 vs 0.83 turns augmentation on; vs 0.81 turns it off and no copy is a fit row of ablation or calibration), and extend the shared-splits run test so selection, the augmentation decision, ablation and calibration receive identical validation indices; verify they pass
- [x] 4.3 Add a test that importing `src.pipelines` (and loading an artifact) does not import `src.training.augmentation`; verify it passes

## 5. Report and manifest

- [x] 5.1 Add `augmentation` (`on`, `copies`, `ranges`, `cv_without` and `cv_with` mean and std, `margin`) to `report.json`, the artifact manifest and a short section of `report.md`; keep `folds.csv` and `feature_stats` on originals only; verify `src/tests/test_training_report.py`, `src/tests/test_training_artifact.py`, and that the `test_pipeline_*` tests still load the artifact

## 6. Documentation

- [x] 6.1 Add decision D21 to `docs/ml-engine/decisions.md`: augmentation after B6 on the CNN block only, per-copy seed, separate copy cache, copies after originals in the matrix, the in-run decision with its margin, and the alternatives (raw-image augmentation, CLI flag, joint selection); verify the entry states its status (Verified on synthetic data, not run on real data)
- [x] 6.2 Update `docs/ml-engine/preprocessing-pipeline.md` (B7 rule: remove "and in the final fit", state that copies vary the CNN block only and that the run decides on CV; B7 traceability row; G3 and G4 noted as tested) and the training section of `app/ml_engine/README.md` (augmentation and its report section); verify the README commands run as written on the synthetic dataset

## 7. Integration check

- [x] 7.1 Run the full `pytest src/ -v` and `black --check src` in `app/ml_engine`, then run the training CLI end to end on the synthetic dataset and confirm `report.json`, `report.md` and the artifact manifest contain the augmentation section and `folds.csv` has no copy rows

## Workflow follow-up

- B10 / #168 shortcut audit comes next, as its own change, with the revised acceptance criteria agreed with the team.
- The single real-data run (export the backbone, train, record results) comes after #168.
- Commit with `AB-167:` messages, open the PR to `develop`, then archive the change after review.
