# Design

## Context

See proposal.md (Why) for the motivation and specs/training-augmentation/spec.md and the model-validation delta for the required behavior.

Current state in `app/ml_engine/src/training/`:

- `run.train` assigns folds on the purged Training samples (`assign_folds`), builds the feature tables through `FeatureCache.table`, maps each surviving row to its fold, and calls `fold_splits` once. Selection, ablation and calibration all receive those `Splits`.
- `FeatureCache` stores one `.npz` per image (`radiomics`, `cnn`, or a rejection `reason`) under `<cache>/<fingerprint>/<sha256[:2]>/<sha256>.npz`. The fingerprint hashes the preprocessing config, the radiomics config and the backbone.
- `prepare_image` returns `radiomics_image` (original resolution), `mask`, and `cnn_image` (224x224, B6). The CNN extractor normalizes `cnn_image` with the bundle statistics (B8).
- `ablate_blocks` reuses the selection score as the hybrid variant. `calibrate` uses `CalibratedClassifierCV(cv=splits, ensemble=True)`.

Constraints: OpenCV and NumPy only, fixed seeds, no change to serving, and the project rule to remove whatever stops being needed.

## Goals / Non-Goals

**Goals:**
- Copies that slot into the existing splits without changing the consumers' signatures.
- Validation rows that are the same index set with and without copies, so scores are comparable fold by fold.
- A cache that never mixes copies made under different settings.

**Non-Goals:**
- Choosing k or tuning the transform ranges.
- Augmenting radiomics or the raw image.
- Selecting the classifier jointly with the augmentation setting.

## Decisions

### D1. Augment the B6 output, not the raw image

The copy is made from `cnn_image`, so B3-B6 and the quality gate are untouched and the copy needs one CNN pass. Its radiomic features are the original's.

*Alternative:* augment the raw image before B4 and re-run the whole pipeline. Rejected: brightness and contrast changes could make a copy fail the gate, rotation adds black corners that B5/B6 would have to absorb, and the cost is a full pipeline run per copy. The evidence for augmentation with a frozen CNN is weak, so the cheaper variant goes to the ablation first.

### D2. One affine warp, then intensity

New module `training/augmentation.py`:

- `AugmentationConfig` (frozen dataclass): `copies = 1`, `max_rotation_deg = 10.0`, `flip_probability = 0.5`, `scale_range = 0.1`, `brightness_range = 0.1`, `contrast_range = 0.1`.
- `draw(config, rng) -> AugmentationParams` draws one set of parameters.
- `augment(image, params) -> np.ndarray`: horizontal flip (`np.fliplr`), then a single `cv2.warpAffine` with `cv2.getRotationMatrix2D(center, angle, scale)`, linear interpolation and `BORDER_CONSTANT` 0. Then contrast around the image mean and multiplicative brightness: `clip(brightness * (contrast * (x - mean) + mean), 0, 1)`, returned as `float32`. Pixels that are 0 after the warp stay 0, so the border and the B6 padding stay black whatever the contrast.

Neutral parameters return the input unchanged. A spec scenario requires it, and a test checks it.

*Why one warp:* rotation and scale in one matrix interpolate once, so they blur less than two warps.

### D3. Per-copy generator from seed, content and index

`np.random.default_rng([seed, int(sha256[:16], 16), index])`. `sha256` is the original file's hash, which the cache already computes. The draw does not depend on processing order or on which other images exist.

### D4. Copies cached apart from originals

Copy entries live under `<cache>/<fingerprint>/aug-<augmentation_fingerprint>/<sha256[:2]>/<sha256>.npz`, one file per original with a `(k, C)` `cnn` array whose row i is copy i. The augmentation fingerprint hashes `asdict(AugmentationConfig)` and the run seed. Since `copies` is part of it, changing k recomputes every copy, which is what the spec asks for, and a single file per original lets the copies share one `prepare_image` call. Original entries are not touched.

When the Training featurizer computes an original, it passes the prepared `cnn_image` to `CopyCache.add`, which makes and stores the copies, so a cold run prepares each image once. Only when the original came from the cache does `CopyCache.cnn` prepare the image again for missing copies. File hashes are memoized for the run and shared by the original and copy caches. This avoids storing `cnn_image` in the original's entry.

*Alternative:* keep `cnn_image` in each original entry. Rejected: it adds about 200 KB per image to the cache for a step that re-runs only when settings change.

### D5. Copies appended after the originals

`run` builds one matrix: the N original rows first, in the existing order, then the N*k copy rows, each with its original's radiomic row and its own CNN row. A per-row `fold_of_row` (inherited for copies) and an `is_copy` mask go with it.

`fold_splits(fold_of_row, is_copy, n_splits)`: validation rows of fold k are `(fold == k) & ~is_copy`, and its fit rows are `fold != k`. Since the originals come first, the validation index arrays are identical whether or not copies are present. The run without copies uses the first N rows and `fold_splits(fold_of_row[:N], is_copy[:N], n)`.

`compare_classifiers`, `cross_validate_scores`, `ablate_blocks` and `calibrate` keep their signatures: they take a matrix and splits.

### D6. Step order and the decision rule

1. Selection on the originals only (unchanged).
2. `cross_validate_scores` of the selected classifier on the matrix with copies and its splits.
3. `augmentation_on = mean_with >= mean_without + std_without`, where `std_without` is the standard deviation across folds of the selected classifier's CV macro-F1 without copies.
4. Block ablation and calibration on the chosen matrix and splits. The hybrid reuse still holds: it reuses the score from step 1 or step 2.
5. Testing scored once, unchanged.

*Why select without copies:* the classifier choice stays independent of augmentation, and the extra cost is one 5-fold CV instead of three.
*Why the margin uses the std of the run without copies:* agreed during exploration as a guard against noise. A paired test on the per-fold differences would be more sensitive. It is not used because, with 5 folds, its threshold is just as arbitrary and harder to explain in the report.

### D7. Report and manifest

`report.json` and the artifact manifest gain `augmentation`: `{on, copies, ranges: asdict(AugmentationConfig), cv_without: {mean, std}, cv_with: {mean, std}, margin}`. `report.md` gains a short section with these numbers. `folds.csv` and `feature_stats` keep using originals only: copies have no file, and their radiomics duplicate the originals'.

## Risks / Trade-offs

- [Duplicated radiomic rows give the radiomic scaler and PCA twice the weight of every original] → the weighting is uniform, so means, variances and PCA directions are those of the originals. Only the PCA's random solver path could differ, and its seed is fixed.
- [`class_weight="balanced"` with copies] → each class doubles, so the class ratios and weights are unchanged.
- [The margin is conservative with 5 noisy folds] → augmentation may stay off on a real gain. Accepted: "on" has to be justified. The two scores are in the report, so the team can see a near miss.
- [Copies are always extracted, even when the decision turns augmentation off] → about 1,830 extra CNN passes, cached once. That is the price of deciding inside the run.
- [sklearn behavior with rows that are fit rows but never validation rows] → `cross_validate` and `CalibratedClassifierCV(ensemble=True)` iterate the given pairs and need no partition. A run test with copies checks that both accept these splits.
- [Copies leak into serving] → augmentation lives in `training/`, and a test asserts that importing `src.pipelines` does not load `src.training.augmentation`.

## Migration Plan

No data migration. The original feature cache stays valid. Copy entries are new files under their own folder. Reports and manifests gain an additive section. Rollback is reverting the change.
