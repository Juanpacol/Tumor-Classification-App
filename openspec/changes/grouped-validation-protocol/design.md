# Design

## Context

See proposal.md (Why) for the motivation and specs/model-validation/spec.md for the required behavior.

Current state in `app/ml_engine/src/training/`:

- `run.train` does `split_samples` (B2), then builds the feature tables through `FeatureCache.table`. That step drops images the quality gate rejects and returns them as `skipped`. Only then does it build `x_train, y_train, groups` from the surviving rows.
- `model.group_folds(n_splits, seed)` returns a `StratifiedGroupKFold(shuffle=True, random_state=seed)`. It is called separately by `cross_validate_macro_f1` (through `cross_val_score(cv=...)`), by `ablate_blocks` (through `cross_validate_macro_f1`) and by `calibration.calibrate` (`list(group_folds().split(...))`).
- `evaluation.evaluate` computes accuracy, macro-F1, per-class metrics and the confusion matrix from `classification_report`.
- `CalibratedClassifierCV` already uses the default `ensemble=True`.

Constraints: scikit-learn 1.8, no new dependency, fixed seeds, and the project rule to remove parameters that stop being needed.

## Goals / Non-Goals

**Goals:**
- One fold assignment per run, keyed by image and independent of which images the gate keeps.
- One representation of "the folds" that selection, ablation and calibration all receive, so they cannot drift apart.
- A fold record that #167 (augmentation) can join on, to place each copy in its original's fold.

**Non-Goals:**
- Supporting augmented rows now. The design only keeps the door open: the splits are built from a per-row fold array, so extra rows with an inherited fold fit in without changing the consumers.
- Nested CV or hyperparameter search.
- Changing the serving artifact or its loader.

## Decisions

### D1. Assign folds on the sample list, before features

New module `training/folds.py`:

- `assign_folds(samples, n_splits, seed) -> dict[Path, int]` runs `StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(n), labels, groups)` on the purged Training `Sample`s and maps each sample path to its fold. It needs only labels and `group_id`, which come from the manifest.
- `fold_splits(fold_of_row: np.ndarray, n_splits) -> list[tuple[np.ndarray, np.ndarray]]` turns a per-row fold array into `(fit_idx, val_idx)` pairs and raises `ValueError` naming any fold with no rows.

`run.train` calls `assign_folds` right after `split_samples`. After the feature tables are built, it looks up each surviving row's fold to get `fold_of_row`, then calls `fold_splits` once.

*Why before features:* the assignment then depends only on the manifest and the seed (spec: "does not depend on the quality gate"), and a later augmentation step only needs the fold of the original.
*Alternative:* keep splitting the feature matrix but pass a frozen `groups` array. Rejected: rows still decide the stratification, so the gate and augmentation would still move images between folds.

### D2. Consumers take explicit splits

`cross_validate`, `compare_classifiers`, `ablate_blocks` and `calibrate` receive `splits: list[tuple[np.ndarray, np.ndarray]]` in place of `groups`, `n_splits` and the seed used for folding. The seed stays where it seeds models (PCA, classifiers). `group_folds` is deleted. Its only remaining role, building the `StratifiedGroupKFold`, moves into `assign_folds`.

*Why a list of index pairs:* scikit-learn's `cross_validate(cv=...)` and `CalibratedClassifierCV(cv=...)` both accept an iterable of `(train, test)` index pairs, which calibration already uses today. `PredefinedSplit` would also work, but it reserves `-1` for "never in validation", which is semantics we do not want yet. That decision belongs to #167.
*Alternative:* pass a fitted CV object around. Rejected: recomputing from data is exactly the drift being removed.

### D3. CV scores with two metrics, one decides

`cross_validate_macro_f1` becomes `cross_validate_scores` and uses `sklearn.model_selection.cross_validate(scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"})`. `CandidateScore` gains `mean_balanced_accuracy`, `std_balanced_accuracy` and `fold_balanced_accuracy`. The selection rule is unchanged (max mean macro-F1, ties by name). The ablation keeps reporting macro-F1 only, as section B9 of the pipeline doc specifies.

### D4. Balanced accuracy on Testing

`EvaluationMetrics` gains `balanced_accuracy`, computed by `balanced_accuracy_score(y, predictions)`. Its classes are the ones present in `y`, which matches the scikit-learn definition. It is computed from the same `predictions` already used for the report, so `predict` is still called once and the G7 test (`evaluate` called once) keeps holding.

### D5. Fold record and counts

`run` builds a `FoldRecord` (per Training sample: relative path, label, `group_id`, fold, used). `report.write_report` writes it as `folds.csv` in the report folder. `report.json` gains `validation`: `{method: "StratifiedGroupKFold", n_splits, seed, used_per_fold_and_class}`, and `report.md` gains a table of used images per fold and class. The CV table gains balanced-accuracy columns, and the headline line gains test balanced accuracy.

*Why CSV next to the report and not inside `report.json`:* about 1,830 rows are easier to diff and join in a CSV, and #167 and #168 consume it as data.

### D6. Keep `ensemble=True`, make it explicit

`calibrate` passes `ensemble=True` explicitly and its docstring states that the result averages one calibrated pipeline per fold. Decision D16 in `docs/ml-engine/decisions.md` records the reasons and the alternative:
- Each sigmoid is fitted on predictions from the model it calibrates, on data that model did not see.
- The per-image cost of 5 classifiers is negligible next to the CNN.
- With `ensemble=False`, one pipeline is refit on all of Training with a single sigmoid fitted on out-of-fold predictions. It is smaller to serve but calibrates a different model than the one used to fit the sigmoid.

## Risks / Trade-offs

- [The gate empties a fold, or a fold's fit rows lack a class] → `fold_splits` raises with the fold number before any fit. With 1,830 images and 1st-percentile thresholds this is not expected. `CalibratedClassifierCV` already raises if a class is missing from the fit rows.
- [Folds become slightly unbalanced after the gate, because stratification happens before it] → accepted: rejections are about 1% by construction of the thresholds. The per-fold, per-class table shows any imbalance.
- [Breaking signatures inside `training/`] → all callers are in the same package and its tests, which are updated in the same change. Nothing outside `training/` imports these functions (checked: `pipelines/` imports only `artifact`-related code).
- [Old reports lack the new fields] → reports are per run and never read back by code. Serving reads only the artifact manifest, whose metric additions are additive.
- [Shuffle semantics of `StratifiedGroupKFold`] → with `shuffle=True` only the order of groups with equal class distribution changes. Determinism under a fixed seed is covered by a test.

## Migration Plan

No data migration. Earlier reports and artifacts stay valid but are not comparable with new ones (they were already not comparable after B2). Rollback is reverting the change.
