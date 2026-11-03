# Design

## Context

See proposal.md (Why) for the motivation. Current state that shapes the approach:

- `train()` in `run.py` is one function: features for both splits (`_features` extracts Train and Test in one call), selection, augmentation decision, out-of-fold metrics, ablation, metadata baseline, calibration, the Testing score and the artifact. There is no way to stop before Testing.
- `DeepFeatureExtractor.from_bundle(config.backbone_bundle, ...)` reads one bundle folder given by `--backbone-bundle` (default `weights`). `export_backbone --backbone <name> --out <dir>` already exports any of the three backbones. Today `weights/` holds the `efficientnet_b0` bundle at its root.
- The feature-cache fingerprint already hashes the backbone's weight SHA-256, input size, mean and std (`config_fingerprint`), so different backbones never share entries and returning to a backbone reuses its entries.
- `calibrate()` uses the module constants `CALIBRATION_METHOD = "sigmoid"` and `ensemble=True` on the shared folds.
- `build_model(..., blocks=BLOCKS)` already supports dropping a block (the ablation uses it); `compare_classifiers`, `cross_validate_predictions` and the final model do not pass `blocks`.
- Cache entries hold only `radiomics` and `cnn`. Quality metrics (snr, entropy) are computed by `prepare_image` but not stored. Width and height are read from file headers by `shortcut.metadata_matrix`.
- Serving (`pipelines/trained.py`) builds the extractor from the artifact's own copy of the bundle and unpickles the calibrated classifier, so it does not care which backbone, blocks or method a run chose. Top radiomic features at serving time need the radiomic block's z-score statistics, which the artifact always stores.
- Baseline per fold: about 366 validation rows; the smallest class (no_tumor) has about 52 per fold.

## Goals / Non-Goals

**Goals:**
- A round is one command that never touches Testing; the kept round becomes a model with one more command (`--final`).
- Every lever named in #222 is a round-file key, and every round is comparable on the same Training-only numbers, calibration included.
- Enough per-image Training evidence to look for the cause of the tumor→no_tumor errors.

**Non-Goals:**
- Choosing the round values here: tasks run the rounds, but which values win is decided by the numbers.
- Model selection across backbones inside one run, or any automated search.
- Comparing Training and Testing distributions (needs Testing labels; left to a team decision).

## Decisions

**1. Tuning by default, `--final` to finish.** `train()` is split into `tune(config, features) -> TuningResult` (everything up to and including the shortcut audit, out-of-fold metrics, calibrated out-of-fold metrics and the breakdown) and `finalize(config, tuning, test_table)` (calibrate, score Testing, write artifact). `__main__` runs `tune` and, only with `--final`, extracts Testing features and runs `finalize`. Both modes write the same report folder, with `mode: "tuning" | "final"` and the Testing section only in final. *Alternative:* an opt-out flag `--skip-test`. Rejected: forgetting a flag would leak Testing into a round, and the default should be the safe one. The baseline README commands gain `--final` where they meant to produce an artifact.

**2. Weights directory with one folder per backbone.** `--weights-dir` (default `weights`) replaces `--backbone-bundle`; the bundle is `<weights-dir>/<model.backbone>/`. Before anything else the run reads the bundle manifest and checks its declared backbone equals `model.backbone` (integrity is already checked by `read_bundle`). Export becomes `export_backbone --backbone resnet50 --out weights/resnet50`. *Alternative:* keep `--backbone-bundle` and add the backbone name only for checking. Rejected: the round file would not be self-contained, and the CLI flag would decide the round's most important parameter outside the recorded config. The cache needs no change (see Context).

**3. `ModelConfig` gains `backbone`, `calibration` and `blocks`.** The round loader's `_model` accepts the three keys and validates them in `ModelConfig.__post_init__` (backbone in `BACKBONES`; calibration in `{"sigmoid", "isotonic"}`; blocks exactly `["radiomic", "cnn"]` or `["cnn"]`; a radiomics-only model scored 0.754 in the ablation and is not offered). `blocks` is threaded through `compare_classifiers`, `cross_validate_predictions`, the ablation baseline score and the final `build_model`. The ablation keeps scoring all three variants regardless of `blocks`. `resolved_config` and the manifest's `config.model` record all three.

**4. Calibrated out-of-fold metrics by leave-one-fold-out inside the fit folds.** For outer fold k: take the fit rows (folds ≠ k, with copies if the run kept them), build a `CalibratedClassifierCV(method=<configured>, ensemble=True)` whose `cv` is the 4 inner splits "one of the other folds as calibration rows, the remaining 3 as fit rows", fit it, and predict probabilities for fold k's originals. Pool over k for macro-F1, log loss and the triage view (share of images with top probability ≥ `TRIAGE_CONFIDENCE_THRESHOLD`, and accuracy among them). This mirrors exactly how the final model is calibrated, minus one fold, and keeps groups intact because inner splits reuse the fold assignment. *Alternatives:* compare methods on Testing (forbidden), or calibrate on the outer validation fold and score it (not out-of-fold). Cost: 5 × 4 extra fits of the selected classifier per run; for logistic regression that is minutes, acceptable for a round.

**5. Quality metrics go into the cache entry.** Cache entries gain a `quality` array (snr, entropy). The fingerprint payload gains `"entry_schema": 2`, so all old entries are recomputed once (about 8 min per backbone, the same as a backbone change). Width and height come from file headers as the shortcut audit already does. *Alternative:* recompute `prepare_image` for the diagnosis. Rejected: a second pass over every image on each round, for data the featurizer already computes.

**6. Diagnosis module.** `training/diagnosis.py` builds the predictions table (one row per usable original, from the same predictions as `cv_out_of_fold`) and the breakdown: per class recall and tumor→no_tumor counts for each cell of three splits — source size (512x512 vs other), group size (1 vs >1) and per-metric terciles of snr and entropy (tercile edges from Training rows, recorded). Written as `oof_predictions.csv` and `cv_error_breakdown` in `report.json` and `report.md`. It runs in `tune`, so it never sees Testing.

**7. How the kept round is chosen.** Highest mean CV macro-F1 of the selected classifier. When two rounds are within one fold std of each other, the one with fewer out-of-fold tumor→no_tumor errors wins, then the simpler one (CNN-only over hybrid, fewer PCA components). Calibration method is chosen on calibrated out-of-fold log loss, with macro-F1 as a tie check. This rule is written in `decisions.md` before round 0.2.0 runs, so it cannot be bent to fit results.

**8. Round order.** One family per round, each starting from the best previous round: 0.2.0 backbone (`resnet50`, `densenet121` as 0.2.0 and 0.3.0), then blocks/PCA, classifier hyperparameters, quality thresholds and augmentation, calibration. The diagnosis of round 0.1.0's configuration (rerun in tuning mode as the reference) comes first and may reorder the families; any reorder is recorded with its reason.

## Risks / Trade-offs

- [Many rounds on the same 5 folds overfit the CV estimate] → one family per round, a fixed kept-round rule written before the rounds, and Testing read once.
- [CV on Training does not show the Testing error mode (19 vs 85 tumor→no_tumor), so rounds may improve CV without closing the gap] → the breakdown looks for a Training-side cause; if none is found, the team decides whether to study the Train/Test shift with Testing data, at the cost of using Testing as a development set.
- [Isotonic calibration overfits with about 52 no_tumor rows per calibration fold] → it is compared on calibrated out-of-fold log loss, so overfitting shows up as a worse score and sigmoid stays.
- [Cache schema bump recomputes every entry once] → accepted, one-off; noted in the README.
- [CLI and weights layout change break existing commands] → README, `status.md` and the D23 commands are updated in the same change; the old `weights/backbone.*` files are moved, not re-downloaded.
- [Target 0.90 may still not be met] → per #112 criterion 9 the results go to the team; nothing is published.

## Migration Plan

1. Move `weights/backbone.{json,pt}` to `weights/efficientnet_b0/`; export `resnet50` and `densenet121` next to it.
2. The first run after the change recomputes the cache once (schema 2).
3. `artifacts/model-v0.1.0/` stays valid: serving reads only the artifact folder, which is unchanged.
Rollback: revert the branch; the old cache folders are still on disk under their old fingerprints.
