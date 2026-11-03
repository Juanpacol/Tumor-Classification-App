# Tasks

Run tests in the trainer from the repository root with `docker compose run --rm --entrypoint pytest trainer src/ -q` (Git Bash: prefix `MSYS_NO_PATHCONV=1`). Work on branch `AB-222-model-tuning-rounds` from an up-to-date `develop`. Commits use `AB-222: type(scope): description`.

## 1. Tuning and final modes

- [x] 1.1 Split `train()` in `src/training/run.py` into `tune` (Training features only, selection through shortcut audit and out-of-fold metrics) and `finalize` (Testing features, calibration, Testing score, artifact), with `mode` in the report and the Testing section only in final (design decision 1); verify new tests in `src/tests/test_training_run.py`: a tuning run writes no artifact, has no `test` key, and leaves the cache with no Testing entry; a final run of the same config gives the same CV and out-of-fold metrics and scores Testing once
- [x] 1.2 Add `--final` to `src/training/__main__.py`; tuning runs print CV macro-F1 and out-of-fold recall per class instead of Testing; verify a CLI test for both modes
- [x] 1.3 Render the mode in `report.md` (`report.py`) and update the README training section and `docs/ml-engine/status.md` "How to resume" so rounds run without `--final` and the kept round with it; verify `test_training_report.py` checks the mode line and the README commands match the CLI help

## 2. Backbone key and weights layout

- [x] 2.1 Add `backbone` (default `efficientnet_b0`, validated against `BACKBONES`) to `ModelConfig` and the round loader's `model` section; verify `test_training_config.py`: override resolves, `vgg16` is rejected naming `model.backbone`
- [x] 2.2 Replace `--backbone-bundle` with `--weights-dir` (default `weights`); resolve `<weights-dir>/<backbone>/` and stop before any image is read when the bundle is missing or its manifest declares another backbone (design decision 2); verify tests for missing and mismatched bundles that assert no featurizer call happened
- [x] 2.3 Add a cache test: a round on `resnet50` then a round back on `efficientnet_b0` with otherwise equal settings recomputes nothing on the second; verify in `test_training_features.py` with a fake extractor
- [x] 2.4 Move `weights/backbone.{json,pt}` to `weights/efficientnet_b0/` and export `resnet50` and `densenet121` with `docker compose run --rm trainer src.deep_learning.export_backbone --backbone <name> --out weights/<name>`; update the export commands in the README and `status.md`; verify each folder passes `read_bundle` and `git status` lists nothing under `weights/`

## 3. Blocks and calibration keys

- [ ] 3.1 Add `blocks` (`["radiomic","cnn"]` default or `["cnn"]`) to `ModelConfig` and thread it through `compare_classifiers`, `cross_validate_predictions`, the ablation reference score and the final `build_model` (design decision 3); verify tests that a CNN-only round fits every candidate on the CNN columns only and that `[]` and `["radiomic"]` are rejected naming `model.blocks`
- [ ] 3.2 Verify serving of a CNN-only artifact: build a synthetic final artifact with `blocks = ["cnn"]` and check `BundleInferencePipeline.from_bundle(...).predict(bytes)` returns top radiomic features; test in `src/tests/`
- [ ] 3.3 Add `calibration` (`sigmoid` default or `isotonic`) to `ModelConfig`, used by `calibrate()` instead of the constant, and record it in the report and the manifest; verify tests that the saved classifier's 5 calibrators use the configured method and `platt` is rejected
- [ ] 3.4 Add calibrated out-of-fold metrics (design decision 4): per outer fold, `CalibratedClassifierCV` with the configured method on the 4 leave-one-fold-out inner splits of the fit folds, probabilities for the fold's originals, pooled macro-F1, log loss and triage share/accuracy at `TRIAGE_CONFIDENCE_THRESHOLD`; report as `cv_calibrated`; verify tests that fold k's predicting model never saw a fold-k row in fit or calibration, and that no Testing feature is read
- [ ] 3.5 Record decisions D25 (modes, weights layout, model keys, calibrated out-of-fold metrics) in `docs/ml-engine/decisions.md`; verify the entry names the CLI and layout breaking changes

## 4. Training error diagnosis

- [ ] 4.1 Store `quality` (snr, entropy) in each cache entry and add `entry_schema: 2` to the fingerprint (design decision 5); verify `test_training_features.py`: entries hold `quality`, and the fingerprint differs from the schema-1 one
- [ ] 4.2 Add `src/training/diagnosis.py`: the out-of-fold predictions table and the breakdown by source size, group size and snr/entropy terciles (design decision 6); write `oof_predictions.csv` and `cv_error_breakdown` from `tune`; verify tests that the CSV's confusion matrix equals `cv_out_of_fold`, cells sum to the usable originals and to the tumor→no_tumor count, and no copy or Testing row appears
- [ ] 4.3 Render the breakdown in `report.md`; verify `test_training_report.py` checks the section and its placement before any Testing section
- [ ] 4.4 Run the full suite and black/ml-validation checks in the trainer; verify all pass

## 5. Reference and rounds

- [ ] 5.1 Write the kept-round rule (design decision 7) and the rounds results table header in `docs/ml-engine/decisions.md` (D26) before any round runs; verify the rule is committed before round 0.2.0's results
- [ ] 5.2 Rerun the 0.1.0 configuration in tuning mode as the reference (`--version 0.1.1`); read `oof_predictions.csv` and the breakdown, record what the tumor→no_tumor errors share and any reorder of the families in D26; verify CV numbers equal D24
- [ ] 5.3 Run backbone rounds (`rounds/0.2.0.toml` resnet50, `0.3.0` densenet121) in tuning mode; record each in the D26 table with wall time; verify each round file is committed and its `config.json` hash matches
- [ ] 5.4 Run the blocks/PCA round(s) from the best backbone; record in D26
- [ ] 5.5 Run the classifier hyperparameter round(s); record in D26
- [ ] 5.6 Run the quality thresholds and augmentation round(s); record in D26
- [ ] 5.7 Run the calibration round (isotonic vs the sigmoid reference, compared on calibrated out-of-fold log loss); record in D26
- [ ] 5.8 Apply the kept-round rule, rerun the kept round once with `--final`; record Testing metrics, shortcut audit and the outcome (ready for #114, or results to the team) in D26 and `status.md`; verify Testing was scored only in this run (all other reports have `mode: "tuning"`)

## Workflow follow-up

- Archive `baseline-training-run` before this change, since this change adds requirements to `training-configuration`, which that change introduces.
- If the target is met, start #114 with the kept artifact; otherwise present D26 to the team.
