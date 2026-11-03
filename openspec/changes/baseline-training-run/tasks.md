# Tasks

Run tests from `app/ml_engine` with `pytest src/ -v`. Once group 3 lands, run them in the trainer with `docker compose run --rm --entrypoint pytest trainer src/ -v`. Work on branch `AB-212-baseline-training-run` from an up-to-date `develop`.

## 1. Repo setup

- [x] 1.1 Add `app/ml_engine/weights/` to `.gitignore`; verify `git check-ignore -v app/ml_engine/weights/backbone.pt` prints the new rule

## 2. Round configuration

- [x] 2.1 Add `pca_components` (default 32) and `classifier_params` (default empty) to `TrainingConfig`, and pass them through `candidate_classifiers(seed, params)` and `build_model(..., pca_components)`, with `class_weight` and `random_state` still set by the code (design decision 6); verify existing tests pass unchanged and a new test in `src/tests/test_training_model.py` shows an override reaches the estimator while `class_weight` stays `"balanced"`
- [x] 2.2 Add the TOML round loader (`tomllib`), which overlays each section on the defaults, converts lists to tuples, rebuilds the dataclasses, rejects unknown sections, keys and classifier names, rejects the guarded keys (`class_weight`, `random_state`, `metrics_version`), and validates classifier params with `set_params` and `_validate_params`; verify new `src/tests/test_training_config.py`: empty file equals defaults, partial override, nested thresholds, list to tuple, and each rejection names the dotted key and happens before any image is read
- [x] 2.3 Add `--config` to `src/training/__main__.py` with CLI `--seed`/`--n-splits` taking precedence when given (design decision 7); verify a CLI test that both the file value and the flag override resolve as specified
- [x] 2.4 Write `config.json` (resolved configuration) into the report folder, and add `config` and `config_source` (`path`, `sha256` or `null`) to `report.json` and the manifest (design decision 8); verify `src/tests/test_training_run.py` checks the three records and that two runs with equal resolved configs give equal metrics
- [x] 2.5 Add a feature-cache test that a round changing only `classifiers.*` or `model.pca_components` recomputes no image or copy features, while a `preprocessing.crop_margin` change does; verify in `src/tests/test_training_features.py`
- [x] 2.6 Add `app/ml_engine/rounds/0.1.0.toml` listing today's defaults explicitly, with a header comment explaining the baseline; verify loading it gives a config equal to the defaults (test)

## 3. Training image and compose service

- [x] 3.1 Add `app/ml_engine/Dockerfile` and `app/ml_engine/.dockerignore` (`*`, `!requirements.lock`) per design decision 9; verify `docker build app/ml_engine` succeeds and the context sent is only a few KB
- [x] 3.2 Add the `trainer` service to `docker-compose.yml` under `profiles: ["train"]` mounting `./app/ml_engine:/work` (design decision 10); verify that `docker compose config --services` without the profile omits `trainer`, and that `docker compose run --rm trainer src.training --help` prints the CLI help including `--config`
- [x] 3.3 Run `docker compose run --rm trainer src.deep_learning.export_backbone --out weights`; verify `app/ml_engine/weights/backbone.json` exists and `git status` does not list `weights/`
- [x] 3.4 Run the full test suite in the trainer; verify `pytest src/ -v` passes inside the container

## 4. Out-of-fold metrics

- [x] 4.1 Add the per-fold fit-and-predict step for the selected pipeline (design decision 1), returning labels and predictions of original rows only; verify new tests in `src/tests/test_training_model.py`: with and without copy rows, every original is predicted exactly once, no copy is predicted, and fold k's model never sees a copy of a fold-k image
- [x] 4.2 In `training/run.py`, call it on `rows` after the augmentation decision and before calibration and the Testing score, turn its output into metrics with `evaluate`, and store it with `augmented` under `cv_out_of_fold` in the report and the manifest `metrics`; verify `src/tests/test_training_run.py`: the matrix sums to the usable Training originals, supports match per class, and the G7 single-scoring test still passes
- [x] 4.3 In `training/report.py`, render "Per-class metrics (cross-validated on Train, out-of-fold, uncalibrated)" after selection and before Testing, plus the resolved-config summary and the line that rounds are compared on this section; verify `src/tests/test_training_report.py` checks the heading, the table rows and that the section is separate from Testing

## 5. Documentation of the round workflow

- [x] 5.1 Rewrite the training section of `app/ml_engine/README.md`, covering: build once, export once, one command per round, round files in `rounds/` with the SemVer rule, which changes reuse the feature cache, the PowerShell `Measure-Command` and POSIX `time` wrappers, the Linux `--user` note, promoting a round to `artifacts/current`, and "compare rounds on the out-of-fold section, not Testing"; verify the commands are copied from the ones run in group 3
- [x] 5.2 Add a decision in `docs/ml-engine/decisions.md` for design decisions 1-3 and 5-10 (out-of-fold metrics, round files, training image), with the CV-overfitting limitation of repeated rounds; verify the entry exists and names the open risks

## 6. Baseline round 0.1.0

- [x] 6.1 Run `docker compose run --rm trainer src.training --data-dir src/data --manifest src/data/processed/dedup_20261028T234450903626Z/manifest.csv --config rounds/0.1.0.toml --version 0.1.0`, timed from the shell; verify `artifacts/model-v0.1.0/manifest.json`, `report.json`, `report.md` and `config.json` exist, and the CLI printed the test macro-F1 and the metadata baseline
- [x] 6.2 Run the dataset limits check from the "Open verifications" list of `decisions.md` in the trainer, and close the item with the measured max bytes and pixels; verify the item is checked and states the values
- [x] 6.3 Record the baseline in a new decision in `docs/ml-engine/decisions.md`, covering: skipped images and reasons, the selected classifier and CV scores, augmentation decision, ablation, the out-of-fold per-class table, Testing metrics, the shortcut audit, guarantee status, and wall time (first run and cached rerun); verify every number matches `report.json`
- [x] 6.4 Update `docs/ml-engine/status.md` with these facts: the dataset is in the repo, skull stripping is gone (D18), #112 code is merged, rounds run through the trainer, and the baseline outcome; verify no line still says the dataset is not in the repo
- [x] 6.5 If Testing macro-F1 < 0.90 or the metadata baseline ≥ 0.70, write the summary for the team (links to the report and the decision) and state in the decision that no model is published until the team reviews it; verify the decision says which case applied

## 7. Integration check

- [x] 7.1 Make a throwaway round file that changes only `classifiers.svm_rbf.C` and run it with a throwaway version, then delete its outputs and file; verify the run extracts no features (cache hit, shown by run time and no new `.npz` files) and records the override in `config.json`
- [x] 7.2 Rerun `rounds/0.1.0.toml` with a throwaway version; verify it reproduces the CV and Testing metrics of 6.1 exactly, then delete its outputs

## Workflow follow-up

- Commit with `AB-212: ...` messages and open the PR against `develop` (`create-commit`, `create-pull-request`).
- After merge, move Azure #212 to Done by hand, update its acceptance criteria to mention round files and the trainer service, and archive this change (`/opsx:archive`).
- Next story: tuning rounds `0.2.0`+, compared only on the out-of-fold section.
