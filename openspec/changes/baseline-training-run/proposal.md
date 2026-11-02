# Proposal

## Why

The training pipeline of #112, #167 and #168 has only run on synthetic data. The next step is to tune its parameters on the real dataset, one round after another, until the results are the best we can get: quality thresholds, crop, augmentation, PCA and classifier hyperparameters, all with an eye on accuracy and per-class recall. Three things get in the way today:

- **Per-class metrics exist only for Testing.** The report gives per-class precision, recall and the confusion matrix only for the official Testing split. Cross-validation gives only mean macro-F1 and balanced accuracy. Tuning would be guided by Testing, and the final Testing score would come out optimistic.
- **Parameters can only be changed in code.** Every parameter except the seed and the number of folds is a code default, so each round means editing code, and nothing records which values a round used.
- **No ready environment.** Nothing is set up to run on this machine, and installing the lock on every run would cost minutes per round.

This change fixes all three, then runs the baseline once (Azure #212).

## What Changes

- **Out-of-fold per-class metrics.** The training report gains an out-of-fold section for the selected classifier: per-class precision, recall, F1 and support, and the confusion matrix. They come from the validation rows of the shared grouped folds, never from augmented copies or Testing. The section is kept apart from Testing in `report.json` and `report.md`, and the artifact manifest's metrics carry it too.
- **Round configuration file.** `python -m src.training` accepts `--config <file>.toml`. The file overrides the defaults of the preprocessing settings (including the quality thresholds), radiomics, augmentation, PCA components and each candidate classifier's hyperparameters, plus the seed and the number of folds. Keys left out keep their defaults. Unknown keys, and values that the guarded rules forbid (balanced class weights, seed-driven randomness, the metrics version), stop the run before any work. The resolved configuration is written next to the report and recorded in the manifest. Round files live in `app/ml_engine/rounds/` and are versioned in git.
- **Training image and compose service.** A training image (`app/ml_engine/Dockerfile`, `python:3.11-slim` with `requirements.lock` installed) and a `trainer` service in `docker-compose.yml` under `profiles: ["train"]`. The service mounts `app/ml_engine`, so code, data, weights, artifacts and the feature cache stay on the host. One round is one command, `docker compose run --rm trainer src.training --config rounds/<v>.toml --version <v>`. The image is rebuilt only when the lock changes. `docker compose up` does not start the trainer.
- `app/ml_engine/weights/` is added to `.gitignore`, as the README already claims.
- **Baseline round `0.1.0`.** Versions follow SemVer, and each later round bumps the minor version. The round runs with `rounds/0.1.0.toml` (today's defaults), and its report, shortcut audit and wall time go into a new decision in `docs/ml-engine/decisions.md`.
- **Stale docs.** `docs/ml-engine/status.md` and the "src/data is not in the repo" check in `decisions.md` are brought up to date.
- If Testing macro-F1 is below 0.90 or the metadata-only baseline reaches 0.70 balanced accuracy, the results go to the team before any model is published (#112 criterion 9, #168).

## Capabilities

### New Capabilities

- `training-configuration`: how a training run takes its parameters from a round file, which parameters a round may change and which it may not, how invalid files are rejected, and how the resolved configuration is recorded.

### Modified Capabilities

- `model-validation`: adds out-of-fold per-class metrics and the confusion matrix on Training for the selected classifier, and defines which rows they may use.

## Impact

- **Code** (`app/ml_engine/src/training/`):
  - A per-fold fit-and-predict step for the selected pipeline. `run.py` calls it on the rows it kept, and `report.py` renders it.
  - A round-config loader based on `tomllib` (stdlib in 3.11, no new dependency).
  - `__main__.py` gets `--config`.
  - `model.py`: `candidate_classifiers` and `build_model` take the hyperparameter overrides and the PCA component count from the config.
- **Docker**: new `app/ml_engine/Dockerfile` and `app/ml_engine/.dockerignore`, and a new `trainer` service in `docker-compose.yml`. The existing services do not change.
- **Tests**: out-of-fold predictions, the config loader (defaults, overrides, rejected keys and guarded values), resolved config in the report and the manifest, and the report rendering.
- **Serving**: none. Serving already rebuilds preprocessing and thresholds strictly from the manifest, so tuned values reach inference unchanged.
- **Cost**: one extra 5-fold fit of the selected classifier per run. A round that changes only PCA or classifier settings reuses every cached feature.
- **Out of scope**: choosing the tuned values (the next story, which uses only cross-validated metrics), automated search such as grid or Optuna, GPU support, changing the calibration method or the candidate list, and publishing the artifact (#114).
