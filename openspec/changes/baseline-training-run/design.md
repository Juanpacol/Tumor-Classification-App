# Design

## Context

- `run.py` assigns folds once, builds `plain` rows (originals) and `augmented_rows` (originals plus copies), and selects a classifier with `compare_classifiers`. It then decides on augmentation and keeps the chosen rows in `rows`. Ablation, calibration and the Testing score all use `rows`.
- `cross_validate_scores` uses sklearn's `cross_validate` and keeps only fold scores, never the predictions. `evaluation.evaluate` already turns labels and predictions into per-class metrics and a confusion matrix, and it is used for Testing.
- With copies, `fold_splits` gives splits whose validation rows are originals only. Copies are never validation rows, so the splits are not a partition of all rows.
- The calibrated model (`CalibratedClassifierCV`, sigmoid, `ensemble=True`) fits each fold's sigmoid on that fold's validation rows.
- `TrainingConfig` already nests frozen dataclasses: `PreprocessingConfig` (with `QualityThresholds`), `RadiomicsConfig` and `AugmentationConfig`. The CLI exposes only `--seed`, `--n-splits`, `--version` and `--cache-dir`. `PCA_COMPONENTS` is a module constant, and `candidate_classifiers` hard-codes each estimator's hyperparameters.
- Feature cache keys hash `asdict` of the preprocessing, radiomics and augmentation configs, plus the backbone (D21, AB-168 audit). PCA and classifier settings are not in them.
- Serving (`pipelines/artifact.py`) rebuilds `PreprocessingConfig` and `QualityThresholds` strictly from the manifest (`_strict`, `_tuples`) and rejects a different `metrics_version`. New manifest metric keys are ignored.
- The backend Dockerfile already installs `requirements.lock` in `python:3.11-slim`. The root `.dockerignore` keeps `src/data` and `artifacts` out of build contexts. The backend reads its model from `./app/ml_engine/artifacts/current`.
- On this machine the global Python is 3.14 and has no torch, `app/ml_engine/weights/` is empty and not ignored by git, and Docker 29.9 is installed.

## Goals / Non-Goals

**Goals:**
- Per-class out-of-fold metrics on Training, from the same pipeline, folds and rows that the run calibrates. See `specs/model-validation/spec.md`.
- A tuning round is: edit one TOML file, run one command, and pay only for the features the change invalidates. See `specs/training-configuration/spec.md`.

**Non-Goals:**
- Out-of-fold metrics for every candidate or ablation variant, per-fold confusion matrices, or out-of-fold probabilities.
- Automated search (grid, Optuna), parallel rounds, or comparing rounds side by side in one report.
- Making the candidate list, the calibration method, the CNN backbone choice or the fold method configurable.
- GPU support, or recording run time inside the report.

## Decisions

**1. Per-fold fit-and-predict loop instead of `cross_val_predict`.** For each `(fit, validation)` split, clone the selected pipeline (`build_model`), fit it on the fit rows and predict the validation rows. Write the predictions at those row positions, then keep only the original rows. `cross_val_predict` raises an error unless the splits partition every row, and with copies they do not. Getting predictions back from `cross_validate` (`return_estimator`) would mean holding five fitted pipelines for no gain.

**2. Uncalibrated selected pipeline, not the calibrated model.** The calibrated ensemble fits each fold's sigmoid on that fold's validation rows. Its predictions on those rows are therefore not out-of-fold. The uncalibrated pipeline is exactly what selection and the augmentation decision scored, so the new per-class view matches the reported CV macro-F1. Trade-off: per-class sigmoid scaling can move a few argmax decisions. The report labels the section as uncalibrated.

**3. Pooled over folds.** `evaluate` runs once on all out-of-fold predictions. Pooled counts show which classes are confused, and the spread across folds is already in the per-fold macro-F1 std. The report keeps the fold-mean macro-F1 as the headline and says the pooled numbers are for per-class analysis.

**4. Placement and shape.** The step runs right after the augmentation decision, on `rows`, and before calibration and the Testing score, so it reads no Testing features (G7). `report.json` and the manifest's `metrics` get `cv_out_of_fold`, holding the `evaluate` output plus `augmented: bool`. `report.md` shows "Per-class metrics (cross-validated on Train, out-of-fold, uncalibrated)" after the selection table and before Testing.

**5. Round files in TOML, loaded with `tomllib`.** TOML is in the stdlib from 3.11, which the lock targets, so it adds no dependency. Its typed values (floats, ints, arrays, nested tables) map directly onto the dataclasses. YAML would need PyYAML. JSON has no comments, and a round file should say why a value changed. Layout:

```toml
# rounds/0.2.0.toml - why this round exists
[run]
seed = 42
n_splits = 5

[preprocessing]
crop_margin = 0.08
[preprocessing.thresholds]
min_snr = 2.74

[radiomics]
lbp_radius = 2

[augmentation]
copies = 2

[model]
pca_components = 32

[classifiers.svm_rbf]
C = 3.0
gamma = "scale"
```

**6. Loader: overlay on the defaults, then rebuild the dataclasses.** The loader starts from `asdict` of each default config, overlays the file's table, converts lists to tuples (the same rule as serving's `_tuples`), and builds the dataclass. `__post_init__` therefore keeps validating values. Unknown keys are found by comparing against `fields()` before construction, and the error names the dotted key. `metrics_version` is rejected explicitly.

Classifier tables are checked against `candidate_classifiers()`: the classifier name must exist, and each key must be in that estimator's `get_params()` and not in the guarded set `{class_weight, random_state}`. Overrides are applied with `set_params` before any fit, so type errors from sklearn's parameter validation surface at load time through an explicit `_validate_params()` call.

The loader returns the existing `TrainingConfig`, which gains `pca_components: int` and `classifier_params: Mapping[str, Mapping[str, Any]]` (frozen, defaults 32 and empty). `candidate_classifiers(seed, params)` and `build_model(..., pca_components)` read them, and nothing else in the pipeline changes.

**7. CLI precedence.** `--config` is optional. `--seed` and `--n-splits` stay for quick runs and win over the file when given. The resolved values are what gets recorded, so a run is never ambiguous. `--version`, `--out` and `--cache-dir` stay CLI-only because they name a round's outputs, not its parameters.

**8. Recording.** The run writes `config.json` (the full resolved configuration) into the report folder and adds `config` and `config_source` (`{path, sha256}`, or `null`) to the manifest and `report.json`. The resolved configuration, not the file, is the reproducibility record. A file can leave keys out, and their defaults can change in code later.

**9. Training image: own Dockerfile, built from a minimal context.** `app/ml_engine/Dockerfile`:

```dockerfile
FROM python:3.11-slim
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock
WORKDIR /work
ENTRYPOINT ["python", "-m"]
CMD ["src.training", "--help"]
```

The build context is `./app/ml_engine`, with its own `.dockerignore` (`*` then `!requirements.lock`), so the dataset, weights and artifacts are never sent to the daemon. Only the lock goes into the image, and the code is mounted. Editing code or round files needs no rebuild, and a lock change does.

Why not reuse the backend image: it carries backend dependencies and build tools, and it is the Cloud Run image, so a backend change would silently change the training environment. Why not a long-lived container or devcontainer: its state drifts from the lock.

**10. `trainer` compose service under a profile.**

```yaml
trainer:
  build: { context: ./app/ml_engine }
  profiles: ["train"]
  working_dir: /work
  volumes:
    - ./app/ml_engine:/work
```

`docker compose run --rm trainer src.training --data-dir src/data --manifest <m> --config rounds/0.2.0.toml --version 0.2.0` runs one round. The README gives the full command with the data and manifest paths, so the CLI does not change for that. `docker compose run --rm trainer src.deep_learning.export_backbone --out weights` runs once. `profiles` keeps `docker compose up` from building or starting the trainer, while `compose run trainer` still works because naming a service activates its profile. Everything written (weights, artifacts, feature cache) lands in the mounted folder on the host and survives the container.

**11. Versions and promotion.** SemVer: the baseline is `0.1.0` with `rounds/0.1.0.toml`, which lists today's defaults explicitly for reference. Each round bumps the minor version and adds its own file. `write_artifact` refuses to overwrite, so each round keeps `model-v<version>/` and a timestamped report. To try a round in the app, copy or link `artifacts/model-v<version>` to `artifacts/current`, which the backend mounts. This is documented, not automated.

**12. Run time measured from the shell.** Wall time is measured around `docker compose run` (`time` or `Measure-Command`), for the first run and for a cached rerun, and recorded in the decision entry. It stays out of the report, which should be identical across reruns.

## Risks / Trade-offs

- [Rounds drift toward Testing because the report still prints Testing per-class metrics] → The README and the decision entry say that rounds are compared only on the out-of-fold section and the CV scores. Testing is read for the final chosen version.
- [Many rounds tuned on the same 5 folds overfit the CV estimate] → Recorded as a known limitation in the decision. Keep the number of rounds small and change one family of parameters per round. A final confirmation on Testing, done once, is the check.
- [Out-of-fold scores are uncalibrated] → Labeled in the report. They can differ from the calibrated model only where sigmoid scaling moves argmax ties.
- [A round file is edited after its run] → `config_source.sha256` in the manifest detects it. `config.json` in the report folder is the record that counts.
- [Container writes as root on Linux hosts] → Not an issue on Docker Desktop for Windows. The README notes `--user "$(id -u):$(id -g)"` with `HOME=/tmp` for Linux.
- [First build needs about 200 MB of downloads, and the export needs network] → Done once. Later rounds need no network.
- [Real images exceed `MAX_FILE_BYTES` or `MAX_PIXELS`] → They are skipped and reported. The open "Limits vs the real dataset" check in `decisions.md` is run in the container and closed.
- [Testing macro-F1 < 0.90 or metadata baseline ≥ 0.70] → No model is published, and the report goes to the team. This is an expected outcome, not a failure of the change.

## Migration Plan

None for existing behavior. Without `--config`, a run resolves to today's defaults. The new manifest and report keys are additive, and serving ignores them. The new compose service is behind a profile. Rollback is reverting the commit.
