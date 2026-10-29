# ML Engine status

Last updated 2026-10-29. Azure Boards is the source of truth for the state of each story. This file records what Boards does not: how the pieces fit, what is blocked and how to resume. The reasons behind each choice are in `decisions.md`.

## Data flow

```
load_mri_image -> prepare_image -> extract_radiomics (45) ----+
 (AB-108)          (AB-109)        DeepFeatureExtractor (1280) +-> sklearn pipeline -> calibrated
 JPG/PNG -> 2D     quality gate,   (AB-110, AB-111)               scaler + PCA(32) / scaler   classifier
 float [0, 1]      optional skull                                 -> classifier (AB-112)      -> PredictionResult
                   stripping,                                                                  (contract, AB-107)
                   mask, CNN input
```

## State

| Area | Stories | State |
|---|---|---|
| Contract, fake pipeline, seeds | AB-107 | Merged |
| Loader, quality gate, optional skull stripping | AB-108, AB-109 | Merged. Quality thresholds are provisional (D7) |
| Radiomic and CNN features | AB-110, AB-111 | Merged. Verified with real ImageNet weights on CPU (D9) |
| Dataset cleaning | AB-150 | Merged. 3,434 images kept (`DATASET_CLEANING_REPORT.md`) |
| Training, selection, calibration, artifact, report, CLI | AB-112 | Code done (part 1 merged, part 2 in review). **Needs a run on the real dataset** |
| End-to-end `InferencePipeline.from_bundle` | AB-113 | Not started. Needs a trained bundle |
| Publish artifacts as a GitHub Release | AB-114 | Not started |
| CI, lock, branching | AB-151 to AB-154 | Merged. Lock is universal with CPU-only PyTorch (D10) |

## Blocked on the real dataset

Everything below needs the Kaggle Brain Tumor MRI Dataset (Masoud Nickparvar) in `app/ml_engine/src/data`, which is not in the repo.

1. Run the training once and read its report: the 0.90 macro-F1 target, the per-class scores and the number of skipped images.
2. Calibrate the quality thresholds (D7). They are guesses until then.
3. Compare the three backbones and the sigmoid against isotonic calibration (D9, D12).
4. Look at real skull stripping masks before any ablation (D7).
5. Measure the extraction time on the real set (estimated at about 5 minutes, not measured).

AB-112 stays open until step 1 produces a report.

## Not verified yet

- Timing and memory on Cloud Run (AB-120).
- Terms of use of the ImageNet weights for commercial use. Check before AB-114.
- EXIF orientation on real uploads, and the upload size cap in the backend (D5).
- Nothing in the pipeline has been run on real MRI images.

## How to resume

```bash
cd app/ml_engine
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.lock
.venv/bin/python -m pytest src/tests/ --cov=src --cov-fail-under=80
.venv/bin/python -m src.deep_learning.export_backbone --out weights
python -m src.preprocessing.deduplication --data-dir src/data      # manifest.csv goes to src/data/processed/dedup_<timestamp>/
python -m src.training --data-dir src/data --manifest src/data/processed/dedup_<timestamp>/manifest.csv --version 0.1.0
```

Run the CI checks from the repository root, as CI does:

```bash
black --check app/backend app/ml_engine    # two paths: from the root it uses 88 columns
ruff check app/backend app/ml_engine
```

Pitfalls found along the way:

- `black app/ml_engine` with one path picks up the 100 columns of the package `pyproject.toml`. Always pass both paths from the root.
- CI installs the latest `ruff`, which adds rules over time. Run the same version before pushing.
- The GitHub remote is named `github`. A branch created from `github/develop` tracks `develop`, so push with `git push -u github <branch>`.
- `app/ml_engine/weights/` and `app/ml_engine/artifacts/` are ignored by git on purpose.
- Open every PR against `develop`. Only hotfixes target `main`.
