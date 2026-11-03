# ML Engine status

Last updated 2026-11-02 (AB-212). Azure Boards is the source of truth for the state of each story. This file records what Boards does not: how the pieces fit, what is blocked and how to resume. The reasons behind each choice are in `decisions.md`.

## Data flow

```
load_mri_image -> prepare_image -> extract_radiomics (45) ----+
 (AB-108)          (AB-109)        DeepFeatureExtractor (1280) +-> sklearn pipeline -> calibrated
 JPG/PNG -> 2D     quality gate,   (AB-110, AB-111)               scaler + PCA(32) / scaler   classifier
 float [0, 1]      brain mask,                                    -> classifier (AB-112)      -> PredictionResult
                   square crop,                                                                (contract, AB-107)
                   CNN input (AB-166)
```

## State

| Area | Stories | State |
|---|---|---|
| Contract, fake pipeline, seeds | AB-107 | Merged |
| Loader, quality gate, brain mask, geometry | AB-108, AB-109, AB-165, AB-166, AB-175 | Merged. Thresholds calibrated on Training (D17); skull stripping removed (D18) |
| Radiomic and CNN features | AB-110, AB-111 | Merged. Verified with real ImageNet weights on CPU (D9) |
| Dataset cleaning | AB-150 | Merged. 3,434 images kept (`DATASET_CLEANING_REPORT.md`) |
| Training, selection, calibration, artifact, report, CLI | AB-112, AB-167, AB-168 | Merged |
| Out-of-fold metrics, round files, trainer container, baseline run | AB-212 | Baseline 0.1.0 run (D24): Test macro-F1 0.807, target 0.90 not met; results go to the team before any model is published |
| Serving pipeline (`BundleInferencePipeline.from_bundle`, `predict(bytes)`) | AB-113 | Merged. Tested on a synthetic artifact (D13) |
| Publish artifacts as a GitHub Release | AB-114 | Not started |
| CI, lock, branching | AB-151 to AB-154 | Merged. Lock is universal with CPU-only PyTorch (D10) |

## Next: tuning rounds

The dataset is versioned in `app/ml_engine/src/data` and every round runs in the `trainer` container (`app/ml_engine/README.md`, *Tuning rounds* and *Training in Docker*). The baseline is round `0.1.0` (D24).

1. Tune in rounds `0.2.0`, `0.3.0`, ... (one family of parameters per round), comparing only the out-of-fold section and the CV scores (D23).
2. Compare the three backbones and the sigmoid against isotonic calibration (D9, D12). Both need code changes; they are not round-file keys.
3. Read Test only for the version that is kept, then publish it (AB-114).

## Not verified yet

- Timing and memory on Cloud Run (AB-120).
- Terms of use of the ImageNet weights for commercial use. Check before AB-114.
- EXIF orientation on real uploads, and the upload size cap in the backend (D5).

## How to resume

From the repository root (Git Bash: prefix with `MSYS_NO_PATHCONV=1`):

```bash
docker compose build trainer
docker compose run --rm --entrypoint pytest trainer src/ -q
docker compose run --rm trainer src.deep_learning.export_backbone --backbone efficientnet_b0 --out weights/efficientnet_b0   # once per backbone
docker compose run --rm trainer src.training --data-dir src/data \
  --manifest src/data/processed/dedup_20261028T234450903626Z/manifest.csv \
  --config rounds/<version>.toml --version <version>      # tuning; add --final once for the kept round
```

A local `python3.11` venv with `pip install -r requirements.lock` still works for the same commands from `app/ml_engine`.

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
