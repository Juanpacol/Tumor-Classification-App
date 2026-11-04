# ml_engine

Standalone hybrid (radiomics + CNN) brain MRI tumor classifier. Current state and how to resume are in `docs/ml-engine/status.md`; decisions and their evidence are in `docs/ml-engine/decisions.md`.

## Setup

```bash
cd app/ml_engine
pip install -r requirements.lock   # exact versions, CPU-only PyTorch on Linux
pytest src/tests -v
```

`requirements.lock` is universal (macOS and Linux) and carries its package indexes, so the command above needs no extra flags. On Linux it installs the CPU-only PyTorch wheels (`+cpu`), which avoids the CUDA libraries that PyPI's `torch` pulls in. To regenerate it after changing dependencies:

```bash
uv pip compile pyproject.toml --extra dev --python-version 3.11 --universal \
  --index https://download.pytorch.org/whl/cpu --index-strategy first-index \
  --emit-index-url -o requirements.lock
```

## Preprocessing

`load_mri_image` (JPG/PNG to a 2D float32 array in [0, 1]) followed by `prepare_image`, which runs the quality gate and builds both model inputs (the grayscale image with its brain mask for radiomics, and the square `cnn_image` in [0, 1] for the CNN).

### Quality thresholds

Both metrics are measured on the original image, inside the brain region (the foreground mask: pixels above half the Otsu threshold, largest component, holes filled).

| Metric | Minimum | Method |
|---|---|---|
| SNR | 2.74 | median intensity inside the brain region divided by a MAD-based noise sigma of the dark background (pixels away from every bright region), floored at 1/255; whole-image wavelet sigma when the background is under 1% of the image |
| Shannon entropy | 3.20 bits | 256-bin intensity histogram of the brain region |

`QualityThresholds.metrics_version` (currently 2) names these definitions. It is part of the feature cache key and of the artifact manifest, and serving refuses an artifact calibrated for another version.

These values were calibrated on 2026-10-30 and reviewed; they reject no Training or Testing image of the versioned dataset (`docs/ml-engine/decisions.md`, D17). The calibration command reads only the leakage-free Training split and never opens a Testing image:

```bash
python -m src.training.thresholds --data-dir src/data --manifest src/data/processed/dedup_20261028T234450903626Z/manifest.csv
```

Each minimum sits in the gap between synthetic unusable references (SNR: blank, uniform noise, heavy Gaussian noise; entropy: blank, a 4-level posterized image) and the lowest value over the Training images: the geometric midpoint for SNR, the arithmetic midpoint for entropy. Every Training image passes by construction, and the command fails when a metric has no gap. It prints `min_snr`, `min_entropy`, `images_used` and `gap` (`low` and `high` per metric). Before changing `QualityThresholds` (`src/preprocessing/quality.py`), review the Testing images the values reject and record the result in `docs/ml-engine/decisions.md`.

### Brain mask

The radiomics region of interest is the brain mask above (`src/preprocessing/brain_mask.py`, pipeline block B5). When it covers less than `PreprocessingConfig.min_mask_fraction` (default 0.05) of the image, the full image is used instead; the quality gate still measures on the mask. There is no skull stripping, so `skull_stripped` is always `false` (`docs/ml-engine/decisions.md`, D18).

## Radiomic features

`extract_radiomics(radiomics_image, mask)` (from `PreprocessedImage`) returns a fixed-length vector of 45 finite features with stable names: GLCM (5), LBP (10), wavelet (21) and first-order intensity (9). All parameters are in `RadiomicsConfig`. Details and deviations from PyRadiomics and IBSI are in `docs/ml-engine/decisions.md` (D8).

## Deep features

`DeepFeatureExtractor.from_bundle(folder)` loads a frozen ImageNet backbone (EfficientNet-B0 by default) and `extract` / `extract_batch` turn the `cnn_image` of `prepare_image` into 1280-d embeddings, normalizing it with the mean and std stored in the bundle (D20). Weights are never downloaded at runtime. Create one bundle per backbone once on a development machine, each in a folder named after it:

```bash
python -m src.deep_learning.export_backbone --backbone efficientnet_b0 --out weights/efficientnet_b0
python -m src.deep_learning.export_backbone --backbone resnet50 --out weights/resnet50
python -m src.deep_learning.export_backbone --backbone densenet121 --out weights/densenet121
```

A training run loads `<weights-dir>/<model.backbone>/` (`--weights-dir` defaults to `weights`) and stops before reading any image if that folder is missing or holds another backbone. The `weights/` folder is ignored by git. The export writes the input size, mean and std that torchvision publishes for the weights. Details are in `docs/ml-engine/decisions.md` (D9, D20).

## Training data and model selection

The cleaned dataset is described by the `manifest.csv` that the deduplication step writes (`python -m src.preprocessing.deduplication --data-dir src/data`). `load_samples` reads the kept images from it, `FeatureCache` computes radiomic and CNN features once per image and `compare_classifiers` cross-validates logistic regression, an RBF SVM and a random forest on grouped folds that `assign_folds` computes once per run. Decisions are in `docs/ml-engine/decisions.md` (D11, D12, D16, D21).

### Train

With the backbone bundle exported (see *Deep features*) and the cleaning `manifest.csv`:

```bash
python -m src.training --data-dir src/data --manifest <path>/manifest.csv --config rounds/0.2.0.toml --version 0.2.0          # tuning round
python -m src.training --data-dir src/data --manifest <path>/manifest.csv --config rounds/0.4.0.toml --version 0.4.0 --final  # kept round
```

A run without `--final` is a **tuning run**: it never reads a Testing image and writes only the report. `--final` is for the round you keep, once: it adds calibration, the single Testing score and the artifact (D25).

It first removes from Train every image whose `group_id` also appears in Test, and prints the removed count per class (also stored as `split_removed_per_class` in the report and the artifact manifest). It then assigns every Train image to one of 5 grouped folds (`StratifiedGroupKFold` on `group_id`), selects a classifier by cross-validated macro-F1 on those folds (balanced accuracy is reported too), cross-validates it again with one augmented copy of each usable Train image (made from its CNN input, used only as fit rows) and keeps augmentation only if macro-F1 rises by at least the std across folds, runs an ablation (CNN-only, radiomics-only, hybrid) and the shortcut audit. A `--final` run then calibrates the model, scores the official Test split once and writes the artifact. Under `artifacts/` (override with `--out`):

- `model-v<version>/` (final runs only): `classifier.joblib`, the backbone files, `feature_stats.json` and `manifest.json` with the configs, versions, metrics and sha256 of every file.
- `reports/<timestamp>/`: `report.json` and `report.md` (the run's mode; for final runs the test metrics including balanced accuracy and confusion matrix; images per fold and class, selection, the out-of-fold per-class metrics on Train, the augmentation decision, ablation, the shortcut audit, the status of each leakage guarantee, and, for final runs, whether the 0.90 macro-F1 target was met), `config.json` with every resolved parameter, `folds.csv` with the fold of every Train image left after the leak removal and whether the quality gate let it through, and `oof_predictions.csv` with the out-of-fold prediction of every usable Train image (fold, group size, width, height, snr, entropy). The report breaks those errors down by source size, group size and snr/entropy tercile, with the tumor images predicted no_tumor in every cell, to look for where the errors cluster without reading Test.

The out-of-fold section gives per-class precision, recall, F1 and the confusion matrix of the selected classifier (uncalibrated), predicting every usable Train image once with the model of the fold that validates it. It never reads Test, so it is the section to tune on (D23). The calibrated section next to it gives macro-F1, log loss and the share of confident predictions of the same classifier calibrated as the final model is, with fold k calibrated on the other folds only; compare calibration methods on its log loss (D25).

The CLI prints the balanced accuracy of a classifier that sees only width, height and file size; at 0.70 or more it warns that the class can be told from the file alone, and that result goes to the team before a model is published.

The same resolved configuration reproduces the same metrics. Features, including those of the augmented copies, are cached under `artifacts/feature-cache`, so a second run skips the extraction. An existing artifact or report is never overwritten.

### Tuning rounds

Each round is a TOML file in `rounds/`, versioned in git, named after the round's SemVer version: `0.1.0` is the baseline, and every round bumps the minor version (`0.2.0`, `0.3.0`, ...). A file overrides only the keys it sets; start from a copy of the previous round and say in a comment why the round exists. `rounds/0.1.0.toml` lists every tunable key:

| Section | What it sets |
|---|---|
| `[run]` | `seed`, `n_splits` |
| `[preprocessing]`, `[preprocessing.thresholds]` | `PreprocessingConfig` and `QualityThresholds` (not `metrics_version`) |
| `[radiomics]` | `RadiomicsConfig` |
| `[augmentation]` | `AugmentationConfig` |
| `[model]` | `backbone` (`efficientnet_b0`, `resnet50` or `densenet121`), `pca_components`, `blocks` (`["radiomic", "cnn"]` or `["cnn"]`), `calibration` (`sigmoid` or `isotonic`) |
| `[classifiers.<name>]` | hyperparameters of `logistic_regression`, `svm_rbf` or `random_forest` (any `get_params()` key except `class_weight` and `random_state`) |

An unknown or guarded key stops the run before any image is read, with an error naming it. `--seed` and `--n-splits` on the command line override the file. Rounds that change only `[model]` settings other than `backbone`, or `[classifiers.*]`, reuse every cached feature; a new backbone extracts its CNN features once, and returning to a backbone reuses them; changes to preprocessing, radiomics or augmentation recompute what they affect.

Compare rounds on the out-of-fold section and the cross-validated scores only. Rounds run without `--final`, so they never read Test; choosing values by Test would make the final Test score optimistic. Rerun only the kept round with `--final`. Many rounds on the same 5 folds also overfit the CV estimate, so change one family of parameters per round and keep the number of rounds small.

### Training in Docker

The `trainer` service of the root `docker-compose.yml` is an image with `requirements.lock` installed (`Dockerfile` here) that mounts `app/ml_engine` at `/work`, so code, data, round files, weights, artifacts and the feature cache stay on the host. Editing code or a round file needs no rebuild; rebuild only after `requirements.lock` changes. `docker compose up` does not start it (profile `train`). CPU only. From the repo root:

```bash
docker compose build trainer
docker compose run --rm trainer src.deep_learning.export_backbone --backbone efficientnet_b0 --out weights/efficientnet_b0   # once per backbone
docker compose run --rm trainer src.training --data-dir src/data \
  --manifest src/data/processed/dedup_20261028T234450903626Z/manifest.csv \
  --config rounds/0.2.0.toml --version 0.2.0                                    # add --final for the kept round
docker compose run --rm --entrypoint pytest trainer src/ -q                       # tests
```

- In Git Bash prefix the commands with `MSYS_NO_PATHCONV=1`, or it rewrites `/work`-style paths.
- To time a round: `time docker compose run ...` (POSIX) or `Measure-Command { docker compose run ... }` (PowerShell).
- On a Linux host the container writes files as root; add `--user "$(id -u):$(id -g)" -e HOME=/tmp` to `run`.

### Publish a model

A trained bundle is published outside git, as one asset of a GitHub Release tagged `model-v<version>`. Publishing is gated: per the training rules, a model that misses the 0.90 macro-F1 target goes to the team for review first and nobody publishes it until they have read the report.

```bash
python -m src.release package artifacts/model-v0.2.0 --out release   # archive, .sha256 and release notes
gh release create model-v0.2.0 release/model-v0.2.0.tar.gz release/model-v0.2.0.tar.gz.sha256 \
  --title model-v0.2.0 --notes-file release/release-notes-model-v0.2.0.md
python -m src.release fetch 0.2.0 --dest artifacts                  # download, verify, extract
```

`package` refuses a folder whose name does not match the manifest version and never overwrites an archive; its release notes carry the Test metrics from `manifest.json`. `fetch` compares the SHA-256 of the download with the published one and extracts nothing on a mismatch. Only regular files under `model-v<version>/` are extracted. `release/` is ignored by git.

The flow is: train (`src.training`) → `package` → `gh release create` → deploy. The backend image fetches the release named by its `MODEL_VERSION` build argument, so Cloud Run picks it up on the next build (`app/backend/README.md`, *Docker*). Locally, `fetch` the version and point compose at it with `MODEL_BUNDLE_DIR`. The deployed version is `0.1.0`, below the target and accepted as a known limitation (D28).

### Pipeline notebook

`notebooks/pipeline_walkthrough.ipynb` runs the pipeline cell by cell with matplotlib plots: dataset and leakage-free split, source geometry, one image through B3-B6, the quality gate over Train, augmentation, the CNN tensor and embedding, radiomics, folds, then selection, augmentation decision, ablation, out-of-fold metrics, the shortcut audit and a single Test evaluation. It only calls `src` functions and never writes an artifact. It runs in the `notebook` image (the training image plus JupyterLab and matplotlib, constrained by the lock):

```bash
docker compose --profile notebook up notebook   # open the printed http://127.0.0.1:8888/lab?token=... link
```

It reuses `artifacts/feature-cache`, so run a round first (or expect about 8 minutes in its feature section). Notebooks are committed without outputs.

To try the kept round in the app, copy the artifact of its `--final` run to the folder the backend mounts and restart it: `cp -r artifacts/model-v0.2.0 artifacts/current` (replace the folder if it exists), then `docker compose restart backend`.
