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

`load_mri_image` (JPG/PNG to a 2D float32 array in [0, 1]) followed by `prepare_image`, which runs the quality gate and builds both model inputs (a masked grayscale image for radiomics and a normalized 3-channel array for the CNN).

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

### Skull stripping

Disabled by default (`PreprocessingConfig(skull_stripping=False)`). Enable it only after an ablation with a patient-level split shows it helps. When enabled and the mask is implausible (under 15% or over 95% of the image) the image is used unchanged and `skull_stripped=false` is reported.

## Radiomic features

`extract_radiomics(radiomics_image, mask)` (from `PreprocessedImage`) returns a fixed-length vector of 45 finite features with stable names: GLCM (5), LBP (10), wavelet (21) and first-order intensity (9). All parameters are in `RadiomicsConfig`. Details and deviations from PyRadiomics and IBSI are in `docs/ml-engine/decisions.md` (D8).

## Deep features

`DeepFeatureExtractor.from_bundle(folder)` loads a frozen ImageNet backbone (EfficientNet-B0 by default) and `extract` / `extract_batch` turn the `cnn_input` of `prepare_image` into 1280-d embeddings. Weights are never downloaded at runtime. Create the bundle once on a development machine:

```bash
python -m src.deep_learning.export_backbone --out weights
```

The `weights/` folder is ignored by git. Details are in `docs/ml-engine/decisions.md` (D9).

## Training data and model selection

The cleaned dataset is described by the `manifest.csv` that the deduplication step writes (`python -m src.preprocessing.deduplication --data-dir src/data`). `load_samples` reads the kept images from it, `FeatureCache` computes radiomic and CNN features once per image and `compare_classifiers` cross-validates logistic regression, an RBF SVM and a random forest on grouped folds that `assign_folds` computes once per run. Decisions are in `docs/ml-engine/decisions.md` (D11, D12, D16).

### Train

With the backbone bundle exported (see *Deep features*) and the cleaning `manifest.csv`:

```bash
python -m src.training --data-dir src/data --manifest <path>/manifest.csv --version 0.1.0
```

It first removes from Train every image whose `group_id` also appears in Test, and prints the removed count per class (also stored as `split_removed_per_class` in the report and the artifact manifest). It then assigns every Train image to one of 5 grouped folds (`StratifiedGroupKFold` on `group_id`), selects a classifier by cross-validated macro-F1 on those folds (balanced accuracy is reported too), runs an ablation (CNN-only, radiomics-only, hybrid), calibrates the model, scores the official Test split once and writes, under `artifacts/` (override with `--out`):

- `model-v<version>/`: `classifier.joblib`, the backbone files, `feature_stats.json` and `manifest.json` with the configs, versions, metrics and sha256 of every file.
- `reports/<timestamp>/`: `report.json` and `report.md` (test metrics including balanced accuracy, confusion matrix, images per fold and class, selection, ablation, and whether the 0.90 macro-F1 target was met), and `folds.csv` with the fold of every Train image left after the leak removal and whether the quality gate let it through.

The same `--seed` (default 42) reproduces the same metrics. Features are cached under `artifacts/feature-cache`, so a second run skips the extraction. An existing artifact or report is never overwritten.
