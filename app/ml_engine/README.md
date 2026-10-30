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

| Metric | Minimum | Method |
|---|---|---|
| SNR | 4.0 | mean intensity above the Otsu threshold divided by a wavelet noise estimate |
| Shannon entropy | 2.0 bits | 256-bin intensity histogram |

**These values are provisional.** A calibration on the training data was run and recorded but not applied, because the current metrics reject usable scans and concentrate the rejections in notumor (`docs/ml-engine/decisions.md`, D15). The calibration command reads only the leakage-free Training split and never opens a Testing image:

```bash
python -m src.training.thresholds --data-dir src/data --manifest src/data/processed/dedup_20261028T234450903626Z/manifest.csv
```

It prints the 1st percentile of each metric, the percentile and the number of images used. Before changing `QualityThresholds` (`src/preprocessing/quality.py`), check the per-class rejections against a visual review and record them in `docs/ml-engine/decisions.md`.

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

The cleaned dataset is described by the `manifest.csv` that the deduplication step writes (`python -m src.preprocessing.deduplication --data-dir src/data`). `load_samples` reads the kept images from it, `FeatureCache` computes radiomic and CNN features once per image and `compare_classifiers` cross-validates logistic regression, an RBF SVM and a random forest with group-aware folds. Decisions are in `docs/ml-engine/decisions.md` (D11, D12).

### Train

With the backbone bundle exported (see *Deep features*) and the cleaning `manifest.csv`:

```bash
python -m src.training --data-dir src/data --manifest <path>/manifest.csv --version 0.1.0
```

It first removes from Train every image whose `group_id` also appears in Test, and prints the removed count per class (also stored as `split_removed_per_class` in the report and the artifact manifest). It then selects a classifier by group-aware cross-validation, runs an ablation (CNN-only, radiomics-only, hybrid), calibrates the model, scores the official Test split once and writes, under `artifacts/` (override with `--out`):

- `model-v<version>/`: `classifier.joblib`, the backbone files, `feature_stats.json` and `manifest.json` with the configs, versions, metrics and sha256 of every file.
- `reports/<timestamp>/`: `report.json` and `report.md` (test metrics, confusion matrix, selection, ablation, and whether the 0.90 macro-F1 target was met).

The same `--seed` (default 42) reproduces the same metrics. Features are cached under `artifacts/feature-cache`, so a second run skips the extraction. An existing artifact or report is never overwritten.
