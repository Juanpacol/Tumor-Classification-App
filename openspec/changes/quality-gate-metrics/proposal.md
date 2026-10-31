# Proposal

## Why

Block B4 of the preprocessing pipeline (`docs/ml-engine/preprocessing-pipeline.md`, Azure #175) has a Training-only calibration (#165), but its values were not applied (decision D15). The current metrics reject usable scans and concentrate in one class: at the 1st percentile the gate rejects notumor 8.1% on Training and 8.0% on Testing, against 0.2-1.8% for the other classes. A visual review of the 61 rejected images traced this to the metrics:

- The wavelet noise estimate reads fine FLAIR/T2 anatomy as noise, so clean scans get a low SNR (14-40).
- Near noise-free re-processed glioma copies reach an SNR above 250,000, so the SNR scale is not comparable across sources.
- Entropy is measured over the whole image, so dark slices with a small brain and a large black background score low.

The threshold rule is also wrong for this data. The dataset is curated, so it has almost no unusable images, and any percentile threshold rejects good scans by construction. Measurements made while planning (design.md, Evidence) confirm this even at the 0.1st percentile.

## What Changes

- **SNR is redefined.** Signal is the median intensity inside the brain mask. Noise is a robust (MAD-based) standard deviation of the dark background, away from every bright region of the image, floored at one 8-bit intensity step. The wavelet estimator is kept only as a fallback for images with almost no background. The floor bounds the SNR, so the scale is comparable across sources.
- **Entropy is measured inside the brain mask only**, so the size of the black background no longer changes it.
- **Thresholds come from a gap, not a percentile.** Each minimum sits between the highest value over synthetic unusable references (blank, uniform noise, heavy Gaussian noise; posterized image for entropy) and the lowest value over the purged Training split. Every Training image passes and every reference fails; calibration refuses to run if there is no gap. The percentile option is removed.
- **Thresholds carry a metric definition version.** It reaches the feature cache fingerprint and the artifact manifest, and serving refuses an artifact calibrated for another version.
- **New defaults.** After a team member reviews the Testing images the new values reject, and if they are unusable and do not concentrate in one class, the calibrated values become the `QualityThresholds` defaults. Otherwise the defaults stay and the decision log says why (existing review rule).
- **Docs.** A new decision in `docs/ml-engine/decisions.md` supersedes the metric part of D7 and the "not applied" part of D15. The B4 parts of `docs/ml-engine/preprocessing-pipeline.md` and its `.tex` version are updated.
- The `QualityMetrics` contract keeps its fields (`snr`, `entropy`, `skull_stripped`). Their values change scale, so the backend and frontend need no code change.

## Capabilities

### New Capabilities
- `image-quality-gate`: how a single image's quality is measured (SNR and entropy definitions and their version) and when the gate rejects it. Used identically at training and inference.

### Modified Capabilities
- `quality-threshold-calibration`: thresholds sit in the gap between unusable references and Training measurements instead of at the 1st percentile; the output and the decision-log entry report each metric's gap instead of a percentile.

## Impact

- **Code** (`app/ml_engine/src/`): `preprocessing/quality.py` (new SNR and entropy, `metrics_version`, defaults), `training/thresholds.py` (references, gap rule, output), `pipelines/artifact.py` (refuses a different `metrics_version`). `preprocessing/prepare.py` and `skull_stripping.foreground_mask` are reused as they are.
- **Tests**: `test_quality.py`, `test_prepare.py`, `test_training_thresholds.py`, `test_pipeline_artifact.py`, `test_training_features.py` (fingerprint), and `test_training_run.py` (manifest thresholds).
- **Artifacts**: the manifest's thresholds gain `metrics_version`. Artifacts trained before this change are refused and must be retrained; none has been published (#114 pending).
- **Feature cache**: the version is in the fingerprint, so old cache entries are never reused.
- **Backend / frontend**: no code change. `snr` values shown in the clinical summary change scale.
- **Dependencies**: none new.
- **Out of scope**: per-fold threshold fitting (accepted limitation in the pipeline doc), skull stripping, new quality metrics, the real-data training run (#112).
