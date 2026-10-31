# Design

## Context

Current code (`app/ml_engine/src/`), observed on 2026-10-30:

- `preprocessing/skull_stripping.py` holds three things: `strip_skull` (Otsu, opening, erosion, largest component, dilation, hole filling; falls back to an all-true mask outside 15-95% coverage), `bright_regions` (pixels above half the Otsu threshold) and `foreground_mask` (`bright_regions`, largest component, holes filled; a flat image is all true if not black, empty otherwise).
- `preprocessing/prepare.py`: `prepare_image` runs `strip_skull` only when `PreprocessingConfig.skull_stripping` is true (default false), computes `foreground_mask(gray)`, measures quality on it, and returns the stripped mask and image when stripping applied, otherwise `gray` and the foreground.
- `preprocessing/quality.py` (B4, `METRICS_VERSION = 2`, D17) imports `bright_regions` for the SNR background and `foreground_mask` for signal and entropy. The thresholds 2.74 / 3.20 were calibrated on this exact mask.
- `classical/extractor.py` extracts radiomics inside the mask; `classical/roi.crop_to_mask` returns `None` below a minimum pixel count, so an empty mask silently gives no texture features.
- `pipelines/artifact._preprocessing` builds `PreprocessingConfig(**fields)` from the manifest. An unknown key raises `TypeError`, which `load_artifact` maps to `ModelUnavailableError`, but a missing key is silently filled with the default.
- `training/features.py` keys the feature cache on `asdict(preprocessing)`.
- `QualityMetrics.skull_stripped` reaches the backend schema and the frontend clinical summary ("Applied" / "Not applied").

Measurement made while planning (purged Training split, 1,830 images, manifest `dedup_20261028T234450903626Z`; `foreground_mask` reimplemented with a NumPy Otsu on 256 bins because no environment with scikit-image was available, so values can differ slightly):

| Class | n | Min coverage | 1st percentile | Median | Below 5% |
|---|---|---|---|---|---|
| glioma | 850 | 22.7% | 27.0% | 48.8% | 0 |
| meningioma | 446 | 15.4% | 21.6% | 50.3% | 0 |
| no_tumor | 260 | 7.0% | 24.2% | 58.8% | 0 |
| pituitary | 274 | 18.2% | 41.2% | 68.1% | 0 |
| all | 1,830 | 7.0% | 24.5% | 52.7% | 0 |

The maximum is 100%: some images are already cropped to the head.

## Goals / Non-Goals

**Goals:**
- One mask function and one fallback rule that B6 can crop to.
- B4 stays bit-identical: same measurements, same thresholds, same `metrics_version`.

**Non-Goals:**
- Tuning the mask threshold (half Otsu) or adding morphology.
- Exposing the fallback in `PredictionResult` or the API.
- Using the mask for the CNN input; that is B6.

## Decisions

### D1. Keep `foreground_mask` as the B5 mask

The pipeline doc says "Otsu threshold, largest connected component, hole filling". `foreground_mask` does that at half the Otsu threshold, which keeps the darker edge of the brain. It is the region B4 was calibrated on (D17) and the region radiomics already uses.

*Alternatives:* full Otsu threshold (would change the B4 signal and entropy, force `metrics_version = 3` and a recalibration with its Testing review, for no measured gain); `strip_skull` as the mask (it is the skull-stripping step the evidence rejects, and it erodes brain tissue). The doc wording is corrected to "half the Otsu threshold".

### D2. Fallback applies after the quality gate, not inside it

New module `preprocessing/brain_mask.py` keeps `bright_regions` and `foreground_mask` as they are and adds `region_of_interest(mask, min_fraction) -> np.ndarray`: `mask` when its mean is at least `min_fraction`, otherwise an all-true mask. It takes the mask, not the image, so `foreground_mask` runs once per image. `prepare_image` computes `foreground_mask` once, passes it to `measure_quality` (as today), and then applies the fallback to the same array for the returned `mask`.

So a degenerate mask cannot raise the measured quality of a bad image (a full-image fallback would add background to the entropy and the median, and make metrics depend on a new setting), and B4 output does not change.

*Alternatives:* fallback inside `foreground_mask` (changes B4 for any image below the fraction and couples the gate to a preprocessing setting); a `BrainMask(mask, fallback)` result type (no consumer reads the flag today; the measurement in D4 calls the function directly).

### D3. Minimum fraction 0.05, stored in `PreprocessingConfig`

`PreprocessingConfig.min_mask_fraction: float = 0.05`. Being a config field, it reaches the artifact manifest and the feature cache key with no extra code. 0.05 is below every Training image (lowest 7.0%), so the fallback only catches degenerate inputs (black frames, a single bright mark). There is no upper bound: a mask over the whole frame is a head already cropped, and its fallback would be the same region.

*Alternatives:* the 15% of `strip_skull` (one no_tumor and one meningioma image are close to it, 7.0% and 15.4%, and would be reshaped by the rule); a percentile of Training (the B4 lesson, D17: a curated set has no degenerate images, so a percentile would cut good ones).

### D4. Record the fallback check as decision D18

The implementation re-runs the coverage measurement with the real `foreground_mask` over the purged Training split (a small function under `training/`, or a one-off command documented in the decision; not a new CLI unless it is reused) and writes D18 in `docs/ml-engine/decisions.md`: the per-class table, the default, and that no Training image falls back. If the real values differ from the table above enough that any Training image falls back, the default is lowered below the minimum before merging.

### D5. Remove skull stripping, keep the contract field

Delete `strip_skull`, `SkullStrippingResult`, `MIN_MASK_FRACTION`, `MAX_MASK_FRACTION`, `_RADIUS_FRACTION`, `_brain_mask` and `PreprocessingConfig.skull_stripping`. `measure_quality` loses its `skull_stripped` parameter and always reports `False`; `training/thresholds.py` and the tests drop the argument. `pipelines/fake.py` reports `False`.

`QualityMetrics.skull_stripped` stays so the backend and frontend need no change; removing it is an API change for its own issue. Frontend `sampleData.ts` may be set to `false` for consistency.

*Alternative:* remove the field now (touches `contracts.py`, backend schemas and their documentation test, frontend types and tests, for a hackathon demo that only shows "Not applied").

### D6. Artifact loader refuses mismatched preprocessing fields

`_preprocessing` compares the manifest keys with the fields of `PreprocessingConfig` and raises `ModelUnavailableError` naming the missing and unexpected keys, so an artifact trained before this change (with `skull_stripping` and without `min_mask_fraction`) gets a clear 503 `MODEL_UNAVAILABLE`, and a manifest missing `min_mask_fraction` is not served with the default. The check runs after the existing `metrics_version` check.

### D7. Module rename

`skull_stripping.py` becomes `brain_mask.py` and `test_skull_stripping.py` becomes `test_brain_mask.py`, matching the traceability table of the pipeline doc. Imports in `quality.py`, `prepare.py` and the tests move with it. No re-export shim: nothing outside `app/ml_engine` imports the module.

## Risks / Trade-offs

- [The NumPy Otsu used for planning differs from scikit-image] → D4 re-measures with the real function before merging.
- [A real upload with a small head (zoomed-out slice) falls back to the full image] → radiomics then include background, which only lowers confidence; B4 has already measured it on the real mask. Accepted.
- [Two masks in `prepare_image` (raw for B4, with fallback for downstream)] → they are the same array except below 5% coverage; the spec states which one each step uses.
- [The B6 change will need the region's bounding box] → the returned `mask` is that input; no further API needed here.

## Migration Plan

1. Archive `quality-gate-metrics` first, so `openspec/specs/image-quality-gate` exists for this change's MODIFIED delta.
2. Implement on its own branch (`AB-166-brain-mask`), not on `AB-175-quality-gate-metrics`.
3. No published artifact exists; local artifacts and the feature cache must be rebuilt (the cache key changes by itself).
4. Rollback: revert the commits; artifacts trained after this change would then be refused by the old loader (`TypeError` on `min_mask_fraction`), so retrain.
