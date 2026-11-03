# Design

## Context

See proposal.md (Why) for the motivation, and `specs/image-quality-gate` and `specs/quality-threshold-calibration` for the required behavior.

Current state in `app/ml_engine/src/`:

- `preprocessing/quality.py`: `estimate_snr` divides the mean intensity above the Otsu threshold by `skimage.restoration.estimate_sigma` on the whole image. `shannon_entropy` uses the whole image. `QualityThresholds` defaults are 4.0 / 2.0. `check_quality` raises `QualityCheckError` with the metrics.
- `preprocessing/prepare.py`: `prepare_image` calls `measure_quality` on the original image before skull stripping, then `check_quality`. The radiomics region of interest is `skull_stripping.foreground_mask(gray)` (pixels above half the Otsu threshold, largest component, holes filled).
- `training/thresholds.py`: `calibrate(paths, percentile=1.0)` returns thresholds and the count used. `main` loads the manifest, keeps the purged Training split and prints JSON.
- `training/features.py`: `config_fingerprint` hashes `asdict(preprocessing)`, thresholds included. `pipelines/artifact.py`: `_preprocessing` rebuilds `QualityThresholds(**manifest["preprocessing"]["thresholds"])`.

The dataset is local at `app/ml_engine/src/data` (manifest `processed/dedup_20261028T234450903626Z/manifest.csv`).

## Evidence gathered while planning (2026-10-30)

Read-only measurements on the 1,830 purged Training images, run with the pinned image libraries in `python:3.11-slim`. Scripts and outputs are outside the repo.

| Definition | Lowest Training SNR | Rejected at 0.1st percentile | Fallback images |
|---|---|---|---|
| Current (wavelet) | - | 4, all notumor | - |
| Background = outside the dilated brain mask | 1.50 | 4 (meningioma 1, notumor 1, pituitary 2) | 9 |
| **Background = outside every dilated bright region (D2)** | **3.75** | 2, both notumor | 1 |

- A visual check of the lowest-scoring images under the second definition found clean sagittal and coronal scans. Neck and face tissue is above half the Otsu threshold but outside the largest component, so it was counted as noise. D2 fixes that.
- Under D2: uniform noise 1.67, real slice plus Gaussian noise σ 0.2 → 1.77, σ 0.1 → 3.15, σ 0.05 → 5.87; the same slice clean → 57. 542 of 1,830 images have a zero background MAD, so their SNR sits at the floor-defined cap.
- Brain-only entropy: lowest Training value 5.32 (a clean meningioma with a large bright lesion); the 0.1st percentile (5.57) rejected two clean scans.
- Conclusion: this curated set has essentially no unusable images, so any percentile threshold rejects good scans. The D2 SNR and brain-only entropy do leave a gap between real images and unusable ones, which D5 uses.

## Goals / Non-Goals

**Goals:**
- An SNR whose noise term ignores all tissue and whose scale is bounded.
- An entropy that does not depend on how much black background a slice has.
- Thresholds that pass every purged Training image and fail known-unusable images, with no class-dependent losses by construction.
- No silent mixing of old and new metric scales in the cache or in artifacts.

**Non-Goals:**
- Physically accurate MRI noise (Rician correction). The gate only needs to separate usable from unusable.
- New metrics (blur, contrast, foreground fraction).
- Changing the `QualityMetrics` contract or the backend error mapping.

## Decisions

### D1. Brain region is the existing foreground mask

`measure_quality` computes `foreground_mask(gray)` on the original image for the signal and the entropy. It is the region radiomics already uses, and it is computed before skull stripping. `prepare_image` computes it again for the radiomics ROI; the cost is a few milliseconds, and passing it would couple the gate to the radiomics path. When #166 moves `foreground_mask` to the B5 module, the import moves with it.

### D2. SNR = median brain intensity / background noise, floored

- **Signal:** median of `gray` inside the brain mask.
- **Bright regions:** `gray > 0.5 * threshold_otsu(gray)`, all components (the same threshold as `foreground_mask`, before it keeps the largest component).
- **Background:** pixels outside the bright regions dilated by `max(3, round(0.02 * min(h, w)))` pixels, so partial-volume edges are excluded.
- **Noise:** `1.4826 * MAD` of the background pixels, floored at `1 / 255`. The MAD ignores burned-in labels. A zero background gives `SNR = median * 255`, at most 255.
- **Flat image:** 0.

*Alternatives:* background outside the brain mask only (rejected by the evidence above: 1.50 on clean scans); background standard deviation (one annotation dominates it); wavelet estimate (reads anatomy as noise, D15).

### D3. Fallback when there is no background

If the background is under 1% of the pixels, noise is `estimate_sigma(gray)` with the same floor. Pure and heavy noise reach this path (bright regions cover the frame) and score 1.67-1.77, so they stay rejectable. Only 1 Training image takes the fallback under D2.

### D4. Entropy inside the mask

Shannon entropy in bits of the 256-bin histogram over [0, 1] of `gray[mask]`; 0 for an empty mask. A flat non-black image has one bin and gives 0.

### D5. Thresholds in the gap between references and Training

`training/thresholds.py` becomes single-purpose functions:

- `measure(paths) -> list[QualityMetrics]`: readable images only.
- `reference_images(anchor) -> dict[str, dict[str, np.ndarray]]`: unusable references per metric, fixed seed. SNR: blank, uniform noise, `anchor` plus Gaussian noise σ 0.2 (clipped). Entropy: blank, `anchor` posterized to 4 levels (at most 2 bits by construction). `anchor` is the calibration image with the median SNR (ties by path), so the degraded references look like real slices and still come from Training only.
- `gap_thresholds(measurements, references) -> Calibration`: for each metric, `low = min` over Training and `high = max` over its references; raises `ValueError` naming the metric and both values when `low <= high`. Minimum SNR = `sqrt(low * high)` (SNR is a ratio, so the midpoint is geometric). Minimum entropy = `(low + high) / 2`. Each is rounded to 2 decimals, and the rounded value is checked to stay strictly inside the gap.

`Calibration` holds the thresholds, the images used and the `low` / `high` pair per metric. The CLI prints them as JSON. The percentile option is removed.

Expected on the current dataset: SNR gap 1.77-3.75, so about 2.58; entropy gap 2.0-5.32, so about 3.66.

*Alternatives:* 0.1st percentile (rejects 2 clean notumor scans); Training minimum only (no guarantee that noise or blank images fail).

### D6. Per-class rejections

On Training the gap rule rejects nothing by construction, so the CLI does not print per-class counts. Testing rejections per class are measured once with an ad-hoc command reusing `measure` on the kept Testing samples (not committed, so the CLI never opens Testing) and written to the decision log.

### D7. Definition version on the thresholds

`QualityThresholds` gains `metrics_version: int = 2` (the current metrics are version 1, never stored). Because it is a field, it reaches the feature cache fingerprint and the artifact manifest through `asdict` with no extra code. `pipelines/artifact.py` refuses (`ModelUnavailableError`) a manifest whose thresholds lack the field or carry a different version, the same pattern as the backbone check (D9 of the decision log). A changed definition must bump the constant.

### D8. Review before the defaults change

The calibrated values become the `QualityThresholds` defaults when a team member reviews a contact sheet of every Testing image the values reject (written outside the repo) and finds them unusable or borderline, with no class above 1% of its Testing images. The decision names the reviewer and date. Otherwise the defaults stay and the decision records why and which ticket owns the fix (existing spec rule). The synthetic tests are a second guard: blank and uniform-noise images fail and a synthetic slice passes with the defaults.

## Risks / Trade-offs

- [The SNR gap on Training is narrow (1.77 vs 3.75) and Testing images may fall below 2.58] → D8 reviews every Testing rejection before the defaults change. If clean Testing scans fall in the gap, the decision records it and the defaults stay.
- [If the review fails, the kept default 4.0 sits above the lowest Training SNR (3.75) on the new scale] → The decision records it; the interim defaults are not calibrated values, and the follow-up ticket owns replacing them.
- [σ 0.1 Gaussian noise scores 3.15, above the threshold, so moderate noise passes] → Accepted: the gate rejects unusable images; moderate noise is the classifier's problem, not the gate's.
- [Background MAD is 0 for 30% of images, so their SNRs sit at the cap] → The gate only uses the low end.
- [`1.4826 * MAD` of a magnitude background is not the Rician sigma] → Accepted and recorded as an assumption.
- [The gap depends on the anchor image] → It is chosen deterministically, and the CLI prints both ends of each gap.
- [The new SNR scale changes the numbers in the clinical summary] → Display only; noted in the decision.

## Migration Plan

1. Land the new metrics, `metrics_version` and calibration code with defaults 4.0 / 2.0 for SNR / entropy, tests updated.
2. Calibrate on the purged Training split, measure Testing rejections, review (D8).
3. If the review passes, change the defaults and the `test_training_run` manifest assertion; otherwise keep them and record why.
4. Rollback: revert the change. No published artifact depends on either scale.
