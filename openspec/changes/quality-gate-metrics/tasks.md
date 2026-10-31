# Tasks

Run tests from `app/ml_engine` with `pytest src/ -v`. Work on a branch `AB-175-quality-gate-metrics` from `develop`; commits use `AB-175:`.

## 1. Quality metrics (B4 gate)

- [x] 1.1 In `preprocessing/quality.py`, implement the SNR of design D2 and D3: signal = median inside `foreground_mask`, background = pixels outside the bright regions (`> 0.5 * Otsu`, all components) dilated by `max(3, round(0.02 * min(h, w)))`, noise = `1.4826 * MAD` floored at `1 / 255`, `estimate_sigma` fallback when the background is under 1% of the pixels, 0 for a flat image. Verify with `src/tests/test_quality.py`: added noise lowers the SNR; zero-mean texture inside the mask and a separate bright structure in the background each change the SNR by under 5%; a zero background gives `median * 255`; a frame-filling image returns a finite SNR; uniform noise scores below 2; a flat image gives 0
- [x] 1.2 Make `shannon_entropy` measure only the pixels inside the mask (0 for an empty mask) and have `measure_quality` compute the mask once for both metrics. Verify in `test_quality.py`: padding a synthetic slice with a black border that doubles its size changes entropy by under 0.05 bits, a flat image gives 0, uniform noise stays near 8 bits
- [x] 1.3 Add `metrics_version: int = 2` to `QualityThresholds` (defaults stay 4.0 / 2.0 for SNR / entropy for now) and make `pipelines/artifact.py` raise `ModelUnavailableError` when the manifest's thresholds lack `metrics_version` or carry another value. Verify with `src/tests/test_pipeline_artifact.py` (manifest without the field refused, matching one loads) and a `test_training_features.py` case that two configs differing only in `metrics_version` have different fingerprints
- [x] 1.4 Check the gate behavior with the new metrics: blank image rejected naming SNR and entropy, uniform noise rejected, synthetic slice passes; add a `test_prepare.py` case that with skull stripping enabled the reported metrics equal `measure_quality` on the original image. Verify `pytest src/tests/test_quality.py src/tests/test_prepare.py src/tests/test_inference_pipeline.py -v` passes
- [x] 1.5 Update the `QualityThresholds` docstring and the "Quality thresholds" table of `app/ml_engine/README.md` with the new definitions and `metrics_version`. Verify `grep -n "wavelet noise estimate" app/ml_engine/README.md` returns nothing

## 2. Calibration (B4 thresholds)

- [x] 2.1 Rewrite `training/thresholds.py` per design D5: `measure(paths)`, `reference_images(anchor)` (fixed seed; SNR: blank, uniform noise, anchor + Gaussian σ 0.2 clipped; entropy: blank, anchor posterized to 4 levels), `gap_thresholds(measurements, references)` returning a `Calibration` (thresholds, images used, `low` / `high` per metric; geometric midpoint for SNR, arithmetic for entropy, rounded to 2 decimals and kept strictly inside the gap); anchor = calibration image with the median SNR, ties by path; remove the percentile option. Verify with `src/tests/test_training_thresholds.py`: every calibration image passes, every reference fails its metric, a set whose lowest value is not above the references raises naming the metric, an unreadable image is skipped and not counted, nothing readable raises
- [x] 2.2 Make the CLI print `min_snr`, `min_entropy`, `images_used` and `gap: {snr: {low, high}, entropy: {low, high}}`. Verify the CLI test checks those keys, and the "Testing images are never read" and "manifest required" tests still pass
- [x] 2.3 Update the calibration section of `app/ml_engine/README.md` (gap rule, output keys, no percentile). Verify `grep -n "percentile" app/ml_engine/README.md` returns nothing about calibration

## 3. Calibrate and review on the real dataset

- [x] 3.1 From `app/ml_engine`, run `python -m src.training.thresholds --data-dir src/data --manifest src/data/processed/dedup_20261028T234450903626Z/manifest.csv` and save the JSON outside the repo. Verify `images_used = 1830` and that both gaps are reported; compare with the planning estimate (SNR about 2.58 in 1.77-3.75, entropy about 3.66 in 2.0-5.32) and record any difference
- [x] 3.2 With an ad-hoc command (not committed) reusing `measure`, count per class the kept Testing images below the calibrated values, and the Training images that take the D3 fallback. Verify the counts are written down for the decision entry
- [x] 3.3 Write a contact sheet of every rejected Testing image (label, SNR, entropy) to a folder outside the repo and have a team member review it against design D8. Verify the outcome is recorded with the reviewer's name and date: per image unusable / borderline / usable, and per-class Testing rejection rates against the 1% limit

## 4. Apply or keep the defaults

- [x] 4.1 If the review passes D8, set the `QualityThresholds` defaults to the calibrated values and update the manifest assertion in `src/tests/test_training_run.py`; if it fails, keep 4.0 / 2.0 and open a follow-up ticket in Azure Boards. Either way verify `pytest src/ -v` passes, including blank and uniform-noise images rejected and the synthetic slice passing with the defaults
- [x] 4.2 Add a new decision to `docs/ml-engine/decisions.md` (AB-175) superseding the metric part of D7 and the "not applied" part of D15: SNR and entropy definitions and status (Assumption: MAD of a magnitude background is not the Rician sigma), the planning evidence of design.md, `metrics_version` and the artifact refusal, calibration date, images used, values and both gaps, fallback count, per-class rejections on Training (0) and Testing, review outcome with reviewer, whether the defaults changed (and the follow-up ticket if not), and the new SNR scale shown in the clinical summary. Update D7's threshold bullet to point to it and close the "Redefine the quality-gate metrics" item in Open verifications. Verify the entry contains each of those items
- [x] 4.3 Update `docs/ml-engine/preprocessing-pipeline.md` and `preprocessing-pipeline.tex` together: step 4 of the training table (gap rule), the B4 block (definitions, calibration command), the "B4 is not fitted per fold" limitation wording, and the B4 traceability state. Verify `grep -n "1st percentile" docs/ml-engine/preprocessing-pipeline.md docs/ml-engine/preprocessing-pipeline.tex` returns nothing

## 5. Integration check

- [x] 5.1 In `app/ml_engine`, run `pytest src/ -v` and `black --check src`, then run the training CLI end to end on the synthetic dataset and confirm the artifact manifest records the current default thresholds with `metrics_version = 2` and that the serving pipeline loads it; in `app/backend`, run `python -m pytest tests/ -v` to confirm the contract is unchanged

## Workflow follow-up

- Update Azure #175 with the decision entry and close it when merged.
- Commit with `AB-175:` messages, open the PR to `develop`, then archive the change after review.
