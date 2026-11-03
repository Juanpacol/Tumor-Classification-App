# Tasks

Run tests from `app/ml_engine` with `pytest src/ -v`. Work on a branch `AB-166-brain-mask` from `develop` after `quality-gate-metrics` is merged; commits use `AB-166:`.

## 1. Brain mask module (B5)

- [x] 1.1 Rename `preprocessing/skull_stripping.py` to `preprocessing/brain_mask.py` and `tests/test_skull_stripping.py` to `tests/test_brain_mask.py` (`git mv`), and move the imports in `preprocessing/quality.py`, `preprocessing/prepare.py` and `tests/test_quality.py`. Verify `grep -rn "skull_stripping" app/ml_engine/src --include=*.py` only shows the code removed in group 2
- [x] 1.2 Add `region_of_interest(mask, min_fraction)` per design D2 (the mask when its mean is at least `min_fraction`, otherwise all true; never raises), keeping `bright_regions` and `foreground_mask` unchanged. Verify with `test_brain_mask.py`: a black image and a single bright spot under the fraction give an all-true region; a head phantom gives its `foreground_mask`; two disconnected bright regions keep only the larger; an enclosed dark hole is inside the mask; the existing `foreground_mask` tests still pass
- [x] 1.3 Add `min_mask_fraction: float = 0.05` to `PreprocessingConfig` and have `prepare_image` return `region_of_interest` as `mask`, measuring quality on the raw `foreground_mask` (computed once). Verify with `test_prepare.py`: an image below the fraction gets an all-true mask while its quality equals `measure_quality` on the raw mask; a normal slice's mask equals `foreground_mask`; and with `test_training_features.py` that two configs differing only in `min_mask_fraction` have different cache fingerprints

## 2. Remove skull stripping (design D5)

- [x] 2.1 Delete `strip_skull`, `SkullStrippingResult`, `MIN_MASK_FRACTION`, `MAX_MASK_FRACTION`, `_RADIUS_FRACTION` and `_brain_mask`, and their tests; delete `PreprocessingConfig.skull_stripping` and the stripped-image branch of `prepare_image`, so `radiomics_image` is always the loaded image. Update the `PreprocessingConfig`, `PreprocessedImage` and `prepare_image` docstrings. Verify `grep -rn "strip_skull\|skull_stripping" app/ml_engine/src --include=*.py` returns nothing and `test_prepare.py` asserts `radiomics_image` equals the input
- [x] 2.2 Remove the `skull_stripped` parameter of `measure_quality` (always reports `False`) and update its callers: `prepare.py`, `training/thresholds.py`, `tests/test_quality.py`, `tests/test_prepare.py`; set `skull_stripped=False` in `pipelines/fake.py`. Verify a test checks a prepared image reports `skull_stripped is False`, and `pytest src/tests/test_quality.py src/tests/test_prepare.py src/tests/test_training_thresholds.py src/tests/test_fake_pipeline.py -v` passes
- [x] 2.3 Set `skull_stripped: false` in `app/frontend/src/services/api/sampleData.ts` and fix the `ClinicalSummary.test.tsx` case if it expects "Applied". Verify `npm test` in `app/frontend` passes for the clinical summary tests

## 3. Artifact compatibility (design D6)

- [x] 3.1 In `pipelines/artifact._preprocessing`, after the `metrics_version` check, raise `ModelUnavailableError` naming missing and unexpected keys when the manifest's preprocessing fields differ from `PreprocessingConfig`. Replace the `skull_stripping=True` case in `test_pipeline_artifact.py` with: a manifest containing `skull_stripping` is refused naming it; one without `min_mask_fraction` is refused naming it; a manifest written by `training/run.py` loads with its `min_mask_fraction`. Verify `pytest src/tests/test_pipeline_artifact.py src/tests/test_training_run.py src/tests/test_inference_pipeline.py -v` passes

## 4. Real-data check and docs

- [x] 4.1 With the real `foreground_mask`, measure the mask coverage of the purged Training split (manifest `src/data/processed/dedup_20261028T234450903626Z/manifest.csv`) per class (min, 1st percentile, median, count below `min_mask_fraction`) using an ad-hoc command, not committed. Verify no Training image falls back; if any does, lower the default below the minimum and re-run group 1 tests. Compare with the planning table of design.md
- [x] 4.2 Add decision D18 (AB-166) to `docs/ml-engine/decisions.md`: B5 mask definition (half Otsu, why not full Otsu: B4 unchanged), the fallback rule and where it applies (after the gate), the 0.05 default with the 4.1 table, skull stripping removed, `skull_stripped` kept and always false (removal deferred), and the artifact refusal of mismatched preprocessing fields. Update D17's file list to `brain_mask.py`. Verify the entry contains each item
- [x] 4.3 Update `docs/ml-engine/preprocessing-pipeline.md` and `preprocessing-pipeline.tex` together: the B5 block (half Otsu, minimum fraction, fallback after the gate), the "`skull_stripped` in the contract" limitation (decided: kept, always false), and the B5 traceability row (`preprocessing/brain_mask.py`, Done). Update `app/ml_engine/README.md` if it mentions skull stripping. Verify `grep -n "skull_stripping.py\|skull stripping is off" docs/ml-engine app/ml_engine/README.md -r` returns nothing

## 5. Integration

- [x] 5.1 Run `pytest src/ -v` in `app/ml_engine` and `python -m pytest tests/ -v` in `app/backend`. Verify both pass with no skipped test caused by this change

## Workflow follow-up

- Archive `quality-gate-metrics` before archiving this change, so the `image-quality-gate` MODIFIED delta has a main spec.
- B6 (crop, pad square, resize) and B8 complete #166 in a later change that crops to this change's `mask`.
- Removing `QualityMetrics.skull_stripped` from the API is its own Boards issue if the team wants it.
