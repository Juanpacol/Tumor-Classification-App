# Proposal

## Why

Block B5 of the preprocessing pipeline (`docs/ml-engine/preprocessing-pipeline.md`, Azure #166) replaces skull stripping with a plain brain mask, because skull stripping lowered accuracy with a patient-level split [6] and had a negligible effect in [7]. The code still carries the old design: `preprocessing/skull_stripping.py` has an optional `strip_skull` step behind `PreprocessingConfig.skull_stripping`, two masks (the stripped mask and `foreground_mask`) and no fallback rule, so an image whose mask is empty reaches radiomics and the coming crop step (B6) with no region at all. B6 needs one mask with a defined fallback before it can crop to it.

Most of B5 already exists: `foreground_mask` (half the Otsu threshold, largest component, holes filled) is the region the quality gate (B4, metrics version 2) and radiomics use today. This change keeps it as the B5 mask and does not redo that work.

## What Changes

- **One brain mask.** `foreground_mask` becomes the B5 mask, unchanged, in a module named after the block (`preprocessing/brain_mask.py`). Its definition is not modified, so the B4 measurements, the calibrated thresholds (D17) and `metrics_version` stay as they are.
- **Fallback rule.** When the mask is empty or covers less than a minimum fraction of the image, the full image is used as the region, and preparation does not fail. The minimum fraction is a preprocessing setting, so it reaches the feature cache key and the artifact manifest. Its default is checked against the purged Training split and recorded in the decision log.
- **BREAKING (artifact manifest): skull stripping is removed.** `strip_skull`, `SkullStrippingResult`, its constants and `PreprocessingConfig.skull_stripping` are deleted. The radiomics image is always the loaded image. A manifest whose preprocessing fields do not match this engine is refused with `ModelUnavailableError` instead of crashing. No artifact has been published (#114 pending).
- **`skull_stripped` in the contract stays and is always `false`.** This settles the open decision of #166 without a backend or frontend API change. `measure_quality` loses its `skull_stripped` parameter.
- **Docs.** B5 in `docs/ml-engine/preprocessing-pipeline.md` (and its `.tex`) states the half-Otsu threshold, the fallback rule and the new module; the `skull_stripped` open decision is closed; a new decision entry in `docs/ml-engine/decisions.md` records the fallback measurement.

## Capabilities

### New Capabilities
- `brain-mask`: the region of interest computed for every image (definition, fallback, no skull stripping), shared by radiomics and the geometry step, identical at training and inference.

### Modified Capabilities
- `image-quality-gate`: the requirement "Quality is measured on the original image" drops its skull-stripping scenario and states that quality is measured inside the brain mask before the fallback. Its main spec is created when the change `quality-gate-metrics` is archived, which must happen first.

## Impact

- **Code** (`app/ml_engine/src/`): `preprocessing/skull_stripping.py` renamed to `preprocessing/brain_mask.py` (strip step removed, fallback added), `preprocessing/prepare.py`, `preprocessing/quality.py` (import, no `skull_stripped` parameter), `training/thresholds.py` (call site), `pipelines/artifact.py` (refuses mismatched preprocessing fields), `pipelines/fake.py` (`skull_stripped=False`).
- **Tests**: `test_skull_stripping.py` becomes `test_brain_mask.py`; `test_prepare.py`, `test_quality.py`, `test_pipeline_artifact.py`, `test_training_features.py` (cache key).
- **Feature cache**: the preprocessing fields change, so old entries are not reused.
- **Backend / frontend**: no code change; `skull_stripped` is always `false` ("Not applied" in the clinical summary). Frontend sample data may be aligned.
- **Dependencies**: none new.
- **Out of scope**: B6 geometry (crop, pad, resize) and B8 tensor changes, which complete #166 in a later change; removing `skull_stripped` from the API; any change to the B4 metrics or thresholds.
