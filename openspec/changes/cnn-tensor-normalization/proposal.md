# Proposal

## Why

Block B8 of the preprocessing pipeline (`docs/ml-engine/preprocessing-pipeline.md`, Azure #166) says the CNN tensor is normalized "with the mean and std of the selected backbone, read from its configuration", and #166 adds "never hard-coded". Today the ImageNet values are hard-coded twice: as defaults of `PreprocessingConfig.cnn_mean` / `cnn_std` (which `prepare_image` applies) and as defaults of `write_bundle` (which the backbone export writes into the bundle manifest without looking at the weights). The extractor only checks that the two copies agree, so any backbone with other statistics (a timm ResNet50V2 uses 0.5 / 0.5 [11]) makes training fail until someone edits the config by hand. B8 is the last open block of #166; B5 is merged and B6 is in review.

## What Changes

- **The backbone bundle is the only source of the normalization.** The export takes mean, std and input size from the metadata torchvision ships with the selected weights, and `write_bundle` requires them (no defaults).
- **B8 runs in the CNN extractor.** The extractor replicates the gray CNN image to 3 channels and normalizes it with its bundle's mean and std. `prepare_image` stops at the B6 output: a `(S, S)` image in [0, 1].
- **BREAKING (preprocessing settings, artifact manifest):** `PreprocessingConfig.cnn_mean` and `cnn_std` are removed, so artifacts trained before this change are refused by the strict manifest check. `PreprocessedImage.cnn_input` (3 channels, normalized) becomes `cnn_image` (one channel, unnormalized). No artifact has been published (#114 pending).
- **The feature cache key covers the normalization.** It includes the bundle's mean, std and input size besides the weights hash, so changing the normalization never reuses stale embeddings.
- **Invalid bundle normalization is refused.** A bundle manifest whose mean or std does not have 3 values, or whose std is not positive, cannot be loaded.
- **Docs.** B8 and the traceability row of the pipeline doc (and its `.tex`), and a new decision entry in `docs/ml-engine/decisions.md`.

## Capabilities

### New Capabilities
- `cnn-tensor`: how the CNN image becomes the tensor the frozen backbone receives (3 channels, normalization from the backbone bundle), where those values come from, and their identity at training and inference.

### Modified Capabilities
- None. The `image-geometry` capability (unarchived change `image-geometry`) defines the CNN image this change consumes; its requirements do not change.

## Impact

- **Code** (`app/ml_engine/src/`): `deep_learning/bundle.py` (required normalization, validation), `deep_learning/export_backbone.py` (reads it from the weights), `deep_learning/extractor.py` (B8 tensor, size check only), `preprocessing/prepare.py` (no normalization, `cnn_image`), `training/features.py` (cache key), `pipelines/trained.py` (call site).
- **Tests**: `test_bundle.py`, `test_deep_extractor.py`, `test_prepare.py`, `test_training_features.py`, `test_pipeline_artifact.py`, `test_training_run.py`, plus every test that writes a bundle (`conftest.py`, `test_training_artifact.py`, `test_training_integration.py`).
- **Artifacts / cache**: retrain after merge; old feature caches are not reused. The local `weights/` bundle must be re-exported.
- **Backend / frontend**: no change.
- **Dependencies**: none new.
- **Depends on**: change `image-geometry` (PR #93) being merged first, since both edit `prepare_image`.
- **Out of scope**: adding a timm backbone, CNN fine-tuning, Grad-CAM, taking `cnn_size` from the bundle (the size check stays as it is).
