# Design

## Context

Current code (`app/ml_engine/src/`), observed on 2026-10-31 on branch `AB-166-image-geometry` (B6, PR #93):

- `preprocessing/prepare.py`: `PreprocessingConfig` has `cnn_size = 224`, `cnn_mean = (0.485, 0.456, 0.406)`, `cnn_std = (0.229, 0.224, 0.225)`. `prepare_image` builds the B6 square, resizes it to `cnn_size`, and `_to_cnn_input` repeats it to 3 channels and normalizes it with the config values. `PreprocessedImage.cnn_input` is `(3, S, S)`.
- `deep_learning/bundle.py`: `BackboneManifest` holds `input_size`, `mean`, `std`. `write_bundle` defaults them to 224 and the ImageNet values. `read_bundle` converts `mean` / `std` to tuples and validates nothing else about them.
- `deep_learning/export_backbone.py`: loads torchvision ImageNet weights and calls `write_bundle` without passing `mean`, `std` or `input_size`, so the manifest gets the hard-coded defaults.
- `deep_learning/extractor.py`: `DeepFeatureExtractor.from_bundle(directory, preprocessing)` runs `_check_preprocessing`, which refuses a bundle whose `input_size` differs from `cnn_size` or whose mean/std differ from the config's. `extract` takes `(3, S, S)`.
- `training/features.py`: the feature cache fingerprint hashes `asdict(preprocessing)`, `asdict(radiomics)` and the weights' sha256 only.
- Callers of `cnn_input`: `training/features.py` and `pipelines/trained.py`, both passing it straight to `extractor.extract`. `deep_learning/explainability/` is empty (no Grad-CAM yet).
- The artifact folder holds a copy of the backbone bundle; serving builds the extractor from it, and `pipelines/artifact._strict` refuses manifests whose preprocessing keys differ from `PreprocessingConfig`.

Torchvision 0.29.2 (the locked version), checked in the `ml-engine-test` image: `weights.transforms()` gives `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`, `crop_size = [224]` for the three registered backbones (`efficientnet_b0`, `resnet50`, `densenet121`, all `IMAGENET1K_V1`).

## Goals / Non-Goals

**Goals:**
- One source for the normalization, the bundle manifest, filled from the weights' own metadata.
- No numeric mean/std literal left in the engine code outside tests.

**Non-Goals:**
- Taking `cnn_size` from the bundle. The geometry step needs the size before any extractor exists, and the existing size check already refuses a mismatch.
- Supporting backbones outside the torchvision registry.

## Decisions

### D1. B8 moves into the extractor

`prepare_image` returns `cnn_image`, the `(S, S)` float32 B6 output in [0, 1]. `DeepFeatureExtractor` gets a method `to_tensor(image) -> (3, S, S)` (repeat, subtract the manifest mean, divide by the manifest std), and `extract` / `extract_batch` take `(S, S)` / `(N, S, S)` CNN images and call it. The extractor already owns the bundle, so the statistics are read where they are used and no copy has to be kept in sync.

*Alternatives:* keep `cnn_mean` / `cnn_std` in `PreprocessingConfig` and have training overwrite them from the bundle (two copies remain, the defaults stay hard-coded, and every caller of `prepare_image` must remember to pass a bundle-derived config); pass the manifest into `prepare_image` (couples the torch-free preprocessing module to the bundle).

### D2. `PreprocessingConfig` loses `cnn_mean` and `cnn_std`

They are removed, so the artifact manifest no longer carries them and `_strict` refuses older artifacts naming `cnn_mean` and `cnn_std` as unexpected. `_check_preprocessing` keeps only the size check. The extractor's normalization error path disappears.

### D3. Export reads the statistics from the weights

`export_backbone` passes `list(transforms.mean)`, `list(transforms.std)` and `transforms.crop_size[0]` from `weights.transforms()`. `write_bundle` makes `input_size`, `mean` and `std` required keyword arguments. Test helpers that write bundles pass them explicitly through one shared helper in `tests/` so the ImageNet literal appears once in test code.

### D4. Bundle validation

`read_bundle` refuses a manifest whose `mean` or `std` is not a sequence of 3 finite numbers, or whose `std` has a value `<= 0`, and an `input_size` that is not a positive integer. These raise `ModelUnavailableError` inside the existing "invalid manifest" path.

### D5. Cache key covers the normalization

`config_fingerprint` takes the backbone manifest's `sha256`, `input_size`, `mean` and `std` instead of the sha256 alone (`make_featurizer` passes them from `extractor.manifest`). Because `cnn_mean` / `cnn_std` leave `PreprocessingConfig`, the fingerprint changes anyway, so old caches are not reused.

### D6. Order with B6

This change edits `prepare_image` right after the B6 change. It is implemented on a branch cut from `develop` after PR #93 is merged, to avoid a conflict in `prepare.py` and the `test_prepare.py` cases B6 added.

## Risks / Trade-offs

- [Embeddings change for nobody today, but the public shape of `PreprocessedImage` changes] → only two callers in the engine; the backend uses `BundleInferencePipeline.predict` and never sees it.
- [Future Grad-CAM needs the normalized tensor] → it gets it from `DeepFeatureExtractor.to_tensor`, which keeps the statistics with the model.
- [A future timm backbone has no `weights.transforms()`] → its exporter must read `model.pretrained_cfg` instead; out of scope, recorded in the decision entry.
- [Bundles exported before this change hold the hard-coded ImageNet values, which are the correct ones for the registered weights] → they have the same fields and pass the new validation, so they keep loading. The repository's `app/ml_engine/weights/` folder is empty today, so nothing needs re-exporting now.

## Migration Plan

1. Merge PR #93 (B6), then branch `AB-166-cnn-tensor` from `develop`.
2. Implement; retrain locally to rebuild the feature cache and the artifact (none published).
3. Rollback: revert the commits; artifacts trained after this change lack `cnn_mean` / `cnn_std` and would be refused by the old loader, so retrain.
