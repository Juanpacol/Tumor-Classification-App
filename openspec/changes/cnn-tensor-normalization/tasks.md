# Tasks

Run tests from `app/ml_engine` with `pytest src/ -v` (the `ml-engine-test` Docker image works). Work on a branch `AB-166-cnn-tensor` cut from `develop` after PR #93 (B6) is merged; commits use `AB-166:`.

## 1. Bundle as the source of the normalization (design D3, D4)

- [x] 1.1 Make `input_size`, `mean` and `std` required keyword arguments of `write_bundle`, and add one test helper in `src/tests/` that writes a bundle with the ImageNet values; switch `conftest.py`, `test_bundle.py`, `test_deep_extractor.py`, `test_training_artifact.py` and `test_training_integration.py` to it. Verify `grep -rn "0.485" app/ml_engine/src --include=*.py` only shows that helper and tests that set values on purpose
- [x] 1.2 Have `export_backbone` pass `mean`, `std` and `crop_size[0]` from `weights.transforms()`. Verify with a test that monkeypatches the torchvision weights to an object whose `transforms()` returns 0.5 / 0.5 / 192 and checks the written manifest holds them
- [x] 1.3 In `read_bundle`, refuse with `ModelUnavailableError` a mean or std that is not 3 finite numbers, a std `<= 0`, or an `input_size` that is not a positive integer. Verify with `test_bundle.py` cases for a zero std, a one-value mean and a zero input size

## 2. B8 in the extractor (design D1, D2)

- [x] 2.1 Add `DeepFeatureExtractor.to_tensor(image)` (repeat to 3 channels, normalize with the manifest mean and std, float32) and make `extract` / `extract_batch` take `(S, S)` / `(N, S, S)` CNN images, checking `S == input_size`. Verify in `test_deep_extractor.py`: a bundle with 0.5 / 0.25 turns a pixel of 1.0 into 2.0 on every channel; a bundle with other statistics changes the embedding of the same image; wrong shapes still raise `ValueError`
- [x] 2.2 Remove `cnn_mean`, `cnn_std` and `_to_cnn_input` from `preprocessing/prepare.py`; rename `PreprocessedImage.cnn_input` to `cnn_image` (`(S, S)` float32 in [0, 1]) and update the docstrings. Reduce `_check_preprocessing` to the size check. Update `training/features.py` and `pipelines/trained.py`. Verify `grep -rn "cnn_input\|cnn_mean\|cnn_std" app/ml_engine/src --include=*.py` returns only the artifact refusal test of 3.1, and `test_prepare.py` checks the `cnn_image` shape, dtype and [0, 1] range instead of normalized channels
- [x] 2.3 Keep the training/inference parity: extend the `test_prepare.py` parity case or add one in `test_inference_pipeline.py` so the same file bytes through `load_mri_image` + extractor and through `decode_mri_image` + extractor give identical embeddings. Verify it passes

## 3. Artifacts and cache (design D2, D5)

- [x] 3.1 Replace the `cnn_mean` case in `test_pipeline_artifact.py` with: a manifest whose preprocessing has `cnn_mean` is refused naming it as unexpected. Update `test_training_run.py` if it asserts the old keys. Verify `pytest src/tests/test_pipeline_artifact.py src/tests/test_training_run.py src/tests/test_inference_pipeline.py -v` passes
- [x] 3.2 Make `config_fingerprint` hash the backbone manifest's `sha256`, `input_size`, `mean` and `std`, passed by `make_featurizer` from `extractor.manifest`. Verify in `test_training_features.py` that two manifests differing only in `mean` give different fingerprints

## 4. Docs

- [x] 4.1 Add decision D20 (AB-166) to `docs/ml-engine/decisions.md`: B8 in the extractor, bundle as the single source filled from `weights.transforms()` (values checked for the three registered backbones on torchvision 0.29.2), validation rules, cache key, removed settings and the artifact refusal, and that a timm exporter would read `pretrained_cfg`. Verify the entry has each item
- [x] 4.2 Update `docs/ml-engine/preprocessing-pipeline.md` and `.tex` together: the B8 block (runs in the extractor, values from the bundle), section 3.2 inference diagram if it names the module, and the B8 traceability row (`deep_learning/extractor.py`, `deep_learning/bundle.py`, Done). Update `app/ml_engine/README.md` where it describes `cnn_input` or the CNN normalization. Verify `grep -rn "cnn_input\|Mostly done" docs/ml-engine app/ml_engine/README.md` returns nothing

## 5. Integration

- [x] 5.1 Run `pytest src/ -v` in `app/ml_engine` and `python -m pytest tests/ -v` in `app/backend` (with ml_engine installed from the working copy). Verify both pass with no new skip

## Workflow follow-up

- Archive `quality-gate-metrics`, `brain-mask` and `image-geometry` before this change.
- Close Azure Boards #166 by hand after this change is merged (B5, B6 and B8 done).
- Re-export the backbone bundle and retrain the artifact (#114).
