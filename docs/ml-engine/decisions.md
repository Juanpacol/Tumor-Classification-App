# ML Engine Decision Log

Why `ml_engine` is built the way it is. Add an entry whenever a choice is not obvious from the code. Each entry says what was decided, the evidence, and what is still unverified.

Status: **Verified** (checked against source, docs or a run), **Assumption** (our choice, no external proof), **Open** (needs a check, see the command).

## D1. Inference contract is a frozen dataclass (AB-107)

- **Decision:** `PredictionResult` and its parts are `@dataclass(frozen=True)` and validate their invariants in `__post_init__` (4 classes, probabilities sum to 1 ± 1e-6, confidence is the max, label is the argmax).
- **Why:** an invalid result cannot be constructed. The backend domain already uses dataclasses, so `ml_engine` stays dependency-free and independent of the backend. Pydantic belongs at the HTTP boundary, where input is untrusted.
- **Evidence:** Pydantic docs and a comparison article agree that frozen stdlib dataclasses suit already-validated internal outputs. Status: **Assumption** (general guidance, not measured here).
- **Deferred:** `schema_version` and `to_dict()`. Add them when persistence (#116) needs to store a result.

## D2. Package is still named `src` (AB-107)

- **Decision:** keep `from src...` imports for now.
- **Known problem:** setuptools treats `src/` as a src-layout, so `pip install -e` exposes `preprocessing`, `pipelines`, ... as top-level packages, not `src.*`. `from src.pipelines import ...` only works when run from `app/ml_engine`, which is how CI runs it. Status: **Verified** (stale `top_level.txt` and `.pth` in the old venv, setuptools discovery docs).
- **Resolved for installs (AB-159, 2026-10-29):** `[tool.setuptools.packages.find]` with `include = ["src*"]`, `exclude = ["src.tests*"]` and `namespaces = false` makes `import src.pipelines` work from any directory after an editable and after a regular install, without packaging the tests or the dataset. The `ml-validation` job now checks both from outside `app/ml_engine`. Status: **Verified** locally with `uv`, the CI check uses `pip`.
- **Still open:** the top-level name `src` is generic and could clash with another project's `src` in the same environment. Rename to a real package (`ml_engine`) if that happens.

## D3. Dependencies: ranges in `pyproject.toml`, exact pins in `requirements.lock` (AB-107)

- **Decision:** ranges for the installable package, lock file for reproducible installs. Regenerate it with the command in `app/ml_engine/README.md` (D10 explains why it is universal and CPU-only).
- **Why:** `==` pins in a library cause resolver conflicts when the backend installs it. Status: **Assumption** (general packaging practice, the search did not cover it).
- **Checked:** `pip-audit -r requirements.lock` reported no known vulnerabilities on 2026-10-28. Status: **Verified**. Re-run periodically.

## D4. Seeds and determinism (AB-107)

- **Decision:** `set_seed` seeds Python, NumPy and PyTorch, enables `torch.use_deterministic_algorithms(True, warn_only=True)` and deterministic cuDNN.
- **Evidence:** MONAI's `set_determinism` does the same (seeds plus backend flags). Status: **Verified** (MONAI paper). Bit-exact GPU results are not guaranteed, so pin the environment too.

## D5. MRI loader (AB-108)

`src/preprocessing/loading.py`, `load_mri_image`.

| Choice | Value | Status |
|---|---|---|
| Output | 2D `float32`, shape `(H, W)`, scaled to [0, 1] | Assumption. Downstream steps assume this range. |
| Format detection | magic bytes (PNG, JPEG), plus `Image.open(..., formats=["PNG", "JPEG"])` | Verified (Pillow security handbook recommends a format allowlist) |
| Single read | file read once, capped at `MAX_FILE_BYTES` (50 MB), decoded from memory | Assumption |
| Size checks before decode | `MIN_SIDE` 64, `MAX_PIXELS` 25,000,000, checked from the header before `load()` | Verified that Pillow's own limit is 89,478,485 px (Pillow 13.0.0, run locally). The 25 MP and 50 MB values are ours. |
| Corruption | full `load()` inside `try`; `Image.verify()` not used | Verified: its docstring says it does not decode and the image must be reopened. |
| EXIF orientation | `ImageOps.exif_transpose` applied | Assumption (blog-level sources). Same function runs for training and inference. |
| Gray conversion | reuses `to_gray_uint8` (16-bit is rescaled, not clipped) | Verified: MONAI's `PILReader` does no grayscale conversion itself, it leaves that to a user `converter`. |

## D6. Model bundle loading and pipeline structure (for AB-113, AB-114, AB-117)

Not implemented yet. Ideas taken from reading the source of MONAI Deploy App SDK (commit `3aee3da`, read only, nothing executed).

- **Three internal stages.** `InferenceOperator` splits inference into `pre_process`, `predict` and `post_process`, each raising `NotImplementedError` until overridden. The real pipeline (#113) should keep the same stages internally while the public protocol stays a single `predict`. Status: **Verified** (source).
- **Unavailable model is a normal state.** MONAI's `Model("")` is a "null model" that evaluates false. #117 needs the same behavior: a missing or invalid bundle must not stop the app, `/health` reports it and uploads return 503. Status: **Verified** (source), the mapping to #117 is our **Assumption**.
- **Bundle layout is validated file by file.** MONAI expects `configs/metadata.json` and `models/model.pt` or `model.ts`, and raises an error naming the exact missing path. Our bundle loader (#114) should do the same, with a manifest next to the weights (model version, classes, expected input, training data hash, metrics) so the backend can check compatibility before loading. Status: **Verified** that MONAI validates this way, the manifest fields are our **Assumption**.
- **Path safety.** MONAI rejects config names containing `..`, `/` or `\` before reading them. Never build file paths from names that come from a manifest or from users without the same check. Status: **Verified** (source).
- **Not adopted:** the operator DAG, the executor and container packaging (MAP). They are more than this project needs. MONAI Deploy also requires Holoscan and CUDA 12.
- **Not verified:** the MAP manifest format (not read) and the MONAI Bundle specification itself (only seen through this SDK's loader).

## D7. Quality gate, skull stripping and model inputs (AB-109)

`src/preprocessing/{quality,skull_stripping,prepare,calibrate}.py`.

- **Quality gate is always on.** `prepare_image` measures SNR and Shannon entropy on the original image (before any stripping, so stripping cannot hide a bad scan) and raises `QualityCheckError` with the measured values below the thresholds.
- **SNR definition (ours):** mean intensity above the Otsu threshold divided by a wavelet noise estimate (`skimage.restoration.estimate_sigma`). It needs no noise-only background, which cropped MRI slices do not have. A flat image returns 0. Status: **Assumption**, not compared against other SNR estimators.
- **Entropy:** Shannon entropy in bits of a 256-bin histogram of the [0, 1] image.
- **Skull stripping is off by default.** Otsu threshold, opening, erosion to cut the scalp links, largest component, dilation and hole filling, CPU only. If the mask covers under 15% or over 95% of the image, or the image is flat, the input is returned unchanged and `skull_stripped=false`. It never raises. Off by default because our literature review rates the evidence as weak and contradictory (`docs/research/preprocessing-2d-mri-cnn-ml.md`, 3/10) and recommends treating it as an ablation variable. Status: **Verified** that the review says so. The algorithm has only been tested on a synthetic phantom.
- **Model inputs:** `radiomics_image` is (H, W) float32 in [0, 1]. `cnn_input` is a NumPy (3, S, S) float32 array, resized (S = 224 by default) and normalized with ImageNet mean and std. Both are configurable in `PreprocessingConfig`, because the real backbone is chosen in AB-111. Status: **Assumption**. Convert with `torch.from_numpy` in the consumer, so preprocessing does not depend on PyTorch.
- **Thresholds are provisional.** `min_snr = 4.0` and `min_entropy = 2.0` are our guesses (uniform noise measures about 2.6 SNR, and a synthetic slice far above). They have **not** been calibrated on the training set because the Kaggle dataset is not available in the repo. Wrong values can reject good images or let bad ones through. Status: **Open**.

## D8. Radiomic feature extractors (AB-110)

`src/classical/{config,roi,glcm,lbp,wavelet,intensity,extractor}.py`. The combined vector has 45 features in a fixed order: GLCM 5, LBP 10, wavelet 21, intensity 9. Names come from `feature_names(config)` and do not depend on the input.

Own implementation on scikit-image, PyWavelets and SciPy, following the conventions of PyRadiomics and IBSI where cheap. PyRadiomics itself is not used: its build compiles a C extension with `scikit-build-core` and needs `numpy>=2.0` at build time (read from its `pyproject.toml`). Status: **Verified** for the build requirement, the extra friction in CI and Cloud Run is our **Assumption**.

| Choice | Value | Status |
|---|---|---|
| Mask handling in GLCM | ROI quantized to levels 1..N, level 0 (background) dropped from the matrix, so only pairs with both pixels in the mask count | **Verified** equivalent to PyRadiomics, which passes the mask to its C routine and drops gray levels absent from the ROI |
| GLCM setup | distance 1, symmetric, 4 angles, features computed per angle then averaged, empty angles ignored | **Verified** against PyRadiomics defaults. This is IBSI's 2D:avg aggregation, which the IBSI manual says can differ from merging the matrices |
| Discretization | 32 fixed bins over the ROI min-max | **Assumption.** PyRadiomics defaults to a fixed bin width of 25, and its FAQ says it differs from IBSI here. We recall IBSI recommending a fixed bin number for arbitrary-unit MRI but did not confirm it |
| Kurtosis | excess kurtosis (`scipy.stats.kurtosis`) | **Verified** that PyRadiomics returns Pearson kurtosis (+3) while IBSI uses excess |
| GLCM `energy` | scikit-image definition (square root of the angular second moment) | PyRadiomics joint energy is the sum of squares, so values are not comparable with it |
| Flat or tiny region | zeros, never NaN or inf; flat is detected by value range because the std of a constant float array is not exactly 0 | Found by a unit test |
| LBP | uniform patterns, 8 points, radius 1, on the 8-bit image; pixels within `radius` of the bounding-box edge are not counted | Found by a unit test: edge pixels were compared with padding |
| Wavelet | Haar, level 2, run on the bounding box with the outside filled by the ROI mean, statistics only where the mask (downsampled per sub-band) is inside | **Assumption**, modeled on how PyRadiomics applies filters before measuring inside the ROI |
| Region of interest | the brain mask when skull stripping was applied, otherwise the foreground (half the Otsu threshold, largest component, holes filled) | **Assumption** |

- **Not validated against IBSI.** Training and inference use the same code, so accuracy does not depend on it. It matters only to compare numbers with the literature.
- **Settings live in `RadiomicsConfig`** and change the vector length and names. Store them with any model trained on these features (AB-114 manifest).
- **Speed:** 8 ms median for one 512x512 image with a half-image mask on this machine (10 runs, CPU), under the 1 s requirement. A unit test enforces the bound. Shared CI runners are slower, so the margin matters.
- **`z_score` of `top_features`** needs training means and stds. That is AB-112.

## D9. Frozen CNN feature extractor and backbone bundle (AB-111)

`src/deep_learning/{backbones,bundle,extractor,export_backbone}.py`.

- **Backbones:** `efficientnet_b0` (default, 1280-d, as the story asks), `resnet50` (2048-d) and `densenet121` (1024-d). The last two are the baselines recommended by our literature review, kept available for ablations. A test checks each declared embedding size against the real output.
- **Weights are never downloaded at runtime.** The architecture is built with `weights=None` and the `state_dict` comes from a bundle: `backbone.pt` plus `backbone.json` (backbone, embedding size, input size, mean, std, sha256, torchvision version, weights version). `export_backbone` is the only code that touches the network, run once on a development machine. File names are fixed, so no path is built from manifest content.
- **Safe loading.** The sha256 is verified before deserializing, and `torch.load(..., weights_only=True)` refuses arbitrary pickled objects (a test saves one with a matching hash and checks it is refused). `ModelUnavailableError` covers a missing, corrupt, tampered or non-fitting bundle, so the backend (AB-117) can map it to 503 `MODEL_UNAVAILABLE`. Status: **Verified** by tests. We did not check what the default of `weights_only` resolves to in torch 2.15 (the signature shows `None`), so it is passed explicitly.
- **Preprocessing must match the backbone.** `from_bundle` compares the manifest with `PreprocessingConfig` (input size, mean, std) and refuses to load on a mismatch, so changing the normalization cannot silently feed the model different inputs than it was exported with.
- **Inference:** `eval()`, parameters frozen, `torch.inference_mode()`. The library does not set the global thread count. Same input gives the same output in a process; the last decimals can differ across CPUs, so never compare against stored values.
- **Measured with real ImageNet weights** (torchvision `IMAGENET1K_V1`, EfficientNet-B0, 4 CPU threads, this machine): downloaded file sha256 `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934` (matches the prefix in the file name), reload with `strict=True` matches every key, 87 ms median per image and about 64 ms per image in a batch of 32. Status: **Verified** here, not on Cloud Run.
- **Grad-CAM (later) cannot reuse this path:** `inference_mode` removes the gradients it needs.

Open items:
- [ ] **Terms of use of the ImageNet weights for commercial use.** Not checked. Review before publishing the bundle in AB-114.
- [ ] **Timing on Cloud Run** (AB-120) and **embedding quality on the real MRI dataset** (AB-112, compare the three backbones).

## D10. Universal, CPU-only lock file (AB-154)

- **Problem:** the first `requirements.lock` was resolved on macOS only (no platform markers). A universal resolve showed that on Linux PyPI `torch` depends on 15 `nvidia-*` packages plus `triton`, which would bloat the container image. Status: **Verified** (resolved with `uv pip compile --universal` on 2026-10-29).
- **Decision:** one universal lock using the CPU-only PyTorch index with `--index-strategy first-index` and `--emit-index-url`, so the lock carries its own indexes and `pip install -r requirements.lock` needs no extra flags. macOS keeps `torch==2.15.0`, other platforms get `torch==2.15.0+cpu`. The result has no `nvidia-*` or `triton`.
- **Verified in Linux containers:** `uv pip install --dry-run` of the lock inside `python:3.11-slim` resolves 42 packages with `torch==2.15.0+cpu` and `torchvision==0.29.2+cpu` on both `linux/aarch64` and `linux/amd64`. Only a dry run: the wheels were not downloaded or imported on Linux.
- **Not changed:** CI still installs with `pip install -e ".[dev]"`, which pulls PyPI `torch`, so the CPU lock does not shorten CI yet. The backend `Dockerfile` does not install `ml_engine` yet (AB-120).

## D11. Training data, feature cache and classifier selection, part 1 (AB-112)

`src/training/{dataset,features,model}.py`. Part 2 will add calibration, the report with the ablation, the bundle and the CLI.

- **Dataset size.** After the cleaning of AB-150 the kept images are 3,434 (Train 2,325, Test 1,109), and Train is unbalanced: glioma 992, meningioma 539, notumor 275, pituitary 519 (`DATASET_CLEANING_REPORT.md`). Status: **Verified** from that report. An earlier estimate in this project used 6,878 images, which was the count of exact-unique images before the rest of the cleaning, so the feature extraction should take about 5 minutes (3,434 images at roughly 90 ms, **not measured** on the real set).
- **Classes are balanced by weight, models are compared by macro-F1.** All three candidates use `class_weight="balanced"`. Status: **Assumption**, to confirm with the per-class report.
- **Cross-validation is grouped.** `StratifiedGroupKFold` (5 folds, shuffled, fixed seed) with the `group_id` of the cleaning manifest, so near-duplicate images never straddle two folds. The story asks for a plain stratified 5-fold, this is a deliberate deviation. **Limitation:** the dataset has no patient id, so the groups are only a proxy. Duplicates of one scan that the cleaning did not link can still leak, and the metrics can be optimistic.
- **Everything fitted lives inside the sklearn pipeline.** Scaler and PCA (32 components) on the radiomic block, scaler on the CNN block, then the classifier, so each fold refits them (a test checks the scaler statistics come from the training rows only). PCA needs at least 32 training rows per fold, which the real data satisfies and tiny datasets do not.
- **Candidates:** logistic regression, SVM with RBF kernel and random forest. XGBoost is not a dependency, and the story accepts a random forest instead.
- **Feature cache.** One `.npz` per image, keyed by the file sha256 and a fingerprint of the preprocessing config, the radiomics config and the backbone sha256, so a changed config, backbone or image recomputes only what is stale. Images the engine rejects are cached with their reason and reported, never silently dropped. The quality thresholds are still provisional (D7), so the count of rejected images on the real dataset must be read before trusting any metric.
- **Status:** validated only on synthetic features and on a synthetic image dataset. The classifier selection on them says nothing about the real data.

## D12. Calibration, test evaluation, artifact bundle and training CLI, part 2 (AB-112)

`src/training/{calibration,evaluation,ablation,stats,artifact,report,run,__main__}.py`. One command runs the whole pipeline: `python -m src.training --data-dir src/data --manifest <manifest.csv>`.

- **Order of a run:** features (cached), classifier selection by cross-validation on Train, ablation, calibration, one evaluation on Test, then artifact and report. The Test split is scored once, by the calibrated model, and a test enforces that `evaluate` is called a single time.
- **Calibration uses the same group-aware folds as the selection** (`CalibratedClassifierCV` with those splits, sigmoid method), so near-duplicate images are never on both sides of a split. The saved classifier is therefore an ensemble with one fitted pipeline per fold, not a single model refitted on all of Train. Sigmoid was chosen because isotonic needs more samples per class, which is an **Assumption**: the 275 'notumor' Train images may or may not be enough, compare both once the real data is available.
- **The ablation uses cross-validated macro-F1 on Train**, not the Test split, because scoring Test three times would break the rule that Test is used once. The report therefore compares CNN-only, radiomics-only and hybrid on the same footing as the classifier selection.
- **Artifact:** `classifier.joblib`, the backbone files copied from a verified bundle (D9), `feature_stats.json` (mean and population std of every radiomic feature, for the z-scores of `top_features`) and `manifest.json`. The manifest records version, date, seed, class names, selected classifier, calibration, the full preprocessing and radiomics configs, the quality thresholds, the feature names, library versions, the cross-validated and test metrics, the dataset counts and the sha256 of every file. The configs are stored so that serving (AB-113) can refuse to run with parameters different from the ones used to train, as D9 does for the backbone.
- **A `.joblib` file is a pickle.** Loading an untrusted one can run code. The manifest has its sha256, so serving must verify it before loading. This only helps if the manifest itself is trusted, which the checksum of the published archive (AB-114) has to guarantee. Status: **Verified** that the hash is written, the check on load is AB-113.
- **Target:** the report states whether the test macro-F1 reaches 0.90 and the CLI prints a warning when it does not, to present the results to the team before continuing, as the story asks.
- **Reproducibility:** the same seed gives the same selection, ablation and test metrics (a test runs the pipeline twice). `joblib` is now a direct dependency because the artifact imports it.
- **Status:** the CLI was run end to end only on a synthetic dataset of four texture classes (160 images, about 19 s, real ImageNet weights, test macro-F1 0.987). That number says nothing about the real data, because those classes are trivially separable.
- **Still needed for AB-112:** a run on the real dataset and its report. The story stays open until then.

## D13. Serving pipeline (AB-113)

`src/pipelines/{artifact,top_features,trained}.py`. `BundleInferencePipeline.from_bundle(folder, triage_threshold=0.75)` loads a training artifact once and `predict(file_bytes)` returns a `PredictionResult`.

- **Name.** The story calls it `InferencePipeline.from_bundle`. `InferencePipeline` stays the protocol, because CI, `CLAUDE.md` and the fake already use that name, and the concrete class is `BundleInferencePipeline`.
- **The contract takes bytes.** The protocol and `FakeInferencePipeline` took a file path while the story asks for `predict(file_bytes)`. Both now take bytes, so the fake and the real pipeline are interchangeable, and `decode_mri_image` decodes from memory with the same size and format checks as the file loader. Nothing outside `ml_engine` used the old signature.
- **Triage threshold 0.75**, as the story asks (the fake used 0.7, now both use the shared constant). It has not been calibrated on real data. Status: **Assumption**.
- **Everything is verified before it is loaded.** The sha256 of each file in the manifest is checked before the pickled classifier or the weights are deserialized, then the manifest classes and radiomic feature names are compared with the code, and the backbone loads through the checks of D9. Any failure is a `ModelUnavailableError`, so the backend can fail fast at start-up and answer 503 `MODEL_UNAVAILABLE`.
- **Serving rebuilds the preprocessing and radiomics settings from the manifest**, not from the code defaults. A test checks the settings survive the JSON round trip, and another that an edited manifest changes them. If the manifest normalization differs from the backbone it is refused.
- **Limit of the integrity check.** The hashes live in the manifest, so they only prove the files match the manifest. They do not prove the manifest is genuine. The checksum of the published archive (AB-114) has to cover that. Loading a `.joblib` stays a pickle load (D12).
- **`top_features`:** the 5 raw radiomic features with the largest |z-score| against the training mean and std. The z-score uses the radiomic features before PCA, because the PCA components have no name. A feature that never varied in training has z-score 0, and ties are ordered by name.
- **No train/serve skew, checked on 5 images:** the probabilities from `predict(bytes)` equal, within 1e-6, those of the classifier fed with the features of the training code path. Status: **Verified** on synthetic images, in one process on one machine.
- **Latency:** 99 ms median per image after loading, 0.07 s to load, measured with a 128x128 synthetic PNG, random backbone weights and a classifier trained on synthetic data. The 3 s limit is enforced by a test. A real 512x512 MRI was not measured end to end, only its parts (about 90 ms for the CNN, 8 ms for the radiomics). Status: **Verified** for the synthetic case.
- **Import weight.** `import src.pipelines` now imports PyTorch and scikit-learn. The backend installs `ml_engine` only in AB-120, so nothing pays for it yet.

## D14. Provenance of third-party assets (AB-156)

The repository is public for the hackathon evaluation, and the artifact published in AB-114 bundles the ImageNet-pretrained backbone weights.

- **Decision:** publish the weights inside the artifact (the simplest option) and make the reservations explicit instead of hiding them. The weights stay out of git (`weights/*.pt` is ignored), and only a Release would make them public.
- **Why it is safe to decide this now:** the backbone is a replaceable component (registry, bundle, manifest, one training command). If cleaner weights are needed, swap them and retrain, with no change to the rest of the pipeline.
- **What was added:** the backbone manifest records `source_url` and `license_note`, with defaults so bundles exported earlier still load, and the artifact manifest carries them as `backbone_provenance`. The README lists every external asset and its status.
- **What is not known:** the terms of the torchvision weights (no license statement was found for them, only the BSD license of the code), whether weights trained on ImageNet inherit its non-commercial terms, and the licenses of the Kaggle dataset and of the Python dependencies. This is not legal advice.
- **Alternative kept for later:** publish the artifact without `backbone.pt` and download the weights from torchvision when the image is built. It avoids redistributing them but not the question of the terms, and it needs the manifest to record the hash of the original file, because re-saving the weights changes the file hash.
- **Open items:**
  - [ ] Ask the hackathon organizers how third-party models and data must be declared.
  - [ ] Check the license of the Kaggle dataset on its page. A secondary source lists CC0, unconfirmed, and the dataset combines figshare, SARTAJ and Br35H, so check those terms too.
  - [ ] Get a legal review of the weights before any commercial use.
  - [x] The repository had no `LICENSE` file, which matters because it is public and the landing page presents it as open source. The team chose the MIT License (copyright holder "Ether contributors", change it if the team prefers named holders). It covers the code, not the third-party assets above.

## GMLP principles checklist

The ten FDA, Health Canada and MHRA Good Machine Learning Practice guiding principles, with where the project stands. This is a design aid, not a compliance claim: the project is not seeking regulatory clearance and must not be presented as compliant.

| # | Principle | Status |
|---|---|---|
| 1 | Multidisciplinary expertise across the product life cycle | Not covered, no clinical review so far |
| 2 | Good software engineering and security practices | Partial: typed contract, input limits in the loader, lock file, `pip-audit` run |
| 3 | Data and study participants represent the intended population | **Limitation:** the Kaggle dataset has no demographic information |
| 4 | Training data independent of test data | Covered by deduplication and leak removal (AB-150, see `DATASET_CLEANING_REPORT.md`) |
| 5 | Reference datasets built with best available methods | Open: labels come from a public dataset, their provenance is not audited |
| 6 | Model design fits the data and the intended use | In progress (hybrid radiomics and CNN) |
| 7 | Performance of the human-AI team | By design: `needs_expert_triage`, uncertainty alert (AB-127), Grad-CAM |
| 8 | Testing under clinically relevant conditions, including subgroups | **Limitation:** no subgroup metadata, so no subgroup evaluation |
| 9 | Users get clear, essential information | Partial: landing safety rubric exists, a model card is missing |
| 10 | Deployed models are monitored | Out of scope for the hackathon |

## Open verifications

- [ ] **Calibrate the quality thresholds (D7).** Needs the dataset in `app/ml_engine/src/data` (`Train/` and `Test/`). Run from `app/ml_engine`:
  ```bash
  python -m src.preprocessing.calibrate --data-dir src/data
  ```
  It prints the 1st percentile of SNR and entropy, so at least 99% of the training images pass. Put the result in `QualityThresholds` and record the numbers and the date here.
- [ ] **Skull stripping on real images.** Run it on real slices and look at the masks before any ablation. Only the synthetic phantom is tested so far.
- [ ] **Limits vs the real dataset.** Not possible yet: `src/data` is not in the repo. Once the Kaggle dataset is downloaded, run from `app/ml_engine`:
  ```bash
  python -c "
  import pathlib
  from PIL import Image
  files = [p for p in pathlib.Path('src/data').rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
  print('max bytes', max(p.stat().st_size for p in files))
  print('max pixels', max(Image.open(p).size[0] * Image.open(p).size[1] for p in files))
  "
  ```
  If either value is close to `MAX_FILE_BYTES` or `MAX_PIXELS`, raise the limit.
- [ ] **Backend upload cap.** The 50 MB cap in the loader is defense in depth. Enforce an upload size limit in the upload endpoint (#118) before the file is stored.
- [ ] **EXIF behavior on real uploads.** Confirm with a phone photo of a screen that orientation comes out right.
- [ ] **Package rename** (see D2).

## Sources

- GMLP guiding principles (UK government copy): https://www.gov.uk/government/publications/good-machine-learning-practice-for-medical-device-development-guiding-principles/good-machine-learning-practice-for-medical-device-development-guiding-principles
- MONAI Deploy App SDK: https://github.com/Project-MONAI/monai-deploy-app-sdk
- Pillow security handbook: https://pillow.readthedocs.io/en/latest/handbook/security.html
- Pydantic dataclasses: https://docs.pydantic.dev/2.5/concepts/dataclasses
- setuptools package discovery: https://setuptools.pypa.io/en/latest/userguide/package_discovery.html
- MONAI paper (determinism): https://arxiv.org/pdf/2211.02701
