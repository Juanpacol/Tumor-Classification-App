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
- **Revisit:** rename to a real package (`ml_engine`) before the backend installs it as a dependency (#117, #120).

## D3. Dependencies: ranges in `pyproject.toml`, exact pins in `requirements.lock` (AB-107)

- **Decision:** ranges for the installable package, lock file for reproducible installs. Regenerate with `uv pip compile pyproject.toml --extra dev --python-version 3.11 -o requirements.lock`.
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
