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

## Open verifications

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

- Pillow security handbook: https://pillow.readthedocs.io/en/latest/handbook/security.html
- Pydantic dataclasses: https://docs.pydantic.dev/2.5/concepts/dataclasses
- setuptools package discovery: https://setuptools.pypa.io/en/latest/userguide/package_discovery.html
- MONAI paper (determinism): https://arxiv.org/pdf/2211.02701
