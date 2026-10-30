# Design

## Context

- `src/training/dataset.py` has `load_samples` (kept manifest rows → `Sample(path, split, label, group_id)`) and `split_samples`, which today only partitions by the `split` column. `training/run.py` calls both in `_feature_tables`. Grouped CV (`StratifiedGroupKFold` on `group_id`) already happens inside Training, so the only cross-split leak is between Training and Testing.
- `src/preprocessing/calibrate.py` exposes `calibrate(paths, percentile)` and a CLI that gets its paths from `discover_images(data_dir)`, which lists `Train/` **and** `Test/` and ignores the manifest.
- `src/training/calibration.py` already exists and means classifier probability calibration, so the threshold module needs a different name.
- The dataset and the manifest `processed/dedup_20261028T234450903626Z/manifest.csv` are versioned in git under `app/ml_engine/src/data`.
- On that manifest, purging by the groups of *kept* Testing images removes 495 images (the #165 reference). Purging by *all* Testing rows would remove 504; the extra 9 are Training images whose only Testing group member was dropped by #150 and is never scored.

## Goals / Non-Goals

**Goals:**
- One function that defines the leakage-free split, used by both the training run and threshold calibration.
- Keep `preprocessing/` limited to steps that do not learn from data.

**Non-Goals:**
- Changing the deduplication (#150) grouping, or making the split patient-level.
- Re-splitting Testing, or moving images between splits.
- Fixing `ml-validation.yml` so it selects and enforces ml_engine tests (separate ticket).
- Rebalancing classes after the purge (class weights already exist, D12).

## Decisions

### D1. `split_samples` becomes the leakage-free split, with no second function
`split_samples(samples) -> Split`, where `Split` is a frozen dataclass with `train`, `test` and `removed_per_class: dict[TumorLabel, int]` (every label present, zeros included). The function collects the `group_id`s of the Test samples and drops Train samples in that set.

- *Why:* the plain folder split has no remaining legitimate caller. Keeping both would leave a leaky option in the codebase, against the project rule to remove what is not needed.
- *Alternative:* a new `leakage_free_split` next to the old one. Rejected for that reason.
- Input is `load_samples` output, which already contains only `keep = 1` rows. This gives the "kept Testing groups" definition (495) for free, with no extra flag.

### D2. Threshold calibration lives in `src/training/thresholds.py`
It keeps the pure `calibrate(paths, percentile) -> (QualityThresholds, used)` unchanged, and replaces the CLI: `--data-dir` and `--manifest` (both required), `--percentile` (default 1.0). The CLI paths are `[s.path for s in split_samples(load_samples(...)).train]`. `preprocessing/calibrate.py` is deleted.

- *Why:* calibration learns from data, so it belongs with the other fitted steps. Importing `training` from `preprocessing` would invert the dependency direction.
- *Alternative:* keep it in `preprocessing/` and import the split from `training/`. Rejected for the inverted dependency.
- The name is `thresholds.py` because `training/calibration.py` is already classifier calibration.
- `load_samples` checks with `is_file()` that every listed image, Testing included, exists. That is a filesystem check and does not open the image, so it satisfies "Testing images are never read".

### D3. Purge counts flow through the existing dataset summary
`_feature_tables` returns the `Split`. `_dataset_summary` adds `split_removed_per_class` (label value → count). That one dict already feeds the report and the artifact `manifest.json` `dataset` section, so no new plumbing is needed. The training CLI prints the counts.

### D4. Reference test reads the real versioned manifest
`src/tests/test_training_split_reference.py` resolves `src/data` relative to the test file, runs `load_samples` + `split_samples` and asserts 142 / 93 / 15 / 245, 1,830 and 1,109. It has no `skipif`, because the data is in git and a silent skip would hide the guarantee. The invariant test ("no group in both splits") uses synthetic `Sample`s with no files.

### D5. Calibrated values are recorded, not applied
Calibration ran once on the versioned dataset on 2026-10-30: `min_snr = 42.8567` and `min_entropy = 4.0914` over 1,830 images at the 1st percentile. With those values the gate rejects notumor 21/260 Train and 10/125 Test, against at most 1.8% for the other classes. A contact sheet of the 61 rejected images showed mostly clean diagnostic scans. The wavelet noise estimate reads fine FLAIR/T2 anatomy as noise. Some glioma copies reach an SNR above 250,000. The low-entropy rejections are dark slices with little brain. So the metrics do not measure quality on this dataset.

- *Decision:* the `QualityThresholds` defaults stay at 4.0 / 2.0. The docstring points to D15 instead of saying "not calibrated". D15 records the values, the per-class rejections and the finding. Redefining the metrics and recalibrating moves to AB-175.
- *Alternative:* apply the values as calibrated. Rejected because it would remove 8% of notumor non-randomly from training and reject valid scans at inference.
- The goal of #165 still holds: the calibration CLI never reads Testing, so the follow-up only has to change the metrics, not the data flow.

## Risks / Trade-offs

Every risk is closed: either accepted with its reason, or mitigated by a named task or ticket.

| # | Risk | Status | Resolution |
|---|---|---|---|
| R1 | Training shrinks to 1,830 images and pituitary to 274 | **Accepted** | `class_weight="balanced"` and stratified grouped folds (D12) already handle the imbalance. Per-class counts after the purge are reported (task 2.1), and the proposal states that metrics are not comparable with earlier runs. |
| R2 | Thresholds calibrated on purged Train reject some Testing images at the gate | **Triggered, resolved by D5** | Task 4.1 measured it: the rejections concentrate in notumor (8% in both splits) and are mostly usable scans. The values are recorded and not applied (D5). Fixing the metrics is tracked in AB-175. |
| R3 | New defaults invalidate the feature cache | **No longer applies** | The defaults do not change (D5), so the cache stays valid. |
| R4 | The reference test pins one manifest file | **Accepted** | Intended: regenerating the manifest must be a deliberate change that updates the test and `preprocessing-pipeline.md` together. |
| R5 | The CI neither selects nor enforces the new tests | **Mitigated by AB-174** | AB-174 (child of #103) makes `ml-validation.yml` run all ml_engine tests and fail on any failure. Until it lands, task 5.1 runs the suite locally. |

## Migration Plan

There are no deployed consumers. Docs that show the old `python -m src.preprocessing.calibrate` command are updated in the same change. Rollback is a git revert.
