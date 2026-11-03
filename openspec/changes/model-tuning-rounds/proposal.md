# Proposal

## Why

Baseline round 0.1.0 (D24) reached CV macro-F1 0.898 on Training but only 0.807 on Testing, below the 0.90 target of #112, so no model can be published (#114) and the backend cannot classify in production (#120). Tuning in rounds is the next step (Azure #222), but today it cannot be done honestly or completely:

- **Every run scores Testing.** `train()` always ends by scoring Testing and writing an artifact. Ten rounds would read Testing ten times, and the final Testing score would be optimistic.
- **Two of the most promising levers are not round-file keys.** The CNN backbone is whatever bundle sits in `weights/`, and the calibration method is the constant `CALIBRATION_METHOD = "sigmoid"` (D9, D12). The radiomic block cannot be dropped, although the ablation shows it adds only +0.005.
- **Nothing helps explain the CV-to-Testing gap.** Almost the whole gap is tumors called no_tumor (85 of 984 on Testing against 19 of 1,570 in CV). The report has only pooled out-of-fold counts, so there is no way to see which Training images the model gets wrong, or whether those errors cluster by image size, quality or group.

## What Changes

- **BREAKING (CLI): tuning runs no longer score Testing.** `python -m src.training` runs selection, augmentation decision, out-of-fold metrics, ablation and shortcut audit, writes the report, and stops: no Testing score, no calibration, no artifact. A new `--final` flag runs the whole pipeline as today (calibrate, score Testing once, write `artifacts/model-v<version>/`). The report says which mode it ran in.
- **Round-file keys for the model family:**
  - `model.backbone`: one of `efficientnet_b0`, `resnet50`, `densenet121`. The run loads the bundle from `<weights dir>/<backbone>/` and stops before any image is read if the bundle is missing or declares another backbone. `--backbone-bundle` is replaced by `--weights-dir` (default `weights`). **BREAKING (layout)**: the existing bundle moves from `weights/` to `weights/efficientnet_b0/`.
  - `model.calibration`: `sigmoid` (default) or `isotonic`, used by `--final` runs.
  - `model.blocks`: `["radiomic", "cnn"]` (default) or `["cnn"]`. Radiomics are still extracted, because serving reports the top radiomic features.
- **Calibrated out-of-fold metrics.** Every run also reports the selected classifier calibrated with the configured method, out-of-fold: macro-F1, log loss and the share and accuracy of confident predictions (triage threshold 0.75). Fold k is calibrated on the other folds only. This is what lets a round compare sigmoid and isotonic without Testing.
- **Out-of-fold predictions file.** Every run writes `oof_predictions.csv` (path, label, fold, `group_id`, group size, width, height, quality metrics, predicted label) for the selected classifier, from the same predictions as `cv_out_of_fold`.
- **Error breakdown on Training.** The report gains `cv_error_breakdown`: out-of-fold recall per class and the tumor→no_tumor error count, split by source size (512x512 vs other), by group size (single vs grouped) and by quality-metric tercile. Training rows only.
- **Rounds.** Rounds 0.2.0, 0.3.0, ... each change one family (backbone, blocks/PCA, classifier hyperparameters, quality thresholds and augmentation, calibration), run in tuning mode, and are compared on CV macro-F1 and the out-of-fold section only. A results table in `docs/ml-engine/decisions.md` lists every round. The kept round is rerun once with `--final`; if Testing macro-F1 ≥ 0.90 and the metadata baseline stays below 0.70 balanced accuracy, it is ready for #114, otherwise the results go to the team.

## Capabilities

### New Capabilities

None. `training-configuration` is introduced by the unarchived change `baseline-training-run`; this change adds requirements to it.

### Modified Capabilities

- `training-configuration`: adds the backbone, calibration and blocks keys, the weights-directory lookup with its mismatch check, and the tuning vs `--final` modes.
- `model-validation`: calibration method becomes configurable (sigmoid or isotonic); Testing is scored only by `--final` runs; adds calibrated out-of-fold metrics, the out-of-fold predictions file and the Training error breakdown.

## Impact

- **Code** (`app/ml_engine/src/training/`): `config.py` (new `model` keys, `ModelConfig` fields), `__main__.py` (`--weights-dir`, `--final`), `run.py` (split into the tuning part and the final part), `calibration.py` (method from config), `model.py` (blocks through `build_model`), `report.py`, a new `diagnosis.py` for the breakdown and the predictions file.
- **Weights layout**: `weights/<backbone>/backbone.{json,pt}`; README, `status.md` and the export commands updated. The trainer compose service is unchanged.
- **Serving**: none. The artifact already carries its own backbone bundle and the pickled calibrated classifier, so any backbone, blocks or calibration choice reaches inference unchanged.
- **Tests**: config keys and rejections, bundle mismatch, tuning mode never reads Testing features nor writes an artifact, isotonic calibration, CNN-only blocks, predictions file and breakdown sums.
- **Cost**: each new backbone extracts CNN features once (about 8 min on CPU for 2,939 images); later rounds on it reuse the cache, which already keys on the backbone's weights.
- **Out of scope**: CNN fine-tuning, a fusion MLP or Grad-CAM (epic #103), automated search, GPU, new candidate classifiers, reading Testing data to explain the gap, publishing the model (#114).
