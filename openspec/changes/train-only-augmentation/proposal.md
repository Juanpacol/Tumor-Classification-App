# Proposal

## Why

Block B7 of the agreed preprocessing pipeline (`docs/ml-engine/preprocessing-pipeline.md`, Azure #167) adds train-only augmentation, and guarantees G3 (no augmented copy in a validation fold, Testing or inference) and G4 (a copy is always in its original's fold) depend on it. Today the training code has no augmentation, so G3 and G4 have nothing to test. B9 (#112) already assigns the folds once, before features, so copies can now inherit their original's fold without re-splitting.

The evidence for augmentation with a frozen CNN is weak ([5], [9] in the pipeline doc), so the run itself has to decide, on cross-validation only, whether augmentation stays on. This change comes before the shortcut audit (#168) and before the single real-data run, so that run is done once with every block in place.

## What Changes

- New augmentation step. Each usable Training image gets k augmented copies (default k = 1) of its B6 output (`cnn_image`, 224x224): rotation +/-10 degrees, horizontal flip, scale +/-10%, brightness and contrast +/-10%, black fill beyond the border. OpenCV and NumPy only.
- Copies vary the CNN block only. A copy's radiomic features are those of its original, since radiomics run on the image at original resolution, before B6.
- Each copy inherits its original's label, `group_id` and fold. Its random draw is seeded from the run seed, the original's content hash and the copy index, so it does not depend on processing order.
- Copy CNN features are cached on disk under a key that includes the transform parameters, k and the copy index.
- Fold splits take a copy flag: copies are fit rows of the folds other than their own and are never validation rows. Selection, ablation and calibration keep the same validation rows.
- New run step after classifier selection: the selected classifier is cross-validated with copies, and augmentation stays on only when its mean CV macro-F1 beats the run without copies by at least the standard deviation across folds of the run without copies. The block ablation and calibration use the chosen setting. Testing is not involved.
- The report and the artifact manifest record the augmentation decision, both CV scores, the margin and the transform parameters. Copies are never written to `folds.csv` and never counted in the training feature statistics.
- Copies are never made for Testing or at inference.
- Docs: new decision in `docs/ml-engine/decisions.md`; B7 section (remove "and in the final fit", which predates the per-fold models of D16), B7 traceability row and G3/G4 in `docs/ml-engine/preprocessing-pipeline.md`; training section of `app/ml_engine/README.md`.

## Capabilities

### New Capabilities
- `training-augmentation`: how augmented copies of Training images are produced, seeded, cached and kept out of validation, Testing and inference, and how a run decides whether augmentation stays on.

### Modified Capabilities
- `model-validation`: fit rows may include augmented copies while validation rows stay the usable original images of the fold; the report records the augmentation decision.

## Impact

- **Code** (`app/ml_engine/src/training/`): new `augmentation.py`; `features.py` (copy CNN features and their cache key), `folds.py` (copy flag in the splits), `run.py` (copies, augmentation decision, step order), `report.py` and the manifest fields written by `run.py` (augmentation section). `model.py`, `ablation.py` and `calibration.py` keep their signatures: they already take splits.
- **Tests**: new `test_training_augmentation.py`; `test_training_folds.py`, `test_training_features.py`, `test_training_run.py`, `test_training_run_folds.py`, `test_training_report.py`, and a test that `src/pipelines` never imports the augmentation module.
- **Artifacts**: the report and the artifact manifest gain an augmentation section. The model file layout does not change and serving (#113) reads none of the new fields.
- **Cost**: one extra CNN pass per copy (about 1,830 with k = 1) and one extra 5-fold CV of the selected classifier.
- **Dependencies**: none new.
- **Out of scope**: the shortcut audit (#168), the real-data run and its report, augmentation of radiomics or of the raw image, other transforms, and choosing k.
