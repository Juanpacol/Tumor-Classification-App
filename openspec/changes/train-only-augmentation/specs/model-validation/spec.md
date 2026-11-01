# Spec Delta

## MODIFIED Requirements

### Requirement: All cross-validated steps share the same folds
Classifier selection, the augmentation decision, the feature-block ablation and probability calibration SHALL use the same fold assignment. In every step, fold k's validation rows SHALL be exactly the usable original images assigned to fold k. Its fit rows SHALL be the usable original images of the other folds plus, when the step uses augmentation, the copies of those images.

#### Scenario: Same validation rows in every step
- **WHEN** a run performs selection, the augmentation decision, ablation and calibration
- **THEN** for each fold the validation rows are identical across the four steps

#### Scenario: Fit rows with augmentation
- **WHEN** a step uses augmentation and fold k is validated
- **THEN** its fit rows are the usable originals of the other folds and their copies, and include no copy of an image in fold k
