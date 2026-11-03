# Spec Delta

## MODIFIED Requirements

### Requirement: Calibrated model is one calibrated pipeline per fold
The saved classifier SHALL be calibrated with the configured method (sigmoid by default, or isotonic) on the shared folds and SHALL average the calibrated probabilities of one model per fold, each fitted on that fold's fit rows and calibrated on its validation rows.

#### Scenario: Saved classifier composition
- **WHEN** a run with 5 folds saves its classifier
- **THEN** the classifier holds 5 calibrated models and its probabilities are their average

#### Scenario: Isotonic method
- **WHEN** a final run is configured with isotonic calibration
- **THEN** each of the 5 calibrated models uses the isotonic method, and the report and manifest name it

### Requirement: Testing is scored exactly once
A final training run SHALL score the held-out Testing split exactly once, with the calibrated classifier, after selection, ablation and calibration are complete. A tuning run SHALL NOT score Testing. No other step SHALL read Testing features.

#### Scenario: Single scoring
- **WHEN** a final training run completes
- **THEN** the Testing split has been scored once, by the calibrated classifier

#### Scenario: Tuning run
- **WHEN** a tuning run completes
- **THEN** the Testing split has not been scored and no Testing feature has been read

## ADDED Requirements

### Requirement: Out-of-fold predictions are recorded per image
Every run SHALL write, next to its report, one row per usable original Training image with its path relative to the data folder, label, fold, `group_id`, group size, width, height, quality metrics and the out-of-fold predicted label of the selected classifier. The predictions SHALL be the same ones the out-of-fold metrics are computed from.

#### Scenario: File matches the metrics
- **WHEN** a run completes
- **THEN** the confusion matrix built from the predictions file equals the out-of-fold confusion matrix in the report

#### Scenario: No copy and no Testing row
- **WHEN** the run keeps augmentation
- **THEN** the file holds no augmented copy and no Testing image

### Requirement: Calibrated out-of-fold metrics on Training
Every run SHALL report out-of-fold metrics of the selected classifier calibrated with the configured method: macro-F1, log loss, and the share and accuracy of images whose top probability reaches the triage threshold. For fold k, calibration SHALL use only the other folds, each serving once as calibration rows, so no fold-k image is used to fit or calibrate the model that predicts it.

#### Scenario: Comparing calibration methods without Testing
- **WHEN** two tuning rounds differ only in `model.calibration`
- **THEN** both reports give calibrated out-of-fold macro-F1, log loss and triage share, and neither reads Testing

#### Scenario: Fold k never calibrates its own model
- **WHEN** the calibrated out-of-fold metrics are computed
- **THEN** the model that predicts fold k's images was fitted and calibrated on rows of the other folds only

### Requirement: Out-of-fold errors are broken down on Training
The report SHALL give, from the out-of-fold predictions only, per-class recall and the count of tumor images predicted no_tumor, split by source size (512x512 vs other), by group size (single vs grouped) and by tercile of each quality metric, with the image count of every cell.

#### Scenario: Cells add up
- **WHEN** a run writes the breakdown
- **THEN** for each split, the image counts of its cells sum to the usable original Training images, and the tumor→no_tumor counts sum to that number in the out-of-fold confusion matrix

#### Scenario: Computed before Testing
- **WHEN** a final run computes the breakdown
- **THEN** no Testing feature has been read yet
