# model-validation Specification

## Purpose
Defines how a training run validates the brain MRI classifier: one grouped fold assignment shared by every cross-validated step, the metrics used to select and report the model, and the single use of the held-out Testing split.

## Requirements

### Requirement: One grouped fold assignment per training run
A training run SHALL assign every image of the purged Training split to exactly one of 5 folds, using stratified group k-fold with the image's `group_id` as the group, its label as the stratum, and the run seed. The assignment SHALL be computed once, from the image list alone, before any feature extraction or quality gate.

#### Scenario: Every Training image gets one fold
- **WHEN** a run starts with N purged Training images
- **THEN** each of the N images is assigned exactly one fold in 0..4

#### Scenario: Assignment does not depend on the quality gate
- **WHEN** two runs use the same manifest and seed but different quality thresholds
- **THEN** every image present in both runs has the same fold in both

#### Scenario: Same seed, same folds
- **WHEN** the fold assignment runs twice on the same images with the same seed
- **THEN** both assignments are identical

### Requirement: No group crosses folds
All images that share a `group_id` SHALL be in the same fold.

#### Scenario: Group kept together
- **WHEN** the fold assignment runs on any valid purged Training split
- **THEN** no `group_id` appears in two different folds

### Requirement: Rejected images keep their fold without contributing rows
An image the quality gate rejects SHALL keep its assigned fold and SHALL NOT be a fit or validation row in any fold. The run SHALL fail with an error naming the fold if the gate leaves a fold with no usable images.

#### Scenario: Image rejected by the gate
- **WHEN** an image assigned to fold 2 is rejected by the quality gate
- **THEN** it appears in the fold record with fold 2 marked as not used, and no fold's fit or validation rows include it

#### Scenario: Empty fold
- **WHEN** the quality gate rejects every image of one fold
- **THEN** the run stops with an error that names that fold, before any model is fitted

### Requirement: All cross-validated steps share the same folds
Classifier selection, the feature-block ablation and probability calibration SHALL use the same fold assignment. In every step, fold k's validation rows SHALL be exactly the usable images assigned to fold k, and its fit rows SHALL be the usable images of the other folds.

#### Scenario: Same validation rows in every step
- **WHEN** a run performs selection, ablation and calibration
- **THEN** for each fold the validation rows are identical across the three steps

### Requirement: Selection by cross-validated macro-F1
The run SHALL select the classifier with the highest mean cross-validated macro-F1. For every candidate, it SHALL report the mean and standard deviation of macro-F1 and of balanced accuracy across folds.

#### Scenario: Scores reported per candidate
- **WHEN** selection completes
- **THEN** the report lists, for every candidate, mean and std of CV macro-F1 and of CV balanced accuracy, and names the candidate with the highest mean macro-F1 as selected

#### Scenario: Balanced accuracy does not decide
- **WHEN** one candidate has the highest mean CV macro-F1 and another has the highest mean CV balanced accuracy
- **THEN** the candidate with the highest mean CV macro-F1 is selected

### Requirement: Calibrated model is one calibrated pipeline per fold
The saved classifier SHALL be calibrated with the sigmoid method on the shared folds and SHALL average the calibrated probabilities of one model per fold, each fitted on that fold's fit rows and calibrated on its validation rows.

#### Scenario: Saved classifier composition
- **WHEN** a run with 5 folds saves its classifier
- **THEN** the classifier holds 5 calibrated models and its probabilities are their average

### Requirement: Testing is scored exactly once
A training run SHALL score the held-out Testing split exactly once, with the calibrated classifier, after selection, ablation and calibration are complete. No other step SHALL read Testing features.

#### Scenario: Single scoring
- **WHEN** a training run completes
- **THEN** the Testing split has been scored once, by the calibrated classifier

### Requirement: Held-out metrics include balanced accuracy
The Testing evaluation SHALL report accuracy, macro-F1, balanced accuracy, per-class precision, recall, F1 and support, and the confusion matrix with rows as the true class.

#### Scenario: Balanced accuracy on an unbalanced test set
- **WHEN** the test set has 3 glioma and 1 notumor image and the classifier gets all glioma right and the notumor wrong
- **THEN** accuracy is 0.75 and balanced accuracy is the mean recall over the classes present, 0.5

### Requirement: Fold assignment is recorded
Every run SHALL write, next to its report, a fold record with one row per purged Training image: path relative to the data folder, label, `group_id`, fold, and whether the image was used. The report SHALL give the number of used images per fold and class.

#### Scenario: Fold record written
- **WHEN** a run completes
- **THEN** its report folder holds the fold record with one row per purged Training image, and the report lists used images per fold and class

#### Scenario: Fold record matches the rows
- **WHEN** the fold record marks M images of fold k as used
- **THEN** fold k's validation rows in selection, ablation and calibration are exactly those M images
