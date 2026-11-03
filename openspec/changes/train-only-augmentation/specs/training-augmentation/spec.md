# Spec Delta

## Purpose

Defines how a training run makes augmented copies of Training images, keeps them out of every validation, Testing and inference path, and decides on cross-validation alone whether they stay in the model.

## ADDED Requirements

### Requirement: Copies are made from the CNN image of usable Training images
For each Training image that passes the quality gate, the run SHALL make k augmented copies (default 1) of the image the CNN receives after the geometry step. A copy SHALL keep that image's size and SHALL fill pixels beyond the source border with black. No copy SHALL be made of an image the gate rejects.

#### Scenario: Default number of copies
- **WHEN** a run with default settings has N usable Training images
- **THEN** it makes exactly N copies, one per usable image, each 224x224

#### Scenario: Rejected image has no copy
- **WHEN** the quality gate rejects a Training image
- **THEN** no copy of that image exists in the run

### Requirement: Transform ranges
Each copy SHALL apply a rotation drawn uniformly in [-10, 10] degrees, a horizontal flip with probability 0.5, a scale factor drawn uniformly in [0.9, 1.1], and brightness and contrast factors each drawn uniformly in [0.9, 1.1]. Output values SHALL stay in [0, 1].

#### Scenario: Parameters stay in range
- **WHEN** many copies are drawn with different seeds
- **THEN** every recorded rotation, scale, brightness and contrast lies in its range, and every output pixel lies in [0, 1]

#### Scenario: Neutral parameters
- **WHEN** a copy is made with rotation 0, no flip, scale 1, brightness 1 and contrast 1
- **THEN** it equals the source image

### Requirement: Copies are reproducible and order-independent
The random draw of a copy SHALL depend only on the run seed, the content of its original image file and the copy index.

#### Scenario: Same inputs, same copy
- **WHEN** two runs with the same seed make copy 0 of the same image file
- **THEN** both copies are identical

#### Scenario: Processing order does not matter
- **WHEN** the Training images are processed in a different order with the same seed
- **THEN** every copy is identical to the one made in the original order

### Requirement: Copies vary the CNN features only
A copy's CNN features SHALL be extracted from the copy. Its radiomic features SHALL be those of its original.

#### Scenario: Feature row of a copy
- **WHEN** the feature matrix holds an original and its copy
- **THEN** their radiomic columns are equal and their CNN columns are the features of their own images

### Requirement: A copy inherits its original's label, group and fold
Each copy SHALL have the label, `group_id` and fold of its original.

#### Scenario: Inherited fold
- **WHEN** an original image is assigned to fold 3
- **THEN** each of its copies has label, `group_id` and fold 3 equal to the original's

### Requirement: Copy features are cached by transform parameters
The CNN features of a copy SHALL be cached on disk under a key that includes the content of the original, the transform ranges, k, the copy index and the run seed. A change to any of them SHALL recompute the copy instead of reusing a stale entry.

#### Scenario: Changed transform range
- **WHEN** a run reuses a cache filled with a rotation range of 10 degrees but now uses 15 degrees
- **THEN** every copy's CNN features are recomputed

#### Scenario: Unchanged settings
- **WHEN** a run is repeated with the same settings and cache
- **THEN** no copy's CNN features are recomputed

### Requirement: Copies never reach validation, Testing or inference
Copies SHALL only ever be fit rows of the folds other than their own. They SHALL NOT be validation rows of any fold, SHALL NOT be made for Testing images, and the serving pipeline SHALL NOT make or use copies.

#### Scenario: No copy in a validation fold
- **WHEN** a run with augmentation on builds its splits
- **THEN** no validation row of any fold is a copy, and every copy of an image in fold k is a fit row of every fold except k and of no split in fold k

#### Scenario: Testing has no copies
- **WHEN** a run scores the Testing split
- **THEN** every scored row is an original Testing image

#### Scenario: Serving pipeline does not augment
- **WHEN** the inference pipeline is loaded
- **THEN** it does not load the augmentation code

### Requirement: The run decides whether augmentation stays on
After classifier selection, the run SHALL cross-validate the selected classifier with copies on the shared folds. Augmentation SHALL stay on only if this mean CV macro-F1 is at least the mean CV macro-F1 without copies plus the standard deviation across folds of the latter. The decision SHALL NOT read Testing.

#### Scenario: Gain above the margin
- **WHEN** the selected classifier scores mean CV macro-F1 0.80 with std 0.02 without copies, and 0.83 with copies
- **THEN** augmentation stays on

#### Scenario: Gain below the margin
- **WHEN** the selected classifier scores mean CV macro-F1 0.80 with std 0.02 without copies, and 0.81 with copies
- **THEN** augmentation is turned off and no copy is a fit row of the ablation or calibration

### Requirement: The augmentation decision is recorded
The report and the model artifact manifest SHALL record whether augmentation is on, k, the transform ranges, the mean and std CV macro-F1 with and without copies, and the margin used.

#### Scenario: Decision in the report
- **WHEN** a run completes
- **THEN** its report and its artifact manifest both state the augmentation decision with the two CV scores and the margin
