# Spec Delta

## RENAMED Requirements

- FROM: `### Requirement: Thresholds are the 1st percentile of training measurements`
- TO: `### Requirement: Thresholds sit between unusable references and training measurements`

## MODIFIED Requirements

### Requirement: Thresholds sit between unusable references and training measurements
Calibration SHALL set each minimum inside the gap between the highest value of that metric over a fixed set of unusable reference images and the lowest value over the readable calibration images: the geometric midpoint for SNR, the arithmetic midpoint for entropy. References for SNR are a blank image, uniform noise and heavy Gaussian noise; for entropy, a blank image and a 4-level posterized image. Unreadable images SHALL be skipped and SHALL NOT count as used.

#### Scenario: Calibration images pass the gate
- **WHEN** thresholds are calibrated on a set of images
- **THEN** every one of those images passes the quality gate with those thresholds

#### Scenario: References fail the gate
- **WHEN** thresholds are calibrated
- **THEN** every SNR reference falls below the minimum SNR and every entropy reference falls below the minimum entropy

#### Scenario: No gap
- **WHEN** the lowest calibration value of a metric is not above the highest reference value of that metric
- **THEN** calibration fails with an error naming the metric and both values

#### Scenario: Unreadable image
- **WHEN** one calibration image cannot be read
- **THEN** it is skipped and the reported number of images used excludes it

#### Scenario: Nothing readable
- **WHEN** no calibration image can be read
- **THEN** calibration fails with an error

### Requirement: Calibration output states its provenance
Calibration SHALL output the minimum SNR, the minimum entropy, the number of images used, and for each metric the lowest calibration value and the highest reference value.

#### Scenario: Command-line output
- **WHEN** calibration is run from the command line
- **THEN** it prints the minimum SNR, minimum entropy, number of images used, and the lowest calibration value and highest reference value of each metric

### Requirement: Calibrated values are recorded with their effect on each class
The project decision log SHALL record the values calibrated on the versioned dataset, the calibration date, the number of images used, the lowest calibration value and highest reference value of each metric, and how many images per class those values reject in the purged training set and in the test set.

#### Scenario: Decision log entry
- **WHEN** the change is complete
- **THEN** the decision log states the calibrated minimum SNR and entropy, the date, the number of images used, the gap of each metric and the per-class rejections for training and test
