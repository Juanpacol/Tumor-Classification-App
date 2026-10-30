# Spec Delta

## Purpose

Derives the quality-gate thresholds (minimum SNR and minimum entropy) from training data only and records the calibrated values, so the gate that rejects poor scans at training and inference is fitted without any information from the held-out test set.

## ADDED Requirements

### Requirement: Calibration uses only the purged training set
Threshold calibration SHALL measure only the images in the training set produced by the leakage-free split of the cleaning manifest. It SHALL NOT open any Testing image or any image the manifest marks `keep = 0`.

#### Scenario: Testing images are never read
- **WHEN** calibration runs on a dataset whose manifest lists Training and Testing images
- **THEN** no Testing image file is opened

#### Scenario: Purged and dropped Training images are excluded
- **WHEN** a Training image is removed by the split or has `keep = 0`
- **THEN** it is not measured during calibration

#### Scenario: Calibration requires a manifest
- **WHEN** calibration is started without a cleaning manifest
- **THEN** it refuses to run instead of scanning the dataset folders

### Requirement: Thresholds are the 1st percentile of training measurements
Calibration SHALL set the minimum SNR and the minimum entropy each to the 1st percentile of that measurement over the readable calibration images, so at least 99% of them pass the gate. Unreadable images SHALL be skipped and SHALL NOT count as used.

#### Scenario: Calibration images pass the gate
- **WHEN** thresholds are calibrated at the 0th percentile on a set of images
- **THEN** every one of those images passes the quality gate with those thresholds

#### Scenario: Unreadable image
- **WHEN** one calibration image cannot be read
- **THEN** it is skipped and the reported number of images used excludes it

#### Scenario: Nothing readable
- **WHEN** no calibration image can be read
- **THEN** calibration fails with an error

### Requirement: Calibration output states its provenance
Calibration SHALL output the minimum SNR, the minimum entropy, the percentile used and the number of images used.

#### Scenario: Command-line output
- **WHEN** calibration is run from the command line
- **THEN** it prints the minimum SNR, minimum entropy, percentile and number of images used

### Requirement: Calibrated values are recorded with their effect on each class
The project decision log SHALL record the values calibrated on the versioned dataset, the calibration date, the percentile, the number of images used, and how many images per class those values reject in the purged training set and in the test set.

#### Scenario: Decision log entry
- **WHEN** the change is complete
- **THEN** the decision log states the calibrated minimum SNR and entropy, the date, the percentile, the number of images used and the per-class rejections for training and test

### Requirement: Calibrated values become defaults only after review
The quality gate's default thresholds SHALL change to calibrated values only when a review of the images those values reject finds that they reject unusable scans and do not concentrate in one class. Until then the defaults SHALL remain unchanged, and the decision log SHALL state why.

#### Scenario: Review finds good scans rejected
- **WHEN** the calibrated values mostly reject usable scans, or their rejections concentrate in one class
- **THEN** the gate defaults stay unchanged and the decision log records the finding and the follow-up that owns the fix

#### Scenario: Default thresholds
- **WHEN** the quality gate is used without explicit thresholds
- **THEN** it applies the defaults the decision log names as current
