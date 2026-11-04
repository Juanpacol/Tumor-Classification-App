# Spec Delta

## ADDED Requirements

### Requirement: Round file chooses the CNN backbone
A round file SHALL be able to set `model.backbone` to `efficientnet_b0`, `resnet50` or `densenet121`; the default SHALL be `efficientnet_b0`. The run SHALL use the backbone bundle stored under that name in the weights directory, and the resolved configuration SHALL record the backbone name.

#### Scenario: Backbone override
- **WHEN** a round file sets `model.backbone = "resnet50"` and the weights directory holds a `resnet50` bundle
- **THEN** the run extracts CNN features with that bundle and the resolved configuration records `resnet50`

#### Scenario: Unknown backbone
- **WHEN** a round file sets `model.backbone = "vgg16"`
- **THEN** the run stops with an error naming `model.backbone` before any image is read

### Requirement: Backbone bundle must match the configured backbone
The run SHALL stop before reading any image when the bundle for the configured backbone is missing, fails its integrity check, or declares a different backbone than the one configured. The error SHALL name the backbone and the folder it looked in.

#### Scenario: Missing bundle
- **WHEN** the round sets `model.backbone = "densenet121"` and no `densenet121` bundle exists in the weights directory
- **THEN** the run stops with an error naming `densenet121` and that folder, and no feature is extracted

#### Scenario: Mismatched bundle
- **WHEN** the folder for `resnet50` holds a bundle whose manifest declares `efficientnet_b0`
- **THEN** the run stops with an error naming both backbones

### Requirement: Changing the backbone recomputes only CNN-dependent cache entries
A run SHALL reuse cached features whose backbone weights, preprocessing and radiomics settings all match, and SHALL recompute features under a new cache key when the backbone changes.

#### Scenario: Return to a previous backbone
- **WHEN** a round uses `resnet50`, and a later round with otherwise identical settings returns to `efficientnet_b0`
- **THEN** the later round recomputes no feature

### Requirement: Round file chooses the calibration method
A round file SHALL be able to set `model.calibration` to `sigmoid` or `isotonic`; the default SHALL be `sigmoid`. Any other value SHALL stop the run with an error naming the key before any image is read. The resolved configuration SHALL record the method.

#### Scenario: Isotonic calibration
- **WHEN** a final run uses a round file with `model.calibration = "isotonic"`
- **THEN** the saved classifier is calibrated with the isotonic method and the manifest records `isotonic`

#### Scenario: Invalid method
- **WHEN** a round file sets `model.calibration = "platt"`
- **THEN** the run stops with an error naming `model.calibration`

### Requirement: Round file chooses the feature blocks
A round file SHALL be able to set `model.blocks` to `["radiomic", "cnn"]` (the default) or `["cnn"]`. The classifier SHALL see only the chosen blocks in selection, out-of-fold metrics, calibration and serving. Radiomic features SHALL still be extracted and recorded so serving can report them. Any other value SHALL stop the run naming the key.

#### Scenario: CNN-only round
- **WHEN** a round file sets `model.blocks = ["cnn"]`
- **THEN** every candidate is fitted on the CNN block only, and the saved artifact still reports top radiomic features at serving time

#### Scenario: Empty blocks
- **WHEN** a round file sets `model.blocks = []`
- **THEN** the run stops with an error naming `model.blocks`

### Requirement: Tuning runs do not read Testing
A run without the final flag SHALL complete selection, the augmentation decision, out-of-fold metrics, ablation and the shortcut audit, write its report, and stop. It SHALL NOT extract or read Testing features, calibrate, score Testing or write a model artifact. Its report SHALL state that it is a tuning run.

#### Scenario: Tuning round
- **WHEN** a run with `rounds/0.2.0.toml` completes without the final flag
- **THEN** its report holds the CV and out-of-fold sections, has no Testing section, is marked as a tuning run, and no `model-v0.2.0` artifact exists

#### Scenario: Testing images untouched
- **WHEN** a tuning run completes on a feature cache with no Testing entries
- **THEN** the cache still holds no Testing entry

### Requirement: Final runs score Testing once and write the artifact
A run with the final flag SHALL perform everything a tuning run does, then calibrate, score Testing exactly once and write the versioned model artifact. Its report SHALL state that it is a final run.

#### Scenario: Final run of the kept round
- **WHEN** the kept round is rerun with the final flag and version `0.4.0`
- **THEN** the report holds a Testing section, is marked as a final run, and `model-v0.4.0` exists with the resolved configuration in its manifest

#### Scenario: Same CV results as the tuning run
- **WHEN** a round is run in tuning mode and then in final mode with the same data and resolved configuration
- **THEN** both reports give the same CV and out-of-fold metrics
