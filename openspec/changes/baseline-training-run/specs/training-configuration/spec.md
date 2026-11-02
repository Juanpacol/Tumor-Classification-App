# Spec Delta

## Purpose

Defines how a training run takes its tunable parameters from a versioned round file, which parameters a round may and may not change, and how the configuration a run used is recorded so every round can be compared and reproduced.

## ADDED Requirements

### Requirement: Round file overrides defaults
A training run SHALL accept an optional TOML round file. Its sections SHALL override the run's defaults for preprocessing (including quality thresholds), radiomics, augmentation, PCA components, each candidate classifier's hyperparameters, the seed and the number of folds. Any key the file leaves out SHALL keep its default. Without a file, the run SHALL use the defaults.

#### Scenario: Partial override
- **WHEN** a round file sets only `preprocessing.crop_margin = 0.1`
- **THEN** the run uses a crop margin of 0.1 and every other parameter at its default

#### Scenario: No round file
- **WHEN** a run starts without a round file
- **THEN** it uses the same parameters as a round file with no keys

#### Scenario: Classifier hyperparameter
- **WHEN** a round file sets `classifiers.svm_rbf.C = 3.0`
- **THEN** the SVM candidate is built with `C = 3.0` in selection, ablation, out-of-fold metrics and calibration

### Requirement: Invalid round files stop the run before any work
The run SHALL stop with an error naming the offending key before reading any image when the round file contains an unknown section, key or classifier name, a value of the wrong type, or a value the parameter's own validation rejects.

#### Scenario: Misspelled key
- **WHEN** a round file contains `preprocessing.crop_margn = 0.1`
- **THEN** the run stops with an error that names `preprocessing.crop_margn`, and no feature is extracted

#### Scenario: Unknown classifier hyperparameter
- **WHEN** a round file sets `classifiers.random_forest.n_estimator = 500`
- **THEN** the run stops with an error that names that key

### Requirement: Guarded parameters cannot be overridden
A round file SHALL NOT set a candidate's class weighting, its random state, or the quality metrics version. Class weights SHALL stay balanced, random states SHALL come from the run seed, and the metrics version SHALL be the engine's. A file that sets any of these SHALL be rejected like an invalid file.

#### Scenario: Class weight override
- **WHEN** a round file sets `classifiers.logistic_regression.class_weight = "none"`
- **THEN** the run stops with an error naming that key

#### Scenario: Metrics version override
- **WHEN** a round file sets `preprocessing.thresholds.metrics_version = 1`
- **THEN** the run stops with an error naming that key

### Requirement: Resolved configuration is recorded
Every run SHALL write the resolved configuration, meaning every parameter with its effective value, to its report folder and to the artifact manifest. It SHALL also record the round file's path and SHA-256, or record that no file was used.

#### Scenario: Recorded round
- **WHEN** a run completes with `rounds/0.2.0.toml`
- **THEN** its report folder and its manifest hold the full resolved configuration, the path `rounds/0.2.0.toml` and that file's SHA-256

#### Scenario: Same resolved configuration, same metrics
- **WHEN** two runs have the same data and the same resolved configuration
- **THEN** they report the same cross-validated and Testing metrics

### Requirement: Feature cache follows the resolved configuration
A change to any resolved parameter that alters extracted features SHALL invalidate those cached features. A change to a parameter that does not alter features, namely PCA components and classifier hyperparameters, SHALL reuse them.

#### Scenario: Classifier-only round
- **WHEN** a round changes only `classifiers.svm_rbf.C` from the previous round
- **THEN** no image or copy features are recomputed

#### Scenario: Preprocessing round
- **WHEN** a round changes `preprocessing.crop_margin`
- **THEN** every image's features are recomputed under a new cache key
