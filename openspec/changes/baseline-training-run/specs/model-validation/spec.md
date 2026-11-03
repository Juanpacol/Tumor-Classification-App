# Spec Delta

## ADDED Requirements

### Requirement: Out-of-fold per-class metrics on Training
A training run SHALL report out-of-fold metrics for the selected classifier: per-class precision, recall, F1 and support, and the confusion matrix with rows as the true class. Each usable original Training image SHALL be predicted once, by the model of the fold it validates. That model SHALL be fitted on the fold's fit rows, with or without copies as the run chose.

#### Scenario: Every usable original predicted once
- **WHEN** a run with N usable original Training images completes
- **THEN** the out-of-fold confusion matrix sums to N and the supports per class equal the usable images per class

#### Scenario: Copies are never predicted
- **WHEN** the run keeps augmentation
- **THEN** no augmented copy is counted in the out-of-fold metrics, and the copies of fold k's images are fit rows only for the other folds

#### Scenario: Same fit rows as the chosen setting
- **WHEN** the run drops augmentation
- **THEN** the out-of-fold models are fitted on originals only

### Requirement: Out-of-fold metrics never read Testing
The out-of-fold metrics SHALL be computed from Training rows only, before Testing is scored. The report SHALL present them in a section separate from the Testing metrics and SHALL label them as cross-validated on Training.

#### Scenario: Report sections
- **WHEN** a run writes its report
- **THEN** the JSON report holds the out-of-fold metrics under their own key, apart from the Testing metrics, and the markdown report shows them under a heading that names cross-validation on Training

#### Scenario: Testing still scored once
- **WHEN** a run computes the out-of-fold metrics
- **THEN** no Testing feature is read by that step, and Testing is scored exactly once in the run
