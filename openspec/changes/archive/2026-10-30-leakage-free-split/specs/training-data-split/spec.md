# Spec Delta

## Purpose

Divides the cleaned brain MRI dataset into a training set and a held-out test set so that no group of near-duplicate images is on both sides, and reports what the division removed, so test metrics measure generalization rather than memorized near-twins.

## ADDED Requirements

### Requirement: Official Testing folder is the held-out set
The split SHALL take as the test set exactly the images the cleaning manifest marks as kept (`keep = 1`) with split `Test`. The split SHALL NOT add, remove or relabel any of them.

#### Scenario: Test set is unchanged by the split
- **WHEN** the split runs on a manifest with N kept Testing images
- **THEN** the test set contains exactly those N images, with their manifest labels and group ids

#### Scenario: Dropped Testing images are not in the test set
- **WHEN** a Testing image has `keep = 0` in the manifest (duplicate, augmented or label conflict)
- **THEN** it is not part of the test set

### Requirement: Training images sharing a group with the test set are removed
The split SHALL remove from the training set every kept Training image whose `group_id` equals the `group_id` of at least one kept Testing image. Groups whose Testing members all have `keep = 0` SHALL NOT cause removals.

#### Scenario: Training image with a near-twin in Testing
- **WHEN** a kept Training image and a kept Testing image share a `group_id`
- **THEN** the Training image is not in the training set

#### Scenario: Group present only through a dropped Testing image
- **WHEN** a kept Training image shares its `group_id` only with Testing images that have `keep = 0`
- **THEN** the Training image stays in the training set

#### Scenario: Training image with no Testing group member
- **WHEN** a kept Training image's `group_id` has no Testing member
- **THEN** the Training image stays in the training set

### Requirement: No group crosses the split
After the split, the set of `group_id` values in the training set and the set in the test set SHALL be disjoint.

#### Scenario: Disjoint groups on any manifest
- **WHEN** the split runs on any valid manifest
- **THEN** no `group_id` appears in both the training set and the test set

### Requirement: Removed images are reported per class
The split SHALL report how many Training images it removed for each class, including classes with zero removals. A training run SHALL include these counts in its console output, its report and the dataset section of the model artifact manifest.

#### Scenario: Counts available to the caller
- **WHEN** the split removes 3 glioma and 0 notumor Training images
- **THEN** the reported counts are glioma 3, meningioma 0, notumor 0, pituitary 0

#### Scenario: Counts recorded by a training run
- **WHEN** a training run completes
- **THEN** its report and the artifact manifest list the Training images removed by the split per class

### Requirement: Reference counts on the versioned dataset
On the cleaning manifest versioned in the repository (`processed/dedup_20261028T234450903626Z/manifest.csv`), the split SHALL remove glioma 142, meningioma 93, notumor 15 and pituitary 245 Training images, leaving 1,830 Training and 1,109 Testing images. A test SHALL check this against the versioned file and SHALL fail, not skip, if the file is missing.

#### Scenario: Versioned manifest
- **WHEN** the split runs on the versioned manifest
- **THEN** the removed counts are 142, 93, 15 and 245, the training set has 1,830 images and the test set has 1,109

#### Scenario: Versioned manifest missing
- **WHEN** the reference test runs and the versioned manifest is not on disk
- **THEN** the test fails
