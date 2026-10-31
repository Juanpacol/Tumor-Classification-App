# image-quality-gate Specification

## Purpose
Measures the quality of one brain MRI image (SNR and entropy) and rejects it before classification when it is unusable, with the same measurements at training and inference.

## Requirements

### Requirement: SNR compares brain signal with background noise
The SNR SHALL be the median intensity inside the brain region divided by the noise of the dark background, which is every pixel away from all bright regions of the image, not only away from the brain. Tissue outside the brain and texture inside it SHALL NOT count as noise. Noise SHALL be floored at one 8-bit intensity step, so the SNR is finite and bounded.

#### Scenario: Added noise lowers the SNR
- **WHEN** Gaussian noise is added to a slice with a dark background
- **THEN** its SNR is lower than the SNR of the clean slice

#### Scenario: Brain texture does not lower the SNR
- **WHEN** zero-mean fine texture is added inside the brain region only and the background is unchanged
- **THEN** the SNR changes by less than 5%

#### Scenario: Tissue outside the brain does not lower the SNR
- **WHEN** a bright structure separate from the brain (such as neck or face tissue) is added to the background of a slice
- **THEN** the SNR changes by less than 5%

#### Scenario: Noise-free background
- **WHEN** the background of a slice is exactly zero
- **THEN** the SNR is finite and equals the median brain intensity divided by one 8-bit intensity step

#### Scenario: Flat image
- **WHEN** the image has a single intensity
- **THEN** its SNR is 0

### Requirement: SNR is defined without a background region
When the dark background covers less than 1% of the image, the noise SHALL be estimated from the whole image with the same floor, and the SNR SHALL be a finite, non-negative number without raising.

#### Scenario: Brain fills the frame
- **WHEN** the bright regions cover almost the whole image
- **THEN** an SNR is returned without error

#### Scenario: Uniform noise
- **WHEN** the image is uniform random noise
- **THEN** its SNR is below 2

### Requirement: Entropy is measured inside the brain region
The entropy SHALL be the Shannon entropy in bits of the intensity histogram of the pixels inside the brain region only, so the amount of black background does not change it.

#### Scenario: Padding with background does not change entropy
- **WHEN** a slice is padded with a black border that doubles its size
- **THEN** its entropy changes by less than 0.05 bits

#### Scenario: Flat image
- **WHEN** the image has a single intensity
- **THEN** its entropy is 0

### Requirement: Measurements carry a definition version
The quality thresholds SHALL carry the version of the SNR and entropy definitions they were calibrated for. A stored model whose thresholds name a different version, or none, SHALL be refused at load time, and the feature cache SHALL NOT reuse entries computed under another version.

#### Scenario: Artifact from the old metrics
- **WHEN** a model artifact whose thresholds have no definition version is loaded
- **THEN** loading fails with a model-unavailable error

#### Scenario: Cache after a definition change
- **WHEN** the definition version changes and the thresholds do not
- **THEN** the feature cache computes new entries instead of reusing the old ones

### Requirement: Gate rejects images below the thresholds
The gate SHALL reject an image whose SNR or entropy is below the minimum SNR or minimum entropy, with an error that carries the measured values and names each failed metric. Images at or above both minimums SHALL pass.

#### Scenario: Blank image
- **WHEN** a blank image is checked with the default thresholds
- **THEN** it is rejected and the error names both SNR and entropy

#### Scenario: Pure noise
- **WHEN** an image of uniform random noise is checked with the default thresholds
- **THEN** it is rejected

#### Scenario: Usable slice
- **WHEN** a synthetic brain slice is checked with the default thresholds
- **THEN** it passes

### Requirement: Quality is measured before geometry and the mask fallback
Quality SHALL be measured on the loaded image before any geometry step, inside the brain mask as computed before its full-image fallback, with the same code at training and inference.

#### Scenario: Geometry does not change quality
- **WHEN** an image is prepared
- **THEN** the reported SNR and entropy equal those measured on the loaded image

#### Scenario: Fallback does not change quality
- **WHEN** an image whose brain mask is below the minimum fraction is measured
- **THEN** the SNR and entropy are computed on that mask, not on the full image
