# Spec Delta

## Purpose

Defines the brain region of interest computed for every loaded 2D MRI image, which radiomics and the geometry step use, so both see the head and not the black background, with the same result at training and inference.

## ADDED Requirements

### Requirement: Brain mask is the largest bright region with holes filled
The brain mask SHALL be a boolean array with the shape of the loaded image, made of the pixels above half the Otsu threshold of that image, keeping only the largest connected component and filling its holes. No skull or scalp tissue SHALL be removed from it.

#### Scenario: Head on a black background
- **WHEN** the mask is computed for a synthetic head on a black background
- **THEN** it covers the center of the head and no corner of the background

#### Scenario: Two separate bright regions
- **WHEN** the image has two disconnected bright regions of different sizes
- **THEN** the mask covers only the larger one

#### Scenario: Dark hole inside the head
- **WHEN** the head contains a dark region fully enclosed by bright tissue
- **THEN** that region is inside the mask

### Requirement: Empty or implausible mask falls back to the full image
When the brain mask is empty or covers less than the configured minimum fraction of the image, the full image SHALL be used as the region of interest. Computing the region SHALL never raise for a loaded image.

#### Scenario: Black image
- **WHEN** the region is computed for an all-black image
- **THEN** it is the full image and no error is raised

#### Scenario: Tiny bright spot
- **WHEN** the only bright region covers less than the minimum fraction of the image
- **THEN** the region is the full image

#### Scenario: Normal head
- **WHEN** the mask covers at least the minimum fraction
- **THEN** the region is the mask itself

### Requirement: The minimum mask fraction is part of the preprocessing settings
The minimum mask fraction SHALL be a preprocessing setting stored with the trained artifact and included in the feature cache key, so a model is always served with the value it was trained with.

#### Scenario: Different minimum fraction
- **WHEN** features are requested with a minimum fraction different from the cached one
- **THEN** cached features are not reused

#### Scenario: Value served from the artifact
- **WHEN** an artifact trained with a given minimum fraction is loaded
- **THEN** inference uses that same value

### Requirement: Radiomics uses the region on the original image
Radiomic features SHALL be extracted from the loaded image at its original resolution, without any pixel removed or changed, inside the region of interest.

#### Scenario: Region and image match
- **WHEN** an image is prepared
- **THEN** the radiomics image equals the loaded image and the region has its shape

### Requirement: Same region at training and inference
The region of interest SHALL be computed by the same code at training and at inference and SHALL depend only on the image and the preprocessing settings.

#### Scenario: Repeated preparation
- **WHEN** the same image bytes are prepared twice with the same settings, once through the training path and once through the inference path
- **THEN** both regions are identical

### Requirement: Skull stripping is not offered
Preprocessing SHALL NOT offer skull stripping. The quality metrics SHALL always report `skull_stripped` as false, and an artifact whose preprocessing settings do not match the settings of this engine SHALL be refused as unavailable.

#### Scenario: Reported flag
- **WHEN** any image is prepared
- **THEN** its quality metrics report `skull_stripped` as false

#### Scenario: Artifact with a skull-stripping setting
- **WHEN** an artifact manifest contains a skull-stripping preprocessing setting
- **THEN** loading it fails with a model-unavailable error that names the mismatch
