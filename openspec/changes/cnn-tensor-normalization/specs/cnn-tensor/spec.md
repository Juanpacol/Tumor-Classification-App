# Spec Delta

## Purpose

Defines how the square CNN image becomes the tensor the frozen backbone receives, with the normalization taken from the backbone bundle so a model is always fed the statistics its weights were trained with, at training and at inference.

## ADDED Requirements

### Requirement: The tensor is the CNN image on three normalized channels
The CNN tensor SHALL have shape (3, S, S) and float32 values, where each channel is the gray CNN image minus that channel's mean, divided by that channel's standard deviation.

#### Scenario: Gray image
- **WHEN** a CNN image is turned into a tensor
- **THEN** the three channels are the same image, each normalized with its own channel's mean and standard deviation

#### Scenario: Known values
- **WHEN** the backbone's mean is 0.5 and its standard deviation 0.25 on every channel, and a pixel of the CNN image is 1.0
- **THEN** that pixel is 2.0 in every channel of the tensor

### Requirement: Normalization comes from the backbone bundle
The mean and standard deviation SHALL be read from the manifest of the backbone bundle that computes the embedding. Preprocessing settings SHALL NOT hold their own copy of them.

#### Scenario: Backbone with non-ImageNet statistics
- **WHEN** the bundle manifest states a mean and standard deviation of 0.5 on every channel
- **THEN** the tensor is normalized with 0.5 and 0.5, with no change to the preprocessing settings

#### Scenario: Artifact with its own normalization setting
- **WHEN** an artifact manifest's preprocessing settings contain a CNN mean or standard deviation
- **THEN** loading it fails with a model-unavailable error that names the unexpected setting

### Requirement: Exported bundles record the statistics of their weights
Exporting a backbone bundle SHALL write the mean, standard deviation and input size published with the selected pretrained weights. Writing a bundle SHALL require those three values instead of assuming defaults.

#### Scenario: Torchvision ImageNet weights
- **WHEN** a torchvision backbone is exported with its ImageNet weights
- **THEN** the bundle manifest holds the mean, standard deviation and input size that torchvision publishes for those weights

### Requirement: Invalid bundle normalization is refused
A bundle SHALL be refused as unavailable when its mean or standard deviation does not have exactly three values or when any standard deviation is not a positive finite number.

#### Scenario: Zero standard deviation
- **WHEN** a bundle manifest has a standard deviation of 0 on one channel
- **THEN** loading the bundle fails with a model-unavailable error

#### Scenario: Wrong number of channels
- **WHEN** a bundle manifest has a mean with one value
- **THEN** loading the bundle fails with a model-unavailable error

### Requirement: Cached embeddings follow the normalization
The feature cache key SHALL change when the bundle's mean, standard deviation or input size changes, even if its weights do not.

#### Scenario: Same weights, other normalization
- **WHEN** two bundles have the same weights but different means
- **THEN** their feature cache keys differ

### Requirement: Same tensor at training and inference
The CNN tensor SHALL be computed by the same code at training and at inference and SHALL depend only on the CNN image and the backbone bundle.

#### Scenario: Training and inference paths
- **WHEN** the same image bytes go through the training path and the inference path with the same settings and bundle
- **THEN** both produce identical embeddings
