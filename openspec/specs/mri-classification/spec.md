# mri-classification Specification

## Purpose
Classifies an uploaded MRI image with the trained ML engine and returns a typed prediction. The backend keeps serving when the model is unavailable, and it explains why an image cannot be used.

## Requirements

### Requirement: Model bundle is loaded once at startup
The system SHALL load the model bundle configured by `MODEL_BUNDLE_PATH` once, during application startup. It SHALL reuse that loaded model for every classification and SHALL NOT reload it per request.

#### Scenario: Bundle loaded a single time
- **WHEN** the application starts with a valid bundle and then serves several classifications
- **THEN** the bundle is loaded exactly once and every classification uses the same loaded model

### Requirement: Application starts without a usable model
The system SHALL start and serve requests when `MODEL_BUNDLE_PATH` is unset, points to a missing folder, or points to a bundle that is incomplete, altered or inconsistent. It SHALL log the reason the model is unavailable.

#### Scenario: Bundle path not configured
- **WHEN** the application starts with `MODEL_BUNDLE_PATH` unset
- **THEN** the application starts and the log states that the model is unavailable because no bundle path is configured

#### Scenario: Bundle folder missing
- **WHEN** the application starts with `MODEL_BUNDLE_PATH` pointing to a folder that does not exist
- **THEN** the application starts and the log names the missing manifest

#### Scenario: Corrupt bundle
- **WHEN** the application starts with a bundle whose manifest is unreadable, or whose files are missing or do not match their recorded hashes
- **THEN** the application starts and the log states why the bundle was rejected

### Requirement: Health reports model availability
`GET /health` SHALL report `model_loaded: true` only when a bundle was loaded successfully, and `model_loaded: false` otherwise. It SHALL still respond with HTTP 200.

#### Scenario: Model loaded
- **WHEN** `GET /health` is called after startup with a valid bundle
- **THEN** the response is HTTP 200 with `model_loaded: true`

#### Scenario: Model unavailable
- **WHEN** `GET /health` is called after startup without a usable bundle
- **THEN** the response is HTTP 200 with `model_loaded: false`

### Requirement: Classification without a model is refused
When no model is loaded, a classification request SHALL fail with error code `MODEL_UNAVAILABLE`, which the HTTP API returns as status 503.

#### Scenario: Classify while unavailable
- **WHEN** a classification is requested and no model is loaded
- **THEN** it fails with code `MODEL_UNAVAILABLE` and the HTTP error response has status 503 and body `{"code": "MODEL_UNAVAILABLE", "message": ...}`

### Requirement: Classification returns a typed prediction
A successful classification SHALL return the prediction for the given study: label, probabilities for glioma, meningioma, pituitary and no_tumor, confidence, expert-triage flag, quality metrics (snr, entropy, skull_stripped), top radiomic features (name, value, z_score) and model version.

#### Scenario: Valid image classified
- **WHEN** a classification is requested for a study with image bytes the model accepts
- **THEN** the prediction carries that study's identifier, a label from the four classes, probabilities for all four classes that sum to 1, the confidence of the predicted label, the quality metrics, the top features and a non-empty model version

### Requirement: Unusable images are rejected with their reason
An image the model cannot read SHALL fail with code `UNSUPPORTED_IMAGE` and its reason. An image that fails the quality gate SHALL fail with code `QUALITY_CHECK_FAILED`, its reason and the measured snr, entropy and skull_stripped. Both map to HTTP 422.

#### Scenario: Unreadable image
- **WHEN** a classification is requested with bytes that are not a supported image (for example, empty bytes or a renamed text file)
- **THEN** it fails with code `UNSUPPORTED_IMAGE`, and the HTTP error response has status 422 and `details.reason`

#### Scenario: Image fails quality gate
- **WHEN** a classification is requested with an image whose quality is below the model's thresholds
- **THEN** it fails with code `QUALITY_CHECK_FAILED`, and the HTTP error response has status 422 with `details.reason`, `details.snr`, `details.entropy` and `details.skull_stripped`

### Requirement: Inference does not block other requests
Classification SHALL run off the request event loop, at most one inference at a time per process, so other endpoints keep responding while an inference is in progress.

#### Scenario: Health responds during inference
- **WHEN** a slow classification is in progress and `GET /health` is called
- **THEN** `/health` answers without waiting for the classification to finish

#### Scenario: Concurrent classifications are serialized
- **WHEN** two classifications are requested at the same time
- **THEN** the second inference starts only after the first one finishes, and both return their own result
