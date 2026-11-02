# Spec Delta

## MODIFIED Requirements

### Requirement: Unusable images are rejected with their reason
An image the model cannot use SHALL fail with code `UNSUPPORTED_IMAGE` and its reason. An image that fails the quality gate SHALL fail with code `QUALITY_CHECK_FAILED`, its reason and the measured snr, entropy and skull_stripped. Both map to HTTP 422. Files that are not decodable JPEG or PNG images are refused at upload before classification (see `study-upload`), so they never reach the model.

#### Scenario: Unreadable image
- **WHEN** a classification is requested with a decodable image the model cannot use (for example, smaller than its minimum size)
- **THEN** it fails with code `UNSUPPORTED_IMAGE`, and the HTTP error response has status 422 and `details.reason`

#### Scenario: Undecodable file never reaches the model
- **WHEN** a file that is not a decodable JPEG or PNG (for example, empty bytes or a renamed text file) is uploaded
- **THEN** the upload is refused with HTTP 415 and the classifier is not called

#### Scenario: Image fails quality gate
- **WHEN** a classification is requested with an image whose quality is below the model's thresholds
- **THEN** it fails with code `QUALITY_CHECK_FAILED`, and the HTTP error response has status 422 with `details.reason`, `details.snr`, `details.entropy` and `details.skull_stripped`
