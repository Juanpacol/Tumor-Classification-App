# Spec Delta

## Purpose

Lets a clinician upload a brain MRI image with optional clinical notes and get its AI classification back in the same request. Images are stored without identifying metadata and every outcome is recorded with a clear status.

## ADDED Requirements

### Requirement: Upload request shape
`POST /api/studies` SHALL accept a multipart form with a required `file` part and an optional `clinical_notes` text field of at most 2000 characters. A request that breaks these rules SHALL be refused with HTTP 422 and nothing SHALL be persisted.

#### Scenario: Notes within the limit
- **WHEN** a valid image is uploaded with `clinical_notes` of 2000 characters
- **THEN** the study is created and the response echoes the notes

#### Scenario: Notes too long
- **WHEN** an image is uploaded with `clinical_notes` of 2001 characters
- **THEN** the response is HTTP 422 and no study or file is stored

#### Scenario: File missing
- **WHEN** the form has no `file` part
- **THEN** the response is HTTP 422 and no study or file is stored

### Requirement: Oversized files are refused
The system SHALL refuse a file larger than 20 MB with HTTP 413 and code `FILE_TOO_LARGE`, and SHALL NOT persist a study or a file.

#### Scenario: File one byte over the limit
- **WHEN** a file of 20 MB plus one byte is uploaded
- **THEN** the response is HTTP 413 with code `FILE_TOO_LARGE` and no study or file is stored

### Requirement: Only decodable JPEG and PNG images are accepted
The system SHALL detect the format from the file content, never from the filename or declared content type. It SHALL refuse DICOM, any other format, and JPEG or PNG files whose image data cannot be fully decoded with HTTP 415 and code `UNSUPPORTED_MEDIA_TYPE`, and SHALL NOT persist a study or a file.

#### Scenario: PNG renamed to .jpg
- **WHEN** a valid PNG is uploaded with filename `scan.jpg` and content type `image/jpeg`
- **THEN** it is accepted and the study's original format is `png`

#### Scenario: DICOM file
- **WHEN** a DICOM file is uploaded, whatever its filename
- **THEN** the response is HTTP 415 and no study or file is stored

#### Scenario: Text file renamed to .png
- **WHEN** a text file named `scan.png` is uploaded
- **THEN** the response is HTTP 415 and no study or file is stored

#### Scenario: Truncated image data
- **WHEN** a JPEG whose headers are valid but whose image data is truncated is uploaded
- **THEN** the response is HTTP 415 and no study or file is stored

### Requirement: Upload is refused while the model is unavailable
When no classification model is loaded, the system SHALL refuse an otherwise valid upload with HTTP 503 and code `MODEL_UNAVAILABLE`, and SHALL NOT persist a study or a file.

#### Scenario: Upload with no model loaded
- **WHEN** a valid image is uploaded while no model is loaded
- **THEN** the response is HTTP 503 with code `MODEL_UNAVAILABLE` and no study or file is stored

### Requirement: Stored images carry no identifying metadata
The system SHALL store the uploaded image without EXIF, XMP, IPTC, comment or PNG text metadata, keyed by the study UUID, with the pixel data unchanged. It SHALL NOT store the original filename anywhere.

#### Scenario: JPEG with EXIF
- **WHEN** a JPEG carrying EXIF tags (for example, camera model and date) is uploaded
- **THEN** the stored file has no EXIF data and decodes to the same pixels as the upload

#### Scenario: PNG with text chunks
- **WHEN** a PNG carrying `tEXt`, `iTXt` or `eXIf` chunks is uploaded
- **THEN** the stored file has none of those chunks and decodes to the same pixels as the upload

#### Scenario: Filename is not kept
- **WHEN** an image named `john_doe_mri.png` is uploaded
- **THEN** neither the storage key, the study record nor the response contains `john_doe_mri`

### Requirement: Successful classification returns the study
The system SHALL classify the stored image synchronously within the request. On success it SHALL save the study as `completed` with its prediction and respond with HTTP 201 and the study, including the prediction.

#### Scenario: Image classified
- **WHEN** a valid image is uploaded and the classifier returns a prediction
- **THEN** the response is HTTP 201 with `status: "completed"` and the prediction, and the same study and prediction are stored

### Requirement: Rejected images are recorded with their reason
When the classifier rejects the image (failed quality gate, or an image the model cannot use), the system SHALL save the study as `rejected` with the specific rejection reason. It SHALL respond with HTTP 422 and a body holding `code`, `message`, `details` (including `reason`) and `study`, which is the rejected study in the same shape as a successful response.

#### Scenario: Quality gate fails
- **WHEN** a valid image is uploaded and the classifier reports `QUALITY_CHECK_FAILED` with reason "too noisy"
- **THEN** the response is HTTP 422 with code `QUALITY_CHECK_FAILED`, `details.reason` "too noisy" and `study.status` "rejected"; the stored study is `rejected` with rejection reason "too noisy" and has no prediction

#### Scenario: Model cannot use the image
- **WHEN** a decodable image is uploaded and the classifier reports `UNSUPPORTED_IMAGE`
- **THEN** the response is HTTP 422 with code `UNSUPPORTED_IMAGE` and `study.status` "rejected", and the stored study is `rejected` with that reason

### Requirement: Unexpected classifier failures are recorded without leaking details
When classification fails for any other reason, the system SHALL save the study as `failed` and respond with HTTP 500 and the body `{"code": "INTERNAL_ERROR", "message": ...}`. The message SHALL be generic and SHALL NOT include the exception text.

#### Scenario: Classifier raises an unexpected error
- **WHEN** a valid image is uploaded and the classifier raises an unexpected exception with message "boom at /data/john_doe"
- **THEN** the response is HTTP 500 with code `INTERNAL_ERROR`, the body does not contain "boom" or "john_doe", and the stored study is `failed`

### Requirement: Upload logs contain no PHI
The system SHALL NOT write clinical notes, original filenames, image content or exception messages from a failed upload to the logs. A failure log entry SHALL identify the study only by its UUID and the error type.

#### Scenario: Failure is logged without PHI
- **WHEN** an upload with notes "Patient Jane Roe, 54" and filename `jane_roe.png` fails with an unexpected classifier error
- **THEN** the captured logs contain the study UUID and the error type, and contain neither "Jane Roe" nor "jane_roe"
