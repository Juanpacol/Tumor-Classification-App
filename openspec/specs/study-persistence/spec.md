# study-persistence Specification

## Purpose
Durably stores and retrieves the study aggregate (study, prediction, clinical report and chat history) so clinicians can return to previous results, without retaining patient-identifying data.

## Requirements

### Requirement: Study records persist across restarts
The system SHALL durably store each study with its identifier (UUID), creation timestamp, status, original file format, storage file key, optional clinical notes and optional rejection reason, and SHALL return the same values when the study is read back after a service restart.

#### Scenario: Saved study is readable after restart
- **WHEN** a study is saved and the service is restarted
- **THEN** reading the study by its identifier returns the same identifier, created_at, status, original_format, file_key, clinical_notes and rejection_reason

#### Scenario: Unknown study
- **WHEN** a study is read with an identifier that was never saved
- **THEN** the system returns no study instead of raising an error

#### Scenario: Status update
- **WHEN** an existing study is saved again with a different status or rejection reason
- **THEN** subsequent reads return the updated values and no duplicate study exists

### Requirement: Study status is restricted to known values
The system SHALL only store study statuses `pending`, `processing`, `completed`, `rejected` and `failed`.

#### Scenario: Invalid status rejected by storage
- **WHEN** a study row with a status outside the allowed set is written directly to the database
- **THEN** the database rejects the write

### Requirement: One prediction per study
The system SHALL store at most one prediction per study, containing label, class probabilities, confidence, expert-triage flag, quality assessment, top features and model version.

#### Scenario: Prediction round-trip
- **WHEN** a prediction is saved for a study and then read by the study identifier
- **THEN** all prediction fields are returned with the same values, including nested probabilities, quality and top features

#### Scenario: Prediction replaced
- **WHEN** a second prediction is saved for a study that already has one
- **THEN** reads return the latest prediction and only one prediction exists for that study

#### Scenario: Study without prediction
- **WHEN** the prediction is read for a study that has none
- **THEN** the system returns no prediction

### Requirement: One clinical report per study with lifecycle
The system SHALL store at most one report per study with status (`pending`, `completed` or `failed`), model identifier, creation timestamp, optional structured content and optional error message.

#### Scenario: Completed report round-trip
- **WHEN** a completed report with structured content is saved and read back
- **THEN** status, model, created_at and every content field (findings, visual observations, impression, differential considerations, suggested next steps, AI agreement) are returned unchanged

#### Scenario: Failed report keeps the error
- **WHEN** a report is saved with status `failed`, no content and an error message
- **THEN** reading it returns status `failed`, empty content and the same error message

#### Scenario: Report progresses from pending to completed
- **WHEN** a `pending` report is later saved as `completed` with content
- **THEN** reads return the completed report and only one report exists for that study

### Requirement: Chat history per study
The system SHALL store any number of chat messages per study, each with role, content and creation timestamp, and SHALL return a study's messages ordered by creation time ascending.

#### Scenario: Messages listed in order
- **WHEN** three messages are added to a study at increasing times
- **THEN** listing the study's messages returns all three oldest first

#### Scenario: Study without messages
- **WHEN** messages are listed for a study that has none
- **THEN** the system returns an empty list

### Requirement: Dependent records belong to an existing study
The system SHALL reject predictions, reports and chat messages that reference a study that does not exist, and SHALL remove them when their study is deleted.

#### Scenario: Orphan prediction rejected
- **WHEN** a prediction is saved for a study identifier that does not exist
- **THEN** the save fails and nothing is stored

#### Scenario: Cascade on study deletion
- **WHEN** a study with a prediction, a report and chat messages is deleted from the database
- **THEN** its prediction, report and chat messages are deleted as well

### Requirement: No patient-identifying data is stored
The persisted schema MUST NOT contain columns for patient name, patient identifier or birth date, and MUST NOT store the original uploaded filename.

#### Scenario: Schema inspection
- **WHEN** the migrated database schema is inspected
- **THEN** no table has a column for patient name, patient ID, birth date or original filename

### Requirement: Schema is created by versioned migrations
The system SHALL create the complete schema on an empty PostgreSQL database by applying versioned migrations, and SHALL support rolling the initial migration back.

#### Scenario: Migrate an empty database
- **WHEN** migrations are applied to an empty PostgreSQL database
- **THEN** the studies, predictions, reports and chat_messages tables exist with their constraints

#### Scenario: Downgrade to empty
- **WHEN** the initial migration is rolled back
- **THEN** the four tables no longer exist

### Requirement: Environment is selected only by connection string
The system SHALL connect to local PostgreSQL and to Supabase using the same code, differing only in the `DATABASE_URL` value, and SHALL accept plain `postgresql://` URLs.

#### Scenario: Plain PostgreSQL URL accepted
- **WHEN** `DATABASE_URL` is set to a `postgresql://` connection string
- **THEN** the service and migrations connect without any other configuration change

#### Scenario: Pooled Supabase connection
- **WHEN** `DATABASE_URL` points to a Supabase transaction-mode pooler and prepared-statement caching is disabled through configuration
- **THEN** reads and writes succeed

### Requirement: Health check reports database reachability
The health endpoint SHALL report whether the database is reachable without failing the request when it is not.

#### Scenario: Database down
- **WHEN** the database is unreachable and `/health` is called
- **THEN** the response is successful with `database_reachable` set to false
