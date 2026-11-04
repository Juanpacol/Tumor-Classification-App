# Spec Delta

## ADDED Requirements

### Requirement: Local environment migrates the schema before serving
When the local compose environment starts the backend, the system SHALL apply all pending versioned migrations to the configured database before the backend accepts requests, with no manual step. A start against a database already at the latest revision SHALL leave the schema unchanged.

#### Scenario: First start on an empty database
- **WHEN** `docker compose up` starts the stack with an empty PostgreSQL volume
- **THEN** the studies, predictions, reports and chat_messages tables exist before the backend answers its first request, and uploading a study does not fail for a missing table

#### Scenario: Restart on a migrated database
- **WHEN** the backend is restarted against a database already at the latest revision
- **THEN** the backend starts serving and the schema and stored studies are unchanged

#### Scenario: Pending migration applied on start
- **WHEN** the backend starts against a database at an older revision
- **THEN** the database is brought to the latest revision before the backend serves requests

### Requirement: Local backend does not serve when migration fails
If applying migrations fails when the local compose environment starts the backend, the backend MUST NOT start serving requests, and the migration error MUST appear in the backend container's logs.

#### Scenario: Database rejects the migration
- **WHEN** the migration step fails, for example because the database URL has wrong credentials
- **THEN** the backend container exits without listening on its port, and its logs show the migration error
