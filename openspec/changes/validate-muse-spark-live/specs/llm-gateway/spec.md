# Spec Delta

## Purpose

Defines how the backend calls the hosted LLM (Muse Spark through the Meta Model API): what each request bounds and declares, how answers and streams are read, how token usage reaches the spend cap, and what data may be sent while free credits are in use.

## ADDED Requirements

### Requirement: Every request bounds its generated tokens
Every request to the LLM SHALL state an upper bound on generated tokens, counting both visible output and reasoning. The bound is configurable, and its default is chosen from measured calls.

#### Scenario: Default bound
- **WHEN** a report or chat request is sent and no bound is configured
- **THEN** the request carries the documented default bound

#### Scenario: Configured bound
- **WHEN** the deployment configures a bound
- **THEN** every structured and streamed request carries exactly that bound

#### Scenario: An invalid bound stops startup
- **WHEN** the configured bound is not a positive integer
- **THEN** the application refuses to start with a settings error

### Requirement: Report and chat each declare their own reasoning effort
Every structured (report) request SHALL carry the report reasoning effort, and every streamed (chat) request SHALL carry the chat reasoning effort. Both are configurable and limited to the values the model accepts (`minimal`, `low`, `medium`, `high`, `xhigh`, `max`). Their defaults are chosen from measured calls.

#### Scenario: Defaults
- **WHEN** nothing is configured
- **THEN** a report request carries `low` and a chat request carries `minimal`

#### Scenario: Configured efforts
- **WHEN** the deployment configures a report effort and a chat effort
- **THEN** report requests carry the report effort and chat requests carry the chat effort

#### Scenario: An effort the model rejects stops startup
- **WHEN** either configured effort is not one of the accepted values, for example `none`
- **THEN** the application refuses to start with a settings error

### Requirement: An image is sent with its real media type
An image attached to a request SHALL be declared with the media type of its actual content, JPEG or PNG. The type is detected from the bytes, not assumed.

#### Scenario: JPEG study image
- **WHEN** a report is generated for a study whose stored image is a JPEG
- **THEN** the image is sent as `data:image/jpeg;base64,...`

#### Scenario: PNG study image
- **WHEN** the stored image is a PNG
- **THEN** the image is sent as `data:image/png;base64,...`

### Requirement: Streamed answers report measured usage
A streamed request SHALL ask the API for token usage. Usage carried by any chunk SHALL be handed to the spend cap, and a chunk with no choices SHALL produce no text.

#### Scenario: Usage on the final content chunk
- **WHEN** a chat answer streams and its last chunk carries `finish_reason` and a `usage` object, as the live API does
- **THEN** the reader receives every text chunk and the spend cap counts the reported tokens

#### Scenario: Usage on a chunk without choices
- **WHEN** the last chunk has an empty `choices` list and a `usage` object, as Meta's schema documents
- **THEN** no error is raised, no text is produced for it, and the spend cap counts the reported tokens

#### Scenario: Stream ends without usage
- **WHEN** a stream is interrupted or ends without a usage chunk
- **THEN** the spend cap counts the character-based estimate for that call, as before

### Requirement: Usage reaches the spend cap once per call
Each call SHALL add its tokens to the spend cap exactly once. Reported usage is used when present, and the estimate or default only when usage is absent.

#### Scenario: Stream with reported usage is not double counted
- **WHEN** a stream reports usage
- **THEN** the spend cap adds the reported total and does not also add the estimate

### Requirement: Only public data reaches the LLM while free credits are in use
While the Meta account runs on free credits, validation calls and demos SHALL send only images from the public training dataset and invented clinical notes. They SHALL NOT use the `-contributor` model variants.

#### Scenario: Live validation run
- **WHEN** the live validation probe is run
- **THEN** its image comes from the public dataset folder, its notes are invented text, and its model id is not a `-contributor` variant

### Requirement: Live validation is documented
The backend documentation SHALL record, for each validated capability (plain text, image, structured JSON, streaming), the date, model, latency, input, cached, output and reasoning tokens, and cost. It SHALL also record the prices used and any claim that remains unconfirmed.

#### Scenario: Reader checks what a call costs
- **WHEN** a developer reads the Muse Spark section of the backend README
- **THEN** they find a dated results table with one row per capability and the per-token prices it was computed from

#### Scenario: Validation is repeatable without touching CI
- **WHEN** the test suite runs, in CI or locally, without an explicit live run
- **THEN** no request is sent to the Meta API
