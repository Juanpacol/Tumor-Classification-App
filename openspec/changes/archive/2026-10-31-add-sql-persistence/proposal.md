# Proposal

## Why

Studies, predictions, clinical reports and chat messages live only in an in-memory repository today, so every restart loses them and clinicians cannot return to previous results (AB-116). The hackathon deployment (Cloud Run + Supabase) needs durable relational storage, and the hexagonal skeleton from AB-115 is small enough that making the repository port async and aggregate-wide is cheapest now.

## What Changes

- Persist the study aggregate in PostgreSQL: `studies`, `predictions` (1:1), `reports` (1:1) and `chat_messages` (1:N), with no PHI columns and no original filename.
- Create the schema from scratch with Alembic migrations that run unchanged against docker-compose Postgres and Supabase; only `DATABASE_URL` differs between environments.
- **BREAKING** `StudyRepositoryPort` becomes async and grows to cover the whole aggregate: prediction, report and chat message read/write methods alongside `get`/`save`/`ping`.
- **BREAKING** Domain model updates: `Study.image_path` is replaced by `file_key`, plus `original_format`, `clinical_notes` and `rejection_reason`; `Prediction` gains `quality` and `top_features`; a new `Report` entity wraps the existing `ClinicalReport` content with a lifecycle (`status`, `model`, `error`, `created_at`).
- Add an async SQLAlchemy (asyncpg) adapter implementing the port; ORM objects are mapped to domain dataclasses and never leave the adapter.
- Update the in-memory fake to the new async port so unit tests keep running without a database.
- Normalize `DATABASE_URL` to the asyncpg dialect and drop the SQLite default, aligning settings with `docs/deployment.md`.
- Add integration tests that run against Postgres (CI service / docker-compose), never SQLite.

## Capabilities

### New Capabilities
- `study-persistence`: durable storage and retrieval of the study aggregate (study, prediction, report, chat history), its schema lifecycle via migrations, and the privacy constraints on stored data.

### Modified Capabilities
<!-- None: no specs exist yet. -->

## Impact

- **Code**: `app/backend/domain/models.py`, `ports/study_repository_port.py`, `adapters/outbound/persistence/` (fake + new SQLAlchemy adapter), `adapters/inbound/http/router_health.py` (awaits `ping`), `main.py` (engine wiring and disposal), `settings.py`.
- **New files**: Alembic config and initial migration under `app/backend/`; integration tests under `app/backend/tests/`.
- **Dependencies**: already declared (`sqlalchemy`, `asyncpg`, `alembic`); a test-only dependency may be needed for async DB fixtures.
- **Systems**: docker-compose Postgres (local), GitHub Actions Postgres service (CI), Supabase (cloud demo). `docs/deployment.md` gains notes on running migrations and on the Supabase connection mode.
- **Out of scope**: use cases and HTTP endpoints that create studies, run classification, generate reports or chat; file storage (`StoragePort`); Celery workers.
