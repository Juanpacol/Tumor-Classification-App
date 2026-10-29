# Tasks

## 1. Domain model

- [x] 1.1 Update `Study` (`file_key`, `original_format`, `clinical_notes`, `rejection_reason`; drop `image_path`) and `Prediction` (`quality`, `top_features`) in `domain/models.py`; verify `tests/test_architecture_boundaries.py` still passes
- [x] 1.2 Add `ReportStatus` enum and `Report` dataclass wrapping `ClinicalReport` (design D3); verify with a unit test that a `Report` builds in pending, completed and failed states

## 2. Async aggregate port and in-memory fake

- [x] 2.1 Make `StudyRepositoryPort` async and add `save_prediction`/`get_prediction`, `save_report`/`get_report`, `add_message`/`list_messages` (design D1); verify the Protocol type-checks with mypy/pyright in CI
- [x] 2.2 Update `InMemoryStudyRepository` to the async port, with upsert semantics for prediction/report and messages ordered by `created_at`; verify with unit tests mirroring the spec scenarios (round-trip, replace, empty, ordering)
- [x] 2.3 Update `router_health.py` to `await repository.ping()`; verify `tests/test_health.py` and `tests/test_api_documentation.py` pass

## 3. Settings and connection

- [x] 3.1 In `settings.py`, replace the SQLite default with the docker-compose Postgres URL and add `async_database_url` (rewrites `postgresql://`/`postgres://` to `postgresql+asyncpg://`) plus `db_statement_cache_size`; verify with unit tests covering both URL schemes and an already-async URL
- [x] 3.2 Add `adapters/outbound/persistence/sqlalchemy/engine.py` building the `AsyncEngine` and `async_sessionmaker` from settings (passing `statement_cache_size` when set); verify with a unit test that `connect_args` reflects the setting

## 4. ORM, mappers and migration

- [x] 4.1 Add `orm.py` with the four tables, naming convention, CHECK constraints, FKs `ON DELETE CASCADE` and the `(study_id, created_at)` index (design D5); verify no column matches patient name/ID/birth date/original filename
- [x] 4.2 Add `mappers.py` (ORM <-> domain, `str` <-> `UUID`, `ClinicalReport` <-> JSONB); verify with unit tests round-tripping every entity, including `Report` with null content and error
- [x] 4.3 Set up Alembic (`alembic.ini`, async `migrations/env.py` reading `async_database_url`, `target_metadata` from `orm.py`); verify `alembic check` runs against docker-compose Postgres
- [x] 4.4 Create and hand-review the initial revision (tables, CHECKs, index, `downgrade`); verify `alembic upgrade head` then `alembic downgrade base` succeed on an empty database

## 5. SQLAlchemy repository

- [x] 5.1 Add integration test infrastructure in `tests/integration/conftest.py`: `integration` marker registered in `pyproject.toml`, session fixture running `alembic upgrade head`, per-test truncation, skip with reason when Postgres is unreachable; verify a placeholder test runs against docker-compose Postgres and skips cleanly without it
- [x] 5.2 Implement `SqlAlchemyStudyRepository.get`/`save`/`ping` (upsert on `studies`, `ping` returns False on error); verify integration tests for study round-trip, unknown study, status update and invalid status rejected by the CHECK
- [x] 5.3 Implement prediction and report methods with `ON CONFLICT` upserts; verify integration tests for round-trip, replacement, missing record, failed report with error and orphan prediction rejected
- [x] 5.4 Implement `add_message`/`list_messages`; verify integration tests for ordering, empty history and cascade deletion of prediction/report/messages when a study row is deleted
- [x] 5.5 Add an integration test inspecting the migrated schema for PHI columns and the four-table upgrade/downgrade; verify it passes against docker-compose Postgres

## 6. App wiring and docs

- [x] 6.1 Wire the engine, session factory and `SqlAlchemyStudyRepository` in `main.py` via a `lifespan` that disposes the engine, keeping an optional repository override for tests (design D9); verify `/health` reports `database_reachable: true` with docker-compose Postgres up and `false` with it stopped
- [x] 6.2 Update `docs/deployment.md` and `app/backend/.env.example` (if present) with migration steps, the Supabase URL choice and `DB_STATEMENT_CACHE_SIZE=0` for the pooler; verify the documented commands run as written locally

## 7. Integration check

- [ ] 7.1 Run the full backend suite with `DATABASE_URL` pointing to docker-compose Postgres and confirm coverage stays at or above the CI gate (80%); verify the GitHub Actions `tests.yml` backend job passes on the branch

## Workflow follow-up

- Open the PR to `develop` with the `create-pull-request` skill (`Closes #116`).
- Archive the change with `/opsx:archive` after merge.
