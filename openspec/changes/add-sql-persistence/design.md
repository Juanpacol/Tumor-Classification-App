# Design

## Context

The backend follows hexagonal architecture (AB-115). `StudyRepositoryPort` is a sync Protocol with `get`/`save`/`ping`, implemented only by `InMemoryStudyRepository` and consumed only by the health router. `application/` has no use cases yet. `domain/` and `application/` are guarded by `tests/test_architecture_boundaries.py` (no `sqlalchemy`/`fastapi` imports). `sqlalchemy`, `asyncpg`, `psycopg2-binary` and `alembic` are already declared dependencies. `settings.database_url` defaults to SQLite, which contradicts `docs/deployment.md` ("one engine everywhere"). CI (`.github/workflows/tests.yml`) already provides a `postgres:15-alpine` service and passes a plain `postgresql://` URL, as does Supabase.

See proposal.md for motivation; requirements live in `specs/study-persistence/spec.md`.

## Goals / Non-Goals

**Goals:**
- One async repository adapter covering the whole study aggregate.
- Domain model shaped by the AB-116 acceptance criteria, still framework-free.
- Identical code path for docker-compose Postgres, CI Postgres and Supabase.

**Non-Goals:**
- Use cases, HTTP endpoints, Celery tasks or file storage.
- Deleting studies through the port (deletion only matters at DB level for cascade).
- Read models, pagination or search over studies.

## Decisions

### D1. One aggregate port (option A) instead of a port per entity
`StudyRepositoryPort` grows to:

```
async get(study_id) -> Study | None
async save(study) -> None
async ping() -> bool
async save_prediction(prediction) -> None
async get_prediction(study_id) -> Prediction | None
async save_report(report) -> None
async get_report(study_id) -> Report | None
async add_message(message) -> None
async list_messages(study_id) -> list[ChatMessage]
```

Predictions, reports and chat messages cannot exist without a study (FK, 1:1 / 1:N), so they form one aggregate. The AB-116 criteria name only `StudyRepositoryPort`, and one port means one fake.
*Alternative*: separate `Prediction`/`Report`/`ChatRepositoryPort`. Better cohesion per entity, but four adapters and four fakes for no current consumer.

### D2. Async port with SQLAlchemy 2 `AsyncSession` + asyncpg
FastAPI handlers are async and asyncpg is already a dependency. Converting now touches only the fake and the health router.
*Alternative*: sync psycopg2 session in the threadpool. Simpler, but it would block threads and contradicts the US.

### D3. `Report` wraps `ClinicalReport` (option 1)
New domain types:

```
class ReportStatus(str, Enum): PENDING, COMPLETED, FAILED

@dataclass
class Report:
    study_id: str
    status: ReportStatus
    model: str
    created_at: datetime
    content: ClinicalReport | None = None
    error: str | None = None
```

`ClinicalReport` stays unchanged so it remains the structured-output schema for `LLMPort.complete_json`. It is stored as JSONB in `reports.content`.
*Alternative*: flatten lifecycle fields into `ClinicalReport`. Rejected because every content field would become optional and partially built reports would be easy to create.

### D4. Domain model changes
- `Study`: `image_path` becomes `file_key`; add `original_format: str`, `clinical_notes: str | None`, `rejection_reason: str | None`.
- `Prediction`: add `quality: dict[str, Any]` and `top_features: list[dict[str, Any]]`. Both are JSON-compatible and opaque to persistence; the ML engine owns their inner shape.
- IDs remain `str` in the domain. The adapter converts to and from `uuid.UUID` so the domain stays free of DB types.
- `ChatMessage.id` stays a UUID string and becomes the `chat_messages` primary key.

### D5. Schema

```
studies         id UUID PK, created_at timestamptz, status varchar CHECK,
                original_format varchar, file_key varchar,
                clinical_notes text NULL, rejection_reason text NULL
predictions     study_id UUID PK FK->studies ON DELETE CASCADE,
                label varchar, probabilities JSONB, confidence double,
                needs_expert_triage bool, quality JSONB, top_features JSONB,
                model_version varchar
reports         study_id UUID PK FK->studies ON DELETE CASCADE,
                status varchar CHECK, content JSONB NULL, model varchar,
                error text NULL, created_at timestamptz
chat_messages   id UUID PK, study_id UUID FK->studies ON DELETE CASCADE,
                role varchar, content text, created_at timestamptz
                INDEX (study_id, created_at)
```

- **1:1 via `study_id` as PK.** This enforces "at most one" in the database, and saves become upserts (`INSERT ... ON CONFLICT (pk) DO UPDATE`).
- **Status as varchar + CHECK, not a native enum.** Native Postgres enums require `ALTER TYPE` migrations to add values, which Alembic autogenerate handles poorly.
- **JSONB for semi-structured ML/LLM output** that the backend does not query by field.
- **`timestamptz` everywhere.** The domain uses timezone-aware UTC datetimes.

### D6. Adapter structure
`adapters/outbound/persistence/sqlalchemy/` contains:
- `orm.py`: declarative models and `MetaData` with a naming convention.
- `mappers.py`: pure ORM <-> domain functions.
- `repository.py`: `SqlAlchemyStudyRepository`, which takes an `async_sessionmaker` and opens one session/transaction per method.
- `engine.py`: builds the `AsyncEngine` from settings.

ORM instances never cross the adapter boundary (spec: no ORM leakage), and mappers are unit-testable without a database. `ping()` runs `SELECT 1`; it catches errors and returns `False`, so `/health` never fails because of the database.

### D7. Connection URL normalization and pooler support
`settings.py` exposes `async_database_url`, which rewrites `postgresql://` / `postgres://` to `postgresql+asyncpg://`. The default `database_url` becomes the docker-compose Postgres URL, and SQLite is dropped. A setting `db_statement_cache_size` (default unset, meaning the asyncpg default) is passed through `connect_args`. It is set to `0` when using Supabase's transaction-mode pooler (port 6543), because PgBouncer transaction mode breaks asyncpg prepared statements.
*Alternative*: always disable the cache. That costs performance locally for no benefit.

### D8. Alembic
- `app/backend/alembic.ini` and `app/backend/migrations/` use an async `env.py` that reads `async_database_url` from settings, never from `alembic.ini`.
- `target_metadata` comes from `orm.py`.
- The initial revision is hand-reviewed after autogenerate, including CHECK constraints, the index and `downgrade()`.
- Migrations run as an explicit step (`alembic upgrade head`): locally via a docker-compose command or manually, and for Cloud Run once before deploying. They do not run on app startup, which avoids races between concurrently starting instances.

### D9. App wiring
`main.py` builds the engine and session factory inside a FastAPI `lifespan` and disposes the engine on shutdown. Tests keep using `InMemoryStudyRepository` through the same factory, so `create_app` accepts an optional repository override.

### D10. Testing
- **Unit**: the async in-memory fake and the mappers. Both run everywhere with no DB.
- **Integration**: marked `@pytest.mark.integration` and placed in `tests/integration/`. They run against the Postgres at `DATABASE_URL`.
  - A session-scoped fixture runs `alembic upgrade head`.
  - Each test truncates the four tables.
  - Tests are skipped with a clear reason when no Postgres is reachable. They never fall back to SQLite.
- **Coverage**: CI already provides Postgres, so integration tests count toward the 80% gate there.
- `pytest-asyncio` (already a dev dependency) drives async tests.

## Risks / Trade-offs

- **[Breaking port and domain changes]** → Only the health router and the fake consume them today. Updating both is part of this change.
- **[Supabase pooler breaks prepared statements]** → `db_statement_cache_size=0` via env (D7). `docs/deployment.md` documents which Supabase URL to use.
- **[Integration tests silently skipped locally]** → The skip reason is printed. CI always has Postgres, so it remains the source of truth.
- **[JSONB fields lose schema guarantees]** → Domain dataclasses define the shape. Mapper tests cover the round-trip.
- **[Forgetting to run migrations before deploy]** → Documented as a deploy step. `/health` still reports the DB as reachable, so a missing table surfaces as a 500 on first write. That is acceptable for the hackathon.

## Migration Plan

1. Merge with the new initial migration. No existing data needs migrating because persistence was in-memory.
2. Local: `docker-compose up -d postgres`, then `alembic upgrade head` from `app/backend`.
3. Supabase: set `DATABASE_URL` (plus `DB_STATEMENT_CACHE_SIZE=0` if using the pooler) and run `alembic upgrade head` once before deploying to Cloud Run.
4. Rollback: `alembic downgrade base` drops the four tables, then redeploy the previous image.
