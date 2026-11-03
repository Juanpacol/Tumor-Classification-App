# Deployment Plan (Hackathon)

Where each part of the system runs for the hackathon demo, and how it differs from local development. This is the demo deployment, not a production architecture — see `CLAUDE.md` for the full system design.

## Services

Two free accounts, one service each:

| Layer | Service | Why |
|---|---|---|
| Frontend (Next.js) | **Vercel** | Native Next.js support, auto-deploy from GitHub, free |
| Backend + ML inference (FastAPI + PyTorch) | **Google Cloud Run** (Docker container, continuous deploy via Developer Connect) | Usage-based free tier with configurable memory (2-4GB), scales to zero when idle, no fixed monthly cost under quota |
| Database | **Supabase** (Postgres, free tier) | Managed Postgres, 500MB storage — plenty for relational metadata (studies, predictions, reports); actual scan files are not stored here |

Rejected options and why:
- **Hugging Face Spaces** (Docker SDK): originally chosen for its RAM headroom, but Docker Spaces now require a paid PRO subscription — no longer free as of when we checked.
- **Render** (free tier): only 512MB RAM. Loading PyTorch + an EfficientNet backbone risks OOM under any concurrent request — too risky right before a live demo.
- **Railway**: no longer has a real free tier (trial credit only as of the services we checked).
- **AWS/Azure free tiers**: both require the same card-verification hold as GCP, and their free compute (AWS EC2 t2/t3.micro, 1GB RAM) has the same RAM ceiling problem as Render. No advantage over Cloud Run for this workload.
- **Two different database engines (SQLite local, Postgres cloud)**: rejected. SQLite and Postgres behave differently (types, foreign key enforcement, concurrency), so code that works locally against SQLite can break only once deployed to Postgres. One engine everywhere removes that risk entirely.

Cloud Run setup notes:
- Deployed via "Connect repository" (Developer Connect), which auto-builds `app/backend/Dockerfile` and redeploys on every push to `main` — no manual deploy step.
- The GitHub App for Developer Connect can only be installed by the repo owner/admin, which is why the repo was transferred to a personal GitHub account rather than staying under a collaborator's account.
- The container must listen on the `$PORT` env var Cloud Run injects (defaults to 8080) — `app/backend/Dockerfile`'s `CMD` uses `${PORT:-8080}`, not a hardcoded port.
- Region: `us-south1` (closest to Colombia with good free-tier coverage).

Vercel setup notes:
- Project's **Root Directory** must be set to `app/frontend` (Settings → General) — the repo root isn't a Next.js project.
- Same GitHub App permission requirement as Cloud Run: Vercel needs explicit access to the repo (Settings → Git → Connect Git Repository), which only works smoothly once the repo is under an account you administer.
- **Deployment Protection** (`vercel_authentication`) is on by default on new projects and blocks public access behind a login. Disabled it (Settings → Deployment Protection, or `vercel project protection disable --sso`) so judges/public can open the demo without a Vercel account.
- `NEXT_PUBLIC_API_URL` must be set for every environment you'll actually view (at least Production; add Preview/Development too if you'll check PR preview deployments).

## Live demo URLs

- Frontend: https://ether-self-five.vercel.app
- Backend: Cloud Run service `ether-backend` (copy the current Endpoint URL from the Cloud Run console — it can change if the service is ever recreated)

## Database strategy

One engine, two places:

- **Local development**: the `postgres:15-alpine` container already defined in `docker-compose.yml`. No changes.
- **Cloud demo**: Supabase (same Postgres engine, hosted). Only the `DATABASE_URL` connection string changes between environments — no code or driver changes. The backend uses async SQLAlchemy with `asyncpg`; a plain `postgresql://` URL (as Supabase and docker-compose provide it) is rewritten to `postgresql+asyncpg://` by `app/backend/settings.py`.

Supabase free-tier note: projects pause after 7 days of inactivity and need a manual restore from the dashboard. Touch the project (or restore it) the day before presenting.

### Which Supabase connection string

Supabase offers three connection strings (Project → Connect):

| Option | Port | Use it? | Notes |
|---|---|---|---|
| Direct connection | 5432 | No | IPv6-only on the free tier; Cloud Run egress is IPv4 by default |
| **Session pooler** | 5432 | **Yes** (app + migrations) | IPv4, behaves like a normal connection |
| Transaction pooler | 6543 | Only with `DB_STATEMENT_CACHE_SIZE=0` | PgBouncer transaction mode breaks asyncpg's prepared-statement cache |

### Migrations

The schema is created by Alembic (`app/backend/migrations/`); the application code never creates it. Only the local compose `backend` service runs the migration on start (below); Cloud Run instances do not, and rely on the `Migrate` step of `cloudbuild.yaml` and the manual Supabase step.

```bash
cd app/backend
alembic upgrade head          # create/upgrade the schema
alembic downgrade base        # drop it (rollback)
```

- **Local (compose)**: the `backend` service runs `alembic upgrade head` before uvicorn on every start (`up`, `restart`, recreate). It is a no-op at head; if the migration fails the container exits before serving and `docker compose logs backend` shows the error.
- **Local (backend on the host)**: `docker compose up -d postgres`, then run `alembic upgrade head` from `app/backend` (the default `DATABASE_URL` points to the compose database).
- **Supabase**: Cloud Run redeploys on every push to `main`, so run the migration **before** merging a change that adds one: `DATABASE_URL=<Supabase session pooler URL> alembic upgrade head` from a local checkout. The Cloud Run service's own `DATABASE_URL` env var must point at the same session pooler connection string, or `/health` reports `database_reachable: false`. Ran for the first time 2026-10-30 (AB-186), after the Cloud Run build was fixed in AB-184.

### Tests

Run the backend suite in compose with the `test` profile, not in the running `backend` container:

```bash
docker compose run --rm --build backend-test
```

It builds the `test` stage of `app/backend/Dockerfile` (with the `dev` extra, so `pytest-asyncio` is there) and starts a throwaway `postgres-test` database kept in memory. `docker compose exec backend pytest` is not supported: that image has no dev tools.

Backend integration tests (`app/backend/tests/integration/`) run against the PostgreSQL at `TEST_DATABASE_URL` (never SQLite, never `DATABASE_URL`). They migrate it to head and truncate the tables between tests, so they refuse a database whose name does not contain `test`: the compose `ether_db` and Supabase are never touched. Without `TEST_DATABASE_URL`, or without a reachable database, they are skipped with the reason shown in the pytest summary (`pytest -rs`). CI sets `TEST_DATABASE_URL` to its own `test_db`.

The unit tests clear `LLM_PROVIDER`, `META_MODEL_API_KEY` and `MODEL_BUNDLE_PATH` before each test (`app/backend/tests/conftest.py`), so the compose settings do not change their results.

## What's simplified for the cloud demo vs local dev

`docker-compose.yml` is the local dev stack: Postgres + backend + frontend. The cloud demo runs two services (frontend on Vercel, backend on Cloud Run) plus Supabase.

Neither environment uses Celery, Redis or S3/MinIO: inference runs synchronously inside the request, and uploaded scans are kept on the backend's filesystem (`STORAGE_DIR`). On Cloud Run that is not persistent across restarts or scale-to-zero, which is acceptable for a demo, not for production.

## Environment variables (cloud demo)

```
DATABASE_URL=<Supabase session pooler connection string>
DB_STATEMENT_CACHE_SIZE=   # set to 0 only if using the transaction pooler (port 6543)
ENVIRONMENT=demo
NEXT_PUBLIC_API_URL=<Cloud Run service URL>
```

(No `REDIS_URL`, `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` in the cloud demo — those stay local-only.)
