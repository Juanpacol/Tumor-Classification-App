# Deployment Plan (Hackathon)

Where each part of the system runs for the hackathon demo, and how it differs from local development. This is the demo deployment, not a production architecture — see `CLAUDE.md` for the full system design.

## Services

Two free accounts, one service each:

| Layer | Service | Why |
|---|---|---|
| Frontend (Next.js) | **Vercel** | Native Next.js support, auto-deploy from GitHub, free |
| Backend + ML inference (FastAPI + PyTorch) | **Hugging Face Spaces** (Docker SDK, CPU) | Free CPU Spaces give far more RAM headroom (~16GB) than typical free app hosts (e.g. Render's 512MB), which matters once PyTorch and the model are loaded in memory |
| Database | **Supabase** (Postgres, free tier) | Managed Postgres, 500MB storage — plenty for relational metadata (studies, predictions, reports); actual scan files are not stored here |

Rejected options and why:
- **Render** (free tier): only 512MB RAM. Loading PyTorch + an EfficientNet backbone risks OOM under any concurrent request — too risky right before a live demo.
- **Railway**: no longer has a real free tier (trial credit only as of the services we checked).
- **Two different database engines (SQLite local, Postgres cloud)**: rejected. SQLite and Postgres behave differently (types, foreign key enforcement, concurrency), so code that works locally against SQLite can break only once deployed to Postgres. One engine everywhere removes that risk entirely.

## Database strategy

One engine, two places:

- **Local development**: the `postgres:15-alpine` container already defined in `docker-compose.yml`. No changes.
- **Cloud demo**: Supabase (same Postgres engine, hosted). Only the `DATABASE_URL` connection string changes between environments — no code or driver changes, since `backend/pyproject.toml` already uses `asyncpg` + `psycopg2-binary`, both standard Postgres drivers.

Supabase free-tier note: projects pause after 7 days of inactivity and need a manual restore from the dashboard. Touch the project (or restore it) the day before presenting.

## What's simplified for the cloud demo vs local dev

`docker-compose.yml` stays as the full local dev stack (Postgres + Redis + MinIO + backend + Celery worker + frontend) — unchanged.

For the cloud demo, these are dropped to keep the deployment to two services:

- **Celery + Redis**: inference runs synchronously inside the request instead of through an async task queue. A hackathon demo doesn't need to handle concurrent background jobs.
- **MinIO / S3 object storage**: uploaded scans are kept in the Hugging Face Space's container filesystem instead of a separate object store. Not persistent across container restarts — acceptable for a demo, not for production.

This means local dev and the cloud demo are not identical environments. If the backend code assumes Celery/Redis or S3 are always present, that assumption needs a feature flag or environment check so the same codebase runs both ways.

## Environment variables (cloud demo)

```
DATABASE_URL=<Supabase connection string>
ENVIRONMENT=demo
NEXT_PUBLIC_API_URL=<Hugging Face Space URL>
```

(No `REDIS_URL`, `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` in the cloud demo — those stay local-only.)
