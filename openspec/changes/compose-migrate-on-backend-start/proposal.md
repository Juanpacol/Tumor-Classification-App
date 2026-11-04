# Proposal

## Why

`docker compose up` starts the backend against an empty database, so every upload fails with a 500 (`relation "studies" does not exist`) until someone runs `docker compose exec backend alembic upgrade head` by hand. The step is easy to miss: on 2026-11-03 classification looked broken even with the model loaded. The local stack should be ready to use once `docker compose up` finishes (Azure #218).

## What Changes

- The compose `backend` service applies `alembic upgrade head` before starting uvicorn, on every start. Migrations are idempotent, so later starts are a no-op when the schema is already at head.
- If the migration fails, the backend container does not start serving and exits with the error in its logs. It never serves on a partially migrated schema.
- The manual "first run only" migration step is removed from the README quick start, and `docs/deployment.md` describes the new local behaviour.
- No change to the Cloud Run / Supabase path. It keeps its dedicated `Migrate` step in `cloudbuild.yaml` and the image's default `CMD` stays as it is. The application code still does not create the schema at startup.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `study-persistence`: adds a requirement that the local compose environment brings the schema to head before the backend serves requests.

## Impact

- `docker-compose.yml`: `backend.command`.
- `README.md` (quick start) and `docs/deployment.md` (Migrations → Local).
- No API, dependency or application-code changes. Existing local databases that are already at head are unaffected. A database created from an older revision is upgraded automatically on the next `docker compose up`.
