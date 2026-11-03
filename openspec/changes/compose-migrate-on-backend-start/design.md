# Design

## Context

- The compose `backend` service runs `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`, with `app/backend` bind-mounted at `/app`. Alembic (`alembic.ini`, `migrations/`) is already installed in the image and reads `DATABASE_URL` through `settings.py`. `docker compose exec backend alembic upgrade head` works today.
- The backend already waits for `postgres` to be `service_healthy`, so the database accepts connections when the backend command runs.
- Cloud Run migrates in its own `Migrate` step in `cloudbuild.yaml`, and the image's `CMD` is shared with that deployment. `docs/deployment.md` states the schema is not created on app startup.

## Goals / Non-Goals

**Goals:**
- Run the migration inside the backend container's start command, so that `docker compose up`, `docker compose restart backend` and container recreation all migrate.

**Non-Goals:**
- Changing the Dockerfile `CMD`, Cloud Run or Supabase migration flow.
- Creating the schema from application code (FastAPI lifespan, `metadata.create_all`).
- Re-running migrations on uvicorn `--reload` file changes. A new migration file needs a backend restart.

## Decisions

**1. Migrate in the compose `backend.command`, not in the image.**
The command becomes `sh -c "alembic upgrade head && exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload"`. Compose does not run a string `command` through a shell, so `&&` needs the explicit `sh -c`. `exec` makes uvicorn PID 1, so `docker compose stop` still sends SIGTERM straight to it.
- *Alternative: change the Dockerfile `CMD`.* Rejected. The image is also the Cloud Run image, where migrating on each instance start would race between instances and run against the Supabase pooler. The `Migrate` build step already covers it.
- *Alternative: a one-shot `migrate` service that `backend` depends on with `service_completed_successfully`.* Rejected. `docker compose restart backend` does not re-run dependencies, so a restart after pulling a new migration would skip it. That is the case the request targets. It also adds a service for one command.
- *Alternative: migrate in the FastAPI lifespan.* Rejected. It mixes schema management into application code, contradicts the documented rule, and would also run on Cloud Run.

**2. Fail closed.**
With `&&`, a failing `alembic upgrade head` ends the container with a non-zero exit before uvicorn binds the port. The Alembic error is already printed to stderr, which is the container's logs. The service has no `restart:` policy, so it stays down and visible in `docker compose ps` instead of looping.

**3. Verify on a scratch database, not the developer's data.**
Checking the empty-database case must not require `docker compose down -v`, which would wipe the developer's stored studies. The tasks use a throwaway database in the same Postgres container and point a one-off backend container at it.

## Risks / Trade-offs

- [Startup is a little slower, since Alembic connects on every start] → It is a no-op at head and takes about a second.
- [A broken migration blocks the local backend entirely] → This is intended (spec: does not serve when migration fails). Logs name the failing revision, and `alembic downgrade` stays available through `docker compose run`.
- [Two developers' branches add different heads] → Alembic refuses to upgrade with multiple heads and the backend fails closed. Same as today's manual step, to be resolved with a merge revision.
- [The behaviour differs between local (auto) and Cloud Run (build step)] → Documented in `docs/deployment.md`. Both run the same `alembic upgrade head`.
