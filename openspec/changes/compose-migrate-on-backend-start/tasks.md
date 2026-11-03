# Tasks

Work on branch `AB-218-compose-migrate-on-backend-start` from an up-to-date `develop`.

## 1. Migrate on backend start

- [x] 1.1 In `docker-compose.yml`, change `backend.command` to `sh -c "alembic upgrade head && exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload"` (design, decision 1). Verify with `docker compose config` that it parses and shows the new command.
- [x] 1.2 Add `app/backend/tests/test_compose_migrations.py`. It reads the repo-root `docker-compose.yml` and asserts that the `backend` command runs `alembic upgrade head` before `uvicorn`, chained with `&&`. Skip it when the file is not present, for example inside the backend container, which mounts only `app/backend`. Use PyYAML only if it is already a backend dependency; otherwise check the raw text. Verify that `pytest tests/test_compose_migrations.py` passes, and that it fails when the command is reverted.
- [x] 1.3 Verify the empty-database scenario without touching the developer's data. Create a scratch database: `docker compose exec postgres createdb -U ether_user ether_verify`. Run `docker compose run --rm -e DATABASE_URL=postgresql://ether_user:ether_password@postgres:5432/ether_verify backend` until uvicorn reports startup, then stop it. Confirm `\dt` in `ether_verify` lists studies, predictions, reports and chat_messages. Drop `ether_verify` afterwards.
- [x] 1.4 Verify the restart and fail-closed scenarios. After `docker compose restart backend`, the logs show Alembic running with no upgrade applied, `/health` reports `database_reachable: true`, and the studies uploaded earlier are still listed by `GET /api/studies`. Run the same one-off container with a wrong password in `DATABASE_URL`. It exits non-zero, uvicorn never starts, and the Alembic or asyncpg error is in the output.

## 2. Documentation

- [x] 2.1 In `README.md` (Quick start → Run the whole project), remove the `docker compose exec backend alembic upgrade head` line and note that the backend applies migrations on start. Verify the quick start works as written on a clean volume, which is covered by 1.3.
- [x] 2.2 In `docs/deployment.md` (Migrations), keep "not created on app startup" for application code. Rewrite the **Local** bullet: compose runs `alembic upgrade head` before uvicorn on every backend start and fails closed; running the backend on the host still needs the manual command. Leave the Supabase/Cloud Run bullet unchanged. Verify that both docs describe the same behaviour.

## Workflow follow-up

- Commit with `AB-218: ...` messages and open the PR against `develop` (`create-commit`, `create-pull-request`).
- After merge, move Azure #218 to Done by hand and archive this change (`/opsx:archive`).
