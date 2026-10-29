---
name: create-commit
description: Create git commits that follow this project's mandatory format `AB-{ID}: type(scope): description`. Use when the user asks to commit, save changes to git, or says "create commit". Derives the ticket ID from the branch name and splits unrelated changes into separate commits.
allowed-tools: Bash(git:*)
---

Create commits following `CONTRIBUTING.md` and the `validate-commits` CI workflow.

## Format (mandatory)

```
AB-{TICKET_ID}: type(scope): description

Optional extended description.
```

- **Ticket ID**: taken from the branch name (`AB-116-persistence-with-sql` -> `AB-116`). If the branch does not start with `AB-{number}`, ask the user for the ticket; never invent one.
- **type**: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`. (CI regex rejects anything else, e.g. `ci`.)
- **scope**: `backend`, `ml_engine`, `frontend`, `infra`. Lowercase letters/underscores only.
- **description**: English, imperative, lowercase start, no trailing period, subject line <= ~72 chars.
- All commit text MUST be in English, even if the user speaks Spanish.

## Steps

1. Run in parallel: `git status --short`, `git diff HEAD --stat`, `git branch --show-current`, `git log -5 --oneline`.
2. Extract the ticket ID from the branch.
3. Group changes by scope and intent. Changes touching different scopes (e.g. `app/backend` and `.github/` or `docker-compose.yml`) or unrelated purposes become separate commits. Mapping: `app/backend/**` -> `backend`, `app/ml_engine/**` -> `ml_engine`, `app/frontend/**` -> `frontend`, everything else (CI, Docker, config, docs about infra) -> `infra`.
4. Stage files explicitly by path (`git add <paths>`), never `git add .` / `-A`. Do not stage secrets (`.env`, credentials), `__pycache__`, or build artifacts; warn the user if you see them.
5. Commit with a heredoc so the body formats correctly:
   ```bash
   git commit -m "$(cat <<'MSG'
   AB-116: refactor(backend): flatten app package into backend root

   Optional body explaining why.

   Co-Authored-By: <attribution line from the session's system reminder, if present>
   MSG
   )"
   ```
6. Verify each subject against the CI regex `^AB-[0-9]+: (feat|fix|docs|refactor|test|chore)\([a-z_]+\):.+`. If a hook fails, fix the cause and create a new commit; never use `--no-verify` or amend unless asked.
7. Finish with `git log --oneline -n <count>` and `git status --short` and report the commits made.

## Rules

- Do NOT push; pushing is a separate, explicit request (see `create-pull-request`).
- Do not commit on `main`; if on `main`, stop and ask which branch to create (`AB-{ID}-{description}`).
- Renames/moves: use `git mv` or stage both sides so Git records them as renames.
