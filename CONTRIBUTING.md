# Contributing Guide

## Where Work Lives

This project uses two platforms for two different jobs — don't mix them up:

- **Azure Boards** (`dev.azure.com/juanpabloorg/Ether`): ticket tracking only. Every unit of work starts here as a work item (`AB-{ID}`).
- **GitHub** (`github.com/Juanpacol/Tumor-Classification-App`): the actual repository. All code review, pull requests, and merges happen here.

There is no PR in Azure DevOps anymore. Azure Boards work items don't auto-close from a GitHub PR merge (no native integration is installed), so the work item is closed manually once the GitHub PR merges. If this gets painful, installing the "Azure Boards" GitHub App on the repo would close that gap, but it requires admin access to the GitHub repo.

## Branch and Commit Conventions

### Branch Format (MANDATORY)
```
AB-{TICKET_ID}-{short-description}
```

**Rule**: Branch name must start with `AB-{TICKET_ID}` to link all work to the Azure DevOps ticket.

**Examples:**
- `AB-88-setup-project-structure`
- `AB-92-implement-dicom-viewer`
- `AB-100-fix-auth-endpoint`
- `AB-89-setup-github-actions-gates`

### Commit Format (MANDATORY)
```
AB-{TICKET_ID}: type(scope): description

Extended description if necessary.
```

**Rule**: Every commit MUST start with `AB-{TICKET_ID}:` to maintain visibility and traceability with the ticket. This links all work to the Azure DevOps ticket.

**Allowed Types:**
- `feat`: New functionality
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring (no functional changes)
- `test`: Add/modify tests
- `chore`: Build, dependencies, etc.

**Mandatory Scopes:**
- `backend`: Backend API changes
- `ml_engine`: ML pipeline changes
- `frontend`: Frontend changes
- `infra`: Infrastructure, CI/CD, Docker, configuration

**Commit Examples (CORRECT):**
```bash
# Working on branch: AB-88-setup-project-structure
git commit -m "AB-88: feat(backend): implement upload endpoint"
git commit -m "AB-88: fix(ml_engine): correct GLCM normalization"
git commit -m "AB-89: ci(infra): implement GitHub Actions gates"
git commit -m "AB-92: docs(frontend): add component documentation"
git commit -m "AB-100: refactor(backend): simplify auth middleware"
```

**Commit Examples (WRONG - will fail validation):**
```bash
git commit -m "add new feature"  # Missing AB-X: prefix, type, scope
git commit -m "feat: fix bug"  # Missing AB-X: prefix and scope
git commit -m "feat(backend): add endpoint"  # Missing AB-X: prefix
git commit -m "AB-88: implement feature"  # Missing type(scope)
git commit -m "88: feat(backend): add endpoint"  # Wrong prefix format
```

### Traceability
- **Work item** → created in Azure Boards (`AB-88`)
- **Branch** → `AB-88-description`
- **Commit** → `AB-88: type(scope): description`
- **PR** → opened in GitHub, titled `AB-88: Description`. "Closes #88" in the PR body is documentation only — it does not auto-close the Azure Boards work item (no GitHub issue #88 exists); close the work item manually after merge.
- **Result**: All code changes visible and traceable to a single ticket throughout git history

## Language

**Rule**: All implementation (code, commit messages, PR titles/descriptions, code comments, in-app copy) MUST be written in English. This is MANDATORY and applies regardless of the language used in Azure Boards discussions or team chat.

## Branching Model

- `develop`: QA branch. Every feature PR targets `develop`, never `main` directly.
- `main`: production. Updated only by a PR from `develop` once QA passes, which promotes all accumulated changes together (see [Promoting `develop` to `main`](#promoting-develop-to-main)).

## Pull Requests

1. **PR Title**: Same as branch name, in English
   ```
   AB-88: Setup project structure
   ```

2. **Description**: MANDATORY - Use the fixed structure from `.github/pull_request_template.md` (GitHub pre-fills it automatically on every new PR). Do not remove required sections:
   - What
   - How to test
   - Related ticket ("Closes #88")
   - Screenshots/links (if applicable, remove section otherwise)

3. **Base branch**: every PR targets `develop`. Only a hotfix targets `main`, and its title gets a `[HOTFIX]` prefix (e.g. `AB-120: [HOTFIX] Fix upload crash`); merge the fix back into `develop` afterward

4. **Reviewers**: Minimum 1 review before merge

5. **Merge Strategy**: MANDATORY - Use **Rebase and merge** ONLY for every PR into `develop`
   - This is the ONLY allowed merge strategy for feature PRs
   - Preserves all commits in linear history
   - All commits remain traceable to the ticket
   - No squashing or merge commits allowed on `develop`
   - The one exception is promoting `develop` to `main`, which uses a merge commit (see below)

## Promoting `develop` to `main`

A promotion is the only exception to the rebase-only rule. GitHub's "Rebase and merge" always rewrites commits with new SHAs, so using it would leave `develop` and `main` holding the same changes under different commits, and the next promotion would list old commits again. A merge commit keeps every SHA, so `develop` stays contained in `main`.

1. **Check that `main` has no commits of its own.** This must print nothing (merge commits of earlier promotions are excluded):
   ```bash
   git fetch github
   git log --no-merges --oneline github/develop..github/main
   ```
   Also check that every check on `develop` is green.
2. **Create a work item** for the promotion and open a PR from `develop` to `main` titled `AB-{ID}: Promote develop to main`. List the included PRs in the "What" section.
3. **Tag a backup of both branches** before merging, so their previous state can always be recovered:
   ```bash
   git tag backup/main-before-promotion-N github/main
   git tag backup/develop-before-promotion-N github/develop
   git push github backup/main-before-promotion-N backup/develop-before-promotion-N
   ```
4. **Wait for every check of the PR to pass.**
5. **Enable merge commits temporarily** (needs repository admin):
   - Settings → General → Pull Requests → check "Allow merge commits".
   - Settings → Branches → the rule for `main` → clear "Require linear history".
6. **Merge the PR with "Create a merge commit".** Never use "Rebase and merge" for a promotion.
7. **Restore both settings** from step 5: uncheck "Allow merge commits" and check "Require linear history" on the `main` rule again.
8. **Verify the result:**
   - `git rev-list --count github/main..github/develop` is `0`, so `develop` is contained in `main`.
   - `git rev-list --count github/develop..github/main` has grown by one: the merge commit. This is expected.
   - The Vercel production deployment is Ready, the new Cloud Run revision is healthy and its `/health` answers.
9. **Close the work item** in Azure Boards.

Do not merge `main` back into `develop`. The next promotion starts from the last `develop` commit that was promoted.

## Contributing Steps

```bash
# 1. Update develop branch (use main only for hotfixes)
git checkout develop
git pull origin develop

# 2. Create branch from issue
git checkout -b AB-{ID}-{description}

# 3. Make changes
# ... edit files ...

# 4. Commit with correct format
git add .
git commit -m "AB-{ID}: type(scope): description"

# 5. Push to branch
git push -u origin AB-{ID}-{description}

# 6. Create PR in GitHub (not Azure DevOps), base branch: develop
# github.com/Juanpacol/Tumor-Classification-App → Pull Requests → New

# 7. Once merged, close the matching Azure Boards work item manually
```

## Code Standards

### Backend (Python)
- Python 3.11+
- PEP 8 + Black formatter
- Type hints mandatory
- Docstrings for public functions
- Flat layout in `app/backend/` (`domain/`, `ports/`, `adapters/`, `application/`, `tests/`, `main.py`); imports have no `app.` prefix
- Every endpoint MUST be documented in Swagger/OpenAPI: `tags`, `summary`, `description` and `response_model` (plus `responses` for error codes). Enforced by `tests/test_api_documentation.py`; docs at `/docs`

### ML Engine (Python)
- Comments only for non-obvious WHY
- Unit tests for pipelines
- Reproducibility: fixed seeds and pinned versions

### Frontend (TypeScript/React)
- TypeScript strict mode
- ESLint + Prettier
- Functional components + hooks
- Props documented with JSDoc

## Testing

- Each feature requires tests
- Tests must pass before PR merge
- Minimum coverage: 80%

```bash
# Backend
cd app/backend && pytest

# ML Engine
cd app/ml_engine && pytest

# Frontend
cd app/frontend && npm test
```

## CI/CD

PRs automatically:
1. Run linting
2. Run tests
3. Generate coverage reports
4. Validate commit messages

All must pass before merge.
