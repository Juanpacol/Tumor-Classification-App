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

## Pull Requests

1. **PR Title**: Same as branch name
   ```
   AB-88: Setup project structure
   ```

2. **Description**: Include:
   - What was implemented
   - How to test it
   - Screenshots/links (if applicable)
   - Issue linking: "Closes #88"

3. **Reviewers**: Minimum 1 review before merge

4. **Merge Strategy**: MANDATORY - Use **Rebase and merge** ONLY
   - This is the ONLY allowed merge strategy
   - Preserves all commits in linear history
   - All commits remain traceable to the ticket
   - No squashing or merge commits allowed

## Contributing Steps

```bash
# 1. Update main branch
git checkout main
git pull origin main

# 2. Create branch from issue
git checkout -b AB-{ID}-{description}

# 3. Make changes
# ... edit files ...

# 4. Commit with correct format
git add .
git commit -m "AB-{ID}: type(scope): description"

# 5. Push to branch
git push -u origin AB-{ID}-{description}

# 6. Create PR in GitHub (not Azure DevOps)
# github.com/Juanpacol/Tumor-Classification-App → Pull Requests → New

# 7. Once merged, close the matching Azure Boards work item manually
```

## Code Standards

### Backend (Python)
- Python 3.11+
- PEP 8 + Black formatter
- Type hints mandatory
- Docstrings for public functions

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
