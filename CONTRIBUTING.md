# Contributing Guide

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
type(scope): description #TICKET_ID

Extended description if necessary.
```

**Rule**: Every commit must reference the ticket ID with `#TICKET_ID` at the end. This links the commit to the Azure DevOps ticket automatically.

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

**Commit Examples:**
```bash
# Working on branch: AB-88-setup-project-structure
git commit -m "feat(backend): implement upload endpoint #88"
git commit -m "fix(ml_engine): correct GLCM normalization #88"
git commit -m "ci(infra): implement GitHub Actions gates #89"
git commit -m "docs(frontend): add component documentation #92"

# WRONG (will fail validation):
git commit -m "add new feature"  # Missing format and ticket
git commit -m "feat: fix bug"  # Missing scope and ticket
git commit -m "feat(backend): add endpoint"  # Missing ticket #
```

### Traceability
- **Branch** → Ticket ID (AB-88)
- **Commit** → References same Ticket ID (#88)
- **PR** → Links to Ticket ("Closes #88")
- **Result**: All code changes traceable to a single ticket

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

4. **Merge**: Use "Squash commit" for feature branches

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
git commit -m "type(scope): description #ID"

# 5. Push to branch
git push -u origin AB-{ID}-{description}

# 6. Create PR in Azure DevOps
# Go to Repos → Pull Requests → New Pull Request
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
cd backend && pytest

# ML Engine
cd ml_engine && pytest

# Frontend
cd frontend && npm test
```

## CI/CD

PRs automatically:
1. Run linting
2. Run tests
3. Generate coverage reports
4. Validate commit messages

All must pass before merge.
