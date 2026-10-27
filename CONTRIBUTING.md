# Contributing Guide

## Branch and Commit Conventions

### Branch Format
```
AB-{ISSUE_ID}-{short-description}
```

**Examples:**
- `AB-88-setup-project-structure`
- `AB-92-implement-dicom-viewer`
- `AB-100-fix-auth-endpoint`

### Commit Format
```
type(scope): description #ISSUE_ID

Extended description if necessary.
```

**Allowed Types:**
- `feat`: New functionality
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring (no functional changes)
- `test`: Add/modify tests
- `chore`: Build, dependencies, etc.

**Suggested Scopes:**
- `backend`
- `ml_engine`
- `frontend`
- `infra`

**Examples:**
```bash
git commit -m "feat(backend): implement upload endpoint #88"
git commit -m "fix(ml_engine): correct GLCM normalization #92"
git commit -m "docs: update README with setup instructions #88"
```

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
