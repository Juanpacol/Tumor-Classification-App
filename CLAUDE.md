# CLAUDE.md - Team Guidelines

## Project Overview

**Project**: Hybrid Brain Tumor Detection Support System
**Stack**: FastAPI + PyTorch + Next.js + Cornerstone3D
**Type**: Clinical Decision Support System (CDSS)
**Goal**: Provide AI-assisted second opinion on brain MRI scans for radiologists in resource-limited environments.

---

## Project Structure

```
Ether/
├── backend/              # FastAPI + Services
├── ml_engine/            # Standalone ML Pipeline
├── frontend/             # Clinical Portal (Next.js)
├── docker-compose.yml    # Local Orchestration
└── docs/                 # Project Documentation
```

---

## Environment Setup

### Prerequisites

1. **Azure DevOps CLI**
   ```bash
   az login
   az devops configure --defaults organization=https://dev.azure.com/juanpabloorg project=Ether
   ```

2. **Git with correct branch naming**
   - Always: `AB-{ISSUE_ID}-{description}`
   - See CONTRIBUTING.md

3. **Docker & Docker Compose**
   ```bash
   docker --version
   docker-compose --version
   ```

### Initial Setup

```bash
# 1. Clone repository
git clone https://dev.azure.com/juanpabloorg/Ether/_git/Ether
cd Ether

# 2. Create branch from user story
git checkout -b AB-{ID}-{description}

# 3. Start local environment
docker-compose up -d
```

---

## Workflow

### 1. Create User Story (US)

In Azure DevOps:
1. https://dev.azure.com/juanpabloorg/Ether
2. **Boards** → **New Work Item** → **Task**
3. Complete: Title, Description, Acceptance Criteria
4. Copy the **ID** (ex: `#88`)

### 2. Create Branch

```bash
git checkout main
git pull origin main
git checkout -b AB-88-feature-name
```

### 3. Implement Feature

Edit files according to architecture. Respect:
- Type hints in backend
- Tests for each component
- Frequent commits with correct format

### 4. Commit & Push

```bash
git add .
git commit -m "type(scope): description #ID"
git push -u origin AB-88-feature-name
```

**Format is mandatory**: See CONTRIBUTING.md

### 5. Create Pull Request

1. Azure DevOps → **Repos** → **Pull Requests** → **New**
2. Title: `AB-88: Feature name`
3. Description includes: What, How to test, Screenshots
4. Add: `Closes #88`
5. Requires minimum 1 review

### 6. Merge

Only after:
- ✅ All tests pass
- ✅ Approved by at least 1 reviewer
- ✅ No conflicts

---

## Architecture

### Backend

**Pattern**: Service-Client (FastAPI)

Layers:
- `app/api/`: HTTP routes (thin controllers)
- `app/services/`: Business logic
- `app/clients/`: External integrations (S3, Llama)
- `app/models/`: SQLAlchemy ORM
- `app/schemas/`: Pydantic DTOs

**Standards**:
- Type hints mandatory
- Docstrings for public functions
- Pytest for tests
- PEP 8 + Black

### ML Engine

**Pattern**: Standalone package (importable, testable)

Modules:
- `preprocessing/`: DICOM preparation
- `classical/`: Radiomics features (GLCM, LBP, Wavelet)
- `deep_learning/`: CNN + Fusion + Grad-CAM
- `pipelines/`: End-to-end orchestration

**Standards**:
- Reproducibility (fixed seeds, pinned versions)
- No obvious comments
- Unit tests per extractor
- Timestamped outputs

### Frontend

**Pattern**: Feature-Sliced Architecture (Next.js)

Modules:
- `features/dicom_viewer/`: Medical visualization (Cornerstone3D)
- `features/clinical_summary/`: Metrics and confidence
- `features/llm_assistant/`: Generative chat

**Standards**:
- TypeScript strict mode
- Functional components + hooks
- ESLint + Prettier
- JSDoc for public props

---

## Key Decisions (Don't Change Without Consensus)

1. **Hybrid ML Architecture**: Radiomics + CNN = better generalization on heterogeneous datasets
2. **FastAPI over Django**: Speed, native async, automatic OpenAPI
3. **Next.js over raw React**: SSR, routing, performance
4. **Cornerstone3D**: Standard for DICOM visualization
5. **Llama (OSS)**: Data sovereignty, no vendor lock-in
6. **Hackathon deployment**: Vercel (frontend) + Hugging Face Spaces (backend/ML) + Supabase (Postgres) — see `docs/deployment.md` for the full plan and what's simplified vs local dev

---

## Debugging

### Backend

```bash
cd backend
python -m pytest tests/ -v
python main.py  # run server
```

Logs in `backend/logs/` (if applicable)

### ML Engine

```bash
cd ml_engine
pytest src/ -v
python -c "from src.pipelines import InferencePipeline; ..." # test import
```

### Frontend

```bash
cd frontend
npm run dev
# http://localhost:3000
```

Browser DevTools: Network, Console, React Profiler

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Device not configured" (Azure CLI) | `az login` |
| Branch conflicts | `git fetch origin && git rebase origin/main` |
| Docker container crash | `docker-compose logs -f {service}` |
| ML model not loading | Check `ml_engine/weights/` exists |
| Cornerstone3D black screen | Check canvas size + WebGL support |

---

## Communication

- **Critical issues**: Slack #ether-dev
- **Design reviews**: Azure Boards comments
- **Questions**: Wiki docs (TBD)

---

## Resources

- Architecture spec: `/docs/architecture.md` (TBD)
- API docs: `http://localhost:8000/docs` (FastAPI auto-docs)
- DICOM standards: `MEDICAL_REFERENCES.md` (TBD)
