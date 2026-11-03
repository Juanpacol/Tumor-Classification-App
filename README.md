# Ether - Hybrid Brain Tumor Detection Support System

A Clinical Decision Support System (CDSS) prototype that combines classical radiomics and deep learning to give an AI-assisted second opinion on brain MRI scans, aimed at radiologists in resource-limited environments.

> **Hackathon prototype, not a medical device.** It has not been validated on clinical data and must not be used to make clinical decisions. It is designed to support a specialist, never to replace one.

## 📌 Project status

Reviewed on 2026-11-02. This table says what exists today, not what is planned.

| Area | Works today | Not built yet |
|---|---|---|
| **ML engine** | JPG/PNG loader, quality gate with calibrated thresholds, brain mask, 45 radiomic features, frozen EfficientNet-B0 embeddings, training pipeline (grouped cross-validation, calibration, shortcut audit) and serving pipeline | A published model that loads with the current engine, so **no accuracy figures yet**. Grad-CAM. DICOM input |
| **Backend** | Upload (JPG/PNG up to 20 MB, metadata removed), classification, study list, detail and image, clinical report with verified citations, streamed chat (Muse Spark, with a spend cap), Alembic schema, end-to-end test | A published model (without one, classification answers 503). DICOM and its de-identification |
| **Frontend** | Landing, "how we built it" and the clinician portal: study list, upload with checks, image viewer, clinical summary, report with citations, chat | DICOM viewing. Explanation heatmap |
| **Infrastructure** | CI on pull requests and on pushes to `develop` and `main` (tests, type checks; commit format on pull requests). Vercel, Cloud Run and Supabase; the migration runs on every Cloud Run deploy | Backend deployed with a model. End-to-end demo checklist |

Live frontend: https://ether-self-five.vercel.app

## 🏗 Architecture

Three independent components:

- **Backend** (`app/backend`): FastAPI with a hexagonal (ports and adapters) structure.
- **ML engine** (`app/ml_engine`): a standalone package with the hybrid pipeline (radiomics + CNN).
- **Frontend** (`app/frontend`): Next.js web app.

Request flow (needs a published model to classify):

```
Clinician            Backend (FastAPI)            ML engine             Muse Spark API
    │                       │                         │                       │
    ├─ Upload MRI ─────────>│                         │                       │
    │                       ├─ Classify ─────────────>│                       │
    │                       │<─ Prediction ───────────│                       │
    │                       ├─ Draft report ─────────────────────────────────>│
    │<─ Result + report ────│<────────────────────────────────────────────────│
```

See [CLAUDE.md](./CLAUDE.md) for the architecture rules and [docs/ml-engine/status.md](./docs/ml-engine/status.md) for how the ML pieces fit together.

## 🚀 Quick start

Prerequisites: Docker, Python 3.11+ and Node.js 20+ (Node 18 also works).

### Run the whole project

```bash
docker compose up -d --build                           # Postgres, backend, frontend
```

The backend applies the database migrations (`alembic upgrade head`) every time it starts; if they fail it does not start, and `docker compose logs backend` shows why.

| Check | Expect |
|---|---|
| http://localhost:8000/health | `database_reachable: true` |
| http://localhost:8000/docs | API docs |
| http://localhost:3000 | landing page; the portal is at `/studies` |

By default it runs with no model and no key:

| To get | Do |
|---|---|
| Real reports and chat | `export LLM_PROVIDER=meta META_MODEL_API_KEY=<key>` before `docker compose up`. Send only public dataset images, never patient data |
| Real classification | Put a trained bundle in `app/ml_engine/artifacts/current` (or set `MODEL_BUNDLE_DIR`). Without it `/health` says `model_loaded: false` and classification answers 503. Training: [app/ml_engine/README.md](./app/ml_engine/README.md) |
| Portal with no backend | `NEXT_PUBLIC_API_MODE=mock` in `app/frontend/.env.local` serves sample data |

Upgrading from the old compose file? `docker compose up -d --remove-orphans` removes the leftover Redis and MinIO containers.

Port 5432 taken? Create `docker-compose.override.yml` (git-ignored) with `services: {postgres: {ports: !override ["5436:5432"]}}`, and use that port in `DATABASE_URL` when running the backend on the host.

### Run each part on the host

```bash
# Backend (needs Postgres: docker compose up -d postgres)
cd app/backend && pip install -e ".[dev]" && cp .env.example .env
alembic upgrade head && uvicorn main:app --reload     # :8000
pytest                                                # integration tests need Postgres

# Frontend
cd app/frontend && npm install && cp .env.example .env.local
npm run dev                                           # :3000

# ML engine
cd app/ml_engine && pip install -r requirements.lock && pytest src/tests
```

## 📁 Project structure

```
Ether/
├── app/
│   ├── backend/                 # FastAPI service
│   │   ├── adapters/            # inbound HTTP and outbound (database, LLM, storage, classifier)
│   │   ├── application/         # use cases
│   │   ├── domain/              # entities and domain errors
│   │   ├── ports/               # interfaces the adapters implement
│   │   ├── migrations/          # Alembic migrations
│   │   ├── tests/
│   │   ├── main.py              # application factory
│   │   └── settings.py
│   │
│   ├── ml_engine/               # standalone ML package
│   │   └── src/
│   │       ├── preprocessing/   # loader, quality gate, skull stripping, deduplication
│   │       ├── classical/       # radiomic extractors
│   │       ├── deep_learning/   # frozen CNN extractor and weight bundles
│   │       ├── pipelines/       # serving pipeline
│   │       ├── training/        # training, selection, calibration, artifacts
│   │       └── tests/
│   │
│   └── frontend/                # Next.js app
│       └── src/
│           ├── app/             # pages and routes
│           ├── features/        # landing and story sections
│           └── hooks/
│
├── docs/                        # deployment plan and ML engine documentation
├── openspec/                    # specification changes
├── docker-compose.yml           # local services
├── CONTRIBUTING.md              # contribution guidelines
├── CLAUDE.md                    # team guidelines and architecture rules
└── LICENSE
```

## 📚 Documentation

- [CLAUDE.md](./CLAUDE.md): team guidelines and architecture rules
- [CONTRIBUTING.md](./CONTRIBUTING.md): Git workflow, code standards and testing requirements
- [docs/deployment.md](./docs/deployment.md): how it is deployed (Vercel, Cloud Run, Supabase)
- [docs/ml-engine/status.md](./docs/ml-engine/status.md): ML engine state and how to resume
- [docs/ml-engine/decisions.md](./docs/ml-engine/decisions.md): the decisions behind the ML engine and their evidence
- [app/ml_engine/DATASET_CLEANING_REPORT.md](./app/ml_engine/DATASET_CLEANING_REPORT.md): how the dataset was cleaned
- [app/backend/README.md](./app/backend/README.md): backend notes, including the Muse Spark integration

## 🛠 Technologies

- **Backend**: FastAPI, SQLAlchemy (async) with asyncpg, Alembic, Pydantic, httpx
- **ML engine**: PyTorch, torchvision, scikit-learn, scikit-image, PyWavelets, OpenCV, joblib
- **Frontend**: Next.js 14, React 18, TypeScript, Recharts
- **Data**: PostgreSQL (Supabase in the cloud)
- **Deployment**: Vercel, Google Cloud Run, Supabase
- **Tooling**: GitHub Actions for CI, Azure Boards for work items

## 🤝 Contributing

Read [CONTRIBUTING.md](./CONTRIBUTING.md). In short:

```bash
git checkout develop && git pull
git checkout -b AB-88-your-feature            # AB-{work item id}-{description}
git commit -m "AB-88: feat(backend): your description"
git push -u origin AB-88-your-feature         # then open a pull request against develop
```

Pull requests target `develop` (QA). Changes reach `main` only through a promotion from `develop`. Work items live in a private Azure Boards project; use GitHub issues for questions.

## ⚠️ Important notes

- Ether is a **Clinical Decision Support System** prototype: it is meant to augment a specialist's evaluation, never to replace it, and it is **not validated on clinical data**.
- No real patient data is processed or stored today. Only JPG and PNG images are supported, and their embedded metadata is removed on upload. DICOM support and its de-identification are planned, not implemented.
- Explanations such as Grad-CAM heatmaps are planned, not implemented.

## 📜 Third-party assets and licenses

The code in this repository is released under the [MIT License](./LICENSE). That license does not cover the third-party assets below, which keep their own terms.

Ether is a hackathon prototype. These are the external assets it depends on and what is known about their terms. "Not verified" means nobody on the team has checked them yet, not that they are permissive.

| Asset | Used for | Status |
|---|---|---|
| EfficientNet-B0 weights pretrained on ImageNet-1K, from [torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html) (`IMAGENET1K_V1`) | Frozen CNN feature extractor | **Terms not verified.** They are never committed to git. The [ImageNet terms of access](https://image-net.org/download) limit the dataset to non-commercial research and education, and whether trained weights inherit that is unresolved. Check before any commercial use |
| [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Masoud Nickparvar, Kaggle) | Training and evaluation | **License not confirmed.** A secondary source lists it as CC0, but we have not checked the Kaggle page, and the dataset combines three public datasets (figshare, SARTAJ and Br35H) whose own terms also matter. The team decided to commit it to `app/ml_engine` to track preprocessing steps, despite the license still being unconfirmed |
| Meta Model API (Muse Spark) | Clinical report and chat | Used through the API with a key from an environment variable. Subject to Meta's terms. **While free credits are in use, Meta may use the content sent for product improvement**, so only public dataset images and invented notes are sent (see `app/backend/README.md`) |
| Python and JavaScript dependencies | Everything | Open-source packages pinned in `app/ml_engine/requirements.lock` and the backend and frontend manifests. Their licenses were not audited |
| Drawn skull illustration (`app/frontend/public/portal-skull.png`) | Fixed background behind the study list | Owned by the team; cleared for use in this repository |
| Drawn hands with vintage scissors illustration (`app/frontend/public/portal-scissors-hands.png`) | Fixed background behind the study list | Owned by the team; cleared for use in this repository |
| Drawn skeletal hand, torso and bird illustrations (`app/frontend/public/portal-brain.png`, `portal-heart.png`, `portal-web.png`) | Fixed background behind each analysis tab | Owned by the team; cleared for use in this repository |
| Drawn skeleton back and IV drip stand illustrations (`app/frontend/public/story-skeleton-back.png`, `story-iv-drip.png`) | Backdrop on the how-we-built-it hero | Owned by the team; cleared for use in this repository |

Every model artifact records where its weights came from (`backbone_provenance` in its `manifest.json`). Details are in `docs/ml-engine/decisions.md` (D9, D14).
