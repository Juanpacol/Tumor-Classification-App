# Ether - Hybrid Brain Tumor Detection Support System

A Clinical Decision Support System (CDSS) prototype that combines classical radiomics and deep learning to give an AI-assisted second opinion on brain MRI scans, aimed at radiologists in resource-limited environments.

> **Hackathon prototype, not a medical device.** It has not been validated on clinical data and must not be used to make clinical decisions. It is designed to support a specialist, never to replace one.

## 📌 Project status

Reviewed on 2026-10-29. This table says what exists today, not what is planned.

| Area | Works today | Not built yet |
|---|---|---|
| **ML engine** | JPG/PNG loader, image quality gate, 45 radiomic features (GLCM, LBP, wavelet, intensity), frozen EfficientNet-B0 embeddings, a training pipeline (cross-validation, calibration, report) and a serving pipeline. All tested on **synthetic** data | A model trained on the real dataset (the dataset is not in this repository), so **there are no accuracy figures yet**. Grad-CAM explanations. DICOM input |
| **Backend** | FastAPI service with a `/health` endpoint and API docs, hexagonal structure, PostgreSQL schema managed with Alembic, a client for the Meta Model API (Muse Spark) that is not yet connected to any endpoint | Study upload and classification endpoints, report and chat endpoints, de-identification. `/health` reports `model_loaded: true` because the service still uses a fake classifier |
| **Frontend** | Landing page and the "how we built it" page, deployed on Vercel | The clinician portal, the MRI viewer (Cornerstone3D is installed but no viewer exists), the results panel and the chat |
| **Infrastructure** | CI on GitHub Actions (tests, type checks, commit format). Frontend on Vercel, backend on Google Cloud Run, database on Supabase | Automatic database migrations on deploy |

Live frontend: https://ether-self-five.vercel.app

## 🏗 Architecture

Three independent components:

- **Backend** (`app/backend`): FastAPI with a hexagonal (ports and adapters) structure.
- **ML engine** (`app/ml_engine`): a standalone package with the hybrid pipeline (radiomics + CNN).
- **Frontend** (`app/frontend`): Next.js web app.

Planned request flow (the backend does not implement it yet):

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

Prerequisites: Python 3.11+, Node.js 18+ and Docker (only for the local database).

### Backend

```bash
docker compose up -d postgres        # PostgreSQL on localhost:5432
cd app/backend
pip install -e ".[dev]"
cp .env.example .env
alembic upgrade head                 # create the schema
uvicorn main:app --reload            # http://localhost:8000, docs at /docs
pytest                               # integration tests need a reachable PostgreSQL
```

If port 5432 is already taken by a local PostgreSQL, change the host port in `docker-compose.yml` and in `DATABASE_URL`.

Do not run the whole stack with `docker compose up` yet. The compose file still defines Redis, MinIO and a Celery worker that the backend no longer uses, and the worker points to a module that does not exist.

### ML engine

```bash
cd app/ml_engine
pip install -r requirements.lock
pytest src/tests
```

Exporting the backbone weights, training and serving are described in [app/ml_engine/README.md](./app/ml_engine/README.md).

### Frontend

```bash
cd app/frontend
npm install
npm run dev                          # http://localhost:3000
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
- No real patient data is processed or stored today. Only JPG and PNG images are supported. De-identification and DICOM support are planned, not implemented.
- Explanations such as Grad-CAM heatmaps are planned, not implemented.

## 📜 Third-party assets and licenses

The code in this repository is released under the [MIT License](./LICENSE). That license does not cover the third-party assets below, which keep their own terms.

Ether is a hackathon prototype. These are the external assets it depends on and what is known about their terms. "Not verified" means nobody on the team has checked them yet, not that they are permissive.

| Asset | Used for | Status |
|---|---|---|
| EfficientNet-B0 weights pretrained on ImageNet-1K, from [torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html) (`IMAGENET1K_V1`) | Frozen CNN feature extractor | **Terms not verified.** They are never committed to git. The [ImageNet terms of access](https://image-net.org/download) limit the dataset to non-commercial research and education, and whether trained weights inherit that is unresolved. Check before any commercial use |
| [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Masoud Nickparvar, Kaggle) | Training and evaluation | **License not confirmed.** A secondary source lists it as CC0, but we have not checked the Kaggle page, and the dataset combines three public datasets (figshare, SARTAJ and Br35H) whose own terms also matter. It is not committed to git: download it from Kaggle yourself |
| Meta Model API (Muse Spark) | Clinical report and chat | Used through the API with a key from an environment variable. Subject to Meta's terms. **While free credits are in use, Meta may use the content sent for product improvement**, so only public dataset images and invented notes are sent (see `app/backend/README.md`) |
| Python and JavaScript dependencies | Everything | Open-source packages pinned in `app/ml_engine/requirements.lock` and the backend and frontend manifests. Their licenses were not audited |
| Drawn skull and surgical instrument illustrations (`app/frontend/public/portal-skull.png`, `portal-instrument.png`) | Fixed background behind the study list | Owned by the team; cleared for use in this repository |
| Drawn skeletal hand, torso and bird illustrations (`app/frontend/public/portal-brain.png`, `portal-heart.png`, `portal-web.png`) | Fixed background behind each analysis tab | Owned by the team; cleared for use in this repository |

Every model artifact records where its weights came from (`backbone_provenance` in its `manifest.json`). Details are in `docs/ml-engine/decisions.md` (D9, D14).
