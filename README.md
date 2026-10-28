# Ether - Hybrid Brain Tumor Detection Support System

A Clinical Decision Support System (CDSS) leveraging hybrid machine learning (classical radiomics + deep learning) to provide AI-assisted second opinions on brain MRI scans for radiologists in resource-limited environments.

## 🎯 Project Goal

Augment radiologist expertise with interpretable, explainable AI to improve diagnostic accuracy and efficiency in decentralized health networks, particularly across Latin America.

## 🏗 Architecture

The system is built on three independent, modular components:

- **Backend**: FastAPI service layer + task orchestration (Celery + Redis)
- **ML Engine**: Standalone hybrid ML pipeline (radiomics + CNN)
- **Frontend**: Clinical portal (Next.js + Cornerstone3D for DICOM visualization)

See [CLAUDE.md](./CLAUDE.md) for detailed architecture and team guidelines.

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local development)
- Node.js 18+ (for frontend development)
- Azure DevOps CLI: `az login`

### Setup

```bash
# Clone repository
git clone https://dev.azure.com/juanpabloorg/Ether/_git/Ether
cd Ether

# Start all services
docker-compose up -d

# Services available at:
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Frontend: http://localhost:3000
# - MinIO Console: http://localhost:9001
# - PostgreSQL: localhost:5432
# - Redis: localhost:6379
```

## 📋 Contributing

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for:
- Branch naming convention: `AB-{ISSUE_ID}-{description}`
- Commit format standards
- Pull request process
- Code style requirements

### Workflow Summary

```bash
# 1. Create branch from issue
git checkout -b AB-88-your-feature

# 2. Make changes and commit
git commit -m "feat(backend): your description #88"

# 3. Push and create PR
git push -u origin AB-88-your-feature
```

## 📁 Project Structure

```
Ether/
├── app/
│   ├── backend/                 # FastAPI application
│   │   ├── app/
│   │   │   ├── api/             # HTTP route controllers
│   │   │   ├── services/        # Business logic
│   │   │   ├── clients/         # External integrations
│   │   │   ├── models/          # Database models
│   │   │   ├── schemas/         # Pydantic DTOs
│   │   │   ├── core/            # Configuration
│   │   │   └── tasks.py         # Celery workers
│   │   └── Dockerfile
│   │
│   ├── ml_engine/                # ML Pipeline (standalone)
│   │   ├── src/
│   │   │   ├── preprocessing/   # DICOM handling & preprocessing
│   │   │   ├── classical/       # Radiomics extractors (GLCM, LBP, Wavelet)
│   │   │   ├── deep_learning/   # CNN backbone + Fusion + Grad-CAM
│   │   │   ├── pipelines/       # End-to-end inference
│   │   │   └── tests/           # Unit tests
│   │   └── weights/              # Pre-trained model files
│   │
│   └── frontend/                 # Next.js Clinical Portal
│       ├── src/
│       │   ├── app/             # Pages and routes
│       │   ├── features/        # Feature-sliced modules
│       │   │   ├── dicom_viewer/ # Medical visualization
│       │   │   ├── clinical_summary/ # Metrics display
│       │   │   └── llm_assistant/ # Generative chat
│       │   ├── services/        # API client
│       │   └── hooks/           # Shared React hooks
│       └── package.json
│
├── docker-compose.yml         # Local orchestration
├── CONTRIBUTING.md            # Contribution guidelines
├── CLAUDE.md                  # Team guidelines & documentation
└── README.md                  # This file
```

## 🔧 Development

### Backend Development

```bash
cd app/backend
pip install -r requirements.txt
pytest  # Run tests
python main.py  # Run locally
```

### ML Engine Development

```bash
cd app/ml_engine
pip install -r requirements.txt
pytest  # Run tests
```

### Frontend Development

```bash
cd app/frontend
npm install
npm run dev
# Open http://localhost:3000
```

## 📚 Documentation

- [CLAUDE.md](./CLAUDE.md) - Team guidelines, architecture details, debugging
- [CONTRIBUTING.md](./CONTRIBUTING.md) - Git workflow, code standards, testing requirements
- [Architecture Report](./docs/architecture.md) - Detailed system design (TBD)

## 🛠 Technologies

- **Backend**: FastAPI, SQLAlchemy, Pydantic, Celery, Redis
- **ML**: PyTorch, scikit-image, SimpleITK, MONAI
- **Frontend**: Next.js, TypeScript, React Hooks, Cornerstone3D, TailwindCSS
- **Data**: PostgreSQL, MinIO (S3-compatible)
- **Deployment**: Docker, Docker Compose, Azure DevOps

## 📋 System Architecture Overview

```
Clinician           FastAPI Router       Celery Worker        ML Engine       Meta Llama API
   │                     │                    │                   │                │
   ├─ Upload MRI ───────>│                    │                   │                │
   │                     ├─ Strip PHI         │                   │                │
   │                     ├─ Queue Task ───────>│                   │                │
   │<─ Return Task ID ───│                    │                   │                │
   │                     │                    ├─ Run Pipeline ────>│                │
   │                     │                    │<─ Predictions ─────│                │
   │                     │                    ├─ Generate Synthesis────────────────>│
   │                     │<─ Save Results ────│<─ Clinical Report ─────────────────│
   │                     │                    │                   │                │
   └─ View Results ──────>│
        & Chat ────────────>│──────────────────────────────────────>│
                           │<─────────────────────────────────────│
```

## ⚠️ Important Notes

- This system is designed as a **Clinical Decision Support System (CDSS)** - it augments but does NOT replace specialist evaluation
- All outputs are explainable via Grad-CAM++ heatmaps
- PHI (Protected Health Information) is stripped before processing
- Offline deployment possible with on-premise quantized Llama model

## 🤝 Team

See Azure DevOps Boards for current team members and assignments.

## 📞 Support

- **Issues**: Use Azure DevOps → Boards
- **Questions**: Comment on relevant tasks
- **Critical Issues**: Slack #ether-dev

---

**Last Updated**: October 26, 2024