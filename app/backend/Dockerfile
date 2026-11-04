FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Built with the repo root as context (see docker-compose.yml and
# docs/deployment.md), so ml_engine is a plain COPY from a sibling directory.
# The root .dockerignore excludes app/ml_engine/src/data, so the dataset
# never lands in a layer; only the installed package stays in the image.
COPY app/ml_engine/requirements.lock /tmp/ml_engine-requirements.lock
RUN pip install --no-cache-dir -r /tmp/ml_engine-requirements.lock
COPY app/ml_engine /tmp/ml_engine
RUN pip install --no-cache-dir --no-deps /tmp/ml_engine && rm -rf /tmp/ml_engine

COPY app/backend/pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY app/backend .

# Backend test suite with the dev extra (pytest-asyncio); see docs/deployment.md.
FROM base AS test
RUN pip install --no-cache-dir -e ".[dev]"

# Last stage, so the default build (Cloud Run) never carries the dev tools.
FROM base AS runtime

# The deployed model: downloaded from its GitHub Release and checked against
# the published SHA-256 before extraction (docs/deployment.md, Model).
ARG MODEL_VERSION=0.1.0
RUN python -m src.release fetch ${MODEL_VERSION} --dest /models \
    && rm /models/model-v${MODEL_VERSION}.tar.gz
ENV MODEL_BUNDLE_PATH=/models/model-v${MODEL_VERSION}

EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
