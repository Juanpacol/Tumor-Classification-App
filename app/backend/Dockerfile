FROM python:3.11-slim

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

EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
