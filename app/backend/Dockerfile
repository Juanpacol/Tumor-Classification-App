FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ml_engine comes from the "ml_engine" build context (see docker-compose.yml or
# `docker build --build-context ml_engine=app/ml_engine app/backend`).
COPY --from=ml_engine requirements.lock pyproject.toml /opt/ml_engine/
RUN pip install --no-cache-dir -r /opt/ml_engine/requirements.lock
COPY --from=ml_engine src /opt/ml_engine/src
RUN pip install --no-cache-dir --no-deps /opt/ml_engine

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY . .

EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
