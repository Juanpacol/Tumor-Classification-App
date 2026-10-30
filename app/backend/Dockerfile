FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ml_engine comes from the "ml_engine" build context (see docker-compose.yml or
# `docker build --build-context ml_engine=app/ml_engine app/backend`).
# The source is bind-mounted, not copied, so the dataset under src/data never
# lands in a layer; only the installed package stays in the image.
COPY --from=ml_engine requirements.lock /tmp/ml_engine-requirements.lock
RUN pip install --no-cache-dir -r /tmp/ml_engine-requirements.lock
RUN --mount=type=bind,from=ml_engine,target=/tmp/ml_engine,rw \
    pip install --no-cache-dir --no-deps /tmp/ml_engine

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY . .

EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
