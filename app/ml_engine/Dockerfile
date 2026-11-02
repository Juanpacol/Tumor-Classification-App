# Training image: only the locked dependencies. Code, data, weights and artifacts
# are mounted from the host (docker-compose.yml, service "trainer"), so editing
# code or a round file needs no rebuild; a requirements.lock change does.
FROM python:3.11-slim

COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock

WORKDIR /work
ENTRYPOINT ["python", "-m"]
CMD ["src.training", "--help"]
