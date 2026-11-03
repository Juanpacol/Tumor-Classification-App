# Training image: only the locked dependencies. Code, data, weights and artifacts
# are mounted from the host (docker-compose.yml, services "trainer" and "notebook"),
# so editing code or a round file needs no rebuild; a requirements.lock change does.
FROM python:3.11-slim AS trainer

COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock

WORKDIR /work
ENTRYPOINT ["python", "-m"]
CMD ["src.training", "--help"]

# Notebook image: the training image plus JupyterLab and matplotlib, constrained by
# the lock so no engine dependency changes version.
FROM trainer AS notebook

RUN pip install --no-cache-dir -c /tmp/requirements.lock \
    jupyterlab==4.6.5 matplotlib==3.11.2

EXPOSE 8888
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.root_dir=/work"]
CMD []
