# Pinned runtime dependencies retained; changed source qualification is pending.
# No data/model/source snapshots enter the image.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:e711c99333fdfe8ae1e677b4972be6c5021f0128a1d31f775c7e58d88921b6a9

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12=3.12.3-1ubuntu0.17 python3.12-venv=3.12.3-1ubuntu0.17 \
      ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && python3.12 -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GDAL_PAM_ENABLED=NO \
    PROJ_NETWORK=OFF \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MPLBACKEND=Agg

COPY requirements/linux-height-window.txt requirements/linux-height-window.constraints.txt /opt/spec/requirements/
COPY containers/window-worker-requirements.txt /opt/spec/containers/
RUN python -m pip install --no-cache-dir --only-binary=:all: \
      -r /opt/spec/containers/window-worker-requirements.txt \
 && python -m pip check

# Keep the previously verified DEP/GIS pins isolated from inference dependencies.
# Both interpreters are built here; neither is a host environment mount.
COPY requirements/dep-window.txt requirements/dep-window.constraints.txt /opt/spec/requirements/
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && python3.12 -m venv /opt/adapter-venv \
 && /opt/adapter-venv/bin/python -m pip install --no-cache-dir pip==24.0 \
 && /opt/adapter-venv/bin/python -m pip install --no-cache-dir \
      --only-binary=:all: -r /opt/spec/requirements/dep-window.txt \
 && /opt/adapter-venv/bin/python -m pip check

WORKDIR /opt/processor
COPY carbon/*.py ./carbon/
COPY tasks/*.py ./tasks/
COPY tasks/inputs/*.py ./tasks/inputs/
COPY configs/scientific-profile.json ./configs/
COPY release.json ./release.json
ENV SPC_WORKER_PYTHON=/opt/venv/bin/python
USER 65534:65534
ENTRYPOINT ["/opt/adapter-venv/bin/python", "-u", "-B", "-m", "tasks"]
