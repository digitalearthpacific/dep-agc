# Supply the pinned model at runtime.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:e711c99333fdfe8ae1e677b4972be6c5021f0128a1d31f775c7e58d88921b6a9

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12=3.12.3-1ubuntu0.17 python3.12-venv=3.12.3-1ubuntu0.17 \
      ca-certificates git libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && python3.12 -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GDAL_PAM_ENABLED=NO \
    PROJ_NETWORK=OFF \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY requirements.txt /opt/spec/requirements.txt
RUN python -m pip install --no-cache-dir pip==24.0 \
 && python -m pip install --no-cache-dir --only-binary=:all: -r /opt/spec/requirements.txt \
 && python -m pip check

WORKDIR /opt/processor
RUN install -d -o 65534 -g 65534 models
COPY src/*.py ./src/

# dep-tools records the build identity in output STAC Items.
ARG DOCKER_IMAGE
ARG COMMIT
ENV IMAGE_TAG=${DOCKER_IMAGE} \
    COMMIT=${COMMIT} \
    IMAGE_REPO=${COMMIT:+https://github.com/digitalearthpacific/dep-agc/tree/${COMMIT}/}

USER 65534:65534
ENTRYPOINT ["/opt/venv/bin/python", "-u", "-B", "-m", "src.run_task"]
