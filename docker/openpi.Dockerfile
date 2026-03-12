# Dockerfile for OpenPI environment management with uv + editable source bind mount.
#
# Build:
# docker build . -t openpi-env -f docker/openpi.Dockerfile
#
# Run:
# docker run --rm -it --network=host --gpus all -v $PWD:/app openpi-env /bin/bash
#
# The image installs OpenPI in editable mode at /app during build, then ships only
# the virtual environment. At runtime, you must bind-mount the repository to /app.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder
SHELL ["/bin/bash", "-c"]

COPY --from=ghcr.io/astral-sh/uv:0.5.6 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    build-essential \
    clang \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Keep the project environment outside /app since /app is expected to be bind-mounted.
ENV HF_HOME=/opt/.cache/huggingface
ENV UV_HTTP_TIMEOUT=120
ENV UV_LINK_MODE=hardlink
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV UV_PYTHON_INSTALL_DIR=/.uv/python
ENV UV_CACHE_DIR=/var/cache/uv
RUN mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

WORKDIR /app

COPY . /app

# Match README installation flow inside the container as closely as possible.
# README:
#   GIT_LFS_SKIP_SMUDGE=1 uv sync
#   GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
RUN uv venv --python 3.11.9 "$UV_PROJECT_ENVIRONMENT"
RUN --mount=type=cache,target=/var/cache/uv \
    GIT_LFS_SKIP_SMUDGE=1 uv sync
RUN --mount=type=cache,target=/var/cache/uv \
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
RUN uv cache prune

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS runtime
SHELL ["/bin/bash", "-c"]

COPY --from=ghcr.io/astral-sh/uv:0.5.6 /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /.venv /.venv
COPY --from=builder /.uv /.uv

ENV HF_HOME=/opt/.cache/huggingface
ENV UV_HTTP_TIMEOUT=120
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV UV_PYTHON_INSTALL_DIR=/.uv/python
ENV UV_LINK_MODE=hardlink
ENV UV_CACHE_DIR=/var/cache/uv
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV PATH="/.venv/bin:${PATH}"
WORKDIR /app

RUN mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
RUN echo "source /.venv/bin/activate" >> /root/.bashrc

CMD ["/bin/bash"]
