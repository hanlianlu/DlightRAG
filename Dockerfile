# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
# DlightRAG - multimodal RAG

ARG UV_VERSION=0.11.21

FROM python:3.14-slim-bookworm AS uv-bin
ARG UV_VERSION
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

FROM python:3.14-slim-bookworm AS builder

WORKDIR /app
COPY --from=uv-bin /usr/local/bin/uv /usr/local/bin/uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 uv sync --frozen --no-dev --no-install-project

FROM python:3.14-slim-bookworm
LABEL maintainer="HanlianLyu"

WORKDIR /app
COPY --from=uv-bin /usr/local/bin/uv /usr/local/bin/uvx /bin/

# Runtime utilities for Git-based dependencies and TLS.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user BEFORE copying files to avoid chown layer duplication
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home app \
    && mkdir -p /app/dlightrag_storage && chown app:app /app/dlightrag_storage

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app pyproject.toml uv.lock README.md ./
COPY --chown=app:app src/ src/

RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

VOLUME /app/dlightrag_storage

EXPOSE 8100 8101

USER app

# Default: start the REST API server
CMD ["dlightrag-api"]
