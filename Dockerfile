# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
# DlightRAG - multimodal RAG

ARG UV_VERSION=0.11.21

FROM python:3.14-slim-bookworm AS uv-bin
ARG UV_VERSION
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

FROM python:3.14-slim-bookworm AS builder

WORKDIR /app
ENV UV_LINK_MODE=copy
COPY --from=uv-bin /usr/local/bin/uv /usr/local/bin/uvx /bin/

COPY pyproject.toml uv.lock ./
# Deps only — binary-only (UV_NO_BUILD): never compile an sdist; the slim base has
# no toolchain, so a missing wheel fails fast. Keep it off the project build below.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 UV_NO_BUILD=1 uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 uv sync --frozen --no-dev --no-editable

FROM python:3.14-slim-bookworm
LABEL maintainer="HanlianLyu"

WORKDIR /app

# Create non-root user BEFORE copying files to avoid chown layer duplication
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home app \
    && mkdir -p /app/dlightrag_storage && chown app:app /app/dlightrag_storage

COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

VOLUME /app/dlightrag_storage

EXPOSE 8100 8101

USER app

# Default: start the REST API server
CMD ["dlightrag-api"]
