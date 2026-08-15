# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
# DlightRAG - multimodal RAG

ARG UV_VERSION=0.11.21

FROM python:3.14-slim-bookworm AS uv-bin
ARG UV_VERSION
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

# Match GitHub Actions so one npm version produces the same cross-platform lock behavior.
FROM node:26-slim AS frontend
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json frontend/
RUN --mount=type=cache,target=/root/.npm npm --prefix frontend ci
COPY frontend/ frontend/
# Vite writes into ../src/dlightrag/web/static/generated, which the wheel picks up.
RUN npm --prefix frontend run build

FROM python:3.14-slim-bookworm AS builder

WORKDIR /app
ENV UV_LINK_MODE=copy
COPY --from=uv-bin /usr/local/bin/uv /bin/

COPY pyproject.toml uv.lock ./
COPY packages/ai/pyproject.toml packages/ai/pyproject.toml
COPY packages/agent-core/pyproject.toml packages/agent-core/pyproject.toml
COPY packages/rag-core/pyproject.toml packages/rag-core/pyproject.toml
# Deps only — binary-only (UV_NO_BUILD): never compile an sdist; the slim base has
# no toolchain, so a missing wheel fails fast. Keep it off the project build below.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 UV_NO_BUILD=1 uv sync --frozen --no-dev --no-install-workspace

COPY LICENSE NOTICE README.md ./
COPY packages/ packages/
COPY src/ src/
COPY --from=frontend /app/src/dlightrag/web/static/generated/ src/dlightrag/web/static/generated/
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

EXPOSE 8100 8101

USER app

# Default image role; deployments can override it for MCP or maintenance commands.
CMD ["dlightrag-api"]
