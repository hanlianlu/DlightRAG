# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
# DlightRAG - multimodal RAG

ARG UV_VERSION=0.11.21

FROM python:3.14.7-slim-bookworm AS uv-bin
ARG UV_VERSION
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

# Resolve current upstream releases at build time. The installer rejects assets
# without GitHub-published SHA-256 provenance and enforces DlightRAG's static minima.
FROM python:3.14.7-slim-bookworm AS search-tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY src/dlightrag/engine/agent/environment/toolchain.py /tmp/toolchain.py
RUN python /tmp/toolchain.py install --allow-runtime-download \
    --cache-root /tmp/search-tool-cache --bin-dir /search-tools

# Match GitHub Actions so one npm version produces the same cross-platform lock behavior.
FROM node:26-slim AS frontend
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json frontend/
RUN --mount=type=cache,target=/root/.npm npm --prefix frontend ci
COPY frontend/ frontend/
# Vite writes HTML and hashed assets into ../src/dlightrag/adapters/http/browser/static/app, which the wheel picks up.
RUN npm --prefix frontend run build

FROM python:3.14.7-slim-bookworm AS builder

WORKDIR /app
ENV UV_LINK_MODE=copy
# Ship bytecode in the venv: without it every container (re)creation compiles each imported
# module from source (seconds on the api's first import, and a SyntaxWarning from lightrag.py).
ENV UV_COMPILE_BYTECODE=1
COPY --from=uv-bin /usr/local/bin/uv /bin/

COPY pyproject.toml uv.lock ./
COPY packages/memory/pyproject.toml packages/memory/pyproject.toml
# Deps only — binary-only (UV_NO_BUILD): never compile an sdist; the slim base has
# no toolchain, so a missing wheel fails fast. Keep it off the project build below.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 UV_NO_BUILD=1 uv sync --frozen --no-dev --no-install-workspace

COPY LICENSE NOTICE README.md ./
COPY packages/ packages/
COPY src/ src/
COPY --from=frontend /app/src/dlightrag/adapters/http/browser/static/app/ src/dlightrag/adapters/http/browser/static/app/
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 uv sync --frozen --no-dev --no-editable

# LightRAG's default tokenizer (tiktoken o200k_base via gpt-4o-mini) downloads its 3.6 MB
# encoding file from the public internet at every process start; bake it into the image.
RUN TIKTOKEN_CACHE_DIR=/opt/tiktoken /app/.venv/bin/python -c "import tiktoken; tiktoken.encoding_for_model('gpt-4o-mini')"

# Charts (built-in charts Skill): the upstream resvg CLI, built from its pinned crates.io
# release because upstream publishes no linux-aarch64 binary. The binary links its crates
# statically, so the license files of every crate the locked build used ship with it, one
# directory per crate.
FROM rust:1.99.0-slim-bookworm AS resvg
ARG RESVG_VERSION=0.48.1
RUN cargo install --locked resvg --version ${RESVG_VERSION} --root /out \
    && mkdir -p /out/licenses \
    && cd /usr/local/cargo/registry/src/*/ \
    && find . -mindepth 2 -maxdepth 2 -type f \
        \( -iname '*licen[cs]e*' -o -iname 'COPYING*' -o -iname 'NOTICE*' -o -iname 'COPYRIGHT*' \) \
        -exec cp --parents {} /out/licenses/ \;

# Charts: ECharts draws on the image's node, so only echarts.min.js and its licenses ship.
FROM node:26-slim AS chart-render
WORKDIR /build
COPY chart-render/package.json chart-render/package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund \
    && mkdir -p /out/licenses \
    && cp node_modules/echarts/dist/echarts.min.js /out/ \
    && cp node_modules/echarts/LICENSE node_modules/echarts/NOTICE node_modules/echarts/licenses/LICENSE-d3 /out/licenses/ \
    && cp node_modules/zrender/LICENSE /out/licenses/LICENSE-zrender
COPY chart-render/echarts_render.py chart-render/ssr.cjs chart-render/theme.json /out/
RUN chmod 755 /out/echarts_render.py

# Charts: Noto Sans SC (OFL-1.1), pinned to one noto-cjk commit and checked by digest. A remote
# ADD arrives 0600, unreadable to app, and ADD --chmod leaves its parent directory closed, so the
# files are opened here before the final stage copies them.
FROM python:3.14.7-slim-bookworm AS chart-fonts
ARG NOTO_CJK=https://raw.githubusercontent.com/notofonts/noto-cjk/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Sans
ADD --checksum=sha256:faa6c9df652116dde789d351359f3d7e5d2285a2b2a1f04a2d7244df706d5ea9 ${NOTO_CJK}/SubsetOTF/SC/NotoSansSC-Regular.otf /fonts/NotoSansSC-Regular.otf
ADD --checksum=sha256:c6cb5a93abaa9edc8ee7463b7ebb7f42d618d40e6ed2f7a5371c97b0b64767c0 ${NOTO_CJK}/SubsetOTF/SC/NotoSansSC-Bold.otf /fonts/NotoSansSC-Bold.otf
ADD --checksum=sha256:6a73f9541c2de74158c0e7cf6b0a58ef774f5a780bf191f2d7ec9cc53efe2bf2 ${NOTO_CJK}/LICENSE /fonts/LICENSE
RUN chmod 755 /fonts && chmod 644 /fonts/*

FROM python:3.14.7-slim-bookworm
LABEL maintainer="HanlianLyu"

WORKDIR /app

# Only the node binary is copied, so install the one library it links that the
# slim base lacks (libatomic1); without it every node process fails to start.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates libatomic1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=frontend /usr/local/bin/node /usr/local/bin/node
COPY --from=search-tools /search-tools/fd /search-tools/rg /usr/local/bin/
# Charts: echarts-render draws with ECharts on the node above and rasterizes with resvg, in the
# one font the image ships.
COPY --from=resvg /out/bin/resvg /usr/local/bin/resvg
COPY --from=resvg /out/licenses/ /usr/local/share/doc/resvg/
COPY --from=chart-fonts /fonts/ /usr/local/share/fonts/noto-sans-sc/
COPY --from=chart-render /out/ /usr/local/lib/echarts-render/
RUN ln -s /usr/local/lib/echarts-render/echarts_render.py /usr/local/bin/echarts-render
# Create non-root user BEFORE copying files to avoid chown layer duplication
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home app \
    && mkdir -p /app/dlightrag_storage /home/app/.dlightrag/agent_workspaces \
    /home/app/.dlightrag/skills /home/app/.dlightrag/owner_skills \
    && chown app:app /app/dlightrag_storage /home/app/.dlightrag/agent_workspaces \
    /home/app/.dlightrag/skills /home/app/.dlightrag/owner_skills

COPY --from=builder --chown=app:app /app/.venv /app/.venv
# The tiktoken encoding file baked in the builder stage: root-owned, read-only for app (it
# never writes it). TIKTOKEN_CACHE_DIR points every process of the image at it.
COPY --from=builder /opt/tiktoken /opt/tiktoken

ENV PATH="/app/.venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

EXPOSE 8100 8101

USER app

# Charts need node, resvg, ECharts and the font together as the app user: fail the build, not a Run.
RUN printf '%s' '{"title":{"text":"冒烟测试 Smoke"},"xAxis":{"type":"category","data":["甲","乙"]},"yAxis":{},"series":[{"type":"bar","data":[1,2]}]}' \
    | echarts-render - /tmp/chart-smoke.png && test -s /tmp/chart-smoke.png && rm /tmp/chart-smoke.png

# Default image role; deployments can override it for MCP or maintenance commands.
CMD ["dlightrag-api"]
