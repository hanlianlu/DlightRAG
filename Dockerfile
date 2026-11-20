# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
# DlightRAG - multimodal RAG

ARG UV_VERSION=0.13.9

FROM python:3.14.10-slim-bookworm AS uv-bin
ARG UV_VERSION
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

# Resolve current upstream releases at build time. The installer rejects assets
# without GitHub-published SHA-256 provenance and enforces DlightRAG's static minima.
FROM python:3.14.10-slim-bookworm AS search-tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY src/dlightrag/engine/agent/environment/toolchain.py /tmp/toolchain.py
RUN python /tmp/toolchain.py install --allow-runtime-download \
    --cache-root /tmp/search-tool-cache --bin-dir /search-tools

# Match GitHub Actions so one npm version produces the same cross-platform lock behavior.
FROM node:26.13.1-slim AS frontend
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json frontend/
RUN --mount=type=cache,target=/root/.npm npm --prefix frontend ci
COPY frontend/ frontend/
# Vite writes HTML and hashed assets into ../src/dlightrag/adapters/http/browser/static/app, which the wheel picks up.
RUN npm --prefix frontend run build

FROM python:3.14.10-slim-bookworm AS builder

WORKDIR /app
ENV UV_LINK_MODE=copy
# Ship bytecode in the venv: without it every container (re)creation compiles each imported
# module from source (seconds on the api's first import, and a SyntaxWarning from lightrag.py).
ENV UV_COMPILE_BYTECODE=1
COPY --from=uv-bin /usr/local/bin/uv /bin/

COPY pyproject.toml uv.lock ./
COPY packages/memory/pyproject.toml packages/memory/pyproject.toml
COPY packages/mineru/pyproject.toml packages/mineru/pyproject.toml
COPY packages/observability/pyproject.toml packages/observability/pyproject.toml
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
FROM rust:1.100.0-slim-bookworm AS resvg
ARG RESVG_VERSION=0.50.0
RUN cargo install --locked resvg --version ${RESVG_VERSION} --root /out \
    && mkdir -p /out/licenses \
    && cd /usr/local/cargo/registry/src/*/ \
    && find . -mindepth 2 -maxdepth 2 -type f \
        \( -iname '*licen[cs]e*' -o -iname 'COPYING*' -o -iname 'NOTICE*' -o -iname 'COPYRIGHT*' \) \
        -exec cp --parents {} /out/licenses/ \;

# Charts: ECharts draws on the image's node, so only echarts.min.js and its licenses ship, beside
# the two commands (echarts-render, html-report), the one palette and house theme they share, and
# the report runtime.
FROM node:26.13.1-slim AS chart-render
WORKDIR /build
COPY chart-render/package.json chart-render/package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund \
    && mkdir -p /out/licenses \
    && cp node_modules/echarts/dist/echarts.min.js /out/ \
    && cp node_modules/echarts/LICENSE node_modules/echarts/NOTICE node_modules/echarts/licenses/LICENSE-d3 /out/licenses/ \
    && cp node_modules/zrender/LICENSE /out/licenses/LICENSE-zrender
COPY chart-render/echarts_render.py chart-render/html_report.py chart-render/ssr.cjs chart-render/theme.js chart-render/theme.json chart-render/palette.json /out/
COPY chart-render/report/core.js chart-render/report/runtime.js chart-render/report/report.css chart-render/report/preview.cjs /out/report/
RUN chmod 755 /out/echarts_render.py /out/html_report.py

# The font set (OFL-1.1) charts and documents draw in: Noto Sans SC, JP and KR for those scripts,
# and Source Sans 3 for the Latin they all stop short of beyond Latin-1. Pinned to one noto-cjk
# commit and one source-sans commit and checked by digest. A remote ADD arrives 0600, unreadable to
# app, and ADD --chmod leaves its parent directory closed, so the files are opened here before the
# final stage copies them; scripts/install-fonts.sh installs this same set on a development machine.
FROM python:3.14.10-slim-bookworm AS fonts
ARG NOTO_CJK=https://raw.githubusercontent.com/notofonts/noto-cjk/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Sans
ARG SOURCE_SANS=https://raw.githubusercontent.com/adobe-fonts/source-sans/87b37a2daaed80fcb8e8ccb0085c4d72ddade12e
ADD --checksum=sha256:faa6c9df652116dde789d351359f3d7e5d2285a2b2a1f04a2d7244df706d5ea9 ${NOTO_CJK}/SubsetOTF/SC/NotoSansSC-Regular.otf /fonts/NotoSansSC-Regular.otf
ADD --checksum=sha256:c6cb5a93abaa9edc8ee7463b7ebb7f42d618d40e6ed2f7a5371c97b0b64767c0 ${NOTO_CJK}/SubsetOTF/SC/NotoSansSC-Bold.otf /fonts/NotoSansSC-Bold.otf
ADD --checksum=sha256:dff723ba59d57d136764a04b9b2d03205544f7cd785a711442d6d2d085ac5073 ${NOTO_CJK}/SubsetOTF/JP/NotoSansJP-Regular.otf /fonts/NotoSansJP-Regular.otf
ADD --checksum=sha256:1b0edfb500b73a4fa8a4fcaae1bbbd403994e08e73e3e0da37e70d3853f42c5f ${NOTO_CJK}/SubsetOTF/JP/NotoSansJP-Bold.otf /fonts/NotoSansJP-Bold.otf
ADD --checksum=sha256:69975a0ac8472717870aefeab0a4d52739308d90856b9955313b2ad5e0148d68 ${NOTO_CJK}/SubsetOTF/KR/NotoSansKR-Regular.otf /fonts/NotoSansKR-Regular.otf
ADD --checksum=sha256:5a6ceb287ed2fc6cfc6213144ebea68cbd94b20fc9eb873d8486493bf02d9bda ${NOTO_CJK}/SubsetOTF/KR/NotoSansKR-Bold.otf /fonts/NotoSansKR-Bold.otf
ADD --checksum=sha256:6a73f9541c2de74158c0e7cf6b0a58ef774f5a780bf191f2d7ec9cc53efe2bf2 ${NOTO_CJK}/LICENSE /fonts/LICENSE-noto-cjk
ADD --checksum=sha256:4644c81b86ec9caaa76b634889968ed3c4f4f52f054855933acc7c2b21e53b0f ${SOURCE_SANS}/TTF/SourceSans3-Regular.ttf /fonts/SourceSans3-Regular.ttf
ADD --checksum=sha256:9214b9d95e4231c609802815c2646c98174e2102d0d37f88978a7f8e71006e6a ${SOURCE_SANS}/TTF/SourceSans3-Bold.ttf /fonts/SourceSans3-Bold.ttf
ADD --checksum=sha256:192afd78f0f54a3c69eaf02d43f4d9a821e9d6110e41d3d25d61a7385cd580e4 ${SOURCE_SANS}/TTF/SourceSans3-It.ttf /fonts/SourceSans3-It.ttf
ADD --checksum=sha256:7978291fc1bf314db887e0366853b33c5cf2e964c7b95cfb9ce403a6ec46a842 ${SOURCE_SANS}/TTF/SourceSans3-BoldIt.ttf /fonts/SourceSans3-BoldIt.ttf
ADD --checksum=sha256:56af9b9c6715597e458284a474dc118a50a4150e9d547c70f7b4a33c3e6a9328 ${SOURCE_SANS}/LICENSE.md /fonts/LICENSE-source-sans-3.md
RUN chmod 755 /fonts && chmod 644 /fonts/*

FROM python:3.14.10-slim-bookworm
LABEL maintainer="HanlianLyu"

WORKDIR /app

# Only the node binary is copied, so install the one library it links that the
# slim base lacks (libatomic1); without it every node process fails to start.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libatomic1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=frontend /usr/local/bin/node /usr/local/bin/node
COPY --from=search-tools /search-tools/fd /search-tools/rg /usr/local/bin/
# Charts: echarts-render draws with ECharts on the node above and rasterizes with resvg, in the
# fonts the image ships.
COPY --from=resvg /out/bin/resvg /usr/local/bin/resvg
COPY --from=resvg /out/licenses/ /usr/local/share/doc/resvg/
COPY --from=fonts /fonts/ /usr/local/share/fonts/dlightrag/
COPY --from=chart-render /out/ /usr/local/lib/echarts-render/
RUN ln -s /usr/local/lib/echarts-render/echarts_render.py /usr/local/bin/echarts-render \
    && ln -s /usr/local/lib/echarts-render/html_report.py /usr/local/bin/html-report
# Create non-root user BEFORE copying files to avoid chown layer duplication
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home app \
    && mkdir -p /app/dlightrag_storage /home/app/.dlightrag/agent_workspaces \
    /home/app/.dlightrag/skills /home/app/.dlightrag/owner_skills \
    && chown app:app /app/dlightrag_storage /home/app/.dlightrag/agent_workspaces \
    /home/app/.dlightrag/skills /home/app/.dlightrag/owner_skills

# The venv stays root-owned, read-only for app. Dependencies are the lockfile's, preflighted rather
# than installed at runtime: neither the service (LightRAG's pipmaster) nor an Agent process, if
# Landlock is unavailable and it runs unconfined, can change what the image ships. Bytecode is
# compiled in the builder, so nothing needs to write here.
COPY --from=builder /app/.venv /app/.venv
# The tiktoken encoding file baked in the builder stage: root-owned, read-only for app (it
# never writes it). TIKTOKEN_CACHE_DIR points every process of the image at it.
COPY --from=builder /opt/tiktoken /opt/tiktoken

ENV PATH="/app/.venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

EXPOSE 8100 8101

USER app

# Charts need node, resvg, ECharts and the font set together as the app user: fail the build, not a
# Run. The second chart draws with the scripts the image only recently gained — a Korean title and
# axis, a Turkish subtext, a Japanese label — one script per field, because resvg gives a text
# element one family and boxes what that family lacks. `echarts-render` exits 0 after naming the
# characters no font has, so its note is the thing to assert on.
RUN printf '%s' '{"title":{"text":"冒烟测试 Smoke"},"xAxis":{"type":"category","data":["甲","乙"]},"yAxis":{},"series":[{"type":"bar","data":[1,2]}]}' \
    | echarts-render - /tmp/chart-smoke.png && test -s /tmp/chart-smoke.png && rm /tmp/chart-smoke.png \
    && printf '%s' '{"title":{"text":"한국어 매출","subtext":"Türkçe ıİşğ · źródło: test"},"xAxis":{"type":"category","data":["서울","부산","東京"]},"yAxis":{},"series":[{"type":"bar","data":[3,2,1]}]}' \
    | echarts-render - /tmp/chart-fonts-smoke.png 2> /tmp/chart-fonts-smoke.err \
    && test -s /tmp/chart-fonts-smoke.png && ! grep -q "no font has" /tmp/chart-fonts-smoke.err \
    && rm /tmp/chart-fonts-smoke.png /tmp/chart-fonts-smoke.err

# Reports need the same tools and the runtime files: build a one-chart report and its previews as the
# app user, so a toolkit that cannot run fails the build, not a Run.
RUN printf '%s' '<h1>冒烟测试 Smoke</h1><figure class="chart" data-chart="a"></figure><script type="application/json" id="chart-a">{"option":{"title":{"text":"冒烟测试","subtext":"单位：个 · 来源：测试"},"xAxis":{"type":"category","data":["甲","乙"]},"yAxis":{},"series":[{"type":"bar","data":[1,2]}]}}</script>' > /tmp/report-smoke.html \
    && html-report build /tmp/report-smoke.html /tmp/report-smoke.out.html --preview /tmp/report-smoke \
    && test -s /tmp/report-smoke.out.html && test -s /tmp/report-smoke/a-360.png && test -s /tmp/report-smoke/a-900.png \
    && rm -r /tmp/report-smoke.html /tmp/report-smoke.out.html /tmp/report-smoke

# Documents (office-documents Skill) embed the same fonts through fpdf2, and app cannot change the
# venv: an Agent that wants either has what it needs and no way to install around it. The page is the
# Skill's own rule — each language in its own family, the others as fallbacks — and the assertion is
# fpdf2's missing-glyph warning, because a glyph the chosen font lacks is otherwise dropped in
# silence.
RUN python -c "import io, logging; from fpdf import FPDF; d = '/usr/local/share/fonts/dlightrag/'; faces = {'latin': ('SourceSans3-Regular.ttf', 'SourceSans3-Bold.ttf'), 'sc': ('NotoSansSC-Regular.otf', 'NotoSansSC-Bold.otf'), 'jp': ('NotoSansJP-Regular.otf', 'NotoSansJP-Bold.otf'), 'kr': ('NotoSansKR-Regular.otf', 'NotoSansKR-Bold.otf')}; pages = {'latin': 'English Ångström Straße ẞ français œuvre Ÿ español ñ ¿Qué? Türkçe ıİşğ Polski ąćęłńóśźż Čeština řčšžůě Magyar őű Română șț', 'sc': '中文 简体 臺灣 单位：个 · 来源：测试', 'jp': '日本語 かな 直骨海 東京都', 'kr': '한국어 서울 대한민국'}; log = io.StringIO(); logging.getLogger().addHandler(logging.StreamHandler(log)); p = FPDF(); p.add_page(); [p.add_font(f, s, d + v) for f, pair in faces.items() for s, v in (('', pair[0]), ('B', pair[1]))]; [(p.set_font(f, 'B', 12), p.set_fallback_fonts([o for o in faces if o != f], exact_match=False), p.cell(text=t), p.ln(8)) for f, t in pages.items()]; assert p.output(); assert 'missing the following glyphs' not in log.getvalue(), log.getvalue()" \
    && ! touch /app/.venv/.app-may-not-write 2>/dev/null

# Default image role; deployments can override it for MCP or maintenance commands.
CMD ["dlightrag-api"]
