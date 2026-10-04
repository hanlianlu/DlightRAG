# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
#
# One Agent Browser pool member (ADR 0032): a Playwright run-server that launches a
# browser for each connection and closes it with that connection. The client's Python
# `playwright` and this image's Playwright are one version, and the version is bumped in
# every pin together (docs/operations.md): pyproject.toml, uv.lock, this ARG, and
# package.json with its lock. The npm package is installed here because the container has
# no route to the internet when it runs.
ARG PLAYWRIGHT_VERSION=1.63.0
FROM mcr.microsoft.com/playwright:v${PLAYWRIGHT_VERSION}-noble
ARG PLAYWRIGHT_VERSION
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
WORKDIR /opt/agent-browser
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund \
 && test "$(node -p "require('playwright/package.json').version")" = "$PLAYWRIGHT_VERSION"
USER pwuser
WORKDIR /home/pwuser
EXPOSE 3000
# --unsafe lets a client pass Chromium's own sandbox request to the launch. It also lets a
# client choose the launch arguments, so the pool's networks admit only DlightRAG. The
# server limits no clients: the PostgreSQL lease alone gives a Run its member (ADR 0032).
CMD ["/opt/agent-browser/node_modules/.bin/playwright","run-server","--port","3000","--host","0.0.0.0","--unsafe"]
