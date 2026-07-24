#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
set -euo pipefail

mineru_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/mineru/env.sh
source "$mineru_script_dir/env.sh"

load_mineru_env_key MINERU_API_HOST
load_mineru_env_key MINERU_API_PORT
load_mineru_env_key MINERU_GRADIO_HOST
load_mineru_env_key MINERU_GRADIO_PORT
load_mineru_env_key MINERU_GRADIO_API_URL
load_mineru_env_key MINERU_SERVICE_VENV

api_host="${MINERU_API_HOST:-127.0.0.1}"
api_port="${MINERU_API_PORT:-8210}"
gradio_host="${MINERU_GRADIO_HOST:-127.0.0.1}"
gradio_port="${MINERU_GRADIO_PORT:-7860}"
venv="${MINERU_SERVICE_VENV:-.venv-mineru}"

# The Gradio WebUI is a thin front end. Point it at the already-running mineru-api
# sidecar via --api-url so parsing REUSES that one backend instead of spawning a
# second model-loading service. When the API binds 0.0.0.0 (all interfaces), the
# WebUI still reaches it over loopback.
api_connect_host="$api_host"
if [[ "$api_connect_host" == "0.0.0.0" ]]; then
  api_connect_host="127.0.0.1"
fi
api_url="${MINERU_GRADIO_API_URL:-http://${api_connect_host}:${api_port}}"

if [[ -x "$venv/bin/mineru-gradio" ]]; then
  mineru_gradio="$venv/bin/mineru-gradio"
elif command -v mineru-gradio >/dev/null 2>&1; then
  mineru_gradio="$(command -v mineru-gradio)"
else
  cat >&2 <<'EOF'
mineru-gradio was not found on PATH.

Install the dedicated local MinerU service env first (the `core` extra bundles
the Gradio WebUI):
  make mineru-install
EOF
  exit 127
fi

exec "$mineru_gradio" \
  --api-url "$api_url" \
  --server-name "$gradio_host" \
  --server-port "$gradio_port" \
  "$@"
