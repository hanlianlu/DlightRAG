#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
# Write the MinerU title-aided LLM config to ~/mineru.json.
#
# Reads credentials from .env.mineru (never committed).  If you need custom
# values for one run, export the env vars before calling this script:
#
#   MINERU_TITLE_AIDED_API_KEY=sk-... make mineru-title-aided
#
# Re-run after changing credentials to update ~/mineru.json in-place.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/../.." && pwd)"
MINERU_ENV_FILE="${MINERU_ENV_FILE:-$REPO_ROOT/.env.mineru}"

# ── load credentials from .env.mineru (env vars take precedence) ──────
load_key() {
  local key="$1"
  if [[ -n "${!key:-}" ]]; then return; fi
  if [[ ! -f "$MINERU_ENV_FILE" ]]; then return; fi
  local line
  line="$(grep -E "^[[:space:]]*${key}=" "$MINERU_ENV_FILE" | tail -n 1 || true)"
  if [[ -z "$line" ]]; then return; fi
  local value="${line#*=}"
  value="${value%$'\r'}"
  if [[ "$value" == \"*\" && "$value" == *\" ]]; then value="${value:1:${#value}-2}"; fi
  if [[ "$value" == \'*\' && "$value" == *\' ]]; then value="${value:1:${#value}-2}"; fi
  export "$key=$value"
}

load_key MINERU_TITLE_AIDED_ENABLE
load_key MINERU_TITLE_AIDED_API_KEY
load_key MINERU_TITLE_AIDED_BASE_URL
load_key MINERU_TITLE_AIDED_MODEL
load_key MINERU_TITLE_AIDED_ENABLE_THINKING

ENABLE="${MINERU_TITLE_AIDED_ENABLE:-true}"
API_KEY="${MINERU_TITLE_AIDED_API_KEY:-}"
BASE_URL="${MINERU_TITLE_AIDED_BASE_URL:-}"
MODEL="${MINERU_TITLE_AIDED_MODEL:-}"
ENABLE_THINKING="${MINERU_TITLE_AIDED_ENABLE_THINKING:-false}"

# ── validate ──────────────────────────────────────────────────────────
if [[ "$ENABLE" != "true" ]]; then
  echo "==> MinerU title-aided is disabled (MINERU_TITLE_AIDED_ENABLE != true)."
  echo "    Set MINERU_TITLE_AIDED_ENABLE=true in .env.mineru and re-run to enable."
  exit 0
fi

if [[ -z "$API_KEY" ]]; then
  echo "ERROR: MINERU_TITLE_AIDED_API_KEY is not set."
  echo "Add it to .env.mineru or export it before running this script."
  exit 1
fi

if [[ -z "$BASE_URL" ]]; then
  echo "ERROR: MINERU_TITLE_AIDED_BASE_URL is not set."
  echo "Add it to .env.mineru or export it before running this script."
  exit 1
fi

if [[ -z "$MODEL" ]]; then
  echo "ERROR: MINERU_TITLE_AIDED_MODEL is not set."
  echo "Add it to .env.mineru or export it before running this script."
  exit 1
fi

# ── generate ~/mineru.json via Python (merges with any existing config) ──
TARGET="${HOME}/mineru.json"
PYTHON_BIN="${REPO_ROOT}/.venv-mineru/bin/python3"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || echo python3)"
fi

"$PYTHON_BIN" -u "$SCRIPTS_DIR/_write_mineru_json.py" \
  "$TARGET" "$API_KEY" "$BASE_URL" "$MODEL" "$ENABLE_THINKING"
