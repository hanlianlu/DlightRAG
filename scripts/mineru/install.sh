#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

load_mineru_env_key MINERU_SERVICE_VENV
load_mineru_env_key MINERU_INSTALL_EXTRAS

default_mineru_install_extras() {
  local system machine
  system="$(uname -s)"
  machine="$(uname -m)"
  if [[ "$system" == "Darwin" && ( "$machine" == "arm64" || "$machine" == "aarch64" ) ]]; then
    printf '%s\n' "core,mlx"
  else
    printf '%s\n' "core"
  fi
}

venv="${MINERU_SERVICE_VENV:-.venv-mineru}"
extras="${MINERU_INSTALL_EXTRAS:-$(default_mineru_install_extras)}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found on PATH; install uv before installing the MinerU service env." >&2
  exit 127
fi

uv venv "$venv"
uv pip install --python "$venv/bin/python" -U "mineru[$extras]"
