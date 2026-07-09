#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# shellcheck source=scripts/mineru/env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

load_mineru_env_key MINERU_SERVICE_VENV
load_mineru_env_key MINERU_PYTHON
load_mineru_env_key MINERU_VERSION
load_mineru_env_key MINERU_MIN_VERSION
load_mineru_env_key MINERU_INSTALL_EXTRAS

# MinerU supports CPython 3.10-3.13 only. Pin the sidecar venv to a compatible
# interpreter (uv downloads it if missing) so hosts whose default python is 3.14+
# still resolve wheels -- e.g. onnxruntime ships no cp314 build. Override with
# MINERU_PYTHON to select a different supported version.
default_mineru_python="3.13"

# Lowest MinerU release DlightRAG supports. Fresh installs and `-U` upgrades must
# land on this version or newer ("and onward"). Override MINERU_MIN_VERSION in
# .env.mineru to raise the floor, or set MINERU_VERSION to pin an exact release.
default_mineru_min_version="3.4.3"

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
python_version="${MINERU_PYTHON:-$default_mineru_python}"
version="${MINERU_VERSION:-}"
min_version="${MINERU_MIN_VERSION:-$default_mineru_min_version}"
extras="${MINERU_INSTALL_EXTRAS:-$(default_mineru_install_extras)}"
package="mineru[$extras]"
if [[ -n "$version" ]]; then
  package="$package==$version"
elif [[ -n "$min_version" ]]; then
  package="$package>=$min_version"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found on PATH; install uv before installing the MinerU service env." >&2
  exit 127
fi

uv venv --python "$python_version" "$venv"
uv pip install --python "$venv/bin/python" -U "$package"
