#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# shellcheck source=scripts/mineru/env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

load_mineru_env_key MINERU_SERVICE_VENV
load_mineru_env_key MINERU_PYTHON
load_mineru_env_key MINERU_VERSION
load_mineru_env_key MINERU_MIN_VERSION
load_mineru_env_key MINERU_MAX_VERSION
load_mineru_env_key MINERU_INSTALL_EXTRAS

# MinerU supports CPython 3.10-3.13 only. Pin the sidecar venv to a compatible
# interpreter (uv downloads it if missing) so hosts whose default python is 3.14+
# still resolve wheels -- e.g. onnxruntime ships no cp314 build. Override with
# MINERU_PYTHON to select a different supported version.
default_mineru_python="3.13"

# Supported MinerU release range. Keep upgrades on the reviewed 3.x API contract;
# operators can narrow it further, while an exact MINERU_VERSION must still satisfy
# both bounds.
default_mineru_min_version="3.4.5"
default_mineru_max_version="4"

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
max_version="${MINERU_MAX_VERSION:-$default_mineru_max_version}"
extras="${MINERU_INSTALL_EXTRAS:-$(default_mineru_install_extras)}"
package="mineru[$extras]"
constraints=()
[[ -n "$version" ]] && constraints+=("==$version")
[[ -n "$min_version" ]] && constraints+=(">=$min_version")
[[ -n "$max_version" ]] && constraints+=("<$max_version")
if (( ${#constraints[@]} )); then
  constraint_spec="$(IFS=,; printf '%s' "${constraints[*]}")"
  package="$package$constraint_spec"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found on PATH; install uv before installing the MinerU service env." >&2
  exit 127
fi

uv venv --python "$python_version" "$venv"
uv pip install --python "$venv/bin/python" -U "$package"
