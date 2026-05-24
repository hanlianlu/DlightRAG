#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

mineru_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mineru_env_file="${MINERU_ENV_FILE:-$mineru_repo_root/.env}"
if [[ "$mineru_env_file" != /* ]]; then
  mineru_env_file="$mineru_repo_root/$mineru_env_file"
fi

load_mineru_env_key() {
  local key="$1"
  local line value

  if [[ -n "${!key:-}" || ! -f "$mineru_env_file" ]]; then
    return
  fi

  line="$(grep -E "^[[:space:]]*${key}=" "$mineru_env_file" | tail -n 1 || true)"
  if [[ -z "$line" ]]; then
    return
  fi

  value="${line#*=}"
  value="${value%$'\r'}"
  if [[ "$value" == \"*\" && "$value" == *\" ]]; then
    value="${value:1:${#value}-2}"
  elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
    value="${value:1:${#value}-2}"
  fi

  export "$key=$value"
}
