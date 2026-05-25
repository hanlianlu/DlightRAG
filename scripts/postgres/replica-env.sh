#!/usr/bin/env bash
set -euo pipefail

dlightrag_load_env_file() {
  local env_file="${1:-.env}"
  if [[ ! -f "${env_file}" ]]; then
    return 0
  fi

  local assignment key
  while IFS= read -r -d '' assignment; do
    key="${assignment%%=*}"
    if [[ -z "${!key+x}" ]]; then
      export "${assignment}"
    fi
  done < <(
    python3 - "${env_file}" <<'PY'
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

env_file = Path(sys.argv[1])
allowed = re.compile(r"^(DLIGHTRAG_POSTGRES_|COMPOSE_PROJECT_NAME$)")
key_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

for raw_line in env_file.read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    if "=" not in line:
        continue
    key, value = line.split("=", 1)
    key = key.strip()
    if not key_re.fullmatch(key) or not allowed.match(key):
        continue
    value = value.strip()
    if value.startswith(("'", '"')):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            quote = value[0]
            if len(value) >= 2 and value.endswith(quote):
                value = value[1:-1]
    else:
        value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    sys.stdout.buffer.write(f"{key}={value}".encode())
    sys.stdout.buffer.write(b"\0")
PY
  )
}
