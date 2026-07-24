#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
#
# macOS launchd manager for the local MinerU sidecar. Manages TWO user agents so
# one command brings up both the parsing backend and the WebUI:
#   mineru-api     -> api.sh     (FastAPI parsing backend, :8210)
#   mineru-gradio  -> gradio.sh  (Gradio WebUI, :7860, REUSES the API backend)
# Set MINERU_GRADIO_ENABLE=false in .env.mineru to manage the API alone.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/mineru/env.sh
source "$script_dir/env.sh"

load_mineru_env_key MINERU_GRADIO_ENABLE

domain="gui/$(id -u)"
launchd_home="${MINERU_LAUNCHD_HOME:-$HOME}"
launch_agents_dir="$launchd_home/Library/LaunchAgents"
log_dir="$launchd_home/Library/Logs/dlightrag"

# Every agent this manager owns (used to fully tear down, so a disabled WebUI is
# still cleaned up).
all_services=(api gradio)
# Agents to bring up. The WebUI is opt-out via MINERU_GRADIO_ENABLE=false.
enabled_services=(api)
if [[ "${MINERU_GRADIO_ENABLE:-true}" != "false" ]]; then
  enabled_services+=(gradio)
fi

label_for() { printf 'com.hanlianlyu.dlightrag.mineru-%s' "$1"; }
plist_for() { printf '%s/com.hanlianlyu.dlightrag.mineru-%s.plist' "$launch_agents_dir" "$1"; }

write_plist() {
  local name="$1"
  local label plist_path program stdout_log stderr_log
  label="$(label_for "$name")"
  plist_path="$(plist_for "$name")"
  program="$script_dir/$name.sh"
  stdout_log="$log_dir/mineru-$name.out.log"
  stderr_log="$log_dir/mineru-$name.err.log"
  mkdir -p "$launch_agents_dir" "$log_dir"
  python3 - "$plist_path" "$label" "$program" "$mineru_repo_root" "$mineru_env_file" "$stdout_log" "$stderr_log" <<'PY'
import plistlib
import sys
from pathlib import Path

path, label, program, repo_root, env_file, stdout_log, stderr_log = sys.argv[1:]
plist = {
    "Label": label,
    "ProgramArguments": [program],
    "WorkingDirectory": repo_root,
    "RunAtLoad": True,
    "KeepAlive": True,
    "EnvironmentVariables": {"MINERU_ENV_FILE": env_file},
    "StandardOutPath": stdout_log,
    "StandardErrorPath": stderr_log,
}
with Path(path).open("wb") as fh:
    plistlib.dump(plist, fh)
PY
}

case "${1:-}" in
  install)
    for name in "${all_services[@]}"; do
      launchctl bootout "$domain/$(label_for "$name")" >/dev/null 2>&1 || true
    done
    for name in "${enabled_services[@]}"; do
      write_plist "$name"
      launchctl bootstrap "$domain" "$(plist_for "$name")"
    done
    ;;
  start)
    for name in "${enabled_services[@]}"; do
      launchctl bootstrap "$domain" "$(plist_for "$name")"
    done
    ;;
  stop)
    for name in "${all_services[@]}"; do
      launchctl bootout "$domain/$(label_for "$name")" || true
    done
    ;;
  status)
    for name in "${enabled_services[@]}"; do
      launchctl print "$domain/$(label_for "$name")" || true
    done
    ;;
  logs)
    mkdir -p "$log_dir"
    logs=()
    for name in "${enabled_services[@]}"; do
      logs+=("$log_dir/mineru-$name.out.log" "$log_dir/mineru-$name.err.log")
    done
    tail -n 200 -f "${logs[@]}"
    ;;
  uninstall)
    for name in "${all_services[@]}"; do
      launchctl bootout "$domain/$(label_for "$name")" >/dev/null 2>&1 || true
      rm -f "$(plist_for "$name")"
    done
    ;;
  *)
    echo "Usage: $0 {install|start|stop|status|logs|uninstall}" >&2
    exit 64
    ;;
esac
