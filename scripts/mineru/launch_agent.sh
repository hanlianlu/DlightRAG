#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/env.sh"

label="com.hanlianlyu.dlightrag.mineru-api"
domain="gui/$(id -u)"
service="$domain/$label"
launchd_home="${MINERU_LAUNCHD_HOME:-$HOME}"
launch_agents_dir="$launchd_home/Library/LaunchAgents"
log_dir="$launchd_home/Library/Logs/dlightrag"
plist_path="$launch_agents_dir/$label.plist"
stdout_log="$log_dir/mineru-api.out.log"
stderr_log="$log_dir/mineru-api.err.log"
api_script="$script_dir/api.sh"

write_plist() {
  mkdir -p "$launch_agents_dir" "$log_dir"
  python3 - "$plist_path" "$label" "$api_script" "$mineru_repo_root" "$mineru_env_file" "$stdout_log" "$stderr_log" <<'PY'
import plistlib
import sys
from pathlib import Path

path, label, api_script, repo_root, env_file, stdout_log, stderr_log = sys.argv[1:]
plist = {
    "Label": label,
    "ProgramArguments": [api_script],
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
    write_plist
    launchctl bootout "$service" >/dev/null 2>&1 || true
    launchctl bootstrap "$domain" "$plist_path"
    ;;
  start)
    launchctl bootstrap "$domain" "$plist_path"
    ;;
  stop)
    launchctl bootout "$service"
    ;;
  status)
    launchctl print "$service"
    ;;
  logs)
    mkdir -p "$log_dir"
    tail -n 200 -f "$stdout_log" "$stderr_log"
    ;;
  uninstall)
    launchctl bootout "$service" >/dev/null 2>&1 || true
    rm -f "$plist_path"
    ;;
  *)
    echo "Usage: $0 {install|start|stop|status|logs|uninstall}" >&2
    exit 64
    ;;
esac
