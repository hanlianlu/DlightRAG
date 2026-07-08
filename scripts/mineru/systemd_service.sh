#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
#
# Linux / WSL2 background MinerU sidecar via a systemd --user unit.
# Mirrors scripts/mineru/launch_agent.sh (macOS launchd); routed here by
# scripts/mineru/service.sh on Linux.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/mineru/env.sh
source "$script_dir/env.sh"

unit_name="dlightrag-mineru-api.service"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
unit_path="$unit_dir/$unit_name"
api_script="$script_dir/api.sh"

require_systemd() {
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "systemctl not found; cannot manage a background service." >&2
    echo "Run the sidecar in the foreground instead:  make mineru-api" >&2
    exit 127
  fi
}

write_unit() {
  mkdir -p "$unit_dir"
  cat >"$unit_path" <<UNIT
[Unit]
Description=DlightRAG MinerU API sidecar
After=network.target

[Service]
Type=simple
WorkingDirectory=${mineru_repo_root}
Environment=MINERU_ENV_FILE=${mineru_env_file}
ExecStart=${api_script}
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
UNIT
}

case "${1:-}" in
install)
  require_systemd
  write_unit
  systemctl --user daemon-reload
  systemctl --user enable --now "$unit_name"
  ;;
start)
  require_systemd
  systemctl --user start "$unit_name"
  ;;
stop)
  require_systemd
  systemctl --user stop "$unit_name"
  ;;
status)
  require_systemd
  systemctl --user status "$unit_name"
  ;;
logs)
  require_systemd
  journalctl --user -u "$unit_name" -n 200 -f
  ;;
uninstall)
  require_systemd
  systemctl --user disable --now "$unit_name" >/dev/null 2>&1 || true
  rm -f "$unit_path"
  systemctl --user daemon-reload
  ;;
*)
  echo "Usage: $0 {install|start|stop|status|logs|uninstall}" >&2
  exit 64
  ;;
esac
