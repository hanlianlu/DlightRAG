#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
#
# Linux / WSL2 background MinerU sidecar via systemd --user units. Mirrors
# scripts/mineru/launch_agent.sh (macOS launchd); routed here by service.sh on
# Linux. Manages TWO units so one command brings up both the parsing backend and
# the WebUI:
#   dlightrag-mineru-api.service     -> api.sh     (FastAPI backend, :8210)
#   dlightrag-mineru-gradio.service  -> gradio.sh  (Gradio WebUI, :7860, reuses the API)
# Set MINERU_GRADIO_ENABLE=false in .env.mineru to manage the API alone.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/mineru/env.sh
source "$script_dir/env.sh"

load_mineru_env_key MINERU_GRADIO_ENABLE

unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

# Every unit this manager owns (used to fully tear down, so a disabled WebUI is
# still cleaned up).
all_services=(api gradio)
# Units to bring up. The WebUI is opt-out via MINERU_GRADIO_ENABLE=false.
enabled_services=(api)
if [[ "${MINERU_GRADIO_ENABLE:-true}" != "false" ]]; then
  enabled_services+=(gradio)
fi

unit_for() { printf 'dlightrag-mineru-%s.service' "$1"; }

require_systemd() {
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "systemctl not found; cannot manage a background service." >&2
    echo "Run the sidecar in the foreground instead:  make mineru-api" >&2
    exit 127
  fi
}

write_unit() {
  local name="$1"
  local description after
  if [[ "$name" == "gradio" ]]; then
    description="DlightRAG MinerU Gradio WebUI"
    after="network.target $(unit_for api)"
  else
    description="DlightRAG MinerU API sidecar"
    after="network.target"
  fi
  mkdir -p "$unit_dir"
  cat >"$unit_dir/$(unit_for "$name")" <<UNIT
[Unit]
Description=${description}
After=${after}

[Service]
Type=simple
WorkingDirectory=${mineru_repo_root}
Environment=MINERU_ENV_FILE=${mineru_env_file}
ExecStart=${script_dir}/${name}.sh
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
UNIT
}

case "${1:-}" in
install)
  require_systemd
  for name in "${all_services[@]}"; do
    systemctl --user disable --now "$(unit_for "$name")" >/dev/null 2>&1 || true
    rm -f "$unit_dir/$(unit_for "$name")"
  done
  for name in "${enabled_services[@]}"; do
    write_unit "$name"
  done
  systemctl --user daemon-reload
  for name in "${enabled_services[@]}"; do
    systemctl --user enable --now "$(unit_for "$name")"
  done
  ;;
start)
  require_systemd
  for name in "${enabled_services[@]}"; do
    systemctl --user start "$(unit_for "$name")"
  done
  ;;
stop)
  require_systemd
  for name in "${all_services[@]}"; do
    systemctl --user stop "$(unit_for "$name")" 2>/dev/null || true
  done
  ;;
status)
  require_systemd
  units=()
  for name in "${enabled_services[@]}"; do
    units+=("$(unit_for "$name")")
  done
  systemctl --user status "${units[@]}"
  ;;
logs)
  require_systemd
  args=()
  for name in "${enabled_services[@]}"; do
    args+=(-u "$(unit_for "$name")")
  done
  journalctl --user "${args[@]}" -n 200 -f
  ;;
uninstall)
  require_systemd
  for name in "${all_services[@]}"; do
    systemctl --user disable --now "$(unit_for "$name")" >/dev/null 2>&1 || true
    rm -f "$unit_dir/$(unit_for "$name")"
  done
  systemctl --user daemon-reload
  ;;
*)
  echo "Usage: $0 {install|start|stop|status|logs|uninstall}" >&2
  exit 64
  ;;
esac
