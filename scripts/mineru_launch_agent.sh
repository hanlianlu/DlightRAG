#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
source "$script_dir/mineru_env.sh"

load_mineru_env_key MINERU_LAUNCHD_LABEL
load_mineru_env_key MINERU_LAUNCHD_HOME
load_mineru_env_key MINERU_LAUNCHD_LOG_DIR
load_mineru_env_key MINERU_LAUNCHD_PLIST

label="${MINERU_LAUNCHD_LABEL:-com.hanlianlyu.dlightrag.mineru-api}"
launchd_home="${MINERU_LAUNCHD_HOME:-$HOME}"
launch_agents_dir="$launchd_home/Library/LaunchAgents"
log_dir="${MINERU_LAUNCHD_LOG_DIR:-$launchd_home/Library/Logs/dlightrag}"
plist_path="${MINERU_LAUNCHD_PLIST:-$launch_agents_dir/$label.plist}"
uid="$(id -u)"
domain="gui/$uid"
service_target="$domain/$label"

xml_escape() {
  local value="$1"
  value="${value//&/&amp;}"
  value="${value//</&lt;}"
  value="${value//>/&gt;}"
  value="${value//\"/&quot;}"
  value="${value//\'/&apos;}"
  printf '%s' "$value"
}

require_launchctl() {
  if ! command -v launchctl >/dev/null 2>&1; then
    echo "launchctl was not found; MinerU LaunchAgent management requires macOS." >&2
    exit 127
  fi
}

write_plist() {
  mkdir -p "$launch_agents_dir" "$log_dir"

  local start_script env_file stdout_path stderr_path
  start_script="$repo_root/scripts/start_mineru_api.sh"
  env_file="$mineru_env_file"
  stdout_path="$log_dir/mineru-api.out.log"
  stderr_path="$log_dir/mineru-api.err.log"

  cat >"$plist_path" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$(xml_escape "$label")</string>
  <key>ProgramArguments</key>
  <array>
    <string>$(xml_escape "$start_script")</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$(xml_escape "$repo_root")</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>MINERU_ENV_FILE</key>
    <string>$(xml_escape "$env_file")</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$(xml_escape "$stdout_path")</string>
  <key>StandardErrorPath</key>
  <string>$(xml_escape "$stderr_path")</string>
</dict>
</plist>
EOF
}

is_loaded() {
  launchctl print "$service_target" >/dev/null 2>&1
}

start_service() {
  require_launchctl
  if is_loaded; then
    launchctl kickstart -k "$service_target"
  else
    launchctl bootstrap "$domain" "$plist_path"
  fi
}

install_service() {
  require_launchctl
  write_plist
  launchctl bootout "$service_target" 2>/dev/null || true
  launchctl bootstrap "$domain" "$plist_path"
}

stop_service() {
  require_launchctl
  launchctl bootout "$service_target" 2>/dev/null || true
}

usage() {
  cat <<EOF
Usage: scripts/mineru_launch_agent.sh <command>

Commands:
  install    Write the LaunchAgent plist and start MinerU API
  start      Start or restart the installed LaunchAgent
  stop       Stop the LaunchAgent
  status     Print launchd status for the LaunchAgent
  logs       Follow MinerU API stdout and stderr logs
  uninstall  Stop the LaunchAgent and remove the plist
EOF
}

command="${1:-}"
case "$command" in
  install)
    install_service
    ;;
  start)
    if [[ ! -f "$plist_path" ]]; then
      write_plist
    fi
    start_service
    ;;
  stop)
    stop_service
    ;;
  status)
    require_launchctl
    launchctl print "$service_target"
    ;;
  logs)
    mkdir -p "$log_dir"
    touch "$log_dir/mineru-api.out.log" "$log_dir/mineru-api.err.log"
    exec tail -n 100 -f "$log_dir/mineru-api.out.log" "$log_dir/mineru-api.err.log"
    ;;
  uninstall)
    stop_service
    rm -f "$plist_path"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 64
    ;;
esac
