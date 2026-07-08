#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
#
# Cross-OS dispatcher for the MinerU background service. Routes the
# make mineru-service-* targets to the platform-native mechanism:
#   macOS  -> launch_agent.sh   (launchd)
#   Linux  -> systemd_service.sh (systemd --user; also covers WSL2)
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$(uname -s)" in
Darwin)
  exec "$script_dir/launch_agent.sh" "$@"
  ;;
Linux)
  exec "$script_dir/systemd_service.sh" "$@"
  ;;
*)
  echo "Unsupported OS for a MinerU background service: $(uname -s)." >&2
  echo "Run the sidecar in the foreground instead:  make mineru-api" >&2
  exit 64
  ;;
esac
