#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
#
# Install the two tools the chart renderer's PNG path needs beyond node, resvg and the font set,
# into DIR (default .chart-tools): DIR/bin/resvg and DIR/fonts/dlightrag. The versions and the
# font checksums are the Dockerfile's, so CI cannot test a renderer the image does not ship.
# Linux x86_64 only: the image builds resvg from its crates.io release, and CI is the one place
# that wants the upstream binary. The fonts come from install-fonts.sh, which runs on any platform,
# so a development machine that only needs them does not need this one. Afterwards:
#   export PATH="DIR/bin:$PATH" FONTS_DIR="DIR/fonts/dlightrag"
set -euo pipefail

dir="${1:-.chart-tools}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dockerfile="${root}/Dockerfile"

if [ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]; then
    echo "install-chart-tools: only Linux x86_64 is supported; install resvg and Noto Sans SC yourself" >&2
    exit 1
fi

resvg_version="$(sed -n 's/^ARG RESVG_VERSION=//p' "${dockerfile}")"
# The checksum of the upstream linux-x86_64 archive of each resvg release the Dockerfile has pinned.
case "${resvg_version}" in
0.50.0) resvg_sha256="5d1c9ffd43c82fb802d92cef8100055c612cd3e248d5a787c20079293a456e21" ;;
*)
    echo "install-chart-tools: add the linux-x86_64 archive checksum of resvg ${resvg_version}" >&2
    exit 1
    ;;
esac

fetch() { # fetch URL SHA256 FILE
    curl --fail --silent --show-error --location --retry 3 --output "$3" "$1"
    echo "$2  $3" | sha256sum --check --quiet -
}

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
mkdir -p "${dir}/bin"

fetch "https://github.com/linebender/resvg/releases/download/v${resvg_version}/resvg-linux-x86_64.tar.gz" \
    "${resvg_sha256}" "${work}/resvg.tar.gz"
tar --extract --gzip --file "${work}/resvg.tar.gz" --directory "${work}"
find "${work}" -type f -name resvg -exec install -m 755 {} "${dir}/bin/resvg" \;
if [ ! -x "${dir}/bin/resvg" ]; then
    echo "install-chart-tools: the resvg ${resvg_version} archive holds no resvg executable" >&2
    exit 1
fi

"${root}/scripts/install-fonts.sh" "${dir}/fonts/dlightrag"

"${dir}/bin/resvg" --version
