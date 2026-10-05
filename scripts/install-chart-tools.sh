#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
#
# Install the two tools the chart renderer's PNG path needs beyond node, resvg and Noto Sans SC,
# into DIR (default .chart-tools): DIR/bin/resvg and DIR/fonts/noto-sans-sc. The versions and the
# font checksums are the Dockerfile's, so CI cannot test a renderer the image does not ship.
# Linux x86_64 only: the image builds resvg from its crates.io release, and CI is the one place
# that wants the upstream binary. Afterwards:
#   export PATH="DIR/bin:$PATH" ECHARTS_RENDER_FONT_DIR="DIR/fonts/noto-sans-sc"
set -euo pipefail

dir="${1:-.chart-tools}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dockerfile="${root}/Dockerfile"

if [ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]; then
    echo "install-chart-tools: only Linux x86_64 is supported; install resvg and Noto Sans SC yourself" >&2
    exit 1
fi

resvg_version="$(sed -n 's/^ARG RESVG_VERSION=//p' "${dockerfile}")"
noto_cjk="$(sed -n 's/^ARG NOTO_CJK=//p' "${dockerfile}")"
# The checksum of the upstream linux-x86_64 archive of each resvg release the Dockerfile has pinned.
case "${resvg_version}" in
0.48.1) resvg_sha256="fa8c26495a187e592c501db15bf9e8a9fdc051d4b2b336b39703d5b59f912b9d" ;;
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
mkdir -p "${dir}/bin" "${dir}/fonts/noto-sans-sc"

fetch "https://github.com/linebender/resvg/releases/download/v${resvg_version}/resvg-linux-x86_64.tar.gz" \
    "${resvg_sha256}" "${work}/resvg.tar.gz"
tar --extract --gzip --file "${work}/resvg.tar.gz" --directory "${work}"
find "${work}" -type f -name resvg -exec install -m 755 {} "${dir}/bin/resvg" \;
if [ ! -x "${dir}/bin/resvg" ]; then
    echo "install-chart-tools: the resvg ${resvg_version} archive holds no resvg executable" >&2
    exit 1
fi

for face in Regular Bold; do
    sha256="$(sed -n "s|^ADD --checksum=sha256:\([0-9a-f]*\) .*/NotoSansSC-${face}.otf .*|\1|p" "${dockerfile}")"
    fetch "${noto_cjk}/SubsetOTF/SC/NotoSansSC-${face}.otf" "${sha256}" \
        "${dir}/fonts/noto-sans-sc/NotoSansSC-${face}.otf"
done

"${dir}/bin/resvg" --version
