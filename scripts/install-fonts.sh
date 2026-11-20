#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
#
# Install the font set the image ships, at the Dockerfile's pins, into DIR (default
# .chart-tools/fonts/dlightrag): Noto Sans SC, JP and KR for those scripts, Source Sans 3 for
# Latin, and both OFL-1.1 licenses. The file list is the Dockerfile's own `/fonts/` ADD lines, so a
# development machine installs the bytes the image carries rather than a font the image lacks. Any
# platform with curl: the chart renderer embeds this set in a PNG, and an Agent embeds it in a PDF
# through fpdf2, which reads no environment of its own — the Skill names DIR. Afterwards:
#   export FONTS_DIR="DIR"
set -euo pipefail

dir="${1:-.chart-tools/fonts/dlightrag}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dockerfile="${root}/Dockerfile"

noto_cjk="$(sed -n 's/^ARG NOTO_CJK=//p' "${dockerfile}")"
source_sans="$(sed -n 's/^ARG SOURCE_SANS=//p' "${dockerfile}")"
if [ -z "${noto_cjk}" ] || [ -z "${source_sans}" ]; then
    echo "install-fonts: ${dockerfile} pins no NOTO_CJK or no SOURCE_SANS" >&2
    exit 1
fi

# The name and the checksum of every file the image's font stage installs, e.g. NotoSansSC-Bold.otf.
pinned_names="$(sed -n 's|^ADD --checksum=sha256:[0-9a-f]\{64\} [^ ]* /fonts/\(.*\)$|\1|p' "${dockerfile}")"
if [ -z "${pinned_names}" ]; then
    echo "install-fonts: ${dockerfile} pins no file under /fonts/" >&2
    exit 1
fi

pinned_sha256() { # pinned_sha256 NAME
    sed -n "s|^ADD --checksum=sha256:\([0-9a-f]\{64\}\) [^ ]* /fonts/$1\$|\1|p" "${dockerfile}"
}

source_url() { # source_url NAME
    case "$1" in
    NotoSansSC-*) echo "${noto_cjk}/SubsetOTF/SC/$1" ;;
    NotoSansJP-*) echo "${noto_cjk}/SubsetOTF/JP/$1" ;;
    NotoSansKR-*) echo "${noto_cjk}/SubsetOTF/KR/$1" ;;
    LICENSE-noto-cjk) echo "${noto_cjk}/LICENSE" ;;
    SourceSans3-*) echo "${source_sans}/TTF/$1" ;;
    LICENSE-source-sans-3.md) echo "${source_sans}/LICENSE.md" ;;
    *) return 1 ;;
    esac
}

# macOS carries shasum and not sha256sum; both read "HASH  FILE" from standard input.
if command -v sha256sum >/dev/null 2>&1; then
    verify() { sha256sum --check --quiet -; }
else
    verify() { shasum --algorithm 256 --check --quiet -; }
fi

fetch() { # fetch URL SHA256 FILE
    curl --fail --silent --show-error --location --retry 3 --output "$3" "$1"
    echo "$2  $3" | verify
}

mkdir -p "${dir}"
for name in ${pinned_names}; do
    sha256="$(pinned_sha256 "${name}")"
    url="$(source_url "${name}")" || {
        echo "install-fonts: no source for the pinned file ${name}" >&2
        exit 1
    }
    fetch "${url}" "${sha256}" "${dir}/${name}"
done
chmod 644 "${dir}"/*

echo "install-fonts: ${dir} holds the image's font set at the Dockerfile's pins" >&2
echo "export FONTS_DIR=\"$(cd "${dir}" && pwd)\""
