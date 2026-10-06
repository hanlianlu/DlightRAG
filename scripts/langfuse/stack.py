#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Prepare the local Langfuse Docker Compose stack used by DlightRAG.

Upstream pins the server images to a floating major tag, so a stack written once
keeps whatever major it was created with unless something compares it against the
major this repository targets. This module owns that comparison: an existing stack
that pins another major is refreshed from the official Compose file, and a stack
that still does not match is refused instead of started.
"""

import argparse
import re
import sys
import urllib.request
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

OFFICIAL_COMPOSE_URL = "https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml"

# The Langfuse server major DlightRAG targets. The Python SDK dependency is pinned to
# the same major (see pyproject.toml), because the v4 data model is what the SDK's
# real-time OpenTelemetry ingestion and the v2 read APIs require. A server major older
# than the SDK's is deprecated upstream, so bump this together with the SDK.
LANGFUSE_IMAGE_MAJOR = "4"

_LANGFUSE_IMAGE = re.compile(
    r"^\s*image:\s*(?:\S*/)?langfuse/langfuse(?:-worker)?:v?(\d+)\s*$", re.MULTILINE
)

_LOCAL_PORT_REPLACEMENTS = {
    "127.0.0.1:3030:3030": "127.0.0.1:${LANGFUSE_WORKER_PORT:-3301}:3030",
    "NEXTAUTH_URL: ${NEXTAUTH_URL:-http://localhost:3000}": (
        "NEXTAUTH_URL: ${NEXTAUTH_URL:-http://localhost:3300}"
    ),
    "LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT: ${LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT:-http://localhost:9090}": (
        "LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT: ${LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT:-http://localhost:3390}"
    ),
    "LANGFUSE_S3_BATCH_EXPORT_EXTERNAL_ENDPOINT: ${LANGFUSE_S3_BATCH_EXPORT_EXTERNAL_ENDPOINT:-http://localhost:9090}": (
        "LANGFUSE_S3_BATCH_EXPORT_EXTERNAL_ENDPOINT: ${LANGFUSE_S3_BATCH_EXPORT_EXTERNAL_ENDPOINT:-http://localhost:3390}"
    ),
    "- 3000:3000": "- 127.0.0.1:${LANGFUSE_WEB_PORT:-3300}:3000",
    "- 127.0.0.1:8123:8123": "- 127.0.0.1:${LANGFUSE_CLICKHOUSE_HTTP_PORT:-18123}:8123",
    "- 127.0.0.1:9000:9000": "- 127.0.0.1:${LANGFUSE_CLICKHOUSE_NATIVE_PORT:-19000}:9000",
    "- 9090:9000": "- 127.0.0.1:${LANGFUSE_MINIO_API_PORT:-3390}:9000",
    "- 127.0.0.1:9091:9001": "- 127.0.0.1:${LANGFUSE_MINIO_CONSOLE_PORT:-3391}:9001",
    "- 127.0.0.1:6379:6379": "- 127.0.0.1:${LANGFUSE_REDIS_PORT:-16379}:6379",
    "- 127.0.0.1:5432:5432": "- 127.0.0.1:${LANGFUSE_POSTGRES_PORT:-15432}:5432",
}


class StackResult(NamedTuple):
    path: Path
    created: bool
    refreshed: bool
    patched: bool


def _read_source(source: Path | None) -> str:
    if source is not None:
        return source.read_text(encoding="utf-8")
    parsed = urlparse(OFFICIAL_COMPOSE_URL)
    if parsed.scheme != "https" or parsed.netloc != "raw.githubusercontent.com":
        raise ValueError(f"unexpected Langfuse compose URL: {OFFICIAL_COMPOSE_URL}")
    with urllib.request.urlopen(OFFICIAL_COMPOSE_URL, timeout=30) as response:  # noqa: S310
        return response.read().decode("utf-8")


def _image_majors(content: str) -> frozenset[str]:
    """Return the Langfuse server majors the web and worker images pin."""
    return frozenset(_LANGFUSE_IMAGE.findall(content))


def _require_image_major(content: str, *, expected: str) -> None:
    majors = _image_majors(content)
    if majors != {expected}:
        found = ", ".join(sorted(majors)) or "none"
        raise RuntimeError(
            "Local Langfuse stack must pin langfuse/langfuse and langfuse/langfuse-worker "
            f"at major {expected}; found: {found}. If upstream moved to a new major, upgrade "
            "the DlightRAG Langfuse SDK dependency and LANGFUSE_IMAGE_MAJOR together."
        )


def _patch_for_local_ports(content: str) -> tuple[str, bool]:
    patched = False
    for old, new in _LOCAL_PORT_REPLACEMENTS.items():
        if old in content:
            content = content.replace(old, new)
            patched = True
    required = [
        "127.0.0.1:${LANGFUSE_WEB_PORT:-3300}:3000",
        "NEXTAUTH_URL: ${NEXTAUTH_URL:-http://localhost:3300}",
    ]
    missing = [needle for needle in required if needle not in content]
    if missing:
        raise RuntimeError(
            "Unable to prepare local Langfuse compose file; missing expected "
            f"local settings: {', '.join(missing)}"
        )
    return content, patched


def prepare_stack(*, target_dir: Path, source: Path | None = None) -> StackResult:
    """Create or patch the local Langfuse compose stack.

    The file is written once and only patched afterwards, so an existing stack is
    kept as-is unless it pins a Langfuse major other than ``LANGFUSE_IMAGE_MAJOR``.
    Delete the file to force a full re-download on a smaller upstream change.
    """
    compose_path = target_dir / "docker-compose.yml"
    created = not compose_path.exists()
    refreshed = False
    if created:
        content = _read_source(source)
    else:
        content = compose_path.read_text(encoding="utf-8")
        if _image_majors(content) != {LANGFUSE_IMAGE_MAJOR}:
            content = _read_source(source)
            refreshed = True
    content, patched = _patch_for_local_ports(content)
    _require_image_major(content, expected=LANGFUSE_IMAGE_MAJOR)
    target_dir.mkdir(parents=True, exist_ok=True)
    compose_path.write_text(content, encoding="utf-8")
    return StackResult(path=compose_path, created=created, refreshed=refreshed, patched=patched)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and prepare the local Langfuse Docker Compose stack",
        suggest_on_error=True,
    )
    parser.add_argument("--dir", type=Path, required=True, help="Target local Langfuse stack dir")
    parser.add_argument("--source", type=Path, default=None, help="Optional compose source file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_stack(target_dir=args.dir, source=args.source)
    action = "created" if result.created else "refreshed" if result.refreshed else "checked"
    suffix = " and patched for local ports" if result.patched else ""
    print(f"Langfuse compose stack {action}{suffix}: {result.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
