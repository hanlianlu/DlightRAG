# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for local Langfuse stack preparation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_script_path = Path(__file__).resolve().parents[2] / "scripts" / "langfuse" / "stack.py"
_spec = importlib.util.spec_from_file_location("langfuse_stack", _script_path)
_langfuse_stack = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_langfuse_stack)

prepare_stack = _langfuse_stack.prepare_stack


OFFICIAL_COMPOSE_SAMPLE = """\
services:
  langfuse-worker:
    ports:
      - 127.0.0.1:3030:3030
    environment:
      NEXTAUTH_URL: ${NEXTAUTH_URL:-http://localhost:3000}
      LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT: ${LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT:-http://localhost:9090}
      LANGFUSE_S3_BATCH_EXPORT_EXTERNAL_ENDPOINT: ${LANGFUSE_S3_BATCH_EXPORT_EXTERNAL_ENDPOINT:-http://localhost:9090}
  langfuse-web:
    ports:
      - 3000:3000
  clickhouse:
    ports:
      - 127.0.0.1:8123:8123
      - 127.0.0.1:9000:9000
  minio:
    ports:
      - 9090:9000
      - 127.0.0.1:9091:9001
  redis:
    ports:
      - 127.0.0.1:6379:6379
  postgres:
    ports:
      - 127.0.0.1:5432:5432
"""


def test_prepare_stack_downloads_and_patches_local_ports(tmp_path: Path) -> None:
    source = tmp_path / "source-compose.yml"
    source.write_text(OFFICIAL_COMPOSE_SAMPLE, encoding="utf-8")
    target_dir = tmp_path / "langfuse-local"

    result = prepare_stack(target_dir=target_dir, source=source)

    assert result.created is True
    assert result.path == target_dir / "docker-compose.yml"
    compose = result.path.read_text(encoding="utf-8")
    assert "127.0.0.1:${LANGFUSE_WEB_PORT:-3300}:3000" in compose
    assert "NEXTAUTH_URL: ${NEXTAUTH_URL:-http://localhost:3300}" in compose
    assert "127.0.0.1:${LANGFUSE_WORKER_PORT:-3301}:3030" in compose
    assert "127.0.0.1:${LANGFUSE_MINIO_API_PORT:-3390}:9000" in compose
    assert "127.0.0.1:${LANGFUSE_POSTGRES_PORT:-15432}:5432" in compose


def test_prepare_stack_is_idempotent_for_existing_stack(tmp_path: Path) -> None:
    target_dir = tmp_path / "langfuse-local"
    target_dir.mkdir()
    compose_path = target_dir / "docker-compose.yml"
    compose_path.write_text(OFFICIAL_COMPOSE_SAMPLE, encoding="utf-8")

    first = prepare_stack(target_dir=target_dir)
    second = prepare_stack(target_dir=target_dir)

    assert first.created is False
    assert second.created is False
    assert first.path.read_text(encoding="utf-8") == second.path.read_text(encoding="utf-8")
