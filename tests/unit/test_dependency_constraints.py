# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Dependency constraint policy tests."""

from __future__ import annotations

import tomllib
from pathlib import Path


def _dependencies() -> list[str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["dependencies"]


def test_lightrag_dependency_has_no_upper_bound() -> None:
    """LightRAG should be minimum-pinned for 1.5+ without an upper cap."""
    dependencies = _dependencies()
    lightrag_deps = [dep for dep in dependencies if dep.startswith("lightrag-hku")]

    assert lightrag_deps == ["lightrag-hku>=1.5.0rc1"]


def test_raganything_dependency_has_no_upper_bound() -> None:
    """RAGAnything should be minimum-pinned for 1.3+ without an upper cap."""
    dependencies = _dependencies()
    raganything_deps = [dep for dep in dependencies if dep.startswith("raganything")]

    assert raganything_deps == ["raganything[all]>=1.3.0"]


def test_langfuse_dependency_has_no_upper_bound() -> None:
    """Langfuse should require the v4 SDK API without an upper cap."""
    dependencies = _dependencies()
    langfuse_deps = [dep for dep in dependencies if dep.startswith("langfuse")]

    assert langfuse_deps == ["langfuse>=4.0.0"]


def test_raganything_uses_registry_release_not_git_main() -> None:
    """RAGAnything should not drift on the GitHub main branch."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["tool"]["uv"].get("sources", {}).get("raganything") is None


def test_pg_trgm_removed_from_runtime_and_postgres_init() -> None:
    """Metadata filtering should not depend on pg_trgm or trigram similarity."""
    runtime_files = list(Path("src/dlightrag").rglob("*.py")) + [Path("postgres/init.sql")]
    offenders = []
    for path in runtime_files:
        text = path.read_text(encoding="utf-8").lower()
        if "pg_trgm" in text or "gin_trgm" in text or "similarity(" in text:
            offenders.append(str(path))

    assert offenders == []


def test_postgres_init_uses_required_pg18_extensions() -> None:
    init_sql = Path("postgres/init.sql").read_text(encoding="utf-8")

    assert "CREATE EXTENSION IF NOT EXISTS vector;" in init_sql
    assert "CREATE EXTENSION IF NOT EXISTS age;" in init_sql
    assert "CREATE EXTENSION IF NOT EXISTS pg_textsearch;" in init_sql


def test_postgres_dockerfile_targets_pg18_ecosystem() -> None:
    dockerfile = Path("postgres/Dockerfile").read_text(encoding="utf-8")

    assert "pgvector/pgvector:pg18" in dockerfile
    assert "postgresql-server-dev-18" in dockerfile
    assert "--branch PG18" in dockerfile
    assert "pg_textsearch" in dockerfile


def test_compose_preloads_postgres_extensions() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "shared_preload_libraries=age,pg_textsearch" in compose
