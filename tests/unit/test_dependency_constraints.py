# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Dependency constraint policy tests."""

from __future__ import annotations

import tomllib
from pathlib import Path


def _dependencies() -> list[str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["dependencies"]


def test_lightrag_dependency_uses_github_main() -> None:
    dependencies = _dependencies()
    lightrag_deps = [dep for dep in dependencies if dep.startswith("lightrag-hku")]

    assert lightrag_deps == [
        "lightrag-hku @ git+https://github.com/HKUDS/LightRAG.git@main"
    ]


def test_legacy_multimodal_dependency_removed() -> None:
    dependencies = _dependencies()
    removed = "rag" + "anything"

    assert not any(dep.startswith(removed) for dep in dependencies)


def test_langfuse_dependency_has_no_upper_bound() -> None:
    """Langfuse should require the v4 SDK API without an upper cap."""
    dependencies = _dependencies()
    langfuse_deps = [dep for dep in dependencies if dep.startswith("langfuse")]

    assert langfuse_deps == ["langfuse>=4.0.0"]


def test_legacy_multimodal_source_removed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["tool"]["uv"].get("sources", {}).get("rag" + "anything") is None


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


def test_runtime_imports_do_not_reference_removed_multimodal_package() -> None:
    source_files = list(Path("src/dlightrag").rglob("*.py"))
    removed = "rag" + "anything"
    offenders = [
        path
        for path in source_files
        if removed in path.read_text(encoding="utf-8").lower()
    ]

    assert offenders == []


def test_dlightrag_does_not_ship_legacy_document_conversion_paths() -> None:
    """Document parsing should be delegated to LightRAG parser sidecars."""
    removed_paths = [
        Path("src/dlightrag/prompts/vision.py"),
        Path("src/dlightrag/core") / ("vlm" + "_ocr.py"),
        Path("src/dlightrag") / "converters" / "office.py",
        Path("tests/unit") / ("test_vlm" + "_ocr.py"),
        Path("tests/unit/test_office_converter.py"),
    ]

    assert [str(path) for path in removed_paths if path.exists()] == []


def test_direct_document_parser_dependencies_removed() -> None:
    """DlightRAG should not directly depend on parser/converter stacks LightRAG owns."""
    dependencies = _dependencies()
    removed_prefixes = ("doc" + "ling", "pypdf" + "ium2", "open" + "pyxl")

    assert [
        dep for dep in dependencies if dep.lower().startswith(removed_prefixes)
    ] == []


def test_default_parser_routing_has_no_legacy_fallback() -> None:
    """Default ingestion must not silently degrade into LightRAG legacy parsing."""
    from dlightrag.config import DlightragConfig, EmbeddingConfig

    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
    )

    assert "legacy" not in cfg.parser.rules.lower()


def test_office_conversion_config_removed() -> None:
    """Office conversion settings belonged to the removed converter path."""
    from dlightrag.config import DlightragConfig, EmbeddingConfig

    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
    )

    for name in (
        "excel_auto_convert_to_pdf",
        "excel_pdf_delete_original",
        "libreoffice_timeout",
        "libreoffice_pdf_quality",
    ):
        assert not hasattr(cfg, name)
