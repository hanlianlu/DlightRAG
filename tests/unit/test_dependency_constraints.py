# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Dependency constraint policy tests."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


def _dependencies() -> list[str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["dependencies"]


def test_removed_multimodal_dependency_absent() -> None:
    dependencies = _dependencies()
    removed = "rag" + "anything"

    assert not any(dep.startswith(removed) for dep in dependencies)


def test_lightrag_dependency_tracks_stable_1_5_release() -> None:
    dependencies = _dependencies()
    lightrag_deps = [dep for dep in dependencies if dep.startswith("lightrag-hku")]

    assert lightrag_deps == ["lightrag-hku>=1.5.4"]


def test_langfuse_dependency_has_no_upper_bound() -> None:
    """Langfuse should require the v4 SDK API without an upper cap."""
    dependencies = _dependencies()
    langfuse_deps = [dep for dep in dependencies if dep.startswith("langfuse")]

    assert langfuse_deps == ["langfuse>=4.12.0"]


def test_language_detection_uses_lingua_not_legacy_detector() -> None:
    dependencies = _dependencies()
    removed = "lang" + "detect"

    assert any(dep.startswith("lingua-language-detector") for dep in dependencies)
    assert not any(dep.startswith(removed) for dep in dependencies)


def test_removed_multimodal_source_absent() -> None:
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
    assert "CREATE EXTENSION IF NOT EXISTS pg_jieba;" in init_sql


def test_postgres_dockerfile_targets_pg18_ecosystem() -> None:
    dockerfile = Path("postgres/Dockerfile").read_text(encoding="utf-8")

    assert "pgvector/pgvector:pg18" in dockerfile
    assert "postgresql-server-dev-18" in dockerfile
    assert "--branch PG18" in dockerfile
    assert "ARG PG_TEXTSEARCH_REF=v1.3.0" in dockerfile
    assert (
        "git clone --branch ${PG_TEXTSEARCH_REF} --depth 1 https://github.com/timescale/pg_textsearch.git"
        in dockerfile
    )
    assert "ARG PG_JIEBA_REF=v2.0.1" in dockerfile
    assert (
        "git clone --branch ${PG_JIEBA_REF} --depth 1 --recurse-submodules https://github.com/jaiminpan/pg_jieba.git"
        in dockerfile
    )
    assert "pg_config --includedir-server" in dockerfile


def test_compose_preloads_postgres_extensions() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "shared_preload_libraries=age,pg_textsearch,pg_jieba" in compose


def test_removed_chinese_parser_is_not_a_supported_runtime_surface() -> None:
    runtime_files = (
        list(Path("src/dlightrag").rglob("*.py"))
        + list(Path("tests").rglob("*.py"))
        + [Path("postgres/Dockerfile"), Path("postgres/init.sql"), Path("config.yaml")]
    )
    removed = "zh" + "parser"
    offenders = [str(path) for path in runtime_files if removed in path.read_text()]

    assert offenders == []


def test_compose_postgres_performance_knobs_are_env_overridable() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    for setting, default in {
        "shared_buffers": "8GB",
        "work_mem": "256MB",
        "maintenance_work_mem": "2GB",
        "effective_cache_size": "18GB",
        "max_connections": "80",
    }.items():
        env_name = f"DLIGHTRAG_POSTGRES_{setting.upper()}"
        assert f"{setting}=${{{env_name}:-{default}}}" in compose


def test_compose_builds_pg18_postgres_image_locally() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/postgres-image.yml").read_text(encoding="utf-8")

    assert "image: dlightrag-postgres:pg18" in compose
    assert "context: postgres" in compose
    assert "ghcr.io/hanlianlu/dlightrag-postgres" not in compose
    assert "ghcr.io/hanlianlu/dlightrag-postgres:latest" not in compose
    assert "dlightrag-postgres:pg18" in workflow


def test_compose_runtime_services_do_not_bind_mount_source_tree() -> None:
    """Default compose should run the built image, not a host source overlay."""
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "./src:/app/src" not in compose


def test_runtime_imports_do_not_reference_removed_multimodal_package() -> None:
    source_files = list(Path("src/dlightrag").rglob("*.py"))
    removed = "rag" + "anything"
    offenders = [
        path for path in source_files if removed in path.read_text(encoding="utf-8").lower()
    ]

    assert offenders == []


def test_dlightrag_does_not_ship_removed_document_conversion_paths() -> None:
    """Document parsing should be delegated to LightRAG parser sidecars."""
    removed_paths = [
        Path("src/dlightrag/prompts/vision.py"),
        Path("src/dlightrag/core") / ("vlm" + "_ocr.py"),
        Path("src/dlightrag") / "converters" / "office.py",
        Path("tests/unit") / ("test_vlm" + "_ocr.py"),
        Path("tests/unit/test_office_converter.py"),
    ]

    assert [str(path) for path in removed_paths if path.exists()] == []


def test_runtime_dockerfile_does_not_install_removed_office_converter_stack() -> None:
    """The app image should not carry converter packages owned by old paths."""
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8").lower()

    for removed in ("libreoffice", "libgl1-mesa-glx"):
        assert removed not in dockerfile


def test_runtime_dockerfile_does_not_depend_on_ghcr_uv_stage() -> None:
    """App image builds should not require GHCR metadata just to obtain uv."""
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "ghcr.io/astral-sh/uv" not in dockerfile
    assert "uv==${UV_VERSION}" in dockerfile
    assert "COPY --from=uv-bin /usr/local/bin/uv /usr/local/bin/uvx /bin/" in dockerfile


def test_direct_document_parser_dependencies_removed() -> None:
    """DlightRAG should not directly depend on parser/converter stacks LightRAG owns."""
    dependencies = _dependencies()
    removed_prefixes = ("doc" + "ling", "pypdf" + "ium2", "open" + "pyxl")

    assert [dep for dep in dependencies if dep.lower().startswith(removed_prefixes)] == []


def test_docx_native_parser_runtime_dependency_is_direct() -> None:
    """LightRAG native DOCX parsing needs python-docx available at DlightRAG runtime."""
    dependencies = _dependencies()

    assert any(dep.lower().startswith("python-docx") for dep in dependencies)


def test_runtime_imports_do_not_reference_old_lightrag_docx_entrypoint() -> None:
    source_files = list(Path("src/dlightrag").rglob("*.py")) + list(Path("tests").rglob("*.py"))
    old_entrypoint = "lightrag.native_parser" + ".docx"
    offenders = [
        str(path) for path in source_files if old_entrypoint in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_default_parser_routing_has_no_unrouted_fallback() -> None:
    """Default ingestion must not silently degrade into an unrouted parser path."""
    from dlightrag.config import DlightragConfig, EmbeddingConfig

    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
    )

    assert ("leg" + "acy") not in cfg.parser.rules.lower()


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


def test_env_example_documents_config_first_parser_sidecar_policy() -> None:
    example = Path(".env.example").read_text(encoding="utf-8")

    assert "parser_sidecars" in example
    assert "DLIGHTRAG_PARSER_SIDECARS__MINERU__API_TOKEN" in example
    for name in ("VLM_PROCESS_ENABLE", "MINERU_API_MODE", "MINERU_LOCAL_ENDPOINT"):
        assert name not in example


def test_env_example_defaults_mineru_to_local_sidecar() -> None:
    example = Path(".env.example").read_text(encoding="utf-8")
    config = Path("config.yaml").read_text(encoding="utf-8")

    assert "parser_sidecars:" in config
    assert re.search(r"(?m)^    api_mode: local$", config)
    assert re.search(r"(?m)^    local_endpoint: http://127\.0\.0\.1:8210$", config)
    for active_non_secret in (
        "MINERU_API_MODE",
        "MINERU_LOCAL_ENDPOINT",
        "MINERU_LOCAL_BACKEND",
        "MINERU_LOCAL_PARSE_METHOD",
        "MINERU_LOCAL_IMAGE_ANALYSIS",
        "MINERU_ENABLE_TABLE",
        "MINERU_ENABLE_FORMULA",
        "VLM_PROCESS_ENABLE",
        "VLM_MAX_IMAGE_BYTES",
    ):
        assert not re.search(rf"(?m)^{active_non_secret}=", example)
    assert "MINERU_API_TOKEN" not in example
    assert "MINERU_OFFICIAL_ENDPOINT" not in example


def test_env_example_active_keys_are_credentials_only() -> None:
    example = Path(".env.example").read_text(encoding="utf-8")
    active_keys = []
    for line in example.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        active_keys.append(stripped.split("=", 1)[0].strip())

    assert active_keys == [
        "DLIGHTRAG_LLM__DEFAULT__API_KEY",
        "DLIGHTRAG_EMBEDDING__API_KEY",
        "DLIGHTRAG_LLM__ROLES__EXTRACT__API_KEY",
        "DLIGHTRAG_LLM__ROLES__KEYWORD__API_KEY",
        "DLIGHTRAG_POSTGRES_PASSWORD",
    ]


def test_env_example_describes_config_as_curated_not_tuning_dump() -> None:
    example = Path(".env.example").read_text(encoding="utf-8")

    assert "high-signal product" in example
    assert "deployment choices" in example
    assert "docs/config-reference.md" in example
    assert "PostgreSQL tuning" not in example


def test_compose_routes_container_mineru_to_host_sidecar_by_default() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert (
        "MINERU_LOCAL_ENDPOINT: ${MINERU_DOCKER_LOCAL_ENDPOINT:-http://host.docker.internal:8210}"
    ) in compose


def test_env_example_has_no_removed_keys() -> None:
    example = Path(".env.example").read_text(encoding="utf-8").lower()

    for removed in (
        "dlightrag_" + "chat__",
        "excel_auto_convert_to_pdf",
        "libreoffice_timeout",
        "libreoffice_pdf_quality",
    ):
        assert removed not in example
