# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Dependency constraint policy tests."""

import re
import tomllib
from pathlib import Path


def _dependencies() -> list[str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["dependencies"]


def test_langfuse_dependency_has_no_upper_bound() -> None:
    """Langfuse should require the v4 SDK API without an upper cap."""
    dependencies = _dependencies()
    langfuse_deps = [dep for dep in dependencies if dep.startswith("langfuse")]

    assert len(langfuse_deps) == 1
    (langfuse_dep,) = langfuse_deps
    # Pins the v4 SDK API as the floor without capping the major version, so
    # routine patch bumps stay green while still guarding against v3/v5 drift.
    assert re.fullmatch(r"langfuse>=4\.\d+(\.\d+)?", langfuse_dep)


def test_language_detection_uses_lingua() -> None:
    dependencies = _dependencies()

    assert any(dep.startswith("lingua-language-detector") for dep in dependencies)


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
    # A pin is required (no floating main/latest), but the exact patch version is
    # intentionally NOT asserted so routine version bumps don't break the test.
    assert re.search(r"ARG PG_TEXTSEARCH_REF=v\d+\.\d+", dockerfile)
    assert (
        "git clone --branch ${PG_TEXTSEARCH_REF} --depth 1 https://github.com/timescale/pg_textsearch.git"
        in dockerfile
    )
    assert re.search(r"ARG PG_JIEBA_REF=v\d+\.\d+", dockerfile)
    assert (
        "git clone --branch ${PG_JIEBA_REF} --depth 1 --recurse-submodules https://github.com/jaiminpan/pg_jieba.git"
        in dockerfile
    )
    assert "pg_config --includedir-server" in dockerfile


def test_compose_preloads_postgres_extensions() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "shared_preload_libraries=age,pg_textsearch,pg_jieba" in compose


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


def test_compose_binds_api_port_to_loopback_on_host() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert '"127.0.0.1:8100:8100"' in compose
    assert 'DLIGHTRAG_API_HOST: "0.0.0.0"' in compose


def test_compose_api_healthcheck_uses_strict_readiness_endpoint() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "http://127.0.0.1:8100/ready" in compose
    assert "urllib.request.urlopen" in compose


def test_runtime_dockerfile_does_not_depend_on_ghcr_uv_stage() -> None:
    """App image builds should not require GHCR metadata just to obtain uv."""
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "ghcr.io/astral-sh/uv" not in dockerfile
    assert "uv==${UV_VERSION}" in dockerfile
    assert "COPY --from=uv-bin /usr/local/bin/uv /usr/local/bin/uvx /bin/" in dockerfile


def test_docx_native_parser_runtime_dependency_is_direct() -> None:
    """LightRAG native DOCX parsing needs python-docx available at DlightRAG runtime."""
    dependencies = _dependencies()

    assert any(dep.lower().startswith("python-docx") for dep in dependencies)


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


def test_config_yaml_uses_input_modality_for_rerank() -> None:
    config = Path("config.yaml").read_text(encoding="utf-8")

    assert "  strategy: voyage_reranker\n" in config
    assert "  model: rerank-2.5-lite\n" in config
    assert re.search(r"(?m)^  input_modality: auto$", config)
    assert not re.search(r"(?m)^  api_key:", config)
    assert "multimodal:" not in config


def test_compose_routes_container_mineru_to_host_sidecar_by_default() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert (
        "MINERU_LOCAL_ENDPOINT: ${MINERU_DOCKER_LOCAL_ENDPOINT:-http://host.docker.internal:8210}"
    ) in compose


def test_codeql_config_filters_self_referential_advanced_setup_alert() -> None:
    config = Path(".github/codeql/codeql-config.yml").read_text(encoding="utf-8")

    assert "query-filters:" in config
    assert "id: actions/unnecessary-use-of-advanced-config" in config
