# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Dependency constraint policy tests."""

import re
import tomllib
from pathlib import Path

import pytest
import yaml

_MANIFESTS = (Path("pyproject.toml"), Path("packages/memory/pyproject.toml"))


def _project(path: Path = Path("pyproject.toml")) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def _dependencies(path: Path = Path("pyproject.toml")) -> list[str]:
    return _project(path)["dependencies"]  # type: ignore[return-value]


def test_workspace_versions_are_lockstep() -> None:
    versions = {_project(path)["version"] for path in _MANIFESTS}

    assert len(versions) == 1


def test_root_is_batteries_included_and_depends_only_on_standalone_memory() -> None:
    dependencies = _dependencies()
    version = _project()["version"]

    assert f"dlightrag-memory=={version}" in dependencies
    assert all(
        any(dependency.startswith(name) for dependency in dependencies)
        for name in (
            "openai",
            "anthropic",
            "google-genai",
            "json-repair",
            "aiofiles",
            "aiobotocore",
            "azure-storage-blob",
            "botocore",
            "lightrag-hku",
            "lingua-language-detector",
        )
    )
    assert [dependency for dependency in dependencies if dependency.startswith("dlightrag-")] == [
        f"dlightrag-memory=={version}"
    ]


def test_root_has_no_server_template_runtime_dependency() -> None:
    names = {
        re.split(r"[<>=!~\[]", dependency.lower(), maxsplit=1)[0] for dependency in _dependencies()
    }

    assert names.isdisjoint({"jinja2", "markupsafe"})


def test_workspace_sources_and_lock_are_exact() -> None:
    root = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert root["tool"]["uv"]["workspace"]["members"] == ["packages/memory"]
    assert root["tool"]["uv"]["sources"] == {
        "dlightrag-memory": {"workspace": True},
    }

    lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))
    workspace_sources = {
        package["name"]: package["source"]
        for package in lock["package"]
        if package["name"].startswith("dlightrag")
    }
    assert workspace_sources == {
        "dlightrag": {"editable": "."},
        "dlightrag-memory": {"editable": "packages/memory"},
    }


def test_eval_dependency_group_uses_lightrag_evaluation_extra() -> None:
    """The eval group may add the evaluation extra, but must not drift off the runtime floor."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    (runtime_pin,) = [dep for dep in _dependencies() if dep.startswith("lightrag-hku")]
    version_spec = runtime_pin.removeprefix("lightrag-hku")

    assert pyproject["dependency-groups"]["eval"] == [f"lightrag-hku[evaluation]{version_spec}"]


def test_langfuse_dependency_has_no_upper_bound() -> None:
    """Langfuse should require the v4 SDK API without an upper cap."""
    dependencies = _dependencies()
    langfuse_deps = [dep for dep in dependencies if dep.startswith("langfuse")]

    assert len(langfuse_deps) == 1
    (langfuse_dep,) = langfuse_deps
    # Pins the v4 SDK API as the floor without capping the major version, so
    # routine patch bumps stay green while still guarding against v3/v5 drift.
    assert re.fullmatch(r"langfuse>=4\.\d+(\.\d+)?", langfuse_dep)


def test_language_detection_dependency_is_owned_by_root() -> None:
    assert any(dep.startswith("lingua-language-detector") for dep in _dependencies())


def test_postgres_init_uses_required_pg18_extensions() -> None:
    init_sql = Path("postgres/init.sql").read_text(encoding="utf-8")

    assert "CREATE EXTENSION IF NOT EXISTS vector;" in init_sql
    assert "CREATE EXTENSION IF NOT EXISTS pg_textsearch;" in init_sql
    assert "CREATE EXTENSION IF NOT EXISTS pg_jieba;" in init_sql


def test_postgres_dockerfile_targets_pg18_ecosystem() -> None:
    dockerfile = Path("postgres/Dockerfile").read_text(encoding="utf-8")

    assert "pgvector/pgvector:pg18" in dockerfile
    assert "postgresql-server-dev-18" in dockerfile
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

    assert "shared_preload_libraries=pg_textsearch,pg_jieba" in compose


def test_compose_postgres_endpoint_is_env_overridable() -> None:
    """The container wires the service host while env still outranks YAML."""
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "DLIGHTRAG_STORAGE__POSTGRES__HOST: postgres" in compose
    assert "DLIGHTRAG_STORAGE__POSTGRES__PORT" not in compose


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
    assert 'DLIGHTRAG_INTERFACES__API__HOST: "0.0.0.0"' in compose


def test_compose_mcp_local_listener_passes_security_validation() -> None:
    from dlightrag.application.config import DlightragConfig

    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    environment = compose["services"]["dlightrag-mcp"]["environment"]

    with pytest.warns(UserWarning, match="allow_insecure_no_auth"):
        config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
            interfaces={
                "mcp": {
                    "transport": environment["DLIGHTRAG_INTERFACES__MCP__TRANSPORT"],
                    "host": environment["DLIGHTRAG_INTERFACES__MCP__HOST"],
                    "port": environment["DLIGHTRAG_INTERFACES__MCP__PORT"],
                },
            },
            access={
                "allow_insecure_no_auth": (
                    environment.get("DLIGHTRAG_ACCESS__ALLOW_INSECURE_NO_AUTH") == "true"
                ),
            },
        )

    assert config.interfaces.mcp.host == "0.0.0.0"


def test_compose_api_healthcheck_uses_strict_readiness_endpoint() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "http://127.0.0.1:8100/ready" in compose
    assert "urllib.request.urlopen" in compose


def test_runtime_dockerfile_does_not_depend_on_ghcr_uv_stage() -> None:
    """App image builds should not require GHCR metadata just to obtain uv."""
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "ghcr.io/astral-sh/uv" not in dockerfile
    assert "uv==${UV_VERSION}" in dockerfile
    assert "COPY --from=uv-bin /usr/local/bin/uv /bin/" in dockerfile


def test_runtime_image_defaults_to_api_and_mcp_overrides_it() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))

    assert 'CMD ["dlightrag-api"]' in dockerfile
    assert "command" not in compose["services"]["dlightrag-api"]
    assert compose["services"]["dlightrag-mcp"]["command"] == ["dlightrag-mcp"]


def test_docx_native_parser_runtime_dependency_is_direct() -> None:
    """LightRAG native DOCX parsing needs python-docx available at DlightRAG runtime."""
    dependencies = _dependencies()

    assert any(dep.lower().startswith("python-docx") for dep in dependencies)


def test_default_parser_routing_has_no_unrouted_fallback() -> None:
    """Default ingestion must not silently degrade into an unrouted parser path."""
    from dlightrag.ai.settings import EmbeddingSettings
    from dlightrag.application.config import DlightragConfig

    cfg = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        models={
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
        },
    )

    assert ("leg" + "acy") not in cfg.corpus.parser_rules.lower()


def test_config_yaml_uses_input_modality_for_rerank() -> None:
    config = Path("config.yaml").read_text(encoding="utf-8")

    assert re.search(r"(?m)^    input_modality: auto$", config)
    assert not re.search(r"(?m)^    api_key:", config)
    assert "multimodal:" not in config


def test_curated_config_routes_container_to_host_native_docling_by_default() -> None:
    config = Path("config.yaml").read_text(encoding="utf-8")
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "endpoint: http://host.docker.internal:5001" in config
    assert "code_formula_preset: granite_docling" in config
    assert "local_endpoint: http://host.docker.internal:8210" not in config
    assert "DLIGHTRAG_CORPUS__SIDECARS__MINERU__LOCAL_ENDPOINT" not in compose
    assert "DLIGHTRAG_CORPUS__SIDECARS__DOCLING__ENDPOINT" not in compose


def test_codeql_config_filters_self_referential_advanced_setup_alert() -> None:
    config = Path(".github/codeql/codeql-config.yml").read_text(encoding="utf-8")

    assert "query-filters:" in config
    assert "id: actions/unnecessary-use-of-advanced-config" in config


def test_ci_runs_frontend_unit_and_browser_tests() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["fast"]["steps"]
    commands = [step["run"] for step in steps if "run" in step]
    expected = [
        "make frontend-install",
        "npx playwright install --with-deps chromium",
        "make frontend-typecheck",
        "make frontend-lint",
        "make frontend-test",
        "make frontend-browser-test",
        "make workspace-wheels",
        "make frontend-audit",
    ]

    assert [commands.index(command) for command in expected] == sorted(
        commands.index(command) for command in expected
    )
    browser_install = next(step for step in steps if step.get("run") == expected[1])
    assert browser_install["working-directory"] == "frontend"


def test_manual_e2e_ci_builds_the_gitignored_frontend() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci-e2e.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["e2e-pg18"]["steps"]
    commands = [step["run"] for step in steps if "run" in step]
    expected = [
        "uv sync --group dev",
        "make frontend-install",
        "uv run playwright install --with-deps chromium",
        "make frontend-build",
        "uv run pytest tests/e2e -v --tb=long -m e2e_pg18",
    ]

    assert any(str(step.get("uses", "")).startswith("actions/setup-node@") for step in steps)
    assert [commands.index(command) for command in expected] == sorted(
        commands.index(command) for command in expected
    )
