# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Dependency constraint policy tests."""

import json
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

_MANIFESTS = (Path("pyproject.toml"), Path("packages/memory/pyproject.toml"))


def _project(path: Path = Path("pyproject.toml")) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def _dependencies(path: Path = Path("pyproject.toml")) -> list[str]:
    return _project(path)["dependencies"]  # type: ignore[return-value]


def test_workspace_requirements_use_floors_without_upper_bounds() -> None:
    for path in _MANIFESTS:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
        project = config["project"]
        requirements = [
            project["requires-python"],
            *project["dependencies"],
            *config["build-system"]["requires"],
        ]
        requirements.extend(
            requirement
            for group in config.get("dependency-groups", {}).values()
            for requirement in group
        )

        assert all(
            "<" not in requirement and "~=" not in requirement for requirement in requirements
        )


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


def test_runtime_uses_the_current_yaml_1_2_parser() -> None:
    dependencies = _dependencies()

    assert "ruamel.yaml>=0.19.1" in dependencies
    assert not any(dependency.startswith("pyyaml") for dependency in dependencies)


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


def _compose() -> dict[str, Any]:
    return yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))


def test_compose_publishes_every_port_on_host_loopback_only() -> None:
    """Exposure beyond this host is the operator's ingress, never a Compose default."""
    for name, service in _compose()["services"].items():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), (name, port)


def test_compose_mounts_the_operator_skills_root_read_only() -> None:
    """The global Skills root is operator-provisioned and read-only for the Agent."""
    for name, service in _compose()["services"].items():
        for mount in service.get("volumes", []):
            if isinstance(mount, dict) and mount.get("target") == "/home/app/.dlightrag/skills":
                assert mount["read_only"] is True, name


def test_compose_mcp_local_listener_passes_security_validation() -> None:
    from dlightrag.application.config import DlightragConfig

    environment = _compose()["services"]["dlightrag-mcp"]["environment"]

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


def test_docx_native_parser_runtime_dependency_is_direct() -> None:
    """LightRAG native DOCX parsing needs python-docx available at DlightRAG runtime."""
    dependencies = _dependencies()

    assert any(dep.lower().startswith("python-docx") for dep in dependencies)


def test_compose_reader_service_is_a_profiled_second_role() -> None:
    """The read-only replica is topology, not a hand-assembled docker run.

    A reader is a second process over the writer's database and configuration. It
    stays out of a default `docker compose up` (profile-gated), and its role must
    never appear in the writer's service definition.
    """
    services = _compose()["services"]
    reader = services["dlightrag-reader"]
    writer = services["dlightrag-api"]
    postgres_host = "DLIGHTRAG_STORAGE__POSTGRES__HOST"

    assert reader["profiles"] == ["reader"]
    assert reader["environment"]["DLIGHTRAG_DEPLOYMENT__SERVICE_ROLE"] == "reader"
    assert "DLIGHTRAG_DEPLOYMENT__SERVICE_ROLE" not in writer["environment"]
    assert reader["environment"][postgres_host] == writer["environment"][postgres_host]
    assert reader["configs"] == writer["configs"]
    assert reader["healthcheck"]["test"] == writer["healthcheck"]["test"]

    # The role is exercised against the shared corpus root, so a reader must see
    # the same storage and Answer-workspace mounts as the writer.
    def mounts(volumes: object) -> list[str]:
        assert isinstance(volumes, list)
        return sorted(
            json.dumps(v, sort_keys=True) if isinstance(v, dict) else str(v) for v in volumes
        )

    assert mounts(reader["volumes"]) == mounts(writer["volumes"])
