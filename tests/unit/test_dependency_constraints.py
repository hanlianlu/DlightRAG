# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Dependency constraint policy tests."""

import json
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

_MANIFESTS = (
    Path("pyproject.toml"),
    Path("packages/memory/pyproject.toml"),
    Path("packages/observability/pyproject.toml"),
)


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


def _compose() -> dict[str, Any]:
    return yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))


def test_compose_publishes_every_port_on_host_loopback_only() -> None:
    """Exposure beyond this host is the operator's ingress, never a Compose default."""
    for name, service in _compose()["services"].items():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), (name, port)


def test_compose_mounts_the_operator_skills_root_read_only() -> None:
    """The global Skills root is operator-provisioned and read-only for the Agent.

    Compose must not create a missing host directory: an empty root it invented would
    hide a mistyped path behind a Run that silently has no Skills.
    """
    services = _compose()["services"]
    for name in ("dlightrag-api", "dlightrag-mcp"):
        (mount,) = (
            mount
            for mount in services[name]["volumes"]
            if isinstance(mount, dict) and mount.get("target") == "/home/app/.dlightrag/skills"
        )
        assert mount["type"] == "bind", name
        assert mount["read_only"] is True, name
        assert mount["bind"]["create_host_path"] is False, name


def test_compose_preloads_postgres_extensions() -> None:
    """Keyword search runs on these; without the preload it fails at query time."""
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "shared_preload_libraries=pg_textsearch,pg_jieba" in compose


def test_runtime_image_installs_the_library_the_copied_node_links() -> None:
    """Only the node binary crosses stages, so the runtime stage owns its libatomic.

    Without it node exits 127 on arm64, and every workspace tool and Skill that runs
    node fails (f8581a4e). CI builds no image, so this is the only guard.
    """
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    runtime_stage = dockerfile.rsplit("\nFROM ", 1)[1]

    assert "COPY --from=frontend /usr/local/bin/node /usr/local/bin/node" in runtime_stage
    assert re.search(
        r"apt-get install -y --no-install-recommends [^\n]*\blibatomic1\b", runtime_stage
    )


def test_compose_mcp_local_listener_passes_security_validation() -> None:
    from dlightrag.application.config.sections import DlightragConfig

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
