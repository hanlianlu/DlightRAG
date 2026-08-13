# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Ownership boundaries around the model-visible tool package.

One package owns every tool contract, its runtime, and its adapters; the
resource domain and the operator commands stay on their own side of that line.
"""

import importlib
import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]

_RETIRED_MODULES = (
    "dlightrag.core.agent.tool_loop",
    "dlightrag.core.agent.tools",
    "dlightrag.core.resources.tools",
    "dlightrag.tools",
)


@pytest.mark.parametrize("module", _RETIRED_MODULES)
def test_retired_tool_modules_are_gone(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_tool_package_exports_the_run_contract() -> None:
    tools = importlib.import_module("dlightrag.core.tools")

    assert set(tools.__all__) <= set(dir(tools))
    assert {"AgentTool", "ExactCallCache", "ToolResult", "compose_research_tools"} <= set(
        tools.__all__
    )


def test_resource_tool_module_exports_only_composition_factory() -> None:
    resources = importlib.import_module("dlightrag.core.tools.resources")

    assert resources.__all__ == ["build_resource_tools"]


def test_maintenance_package_is_covered_by_transport_boundary() -> None:
    config = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    contracts = config["tool"]["importlinter"]["contracts"]
    no_transport = next(
        contract for contract in contracts if contract["id"] == "no-transport-internals"
    )

    assert "dlightrag.maintenance" in no_transport["source_modules"]


def test_architecture_documents_all_import_contracts() -> None:
    architecture = (_REPO / "docs" / "architecture.md").read_text(encoding="utf-8")

    assert "`lint-imports` enforces six contracts" in architecture
    assert "resource domain" in architecture
    assert "model-visible tool" in architecture


def test_maintenance_owns_the_operator_console_scripts() -> None:
    scripts = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]

    assert scripts["dlightrag-rebuild-bm25"] == "dlightrag.maintenance.rebuild_bm25:main"
    assert scripts["dlightrag-rebuild-vdb"] == "dlightrag.maintenance.rebuild_vdb:main"
    assert callable(importlib.import_module("dlightrag.maintenance.rebuild_bm25").main)
    assert callable(importlib.import_module("dlightrag.maintenance.rebuild_vdb").main)
