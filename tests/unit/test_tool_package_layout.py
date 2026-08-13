# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Ownership boundaries around the model-visible tool package.

One package owns every tool contract, its runtime, and its adapters; the
resource domain and the operator commands stay on their own side of that line.
"""

import ast
import importlib
import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "dlightrag"

_TOOL_MODULES = (
    "dlightrag.core.tools.models",
    "dlightrag.core.tools.executor",
    "dlightrag.core.tools.cache",
    "dlightrag.core.tools.composition",
    "dlightrag.core.tools.search",
    "dlightrag.core.tools.resources",
)

_RETIRED_MODULES = (
    "dlightrag.core.agent.tool_loop",
    "dlightrag.core.agent.tools",
    "dlightrag.core.resources.tools",
    "dlightrag.tools",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("module", _TOOL_MODULES)
def test_tool_package_owns_every_model_visible_module(module: str) -> None:
    assert importlib.import_module(module).__name__ == module


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


def test_agent_package_keeps_only_orchestration() -> None:
    modules = {path.stem for path in (_SRC / "core" / "agent").glob("*.py")}

    assert modules == {"__init__", "context", "orchestrator"}


def test_resource_domain_has_no_tool_back_edge() -> None:
    offenders = {
        path.name: back_edges
        for path in (_SRC / "core" / "resources").glob("*.py")
        if (
            back_edges := sorted(
                name for name in _imported_modules(path) if name.startswith("dlightrag.core.tools")
            )
        )
    }

    assert offenders == {}


def test_maintenance_owns_the_operator_console_scripts() -> None:
    scripts = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]

    assert scripts["dlightrag-rebuild-bm25"] == "dlightrag.maintenance.rebuild_bm25:main"
    assert scripts["dlightrag-rebuild-vdb"] == "dlightrag.maintenance.rebuild_vdb:main"
    assert callable(importlib.import_module("dlightrag.maintenance.rebuild_bm25").main)
    assert callable(importlib.import_module("dlightrag.maintenance.rebuild_vdb").main)
