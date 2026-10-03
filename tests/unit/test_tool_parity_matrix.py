# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Test-maintained parity matrix: DlightRAG base tools vs the Pi baseline.

DlightRAG adds direct pixel view beside the Pi-shaped read, bash, edit, write,
grep, find and ls tools. This matrix owns their current argument surfaces,
persistence, cursor and safety contracts, as an Answer run composes them: rooted
in an Agent Workspace and backed by the resource reader and viewer.
"""

from pathlib import Path
from unittest.mock import AsyncMock

from dlightrag.engine.agent.environment import SearchToolchain
from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
from dlightrag.engine.agent.tools.contracts import AgentTool
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.tools.composition import compose_research_tools

# tool name -> (required params, optional params with defaults, replay policy, contract)
MATRIX: dict[str, tuple[tuple[str, ...], dict[str, object], str, int]] = {
    "read": (
        (),
        {
            "path": None,
            "resource_id": None,
            "url": None,
            "http": None,
            "offset": None,
            "limit": None,
            "focus": None,
            "cursor": None,
        },
        "replayable",
        4,
    ),
    "view": (
        (),
        {
            "path": None,
            "resource_id": None,
            "url": None,
            "http": None,
            "locator": None,
            "cursor": None,
        },
        "replayable",
        2,
    ),
    "bash": (("command",), {"timeout_seconds": None}, "never", 3),
    "edit": (
        ("path", "edits"),
        {},
        "never",
        3,
    ),
    "write": (("path", "content"), {}, "never", 3),
    "grep": (
        ("pattern",),
        {
            "path": ".",
            "glob": None,
            "ignore_case": False,
            "literal": False,
            "context": None,
            "limit": 100,
        },
        "replayable",
        3,
    ),
    "find": (("pattern",), {"path": ".", "limit": 1000}, "replayable", 2),
    "ls": ((), {"path": ".", "limit": 500, "cursor": None}, "replayable", 2),
}


def _tools(tmp_path: Path) -> list[AgentTool]:
    composed = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=AsyncMock(),
        search_web=None,
        injected_tools=[],
        register_web_source=None,
        resource_reader=AsyncMock(),
        resource_viewer=AsyncMock(),
        environment=LocalExecutionEnvironment(tmp_path),
        search_toolchain=SearchToolchain(),
    )
    return [tool for tool in composed if tool.name in MATRIX]


def test_tool_names_and_order_match_the_current_contract(tmp_path: Path) -> None:
    assert [tool.name for tool in _tools(tmp_path)] == list(MATRIX)


def test_argument_surfaces_and_defaults_match_the_matrix(tmp_path: Path) -> None:
    for tool in _tools(tmp_path):
        required, defaults, _, _ = MATRIX[tool.name]
        fields = tool.input_model.model_fields
        assert set(fields) == set(required) | set(defaults), tool.name
        for name in required:
            assert fields[name].is_required(), (tool.name, name)
        for name, expected in defaults.items():
            assert fields[name].default == expected, (tool.name, name, expected)


def test_replay_policies_and_contract_versions_match_the_matrix(tmp_path: Path) -> None:
    for tool in _tools(tmp_path):
        _, _, replay_policy, contract_version = MATRIX[tool.name]
        assert tool.replay_policy == replay_policy, tool.name
        assert tool.contract_version == contract_version, tool.name
