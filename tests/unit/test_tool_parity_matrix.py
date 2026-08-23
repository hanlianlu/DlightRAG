# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Test-maintained parity matrix: DlightRAG base tools vs the Pi baseline.

Pi 0.84.2 exposes seven first-class filesystem tools in this exact order:
read, bash, edit, write, grep, find, ls. This matrix pins names, order,
argument surfaces, defaults, and replay/contract facts so drift from the
negotiated baseline fails CI instead of passing silently.
"""

from pathlib import Path

from dlightrag.agent.environment import AccessScheduler, LocalExecutionEnvironment
from dlightrag.agent.tools.files import path_tools

# tool name -> (required params, optional params with defaults, replay policy, contract)
MATRIX: dict[str, tuple[tuple[str, ...], dict[str, object], str, int]] = {
    "read": (
        (),
        {
            "path": None,
            "resource_id": None,
            "offset": None,
            "limit": None,
            "focus": None,
            "cursor": None,
        },
        "safe",
        2,
    ),
    "bash": (("command",), {"timeout_seconds": None}, "never", 2),
    "edit": (
        ("path", "edits"),
        {},
        "never",
        2,
    ),
    "write": (("path", "content"), {}, "never", 2),
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
        "safe",
        2,
    ),
    "find": (("pattern",), {"path": ".", "limit": 1000}, "safe", 1),
    "ls": ((), {"path": ".", "limit": 500}, "safe", 1),
}


def _tools(tmp_path: Path):
    environment = LocalExecutionEnvironment(tmp_path)
    return path_tools(environment, scheduler=AccessScheduler())


def test_tool_names_and_order_match_the_pi_baseline(tmp_path: Path) -> None:
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


def test_every_tool_carries_short_usage_guidance(tmp_path: Path) -> None:
    for tool in _tools(tmp_path):
        assert tool.guidance.strip(), tool.name
        assert len(tool.guidance) <= 400, tool.name
