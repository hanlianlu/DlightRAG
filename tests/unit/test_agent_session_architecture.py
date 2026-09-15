# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Deletion and dependency guards for the final durable Agent architecture."""

import ast
from pathlib import Path

from dlightrag.engine.agent.session.repository import AgentSessionRepository

ROOT = Path(__file__).parents[2]
SRC = ROOT / "src" / "dlightrag"
AGENT = SRC / "engine" / "agent"


def test_agent_session_runtime_never_depends_on_answer_product_modules() -> None:
    violations: list[str] = []
    for path in (AGENT / "session").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "dlightrag.engine.answer"
            ):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.relative_to(ROOT)}:{node.lineno}"
                    for alias in node.names
                    if alias.name.startswith("dlightrag.engine.answer")
                )
    assert violations == []


def test_removed_extension_entry_and_store_passthroughs_cannot_return() -> None:
    assert not (AGENT / "extensions.py").exists()
    forbidden = (
        "TrustedExtensions",
        "AdoptionEntry",
        "AgentSessionStore",
        "append_to_lane",
        "fork_lane",
        "archive_lane",
    )
    matches: dict[str, list[str]] = {name: [] for name in forbidden}
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            if name in text:
                matches[name].append(str(path.relative_to(ROOT)))
    assert matches == {name: [] for name in forbidden}


def test_session_repository_exposes_only_snapshot_and_transaction_primitives() -> None:
    public_methods = {
        name
        for name, value in AgentSessionRepository.__dict__.items()
        if callable(value) and not name.startswith("_")
    }
    assert public_methods == {"load", "refresh", "transact"}


def test_the_legacy_event_driven_tool_loop_cannot_return() -> None:
    """One live tool path: the durable AgentSessionRuntime.

    The event-driven executor was the second implementation of tool dispatch; it
    kept its own copy of settlement metadata, which is how the browser ended up
    with a dead ``duration_ms`` field. Its module and event type are gone.
    """
    assert not (AGENT / "tools" / "executor.py").exists()
    assert not (AGENT / "events.py").exists()
    forbidden = (
        "ToolTurnExecutor",
        "PreparedToolTurn",
        "ToolPreflight",
        "preflight_tool_calls",
        "DuplicateToolCallIdError",
        "ToolObservation",
        "ToolExecution",
        "AgentEvent",
    )
    matches: dict[str, list[str]] = {name: [] for name in forbidden}
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            if name in text:
                matches[name].append(str(path.relative_to(ROOT)))
    assert matches == {name: [] for name in forbidden}
