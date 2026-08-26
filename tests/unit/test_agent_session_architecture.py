# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Deletion and dependency guards for the final durable Agent architecture."""

import ast
from pathlib import Path

from dlightrag.agent.session.repository import AgentSessionRepository

ROOT = Path(__file__).parents[2]
SRC = ROOT / "src" / "dlightrag"


def test_agent_session_runtime_never_depends_on_answer_product_modules() -> None:
    violations: list[str] = []
    for path in (SRC / "agent" / "session").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "dlightrag.answer"
            ):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.relative_to(ROOT)}:{node.lineno}"
                    for alias in node.names
                    if alias.name.startswith("dlightrag.answer")
                )
    assert violations == []


def test_removed_extension_entry_and_store_passthroughs_cannot_return() -> None:
    assert not (SRC / "agent" / "extensions.py").exists()
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


def test_session_repository_exposes_only_open_and_transaction_primitives() -> None:
    public_methods = {
        name
        for name, value in AgentSessionRepository.__dict__.items()
        if callable(value) and not name.startswith("_")
    }
    assert public_methods == {"load", "transact"}
