# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Every Agent Browser context is made one way: it inherits the launch proxy and nothing else (ADR 0032)."""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src" / "dlightrag"


def _uses(name: str) -> list[tuple[str, int]]:
    """Where the product source calls or names ``name`` as code, never in a docstring or comment."""
    found = []
    for path in sorted(SOURCE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Attribute) and node.attr == name) or (
                isinstance(node, ast.keyword) and node.arg == name
            ):
                found.append((path.name, node.lineno))
    return found


def test_the_one_place_a_context_is_made_is_the_agent_browser_adapter() -> None:
    assert [name for name, _ in _uses("new_context")] == ["leased_browser.py"]
    assert _uses("expose_network") == []
