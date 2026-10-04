# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Every Agent Browser context is made one way: it inherits the launch proxy and nothing else (ADR 0032)."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

import pytest
from playwright.async_api import Browser

from dlightrag.adapters.agent_browser.playwright_session import new_agent_context

SOURCE = Path(__file__).resolve().parents[2] / "src" / "dlightrag"


class RecordingBrowser:
    """A browser that keeps the options each context was made with."""

    def __init__(self) -> None:
        self.contexts: list[dict[str, Any]] = []

    async def new_context(self, **options: Any) -> object:
        self.contexts.append(options)
        return object()


@pytest.mark.parametrize("downloads", [True, False])
async def test_a_context_is_made_with_no_proxy_of_its_own_and_no_service_workers(
    downloads: bool,
) -> None:
    browser = RecordingBrowser()

    await new_agent_context(cast(Browser, browser), accept_downloads=downloads)

    assert browser.contexts == [{"accept_downloads": downloads, "service_workers": "block"}]


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
    assert [name for name, _ in _uses("new_context")] == ["playwright_session.py"]
    assert _uses("expose_network") == []
