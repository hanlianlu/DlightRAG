# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The one door every Tool Subject passes through, and where its wire spelling may appear."""

from pathlib import Path

from dlightrag.engine.agent.tool_content import ToolTextPart
from dlightrag.engine.agent.tools import ToolResult
from dlightrag.engine.agent.tools.contracts import TOOL_SUBJECT_MAX_CHARS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WIRE_SPELLING = "object_label"
# The Engine sink that projects a reported subject onto the run event, plus the browser edge
# that keeps its own display cap. No other Engine module may name the wire field.
_WIRE_SPELLING_OWNERS = {
    Path("src/dlightrag/engine/answer/research/runtime.py"),
    Path("src/dlightrag/adapters/http/browser/answer_events.py"),
}


def test_the_published_subject_bound_is_sixty_four_characters() -> None:
    assert TOOL_SUBJECT_MAX_CHARS == 64


def test_a_reported_subject_becomes_one_bounded_line() -> None:
    collapsed = ToolResult.text("", subject="  quarterly\n\trevenue  ")

    assert collapsed.subject == "quarterly revenue"
    # Control characters are invisible in a one-line row, and U+0000 cannot be stored in
    # the durable JSON payload at all.
    assert ToolResult.text("", subject="a\x00b\x07c").subject == "abc"
    assert ToolResult.text("", subject="x" * 64).subject == "x" * 64
    assert ToolResult.text("", subject="x" * 65).subject == "x" * 63 + "…"


def test_the_bound_holds_on_every_construction_path() -> None:
    constructed = ToolResult(parts=(ToolTextPart(""),), subject="y" * 200)

    assert constructed.subject == "y" * 63 + "…"


def test_a_subject_never_reaches_the_model_visible_text() -> None:
    result = ToolResult.text("body", subject="quarterly revenue")

    assert result.text_content == "body"
    assert ToolResult.text("body").subject is None


def test_the_wire_spelling_stays_at_the_engine_sink_and_the_browser_edge() -> None:
    owners = {
        path.relative_to(_REPO_ROOT)
        for path in (_REPO_ROOT / "src").rglob("*.py")
        if _WIRE_SPELLING in path.read_text(encoding="utf-8")
    }

    assert owners == _WIRE_SPELLING_OWNERS
