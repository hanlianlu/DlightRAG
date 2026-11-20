# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The edit pipeline stays linear in the lines an edit touches (ADR 0072, M2).

A whole-file ``difflib`` comparison is quadratic in the worst case; a batch of scattered
edits over a large file used to cost seconds. The spans are known, so the hunk renderer
renders from them and never compares the whole file. These tests count the operations that
would make it quadratic instead of timing them, so a loaded machine cannot fail them.
"""

from __future__ import annotations

import ast
from unittest.mock import MagicMock

import pytest

from dlightrag.engine.agent.tools import editing
from dlightrag.engine.agent.tools.editing import apply_edits, render_hunks, resolve_edits

#: A hunk around one rewritten line: its header, three lines of context either side, the
#: removed line and the added line.
_LINES_PER_HUNK = 1 + 3 + 2 + 3
_DIFF_HEADER_LINES = 2


class _Target:
    def __init__(
        self,
        *,
        new_text: str = "",
        old_text: str | None = None,
        symbol: str | None = None,
    ) -> None:
        self.new_text: str = new_text
        self.old_text: str | None = old_text
        self.symbol: str | None = symbol
        self.start_marker: str | None = None
        self.end_marker: str | None = None


def _forbid_the_whole_file_comparison(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    whole_file = MagicMock(side_effect=AssertionError("compared the whole file"))
    monkeypatch.setattr(editing.difflib, "unified_diff", whole_file)
    return whole_file


@pytest.mark.parametrize("line_count", [50_000, 100_000])
def test_scattered_edits_render_from_their_spans_and_the_diff_grows_with_the_edits_only(
    monkeypatch: pytest.MonkeyPatch, line_count: int
) -> None:
    _forbid_the_whole_file_comparison(monkeypatch)
    edit_count = 50
    step = line_count // edit_count
    original = "".join(
        f"line {index} with some realistic text for the tool\n" for index in range(line_count)
    )
    operations = [
        _Target(old_text=f"line {index} with some", new_text=f"line {index} rewritten with")
        for index in range(0, line_count, step)
    ]

    resolved = resolve_edits(original, operations, path="large.txt")
    updated = apply_edits(original, resolved)
    diff = render_hunks(original, updated, resolved, path="large.txt")

    assert len(resolved) == edit_count
    assert "diff omitted" not in diff
    # Twice the lines with the same edits renders the same diff size: it follows the spans.
    assert len(diff.split("\n")) <= _DIFF_HEADER_LINES + edit_count * _LINES_PER_HUNK


def test_a_structured_replacement_of_a_large_python_file_parses_it_once_and_renders_its_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_the_whole_file_comparison(monkeypatch)
    parse = MagicMock(side_effect=ast.parse)
    monkeypatch.setattr(editing.ast, "parse", parse)
    body = "".join(f"def function_{index}():\n    return {index}\n\n" for index in range(2_000))
    text = "TARGET = 1\n\n" + body + "".join(f"VALUE_{index} = {index}\n" for index in range(2_000))

    resolved = resolve_edits(
        text, [_Target(symbol="TARGET", new_text="TARGET = 2")], path="large.py"
    )
    updated = apply_edits(text, resolved)
    diff = render_hunks(text, updated, resolved, path="large.py")

    assert updated.startswith("TARGET = 2\n")
    assert "@@" in diff
    assert parse.call_count == 1
