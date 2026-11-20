# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The edit resolver: a target resolves to one region, and nothing else is touched.

These tests cross the resolver's own seam (ADR 0072): text in, spans or a refusal out.
The tool's filesystem/contract behaviour stays in test_path_tools.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from dlightrag.engine.agent.tools.editing import (
    EditResolutionError,
    ResolvedEdit,
    apply_edits,
    edit_summary_lines,
    render_hunks,
    resolve_edits,
)


class _Target:
    """One edit operation as the resolver reads it (the Tool's model supplies the same)."""

    def __init__(
        self,
        *,
        new_text: str = "",
        old_text: str | None = None,
        symbol: str | None = None,
        start_marker: str | None = None,
        end_marker: str | None = None,
    ) -> None:
        self.new_text = new_text
        self.old_text = old_text
        self.symbol = symbol
        self.start_marker = start_marker
        self.end_marker = end_marker


def _resolve(
    text: str,
    *operations: _Target,
    path: str = "note.py",
) -> tuple[ResolvedEdit, ...]:
    return resolve_edits(text, list(operations), path=path)


def _apply(text: str, *operations: _Target, path: str = "note.py") -> str:
    return apply_edits(text, _resolve(text, *operations, path=path))


# --------------------------------------------------------------------------------------
# old_text
# --------------------------------------------------------------------------------------


def test_exact_text_resolves_to_its_one_occurrence_and_applies() -> None:
    text = "alpha beta\nlast line\n"
    resolved = _resolve(text, _Target(old_text="last line", new_text="final"))
    assert [(edit.kind, edit.first_line, edit.last_line) for edit in resolved] == [
        ("old_text", 2, 2)
    ]
    assert _apply(text, _Target(old_text="last line", new_text="final")) == "alpha beta\nfinal\n"


def test_exact_text_refusal_names_the_lines_each_match_sits_on() -> None:
    with pytest.raises(EditResolutionError) as caught:
        _resolve("alpha\nbeta\nalpha\n", _Target(old_text="alpha", new_text="A"))
    assert "matches 2 times (lines 1, 3)" in str(caught.value)
    assert "matches 0 times" in _resolution_error(
        "alpha\n", _Target(old_text="gamma", new_text="G")
    )


def test_exact_text_that_only_rewrites_itself_is_refused() -> None:
    with pytest.raises(EditResolutionError) as caught:
        _resolve("alpha\n", _Target(old_text="alpha", new_text="alpha"))
    assert "old_text and new_text are identical" in str(caught.value)


# --------------------------------------------------------------------------------------
# symbol
# --------------------------------------------------------------------------------------


def test_symbol_replaces_a_module_level_assignment_by_name_without_its_text() -> None:
    text = 'HISTORY = me.build_history(\n    [{"year": 2021}],\n    source="old",\n)\nNEXT = 1\n'
    updated = _apply(
        text,
        _Target(symbol="HISTORY", new_text="HISTORY = me.build_history([], source='new')"),
    )
    assert updated == "HISTORY = me.build_history([], source='new')\nNEXT = 1\n"


def test_symbol_starts_at_the_decorator_and_keeps_the_body_layout() -> None:
    text = "@cache\ndef report():\n    return 1\n\nVALUE = 2\n"
    resolved = _resolve(
        text, _Target(symbol="report", new_text="@cache\ndef report():\n    return 3\n")
    )
    assert (resolved[0].first_line, resolved[0].last_line) == (1, 3)
    assert (
        _apply(text, _Target(symbol="report", new_text="@cache\ndef report():\n    return 3\n"))
        == "@cache\ndef report():\n    return 3\n\nVALUE = 2\n"
    )


def test_symbol_reaches_a_dotted_class_member() -> None:
    text = "class Case:\n    def validate(self, row):\n        return row * 2\n\n    scale = 1.0\n"
    updated = _apply(
        text,
        _Target(
            symbol="Case.validate",
            new_text="def validate(self, row):\n    return row * 3\n",
        ),
    )
    assert "return row * 3" in updated
    assert updated.startswith("class Case:\n    def validate(self, row):\n        return row * 3")


def test_symbol_indents_a_member_replacement_written_at_column_zero() -> None:
    text = "class Case:\n    scale = 1.0\n"
    updated = _apply(text, _Target(symbol="Case.scale", new_text="scale = 2.5"))
    assert updated == "class Case:\n    scale = 2.5\n"


def test_symbol_keeps_an_already_indented_member_replacement() -> None:
    text = "class Case:\n    scale = 1.0\n"
    updated = _apply(text, _Target(symbol="Case.scale", new_text="    scale = 2.5"))
    assert updated == "class Case:\n    scale = 2.5\n"


@pytest.mark.parametrize(
    ("text", "symbol", "new_text", "reasons"),
    [
        (
            "class Case:\n    scale = 1.0\n",
            "Case.scale",
            "  scale = 2.5",
            ("must start at column 4",),
        ),
        ("VALUE = 1\nVALUE = 2\n", "VALUE", "VALUE = 3", ("defined 2 times", "lines 1, 2")),
        ("VALUE = 1\n", "VALUE.member", "x = 1", ("which is not a class",)),
        ("A = 1; B = 2\n", "A", "replaced = 0", ("use old_text",)),
        ("A = 1; B = 2\n", "B", "replaced = 0", ("use old_text",)),
        ("class C: x = 1\n", "C.x", "replaced = 0", ("use old_text",)),
    ],
    ids=[
        "wrong-indentation",
        "duplicate-names",
        "walk-through-a-non-class",
        "shares-its-line-first",
        "shares-its-line-second",
        "shares-its-line-member",
    ],
)
def test_symbol_refuses_an_edit_it_cannot_address_and_says_why(
    text: str, symbol: str, new_text: str, reasons: tuple[str, ...]
) -> None:
    with pytest.raises(EditResolutionError) as caught:
        _resolve(text, _Target(symbol=symbol, new_text=new_text))
    for reason in reasons:
        assert reason in str(caught.value)


def test_symbol_reads_an_annotated_assignment() -> None:
    text = "SCALE: float = 1.0\nOTHER = 2\n"
    assert _apply(text, _Target(symbol="SCALE", new_text="SCALE: float = 2.0")) == (
        "SCALE: float = 2.0\nOTHER = 2\n"
    )


def test_symbol_refusal_suggests_the_nearest_definition() -> None:
    text = "HISTORY_SOURCE = 'x'\n"
    with pytest.raises(EditResolutionError) as caught:
        _resolve(text, _Target(symbol="HISTORY", new_text="HISTORY = []"))
    assert "closest: HISTORY_SOURCE" in str(caught.value)


def test_symbol_names_the_parse_failure_instead_of_guessing() -> None:
    with pytest.raises(EditResolutionError) as caught:
        _resolve("def broken(:\n", _Target(symbol="broken", new_text="def broken():\n    pass\n"))
    assert "does not parse as Python" in str(caught.value)
    assert "old_text" in str(caught.value)


def test_symbol_is_refused_for_a_non_python_file() -> None:
    with pytest.raises(EditResolutionError) as caught:
        _resolve("# notes\n", _Target(symbol="NOTES", new_text="# other\n"), path="notes.md")
    assert "not a .py file" in str(caught.value)


def test_symbol_addresses_a_name_bound_by_a_multi_target_assignment() -> None:
    updated = _apply("A = B = 1\n", _Target(symbol="A", new_text="A = 10\nB = 1"))
    assert updated == "A = 10\nB = 1\n"


def test_symbol_addresses_a_name_the_starter_unpacks() -> None:
    starter = _starter_text()
    resolved = _resolve(
        starter,
        _Target(
            symbol="REPORTING_CURRENCY",
            new_text='REPORTING_CURRENCY, UNITS = "SEK", "millions"',
        ),
    )
    assert resolved[0].kind == "symbol"
    assert 'REPORTING_CURRENCY, UNITS = "SEK"' in _apply(
        starter,
        _Target(
            symbol="REPORTING_CURRENCY",
            new_text='REPORTING_CURRENCY, UNITS = "SEK", "millions"',
        ),
    )


def test_symbol_composes_with_one_batch_atomically() -> None:
    text = "FIRST = 1\nSECOND = 2\n"
    updated = _apply(
        text,
        _Target(symbol="FIRST", new_text="FIRST = 10"),
        _Target(symbol="SECOND", new_text="SECOND = 20"),
    )
    assert updated == "FIRST = 10\nSECOND = 20\n"


# --------------------------------------------------------------------------------------
# markers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "start", "end", "new_text", "expected"),
    [
        (
            "# A\nbody one\nbody two\n# B\ntail\n",
            "# A",
            "# B",
            "replacement",
            "# A\nreplacement\n# B\ntail\n",
        ),
        ("# A\nbody\n# B\n", "# A", "# B", "", "# A\n# B\n"),
        ("# A\n# B\ntail\n", "# A", "# B", "inserted", "# A\ninserted\n# B\ntail\n"),
        (
            "begin\n# section\ntitle\nbody\nend\n",
            "begin\n# section",
            "end",
            "new body",
            "begin\n# section\nnew body\nend\n",
        ),
    ],
    ids=[
        "replace-keeps-both-marker-lines",
        "empty-deletes",
        "adjacent-inserts",
        "multi-line-marker",
    ],
)
def test_markers_replace_the_lines_between_them_and_keep_both_marker_lines(
    text: str, start: str, end: str, new_text: str, expected: str
) -> None:
    updated = _apply(text, _Target(start_marker=start, end_marker=end, new_text=new_text))
    assert updated == expected


def test_markers_replace_the_whole_starter_section_shape() -> None:
    text = (
        "# ===== SECTION 1 — FILL: identity =====\nIDENTITY = {}\n"
        "# ===== SECTION 2 — FILL: history =====\nHISTORY = []\n"
        "# ===== SECTION 3 — FILL: assumptions =====\nX = 1\n"
    )
    updated = _apply(
        text,
        _Target(
            start_marker="# ===== SECTION 2 — FILL: history =====",
            end_marker="# ===== SECTION 3 — FILL: assumptions =====",
            new_text="HISTORY = [1, 2]",
        ),
    )
    assert updated == (
        "# ===== SECTION 1 — FILL: identity =====\nIDENTITY = {}\n"
        "# ===== SECTION 2 — FILL: history =====\nHISTORY = [1, 2]\n"
        "# ===== SECTION 3 — FILL: assumptions =====\nX = 1\n"
    )


def test_marker_refusals_name_the_marker_and_its_lines() -> None:
    duplicates = _resolution_error(
        "# A\n# A\nbody\n# B\n",
        _Target(start_marker="# A", end_marker="# B", new_text="x"),
    )
    assert "start_marker occurs 2 times (lines 1, 2)" in duplicates
    missing = _resolution_error(
        "# A\nbody\n# B\n",
        _Target(start_marker="# C", end_marker="# B", new_text="x"),
    )
    assert "start_marker is not a whole line" in missing
    not_a_line = _resolution_error(
        "# A extra\nbody\n# B\n",
        _Target(start_marker="# A", end_marker="# B", new_text="x"),
    )
    assert "not a whole line" in not_a_line
    reversed_markers = _resolution_error(
        "# A\nbody\n# B\n",
        _Target(start_marker="# B", end_marker="# A", new_text="x"),
    )
    assert "before or inside start_marker" in reversed_markers
    identical = _resolution_error(
        "# A\nbody\n# B\n",
        _Target(start_marker="# A", end_marker="# A", new_text="x"),
    )
    assert "identical" in identical
    blank = _resolution_error(
        "\n# B\n",
        _Target(start_marker="  ", end_marker="# B", new_text="x"),
    )
    assert "visible text" in blank


def test_marker_replacement_that_changes_nothing_is_refused() -> None:
    with pytest.raises(EditResolutionError) as caught:
        _resolve(
            "# A\nbody\n# B\n",
            _Target(start_marker="# A", end_marker="# B", new_text="body"),
        )
    assert "equals the region it replaces" in str(caught.value)


# --------------------------------------------------------------------------------------
# target form and batch validation
# --------------------------------------------------------------------------------------


def test_one_target_form_per_edit() -> None:
    no_target = _resolution_error("x\n", _Target(new_text="y"))
    assert "name exactly one target" in no_target
    two_targets = _resolution_error("x\n", _Target(old_text="x", symbol="X", new_text="y"))
    assert "name exactly one target" in two_targets
    half_markers = _resolution_error("x\n", _Target(start_marker="# A", new_text="y"))
    assert "give both" in half_markers


def test_overlapping_edits_are_refused_before_anything_is_applied() -> None:
    with pytest.raises(EditResolutionError) as caught:
        _resolve(
            "alpha beta\n",
            _Target(old_text="alpha beta", new_text="first"),
            _Target(old_text="beta", new_text="second"),
        )
    assert "edits 1 and 2 overlap" in str(caught.value)


def test_batch_order_does_not_change_the_result() -> None:
    text = "FIRST = 1\nSECOND = 2\n"
    first_then_second = _apply(
        text,
        _Target(symbol="FIRST", new_text="FIRST = 10"),
        _Target(symbol="SECOND", new_text="SECOND = 20"),
    )
    second_then_first = _apply(
        text,
        _Target(symbol="SECOND", new_text="SECOND = 20"),
        _Target(symbol="FIRST", new_text="FIRST = 10"),
    )
    assert first_then_second == second_then_first == "FIRST = 10\nSECOND = 20\n"


# --------------------------------------------------------------------------------------
# result summary and hunks
# --------------------------------------------------------------------------------------


def test_summary_names_the_target_the_lines_and_the_resulting_count() -> None:
    text = "HISTORY = 1\nVALUE = 2\n"
    resolved = _resolve(
        text,
        _Target(symbol="HISTORY", new_text="HISTORY = 10"),
        _Target(old_text="VALUE = 2", new_text="VALUE = 20\nVALUE_2 = 21"),
    )
    assert edit_summary_lines(resolved) == [
        'edit 1: symbol "HISTORY", line 1 -> 1 line',
        'edit 2: old_text "VALUE = 2", line 2 -> 2 lines',
    ]


def test_summary_is_capped_and_says_how_many_remain() -> None:
    text = "".join(f"VALUE_{index} = {index}\n" for index in range(20))
    operations = [
        _Target(symbol=f"VALUE_{index}", new_text=f"VALUE_{index} = {index + 100}")
        for index in range(20)
    ]
    summary = edit_summary_lines(_resolve(text, *operations))
    assert len(summary) == 13
    assert summary[-1] == "... 8 more edits"


def test_hunk_diff_shows_only_the_changed_lines_and_their_context() -> None:
    text = "one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\n"
    updated = _apply(text, _Target(old_text="five", new_text="FIVE"))
    diff = render_hunks(
        text, updated, _resolve(text, _Target(old_text="five", new_text="FIVE")), path="in.txt"
    )
    assert diff.splitlines() == [
        "--- in.txt",
        "+++ in.txt",
        "@@ -2,7 +2,7 @@",
        " two",
        " three",
        " four",
        "-five",
        "+FIVE",
        " six",
        " seven",
        " eight",
    ]


def test_hunk_diff_renders_two_far_changes_as_two_hunks() -> None:
    text = "".join(f"line {index}\n" for index in range(20))
    operations = [
        _Target(old_text="line 1\n", new_text="LINE 1\n"),
        _Target(old_text="line 17\n", new_text="LINE 17\n"),
    ]
    resolved = _resolve(text, *operations)
    updated = apply_edits(text, resolved)
    diff = render_hunks(text, updated, resolved, path="in.txt", context=1)
    assert len([line for line in diff.splitlines() if line.startswith("@@")]) == 2
    assert "-line 1" in diff and "+LINE 1" in diff
    assert "-line 17" in diff and "+LINE 17" in diff


def test_hunk_diff_hands_a_large_structural_change_to_the_reader() -> None:
    text = "head\n" + "".join(f"old {index}\n" for index in range(500)) + "tail\n"
    resolved = _resolve(
        text,
        _Target(old_text=text[len("head\n") : -len("tail\n")], new_text="one line\n"),
    )
    updated = apply_edits(text, resolved)
    diff = render_hunks(text, updated, resolved, path="in.txt")
    assert diff.startswith("diff omitted: 500 lines removed and 1 line added")


def test_hunks_are_empty_when_nothing_changed() -> None:
    assert render_hunks("a\n", "a\n", (), path="in.txt") == ""


def test_the_real_starter_fills_by_section_headers_and_still_parses() -> None:
    """The starter's own `# ===== SECTION n — FILL …` lines are the intended target handles."""
    text = _starter_text()
    headers = [line for line in text.splitlines() if line.startswith("# ===== SECTION ")]
    operations = [
        _Target(
            start_marker=header,
            end_marker=headers[index + 1],
            new_text=f"_SECTION_{index + 1}_FILLED = {index + 1}",
        )
        for index, header in enumerate(headers[:-1])
        if "— FILL" in header
    ]
    resolved = resolve_edits(text, operations, path="model-starter.py")
    updated = apply_edits(text, resolved)
    assert len(resolved) == 9
    ast.parse(updated)
    assert all(header in updated for header in headers)


def test_symbol_handles_a_parenthesized_multi_line_decorator() -> None:
    text = "@(\n    deco\n)\ndef f():\n    return 1\n"
    updated = _apply(
        text,
        _Target(symbol="f", new_text="@(\n    deco\n)\ndef f():\n    return 2\n"),
    )
    assert updated == "@(\n    deco\n)\ndef f():\n    return 2\n"
    ast.parse(updated)


def test_overlap_refusal_does_not_depend_on_the_request_order() -> None:
    text = "# A\nfoo\n"
    insertion = _Target(start_marker="# A", end_marker="foo", new_text="x")
    replacement = _Target(old_text="foo", new_text="bar")
    first = _resolve(text, insertion, replacement)
    second = _resolve(text, replacement, insertion)
    assert [(edit.start, edit.end) for edit in first] == [(edit.start, edit.end) for edit in second]


def test_empty_old_text_is_refused_instead_of_looping() -> None:
    with pytest.raises(EditResolutionError) as caught:
        _resolve("abc\n", _Target(old_text="", new_text="X"))
    assert "old_text is empty" in str(caught.value)


def test_a_form_feed_indentation_is_recognised() -> None:
    text = "class C:\n\x0c    x = 1\n"
    updated = _apply(text, _Target(symbol="C.x", new_text="x = 2"))
    assert updated == "class C:\n\x0c    x = 2\n"
    ast.parse(updated)


@pytest.mark.parametrize(
    ("original", "operations", "path"),
    [
        ("TODO: fix this\nkeep\n", [_Target(old_text="TODO: ", new_text="")], "t.txt"),
        ("foo bar\nnext\n", [_Target(old_text="foo ", new_text="FOO\n")], "t.txt"),
        (
            "A = 1; B = 2\nnext\n",
            [_Target(old_text="1", new_text="10"), _Target(old_text="2", new_text="20")],
            "t.txt",
        ),
        (
            "# A\nfirst\nsecond\n# B\ntail\n",
            [_Target(start_marker="# A", end_marker="# B", new_text="only")],
            "t.txt",
        ),
        (
            "class Case:\n    scale = 1.0\n\nVALUE = 2\n",
            [_Target(symbol="Case.scale", new_text="scale = 2.0")],
            "t.py",
        ),
        ("alpha beta\n", [_Target(old_text="alpha beta", new_text="")], "t.txt"),
        ("alpha\n", [_Target(old_text="alpha", new_text="one\ntwo\n")], "t.txt"),
        ("a\nb\nc\n", [_Target(old_text="b\n", new_text="X")], "t.txt"),
        ("a\nb\nc\n", [_Target(old_text="b\n", new_text="X\n")], "t.txt"),
        ("a\nb", [_Target(old_text="b", new_text="B")], "t.txt"),
        ("a\nb\n", [_Target(old_text="b\n", new_text="")], "t.txt"),
        ("a\n", [_Target(old_text="a\n", new_text="A")], "t.txt"),
        ("class C:\n\x0c    x = 1\n", [_Target(old_text="x = 1", new_text="x = 2")], "t.py"),
        (
            "A = 1; B = 2\nnext\n",
            [_Target(old_text="A = ", new_text=""), _Target(old_text="B = ", new_text="")],
            "t.txt",
        ),
        ("ab\ncd\n", [_Target(old_text="b\n", new_text="")], "t.txt"),
        (
            "a\nb\nc\nd\n",
            [_Target(old_text="a\nb\n", new_text="X"), _Target(old_text="c\n", new_text="")],
            "t.txt",
        ),
        ("one\ntwo", [_Target(old_text="two", new_text="THREE\n")], "t.txt"),
    ],
)
def test_every_rendered_hunk_applies_to_the_original(
    original: str,
    operations: list[_Target],
    path: str,
) -> None:
    resolved = _resolve(original, *operations, path=path)
    updated = apply_edits(original, resolved)
    diff = render_hunks(original, updated, resolved, path=path)
    assert _apply_diff(original, diff) == updated


def test_hunks_apply_over_a_deterministic_sample_of_batches() -> None:
    import random

    generator = random.Random(20261025)
    for _ in range(120):
        line_count = generator.randint(2, 9)
        lines = [f"value {generator.randint(0, 9)} line {index}" for index in range(line_count)]
        original = "\n".join(lines) + "\n"
        first = generator.randrange(1, line_count + 1)
        second = generator.randrange(first, line_count + 1)
        target = f"line {first - 1}" if first == second else f"line {second - 1}"
        if original.count(target) != 1:
            continue
        resolved = _resolve(original, _Target(old_text=target, new_text="REPLACED"))
        updated = apply_edits(original, resolved)
        diff = render_hunks(original, updated, resolved, path="sample.txt")
        assert _apply_diff(original, diff) == updated


def test_two_insertions_at_the_same_place_are_refused() -> None:
    text = "# A\n# B\n"
    with pytest.raises(EditResolutionError) as caught:
        _resolve(
            text,
            _Target(start_marker="# A", end_marker="# B", new_text="first"),
            _Target(start_marker="# A", end_marker="# B", new_text="second"),
        )
    assert "insert at the same place" in str(caught.value)


def test_a_trailing_comment_does_not_make_a_definition_unaddressable() -> None:
    text = "VALUE = 1  # a note\nNEXT = 2\n"
    updated = _apply(text, _Target(symbol="VALUE", new_text="VALUE = 10"))
    assert updated.startswith("VALUE = 10\n")


def test_symbol_addresses_a_starred_unpacking_target() -> None:
    text = "A, *REST = [1, 2, 3]\n"
    resolved = _resolve(text, _Target(symbol="REST", new_text="A, *REST = [4, 5, 6]"))
    assert resolved[0].kind == "symbol"


def test_nearest_name_hint_does_not_offer_an_unaddressable_name() -> None:
    duplicate = _resolution_error(
        "VALUE = 1\nVALUE = 2\n", _Target(symbol="VALU", new_text="x = 1")
    )
    assert "closest" not in duplicate
    shared = _resolution_error("A = 1; B = 2\n", _Target(symbol="AA", new_text="x = 1"))
    assert "closest" not in shared


def test_a_decorator_comment_is_refused_rather_than_guessed() -> None:
    text = "@(\n    # why\n    deco\n)\ndef f():\n    return 1\n"
    with pytest.raises(EditResolutionError) as caught:
        _resolve(text, _Target(symbol="f", new_text="def f():\n    return 2\n"))
    assert "cannot locate the @ line" in str(caught.value)


def test_a_missing_final_newline_is_marked_in_the_hunk() -> None:
    text = "one\ntwo"
    resolved = _resolve(text, _Target(old_text="two", new_text="THREE"))
    updated = apply_edits(text, resolved)
    diff = render_hunks(text, updated, resolved, path="t.txt")
    assert "\\ No newline at end of file" in diff
    assert _apply_diff(text, diff) == "one\nTHREE"


def test_rendered_hunks_apply_with_git(tmp_path: Path) -> None:
    import subprocess

    cases = [
        ("a\nb\nc\n", [_Target(old_text="b\n", new_text="X")], "case0.txt"),
        ("a\nb", [_Target(old_text="b", new_text="B")], "case1.txt"),
        ("one\n", [_Target(old_text="one\n", new_text="1\n2\n")], "case2.txt"),
        ("ab\ncd\n", [_Target(old_text="b\n", new_text="")], "case3.txt"),
        (
            "class C:\n    scale = 1.0\n",
            [_Target(symbol="C.scale", new_text="scale = 2.0")],
            "case4.py",
        ),
    ]
    for index, (original, operations, path) in enumerate(cases):
        resolved = _resolve(original, *operations, path=path)
        updated = apply_edits(original, resolved)
        diff = render_hunks(original, updated, resolved, path=path)
        repo = tmp_path / f"repo{index}"
        repo.mkdir()
        (repo / path).write_text(original, encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(
            ["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "base"],
            cwd=repo,
            check=True,
        )
        patch = repo / "change.diff"
        patch.write_text(diff + "\n", encoding="utf-8")
        checked = subprocess.run(
            ["git", "apply", "--check", str(patch)], cwd=repo, capture_output=True, text=True
        )
        assert checked.returncode == 0, (diff, checked.stderr)
        subprocess.run(["git", "apply", str(patch)], cwd=repo, check=True)
        assert (repo / path).read_text(encoding="utf-8") == updated


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _starter_text() -> str:
    """A long sectioned script: twelve `# ===== SECTION n` headers (nine FILL, three ENGINE) over big bodies."""
    kinds = ("FILL",) * 3 + ("ENGINE", "FILL", "ENGINE") + ("FILL",) * 5 + ("ENGINE",)
    lines = ['REPORTING_CURRENCY, UNITS = "USD", "thousands"']
    for index, kind in enumerate(kinds, start=1):
        lines.append(f"# ===== SECTION {index} — {kind}: part {index} =====")
        lines.extend(f"VALUE_{index}_{row} = {row}" for row in range(100))
    return "\n".join(lines) + "\n"


def _apply_diff(original: str, diff: str) -> str:
    """Apply our rendered hunks back to the original; a corrupt or miscounted hunk raises."""
    import re

    lines = original.split("\n")
    result: list[str] = []
    cursor = 0
    header = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@$")
    counts = {"old": -1, "new": -1, "seen_old": 0, "seen_new": 0}
    unterminated = False
    last_prefix = ""

    def flush_hunk() -> None:
        if counts["old"] < 0:
            return
        assert counts["seen_old"] == counts["old"], f"header old count disagrees with body: {diff}"
        assert counts["seen_new"] == counts["new"], f"header new count disagrees with body: {diff}"

    for line in diff.split("\n"):
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        if line == "\\ No newline at end of file":
            unterminated = True
            continue
        if line.startswith("@@"):
            flush_hunk()
            match = header.match(line)
            assert match, line
            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) is not None else 1
            counts["old"], counts["new"] = (
                old_count,
                (int(match.group(4)) if match.group(4) is not None else 1),
            )
            counts["seen_old"] = counts["seen_new"] = 0
            start_index = old_start - 1 if old_count else old_start
            assert start_index >= cursor, (line, cursor)
            result.extend(lines[cursor:start_index])
            cursor = start_index
            continue
        assert line, "a unified diff line carries a prefix"
        prefix, body = line[0], line[1:]
        if prefix == " ":
            assert lines[cursor] == body, (lines[cursor], body)
            result.append(body)
            cursor += 1
            counts["seen_old"] += 1
            counts["seen_new"] += 1
        elif prefix == "-":
            assert lines[cursor] == body, (lines[cursor], body)
            cursor += 1
            counts["seen_old"] += 1
        elif prefix == "+":
            result.append(body)
            counts["seen_new"] += 1
        else:  # pragma: no cover - a renderer bug
            raise AssertionError(line)
        unterminated = False
        last_prefix = prefix
    flush_hunk()
    result.extend(lines[cursor:])
    if unterminated and result and result[-1] == "":
        result.pop()
    if not unterminated and last_prefix == "+" and result and result[-1] != "":
        result.append("")
    return "\n".join(result)


def _resolution_error(text: str, operation: _Target, *, path: str = "note.py") -> str:
    with pytest.raises(EditResolutionError) as caught:
        resolve_edits(text, [operation], path=path)
    return str(caught.value)
