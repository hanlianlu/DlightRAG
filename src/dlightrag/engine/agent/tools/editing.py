# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Resolve an edit call's targets to regions, apply them, and render the hunks.

The edit Tool is a span pipeline. Every target names a region — an exact string, a
named definition, or the lines between two whole-line markers — and the resolver turns
the whole batch into non-overlapping spans against the original text before the caller
writes anything. Nothing here touches the filesystem or the workspace text layer: the
caller hands in the presented text and writes the result through that layer.
"""

from __future__ import annotations

import ast
import bisect
import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

type EditKind = Literal["old_text", "symbol", "between"]

#: The most diff lines a result carries before the caller is told to re-read the region.
MAX_DIFF_LINES = 400

#: The most per-edit summary lines a result carries.
MAX_SUMMARY_EDITS = 12

#: The most match positions a non-unique refusal lists.
_MAX_POSITIONS = 5

#: The most characters of a target a result summary quotes.
_MAX_LABEL_CHARS = 48


class EditTarget(Protocol):
    """What one requested edit must expose to the resolver (the Tool's argument model).

    Each target names exactly one region: ``old_text`` (exact text), ``symbol`` (a named
    definition), or ``start_marker`` + ``end_marker`` (the lines between two whole-line
    markers).
    """

    new_text: str
    old_text: str | None
    symbol: str | None
    start_marker: str | None
    end_marker: str | None


class EditResolutionError(ValueError):
    """One target did not resolve to exactly one region, so nothing is written."""


@dataclass(frozen=True, slots=True)
class ResolvedEdit:
    """One edit's region in the original text and the text that replaces it."""

    index: int
    kind: EditKind
    label: str
    start: int
    end: int
    replacement: str
    first_line: int
    last_line: int

    @property
    def added_lines(self) -> int:
        return _line_count(self.replacement)


def resolve_edits(
    text: str,
    operations: Sequence[EditTarget],
    *,
    path: str,
) -> tuple[ResolvedEdit, ...]:
    """Resolve every operation against ``text``, or raise before anything is written.

    Every target must name exactly one region. The resolved spans may not overlap, and a
    replacement may not equal the region it replaces; both are refused here rather than
    half-applied later.
    """
    source = _Source(text)
    resolved = [
        _resolve_one(operation, source, path=path, index=index)
        for index, operation in enumerate(operations, start=1)
    ]
    # An empty span at an offset comes before a non-empty span at the same offset: the two
    # are disjoint, and the order must not depend on how the caller listed them.
    ordered = sorted(resolved, key=lambda edit: (edit.start, edit.end))
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if previous.start == previous.end == current.start == current.end:
            raise EditResolutionError(
                f"edits {previous.index} and {current.index} insert at the same place; merge "
                "them into one target"
            )
        if previous.end > current.start:
            raise EditResolutionError(
                f"edits {previous.index} and {current.index} overlap "
                f"({source.span(previous.start, previous.end)} and "
                f"{source.span(current.start, current.end)}); use old_text, "
                "or merge them into one target"
            )
    return tuple(ordered)


def apply_edits(text: str, edits: Sequence[ResolvedEdit]) -> str:
    """Apply already-resolved (non-overlapping) edits to the original text."""
    parts: list[str] = []
    cursor = 0
    for edit in edits:
        parts.append(text[cursor : edit.start])
        parts.append(edit.replacement)
        cursor = edit.end
    parts.append(text[cursor:])
    return "".join(parts)


def edit_summary_lines(edits: Sequence[ResolvedEdit]) -> list[str]:
    """One line per edit naming its target and the lines it replaced, in request order."""
    lines: list[str] = []
    by_index = sorted(edits, key=lambda edit: edit.index)
    for edit in by_index[:MAX_SUMMARY_EDITS]:
        span = (
            f"line {edit.first_line}"
            if edit.first_line == edit.last_line
            else f"lines {edit.first_line}-{edit.last_line}"
        )
        noun = "line" if edit.added_lines == 1 else "lines"
        lines.append(f"edit {edit.index}: {edit.label}, {span} -> {edit.added_lines} {noun}")
    if len(by_index) > MAX_SUMMARY_EDITS:
        lines.append(f"... {len(by_index) - MAX_SUMMARY_EDITS} more edits")
    return lines


def render_hunks(
    original: str,
    updated: str,
    edits: Sequence[ResolvedEdit],
    *,
    path: str,
    context: int = 3,
    max_changed_lines: int = MAX_DIFF_LINES,
) -> str:
    """Render the changed regions as unified-diff hunks that replay to ``updated``.

    The fast path renders hunks straight from the resolved spans (linear in the lines a span
    touches). The result is then replayed against the original text; when a span shape cannot
    be rendered faithfully that way (edits that merge lines across each other, for example),
    the standard whole-file comparison produces the diff instead, so the result is still a
    patch a reader or a patch tool can apply. A file too large for that fallback reports the
    omission instead of a diff no one can use.
    """
    changes = _changes(original, updated, edits)
    if not changes:
        return ""
    removed = sum(change.old_last - change.old_first for change in changes)
    added = sum(change.new_last - change.new_first for change in changes)
    if removed + added > max_changed_lines:
        return _diff_omitted(removed, added, max_changed_lines)
    fast = _custom_hunks(original, updated, changes, path=path, context=context)
    if _replays(fast, original, updated) is not None:
        return "\n".join(fast)
    if len(original.split("\n")) + len(updated.split("\n")) <= _FALLBACK_LINE_LIMIT:
        fallback = _diff_hunks(original, updated, path=path, context=context)
        if _replays(fallback, original, updated) is not None:
            fallback_removed = sum(
                1 for line in fallback if line.startswith("-") and not line.startswith("---")
            )
            fallback_added = sum(
                1 for line in fallback if line.startswith("+") and not line.startswith("+++")
            )
            if fallback_removed + fallback_added <= max_changed_lines:
                return "\n".join(fallback)
    return (
        "diff omitted: the changed lines could not be rendered as a faithful patch; read the "
        "changed lines to inspect them"
    )


#: Above this many lines the whole-file fallback comparison is not worth its cost.
_FALLBACK_LINE_LIMIT = 200_000

_NO_NEWLINE = "\\ No newline at end of file"

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@$")


def _hunk_counts(line: str) -> tuple[int, int, int, int] | None:
    """(old_start, old_count, new_start, new_count); difflib omits a one-line count."""
    match = _HUNK_HEADER.match(line)
    if match is None:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)) if match.group(2) is not None else 1,
        int(match.group(3)),
        int(match.group(4)) if match.group(4) is not None else 1,
    )


def _diff_omitted(removed: int, added: int, max_changed_lines: int) -> str:
    return (
        f"diff omitted: {removed} line{'s' if removed != 1 else ''} removed and {added} "
        f"line{'s' if added != 1 else ''} added, over the {max_changed_lines}-line result "
        "budget; read the changed lines to inspect them"
    )


def _custom_hunks(
    original: str,
    updated: str,
    changes: Sequence[_Change],
    *,
    path: str,
    context: int,
) -> list[str]:
    """Hunks rendered straight from the resolved line ranges (the fast path)."""
    old_lines = _real_lines(original)
    new_lines = _real_lines(updated)
    old_count = len(old_lines)
    new_count = len(new_lines)
    old_final_newline = not original or original.endswith("\n")
    new_final_newline = not updated or updated.endswith("\n")
    groups: list[list[_Change]] = []
    for change in changes:
        if groups and change.old_first <= groups[-1][-1].old_last + 2 * context:
            groups[-1].append(change)
        else:
            groups.append([change])
    rendered = [f"--- {path}", f"+++ {path}"]
    for group in groups:
        old_from = max(0, min(group[0].old_first, old_count) - context)
        old_to = max(old_from, min(old_count, group[-1].old_last + context))
        new_from = max(0, min(group[0].new_first, new_count) - context)
        new_to = max(new_from, min(new_count, group[-1].new_last + context))
        rendered.append(_hunk_header(old_from, old_to - old_from, new_from, new_to - new_from))
        cursor = old_from
        for change in group:
            while cursor < min(change.old_first, old_count):
                rendered.append(f" {old_lines[cursor]}")
                if cursor == old_count - 1 and not old_final_newline:
                    rendered.append(_NO_NEWLINE)
                cursor += 1
            for index in range(change.old_first, min(change.old_last, old_count)):
                rendered.append(f"-{old_lines[index]}")
                if index == old_count - 1 and not old_final_newline:
                    rendered.append(_NO_NEWLINE)
            for index in range(change.new_first, min(change.new_last, new_count)):
                rendered.append(f"+{new_lines[index]}")
                if index == new_count - 1 and not new_final_newline:
                    rendered.append(_NO_NEWLINE)
            cursor = max(cursor, change.old_last)
        while cursor < old_to:
            rendered.append(f" {old_lines[cursor]}")
            if cursor == old_count - 1 and not old_final_newline:
                rendered.append(_NO_NEWLINE)
            cursor += 1
    return rendered


def _real_lines(text: str) -> list[str]:
    """The file's lines without the phantom empty line a trailing newline would add."""
    if not text:
        return []
    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()
    return lines


def _diff_hunks(original: str, updated: str, *, path: str, context: int) -> list[str]:
    """The whole-file comparison used when the span rendering cannot be replayed."""
    old_lines = _real_lines(original)
    new_lines = _real_lines(updated)
    raw = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=path,
            tofile=path,
            lineterm="",
            n=context,
        )
    )
    return _mark_final_newlines(raw, original, updated)


def _mark_final_newlines(lines: list[str], original: str, updated: str) -> list[str]:
    """Append the standard no-newline marker where a file ends without a terminator."""
    old_missing = bool(original) and not original.endswith("\n")
    new_missing = bool(updated) and not updated.endswith("\n")
    marked: list[str] = []
    old_index = new_index = 0
    old_count = new_count = 0
    in_hunk = False
    for line in lines:
        if line.startswith("@@") and not line.startswith("@@@"):
            counts = _hunk_counts(line)
            if counts is not None:
                old_start, old_count, new_start, new_count = counts
                old_index = old_start - 1 if old_count else old_start
                new_index = new_start - 1 if new_count else new_start
                in_hunk = True
            marked.append(line)
            continue
        if not in_hunk or not line or line.startswith(("--- ", "+++ ")):
            marked.append(line)
            continue
        prefix = line[0]
        if prefix == " ":
            marked.append(line)
            if old_missing and old_index == old_count - 1:
                marked.append(_NO_NEWLINE)
            elif new_missing and new_index == new_count - 1:
                marked.append(_NO_NEWLINE)
            old_index += 1
            new_index += 1
        elif prefix == "-":
            marked.append(line)
            if old_missing and old_index == old_count - 1:
                marked.append(_NO_NEWLINE)
            old_index += 1
        elif prefix == "+":
            marked.append(line)
            if new_missing and new_index == new_count - 1:
                marked.append(_NO_NEWLINE)
            new_index += 1
        else:  # pragma: no cover - difflib lines always carry a prefix here
            marked.append(line)
    return marked


def _replays(lines: Sequence[str], original: str, expected: str) -> str | None:
    """Apply rendered hunks back to ``original``; None unless the result is ``expected``.

    A unified diff cannot always say whether a final newline is present unless it carries
    the no-newline marker; the comparison accepts the one-line-ending difference that the
    marker legibility leaves open, and nothing else.
    """
    if not lines:
        return None
    old_lines = _real_lines(original)
    result: list[str] = []
    cursor = 0
    old_count = new_count = 0
    seen_old = seen_new = 0
    in_hunk = False

    def finish_hunk() -> bool:
        return (not in_hunk) or (seen_old == old_count and seen_new == new_count)

    for line in lines:
        if line.startswith(("--- ", "+++ ")) or line == _NO_NEWLINE:
            continue
        if line.startswith("@@"):
            if not finish_hunk():
                return None
            counts = _hunk_counts(line)
            if counts is None:
                return None
            old_start, old_count, _new_start, new_count = counts
            start_index = old_start - 1 if old_count else old_start
            if start_index < cursor:
                return None
            result.extend(old_lines[cursor:start_index])
            cursor = start_index
            seen_old = seen_new = 0
            in_hunk = True
            continue
        if not in_hunk or not line:
            return None
        prefix, body = line[0], line[1:]
        if prefix == " ":
            if cursor >= len(old_lines) or old_lines[cursor] != body:
                return None
            result.append(body)
            cursor += 1
            seen_old += 1
            seen_new += 1
        elif prefix == "-":
            if cursor >= len(old_lines) or old_lines[cursor] != body:
                return None
            cursor += 1
            seen_old += 1
        elif prefix == "+":
            result.append(body)
            seen_new += 1
        else:  # pragma: no cover - rendered lines always carry a prefix
            return None
    if not finish_hunk():
        return None
    result.extend(old_lines[cursor:])
    candidate = "\n".join(result)
    for text in (candidate, f"{candidate}\n" if candidate else "\n"):
        if text == expected:
            return expected
    return None


# --------------------------------------------------------------------------------------
# Target resolution
# --------------------------------------------------------------------------------------


def _resolve_one(
    operation: EditTarget,
    source: _Source,
    *,
    path: str,
    index: int,
) -> ResolvedEdit:
    target = _target_kind(operation, index=index)
    if target == "old_text":
        return _resolve_old_text(operation, source, index=index)
    if target == "symbol":
        return _resolve_symbol(operation, source, path=path, index=index)
    return _resolve_between(operation, source, index=index)


def _target_kind(operation: EditTarget, *, index: int) -> EditKind:
    """Name the one target form the operation carries, or refuse the call."""
    has_markers = operation.start_marker is not None or operation.end_marker is not None
    forms = (operation.old_text is not None, operation.symbol is not None, has_markers)
    if sum(forms) != 1:
        raise EditResolutionError(
            f"edit {index}: name exactly one target — old_text, symbol, or start_marker with "
            "end_marker"
        )
    if has_markers and (operation.start_marker is None or operation.end_marker is None):
        raise EditResolutionError(
            f"edit {index}: start_marker and end_marker name a region together; give both"
        )
    if operation.old_text is not None:
        return "old_text"
    if operation.symbol is not None:
        return "symbol"
    return "between"


def _resolve_old_text(operation: EditTarget, source: _Source, *, index: int) -> ResolvedEdit:
    old_text = operation.old_text
    if not old_text:
        raise EditResolutionError(
            f"edit {index}: old_text is empty; name the text to replace, or a symbol or the "
            "lines between two markers"
        )
    if old_text == operation.new_text:
        raise EditResolutionError(f"edit {index} rejected: old_text and new_text are identical")
    count = source.text.count(old_text)
    if count != 1:
        positions = _positions(source.text, old_text)
        lines = ", ".join(str(source.line_number(position)) for position in positions)
        if count == 0:
            raise EditResolutionError(
                f"edit {index}: old_text matches 0 times; it must match exactly once — read the "
                "file and use its current text, or name a symbol or the lines between two markers"
            )
        listed = f" (lines {lines})" if positions else ""
        raise EditResolutionError(
            f"edit {index}: old_text matches {count} times{listed}; each match must be unique — "
            "add surrounding context, or name a symbol or the lines between two markers"
        )
    start = source.text.index(old_text)
    return _resolved(
        source,
        index=index,
        kind="old_text",
        label=_quoted_label("old_text", old_text),
        start=start,
        end=start + len(old_text),
        replacement=operation.new_text,
    )


def _resolve_symbol(
    operation: EditTarget,
    source: _Source,
    *,
    path: str,
    index: int,
) -> ResolvedEdit:
    symbol = operation.symbol
    if not symbol:
        raise EditResolutionError(
            f"edit {index}: symbol is empty; name a definition, or use old_text or the lines "
            "between two markers"
        )
    if not path.lower().endswith(".py"):
        raise EditResolutionError(
            f"edit {index}: symbol targets a Python definition, but {path} is not a .py file — "
            "use old_text, or start_marker with end_marker"
        )
    parts = symbol.split(".")
    if not all(part.isidentifier() for part in parts):
        raise EditResolutionError(
            f"edit {index}: symbol {symbol!r} must be dotted Python names, for example HISTORY "
            "or Case.validate"
        )
    tree = _parse_python(source.text, path=path, index=index)
    scope: Sequence[ast.stmt] = tree.body
    scope_label = "top-level definition"
    node = _single_definition(scope, parts[0], index=index, scope=scope_label, source=source)
    for part in parts[1:]:
        if not isinstance(node, ast.ClassDef):
            raise EditResolutionError(
                f"edit {index}: {symbol!r} walks through "
                f"{getattr(node, 'name', parts[0])!r}, which is not a class — only dotted "
                "Class.member paths reach inside a class"
            )
        scope = node.body
        scope_label = f'member of class "{node.name}"'
        node = _single_definition(scope, part, index=index, scope=scope_label, source=source)
    start, end, start_line, start_char = _definition_span(source, node, index=index, symbol=symbol)
    indent = source.lines[start_line - 1][:start_char]
    replacement = _indented(operation.new_text, indent, symbol=symbol, index=index, line=start_line)
    replacement = _terminated(replacement, source.text, start=start, end=end)
    return _resolved(
        source,
        index=index,
        kind="symbol",
        label=f'symbol "{symbol}"',
        start=start,
        end=end,
        replacement=replacement,
    )


def _resolve_between(operation: EditTarget, source: _Source, *, index: int) -> ResolvedEdit:
    start_marker = operation.start_marker
    end_marker = operation.end_marker
    if start_marker is None or end_marker is None:  # pragma: no cover - _target_kind guards
        raise EditResolutionError(f"edit {index}: start_marker and end_marker name a region")
    if not start_marker.strip() or not end_marker.strip():
        raise EditResolutionError(
            f"edit {index}: markers must contain visible text; a blank line is not a target — "
            "use old_text for a blank line"
        )
    if start_marker == end_marker:
        raise EditResolutionError(
            f"edit {index}: start_marker and end_marker are identical; a between edit needs two "
            "different lines — use old_text to replace a single line"
        )
    start_index = _single_marker_line(source.lines, start_marker, index=index, name="start_marker")
    end_index = _single_marker_line(source.lines, end_marker, index=index, name="end_marker")
    start_last = start_index + _marker_line_count(start_marker) - 1
    if end_index <= start_last:
        raise EditResolutionError(
            f"edit {index}: end_marker is on line {end_index + 1}, before or inside start_marker "
            f"on line {start_index + 1}; markers must be ordered top to bottom"
        )
    first_content_line = start_last + 1
    start = (
        source.starts[first_content_line]
        if first_content_line < len(source.starts)
        else len(source.text)
    )
    end = source.starts[end_index]
    replacement = operation.new_text
    if replacement and not replacement.endswith("\n"):
        replacement = f"{replacement}\n"
    return _resolved(
        source,
        index=index,
        kind="between",
        label=_quoted_label("between", start_marker.splitlines()[0]),
        start=start,
        end=end,
        replacement=replacement,
    )


# --------------------------------------------------------------------------------------
# Python definitions
# --------------------------------------------------------------------------------------


def _parse_python(text: str, *, path: str, index: int) -> ast.Module:
    try:
        return ast.parse(text)
    except SyntaxError as exc:
        line = exc.lineno if exc.lineno is not None else 1
        column = exc.offset if exc.offset is not None else 1
        raise EditResolutionError(
            f"edit {index}: {path} does not parse as Python (line {line}, column {column}: "
            f"{exc.msg}); a symbol target needs a parseable file — use old_text, or start_marker "
            "with end_marker"
        ) from exc


def _single_definition(
    body: Sequence[ast.stmt],
    name: str,
    *,
    index: int,
    scope: str,
    source: _Source,
) -> ast.stmt:
    found = _definitions_named(body, name)
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        lines = ", ".join(str(node.lineno) for node in found[:5])
        raise EditResolutionError(
            f"edit {index}: symbol {name!r} is defined {len(found)} times at the {scope} level "
            f"(lines {lines}); name a unique one with old_text, or the lines between two markers"
        )
    available = _addressable_names(body, source)
    nearest = difflib.get_close_matches(name, available, n=4)
    hint = f"; closest: {', '.join(nearest)}" if nearest else ""
    raise EditResolutionError(
        f"edit {index}: no {scope} named {name!r}{hint}; read the file to see its definitions, "
        "or use old_text or the lines between two markers"
    )


def _definitions_named(body: Sequence[ast.stmt], name: str) -> list[ast.stmt]:
    """The statements that bind ``name``: a def/class, or any assignment binding it."""
    found: list[ast.stmt] = []
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.name == name:
                found.append(node)
        elif isinstance(node, ast.Assign):
            if any(name in _binding_names(target) for target in node.targets):
                found.append(node)
        elif isinstance(node, ast.AnnAssign):
            if name in _binding_names(node.target):
                found.append(node)
    return found


def _addressable_names(body: Sequence[ast.stmt], source: _Source) -> list[str]:
    """The names a symbol target could actually edit here: unique and line-exclusive."""
    counts: dict[str, int] = {}
    for candidate in _definition_names(body):
        counts[candidate] = counts.get(candidate, 0) + 1
    addressable: list[str] = []
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [bound for target in node.targets for bound in _binding_names(target)]
        elif isinstance(node, ast.AnnAssign):
            names = _binding_names(node.target)
        else:
            continue
        for candidate in names:
            if counts.get(candidate, 0) != 1:
                continue
            try:
                _definition_span(source, node, index=0, symbol=candidate)
            except EditResolutionError:
                continue
            addressable.append(candidate)
    return addressable


def _definition_names(body: Sequence[ast.stmt]) -> list[str]:
    names: list[str] = []
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.extend(_binding_names(target))
        elif isinstance(node, ast.AnnAssign):
            names.extend(_binding_names(node.target))
    return names


def _binding_names(target: ast.expr) -> list[str]:
    """Every name a single assignment target binds (``A``, ``A, B``, ``(A, B)``)."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple | ast.List):
        names: list[str] = []
        for element in target.elts:
            names.extend(_binding_names(element))
        return names
    if isinstance(target, ast.Starred):
        return _binding_names(target.value)
    return []


def _definition_span(
    source: _Source,
    node: ast.stmt,
    *,
    index: int,
    symbol: str,
) -> tuple[int, int, int, int]:
    """The whole-line span of ``node`` (decorators included), refusing shared lines."""
    end_line = node.end_lineno
    if end_line is None:  # pragma: no cover - ast always populates it on 3.14
        raise EditResolutionError(
            f"edit {index}: cannot determine the end of {symbol!r}; use old_text, or "
            "start_marker with end_marker"
        )
    decorators: tuple[ast.expr, ...] = (
        tuple(node.decorator_list)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        else ()
    )
    if decorators:
        at_lines = [_decorator_at_line(source, decorator) for decorator in decorators]
        if any(line is None for line in at_lines):
            raise EditResolutionError(
                f"edit {index}: cannot locate the @ line of a multi-line decorator on "
                f"{symbol!r}; use old_text, or start_marker with end_marker"
            )
        start_line = min(line for line in at_lines if line is not None)
        start_char = len(source.lines[start_line - 1]) - len(source.lines[start_line - 1].lstrip())
    else:
        start_line = node.lineno
        start_char = _char_column(source.lines[start_line - 1], node.col_offset)
    end_column = node.end_col_offset
    if end_column is None:  # pragma: no cover - ast always populates it on 3.14
        raise EditResolutionError(
            f"edit {index}: cannot determine the end of {symbol!r}; use old_text, or "
            "start_marker with end_marker"
        )
    end_char = _char_column(source.lines[end_line - 1], end_column)
    _refuse_shared_line(
        source, start_line, start_char, end_line, end_char, index=index, symbol=symbol
    )
    start = source.starts[start_line - 1]
    end = source.starts[end_line] if end_line < len(source.starts) else len(source.text)
    return start, end, start_line, start_char


def _decorator_at_line(source: _Source, decorator: ast.expr) -> int | None:
    """The 1-based line carrying the decorator's ``@``, or None when it cannot be found.

    ``decorator.lineno`` is the line of the expression, which for a parenthesized
    decorator (``@(`` newline ``deco`` newline ``)``) is not the line carrying the ``@``.
    Scanning back over the continuation punctuation finds it; anything else means the
    span cannot be established and the caller refuses instead of guessing.
    """
    start = source.starts[decorator.lineno - 1] + _char_column(
        source.lines[decorator.lineno - 1], decorator.col_offset
    )
    cursor = start - 1
    while cursor >= 0 and source.text[cursor] in " \t\r\n\\,([{":
        cursor -= 1
    if cursor < 0 or source.text[cursor] != "@":
        return None
    line = source.line_number(cursor)
    if not source.lines[line - 1].lstrip().startswith("@"):
        return None
    return line


def _refuse_shared_line(
    source: _Source,
    start_line: int,
    start_char: int,
    end_line: int,
    end_char: int,
    *,
    index: int,
    symbol: str,
) -> None:
    """A symbol edit replaces whole lines, so the definition must own both of them."""
    prefix = source.lines[start_line - 1][:start_char]
    if prefix.strip():
        raise EditResolutionError(
            f"edit {index}: symbol {symbol!r} starts on line {start_line} after other code "
            f"({prefix.strip()[:40]!r}); a symbol target replaces the whole statement, so use "
            "old_text, or start_marker with end_marker"
        )
    suffix = source.lines[end_line - 1][end_char:]
    if suffix.strip() and not suffix.lstrip().startswith("#"):
        raise EditResolutionError(
            f"edit {index}: symbol {symbol!r} ends on line {end_line} before other code "
            f"({suffix.strip()[:40]!r}); a symbol target replaces the whole statement, so use "
            "old_text, or start_marker with end_marker"
        )


# --------------------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Source:
    """The presented text, its lines, and the line starts every resolver measures against."""

    text: str
    lines: tuple[str, ...]
    starts: tuple[int, ...]

    def __init__(self, text: str) -> None:
        starts = [0]
        for position, character in enumerate(text):
            if character == "\n":
                starts.append(position + 1)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "lines", tuple(text.split("\n")))
        object.__setattr__(self, "starts", tuple(starts))

    def line_of(self, offset: int) -> int:
        """The 0-based line index containing ``offset``."""
        return bisect.bisect_right(self.starts, offset) - 1

    def line_number(self, offset: int) -> int:
        """The 1-based line number containing ``offset``."""
        return self.line_of(offset) + 1

    def span(self, start: int, end: int) -> str:
        first = self.line_number(start)
        last = self.line_number(max(start, end - 1))
        return f"line {first}" if first == last else f"lines {first}-{last}"


def _positions(text: str, needle: str) -> list[int]:
    """The first few match offsets; a refusal lists lines, never every occurrence."""
    positions: list[int] = []
    cursor = 0
    while len(positions) < _MAX_POSITIONS and (position := text.find(needle, cursor)) >= 0:
        positions.append(position)
        cursor = position + len(needle)
    return positions


def _marker_line_count(marker: str) -> int:
    lines = marker.split("\n")
    if len(lines) > 1 and lines[-1] == "":
        lines.pop()
    return len(lines)


def _single_marker_line(
    file_lines: Sequence[str],
    marker: str,
    *,
    index: int,
    name: str,
) -> int:
    marker_lines = marker.split("\n")
    if len(marker_lines) > 1 and marker_lines[-1] == "":
        marker_lines.pop()
    width = len(marker_lines)
    matches = [
        position
        for position in range(0, len(file_lines) - width + 1)
        if list(file_lines[position : position + width]) == marker_lines
    ][: _MAX_POSITIONS + 1]
    if not matches:
        raise EditResolutionError(
            f"edit {index}: {name} is not a whole line in this file: no line equals "
            f"{marker_lines[0]!r}; a marker is the exact text of complete line(s) — use "
            "old_text, or copy the line from the file"
        )
    if len(matches) > 1:
        lines = ", ".join(str(position + 1) for position in matches[:5])
        raise EditResolutionError(
            f"edit {index}: {name} occurs {len(matches)} times (lines {lines}); each marker must "
            "occur once — add a neighbouring line to the marker, or use old_text"
        )
    return matches[0]


def _leading_whitespace(line: str) -> str:
    width = 0
    while width < len(line) and line[width] in " \t\x0c":
        width += 1
    return line[:width]


def _indented(
    new_text: str,
    indent: str,
    *,
    symbol: str,
    index: int,
    line: int,
) -> str:
    """Give a nested definition's replacement the indentation its target had."""
    if not indent or not new_text:
        return new_text
    lines = new_text.split("\n")
    non_empty = [text_line for text_line in lines if text_line.strip()]
    if not non_empty:
        return new_text
    leading = _leading_whitespace(non_empty[0])
    if not leading:
        # Written at column 0: indent the block, keeping its relative indentation.
        return "\n".join(
            "" if not text_line.strip() else f"{indent}{text_line}" for text_line in lines
        )
    if leading == indent:
        return new_text
    raise EditResolutionError(
        f"edit {index}: the replacement for nested symbol {symbol!r} (line {line}) must start "
        f"at column {len(indent)} (its indentation), or at column 0 to be indented; it starts "
        f"at column {len(leading)} — use old_text, or the lines between two markers"
    )


def _line_count(text: str) -> int:
    """Logical lines in a presented (LF) text, counting a missing final newline."""
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _terminated(new_text: str, text: str, *, start: int, end: int) -> str:
    """Keep the line ending the target region had: a replaced block ends with a newline."""
    if not new_text or new_text.endswith("\n"):
        return new_text
    region_ended_with_newline = end > start and text[end - 1] == "\n"
    return f"{new_text}\n" if region_ended_with_newline else new_text


def _char_column(line: str, byte_column: int) -> int:
    """Convert a UTF-8 byte column (ast's convention) to a character index in the line."""
    if byte_column <= 0:
        return 0
    encoded = line.encode("utf-8")
    if byte_column >= len(encoded):
        return len(line)
    return len(encoded[:byte_column].decode("utf-8"))


def _quoted_label(kind: str, text: str) -> str:
    escaped = text.replace("\n", "\\n")
    clipped = (
        escaped if len(escaped) <= _MAX_LABEL_CHARS else f"{escaped[: _MAX_LABEL_CHARS - 3]}..."
    )
    return f'{kind} "{clipped}"'


def _resolved(
    source: _Source,
    *,
    index: int,
    kind: EditKind,
    label: str,
    start: int,
    end: int,
    replacement: str,
) -> ResolvedEdit:
    if source.text[start:end] == replacement:
        raise EditResolutionError(
            f"edit {index}: the replacement equals the region it replaces; nothing to do"
        )
    first_line = source.line_number(start)
    return ResolvedEdit(
        index=index,
        kind=kind,
        label=label,
        start=start,
        end=end,
        replacement=replacement,
        first_line=first_line,
        last_line=first_line if end <= start else source.line_number(end - 1),
    )


# --------------------------------------------------------------------------------------
# Hunk rendering
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Change:
    old_first: int
    old_last: int
    new_first: int
    new_last: int


def _changes(original: str, updated: str, edits: Sequence[ResolvedEdit]) -> list[_Change]:
    """The whole-line ranges each edit changes on the old and the new side of the diff."""
    old_source = _Source(original)
    new_source = _Source(updated)
    old_line_count = (
        len(old_source.lines) - (1 if original.endswith("\n") else 0) if original else 0
    )
    new_line_count = len(new_source.lines) - (1 if updated.endswith("\n") else 0) if updated else 0
    changes: list[_Change] = []
    delta = 0
    for edit in sorted(edits, key=lambda edit: (edit.start, edit.end)):
        old_first = old_source.line_of(edit.start)
        if edit.end > edit.start:
            old_last = old_source.line_of(edit.end - 1) + 1
            end_line = old_source.line_of(edit.end - 1)
            suffix_remains = max(0, _line_content_end(old_source, end_line) - edit.end)
            consumed_terminator = original[edit.end - 1] == "\n"
            terminator_remains = not consumed_terminator and edit.end < len(original)
            merges_next_line = (
                bool(edit.replacement)
                and not edit.replacement.endswith("\n")
                and consumed_terminator
                and edit.end < len(original)
            )
            includes_joined_line = merges_next_line or (
                bool(edit.replacement)
                and (suffix_remains > 0 or (edit.replacement.endswith("\n") and terminator_remains))
            )
        else:
            old_last = old_first
            suffix_remains = 0
            merges_next_line = False
            includes_joined_line = False
        new_start = edit.start + delta
        new_first = new_source.line_of(new_start)
        tail = new_start + len(edit.replacement)
        if edit.replacement:
            if includes_joined_line:
                new_last = new_source.line_of(tail) + 1
            else:
                new_last = new_source.line_of(tail - 1) + 1
        elif edit.end > edit.start:
            prefix_remains = edit.start - old_source.starts[old_first]
            line_survives = edit.end < len(original) and original[edit.end - 1] != "\n"
            if (line_survives or prefix_remains > 0 or suffix_remains > 0) and (
                new_first < new_line_count
            ):
                new_last = new_first + 1
            else:
                new_last = new_first
        else:
            new_last = new_first
        if includes_joined_line and suffix_remains == 0 and not edit.replacement.endswith("\n"):
            # The replacement's last line merges into the following old line; that old
            # line is changed on the old side too, or the hunk shows it as context.
            old_last = min(old_line_count, old_last + 1)
        changes.append(_Change(old_first, old_last, new_first, new_last))
        delta += len(edit.replacement) - (edit.end - edit.start)
    return _merged(changes)


def _line_content_end(source: _Source, index: int) -> int:
    """The character offset where line ``index`` ends, before its terminator."""
    if index + 1 < len(source.starts):
        return source.starts[index + 1] - 1
    return len(source.text)


def _merged(changes: Sequence[_Change]) -> list[_Change]:
    merged: list[_Change] = []
    for change in changes:
        while merged and _touches(merged[-1], change):
            previous = merged.pop()
            change = _Change(
                min(previous.old_first, change.old_first),
                max(previous.old_last, change.old_last),
                min(previous.new_first, change.new_first),
                max(previous.new_last, change.new_last),
            )
        merged.append(change)
    return merged


def _touches(left: _Change, right: _Change) -> bool:
    """Whether two line ranges overlap or share lines, empty ranges included."""
    return _ranges_touch(left.old_first, left.old_last, right.old_first, right.old_last) or (
        _ranges_touch(left.new_first, left.new_last, right.new_first, right.new_last)
    )


def _ranges_touch(left_first: int, left_last: int, right_first: int, right_last: int) -> bool:
    if left_first == left_last:
        return right_first <= left_first < right_last
    if right_first == right_last:
        return left_first <= right_first < left_last
    return left_first < right_last and right_first < left_last


def _hunk_header(old_from: int, old_count: int, new_from: int, new_count: int) -> str:
    old_line = old_from + 1 if old_count else old_from
    new_line = new_from + 1 if new_count else new_from
    return f"@@ -{old_line},{old_count} +{new_line},{new_count} @@"
