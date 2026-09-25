# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Exact source positions through markdown-it's existing grammar.

Markdown's block parser removes container prefixes and table escapes before its
inline pass. Carry compact source runs through those string operations instead
of finding rendered text back in the document. The grammar remains upstream's;
normalization, container extraction and table cells retain their source runs.
"""

import re
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Any, overload

from markdown_it import MarkdownIt
from markdown_it.common.utils import isStrSpace
from markdown_it.parser_block import ParserBlock
from markdown_it.rules_block.state_block import StateBlock
from markdown_it.rules_block.table import table
from markdown_it.rules_core.state_core import StateCore
from markdown_it.token import Token


@dataclass(frozen=True, slots=True)
class _Run:
    start: int
    end: int
    source: int | None


class SourceText(str):
    """Parser-private text whose contiguous runs retain original positions."""

    runs: tuple[_Run, ...]

    def __new__(cls, text: str, runs: tuple[_Run, ...]) -> SourceText:
        value = super().__new__(cls, text)
        value.runs = runs
        return value

    @overload
    def __getitem__(self, key: int) -> str: ...

    @overload
    def __getitem__(self, key: slice) -> SourceText: ...

    def __getitem__(self, key: int | slice) -> str | SourceText:
        value = super().__getitem__(key)
        if isinstance(key, int):
            return value
        start, stop, step = key.indices(len(self))
        if step != 1:
            raise ValueError("Markdown source mapping requires contiguous slices")
        runs = []
        index = max(0, bisect_right(self.runs, start, key=lambda run: run.start) - 1)
        for run_index in range(index, len(self.runs)):
            run = self.runs[run_index]
            if run.start >= stop:
                break
            lo, hi = max(start, run.start), min(stop, run.end)
            if lo < hi:
                source = None if run.source is None else run.source + lo - run.start
                runs.append(_Run(lo - start, hi - start, source))
        return SourceText(value, tuple(runs))

    def strip(self, chars: str | None = None) -> SourceText:
        start = len(self) - len(super().lstrip(chars))
        return self[start : len(super().rstrip(chars))]

    def __add__(self, other: str) -> SourceText:
        return _join((self, other))

    def __radd__(self, other: str) -> SourceText:
        return _join((other, self))

    def source_span(self, start: int, end: int) -> tuple[int, int]:
        """A marker must occupy an unmodified, contiguous original span."""
        selected = self[start:end]
        if len(selected.runs) != 1 or selected.runs[0].source is None:
            raise ValueError("Citation marker lost its contiguous Markdown source span")
        source = selected.runs[0].source
        return source, source + end - start

    def source_extent(self) -> tuple[int, int]:
        first = next((run for run in self.runs if run.source is not None), None)
        last = next((run for run in reversed(self.runs) if run.source is not None), None)
        if first is None or last is None or first.source is None or last.source is None:
            raise ValueError("Markdown inline content has no source extent")
        return first.source, last.source + last.end - last.start


def _join(parts: Iterable[str]) -> SourceText:
    texts: list[str] = []
    runs: list[_Run] = []
    position = 0
    for part in parts:
        texts.append(part)
        incoming = part.runs if isinstance(part, SourceText) else (_Run(0, len(part), None),)
        for run in incoming:
            if run.start == run.end:
                continue
            shifted = _Run(position + run.start, position + run.end, run.source)
            if runs:
                last = runs[-1]
                contiguous = (last.source is None and shifted.source is None) or (
                    last.source is not None
                    and shifted.source == last.source + last.end - last.start
                )
                if contiguous and last.end == shifted.start:
                    runs[-1] = _Run(last.start, shifted.end, last.source)
                    continue
            runs.append(shifted)
        position += len(part)
    return SourceText("".join(texts), tuple(runs))


_NORMALIZED = re.compile(r"\r\n?|\x00")


def _normalize(state: StateCore) -> None:
    original = state.src
    parts: list[SourceText] = []
    cursor = 0
    for match in _NORMALIZED.finditer(original):
        start, end = match.span()
        parts.append(SourceText(original[cursor:start], (_Run(0, start - cursor, cursor),)))
        value = "\ufffd" if match[0] == "\x00" else "\n"
        parts.append(SourceText(value, (_Run(0, 1, start),)))
        cursor = end
    parts.append(SourceText(original[cursor:], (_Run(0, len(original) - cursor, cursor),)))
    if not state.inlineMode and original and original[-1] not in "\r\n":
        # markdown-it's table terminators can probe one character past a final
        # empty container line. Supply a parser-only sentinel, never source data.
        parts.append(SourceText("\n", (_Run(0, 1, None),)))
    state.src = _join(parts)


class _SourceBlockState(StateBlock):
    @cached_property
    def plain_source(self) -> str:
        # Every block tries the table rule. Copy the str subclass at most once
        # per document, not once per attempted table or paragraph.
        return str(self.src)

    def getLines(self, begin: int, end: int, indent: int, keepLastLF: bool) -> str:
        """Upstream line extraction, with a source-preserving join.

        Container rules already updated bMarks/tShift/bsCount before this call;
        using that state also handles nested lists, blockquotes and expanded tabs.
        """
        parts: list[str] = []
        for line in range(begin, end):
            line_indent = 0
            first = line_start = self.bMarks[line]
            last = self.eMarks[line] + int(line + 1 < end or keepLastLF)
            while first < last and line_indent < indent:
                char = self.src[first]
                if isStrSpace(char):
                    line_indent += 4 - (line_indent + self.bsCount[line]) % 4 if char == "\t" else 1
                elif first - line_start < self.tShift[line]:
                    line_indent += 1
                else:
                    break
                first += 1
            if line_indent > indent:
                parts.append(" " * (line_indent - indent))
            parts.append(self.src[first:last])
        content = _join(parts)
        if content.endswith("\n") and content.runs[-1].source is None:
            # An unterminated fence must not acquire the parser-only LF in its
            # code content. Real source newlines retain their mapped run.
            return content[:-1]
        return content


class _SourceBlockParser(ParserBlock):
    def parse(
        self, src: str, md: MarkdownIt, env: Any, outTokens: list[Token]
    ) -> list[Token] | None:
        if not src:
            return None
        state = _SourceBlockState(src, md, env, outTokens)
        self.tokenize(state, state.line, state.lineMax)
        return state.tokens


def _table_cells(line: SourceText) -> list[SourceText]:
    """Map upstream escapedSplit's pipe removal, joining each cell only once."""
    cells: list[SourceText] = []
    parts: list[str] = []
    cursor = 0
    escaped = False
    for position, char in enumerate(line):
        if char == "|":
            parts.append(line[cursor : position - 1 if escaped else position])
            if escaped:
                cursor = position
            else:
                cells.append(_join(parts))
                parts = []
                cursor = position + 1
        escaped = char == "\\"
    parts.append(line[cursor:])
    cells.append(_join(parts))
    if cells and not cells[0]:
        cells.pop(0)
    if cells and not cells[-1]:
        cells.pop()
    return [cell.strip() for cell in cells]


def _table(state: StateBlock, start: int, end: int, silent: bool) -> bool:
    """Run the upstream grammar, then map its emitted cells by row and column.

    Its escapedSplit accumulates strings with +=. Let it use ordinary strings
    so a cell with thousands of escaped pipes does not repeatedly copy source
    runs. The mapping below performs only that cell extraction, never recognition.
    """
    source = state.src
    if not isinstance(source, SourceText):
        return table(state, start, end, silent)
    if not isinstance(state, _SourceBlockState):
        raise ValueError("Mapped table parsing requires mapped block state")
    first_token = len(state.tokens)
    state.src = state.plain_source
    try:
        matched = table(state, start, end, silent)
    finally:
        state.src = source
    if silent or not matched:
        return matched
    row = -1
    column = 0
    cells: list[SourceText] = []
    for token in state.tokens[first_token:]:
        if token.type != "inline" or token.map is None:
            continue
        if token.map[0] != row:
            row, column = token.map[0], 0
            line = source[state.bMarks[row] + state.tShift[row] : state.eMarks[row]].strip()
            cells = _table_cells(line)
        content = cells[column] if column < len(cells) else SourceText("", ())
        if content != token.content:
            raise ValueError("Markdown table source mapping differs from the parser's cell")
        token.content = content
        column += 1
    return True


def install_source_mapping(md: MarkdownIt) -> None:
    """Adapt one parser instance; no global patch or alternate grammar."""
    parser = _SourceBlockParser()
    parser.ruler = md.block.ruler
    md.block = parser
    md.core.ruler.at("normalize", _normalize)
    md.block.ruler.at("table", _table)
