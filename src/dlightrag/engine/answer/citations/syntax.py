# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The single citation syntax shared by validation and all projections."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.rules_inline import StateInline, image
from markdown_it.token import Token

from dlightrag.engine.answer.markdown_source import SourceText

from .contracts import CITATION_PATTERN, DOC_CITATION_PATTERN


@dataclass(frozen=True, slots=True)
class Citation:
    ref_id: str
    chunk_idx: int | None
    start: int
    end: int
    marker: str
    scope: tuple[int, int]

    @property
    def key(self) -> str:
        return self.ref_id if self.chunk_idx is None else f"{self.ref_id}-{self.chunk_idx}"


@dataclass(frozen=True, slots=True)
class CitationDocument:
    text: str
    citations: tuple[Citation, ...]

    def rewrite(self, replacement: Callable[[Citation], str]) -> str:
        parts: list[str] = []
        cursor = 0
        for citation in self.citations:
            parts.extend((self.text[cursor : citation.start], replacement(citation)))
            cursor = citation.end
        parts.append(self.text[cursor:])
        return "".join(parts)


def citation_reference(marker: str) -> str | None:
    """The source id of one complete marker, for already-parsed link labels."""
    match = CITATION_PATTERN.fullmatch(marker) or DOC_CITATION_PATTERN.fullmatch(marker)
    return match[1] if match is not None else None


def _citation(state: StateInline, silent: bool) -> bool:
    # Silent parsing discovers balanced link labels. Consuming a bracket pair
    # there would change Markdown syntax before the link rule can claim it.
    if silent or state.linkLevel or state.env.get("_citation_image"):
        return False
    match = CITATION_PATTERN.match(state.src, state.pos)
    chunk_idx = int(match[2]) if match is not None else None
    if match is None:
        match = DOC_CITATION_PATTERN.match(state.src, state.pos)
    if match is None:
        return False
    if not isinstance(state.src, SourceText):
        raise ValueError("Citation parsing requires mapped Markdown source")
    start, end = state.src.source_span(*match.span())
    token = state.push("citation", "", 0)
    token.content = match[0]
    token.meta["citation"] = Citation(
        match[1], chunk_idx, start, end, match[0], state.src.source_extent()
    )
    state.pos = match.end()
    return True


def _image(state: StateInline, silent: bool) -> bool:
    previous = state.env.get("_citation_image", False)
    state.env["_citation_image"] = True
    try:
        return image(state, silent)
    finally:
        state.env["_citation_image"] = previous


def _render_citation(_renderer: Any, tokens: list[Token], idx: int, _options: Any, env: Any) -> str:
    citation = tokens[idx].meta["citation"]
    render = env.get("render_citation")
    return render(citation) if render is not None else escape(citation.marker)


def install_citation_syntax(md: MarkdownIt) -> None:
    md.inline.ruler.after("image", "citation", _citation)
    md.inline.ruler.at("image", _image)
    md.add_render_rule("citation", _render_citation)


@lru_cache(maxsize=1)
def _parser() -> MarkdownIt:
    # The grammar factory imports this plugin, so construction stays lazy.
    from dlightrag.engine.answer.markdown import answer_markdown

    return answer_markdown()


def parse_citations(text: str) -> CitationDocument:
    citations = tuple(
        token.meta["citation"]
        for block in _parser().parse(text)
        for token in block.children or ()
        if token.type == "citation"
    )
    for citation in citations:
        if text[citation.start : citation.end] != citation.marker:
            raise ValueError("Citation source span does not match the original marker")
    return CitationDocument(text, citations)
