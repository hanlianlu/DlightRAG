# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Safe semantic rich-content projection for the Lit browser presentation."""

import re
from typing import Any

import nh3
from dlightrag_rag.sourcing.url import validate_public_web_url

from dlightrag.answer.citations.parser import CITATION_PATTERN, DOC_CITATION_PATTERN
from dlightrag.answer.citations.schemas import SourceReferencePayload
from dlightrag.answer.client_contracts import ClientContractModel
from dlightrag.web.markdown import (
    inject_highlights,
    normalize_chunk_source,
    render_chunk_content,
    render_markdown,
)
from dlightrag.web.safe_html import sanitize_html_fragment

_CHUNK_ALLOWED_TAGS = {
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "th",
    "td",
    "caption",
    "colgroup",
    "col",
    "p",
    "br",
    "hr",
    "b",
    "i",
    "em",
    "strong",
    "u",
    "s",
    "del",
    "sub",
    "sup",
    "mark",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "div",
    "span",
    "ul",
    "ol",
    "li",
    "dl",
    "dt",
    "dd",
    "pre",
    "code",
    "a",
    "blockquote",
    "abbr",
    "details",
    "summary",
}
_CHUNK_ALLOWED_ATTRS: dict[str, set[str]] = {
    "*": {"class"},
    "a": {"href", "title"},
    "td": {"colspan", "rowspan"},
    "th": {"colspan", "rowspan", "scope"},
    "col": {"span"},
    "colgroup": {"span"},
}


class PresentationImage(ClientContractModel):
    id: str = ""
    chunk_id: str = ""
    source_ref: str = ""
    url: str
    thumbnail_url: str
    label: str = ""
    answer_image_sent: bool = True


class PresentationSourceChunk(ClientContractModel):
    chunk_idx: int | None = None
    page_number: int | None = None
    content_html: str = ""
    image_url: str | None = None
    thumbnail_url: str | None = None


class PresentationSource(ClientContractModel):
    id: str
    title: str
    source_url: str | None = None
    download_url: str | None = None
    chunks: list[PresentationSourceChunk]


class AnswerPresentation(ClientContractModel):
    answer_text: str
    answer_html: str
    sources: list[PresentationSource]
    answer_images: list[PresentationImage]
    primary_report: str | None = None


def _protect_code_blocks(html: str) -> tuple[str, list[str]]:
    protected: list[str] = []

    def replace(match: re.Match[str]) -> str:
        index = len(protected)
        protected.append(match.group(0))
        return f"\x00CODE{index}\x00"

    html = re.sub(r"<pre[^>]*>.*?</pre>", replace, html, flags=re.DOTALL)
    html = re.sub(r"<code[^>]*>.*?</code>", replace, html, flags=re.DOTALL)
    return html, protected


def _restore_code_blocks(html: str, protected: list[str]) -> str:
    for index in range(len(protected) - 1, -1, -1):
        html = html.replace(f"\x00CODE{index}\x00", protected[index])
    return html


def _reference_label(ref_id: Any, chunk_idx: Any | None = None) -> str:
    ref = str(ref_id)
    return ref if chunk_idx is None or chunk_idx == "" else f"{ref}-{chunk_idx}"


def _reference_aria_label(ref_id: Any, chunk_idx: Any | None = None) -> str:
    ref = str(ref_id)
    return (
        f"Source {ref}"
        if chunk_idx is None or chunk_idx == ""
        else f"Source {ref}, chunk {chunk_idx}"
    )


def render_answer_html(answer: str) -> str:
    """Render answer Markdown with semantic interactive citation elements."""
    html = render_markdown(answer)
    html, protected = _protect_code_blocks(html)

    def chunk_citation(match: re.Match[str]) -> str:
        ref_id, chunk_idx = match.group(1), match.group(2)
        return (
            f'<cite class="citation-badge" data-ref="{ref_id}" data-chunk="{chunk_idx}" '
            f'role="button" tabindex="0" '
            f'aria-label="{_reference_aria_label(ref_id, chunk_idx)}">'
            f"{_reference_label(ref_id, chunk_idx)}</cite>"
        )

    def document_citation(match: re.Match[str]) -> str:
        ref_id = match.group(1)
        return (
            f'<cite class="citation-badge" data-ref="{ref_id}" role="button" tabindex="0" '
            f'aria-label="{_reference_aria_label(ref_id)}">'
            f"{_reference_label(ref_id)}</cite>"
        )

    html = CITATION_PATTERN.sub(chunk_citation, html)
    html = DOC_CITATION_PATTERN.sub(document_citation, html)
    return sanitize_html_fragment(_restore_code_blocks(html, protected))


def render_source_chunk_html(content: str, phrases: list[str] | None = None) -> str:
    source = normalize_chunk_source(content)
    html = render_chunk_content(source)
    html = nh3.clean(html, tags=_CHUNK_ALLOWED_TAGS, attributes=_CHUNK_ALLOWED_ATTRS)
    if phrases:
        html = inject_highlights(html, source, phrases)
    return html


def _public_source_url(value: str) -> str | None:
    try:
        return validate_public_web_url(value.strip())
    except ValueError:
        return None


def _presentation_source(value: SourceReferencePayload | dict[str, Any]) -> PresentationSource:
    source = SourceReferencePayload.model_validate(value)
    return PresentationSource(
        id=source.id,
        title=source.title or "Source",
        source_url=_public_source_url(source.source_uri),
        download_url=source.download_url,
        chunks=[
            PresentationSourceChunk(
                chunk_idx=chunk.chunk_idx,
                page_number=chunk.page_number,
                content_html=render_source_chunk_html(chunk.content, chunk.highlight_phrases)
                if chunk.content
                else "",
                image_url=chunk.image_url,
                thumbnail_url=chunk.thumbnail_url,
            )
            for chunk in (source.chunks or [])
        ],
    )


def build_answer_presentation(
    *,
    answer: str,
    sources: list[SourceReferencePayload] | list[dict[str, Any]],
    answer_images: list[dict[str, Any]],
    primary_report: str | None = None,
) -> AnswerPresentation:
    """Build the one safe presentation used by SSE, history, and reports."""
    return AnswerPresentation(
        answer_text=answer,
        answer_html=render_answer_html(answer),
        sources=[_presentation_source(source) for source in sources],
        answer_images=[PresentationImage.model_validate(image) for image in answer_images],
        primary_report=primary_report,
    )


__all__ = [
    "AnswerPresentation",
    "PresentationImage",
    "PresentationSource",
    "PresentationSourceChunk",
    "build_answer_presentation",
    "render_answer_html",
    "render_source_chunk_html",
]
