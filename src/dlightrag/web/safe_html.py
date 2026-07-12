# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Safe HTML fragment rendering for browser-inserted web UI payloads."""

import re
from typing import Any
from urllib.parse import urlparse

import nh3

from dlightrag.citations.schemas import SourceReferencePayload
from dlightrag.web.deps import render_partial
from dlightrag.web.markdown import render_markdown

_ALLOWED_TAGS = {
    "a",
    "abbr",
    "b",
    "blockquote",
    "br",
    "button",
    "caption",
    "code",
    "col",
    "colgroup",
    "dd",
    "del",
    "details",
    "div",
    "dl",
    "dt",
    "em",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "i",
    "img",
    "li",
    "mark",
    "ol",
    "p",
    "pre",
    "s",
    "span",
    "strong",
    "sub",
    "summary",
    "sup",
    "svg",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "u",
    "ul",
    # SVG shape elements (used by source panel download icons)
    "path",
    "line",
    "polyline",
}

_ALLOWED_ATTRS: dict[str, set[str]] = {
    "*": {
        "class",
        "hidden",
        "id",
        "role",
        "tabindex",
        "title",
        # SVG presentation attributes (download icon, upload icons)
        "d",
        "fill",
        "points",
        "stroke",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-width",
        "viewbox",
        "viewBox",
        "width",
        "height",
        "x1",
        "x2",
        "y1",
        "y2",
    },
    "a": {"aria-label", "download", "href", "title"},
    "button": {"type"},
    "col": {"span"},
    "colgroup": {"span"},
    "img": {"alt", "loading", "src"},
    "td": {"colspan", "rowspan"},
    "th": {"colspan", "rowspan", "scope"},
}
_URL_ATTRS = {"href", "src", "data-src", "data-full-src"}
_DATA_ACTIONS = {
    "filter-source",
    "open-lightbox",
    "open-ref-source",
    "show-all-sources",
    "toggle-doc",
}
_ALLOWED_IDS = {"answer-content", "source-data"}
_CLEAN_CONTENT_TAGS = {"iframe", "object", "script", "style", "template"}
_URL_SCHEMES = {"http", "https", "data", "blob"}
_DATA_IMAGE_RE = re.compile(r"^data:image/(?:gif|jpe?g|png|webp);base64,", re.IGNORECASE)


def _is_safe_url(value: str) -> bool:
    candidate = value.strip()
    if not candidate:
        return True
    lowered = candidate.lower()
    if lowered.startswith("/") and not lowered.startswith("//"):
        return True
    if lowered.startswith("data:image/"):
        return bool(_DATA_IMAGE_RE.match(candidate))
    parsed = urlparse(candidate)
    if parsed.scheme == "blob":
        return True
    return parsed.scheme in {"http", "https"}


def _attribute_filter(_tag: str, attr: str, value: str) -> str | None:
    lowered = attr.lower()
    if lowered.startswith("on"):
        return None
    if lowered == "id":
        return value if value in _ALLOWED_IDS else None
    if lowered in _URL_ATTRS:
        return value if _is_safe_url(value) else None
    if lowered == "data-action":
        return value if value in _DATA_ACTIONS else None
    return value


def sanitize_html_fragment(html: str) -> str:
    """Sanitize app-rendered HTML before it is inserted via browser innerHTML."""
    return nh3.clean(
        html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRS,
        clean_content_tags=_CLEAN_CONTENT_TAGS,
        attribute_filter=_attribute_filter,
        generic_attribute_prefixes={"data-"},
        url_schemes=_URL_SCHEMES,
    )


def safe_answer_preview(markdown_text: str) -> str:
    """Render and sanitize a streaming answer preview."""
    return sanitize_html_fragment(render_markdown(markdown_text))


def safe_answer_done(
    *,
    answer: str,
    sources: list[SourceReferencePayload],
    answer_images: list[dict[str, Any]],
) -> str:
    """Render and sanitize the final answer partial."""
    html = render_partial(
        "partials/answer_done.html",
        answer=answer,
        sources=sources,
        answer_images=answer_images,
    )
    return sanitize_html_fragment(html)


def safe_source_panel(*, sources: list[SourceReferencePayload]) -> str:
    """Render and sanitize the source panel fragment."""
    return sanitize_html_fragment(render_partial("partials/source_panel.html", sources=sources))


__all__ = [
    "safe_answer_done",
    "safe_answer_preview",
    "safe_source_panel",
    "sanitize_html_fragment",
]
