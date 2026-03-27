"""FastAPI dependency injection and Jinja2 environment for web routes."""

from __future__ import annotations

import re
from pathlib import Path

import nh3
from fastapi import Cookie, Request
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from dlightrag.citations.parser import CITATION_PATTERN
from dlightrag.web.markdown import render_chunk_content, render_markdown

_TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

DEFAULT_WORKSPACE = "default"

# Allowlist for nh3 sanitisation of source chunk content.
_CHUNK_ALLOWED_TAGS = {
    # table
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
    # text formatting
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
    # structure
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "div",
    "span",
    # lists
    "ul",
    "ol",
    "li",
    "dl",
    "dt",
    "dd",
    # code
    "pre",
    "code",
    # links
    "a",
    # misc
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


# ---------------------------------------------------------------------------
# Custom Jinja2 filters
# ---------------------------------------------------------------------------


def _protect_code_blocks(html: str) -> tuple[str, list[str]]:
    """Replace <pre>...</pre> and <code>...</code> with indexed placeholders.

    Protects code content from citation regex replacement.
    """
    protected: list[str] = []

    def _replace(m: re.Match) -> str:
        idx = len(protected)
        protected.append(m.group(0))
        return f"\x00CODE{idx}\x00"

    # Protect <pre> blocks first (they contain <code>)
    html = re.sub(r"<pre[^>]*>.*?</pre>", _replace, html, flags=re.DOTALL)
    # Then protect remaining inline <code>
    html = re.sub(r"<code[^>]*>.*?</code>", _replace, html, flags=re.DOTALL)
    return html, protected


def _restore_code_blocks(html: str, protected: list[str]) -> str:
    """Restore code block placeholders with original content."""
    for idx in range(len(protected) - 1, -1, -1):
        html = html.replace(f"\x00CODE{idx}\x00", protected[idx])
    return html


def _citation_badges(text: str) -> Markup:
    """Replace citation patterns with clickable badge HTML.

    Handles both [ref_id-chunk_idx] and [n] doc-level formats.
    Renders Markdown first, then injects badges while protecting code blocks.

    The LLM-generated ``### References`` section is preserved — it provides
    a document-level citation overview without opening the sources panel.
    """
    from dlightrag.citations.parser import DOC_CITATION_PATTERN

    # Render Markdown to HTML (replaces Markup.escape)
    html = render_markdown(str(text))

    # Protect code blocks from citation regex
    html, protected = _protect_code_blocks(html)

    # First pass: replace chunk-level [ref-chunk] with badges
    def _chunk_badge(m: re.Match) -> str:
        ref_id = m.group(1)
        chunk_idx = m.group(2)
        return (
            f'<span class="citation-badge" data-ref="{ref_id}" '
            f'data-chunk="{chunk_idx}" onclick="filterSource(this)">'
            f"[{ref_id}-{chunk_idx}]</span>"
        )

    result = CITATION_PATTERN.sub(_chunk_badge, html)

    # Second pass: replace doc-level [n] with badges
    def _doc_badge(m: re.Match) -> str:
        ref_id = m.group(1)
        return (
            f'<span class="citation-badge" data-ref="{ref_id}" '
            f'onclick="filterSource(this)">'
            f"[{ref_id}]</span>"
        )

    result = DOC_CITATION_PATTERN.sub(_doc_badge, result)

    # Restore code blocks
    result = _restore_code_blocks(result, protected)

    return Markup(result)


def _highlight_content(text: str, phrases: list[str] | None = None) -> Markup:
    """Render chunk content as HTML, sanitize, then inject highlight spans."""
    html = render_chunk_content(text)
    html = nh3.clean(html, tags=_CHUNK_ALLOWED_TAGS, attributes=_CHUNK_ALLOWED_ATTRS)

    if phrases:
        for phrase in phrases:
            # HTML-escape the phrase to match entity-encoded text in rendered output
            # (e.g., "&" in phrase matches "&amp;" in HTML)
            safe_phrase = re.escape(str(Markup.escape(phrase)))
            html = re.sub(
                f"(?<=>)([^<]*?)({safe_phrase})",
                r'\1<span class="highlight">\2</span>',
                html,
                flags=re.IGNORECASE,
                count=1,
            )

    return Markup(html)


def _basename(path: str) -> str:
    """Extract filename from a path string."""
    return Path(path).name


# Register filters on the Jinja2 environment
templates.env.filters["citation_badges"] = _citation_badges
templates.env.filters["highlight_content"] = _highlight_content
templates.env.filters["basename"] = _basename


def _compute_static_hash() -> str:
    """Compute a short content hash of all static files for cache busting."""
    import hashlib

    static_dir = Path(__file__).parent / "static"
    h = hashlib.md5(usedforsecurity=False)
    for p in sorted(static_dir.rglob("*")):
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()[:10]


_STATIC_HASH: str = _compute_static_hash()


def _static_version() -> str:
    return _STATIC_HASH


templates.env.globals["static_version"] = _static_version


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


def get_workspace(dlightrag_workspace: str = Cookie(default=DEFAULT_WORKSPACE)) -> str:
    """Read current workspace from cookie, normalized to safe PG identifier."""
    from dlightrag.utils import normalize_workspace

    return normalize_workspace(dlightrag_workspace)


def get_manager(request: Request):
    """Get RAGServiceManager from app state."""
    return request.app.state.manager
