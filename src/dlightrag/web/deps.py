"""FastAPI dependency injection and Jinja2 environment for web routes."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import Cookie, Request
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from dlightrag.citations.parser import CITATION_PATTERN
from dlightrag.web.markdown import render_markdown

_TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

DEFAULT_WORKSPACE = "default"


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
    for idx, original in enumerate(protected):
        html = html.replace(f"\x00CODE{idx}\x00", original)
    return html


def _citation_badges(text: str) -> Markup:
    """Replace citation patterns with clickable badge HTML.

    Handles both [ref_id-chunk_idx] and [n] doc-level formats.
    Renders Markdown first, then injects badges while protecting code blocks.
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
    """Escape content then inject <span class="highlight"> for each phrase."""
    safe = str(Markup.escape(text))
    if phrases:
        for phrase in phrases:
            safe_phrase = str(Markup.escape(phrase))
            pattern = re.escape(safe_phrase)
            safe = re.sub(
                f"({pattern})",
                r'<span class="highlight">\1</span>',
                safe,
                flags=re.IGNORECASE,
                count=1,
            )
    return Markup(safe)


def _basename(path: str) -> str:
    """Extract filename from a path string."""
    return Path(path).name


# Register filters on the Jinja2 environment
templates.env.filters["citation_badges"] = _citation_badges
templates.env.filters["highlight_content"] = _highlight_content
templates.env.filters["basename"] = _basename


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


def get_workspace(dlightrag_workspace: str = Cookie(default=DEFAULT_WORKSPACE)) -> str:
    """Read current workspace from cookie."""
    return dlightrag_workspace


def get_manager(request: Request):
    """Get RAGServiceManager from app state."""
    return request.app.state.manager
