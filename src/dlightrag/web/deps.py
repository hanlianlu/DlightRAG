"""FastAPI dependency injection and Jinja2 environment for web routes."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import Cookie, Request
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from dlightrag.citations.parser import CITATION_PATTERN

_TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

DEFAULT_WORKSPACE = "default"


# ---------------------------------------------------------------------------
# Custom Jinja2 filters
# ---------------------------------------------------------------------------


def _citation_badges(text: str) -> Markup:
    """Replace citation patterns with clickable badge HTML.

    Handles both [ref_id-chunk_idx] and [n] doc-level formats.
    Escapes all text first (XSS safe), then replaces citations with badges.
    """
    from dlightrag.citations.parser import DOC_CITATION_PATTERN

    # Escape entire text first — citation patterns survive HTML escaping
    escaped = str(Markup.escape(str(text)))

    # First pass: replace chunk-level [ref-chunk] with badges
    def _chunk_badge(m: re.Match) -> str:
        ref_id = m.group(1)  # already escaped
        chunk_idx = m.group(2)  # already escaped
        return (
            f'<span class="citation-badge" data-ref="{ref_id}" '
            f'data-chunk="{chunk_idx}" onclick="filterSource(this)">'
            f"[{ref_id}-{chunk_idx}]</span>"
        )

    result = CITATION_PATTERN.sub(_chunk_badge, escaped)

    # Second pass: replace doc-level [n] with badges
    def _doc_badge(m: re.Match) -> str:
        ref_id = m.group(1)  # already escaped
        return (
            f'<span class="citation-badge" data-ref="{ref_id}" '
            f'onclick="filterSource(this)">'
            f"[{ref_id}]</span>"
        )

    result = DOC_CITATION_PATTERN.sub(_doc_badge, result)
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
