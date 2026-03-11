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
    """Replace [ref_id-chunk_idx] patterns with clickable citation badge HTML.

    Input text is auto-escaped by Jinja2 before this filter runs when used
    as ``{{ answer | citation_badges }}``. We split on the citation pattern,
    so non-citation segments are already escaped and citation groups are safe
    alphanumeric/digit strings.
    """
    parts = CITATION_PATTERN.split(str(text))
    # split gives: [text, ref_id, chunk_idx, text, ref_id, chunk_idx, ...]
    result: list[str] = []
    i = 0
    while i < len(parts):
        if i % 3 == 0:
            # Text segment — already escaped by Jinja2 auto-escape
            result.append(Markup.escape(parts[i]))
        else:
            ref_id = Markup.escape(parts[i])
            chunk_idx = Markup.escape(parts[i + 1])
            result.append(
                Markup(
                    '<span class="citation-badge" data-ref="{}" '
                    'data-chunk="{}" onclick="filterSource(this)">'
                    "[{}-{}]</span>"
                ).format(ref_id, chunk_idx, ref_id, chunk_idx)
            )
            i += 1  # Skip chunk_idx, already consumed
        i += 1
    return Markup("".join(result))


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
