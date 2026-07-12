"""FastAPI dependency injection and Jinja2 environment for web routes."""

import re
from pathlib import Path
from typing import Any

import nh3
from fastapi import Cookie, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from dlightrag.access_control import AccessDeniedError, access_control_from_config
from dlightrag.app_state import request_config
from dlightrag.citations.parser import CITATION_PATTERN
from dlightrag.core.scope import RequestScope
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


def _reference_label(ref_id: Any, chunk_idx: Any | None = None) -> str:
    """Return the compact visible label for citation/reference surfaces."""
    ref = str(ref_id)
    if chunk_idx is None or chunk_idx == "":
        return ref
    return f"{ref}-{chunk_idx}"


def _reference_aria_label(ref_id: Any, chunk_idx: Any | None = None) -> str:
    """Return the accessible label for citation/reference controls."""
    ref = str(ref_id)
    if chunk_idx is None or chunk_idx == "":
        return f"Source {ref}"
    return f"Source {ref}, chunk {chunk_idx}"


def _citation_badges(text: str) -> Markup:
    """Replace citation patterns with clickable badge HTML.

    Handles both [ref_id-chunk_idx] and [n] doc-level formats.
    Renders Markdown first, then injects badges while protecting code blocks.

    Source data is rendered separately in the hidden source panel; the answer
    body should only contain validated inline citation markers.
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
        label = _reference_label(ref_id, chunk_idx)
        aria_label = _reference_aria_label(ref_id, chunk_idx)
        return (
            f'<span class="citation-badge" data-ref="{ref_id}" '
            f'data-chunk="{chunk_idx}" data-action="filter-source" '
            f'role="button" tabindex="0" '
            f'aria-label="{aria_label}">'
            f"{label}</span>"
        )

    result = CITATION_PATTERN.sub(_chunk_badge, html)

    # Second pass: replace doc-level [n] with badges
    def _doc_badge(m: re.Match) -> str:
        ref_id = m.group(1)
        label = _reference_label(ref_id)
        aria_label = _reference_aria_label(ref_id)
        return (
            f'<span class="citation-badge" data-ref="{ref_id}" '
            f'data-action="filter-source" role="button" tabindex="0" '
            f'aria-label="{aria_label}">{label}</span>'
        )

    result = DOC_CITATION_PATTERN.sub(_doc_badge, result)

    # Restore code blocks
    result = _restore_code_blocks(result, protected)

    return Markup(result)  # noqa: S704 - markdown escapes raw HTML; final partial is nh3-cleaned


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

    return Markup(html)  # noqa: S704 - chunk HTML is nh3-cleaned before marking safe


def _basename(path: str) -> str:
    """Extract filename from a path string."""
    return Path(path).name


# Register filters on the Jinja2 environment
templates.env.filters["citation_badges"] = _citation_badges
templates.env.filters["highlight_content"] = _highlight_content
templates.env.filters["reference_label"] = _reference_label
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


def render_partial(name: str, **ctx: Any) -> str:
    """Render a Jinja2 partial template to string."""
    return templates.env.get_template(name).render(**ctx)


def error_response(message: str, status_code: int = 400) -> HTMLResponse:
    """Return an HTML error fragment with the given status code."""
    return HTMLResponse(
        render_partial("partials/error.html", message=message),
        status_code=status_code,
    )


def get_workspace(dlightrag_workspace: str = Cookie(default=DEFAULT_WORKSPACE)) -> str:
    """Read current workspace from cookie, normalized to safe PG identifier."""
    from dlightrag.utils import normalize_workspace

    return normalize_workspace(dlightrag_workspace)


def get_manager(request: Request):
    """Get RAGServiceManager from app state."""
    return request.app.state.manager


def get_web_conversation_service(request: Request) -> Any:
    """Return the one app-scoped Web conversation service."""
    return request.app.state.web_conversation_service


def get_request_scope(
    request: Request,
    workspaces: list[str] | tuple[str, ...] | None = None,
) -> RequestScope:
    """Return the authenticated request scope for browser routes."""
    user = getattr(request.state, "user_context", None)
    return RequestScope.from_user(user).for_workspaces(workspaces)


async def enforce_web_access(request: Request, action: str, workspace: str | None) -> None:
    user = getattr(request.state, "user_context", None)
    access_control = getattr(
        request.app.state, "access_control", None
    ) or access_control_from_config(request_config(request))
    try:
        await access_control.check(user, action, workspace=workspace)
    except AccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None


async def filter_web_workspace_records(
    request: Request,
    action: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    user = getattr(request.state, "user_context", None)
    workspaces = [str(row["workspace"]) for row in records]
    access_control = getattr(
        request.app.state, "access_control", None
    ) or access_control_from_config(request_config(request))
    allowed = set(await access_control.filter_workspaces(user, action, workspaces))
    return [row for row in records if str(row["workspace"]) in allowed]
