"""Markdown-to-HTML renderer for Web UI answers.

Uses markdown-it-py (GFM-like preset) for Markdown/tables/lists and
Pygments for fenced code block syntax highlighting. LaTeX $...$ passes
through as literal text for client-side KaTeX rendering.
"""

from __future__ import annotations

import html as _html

from markdown_it import MarkdownIt
from pygments import highlight as pygments_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

_FORMATTER = HtmlFormatter(cssclass="highlight", wrapcode=True)


def _highlight_fn(code: str, lang: str, _attrs: str) -> str:
    """Pygments highlight callback for markdown-it-py fenced code blocks.

    Returns highlighted HTML if language is known, a plain ``<pre><code>``
    block for unknown languages, or empty string (no lang) to fall back to
    the default ``<pre><code>`` wrapper.
    """
    if not lang:
        return ""
    try:
        lexer = get_lexer_by_name(lang)
    except ClassNotFound:
        # Return plain pre/code so the renderer does not add a language class
        return "<pre><code>" + _html.escape(code) + "</code></pre>"
    return pygments_highlight(code, lexer, _FORMATTER)


_md = MarkdownIt("gfm-like", {"html": False, "highlight": _highlight_fn}).disable("linkify")


def render_markdown(text: str) -> str:
    """Convert Markdown text to HTML with syntax-highlighted code blocks."""
    return _md.render(text)
