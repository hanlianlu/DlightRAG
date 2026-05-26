r"""Markdown-to-HTML renderers for Web UI.

Uses markdown-it-py (GFM-like preset) for Markdown/tables/lists and
Pygments for fenced code block syntax highlighting. LaTeX math blocks
($$...$$, $...$, \[...\], \(...\)) are protected from markdown processing
so underscores and other LaTeX syntax survive intact for client-side MathJax.

Two renderers are provided:
- ``render_markdown``: For answer content (``html: False`` — escapes raw HTML).
- ``render_chunk_content``: For source chunks (``html: True`` — allows HTML
  passthrough for tables from LightRAG parsers).
"""

from __future__ import annotations

import html as _html
import re

from markdown_it import MarkdownIt
from pygments import highlight as pygments_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

_FORMATTER = HtmlFormatter(nowrap=True)

# ---------------------------------------------------------------------------
# Math-block protection — prevent markdown-it-py from corrupting LaTeX
# (e.g., underscores in $$x_{1}$$ becoming <em> tags).
# ---------------------------------------------------------------------------

_MATH_DISPLAY_RE = re.compile(r"(\$\$|\\\[)(.+?)(\$\$|\\\])", re.DOTALL)
_MATH_INLINE_RE = re.compile(r"\\\((.+?)\\\)|(?<!\$)\$(?![\s\d$])(.+?)(?<![\s])\$(?!\d)")


_MATH_PLACEHOLDER_RE = re.compile(r"⧸MATH(\d+)⧸")


def _protect_math(text: str) -> tuple[str, list[str]]:
    """Replace math blocks with indexed placeholders."""
    protected: list[str] = []

    def _replace(m: re.Match) -> str:
        idx = len(protected)
        protected.append(m.group(0))
        return f"⧸MATH{idx}⧸"

    text = _MATH_DISPLAY_RE.sub(_replace, text)
    text = _MATH_INLINE_RE.sub(_replace, text)
    return text, protected


def _restore_math(html: str, protected: list[str]) -> str:
    """Restore math blocks from placeholders."""
    for idx, original in enumerate(protected):
        html = html.replace(f"⧸MATH{idx}⧸", original)
    return html


# ---------------------------------------------------------------------------
# Code highlighting
# ---------------------------------------------------------------------------


def _highlight_fn(code: str, lang: str, _attrs: str) -> str:
    """Pygments highlight callback for markdown-it-py fenced code blocks.

    Returns highlighted HTML if language is known, a plain ``<pre><code>``
    block for unknown languages, or empty string (no lang) to fall back to
    the default ``<pre><code>`` wrapper.

    markdown-it-py uses the returned string as-is only when it starts with
    ``<pre``.  Using ``nowrap=True`` in the formatter means Pygments emits
    bare spans without its own ``<div>``/``<pre>`` wrapper, so we can add our
    own ``<pre>`` here and avoid double-wrapping.
    """
    if not lang:
        return ""
    try:
        lexer = get_lexer_by_name(lang)
    except ClassNotFound:
        # Return plain pre/code so the renderer does not add a language class
        return "<pre><code>" + _html.escape(code) + "</code></pre>"
    highlighted = pygments_highlight(code, lexer, _FORMATTER)
    # Wrap in <pre> so markdown-it uses it as-is (starts with <pre)
    return f'<pre class="highlight"><code>{highlighted}</code></pre>'


_md = MarkdownIt("gfm-like", {"html": False, "highlight": _highlight_fn}).disable("linkify")

_md_chunk = MarkdownIt("gfm-like", {"html": True, "highlight": _highlight_fn}).disable("linkify")


def render_markdown(text: str) -> str:
    """Convert Markdown text to HTML with syntax-highlighted code blocks."""
    text, math_blocks = _protect_math(text)
    html = _md.render(text)
    return _restore_math(html, math_blocks)


def render_chunk_content(text: str) -> str:
    """Render chunk content to HTML, allowing HTML passthrough for tables etc."""
    text, math_blocks = _protect_math(text)
    html = _md_chunk.render(text)
    return _restore_math(html, math_blocks)
