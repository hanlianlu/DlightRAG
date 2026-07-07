r"""Markdown-to-HTML renderers for Web UI.

Uses markdown-it-py (GFM-like preset) for Markdown/tables/lists and
Pygments for fenced code block syntax highlighting.  A custom inline
math rule recognises ``$...$`` and ``\(...\)`` as math tokens and a
block rule recognises ``$$...$$`` and ``\[...\]`` so that LaTeX
survives markdown processing intact for client-side MathJax v4.

Two renderers are provided:
- ``render_markdown``: For answer content (``html: False`` — escapes raw HTML).
- ``render_chunk_content``: For source chunks (``html: True`` — allows HTML
  passthrough for tables from LightRAG parsers).
"""

import html as _html

from markdown_it import MarkdownIt
from markdown_it.rules_inline import StateInline
from pygments import highlight as pygments_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

_FORMATTER = HtmlFormatter(nowrap=True)

# ---------------------------------------------------------------------------
# Custom inline math rule — recognises $...$ and \(...\) as math tokens
# so markdown-it-py never tries to interpret underscores etc. inside them.
# ---------------------------------------------------------------------------


def _math_inline_rule(state: StateInline, silent: bool) -> bool:
    r"""Match inline ``$…$``, ``\(…\)`` and display ``$$…$$``, ``\[…\]``.

    Emits a ``math_inline`` token whose content is the inner LaTeX.
    The renderer re-wraps it with the correct delimiters so MathJax
    can pick it up client-side.

    Returns ``True`` on a successful match, advancing ``state.pos`` past
    the closing delimiter.  When ``silent`` the parser only validates
    without emitting tokens (used for emphasis/delimiter resolution).
    """
    pos = state.pos
    src = state.src

    # --- \(...\) (inline) --------------------------------------------------
    if src[pos : pos + 2] == "\\(":
        end = src.find("\\)", pos + 2)
        if end == -1:
            return False
        if not silent:
            token = state.push("math_inline", "", 0)
            token.content = src[pos + 2 : end]
            token.markup = "\\("
        state.pos = end + 2
        return True

    # --- \[...\] (display) -------------------------------------------------
    if src[pos : pos + 2] == "\\[":
        end = src.find("\\]", pos + 2)
        if end == -1:
            return False
        if not silent:
            token = state.push("math_inline", "", 0)
            token.content = src[pos + 2 : end]
            token.markup = "\\["
        state.pos = end + 2
        return True

    # --- $...$ or $$...$$ --------------------------------------------------
    if src[pos] != "$":
        return False

    # \$ is escaped — let the escape rule handle it
    if pos > 0 and src[pos - 1] == "\\":
        return False

    # Display math $$...$$ (crosses lines)
    if pos + 1 < state.posMax and src[pos + 1] == "$":
        if pos + 2 >= state.posMax:
            return False
        end = src.find("$$", pos + 2)
        if end == -1:
            return False
        if not silent:
            token = state.push("math_inline", "", 0)
            token.content = src[pos + 2 : end]
            token.markup = "$$"
        state.pos = end + 2
        return True

    # Inline math $...$ (single line)
    if pos + 1 >= state.posMax:
        return False
    nxt = src[pos + 1]
    if nxt.isspace() or nxt.isdigit() or nxt == "$":
        return False

    end = src.find("$", pos + 1)
    if end == -1:
        return False
    if "\n" in src[pos + 1 : end]:
        return False
    if end > pos + 1 and src[end - 1].isspace():
        return False

    if not silent:
        token = state.push("math_inline", "", 0)
        token.content = src[pos + 1 : end]

    state.pos = end + 1
    return True


def _render_math_inline(_renderer, tokens: list, idx: int, _options, _env) -> str:
    """Re-wrap math content with its original delimiters for MathJax.

    The content is HTML-escaped so the downstream sanitizer (nh3) is not the
    sole barrier against markup smuggled between math delimiters; MathJax reads
    the decoded text content, so escaping does not affect rendering.
    """
    token = tokens[idx]
    content = _html.escape(token.content, quote=False)
    markup = getattr(token, "markup", "$")
    if markup == "$$":
        return f"$${content}$$"
    if markup == "\\[":
        return f"\\[{content}\\]"
    if markup == "\\(":
        return f"\\({content}\\)"
    return f"${content}$"


# ---------------------------------------------------------------------------
# Code highlighting callback
# ---------------------------------------------------------------------------


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
        return "<pre><code>" + _html.escape(code) + "</code></pre>"
    highlighted = pygments_highlight(code, lexer, _FORMATTER)
    return f'<pre class="highlight"><code>{highlighted}</code></pre>'


# ---------------------------------------------------------------------------
# Shared markdown-it-py instances
# ---------------------------------------------------------------------------

_md_opts_answer = {
    "html": False,
    "highlight": _highlight_fn,
}


def _make_md() -> MarkdownIt:
    """Create a fresh markdown-it-py instance with the math inline rule."""
    md = MarkdownIt("gfm-like", _md_opts_answer).disable("linkify")
    # Insert BEFORE the escape rule so \$ still works for literal dollars
    md.inline.ruler.before("escape", "math_inline", _math_inline_rule)
    md.add_render_rule("math_inline", _render_math_inline)
    return md


_md = _make_md()
_md_chunk = MarkdownIt("gfm-like", {"html": True, "highlight": _highlight_fn}).disable("linkify")
# Also protect math in chunk content
_md_chunk.inline.ruler.before("escape", "math_inline", _math_inline_rule)
_md_chunk.add_render_rule("math_inline", _render_math_inline)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_markdown(text: str) -> str:
    r"""Convert Markdown text to HTML with syntax-highlighted code blocks.

    Inline math (``$...$``, ``\(...\)``) and display math
    (``$$...$$``, ``\[...\]``) are passed through verbatim for
    client-side MathJax rendering.
    """
    return _md.render(text)


def render_chunk_content(text: str) -> str:
    """Render chunk content to HTML, allowing HTML passthrough for tables etc."""
    return _md_chunk.render(text)
