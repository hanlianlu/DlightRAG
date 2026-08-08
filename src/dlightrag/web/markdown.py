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
import re
from collections.abc import Sequence

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

    Returns highlighted HTML if language is known, a marked escaped source
    block for Mermaid (so the client can lazily upgrade it to a diagram), a
    plain ``<pre><code>`` block for other unknown languages, or empty string
    (no lang) to fall back to the default ``<pre><code>`` wrapper.
    """
    if not lang:
        return ""
    if lang.lower() == "mermaid":
        # Mermaid has no Pygments lexer. Emit a marked, escaped source block:
        # the client renders it to an SVG when possible and it degrades to
        # readable source otherwise. The class/data-* marker survives nh3.
        return (
            '<pre class="mermaid-source" data-lang="mermaid"><code>'
            + _html.escape(code)
            + "</code></pre>"
        )
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
    return _md_chunk.render(separate_html_blocks(text))


# CommonMark closes an HTML block only at a blank line, so a parser that emits a
# whole table on one line takes the rest of the chunk down with it. Only the tags
# that open such a block are listed: an unknown tag never starts one.
_HTML_BLOCK_TAGS = "blockquote|div|dl|figure|footer|header|main|nav|ol|p|pre|section|table|ul"
_ONE_LINE_HTML_BLOCK = re.compile(
    rf"^\s*<(?P<tag>{_HTML_BLOCK_TAGS})\b[^\n]*</(?P=tag)>\s*$", re.IGNORECASE
)


def separate_html_blocks(text: str) -> str:
    """End a one-line HTML block with a blank line so the Markdown after it renders.

    Callers that align highlights against the source must pass the same result,
    since the inserted lines shift every offset after them.
    """
    lines = text.split("\n")
    out: list[str] = []
    for index, line in enumerate(lines):
        out.append(line)
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if following.strip() and _ONE_LINE_HTML_BLOCK.match(line):
            out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Semantic highlight injection
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]*>")
_ENTITY_RE = re.compile(r"&(?:#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);")
# Source spans that never reach the rendered text: passthrough HTML tags and
# Markdown link destinations. Skipping them stops the alignment walk from
# anchoring on markup that happens to share a prefix with the visible text.
_SOURCE_MARKUP_RE = re.compile(r"<[^<>]*>|\]\([^()]*\)")
_HIGHLIGHT_OPEN = '<span class="highlight">'
_HIGHLIGHT_CLOSE = "</span>"


def _visible_text(html: str) -> tuple[str, list[tuple[int, int]]]:
    """Return the visible text of ``html`` and the HTML slice backing each char."""
    chars: list[str] = []
    spans: list[tuple[int, int]] = []

    def scan(start: int, end: int) -> None:
        pos = start
        while pos < end:
            entity = _ENTITY_RE.match(html, pos, end)
            if entity is None:
                chars.append(html[pos])
                spans.append((pos, pos + 1))
                pos += 1
                continue
            decoded = _html.unescape(entity.group(0))
            chars.extend(decoded)
            spans.extend([(pos, entity.end())] * len(decoded))
            pos = entity.end()

    cursor = 0
    for tag in _TAG_RE.finditer(html):
        scan(cursor, tag.start())
        cursor = tag.end()
    scan(cursor, len(html))
    return "".join(chars), spans


def _align_source_to_visible(source: str, visible: str) -> list[int | None]:
    """Map each source character to its index in the rendered visible text.

    Rendering only deletes source characters (Markdown syntax, tags, link
    targets) and inserts layout whitespace, so a single monotone walk recovers
    the correspondence; deleted characters map to ``None``.
    """
    markup = bytearray(len(source))
    for match in _SOURCE_MARKUP_RE.finditer(source):
        markup[match.start() : match.end()] = b"\x01" * (match.end() - match.start())

    mapping: list[int | None] = [None] * len(source)
    i = j = 0
    while i < len(source) and j < len(visible):
        if markup[i]:
            i += 1
        elif source[i] == visible[j] or (source[i].isspace() and visible[j].isspace()):
            mapping[i] = j
            i += 1
            j += 1
        elif visible[j].isspace():
            j += 1
        else:
            i += 1
    return mapping


def _text_runs(spans: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    """Split a visible-text range into contiguous HTML slices (one per text node)."""
    runs: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    previous: tuple[int, int] | None = None
    for index in range(start, end):
        span = spans[index]
        if span == previous:
            continue
        if current is not None and span[0] == current[1]:
            current = (current[0], span[1])
        else:
            if current is not None:
                runs.append(current)
            current = span
        previous = span
    if current is not None:
        runs.append(current)
    return runs


def inject_highlights(html: str, source: str, phrases: Sequence[str]) -> str:
    """Wrap each phrase of ``source`` in ``<span class="highlight">`` inside ``html``.

    Phrases are verbatim substrings of ``source`` (guaranteed by highlight
    validation), so they are anchored by position rather than re-matched against
    the rendered text.
    """
    visible, spans = _visible_text(html)
    if not visible:
        return html
    mapping = _align_source_to_visible(source, visible)

    matched: list[tuple[int, int]] = []
    for phrase in phrases:
        found = source.find(str(phrase))
        if found < 0:
            continue
        indices = [v for v in mapping[found : found + len(str(phrase))] if v is not None]
        if not indices:
            continue
        start, end = indices[0], indices[-1] + 1
        if any(start < other_end and other_start < end for other_start, other_end in matched):
            continue
        matched.append((start, end))

    runs = [run for start, end in matched for run in _text_runs(spans, start, end)]
    # Whitespace-only runs are the gaps between block tags (table cells, list
    # items); wrapping them would place a span where no text node exists.
    runs = sorted(run for run in runs if html[run[0] : run[1]].strip())
    if not runs:
        return html

    out: list[str] = []
    cursor = 0
    for start, end in runs:
        out.append(html[cursor:start])
        out.append(_HIGHLIGHT_OPEN)
        out.append(html[start:end])
        out.append(_HIGHLIGHT_CLOSE)
        cursor = end
    out.append(html[cursor:])
    return "".join(out)
