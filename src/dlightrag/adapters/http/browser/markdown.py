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

from linkify_it import LinkifyIt
from markdown_it import MarkdownIt
from markdown_it.rules_inline import StateInline
from pygments import highlight as pygments_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

_FORMATTER = HtmlFormatter(nowrap=True)

# CJK punctuation ends an autolinked address; CJK letters may belong to one of
# its components. The fullwidth block holds both kinds, so it is split: its
# punctuation forms U+FF01-FF0F, FF1A-FF20, FF3B-FF40, FF5B-FF65 end an address
# while halfwidth Katakana (FF66-FF9F, including the voiced and semi-voiced
# sound marks that only modify the kana before them) and halfwidth Hangul
# (FFA0-FFDC) are letters a path may legitimately end with.
_CJK_PUNCTUATION = (
    "[\u3000-\u303f\ufe30-\ufe4f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65]"
)
_CJK_LETTERS = (
    "[\u1100-\u11ff\u2e80-\u2eff\u3040-\u30ff\u3130-\u318f\u31f0-\u31ff"
    "\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff66-\uff9f\uffa0-\uffdc]"
)
# Characters that open a new component of an address: a CJK run that follows one
# is a path, query, or host label; a run that follows a letter continues a
# segment, which is where the address's meaning becomes a guess. A dot opens a
# host label but nothing in a path — `www.例子.com` is a name, while
# `…/report.中文` is a sentence — so it is admitted only inside the authority.
_COMPONENT_OPENERS = "/=&?#"
_HOST_OPENERS = _COMPONENT_OPENERS + "."
# A character that would make an address a continuation of a preceding token:
# `blob:https://…`, `data:…`, or `xhttps://…` are not addresses of their own.
_ATTACHED_TO_PREFIX = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._+-:@"
)
# The scheme linkify matched in front of the tail it validates.
_SCHEME_END = re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*:$")

# ---------------------------------------------------------------------------
# Custom inline math rule — recognises $...$ and \(...\) as math tokens
# so markdown-it-py never tries to interpret underscores etc. inside them.
# ---------------------------------------------------------------------------


def _address_start(text: str, position: int) -> int:
    """Return where the address begins, given linkify's position of its ``//``.

    linkify validates one schema tail starting at the ``//`` and composes the
    address from the scheme it matched in front of it, so the scheme has to be
    read back out of the text to see what precedes the address at all.
    """
    scheme = _SCHEME_END.search(text, 0, position)
    return position - len(scheme.group(0)) if scheme else position


def _bounded_address(text: str, position: int, length: int) -> int:
    """Return how much of one validated http(s) tail is the address to follow.

    A Chinese sentence is usually written without spaces, so this is where an
    address stops being one:

    - CJK punctuation always ends it, so `参考 https://example.com/x。` does not
      link the sentence's own full stop.
    - A CJK letter run that opens a component (`…/wiki/中文条目`, `…?q=中文`,
      `www.例子.com`) belongs to the address.
    - A CJK letter run that continues a segment (`…/report即可`, even when a query
      follows) leaves the address ambiguous. Ambiguity returns `0`: the text
      stays text, because a wrong link is worse than a plain one.
    - An address that continues a preceding token (`blob:https://…`) is not an
      address of its own.

    The returned length is measured from `position`, as the schema contract
    requires, and `0` means "do not autolink this".
    """
    start = _address_start(text, position)
    if start and text[start - 1] in _ATTACHED_TO_PREFIX:
        return 0
    address = text[start : position + length]
    punctuation = re.search(_CJK_PUNCTUATION, address)
    if punctuation:
        address = address[: punctuation.start()]
    # The authority ends where the path, query, or fragment begins.
    authority_end = len(address)
    for boundary in "/?#":
        found = address.find(boundary, position - start + 2)
        if found != -1:
            authority_end = min(authority_end, found)
    for run in re.finditer(_CJK_LETTERS + "+", address):
        if not run.start():
            continue
        openers = _HOST_OPENERS if run.start() < authority_end else _COMPONENT_OPENERS
        if address[run.start() - 1] not in openers:
            return 0
    return start + len(address.rstrip("*")) - position


class _AddressBoundary:
    """Bound the schema's own match at the characters a sentence owns.

    Lengths come from a stock linkifier asked through its documented
    ``test_schema_at``, so the http matcher itself stays the dependency's and this
    module never reaches for its internals. The trailing ``*`` is trimmed here
    rather than left to markdown-it's inline pass, which trims it after the
    validator and would otherwise bound the same address differently from the
    core pass (ADR 0026).
    """

    def __init__(self, reference: LinkifyIt) -> None:
        self._reference = reference

    def validate(self, text: str, pos: int) -> int:
        length = self._reference.test_schema_at(text, "http:", pos)
        return _bounded_address(text, pos, length) if length else 0


def _configure_autolinking(md: MarkdownIt) -> None:
    """Turn on address autolinking for Model-written text, and bound it.

    Only an address that names its own scheme qualifies. linkify's fuzzy matching
    also guesses from a top-level domain, and this product's own vocabulary is
    full of words that end in a real one — `report.md`, `build.sh`, `clip.mov`,
    `archive.zip` — each of which would become `http://report.md`. Its other
    built-in schemas are refused for the same reason they are useless here: the
    fragment sanitiser admits only http, https, data, and blob, so a
    `mailto:`, `ftp:`, or protocol-relative href would be stripped and leave an
    inert link styled like a working one (ADR 0026).

    The boundary is applied through linkify's own schema-validation seam, so the
    match the library builds is the match this renderer means. markdown-it-py
    autolinks through two passes over that one instance — an inline rule at `://`
    and a core rule over finished text tokens — and both therefore bound an
    address identically. They are not interchangeable: the core pass needs an
    address to start after whitespace or punctuation, so a Chinese answer's
    `见https://example.com/report`, where the address touches the sentence with no
    space at all, is only reachable by the inline pass. Keeping both means the
    inline pass cannot see what precedes the scheme it matched, which is the one
    accepted edge: `blob:https://example.com/x` links its http(s) part
    (ADR 0026).
    """
    linkify = md.linkify
    if linkify is None:
        raise RuntimeError("the gfm-like preset must provide a linkify instance")
    linkify.set({"fuzzy_link": False, "fuzzy_email": False})
    boundary = _AddressBoundary(LinkifyIt())
    linkify.add("http:", {"validate": boundary.validate})
    linkify.add("https:", "http:")
    for disabled in ("//", "mailto:", "ftp:"):
        linkify.add(disabled, None)


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


def _render_link_open(renderer, tokens: list, idx: int, options, env) -> str:
    """Open external links in a new tab.

    An answer or an Artifact may cite a public URL directly (and a published
    file's projected citations are plain Markdown links). Following one in place
    would navigate the running application away from itself.
    """
    token = tokens[idx]
    href = str(token.attrGet("href") or "")
    if href.startswith(("http://", "https://")):
        token.attrSet("target", "_blank")
    return renderer.renderToken(tokens, idx, options, env)


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
    md = MarkdownIt("gfm-like", _md_opts_answer)
    _configure_autolinking(md)
    # Insert BEFORE the escape rule so \$ still works for literal dollars
    md.inline.ruler.before("escape", "math_inline", _math_inline_rule)
    md.add_render_rule("math_inline", _render_math_inline)
    md.add_render_rule("link_open", _render_link_open)
    return md


_md = _make_md()
# Source chunks quote a document, so their text keeps the shape the parser
# produced: a bare address inside a quotation is not this product's link to make.
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
    return _md_chunk.render(normalize_chunk_source(text))


# CommonMark closes an HTML block only at a blank line, so a parser that emits a
# whole table on one line takes the rest of the chunk down with it. Only the tags
# that open such a block are listed: an unknown tag never starts one.
_HTML_BLOCK_TAGS = "blockquote|div|dl|figure|footer|header|main|nav|ol|p|pre|section|table|ul"
_ONE_LINE_HTML_BLOCK = re.compile(
    rf"^\s*<(?P<tag>{_HTML_BLOCK_TAGS})\b[^\n]*</(?P=tag)>\s*$", re.IGNORECASE
)


def separate_html_blocks(text: str) -> str:
    """End a one-line HTML block with a blank line so the Markdown after it renders."""
    lines = text.split("\n")
    out: list[str] = []
    for index, line in enumerate(lines):
        out.append(line)
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if following.strip() and _ONE_LINE_HTML_BLOCK.match(line):
            out.append("")
    return "\n".join(out)


# A parser that numbers items as **1.** states the structure in an inline mark,
# which leaves every item merged into one paragraph. Bullets, ordered markers and
# headings need no help: each already interrupts a paragraph on its own.
_BOLD_ITEM_START = re.compile(r"^\s*\*\*\S")
_HARD_BREAK = "  "


def break_before_bold_items(text: str) -> str:
    """Keep a line that opens with bold on a line of its own."""
    lines = text.split("\n")
    for index in range(len(lines) - 1):
        line = lines[index]
        if not line.strip() or line.endswith(_HARD_BREAK):
            continue
        if _BOLD_ITEM_START.match(lines[index + 1]):
            lines[index] = line + _HARD_BREAK
    return "\n".join(lines)


def normalize_chunk_source(text: str) -> str:
    """Recover the line structure a parser meant but could not express in Markdown.

    Callers that align highlights against the source must pass the same result,
    since the edits shift every offset after them.
    """
    return break_before_bold_items(separate_html_blocks(text))


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

# MathJax pairs its delimiters within one text node, so a highlight that stops in
# the middle of a formula silently costs the reader the formula. A candidate that
# already holds markup is past saving and is deliberately left unmatched.
_MATH_REGION_RE = re.compile(
    r"\$\$[^<]+?\$\$|\\\[[^<]+?\\\]|\\\([^<]+?\\\)|\$[^$<\n]+?\$", re.DOTALL
)


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


def _widen_over_math(html: str, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Grow a run that reaches into a formula until it holds the whole formula."""
    regions = [region.span() for region in _MATH_REGION_RE.finditer(html)]
    if not regions:
        return runs

    widened: list[tuple[int, int]] = []
    for start, end in runs:
        for region_start, region_end in regions:
            if start < region_end and region_start < end:
                start, end = min(start, region_start), max(end, region_end)
        widened.append((start, end))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(widened):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


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
    runs = _widen_over_math(html, sorted(run for run in runs if html[run[0] : run[1]].strip()))
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
