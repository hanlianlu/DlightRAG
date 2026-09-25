# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Markdown grammar of model-written answers, independent of HTML delivery.

Both page collection and browser rendering use this factory. Link targets come
from link tokens, never from scanning source text or reconstructing code spans.
HTML rendering, highlighting and sanitizing remain the browser adapter's work.
"""

import re

from linkify_it import LinkifyIt
from markdown_it import MarkdownIt
from markdown_it.rules_inline import StateInline

from dlightrag.engine.answer.citations.syntax import install_citation_syntax
from dlightrag.engine.answer.markdown_source import install_source_mapping

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


def math_inline_rule(state: StateInline, silent: bool) -> bool:
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


def answer_markdown() -> MarkdownIt:
    """Create the answer grammar; callers may attach rendering rules, not syntax."""
    md = MarkdownIt("gfm-like", {"html": False})
    _configure_autolinking(md)
    md.inline.ruler.before("escape", "math_inline", math_inline_rule)
    install_source_mapping(md)
    install_citation_syntax(md)
    return md


_LINK_PARSER = answer_markdown()


def link_targets(answer: str) -> list[str]:
    """Distinct HTTP(S) link destinations in reading order, excluding image alt text.

    Code, math, titles and unused reference definitions do not emit link_open
    tokens. Repeated destinations are read once, even though every actual link
    occurrence can be presented as a card.
    """
    targets: dict[str, None] = {}
    for block in _LINK_PARSER.parse(answer):
        if block.type != "inline":
            continue
        for token in block.children or ():
            if token.type == "link_open":
                href = str(token.attrGet("href") or "")
                if href.startswith(("http://", "https://")):
                    targets.setdefault(href, None)
    return list(targets)
