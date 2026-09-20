# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Video link cards: a link a page declares as a video, read from its own metadata.

The Model writes an address; the page it points at decides whether it is a video
by publishing Open Graph metadata, and the card carries only what the page said.
A page that answers with anything else — no declaration, no answer at all, a
network this deployment cannot reach — leaves the link exactly as the Model wrote
it. Nothing here is a site list: YouTube (`og:video:url`) and Bilibili
(`og:video` to its own player) declare it in the same shape, and an article
(`og:type: website`) declares that it is not one.

Cards are bounded on every axis: how many are read per answer, how many bytes each
page may return, how long the whole read may take including waiting for the shared
network admission, and how much of a page is parsed. ADR 0026 records the
decision, including that the card is a link out rather than an embed.
"""

from __future__ import annotations

import asyncio
import html as _html
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from markdown_it import MarkdownIt

from dlightrag.engine.public_http import fetch_public_http, validate_public_web_url

#: How many pages one answer may ask about. The reader gets a card or two, and an
#: answer with a dozen links does not become a dozen outbound reads.
MAX_CARDS = 3
_MAX_PAGE_BYTES = 256 * 1024
#: One page gets a short deadline, and the reads run together, so an answer with
#: three links costs about one deadline rather than three. The deadline covers
#: admission and reading, not reading alone: the shared slot a fetch waits for is
#: part of what the Answer's settlement is paying for.
_PAGE_TIMEOUT_SECONDS = 4.0
_MAX_TITLE = 200
_MAX_DESCRIPTION = 400

# One address as it appears in answer Markdown. Sentence punctuation a reader owns
# is not part of it, and neither is a trailing dot.
_ADDRESS = re.compile(r"https?://[^\s<>\"'`\u3000-\u303f\uff00-\uffef)\]},;]+")
# One Markdown link, whose destination is the address the card describes.
_MARKDOWN_LINK = re.compile(
    r"\[[^\]]*\]\(\s*<?(?P<url>https?://[^\s>)]+)>?(?:\s+[\"'][^\"\']*[\"\'])?\s*\)"
)
# A meta tag's attributes are read from a bounded window, and only the first few
# declarations are read at all: a page cannot make parsing cost more than that.
_MAX_META_TAGS = 64
_MAX_META_WINDOW = 8192
# Every part of an attribute is length-bounded, so a page cannot make attribute
# matching scan quadratically by never writing the `=` it looks for.
_ATTRIBUTE = re.compile(
    r"(?P<name>[a-zA-Z:_-]{1,64})\s*=\s*"
    r"(?P<value>\"[^\"]{0,4096}\"|'[^']{0,4096}'|[^\s\"'>]{1,4096})"
)
_BLOCK_PARSER = MarkdownIt("commonmark")
_ATTRIBUTE = re.compile(
    r"(?P<name>[a-zA-Z:_-]{1,64})\s*=\s*"
    r"(?P<value>\"[^\"]{0,4096}\"|'[^']{0,4096}'|[^\s\"'>]{1,4096})"
)
_FENCE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})", re.MULTILINE)
_INDENTED_CODE = re.compile(r"^(?: {4}|\t)\S", re.MULTILINE)
_INLINE_CODE = re.compile(r"(?P<ticks>`+)(?:(?!(?P=ticks)).)*(?P=ticks)", re.DOTALL)
# The Open Graph video types. `video.other` is the one YouTube and Bilibili use.
_VIDEO_TYPES = frozenset({"video.movie", "video.episode", "video.tv_show", "video.other"})
_HTML_MEDIA_TYPES = frozenset({"text/html", "application/xhtml+xml"})


# The parser normalizes every line break to one newline, so offsets count them
# the same way rather than assuming LF.
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


@dataclass(frozen=True, slots=True)
class LinkCard:
    """One link a page declared to be a video, described by that page."""

    url: str
    title: str
    description: str
    site: str
    image: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "site": self.site,
            "image": self.image,
        }


def _line_spans(
    answer: str, offsets: Sequence[int], kinds: frozenset[str]
) -> list[tuple[int, int]]:
    """Return the character spans of the blocks the parser reports as ``kinds``."""
    spans: list[tuple[int, int]] = []
    for token in _BLOCK_PARSER.parse(answer):
        if token.type in kinds and token.map:
            start, end = token.map
            spans.append((offsets[start], offsets[min(end, len(offsets) - 1)]))
    return [span for span in spans if span[0] < span[1]]


def _block_code_spans(answer: str, offsets: Sequence[int]) -> list[tuple[int, int]]:
    """Return the spans the renderer itself calls block code."""
    return _line_spans(answer, offsets, frozenset({"fence", "code_block"}))


def _normalized_span(interior: str) -> str:
    """Return a code span interior as the parser reports it.

    Line endings become spaces, and one space is stripped from each end when both
    ends have one and the content is not all spaces.
    """
    normalized = _LINE_BREAK.sub(" ", interior)
    if len(normalized) >= 2 and normalized.startswith(" ") and normalized.endswith(" "):
        if normalized.strip():
            normalized = normalized[1:-1]
    return normalized


def _locate_code_span(
    answer: str, start: int, end: int, markup: str, content: str
) -> tuple[int, int] | None:
    """Locate one inline code span the parser already recognized, in order.

    The structure is the parser's: it decided there is a ``code_inline`` child with
    this content and this markup. Two rules locate it, and both are needed: a pair
    of backtick runs counts only when its interior normalizes to exactly that
    content, and that pair has to be the only one in the region. Verification
    handles newlines and padding inside a genuine span; uniqueness handles the
    duplicate a link title can hold. Anything else excludes the region rather than
    protecting the wrong characters.
    """
    length = len(markup)
    candidates: list[tuple[int, int]] = []
    scan = start
    while scan < end:
        opening = answer.find("`", scan, end)
        if opening == -1:
            break
        if answer[opening : opening + length] != markup:
            scan = opening + 1
            continue
        closing = opening + length
        while closing < end:
            found = answer.find("`", closing, end)
            if found == -1:
                break
            run = 0
            while found + run < end and answer[found + run] == "`":
                run += 1
            if run == length:
                interior = answer[opening + length : found]
                if _normalized_span(interior) == content:
                    candidates.append((opening, found + length))
                break
            closing = found + run
        scan = opening + length
    return candidates[0] if len(candidates) == 1 else None


def _inline_code_spans(answer: str, offsets: Sequence[int]) -> list[tuple[int, int]]:
    """Return every inline code span, located from the parser's own token stream.

    This module never decides which backticks pair: the parser owns the pairing,
    and escapes, link titles, and paragraph boundaries are its decisions too.
    """
    spans: list[tuple[int, int]] = []
    for token in _BLOCK_PARSER.parse(answer):
        if token.type != "inline" or not token.map:
            continue
        region_start = offsets[token.map[0]]
        region_end = offsets[min(token.map[1], len(offsets) - 1)]
        cursor = region_start
        located_spans: list[tuple[int, int]] = []
        for child in token.children or ():
            if child.type != "code_inline":
                continue
            located = _locate_code_span(
                answer, cursor, region_end, child.markup or "`", child.content
            )
            # A span that cannot be located, or one that would sit before the
            # previous one, means this region's spans are not accounted for. Which
            # text is code cannot then be trusted, so the whole region is treated as
            # code: no address inside it is read or replaced.
            if located is None or located[0] < cursor:
                located_spans = [(region_start, region_end)]
                break
            located_spans.append(located)
            cursor = located[1]
        spans.extend(located_spans)
    return spans


def _line_offsets(answer: str) -> list[int]:
    """Return the offset each of the parser's lines starts at.

    Line numbers are the parser's, so every line-break form counts as one break
    and Python's wider `splitlines` rules never enter the arithmetic.
    """
    offsets = [0]
    for line_break in _LINE_BREAK.finditer(answer):
        offsets.append(line_break.end())
    if offsets[-1] != len(answer):
        offsets.append(len(answer))
    return offsets


def code_spans(answer: str) -> list[tuple[int, int]]:
    """Return the answer spans that quote code rather than write an address.

    Both halves are the renderer's own classification: block code comes from its
    block tokens, inline code from its inline children, located in the source. No
    Markdown rule about pairing, escaping, or paragraph boundaries is re-derived
    here.
    """
    offsets = _line_offsets(answer)
    return _block_code_spans(answer, offsets) + _inline_code_spans(answer, offsets)


def card_key(url: str) -> str:
    """Return the identity one address is matched by, without a trailing dot."""
    return url.rstrip(".")


@dataclass(frozen=True, slots=True)
class WrittenAddress:
    """One address an answer wrote, and the text a card would replace for it.

    ``start``/``end`` cover the whole Markdown link when the address is that
    link's destination, and just the address when it was written as text, so a
    card never leaves a link's brackets behind.
    """

    key: str
    start: int
    end: int


def written_addresses(answer: str) -> list[WrittenAddress]:
    """Return every address the answer actually wrote, in order.

    One recognizer serves both the card read and the Answer's parts: they cannot
    disagree about where an address ends, which link it belongs to, or whether it
    was quoted as code.
    """
    quoted = code_spans(answer)
    excluded = list(quoted)
    written: list[WrittenAddress] = []
    for match in _MARKDOWN_LINK.finditer(answer):
        # The destination is what has to be written text: a link whose label holds
        # code is still a link, while one whose destination lies inside a fence is
        # quoted text that merely looks like a link.
        if _in_spans(match.start(), excluded) or _overlaps(match.span("url"), excluded):
            continue
        excluded.append(match.span())
        written.append(
            WrittenAddress(card_key(match.group("url").strip()), match.start(), match.end())
        )
    for match in _ADDRESS.finditer(answer):
        if _in_spans(match.start(), excluded):
            continue
        key = card_key(match.group(0))
        written.append(WrittenAddress(key, match.start(), match.start() + len(key)))
    written.sort(key=lambda address: address.start)
    return written


def _in_spans(position: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


def _overlaps(span: tuple[int, int], spans: Sequence[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and other_start < end for other_start, other_end in spans)


def addresses_in(answer: str) -> list[str]:
    """Return the distinct public addresses one answer writes, in order."""
    seen: dict[str, None] = {}
    for address in written_addresses(answer):
        seen.setdefault(address.key, None)
    return list(seen)


def _attributes(tag: str) -> dict[str, str]:
    if "=" not in tag:
        return {}
    return {
        match.group("name").lower(): _html.unescape(match.group("value").strip("\"'"))
        for match in _ATTRIBUTE.finditer(tag)
    }


def _is_meta_tag(page: str, start: int) -> bool:
    """Whether the tag at ``start`` is a meta element rather than a longer name."""
    if page[start : start + 5].lower() != "<meta":
        return False
    after = page[start + 5 : start + 6]
    return not after or not (after.isalnum() or after in "_-:")


def _metadata(page: str) -> dict[str, str]:
    """Return the page's own Open Graph metadata, first declaration wins.

    The scan is one left-to-right walk with an explicit budget: a comment is
    skipped in one step wherever it appears outside a tag, so a page of comment
    openers cannot make parsing quadratic, and a tag's attributes are read from a
    bounded window so an unclosed tag cannot either.
    """
    values: dict[str, str] = {}
    position = 0
    read = 0
    while read < _MAX_META_TAGS:
        start = page.find("<", position)
        if start == -1:
            break
        if page.startswith("<!--", start):
            end = page.find("-->", start + 4)
            position = len(page) if end == -1 else end + 3
            continue
        end = page.find(">", start + 1)
        if end == -1:
            break
        if _is_meta_tag(page, start):
            read += 1
            window = page[start : min(end + 1, start + _MAX_META_WINDOW)]
            attributes = _attributes(window)
            key = (attributes.get("property") or attributes.get("name") or "").lower()
            content = attributes.get("content")
            if key.startswith("og:") and content and key not in values:
                values[key] = content
        position = end + 1
    return values


def _declares_video(values: dict[str, str]) -> bool:
    """Whether the page itself says it holds a video."""
    if any(key.startswith("og:video") for key in values):
        return True
    return values.get("og:type", "").strip().lower() in _VIDEO_TYPES


def _is_html(media_type: Any) -> bool:
    """Whether the response is an HTML document rather than something HTML-ish."""
    essence = str(media_type or "").split(";", 1)[0].strip().lower()
    return not essence or essence in _HTML_MEDIA_TYPES


def _card(url: str, values: dict[str, str]) -> LinkCard | None:
    """Build one card from a declaring page, or ``None`` when it said too little."""
    title = " ".join(values.get("og:title", "").split())[:_MAX_TITLE]
    if not title:
        return None
    image: str | None = None
    declared_image = values.get("og:image", "").strip()
    if declared_image:
        try:
            image = validate_public_web_url(declared_image)
        except ValueError:
            image = None
    return LinkCard(
        url=url,
        title=title,
        description=" ".join(values.get("og:description", "").split())[:_MAX_DESCRIPTION],
        site=" ".join(values.get("og:site_name", "").split())[:80],
        image=image,
    )


async def _read_page(
    address: str,
    *,
    fetch: Callable[..., Awaitable[Any]],
    deadline: float,
) -> Any:
    """Read one page within the whole deadline, admission included.

    ``fetch_public_http`` applies its own timeout after taking a shared network
    slot, so the deadline is applied here as well: an answer waits for a card for
    the deadline, not for the queue that serves it.
    """
    async with asyncio.timeout(deadline):
        return await fetch(
            address,
            max_bytes=_MAX_PAGE_BYTES,
            timeout=deadline,
            agent_url=True,
        )


async def collect_link_cards(
    answer: str,
    *,
    fetch: Callable[..., Awaitable[Any]] = fetch_public_http,
    limit: int = MAX_CARDS,
    deadline: float = _PAGE_TIMEOUT_SECONDS,
) -> tuple[LinkCard, ...]:
    """Read at most ``limit`` declared videos out of one answer's addresses.

    Every failure is silent by design: an address that cannot be read, answers too
    slowly, declares nothing, or answers with something that is not HTML stays an
    ordinary link in the Answer. An address that names credentials in its query is
    not read at all — the read is anonymous, as every Agent URL read is.

    An address the answer writes more than once is not read either: this runs before
    any surface exists to ask the renderer which occurrence it would link, so an
    address whose occurrences cannot be told apart is left alone rather than
    fetched on a guess.
    """
    written = written_addresses(answer)
    occurrences: dict[str, int] = {}
    for address in written:
        occurrences[address.key] = occurrences.get(address.key, 0) + 1
    addresses = [address.key for address in written if occurrences[address.key] == 1][
        : max(0, limit)
    ]
    if not addresses:
        return ()
    pages = await asyncio.gather(
        *(_read_page(address, fetch=fetch, deadline=deadline) for address in addresses),
        return_exceptions=True,
    )
    cards: list[LinkCard] = []
    for address, page in zip(addresses, pages, strict=True):
        if isinstance(page, BaseException):
            # One unreachable link never fails an answer: it stays a plain link.
            continue
        if not _is_html(getattr(page, "media_type", None)):
            continue
        values = _metadata(getattr(page, "content", b"").decode("utf-8", errors="replace"))
        if not _declares_video(values):
            continue
        card = _card(address, values)
        if card is not None:
            cards.append(card)
    return tuple(cards)


def project_link_cards(cards: Sequence[Any]) -> list[dict[str, str | None]]:
    """Project stored cards for one authenticated reader, dropping unsafe URLs."""
    projected: list[dict[str, str | None]] = []
    for item in cards:
        if not isinstance(item, dict):
            continue
        try:
            url = validate_public_web_url(str(item.get("url") or ""))
        except ValueError:
            continue
        image = item.get("image")
        if image:
            try:
                image = validate_public_web_url(str(image))
            except ValueError:
                image = None
        projected.append(
            {
                "url": url,
                "title": str(item.get("title") or ""),
                "description": str(item.get("description") or ""),
                "site": str(item.get("site") or ""),
                "image": image,
            }
        )
    return projected


__all__ = [
    "MAX_CARDS",
    "LinkCard",
    "WrittenAddress",
    "addresses_in",
    "card_key",
    "code_spans",
    "collect_link_cards",
    "project_link_cards",
    "written_addresses",
]
