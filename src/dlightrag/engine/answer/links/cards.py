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


def _line_spans(answer: str, kinds: frozenset[str]) -> list[tuple[int, int]]:
    """Return the character spans of the blocks the parser reports as ``kinds``."""
    # Line numbers are the parser's: it counts newlines, while `splitlines` would
    # also break on U+2028 and U+2029 and shift every span after them.
    lines = answer.split("\n")
    offsets = [0]
    for index, line in enumerate(lines):
        offsets.append(offsets[-1] + len(line) + (1 if index < len(lines) - 1 else 0))
    spans: list[tuple[int, int]] = []
    for token in _BLOCK_PARSER.parse(answer):
        if token.type in kinds and token.map:
            start, end = token.map
            spans.append((offsets[start], offsets[min(end, len(offsets) - 1)]))
    return spans


def _block_code_spans(answer: str) -> list[tuple[int, int]]:
    """Return the spans the renderer itself calls block code.

    Fences, indented blocks, headings that end a paragraph, fence lengths: the
    same parser the Answer is rendered with decides all of it, so this module
    never has to guess at Markdown's block rules.
    """
    return _line_spans(answer, frozenset({"fence", "code_block"}))


def _inline_regions(answer: str) -> list[tuple[int, int]]:
    """Return the parser's own inline contexts: one paragraph, heading, or cell.

    Inline code lives inside exactly one of these, so recognizing spans within
    them cannot pair a backtick in one paragraph with a backtick in the next.
    """
    return _line_spans(answer, frozenset({"inline"}))


def _inline_code_spans(answer: str, segments: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return inline code spans inside prose, closed only by a run of equal length."""
    spans: list[tuple[int, int]] = []
    for start, end in segments:
        position = start
        while position < end:
            opening = answer.find("`", position, end)
            if opening == -1:
                break
            length = _run_length(answer, opening, end)
            if _is_escaped(answer, opening):
                position = opening + length
                continue
            closing = _matching_run(answer, opening + length, end, length)
            if closing == -1:
                position = opening + length
                continue
            spans.append((opening, closing + length))
            position = closing + length
    return spans


def _run_length(answer: str, position: int, end: int) -> int:
    length = 0
    while position + length < end and answer[position + length] == "`":
        length += 1
    return length


def _is_escaped(answer: str, position: int) -> bool:
    """Whether a backslash escapes the character at ``position``."""
    backslashes = 0
    scan = position - 1
    while scan >= 0 and answer[scan] == "\\":
        backslashes += 1
        scan -= 1
    return backslashes % 2 == 1


def _matching_run(answer: str, position: int, end: int, length: int) -> int:
    """Return the start of the next backtick run of exactly ``length``, or ``-1``."""
    scan = position
    while scan < end:
        found = answer.find("`", scan, end)
        if found == -1:
            return -1
        run = _run_length(answer, found, end)
        if run == length:
            return found
        scan = found + run
    return -1


def code_spans(answer: str) -> list[tuple[int, int]]:
    """Return the answer spans that quote code rather than write an address.

    Block code is the renderer's own classification, and inline runs are
    recognized inside the prose only, so a quoting character can never pair
    across a code block.
    """
    blocks = _block_code_spans(answer)
    prose = [
        segment
        for region in _inline_regions(answer)
        for segment in _overlap_of(region, _prose_segments(len(answer), blocks))
    ]
    return blocks + _inline_code_spans(answer, prose)


def _overlap_of(
    region: tuple[int, int], segments: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Return the parts of ``segments`` that fall inside ``region``."""
    start, end = region
    return [
        (max(start, segment_start), min(end, segment_end))
        for segment_start, segment_end in segments
        if segment_start < end and start < segment_end
    ]


def _prose_segments(length: int, blocks: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the stretches of an answer that are not inside a block of code."""
    segments: list[tuple[int, int]] = []
    position = 0
    for start, end in sorted(blocks):
        if start > position:
            segments.append((position, start))
        position = max(position, end)
    if position < length:
        segments.append((position, length))
    return segments


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
    """
    addresses = addresses_in(answer)[: max(0, limit)]
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
