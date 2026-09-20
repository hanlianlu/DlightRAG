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
# Fenced blocks and inline code, where an address is quoted rather than written.
_CODE = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]*`", re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# A meta tag, and the attributes inside it, are bounded before any pattern sees
# them: an unclosed tag in a large page would otherwise be scanned per start.
_META_TAG = re.compile(r"<meta\b[^>]{0,1024}>", re.IGNORECASE)
_ATTRIBUTE = re.compile(r"(?P<name>[a-zA-Z:_-]+)\s*=\s*(?P<value>\"[^\"]*\"|'[^']*'|[^\s\"'>]+)")
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


def code_spans(answer: str) -> list[tuple[int, int]]:
    """Return the answer spans that quote code rather than write an address."""
    return [match.span() for match in _CODE.finditer(answer)]


def card_key(url: str) -> str:
    """Return the identity one address is matched by, without a trailing dot."""
    return url.rstrip(".")


def address_spans(answer: str) -> list[tuple[int, int, str]]:
    """Return one span per written address, in order, as ``(start, end, key)``.

    Both the card read and the Answer's parts replace addresses by these spans, so
    they cannot disagree about where an address ends or drop the punctuation a
    sentence owns. Addresses inside code are quoted, not written.
    """
    quoted = code_spans(answer)
    spans: list[tuple[int, int, str]] = []
    for match in _ADDRESS.finditer(answer):
        if any(start <= match.start() < end for start, end in quoted):
            continue
        text = match.group(0)
        key = card_key(text)
        spans.append((match.start(), match.start() + len(key), key))
    return spans


def addresses_in(answer: str) -> list[str]:
    """Return the distinct public addresses one answer writes, in order."""
    seen: dict[str, None] = {}
    for _, _, key in address_spans(answer):
        seen.setdefault(key, None)
    return list(seen)


def _attributes(tag: str) -> dict[str, str]:
    return {
        match.group("name").lower(): _html.unescape(match.group("value").strip("\"'"))
        for match in _ATTRIBUTE.finditer(tag)
    }


def _metadata(page: str) -> dict[str, str]:
    """Return the page's own Open Graph metadata, first declaration wins."""
    values: dict[str, str] = {}
    for tag in _META_TAG.finditer(_COMMENT.sub("", page)):
        attributes = _attributes(tag.group(0))
        key = (attributes.get("property") or attributes.get("name") or "").lower()
        content = attributes.get("content")
        if key.startswith("og:") and content and key not in values:
            values[key] = content
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
    "address_spans",
    "addresses_in",
    "card_key",
    "code_spans",
    "collect_link_cards",
    "project_link_cards",
]
