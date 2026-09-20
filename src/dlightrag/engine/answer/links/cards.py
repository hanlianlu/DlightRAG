# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Video link cards: a link a page declares as a video, read from its own metadata.

The Model writes an address; the page it points at decides whether it is a video
by publishing Open Graph metadata, and the card carries only what the page said.
A page that answers with anything else — no declaration, no answer at all, a
network this deployment cannot reach — leaves the link exactly as the Model wrote
it. Nothing here is a site list: YouTube (`og:video:url`) and Bilibili
(`og:video` to its own player) declare it in the same shape, and an article
(`og:type: website`) declares that it is not one.

Cards are bounded on every axis: how many are read per answer, how many bytes
each page may return, and how long each read may take. ADR 0026 records the
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
#: three links costs about one deadline rather than three.
_PAGE_TIMEOUT_SECONDS = 4.0
_MAX_TITLE = 200
_MAX_DESCRIPTION = 400

# An address as it appears in answer Markdown. Trailing punctuation that a
# sentence owns is excluded, matching what the renderer will link.
_ADDRESS = re.compile(r"https?://[^\s<>\"'`\u3000-\u303f\uff00-\uffef)\]},;]+")
_META_TAG = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_ATTRIBUTE = re.compile(
    r"(?P<name>[a-zA-Z:_-]+)\s*=\s*(?P<value>\"[^\"]*\"|'[^']*'|[^\s\"'>]+)",
)
_TITLE_TAG = re.compile(r"<title\b[^>]*>(?P<title>.*?)</title>", re.IGNORECASE | re.DOTALL)


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


def addresses_in(answer: str) -> list[str]:
    """Return the distinct public addresses one answer writes, in order."""
    seen: dict[str, None] = {}
    for match in _ADDRESS.finditer(answer):
        address = match.group(0).rstrip(".")
        if address not in seen:
            seen[address] = None
    return list(seen)


def _attributes(tag: str) -> dict[str, str]:
    return {
        match.group("name").lower(): _html.unescape(match.group("value").strip("\"'"))
        for match in _ATTRIBUTE.finditer(tag)
    }


def _metadata(page: str) -> dict[str, str]:
    """Return the page's own Open Graph and title metadata, first declaration wins."""
    values: dict[str, str] = {}
    for tag in _META_TAG.finditer(page):
        attributes = _attributes(tag.group(0))
        key = (attributes.get("property") or attributes.get("name") or "").lower()
        content = attributes.get("content")
        if key and content and key not in values:
            values[key] = content
    if "og:title" not in values:
        title = _TITLE_TAG.search(page)
        if title:
            values["og:title"] = _html.unescape(title.group("title"))
    return values


def _declares_video(values: dict[str, str]) -> bool:
    """Whether the page itself says it holds a video."""
    if any(key.startswith("og:video") for key in values):
        return True
    return values.get("og:type", "").strip().lower().startswith("video")


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


async def collect_link_cards(
    answer: str,
    *,
    fetch: Callable[..., Awaitable[Any]] = fetch_public_http,
    limit: int = MAX_CARDS,
) -> tuple[LinkCard, ...]:
    """Read at most ``limit`` declared videos out of one answer's addresses.

    Every failure is silent by design: an address that cannot be read, answers too
    slowly, or declares nothing stays an ordinary link in the Answer.
    """
    addresses = addresses_in(answer)[: max(0, limit)]
    if not addresses:
        return ()
    pages = await asyncio.gather(
        *(
            fetch(address, max_bytes=_MAX_PAGE_BYTES, timeout=_PAGE_TIMEOUT_SECONDS)
            for address in addresses
        ),
        return_exceptions=True,
    )
    cards: list[LinkCard] = []
    for address, page in zip(addresses, pages, strict=True):
        if isinstance(page, BaseException):
            # One unreachable link never fails an answer: it stays a plain link.
            continue
        content = getattr(page, "content", b"")
        media_type = getattr(page, "media_type", None)
        if media_type is not None and "html" not in str(media_type).lower():
            continue
        values = _metadata(content.decode("utf-8", errors="replace"))
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


def link_card_for(url: str, cards: Sequence[dict[str, str | None]]) -> dict[str, str | None] | None:
    """Return the card one answer address placed, matching the address as written."""
    for card in cards:
        if card.get("url") == url:
            return card
    return None


__all__ = [
    "MAX_CARDS",
    "LinkCard",
    "addresses_in",
    "collect_link_cards",
    "link_card_for",
    "project_link_cards",
]
