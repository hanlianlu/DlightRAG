# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Video link cards carry what the page declared, and nothing when it declared nothing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from dlightrag.engine.answer.links.cards import (
    MAX_CARDS,
    LinkCard,
    addresses_in,
    collect_link_cards,
    link_card_for,
    project_link_cards,
)


@dataclass
class _Page:
    content: bytes
    media_type: str | None = "text/html"
    final_url: str = ""
    status_code: int = 200


def _page(*metadata: str, title: str = "") -> _Page:
    body = "".join(f'<meta property="{tag}">' for tag in metadata)
    head = f"<title>{title}</title>" if title else ""
    return _Page(content=f"<html><head>{head}{body}</head><body></body></html>".encode())


def _video_page(**overrides: str) -> _Page:
    values = {
        "og:type": "video.other",
        "og:title": "Big Buck Bunny",
        "og:description": "A short film.",
        "og:site_name": "Example",
        "og:image": "https://cdn.example.com/cover.jpg",
    }
    values.update(overrides)
    tags = [f'{key}" content="{value}' for key, value in values.items() if value]
    return _Page(
        content=("<html><head>" + "".join(f'<meta property="{tag}">' for tag in tags)).encode()
        + b"</head></html>"
    )


class _Fetcher:
    """One scripted fetch surface: every call is recorded and answered by URL."""

    def __init__(self, pages: dict[str, Any]) -> None:
        self.pages = pages
        self.calls: list[tuple[str, int, float]] = []

    async def __call__(self, url: str, *, max_bytes: int, timeout: float) -> Any:
        self.calls.append((url, max_bytes, timeout))
        answer = self.pages.get(url)
        if isinstance(answer, Exception):
            raise answer
        if answer is None:
            raise ValueError("unreachable")
        return answer


def test_addresses_are_distinct_ordered_and_without_sentence_punctuation() -> None:
    answer = "见 https://example.com/a。 还有 https://example.com/b, https://example.com/a"

    assert addresses_in(answer) == ["https://example.com/a", "https://example.com/b"]


async def test_a_declared_video_becomes_a_card() -> None:
    page = _video_page()
    fetcher = _Fetcher({"https://example.com/watch": page})

    cards = await collect_link_cards("Watch https://example.com/watch now", fetch=fetcher)

    assert cards == (
        LinkCard(
            url="https://example.com/watch",
            title="Big Buck Bunny",
            description="A short film.",
            site="Example",
            image="https://cdn.example.com/cover.jpg",
        ),
    )
    assert fetcher.calls == [("https://example.com/watch", 256 * 1024, 4.0)]


async def test_a_page_that_declares_only_a_player_is_a_video() -> None:
    page = _page('og:video" content="https://player.example.com/embed/1', title="Clip")
    fetcher = _Fetcher({"https://example.com/clip": page})

    cards = await collect_link_cards("https://example.com/clip", fetch=fetcher)

    assert [card.title for card in cards] == ["Clip"]
    assert cards[0].image is None


async def test_an_ordinary_page_stays_a_link() -> None:
    page = _page('og:type" content="website', 'og:title" content="Report')
    fetcher = _Fetcher({"https://example.com/report": page})

    assert await collect_link_cards("See https://example.com/report", fetch=fetcher) == ()


async def test_an_unreadable_or_undeclared_page_stays_a_link() -> None:
    fetcher = _Fetcher(
        {
            "https://example.com/slow": TimeoutError("too slow"),
            "https://example.com/binary": _Page(content=b"%PDF-1.7", media_type="application/pdf"),
            "https://example.com/untitled": _page('og:type" content="video.other'),
        }
    )

    cards = await collect_link_cards(
        "See https://example.com/slow https://example.com/binary https://example.com/untitled",
        fetch=fetcher,
    )

    assert cards == ()


async def test_only_the_first_declared_videos_are_read() -> None:
    pages = {
        f"https://example.com/{index}": _video_page(description=str(index)) for index in range(6)
    }
    fetcher = _Fetcher(pages)
    answer = " ".join(pages)

    cards = await collect_link_cards(answer, fetch=fetcher)

    assert len(cards) == MAX_CARDS
    assert len(fetcher.calls) == MAX_CARDS


async def test_an_unsafe_cover_image_is_dropped_and_the_card_survives() -> None:
    page = _video_page(**{"og:image": "http://127.0.0.1/cover.jpg"})
    fetcher = _Fetcher({"https://example.com/watch": page})

    cards = await collect_link_cards("https://example.com/watch", fetch=fetcher)

    assert cards[0].image is None


def test_projection_keeps_only_public_addresses() -> None:
    projected = project_link_cards(
        [
            {
                "url": "https://example.com/watch",
                "title": "Clip",
                "description": "",
                "site": "",
                "image": "https://cdn.example.com/cover.jpg",
            },
            {"url": "http://127.0.0.1/watch", "title": "Local", "description": "", "site": ""},
            "not-a-card",
        ]
    )

    assert [card["url"] for card in projected] == ["https://example.com/watch"]


def test_a_card_is_found_by_the_address_as_written() -> None:
    cards: list[dict[str, str | None]] = [{"url": "https://example.com/watch", "title": "Clip"}]

    assert link_card_for("https://example.com/watch", cards) == cards[0]
    assert link_card_for("https://example.com/other", cards) is None


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("video.other", True),
        ("video.movie", True),
        ("website", False),
        ("article", False),
        ("", False),
    ],
)
async def test_only_a_video_declaration_is_honoured(declared: str, expected: bool) -> None:
    page = _video_page(**{"og:type": declared, "og:video": ""})
    fetcher = _Fetcher({"https://example.com/page": page})

    cards = await collect_link_cards("https://example.com/page", fetch=fetcher)

    assert bool(cards) is expected


def test_a_markdown_link_a_card_covers_becomes_one_part() -> None:
    from dlightrag.engine.answer.results import answer_parts_from_markdown

    card = {
        "url": "https://www.youtube.com/watch?v=abc",
        "title": "Big Buck Bunny",
        "description": "",
        "site": "YouTube",
        "image": None,
    }

    parts = answer_parts_from_markdown(
        "Before [the film](https://www.youtube.com/watch?v=abc) after.",
        artifacts=[],
        evidence_images=[],
        link_cards=[card],
    )

    assert [part["type"] for part in parts] == ["markdown", "link_card", "markdown"]
    assert parts[1]["card"] == card
    assert parts[0]["text"] == "Before "
    assert parts[2]["text"] == " after."


def test_a_bare_address_a_card_covers_becomes_one_part() -> None:
    from dlightrag.engine.answer.results import answer_parts_from_markdown

    card = {"url": "https://example.com/clip", "title": "Clip", "description": "", "site": ""}

    parts = answer_parts_from_markdown(
        "See https://example.com/clip for it.",
        artifacts=[],
        evidence_images=[],
        link_cards=[card],
    )

    assert [part["type"] for part in parts] == ["markdown", "link_card", "markdown"]
    assert parts[1]["card"] == card


def test_a_cited_source_never_becomes_a_card() -> None:
    from dlightrag.engine.answer.results import answer_parts_from_markdown

    card = {"url": "https://example.com/source", "title": "Clip", "description": "", "site": ""}

    parts = answer_parts_from_markdown(
        "Fact [1](https://example.com/source).",
        artifacts=[],
        evidence_images=[],
        link_cards=[card],
        citation_urls=frozenset({"https://example.com/source"}),
    )

    assert [part["type"] for part in parts] == ["markdown"]


def test_an_address_without_a_card_stays_markdown() -> None:
    from dlightrag.engine.answer.results import answer_parts_from_markdown

    parts = answer_parts_from_markdown(
        "See https://example.com/plain now.",
        artifacts=[],
        evidence_images=[],
        link_cards=[{"url": "https://example.com/other", "title": "Other"}],
    )

    assert [part["type"] for part in parts] == ["markdown"]
