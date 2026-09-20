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
    card_key,
    code_spans,
    collect_link_cards,
    project_link_cards,
    written_addresses,
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

    async def __call__(
        self, url: str, *, max_bytes: int, timeout: float, agent_url: bool = False
    ) -> Any:
        self.calls.append((url, max_bytes, timeout))
        assert agent_url, "a card read must be anonymous"
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


async def test_a_page_that_declares_a_player_needs_its_own_title() -> None:
    """A card carries what the page declared; a document title is not a declaration."""
    declared = _page(
        'og:video" content="https://player.example.com/embed/1',
        'og:title" content="Clip',
    )
    only_a_title = _page('og:video" content="https://player.example.com/embed/1', title="Clip")
    fetcher = _Fetcher(
        {"https://example.com/declared": declared, "https://example.com/title": only_a_title}
    )

    cards = await collect_link_cards(
        "https://example.com/declared https://example.com/title", fetch=fetcher
    )

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


async def test_a_read_is_anonymous_and_bounded_by_its_deadline() -> None:
    """The deadline covers waiting for the shared network slot, not only reading."""
    import asyncio

    class _Slow:
        def __init__(self) -> None:
            self.started = 0

        async def __call__(
            self, url: str, *, max_bytes: int, timeout: float, agent_url: bool
        ) -> Any:
            self.started += 1
            assert agent_url, "a card read must be anonymous"
            await asyncio.sleep(5)
            raise AssertionError("a read past the deadline must be abandoned")

    slow = _Slow()
    started = asyncio.get_event_loop().time()

    cards = await collect_link_cards(
        "See https://example.com/one https://example.com/two", fetch=slow, deadline=0.05
    )

    elapsed = asyncio.get_event_loop().time() - started

    assert cards == ()
    assert slow.started == 2
    assert elapsed < 1.0, f"the deadline must bound the whole read, took {elapsed}"


def test_written_addresses_keep_the_punctuation_a_sentence_owns() -> None:
    answer = "See https://example.com/clip. And https://example.com/x。"
    written = written_addresses(answer)

    assert [address.key for address in written] == [
        "https://example.com/clip",
        "https://example.com/x",
    ]
    assert answer[written[0].end] == ".", "the period stays outside the address"
    assert answer[written[1].end] == "。"
    assert card_key("https://example.com/clip.") == "https://example.com/clip"


def test_addresses_quoted_as_code_are_not_written_addresses() -> None:
    answer = "Run `curl https://example.com/a` and:\n\n```\nhttps://example.com/b\n```\n"

    assert addresses_in(answer) == []
    assert code_spans(answer)


async def test_a_quoted_address_is_never_read() -> None:
    fetcher = _Fetcher({"https://example.com/b": _video_page()})

    cards = await collect_link_cards("```\nhttps://example.com/b\n```\n", fetch=fetcher)

    assert cards == ()
    assert fetcher.calls == []


@pytest.mark.parametrize(
    "declared",
    ["videogame", "video", "videoplaylist"],
)
async def test_a_near_miss_type_is_not_a_video(declared: str) -> None:
    page = _video_page(**{"og:type": declared, "og:video": ""})
    fetcher = _Fetcher({"https://example.com/page": page})

    assert await collect_link_cards("https://example.com/page", fetch=fetcher) == ()


async def test_a_commented_out_declaration_is_not_a_declaration() -> None:
    page = _Page(
        content=(
            b'<html><head><!-- <meta property="og:type" content="video.other"> -->'
            b'<meta property="og:title" content="Clip"></head></html>'
        )
    )
    fetcher = _Fetcher({"https://example.com/page": page})

    assert await collect_link_cards("https://example.com/page", fetch=fetcher) == ()


async def test_a_non_html_answer_with_html_words_is_not_a_page() -> None:
    page = _Page(
        content=b'og:type="video.other" og:title="Clip"',
        media_type="text/plain; note=html",
    )
    fetcher = _Fetcher({"https://example.com/page": page})

    assert await collect_link_cards("https://example.com/page", fetch=fetcher) == ()


def test_a_sentence_period_does_not_break_the_card_and_is_not_swallowed() -> None:
    """The card key drops the period; the span must too, and the period stays text."""
    from dlightrag.engine.answer.results import answer_parts_from_markdown

    card = {"url": "https://example.com/clip", "title": "Clip", "description": "", "site": ""}

    for answer, tail in [
        ("See https://example.com/clip.", "."),
        ("See https://example.com/clip, and more.", ", and more."),
        ("看 https://example.com/clip。", "。"),
    ]:
        parts = answer_parts_from_markdown(
            answer, artifacts=[], evidence_images=[], link_cards=[card]
        )

        assert [part["type"] for part in parts] == ["markdown", "link_card", "markdown"], answer
        assert parts[1]["card"] == card, answer
        assert parts[2]["text"] == tail, answer


def test_a_markdown_link_inside_code_is_not_carded() -> None:
    from dlightrag.engine.answer.results import answer_parts_from_markdown

    card = {"url": "https://example.com/clip", "title": "Clip", "description": "", "site": ""}
    answer = "```\n[clip](https://example.com/clip)\n```\n"

    parts = answer_parts_from_markdown(answer, artifacts=[], evidence_images=[], link_cards=[card])

    assert [part["type"] for part in parts] == ["markdown"]


def test_a_card_covers_the_whole_markdown_link_it_describes() -> None:
    """A card replaces the link, not just the destination inside it."""
    from dlightrag.engine.answer.results import answer_parts_from_markdown

    card = {
        "url": "https://example.com/a,b",
        "title": "Clip",
        "description": "",
        "site": "",
    }
    answer = "Watch [the film](https://example.com/a,b) now."

    parts = answer_parts_from_markdown(answer, artifacts=[], evidence_images=[], link_cards=[card])

    assert [part["type"] for part in parts] == ["markdown", "link_card", "markdown"]
    assert parts[0]["text"] == "Watch "
    assert parts[2]["text"] == " now."
    assert "[" not in "".join(str(part.get("text") or "") for part in parts)


@pytest.mark.parametrize(
    "answer",
    [
        "    https://example.com/clip\n",
        "```\nhttps://example.com/clip\n",
        "``https://example.com/clip``\n",
        "~~~\nhttps://example.com/clip\n~~~\n",
    ],
    ids=["indented", "unterminated-fence", "double-backtick", "tilde-fence"],
)
async def test_a_quoted_address_is_not_written_nor_replaced(answer: str) -> None:
    from dlightrag.engine.answer.results import answer_parts_from_markdown

    card = {"url": "https://example.com/clip", "title": "Clip", "description": "", "site": ""}
    fetcher = _Fetcher({"https://example.com/clip": _video_page()})

    assert await collect_link_cards(answer, fetch=fetcher) == ()
    assert fetcher.calls == []
    parts = answer_parts_from_markdown(answer, artifacts=[], evidence_images=[], link_cards=[card])
    assert [part["type"] for part in parts] == ["markdown"]


async def test_a_long_declared_title_still_produces_a_card() -> None:
    """A page's own title is truncated for the card, never dropped with the page."""
    page = _video_page(**{"og:title": "T" * 1200})
    fetcher = _Fetcher({"https://example.com/clip": page})

    cards = await collect_link_cards("https://example.com/clip", fetch=fetcher)

    assert len(cards) == 1
    assert cards[0].title == "T" * 200


async def test_an_unterminated_comment_hides_everything_after_it() -> None:
    page = _Page(
        content=(
            b'<html><head><!-- <meta property="og:type" content="video.other">'
            b'<meta property="og:title" content="Clip"></head></html>'
        )
    )
    fetcher = _Fetcher({"https://example.com/clip": page})

    assert await collect_link_cards("https://example.com/clip", fetch=fetcher) == ()


async def test_a_comment_inside_an_attribute_invents_no_declaration() -> None:
    page = _Page(
        content=(
            b'<html><head><meta property="og:<!-- -->type" content="video.other">'
            b'<meta property="og:title" content="Clip"></head></html>'
        )
    )
    fetcher = _Fetcher({"https://example.com/clip": page})

    assert await collect_link_cards("https://example.com/clip", fetch=fetcher) == ()


async def test_a_page_of_comment_openers_parses_within_its_budget() -> None:
    """The parse is linear: a page cannot buy parsing time with comment openers."""
    import time

    page = _Page(content=b"<html><head>" + b"<!--" * 65536 + b"</head></html>")
    fetcher = _Fetcher({"https://example.com/clip": page})
    started = time.monotonic()

    cards = await collect_link_cards("https://example.com/clip", fetch=fetcher)

    assert cards == ()
    assert time.monotonic() - started < 2.0
