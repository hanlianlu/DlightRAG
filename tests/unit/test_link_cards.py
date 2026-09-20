# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Video link cards carry what the page declared, and nothing when it declared nothing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from dlightrag.engine.answer.links.cards import (
    MAX_CARDS,
    LinkCard,
    collect_link_cards,
    project_link_cards,
)
from dlightrag.engine.answer.markdown import link_targets


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

    assert link_targets(answer) == ["https://example.com/a", "https://example.com/b"]


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
    assert fetcher.calls == [("https://example.com/watch", 2 * 1024 * 1024, 4.0)]


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


def test_addresses_quoted_as_code_are_not_written_addresses() -> None:
    answer = "Run `curl https://example.com/a` and:\n\n```\nhttps://example.com/b\n```\n"

    assert link_targets(answer) == []


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
    fetcher = _Fetcher({"https://example.com/clip": _video_page()})

    assert await collect_link_cards(answer, fetch=fetcher) == ()
    assert fetcher.calls == []


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


async def test_a_page_of_long_attributes_parses_within_its_budget() -> None:
    """Attributes are length-bounded: no `=` anywhere cannot cost a quadratic scan."""
    import time

    page = _Page(content=b"<html><head>" + (b"<meta " + b"a" * 8185 + b">") * 32)
    fetcher = _Fetcher({"https://example.com/clip": page})
    started = time.monotonic()

    cards = await collect_link_cards("https://example.com/clip", fetch=fetcher)

    assert cards == ()
    assert time.monotonic() - started < 1.0


async def test_a_longer_element_name_is_not_a_meta_tag() -> None:
    """An XML or RDF metadata element is not an HTML meta declaration."""
    page = _Page(
        content=(
            b'<html><head><metadata property="og:type" content="video.other">'
            b'<metadata property="og:title" content="Clip"></head></html>'
        )
    )
    fetcher = _Fetcher({"https://example.com/clip": page})

    assert await collect_link_cards("https://example.com/clip", fetch=fetcher) == ()

    declared = _Fetcher(
        {
            "https://example.com/clip": _Page(
                content=(
                    b'<html><head><meta property="og:type" content="video.other">'
                    b'<meta property="og:title" content="Clip"></head></html>'
                )
            )
        }
    )
    assert [
        card.title for card in await collect_link_cards("https://example.com/clip", fetch=declared)
    ] == ["Clip"]


def test_a_stray_backtick_before_a_fence_does_not_hide_the_prose_after_it() -> None:
    """Inline code is recognized inside prose only, so a run cannot cross a fence."""
    answer = "see `x` before\n```\ncode\n```\nhttps://example.com/clip`\n"

    assert link_targets(answer) == ["https://example.com/clip%60"]
    assert link_targets("```\ncode\n```\nhttps://example.com/clip\n") == [
        "https://example.com/clip"
    ]


def test_a_link_whose_destination_is_quoted_code_is_not_written() -> None:
    """A fenced block can contain text that merely looks like a Markdown link."""
    answer = "Read [label\n~~~\n](https://example.com/clip)\n~~~\n"

    assert link_targets(answer) == []


def test_a_link_may_still_quote_code_in_its_label() -> None:
    answer = "See [run `curl` first](https://example.com/clip) now."

    assert link_targets(answer) == ["https://example.com/clip"]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        # A multi-line inline span: the indented line is a paragraph continuation,
        # so the address is inside the span rather than in a block.
        ("Say `\n    x\nhttps://example.com/clip`\n", []),
        # A fence, then a genuine inline span on the last line.
        ("~~~\n`\n~~~\n`https://example.com/clip`\n", []),
        # An indented line inside a paragraph is prose, which is what is rendered.
        ("Some text\n    https://example.com/clip\n", ["https://example.com/clip"]),
        # An indented block after a blank line is code.
        ("\n    https://example.com/clip\n", []),
        # A stray backtick before a fence cannot pair with one after it.
        (
            "see `x` before\n```\ncode\n```\nhttps://example.com/clip`\n",
            ["https://example.com/clip%60"],
        ),
    ],
    ids=[
        "multiline-inline",
        "fence-then-inline",
        "indented-continuation",
        "indented-block",
        "stray-backtick",
    ],
)
def test_code_quoting_agrees_with_what_the_renderer_shows(answer: str, expected: list[str]) -> None:
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    rendered = render_answer_html(answer, known_sources={})

    assert link_targets(answer) == expected
    # The renderer links exactly the addresses this module calls written.
    if expected:
        assert "<a " in rendered
    else:
        assert "<a " not in rendered


def test_a_title_duplicate_cannot_hide_a_real_link() -> None:
    """Only the link destination is read; title text and code are not links."""
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    answer = '[x](https://example.com/a "`https://example.com/clip`") `https://example.com/clip`'

    assert link_targets(answer) == ["https://example.com/a"]
    assert "https://example.com/clip" in render_answer_html(answer, known_sources={})


def test_padded_and_unpadded_code_are_neither_link_targets() -> None:
    """Code tokens are excluded without trying to distinguish normalized content."""
    answer = "` https://example.com/clip ` `https://example.com/clip`"

    assert link_targets(answer) == []
    # A single padded span is still code, and still excluded.
    assert link_targets("A ` https://example.com/clip ` paragraph.") == []


def test_multiline_and_single_line_code_emit_no_links() -> None:
    """Equivalent code content needs no source-location reconstruction."""
    answer = "`\nhttps://example.com/clip\n` `https://example.com/clip`"

    assert link_targets(answer) == []
    # Addresses between two spans are still written, so protection is neither lost
    # nor spread wider than the spans themselves.
    assert link_targets("`x` https://example.com/clip `y`") == ["https://example.com/clip"]


async def test_repeated_quoted_addresses_are_not_read() -> None:
    """Code occurrences do not emit link tokens, regardless of normalization."""
    fetcher = _Fetcher({"https://example.com/clip": _video_page()})
    answer = "> `\n> https://example.com/clip\n> ` `https://example.com/clip`"

    assert await collect_link_cards(answer, fetch=fetcher) == ()
    assert fetcher.calls == []

    # Written once, so it is read.
    fetcher.calls.clear()
    assert await collect_link_cards("See https://example.com/clip", fetch=fetcher)
    assert len(fetcher.calls) == 1
