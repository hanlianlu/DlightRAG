# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The same fixtures cross settlement, sanitized wire HTML and the browser DOM."""

import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from dlightrag.adapters.http.browser.presentation import build_answer_presentation
from dlightrag.engine.answer.links.cards import collect_link_cards
from dlightrag.engine.answer.markdown import link_targets

_FIXTURES = json.loads(
    (Path(__file__).parents[2] / "frontend/ui/fixtures/link-card-occurrences.json").read_text()
)


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


@pytest.mark.parametrize("case", _FIXTURES, ids=lambda case: case["name"])
async def test_only_actual_link_occurrences_can_trigger_reads(case: dict[str, Any]) -> None:
    calls: list[str] = []

    async def fetch(url: str, **_kwargs: Any) -> None:
        calls.append(url)
        raise TimeoutError("a card failure must leave the original link intact")

    expected = list(dict.fromkeys(case["hrefs"]))
    assert link_targets(case["markdown"]) == expected
    assert await collect_link_cards(case["markdown"], fetch=fetch) == ()
    assert calls == expected

    # Browser tests consume this exact sanitized HTML. In particular they offer
    # metadata for BOTH /x and /y, so an extraneous quoted occurrence cannot hide
    # behind the read-side exclusion or missing metadata.
    value = build_answer_presentation(answer=case["markdown"], sources=[], evidence_images=[])
    assert len(value.parts) == 1
    assert value.parts[0].text == case["markdown"]
    assert value.parts[0].html == case["html"]
    links = _Links()
    links.feed(value.parts[0].html)
    assert links.hrefs == case["hrefs"]


def test_resources_are_placed_from_tokens_without_fragment_reparsing() -> None:
    fixture = json.loads(
        (
            Path(__file__).parents[2] / "frontend/ui/fixtures/mixed-resource-presentation.json"
        ).read_text()
    )
    value = build_answer_presentation(**fixture["input"])
    assert value.model_dump() == fixture["wire"]
    assert [part.type for part in value.parts] == [
        "markdown",
        "artifact",
        "artifact",
        "evidence_image",
    ]
    assert [part.slot for part in value.parts] == [None, 0, 1, 2]
    assert value.evidence_images == []
    links = _Links()
    links.feed(value.parts[0].html)
    assert links.hrefs == ["https://example.com/x", "https://example.com/y"]
    assert (
        "<code>before [quoted](artifact:clip.mp4) https://example.com/x after</code>"
        in value.parts[0].html
    )


def test_resource_slot_contract_rejects_negative_identities() -> None:
    from pydantic import ValidationError

    from dlightrag.adapters.http.browser.presentation import PresentationPart

    with pytest.raises(ValidationError):
        PresentationPart(type="artifact", slot=-1)
    assert PresentationPart(type="artifact", slot=0).slot == 0
    assert PresentationPart(type="artifact").slot is None


async def test_duplicate_links_share_one_read_not_one_placement() -> None:
    calls: list[str] = []

    async def fetch(url: str, **_kwargs: Any) -> None:
        calls.append(url)
        raise TimeoutError

    url = "https://example.com/clip"
    await collect_link_cards(f"`{url}` {url} [again]({url})", fetch=fetch)
    assert calls == [url]
