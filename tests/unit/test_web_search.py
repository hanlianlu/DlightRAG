# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Exa web search."""

import json

import httpx
import pytest

from dlightrag.core.retrieval.web_search import ExaSearch, WebSearchUnavailable

_PAGE = {
    "url": "https://example.org/taylor",
    "title": "The Taylor rule",
    "publishedDate": "2026-01-02T00:00:00.000Z",
    "image": "https://example.org/figure-1.png",
    "highlights": ["a is usually about 1.5", "b is usually about 0.5"],
}


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _responds(payload: dict, status: int = 200):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


@pytest.mark.asyncio
async def test_every_highlight_arrives_with_the_page_it_came_from() -> None:
    search = ExaSearch("k", client=_client(_responds({"results": [_PAGE]})))

    result = await search.search("taylor rule coefficients")

    assert [hit.text for hit in result.hits] == _PAGE["highlights"]
    assert {hit.url for hit in result.hits} == {_PAGE["url"]}
    assert result.hits[0].published_date == _PAGE["publishedDate"]
    assert result.hits[0].image_url == _PAGE["image"]


@pytest.mark.asyncio
async def test_a_passage_is_never_asked_for_at_its_provider_default_length() -> None:
    asked: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(json.loads(request.content)["contents"])
        return httpx.Response(200, json={"results": [_PAGE]})

    search = ExaSearch("k", client=_client(handler))

    await search.search("q")
    await search.search("q", full_text=True)

    assert all("maxCharacters" in asked[0]["highlights"] for _ in asked)
    assert "text" not in asked[0]
    assert "maxCharacters" in asked[1]["text"]


@pytest.mark.asyncio
async def test_a_returned_page_body_becomes_a_passage_of_its_own() -> None:
    page = {**_PAGE, "text": "Full article body."}
    search = ExaSearch("k", client=_client(_responds({"results": [page]})))

    result = await search.search("q", full_text=True)

    assert result.hits[-1].text == "Full article body."
    assert result.hits[-1].url == _PAGE["url"]


@pytest.mark.asyncio
async def test_the_reported_cost_is_carried_back_to_the_caller() -> None:
    payload = {"results": [_PAGE], "costDollars": {"total": 0.007}}
    search = ExaSearch("k", client=_client(_responds(payload)))

    assert (await search.search("q")).cost_dollars == 0.007


@pytest.mark.asyncio
async def test_an_empty_balance_stops_the_next_search_before_it_leaves_the_process() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(402, json={})

    search = ExaSearch("k", client=_client(handler))

    with pytest.raises(WebSearchUnavailable) as first:
        await search.search("q")
    with pytest.raises(WebSearchUnavailable) as second:
        await search.search("q")

    assert first.value.reason == "payment_required"
    assert second.value.reason == "payment_required"
    assert calls == 1


@pytest.mark.asyncio
async def test_a_parked_search_tries_again_once_the_wait_is_over(monkeypatch) -> None:
    now = 0.0
    monkeypatch.setattr("dlightrag.core.retrieval.web_search.time.monotonic", lambda: now)
    replies = [httpx.Response(402, json={}), httpx.Response(200, json={"results": [_PAGE]})]
    search = ExaSearch("k", client=_client(lambda _request: replies.pop(0)))

    with pytest.raises(WebSearchUnavailable):
        await search.search("q")
    now = 15 * 60

    assert len((await search.search("q")).hits) == 2


@pytest.mark.asyncio
async def test_a_slow_provider_is_reported_rather_than_raised_at_the_caller() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    search = ExaSearch("k", client=_client(handler))

    with pytest.raises(WebSearchUnavailable) as failure:
        await search.search("q")

    assert failure.value.reason == "timeout"


@pytest.mark.asyncio
async def test_a_server_error_does_not_park_the_client() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, json={})

    search = ExaSearch("k", client=_client(handler))

    for _ in range(2):
        with pytest.raises(WebSearchUnavailable):
            await search.search("q")

    assert calls == 2
