# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Exa web search."""

import json

import httpx
import pytest

from dlightrag.answer.tools.web import (
    ExaSearch,
    WebSearchHit,
    WebSearchUnavailable,
    web_context_rows,
)

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
        asked.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [_PAGE]})

    search = ExaSearch("k", client=_client(handler))

    await search.search("q")

    assert asked[0]["type"] == "auto"
    assert "maxCharacters" in asked[0]["contents"]["highlights"]


@pytest.mark.asyncio
async def test_a_returned_page_body_becomes_a_passage_of_its_own() -> None:
    page = {**_PAGE, "text": "Full article body."}
    search = ExaSearch("k", client=_client(_responds({"results": [page]})))

    result = await search.search("q")

    assert result.hits[-1].text == "Full article body."
    assert result.hits[-1].url == _PAGE["url"]


@pytest.mark.asyncio
async def test_the_reported_cost_is_carried_back_to_the_caller() -> None:
    payload = {"results": [_PAGE], "costDollars": {"total": 0.007}}
    search = ExaSearch("k", client=_client(_responds(payload)))

    assert (await search.search("q")).cost_dollars == 0.007


@pytest.mark.asyncio
async def test_contents_fetches_known_url_text_via_the_contents_endpoint() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "body": json.loads(request.content)})
        return httpx.Response(200, json={"results": [{**_PAGE, "text": "Body text."}]})

    search = ExaSearch("k", client=_client(handler))

    result = await search.contents("https://example.org/taylor")

    assert seen[0]["url"] == "https://api.exa.ai/contents"
    assert seen[0]["body"]["urls"] == ["https://example.org/taylor"]
    assert seen[0]["body"]["text"] == {"maxCharacters": 10_000}
    assert "highlights" not in seen[0]["body"]
    assert result.hits[-1].text == "Body text."


@pytest.mark.asyncio
async def test_contents_respects_the_same_parking_as_search() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={})

    search = ExaSearch("k", client=_client(handler))

    with pytest.raises(WebSearchUnavailable) as first:
        await search.contents("https://a/x")
    with pytest.raises(WebSearchUnavailable) as second:
        await search.contents("https://a/x")

    assert first.value.reason == "unauthorized"
    assert second.value.reason == "unauthorized"
    assert calls == 1


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
    monkeypatch.setattr("dlightrag.answer.tools.web.time.monotonic", lambda: now)
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


@pytest.mark.asyncio
async def test_the_wait_outlasts_a_live_crawl() -> None:
    search = ExaSearch("k")
    try:
        assert search._client.timeout.read is not None
        assert search._client.timeout.read > 10.0
    finally:
        await search.aclose()


@pytest.mark.asyncio
async def test_pages_that_stop_reading_are_complained_about_not_dropped(caplog) -> None:
    unreadable = {"results": [{"title": "no url here"}]}
    search = ExaSearch("k", client=_client(_responds(unreadable)))

    with caplog.at_level("WARNING"):
        result = await search.search("q")

    assert result.hits == ()
    assert "no usable passage" in caplog.text


@pytest.mark.asyncio
async def test_finding_nothing_is_not_complained_about() -> None:
    search = ExaSearch("k", client=_client(_responds({"results": []})))

    assert (await search.search("q")).hits == ()


@pytest.mark.asyncio
async def test_a_payload_without_the_required_field_is_complained_about(caplog) -> None:
    search = ExaSearch("k", client=_client(_responds({"items": [_PAGE]})))

    with caplog.at_level("WARNING"):
        result = await search.search("q")

    assert result.hits == ()
    assert "without a results field" in caplog.text


def _hit(url: str, text: str, **kw) -> WebSearchHit:
    return WebSearchHit(url=url, title=kw.pop("title", "T"), text=text, **kw)


def test_passages_from_one_page_share_one_source() -> None:
    rows = web_context_rows([_hit("https://a/x#one", "one"), _hit("https://a/x#two", "two")])

    assert len({row["reference_id"] for row in rows}) == 1
    assert {row["metadata"]["source_uri"] for row in rows} == {"https://a/x"}
    assert [row["chunk_id"] for row in rows] == [
        f"{rows[0]['reference_id']}-1",
        f"{rows[0]['reference_id']}-2",
    ]


def test_a_repeated_passage_is_dropped_but_a_new_angle_is_kept() -> None:
    rows = web_context_rows(
        [
            _hit("https://a/x", "same"),
            _hit("https://a/x", "same"),
            _hit("https://a/x", "different"),
        ]
    )

    assert [row["content"] for row in rows] == ["same", "different"]


def test_a_web_passage_says_it_came_from_the_web() -> None:
    (row,) = web_context_rows([_hit("https://a/x", "body", published_date="2026-01-01")])

    assert row["metadata"]["source_type"] == "web_search"
    assert row["metadata"]["source_uri"] == "https://a/x"
    assert row["metadata"]["source_download_locator"] == "https://a/x"
    assert row["metadata"]["published_date"] == "2026-01-01"
    assert row["_workspace"] == "__web_search__"


def test_a_remote_image_is_carried_but_not_yet_shown() -> None:
    (row,) = web_context_rows([_hit("https://a/x", "body", image_url="https://a/pic.png")])

    assert row["metadata"]["remote_image_url"] == "https://a/pic.png"
    assert "image_url" not in row


def test_an_empty_passage_never_becomes_a_source() -> None:
    assert web_context_rows([_hit("https://a/x", "   ")]) == []


def test_a_web_passage_survives_the_citation_builder_as_a_real_source() -> None:
    from dlightrag.answer.citations.source_builder import build_sources_from_chunks

    rows = web_context_rows(
        [_hit("https://a/x", "one", title="A page"), _hit("https://a/x", "two", title="A page")]
    )

    (source,) = build_sources_from_chunks(rows, image_url_prefix="/web/images")

    assert source.source_uri == "https://a/x"
    assert source.download_locator == "https://a/x"
    assert source.title is None or isinstance(source.title, str)
    assert source.chunks is not None
    assert [chunk.content for chunk in source.chunks] == ["one", "two"]
    # A remote address would be rejected by the browser's same-origin image rule,
    # so nothing is offered until it is served from here.
    assert [chunk.image_url for chunk in source.chunks] == [None, None]
    assert [chunk.thumbnail_url for chunk in source.chunks] == [None, None]
