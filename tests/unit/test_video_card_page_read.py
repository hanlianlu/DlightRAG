# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Real HTTP streaming semantics, not an already-materialized tiny OG fixture."""

import socket
from functools import partial

import httpx
import pytest

from dlightrag.engine.answer.links.cards import collect_link_cards
from dlightrag.engine.public_http import fetch_public_http, fetch_public_http_prefix


class _Stream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.read = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.read += len(chunk)
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, port, *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )


async def test_large_video_page_can_declare_metadata_after_256_kib() -> None:
    # The user's real YouTube page was 1.22 MB with OG at byte ~697k.
    # Its irrelevant tail must not invalidate metadata already inside the budget.
    meta = (
        b'<meta property="og:type" content="video.other"><meta property="og:title" content="Film">'
    )
    stream = _Stream(
        [
            b"<html><head><script>" + b"x" * 700_000 + b"</script>" + meta + b"</head>",
            *([b"y" * (64 * 1024)] * 64),
        ]
    )

    def serve(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "example.com"
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        return httpx.Response(200, headers={"content-type": "text/html"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
        cards = await collect_link_cards(
            "https://example.com/watch", fetch=partial(fetch_public_http_prefix, client=client)
        )
    assert [card.title for card in cards] == ["Film"]
    assert stream.closed
    assert stream.read < 3 * 1024 * 1024, "do not drain the irrelevant page tail"


async def test_prefix_closes_at_cap_but_complete_fetch_still_rejects_oversize() -> None:
    streams: list[_Stream] = []

    def serve(request: httpx.Request) -> httpx.Response:
        stream = _Stream([b"a" * 32, b"b" * 32, b"not-read"])
        streams.append(stream)
        return httpx.Response(200, headers={"content-type": "text/html"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
        prefix = await fetch_public_http_prefix(
            "https://example.com/page", max_bytes=40, client=client, agent_url=True
        )
        assert prefix.content == b"a" * 32 + b"b" * 8
        assert prefix.media_type == "text/html"
        assert prefix.final_url == "https://example.com/page"
        assert streams[0].read == 64
        assert streams[0].closed
        with pytest.raises(ValueError, match="exceeds maximum size"):
            await fetch_public_http("https://example.com/page", max_bytes=40, client=client)
        assert streams[1].closed


async def test_declarations_past_the_prefix_limit_do_not_authorize_a_card() -> None:
    stream = _Stream(
        [
            *([b"x" * (64 * 1024)] * 32),
            b'<meta property="og:type" content="video.other"><meta property="og:title" content="Too late">',
        ]
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/html"}, stream=stream
            )
        )
    ) as client:
        assert (
            await collect_link_cards(
                "https://example.com/page", fetch=partial(fetch_public_http_prefix, client=client)
            )
            == ()
        )
    assert stream.read == 2 * 1024 * 1024
    assert stream.closed


async def test_prefix_does_not_inherit_auth_cookies_or_headers() -> None:
    requests: list[httpx.Request] = []

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"ok")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(serve),
        auth=("user", "password"),
        cookies={"session": "secret"},
        headers={"x-api-key": "secret"},
    ) as client:
        result = await fetch_public_http_prefix(
            "https://example.com/page", max_bytes=40, client=client, agent_url=True
        )
    assert result.content == b"ok"
    assert len(requests) == 1
    assert requests[0].url.host == "93.184.216.34"
    assert requests[0].extensions["sni_hostname"] == "example.com"
    for name in ("authorization", "cookie", "x-api-key"):
        assert name not in requests[0].headers


@pytest.mark.parametrize(
    "target",
    [
        "http://127.0.0.1/private",
        "http://example.com/downgrade",
        "https://example.com/watch?token=secret",
    ],
)
async def test_prefix_revalidates_every_redirect(target: str) -> None:
    requests: list[httpx.Request] = []
    stream = _Stream([])

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, headers={"location": target}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
        with pytest.raises(ValueError):
            await fetch_public_http_prefix(
                "https://example.com/start", max_bytes=40, client=client, agent_url=True
            )
    assert len(requests) == 1
    assert stream.closed


@pytest.mark.parametrize("url", ["http://127.0.0.1/page", "https://example.com/?token=secret"])
async def test_prefix_rejects_private_or_credential_bearing_targets_before_fetch(url: str) -> None:
    def serve(request: httpx.Request) -> httpx.Response:
        raise AssertionError("unsafe target reached transport")

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
        with pytest.raises(ValueError):
            await fetch_public_http_prefix(url, max_bytes=40, client=client, agent_url=True)
