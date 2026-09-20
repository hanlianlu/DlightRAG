# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Reader-activated playback is independent of settlement metadata."""

import pytest

from dlightrag.adapters.http.browser import video_playback
from dlightrag.adapters.http.browser.presentation import build_answer_presentation
from dlightrag.engine.public_http import PublicHttpFetch


def test_metadata_free_video_links_offer_playback_without_rewriting_the_answer():
    url = "https://www.youtube.com/watch?v=abcdefghijk"
    answer = f"Watch {url}. `https://vimeo.com/123456`"
    presentation = build_answer_presentation(answer=answer, sources=[], evidence_images=[])
    wire = presentation.model_dump()
    assert wire["video_links"] == [
        {
            "url": url,
            "provider": "YouTube",
            "player_domains": ["www.youtube-nocookie.com", "youtube.com"],
        }
    ]
    assert presentation.answer_text == answer
    assert presentation.link_cards == []
    assert "<iframe" not in presentation.parts[0].html


@pytest.mark.parametrize("query", ["", "?text=介绍"])
def test_published_citation_keeps_its_role_when_a_recommendation_shares_its_url(query):
    from markdown_it.common.normalize_url import normalizeLink

    from dlightrag.engine.answer.citations.contracts import SourceReference
    from dlightrag.engine.answer.citations.projection import link_public_citations

    url = "https://vimeo.com/123456" + query
    source = SourceReference(
        id="9", title="Video", source_uri=url, workspace="default", download_locator=url
    )
    projected = link_public_citations("Fact [9] and [9-1].", [source])
    sources = [{"id": "9", "title": "Video", "type": "web", "source_uri": url}]
    citation_only = build_answer_presentation(answer=projected, sources=sources, evidence_images=[])
    assert citation_only.video_links == []
    assert citation_only.parts[0].html.count('class="answer-citation-link"') == 2
    assert ">9</a>" in citation_only.parts[0].html
    assert ">9-1</a>" in citation_only.parts[0].html
    assert "<cite" not in citation_only.parts[0].html

    mixed = build_answer_presentation(
        answer=projected + f" Watch [the video]({url}).",
        sources=sources,
        evidence_images=[],
    )
    assert [link.url for link in mixed.video_links] == [normalizeLink(url)]
    assert mixed.parts[0].html.count('class="answer-citation-link"') == 2
    assert ">the video</a>" in mixed.parts[0].html


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("![9](https://example.com/thumb.png)", '<img src="https://example.com/thumb.png"'),
        ("**9**", "<strong>9</strong>"),
        ("`9`", "<code>9</code>"),
    ],
)
def test_numeric_rich_link_labels_are_not_replaced_by_citation_classification(label, expected):
    url = "https://example.com/article"
    presentation = build_answer_presentation(
        answer=f"[{label}]({url})",
        sources=[{"id": "9", "title": "Article", "source_uri": url}],
        evidence_images=[],
    )
    assert expected in presentation.parts[0].html
    assert 'class="answer-citation-link"' not in presentation.parts[0].html


@pytest.mark.parametrize(
    "answer",
    [
        "`https://www.youtube.com/watch?v=abcdefghijk`",
        "![poster](https://www.youtube.com/watch?v=abcdefghijk)",
        "[unused]: https://www.youtube.com/watch?v=abcdefghijk",
    ],
)
def test_non_link_occurrences_do_not_offer_playback(answer):
    presentation = build_answer_presentation(answer=answer, sources=[], evidence_images=[])
    assert presentation.model_dump()["video_links"] == []


@pytest.mark.parametrize("media_type", ["application/json", "application/json; charset=utf-8"])
async def test_clicked_video_uses_bounded_oembed_but_never_executes_provider_html(media_type):
    import json

    calls = []

    async def fetch(url, **kwargs):
        calls.append((url, kwargs))
        return PublicHttpFetch(
            json.dumps(
                {
                    "type": "video",
                    "width": 270,
                    "height": 480,
                    "html": '<iframe src="https://www.youtube.com/embed/abcdefghijk?feature=oembed" onload="evil()"></iframe>',
                }
            ).encode(),
            url,
            media_type,
            200,
        )

    player = await video_playback.resolve_video_playback(
        "https://youtu.be/abcdefghijk?si=tracking",
        fetch=fetch,
    )
    assert player is not None
    assert (
        player.embed_url
        == "https://www.youtube-nocookie.com/embed/abcdefghijk?autoplay=1&playsinline=1"
    )
    assert player.aspect_ratio == 270 / 480
    assert len(calls) == 1
    assert calls[0][1] == {"max_bytes": 65536, "timeout": 4.0, "agent_url": True}
    assert "si=" not in calls[0][0]
    assert "evil" not in player.model_dump_json()


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/channel/abcdefghijk",
        "https://www.youtube.com/playlist?list=abcdefghijk",
        "https://www.youtube.com/watch?v=abcdefghijk&v=lmnopqrstuv",
        "https://www.youtube.com/watch?v=abcdefghij_注",
        "https://www.youtube.com.evil.example/watch?v=abcdefghijk",
        "https://www.youtube.com:444/watch?v=abcdefghijk",
        "https://user:password@www.youtube.com/watch?v=abcdefghijk",
        "https://www.youtube.com/watch?v=abcdefghijk&access_token=secret",
        "https://127.0.0.1/watch?v=abcdefghijk",
        "https://[::1]/watch?v=abcdefghijk",
        "javascript:alert(1)",
        "artifact:clip.mp4",
        "https://vimeo.com/123456/private-token",
        "https://example.com/video",
    ],
)
async def test_unsupported_or_credential_urls_never_trigger_resolution(url):
    async def fetch(*args, **kwargs):
        pytest.fail("unrecognized/credential URLs must not reach transport")

    assert video_playback.video_playback_link(url) is None
    assert await video_playback.resolve_video_playback(url, fetch=fetch) is None


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://youtu.be/abcdefghijk?t=1m30s&si=tracking",
            "https://www.youtube-nocookie.com/embed/abcdefghijk?autoplay=1&playsinline=1&start=90",
        ),
        (
            "https://www.youtube.com/shorts/abcdefghijk",
            "https://www.youtube-nocookie.com/embed/abcdefghijk?autoplay=1&playsinline=1",
        ),
        (
            "https://www.youtube.com/live/abcdefghijk",
            "https://www.youtube-nocookie.com/embed/abcdefghijk?autoplay=1&playsinline=1",
        ),
        (
            "https://vimeo.com/123456#t=90s",
            "https://player.vimeo.com/video/123456?autoplay=1#t=90s",
        ),
        (
            "https://www.bilibili.com/video/BV1234567890?p=2&t=30",
            "https://player.bilibili.com/player.html?bvid=BV1234567890&autoplay=1&p=2&t=30",
        ),
    ],
)
async def test_official_mappings_survive_missing_metadata_and_preserve_playback_offsets(
    url, expected
):
    async def fetch(*args, **kwargs):
        raise TimeoutError("metadata is not playback permission")

    player = await video_playback.resolve_video_playback(url, fetch=fetch)
    assert player is not None
    assert player.embed_url == expected
    assert player.aspect_ratio == 16 / 9


@pytest.mark.parametrize(
    "markup",
    [
        '<iframe src="https://evil.example/player"></iframe>',
        '<iframe src="javascript:alert(1)"></iframe>',
        '<iframe src="https://www.youtube.com/embed/anotherfilm"></iframe>',
        '<iframe src="https://www.youtube.com/embed/abcdefghijk" src="https://evil.example"></iframe>',
        "<script>evil()</script>",
    ],
)
async def test_untrusted_oembed_cannot_change_the_selected_official_player(markup):
    import json

    async def fetch(url, **kwargs):
        return PublicHttpFetch(
            json.dumps({"type": "video", "html": markup, "width": 1, "height": 1}).encode(),
            url,
            "application/json",
            200,
        )

    player = await video_playback.resolve_video_playback(
        "https://youtu.be/abcdefghijk", fetch=fetch
    )
    assert player is not None
    assert (
        player.embed_url
        == "https://www.youtube-nocookie.com/embed/abcdefghijk?autoplay=1&playsinline=1"
    )
    assert player.aspect_ratio == 16 / 9


@pytest.mark.parametrize(
    ("status", "media_type", "body"),
    [
        (503, None, b"Unavailable"),
        (200, None, b"{}"),
        (200, "text/html", b"<html>"),
        (200, "application/json", b"invalid JSON"),
    ],
)
async def test_unavailable_or_invalid_oembed_keeps_the_official_mapping(status, media_type, body):
    async def fetch(url, **kwargs):
        return PublicHttpFetch(body, url, media_type, status)

    player = await video_playback.resolve_video_playback(
        "https://youtu.be/abcdefghijk", fetch=fetch
    )
    assert player is not None
    assert (
        player.embed_url
        == "https://www.youtube-nocookie.com/embed/abcdefghijk?autoplay=1&playsinline=1"
    )
    assert player.aspect_ratio == 16 / 9


async def test_provider_http_reads_are_anonymous_pinned_and_redirects_are_revalidated(monkeypatch):
    import socket
    from functools import partial

    import httpx

    from dlightrag.engine.public_http import fetch_public_http

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, port, *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
        ],
    )
    calls = []

    def serve(request):
        calls.append(request)
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "www.youtube.com"
        assert not any(
            name in request.headers for name in ("cookie", "authorization", "referer", "x-private")
        )
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(serve),
        headers={
            "authorization": "secret",
            "cookie": "session=secret",
            "referer": "private",
            "x-private": "secret",
        },
    ) as client:
        player = await video_playback.resolve_video_playback(
            "https://youtu.be/abcdefghijk", fetch=partial(fetch_public_http, client=client)
        )
    assert len(calls) == 1
    assert player is not None
    assert player.embed_url.startswith("https://www.youtube-nocookie.com/embed/abcdefghijk?")


def test_browser_playback_route_authentication_csrf_and_no_history_dependency():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from dlightrag.adapters.http.browser.auth import WEB_CSRF_COOKIE, WebAuthMiddleware
    from dlightrag.adapters.http.browser.routes import router
    from dlightrag.application.config import DlightragConfig

    app = FastAPI()
    app.include_router(router)
    cfg = DlightragConfig()
    app.add_middleware(WebAuthMiddleware, config_getter=lambda: cfg)
    with TestClient(app) as client:
        body = {"url": "https://www.bilibili.com/video/BV1234567890"}
        # No Application, database, Run or conversation is needed to activate a
        # reviewed public player, even on an old settled answer.
        response = client.post("/web/api/video-playback", json=body)
        assert response.status_code == 200
        assert (
            response.json()["embed_url"]
            == "https://player.bilibili.com/player.html?bvid=BV1234567890&autoplay=1"
        )
        assert (
            client.post(
                "/web/api/video-playback", json=body, headers={"Origin": "https://evil.example"}
            ).status_code
            == 403
        )
        client.cookies.set(WEB_CSRF_COOKIE, "token")
        assert client.post("/web/api/video-playback", json=body).status_code == 403
        assert (
            client.post(
                "/web/api/video-playback", json=body, headers={"X-CSRF-Token": "token"}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/web/api/video-playback",
                json={"url": "http://127.0.0.1"},
                headers={"X-CSRF-Token": "token"},
            ).status_code
            == 422
        )

    bare = FastAPI()
    bare.include_router(router)
    with TestClient(bare) as client:
        assert client.post("/web/api/video-playback", json=body).status_code == 401


@pytest.mark.parametrize("mode", ["simple", "jwt"])
def test_playback_route_requires_real_credentials_and_cookie_bound_csrf(mode):
    from datetime import UTC, datetime, timedelta

    import jwt
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from dlightrag.adapters.http.browser.auth import WEB_CSRF_COOKIE, WebAuthMiddleware
    from dlightrag.adapters.http.browser.routes import router
    from dlightrag.application.config import DlightragConfig

    secret = "fixture-secret-never-used-outside-this-test"
    cfg = DlightragConfig.model_validate(
        {
            "access": {
                "auth_mode": mode,
                "api_token": secret,
                "jwt_verification_key": secret,
            }
        }
    )
    token = (
        secret
        if mode == "simple"
        else jwt.encode(
            {"sub": "reader", "exp": datetime.now(UTC) + timedelta(minutes=5)},
            secret,
            algorithm="HS256",
        )
    )
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(WebAuthMiddleware, config_getter=lambda: cfg)
    body = {"url": "https://www.bilibili.com/video/BV1234567890"}
    auth = {"Authorization": f"Bearer {token}"}
    with TestClient(app) as client:
        assert client.post("/web/api/video-playback", json=body).status_code == 401
        assert (
            client.post(
                "/web/api/video-playback",
                json=body,
                headers={"Authorization": "Bearer invalid"},
            ).status_code
            == 401
        )
        assert client.post("/web/api/video-playback", json=body, headers=auth).status_code == 200
        # The browser gets its double-submit cookie on an authenticated GET;
        # this endpoint is intentionally POST-only, even for the same reader.
        assert client.get("/web/api/video-playback", headers=auth).status_code == 405
        csrf = client.cookies.get(WEB_CSRF_COOKIE)
        assert csrf
        assert client.post("/web/api/video-playback", json=body, headers=auth).status_code == 403
        assert (
            client.post(
                "/web/api/video-playback",
                json=body,
                headers={**auth, "X-CSRF-Token": csrf},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/web/api/video-playback",
                json=body,
                headers={**auth, "X-CSRF-Token": csrf, "Origin": "https://evil.example"},
            ).status_code
            == 403
        )
