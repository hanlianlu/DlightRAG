# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Publishers outside the qualification examples use one registry-driven path."""

import json
from urllib.parse import parse_qs, urlsplit

import pytest

from dlightrag.adapters.http.browser.presentation import build_answer_presentation
from dlightrag.adapters.http.browser.video_playback import (
    resolve_video_playback,
    video_playback_link,
)
from dlightrag.engine.public_http import PublicHttpFetch


@pytest.mark.parametrize(
    ("url", "provider", "endpoint", "frame"),
    [
        (
            "https://www.dailymotion.com/video/xfixture",
            "Dailymotion",
            "https://www.dailymotion.com/services/oembed",
            "https://geo.dailymotion.com/player.html?video=xfixture",
        ),
        (
            "https://www.ted.com/talks/fixture",
            "TED",
            "https://www.ted.com/services/v1/oembed.json",
            "https://embed.ted.com/talks/fixture",
        ),
        (
            "https://example.wistia.com/medias/fixture",
            "Wistia, Inc.",
            "https://fast.wistia.com/oembed.json",
            "https://fast.wistia.net/embed/iframe/fixture",
        ),
        (
            "https://www.loom.com/share/fixture",
            "Loom",
            "https://www.loom.com/v1/oembed",
            "https://www.loom.com/embed/fixture",
        ),
    ],
)
async def test_registry_publishers_use_generic_video_resolution_without_site_adapters(
    url, provider, endpoint, frame
):
    calls = []

    async def fetch(destination, **kwargs):
        calls.append(destination)
        assert destination.startswith(endpoint + "?")
        assert parse_qs(urlsplit(destination).query) == {"url": [url], "format": ["json"]}
        assert kwargs == {"max_bytes": 65536, "timeout": 4.0, "agent_url": True}
        payload = {
            "type": "video",
            "html": f'<iframe src="{frame}" onload="evil()"></iframe><script>evil()</script>',
            "width": 640,
            "height": 480,
        }
        return PublicHttpFetch(
            json.dumps(payload).encode(), destination, "application/json; charset=utf-8", 200
        )

    presentation = build_answer_presentation(answer=f"Watch {url}.", sources=[], evidence_images=[])
    assert len(presentation.video_links) == 1
    assert presentation.video_links[0].provider == provider
    assert presentation.link_cards == []
    assert "iframe" not in presentation.parts[0].html
    player = await resolve_video_playback(url, fetch=fetch)
    assert player is not None
    assert player.embed_url == frame
    assert player.aspect_ratio == 4 / 3
    assert "evil" not in player.model_dump_json()
    assert len(calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {
            "type": "photo",
            "html": '<iframe src="https://www.dailymotion.com/embed/video/fixture"></iframe>',
        },
        {"type": "rich", "html": '<script src="https://www.dailymotion.com/sdk.js"></script>'},
        {"type": "video", "html": '<iframe src="https://evil.example/player"></iframe>'},
        {
            "type": "video",
            "html": '<iframe src="https://www.dailymotion.com.evil.example/player"></iframe>',
        },
        {
            "type": "video",
            "html": '<iframe src="https://user:secret@www.dailymotion.com/embed/video/fixture"></iframe>',
        },
        {
            "type": "video",
            "html": '<iframe src="http://www.dailymotion.com/embed/video/fixture"></iframe>',
        },
        {"type": "video", "html": '<iframe src="https://127.0.0.1/player"></iframe>'},
        {
            "type": "video",
            "html": '<iframe src="https://www.dailymotion.com/a"></iframe><iframe src="https://www.dailymotion.com/b"></iframe>',
        },
    ],
)
async def test_registry_response_cannot_authorize_non_video_or_unrelated_frames(payload):
    async def fetch(url, **kwargs):
        return PublicHttpFetch(json.dumps(payload).encode(), url, "application/json", 200)

    assert (
        await resolve_video_playback("https://www.dailymotion.com/video/fixture", fetch=fetch)
        is None
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://www.dailymotion.com.evil.example/video/fixture",
        "https://evil.example/path/.wistia.com/medias/fixture",
        "https://evil.example/path/www.ted.com/talks/fixture",
        "https://www.dailymotion.com:444/video/fixture",
        "https://user:secret@www.dailymotion.com/video/fixture",
        "https://www.dailymotion.com/video/fixture?access_token=secret",
    ],
)
async def test_registry_schemes_do_not_treat_paths_as_provider_authorities(url):
    async def fetch(*args, **kwargs):
        pytest.fail("unrecognized URLs cannot reach a provider endpoint")

    assert video_playback_link(url) is None
    assert await resolve_video_playback(url, fetch=fetch) is None


async def test_registry_candidates_fall_back_to_the_link_when_metadata_is_unavailable():
    async def fetch(*args, **kwargs):
        raise TimeoutError

    url = "https://www.dailymotion.com/video/fixture"
    assert video_playback_link(url) is not None
    assert await resolve_video_playback(url, fetch=fetch) is None
