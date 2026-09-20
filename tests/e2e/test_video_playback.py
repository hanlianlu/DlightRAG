# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Chromium coverage for in-card external video Play: native keyboard and contrast."""

from urllib.parse import urlparse

import pytest
from playwright.sync_api import Page, Route, expect

pytestmark = pytest.mark.e2e

_TIMESTAMP = "2026-08-20T12:00:00Z"
_CONVERSATION_ID = "video-playback-history"
_RUN_ID = "video-playback-run"
_VIDEO_URL = "https://www.youtube.com/watch?v=abcdefghijk"
_PLAYER = "https://www.youtube-nocookie.com/embed/abcdefghijk?autoplay=1&playsinline=1"
_CONVERSATION = {
    "conversation_id": _CONVERSATION_ID,
    "title": "Video",
    "created_at": _TIMESTAMP,
    "updated_at": _TIMESTAMP,
}


def _presentation_wire() -> dict[str, object]:
    return {
        "answer_text": f"Watch {_VIDEO_URL}",
        "parts": [
            {
                "type": "markdown",
                "text": "",
                "html": f'<p><a href="{_VIDEO_URL}" target="_blank">Watch this</a></p>',
                "artifact": None,
                "evidence_image": None,
                "card": None,
                "inline": False,
            }
        ],
        "video_links": [
            {
                "url": _VIDEO_URL,
                "provider": "YouTube",
                "player_domains": ["www.youtube-nocookie.com", "youtube.com"],
            }
        ],
        "link_cards": [],
        "sources": [],
        "evidence_images": [],
        "artifacts": [],
        "artifact_outcome": {"status": "complete", "issues": []},
    }


def _turn(presentation: dict[str, object]) -> dict[str, object]:
    return {
        "turn_id": "video-turn",
        "turn_number": 1,
        "answer_run_id": _RUN_ID,
        "submission_id": "video-submission",
        "status": "succeeded",
        "cancel_requested": False,
        "user_text": "Watch this",
        "assistant_text": str(presentation["answer_text"]),
        "user_attachments": [],
        "presentation": presentation,
        "usage": {},
        "evidence": {},
        "error_kind": None,
        "error_message": None,
        "created_at": _TIMESTAMP,
    }


def _flush_paint(page: Page) -> None:
    page.evaluate("() => new Promise((resolve) => requestAnimationFrame(() => resolve()))")
    page.evaluate("() => new Promise((resolve) => requestAnimationFrame(() => resolve()))")


def _install_card(page: Page) -> list[str]:
    playback_calls: list[str] = []

    def conversations(route: Route) -> None:
        path = urlparse(route.request.url).path
        if path == "/web/api/conversations":
            route.fulfill(json={"items": [_CONVERSATION], "next_cursor": None})
            return
        if path == f"/web/api/conversations/{_CONVERSATION_ID}/history":
            route.fulfill(
                json={"conversation": _CONVERSATION, "turns": [_turn(_presentation_wire())]}
            )
            return
        route.continue_()

    def playback(route: Route) -> None:
        playback_calls.append(route.request.url)
        route.fulfill(json={"embed_url": _PLAYER, "aspect_ratio": 16 / 9})

    page.route("**/web/api/conversations**", conversations)
    page.route("**/web/api/video-playback", playback)
    # Exercise frame placement without contacting a real media provider.
    player_origin = urlparse(_PLAYER)
    page.route(
        f"{player_origin.scheme}://{player_origin.netloc}/**",
        lambda route: route.fulfill(content_type="text/html", body="<p>Player fixture</p>"),
    )
    return playback_calls


def _open_card(page: Page) -> None:
    with page.expect_response(
        lambda response: response.url.endswith("/history") and response.ok,
        timeout=10000,
    ):
        page.goto(f"/web/conversations/{_CONVERSATION_ID}")
    page.wait_for_selector("[data-video-card] [data-video-play]", timeout=10000)


def test_native_space_and_enter_activate_play_once(page: Page) -> None:
    calls = _install_card(page)
    _open_card(page)
    play = page.locator("[data-video-card] [data-video-play]")
    expect(play).to_have_count(1)
    assert calls == []
    media_box = page.locator("[data-video-media]").bounding_box()
    play_box = play.bounding_box()
    assert media_box is not None and play_box is not None
    assert media_box["height"] >= 200
    assert play_box["width"] == pytest.approx(media_box["width"], abs=1)
    assert play_box["height"] == pytest.approx(media_box["height"], abs=1)

    play.focus()
    expect(play).to_be_focused()
    page.keyboard.down("Space")
    _flush_paint(page)
    assert calls == []
    expect(play).to_have_count(1)

    page.keyboard.up("Space")
    page.locator("[data-video-card] iframe[data-external-video]").wait_for(timeout=10000)
    assert len(calls) == 1
    expect(play).to_have_count(0)
    frame_box = page.locator("[data-external-video]").bounding_box()
    assert frame_box is not None
    for dimension in ("x", "y", "width", "height"):
        assert frame_box[dimension] == pytest.approx(media_box[dimension], abs=1)

    page.locator("[data-video-stop]").click()
    play.wait_for(timeout=10000)
    calls.clear()
    play.focus()
    expect(play).to_be_focused()
    page.keyboard.down("Enter")
    page.locator("[data-video-card] iframe[data-external-video]").wait_for(timeout=10000)
    assert len(calls) == 1
    page.keyboard.up("Enter")
    _flush_paint(page)
    assert len(calls) == 1
    expect(play).to_have_count(0)


def test_play_icon_uses_on_scrim_color_in_light_theme(page: Page) -> None:
    page.emulate_media(color_scheme="light")
    _install_card(page)
    _open_card(page)
    expect(page.locator("html")).to_have_attribute("data-color-mode", "light")

    measured = page.locator("[data-video-play]").evaluate(
        """(play) => {
          const overlay = play.querySelector('svg')?.parentElement;
          if (!overlay) return null;
          const probe = document.createElement('span');
          play.append(probe);
          const token = (name, property) => {
            probe.style.color = '';
            probe.style.backgroundColor = '';
            probe.style[property] = `var(${name})`;
            return getComputedStyle(probe)[property];
          };
          const caption = token('--color-image-caption', 'color');
          const primary = token('--color-text-primary', 'color');
          const scrim = token('--color-scrim-strong', 'backgroundColor');
          probe.remove();
          const icon = getComputedStyle(overlay);
          return {
            color: icon.color,
            background: icon.backgroundColor,
            caption,
            primary,
            scrim,
          };
        }"""
    )
    assert measured is not None
    assert measured["color"] == measured["caption"]
    assert measured["color"] != measured["primary"]
    assert measured["background"] == measured["scrim"]
