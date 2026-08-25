# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser-level Web workflows that replace frontend source-text assertions."""

import pytest
from playwright.sync_api import Locator, Page, expect

pytestmark = pytest.mark.e2e


def _open_ready_chat(page: Page) -> Locator:
    page.goto("/web/")
    composer = page.get_by_role("textbox", name="Message", exact=True)
    expect(composer).to_be_visible()
    expect(page.locator('[data-conversation-id][aria-current="page"]')).to_have_count(0)
    return composer


def test_answer_submission_uses_active_conversation_and_restores_saved_history(page: Page) -> None:
    composer = _open_ready_chat(page)
    query = "How does DlightRAG work?"
    composer.fill(query)

    with page.expect_request(lambda request: request.url.endswith("/web/api/answer")) as captured:
        page.get_by_label("Send").click()

    payload = captured.value.post_data_json
    assert isinstance(payload, dict)
    assert set(payload) == {
        "query",
        "workspaces",
        "conversation_id",
        "submission_id",
    }
    assert payload["query"] == query
    workspaces = payload["workspaces"]
    assert isinstance(workspaces, list)
    assert set(workspaces) == {"default", "research"}
    assert payload["conversation_id"] is None
    assert payload["submission_id"]
    expect(page.get_by_text("DlightRAG is a multimodal RAG system.", exact=True)).to_be_visible()
    page.wait_for_url("**/web/conversations/*")
    expect(page.locator('[data-conversation-id][aria-current="page"]')).to_have_count(1)

    page.reload()

    chat_messages = page.locator("#chat-messages")
    expect(chat_messages.get_by_text(query, exact=True)).to_be_visible()
    expect(
        chat_messages.get_by_text("DlightRAG is a multimodal RAG system.", exact=True)
    ).to_be_visible()


def test_composing_line_break_does_not_submit_but_plain_line_break_does(page: Page) -> None:
    composer = _open_ready_chat(page)
    composer.fill("IME draft")
    answer_requests: list[str] = []
    page.on(
        "request",
        lambda request: (
            answer_requests.append(request.url) if request.url.endswith("/web/api/answer") else None
        ),
    )

    composer.evaluate(
        """element => element.dispatchEvent(new InputEvent('beforeinput', {
            bubbles: true,
            cancelable: true,
            inputType: 'insertLineBreak',
            isComposing: true,
        }))"""
    )
    page.wait_for_timeout(100)
    assert answer_requests == []
    expect(composer).to_have_value("IME draft")

    with page.expect_request(lambda request: request.url.endswith("/web/api/answer")):
        composer.evaluate(
            """element => element.dispatchEvent(new InputEvent('beforeinput', {
                bubbles: true,
                cancelable: true,
                inputType: 'insertLineBreak',
                isComposing: false,
            }))"""
        )

    expect(page.get_by_text("DlightRAG is a multimodal RAG system.", exact=True)).to_be_visible()


def test_mobile_shell_tracks_viewport_height(page: Page) -> None:
    page.set_viewport_size({"width": 390, "height": 700})
    page.goto("/web/")

    for height in (700, 560):
        page.set_viewport_size({"width": 390, "height": height})
        dimensions = page.evaluate(
            """() => ({
                viewport: window.innerHeight,
                app: document.querySelector('.app')?.getBoundingClientRect().height,
                panel: document.querySelector('#panel')?.getBoundingClientRect().height,
            })"""
        )
        assert dimensions == {"viewport": height, "app": height, "panel": height}


def test_soft_shell_keeps_all_structural_chrome_square_on_desktop(page: Page) -> None:
    page.set_viewport_size({"width": 1440, "height": 900})
    _open_ready_chat(page)

    radii = page.evaluate(
        """() => {
            const style = selector => getComputedStyle(document.querySelector(selector));
            const radius = selector => style(selector).borderRadius;
            document.body.classList.add('sources-panel-open');
            document.querySelector('#artifact-canvas').classList.add('open');
            return {
                app: radius('.app'),
                appClip: style('.app').clipPath,
                conversations: radius('#chat-sidebar'),
                outerPanel: radius('#panel'),
                internalArtifact: radius('#artifact-canvas'),
            };
        }"""
    )
    assert radii == {
        "app": "0px",
        "appClip": "none",
        "conversations": "0px",
        "outerPanel": "0px",
        "internalArtifact": "0px",
    }


def test_soft_shell_stays_full_bleed_on_mobile(page: Page) -> None:
    page.set_viewport_size({"width": 390, "height": 700})
    _open_ready_chat(page)

    radii = page.evaluate(
        """() => {
            const style = selector => getComputedStyle(document.querySelector(selector));
            const radius = selector => style(selector).borderRadius;
            return {
                app: radius('.app'),
                appClip: style('.app').clipPath,
                conversations: radius('#chat-sidebar'),
                panel: radius('#panel'),
            };
        }"""
    )
    assert radii == {
        "app": "0px",
        "appClip": "none",
        "conversations": "0px",
        "panel": "0px",
    }
