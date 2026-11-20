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


def test_a_reload_restores_the_saved_question_and_answer(page: Page) -> None:
    composer = _open_ready_chat(page)
    query = "How does DlightRAG work?"
    composer.fill(query)

    with page.expect_request(lambda request: request.url.endswith("/web/api/answer")) as captured:
        page.get_by_label("Send").click()

    # The answer searches what the scope picker shows: here, every workspace.
    payload = captured.value.post_data_json
    assert isinstance(payload, dict)
    assert payload["query"] == query
    assert set(payload["workspaces"]) == {"default", "research"}
    expect(page.get_by_text("DlightRAG is a durable agent.", exact=True)).to_be_visible()
    page.wait_for_url("**/web/conversations/*")
    expect(page.locator('[data-conversation-id][aria-current="page"]')).to_have_count(1)

    page.reload()

    chat_messages = page.locator("#chat-messages")
    expect(chat_messages.get_by_text(query, exact=True)).to_be_visible()
    expect(chat_messages.get_by_text("DlightRAG is a durable agent.", exact=True)).to_be_visible()


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

    expect(page.get_by_text("DlightRAG is a durable agent.", exact=True)).to_be_visible()


def test_mobile_shell_tracks_viewport_height(page: Page) -> None:
    page.set_viewport_size({"width": 390, "height": 700})
    page.goto("/web/")
    page.wait_for_selector(".app", timeout=10000)

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
