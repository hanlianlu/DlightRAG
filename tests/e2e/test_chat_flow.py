# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""E2E tests for chat flow: durable run submission, streaming, and reload."""

from typing import Any
from urllib.parse import urlparse

import pytest
from playwright.sync_api import Page, Route


@pytest.mark.e2e
def test_chat_submit_streams_answer(page):
    """Submit a query via the composer and verify the AI response appears in the DOM.

    The mocked backend accepts a durable run, then replays its progress, token,
    and terminal events; the frontend renders them into .ai-message-content.
    """
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)

    composer = page.locator(".composer-input")
    composer.fill("What is DlightRAG?")
    page.click(".composer-send")

    # After submission the composer clears
    page.wait_for_function("document.querySelector('.composer-input').value === ''")

    # AI message container should appear with progressive content
    page.wait_for_selector(".app.has-messages", timeout=10000)
    ai_messages = page.locator('[class*="aiMessageContent"]')
    assert ai_messages.count() >= 1


@pytest.mark.e2e
def test_chat_answer_shows_text(page):
    """Verify the answer text is rendered and visible after stream completion."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)

    page.locator(".composer-input").fill("test")
    page.click(".composer-send")

    # Wait for text content to appear in any AI message
    page.wait_for_function(
        """
        () => {
          const msgs = document.querySelectorAll('[class*="aiMessageContent"]');
          return Array.from(msgs).some(m => m.textContent.trim().length > 0);
        }
        """,
        timeout=15000,
    )

    ai_block = page.locator('[class*="aiMessageContent"]').first
    assert "DlightRAG" in ai_block.text_content()


@pytest.mark.e2e
def test_chat_history_appends_turns(page):
    """Verify that submitting a second query adds another user-message to the DOM."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)

    # First query
    page.locator(".composer-input").fill("First query")
    page.click(".composer-send")
    page.wait_for_function("document.querySelector('.composer-input').value === ''")
    page.wait_for_selector(".app.has-messages", timeout=10000)
    # One answer at a time: wait for the first stream to finish (the Send button
    # returns from its Stop state) before submitting the next query.
    page.wait_for_selector(".composer-send:not(.is-stop)", timeout=10000)

    initial_user_messages = page.locator('[class*="userMessageWrapper"]').count()

    # Second query
    page.locator(".composer-input").fill("Second query")
    page.click(".composer-send")
    page.wait_for_function("document.querySelector('.composer-input').value === ''")

    # Should have at least one more user message wrapper
    final_user_messages = page.locator('[class*="userMessageWrapper"]').count()
    assert final_user_messages > initial_user_messages


@pytest.mark.e2e
def test_chat_submit_keeps_open_panel_visible(page):
    """Submitting a query should not dismiss a panel the user already opened."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)

    page.click("#files-btn")
    page.wait_for_function("document.querySelector('#panel').classList.contains('open')")

    page.locator(".composer-input").fill("Keep the panel open")
    page.click(".composer-send")

    page.wait_for_function("document.querySelector('.composer-input').value === ''")
    assert page.locator("#panel").evaluate("el => el.classList.contains('open')")
    assert page.locator("body").evaluate("el => el.classList.contains('panel-open')")


@pytest.mark.e2e
def test_reloading_recovers_the_answer_from_the_run_not_the_response(page):
    """The 202 response is not the answer; the reloaded page reads the run."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)
    page.locator(".composer-input").fill("What is DlightRAG?")
    page.click(".composer-send")
    page.wait_for_function(
        """
        () => Array.from(document.querySelectorAll('[class*="aiMessageContent"]'))
            .some(node => node.textContent.includes('DlightRAG is a multimodal'))
        """,
        timeout=15000,
    )

    page.reload()

    # Nothing from the original response survives a reload, so this text can only
    # have come from the run the conversation still links to.
    page.wait_for_function(
        """
        () => Array.from(document.querySelectorAll('[class*="aiMessageContent"]'))
            .some(node => node.textContent.includes('DlightRAG is a multimodal'))
        """,
        timeout=15000,
    )
    assert page.locator('[class*="userMessageWrapper"]').count() == 1


@pytest.mark.e2e
def test_a_replayed_submission_never_creates_a_second_turn(page, e2e_base_url):
    """Resending one submission id returns the run it already created."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)
    page.locator(".composer-input").fill("What is DlightRAG?")
    page.click(".composer-send")
    page.wait_for_selector(".composer-send:not(.is-stop)", timeout=15000)

    replay = page.evaluate(
        """
        async () => {
          const history = await (await fetch('/web/api/conversations')).json();
          const conversation = history[0].conversation_id;
          const turns = await (
            await fetch(`/web/api/conversations/${conversation}/history`)
          ).json();
          const turn = turns.turns[0];
          const response = await fetch('/web/api/answer', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              query: 'What is DlightRAG?',
              workspaces: ['default'],
              conversation_id: conversation,
              submission_id: turn.submission_id,
            }),
          });
          const descriptor = await response.json();
          const after = await (
            await fetch(`/web/api/conversations/${conversation}/history`)
          ).json();
          return {
            status: response.status,
            runId: descriptor.run_id,
            originalRunId: turn.answer_run_id,
            turnCount: after.turns.length,
          };
        }
        """
    )

    assert replay["status"] == 202
    assert replay["runId"] == replay["originalRunId"]
    assert replay["turnCount"] == 1


def _frame(sequence: int, event_type: str, data: str) -> str:
    return f"id: {sequence}\nevent: {event_type}\ndata: {data}\n\n"


_PROGRESS = _frame(1, "progress", '{"phase": "planning"}')
_TOKENS = [_frame(2, "token", '"DlightRAG is a "'), _frame(3, "token", '"multimodal RAG system."')]
_DONE = _frame(
    4,
    "done",
    '{"status": "succeeded", "html": "<p>DlightRAG is a multimodal RAG system.</p>",'
    ' "answer": "DlightRAG is a multimodal RAG system.", "answer_images": []}',
)
#: What the server actually renders: the answer inside its ``#answer-content`` host.
_DONE_RENDERED = _frame(
    4,
    "done",
    '{"status": "succeeded", "html": "<div id=\\"answer-content\\">'
    '<p>DlightRAG is a multimodal RAG system.</p></div>",'
    ' "answer": "DlightRAG is a multimodal RAG system.", "answer_images": []}',
)


def _install_event_transport(
    page: Page,
    slices: list[list[str]],
    service: Any = None,
) -> dict[str, Any]:
    """Serve one run's durable event log as a scripted sequence of dropped connections.

    Each entry is the frames one connection delivers before the transport ends
    without a terminal event. Entries beyond the script repeat the last one, so a
    test can express either separated drops that make progress or a barren stall.
    Serving a terminal frame also records the run's terminal state, because a
    ``done`` event only ever reaches a browser after the run committed it.
    """
    state: dict[str, Any] = {"attempts": 0, "cursors": [], "cancelled": 0}

    def handle(route: Route) -> None:
        index = state["attempts"]
        state["attempts"] = index + 1
        state["cursors"].append(route.request.headers.get("last-event-id"))
        frames = slices[index] if index < len(slices) else slices[-1]
        if service is not None and any("event: done" in frame for frame in frames):
            service.finish_run(urlparse(route.request.url).path.split("/")[4])
        route.fulfill(
            status=200,
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
            body="".join(frames),
        )

    def count_cancel(route: Route) -> None:
        if route.request.method == "DELETE":
            state["cancelled"] += 1
        route.continue_()

    page.route("**/web/api/answer/*/events", handle)
    page.route("**/web/api/answer/*", count_cancel)
    return state


def _submit(page: Page, query: str) -> None:
    page.locator(".composer-input").fill(query)
    page.click(".composer-send")
    page.wait_for_function("document.querySelector('.composer-input').value === ''")


@pytest.mark.e2e
def test_navigating_away_from_a_pending_run_detaches_instead_of_blocking(page: Page) -> None:
    """Following a durable run is this tab's business, not the conversation's."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)
    # Every attempt delivers a new sequence, so the run stays pending and
    # followed for the whole test instead of exhausting the reconnect budget.
    transport = _install_event_transport(
        page,
        [[_frame(index + 1, "progress", '{"phase": "planning"}')] for index in range(200)],
    )

    first = page.locator("[aria-current='page']").get_attribute("data-conversation-id")
    page.locator("#new-conversation-btn").click()
    page.wait_for_function(
        "id => document.querySelector('[aria-current=\"page\"]')?.dataset.conversationId !== id",
        arg=first,
    )
    second = page.locator("[aria-current='page']").get_attribute("data-conversation-id")
    _submit(page, "What is DlightRAG?")
    page.wait_for_selector(".composer-send.is-stop", timeout=10000)

    # Following must leave the conversation shell usable.
    assert page.locator("#new-conversation-btn").is_enabled()
    page.locator(f'[data-conversation-id="{first}"]').get_by_role("button").first.click(
        timeout=2000
    )

    page.wait_for_function(
        "id => document.querySelector('[aria-current=\"page\"]')?.dataset.conversationId === id",
        arg=first,
        timeout=5000,
    )
    page.wait_for_selector(".composer-send:not(.is-stop)", timeout=5000)
    assert page.locator('[class*="userMessageWrapper"]').count() == 0
    assert transport["cancelled"] == 0
    # The run the other conversation owns is untouched and still pending.
    status = page.evaluate(
        "id => fetch(`/web/api/conversations/${id}/history`)"
        ".then(r => r.json()).then(h => h.turns[0].status)",
        second,
    )
    assert status in ("queued", "running")


@pytest.mark.e2e
def test_separated_drops_that_make_progress_never_exhaust_the_reconnect_budget(
    page: Page,
    e2e_conversation_service: Any,
) -> None:
    """The budget bounds consecutive barren attempts, not total reconnects."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)
    script = [[_PROGRESS], *[[token] for token in _TOKENS], [], [], [], [], [], [_DONE]]
    transport = _install_event_transport(page, script, e2e_conversation_service)

    _submit(page, "What is DlightRAG?")

    page.wait_for_function(
        """
        () => Array.from(document.querySelectorAll('[class*="aiMessageContent"]'))
            .some(node => node.textContent.includes('DlightRAG is a multimodal RAG system.'))
        """,
        timeout=20000,
    )
    page.wait_for_selector(".composer-send:not(.is-stop)", timeout=10000)
    answer = page.evaluate(
        """() => Array.from(document.querySelectorAll('[class*="aiMessageContent"]'))
            .map(node => node.textContent).join('')"""
    )
    assert answer.count("DlightRAG is a multimodal RAG system.") == 1
    assert transport["attempts"] == len(script)
    # Every reconnect resumed after the last durable sequence it consumed.
    assert transport["cursors"][:4] == [None, "1", "2", "3"]


@pytest.mark.e2e
def test_a_run_still_pending_after_the_budget_offers_an_explicit_reconnect(page: Page) -> None:
    """Exhausting the budget is a recoverable connection error, not a dead spinner."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)
    transport = _install_event_transport(page, [[]])

    _submit(page, "What is DlightRAG?")

    page.get_by_role("button", name="Reconnect").wait_for(timeout=20000)
    page.wait_for_selector(".composer-send:not(.is-stop)", timeout=10000)
    assert transport["cancelled"] == 0


@pytest.mark.e2e
def test_a_second_failed_reconnect_replaces_the_offer_instead_of_stacking_it(
    page: Page,
    e2e_conversation_service: Any,
) -> None:
    """One offer at a time, and a reconnect that succeeds leaves none behind."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)
    # Six barren attempts exhaust the budget, so the first twelve strand the run
    # twice; the thirteenth delivers the run's terminal event.
    _install_event_transport(
        page, [*([[]] * 12), [_PROGRESS, _DONE_RENDERED]], e2e_conversation_service
    )

    _submit(page, "What is DlightRAG?")

    reconnect = page.get_by_role("button", name="Reconnect")
    reconnect.wait_for(timeout=20000)
    reconnect.click()
    page.wait_for_selector(".composer-send.is-stop", timeout=10000)
    page.wait_for_selector(".composer-send:not(.is-stop)", timeout=20000)
    assert reconnect.count() == 1
    assert page.get_by_text("Connection lost. This answer is still running.").count() == 1

    reconnect.click()
    page.wait_for_function(
        """
        () => Array.from(document.querySelectorAll('[class*="aiMessageContent"]'))
            .some(node => node.textContent.includes('DlightRAG is a multimodal RAG system.'))
        """,
        timeout=20000,
    )
    assert page.get_by_role("button", name="Reconnect").count() == 0
    assert page.get_by_text("Connection lost. This answer is still running.").count() == 0
