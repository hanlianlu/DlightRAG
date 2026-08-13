# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""E2E tests for chat flow: durable run submission, streaming, and reload."""

import pytest


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
          const history = await (await fetch('/web/conversations')).json();
          const conversation = history[0].conversation_id;
          const turns = await (
            await fetch(`/web/conversations/${conversation}/history`)
          ).json();
          const turn = turns.turns[0];
          const response = await fetch('/web/answer', {
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
            await fetch(`/web/conversations/${conversation}/history`)
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

    assert replay["status"] == 200
    assert replay["runId"] == replay["originalRunId"]
    assert replay["turnCount"] == 1
