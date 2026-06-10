# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""E2E tests for chat flow: SSE streaming, citations, image rendering."""

from __future__ import annotations

import pytest


@pytest.mark.e2e
def test_chat_submit_streams_answer(page):
    """Submit a query via the composer and verify the AI response appears in the DOM.

    The mocked backend yields token → preview → done SSE events.
    The frontend progressively renders them into .ai-message-content.
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

    initial_user_messages = page.locator('[class*="userMessageWrapper"]').count()

    # Second query
    page.locator(".composer-input").fill("Second query")
    page.click(".composer-send")
    page.wait_for_function("document.querySelector('.composer-input').value === ''")

    # Should have at least one more user message wrapper
    final_user_messages = page.locator('[class*="userMessageWrapper"]').count()
    assert final_user_messages > initial_user_messages
