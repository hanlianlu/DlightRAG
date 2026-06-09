# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""E2E tests for chat flow: SSE streaming, citations, image rendering."""

from __future__ import annotations

import pytest


@pytest.mark.e2e
def test_chat_submit_streams_answer(_page):
    """Submit a query and verify SSE streaming produces answer content."""
    # TODO: Wire app transport, then:
    # page.goto("/")
    # page.fill(".composer-input", "What is DlightRAG?")
    # page.click(".composer-send")
    # page.wait_for_selector(".ai-message-content")
    # assert page.locator(".ai-message-content").text_content() != ""


@pytest.mark.e2e
def test_chat_answer_shows_citations(_page):
    """Verify citation badges render after answer completion."""
    pass


@pytest.mark.e2e
def test_chat_history_appends_turns(_page):
    """Verify conversation history accumulates in the DOM."""
    pass
