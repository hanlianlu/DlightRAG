"""Tests for web UI conversational query rewriting."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


class TestRewriteQuery:
    """Test the _rewrite_query helper."""

    async def test_no_history_returns_original(self):
        """First turn (no history) returns message unchanged, no LLM call."""
        from dlightrag.web.routes import _rewrite_query

        llm = AsyncMock()
        result = await _rewrite_query("What is revenue?", None, llm)
        assert result == "What is revenue?"
        llm.assert_not_awaited()

    async def test_empty_history_returns_original(self):
        """Empty history list returns message unchanged."""
        from dlightrag.web.routes import _rewrite_query

        llm = AsyncMock()
        result = await _rewrite_query("What is revenue?", [], llm)
        assert result == "What is revenue?"
        llm.assert_not_awaited()

    async def test_with_history_calls_llm(self):
        """With conversation history, calls LLM and returns rewritten query."""
        from dlightrag.web.routes import _rewrite_query

        llm = AsyncMock(return_value="What were the Q3 revenue numbers?")
        history = [
            {"role": "user", "content": "Tell me about revenue"},
            {"role": "assistant", "content": "Revenue grew 20% in Q3."},
        ]
        result = await _rewrite_query("more details", history, llm)
        assert result == "What were the Q3 revenue numbers?"
        llm.assert_awaited_once()

    async def test_uses_last_20_messages(self):
        """Only last 20 messages (10 turns) are included."""
        from dlightrag.web.routes import _rewrite_query

        llm = AsyncMock(return_value="rewritten")
        history = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
        await _rewrite_query("follow up", history, llm)
        call_args = llm.call_args
        messages = call_args[1]["messages"]
        # User message is the last message in the messages list
        user_content = messages[-1]["content"]
        assert "msg 10" in user_content
        assert "msg 9" not in user_content  # trimmed

    async def test_llm_failure_propagates(self):
        """LLM failure raises (no fallback)."""
        from dlightrag.web.routes import _rewrite_query

        llm = AsyncMock(side_effect=RuntimeError("LLM down"))
        history = [{"role": "user", "content": "hello"}]
        with pytest.raises(RuntimeError, match="LLM down"):
            await _rewrite_query("follow up", history, llm)
