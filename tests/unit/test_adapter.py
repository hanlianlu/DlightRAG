"""Tests for _adapt_for_lightrag wrapper."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.models.llm import _adapt_for_lightrag


class TestAdaptForLightrag:
    @pytest.mark.asyncio
    async def test_prompt_to_messages(self):
        """Converts (prompt, system_prompt) to messages array."""
        inner = AsyncMock(return_value="response")
        adapted = _adapt_for_lightrag(inner)

        result = await adapted("What is AI?", system_prompt="You are helpful")
        assert result == "response"

        call_args = inner.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are helpful"}
        assert messages[1] == {"role": "user", "content": "What is AI?"}

    @pytest.mark.asyncio
    async def test_no_system_prompt(self):
        inner = AsyncMock(return_value="response")
        adapted = _adapt_for_lightrag(inner)

        await adapted("Hello")
        messages = inner.call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_strips_lightrag_kwargs(self):
        """LightRAG-internal kwargs are stripped before forwarding."""
        inner = AsyncMock(return_value="ok")
        adapted = _adapt_for_lightrag(inner)

        await adapted(
            "test",
            keyword_extraction="foo",
            token_tracker=MagicMock(),
            use_azure=False,
            temperature=0.5,  # should be preserved
        )
        call_kwargs = inner.call_args.kwargs
        assert "keyword_extraction" not in call_kwargs
        assert "token_tracker" not in call_kwargs
        assert "use_azure" not in call_kwargs
        assert call_kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_hashing_kv_cache_miss(self):
        """hashing_kv: cache miss -> call inner -> store result."""
        inner = AsyncMock(return_value="computed result")
        adapted = _adapt_for_lightrag(inner)

        mock_kv = AsyncMock()
        mock_kv.get_by_id.return_value = None  # cache miss

        result = await adapted("test", hashing_kv=mock_kv)
        assert result == "computed result"
        inner.assert_called_once()
        mock_kv.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_hashing_kv_cache_hit(self):
        """hashing_kv: cache hit -> return cached, don't call inner."""
        inner = AsyncMock()
        adapted = _adapt_for_lightrag(inner)

        mock_kv = AsyncMock()
        mock_kv.get_by_id.return_value = {"return": "cached result"}

        result = await adapted("test", hashing_kv=mock_kv)
        assert result == "cached result"
        inner.assert_not_called()

    @pytest.mark.asyncio
    async def test_forwards_stream_and_response_format(self):
        """Non-LightRAG kwargs like stream and response_format pass through."""
        inner = AsyncMock(return_value="ok")
        adapted = _adapt_for_lightrag(inner)

        await adapted("test", stream=True, response_format={"type": "json_object"})
        call_kwargs = inner.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["response_format"] == {"type": "json_object"}
