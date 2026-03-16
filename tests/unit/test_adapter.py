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
    async def test_lightrag_kwargs_not_forwarded(self):
        """LightRAG-internal kwargs are accepted explicitly and not forwarded."""
        inner = AsyncMock(return_value="ok")
        adapted = _adapt_for_lightrag(inner)

        await adapted(
            "test",
            keyword_extraction=True,
            token_tracker=MagicMock(),
            use_azure=False,
            enable_cot=True,
            hashing_kv=MagicMock(),
            azure_deployment="my-deploy",
            api_version="2024-02",
            temperature=0.5,  # should be preserved
        )
        call_kwargs = inner.call_args.kwargs
        assert "keyword_extraction" not in call_kwargs
        assert "token_tracker" not in call_kwargs
        assert "use_azure" not in call_kwargs
        assert "enable_cot" not in call_kwargs
        assert "hashing_kv" not in call_kwargs
        assert "azure_deployment" not in call_kwargs
        assert "api_version" not in call_kwargs
        assert call_kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_history_messages_incorporated(self):
        """history_messages are inserted between system prompt and user message."""
        inner = AsyncMock(return_value="ok")
        adapted = _adapt_for_lightrag(inner)

        history = [
            {"role": "user", "content": "prior question"},
            {"role": "assistant", "content": "prior answer"},
        ]
        await adapted("new question", system_prompt="Be helpful", history_messages=history)

        messages = inner.call_args.kwargs["messages"]
        assert len(messages) == 4
        assert messages[0] == {"role": "system", "content": "Be helpful"}
        assert messages[1] == {"role": "user", "content": "prior question"}
        assert messages[2] == {"role": "assistant", "content": "prior answer"}
        assert messages[3] == {"role": "user", "content": "new question"}
        assert "history_messages" not in inner.call_args.kwargs

    @pytest.mark.asyncio
    async def test_forwards_stream_and_response_format(self):
        """Non-LightRAG kwargs like stream and response_format pass through."""
        inner = AsyncMock(return_value="ok")
        adapted = _adapt_for_lightrag(inner)

        await adapted("test", stream=True, response_format={"type": "json_object"})
        call_kwargs = inner.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["response_format"] == {"type": "json_object"}
