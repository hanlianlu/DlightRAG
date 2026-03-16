"""Tests for messages-first completion callables."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestOpenAICompletion:
    @pytest.mark.asyncio
    async def test_non_streaming(self):
        from dlightrag.models.completion import _openai_completion

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello"))]
        mock_client.chat.completions.create.return_value = mock_response

        result = await _openai_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4.1-mini",
            api_key="sk-test",
            _client=mock_client,
        )
        assert result == "Hello"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4.1-mini"
        assert call_kwargs.kwargs["messages"] == [{"role": "user", "content": "Hi"}]

    @pytest.mark.asyncio
    async def test_with_response_format(self):
        from dlightrag.models.completion import _openai_completion

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"answer": "test"}'))]
        mock_client.chat.completions.create.return_value = mock_response

        await _openai_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4.1-mini",
            api_key="sk-test",
            _client=mock_client,
            response_format={"type": "json_object"},
        )
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_streaming(self):
        from dlightrag.models.completion import _openai_completion

        mock_client = AsyncMock()

        async def mock_stream():
            for text in ["Hel", "lo"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content=text))]
                yield chunk

        mock_client.chat.completions.create.return_value = mock_stream()

        result = await _openai_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4.1-mini",
            api_key="sk-test",
            _client=mock_client,
            stream=True,
        )
        chunks = [c async for c in result]
        assert chunks == ["Hel", "lo"]

    @pytest.mark.asyncio
    async def test_extra_kwargs_forwarded(self):
        from dlightrag.models.completion import _openai_completion

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_response

        await _openai_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4.1-mini",
            api_key="sk-test",
            _client=mock_client,
            temperature=0.5,
        )
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.5


class TestLiteLLMCompletion:
    @pytest.mark.asyncio
    async def test_non_streaming(self):
        from dlightrag.models.completion import _litellm_completion

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello"))]

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await _litellm_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="anthropic/claude-3",
                api_key="sk-test",
            )
        assert result == "Hello"

    @pytest.mark.asyncio
    async def test_base_url_mapped_to_api_base(self):
        from dlightrag.models.completion import _litellm_completion

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]

        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=mock_response
        ) as mock:
            await _litellm_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="ollama/qwen3:8b",
                api_key="ollama",
                base_url="http://localhost:11434",
            )
        call_kwargs = mock.call_args
        assert call_kwargs.kwargs["api_base"] == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_streaming(self):
        from dlightrag.models.completion import _litellm_completion

        async def mock_stream():
            for text in ["He", "llo"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content=text))]
                yield chunk

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_stream()):
            result = await _litellm_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="anthropic/claude-3",
                api_key="sk-test",
                stream=True,
            )
        chunks = [c async for c in result]
        assert chunks == ["He", "llo"]
