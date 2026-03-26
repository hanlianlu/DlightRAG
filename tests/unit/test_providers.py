# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for provider ABC, registry, and concrete implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.models.providers import get_provider
from dlightrag.models.providers.base import CompletionProvider


class TestCompletionProviderABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            CompletionProvider(api_key="k", timeout=10.0, max_retries=1)


class TestProviderRegistry:
    def test_get_openai_provider(self):
        p = get_provider("openai", api_key="test-key")
        assert isinstance(p, CompletionProvider)

    def test_get_anthropic_provider(self):
        p = get_provider("anthropic", api_key="test-key")
        assert isinstance(p, CompletionProvider)

    def test_get_gemini_provider(self):
        p = get_provider("gemini", api_key="test-key")
        assert isinstance(p, CompletionProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown_provider")

    def test_error_message_lists_available(self):
        with pytest.raises(ValueError, match="openai"):
            get_provider("bad")


class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_complete_extracts_system_message(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="reply")]
        with patch("dlightrag.models.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            p._client = None
            result = await p.complete(
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hi"},
                ],
                "claude-sonnet-4-20250514",
            )
            call_kwargs = MockSDK.return_value.messages.create.call_args[1]
            assert call_kwargs["system"] == "You are helpful."
            assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert result == "reply"

    @pytest.mark.asyncio
    async def test_complete_defaults_max_tokens(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        with patch("dlightrag.models.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            p._client = None
            await p.complete([{"role": "user", "content": "hi"}], "claude-sonnet-4-20250514")
            call_kwargs = MockSDK.return_value.messages.create.call_args[1]
            assert call_kwargs["max_tokens"] == 8192

    @pytest.mark.asyncio
    async def test_complete_routes_thinking_to_top_level(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="thought")]
        with patch("dlightrag.models.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            p._client = None
            await p.complete(
                [{"role": "user", "content": "hi"}],
                "claude-sonnet-4-20250514",
                model_kwargs={"thinking": {"type": "enabled", "budget_tokens": 1024}},
            )
            call_kwargs = MockSDK.return_value.messages.create.call_args[1]
            assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 1024}
            assert "extra_body" not in call_kwargs

    @pytest.mark.asyncio
    async def test_json_response_format_adds_system_instruction(self):
        """Anthropic doesn't support response_format natively; inject JSON instruction."""
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"key": "val"}')]
        with patch("dlightrag.models.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            p._client = None
            await p.complete(
                [{"role": "user", "content": "hi"}],
                "claude-sonnet-4-20250514",
                response_format={"type": "json_object"},
            )
            call_kwargs = MockSDK.return_value.messages.create.call_args[1]
            assert "json" in call_kwargs["system"].lower()


class TestOpenAICompatProvider:
    @pytest.mark.asyncio
    async def test_complete_returns_content(self):
        p = get_provider("openai", api_key="test-key")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="hello"))]
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            result = await p.complete([{"role": "user", "content": "hi"}], "gpt-4.1")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_complete_routes_model_kwargs_to_extra_body(self):
        p = get_provider("openai", api_key="test-key")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        with patch.object(p, "_get_client") as mock_client:
            create_mock = AsyncMock(return_value=mock_response)
            mock_client.return_value.chat.completions.create = create_mock
            await p.complete(
                [{"role": "user", "content": "hi"}],
                "gpt-4.1",
                model_kwargs={"enable_thinking": True},
            )
            call_kwargs = create_mock.call_args[1]
            assert call_kwargs["extra_body"] == {"enable_thinking": True}

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        p = get_provider("openai", api_key="test-key")

        async def fake_stream():
            for text in ["hel", "lo"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content=text))]
                yield chunk

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=fake_stream())
            tokens = []
            async for t in p.stream([{"role": "user", "content": "hi"}], "gpt-4.1"):
                tokens.append(t)
        assert tokens == ["hel", "lo"]


class TestGeminiProvider:
    @pytest.mark.asyncio
    async def test_complete_extracts_system_instruction(self):
        p = get_provider("gemini", api_key="test-key")
        mock_response = MagicMock()
        mock_response.text = "reply"
        with patch("dlightrag.models.providers.gemini_native.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content = AsyncMock(return_value=mock_response)
            p._client = None
            result = await p.complete(
                [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "hi"},
                ],
                "gemini-2.0-flash",
            )
            call_kwargs = mock_client.models.generate_content.call_args[1]
            assert "Be concise." in str(call_kwargs.get("config", {}).get("system_instruction", ""))
        assert result == "reply"

    @pytest.mark.asyncio
    async def test_role_mapping_assistant_to_model(self):
        p = get_provider("gemini", api_key="test-key")
        mock_response = MagicMock()
        mock_response.text = "ok"
        with patch("dlightrag.models.providers.gemini_native.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content = AsyncMock(return_value=mock_response)
            p._client = None
            await p.complete(
                [
                    {"role": "assistant", "content": "I said hi"},
                    {"role": "user", "content": "continue"},
                ],
                "gemini-2.0-flash",
            )
            call_args = mock_client.models.generate_content.call_args
            contents = call_args[1].get(
                "contents", call_args[0][1] if len(call_args[0]) > 1 else None
            )
            # Verify assistant → model role mapping
            assert any(c.get("role") == "model" for c in contents if isinstance(c, dict))
