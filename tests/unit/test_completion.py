# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for CompletionProvider implementations via the provider registry."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.models.providers import get_provider


class TestOpenAIProvider:
    def test_instantiation(self):
        p = get_provider("openai", api_key="sk-test")
        assert p is not None

    @pytest.mark.asyncio
    async def test_complete(self):
        p = get_provider("openai", api_key="sk-test")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello"))]
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(p, "_client", mock_client):
            result = await p.complete(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4.1-mini",
            )
        assert result == "Hello"

    @pytest.mark.asyncio
    async def test_complete_with_response_format(self):
        p = get_provider("openai", api_key="sk-test")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"answer": "test"}'))]
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(p, "_client", mock_client):
            result = await p.complete(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4.1-mini",
                response_format={"type": "json_object"},
            )
        assert '"answer"' in result


class TestAnthropicProvider:
    def test_instantiation(self):
        p = get_provider("anthropic", api_key="sk-ant-test")
        assert p is not None


class TestGeminiProvider:
    def test_instantiation(self):
        p = get_provider("gemini", api_key="gemini-test")
        assert p is not None
