# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for vision probe."""

from unittest.mock import AsyncMock

from dlightrag.core.vision_probe import probe_vision_support
from dlightrag.models.providers.anthropic_native import AnthropicProvider
from dlightrag.models.providers.gemini_native import GeminiProvider


class TestProbeVisionSupport:
    async def test_returns_true_when_model_says_ok(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value="ok")
        result = await probe_vision_support(provider, model="test-model")
        assert result is True
        provider.complete.assert_awaited_once()

    async def test_returns_false_on_exception(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=Exception("no vision"))
        result = await probe_vision_support(provider, model="test-model")
        assert result is False

    async def test_returns_false_when_response_lacks_ok(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value="error: model does not support images")
        result = await probe_vision_support(provider, model="test-model")
        assert result is False

    async def test_sends_one_pixel_image(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value="ok")
        await probe_vision_support(provider, model="gemini-2.5-pro")

        # complete(self, messages, model, *, ...) → messages is arg 0
        call_messages = provider.complete.call_args.args[0]
        user_content = call_messages[0]["content"]
        assert isinstance(user_content, list)
        assert any(block.get("type") == "image_url" for block in user_content)

    def test_native_providers_start_unprobed(self) -> None:
        assert AnthropicProvider(api_key="test").supports_vision is None
        assert GeminiProvider(api_key="test").supports_vision is None
