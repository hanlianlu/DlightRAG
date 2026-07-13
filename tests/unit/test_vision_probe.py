# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for vision probe."""

from unittest.mock import AsyncMock

from dlightrag.core.vision_probe import probe_image_capability, probe_vision_support


class TestProbeImageCapability:
    async def test_supported_does_not_require_ok_word(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value="any text, magic word not required")
        outcome = await probe_image_capability(provider, model="m", ceiling=1)
        assert outcome.status == "supported"

    async def test_explicit_reject_is_unsupported(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(
            side_effect=RuntimeError("this model does not support image input")
        )
        outcome = await probe_image_capability(provider, model="m", ceiling=1)
        assert outcome.status == "unsupported"
        assert outcome.failure_kind == "explicit_unsupported"

    async def test_transient_error_is_unknown_not_unsupported(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=TimeoutError("upstream 429"))
        outcome = await probe_image_capability(provider, model="m", ceiling=1)
        assert outcome.status == "unknown"
        assert outcome.failure_kind is not None

    async def test_zero_ceiling_is_config_disabled(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value="ok")
        outcome = await probe_image_capability(provider, model="m", ceiling=0)
        assert outcome.status == "unsupported"
        assert outcome.failure_kind == "config_disabled"
        provider.complete.assert_not_awaited()


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

    async def test_allows_reasoning_models_room_to_reach_content(self) -> None:
        provider = AsyncMock()

        async def complete(*args: object, **kwargs: object) -> str:
            return "ok" if kwargs.get("max_tokens") == 512 else ""

        provider.complete = AsyncMock(side_effect=complete)

        result = await probe_vision_support(provider, model="test-model")

        assert result is True
        assert provider.complete.call_args.kwargs["max_tokens"] == 512

    async def test_forwards_model_kwargs(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value="ok")
        model_kwargs = {"reasoning_effort": "none"}

        result = await probe_vision_support(
            provider,
            model="test-model",
            model_kwargs=model_kwargs,
        )

        assert result is True
        assert provider.complete.call_args.kwargs["model_kwargs"] == model_kwargs

    async def test_sends_one_pixel_image(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value="ok")
        await probe_vision_support(provider, model="gemini-2.5-pro")

        # complete(self, messages, model, *, ...) → messages is arg 0
        call_messages = provider.complete.call_args.args[0]
        user_content = call_messages[0]["content"]
        assert isinstance(user_content, list)
        assert any(block.get("type") == "image_url" for block in user_content)
