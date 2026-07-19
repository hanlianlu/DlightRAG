# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for vision probe."""

from unittest.mock import AsyncMock

from dlightrag.core.vision_probe import probe_image_capability


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

    async def test_openrouter_no_image_endpoint_is_unsupported(self) -> None:
        # OpenRouter reports a text-only model with a 404 whose message differs
        # from the generic markers: "No endpoints found that support image input".
        provider = AsyncMock()
        provider.complete = AsyncMock(
            side_effect=RuntimeError(
                "Error code: 404 - {'error': {'message': "
                "'No endpoints found that support image input', 'code': 404}}"
            )
        )
        outcome = await probe_image_capability(provider, model="m", ceiling=1)
        assert outcome.status == "unsupported"
        assert outcome.failure_kind == "explicit_unsupported"

    async def test_openrouter_model_not_found_stays_unknown(self) -> None:
        # A wrong/unavailable slug ("No endpoints found for <model>") is a config
        # error, not a proven text-only capability -- must remain unknown.
        provider = AsyncMock()
        provider.complete = AsyncMock(
            side_effect=RuntimeError(
                "Error code: 404 - {'error': {'message': "
                "'No endpoints found for foo/bar.', 'code': 404}}"
            )
        )
        outcome = await probe_image_capability(provider, model="m", ceiling=1)
        assert outcome.status == "unknown"

    async def test_zero_ceiling_is_config_disabled(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value="ok")
        outcome = await probe_image_capability(provider, model="m", ceiling=0)
        assert outcome.status == "unsupported"
        assert outcome.failure_kind == "config_disabled"
        provider.complete.assert_not_awaited()
