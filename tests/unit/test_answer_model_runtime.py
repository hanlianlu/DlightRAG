# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-owned model runtime behavior."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import ModelRoleSettings, ModelSettings
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.answer.images import AnswerImagePolicy
from dlightrag.engine.answer.model_runtime import (
    AnswerModelRuntime,
    AnswerModelRuntimeClosedError,
    AnswerModelRuntimeSettings,
    WebSourceRuntimeSettings,
)


def _policy(profile: ModelProfile) -> AnswerImagePolicy:
    return AnswerImagePolicy(
        max_images=3 if profile.supports_images else 0,
        max_total_bytes=3_000_000,
        max_bytes_per_image=1_000_000,
        max_pixels=4_000_000,
        max_px=1536,
        min_px=1024,
        quality=89,
        min_quality=79,
    )


def _runtime(*, vlm_profile: ModelProfile | None = None) -> AnswerModelRuntime:
    settings = AnswerModelRuntimeSettings(
        model_roles=ModelRoleSettings(
            default=ModelSettings(provider="openai", model="test-model", api_key="test")
        ),
        web_sources=WebSourceRuntimeSettings(
            exa_api_key="exa-test",
            search_providers=("exa",),
            extract_providers=("exa",),
        ),
        query_image_limit=3,
    )
    return AnswerModelRuntime(
        settings=settings,
        scheduler=ModelScheduler(max_concurrency=1),
        telemetry=NOOP_TELEMETRY,
        answer_image_policy=_policy,
        vlm_image_policy=_policy,
        vlm_profile=lambda: (
            vlm_profile or ModelProfile(context_window_tokens=10_000, supports_images=True)
        ),
    )


def test_synthesizer_cache_reuses_one_model_across_profiles() -> None:
    runtime = _runtime()
    live = ModelProfile(context_window_tokens=10_000, supports_images=True)
    pinned = ModelProfile(context_window_tokens=8_000, supports_images=False)
    model = MagicMock()

    with patch(
        "dlightrag.engine.answer.model_runtime.CompletionModel", return_value=model
    ) as create:
        live_synthesizer = runtime.answer_synthesizer(live)
        pinned_synthesizer = runtime.answer_synthesizer(pinned)

    assert runtime.answer_synthesizer(live) is live_synthesizer
    assert runtime.answer_synthesizer(pinned) is pinned_synthesizer
    assert live_synthesizer.model_func is model
    assert pinned_synthesizer.model_func is model
    create.assert_called_once()


def test_synthesizer_failure_does_not_construct_provider() -> None:
    runtime = _runtime()
    profile = ModelProfile(context_window_tokens=10_000)

    with (
        patch(
            "dlightrag.engine.answer.model_runtime.AnswerSynthesizer",
            side_effect=RuntimeError("synthesizer failed"),
        ),
        patch("dlightrag.engine.answer.model_runtime.CompletionModel") as completion,
        pytest.raises(RuntimeError, match="synthesizer failed"),
    ):
        runtime.answer_synthesizer(profile)

    completion.assert_not_called()


async def test_close_is_idempotent_and_prevents_recreation() -> None:
    runtime = _runtime()
    components = [AsyncMock(), AsyncMock(), AsyncMock(), AsyncMock()]
    runtime._tool_models["query"] = components[0]
    runtime._answer_model = components[1]
    runtime._vlm_model = components[2]
    runtime._web_sources = components[3]

    await runtime.aclose()
    await runtime.aclose()

    for component in components:
        component.aclose.assert_awaited_once()
    with pytest.raises(AnswerModelRuntimeClosedError):
        runtime.query_tool_model()
    with pytest.raises(AnswerModelRuntimeClosedError):
        runtime.vlm_func()
    with pytest.raises(AnswerModelRuntimeClosedError):
        runtime.web_sources()


def test_query_image_describer_follows_vlm_profile() -> None:
    runtime = _runtime()

    with patch("dlightrag.engine.answer.model_runtime.CompletionModel", return_value=MagicMock()):
        describer = runtime.query_image_describer()

    assert describer._max_images == 3
    assert describer._image_policy.max_images == 3


async def test_query_image_describer_is_disabled_without_vlm_support() -> None:
    runtime = _runtime(
        vlm_profile=ModelProfile(context_window_tokens=10_000, supports_images=False)
    )

    with patch("dlightrag.engine.answer.model_runtime.CompletionModel") as create:
        describer = runtime.query_image_describer()

    assert describer._max_images == 0
    assert await describer.describe([{"type": "image_url", "image_url": {"url": "data:x"}}]) == []
    create.assert_not_called()


async def test_repeated_cancellation_still_joins_model_shutdown() -> None:
    runtime = _runtime()
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    model = AsyncMock()

    async def close() -> None:
        close_started.set()
        await release_close.wait()

    model.aclose.side_effect = close
    runtime._answer_model = model
    closing = asyncio.create_task(runtime.aclose())
    await close_started.wait()

    closing.cancel()
    await asyncio.sleep(0)
    closing.cancel()
    release_close.set()

    with pytest.raises(asyncio.CancelledError):
        await closing
    model.aclose.assert_awaited_once()
