# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Every provider stream closes its SDK response however its consumer leaves."""

import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.engine.ai.providers import get_provider
from dlightrag.engine.ai.providers.base import CompletionProvider

_MESSAGES = [{"role": "user", "content": "hi"}]


class _SdkStream:
    """An SDK response stream: one chunk, then a stalled connection.

    The OpenAI and Anthropic SDKs expose ``close()``; a Gemini stream is an
    async generator, so it only has ``aclose()``.
    """

    def __init__(self, first_chunk: Any, *, generator: bool) -> None:
        self._first_chunk = first_chunk
        self._sent = False
        self._stalled = asyncio.Event()
        self.closed = False
        self.close_error: Exception | None = None
        if generator:
            self.aclose = self._close
        else:
            self.close = self._close

    def __aiter__(self) -> _SdkStream:
        return self

    async def __anext__(self) -> Any:
        if not self._sent:
            self._sent = True
            return self._first_chunk
        await self._stalled.wait()
        raise StopAsyncIteration

    async def _close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def _openai_chunk() -> Any:
    delta = SimpleNamespace(content="partial", model_extra=None, tool_calls=None)
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason=None, delta=delta)])


def _anthropic_chunk() -> Any:
    delta = SimpleNamespace(type="text_delta", text="partial")
    return SimpleNamespace(type="content_block_delta", index=0, delta=delta)


def _gemini_chunk() -> Any:
    part = SimpleNamespace(text="partial", thought=False, function_call=None)
    candidate = SimpleNamespace(finish_reason=None, content=SimpleNamespace(parts=[part]))
    return SimpleNamespace(usage_metadata=None, text="partial", candidates=[candidate])


def _provider_with_stream(name: str) -> tuple[CompletionProvider, _SdkStream, MagicMock]:
    provider = get_provider(name, api_key="test-key")
    client = MagicMock()
    if name == "openai":
        stream = _SdkStream(_openai_chunk(), generator=False)
        client.chat.completions.create = AsyncMock(return_value=stream)
    elif name == "anthropic":
        stream = _SdkStream(_anthropic_chunk(), generator=False)
        client.messages.create = AsyncMock(return_value=stream)
    else:
        stream = _SdkStream(_gemini_chunk(), generator=True)
        client.aio.models.generate_content_stream = AsyncMock(return_value=stream)
    return provider, stream, client


_TEXT_STREAMS = [
    ("openai", "stream"),
    ("openai", "stream_tool_text"),
    ("anthropic", "stream"),
    ("anthropic", "stream_tool_text"),
    ("gemini", "stream"),
    ("gemini", "stream_tool_text"),
]


def _text_stream(provider: CompletionProvider, entrypoint: str) -> AsyncGenerator[str]:
    return getattr(provider, entrypoint)(_MESSAGES, "model")


@pytest.mark.parametrize(("name", "entrypoint"), _TEXT_STREAMS)
async def test_text_stream_closes_the_sdk_stream_when_the_consumer_stops_early(
    name: str,
    entrypoint: str,
) -> None:
    provider, stream, client = _provider_with_stream(name)
    with patch.object(provider, "_get_client", return_value=client):
        tokens = _text_stream(provider, entrypoint)
        assert await anext(tokens) == "partial"
        await tokens.aclose()

    assert stream.closed is True


@pytest.mark.parametrize(("name", "entrypoint"), _TEXT_STREAMS)
async def test_text_stream_closes_the_sdk_stream_when_its_task_is_cancelled(
    name: str,
    entrypoint: str,
) -> None:
    provider, stream, client = _provider_with_stream(name)
    first_token = asyncio.Event()

    async def consume() -> None:
        async for _token in _text_stream(provider, entrypoint):
            first_token.set()

    with patch.object(provider, "_get_client", return_value=client):
        task = asyncio.create_task(consume())
        await asyncio.wait_for(first_token.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert stream.closed is True


@pytest.mark.parametrize("name", ["openai", "anthropic", "gemini"])
async def test_tool_turn_stream_closes_the_sdk_stream_when_the_consumer_fails(
    name: str,
) -> None:
    provider, stream, client = _provider_with_stream(name)

    async def emit_text(_text: str) -> None:
        raise RuntimeError("consumer stopped")

    with patch.object(provider, "_get_client", return_value=client):
        with pytest.raises(RuntimeError, match="consumer stopped"):
            await provider.complete_tool_turn_streaming(
                _MESSAGES,
                "model",
                tools=[],
                emit_text=emit_text,
            )

    assert stream.closed is True


@pytest.mark.parametrize("name", ["openai", "anthropic", "gemini"])
async def test_tool_turn_stream_closes_the_sdk_stream_when_its_task_is_cancelled(
    name: str,
) -> None:
    provider, stream, client = _provider_with_stream(name)
    first_text = asyncio.Event()

    async def emit_text(_text: str) -> None:
        first_text.set()

    with patch.object(provider, "_get_client", return_value=client):
        task = asyncio.create_task(
            provider.complete_tool_turn_streaming(
                _MESSAGES,
                "model",
                tools=[],
                emit_text=emit_text,
            )
        )
        await asyncio.wait_for(first_text.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert stream.closed is True


@pytest.mark.parametrize("name", ["openai", "anthropic", "gemini"])
async def test_an_exhausted_stream_is_closed_too(name: str) -> None:
    provider, stream, client = _provider_with_stream(name)
    stream._stalled.set()  # pyright: ignore[reportPrivateUsage]

    with patch.object(provider, "_get_client", return_value=client):
        tokens = [token async for token in _text_stream(provider, "stream")]

    assert tokens == ["partial"]
    assert stream.closed is True


_CLOSE_FAILURE = "Failed to close a provider stream while unwinding"


@pytest.mark.parametrize("name", ["openai", "anthropic", "gemini"])
async def test_a_failed_close_does_not_replace_the_consumer_failure(
    name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, stream, client = _provider_with_stream(name)
    stream.close_error = OSError("connection already gone")

    async def emit_text(_text: str) -> None:
        raise RuntimeError("consumer stopped")

    with patch.object(provider, "_get_client", return_value=client):
        with pytest.raises(RuntimeError, match="consumer stopped"):
            await provider.complete_tool_turn_streaming(
                _MESSAGES,
                "model",
                tools=[],
                emit_text=emit_text,
            )

    assert stream.closed is True
    assert _CLOSE_FAILURE in caplog.text


@pytest.mark.parametrize(("name", "entrypoint"), _TEXT_STREAMS)
async def test_a_failed_close_does_not_replace_an_early_stop(
    name: str,
    entrypoint: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, stream, client = _provider_with_stream(name)
    stream.close_error = OSError("connection already gone")

    with patch.object(provider, "_get_client", return_value=client):
        tokens = _text_stream(provider, entrypoint)
        assert await anext(tokens) == "partial"
        await tokens.aclose()

    assert stream.closed is True
    assert _CLOSE_FAILURE in caplog.text


@pytest.mark.parametrize(("name", "entrypoint"), _TEXT_STREAMS)
async def test_a_failed_close_does_not_replace_a_cancellation(
    name: str,
    entrypoint: str,
) -> None:
    provider, stream, client = _provider_with_stream(name)
    stream.close_error = OSError("connection already gone")
    first_token = asyncio.Event()

    async def consume() -> None:
        async for _token in _text_stream(provider, entrypoint):
            first_token.set()

    with patch.object(provider, "_get_client", return_value=client):
        task = asyncio.create_task(consume())
        await asyncio.wait_for(first_token.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert stream.closed is True


async def test_a_failed_close_after_a_complete_stream_is_reported() -> None:
    # Nothing else is in flight, so the close failure is the error to report.
    provider, stream, client = _provider_with_stream("openai")
    stream._stalled.set()  # pyright: ignore[reportPrivateUsage]
    stream.close_error = OSError("connection already gone")

    with patch.object(provider, "_get_client", return_value=client):
        with pytest.raises(OSError, match="connection already gone"):
            [token async for token in _text_stream(provider, "stream")]
