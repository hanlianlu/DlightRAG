# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for provider ABC, registry, and concrete implementations."""

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest

from dlightrag.engine.ai.messages import ToolDefinition
from dlightrag.engine.ai.providers import get_provider
from dlightrag.engine.ai.providers.base import (
    CompletionOutput,
    CompletionProvider,
    provider_cache_hit_tokens,
    provider_input_tokens,
    provider_status_code,
    usage_counters,
)
from dlightrag.engine.ai.providers.base import (
    is_provider_reasoning_rejection as provider_reasoning_rejection,
)
from dlightrag.engine.ai.providers.openai_compatible import (
    OpenAICompatibleProvider,
    _openai_tool_messages,
)
from dlightrag.engine.ai.providers.openai_response import ResponseStatusError


def _openai_error_response(status_code: int) -> httpx2.Response:
    """Build the HTTP response type used by the OpenAI SDK transport."""
    return httpx2.Response(
        status_code,
        request=httpx2.Request("POST", "https://t/v1"),
    )


class TestCompletionProviderABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            cast(Any, CompletionProvider)(api_key="k", timeout=10.0, max_retries=1)


class TestProviderRegistry:
    @pytest.mark.parametrize("provider_name", ["openai", "anthropic", "gemini"])
    def test_get_provider_returns_completion_provider(self, provider_name: str):
        p = get_provider(provider_name, api_key="test-key")
        assert isinstance(p, CompletionProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown_provider")

    def test_error_message_lists_available(self):
        with pytest.raises(ValueError, match="openai"):
            get_provider("bad")

    def test_response_family_is_bound_only_to_the_openai_provider(self):
        provider = get_provider("openai", api_key="test-key", api_family="response")

        assert cast(Any, provider)._api_family == "response"
        with pytest.raises(ValueError, match="requires the openai provider"):
            get_provider("anthropic", api_key="test-key", api_family="response")


class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_complete_extracts_system_message(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="reply")]
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
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
    async def test_complete_preserves_token_limit_stop_reason(self):
        p = get_provider("anthropic", api_key="test-key")
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="partial")],
            stop_reason="max_tokens",
            usage=None,
        )
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as sdk:
            sdk.return_value.messages.create = AsyncMock(return_value=response)
            cast(Any, p)._client = None
            result = await p.complete(
                [{"role": "user", "content": "hi"}],
                "claude-sonnet-4-20250514",
            )

        assert result == "partial"
        assert result.stop_reason == "length"

    @pytest.mark.asyncio
    async def test_complete_defaults_max_tokens(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="ok")]
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            await p.complete([{"role": "user", "content": "hi"}], "claude-sonnet-4-20250514")
            call_kwargs = MockSDK.return_value.messages.create.call_args[1]
            assert call_kwargs["max_tokens"] == 8192

    @pytest.mark.asyncio
    async def test_complete_tool_turn_converts_tools_history_and_response(self):
        p = get_provider("anthropic", api_key="test-key")
        tool = ToolDefinition(
            name="search_web",
            description="Search the open web.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )
        response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="thinking",
                    thinking="Need another source.",
                    signature="anthropic-signature",
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="call-2",
                    name="search_web",
                    input={"query": "inflation"},
                ),
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=8, output_tokens=3),
        )
        messages = [
            {
                "role": "assistant",
                "content": "",
                "provider_state": {
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "Previous thought.",
                            "signature": "previous-signature",
                        }
                    ]
                },
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"prices"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search_web",
                "content": "price evidence",
                "is_error": False,
            },
        ]

        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as sdk:
            create = AsyncMock(return_value=response)
            sdk.return_value.messages.create = create
            cast(Any, p)._client = None
            turn = await p.complete_tool_turn(
                messages,
                "claude-sonnet-4-20250514",
                tools=[tool],
                tool_choice="required",
            )

        await_args = create.await_args
        assert await_args is not None
        request = await_args.kwargs
        assert request["tools"] == [
            {
                "name": "search_web",
                "description": "Search the open web.",
                "input_schema": tool.parameters,
            }
        ]
        assert request["tool_choice"] == {"type": "any"}
        assert request["messages"] == [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Previous thought.",
                        "signature": "previous-signature",
                    },
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "search_web",
                        "input": {"query": "prices"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "content": "price evidence",
                        "is_error": False,
                    }
                ],
            },
        ]
        assert turn.stop_reason == "tool_use"
        assert turn.reasoning == "Need another source."
        assert turn.provider_state == {
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Need another source.",
                    "signature": "anthropic-signature",
                }
            ]
        }
        assert turn.tool_calls[0].arguments == {"query": "inflation"}
        assert turn.usage_details == {"input_tokens": 8, "output_tokens": 3}

    @pytest.mark.asyncio
    async def test_complete_tool_turn_streaming_preserves_text_thinking_and_tool_calls(self):
        p = get_provider("anthropic", api_key="test-key")

        async def fake_stream():
            yield SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=5)),
            )
            yield SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(type="thinking", thinking="", signature=""),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="thinking_delta", thinking="Think."),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="signature_delta", signature="signed"),
            )
            yield SimpleNamespace(
                type="content_block_start",
                index=1,
                content_block=SimpleNamespace(type="text", text="Draft "),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                index=1,
                delta=SimpleNamespace(type="text_delta", text="answer"),
            )
            yield SimpleNamespace(
                type="content_block_start",
                index=2,
                content_block=SimpleNamespace(
                    type="tool_use", id="call-1", name="search_web", input={}
                ),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                index=2,
                delta=SimpleNamespace(type="input_json_delta", partial_json='{"query":'),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                index=2,
                delta=SimpleNamespace(type="input_json_delta", partial_json='"inflation"}'),
            )
            yield SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="tool_use"),
                usage=SimpleNamespace(output_tokens=7),
            )

        emitted: list[str] = []

        async def emit_text(text: str) -> None:
            emitted.append(text)

        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as sdk:
            sdk.return_value.messages.create = AsyncMock(return_value=fake_stream())
            cast(Any, p)._client = None
            turn = await p.complete_tool_turn_streaming(
                [{"role": "user", "content": "latest inflation"}],
                "claude-sonnet-4-20250514",
                tools=[],
                emit_text=emit_text,
            )

        assert emitted == ["Draft ", "answer"]
        assert turn.text == "Draft answer"
        assert turn.reasoning == "Think."
        assert turn.tool_calls[0].arguments == {"query": "inflation"}
        assert turn.provider_state == {
            "thinking_blocks": [{"type": "thinking", "thinking": "Think.", "signature": "signed"}]
        }
        assert turn.usage_details == {"input_tokens": 5, "output_tokens": 7}

    @pytest.mark.asyncio
    async def test_complete_routes_thinking_to_top_level(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="thought")]
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            await p.complete(
                [{"role": "user", "content": "hi"}],
                "claude-sonnet-4-20250514",
                model_kwargs={"thinking": {"type": "enabled", "budget_tokens": 1024}},
            )
            call_kwargs = MockSDK.return_value.messages.create.call_args[1]
            assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 1024}
            assert "extra_body" not in call_kwargs

    @pytest.mark.asyncio
    async def test_json_object_response_format_is_rejected(self):
        p = get_provider("anthropic", api_key="test-key")
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            cast(Any, p)._client = None
            with pytest.raises(ValueError, match="json_schema"):
                await p.complete(
                    [{"role": "user", "content": "hi"}],
                    "claude-sonnet-4-20250514",
                    response_format={"type": "json_object"},
                )
            MockSDK.return_value.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_json_schema_response_format_uses_output_config(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text='{"answer": "ok"}')]
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            await p.complete(
                [{"role": "user", "content": "hi"}],
                "claude-sonnet-4-20250514",
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "demo_plan", "schema": schema, "strict": True},
                },
            )
            call_kwargs = MockSDK.return_value.messages.create.call_args[1]
            assert call_kwargs["output_config"] == {
                "format": {"type": "json_schema", "schema": schema}
            }
            assert "system" not in call_kwargs

    @pytest.mark.asyncio
    async def test_json_schema_and_reasoning_effort_share_output_config(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text='{"answer": "ok"}')]
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            await p.complete(
                [{"role": "user", "content": "hi"}],
                "claude-opus-5",
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "demo_plan", "schema": schema, "strict": True},
                },
                model_kwargs={
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "high"},
                },
            )
            call_kwargs = MockSDK.return_value.messages.create.call_args[1]
            assert call_kwargs["thinking"] == {"type": "adaptive"}
            assert call_kwargs["output_config"] == {
                "format": {"type": "json_schema", "schema": schema},
                "effort": "high",
            }

    @pytest.mark.asyncio
    async def test_complete_converts_https_image_url(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="ok")]
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            await p.complete(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/chart.png"},
                            },
                            {"type": "text", "text": "describe"},
                        ],
                    }
                ],
                "claude-sonnet-4-20250514",
            )
            call_kwargs = MockSDK.return_value.messages.create.call_args[1]
            assert call_kwargs["messages"][0]["content"][0] == {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/chart.png"},
            }

    @pytest.mark.asyncio
    async def test_complete_handles_thinking_blocks_and_usage(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="thinking", thinking="let me think"),
            MagicMock(type="text", text="answer"),
        ]
        mock_response.usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=3,
            cache_creation=SimpleNamespace(ephemeral_5m_input_tokens=7),
        )
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            result = await p.complete(
                [{"role": "user", "content": "hi"}],
                "claude-sonnet-4-20250514",
                model_kwargs={"thinking": {"type": "enabled", "budget_tokens": 1024}},
            )
        assert result == "answer"
        assert cast(Any, p).last_reasoning == "let me think"
        assert result.usage_details == {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
            "cache_creation.ephemeral_5m_input_tokens": 7,
        }

    @pytest.mark.asyncio
    async def test_stream_merges_message_start_and_delta_usage(self):
        p = get_provider("anthropic", api_key="test-key")
        holder: dict[str, Any] = {}

        async def fake_stream():
            yield SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=0)),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="hi"),
            )
            yield SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=6))

        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=fake_stream())
            cast(Any, p)._client = None
            tokens = [
                t
                async for t in cast(Any, p).stream(
                    [{"role": "user", "content": "hi"}],
                    "claude-sonnet-4-20250514",
                    usage_holder=holder,
                )
            ]

        assert tokens == ["hi"]
        assert holder == {"usage_details": {"input_tokens": 10, "output_tokens": 6}}

    async def test_stream_tool_text_replays_native_tool_history(self):

        p = get_provider("anthropic", api_key="test-key")

        async def fake_stream():
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="final "),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="answer"),
            )

        messages = [
            {
                "role": "assistant",
                "content": "",
                "provider_state": {
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "thought",
                            "signature": "signature",
                        }
                    ]
                },
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search",
                "content": "evidence",
                "is_error": False,
            },
        ]
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as sdk:
            create = AsyncMock(return_value=fake_stream())
            sdk.return_value.messages.create = create
            cast(Any, p)._client = None
            tokens = [
                token
                async for token in p.stream_tool_text(
                    messages,
                    "claude-sonnet-4-20250514",
                )
            ]

        assert tokens == ["final ", "answer"]
        await_args = create.await_args
        assert await_args is not None
        assert await_args.kwargs["messages"][0]["content"][0]["signature"] == "signature"
        assert await_args.kwargs["messages"][1]["content"][0]["type"] == "tool_result"


def _response_event(event_type: str, **values: Any) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, **values)


class _ResponseEventStream(AsyncIterator[Any]):
    def __init__(self, events: list[SimpleNamespace]) -> None:
        self._events = iter(events)
        self.closed = False

    def __aiter__(self) -> _ResponseEventStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def close(self) -> None:
        self.closed = True


class _BlockingResponseEventStream(AsyncIterator[Any]):
    def __init__(self) -> None:
        self._emitted = False
        self._blocked = asyncio.Event()
        self.closed = False

    def __aiter__(self) -> _BlockingResponseEventStream:
        return self

    async def __anext__(self) -> Any:
        if not self._emitted:
            self._emitted = True
            return _response_event("response.output_text.delta", delta="partial")
        await self._blocked.wait()
        raise StopAsyncIteration

    async def close(self) -> None:
        self.closed = True


class TestOpenAICompatibleProvider:
    async def test_complete_returns_content(self):
        p = get_provider("openai", api_key="test-key")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="hello"))]
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            result = await p.complete([{"role": "user", "content": "hi"}], "gpt-5.4-mini")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_complete_preserves_token_limit_stop_reason(self):
        p = get_provider("openai", api_key="test-key")
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="partial", model_extra=None),
                    finish_reason="length",
                )
            ],
            usage=None,
        )
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=response)
            result = await p.complete([{"role": "user", "content": "hi"}], "gpt-5.4-mini")

        assert result == "partial"
        assert result.stop_reason == "length"

    @pytest.mark.parametrize(
        ("response_format", "expected_text"),
        [
            (None, None),
            ({"type": "json_object"}, {"format": {"type": "json_object"}}),
        ],
    )
    async def test_response_complete_maps_plain_text_and_json_object_formats(
        self,
        response_format: dict[str, Any] | None,
        expected_text: dict[str, Any] | None,
    ) -> None:
        p = get_provider(
            "openai",
            api_key="test-key",
            api_family="response",
        )
        response = SimpleNamespace(
            status="completed",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="answer")],
                )
            ],
            usage=None,
        )
        with patch.object(p, "_get_client") as mock_client:
            create = AsyncMock(return_value=response)
            mock_client.return_value.responses.create = create
            result = await p.complete(
                [{"role": "user", "content": "hi"}],
                "gpt-5.4",
                response_format=response_format,
            )

        assert result == "answer"
        await_args = create.await_args
        assert await_args is not None
        if expected_text is None:
            assert "text" not in await_args.kwargs
        else:
            assert await_args.kwargs["text"] == expected_text

    @pytest.mark.asyncio
    async def test_response_complete_maps_token_limit_and_nested_usage(self):
        p = get_provider(
            "openai",
            api_key="test-key",
            api_family="response",
        )
        response = SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="partial")],
                )
            ],
            usage=SimpleNamespace(
                input_tokens=4,
                input_tokens_details=SimpleNamespace(cached_tokens=3),
                output_tokens=2,
            ),
        )
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.responses.create = AsyncMock(return_value=response)
            result = await p.complete([{"role": "user", "content": "hi"}], "gpt-5.4")

        assert result == "partial"
        assert result.stop_reason == "length"
        assert result.usage_details == {
            "input_tokens": 4,
            "input_tokens_details.cached_tokens": 3,
            "output_tokens": 2,
        }

    @pytest.mark.parametrize(
        ("status", "output", "incomplete_reason", "message"),
        [
            ("failed", [], None, "failed"),
            ("cancelled", [], None, "unsupported status"),
            ("incomplete", [], "content_filter", "incomplete"),
            ("completed", [], None, "without text"),
            (
                "completed",
                [
                    SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="refusal", refusal="no")],
                    )
                ],
                None,
                "refused",
            ),
        ],
    )
    async def test_response_complete_rejects_non_output_terminal_states(
        self,
        status: str,
        output: list[SimpleNamespace],
        incomplete_reason: str | None,
        message: str,
    ) -> None:
        p = get_provider(
            "openai",
            api_key="test-key",
            api_family="response",
        )
        response = SimpleNamespace(
            status=status,
            error=SimpleNamespace(code="provider_error") if status == "failed" else None,
            incomplete_details=(
                SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None
            ),
            output=output,
            usage=None,
        )
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.responses.create = AsyncMock(return_value=response)
            with pytest.raises(ResponseStatusError, match=message):
                await p.complete([{"role": "user", "content": "hi"}], "gpt-5.4")

    @pytest.mark.asyncio
    async def test_response_tool_turn_preserves_invalid_arguments_for_local_rejection(self):
        p = get_provider(
            "openai",
            api_key="test-key",
            api_family="response",
        )
        response = SimpleNamespace(
            status="completed",
            output=[
                SimpleNamespace(
                    id="fc-item",
                    type="function_call",
                    status="completed",
                    call_id="call-1",
                    name="lookup",
                    arguments='{"value":',
                )
            ],
            usage=None,
        )
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.responses.create = AsyncMock(return_value=response)
            turn = await p.complete_tool_turn(
                [{"role": "user", "content": "hi"}],
                "gpt-5.4",
                tools=[],
            )

        assert turn.stop_reason == "tool_use"
        assert turn.tool_calls[0].id == "call-1"
        assert turn.tool_calls[0].id != "fc-item"
        assert turn.tool_calls[0].arguments == {}
        assert turn.tool_calls[0].argument_error is not None

    @pytest.mark.asyncio
    async def test_response_incomplete_never_exposes_calls_for_execution(self):
        p = get_provider(
            "openai",
            api_key="test-key",
            api_family="response",
        )
        response = SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            output=[
                SimpleNamespace(
                    id="fc-item",
                    type="function_call",
                    status="incomplete",
                    call_id="call-1",
                    name="lookup",
                    arguments='{"value":"partial',
                )
            ],
            usage=None,
        )
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.responses.create = AsyncMock(return_value=response)
            turn = await p.complete_tool_turn(
                [{"role": "user", "content": "hi"}],
                "gpt-5.4",
                tools=[],
            )

        assert turn.stop_reason == "length"
        assert turn.tool_calls == ()
        assert turn.provider_state is None

    @pytest.mark.asyncio
    async def test_response_streaming_tool_turn_reconciles_final_items_without_duplication(self):
        p = get_provider("openai", api_key="test-key", api_family="response")
        message = SimpleNamespace(
            id="msg-1",
            type="message",
            status="completed",
            role="assistant",
            content=[SimpleNamespace(type="output_text", text="Checking.")],
        )
        call = SimpleNamespace(
            id="fc-item-1",
            type="function_call",
            status="completed",
            call_id="call-1",
            name="lookup",
            arguments='{"value":"one"}',
        )
        response = SimpleNamespace(
            status="completed",
            output=[message, call],
            usage=SimpleNamespace(input_tokens=4, output_tokens=3),
        )
        stream = _ResponseEventStream(
            [
                _response_event("response.output_text.delta", delta=""),
                _response_event("response.output_text.delta", delta="Check"),
                _response_event("response.output_text.delta", delta="ing."),
                _response_event("response.output_item.done", output_index=0, item=message),
                _response_event(
                    "response.function_call_arguments.delta",
                    output_index=1,
                    item_id="fc-item-1",
                    delta='{"value":',
                ),
                _response_event(
                    "response.function_call_arguments.delta",
                    output_index=1,
                    item_id="fc-item-1",
                    delta='"one"}',
                ),
                _response_event(
                    "response.function_call_arguments.done",
                    output_index=1,
                    item_id="fc-item-1",
                    name="lookup",
                    arguments='{"value":"one"}',
                ),
                _response_event("response.output_item.done", output_index=1, item=call),
                _response_event("response.completed", response=response),
            ]
        )
        emitted: list[str] = []

        async def emit_text(text: str) -> None:
            emitted.append(text)

        with patch.object(p, "_get_client") as mock_client:
            create = AsyncMock(return_value=stream)
            mock_client.return_value.responses.create = create
            turn = await p.complete_tool_turn_streaming(
                [{"role": "user", "content": "hi"}],
                "gpt-5.4",
                tools=[],
                emit_text=emit_text,
            )

        assert emitted == ["", "Check", "ing."]
        assert turn.text == "Checking."
        assert turn.tool_calls[0].arguments == {"value": "one"}
        assert turn.usage_details == {"input_tokens": 4, "output_tokens": 3}
        assert turn.provider_state is not None
        replay = turn.provider_state["response_replay"]
        assert replay["items"][1]["arguments"] == '{"value":"one"}'
        assert stream.closed is True
        assert create.await_args is not None
        assert create.await_args.kwargs["stream"] is True
        mock_client.return_value.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("entrypoint", ["stream", "stream_tool_text"])
    async def test_response_text_stream_entrypoints_share_terminal_usage(
        self, entrypoint: str
    ) -> None:
        p = get_provider("openai", api_key="test-key", api_family="response")
        message = SimpleNamespace(
            id="msg-1",
            type="message",
            status="completed",
            role="assistant",
            content=[SimpleNamespace(type="output_text", text="hello")],
        )
        response = SimpleNamespace(
            status="completed",
            output=[message],
            usage=SimpleNamespace(input_tokens=4, output_tokens=1),
        )
        stream = _ResponseEventStream(
            [
                _response_event("response.output_text.delta", delta="hel"),
                _response_event("response.output_text.delta", delta="lo"),
                _response_event("response.output_item.done", output_index=0, item=message),
                _response_event("response.completed", response=response),
            ]
        )
        holder: dict[str, Any] = {}
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.responses.create = AsyncMock(return_value=stream)
            method = getattr(p, entrypoint)
            tokens = [
                token
                async for token in method(
                    [{"role": "user", "content": "hi"}],
                    "gpt-5.4",
                    usage_holder=holder,
                )
            ]

        assert tokens == ["hel", "lo"]
        assert holder == {"usage_details": {"input_tokens": 4, "output_tokens": 1}}
        assert stream.closed is True

    @pytest.mark.asyncio
    async def test_response_incomplete_stream_never_exposes_partial_call(self):
        p = get_provider("openai", api_key="test-key", api_family="response")
        partial_call = SimpleNamespace(
            id="fc-item-1",
            type="function_call",
            status="incomplete",
            call_id="call-1",
            name="lookup",
            arguments='{"value":"par',
        )
        response = SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            output=[partial_call],
            usage=None,
        )
        stream = _ResponseEventStream(
            [
                _response_event(
                    "response.function_call_arguments.delta",
                    output_index=0,
                    item_id="fc-item-1",
                    delta='{"value":"par',
                ),
                _response_event("response.output_item.done", output_index=0, item=partial_call),
                _response_event("response.incomplete", response=response),
            ]
        )
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.responses.create = AsyncMock(return_value=stream)
            turn = await p.complete_tool_turn_streaming(
                [{"role": "user", "content": "hi"}],
                "gpt-5.4",
                tools=[],
                emit_text=AsyncMock(),
            )

        assert turn.stop_reason == "length"
        assert turn.tool_calls == ()
        assert turn.provider_state is None
        assert stream.closed is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("events", "message"),
        [
            ([_response_event("error", code="bad", message="provider broke")], "provider broke"),
            ([], "terminal"),
            (
                [
                    _response_event(
                        "response.failed",
                        response=SimpleNamespace(
                            status="failed",
                            error=SimpleNamespace(code="provider_error"),
                            output=[],
                            usage=None,
                        ),
                    )
                ],
                "failed",
            ),
            (
                [
                    _response_event(
                        "response.incomplete",
                        response=SimpleNamespace(
                            status="incomplete",
                            incomplete_details=SimpleNamespace(reason="content_filter"),
                            output=[],
                            usage=None,
                        ),
                    )
                ],
                "incomplete",
            ),
            (
                [
                    _response_event(
                        "response.cancelled",
                        response=SimpleNamespace(
                            status="cancelled",
                            output=[],
                            usage=None,
                        ),
                    )
                ],
                "cancelled",
            ),
            (
                [
                    _response_event(
                        "response.output_item.done",
                        output_index=0,
                        item=SimpleNamespace(
                            id="msg-1",
                            type="message",
                            status="completed",
                            role="assistant",
                            content=[SimpleNamespace(type="refusal", refusal="no")],
                        ),
                    ),
                    _response_event(
                        "response.completed",
                        response=SimpleNamespace(
                            status="completed",
                            output=[
                                SimpleNamespace(
                                    id="msg-1",
                                    type="message",
                                    status="completed",
                                    role="assistant",
                                    content=[SimpleNamespace(type="refusal", refusal="no")],
                                )
                            ],
                            usage=None,
                        ),
                    ),
                ],
                "refused",
            ),
        ],
    )
    async def test_response_stream_rejects_error_failed_or_missing_terminal(
        self, events: list[SimpleNamespace], message: str
    ) -> None:
        p = get_provider("openai", api_key="test-key", api_family="response")
        stream = _ResponseEventStream(events)
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.responses.create = AsyncMock(return_value=stream)
            with pytest.raises(ResponseStatusError, match=message):
                _ = [
                    token
                    async for token in p.stream([{"role": "user", "content": "hi"}], "gpt-5.4")
                ]
        assert stream.closed is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("entrypoint", ["stream", "stream_tool_text"])
    async def test_response_text_stream_close_closes_the_http_stream(self, entrypoint: str) -> None:
        p = get_provider("openai", api_key="test-key", api_family="response")
        stream = _BlockingResponseEventStream()
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.responses.create = AsyncMock(return_value=stream)
            method = getattr(p, entrypoint)
            tokens = method([{"role": "user", "content": "hi"}], "gpt-5.4")
            assert await anext(tokens) == "partial"
            await tokens.aclose()

        assert stream.closed is True

    @pytest.mark.asyncio
    async def test_response_stream_cancellation_closes_the_http_stream(self):
        p = get_provider("openai", api_key="test-key", api_family="response")
        stream = _BlockingResponseEventStream()
        emitted = asyncio.Event()

        async def emit_text(_text: str) -> None:
            emitted.set()

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.responses.create = AsyncMock(return_value=stream)
            task = asyncio.create_task(
                p.complete_tool_turn_streaming(
                    [{"role": "user", "content": "hi"}],
                    "gpt-5.4",
                    tools=[],
                    emit_text=emit_text,
                )
            )
            await asyncio.wait_for(emitted.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert stream.closed is True

    @pytest.mark.asyncio
    async def test_complete_tool_turn_sends_tools_and_normalizes_calls(self):
        p = get_provider("openai", api_key="test-key")
        function = SimpleNamespace(name="search_web", arguments='{"query":"inflation"}')
        tool_call = SimpleNamespace(id="call-1", type="function", function=function)
        message = SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
            model_extra={
                "reasoning_content": "Need current evidence.",
                "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
            },
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
        )
        tool = ToolDefinition(
            name="search_web",
            description="Search the open web.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        )

        with patch.object(p, "_get_client") as mock_client:
            create = AsyncMock(return_value=response)
            mock_client.return_value.chat.completions.create = create
            turn = await p.complete_tool_turn(
                [{"role": "user", "content": "latest inflation"}],
                "mimo-v2.5",
                tools=[tool],
                tool_choice="required",
            )

        await_args = create.await_args
        assert await_args is not None
        request = await_args.kwargs
        assert request["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the open web.",
                    "parameters": tool.parameters,
                    "strict": False,
                },
            }
        ]
        assert request["tool_choice"] == "required"
        assert turn.stop_reason == "tool_use"
        assert turn.text == ""
        assert turn.reasoning == "Need current evidence."
        assert turn.provider_state == {
            "reasoning_content": "Need current evidence.",
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
        }
        assert turn.tool_calls[0].id == "call-1"
        assert turn.tool_calls[0].name == "search_web"
        assert turn.tool_calls[0].arguments == {"query": "inflation"}
        assert turn.tool_calls[0].argument_error is None
        assert turn.usage_details == {"prompt_tokens": 4, "completion_tokens": 2}

        replay = {
            "role": "assistant",
            "content": "",
            "tool_calls": [],
            "provider_state": turn.provider_state,
        }
        with patch.object(p, "_get_client") as replay_client:
            replay_client.return_value.chat.completions.create = AsyncMock(
                return_value=SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="done", tool_calls=None, model_extra=None
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )
            )
            await p.complete_tool_turn([replay], "mimo-v2.5", tools=[])
        replay_args = replay_client.return_value.chat.completions.create.await_args
        assert replay_args is not None
        replay_message = replay_args.kwargs["messages"][0]
        assert "provider_state" not in replay_message
        assert replay_message["reasoning_content"] == "Need current evidence."
        assert replay_message["reasoning_details"] == [
            {"type": "reasoning.encrypted", "data": "opaque"}
        ]

    @pytest.mark.asyncio
    async def test_complete_tool_turn_preserves_bad_arguments_for_the_loop_to_reject(self):
        p = get_provider("openai", api_key="test-key")
        function = SimpleNamespace(name="search_web", arguments='{"query":')
        message = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(id="call-1", type="function", function=function)],
            model_extra=None,
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
            usage=None,
        )

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=response)
            turn = await p.complete_tool_turn(
                [{"role": "user", "content": "q"}],
                "mimo-v2.5",
                tools=[],
            )

        assert turn.tool_calls[0].arguments == {}
        assert turn.tool_calls[0].argument_error is not None

    @pytest.mark.asyncio
    async def test_complete_tool_turn_maps_plain_text_stop(self):
        p = get_provider("openai", api_key="test-key")
        message = SimpleNamespace(content="final answer", tool_calls=None, model_extra=None)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=None,
        )

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=response)
            turn = await p.complete_tool_turn(
                [{"role": "user", "content": "q"}],
                "mimo-v2.5",
                tools=[],
            )

        assert turn.stop_reason == "stop"
        assert turn.text == "final answer"
        assert turn.tool_calls == ()

    @pytest.mark.asyncio
    async def test_complete_tool_turn_streaming_accumulates_fragmented_calls(self):
        p = get_provider("openai", api_key="test-key")

        async def fake_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="Draft ",
                            model_extra={"reasoning_content": "Think."},
                            tool_calls=None,
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            model_extra=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call-1",
                                    function=SimpleNamespace(
                                        name="search_web", arguments='{"query":'
                                    ),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            model_extra=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id=None,
                                    function=SimpleNamespace(name=None, arguments='"inflation"}'),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[], usage=SimpleNamespace(prompt_tokens=4, completion_tokens=3)
            )

        emitted: list[str] = []

        async def emit_text(text: str) -> None:
            emitted.append(text)

        with patch.object(p, "_open_stream", AsyncMock(return_value=fake_stream())):
            turn = await p.complete_tool_turn_streaming(
                [{"role": "user", "content": "latest inflation"}],
                "mimo-v2.5",
                tools=[],
                emit_text=emit_text,
            )

        assert emitted == ["Draft "]
        assert turn.text == "Draft "
        assert turn.reasoning == "Think."
        assert turn.stop_reason == "tool_use"
        assert turn.tool_calls[0].id == "call-1"
        assert turn.tool_calls[0].arguments == {"query": "inflation"}
        assert turn.usage_details == {"prompt_tokens": 4, "completion_tokens": 3}

    @pytest.mark.asyncio
    async def test_stream_captures_usage_and_cost_from_final_chunk(self):
        p = get_provider("openai", api_key="test-key")
        holder: dict[str, Any] = {}

        async def _fake_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="he", model_extra=None))],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="llo", model_extra=None))],
                usage=None,
            )
            # Final usage-only chunk (empty choices), as sent with include_usage.
            yield SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=5, completion_tokens=2, total_tokens=7, cost=0.0012
                ),
            )

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(
                return_value=_fake_stream()
            )
            stream = cast(Any, p).stream(
                [{"role": "user", "content": "hi"}], "gpt", usage_holder=holder
            )
            chunks = [c async for c in stream]

        assert chunks == ["he", "llo"]
        assert holder == {
            "usage_details": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            "cost_details": {"total": 0.0012},
        }

    @pytest.mark.asyncio
    async def test_stream_falls_back_when_stream_options_unsupported(self):
        from openai import BadRequestError

        p = get_provider("openai", api_key="test-key")
        holder: dict[str, Any] = {}
        calls: list[dict[str, Any]] = []

        async def _fake_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hi", model_extra=None))],
                usage=None,
            )

        async def _create(**kwargs: Any):
            calls.append(kwargs)
            if "stream_options" in kwargs:
                raise BadRequestError(
                    "stream_options unsupported",
                    response=_openai_error_response(400),
                    body=None,
                )
            return _fake_stream()

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = _create
            stream = cast(Any, p).stream(
                [{"role": "user", "content": "hi"}], "gpt", usage_holder=holder
            )
            chunks = [c async for c in stream]

        assert chunks == ["hi"]
        assert len(calls) == 2
        assert "stream_options" in calls[0]
        assert "stream_options" not in calls[1]
        assert holder == {}  # the fallback stream carries no usage

    @pytest.mark.parametrize(
        ("message", "body"),
        (
            ("invalid parameter: stream_options", None),
            ("stream_options is not permitted", None),
            (
                "Request validation failed",
                {
                    "detail": [
                        {
                            "loc": ["body", "stream_options"],
                            "msg": "extra inputs are not permitted",
                        }
                    ]
                },
            ),
        ),
    )
    @pytest.mark.asyncio
    async def test_stream_falls_back_for_explicit_stream_options_rejections(
        self,
        message: str,
        body: dict[str, Any] | None,
    ):
        from openai import BadRequestError

        p = get_provider("openai", api_key="test-key")
        calls: list[dict[str, Any]] = []

        async def _fake_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hi", model_extra=None))],
                usage=None,
            )

        async def _create(**kwargs: Any):
            calls.append(kwargs)
            if "stream_options" in kwargs:
                raise BadRequestError(
                    message,
                    response=_openai_error_response(400),
                    body=body,
                )
            return _fake_stream()

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = _create
            chunks = [
                chunk
                async for chunk in cast(Any, p).stream(
                    [{"role": "user", "content": "hi"}],
                    "gpt",
                )
            ]

        assert chunks == ["hi"]
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_stream_falls_back_for_422_stream_options_validation(self):
        from openai import UnprocessableEntityError

        p = get_provider("openai", api_key="test-key")
        calls: list[dict[str, Any]] = []

        async def _fake_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hi", model_extra=None))],
                usage=None,
            )

        async def _create(**kwargs: Any):
            calls.append(kwargs)
            if "stream_options" in kwargs:
                raise UnprocessableEntityError(
                    "Request validation failed",
                    response=_openai_error_response(422),
                    body={
                        "detail": [
                            {
                                "loc": ["body", "stream_options"],
                                "msg": "extra inputs are not permitted",
                            }
                        ]
                    },
                )
            return _fake_stream()

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = _create
            chunks = [
                chunk
                async for chunk in cast(Any, p).stream(
                    [{"role": "user", "content": "hi"}],
                    "gpt",
                )
            ]

        assert chunks == ["hi"]
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_stream_does_not_retry_provider_content_inspection_error(self):
        from openai import BadRequestError

        p = get_provider("openai", api_key="test-key")
        calls: list[dict[str, Any]] = []
        error_body = {
            "error": {
                "message": "Provider returned error",
                "code": 400,
                "metadata": {
                    "raw": (
                        'data: {"error":{"code":"data_inspection_failed",'
                        '"message":"Input text data may contain inappropriate content."}}'
                    ),
                    "provider_name": "Alibaba",
                },
            }
        }

        async def _create(**kwargs: Any):
            calls.append(kwargs)
            raise BadRequestError(
                "Provider returned error",
                response=_openai_error_response(400),
                body=error_body,
            )

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = _create
            stream = cast(Any, p).stream([{"role": "user", "content": "hi"}], "qwen")
            with pytest.raises(BadRequestError, match="Provider returned error"):
                _ = [chunk async for chunk in stream]

        assert len(calls) == 1
        assert "stream_options" in calls[0]

    @pytest.mark.asyncio
    async def test_stream_does_not_retry_on_non_badrequest_error(self):
        p = get_provider("openai", api_key="test-key")
        calls: list[dict[str, Any]] = []

        async def _create(**kwargs: Any):
            calls.append(kwargs)
            raise RuntimeError("network down")

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = _create
            stream = cast(Any, p).stream([{"role": "user", "content": "hi"}], "gpt")
            with pytest.raises(RuntimeError, match="network down"):
                _ = [c async for c in stream]

        assert len(calls) == 1  # genuine errors are not retried

    @pytest.mark.asyncio
    async def test_complete_returns_usage_and_cost_metadata(self):
        p = get_provider("openai", api_key="test-key")
        usage = SimpleNamespace(
            prompt_tokens=4,
            completion_tokens=3,
            total_tokens=7,
            cost=0.002,
        )
        mock_response = SimpleNamespace(usage=usage)
        mock_response.choices = [MagicMock(message=MagicMock(content="hello"))]

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            result = await p.complete([{"role": "user", "content": "hi"}], "gpt-5.4-mini")

        assert isinstance(result, CompletionOutput)
        assert result == "hello"
        assert result.usage_details == {
            "prompt_tokens": 4,
            "completion_tokens": 3,
            "total_tokens": 7,
        }
        assert result.cost_details == {"total": 0.002}

    @pytest.mark.asyncio
    async def test_complete_captures_provider_extra_token_counters(self):
        # DeepSeek-style flat counters arrive as SDK ``model_extra`` fields.
        class _Usage:
            model_extra = {"prompt_cache_hit_tokens": 8, "prompt_cache_miss_tokens": 2}

            def __init__(self) -> None:
                self.prompt_tokens = 10
                self.completion_tokens = 5
                self.total_tokens = 15

        p = get_provider("openai", api_key="test-key")
        mock_response = SimpleNamespace(usage=_Usage())
        mock_response.choices = [MagicMock(message=MagicMock(content="hi"))]
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            result = await p.complete([{"role": "user", "content": "hi"}], "deepseek-flash")
        assert result.usage_details == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_cache_hit_tokens": 8,
            "prompt_cache_miss_tokens": 2,
        }

    @pytest.mark.asyncio
    async def test_complete_flattens_nested_token_details(self):
        # OpenAI/Azure/Zhipu-style nested detail objects are flattened.
        p = get_provider("openai", api_key="test-key")
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=SimpleNamespace(cached_tokens=80),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=20),
        )
        mock_response = SimpleNamespace(usage=usage)
        mock_response.choices = [MagicMock(message=MagicMock(content="hi"))]
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            result = await p.complete([{"role": "user", "content": "hi"}], "gpt-5.4-mini")
        assert result.usage_details == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details.cached_tokens": 80,
            "completion_tokens_details.reasoning_tokens": 20,
        }

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
                "gpt-5.4-mini",
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
            async for t in cast(Any, p).stream([{"role": "user", "content": "hi"}], "gpt-5.4-mini"):
                tokens.append(t)
        assert tokens == ["hel", "lo"]

    @pytest.mark.asyncio
    async def test_stream_tool_text_replays_reasoning_state(self):
        p = get_provider("openai", api_key="test-key")

        async def fake_stream():
            for text in ("final ", "answer"):
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(delta=SimpleNamespace(content=text, model_extra=None))
                    ],
                    usage=None,
                )

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [],
                "provider_state": {
                    "reasoning_content": "thought",
                    "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
                },
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search",
                "content": "evidence",
                "is_error": False,
            },
        ]
        with patch.object(p, "_get_client") as client:
            create = AsyncMock(return_value=fake_stream())
            client.return_value.chat.completions.create = create
            tokens = [
                token
                async for token in p.stream_tool_text(
                    messages,
                    "mimo-v2.5",
                )
            ]

        assert tokens == ["final ", "answer"]
        first_request = create.await_args_list[0].kwargs
        replayed = first_request["messages"][0]
        assert "provider_state" not in replayed
        assert replayed["reasoning_content"] == "thought"
        assert replayed["reasoning_details"][0]["data"] == "opaque"


class TestGeminiProvider:
    @pytest.mark.asyncio
    async def test_complete_extracts_system_instruction(self):
        p = get_provider("gemini", api_key="test-key")
        mock_response = MagicMock()
        mock_response.text = "reply"
        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            result = await p.complete(
                [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "hi"},
                ],
                "gemini-2.0-flash",
            )
            call_kwargs = mock_client.aio.models.generate_content.call_args[1]
            assert "Be concise." in str(call_kwargs.get("config", {}).get("system_instruction", ""))
        assert result == "reply"

    @pytest.mark.asyncio
    async def test_complete_preserves_token_limit_stop_reason(self):
        p = get_provider("gemini", api_key="test-key")
        response = SimpleNamespace(
            text="partial",
            candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
            usage_metadata=None,
        )
        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as genai:
            client = MagicMock()
            genai.Client.return_value = client
            client.aio.models.generate_content = AsyncMock(return_value=response)
            cast(Any, p)._client = None
            result = await p.complete(
                [{"role": "user", "content": "hi"}],
                "gemini-2.0-flash",
            )

        assert result == "partial"
        assert result.stop_reason == "length"

    @pytest.mark.asyncio
    async def test_role_mapping_assistant_to_model(self):
        p = get_provider("gemini", api_key="test-key")
        mock_response = MagicMock()
        mock_response.text = "ok"
        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            await p.complete(
                [
                    {"role": "assistant", "content": "I said hi"},
                    {"role": "user", "content": "continue"},
                ],
                "gemini-2.0-flash",
            )
            call_args = mock_client.aio.models.generate_content.call_args
            contents = call_args[1].get(
                "contents", call_args[0][1] if len(call_args[0]) > 1 else None
            )
            # Verify assistant → model role mapping
            assert any(c.get("role") == "model" for c in contents if isinstance(c, dict))

    @pytest.mark.asyncio
    async def test_complete_tool_turn_converts_tools_history_and_response(self):
        p = get_provider("gemini", api_key="test-key")
        tool = ToolDefinition(
            name="search_web",
            description="Search the open web.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )
        function_call = SimpleNamespace(
            id="call-2",
            name="search_web",
            args={"query": "inflation"},
        )
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    finish_reason="STOP",
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                text=None,
                                thought=False,
                                function_call=function_call,
                                thought_signature="gemini-signature",
                            )
                        ]
                    ),
                )
            ],
            usage_metadata=SimpleNamespace(prompt_token_count=8, candidates_token_count=3),
        )
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"prices"}',
                        },
                        "thought_signature": "previous-gemini-signature",
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search_web",
                "content": "price evidence",
                "is_error": False,
            },
        ]

        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as sdk:
            client = MagicMock()
            sdk.Client.return_value = client
            create = AsyncMock(return_value=response)
            client.aio.models.generate_content = create
            cast(Any, p)._client = None
            turn = await p.complete_tool_turn(
                messages,
                "gemini-2.5-flash",
                tools=[tool],
                tool_choice="required",
            )

        await_args = create.await_args
        assert await_args is not None
        request = await_args.kwargs
        assert request["config"]["tools"] == [
            {
                "function_declarations": [
                    {
                        "name": "search_web",
                        "description": "Search the open web.",
                        "parameters": tool.parameters,
                    }
                ]
            }
        ]
        assert request["config"]["tool_config"] == {"function_calling_config": {"mode": "ANY"}}
        assert request["contents"] == [
            {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "id": "call-1",
                            "name": "search_web",
                            "args": {"query": "prices"},
                        },
                        "thought_signature": "previous-gemini-signature",
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": "call-1",
                            "name": "search_web",
                            "response": {"output": "price evidence", "is_error": False},
                        }
                    }
                ],
            },
        ]
        assert turn.stop_reason == "tool_use"
        assert turn.tool_calls[0].id == "call-2"
        assert turn.tool_calls[0].arguments == {"query": "inflation"}
        assert turn.tool_calls[0].thought_signature == "gemini-signature"
        assert turn.usage_details == {"prompt_tokens": 8, "candidates_tokens": 3}

    @pytest.mark.asyncio
    async def test_complete_tool_turn_streaming_preserves_text_thoughts_and_calls(self):
        p = get_provider("gemini", api_key="test-key")
        function_call = SimpleNamespace(id="call-1", name="search_web", args={"query": "inflation"})

        async def fake_stream():
            yield SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        finish_reason=None,
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(text="Think.", thought=True, function_call=None),
                                SimpleNamespace(text="Draft ", thought=False, function_call=None),
                            ]
                        ),
                    ),
                    SimpleNamespace(
                        finish_reason=None,
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(text="alternate", thought=False, function_call=None)
                            ]
                        ),
                    ),
                ],
                usage_metadata=None,
            )
            yield SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        finish_reason="STOP",
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    text=None,
                                    thought=False,
                                    function_call=function_call,
                                    thought_signature="signed",
                                )
                            ]
                        ),
                    )
                ],
                usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=3),
            )

        emitted: list[str] = []

        async def emit_text(text: str) -> None:
            emitted.append(text)

        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as sdk:
            client = MagicMock()
            sdk.Client.return_value = client
            client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())
            cast(Any, p)._client = None
            turn = await p.complete_tool_turn_streaming(
                [{"role": "user", "content": "latest inflation"}],
                "gemini-2.5-flash",
                tools=[],
                emit_text=emit_text,
            )

        assert emitted == ["Draft "]
        assert turn.text == "Draft "
        assert turn.reasoning == "Think."
        assert turn.stop_reason == "tool_use"
        assert turn.tool_calls[0].arguments == {"query": "inflation"}
        assert turn.tool_calls[0].thought_signature == "signed"
        assert turn.usage_details == {"prompt_tokens": 5, "candidates_tokens": 3}

    @pytest.mark.asyncio
    async def test_stream_uses_gemini_async_stream_api(self):
        p = get_provider("gemini", api_key="test-key")

        async def fake_stream():
            for text in ("hel", "lo"):
                yield SimpleNamespace(text=text)

        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())
            cast(Any, p)._client = None
            tokens = [
                token
                async for token in cast(Any, p).stream(
                    [{"role": "user", "content": "hi"}],
                    "gemini-2.0-flash",
                )
            ]

        assert tokens == ["hel", "lo"]
        mock_client.aio.models.generate_content_stream.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stream_captures_usage_metadata(self):
        p = get_provider("gemini", api_key="test-key")
        holder: dict[str, Any] = {}

        async def fake_stream():
            yield SimpleNamespace(text="hel", usage_metadata=None)
            yield SimpleNamespace(
                text="lo",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12, candidates_token_count=4, total_token_count=16
                ),
            )

        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())
            cast(Any, p)._client = None
            tokens = [
                t
                async for t in cast(Any, p).stream(
                    [{"role": "user", "content": "hi"}],
                    "gemini-2.0-flash",
                    usage_holder=holder,
                )
            ]

        assert tokens == ["hel", "lo"]
        assert holder == {
            "usage_details": {
                "prompt_token_count": 12,
                "candidates_token_count": 4,
                "total_token_count": 16,
            }
        }

    @pytest.mark.asyncio
    async def test_stream_tool_text_replays_thought_signature(self):
        p = get_provider("gemini", api_key="test-key")

        async def fake_stream():
            yield SimpleNamespace(text="final ", usage_metadata=None)
            yield SimpleNamespace(text="answer", usage_metadata=None)

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                        "thought_signature": "signature",
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search",
                "content": "evidence",
                "is_error": False,
            },
        ]
        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as sdk:
            client = MagicMock()
            sdk.Client.return_value = client
            stream = AsyncMock(return_value=fake_stream())
            client.aio.models.generate_content_stream = stream
            cast(Any, p)._client = None
            tokens = [
                token
                async for token in p.stream_tool_text(
                    messages,
                    "gemini-2.5-flash",
                )
            ]

        assert tokens == ["final ", "answer"]
        await_args = stream.await_args
        assert await_args is not None
        first_part = await_args.kwargs["contents"][0]["parts"][0]
        assert first_part["thought_signature"] == "signature"

    @pytest.mark.asyncio
    async def test_json_schema_response_format_uses_response_schema(self):
        p = get_provider("gemini", api_key="test-key")
        mock_response = MagicMock()
        mock_response.text = '{"answer": "ok"}'
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            await p.complete(
                [{"role": "user", "content": "hi"}],
                "gemini-2.5-flash",
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "demo_plan", "schema": schema, "strict": True},
                },
            )
            call_kwargs = mock_client.aio.models.generate_content.call_args[1]
            assert call_kwargs["config"]["response_mime_type"] == "application/json"
            assert call_kwargs["config"]["response_schema"] == schema

    @pytest.mark.asyncio
    async def test_aclose_closes_async_client(self):
        p = get_provider("gemini", api_key="test-key")
        mock_client = MagicMock()
        mock_client.aio.aclose = AsyncMock()
        cast(Any, p)._client = mock_client
        await p.aclose()
        mock_client.aio.aclose.assert_awaited_once()
        assert cast(Any, p)._client is None

    @pytest.mark.asyncio
    async def test_complete_captures_cache_and_thought_tokens(self):
        p = get_provider("gemini", api_key="test-key")
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_response.usage_metadata = SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150,
            cached_content_token_count=80,
            thoughts_token_count=20,
        )
        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            result = await p.complete(
                [{"role": "user", "content": "hi"}],
                "gemini-2.5-flash",
            )
        assert result.usage_details == {
            "prompt_tokens": 100,
            "candidates_tokens": 50,
            "total_tokens": 150,
            "cached_content_tokens": 80,
            "thoughts_tokens": 20,
        }


async def test_empty_tool_calls_arrays_are_stripped_for_strict_endpoints():
    OpenAICompatibleProvider(
        api_key="test-key",
        base_url="http://localhost:8888/v1",
        timeout=10.0,
        max_retries=1,
    )
    messages = [
        {"role": "assistant", "content": "text", "tool_calls": []},
        {
            "role": "assistant",
            "content": "tools",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
    ]

    converted = _openai_tool_messages(messages)

    assert "tool_calls" not in converted[0]
    assert converted[1]["tool_calls"] == messages[1]["tool_calls"]


def test_provider_status_code_reads_the_rejection_without_prompt_text() -> None:
    """A status code classifies a rejection and carries no prompt content."""

    class BadRequest(Exception):
        def __init__(self, message: str) -> None:
            super().__init__(message)
            self.status_code = 400

    assert provider_status_code(BadRequest("max_tokens is too large")) == 400

    # SDK wrappers hide the status behind a cause, and httpx behind a response.
    try:
        raise ValueError("wrapped") from BadRequest("max_tokens is too large")
    except ValueError as wrapped:
        assert provider_status_code(wrapped) == 400

    response = SimpleNamespace(status_code=429)

    class RateLimited(Exception):
        def __init__(self) -> None:
            super().__init__("slow down")
            self.response = response

    assert provider_status_code(RateLimited()) == 429

    # Absent, non-HTTP, and non-error statuses classify nothing.
    assert provider_status_code(ValueError("just bad")) is None

    class Ok(Exception):
        def __init__(self) -> None:
            super().__init__("fine")
            self.status_code = 200

    class Misdeclared(Exception):
        def __init__(self) -> None:
            super().__init__("fine")
            self.status_code = "400"

    assert provider_status_code(Ok()) is None
    assert provider_status_code(Misdeclared()) is None


class TestProviderUsageDialects:
    """One fact — the prompt a provider billed — read from each provider's counters."""

    def test_a_total_counter_is_the_billed_prompt_including_its_cache(self) -> None:
        # DeepSeek and OpenAI state the total, and the hit is a subset of it.
        assert (
            provider_input_tokens(
                {
                    "prompt_tokens": 47_442,
                    "prompt_cache_hit_tokens": 0,
                    "prompt_cache_miss_tokens": 47_442,
                }
            )
            == 47_442
        )
        assert provider_cache_hit_tokens({"prompt_tokens": 47_442}) is None

    def test_anthropic_excludes_its_cache_siblings_from_the_input_counter(self) -> None:
        usage = {
            "input_tokens": 1_000,
            "cache_read_input_tokens": 9_000,
            "cache_creation_input_tokens": 500,
        }

        assert provider_input_tokens(usage) == 10_500
        assert provider_cache_hit_tokens(usage) == 9_000

    def test_an_unstated_prompt_is_unknown_rather_than_zero(self) -> None:
        assert provider_input_tokens(None) is None
        assert provider_input_tokens({}) is None
        assert provider_input_tokens({"completion_tokens": 12}) is None

    def test_a_reported_zero_hit_is_a_measured_miss(self) -> None:
        assert provider_cache_hit_tokens({"prompt_cache_hit_tokens": 0}) == 0

    def test_a_recorded_usage_record_yields_the_counters_it_wraps(self) -> None:
        """A Run's usage record is not the provider's payload, but it is a usage shape.

        The record nests the counters under ``usage_details`` with child and inclusive
        breakdowns beside them; a Session Entry has recorded one since Fast turns began
        committing, and a reader that took it for provider counters failed the Run.
        """
        record = {
            "usage_details": {"prompt_tokens": 18_211, "prompt_cache_hit_tokens": 384},
            "child_usage_details": {"prompt_tokens": 900},
            "inclusive_usage_details": {"prompt_tokens": 19_111},
        }

        assert usage_counters(record) == {"prompt_tokens": 18_211, "prompt_cache_hit_tokens": 384}
        assert provider_input_tokens(usage_counters(record)) == 18_211
        assert provider_cache_hit_tokens(usage_counters(record)) == 384

    def test_counters_pass_through_and_unstated_usage_stays_unstated(self) -> None:
        assert usage_counters({"prompt_tokens": 4_096}) == {"prompt_tokens": 4_096}
        assert usage_counters({"prompt_tokens": 4_096, "extra": {"deep": 1}}) == {
            "prompt_tokens": 4_096,
            "extra.deep": 1,
        }
        assert usage_counters(None) is None
        assert usage_counters({}) is None


class TestReasoningControlRejection:
    """An endpoint that refuses our reasoning control says so, by name.

    An uncatalogued endpoint resolves to a best-effort, unverified level map, so the
    configured level travels as-is and the provider decides. Without this
    classification that decision reads as "Model provider rejected the request (HTTP
    400)", which cannot be told apart from a bad question or a broken prompt.
    """

    class _Rejection(Exception):
        def __init__(self, message: str, status: int | None = 400) -> None:
            super().__init__(message)
            if status is not None:
                self.status_code = status

    @pytest.mark.parametrize(
        "message",
        [
            # DeepSeek/OpenAI dialects name the parameter they refused.
            "Error code: 400 - Unsupported value: 'reasoning_effort' does not support 'max'",
            "400 invalid_request_error: unknown parameter 'thinking'",
            # Gemini and vLLM-shaped refusals name their own control.
            "422 Unprocessable Entity: thinking_config.thinking_level must be one of LOW, HIGH",
            "400 Bad Request: enable_thinking is not supported by this model",
            # Anthropic's adaptive thinking rides in output_config.
            "400 invalid_request_error: output_config: unexpected field",
            # A bare control name still counts, beside rejection wording.
            "400 invalid_request_error: reasoning is not allowed for this model",
        ],
    )
    def test_a_refused_control_classifies(self, message: str) -> None:
        assert provider_reasoning_rejection(self._Rejection(message)) is True

    @pytest.mark.parametrize(
        ("message", "status"),
        [
            # A rejection of something else is not this failure.
            ("400 Unsupported value: 'temperature' does not support 0.9", 400),
            ("400 prompt is too long: 300000 tokens > 200000 maximum", 400),
            # The words alone are not a refusal: usage prose reports reasoning tokens.
            ("reasoning tokens: 512", None),
            ("reasoning_effort: max", 400),
            # A 5xx is not a verdict about the request's shape.
            ("400 unsupported parameter: reasoning_effort", 503),
            # Our own echo fields carry reasoning *data*; complaining about one of them
            # means our replay is wrong, not that the endpoint lacks the level.
            ("400 invalid_request_error: reasoning_content is missing in assistant message", 400),
            ("400 invalid reasoning_details signature", 400),
        ],
    )
    def test_everything_else_stays_unclassified(self, message: str, status: int | None) -> None:
        assert provider_reasoning_rejection(self._Rejection(message, status)) is False

    def test_the_failure_detail_names_the_control_and_the_remedy(self) -> None:
        from dlightrag.engine.answer.errors import (
            REASONING_CONTROL_REJECTED_MESSAGE,
            reasoning_control_rejection_message,
        )
        from dlightrag.engine.answer.research.runtime import provider_attempt_detail

        rejection = self._Rejection(
            "400 Unsupported value: 'reasoning_effort' does not support 'max'"
        )
        overflow = self._Rejection("400 prompt is too long: 300000 tokens > 200000 maximum")

        assert (
            provider_attempt_detail(rejection, retryable=False)
            == f"{REASONING_CONTROL_REJECTED_MESSAGE} (HTTP 400)"
        )
        # A refusal of anything else keeps the generic verdict.
        assert provider_attempt_detail(overflow, retryable=False) == (
            "Model provider rejected the request (HTTP 400)"
        )
        assert reasoning_control_rejection_message(overflow) is None
        # A temporal verdict keeps its own wording even if the text matches.
        assert "temporarily unavailable" in provider_attempt_detail(rejection, retryable=True)
