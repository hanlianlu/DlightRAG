# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""SDK-native attachment and replay contract.

Every case drives a production provider or model binding through the installed
SDK's validation and serialization. Only HTTP transport is replaced.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI

from dlightrag.engine.agent.environment import AccessScheduler
from dlightrag.engine.agent.tool_content import (
    tool_content_attachments,
    tool_content_message_fields,
)
from dlightrag.engine.agent.tools.files import PreparedImageAttachment, view_tool
from dlightrag.engine.ai.completion import CompletionModel
from dlightrag.engine.ai.fingerprints import model_invocation_fingerprint
from dlightrag.engine.ai.media import decode_image_base64
from dlightrag.engine.ai.messages import (
    AssistantTurn,
    ToolCall,
    ToolDefinition,
    tool_call_message,
)
from dlightrag.engine.ai.providers import get_provider
from dlightrag.engine.ai.providers.anthropic_native import AnthropicProvider
from dlightrag.engine.ai.providers.gemini_native import GeminiProvider
from dlightrag.engine.ai.providers.openai_compatible import OpenAICompatibleProvider
from dlightrag.engine.ai.replay import bind_provider_replay
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import ModelSettings
from dlightrag.engine.ai.tool_model import ToolModel
from dlightrag.engine.answer.resources.models import ResourceInput
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.tools.resources import make_resource_viewer
from tests.unit.conftest import answer_image_policy
from tests.unit.test_resource_tools import call
from tests.unit.test_resource_visual import pdf_bytes

PAGE_ONE = b"\x89PNG\r\n\x1a\npage-one"
PAGE_TWO = b"\x89PNG\r\n\x1a\npage-two"
_PAYLOADS = (PAGE_ONE, PAGE_TWO)
_TOOL_TEXT = "pages rendered"
_MODEL_TEXT = "ok"
_PRIVATE_WIRE_KEYS = frozenset({"attachments", "provider_state", "is_error", "untrusted_tool_data"})
_OPENAI_USER_KEYS = frozenset({"role", "content", "name"})
_OPENAI_TOOL_KEYS = frozenset({"role", "content", "tool_call_id", "name"})
_GEMINI_THOUGHT_SIGNATURE = b"native-sig-bytes"
_GEMINI_THOUGHT_SIGNATURE_B64 = base64.b64encode(_GEMINI_THOUGHT_SIGNATURE).decode()
_ANTHROPIC_THINKING_SIGNATURE = "anth-thinking-signature"
_VIEW_TOOL = ToolDefinition(
    name="view",
    description="view a page",
    parameters={"type": "object", "properties": {"locator": {"type": "string"}}},
)
_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.0-flash",
}
_ENTRYPOINTS = (
    "complete",
    "stream",
    "complete_tool_turn",
    "complete_tool_turn_streaming",
    "stream_tool_text",
)


def _data_url(payload: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(payload).decode()}"


def _page_attachment(index: int) -> dict[str, object]:
    payload = _PAYLOADS[index % 2]
    return {
        "resource_id": f"page-{index}",
        "safe_name": f"page-{index}.png",
        "media_type": "image/png",
        "content_digest": "a" * 64,
        "size_bytes": len(payload),
        "data_url": _data_url(payload),
        "source": {"kind": "pdf_page", "page": index + 1, "resource_id": f"doc-{index}"},
    }


def _tool_turn(count: int) -> list[dict[str, object]]:
    message: dict[str, object] = {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "view",
        "content": _TOOL_TEXT,
        "is_error": False,
    }
    if count:
        message["attachments"] = [_page_attachment(index) for index in range(count)]
    return [{"role": "user", "content": "look"}, message]


def _expected_payloads(count: int) -> list[bytes]:
    return [_PAYLOADS[index % 2] for index in range(count)]


def _openai_complete_json() -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _MODEL_TEXT},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
    }


def _openai_response_json() -> dict[str, Any]:
    return {
        "id": "resp-1",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": 128,
        "model": "gpt-5.4",
        "output": [
            {
                "id": "msg-1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": _MODEL_TEXT,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": 0.2,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 0.95,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 11,
            "input_tokens_details": {"cached_tokens": 7},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 1},
            "total_tokens": 13,
        },
    }


def _openai_stream_sse() -> bytes:
    chunks = [
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": _MODEL_TEXT},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
        },
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return f"{body}data: [DONE]\n\n".encode()


def _anthropic_complete_json() -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": _MODEL_TEXT}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 2},
    }


def _anthropic_stream_sse() -> bytes:
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-sonnet-4-20250514",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": _MODEL_TEXT},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(
        f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events
    ).encode()


def _gemini_complete_json(*, thought_signature: bytes | None = None) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": _MODEL_TEXT}]
    if thought_signature is not None:
        parts.append(
            {
                "functionCall": {"id": "call-2", "name": "view", "args": {"locator": "1"}},
                "thoughtSignature": base64.b64encode(thought_signature).decode(),
            }
        )
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": parts},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 2,
            "totalTokenCount": 13,
        },
    }


def _gemini_stream_sse() -> bytes:
    payload = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": _MODEL_TEXT}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 2,
            "totalTokenCount": 13,
        },
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


class _HttpCapture:
    def __init__(
        self,
        provider: str,
        *,
        response_jsons: list[dict[str, Any]] | None = None,
    ) -> None:
        self.provider = provider
        self.requests: list[dict[str, Any]] = []
        self.response_jsons = list(response_jsons or [])

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        url = str(request.url)
        if self.provider == "openai" and "/chat/completions" not in url:
            raise AssertionError(f"unexpected OpenAI URL {url}")
        if self.provider == "openai_response" and not url.endswith("/responses"):
            raise AssertionError(f"unexpected OpenAI Responses URL {url}")
        if self.provider == "anthropic" and "/v1/messages" not in url:
            raise AssertionError(f"unexpected Anthropic URL {url}")
        if self.provider == "gemini" and "generativelanguage.googleapis.com" not in url:
            raise AssertionError(f"unexpected Gemini URL {url}")
        body = json.loads(request.content.decode())
        self.requests.append({"url": url, "body": body})
        streamed = bool(body.get("stream")) or "streamGenerateContent" in url
        if self.provider == "openai":
            if streamed:
                return httpx2.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_openai_stream_sse(),
                )
            return httpx2.Response(200, json=_openai_complete_json())
        if self.provider == "openai_response":
            payload = self.response_jsons.pop(0) if self.response_jsons else _openai_response_json()
            return httpx2.Response(200, json=payload)
        if self.provider == "anthropic":
            if streamed:
                return httpx2.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_anthropic_stream_sse(),
                )
            return httpx2.Response(200, json=_anthropic_complete_json())
        if streamed:
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_gemini_stream_sse(),
            )
        return httpx2.Response(200, json=_gemini_complete_json())

    @property
    def body(self) -> dict[str, Any]:
        assert self.requests, "SDK made no HTTP request"
        return self.requests[-1]["body"]


def bind_mock_http(provider: object, handler: Callable[..., Any]) -> None:
    """Factory wrapper: keep the real SDK client, inject only HTTP transport."""
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    if isinstance(provider, OpenAICompatibleProvider):
        provider._client = AsyncOpenAI(
            api_key=provider._api_key or "test-key",
            base_url=provider._base_url or "https://api.openai.com/v1",
            timeout=provider._timeout,
            max_retries=0,
            http_client=http,
        )
        return
    if isinstance(provider, AnthropicProvider):
        provider._client = AsyncAnthropic(
            api_key=provider._api_key or "test-key",
            timeout=provider._timeout,
            max_retries=0,
            http_client=http,
        )
        return
    if isinstance(provider, GeminiProvider):
        provider._client = genai.Client(
            api_key=provider._api_key or "test-key",
            http_options=genai.types.HttpOptions(httpx_async_client=http),
        )
        return
    raise TypeError(f"unsupported provider {type(provider)!r}")


def _provider(name: str, capture: _HttpCapture) -> Any:
    provider = get_provider(name, api_key="test-key", max_retries=0)
    bind_mock_http(provider, capture.handler)
    return provider


def _wire_images(provider: str, body: dict[str, Any]) -> list[bytes]:
    if provider == "openai":
        images: list[bytes] = []
        for message in body["messages"]:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") == "image_url":
                    url = str(part["image_url"]["url"])
                    images.append(base64.b64decode(url.partition(",")[2]))
        return images
    if provider == "anthropic":
        images = []
        for message in body["messages"]:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                inner = block.get("content") if block.get("type") == "tool_result" else block
                parts = inner if isinstance(inner, list) else [inner]
                for part in parts:
                    if not isinstance(part, dict) or part.get("type") != "image":
                        continue
                    source = part.get("source") or {}
                    if source.get("type") == "base64":
                        images.append(base64.b64decode(source["data"]))
        return images
    images = []
    for content in body["contents"]:
        for part in content.get("parts") or ():
            inline = part.get("inlineData")
            if inline:
                images.append(base64.b64decode(inline["data"]))
    return images


def _assert_no_private_fields(provider: str, body: dict[str, Any]) -> None:
    payload = json.dumps(body)
    for key in ("attachments", "provider_state", "untrusted_tool_data"):
        assert f'"{key}"' not in payload
    if provider == "openai":
        assert '"is_error"' not in payload
        for message in body["messages"]:
            assert _PRIVATE_WIRE_KEYS.isdisjoint(message)
            if message.get("role") == "user":
                assert set(message) <= _OPENAI_USER_KEYS
            if message.get("role") == "tool":
                assert set(message) <= _OPENAI_TOOL_KEYS
                assert message["tool_call_id"] == "call-1"
                assert message["content"] == _TOOL_TEXT
    elif provider == "anthropic":
        for message in body["messages"]:
            assert set(message) <= {"role", "content"}
            assert "is_error" not in message
    else:
        for content in body["contents"]:
            assert set(content) <= {"role", "parts"}
            assert isinstance(content["parts"], list)
            for part in content["parts"]:
                assert isinstance(part, dict)
                assert "is_error" not in part


def _assert_gemini_user_text(body: dict[str, Any]) -> None:
    first = body["contents"][0]
    assert first["role"] == "user"
    assert first["parts"] == [{"text": "look"}]


async def _invoke(
    provider: Any, entrypoint: str, messages: list[dict[str, object]], model: str
) -> Any:
    if entrypoint == "complete":
        result = await provider.complete(messages, model)
        assert result.usage_details
        assert str(result) == _MODEL_TEXT
        return result
    if entrypoint == "stream":
        holder: dict[str, Any] = {}
        tokens = [token async for token in provider.stream(messages, model, usage_holder=holder)]
        assert tokens == [_MODEL_TEXT]
        assert holder.get("usage_details")
        return tokens
    if entrypoint == "complete_tool_turn":
        turn = await provider.complete_tool_turn(messages, model, tools=[_VIEW_TOOL])
        assert turn.text == _MODEL_TEXT
        assert turn.usage_details
        return turn
    if entrypoint == "complete_tool_turn_streaming":
        emitted: list[str] = []

        async def emit_text(text: str) -> None:
            emitted.append(text)

        turn = await provider.complete_tool_turn_streaming(
            messages, model, tools=[_VIEW_TOOL], emit_text=emit_text
        )
        assert emitted == [_MODEL_TEXT]
        assert turn.text == _MODEL_TEXT
        assert turn.usage_details
        return turn
    if entrypoint == "stream_tool_text":
        holder = {}
        tokens = [
            token async for token in provider.stream_tool_text(messages, model, usage_holder=holder)
        ]
        assert tokens == [_MODEL_TEXT]
        assert holder.get("usage_details")
        return tokens
    raise AssertionError(f"unhandled entrypoint {entrypoint!r}")


async def test_unknown_invoke_entrypoint_fails_closed() -> None:
    with pytest.raises(AssertionError, match="unhandled entrypoint 'not_an_entrypoint'"):
        await _invoke(object(), "not_an_entrypoint", [], "gpt-4o")


@pytest.mark.parametrize("provider_name", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("count", [0, 1, 2])
@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
async def test_sdk_http_projects_tool_attachments(
    provider_name: str, count: int, entrypoint: str
) -> None:
    capture = _HttpCapture(provider_name)
    provider = _provider(provider_name, capture)
    original = _tool_turn(count)
    snapshot = json.loads(json.dumps(original))
    try:
        await _invoke(provider, entrypoint, original, _MODELS[provider_name])
    finally:
        await provider.aclose()
    assert original == snapshot
    body = capture.body
    _assert_no_private_fields(provider_name, body)
    assert _wire_images(provider_name, body) == _expected_payloads(count)
    if provider_name == "openai":
        roles = [message["role"] for message in body["messages"]]
        assert roles == (["user", "tool"] if count == 0 else ["user", "tool", "user"])
        if count:
            follow = body["messages"][2]["content"]
            # Images ride alone. Repeating the tool text here would also fabricate a
            # user turn for an attachment that carries no bytes, and any user turn
            # between tool messages ends the batch the provider is still matching.
            assert [part["type"] for part in follow] == ["image_url"] * count
    elif provider_name == "anthropic":
        block = body["messages"][-1]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "call-1"
        if count == 0:
            assert block["content"] == _TOOL_TEXT
        else:
            assert block["content"][0] == {"type": "text", "text": _TOOL_TEXT}
    else:
        _assert_gemini_user_text(body)
        last = body["contents"][-1]
        assert last["role"] == "user"
        response = last["parts"][0]["functionResponse"]
        assert response["id"] == "call-1"
        assert response["name"] == "view"
        assert response["response"]["output"] == _TOOL_TEXT
        assert response["response"]["is_error"] is False


async def test_sdk_http_keeps_an_image_bearing_tool_batch_contiguous() -> None:
    """One image-bearing result must not cut its batch: the provider answers the
    remaining calls with HTTP 400 'insufficient tool messages following tool_calls
    message', which is how a single rendered page made every later request on a
    session fail."""
    capture = _HttpCapture("openai")
    provider = _provider("openai", capture)
    tool_calls = [
        {
            "id": f"call-{index}",
            "type": "function",
            "function": {"name": "view", "arguments": "{}"},
        }
        for index in (1, 2)
    ]
    messages = [
        {"role": "user", "content": "look"},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        _tool_turn(1)[1],
        {"role": "tool", "tool_call_id": "call-2", "name": "view", "content": "second page"},
    ]
    try:
        await _invoke(provider, "complete", messages, _MODELS["openai"])
    finally:
        await provider.aclose()
    serialized = json.dumps(capture.body)
    for key in ("attachments", "provider_state", "untrusted_tool_data", "is_error"):
        assert f'"{key}"' not in serialized
    assert all(_PRIVATE_WIRE_KEYS.isdisjoint(message) for message in capture.body["messages"])
    assert [message["role"] for message in capture.body["messages"]] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
    ]
    assert _wire_images("openai", capture.body) == _expected_payloads(1)


async def test_response_tool_loop_replays_finalized_native_items_and_exact_call_ids() -> None:
    first = _openai_response_json()
    first_output = [
        {
            "id": "rs-1",
            "type": "reasoning",
            "status": "completed",
            "summary": [{"type": "summary_text", "text": "Need two lookups."}],
            "content": [],
            "encrypted_content": "opaque-ciphertext",
        },
        {
            "id": "msg-1",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "I will check.",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        },
        {
            "id": "fc-item-1",
            "type": "function_call",
            "status": "completed",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"value":"one"}',
        },
        {
            "id": "fc-item-2",
            "type": "function_call",
            "status": "completed",
            "call_id": "call-2",
            "name": "lookup",
            "arguments": '{"value":"two"}',
        },
    ]
    first["output"] = first_output
    capture = _HttpCapture(
        "openai_response",
        response_jsons=[first, _openai_response_json()],
    )
    model = ToolModel(
        ModelSettings(
            provider="openai",
            model="gpt-5.4",
            api_family="response",
            api_key="test-key",
            max_retries=0,
        ),
        scheduler=ModelScheduler(max_concurrency=1),
    )
    provider = model._provider
    bind_mock_http(provider, capture.handler)
    lookup = ToolDefinition(
        name="lookup",
        description="Look up one value.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    user = {"role": "user", "content": "Look up two values."}

    try:
        first_turn = await model(messages=[user], tools=[lookup])
        assistant = {
            "role": "assistant",
            "content": first_turn.text,
            "tool_calls": [tool_call_message(call) for call in first_turn.tool_calls],
            "provider_state": first_turn.provider_state,
        }
        second_turn = await model(
            messages=[
                user,
                assistant,
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "lookup",
                    "content": "result one",
                    "is_error": False,
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-2",
                    "name": "lookup",
                    "content": "result two",
                    "is_error": False,
                },
            ],
            tools=[lookup],
        )
    finally:
        await model.aclose()

    assert first_turn.text == "I will check."
    assert first_turn.reasoning == "Need two lookups."
    assert "opaque-ciphertext" not in first_turn.reasoning
    assert [call.id for call in first_turn.tool_calls] == ["call-1", "call-2"]
    assert [call.name for call in first_turn.tool_calls] == ["lookup", "lookup"]
    assert first_turn.stop_reason == "tool_use"
    assert first_turn.provider_state is not None
    assert first_turn.provider_state["payload"] == {
        "response_replay": {"v": 1, "items": first_output}
    }
    assert second_turn.text == _MODEL_TEXT
    assert second_turn.stop_reason == "stop"

    first_body = capture.requests[0]["body"]
    assert first_body["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Look up one value.",
            "parameters": lookup.parameters,
            "strict": False,
        }
    ]
    assert first_body["tool_choice"] == "auto"
    assert first_body["parallel_tool_calls"] is True
    assert "messages" not in first_body

    second_input = capture.requests[1]["body"]["input"]
    assert second_input == [
        user,
        *first_output,
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "result one",
        },
        {
            "type": "function_call_output",
            "call_id": "call-2",
            "output": "result two",
        },
    ]
    assert sum(item.get("id") == "fc-item-1" for item in second_input) == 1
    assert sum(item.get("call_id") == "call-1" for item in second_input) == 2


async def test_response_complete_uses_stateless_responses_wire_without_mutating_input() -> None:
    capture = _HttpCapture("openai_response")
    provider = get_provider(
        "openai",
        api_key="test-key",
        api_family="response",
        max_retries=0,
    )
    bind_mock_http(provider, capture.handler)
    messages = [
        {"role": "system", "content": "You are exact."},
        {"role": "user", "content": [{"type": "text", "text": "look"}]},
        {"role": "assistant", "content": "Earlier answer."},
        {"role": "user", "content": "Return JSON."},
    ]
    original = json.loads(json.dumps(messages))
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    try:
        result = await provider.complete(
            messages,
            "gpt-5.4",
            temperature=0.2,
            max_tokens=128,
            response_format=response_format,
            model_kwargs={"top_p": 0.95},
        )
    finally:
        await provider.aclose()

    assert messages == original
    assert result == _MODEL_TEXT
    assert result.stop_reason == "stop"
    assert result.usage_details == {
        "input_tokens": 11,
        "input_tokens_details.cached_tokens": 7,
        "output_tokens": 2,
        "output_tokens_details.reasoning_tokens": 1,
        "total_tokens": 13,
    }
    assert capture.requests[-1]["url"] == "https://api.openai.com/v1/responses"
    assert capture.body == {
        "model": "gpt-5.4",
        "input": [
            {"role": "system", "content": "You are exact."},
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "look"}],
            },
            {"role": "assistant", "content": "Earlier answer."},
            {"role": "user", "content": "Return JSON."},
        ],
        "background": False,
        "store": False,
        "truncation": "disabled",
        "temperature": 0.2,
        "max_output_tokens": 128,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": response_format["json_schema"]["schema"],
                "strict": True,
            }
        },
        "top_p": 0.95,
    }


async def test_openai_plain_text_complete_has_no_follow_up_user_turn() -> None:
    capture = _HttpCapture("openai")
    provider = _provider("openai", capture)
    try:
        result = await provider.complete([{"role": "user", "content": "look"}], "gpt-4o")
    finally:
        await provider.aclose()
    assert result == _MODEL_TEXT
    assert [message["role"] for message in capture.body["messages"]] == ["user"]
    assert capture.body["messages"][0]["content"] == "look"
    _assert_no_private_fields("openai", capture.body)


async def test_completion_model_replays_same_model_anthropic_thinking_bytes() -> None:
    settings = ModelSettings(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        max_retries=0,
    )
    capture = _HttpCapture("anthropic")
    model = CompletionModel(settings, scheduler=ModelScheduler(max_concurrency=1))
    bind_mock_http(model._provider, capture.handler)
    bound = bind_provider_replay(
        AssistantTurn(
            text="hi",
            tool_calls=(),
            stop_reason="stop",
            provider_state={
                "thinking_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "Think.",
                        "signature": _ANTHROPIC_THINKING_SIGNATURE,
                    }
                ]
            },
        ),
        model.fingerprint,
    )
    messages = [
        {"role": "user", "content": "look"},
        {
            "role": "assistant",
            "content": bound.text,
            "provider_state": bound.provider_state,
        },
        *_tool_turn(1)[1:],
    ]
    original = json.loads(json.dumps(messages))
    try:
        result = await model(messages)
    finally:
        await model.aclose()
    assert result == _MODEL_TEXT
    assert messages == original
    thinking = capture.body["messages"][1]["content"][0]
    assert thinking == {
        "type": "thinking",
        "thinking": "Think.",
        "signature": _ANTHROPIC_THINKING_SIGNATURE,
    }
    assert _wire_images("anthropic", capture.body) == [PAGE_ONE]


async def test_completion_model_strips_cross_model_anthropic_thinking() -> None:
    source = ModelSettings(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        max_retries=0,
    )
    target = ModelSettings(
        provider="anthropic",
        model="claude-opus-4-20250514",
        api_key="test-key",
        max_retries=0,
    )
    capture = _HttpCapture("anthropic")
    model = CompletionModel(target, scheduler=ModelScheduler(max_concurrency=1))
    bind_mock_http(model._provider, capture.handler)
    bound = bind_provider_replay(
        AssistantTurn(
            text="hi",
            tool_calls=(),
            stop_reason="stop",
            provider_state={
                "thinking_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "Think.",
                        "signature": _ANTHROPIC_THINKING_SIGNATURE,
                    }
                ]
            },
        ),
        model_invocation_fingerprint(source),
    )
    try:
        await model(
            [
                {"role": "user", "content": "look"},
                {
                    "role": "assistant",
                    "content": bound.text,
                    "provider_state": bound.provider_state,
                },
            ]
        )
    finally:
        await model.aclose()
    content = capture.body["messages"][1]["content"]
    assert content == [{"type": "text", "text": "hi"}]
    serialized = json.dumps(capture.body)
    assert _ANTHROPIC_THINKING_SIGNATURE not in serialized
    assert "Think." not in serialized


async def test_tool_model_gemini_signature_round_trips_native_bytes() -> None:
    settings = ModelSettings(
        provider="gemini",
        model="gemini-2.0-flash",
        api_key="test-key",
        max_retries=0,
    )
    capture = _HttpCapture("gemini")
    model = ToolModel(settings, scheduler=ModelScheduler(max_concurrency=1))
    bind_mock_http(model._provider, capture.handler)
    bound = bind_provider_replay(
        AssistantTurn(
            text="",
            tool_calls=(
                ToolCall(
                    id="call-1",
                    name="view",
                    arguments={"locator": "1"},
                    thought_signature=_GEMINI_THOUGHT_SIGNATURE,
                ),
            ),
            stop_reason="tool_use",
        ),
        model.fingerprint,
    )
    messages = [
        {"role": "user", "content": "look"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": bound.tool_calls[0].id,
                    "type": "function",
                    "function": {
                        "name": "view",
                        "arguments": '{"locator":"1"}',
                    },
                    "thought_signature": bound.tool_calls[0].thought_signature,
                }
            ],
            "provider_state": bound.provider_state,
        },
        _tool_turn(2)[1],
    ]
    try:
        turn = await model(messages=messages, tools=[_VIEW_TOOL])
    finally:
        await model.aclose()
    assert turn.text == _MODEL_TEXT
    part = capture.body["contents"][1]["parts"][0]
    assert part["thoughtSignature"] == _GEMINI_THOUGHT_SIGNATURE_B64
    assert base64.b64decode(part["thoughtSignature"]) == _GEMINI_THOUGHT_SIGNATURE
    assert _wire_images("gemini", capture.body) == [PAGE_ONE, PAGE_TWO]


async def test_completion_model_strips_cross_model_gemini_signature() -> None:
    source = ModelSettings(
        provider="gemini",
        model="gemini-2.0-flash",
        api_key="test-key",
        max_retries=0,
    )
    target = ModelSettings(
        provider="gemini",
        model="gemini-2.5-pro",
        api_key="test-key",
        max_retries=0,
    )
    capture = _HttpCapture("gemini")
    model = CompletionModel(target, scheduler=ModelScheduler(max_concurrency=1))
    bind_mock_http(model._provider, capture.handler)
    bound = bind_provider_replay(
        AssistantTurn(
            text="",
            tool_calls=(
                ToolCall(
                    id="call-1",
                    name="view",
                    arguments={"locator": "1"},
                    thought_signature=_GEMINI_THOUGHT_SIGNATURE,
                ),
            ),
            stop_reason="tool_use",
        ),
        model_invocation_fingerprint(source),
    )
    try:
        stream = await model(
            [
                {"role": "user", "content": "look"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "view",
                                "arguments": '{"locator":"1"}',
                            },
                            "thought_signature": bound.tool_calls[0].thought_signature,
                        }
                    ],
                    "provider_state": bound.provider_state,
                },
            ],
            stream=True,
        )
        tokens = [token async for token in stream]
    finally:
        await model.aclose()
    assert tokens == [_MODEL_TEXT]
    serialized = json.dumps(capture.body)
    assert "thoughtSignature" not in serialized
    assert _GEMINI_THOUGHT_SIGNATURE_B64 not in serialized
    first = capture.body["contents"][1]["parts"][0]
    assert "thoughtSignature" not in first
    assert first["functionCall"]["id"] == "call-1"


async def test_anthropic_multipage_view_uses_two_rendered_pages_through_tool_model() -> None:
    capture = _HttpCapture("anthropic")
    settings = ModelSettings(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        max_retries=0,
    )
    model = ToolModel(settings, scheduler=ModelScheduler(max_concurrency=1))
    bind_mock_http(model._provider, capture.handler)
    budget = answer_image_policy(max_images=4).new_budget()

    def prepare(data: bytes, label: str) -> PreparedImageAttachment | None:
        block = budget.add_base64(base64.b64encode(data).decode(), label=label)
        if block is None:
            return None
        content, media = decode_image_base64(block["image_url"]["url"])
        return PreparedImageAttachment(content, media or "image/png", content != data)

    async with ResourceRegistry() as registry:
        resource = registry.register(ResourceInput(filename="pages.pdf", content=pdf_bytes(2)))
        view = view_tool(
            None,
            AccessScheduler(),
            resource_viewer=make_resource_viewer(registry),
            image_preparer=prepare,
        )
        first = await call(view, resource_id=resource, locator="1")
        second = await call(view, resource_id=resource, locator="2")
    charged = budget.count
    assert charged == 2
    attachments = (
        *tool_content_attachments(first.parts),
        *tool_content_attachments(second.parts),
    )
    assert len(attachments) == 2
    rendered = [part.data for part in attachments]
    assert rendered[0] != rendered[1]
    assert all(payload.startswith(b"\x89PNG") for payload in rendered)
    messages = [
        {"role": "user", "content": "look"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "view",
                        "arguments": '{"locator":"1"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "view",
            **tool_content_message_fields(first.parts),
            "is_error": first.is_error,
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "view",
                        "arguments": '{"locator":"2"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "name": "view",
            **tool_content_message_fields(second.parts),
            "is_error": second.is_error,
        },
    ]
    try:
        turn = await model(messages=messages, tools=[_VIEW_TOOL])
    finally:
        await model.aclose()
    assert turn.text == _MODEL_TEXT
    assert _wire_images("anthropic", capture.body) == rendered
    ids = [
        block["tool_use_id"]
        for message in capture.body["messages"]
        if message["role"] == "user" and isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert ids == ["call-1", "call-2"]
    assert budget.count == charged


async def test_gemini_answers_a_whole_tool_batch_in_one_matched_turn() -> None:
    """One model turn's calls are answered by one turn holding every response.

    Projecting a turn per tool message leaves the model turn under-answered and
    strands the trailing result behind an unrelated user turn.
    """
    settings = ModelSettings(
        provider="gemini",
        model="gemini-2.0-flash",
        api_key="test-key",
        max_retries=0,
    )
    capture = _HttpCapture("gemini")
    model = ToolModel(settings, scheduler=ModelScheduler(max_concurrency=1))
    bind_mock_http(model._provider, capture.handler)
    messages = [
        {"role": "user", "content": "look"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "view", "arguments": '{"locator":"1"}'},
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "view", "arguments": '{"locator":"2"}'},
                },
            ],
        },
        _tool_turn(2)[1],
        {**_tool_turn(0)[1], "tool_call_id": "call-2"},
    ]
    try:
        turn = await model(messages=messages, tools=[_VIEW_TOOL])
    finally:
        await model.aclose()
    assert turn.text == _MODEL_TEXT
    contents = capture.body["contents"]
    assert [content["role"] for content in contents] == ["user", "model", "user"]
    calls = contents[1]["parts"]
    assert [part["functionCall"]["id"] for part in calls] == ["call-1", "call-2"]
    answers = contents[2]["parts"]
    responses = [part["functionResponse"] for part in answers if "functionResponse" in part]
    assert [response["id"] for response in responses] == ["call-1", "call-2"]
    # The image rides as an ordinary part of the answering turn, and the wire
    # keeps camelCase because only top-level parts get the alias conversion.
    assert [part for part in answers if "inlineData" in part] == [
        {
            "inlineData": {
                "data": base64.b64encode(PAGE_ONE).decode(),
                "mimeType": "image/png",
            }
        },
        {
            "inlineData": {
                "data": base64.b64encode(PAGE_TWO).decode(),
                "mimeType": "image/png",
            }
        },
    ]
    assert _wire_images("gemini", capture.body) == _expected_payloads(2)
