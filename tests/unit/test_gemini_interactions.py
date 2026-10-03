# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Gemini through the Interactions API: what Gemini receives and what a Session keeps.

Each case drives production models through the client the provider builds itself,
with google-genai's own Interactions transport replaced by a scripted endpoint. The
assertions read the REST bodies sent. Streams follow the shape of Google's own SSE
example: a thought starts with an empty signature and its first summary, and the
completed event carries status and usage but no steps.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx2
import pytest
from google.genai._gaos.utils import retries
from pydantic import BaseModel, ValidationError

from dlightrag.adapters.observability.tracing import _langfuse_usage_details
from dlightrag.engine.agent.session.effects import canonical_json
from dlightrag.engine.agent.session.entries import AssistantMessageEntry, decode_entry_payload
from dlightrag.engine.agent.session.fold import fold_assistant_message
from dlightrag.engine.agent.session.ids import EntryId, SessionId
from dlightrag.engine.ai.completion import CompletionModel
from dlightrag.engine.ai.fingerprints import (
    ModelInvocationFingerprint,
    model_invocation_fingerprint,
)
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall, ToolDefinition
from dlightrag.engine.ai.providers import get_provider, provider_for
from dlightrag.engine.ai.providers.base import (
    CompletionProvider,
    is_provider_context_overflow,
    is_provider_reasoning_rejection,
    provider_cache_hit_tokens,
    provider_input_tokens,
)
from dlightrag.engine.ai.providers.gemini_native import (
    InteractionRequestError,
    InteractionStatusError,
)
from dlightrag.engine.ai.reasoning import ReasoningConfigurationError
from dlightrag.engine.ai.replay import bind_provider_replay
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import ModelRoleSettings, ModelSettings, RerankSettings
from dlightrag.engine.ai.structured import StructuredOutput
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.ai.tool_model import ToolModel
from dlightrag.engine.ai.vision import probe_image_capability
from dlightrag.engine.dependencies import classify_transient_dependency
from tests.support.loopback import (
    bypass_proxies,
    loopback_certificate,
    loopback_server,
    reset_on_accept,
)

_MODEL = "gemini-3.8-flash"
_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
_PAGE = base64.b64encode(b"\x89PNG\r\n\x1a\npage-two").decode()
_PAGE_URL = f"data:image/png;base64,{_PAGE}"
_VIEW = ToolDefinition(
    name="view",
    description="View one page.",
    parameters={
        "type": "object",
        "properties": {"page": {"type": "integer"}},
        "required": ["page"],
    },
)
_QUESTION = {"role": "user", "content": "What does page 2 show?"}
_ASKED = {"type": "user_input", "content": [{"type": "text", "text": "What does page 2 show?"}]}
_SIGNATURE = "c2lnbmVkIHRob3VnaHQgb25l"
_THOUGHT = {
    "type": "thought",
    "signature": _SIGNATURE,
    "summary": [{"type": "text", "text": "Page 2 holds the table."}],
}
# Google's own stream example: 62 + 171 + 297 = 530, so the output excludes thoughts.
_USAGE = {
    "total_input_tokens": 62,
    "total_cached_tokens": 40,
    "total_output_tokens": 171,
    "total_thought_tokens": 297,
    "total_tokens": 530,
}
_COUNTERS = {
    "prompt_tokens": 62,
    "cached_content_tokens": 40,
    "completion_tokens": 468,
    "thoughts_tokens": 297,
    "total_tokens": 530,
}
_ENTRYPOINTS = (
    "complete",
    "stream",
    "complete_tool_turn",
    "complete_tool_turn_streaming",
    "stream_tool_text",
)


def _text(text: str) -> dict[str, Any]:
    return {"type": "model_output", "content": [{"type": "text", "text": text}]}


def _call(call_id: str, page: int) -> dict[str, Any]:
    return {"type": "function_call", "id": call_id, "name": "view", "arguments": {"page": page}}


def _function_result(call_id: str, result: Any, *, is_error: bool = False) -> dict[str, Any]:
    return {
        "type": "function_result",
        "call_id": call_id,
        "name": "view",
        "result": result,
        "is_error": is_error,
    }


def _interaction(*steps: dict[str, Any], status: str = "completed", **fields: Any) -> dict:
    return {
        "id": "interaction-1",
        "status": status,
        "steps": list(steps),
        "usage": _USAGE,
        **fields,
    }


def _sse(*events: dict[str, Any]) -> httpx2.Response:
    body = "".join(f"event: {e['event_type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    body += "event: done\ndata: [DONE]\n\n"
    return httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, content=body.encode()
    )


_CREATED = {
    "event_type": "interaction.created",
    "interaction": {"id": "i-1", "status": "in_progress", "object": "interaction"},
}


def _start(index: int, step: dict[str, Any]) -> dict[str, Any]:
    return {"event_type": "step.start", "index": index, "step": step}


def _delta(index: int, **delta: Any) -> dict[str, Any]:
    return {"event_type": "step.delta", "index": index, "delta": delta}


def _stop(index: int) -> dict[str, Any]:
    return {"event_type": "step.stop", "index": index}


def _completed(status: str = "completed") -> dict[str, Any]:
    """The terminal event as Google's example sends it: status and usage, no steps."""
    return {
        "event_type": "interaction.completed",
        "interaction": {"id": "i-1", "status": status, "usage": _USAGE},
    }


def _thought_events(index: int) -> list[dict[str, Any]]:
    """A thought as Google streams it: an empty signature and the first summary, then the
    signature in a delta of its own."""
    first, rest = "Page 2 holds ", "the table."
    return [
        _start(
            index,
            {"signature": "", "summary": [{"text": first, "type": "text"}], "type": "thought"},
        ),
        _delta(index, type="thought_summary", content={"type": "text", "text": rest}),
        _delta(index, type="thought_signature", signature=_SIGNATURE),
        _stop(index),
    ]


def _streamed_text(first: str, *rest: str, status: str = "completed") -> httpx2.Response:
    return _sse(
        _CREATED,
        _start(0, {"content": [{"text": first, "type": "text"}], "type": "model_output"}),
        *(_delta(0, type="text", text=text) for text in rest),
        _stop(0),
        _completed(status),
    )


class _Gemini:
    """A scripted Interactions endpoint: each request it got, and the replies it gives."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        return reply if isinstance(reply, httpx2.Response) else httpx2.Response(200, json=reply)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]


def _bind(provider: CompletionProvider, gemini: _Gemini) -> None:
    """Keep the client and options the provider builds; replace only its HTTP transport."""
    client = provider._get_client()  # pyright: ignore[reportAttributeAccessIssue]
    client._api_client._async_httpx_client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(gemini)
    )


class _WithoutPauses(ModuleType):
    """asyncio as the SDK's retry loop sees it, minus its backoff sleeps."""

    def __getattr__(self, name: str) -> object:
        return getattr(asyncio, name)

    @staticmethod
    async def sleep(_delay: float, result: object = None) -> object:
        return result


@pytest.fixture
def no_retry_pauses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retries, "asyncio", _WithoutPauses("asyncio"))


def _settings(**values: Any) -> ModelSettings:
    return ModelSettings.model_validate(
        {"provider": "gemini", "model": _MODEL, "api_key": "test-key", "max_retries": 1, **values}
    )


def _provider(gemini: _Gemini, **values: Any) -> CompletionProvider:
    provider = provider_for(_settings(**values))
    _bind(provider, gemini)
    return provider


def _tool_model(gemini: _Gemini, **values: Any) -> ToolModel:
    model = ToolModel(
        _settings(**values), scheduler=ModelScheduler(max_concurrency=1), telemetry=NOOP_TELEMETRY
    )
    _bind(model._provider, gemini)
    return model


def _completion_model(gemini: _Gemini, **values: Any) -> CompletionModel:
    model = CompletionModel(
        _settings(**values), scheduler=ModelScheduler(max_concurrency=1), telemetry=NOOP_TELEMETRY
    )
    _bind(model._provider, gemini)
    return model


def _stored(turn: AssistantTurn) -> dict[str, Any]:
    """The turn as a Session stores it, and the message a later request replays from it."""
    entry = AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=SessionId.new(),
        timestamp=datetime.now(UTC),
        content=turn.text,
        reasoning=turn.reasoning,
        stop_reason=turn.stop_reason,
        tool_calls=turn.tool_calls,
        usage=turn.usage_details,
        provider_state=turn.provider_state,
    )
    # The round trip the PostgreSQL repository makes: canonical JSON, then decode.
    payload = json.loads(canonical_json(entry.canonical_payload()))
    restored = decode_entry_payload(
        entry_type=entry.entry_type,
        entry_id=entry.entry_id,
        session_id=entry.session_id,
        sequence=entry.sequence,
        timestamp=entry.timestamp,
        payload=payload,
    )
    assert isinstance(restored, AssistantMessageEntry)
    return fold_assistant_message(restored)


def _result(call_id: str, text: str, *pages: str, is_error: bool = False) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": call_id,
        "name": "view",
        "content": text,
        "is_error": is_error,
    }
    if pages:
        message["attachments"] = [
            {
                "resource_id": f"page-{index}",
                "safe_name": f"page-{index}.png",
                "media_type": "image/png",
                "content_digest": "a" * 64,
                "size_bytes": 10,
                "data_url": page,
            }
            for index, page in enumerate(pages)
        ]
    return message


async def _invoke(provider: CompletionProvider, entrypoint: str) -> str:
    messages: list[dict[str, Any]] = [_QUESTION]
    if entrypoint == "complete":
        return await provider.complete(messages, _MODEL)
    if entrypoint == "complete_tool_turn":
        return (await provider.complete_tool_turn(messages, _MODEL, tools=[_VIEW])).text
    if entrypoint == "complete_tool_turn_streaming":
        emitted: list[str] = []

        async def emit(text: str) -> None:
            emitted.append(text)

        turn = await provider.complete_tool_turn_streaming(
            messages, _MODEL, tools=[_VIEW], emit_text=emit
        )
        assert "".join(emitted) == turn.text
        return turn.text
    stream = getattr(provider, entrypoint)(messages, _MODEL)
    return "".join([token async for token in stream])


async def _stream_turn(model: ToolModel) -> tuple[AssistantTurn, list[str]]:
    emitted: list[str] = []

    async def emit(text: str) -> None:
        emitted.append(text)

    turn = await model.stream_turn(messages=[_QUESTION], tools=[_VIEW], emit_text=emit)
    return turn, emitted


async def test_a_completion_is_one_stateless_interaction_with_its_thinking_controls() -> None:
    gemini = _Gemini(_interaction(_THOUGHT, _text("Revenue by quarter.")))
    model = _completion_model(gemini, reasoning="high")
    try:
        result = await model(
            [{"role": "system", "content": "Answer briefly."}, _QUESTION], max_tokens=256
        )
    finally:
        await model.aclose()

    assert result.stop_reason == "stop"
    assert result == "Revenue by quarter."
    assert model._provider.last_reasoning == "Page 2 holds the table."
    (request,) = gemini.requests
    assert str(request.url) == _URL
    assert request.headers["x-goog-api-key"] == "test-key"
    assert gemini.bodies == [
        {
            "model": _MODEL,
            "input": [_ASKED],
            "system_instruction": "Answer briefly.",
            "generation_config": {
                "max_output_tokens": 256,
                "thinking_level": "high",
                "thinking_summaries": "auto",
            },
            "store": False,
        }
    ]


@pytest.mark.parametrize(
    ("requested", "sent"), [("low", "low"), ("medium", "medium"), ("max", "high")]
)
async def test_a_reasoning_level_is_sent_as_the_catalogued_thinking_level(
    requested: str, sent: str
) -> None:
    gemini = _Gemini(_interaction(_text("ok")))
    model = _completion_model(gemini, reasoning=requested)
    try:
        await model([_QUESTION])
    finally:
        await model.aclose()

    assert gemini.bodies[0]["generation_config"] == {
        "thinking_level": sent,
        "thinking_summaries": "auto",
    }


async def test_reasoning_off_fails_before_gemini_is_asked() -> None:
    gemini = _Gemini()
    model = _completion_model(gemini, reasoning="off")
    try:
        with pytest.raises(ReasoningConfigurationError, match="cannot be honored"):
            await model([_QUESTION])
    finally:
        await model.aclose()

    assert gemini.requests == []


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
async def test_every_entrypoint_sends_full_context_with_store_false(entrypoint: str) -> None:
    streamed = "stream" in entrypoint
    gemini = _Gemini(_streamed_text("o", "k") if streamed else _interaction(_text("ok")))
    provider = _provider(gemini)
    try:
        assert await _invoke(provider, entrypoint) == "ok"
    finally:
        await provider.aclose()

    (body,) = gemini.bodies
    assert body["store"] is False
    assert body["input"] == [_ASKED]
    assert body.get("stream", False) is streamed
    for remote_state in ("previous_interaction_id", "background", "webhook_config", "agent"):
        assert remote_state not in body


async def test_streamed_text_arrives_as_it_comes_and_the_terminal_usage_is_kept() -> None:
    gemini = _Gemini(_streamed_text("Revenue ", "by ", "quarter."))
    model = _completion_model(gemini)
    usage: dict[str, Any] = {}
    try:
        stream = await model([_QUESTION], stream=True, usage_holder=usage)
        tokens = [token async for token in stream]
    finally:
        await model.aclose()

    assert tokens == ["Revenue ", "by ", "quarter."]
    assert usage == {"usage_details": _COUNTERS}
    assert gemini.bodies == [{"model": _MODEL, "input": [_ASKED], "store": False, "stream": True}]
    assert gemini.requests[0].headers["accept"] == "text/event-stream"


class _Answer(BaseModel):
    answer: str
    page: int


async def test_structured_output_asks_for_json_text_against_the_strict_schema() -> None:
    output = StructuredOutput(name="answer", schema=_Answer)
    gemini = _Gemini(_interaction(_text('{"answer":"revenue","page":2}')))
    model = _completion_model(gemini)
    try:
        raw = await model([_QUESTION], structured_output=output)
    finally:
        await model.aclose()

    assert output.parse(raw) == _Answer(answer="revenue", page=2)
    (body,) = gemini.bodies
    assert body["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": output.json_schema_response_format()["json_schema"]["schema"],
    }
    assert "response_mime_type" not in body


async def test_the_json_object_opt_out_asks_for_json_text_without_a_schema() -> None:
    gemini = _Gemini(_interaction(_text('{"answer":"revenue","page":2}')))
    model = _completion_model(gemini, structured_output="json_object")
    try:
        await model([_QUESTION], structured_output=StructuredOutput(name="answer", schema=_Answer))
    finally:
        await model.aclose()

    assert gemini.bodies[0]["response_format"] == {"type": "text", "mime_type": "application/json"}


async def test_a_tool_turn_round_trips_through_a_session_with_its_thought_in_place() -> None:
    first = _interaction(
        _THOUGHT, _text("Let me look."), _call("call-1", 2), status="requires_action"
    )
    gemini = _Gemini(first, _interaction(_text("Page 2 shows revenue by quarter.")))
    model = _tool_model(gemini)
    try:
        turn = await model(messages=[_QUESTION], tools=[_VIEW], tool_choice="required")
        replayed = _stored(turn)
        answer = await model(
            messages=[_QUESTION, replayed, _result("call-1", "Page 2 rendered.", _PAGE_URL)],
            tools=[_VIEW],
        )
    finally:
        await model.aclose()

    assert turn.text == "Let me look."
    assert turn.reasoning == "Page 2 holds the table."
    assert turn.tool_calls == (ToolCall(id="call-1", name="view", arguments={"page": 2}),)
    assert turn.stop_reason == "tool_use"
    assert turn.usage_details == _COUNTERS
    assert replayed["provider_state"]["payload"] == {"interaction_steps": first["steps"]}
    assert answer.text == "Page 2 shows revenue by quarter."
    asked, answered = gemini.bodies
    assert asked["tools"] == [
        {
            "type": "function",
            "name": "view",
            "description": "View one page.",
            "parameters": _VIEW.parameters,
        }
    ]
    assert asked["generation_config"] == {"tool_choice": "any"}
    # The thought goes back exactly as Gemini sent it, before the text and call it made.
    assert answered["input"] == [
        _ASKED,
        _THOUGHT,
        _text("Let me look."),
        _call("call-1", 2),
        _function_result(
            "call-1",
            [
                {"type": "text", "text": "Page 2 rendered."},
                {"type": "image", "mime_type": "image/png", "data": _PAGE},
            ],
        ),
    ]


async def test_parallel_calls_are_answered_in_order_after_the_thought_that_made_them() -> None:
    first = _interaction(_THOUGHT, _call("call-1", 1), _call("call-2", 2), status="requires_action")
    gemini = _Gemini(first, _interaction(_text("Pages 1 and 2 agree.")))
    model = _tool_model(gemini)
    try:
        turn = await model(messages=[_QUESTION], tools=[_VIEW])
        await model(
            messages=[
                _QUESTION,
                _stored(turn),
                _result("call-1", "Page 1."),
                _result("call-2", "Page 2 failed.", is_error=True),
            ],
            tools=[_VIEW],
        )
    finally:
        await model.aclose()

    assert turn.tool_calls == (
        ToolCall(id="call-1", name="view", arguments={"page": 1}),
        ToolCall(id="call-2", name="view", arguments={"page": 2}),
    )
    assert gemini.bodies[1]["input"] == [
        _ASKED,
        _THOUGHT,
        _call("call-1", 1),
        _call("call-2", 2),
        _function_result("call-1", "Page 1."),
        _function_result("call-2", "Page 2 failed.", is_error=True),
    ]


@pytest.mark.parametrize(
    "source",
    [
        ModelInvocationFingerprint("gemini", "gemini-3.8-pro", None, "interactions"),
        # What the generateContent wire recorded for this very model.
        ModelInvocationFingerprint("gemini", _MODEL, None, "chat_completion"),
    ],
    ids=["another-model", "another-api-family"],
)
async def test_another_invocations_state_is_stripped_to_the_canonical_turn(
    source: ModelInvocationFingerprint,
) -> None:
    turn = bind_provider_replay(
        AssistantTurn(
            text="Let me look.",
            tool_calls=(ToolCall(id="call-1", name="view", arguments={"page": 2}),),
            stop_reason="tool_use",
            provider_state={
                "interaction_steps": [_THOUGHT, _text("Let me look."), _call("call-1", 2)]
            },
        ),
        source,
    )
    gemini = _Gemini(_interaction(_text("Done.")))
    model = _tool_model(gemini)
    try:
        await model(
            messages=[_QUESTION, _stored(turn), _result("call-1", "Page 2 rendered.")],
            tools=[_VIEW],
        )
    finally:
        await model.aclose()

    (body,) = gemini.bodies
    assert body["input"] == [
        _ASKED,
        _text("Let me look."),
        _call("call-1", 2),
        _function_result("call-1", "Page 2 rendered."),
    ]
    assert _SIGNATURE not in json.dumps(body)


async def test_a_same_model_replay_that_contradicts_its_canonical_turn_fails() -> None:
    gemini = _Gemini(_interaction(_THOUGHT, _call("call-1", 2), status="requires_action"))
    model = _tool_model(gemini)
    try:
        turn = await model(messages=[_QUESTION], tools=[_VIEW])
        edited = {**_stored(turn), "content": "Words Gemini never wrote."}
        with pytest.raises(InteractionRequestError, match="does not match"):
            await model(messages=[_QUESTION, edited, _result("call-1", "Page 2.")], tools=[_VIEW])
    finally:
        await model.aclose()

    assert len(gemini.requests) == 1


def _streamed_tool_turn(*terminal: dict[str, Any]) -> httpx2.Response:
    """A thought, a sentence and a call, streamed as Google streams them."""
    return _sse(
        _CREATED,
        *_thought_events(0),
        _start(1, {"content": [{"text": "Let me ", "type": "text"}], "type": "model_output"}),
        _delta(1, type="text", text="look."),
        _stop(1),
        _start(2, {"arguments": {}, "id": "call-1", "name": "view", "type": "function_call"}),
        _delta(2, type="arguments_delta", arguments='{"pa'),
        _delta(2, type="arguments_delta", arguments='ge": 2}'),
        {**_stop(2), "usage": _USAGE},
        *terminal,
    )


@pytest.mark.parametrize(
    "terminal",
    [
        [_completed("requires_action")],
        # No completed event: the terminal status update ends the turn just the same.
        [
            {
                "event_type": "interaction.status_update",
                "interaction_id": "i-1",
                "status": "requires_action",
            }
        ],
    ],
    ids=["completed", "status-update"],
)
async def test_a_streamed_tool_turn_is_assembled_from_its_deltas(
    terminal: list[dict[str, Any]],
) -> None:
    gemini = _Gemini(_streamed_tool_turn(*terminal), _interaction(_text("Page 2 shows revenue.")))
    model = _tool_model(gemini)
    try:
        turn, emitted = await _stream_turn(model)
        await model(
            messages=[_QUESTION, _stored(turn), _result("call-1", "Page 2 rendered.")],
            tools=[_VIEW],
        )
    finally:
        await model.aclose()

    assert emitted == ["Let me ", "look."]
    assert turn.text == "Let me look."
    assert turn.reasoning == "Page 2 holds the table."
    assert turn.tool_calls == (ToolCall(id="call-1", name="view", arguments={"page": 2}),)
    assert turn.stop_reason == "tool_use"
    assert turn.usage_details == _COUNTERS
    assert gemini.bodies[0]["stream"] is True
    # The signature that arrived after the empty one goes back with the whole summary.
    assert gemini.bodies[1]["input"][1:4] == [_THOUGHT, _text("Let me look."), _call("call-1", 2)]


async def test_a_streamed_turn_keeps_steps_a_completion_does_carry() -> None:
    final = [_THOUGHT, _text("Revenue.")]
    completed = _completed()
    completed["interaction"]["steps"] = final
    gemini = _Gemini(
        _sse(
            _start(0, {"content": [{"text": "Revenue.", "type": "text"}], "type": "model_output"}),
            completed,
        )
    )
    model = _tool_model(gemini)
    try:
        turn, emitted = await _stream_turn(model)
    finally:
        await model.aclose()

    assert emitted == ["Revenue."]
    assert (turn.text, turn.stop_reason) == ("Revenue.", "stop")
    assert turn.provider_state is not None
    assert turn.provider_state["payload"] == {"interaction_steps": final}


async def test_a_questions_attachments_become_its_own_named_images() -> None:
    attachment = {
        "resource_id": "att-1",
        "safe_name": "chart.png",
        "media_type": "image/png",
        "content_digest": "a" * 64,
        "size_bytes": 10,
    }
    question = {
        **_QUESTION,
        # The second was never hydrated for this Run, so it has no pixels to send.
        "attachments": [{**attachment, "data_url": _PAGE_URL}, {**attachment, "safe_name": "x"}],
    }
    gemini = _Gemini(_interaction(_text("A chart.")))
    model = _completion_model(gemini)
    try:
        await model([question])
    finally:
        await model.aclose()

    assert gemini.bodies[0]["input"] == [
        {
            "type": "user_input",
            "content": [
                {"type": "text", "text": "What does page 2 show?"},
                {"type": "text", "text": "[chart.png]"},
                {"type": "image", "mime_type": "image/png", "data": _PAGE},
            ],
        }
    ]


async def test_usage_counts_thinking_as_output_and_keeps_cache_hits() -> None:
    usage = {**_USAGE, "input_tokens_by_modality": [{"modality": "text", "tokens": 62}]}
    gemini = _Gemini(_interaction(_THOUGHT, _text("ok"), usage=usage))
    model = _tool_model(gemini)
    try:
        turn = await model(messages=[_QUESTION], tools=[])
    finally:
        await model.aclose()

    # Output is 171 visible plus 297 thought tokens, as Google bills it and as every other
    # provider reports reasoning; input plus output is the total again.
    assert turn.usage_details == _COUNTERS
    assert provider_input_tokens(turn.usage_details) == 62
    assert provider_cache_hit_tokens(turn.usage_details) == 40
    assert _langfuse_usage_details(_COUNTERS) == {
        "input": 62,
        "output": 468,
        "total": 530,
        "input_cached_tokens": 40,
    }


def test_gemini_always_uses_the_interactions_api_family() -> None:
    settings = _settings()
    fingerprint = model_invocation_fingerprint(settings)

    assert settings.api_family == "interactions"
    assert _settings(api_family="interactions").api_family == "interactions"
    assert fingerprint.api_family == "interactions"
    assert ModelInvocationFingerprint.from_json(fingerprint.as_json()) == fingerprint
    with pytest.raises(ValidationError, match="gemini always uses the interactions API family"):
        _settings(api_family="chat_completion")
    with pytest.raises(ValidationError, match="response API family requires the openai provider"):
        _settings(api_family="response")
    with pytest.raises(ValidationError, match="interactions API family requires the gemini"):
        ModelSettings(model="gpt-x", api_family="interactions")
    with pytest.raises(ValueError, match="interactions API family requires the gemini"):
        get_provider("openai", api_family="interactions")


def test_a_configured_temperature_is_refused_for_gemini() -> None:
    refused = "temperature is not supported for gemini"
    with pytest.raises(ValidationError, match=refused):
        _settings(temperature=0.2)
    with pytest.raises(ValidationError, match=refused):
        RerankSettings(provider="gemini", model=_MODEL, temperature=0.0)
    with pytest.raises(ValidationError, match=refused):
        ModelRoleSettings.model_validate(
            {"default": {"provider": "gemini", "model": _MODEL, "temperature": 1.0}}
        )
    # A default on another provider is its own endpoint: it inherits neither the shipped
    # OpenRouter URL nor that endpoint's sampling.
    default = ModelRoleSettings.model_validate(
        {"default": {"provider": "gemini", "model": _MODEL}}
    ).default
    assert (default.base_url, default.temperature, default.api_family) == (
        None,
        None,
        "interactions",
    )


async def test_internal_sampling_defaults_never_reach_gemini() -> None:
    # The image probe asks every provider for temperature 0, and a chat reranker scores
    # at 0 wherever the wire samples.
    gemini = _Gemini(_interaction(_text("ok")))
    provider = _provider(gemini)
    try:
        outcome = await probe_image_capability(provider, model=_MODEL)
    finally:
        await provider.aclose()

    assert outcome.status == "supported"
    assert "temperature" not in json.dumps(gemini.bodies[0])
    reranker = RerankSettings(provider="gemini", model=_MODEL, api_key="test-key")
    assert reranker.scoring_model(_settings()).temperature is None


async def test_only_safety_settings_and_service_tier_pass_through() -> None:
    safety = [{"type": "dangerous_content", "threshold": "block_only_high"}]
    gemini = _Gemini(_interaction(_text("ok")))
    model = _completion_model(
        gemini, model_kwargs={"safety_settings": safety, "service_tier": "flex"}
    )
    try:
        await model([_QUESTION])
    finally:
        await model.aclose()

    (body,) = gemini.bodies
    assert (body["safety_settings"], body["service_tier"]) == (safety, "flex")

    refused = _Gemini()
    model = _completion_model(refused, model_kwargs={"cached_content": "cachedContents/abc"})
    try:
        with pytest.raises(InteractionRequestError, match="got cached_content"):
            await model([_QUESTION])
    finally:
        await model.aclose()
    assert refused.requests == []


async def test_a_turn_cut_off_by_its_token_cap_keeps_its_text_and_reports_its_call() -> None:
    cut = _interaction(
        _THOUGHT,
        _text("The table shows"),
        {"type": "function_call", "id": "call-1", "name": "view", "arguments": {}},
        status="incomplete",
    )
    model = _tool_model(_Gemini(cut))
    try:
        turn = await model(messages=[_QUESTION], tools=[_VIEW])
    finally:
        await model.aclose()

    assert turn == AssistantTurn(
        text="The table shows",
        reasoning="Page 2 holds the table.",
        tool_calls=(ToolCall(id="call-1", name="view", arguments={}),),
        stop_reason="length",
        usage_details=_COUNTERS,
    )


async def test_a_stream_cut_off_mid_call_keeps_its_text_and_reports_the_call() -> None:
    gemini = _Gemini(
        _sse(
            _start(
                0, {"content": [{"text": "Let me look.", "type": "text"}], "type": "model_output"}
            ),
            _stop(0),
            _start(1, {"arguments": {}, "id": "call-1", "name": "view", "type": "function_call"}),
            _delta(1, type="arguments_delta", arguments='{"pa'),
            _completed("incomplete"),
        )
    )
    model = _tool_model(gemini)
    try:
        turn, emitted = await _stream_turn(model)
    finally:
        await model.aclose()

    assert emitted == ["Let me look."]
    assert (turn.text, turn.stop_reason) == ("Let me look.", "length")
    assert [(call.id, call.arguments) for call in turn.tool_calls] == [("call-1", {})]
    assert turn.provider_state is None


@pytest.mark.parametrize(
    ("status", "steps", "code", "failure", "transient"),
    [
        ("failed", [], "resource_exhausted", "ended 'failed' (resource_exhausted)", True),
        ("failed", [], "invalid_argument", "ended 'failed' (invalid_argument)", False),
        ("cancelled", [], None, "ended 'cancelled'", False),
        ("requires_action", [_text("I will look.")], None, "requires action but made no", False),
        ("completed", [_THOUGHT], None, "completed without text or calls", False),
    ],
)
async def test_an_interaction_without_usable_output_is_a_provider_failure(
    status: str, steps: list[dict[str, Any]], code: str | None, failure: str, transient: bool
) -> None:
    errors = [{"code": code, "message": "Quota for user 42 exceeded."}] if code else []
    gemini = _Gemini(_interaction(*steps, status=status, errors=errors))
    model = _tool_model(gemini)
    try:
        with pytest.raises(InteractionStatusError, match=re.escape(failure)) as raised:
            await model(messages=[_QUESTION], tools=[_VIEW])
    finally:
        await model.aclose()

    assert "user 42" not in str(raised.value)
    # An overload reported in-band is the outage an HTTP 503 is: retried, then deferred.
    expected = "providers" if transient else None
    assert classify_transient_dependency(raised.value, component_hint="providers") == expected


@pytest.mark.parametrize(
    ("code", "transient"),
    [
        ("unavailable", True),
        ("DEADLINE_EXCEEDED", True),
        # The API documents a code as a URI naming the error type.
        ("https://errors.example/gemini#resource-exhausted", True),
        ("invalid_argument", False),
    ],
)
async def test_a_stream_error_event_fails_the_turn_retryably_when_its_code_is_an_outage(
    code: str, transient: bool
) -> None:
    gemini = _Gemini(
        _sse(
            _start(0, {"content": [{"text": "Let me", "type": "text"}], "type": "model_output"}),
            {"event_type": "error", "error": {"code": code, "message": "Backend failure."}},
        )
    )
    model = _tool_model(gemini)
    emitted: list[str] = []

    async def emit(text: str) -> None:
        emitted.append(text)

    try:
        with pytest.raises(
            InteractionStatusError, match=re.escape(f"stream failed ({code})")
        ) as raised:
            await model.stream_turn(messages=[_QUESTION], tools=[_VIEW], emit_text=emit)
    finally:
        await model.aclose()

    assert emitted == ["Let me"]
    expected = "providers" if transient else None
    assert classify_transient_dependency(raised.value, component_hint="providers") == expected


async def test_a_stream_that_ends_without_a_terminal_status_was_cut_short() -> None:
    gemini = _Gemini(
        _sse(
            _CREATED,
            _start(0, {"content": [{"text": "Le", "type": "text"}], "type": "model_output"}),
            {
                "event_type": "interaction.status_update",
                "interaction_id": "i-1",
                "status": "in_progress",
            },
        )
    )
    provider = _provider(gemini)
    try:
        with pytest.raises(InteractionStatusError, match="ended before completing") as raised:
            [token async for token in provider.stream([_QUESTION], _MODEL)]
    finally:
        await provider.aclose()

    assert classify_transient_dependency(raised.value, component_hint="providers") == "providers"


_OVERLOADED = {"error": {"code": "unavailable", "message": "The model is overloaded."}}
# A frontend error in google.rpc form: its integer code does not fit the SDK's error
# schema, which must not turn an outage into a request fault.
_OVERLOADED_RPC = {
    "error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}
}


@pytest.mark.usefixtures("no_retry_pauses")
@pytest.mark.parametrize(
    ("max_retries", "requests"),
    # google-genai raises an attempts of 0 to 1 before its Interactions client reads it.
    [(0, 2), (1, 2), (3, 4)],
)
@pytest.mark.parametrize("body", [_OVERLOADED, _OVERLOADED_RPC], ids=["interactions", "rpc"])
async def test_an_overloaded_model_is_retried_then_deferred_as_an_outage(
    body: dict[str, Any], max_retries: int, requests: int
) -> None:
    gemini = _Gemini(*(httpx2.Response(503, json=body) for _ in range(requests)))
    model = _tool_model(gemini, max_retries=max_retries)
    try:
        with pytest.raises(Exception) as raised:  # noqa: B017 - the SDK's status error
            await model(messages=[_QUESTION], tools=[_VIEW])
    finally:
        await model.aclose()

    assert len(gemini.requests) == requests
    assert getattr(raised.value, "status_code", None) == 503
    assert classify_transient_dependency(raised.value, component_hint="providers") == "providers"


@pytest.mark.parametrize(
    ("message", "overflow", "reasoning"),
    [
        (
            "The input token count (1048577) exceeds the maximum number of tokens allowed "
            "(1048576).",
            True,
            False,
        ),
        ("Invalid value for generation_config.thinking_level: 'minimal'.", False, True),
    ],
)
async def test_a_rejected_request_is_named_and_never_retried(
    message: str, overflow: bool, reasoning: bool
) -> None:
    body = {"error": {"code": "invalid_argument", "message": message}}
    gemini = _Gemini(httpx2.Response(400, json=body))
    model = _tool_model(gemini, max_retries=2)
    try:
        with pytest.raises(Exception) as raised:  # noqa: B017 - the SDK's status error
            await model(messages=[_QUESTION], tools=[_VIEW])
    finally:
        await model.aclose()

    assert len(gemini.requests) == 1
    assert getattr(raised.value, "status_code", None) == 400
    assert is_provider_context_overflow(raised.value) is overflow
    assert is_provider_reasoning_rejection(raised.value) is reasoning
    assert classify_transient_dependency(raised.value, component_hint="providers") is None


@pytest.mark.usefixtures("no_retry_pauses")
async def test_the_endpoint_and_timeout_reach_the_wire() -> None:
    gemini = _Gemini(httpx2.Response(503, json=_OVERLOADED), _interaction(_text("ok")))
    provider = _provider(gemini, base_url="https://gateway.example/gemini", timeout=12.5)
    try:
        assert await provider.complete([_QUESTION], _MODEL) == "ok"
    finally:
        await provider.aclose()

    assert [str(request.url) for request in gemini.requests] == [
        "https://gateway.example/gemini/v1beta/interactions"
    ] * 2
    assert gemini.requests[0].extensions["timeout"]["read"] == 12.5


async def _close_after_request(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    await reader.read(65_536)
    writer.close()


async def _transport_failure(base_url: str) -> BaseException:
    provider = provider_for(_settings(base_url=base_url))
    try:
        await provider.complete([_QUESTION], _MODEL)
    except Exception as exc:  # noqa: BLE001 - the failure is the subject
        return exc
    finally:
        await provider.aclose()
    raise AssertionError("the request did not fail")


@pytest.mark.usefixtures("no_retry_pauses")
async def test_a_dropped_connection_is_a_provider_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    bypass_proxies(monkeypatch)
    async with loopback_server(reset_on_accept) as port:
        failure = await _transport_failure(f"https://127.0.0.1:{port}")

    assert classify_transient_dependency(failure, component_hint="providers") == "providers"


@pytest.mark.usefixtures("no_retry_pauses")
async def test_an_untrusted_endpoint_is_misconfiguration_not_an_outage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bypass_proxies(monkeypatch)
    tls = loopback_certificate(tmp_path).server_context()
    async with loopback_server(_close_after_request, tls=tls) as port:
        failure = await _transport_failure(f"https://127.0.0.1:{port}")

    assert classify_transient_dependency(failure, component_hint="providers") is None
