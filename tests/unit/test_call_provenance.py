# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What a step records of the call that made it: the allowlist, the served model and the timings
(ADR 0041), checked against what a scripted provider was sent and, for the adapters, against what
a vendor's SDK reads off the wire.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from typing import Any

import pytest

from dlightrag.engine.agent.session.provenance import call_payload, prompt_digests
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.contracts import ApiFamily
from dlightrag.engine.ai.messages import AssistantTurn, ToolDefinition
from dlightrag.engine.ai.provenance import options_digest, served_model_name
from dlightrag.engine.ai.providers import get_provider
from dlightrag.engine.ai.reasoning import best_effort_reasoning_profile
from dlightrag.engine.ai.settings import ModelSettings
from tests.support.model_provider import (
    ProviderRequest,
    RecordingProvider,
    canonical,
    completion_model_over,
    expected_call,
    sha256,
    tool_model_over,
    untimed,
)
from tests.unit import test_provider_attachment_contract as wire

_TOOL = ToolDefinition(name="lookup", description="Look it up.", parameters={"type": "object"})
_GATEWAY = "https://gateway.example/v1"
_PROFILE = ModelProfile(
    context_window_tokens=100_000,
    max_output_tokens=16_000,
    reasoning=best_effort_reasoning_profile("openai"),
)


def _answer(_request: ProviderRequest) -> AssistantTurn:
    return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")


def test_the_options_digest_holds_no_credential_and_moves_with_what_shapes_the_model() -> None:
    shaped = {"top_p": 0.9, "reasoning_effort": "high"}
    one = {
        **shaped,
        "api_key": "sk-live-ONE",
        "extra_headers": {"Authorization": "Bearer ONE", "X-Custom": "ONE"},
        "base_url": "https://one.example/v1?signature=ONE",
    }
    two = {**one, "api_key": "sk-live-TWO", "extra_headers": {"X-Other": "TWO"}, "base_url": "x"}

    # A request container is dropped whole, `X-Custom` included; a key is masked and never hashed.
    assert options_digest(one) == options_digest(two)
    assert options_digest(one) == sha256(canonical({**shaped, "api_key": "[redacted]"}))
    assert options_digest(one) != options_digest({**one, "top_p": 0.5})


async def test_a_tool_turn_states_what_its_provider_was_sent_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = RecordingProvider(reply=_answer, served_model="query-model-2026-10")
    settings = ModelSettings(
        provider="openai",
        model="query-model",
        api_key="sk-live-KEY",
        base_url=f"{_GATEWAY}?signature=SIGNED",
        temperature=0.4,
        reasoning="high",
        model_kwargs={"top_p": 0.9, "extra_headers": {"X-Custom": "SECRET"}},
    )
    model = tool_model_over(monkeypatch, provider, settings)
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]

    turn = await model(
        messages=messages,
        tools=[_TOOL],
        tool_choice="required",
        max_tokens=321,
        model_profile=_PROFILE,
    )
    (sent,) = provider.requests
    prompt = prompt_digests(messages, [asdict(_TOOL)])
    stated = call_payload(turn.call, role="query", prompt=prompt, plan_digest="p")

    assert (sent.temperature, sent.max_tokens, sent.tool_choice) == (0.4, 321, "required")
    assert untimed(stated) == expected_call(
        sent,
        role="query",
        endpoint=_GATEWAY,
        plan_digest="p",
        reasoning={"requested": "high", "effective": "high"},
        served_model="query-model-2026-10",
        options={"top_p": 0.9, "reasoning_effort": "high"},
    )
    for secret in ("sk-live-KEY", "SIGNED", "SECRET", "gateway.example"):
        assert secret not in canonical(stated)


@pytest.mark.parametrize(
    ("stated", "kept"),
    [("gpt-5.4", "gpt-5.4"), (" padded ", "padded"), (None, None), (7, None), ("", None)]
    + [("x" * 201, None), ("two\nlines", None)],
)
def test_a_served_model_is_kept_only_when_the_server_states_a_plain_name(
    stated: object, kept: str | None
) -> None:
    assert served_model_name(stated) == kept


_WIRES = [
    pytest.param(
        "openai", "chat_completion", "openai", wire._MODELS["openai"], id="chat-completions"
    ),
    pytest.param("openai", "response", "openai_response", "gpt-5.4", id="responses"),
    pytest.param(
        "anthropic", "chat_completion", "anthropic", wire._MODELS["anthropic"], id="anthropic"
    ),
]


@pytest.mark.parametrize(("name", "family", "kind", "served"), _WIRES)
async def test_an_adapter_hands_up_the_model_its_server_stated(
    name: str, family: ApiFamily, kind: str, served: str
) -> None:
    provider = get_provider(name, api_key="test-key", api_family=family, max_retries=0)
    wire.bind_mock_http(provider, wire._HttpCapture(kind).handler)
    messages, holder = [{"role": "user", "content": "look"}], {}

    async def emit_text(_text: str) -> None:
        return None

    try:
        turn = await provider.complete_tool_turn(messages, "asked-for", tools=[wire._VIEW_TOOL])
        streamed = await provider.complete_tool_turn_streaming(
            messages, "asked-for", tools=[wire._VIEW_TOOL], emit_text=emit_text
        )
        _ = [t async for t in provider.stream(messages, "asked-for", usage_holder=holder)]
    finally:
        await provider.aclose()

    assert turn.served_model == streamed.served_model == holder["served_model"] == served
    # The name is kept apart from the counters, which stay counters.
    assert all(isinstance(value, int) for value in holder["usage_details"].values())


@pytest.mark.parametrize("way", ["tool-turn", "summary", "answer"])
async def test_a_call_states_the_wait_for_a_slot_apart_from_how_long_its_request_took(
    monkeypatch: pytest.MonkeyPatch, way: str
) -> None:
    """Two requests under one slot: the second waits, and the wait is not time its request took."""
    provider = RecordingProvider(reply=_answer, delay=0.05)
    settings = ModelSettings(provider="openai", model="m")
    tools = tool_model_over(monkeypatch, provider, settings, max_concurrency=1)
    answers = completion_model_over(monkeypatch, provider, settings)
    messages = [{"role": "user", "content": "q"}]

    async def ask() -> tuple[Any, float]:
        began, holder = time.monotonic(), {}
        if way == "tool-turn":
            call = (await tools(messages=messages, tools=[_TOOL])).call
        elif way == "summary":
            _ = [t async for t in tools.stream_text(messages=messages, call_holder=holder)]
            call = holder["call"]
        else:
            stream = await answers(messages=messages, stream=True, call_holder=holder)
            _ = [t async for t in stream]
            call = holder["call"]
        return call, (time.monotonic() - began) * 1000

    first, second = await asyncio.gather(ask(), ask())

    for call, spent in (first, second):
        assert call.duration_ms >= 45
        assert call.queued_ms + call.duration_ms <= spent + 1
    waited, unimpeded = sorted((first[0], second[0]), key=lambda c: c.queued_ms, reverse=True)
    assert waited.queued_ms >= 45 > unimpeded.queued_ms
