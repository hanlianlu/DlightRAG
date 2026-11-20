# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What a step records of the call that made it: the allowlist, the digests, the timings and the
shape (ADR 0043).

The wrappers are driven over a scripted provider that records what it is sent, and every claim a
step makes is checked against that record: the model, the sampling and output limits, the
reasoning levels and the digests of the exact request. The digests are recomputed here from
``hashlib`` and ``json``, so these tests pin the definition and not the code that implements it.
The timings are checked against a provider that takes a known time over each request and a
scheduler that holds a second request until the first one's response has ended.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel

from dlightrag.engine.agent.session.entries import (
    AssistantMessageEntry,
    CompactionEntry,
    decode_entry_payload,
)
from dlightrag.engine.agent.session.ids import EntryId, OperationId, ProjectionId, SessionId
from dlightrag.engine.agent.session.provenance import (
    call_payload,
    prompt_digests,
    request_prompt_digests,
)
from dlightrag.engine.agent.session.registers import RequestSnapshot
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.completion import CompletionModel
from dlightrag.engine.ai.fingerprints import model_invocation_fingerprint
from dlightrag.engine.ai.messages import AssistantTurn, ToolDefinition
from dlightrag.engine.ai.provenance import ModelCall, options_digest, served_model_name
from dlightrag.engine.ai.reasoning import (
    ReasoningLevels,
    ReasoningProfile,
    ResolvedReasoning,
    best_effort_reasoning_profile,
)
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import ModelSettings
from dlightrag.engine.ai.structured import StructuredOutput
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from tests.support.model_provider import (
    ProviderRequest,
    RecordingProvider,
    canonical,
    sha256,
    tool_model_over,
)

_TOOL = ToolDefinition(
    name="lookup",
    description="Look one thing up.",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)
_HIGH_AT_MOST = ModelProfile(
    context_window_tokens=100_000,
    max_output_tokens=16_000,
    reasoning=ReasoningProfile(
        format="openai",
        levels=ReasoningLevels(
            off=None,
            minimal=None,
            low="low",
            medium=None,
            high="high",
            xhigh=None,
            max=None,
        ),
    ),
)


def _answer(request: ProviderRequest) -> AssistantTurn:
    del request
    return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")


# ── the options digest ───────────────────────────────────────────────────────────────────


def test_the_options_digest_does_not_depend_on_any_credential() -> None:
    options = {
        "top_p": 0.9,
        "extra_headers": {"Authorization": "Bearer one", "X-Custom": "one"},
        "api_key": "one",
        "metadata": {"user_token": "one"},
    }
    rotated = {
        "top_p": 0.9,
        "extra_headers": {"Authorization": "Bearer two", "X-Custom": "two"},
        "api_key": "two",
        "metadata": {"user_token": "two"},
    }

    assert options_digest(options) == options_digest(rotated)


def test_a_header_container_is_dropped_for_what_it_is_not_for_the_names_inside_it() -> None:
    """``X-Custom`` looks like no secret, so only its container can keep it out."""
    assert options_digest({"top_p": 0.9, "extra_headers": {"X-Custom": "one"}}) == options_digest(
        {"top_p": 0.9}
    )
    nested = {"extra_body": {"default_headers": {"X-Custom": "one"}, "seed": 1}}
    assert options_digest(nested) == options_digest({"extra_body": {"seed": 1}})
    for container in ("headers", "auth", "cookies", "credentials", "extra_query", "base_url"):
        assert options_digest({container: {"x": "one"}}) == options_digest({}), container


def test_options_that_shape_the_model_move_the_digest_and_their_order_does_not() -> None:
    assert options_digest({"top_p": 0.9, "seed": 1}) == options_digest({"seed": 1, "top_p": 0.9})
    assert options_digest({"top_p": 0.9}) != options_digest({"top_p": 0.8})
    assert options_digest({"top_p": 0.9}) != options_digest({})
    # A count under a token name is an option and not a credential, so it stays in.
    assert options_digest({"limits": {"max_tokens": 5}}) != options_digest(
        {"limits": {"max_tokens": 6}}
    )


def test_the_options_digest_is_sha256_over_canonical_options() -> None:
    options = {"top_p": 0.9, "provider": {"order": ["a", "b"]}, "usage": {"include": True}}

    assert options_digest(options) == sha256(canonical(options))


def test_the_options_digest_never_refuses_an_option_it_cannot_describe() -> None:
    odd = {
        1: {"a": {2}},
        "when": datetime(2026, 10, 10, tzinfo=UTC),
        "raw": b"x",
        ("t",): float("nan"),
    }

    assert len(options_digest(odd)) == 64


# ── what a server may say about the model that served it ─────────────────────────────────


@pytest.mark.parametrize(
    ("stated", "kept"),
    [
        pytest.param("deepseek-v4-flash-2026-10-01", "deepseek-v4-flash-2026-10-01", id="dated"),
        pytest.param("google/gemini-4-hydrogen:free", "google/gemini-4-hydrogen:free", id="routed"),
        pytest.param("  padded  ", "padded", id="padded"),
        pytest.param("", None, id="empty"),
        pytest.param("   ", None, id="blank"),
        pytest.param("x" * 201, None, id="too-long"),
        pytest.param("bad\x00name", None, id="nul"),
        pytest.param("bad\nname", None, id="newline"),
        pytest.param(None, None, id="absent"),
        pytest.param(7, None, id="number"),
        pytest.param(["model"], None, id="list"),
    ],
)
def test_a_served_model_is_kept_only_when_the_server_states_a_plain_name(
    stated: object, kept: str | None
) -> None:
    assert served_model_name(stated) == kept


# ── what a wrapper states of one call ────────────────────────────────────────────────────


def test_a_model_call_states_the_allowlist_and_nothing_a_reader_could_trace_to_a_secret() -> None:
    settings = ModelSettings(
        provider="openai",
        model="query-model",
        api_key="sk-SECRET-KEY",
        base_url="https://gateway.example/v1?signature=SECRET-QUERY",
    )
    fingerprint = model_invocation_fingerprint(settings)
    resolved = ResolvedReasoning(
        requested="max", effective="high", profile=best_effort_reasoning_profile("openai")
    )

    call = ModelCall.of(
        fingerprint,
        served_model="query-model-2026-10",
        temperature=0.5,
        max_tokens=100,
        tool_choice="auto",
        response_format={"type": "json_schema", "json_schema": {"name": "mode", "schema": {}}},
        reasoning=resolved,
        model_kwargs={"top_p": 0.9, "extra_headers": {"Authorization": "Bearer SECRET-HEADER"}},
        queued_seconds=0.0126,
        duration_seconds=1.5004,
    )
    stated = call.as_json()

    assert set(stated) == {"model", "params", "queued_ms", "duration_ms"}
    # Whole milliseconds, rounded, as plain integers.
    assert (stated["queued_ms"], stated["duration_ms"]) == (13, 1500)
    assert stated["model"] == {
        "provider": "openai",
        "model": "query-model",
        "endpoint_fingerprint": fingerprint.endpoint_fingerprint,
        "api_family": "chat_completion",
        "served_model": "query-model-2026-10",
    }
    assert stated["params"] == {
        "temperature": 0.5,
        "max_tokens": 100,
        "tool_choice": "auto",
        # Only the type of the format: the schema is not part of what a step records.
        "response_format": "json_schema",
        "reasoning": {"requested": "max", "effective": "high"},
        "options_digest": sha256(canonical({"top_p": 0.9})),
    }
    written = json.dumps(stated)
    for trace in ("SECRET", "gateway.example", "signature", "sk-"):
        assert trace not in written
    assert len(str(fingerprint.endpoint_fingerprint)) == 64


@pytest.mark.parametrize(
    ("seconds", "milliseconds"),
    [
        pytest.param(0.0, 0, id="did-not-wait"),
        pytest.param(0.0004, 0, id="under-half-a-millisecond"),
        pytest.param(0.0006, 1, id="over-half-a-millisecond"),
        pytest.param(61.2504, 61250, id="a-minute"),
        pytest.param(-0.2, 0, id="never-negative"),
    ],
)
def test_the_timings_are_whole_non_negative_milliseconds(seconds: float, milliseconds: int) -> None:
    call = ModelCall.of(
        model_invocation_fingerprint(ModelSettings(provider="openai", model="m")),
        served_model=None,
        temperature=None,
        max_tokens=None,
        tool_choice=None,
        response_format=None,
        reasoning=None,
        model_kwargs={},
        queued_seconds=seconds,
        duration_seconds=seconds,
    )
    stated = call.as_json()

    for name in ("queued_ms", "duration_ms"):
        assert type(stated[name]) is int
        assert stated[name] == milliseconds


# ── the prompt digests ───────────────────────────────────────────────────────────────────


def test_the_prompt_digests_follow_their_definition() -> None:
    messages = [
        {"role": "system", "content": "You are brief."},
        {"role": "user", "content": "héllo ✓"},
    ]
    tools = [{"name": "lookup", "description": "d", "parameters": {"type": "object"}}]

    digests = prompt_digests(messages, tools)

    assert digests.system == sha256("You are brief.")
    assert digests.tools == sha256(canonical(tools))
    assert digests.request == sha256(canonical({"messages": messages, "tools": tools}))
    assert prompt_digests(messages).tools == sha256("[]")


def test_a_request_snapshot_and_the_lists_it_was_made_from_have_the_same_digests() -> None:
    messages = [
        {"role": "system", "content": "You are brief."},
        {"role": "user", "content": [{"type": "text", "text": "a"}]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function"}]},
    ]
    tools = [{"name": "lookup", "description": "d", "parameters": {"type": "object"}}]
    snapshot = RequestSnapshot.from_values(
        operation_id=OperationId.new(),
        turn_number=1,
        plan_digest="a" * 64,
        model_role="query",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=10,
    )

    assert request_prompt_digests(snapshot) == prompt_digests(messages, tools)


def test_the_system_digest_covers_the_system_text_alone() -> None:
    base = [{"role": "system", "content": "S"}, {"role": "user", "content": "a"}]
    other = [{"role": "system", "content": "S"}, {"role": "user", "content": "b"}]
    blocks = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "S"}, {"type": "text", "text": "T"}],
        },
        {"role": "system", "content": "U"},
    ]

    assert prompt_digests(base).system == prompt_digests(other).system
    assert prompt_digests(base).request != prompt_digests(other).request
    assert prompt_digests(base).tools == prompt_digests(other).tools
    assert prompt_digests(blocks).system == sha256("S\nT\n\nU")
    assert (
        prompt_digests([{"role": "system", "content": "S2"}]).system != prompt_digests(base).system
    )


def test_a_call_payload_adds_what_only_the_committer_knows_and_no_token_count() -> None:
    settings = ModelSettings(provider="openai", model="m", temperature=0.2)
    call = ModelCall.of(
        model_invocation_fingerprint(settings),
        served_model=None,
        temperature=0.2,
        max_tokens=None,
        tool_choice=None,
        response_format=None,
        reasoning=None,
        model_kwargs={},
        queued_seconds=0.0,
        duration_seconds=0.25,
    )
    prompt = prompt_digests([{"role": "system", "content": "S"}])

    research = call_payload(call, role="query", prompt=prompt, plan_digest="b" * 64)
    fast = call_payload(call, role="query", prompt=prompt)

    assert research is not None and fast is not None
    assert research["model"]["role"] == "query"
    assert research["prompt"] == prompt.as_json()
    # The wrapper's timings travel with its half, whoever commits the step.
    assert (research["queued_ms"], research["duration_ms"]) == (0, 250)
    assert (fast["queued_ms"], fast["duration_ms"]) == (0, 250)
    assert research["plan_digest"] == "b" * 64 and "plan_digest" not in fast
    # The shape is closed: what a step states is these keys, and a code version is not among them.
    shape = {"model", "params", "queued_ms", "duration_ms", "prompt"}
    assert set(fast) == shape and set(research) == shape | {"plan_digest"}
    assert call_payload(None, role="query", prompt=prompt) is None

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return {str(key) for key in value} | {k for item in value.values() for k in keys(item)}
        return set()

    counted = {key for key in keys(research) if any(w in key for w in ("token", "usage", "cost"))}
    assert counted == {"max_tokens"}


# ── the entries that carry it ────────────────────────────────────────────────────────────


def _decode(entry_type: str, payload: dict[str, Any]) -> Any:
    return decode_entry_payload(
        entry_type=entry_type,
        entry_id=EntryId.new(),
        session_id=SessionId.new(),
        sequence=1,
        timestamp=datetime.now(UTC),
        payload=payload,
    )


def test_a_call_survives_the_trip_through_a_stored_payload() -> None:
    call = {"model": {"role": "query"}, "params": {"temperature": 0.5}, "duration_ms": 12}
    assistant = AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=SessionId.new(),
        timestamp=datetime.now(UTC),
        content="x",
        stop_reason="stop",
        call=call,
    )
    compaction = CompactionEntry(
        entry_id=EntryId.new(),
        session_id=SessionId.new(),
        timestamp=datetime.now(UTC),
        projection_id=ProjectionId.new(),
        summary="s",
        covered_through_sequence=0,
        first_retained_sequence=1,
        call=call,
    )

    for entry in (assistant, compaction):
        stored = json.loads(json.dumps(entry.canonical_payload()))
        assert stored["call"] == call
        assert _decode(entry.entry_type, stored).call == call


def test_an_entry_without_a_call_stores_and_decodes_as_it_always_did() -> None:
    old_assistant = {"content": "x", "reasoning": "", "stop_reason": "stop", "tool_calls": []}
    old_compaction = {
        "projection_id": ProjectionId.new().value,
        "summary": "s",
        "covered_through_sequence": 0,
        "first_retained_sequence": 1,
        "covered_through_entry_id": None,
        "first_retained_entry_id": None,
        "source_digest": "",
    }

    for entry_type, payload in (
        ("assistant_message", old_assistant),
        ("compaction", old_compaction),
    ):
        entry = _decode(entry_type, payload)
        assert entry.call is None
        # Nothing is written for an absent call, so an entry that has none is byte for byte
        # the entry that was written before the field existed.
        assert entry.canonical_payload() == payload


# ── the wrappers state the attempt that produced the result ──────────────────────────────

_PLAIN_USAGE = {"usage_details": {"prompt_tokens": 7, "completion_tokens": 3}}
_SUMMARY_PROFILE = ModelProfile(
    context_window_tokens=100_000,
    max_output_tokens=16_000,
    reasoning=best_effort_reasoning_profile("openai"),
)
#: Agentic options first, ordinary ones when the agentic attempt says nothing.
_AGENTIC = ModelSettings(
    provider="openai",
    model="query-model",
    reasoning="low",
    agentic_reasoning="high",
    model_kwargs={"top_p": 0.5},
    agentic_model_kwargs={"top_p": 0.9},
)
_COMPAT = ModelSettings(provider="openai", model="compat", base_url="https://openrouter.ai/api/v1")


class _Mode(BaseModel):
    mode: str


_MODE = StructuredOutput(name="mode", schema=_Mode)
Delay = float | Callable[[ProviderRequest], float]
Fallen = tuple[RecordingProvider, str, dict[str, Any]]


def _completion_model(
    monkeypatch: pytest.MonkeyPatch,
    provider: RecordingProvider,
    settings: ModelSettings,
    *,
    slots: int = 1,
) -> CompletionModel:
    """A real ``CompletionModel`` for ``settings`` over the scripted provider."""
    monkeypatch.setattr(
        "dlightrag.engine.ai.providers.get_provider", lambda *_args, **_kwargs: provider
    )
    return CompletionModel(
        settings, scheduler=ModelScheduler(max_concurrency=slots), telemetry=NOOP_TELEMETRY
    )


async def _streamed(model: CompletionModel, messages: list[dict[str, Any]], **options: Any) -> str:
    """The text a stream of ``model`` answers, read to its end."""
    stream = await model(messages=messages, stream=True, **options)
    return "".join([piece async for piece in stream])


def _refuse_the_schema(request: ProviderRequest) -> Exception | None:
    if (request.response_format or {}).get("type") == "json_schema":
        return RuntimeError("response_format json_schema is not supported by this model")
    return None


def _agentic_says_nothing(request: ProviderRequest) -> str:
    return " " if request.model_kwargs["top_p"] == 0.9 else "## Goal\nsummary"


async def _the_schema_is_refused(
    monkeypatch: pytest.MonkeyPatch, delay: Delay = 0.0, usage: dict[str, Any] | None = None
) -> Fallen:
    """A structured stream whose schema the model refuses, so it is asked again as a json object."""
    provider = RecordingProvider(text="{}", fail=_refuse_the_schema, delay=delay)
    model = _completion_model(monkeypatch, provider, _COMPAT)
    holder: dict[str, Any] = {}
    text = await _streamed(
        model,
        [{"role": "system", "content": "Pick a mode."}],
        structured_output=_MODE,
        usage_holder=usage,
        call_holder=holder,
    )
    return provider, text, holder


async def _the_agentic_attempt_says_nothing(
    monkeypatch: pytest.MonkeyPatch, delay: Delay = 0.0
) -> Fallen:
    """A compaction whose agentic attempt writes nothing, so the ordinary one writes the text."""
    provider = RecordingProvider(text=_agentic_says_nothing, delay=delay)
    model = tool_model_over(monkeypatch, provider, _AGENTIC)
    holder: dict[str, Any] = {}
    text = "".join(
        [
            piece
            async for piece in model.stream_text(
                messages=[{"role": "user", "content": "summarize"}],
                model_profile=_SUMMARY_PROFILE,
                call_holder=holder,
            )
        ]
    )
    return provider, text, holder


async def test_a_tool_turn_states_what_its_provider_was_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = RecordingProvider(reply=_answer, served_model="query-model-2026-10")
    settings = ModelSettings(
        provider="openai",
        model="query-model",
        api_key="k",
        base_url="https://gateway.example/v1",
        temperature=0.4,
        reasoning="max",
        model_kwargs={"top_p": 0.9},
    )
    model = tool_model_over(monkeypatch, provider, settings)

    turn = await model(
        messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "q"}],
        tools=[_TOOL],
        tool_choice="required",
        max_tokens=321,
        model_profile=_HIGH_AT_MOST,
    )
    sent = provider.requests[0]

    # The level the model was asked for is `max`; this endpoint tops out at `high`.
    assert sent.model_kwargs == {"top_p": 0.9, "reasoning_effort": "high"}
    assert turn.served_model == "query-model-2026-10"
    assert turn.call is not None
    # What the call states, apart from how long it took, which the timing tests below pin.
    assert replace(turn.call, queued_ms=0, duration_ms=0) == ModelCall(
        fingerprint=model_invocation_fingerprint(settings),
        served_model="query-model-2026-10",
        temperature=sent.temperature,
        max_tokens=sent.max_tokens,
        tool_choice=sent.tool_choice,
        response_format=None,
        reasoning=("max", "high"),
        options_digest=sha256(canonical(sent.model_kwargs)),
        queued_ms=0,
        duration_ms=0,
    )
    assert (sent.temperature, sent.max_tokens, sent.tool_choice) == (0.4, 321, "required")


async def test_a_compaction_that_fell_back_states_the_attempt_that_wrote_the_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, text, holder = await _the_agentic_attempt_says_nothing(monkeypatch)
    agentic, ordinary = provider.requests

    assert text.endswith("## Goal\nsummary")
    assert agentic.model_kwargs == {"top_p": 0.9, "reasoning_effort": "high"}
    assert ordinary.model_kwargs == {"top_p": 0.5, "reasoning_effort": "low"}
    call = holder["call"]
    assert call.reasoning == ("low", "low")
    assert call.options_digest == sha256(canonical(ordinary.model_kwargs))
    assert call.options_digest != sha256(canonical(agentic.model_kwargs))
    assert call.served_model == "served-model-1"


async def test_a_compaction_that_never_finished_states_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = RecordingProvider(fail=lambda _request: RuntimeError("rejected"))
    model = tool_model_over(monkeypatch, provider, ModelSettings(provider="openai", model="m"))
    holder: dict[str, Any] = {}

    with pytest.raises(RuntimeError, match="rejected"):
        _ = [
            piece
            async for piece in model.stream_text(
                messages=[{"role": "user", "content": "s"}], model_kwargs={}, call_holder=holder
            )
        ]

    assert holder == {}


async def test_a_stream_that_fell_back_to_a_json_object_states_the_format_it_was_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage: dict[str, Any] = {}

    provider, text, holder = await _the_schema_is_refused(monkeypatch, usage=usage)

    first, second = provider.requests
    assert text == "{}"
    assert first.response_format is not None and first.response_format["type"] == "json_schema"
    assert second.response_format == {"type": "json_object"}
    assert holder["call"].response_format == "json_object"
    # Usage keeps its counters alone: the model that served the request is not usage.
    assert usage == _PLAIN_USAGE
    assert holder["call"].served_model == "served-model-1"
    # The holders are the wrapper's own out-parameters, never provider options.
    assert [request.model_kwargs for request in provider.requests] == [{}, {}]


async def test_a_stream_that_needed_no_fallback_states_its_call_and_keeps_usage_to_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = RecordingProvider(text="a plain answer", served_model="served-plain")
    settings = ModelSettings(provider="openai", model="plain", temperature=0.3)
    model = _completion_model(monkeypatch, provider, settings)
    usage: dict[str, Any] = {}
    holder: dict[str, Any] = {}

    text = await _streamed(
        model,
        [{"role": "user", "content": "q"}],
        max_tokens=64,
        usage_holder=usage,
        call_holder=holder,
    )

    (sent,) = provider.requests
    call = holder["call"]
    assert text == "a plain answer"
    assert (sent.temperature, sent.max_tokens) == (0.3, 64)
    assert call.fingerprint == model_invocation_fingerprint(settings)
    assert (call.temperature, call.max_tokens) == (sent.temperature, sent.max_tokens)
    assert call.response_format is None
    assert call.options_digest == sha256(canonical(sent.model_kwargs))
    assert call.served_model == "served-plain"
    # What the server stated about the model is the call's, so the caller's usage keeps the
    # counters the provider stated and nothing else.
    assert usage == _PLAIN_USAGE


async def test_a_stream_that_failed_for_good_states_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = RecordingProvider(fail=lambda _request: RuntimeError("down"))
    model = _completion_model(monkeypatch, provider, ModelSettings(provider="openai", model="m"))
    holder: dict[str, Any] = {}

    with pytest.raises(RuntimeError, match="down"):
        await _streamed(model, [{"role": "user", "content": "q"}], call_holder=holder)

    assert holder == {}


# ── how long a call took, and how long it waited for a slot ──────────────────────────────

_WAYS = ["tool-turn", "compaction", "answer-stream"]
#: What each scripted request takes: long enough to read off a clock, short enough to be quick.
_REQUEST_SECONDS = 0.05


def _asker(
    monkeypatch: pytest.MonkeyPatch, provider: RecordingProvider, way: str, *, slots: int
) -> Callable[[], Awaitable[ModelCall]]:
    """One way a step is asked for, as a function that asks once and returns the call it states.

    The three ways are the three paths of a wrapper that state a call: an Agent turn, the stream
    of a compaction summary and the stream of a Fast answer. All run under a real scheduler of
    ``slots`` slots, over the scripted provider.
    """
    settings = ModelSettings(provider="openai", model="m")
    if way == "answer-stream":
        answers = _completion_model(monkeypatch, provider, settings, slots=slots)

        async def ask_for_an_answer() -> ModelCall:
            holder: dict[str, Any] = {}
            await _streamed(answers, [{"role": "user", "content": "q"}], call_holder=holder)
            return holder["call"]

        return ask_for_an_answer

    model = tool_model_over(monkeypatch, provider, settings, max_concurrency=slots)

    async def ask_for_a_turn() -> ModelCall:
        turn = await model(messages=[{"role": "user", "content": "q"}], tools=[_TOOL])
        assert turn.call is not None
        return turn.call

    async def ask_for_a_summary() -> ModelCall:
        holder: dict[str, Any] = {}
        _ = [
            piece
            async for piece in model.stream_text(
                messages=[{"role": "user", "content": "s"}], model_kwargs={}, call_holder=holder
            )
        ]
        return holder["call"]

    return ask_for_a_turn if way == "tool-turn" else ask_for_a_summary


async def _measured(ask: Callable[[], Awaitable[ModelCall]]) -> tuple[ModelCall, float]:
    """The call a request states, and the milliseconds the caller itself spent on it."""
    began = time.monotonic()
    call = await ask()
    return call, (time.monotonic() - began) * 1000


@pytest.mark.parametrize("way", _WAYS)
async def test_a_call_states_how_long_its_request_took_and_how_long_it_waited_for_a_slot(
    monkeypatch: pytest.MonkeyPatch, way: str
) -> None:
    provider = RecordingProvider(reply=_answer, delay=_REQUEST_SECONDS)
    ask = _asker(monkeypatch, provider, way, slots=1)

    (first, first_spent), (second, second_spent) = await asyncio.gather(
        _measured(ask), _measured(ask)
    )

    for call in (first, second):
        assert type(call.duration_ms) is int and type(call.queued_ms) is int
        # The request took what the provider took to answer it, and nothing waits a negative time.
        assert call.duration_ms >= 45
        assert call.queued_ms >= 0
    # One slot: the second request is held until the first one's response has ended.
    assert second.queued_ms >= 45
    assert first.queued_ms < second.queued_ms
    # The wait is not part of the request: each call's two intervals fit in the time its caller
    # spent, which they could not if the second request's duration counted its wait too.
    for call, spent in ((first, first_spent), (second, second_spent)):
        assert call.queued_ms + call.duration_ms <= spent + 1


def _slow_first_attempt() -> Callable[[ProviderRequest], float]:
    """The first request takes 200 ms of the provider's, every later one 50 ms."""
    attempts = itertools.count()
    return lambda _request: 0.2 if next(attempts) == 0 else _REQUEST_SECONDS


@pytest.mark.parametrize(
    ("fall_back", "attempt_of", "attempts"),
    [
        pytest.param(
            _the_schema_is_refused,
            lambda request: (request.response_format or {}).get("type"),
            ["json_schema", "json_object"],
            id="refused-schema",
        ),
        pytest.param(
            _the_agentic_attempt_says_nothing,
            lambda request: request.model_kwargs["top_p"],
            [0.9, 0.5],
            id="silent-agentic-attempt",
        ),
    ],
)
async def test_the_time_a_fallen_back_attempt_took_is_not_part_of_the_call(
    monkeypatch: pytest.MonkeyPatch,
    fall_back: Callable[[pytest.MonkeyPatch, Delay], Awaitable[Fallen]],
    attempt_of: Callable[[ProviderRequest], object],
    attempts: list[object],
) -> None:
    began = time.monotonic()
    provider, _text, holder = await fall_back(monkeypatch, _slow_first_attempt())
    spent = (time.monotonic() - began) * 1000

    assert [attempt_of(request) for request in provider.requests] == attempts
    call = holder["call"]
    # The call is the fallback's: its request took its own time, and the 200 ms the first attempt
    # took went neither into the duration nor into a wait.
    assert call.duration_ms >= 45
    assert spent - call.duration_ms >= 190
