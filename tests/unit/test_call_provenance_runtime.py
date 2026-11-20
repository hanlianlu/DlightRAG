# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A Research Run records, in each step it writes, the call that made it (ADR 0043).

Real ``ToolModel`` wrappers sit over a scripted provider, under the real orchestrator and
``AgentSessionRuntime``; what the provider was sent is compared with what the Session stored.
The Session here is the in-memory repository, so the entries are read exactly as the Runtime
committed them.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dlightrag.engine.agent.environment.toolchain import (
    SearchToolchain,
)
from dlightrag.engine.agent.session.entries import AssistantMessageEntry, CompactionEntry
from dlightrag.engine.agent.session.ids import LaneId, SessionId
from dlightrag.engine.agent.session.operation import OperationCompleted
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.runtime import AgentSessionRuntime
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall
from dlightrag.engine.ai.reasoning import ReasoningLevels, ReasoningProfile
from dlightrag.engine.ai.settings import ModelSettings
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.ai.tool_model import ToolModel
from dlightrag.engine.answer.orchestration.orchestrator import AnswerOrchestrator
from dlightrag.engine.answer.research.runtime import FetchedResourceBuffer, ResearchRuntimeEffects
from dlightrag.engine.answer.tools.memory import MemoryHost
from dlightrag.engine.dependencies import ProviderUnavailableError
from dlightrag.engine.runtime.settlements import EffectHostUpdate
from tests.support.in_memory_session_repository import MemoryAgentSessionRepository
from tests.support.model_provider import (
    ProviderRequest,
    RecordingProvider,
    canonical,
    expected_call,
    tool_model_over,
    untimed,
)
from tests.unit.test_research_runtime_migration import _one_fact, _Session

#: A verified reasoning profile: `high` is the strongest level it offers, and `off` is one it has.
_PROFILE = ModelProfile(
    context_window_tokens=1_000_000,
    max_output_tokens=64_000,
    reasoning=ReasoningProfile(
        format="openai",
        levels=ReasoningLevels(
            off="none",
            minimal=None,
            low="low",
            medium="medium",
            high="high",
            xhigh=None,
            max=None,
        ),
    ),
)
_GATEWAY = "https://gateway.example/v1"
_SEARCH = ToolCall("search-1", "search_knowledge_base", {"query": "one fact"})
_QUESTION = "What was the figure?"


def _settings(**overrides: Any) -> ModelSettings:
    values: dict[str, Any] = {
        "provider": "openai",
        "model": "query-model",
        "api_key": "test-key",
        "base_url": _GATEWAY,
        "temperature": 0.2,
        "reasoning": "max",
        "model_kwargs": {"top_p": 0.9},
    }
    return ModelSettings(**{**values, **overrides})


def _search_then_answer(request: ProviderRequest) -> AssistantTurn:
    """Search on the first turn of a Run, and answer once the search has come back."""
    if any(message["role"] == "tool" for message in request.messages):
        return AssistantTurn(
            text="grounded answer",
            tool_calls=(),
            stop_reason="stop",
            usage_details={"input_tokens": 5, "output_tokens": 2},
        )
    return AssistantTurn(
        text="",
        tool_calls=(_SEARCH,),
        stop_reason="tool_use",
        usage_details={"input_tokens": 3, "output_tokens": 1},
    )


@dataclass
class Driven:
    """One Run driven to the end: the plan it ran under and the Session it wrote."""

    plan: AgentRunPlan
    session_id: SessionId
    repository: MemoryAgentSessionRepository[EffectHostUpdate]

    async def entries(self) -> list[Any]:
        return list((await self.repository.load(self.session_id)).entries)

    async def assistant_entries(self) -> list[AssistantMessageEntry]:
        return [entry for entry in await self.entries() if isinstance(entry, AssistantMessageEntry)]


def _research(
    model: ToolModel, *, memory: bool = False, question: str = _QUESTION
) -> tuple[AnswerOrchestrator, Any, AgentRunPlan]:
    orchestrator = AnswerOrchestrator(
        synthesizer=cast(Any, SimpleNamespace()),
        retrieve_knowledge_base=_one_fact,
        model_func=model,
        stream_model_func=model.stream_text,
        model_profile=_PROFILE,
        telemetry=NOOP_TELEMETRY,
        resolved_mode="research",
        search_toolchain=SearchToolchain(),
        memory_host=MemoryHost() if memory else None,
    )
    prepared = orchestrator.prepare_run(question)
    plan = AgentRunPlan.from_tools(
        prepared.tools, model_role="query", context_policy_revision=CONTEXT_POLICY_REVISION
    )
    return orchestrator, prepared, plan


def _runtime(
    repository: MemoryAgentSessionRepository[EffectHostUpdate],
    orchestrator: AnswerOrchestrator,
    prepared: Any,
    plan: AgentRunPlan,
    session_id: SessionId,
) -> AgentSessionRuntime[EffectHostUpdate]:
    return AgentSessionRuntime(
        repository=repository,
        effects=ResearchRuntimeEffects(
            telemetry=NOOP_TELEMETRY,
            orchestrator=orchestrator,
            prepared=prepared,
            session=_Session(),  # type: ignore[arg-type]
            session_id=session_id,
            fetched_buffer=FetchedResourceBuffer(),
            persist_child_intent=None,
        ),
        tools=prepared.tools,
        fencing_epoch=1,
        holder="run",
        provider_attempt_limit=plan.provider_attempt_limit,
    )


async def _drive(
    monkeypatch: pytest.MonkeyPatch,
    provider: RecordingProvider,
    *,
    settings: ModelSettings | None = None,
    model: ToolModel | None = None,
    memory: bool = False,
    question: str = _QUESTION,
) -> Driven:
    """Drive one Run to its end, on ``model`` when the test shares one between Runs."""
    model = model or tool_model_over(monkeypatch, provider, settings or _settings())
    orchestrator, prepared, plan = _research(model, memory=memory, question=question)
    repository = MemoryAgentSessionRepository[EffectHostUpdate]()
    session_id = SessionId.new()
    runtime = _runtime(repository, orchestrator, prepared, plan, session_id)
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="answer-run",
        content=question,
        plan=plan,
    )
    final = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
    assert isinstance(final.state, OperationCompleted)
    return Driven(plan, session_id, repository)


def _expected(
    request: ProviderRequest, *, plan: AgentRunPlan, reasoning: dict[str, str] | None
) -> dict[str, Any]:
    return expected_call(
        request, role="query", endpoint=_GATEWAY, plan_digest=plan.digest, reasoning=reasoning
    )


async def test_every_research_turn_states_the_call_its_provider_was_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each request takes 50 ms of the provider's, so each step must say it took that long.
    provider = RecordingProvider(reply=_search_then_answer, delay=0.05)
    settings = _settings()

    run = await _drive(monkeypatch, provider, settings=settings)
    turns = await run.assistant_entries()

    first, second = provider.requests
    assert [turn.content for turn in turns] == ["", "grounded answer"]
    assert len(provider.requests) == len(turns)
    # `max` was asked for and this model's strongest level is `high`: both are stated.
    reasoning = {"requested": "max", "effective": "high"}
    for turn, request in zip(turns, provider.requests, strict=True):
        # `untimed` also holds each timing to be a whole, non-negative number of milliseconds.
        assert untimed(turn.call) == _expected(request, plan=run.plan, reasoning=reasoning)
        assert turn.call is not None
        assert turn.call["duration_ms"] >= 45
        assert turn.call["queued_ms"] >= 0
    # The second request is the first plus the tool exchange, so its digests moved with it.
    assert first.digests()["system"] == second.digests()["system"]
    assert first.digests()["tools"] == second.digests()["tools"]
    assert first.digests()["request"] != second.digests()["request"]
    # The second request is built from the first turn's entry, which holds a call; the model is
    # never shown how an earlier turn was made.
    shown = canonical(second.messages)
    assert all("call" not in message for message in second.messages)
    for name in ("served_model", "options_digest", "endpoint_fingerprint", "plan_digest"):
        assert name not in shown
    assert first.digests()["request"] not in shown
    # What the provider received is what the call names: the model, sampling and options.
    assert (first.model, first.temperature, first.tool_choice) == ("query-model", 0.2, "auto")
    assert first.model_kwargs == {"top_p": 0.9, "reasoning_effort": "high"}


async def test_a_step_records_no_usage_cost_or_token_count_beside_its_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = RecordingProvider(reply=_search_then_answer)

    run = await _drive(monkeypatch, provider)
    turns = await run.assistant_entries()

    # The counters are exactly the ones the provider stated, and the call holds none of them.
    assert [turn.usage for turn in turns] == [
        {"input_tokens": 3, "output_tokens": 1},
        {"input_tokens": 5, "output_tokens": 2},
    ]
    for turn in turns:
        assert turn.cost is None
        flat = canonical(turn.call)
        assert "input_tokens" not in flat and "output_tokens" not in flat and "usage" not in flat


@pytest.mark.parametrize("memory", [False, True], ids=["same-flags-and-tools", "memory-on"])
async def test_a_step_states_the_system_and_tools_its_flags_and_tools_made(
    monkeypatch: pytest.MonkeyPatch, memory: bool
) -> None:
    """Two Runs asked the same question: they state one request unless memory moves its tools."""
    first = await _drive(monkeypatch, RecordingProvider(reply=_search_then_answer))
    second = await _drive(monkeypatch, RecordingProvider(reply=_search_then_answer), memory=memory)

    [one, *_] = await first.assistant_entries()
    [two, *_] = await second.assistant_entries()

    assert one.call is not None and two.call is not None
    same = not memory
    assert (one.call["prompt"]["system"] == two.call["prompt"]["system"]) is same
    assert (one.call["prompt"]["tools"] == two.call["prompt"]["tools"]) is same
    assert (one.call["plan_digest"] == two.call["plan_digest"]) is same
    if same:
        # Different Sessions asked the same question, so the whole request matches too.
        assert one.call["prompt"]["request"] == two.call["prompt"]["request"]


async def test_a_temperature_changed_after_acceptance_is_what_a_reclaimed_turn_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first worker records turn one at 0.2 and loses its provider; another resumes at 0.9."""

    def first_worker(request: ProviderRequest) -> AssistantTurn:
        if any(message["role"] == "tool" for message in request.messages):
            raise ConnectionError("provider unreachable")
        return _search_then_answer(request)

    repository = MemoryAgentSessionRepository[EffectHostUpdate]()
    before = _settings(temperature=0.2)
    model = tool_model_over(monkeypatch, RecordingProvider(reply=first_worker), before)
    orchestrator, prepared, plan = _research(model)
    session_id = SessionId.new()
    runtime = _runtime(repository, orchestrator, prepared, plan, session_id)
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="answer-run",
        content=_QUESTION,
        plan=plan,
    )
    with pytest.raises(ProviderUnavailableError):
        await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)

    # Configuration moved on while the Run waited: the same model at another temperature.
    after = _settings(temperature=0.9)
    resumed_provider = RecordingProvider(reply=_search_then_answer)
    resumed = tool_model_over(monkeypatch, resumed_provider, after)
    orchestrator, prepared, _ = _research(resumed)
    runtime = _runtime(repository, orchestrator, prepared, plan, session_id)
    final = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
    assert isinstance(final.state, OperationCompleted)

    [early, late] = await Driven(plan, session_id, repository).assistant_entries()
    assert early.call is not None and late.call is not None
    assert early.call["params"]["temperature"] == 0.2
    assert late.call["params"]["temperature"] == 0.9
    assert late.call["params"]["temperature"] == resumed_provider.requests[-1].temperature
    # Only the temperature moved; the Run is the same Run, under the same Plan.
    assert early.call["plan_digest"] == late.call["plan_digest"] == plan.digest
    assert early.call["model"] == late.call["model"] | {"served_model": "served-model-1"}


async def test_a_compaction_states_the_summarizer_call_that_wrote_its_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request the provider refuses as too long is summarized, then asked for again."""
    overflowed = {"done": False}

    def reply(request: ProviderRequest) -> AssistantTurn:
        has_tool_result = any(message["role"] == "tool" for message in request.messages)
        if has_tool_result and not overflowed["done"]:
            overflowed["done"] = True
            raise RuntimeError("prompt is too long: 300000 tokens > 200000 maximum")
        return _search_then_answer(request)

    provider = RecordingProvider(
        reply=reply,
        text="## Goal\nAnswer the figure.\n\n## Progress\nSearched once.\n\n## Next Steps\nAnswer.",
    )
    settings = _settings()

    # Large enough that covering it shrinks the request, as a compaction must.
    run = await _drive(
        monkeypatch, provider, settings=settings, question="What changed? " + "x" * 40_000
    )
    compactions = [entry for entry in await run.entries() if isinstance(entry, CompactionEntry)]
    turns = await run.assistant_entries()

    [compaction] = compactions
    [summarizer] = [request for request in provider.requests if not request.tools]
    assert compaction.call is not None
    # The summarizer ran on the cheapest level this model has (`off`), with an output cap of
    # its own and the options of that one attempt, and the digests are those of the exact
    # request the provider was sent.
    assert summarizer.model_kwargs == {"reasoning_effort": "none"}
    assert summarizer.max_tokens is not None
    assert untimed(compaction.call) == _expected(
        summarizer, plan=run.plan, reasoning={"requested": "off", "effective": "off"}
    )
    # The turns around it state their own calls, and the refused request left none.
    assert all(turn.call is not None for turn in turns)
    assert len(turns) == 2 and len(provider.requests) == 4
