# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A Research Run records, in each step it writes, the call that made it (ADR 0041).

Real ``ToolModel`` wrappers sit over a scripted provider, under the real orchestrator and
``AgentSessionRuntime``; what the provider was sent is compared with what the Session stored.
"""

from __future__ import annotations

import pytest

from dlightrag.engine.agent.session.entries import AssistantMessageEntry, CompactionEntry
from dlightrag.engine.agent.session.ids import SessionId
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall
from dlightrag.engine.ai.settings import ModelSettings
from dlightrag.engine.runtime.settlements import EffectHostUpdate
from tests.in_memory_session_repository import MemoryAgentSessionRepository
from tests.support.model_provider import (
    ProviderRequest,
    RecordingProvider,
    canonical,
    expected_call,
    tool_model_over,
    untimed,
)
from tests.unit.test_research_runtime_migration import _drive_research_run

_GATEWAY = "https://gateway.example/v1"
_SETTINGS = ModelSettings(
    provider="openai",
    model="query-model",
    api_key="sk-live-KEY",
    base_url=f"{_GATEWAY}?signature=SIGNED",
    temperature=0.2,
    model_kwargs={"top_p": 0.9, "extra_headers": {"X-Custom": "SECRET"}},
)
_SEARCH = ToolCall("search-1", "search_knowledge_base", {"query": "one fact"})
_SUMMARY = "## Goal\nAnswer.\n\n## Progress\nSearched once.\n\n## Next Steps\nAnswer."


async def test_each_research_step_states_the_call_its_provider_was_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Run searches, is refused as too long, is summarized and asks again: its two turns and its
    compaction each state the request that made them, and the refused one is not among them."""
    refused: list[bool] = []

    def reply(request: ProviderRequest) -> AssistantTurn:
        done = any(message["role"] == "tool" for message in request.messages)
        calls, usage = ((), 5) if done else ((_SEARCH,), 3)
        stop = "stop" if done else "tool_use"
        return AssistantTurn(
            "done" if done else "", calls, stop, usage_details={"input_tokens": usage}
        )

    def refuse_the_first_request_after_the_search(request: ProviderRequest) -> Exception | None:
        if any(message["role"] == "tool" for message in request.messages) and not refused:
            refused.append(True)
            return RuntimeError("prompt is too long: 300000 tokens > 200000 maximum")
        return None

    provider = RecordingProvider(
        reply=reply, fail=refuse_the_first_request_after_the_search, text=_SUMMARY
    )
    model = tool_model_over(monkeypatch, provider, _SETTINGS)
    repository = MemoryAgentSessionRepository[EffectHostUpdate]()
    session_id = SessionId.new()
    await _drive_research_run(
        repository,
        session_id,
        "What changed? " + "x" * 40_000,
        model=model,
        stream_model=model.stream_text,
    )
    entries = list((await repository.load(session_id)).entries)

    turns = [entry for entry in entries if isinstance(entry, AssistantMessageEntry)]
    [compaction] = [entry for entry in entries if isinstance(entry, CompactionEntry)]
    asked = [request for request in provider.requests if request.tools]
    [summarizer] = [request for request in provider.requests if not request.tools]
    assert len(turns) == 2 and len(asked) == 3
    assert turns[0].call is not None
    plan_digest = turns[0].call["plan_digest"]
    assert len(plan_digest) == 64
    for step, request in ((turns[0], asked[0]), (turns[1], asked[2]), (compaction, summarizer)):
        assert untimed(step.call) == expected_call(
            request,
            role="query",
            endpoint=_GATEWAY,
            plan_digest=plan_digest,
            reasoning=None,
            options=None if step is compaction else {"top_p": 0.9},
        )
    # The Run's counters are the provider's alone: the call holds none, and no secret is stored.
    assert [turn.usage for turn in turns] == [{"input_tokens": 3}, {"input_tokens": 5}]
    assert "input_tokens" not in canonical(turns[0].call)
    written = canonical([entry.canonical_payload() for entry in entries])
    for secret in ("sk-live-KEY", "SIGNED", "SECRET", "gateway.example"):
        assert secret not in written
