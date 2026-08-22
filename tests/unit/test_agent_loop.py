# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""AgentLoop stop contract: silence, cancel, provider error, all-terminate."""

import pytest

from dlightrag.agent.loop import AgentLoop, LoopCancelled, LoopOutcome, LoopProviderError
from dlightrag.agent.tools.contracts import (
    ExecutedTurn,
    ToolExecution,
    ToolObservation,
    ToolResult,
)
from dlightrag.ai.messages import AssistantTurn, ToolCall


def _turn(*names: str, terminate: bool = False) -> ExecutedTurn:
    calls = tuple(ToolCall(id=f"c{i}", name=name, arguments={}) for i, name in enumerate(names))
    results = tuple(
        ToolExecution(
            call=call,
            result=ToolResult(content="ok", terminate=terminate),
            observation=ToolObservation(
                tool=call.name,
                call_id=call.id,
                outcome="ok",
                duration_ms=1.0,
                cached=False,
                is_error=False,
                content_chars=2,
            ),
            is_error=False,
        )
        for call in calls
    )
    return ExecutedTurn(
        assistant=AssistantTurn(text="x", tool_calls=calls, stop_reason="stop"),
        results=results,
        messages=[],
    )


class _ScriptedHost:
    def __init__(self, turns: list[ExecutedTurn], *, cancel_on: int | None = None) -> None:
        self.turns = list(turns)
        self.calls = 0
        self.cancel_on = cancel_on

    async def check_cancelled(self) -> None:
        if self.cancel_on is not None and self.calls >= self.cancel_on:
            raise LoopCancelled

    async def run_turn(self) -> ExecutedTurn:
        if not self.turns:
            raise AssertionError("host ran out of scripted turns")
        self.calls += 1
        return self.turns.pop(0)


class _ErrorHost:
    async def check_cancelled(self) -> None:
        return None

    async def run_turn(self) -> ExecutedTurn:
        raise LoopProviderError("upstream")


@pytest.mark.asyncio
async def test_silence_stops_without_another_turn() -> None:
    silent = _turn()
    host = _ScriptedHost([silent, _turn("search")])
    outcome = await AgentLoop().run(host)
    assert outcome == LoopOutcome(reason="model_stop", last_turn=silent)
    assert host.calls == 1


@pytest.mark.asyncio
async def test_cancel_at_boundary_stops() -> None:
    host = _ScriptedHost([_turn("search")], cancel_on=0)
    outcome = await AgentLoop().run(host)
    assert outcome.reason == "cancelled"
    assert outcome.last_turn is None


@pytest.mark.asyncio
async def test_all_terminate_stops_after_the_batch() -> None:
    batch = _turn("done", terminate=True)
    host = _ScriptedHost([batch, _turn("search")])
    outcome = await AgentLoop().run(host)
    assert outcome.reason == "all_terminate"
    assert host.calls == 1


@pytest.mark.asyncio
async def test_provider_error_ends_the_attempt() -> None:
    outcome = await AgentLoop().run(_ErrorHost())
    assert outcome.reason == "provider_error"
    assert outcome.last_turn is None


def test_loop_module_imports_no_product() -> None:
    import dlightrag.agent.loop as loop

    forbidden = (
        "dlightrag.answer",
        "dlightrag.rag",
        "dlightrag.adapters",
        "dlightrag.services",
        "asyncpg",
        "fastapi",
    )
    source = loop.__file__
    assert source is not None
    text = open(source, encoding="utf-8").read()
    assert "dlightrag.agent" in text
    assert all(name not in text for name in forbidden)
