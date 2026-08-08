# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for provider-neutral tool turn execution."""

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict

from dlightrag.core.agent.tool_loop import AgentTool, ToolResult, ToolTurnExecutor
from dlightrag.models.tool_turn import AssistantTurn, ToolCall


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str


class ScriptedModel:
    def __init__(self, turn: AssistantTurn) -> None:
        self.turn = turn
        self.calls: list[dict[str, Any]] = []

    async def complete_turn(self, **kwargs: Any) -> AssistantTurn:
        self.calls.append(kwargs)
        return self.turn


def _turn(*calls: ToolCall, stop_reason: str = "tool_use") -> AssistantTurn:
    return AssistantTurn(
        text="",
        tool_calls=tuple(calls),
        stop_reason=stop_reason,  # type: ignore[arg-type]
    )


async def test_valid_calls_execute_in_parallel_and_results_replay_in_source_order() -> None:
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release = asyncio.Event()

    async def execute(args: BaseModel) -> ToolResult:
        assert isinstance(args, SearchArgs)
        (first_started if args.query == "first" else second_started).set()
        await release.wait()
        return ToolResult(content=f"result:{args.query}")

    model = ScriptedModel(
        _turn(
            ToolCall(id="1", name="search", arguments={"query": "first"}),
            ToolCall(id="2", name="search", arguments={"query": "second"}),
        )
    )
    executor = ToolTurnExecutor(model.complete_turn)
    task = asyncio.create_task(
        executor.run_turn(
            [{"role": "user", "content": "q"}],
            [AgentTool("search", "Search.", SearchArgs, execute)],
        )
    )
    await asyncio.wait_for(first_started.wait(), timeout=1)
    await asyncio.wait_for(second_started.wait(), timeout=1)
    release.set()

    executed = await task

    assert [result.call.id for result in executed.results] == ["1", "2"]
    assert [message["content"] for message in executed.messages[-2:]] == [
        "result:first",
        "result:second",
    ]


async def test_invalid_arguments_are_returned_to_the_model_without_execution() -> None:
    calls = 0

    async def execute(_args: BaseModel) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult(content="unreachable")

    model = ScriptedModel(_turn(ToolCall(id="1", name="search", arguments={"unexpected": True})))
    executed = await ToolTurnExecutor(model.complete_turn).run_turn(
        [{"role": "user", "content": "q"}],
        [AgentTool("search", "Search.", SearchArgs, execute)],
    )

    assert calls == 0
    assert executed.results[0].is_error is True
    assert "query" in executed.results[0].result.content


async def test_malformed_provider_arguments_are_never_executed() -> None:
    calls = 0

    async def execute(_args: BaseModel) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult(content="unreachable")

    model = ScriptedModel(
        _turn(
            ToolCall(
                id="1",
                name="search",
                arguments={},
                argument_error="invalid JSON",
            )
        )
    )
    executed = await ToolTurnExecutor(model.complete_turn).run_turn(
        [{"role": "user", "content": "q"}],
        [AgentTool("search", "Search.", SearchArgs, execute)],
    )

    assert calls == 0
    assert executed.results[0].is_error is True
    assert "invalid JSON" in executed.results[0].result.content


async def test_length_stop_never_executes_possibly_truncated_calls() -> None:
    calls = 0

    async def execute(_args: BaseModel) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult(content="unreachable")

    model = ScriptedModel(
        _turn(
            ToolCall(id="1", name="search", arguments={"query": "q"}),
            stop_reason="length",
        )
    )
    executed = await ToolTurnExecutor(model.complete_turn).run_turn(
        [{"role": "user", "content": "q"}],
        [AgentTool("search", "Search.", SearchArgs, execute)],
    )

    assert calls == 0
    assert executed.results[0].is_error is True
    assert "token limit" in executed.results[0].result.content


async def test_unknown_tool_is_an_error_result_not_an_exception() -> None:
    model = ScriptedModel(_turn(ToolCall(id="1", name="invented", arguments={})))

    executed = await ToolTurnExecutor(model.complete_turn).run_turn(
        [{"role": "user", "content": "q"}],
        [],
    )

    assert executed.results[0].is_error is True
    assert "not available" in executed.results[0].result.content


async def test_assistant_provider_state_is_preserved_for_native_replay() -> None:
    turn = AssistantTurn(
        text="",
        tool_calls=(ToolCall(id="1", name="search", arguments={"query": "q"}),),
        stop_reason="tool_use",
        provider_state={"thinking_blocks": [{"signature": "opaque"}]},
    )
    model = ScriptedModel(turn)

    executed = await ToolTurnExecutor(model.complete_turn).run_turn(
        [{"role": "user", "content": "q"}],
        [],
    )

    assert executed.messages[-2]["provider_state"] == turn.provider_state
