# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for provider-neutral tool turn execution."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from dlightrag.agent.tools import (
    AgentTool,
    ToolResult,
    ToolResultCapacityError,
    ToolTurnExecutor,
)
from dlightrag.ai.messages import AssistantTurn, ToolCall
from dlightrag.ai.tokens import estimate_tokens


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


class RecordingObservation:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class RecordingTelemetry:
    capture_sensitive_data = False

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.observation = RecordingObservation()

    @asynccontextmanager
    async def observe(self, name: str, **kwargs: Any):
        self.calls.append({"name": name, **kwargs})
        yield self.observation


def _turn(*calls: ToolCall, stop_reason: str = "tool_use") -> AssistantTurn:
    return AssistantTurn(
        text="",
        tool_calls=tuple(calls),
        stop_reason=stop_reason,  # type: ignore[arg-type]
    )


async def _run_turn(
    executor: ToolTurnExecutor,
    messages: list[dict[str, Any]],
    tools: list[AgentTool],
    **kwargs: Any,
):
    prepared = await executor.prepare_turn(
        messages,
        tools,
        tool_choice=kwargs.pop("tool_choice", "auto"),
        max_tokens=kwargs.pop("max_tokens", None),
    )
    observation_budget = kwargs.pop("observation_budget", None)
    on_result = kwargs.pop("on_result", None)
    assert not kwargs
    return await executor.execute_prepared(
        prepared,
        tools,
        observation_budget=observation_budget,
        on_result=on_result,
    )


async def test_tool_result_error_flag_reaches_execution_and_transcript() -> None:
    async def execute(_args: BaseModel) -> ToolResult:
        return ToolResult(content="remote failed", is_error=True)

    call = ToolCall(id="call-error", name="remote", arguments={"query": "x"})
    executor = ToolTurnExecutor(ScriptedModel(_turn(call)).complete_turn)  # type: ignore[arg-type]

    turn = await _run_turn(
        executor,
        [{"role": "user", "content": "go"}],
        [AgentTool("remote", "Remote tool.", SearchArgs, execute)],
    )

    assert turn.results[0].is_error is True
    assert turn.results[0].observation.is_error is True
    assert turn.messages[-1]["is_error"] is True


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
        _run_turn(
            executor,
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


async def test_parallel_results_share_the_current_request_residual_budget() -> None:
    async def execute(args: BaseModel) -> ToolResult:
        assert isinstance(args, SearchArgs)
        return ToolResult(content=args.query * 400)

    model = ScriptedModel(
        _turn(
            ToolCall(id="1", name="search", arguments={"query": "a"}),
            ToolCall(id="2", name="search", arguments={"query": "b"}),
        )
    )
    measured_transcript: list[dict[str, Any]] = []

    def observation_budget(transcript: list[dict[str, Any]]) -> int:
        measured_transcript.extend(transcript)
        return 150

    executed = await _run_turn(
        ToolTurnExecutor(model.complete_turn),
        [{"role": "user", "content": "q"}],
        [AgentTool("search", "Search.", SearchArgs, execute)],
        observation_budget=observation_budget,
    )

    assert measured_transcript[-1]["role"] == "assistant"
    assert sum(estimate_tokens(result.result.content) for result in executed.results) <= 150
    assert 0 < len(executed.results[0].result.content) < 400
    assert 0 < len(executed.results[1].result.content) < 400


async def test_durable_callback_receives_fitted_error_preserving_results() -> None:
    async def execute(args: BaseModel) -> ToolResult:
        assert isinstance(args, SearchArgs)
        if args.query == "error":
            raise RuntimeError("failed " + "x" * 400)
        return ToolResult(content="y" * 800)

    settled: list[Any] = []

    async def on_result(_intent: Any, execution: Any, _is_last: bool) -> None:
        settled.append(execution)

    model = ScriptedModel(
        _turn(
            ToolCall(id="1", name="search", arguments={"query": "ok"}),
            ToolCall(id="2", name="search", arguments={"query": "error"}),
        )
    )
    executed = await _run_turn(
        ToolTurnExecutor(model.complete_turn),
        [{"role": "user", "content": "q"}],
        [AgentTool("search", "Search.", SearchArgs, execute)],
        observation_budget=lambda _transcript: 80,
        on_result=on_result,
    )

    assert settled == list(executed.results)
    assert settled[1].is_error is True
    assert sum(estimate_tokens(item.result.content) for item in settled) <= 80


async def test_parallel_result_fitting_preserves_every_continuation_suffix() -> None:
    suffix = "[more text available; cursor=opaque-cursor]"

    async def execute(args: BaseModel) -> ToolResult:
        assert isinstance(args, SearchArgs)
        return ToolResult(
            content=f"{args.query * 400}\n{suffix}",
            protected_suffix=suffix,
        )

    model = ScriptedModel(
        _turn(
            ToolCall(id="1", name="search", arguments={"query": "a"}),
            ToolCall(id="2", name="search", arguments={"query": "b"}),
        )
    )

    executed = await _run_turn(
        ToolTurnExecutor(model.complete_turn),
        [{"role": "user", "content": "q"}],
        [AgentTool("search", "Search.", SearchArgs, execute)],
        observation_budget=lambda _transcript: 80,
    )

    for result in executed.results:
        assert result.result.content.endswith(suffix)
        assert "tool result truncated" in result.result.content
        assert result.result.content
    assert sum(estimate_tokens(result.result.content) for result in executed.results) <= 80


async def test_unfit_protected_suffix_raises_a_capacity_error() -> None:
    suffix = "[continuation " + "x" * 200 + "]"

    async def execute(_args: BaseModel) -> ToolResult:
        return ToolResult(content=f"body\n{suffix}", protected_suffix=suffix)

    model = ScriptedModel(_turn(ToolCall(id="1", name="search", arguments={"query": "q"})))

    with pytest.raises(ToolResultCapacityError, match="continuation"):
        await _run_turn(
            ToolTurnExecutor(model.complete_turn),
            [{"role": "user", "content": "q"}],
            [AgentTool("search", "Search.", SearchArgs, execute)],
            observation_budget=lambda _transcript: 5,
        )


async def test_tool_execution_uses_injected_telemetry() -> None:
    async def execute(_args: BaseModel) -> ToolResult:
        return ToolResult(content="found")

    telemetry = RecordingTelemetry()
    model = ScriptedModel(_turn(ToolCall(id="call-1", name="search", arguments={"query": "q"})))

    executed = await _run_turn(
        ToolTurnExecutor(model.complete_turn, telemetry=telemetry),
        [{"role": "user", "content": "q"}],
        [AgentTool("search", "Search.", SearchArgs, execute)],
    )

    assert executed.results[0].is_error is False
    assert telemetry.calls == [
        {
            "name": "agent_tool",
            "as_type": "tool",
            "metadata": {"tool": "search", "call_id": "call-1"},
        }
    ]
    assert telemetry.observation.updates == [
        {
            "output": {
                "tool": "search",
                "call_id": "call-1",
                "outcome": "ok",
                "duration_ms": executed.results[0].observation.duration_ms,
                "cached": False,
                "is_error": False,
                "content_chars": 5,
            }
        }
    ]


async def test_cancelled_tool_call_cancels_and_joins_siblings() -> None:
    sibling_started = asyncio.Event()
    sibling_finished = asyncio.Event()
    release_sibling = asyncio.Event()

    async def execute(args: BaseModel) -> ToolResult:
        assert isinstance(args, SearchArgs)
        if args.query == "cancel":
            await sibling_started.wait()
            raise asyncio.CancelledError
        sibling_started.set()
        try:
            await release_sibling.wait()
            return ToolResult(content="late")
        finally:
            sibling_finished.set()

    model = ScriptedModel(
        _turn(
            ToolCall(id="1", name="search", arguments={"query": "cancel"}),
            ToolCall(id="2", name="search", arguments={"query": "wait"}),
        )
    )

    try:
        with pytest.raises(asyncio.CancelledError):
            await _run_turn(
                ToolTurnExecutor(model.complete_turn),
                [{"role": "user", "content": "q"}],
                [AgentTool("search", "Search.", SearchArgs, execute)],
            )
        assert sibling_finished.is_set()
    finally:
        release_sibling.set()
        await asyncio.sleep(0)


async def test_base_exception_tool_call_cancels_and_joins_siblings() -> None:
    class ToolAbort(BaseException):
        pass

    sibling_started = asyncio.Event()
    sibling_finished = asyncio.Event()

    async def execute(args: BaseModel) -> ToolResult:
        assert isinstance(args, SearchArgs)
        if args.query == "abort":
            await sibling_started.wait()
            raise ToolAbort
        sibling_started.set()
        try:
            await asyncio.Event().wait()
            return ToolResult(content="unreachable")
        finally:
            sibling_finished.set()

    model = ScriptedModel(
        _turn(
            ToolCall(id="1", name="search", arguments={"query": "abort"}),
            ToolCall(id="2", name="search", arguments={"query": "wait"}),
        )
    )

    with pytest.raises(ToolAbort):
        await _run_turn(
            ToolTurnExecutor(model.complete_turn),
            [{"role": "user", "content": "q"}],
            [AgentTool("search", "Search.", SearchArgs, execute)],
        )
    assert sibling_finished.is_set()


async def test_invalid_arguments_are_returned_to_the_model_without_execution() -> None:
    calls = 0

    async def execute(_args: BaseModel) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult(content="unreachable")

    model = ScriptedModel(_turn(ToolCall(id="1", name="search", arguments={"unexpected": True})))
    executed = await _run_turn(
        ToolTurnExecutor(model.complete_turn),
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
    executed = await _run_turn(
        ToolTurnExecutor(model.complete_turn),
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
    executed = await _run_turn(
        ToolTurnExecutor(model.complete_turn),
        [{"role": "user", "content": "q"}],
        [AgentTool("search", "Search.", SearchArgs, execute)],
    )

    assert calls == 0
    assert executed.results[0].is_error is True
    assert "token limit" in executed.results[0].result.content


async def test_unknown_tool_is_an_error_result_not_an_exception() -> None:
    model = ScriptedModel(_turn(ToolCall(id="1", name="invented", arguments={})))

    executed = await _run_turn(
        ToolTurnExecutor(model.complete_turn),
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

    executed = await _run_turn(
        ToolTurnExecutor(model.complete_turn),
        [{"role": "user", "content": "q"}],
        [],
    )

    assert executed.messages[-2]["provider_state"] == turn.provider_state


async def _search_tool(_args: BaseModel) -> ToolResult:
    return ToolResult(content="found")


async def test_preflight_creates_ordered_intents_for_valid_calls() -> None:
    from dlightrag.agent.tools import preflight_tool_calls

    tools = [
        AgentTool(
            name="search",
            description="search docs",
            input_model=SearchArgs,
            execute=_search_tool,
        )
    ]
    turn = _turn(
        ToolCall(id="1", name="search", arguments={"query": "a"}),
        ToolCall(id="2", name="search", arguments={"query": "b"}),
    )
    preflight = preflight_tool_calls(turn, tools)

    assert [intent.source_call_id for intent in preflight.intents] == ["1", "2"]
    assert [intent.tool_name for intent in preflight.intents] == ["search", "search"]
    assert all(intent.replay_policy == "safe" for intent in preflight.intents)
    assert all(len(intent.input_schema_digest) == 64 for intent in preflight.intents)
    assert all(intent.contract_version == 1 for intent in preflight.intents)
    assert preflight.validation_results == ()


async def test_preflight_orders_invalid_calls_as_validation_results() -> None:
    from dlightrag.agent.tools import preflight_tool_calls

    tools = [
        AgentTool(
            name="search",
            description="search docs",
            input_model=SearchArgs,
            execute=_search_tool,
        )
    ]
    turn = _turn(
        ToolCall(id="1", name="invented", arguments={}),
        ToolCall(id="2", name="search", arguments={"query": "ok"}),
    )
    preflight = preflight_tool_calls(turn, tools)

    assert [intent.source_call_id for intent in preflight.intents] == ["2"]
    assert len(preflight.validation_results) == 1
    validation = preflight.validation_results[0]
    assert validation.call_id == "1"
    assert validation.outcome == "unknown_tool"
    assert "not available" in validation.content


async def test_preflight_is_never_policy_for_web_and_contracts_are_pinned() -> None:
    from dlightrag.agent.tools import preflight_tool_calls

    web_tool = AgentTool(
        name="search_web",
        description="web search",
        input_model=SearchArgs,
        execute=_search_tool,
        replay_policy="never",
        contract_version=7,
    )
    turn = _turn(ToolCall(id="1", name="search_web", arguments={"query": "q"}))
    preflight = preflight_tool_calls(turn, [web_tool])

    (intent,) = preflight.intents
    assert intent.replay_policy == "never"
    assert intent.contract_version == 7


async def test_length_stopped_turn_produces_no_intents() -> None:
    model = ScriptedModel(
        _turn(
            ToolCall(id="1", name="search", arguments={"query": "q"}),
            stop_reason="length",
        )
    )
    tools = [
        AgentTool(
            name="search",
            description="search docs",
            input_model=SearchArgs,
            execute=_search_tool,
        )
    ]
    executed = await _run_turn(
        ToolTurnExecutor(model.complete_turn), [{"role": "user", "content": "q"}], tools
    )
    assert executed.intents == ()
    assert executed.results[0].is_error is True


async def test_prepare_turn_never_executes_tools() -> None:
    executed = asyncio.Event()

    async def execute(args: BaseModel) -> ToolResult:
        executed.set()
        assert isinstance(args, SearchArgs)
        return ToolResult(content="ran")

    tools = [AgentTool("search", "Search.", SearchArgs, execute)]
    model = ScriptedModel(_turn(ToolCall(id="1", name="search", arguments={"query": "q"})))
    executor = ToolTurnExecutor(model.complete_turn)

    prepared = await executor.prepare_turn([{"role": "user", "content": "q"}], tools)

    assert not executed.is_set()
    (intent,) = prepared.preflight.intents
    assert intent.source_call_id == "1"

    executed_turn = await executor.execute_prepared(prepared, tools)
    assert executed.is_set()
    assert executed_turn.results[0].call.id == "1"


async def test_execute_prepared_settles_in_source_order_despite_completion_order() -> None:
    """A fast second call must wait behind the first intent's settlement."""
    release_first = asyncio.Event()
    second_completed = asyncio.Event()

    async def execute(args: BaseModel) -> ToolResult:
        assert isinstance(args, SearchArgs)
        if args.query == "first":
            await release_first.wait()
        else:
            second_completed.set()
        return ToolResult(content=f"result:{args.query}")

    tools = [AgentTool("search", "Search.", SearchArgs, execute)]
    model = ScriptedModel(
        _turn(
            ToolCall(id="1", name="search", arguments={"query": "first"}),
            ToolCall(id="2", name="search", arguments={"query": "second"}),
        )
    )
    settled: list[str] = []

    async def on_result(intent: Any, execution: Any, is_last: bool) -> None:
        del intent, is_last
        if execution is not None:
            settled.append(execution.call.id)

    executor = ToolTurnExecutor(model.complete_turn)
    prepared = await executor.prepare_turn([{"role": "user", "content": "q"}], tools)
    task = asyncio.create_task(executor.execute_prepared(prepared, tools, on_result=on_result))

    await asyncio.wait_for(second_completed.wait(), timeout=1)
    assert settled == []  # the second result waits behind the first intent

    release_first.set()
    executed = await task

    assert settled == ["1", "2"]
    assert [result.call.id for result in executed.results] == ["1", "2"]
    assert [message["content"] for message in executed.messages[-2:]] == [
        "result:first",
        "result:second",
    ]
