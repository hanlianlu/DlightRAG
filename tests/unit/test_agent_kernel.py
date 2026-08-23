# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Product-neutral Agent kernel contracts."""

from datetime import UTC, datetime

from pydantic import BaseModel

from dlightrag.agent.context import ContextContribution, ContextProjector
from dlightrag.agent.loop import AgentLoop, AgentLoopCancelled
from dlightrag.agent.session.entries import RunSegmentEntry, UserMessageEntry
from dlightrag.agent.session.graph import AgentSessionGraph
from dlightrag.agent.session.ids import EntryId, SessionId
from dlightrag.agent.tools import AgentTool, ExecutedTurn, ToolResult
from dlightrag.agent.tools.registry import DuplicateToolError, ToolRegistry
from dlightrag.ai.messages import AssistantTurn, ToolCall


class _Input(BaseModel):
    value: str


async def _execute(arguments: BaseModel, _runtime: object) -> ToolResult:
    return ToolResult.text(str(arguments))


def _tool(name: str) -> AgentTool:
    return AgentTool(name, f"{name} tool", _Input, _execute)


def test_context_projector_orders_authorities_and_keeps_source_order() -> None:
    projected = ContextProjector().project(
        [
            ContextContribution(
                source="working",
                authority="working",
                messages=({"role": "assistant", "content": "work"},),
            ),
            ContextContribution(
                source="system",
                authority="system",
                messages=({"role": "system", "content": "rules"},),
                compressible=False,
            ),
            ContextContribution(
                source="question",
                authority="user",
                messages=({"role": "user", "content": "question"},),
                compressible=False,
            ),
        ]
    )

    assert [message["content"] for message in projected.messages] == [
        "rules",
        "question",
        "work",
    ]
    assert projected.sources == ("system", "question", "working")


def test_tool_registry_preserves_order_and_rejects_duplicates() -> None:
    registry = ToolRegistry((_tool("read"), _tool("grep")))
    assert registry.names == ("read", "grep")
    assert [tool.name for tool in registry.resolve(("grep",))] == ["grep"]

    try:
        registry.register(_tool("read"))
    except DuplicateToolError as exc:
        assert exc.names == ("read",)
    else:  # pragma: no cover
        raise AssertionError("duplicate tool was accepted")


def test_session_graph_rejects_a_resume_from_the_wrong_head() -> None:
    session_id = SessionId.new()
    first = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        sequence=1,
        timestamp=datetime.now(UTC),
        content="one",
    )
    segment = RunSegmentEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        sequence=2,
        timestamp=datetime.now(UTC),
        segment_id=EntryId.new().value,
        kind="resume",
        parent_head_id=EntryId.new().value,
    )

    try:
        AgentSessionGraph.from_linear_entries(session_id, (first, segment))
    except ValueError as exc:
        assert "parent head" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mismatched run segment head was accepted")


def test_linear_session_graph_derives_parent_links_and_head() -> None:
    session_id = SessionId.new()
    first = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        sequence=1,
        timestamp=datetime.now(UTC),
        content="one",
    )
    second = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        sequence=2,
        timestamp=datetime.now(UTC),
        content="two",
    )

    graph = AgentSessionGraph.from_linear_entries(session_id, (first, second))

    assert graph.head_entry_id == second.entry_id
    assert graph.nodes[0].parent_entry_id is None
    assert graph.nodes[1].parent_entry_id == first.entry_id
    assert graph.ancestry() == (first, second)
    assert graph.select_head(first.entry_id).ancestry() == (first,)


class _Driver:
    def __init__(self, *, cancel: bool = False) -> None:
        self.cancel = cancel
        self.turns = 0

    async def check_cancelled(self) -> None:
        if self.cancel:
            raise AgentLoopCancelled

    async def run_turn(self, turn_number: int) -> ExecutedTurn:
        self.turns = turn_number
        calls = (ToolCall(id="call", name="read", arguments={}),) if turn_number == 1 else ()
        return ExecutedTurn(
            assistant=AssistantTurn(
                text="working" if calls else "done",
                tool_calls=calls,
                stop_reason="tool_use" if calls else "stop",
            ),
            results=(),
            messages=[],
        )


async def test_agent_loop_emits_ordered_events_until_model_silence() -> None:
    events = []

    async def collect(event) -> None:
        events.append((event.kind, event.turn_number))

    result = await AgentLoop(on_event=collect).run(_Driver())

    assert result.turn_count == 2
    assert result.stop_reason == "model_stop"
    assert result.last_turn is not None and result.last_turn.assistant.text == "done"
    assert events == [
        ("agent_start", None),
        ("turn_start", 1),
        ("turn_end", 1),
        ("turn_start", 2),
        ("turn_end", 2),
        ("agent_end", 2),
    ]


async def test_agent_loop_admits_a_control_arriving_during_terminal_turn() -> None:
    class ControlledDriver:
        turns = 0
        checks = 0

        async def check_cancelled(self) -> None:
            return None

        async def run_turn(self, turn_number: int) -> ExecutedTurn:
            self.turns = turn_number
            return ExecutedTurn(
                assistant=AssistantTurn(
                    text=f"draft {turn_number}", tool_calls=(), stop_reason="stop"
                ),
                results=(),
                messages=[],
            )

        async def continue_after_stop(self) -> bool:
            self.checks += 1
            return self.checks == 1

    driver = ControlledDriver()
    result = await AgentLoop().run(driver)

    assert result.turn_count == 2
    assert result.last_turn is not None
    assert result.last_turn.assistant.text == "draft 2"


async def test_agent_loop_returns_cancelled_without_a_turn() -> None:
    result = await AgentLoop().run(_Driver(cancel=True))
    assert result.turn_count == 0
    assert result.stop_reason == "cancelled"
    assert result.last_turn is None
