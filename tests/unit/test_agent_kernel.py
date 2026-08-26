# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Product-neutral Agent kernel contracts."""

from datetime import UTC, datetime

from pydantic import BaseModel

from dlightrag.agent.context import ContextContribution, ContextProjector
from dlightrag.agent.session.entries import UserMessageEntry
from dlightrag.agent.session.graph import AgentSessionGraph
from dlightrag.agent.session.ids import EntryId, SessionId
from dlightrag.agent.tools import AgentTool, ToolResult
from dlightrag.agent.tools.registry import DuplicateToolError, ToolRegistry


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


def test_tool_registry_preserves_order_and_rejects_duplicates() -> None:
    registry = ToolRegistry((_tool("read"), _tool("grep")))
    assert registry.names == ("read", "grep")
    try:
        registry.register(_tool("read"))
    except DuplicateToolError as exc:
        assert exc.names == ("read",)
    else:  # pragma: no cover
        raise AssertionError("duplicate Tool was accepted")


def test_session_graph_requires_physical_parent_links() -> None:
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
        parent_entry_id=first.entry_id,
        content="two",
    )
    graph = AgentSessionGraph.from_entries(session_id, (first, second))
    assert graph.select_head(second.entry_id).ancestry() == (first, second)
