# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""delegate_research composition and replay."""

from unittest.mock import AsyncMock

from dlightrag_agent.session.ids import SessionId
from dlightrag_agent.tools.context import bind_tool_call, reset_tool_call
from dlightrag_ai.messages import AssistantTurn

from dlightrag.answer.evidence import EvidenceLedger
from dlightrag.answer.tools.composition import compose_research_tools
from dlightrag.answer.tools.delegate import DelegateHost, DelegateInput, delegate_research_tool


async def _retrieve(_query: str) -> object:
    raise RuntimeError("unused")


def test_parent_tools_include_delegate_and_child_omits_it() -> None:
    host = DelegateHost()
    parent = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        delegate_host=host,
    )
    child = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        child=True,
    )
    assert "delegate_research" in {tool.name for tool in parent}
    assert {tool.name for tool in child} == {"search_knowledge_base"}
    assert "write" not in {tool.name for tool in child}
    assert "bash" not in {tool.name for tool in child}


async def test_finished_child_replays_stored_summary() -> None:
    model = AsyncMock()
    host = DelegateHost(
        parent_session_id=SessionId.new(),
        run_id=str(SessionId.new().value),
        owner_id="owner",
        load_child=AsyncMock(
            return_value={"status": "succeeded", "summary": "Prior child finding."}
        ),
        persist=AsyncMock(),
        model_func=model,
    )
    tool = delegate_research_tool(host=host)
    token = bind_tool_call("call-1", "delegate_research")
    try:
        result = await tool.execute(DelegateInput(objective="what happened?"))
    finally:
        reset_tool_call(token)
    assert result.content == "Prior child finding."


async def test_delegate_runs_a_silent_child_turn() -> None:
    async def model(**_kwargs: object) -> AssistantTurn:
        return AssistantTurn(text="Child summary.", tool_calls=(), stop_reason="stop")

    persist = AsyncMock()
    finish = AsyncMock()
    evidence = EvidenceLedger()
    evidence.add_rows(
        [
            {
                "chunk_id": "c1",
                "reference_id": "src",
                "content": "parent already had this",
                "file_path": "old.pdf",
                "_workspace": "ws",
                "metadata": {"title": "Old"},
            }
        ]
    )
    host = DelegateHost(
        parent_session_id=SessionId.new(),
        run_id=str(SessionId.new().value),
        owner_id="owner",
        model_func=model,
        persist=persist,
        load_child=AsyncMock(return_value=None),
        finish_child=finish,
        child_tools=[],
        evidence=evidence,
    )
    tool = delegate_research_tool(host=host)
    token = bind_tool_call("call-9", "delegate_research")
    try:
        result = await tool.execute(DelegateInput(objective="summarize filings"))
    finally:
        reset_tool_call(token)
    assert "Child summary." in result.content
    assert "Evidence handles" not in result.content
    persist.assert_awaited()
    finish.assert_awaited()
