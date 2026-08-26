# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Parent-only Profile Memory tools and typed operation details."""

from dlightrag_memory import Memory, MemoryProvenance
from dlightrag_memory.store import InMemoryMemoryStore

from dlightrag.answer.evidence import EvidenceLedger
from dlightrag.answer.tools.composition import compose_research_tools
from dlightrag.answer.tools.memory import (
    ForgetInput,
    MemoryHost,
    RecallInput,
    RememberInput,
    forget_tool,
    recall_memory_tool,
    remember_tool,
)
from tests.tool_helpers import tool_runtime


async def _retrieve(_query: str) -> object:
    raise RuntimeError("unused")


def _host(*, enabled: bool = True) -> MemoryHost:
    return MemoryHost(
        owner_id="o",
        auth_mode="jwt",
        run_id="11111111-1111-1111-1111-111111111111",
        session_id="22222222-2222-2222-2222-222222222222",
        memory=Memory(InMemoryMemoryStore()),
        enabled=enabled,
    )


def test_child_can_recall_but_cannot_mutate_profile() -> None:
    host = MemoryHost()
    parent = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        memory_host=host,
    )
    child = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        memory_host=host,
        child=True,
    )
    assert {"remember", "forget", "recall_memory"} <= {tool.name for tool in parent}
    assert {tool.name for tool in child} & {"remember", "forget"} == set()
    assert "recall_memory" in {tool.name for tool in child}


async def test_remember_then_forget_returns_typed_receipts() -> None:
    host = _host()
    remembered = await remember_tool(host=host).execute(
        RememberInput(kind="preference", body="No email."),
        tool_runtime(call_id="call-1"),
    )
    operation = (remembered.details or {})["memory_operation"]
    assert operation["outcome"] == "changed"
    memory_id = str(operation["memory_ids"][0])

    forgotten = await forget_tool(host=host).execute(
        ForgetInput(memory_id=memory_id), tool_runtime(call_id="call-2")
    )
    assert (forgotten.details or {})["memory_operation"]["outcome"] == "changed"


async def test_forget_miss_is_unchanged() -> None:
    result = await forget_tool(host=_host()).execute(
        ForgetInput(memory_id="33333333-3333-3333-3333-333333333333"),
        tool_runtime(),
    )
    assert (result.details or {})["memory_operation"]["outcome"] == "unchanged"


async def test_recall_returns_ids_with_relevant_records() -> None:
    host = _host()
    memory = host.memory
    assert memory is not None
    receipt = await memory.remember(
        owner_id="o",
        kind="preference",
        body="No email.",
        provenance=MemoryProvenance(origin_kind="answer_run", origin_id="seed"),
        idempotency_key="seed",
    )

    result = await recall_memory_tool(host=host).execute(RecallInput(query="email"), tool_runtime())

    assert receipt.memory_id is not None
    assert receipt.memory_id in result.text_content
    assert "No email." in result.text_content


async def test_disabled_or_stale_capability_rejects_tools() -> None:
    disabled = _host(enabled=False)
    result = await remember_tool(host=disabled).execute(
        RememberInput(kind="preference", body="No email."), tool_runtime()
    )
    assert result.is_error
    assert (result.details or {})["memory_operation"]["outcome"] == "rejected"

    stale = _host()

    async def no_longer_current(**_kwargs) -> bool:
        return False

    stale.capability_current = no_longer_current
    forgotten = await forget_tool(host=stale).execute(
        ForgetInput(memory_id="33333333-3333-3333-3333-333333333333"), tool_runtime()
    )
    assert forgotten.is_error
    recalled = await recall_memory_tool(host=stale).execute(
        RecallInput(query="anything"), tool_runtime()
    )
    assert recalled.is_error


def test_remember_is_safe_to_replay() -> None:
    assert remember_tool(host=_host()).replay_policy == "replayable"
