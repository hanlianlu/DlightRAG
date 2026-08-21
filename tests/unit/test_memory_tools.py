# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""remember / forget / recall_memory composition and forget-miss."""

from dataclasses import replace

import pytest
from dlightrag_memory import InMemoryMemoryStore, Memory, commit_memory_write

from dlightrag.answer.evidence import EvidenceLedger
from dlightrag.answer.memory import MemoryProvenance, MemoryWrite
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


async def _retrieve(_query: str) -> object:
    raise RuntimeError("unused")


def test_parent_research_gets_memory_tools_child_does_not() -> None:
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
    assert {"remember", "forget", "recall_memory"}.isdisjoint({tool.name for tool in child})


async def test_forget_miss_is_reported() -> None:
    store = InMemoryMemoryStore()
    host = MemoryHost(
        owner_id="o",
        auth_mode="jwt",
        run_id="11111111-1111-1111-1111-111111111111",
        session_id="22222222-2222-2222-2222-222222222222",
        memory=Memory(store),
    )
    tool = forget_tool(host=host)
    result = await tool.execute(ForgetInput(memory_id="33333333-3333-3333-3333-333333333333"))
    assert "No matching memory" in result.content


async def test_remember_then_forget() -> None:
    store = InMemoryMemoryStore()
    host = MemoryHost(
        owner_id="o",
        auth_mode="jwt",
        run_id="11111111-1111-1111-1111-111111111111",
        session_id="22222222-2222-2222-2222-222222222222",
        memory=Memory(store),
    )
    remembered = await remember_tool(host=host).execute(
        RememberInput(kind="preference", body="No email.", confidence=0.9)
    )
    assert remembered.details is not None
    memory_id = str(remembered.details["memory_id"])
    forgotten = await forget_tool(host=host).execute(ForgetInput(memory_id=memory_id))
    assert forgotten.content == "Forgotten."
    assert await store.get(owner_id="o", memory_id=memory_id) is None


async def test_disabled_memory_rejects_model_tools() -> None:
    """Disabled stops model writes and recall; the rejection is explicit."""
    from dlightrag_memory import InMemoryMemoryStore, Memory

    host = MemoryHost(
        owner_id="o",
        auth_mode="jwt",
        run_id="11111111-1111-1111-1111-111111111111",
        session_id="22222222-2222-2222-2222-222222222222",
        memory=Memory(InMemoryMemoryStore()),
        enabled=False,
    )

    remembered = await remember_tool(host=host).execute(
        RememberInput(kind="preference", body="No email.", confidence=0.9)
    )
    assert "disabled" in remembered.content

    forgotten = await forget_tool(host=host).execute(
        ForgetInput(memory_id="33333333-3333-3333-3333-333333333333")
    )
    assert "disabled" in forgotten.content

    recalled = await recall_memory_tool(host=host).execute(RecallInput())
    assert "disabled" in recalled.content


async def test_supersede_rejects_other_owner() -> None:
    store = InMemoryMemoryStore()
    first = await commit_memory_write(
        store,
        MemoryWrite(
            owner_id="alpha",
            kind="fact",
            body="Alpha fact.",
            confidence=1.0,
            provenance=MemoryProvenance(run_id="r", session_id="s"),
        ),
    )
    assert first is not None
    with pytest.raises(ValueError, match="cannot change owner"):
        await store.supersede(
            owner_id="alpha",
            old_id=first.memory_id,
            new=replace(first, owner_id="beta", memory_id="other"),
        )
