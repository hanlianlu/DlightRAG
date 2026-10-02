# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Parent-only Profile Memory tools and the refusals they settle before storage.

The tools' receipts over a real store run against PostgreSQL in
tests/integration/test_memory_pg.py.
"""

from dlightrag_memory import Memory
from dlightrag_memory.postgres import PostgresMemoryStore

from dlightrag.engine.agent.environment import SearchToolchain
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.tools.composition import compose_research_tools
from dlightrag.engine.answer.tools.memory import (
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


async def _unreachable_pool() -> object:
    raise AssertionError("this test must not reach storage")


def _host(*, enabled: bool = True) -> MemoryHost:
    return MemoryHost(
        owner_id="o",
        run_id="11111111-1111-1111-1111-111111111111",
        session_id="22222222-2222-2222-2222-222222222222",
        memory=Memory(PostgresMemoryStore(pool_factory=_unreachable_pool)),
        enabled=enabled,
    )


def test_child_can_recall_but_cannot_mutate_profile() -> None:
    host = MemoryHost()
    parent = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        injected_tools=[],
        register_web_source=None,
        memory_host=host,
        search_toolchain=SearchToolchain(),
    )
    child = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        injected_tools=[],
        register_web_source=None,
        memory_host=host,
        child=True,
        search_toolchain=SearchToolchain(),
    )
    assert {"remember", "forget", "recall_memory"} <= {tool.name for tool in parent}
    assert {tool.name for tool in child} & {"remember", "forget"} == set()
    assert "recall_memory" in {tool.name for tool in child}


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
