# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Parent-only Profile Memory tools and the refusals they settle before storage.

The tools' receipts over a real store run against PostgreSQL in
tests/integration/test_memory_pg.py.
"""

from collections.abc import Callable
from typing import Any, cast

import pytest
from dlightrag_memory import (
    Memory,
    MemoryOperationReceipt,
    MemoryProvenance,
    MemoryRecord,
    MemoryStatus,
)
from dlightrag_memory.postgres import PostgresMemoryStore
from pydantic import BaseModel, ValidationError

from dlightrag.engine.agent.environment.toolchain import (
    SearchToolchain,
)
from dlightrag.engine.agent.tools.contracts import ToolDeclaration
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.tools.composition import compose_research_tools
from dlightrag.engine.answer.tools.memory import (
    ForgetInput,
    LookupInput,
    MemoryHost,
    RememberInput,
    UndoInput,
    forget_declaration,
    forget_tool,
    lookup_tool,
    remember_declaration,
    remember_tool,
    undo_declaration,
    undo_tool,
)
from tests.support.tool_helpers import tool_runtime


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
    mutations = {"memory_remember", "memory_forget", "memory_undo"}
    assert {*mutations, "memory_lookup"} <= {tool.name for tool in parent}
    assert {tool.name for tool in child} & mutations == set()
    assert "memory_lookup" in {tool.name for tool in child}
    # A merge of several records is never handed to the model that answers.
    assert "memory_fuse" not in {tool.name for tool in parent}


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
    undone = await undo_tool(host=stale).execute(
        UndoInput(memory_id="33333333-3333-3333-3333-333333333333"), tool_runtime()
    )
    assert undone.is_error
    assert (undone.details or {})["memory_operation"]["operation"] == "undo"
    looked_up = await lookup_tool(host=stale).execute(LookupInput(query="anything"), tool_runtime())
    assert looked_up.is_error


async def test_undo_needs_exactly_one_record_selector() -> None:
    # Refused by the checklist before any storage is reached.
    result = await undo_tool(host=_host()).execute(UndoInput(), tool_runtime())
    assert result.is_error
    assert (result.details or {})["memory_operation"]["outcome"] == "rejected"


def test_remember_is_safe_to_replay() -> None:
    assert remember_tool(host=_host()).replay_policy == "replayable"
    assert undo_tool(host=_host()).replay_policy == "replayable"


_UNKNOWN = "33333333-3333-3333-3333-333333333333"
_PROVENANCE = MemoryProvenance(origin_kind="answer_run", origin_id="run")


class _ScriptedStore:
    """A Memory whose forget and undo settle as scripted and whose records are the ones given."""

    def __init__(
        self,
        outcome: str,
        records: dict[str, MemoryStatus] | None = None,
        target_change_id: str | None = None,
    ) -> None:
        self.outcome = outcome
        self.records = records or {}
        self.target_change_id = target_change_id
        self.reads: list[str] = []

    async def forget(self, **kwargs: Any) -> MemoryOperationReceipt:
        return self._receipt("forget", kwargs["memory_id"])

    async def undo(self, **kwargs: Any) -> MemoryOperationReceipt:
        return self._receipt("undo", kwargs["memory_id"])

    def _receipt(self, action: str, memory_id: str | None) -> MemoryOperationReceipt:
        changed = self.outcome == "changed"
        return MemoryOperationReceipt(
            change_id="change-1",
            action=action,  # type: ignore[arg-type]
            outcome=self.outcome,  # type: ignore[arg-type]
            memory_ids=(memory_id,) if changed and memory_id else (),
            provenance=_PROVENANCE,
            body="Likes tea." if changed else "",
            target_change_id=self.target_change_id,
        )

    async def get(self, *, owner_id: str, memory_id: str) -> MemoryRecord | None:
        self.reads.append(memory_id)
        status = self.records.get(memory_id)
        if status is None:
            return None
        return MemoryRecord(
            owner_id=owner_id,
            memory_id=memory_id,
            kind="fact",
            body="Likes tea.",
            provenance=_PROVENANCE,
            status=status,
        )


async def _forget(store: _ScriptedStore, **arguments: Any) -> Any:
    host = _host()
    host.memory = cast(Memory, store)
    return await forget_tool(host=host).execute(ForgetInput(**arguments), tool_runtime())


async def _undo(store: _ScriptedStore, **arguments: Any) -> Any:
    host = _host()
    host.memory = cast(Memory, store)
    return await undo_tool(host=host).execute(UndoInput(**arguments), tool_runtime())


async def test_forgetting_an_id_no_memory_has_is_refused_and_says_where_ids_come_from() -> None:
    store = _ScriptedStore("unchanged")

    result = await _forget(store, memory_id=_UNKNOWN)

    assert result.is_error
    assert result.text_content == (
        f"No Profile Memory has id {_UNKNOWN}. Call memory_lookup for current ids."
    )
    # The receipt the store settled is still what is recorded for the call.
    assert result.details["memory_operation"]["outcome"] == "unchanged"


@pytest.mark.parametrize(
    ("status", "said"),
    [("forgotten", "was already forgotten"), ("superseded", "was already replaced")],
)
async def test_forgetting_a_memory_that_is_no_longer_active_is_not_an_error(
    status: MemoryStatus, said: str
) -> None:
    result = await _forget(_ScriptedStore("unchanged", {_UNKNOWN: status}), memory_id=_UNKNOWN)

    assert not result.is_error
    assert said in result.text_content


async def test_forgetting_a_body_no_active_memory_has_is_refused() -> None:
    store = _ScriptedStore("unchanged")

    result = await _forget(store, body="Likes coffee.")

    assert result.is_error
    assert result.text_content == (
        "No active Profile Memory has exactly that body. Call memory_lookup to find its id."
    )
    assert store.reads == []


async def test_a_forget_that_changed_something_is_never_second_guessed() -> None:
    store = _ScriptedStore("changed")

    result = await _forget(store, memory_id=_UNKNOWN)

    assert not result.is_error
    assert result.text_content == f"Forgot Profile Memory {_UNKNOWN}."
    assert store.reads == []


_ID_ARGUMENTS = [
    pytest.param(ForgetInput, "memory_id", {}, id="forget"),
    pytest.param(UndoInput, "memory_id", {}, id="undo"),
    pytest.param(
        RememberInput, "supersedes_id", {"kind": "fact", "body": "Likes tea."}, id="remember"
    ),
]


@pytest.mark.parametrize(("model", "field", "arguments"), _ID_ARGUMENTS)
def test_a_memory_id_that_is_not_an_id_is_refused_before_storage(
    model: type[BaseModel], field: str, arguments: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError, match="'mem-1' is not a memory id; call memory_lookup"):
        model(**arguments, **{field: "mem-1"})

    assert getattr(model(**arguments, **{field: _UNKNOWN}), field) == _UNKNOWN
    assert getattr(model(**arguments), field) is None
    assert getattr(model(**arguments, **{field: " "}), field) is None


@pytest.mark.parametrize(
    ("declaration", "field"),
    [
        pytest.param(remember_declaration, "supersedes_id", id="remember"),
        pytest.param(forget_declaration, "memory_id", id="forget"),
        pytest.param(undo_declaration, "memory_id", id="undo"),
    ],
)
def test_the_id_check_leaves_the_pinned_tool_schema_a_plain_string(
    declaration: Callable[[], ToolDeclaration], field: str
) -> None:
    property_schema = declaration().definition.parameters["properties"][field]

    assert property_schema["anyOf"] == [{"type": "string"}, {"type": "null"}]


async def test_undoing_an_id_no_memory_has_is_refused_and_says_where_ids_come_from() -> None:
    store = _ScriptedStore("conflict")

    result = await _undo(store, memory_id=_UNKNOWN)

    assert result.is_error
    assert result.text_content == (
        f"No Profile Memory has id {_UNKNOWN}. Call memory_lookup for current ids."
    )
    # The receipt the store settled is still what is recorded for the call.
    assert result.details["memory_operation"]["outcome"] == "conflict"


@pytest.mark.parametrize(
    ("status", "said"),
    [("forgotten", "was forgotten"), ("superseded", "was replaced by a newer memory")],
)
async def test_undoing_a_memory_that_is_no_longer_active_says_so_and_points_to_lookup(
    status: MemoryStatus, said: str
) -> None:
    result = await _undo(_ScriptedStore("conflict", {_UNKNOWN: status}), memory_id=_UNKNOWN)

    assert result.is_error
    assert said in result.text_content
    assert "memory_undo names an active memory" in result.text_content
    assert "memory_lookup" in result.text_content


async def test_undoing_a_body_no_active_memory_has_is_refused() -> None:
    store = _ScriptedStore("conflict")

    result = await _undo(store, body="Likes coffee.")

    assert result.is_error
    assert result.text_content == (
        "No active Profile Memory has exactly that body. Call memory_lookup to find its id."
    )
    assert store.reads == []


async def test_an_undo_of_an_active_memory_keeps_the_concurrency_message() -> None:
    # An active record the store still found nothing to compensate for, such as one with no change
    # behind it, is not a wrong id: the generic receipt says what it can.
    result = await _undo(_ScriptedStore("conflict", {_UNKNOWN: "active"}), memory_id=_UNKNOWN)

    assert result.is_error
    assert result.text_content == (
        "Profile Memory changed concurrently; recall it again before retrying."
    )


async def test_an_undo_conflict_that_names_its_change_is_never_second_guessed() -> None:
    store = _ScriptedStore("conflict", target_change_id="change-0")

    result = await _undo(store, memory_id=_UNKNOWN)

    assert result.is_error
    assert result.text_content == (
        "Profile Memory changed concurrently; recall it again before retrying."
    )
    assert store.reads == []


async def test_an_undo_that_changed_something_is_never_second_guessed() -> None:
    store = _ScriptedStore("changed", target_change_id="change-0")

    result = await _undo(store, memory_id=_UNKNOWN)

    assert not result.is_error
    assert result.text_content == f"Undid Profile Memory {_UNKNOWN}."
    assert store.reads == []
