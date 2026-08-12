# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for the durable Answer coordinator on PostgreSQL 18.

Exercises the coordinator against the real fenced store: claim, per-turn
checkpoint, process restart from the latest committed checkpoint, gap-free event
replay across a reconnect, observed cancellation, graceful-shutdown requeue, and
the checkpoint codec's round trip through JSONB.

Every test runs inside a throwaway database created and dropped per test, so the
developer's ``dlightrag`` database is never mutated.

Requires PostgreSQL at localhost:5432 (dlightrag/dlightrag); skipped otherwise.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any

import asyncpg
import pytest

from dlightrag.core.agent.tools import _ToolCallCache
from dlightrag.core.answer_runs.checkpoints import encode_checkpoint_state, restore_agent_state
from dlightrag.core.answer_runs.coordinator import AnswerRunCoordinator, RunSession
from dlightrag.core.answer_runs.models import AgentRunState
from dlightrag.core.memory.episode import RunEpisode
from dlightrag.core.memory.evidence import EvidenceLedger
from dlightrag.core.resources.models import ResourceInput
from dlightrag.core.resources.registry import ResourceRegistry
from dlightrag.storage.answer_runs import PGAnswerRunStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_PG_CONN_KWARGS: dict[str, Any] = dict(
    host="localhost",
    port=5432,
    user="dlightrag",
    password="dlightrag",
    database="dlightrag",
)

_OWNER = "owner-alpha"
_REQUEST: dict[str, Any] = {"query": "why", "workspaces": ["default"]}


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def store() -> AsyncIterator[PGAnswerRunStore]:
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")

    db_name = f"dlightrag_runtime_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()

    pool = await asyncpg.create_pool(
        **{**_PG_CONN_KWARGS, "database": db_name}, min_size=1, max_size=8
    )
    try:
        assert pool is not None
        created = PGAnswerRunStore(pool=pool)
        await created.initialize()
        yield created
    finally:
        if pool is not None:
            await pool.close()
        admin = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


class _Executor:
    def __init__(self, body: Any) -> None:
        self._body = body

    async def execute(self, session: RunSession) -> Mapping[str, Any]:
        return await self._body(session)


async def _settle(predicate: Any, *, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition never became true")


def _status_is(store: PGAnswerRunStore, run_id: str, status: str) -> Any:
    async def _check() -> bool:
        run = await store.get_run(owner_id=_OWNER, run_id=run_id)
        return run is not None and run.status == status

    return _check


async def test_checkpointed_turn_survives_a_new_worker(store: PGAnswerRunStore) -> None:
    creation = await store.create_run(owner_id=_OWNER, request=_REQUEST)
    run_id = creation.run.run_id
    seen: list[int] = []

    async def body(session: RunSession) -> Mapping[str, Any]:
        seen.append(session.completed_turns)
        if session.completed_turns == 0:
            await session.commit_checkpoint(
                {"version": 1, "completed_turns": 1, "state": {"episode": {"exchanges": []}}}
            )
            await asyncio.sleep(30)
        return {"answer": "second attempt", "turns": session.completed_turns}

    first = AnswerRunCoordinator(store=store, executor=_Executor(body), max_async=1)
    await first.start()
    await _settle(_checkpoint_committed(store, run_id))
    await first.aclose()

    second = AnswerRunCoordinator(store=store, executor=_Executor(body), max_async=1)
    await second.start()
    try:
        await _settle(_status_is(store, run_id, "succeeded"))
    finally:
        await second.aclose()

    run = await store.get_run(owner_id=_OWNER, run_id=run_id)
    assert run is not None
    assert run.completed_turns == 1
    assert run.result == {"answer": "second attempt", "turns": 1}
    assert seen == [0, 1]


def _checkpoint_committed(store: PGAnswerRunStore, run_id: str) -> Any:
    async def _check() -> bool:
        run = await store.get_run(owner_id=_OWNER, run_id=run_id)
        return run is not None and run.completed_turns == 1

    return _check


async def test_graceful_shutdown_requeues_without_crash_recovery(
    store: PGAnswerRunStore,
) -> None:
    creation = await store.create_run(owner_id=_OWNER, request=_REQUEST)
    run_id = creation.run.run_id
    running = asyncio.Event()

    async def body(session: RunSession) -> Mapping[str, Any]:
        running.set()
        await asyncio.sleep(30)
        return {"answer": "unreachable"}

    coordinator = AnswerRunCoordinator(store=store, executor=_Executor(body), max_async=1)
    await coordinator.start()
    await asyncio.wait_for(running.wait(), timeout=10)
    await coordinator.aclose()

    run = await store.get_run(owner_id=_OWNER, run_id=run_id)
    assert run is not None
    assert run.status == "queued"
    assert run.recovery_count == 0
    assert run.lease_owner is None


async def test_reconnecting_subscriber_replays_without_gaps_or_duplicates(
    store: PGAnswerRunStore,
) -> None:
    creation = await store.create_run(owner_id=_OWNER, request=_REQUEST)
    run_id = creation.run.run_id

    async def body(session: RunSession) -> Mapping[str, Any]:
        await session.enter_phase("generating")
        await session.emit_token("hello ")
        await session.flush_tokens()
        await session.emit_token("world")
        await session.flush_tokens()
        return {"answer": "hello world"}

    coordinator = AnswerRunCoordinator(store=store, executor=_Executor(body), max_async=1)
    await coordinator.start()
    try:
        await _settle(_status_is(store, run_id, "succeeded"))
        first: list[int] = []
        async for event in coordinator.subscribe(owner_id=_OWNER, run_id=run_id):
            first.append(event.sequence)
            if len(first) == 2:
                break
        second = [
            event.sequence
            async for event in coordinator.subscribe(
                owner_id=_OWNER, run_id=run_id, after_sequence=first[-1]
            )
        ]
    finally:
        await coordinator.aclose()

    assert first == [1, 2]
    assert second == [3, 4]
    events = await store.read_event_page(owner_id=_OWNER, run_id=run_id)
    assert [event.event_type for event in events] == ["progress", "token", "token", "done"]


async def test_running_run_observes_cancellation_and_commits_cancelled(
    store: PGAnswerRunStore,
) -> None:
    creation = await store.create_run(owner_id=_OWNER, request=_REQUEST)
    run_id = creation.run.run_id
    started = asyncio.Event()

    async def body(session: RunSession) -> Mapping[str, Any]:
        started.set()
        for _ in range(2000):
            await session.check_cancelled()
            await asyncio.sleep(0.01)
        return {"answer": "unreachable"}

    coordinator = AnswerRunCoordinator(
        store=store, executor=_Executor(body), max_async=1, heartbeat_seconds=0.05
    )
    await coordinator.start()
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        outcome = await store.request_cancellation(owner_id=_OWNER, run_id=run_id)
        assert outcome.outcome == "pending"
        await _settle(_status_is(store, run_id, "cancelled"))
    finally:
        await coordinator.aclose()

    events = await store.read_event_page(owner_id=_OWNER, run_id=run_id)
    assert events[-1].event_type == "done"
    assert events[-1].payload == {"status": "cancelled"}


async def test_checkpoint_round_trips_through_jsonb(store: PGAnswerRunStore) -> None:
    creation = await store.create_run(owner_id=_OWNER, request=_REQUEST)
    run_id = creation.run.run_id
    claimed = await store.claim_next(worker_id="worker-1")
    assert claimed is not None

    evidence = EvidenceLedger()
    evidence.add_contexts({"chunks": [{"chunk_id": "c1", "content": "text", "_workspace": "ws"}]})
    episode = RunEpisode()
    episode.record([{"role": "assistant", "content": "", "provider_state": {"native": True}}])
    registry = ResourceRegistry()
    registry.register(ResourceInput(content=b"bytes", filename="a.txt"))
    state = AgentRunState(
        evidence=evidence,
        episode=episode,
        tool_cache=_ToolCallCache(),
        registry=registry,
        trace={"agent_turns": 1},
        completed_turns=1,
    )
    envelope = await encode_checkpoint_state(state, owner_id=_OWNER, run_id=run_id, store=store)
    commit = await store.commit_checkpoint(
        owner_id=_OWNER,
        run_id=run_id,
        worker_id="worker-1",
        fencing_epoch=claimed.run.fencing_epoch,
        expected_completed_turns=0,
        version=int(envelope["version"]),
        state=envelope["state"],
    )
    assert commit.outcome == "committed"

    await store.release_for_shutdown(
        owner_id=_OWNER,
        run_id=run_id,
        worker_id="worker-1",
        fencing_epoch=claimed.run.fencing_epoch,
    )
    reclaimed = await store.claim_next(worker_id="worker-2")
    assert reclaimed is not None
    assert reclaimed.checkpoint is not None

    resumed_registry = ResourceRegistry()
    resumed_registry.register(ResourceInput(content=b"bytes", filename="a.txt"))
    resumed = AgentRunState(
        evidence=EvidenceLedger(),
        episode=RunEpisode(),
        tool_cache=_ToolCallCache(),
        registry=resumed_registry,
        trace={},
    )
    await restore_agent_state(
        resumed,
        {
            "version": reclaimed.checkpoint.version,
            "completed_turns": reclaimed.checkpoint.completed_turns,
            "state": reclaimed.checkpoint.state,
        },
        owner_id=_OWNER,
        run_id=run_id,
        store=store,
        expected_completed_turns=1,
    )

    assert resumed.completed_turns == 1
    assert resumed.episode.messages() == episode.messages()
    assert resumed.evidence.contexts["chunks"][0]["chunk_id"] == "c1"
    assert [entry.resource_id for entry in resumed_registry.manifest()] == [
        entry.resource_id for entry in registry.manifest()
    ]
    await state.tool_cache.aclose()
    await resumed.tool_cache.aclose()
    await registry.aclose()
    await resumed_registry.aclose()
