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
import base64
import json
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast
from unittest.mock import MagicMock

import asyncpg
import pytest
from dlightrag_ai.telemetry import NOOP_TELEMETRY

from dlightrag.citations.streaming import AnswerStream
from dlightrag.core.agent.orchestrator import AnswerOrchestrator
from dlightrag.core.answer.synthesizer import AnswerSynthesizer
from dlightrag.core.answer_runs.checkpoints import encode_checkpoint_state, restore_agent_state
from dlightrag.core.answer_runs.coordinator import (
    AnswerRunCoordinator,
    DurableWrites,
    RunSession,
)
from dlightrag.core.answer_runs.execution import AnswerRunInput
from dlightrag.core.answer_runs.models import AgentRunState
from dlightrag.core.answer_runs.subscription import RunEventBroker
from dlightrag.core.memory.conversation import PriorTurns
from dlightrag.core.memory.episode import RunEpisode
from dlightrag.core.memory.evidence import EvidenceLedger
from dlightrag.core.resources import registry as registry_module
from dlightrag.core.resources.models import ResourceInput
from dlightrag.core.resources.registry import ResourceRegistry
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.core.servicemanager import (
    RAGServiceManager,
    _fetched_bytes_sink,
    _OrchestratorRun,
)
from dlightrag.core.tools import ExactCallCache
from dlightrag.storage.answer_runs import PGAnswerRunStore
from dlightrag.storage.web_conversations import PGWebConversationStore

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
_VISUAL_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfake-corpus-visual").decode("ascii")


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
        # Retention exempts conversation-linked runs, so the whole operational
        # schema is established here exactly as a real process establishes it.
        await PGWebConversationStore(pool=pool, run_store=created).initialize()
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


async def test_the_coordinator_applies_retention_without_an_execution_slot(
    store: PGAnswerRunStore,
) -> None:
    """Every run-owning process trims expired event logs and prunes expired runs."""
    expired = await store.create_run(owner_id=_OWNER, request=_REQUEST)
    live = await store.create_run(owner_id=_OWNER, request={**_REQUEST, "query": "recent"})
    for creation in (expired, live):
        claim = await store.claim_next(worker_id="retention-setup")
        assert claim is not None
        await store.finish_success(
            owner_id=_OWNER,
            run_id=claim.run.run_id,
            worker_id="retention-setup",
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": creation.run.run_id},
        )
    async with store._pool.acquire() as conn:  # noqa: SLF001 - backdating is test-only setup
        await conn.execute(
            "UPDATE dlightrag_answer_runs SET finished_at = NOW() - INTERVAL '31 days' "
            "WHERE run_id = $1",
            uuid.UUID(expired.run.run_id),
        )

    held = asyncio.Event()

    async def body(session: RunSession) -> Mapping[str, Any]:
        await held.wait()
        return {"answer": "held"}

    coordinator = AnswerRunCoordinator(
        store=store,
        executor=_Executor(body),
        max_async=1,
        maintenance_seconds=0.05,
    )
    await coordinator.start()
    try:

        async def _pruned() -> bool:
            return await store.get_run(owner_id=_OWNER, run_id=expired.run.run_id) is None

        await _settle(_pruned)
    finally:
        held.set()
        await coordinator.aclose()

    survivor = await store.get_run(owner_id=_OWNER, run_id=live.run.run_id)
    assert survivor is not None
    assert survivor.events_trimmed_at is None


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
        tool_cache=ExactCallCache(),
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
        tool_cache=ExactCallCache(),
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


async def test_restored_fetched_resource_never_refetches_or_rebinds_its_slot(
    store: PGAnswerRunStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durable web bytes survive a restart even when the page no longer resolves."""
    creation = await store.create_run(owner_id=_OWNER, request=_REQUEST)
    run_id = creation.run.run_id
    claimed = await store.claim_next(worker_id="worker-1")
    assert claimed is not None
    session = RunSession(store, claimed, broker=RunEventBroker(), writes=DurableWrites())

    reached: list[str] = []

    async def _validate(url: str, **kwargs: Any) -> None:
        reached.append(url)

    async def _fetch_original(url: str, **kwargs: Any) -> bytes:
        reached.append(url)
        return b"<html>the page as the run first read it</html>"

    monkeypatch.setattr(registry_module, "avalidate_public_https_url", _validate)
    monkeypatch.setattr(registry_module, "afetch_public_https_bytes", _fetch_original)

    registry = ResourceRegistry(fetched_bytes_sink=_fetched_bytes_sink(session, store))
    resource_id = registry.register_discovered_link("https://example.com/a.html")
    assert resource_id is not None
    original = await registry.materialize(resource_id)
    assert original == b"<html>the page as the run first read it</html>"

    state = AgentRunState(
        evidence=EvidenceLedger(),
        episode=RunEpisode(),
        tool_cache=ExactCallCache(),
        registry=registry,
        trace={},
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
    before = await _references_by_resource(store, run_id)

    await store.release_for_shutdown(
        owner_id=_OWNER,
        run_id=run_id,
        worker_id="worker-1",
        fencing_epoch=claimed.run.fencing_epoch,
    )
    reclaimed = await store.claim_next(worker_id="worker-2")
    assert reclaimed is not None
    assert reclaimed.checkpoint is not None
    resumed_session = RunSession(store, reclaimed, broker=RunEventBroker(), writes=DurableWrites())

    async def _dead(url: str, **kwargs: Any) -> Any:
        reached.append(url)
        raise AssertionError(f"a restored run must not reach the network for {url}")

    reached.clear()
    monkeypatch.setattr(registry_module, "avalidate_public_https_url", _dead)
    monkeypatch.setattr(registry_module, "afetch_public_https_bytes", _dead)

    resumed_registry = ResourceRegistry(
        fetched_bytes_sink=_fetched_bytes_sink(resumed_session, store)
    )
    resumed = AgentRunState(
        evidence=EvidenceLedger(),
        episode=RunEpisode(),
        tool_cache=ExactCallCache(),
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

    assert await resumed_registry.materialize(resource_id) == original
    assert reached == []
    assert await _references_by_resource(store, run_id) == before

    monkeypatch.setattr(registry_module, "avalidate_public_https_url", _validate)

    async def _fetch_later(url: str, **kwargs: Any) -> bytes:
        return b"<html>a page discovered after the resume</html>"

    monkeypatch.setattr(registry_module, "afetch_public_https_bytes", _fetch_later)
    later_id = resumed_registry.register_discovered_link("https://example.com/b.html")
    assert later_id is not None
    await resumed_registry.materialize(later_id)

    after = await _references_by_resource(store, run_id)
    assert after[resource_id] == before[resource_id]
    assert after[later_id][1] > after[resource_id][1]

    await state.tool_cache.aclose()
    await resumed.tool_cache.aclose()
    await registry.aclose()
    await resumed_registry.aclose()


async def _references_by_resource(
    store: PGAnswerRunStore, run_id: str
) -> dict[str, tuple[str, int]]:
    references = await store.list_run_artifacts(owner_id=_OWNER, run_id=run_id)
    return {
        reference.resource_id: (reference.digest, reference.ordinal)
        for reference in references
        if reference.reference_kind == "fetched_resource"
    }


async def test_accepted_run_executes_and_stores_a_projected_result_without_a_subscriber(
    store: PGAnswerRunStore,
) -> None:
    """A descriptor-only caller still gets a finished run and a safe canonical result."""
    manager = _answer_manager(store)
    creation = await manager.astart_answer_run(
        owner_id=_OWNER, request=AnswerRunInput(query="why", workspaces=("default",))
    )
    run_id = creation.run.run_id
    try:
        await _settle(_status_is(store, run_id, "succeeded"))
    finally:
        coordinator = manager._answer_coordinator
        manager._answer_coordinator = None
        if coordinator is not None:
            await coordinator.aclose()

    run = await manager.aget_answer_run(owner_id=_OWNER, run_id=run_id)
    assert run is not None
    result = run.result
    assert result is not None
    chunk = result["contexts"]["chunks"][0]
    assert "image_data" not in chunk
    assert "_evidence_key" not in chunk
    assert "source_uri" not in chunk["metadata"]
    assert "source_download_locator" not in chunk["metadata"]

    events = await store.read_event_page(owner_id=_OWNER, run_id=run_id)
    done = events[-1]
    assert done.event_type == "done"
    assert done.payload["status"] == "succeeded"
    assert done.payload["result"] == result

    serialized = json.dumps(dict(result))
    assert _VISUAL_B64 not in serialized
    assert "data:image" not in serialized
    assert "/srv/private/book.pdf" not in serialized
    assert result["sources"][0]["source_uri"] == "corpus://book.pdf"
    assert result["answer_images"][0]["chunk_id"] == "c1"
    assert result["trace"]["retrieval"] == "ok"


def _answer_manager(store: PGAnswerRunStore) -> RAGServiceManager:
    """The manager surface a durable run needs, bound to the throwaway database."""
    config = MagicMock()
    config.max_async = 1
    manager = RAGServiceManager.__new__(RAGServiceManager)
    manager._config = config
    manager._closed = False
    manager._answer_run_store = store
    manager._answer_coordinator = None
    manager._answer_store_lock = asyncio.Lock()
    manager._answer_runtime_lock = asyncio.Lock()
    orchestrator = AnswerOrchestrator(
        synthesizer=cast(AnswerSynthesizer, _CitingSynthesizer()),
        retrieve_knowledge_base=_retrieve_visual,
        telemetry=NOOP_TELEMETRY,
    )

    async def _prepare(turn: Any, **kwargs: Any) -> _OrchestratorRun:
        return _OrchestratorRun(
            orchestrator=orchestrator,
            image_descriptions=[],
            query_images=None,
            history=PriorTurns(),
            current_image_count=0,
            ws_list=["default"],
            registry=None,
        )

    manager._prepare_orchestrated_run = _prepare  # type: ignore[method-assign]
    return manager


class _CitingSynthesizer:
    async def generate_stream(
        self,
        query: str,
        contexts: Any,
        *,
        conversation_history: PriorTurns | None = None,
    ) -> tuple[Any, AsyncIterator[str]]:
        async def _stream() -> AsyncIterator[str]:
            yield "the drawing shows it [1]"

        return contexts, AnswerStream(_stream())


async def _retrieve_visual(query: str) -> RetrievalResult:
    return RetrievalResult(
        contexts={
            "chunks": [
                {
                    "chunk_id": "c1",
                    "content": "evidence",
                    "reference_id": "1",
                    "file_path": "book.pdf",
                    "_workspace": "default",
                    "_evidence_key": "search_knowledge_base:c1",
                    "image_data": _VISUAL_B64,
                    "metadata": {
                        "source_type": "corpus",
                        "source_uri": "corpus://book.pdf",
                        "source_download_locator": "/srv/private/book.pdf",
                        "title": "book.pdf",
                    },
                }
            ],
            "entities": [],
            "relationships": [],
        },
        trace={"retrieval": "ok"},
    )
