# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for the durable Answer coordinator on PostgreSQL 18.

Exercises the coordinator against the real fenced store: claim, durable
journal progress, process restart from the committed journal, gap-free event
replay across a reconnect, observed cancellation, graceful-shutdown requeue,
and the journal's round trip through JSONB.

Every test runs inside a throwaway database created and dropped per test, so the
developer's ``dlightrag`` database is never mutated.

Requires PostgreSQL at localhost:5432 (dlightrag/dlightrag); skipped otherwise.
"""

import asyncio
import base64
import datetime
import json
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

import asyncpg
import pytest
from dlightrag_agent.session.fold import PriorTurns
from dlightrag_agent.session.fold import SessionEpisode as _RunEpisode
from dlightrag_ai.capacity import ModelProfile
from dlightrag_ai.fingerprints import ModelFingerprint
from dlightrag_ai.telemetry import NOOP_TELEMETRY
from dlightrag_rag.retrieval import RetrievalResult

from dlightrag.adapters.postgres.answer_runs import PGAnswerRunStore
from dlightrag.adapters.postgres.web_conversations import PGWebConversationStore
from dlightrag.answer.agent.orchestrator import AnswerOrchestrator
from dlightrag.answer.citations.streaming import AnswerStream
from dlightrag.answer.executor import (
    AnswerExecutor,
    AnswerResourceResolver,
    OrchestratorRun,
)
from dlightrag.answer.resources.models import TextWindowBudget
from dlightrag.answer.runs.execution import AnswerRunInput, PinnedModelProfile
from dlightrag.answer.synthesizer import AnswerSynthesizer
from dlightrag.application import Application, _compose
from dlightrag.config import DlightragConfig, RuntimeConfig
from dlightrag.model_settings import answer_executor_settings, answer_resource_settings
from dlightrag.runtime import (
    RunCoordinator,
    RunSession,
    answer_run_request_fingerprint,
)
from tests.conftest import FingerprintingAnswerRunStore

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
_REQUEST_FINGERPRINT = answer_run_request_fingerprint(_REQUEST)
_VISUAL_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfake-corpus-visual").decode("ascii")


def _episode() -> _RunEpisode:
    return _RunEpisode(retained_tail_tokens=20_000)


def _answer_run_input() -> AnswerRunInput:
    return AnswerRunInput(
        query="why",
        workspaces=("default",),
        pinned_models=tuple(
            PinnedModelProfile(
                role=role,
                fingerprint=ModelFingerprint("openai", f"test-{role}-model", None),
                profile=ModelProfile(context_window_tokens=1_000_000),
            )
            for role in ("extract", "keyword", "query", "vlm")
        ),
        context_policy_revision="m1-v1",
        model_catalog_revision="2026-08-14",
        idempotency_fingerprint="public-request-hash",
    )


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def store() -> AsyncIterator[FingerprintingAnswerRunStore]:
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
        created = FingerprintingAnswerRunStore(pool=pool)
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


async def test_journaled_turn_survives_a_new_worker(store: FingerprintingAnswerRunStore) -> None:
    creation = await store.create_run(
        owner_id=_OWNER,
        request=_REQUEST,
        idempotency_fingerprint=_REQUEST_FINGERPRINT,
    )
    run_id = creation.run.run_id
    seen: list[int] = []

    from dlightrag_agent.session.entries import AssistantMessageEntry
    from dlightrag_agent.session.ids import EntryId, SessionId

    async def body(session: RunSession) -> Mapping[str, Any]:
        journal = session.execution.session_store
        assert session.prepared_input is not None
        session_id = SessionId(str(session.prepared_input["session_id"]))
        snapshot = await journal.load(session_id)
        seen.append(snapshot.version)
        if snapshot.version == 0:
            entry = AssistantMessageEntry(
                entry_id=EntryId.new(),
                session_id=session_id,
                timestamp=datetime.datetime.now(datetime.UTC),
                content="searched",
                stop_reason="stop",
            )
            committed = await journal.append(
                session_id=session_id, expected_version=0, entries=[entry]
            )
            assert committed.__class__.__name__ == "SessionCommit"
            await asyncio.sleep(30)
        return {"answer": "second attempt", "turns": snapshot.version}

    first = RunCoordinator(store=store, executor=_Executor(body), answer_worker_concurrency=1)
    await first.start()
    await _settle(_journal_committed(store, run_id))
    await first.aclose()

    second = RunCoordinator(store=store, executor=_Executor(body), answer_worker_concurrency=1)
    await second.start()
    try:
        await _settle(_status_is(store, run_id, "succeeded"))
    finally:
        await second.aclose()

    run = await store.get_run(owner_id=_OWNER, run_id=run_id)
    assert run is not None
    assert run.durable_progress_version == 1
    assert run.result == {"answer": "second attempt", "turns": 1}
    assert seen == [0, 1]


def _journal_committed(store: PGAnswerRunStore, run_id: str) -> Any:
    async def _check() -> bool:
        run = await store.get_run(owner_id=_OWNER, run_id=run_id)
        return run is not None and run.durable_progress_version == 1

    return _check


async def test_the_coordinator_applies_retention_without_an_execution_slot(
    store: FingerprintingAnswerRunStore,
) -> None:
    """Every run-owning process trims expired event logs and prunes expired runs."""
    expired = await store.create_run(
        owner_id=_OWNER,
        request=_REQUEST,
        idempotency_fingerprint=_REQUEST_FINGERPRINT,
    )
    live_request = {**_REQUEST, "query": "recent"}
    live = await store.create_run(
        owner_id=_OWNER,
        request=live_request,
        idempotency_fingerprint=answer_run_request_fingerprint(live_request),
    )
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
    pool = store._operation_pool  # noqa: SLF001 - backdating is test-only setup
    assert pool is not None
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_answer_runs SET finished_at = NOW() - INTERVAL '31 days' "
            "WHERE run_id = $1",
            uuid.UUID(expired.run.run_id),
        )

    held = asyncio.Event()

    async def body(session: RunSession) -> Mapping[str, Any]:
        await held.wait()
        return {"answer": "held"}

    coordinator = RunCoordinator(
        store=store,
        executor=_Executor(body),
        answer_worker_concurrency=1,
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
    store: FingerprintingAnswerRunStore,
) -> None:
    creation = await store.create_run(
        owner_id=_OWNER,
        request=_REQUEST,
        idempotency_fingerprint=_REQUEST_FINGERPRINT,
    )
    run_id = creation.run.run_id
    running = asyncio.Event()

    async def body(session: RunSession) -> Mapping[str, Any]:
        running.set()
        await asyncio.sleep(30)
        return {"answer": "unreachable"}

    coordinator = RunCoordinator(store=store, executor=_Executor(body), answer_worker_concurrency=1)
    await coordinator.start()
    await asyncio.wait_for(running.wait(), timeout=10)
    await coordinator.aclose()

    run = await store.get_run(owner_id=_OWNER, run_id=run_id)
    assert run is not None
    assert run.status == "queued"
    assert run.reclaims_without_progress == 0
    assert run.lease_owner is None


async def test_reconnecting_subscriber_replays_without_gaps_or_duplicates(
    store: FingerprintingAnswerRunStore,
) -> None:
    creation = await store.create_run(
        owner_id=_OWNER,
        request=_REQUEST,
        idempotency_fingerprint=_REQUEST_FINGERPRINT,
    )
    run_id = creation.run.run_id

    async def body(session: RunSession) -> Mapping[str, Any]:
        await session.enter_phase("generating")
        await session.emit_token("hello ")
        await session.flush_tokens()
        await session.emit_token("world")
        await session.flush_tokens()
        return {"answer": "hello world"}

    coordinator = RunCoordinator(store=store, executor=_Executor(body), answer_worker_concurrency=1)
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
    store: FingerprintingAnswerRunStore,
) -> None:
    creation = await store.create_run(
        owner_id=_OWNER,
        request=_REQUEST,
        idempotency_fingerprint=_REQUEST_FINGERPRINT,
    )
    run_id = creation.run.run_id
    started = asyncio.Event()

    async def body(session: RunSession) -> Mapping[str, Any]:
        started.set()
        for _ in range(2000):
            await session.check_cancelled()
            await asyncio.sleep(0.01)
        return {"answer": "unreachable"}

    coordinator = RunCoordinator(
        store=store, executor=_Executor(body), answer_worker_concurrency=1, heartbeat_seconds=0.05
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


async def test_journal_round_trips_through_jsonb(store: FingerprintingAnswerRunStore) -> None:
    creation = await store.create_run(
        owner_id=_OWNER,
        request=_REQUEST,
        idempotency_fingerprint=_REQUEST_FINGERPRINT,
    )
    run_id = creation.run.run_id
    claimed = await store.claim_next(worker_id="worker-1")
    assert claimed is not None
    journal = claimed.execution.session_store

    from dlightrag_agent.session.entries import AssistantMessageEntry, UserMessageEntry
    from dlightrag_agent.session.fold import fold_entries
    from dlightrag_agent.session.ids import EntryId, SessionId

    assert creation.run.prepared_input is not None
    session_id = SessionId(str(creation.run.prepared_input["session_id"]))
    entries = [
        UserMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=datetime.datetime.now(datetime.UTC),
            content="question",
        ),
        AssistantMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=datetime.datetime.now(datetime.UTC),
            content="searched",
            stop_reason="stop",
            provider_state={"native": True},
        ),
    ]
    committed = await journal.append(session_id=session_id, expected_version=0, entries=entries)
    assert committed.__class__.__name__ == "SessionCommit"

    await store.release_for_shutdown(
        owner_id=_OWNER,
        run_id=run_id,
        worker_id="worker-1",
        fencing_epoch=claimed.run.fencing_epoch,
    )
    reclaimed = await store.claim_next(worker_id="worker-2")
    assert reclaimed is not None

    snapshot = await reclaimed.execution.session_store.load(session_id)
    assert snapshot.version == 1
    folded = fold_entries(snapshot.entries)
    assert [message["role"] for message in folded] == ["user", "assistant"]
    assert folded[1]["provider_state"] == {"native": True}


async def test_accepted_run_executes_and_stores_a_projected_result_without_a_subscriber(
    store: FingerprintingAnswerRunStore,
) -> None:
    """A descriptor-only caller still gets a finished run and a safe canonical result."""
    application, coordinator = _answer_runtime(store)
    await coordinator.start()
    creation = await store.create_run(
        owner_id=_OWNER,
        request=_answer_run_input().as_request(),
        idempotency_fingerprint=_REQUEST_FINGERPRINT,
    )
    run_id = creation.run.run_id
    coordinator.wake()
    try:
        await _settle(_status_is(store, run_id, "succeeded"))
    finally:
        await coordinator.aclose()
        await application.aclose()

    run = await store.get_run(owner_id=_OWNER, run_id=run_id)
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


def _answer_runtime(store: FingerprintingAnswerRunStore) -> tuple[Application, RunCoordinator]:
    """Compose the final executor and coordinator over the throwaway database."""
    config = DlightragConfig(runtime=RuntimeConfig(answer_worker_concurrency=1))
    components = _compose(config)
    application = Application(config, components)
    orchestrator = AnswerOrchestrator(
        synthesizer=cast(AnswerSynthesizer, _CitingSynthesizer()),
        retrieve_knowledge_base=_retrieve_visual,
        model_profile=ModelProfile(context_window_tokens=1_000_000),
        telemetry=NOOP_TELEMETRY,
        text_window_budget=TextWindowBudget(tokens=850_000),
        resolved_mode="fast",
    )

    executor = AnswerExecutor(
        store=store,
        pool=components.pool,
        retrieve=components.retrieval.retrieve_result,
        models=components.models,
        capabilities=components.capabilities,
        resources=AnswerResourceResolver(
            settings=answer_resource_settings(config),
            models=components.models,
            capabilities=components.capabilities,
        ),
        settings=answer_executor_settings(config),
        telemetry=NOOP_TELEMETRY,
    )

    async def _prepare(**kwargs: Any) -> OrchestratorRun:
        return OrchestratorRun(
            orchestrator=orchestrator,
            image_descriptions=[],
            query_images=None,
            history=PriorTurns(),
            current_image_count=0,
            workspaces=["default"],
            registry=None,
        )

    executor.prepare_orchestrated_run = _prepare  # type: ignore[method-assign]
    coordinator = RunCoordinator(
        store=store,
        executor=executor,
        answer_worker_concurrency=1,
    )
    return application, coordinator


class _CitingSynthesizer:
    async def generate_stream(
        self,
        query: str,
        contexts: Any,
        *,
        conversation_history: PriorTurns | None = None,
        memory_text: str = "",
    ) -> tuple[Any, AsyncIterator[str]]:
        del memory_text

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
