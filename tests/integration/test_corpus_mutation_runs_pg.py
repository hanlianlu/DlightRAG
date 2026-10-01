# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL recovery and Workspace ordering for Corpus Mutation Runs."""

import datetime
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.corpus.promotion_jobs import PGPromotionJobStore
from dlightrag.adapters.postgres.corpus.workspaces import PGWorkspaceRegistry
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.engine.runtime.records import (
    PreparedRunEnvelope,
    RunAccessScope,
)
from tests.support.pg import PG_CONN_KWARGS, drop_database, skip_without_postgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@asynccontextmanager
async def _scratch_pool() -> AsyncIterator[asyncpg.Pool]:
    await skip_without_postgres()
    database = f"dlightrag_corpus_run_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{database}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(
        **{**PG_CONN_KWARGS, "database": database}, min_size=1, max_size=8
    )
    try:
        yield pool
    finally:
        await pool.close()
        await drop_database(database)


@pytest.fixture
async def corpus_run_pg() -> AsyncIterator[PGRunStore]:
    async with _scratch_pool() as pool:
        store = PGRunStore(pool=pool)
        await store.initialize()
        yield store


def _envelope(
    workspace: str,
    *,
    key: str,
    action: str = "delete",
    supersedes_run_id: str | None = None,
) -> PreparedRunEnvelope:
    payload: dict[str, Any] = {
        "action": action,
        "workspace": workspace,
        "track_id": f"dlightrag-corpus-{uuid.uuid4()}",
        **(
            {"supersedes_run_id": supersedes_run_id}
            if action == "reset"
            else {
                "file_paths": [],
                "filenames": [],
                "document_ids": [f"doc-{key}"],
            }
        ),
    }
    return PreparedRunEnvelope(
        run_kind="corpus_mutation",
        lane="corpus_mutation",
        submitted_by="corpus-operator",
        access_scope=RunAccessScope(kind="workspace", scope_id=workspace),
        submission_key=key,
        request_fingerprint=f"fingerprint-{key}",
        payload=payload,
        accepted_input={"action": action, "workspace": workspace},
        retention_seconds=7 * 24 * 3600,
        supersedes_run_id=supersedes_run_id,
    )


async def _accept(store: PGRunStore, workspace: str, *, key: str):
    return await store.accept_run(
        envelope=_envelope(workspace, key=key),
        run_id=str(uuid.uuid4()),
    )


async def test_corpus_repair_resume_is_explicit_and_keeps_the_workspace_barrier(
    corpus_run_pg: PGRunStore,
) -> None:
    first = await _accept(corpus_run_pg, "alpha", key="alpha-first")
    second = await _accept(corpus_run_pg, "alpha", key="alpha-second")
    other = await _accept(corpus_run_pg, "beta", key="beta-first")

    first_claim = await corpus_run_pg.claim_next(
        worker_id="corpus-worker-1",
        run_kinds=("corpus_mutation",),
        lanes=("corpus_mutation",),
    )
    assert first_claim is not None
    assert first_claim.run.run_id == first.run.run_id
    assert await corpus_run_pg.start_handoff(
        owner_id="alpha",
        run_id=first.run.run_id,
        worker_id="corpus-worker-1",
        fencing_epoch=first_claim.run.fencing_epoch,
        checkpoint={"phase": "handoff_started"},
    )
    cancellation = await corpus_run_pg.request_cancellation(
        owner_id="alpha", run_id=first.run.run_id
    )
    assert cancellation.outcome == "rejected"

    assert await corpus_run_pg.wait_for_repair(
        owner_id="alpha",
        run_id=first.run.run_id,
        worker_id="corpus-worker-1",
        fencing_epoch=first_claim.run.fencing_epoch,
        checkpoint={
            "phase": "waiting_for_repair",
            "repair_reason": "upstream outcome is uncertain",
            "repair_remedy": "inspect and repair, then resume",
        },
    )

    # The later alpha Run remains blocked; another Workspace can still use the lane.
    other_claim = await corpus_run_pg.claim_next(
        worker_id="corpus-worker-2",
        run_kinds=("corpus_mutation",),
        lanes=("corpus_mutation",),
    )
    assert other_claim is not None
    assert other_claim.run.run_id == other.run.run_id
    assert other_claim.run.run_id != second.run.run_id

    assert await corpus_run_pg.resume_repair(owner_id="alpha", run_id=first.run.run_id)
    resumed = await corpus_run_pg.get_run(owner_id="alpha", run_id=first.run.run_id)
    assert resumed is not None
    assert resumed.checkpoint is not None
    assert resumed.checkpoint["repair_resume_confirmed"] is True

    resumed_claim = await corpus_run_pg.claim_next(
        worker_id="corpus-worker-3",
        run_kinds=("corpus_mutation",),
        lanes=("corpus_mutation",),
    )
    assert resumed_claim is not None
    assert resumed_claim.run.run_id == first.run.run_id


async def test_pre_handoff_cancel_is_terminal_once_and_workspace_authorization_is_closed(
    corpus_run_pg: PGRunStore,
) -> None:
    accepted = await _accept(corpus_run_pg, "alpha", key="cancel-before-handoff")

    cancelled = await corpus_run_pg.request_cancellation(
        owner_id="alpha", run_id=accepted.run.run_id
    )

    assert cancelled.outcome == "cancelled"
    assert cancelled.run is not None and cancelled.run.status == "cancelled"
    assert cancelled.run.purge_after is not None
    assert cancelled.run.finished_at is not None
    assert cancelled.run.purge_after - cancelled.run.finished_at == datetime.timedelta(days=7)
    assert await corpus_run_pg.get_run(owner_id="beta", run_id=accepted.run.run_id) is None
    assert await corpus_run_pg.read_event_page(owner_id="beta", run_id=accepted.run.run_id) == ()
    events = await corpus_run_pg.read_event_page(owner_id="alpha", run_id=accepted.run.run_id)
    assert [(event.sequence, event.event_type) for event in events] == [(1, "done")]
    repeated = await corpus_run_pg.request_cancellation(
        owner_id="alpha", run_id=accepted.run.run_id
    )
    assert repeated.outcome == "already_terminal"
    assert (
        len(await corpus_run_pg.read_event_page(owner_id="alpha", run_id=accepted.run.run_id)) == 1
    )


async def test_reset_explicitly_supersedes_only_the_same_workspace_waiting_run(
    corpus_run_pg: PGRunStore,
) -> None:
    waiting = await _accept(corpus_run_pg, "alpha", key="ambiguous-delete")
    claim = await corpus_run_pg.claim_next(
        worker_id="writer",
        run_kinds=("corpus_mutation",),
        lanes=("corpus_mutation",),
    )
    assert claim is not None and claim.run.run_id == waiting.run.run_id
    assert await corpus_run_pg.start_handoff(
        owner_id="alpha",
        run_id=waiting.run.run_id,
        worker_id="writer",
        fencing_epoch=claim.run.fencing_epoch,
        checkpoint={"phase": "handoff_started"},
    )
    assert await corpus_run_pg.wait_for_repair(
        owner_id="alpha",
        run_id=waiting.run.run_id,
        worker_id="writer",
        fencing_epoch=claim.run.fencing_epoch,
        checkpoint={
            "phase": "waiting_for_repair",
            "repair_reason": "outcome uncertain",
            "repair_remedy": "inspect then reset",
        },
    )

    with pytest.raises(ValueError, match="this Workspace"):
        await corpus_run_pg.accept_run(
            envelope=_envelope(
                "beta",
                key="wrong-workspace-reset",
                action="reset",
                supersedes_run_id=waiting.run.run_id,
            ),
            run_id=str(uuid.uuid4()),
        )

    reset = await corpus_run_pg.accept_run(
        envelope=_envelope(
            "alpha",
            key="authorized-reset",
            action="reset",
            supersedes_run_id=waiting.run.run_id,
        ),
        run_id=str(uuid.uuid4()),
    )
    superseded = await corpus_run_pg.get_run(owner_id="alpha", run_id=waiting.run.run_id)
    assert superseded is not None
    assert superseded.status == "failed"
    assert superseded.error_kind == "repair_superseded"
    assert superseded.superseded_by_run_id == reset.run.run_id
    assert superseded.result is not None
    assert superseded.result["superseded_by_run_id"] == reset.run.run_id
    events = await corpus_run_pg.read_event_page(owner_id="alpha", run_id=waiting.run.run_id)
    assert [(event.sequence, event.event_type) for event in events] == [(1, "error")]

    reset_claim = await corpus_run_pg.claim_next(
        worker_id="reset-writer",
        run_kinds=("corpus_mutation",),
        lanes=("corpus_mutation",),
    )
    assert reset_claim is not None and reset_claim.run.run_id == reset.run.run_id


async def test_mutation_windows_count_once_and_queue_promotion_at_the_threshold() -> None:
    async with _scratch_pool() as pool:
        store = PGRunStore(pool=pool, promotion_doc_threshold=5)
        await store.initialize()
        registry = PGWorkspaceRegistry(pool=pool)
        await registry.initialize()
        await PGPromotionJobStore(pool=pool).initialize()
        await registry.upsert(workspace="alpha", display_name="Alpha", embedding_model="embed")
        run_id = (await _accept(store, "alpha", key="alpha-ingest")).run.run_id

        async def window(number: int, *, docs: int, chunks: int) -> bool:
            return await store.record_corpus_window(
                run_id=run_id, workspace="alpha", window_number=number, docs=docs, chunks=chunks
            )

        async def promotion() -> tuple[Any, ...]:
            row = await registry.get_row("alpha")
            assert row is not None
            async with pool.acquire() as conn:
                jobs = await conn.fetch("SELECT state FROM dlightrag_promotion_jobs")
            return (
                row["ingested_docs_total"],
                row["ingested_chunks_total"],
                row["promotion_state"],
                [job["state"] for job in jobs],
            )

        assert await window(1, docs=3, chunks=41)
        # A window replayed after a lost response counts once.
        assert not await window(1, docs=3, chunks=41)
        assert await promotion() == (3, 41, "none", [])

        assert await window(2, docs=2, chunks=9)
        assert await promotion() == (5, 50, "pending", ["pending"])

        # The pending promotion is queued once, however many windows follow.
        assert await window(3, docs=1, chunks=1)
        assert await promotion() == (6, 51, "pending", ["pending"])
