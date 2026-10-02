# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Queue a Workspace's promotion the one way the product does.

A Corpus Mutation Run that settles an ingestion window carrying the Workspace's
counters over the promotion threshold queues the promotion job in the same
statement (``PGRunStore.record_corpus_window``). Suites that start from a queued
promotion get there through that path, never by writing the job themselves.
"""

import uuid
from typing import Any

from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.engine.runtime.records import PreparedRunEnvelope, RunAccessScope


async def queue_promotion(
    workspace: str,
    *,
    docs: int = 1,
    chunks: int = 1,
    pool: Any | None = None,
) -> None:
    """Accept one ingest Run for ``workspace`` and settle a window that crosses the threshold.

    The Workspace's registry row must exist. While it is shared with no promotion in
    flight, the window queues its promotion; otherwise it only adds to the counters.
    """
    store = PGRunStore(pool=pool, promotion_doc_threshold=1)
    await store.initialize()
    run_id = str(uuid.uuid4())
    accepted = await store.accept_run(
        envelope=PreparedRunEnvelope(
            run_kind="corpus_mutation",
            lane="corpus_mutation",
            submitted_by="corpus-operator",
            access_scope=RunAccessScope(kind="workspace", scope_id=workspace),
            submission_key=run_id,
            request_fingerprint=f"fingerprint-{run_id}",
            payload={
                "action": "ingest",
                "workspace": workspace,
                "staged_sources": [],
                "track_id": f"dlightrag-corpus-{run_id}",
            },
            accepted_input={"action": "ingest", "workspace": workspace},
            retention_seconds=7 * 24 * 3600,
        ),
        run_id=run_id,
    )
    await store.record_corpus_window(
        run_id=accepted.run.run_id,
        workspace=workspace,
        window_number=1,
        docs=docs,
        chunks=chunks,
    )
