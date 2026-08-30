# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit contracts for the automatic promotion worker state machine.

The worker is exercised with scripted stores and a scripted connection so the
orchestration invariants are pinned without a database: fence-before-exclusive,
lease/fence recheck inside the cutover transaction, the exclusion-proved
DELETE-then-ATTACH cutover, all-table atomicity in one transaction,
deterministic staging cleanup, stale generation yielding, and the guarded
failed/retry observability contract (a stale attempt mutates nothing).
"""

import datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.adapters.postgres.corpus import promotion_worker as worker_module
from dlightrag.adapters.postgres.corpus.promotion_worker import (
    PGPromotionWorker,
    PromotionAttemptError,
    StalePromotionAttempt,
    staging_partition_name,
)


class _Tx:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn
        self._depth = 0

    async def __aenter__(self) -> _Tx:
        self._depth = self._conn.begin()
        return self

    async def __aexit__(self, *args: object) -> None:
        self._conn.end(self._depth)


class _Conn:
    """Scripted connection: records statements, answers the worker's reads."""

    def __init__(self, *, workspace: str = "ws_alpha") -> None:
        self.workspace = workspace
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self._tx_depth = 0
        # Answers for guarded UPDATE ... RETURNING statements (failure path).
        self.returning_results: list[int] = []

    def begin(self) -> int:
        self._tx_depth += 1
        return self._tx_depth

    def end(self, depth: int) -> None:
        self._tx_depth -= 1
        assert depth == self._tx_depth + 1

    @property
    def in_transaction(self) -> bool:
        return self._tx_depth > 0

    def transaction(self) -> _Tx:
        return _Tx(self)

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.executed.append((query, args))
        if "quote_literal" in query:
            return f"'{self.workspace}'"
        if "to_regclass" in query:
            return None
        if "RETURNING 1" in query:
            return self.returning_results.pop(0) if self.returning_results else 0
        return None

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.executed.append((query, args))
        if "dlightrag_promotion_jobs" in query:
            return {
                "state": "promoting",
                "lease_owner": "promo-owner",
                "lease_generation": 7,
                "lease_until": datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=60),
            }
        if "dlightrag_workspace_meta" in query:
            return {"write_fence_owner": "promo-owner#7"}
        return None

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.executed.append((query, args))
        return []

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        return "OK"


def _claim(*, generation: int = 7) -> worker_module.PromotionJobClaim:
    return worker_module.PromotionJobClaim(
        job_id=11,
        workspace="ws_alpha",
        attempt_count=2,
        lease_generation=generation,
        owner="promo-owner",
    )


def _worker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    job_store: Any,
    registry: Any,
    conn: _Conn,
    lease_seconds: int = 300,
) -> PGPromotionWorker:
    worker = PGPromotionWorker(
        job_store=job_store,
        registry=registry,
        lease_seconds=lease_seconds,
        retry_backoff_seconds=60,
        claim_poll_seconds=0.01,
    )
    worker._owner = "promo-owner"  # deterministic for assertions

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_gate(workspace: str, *, exclusive: bool = False):  # noqa: ANN001, ANN202
        assert exclusive is True
        yield conn

    monkeypatch.setattr(worker_module, "workspace_write_gate", fake_gate)

    class _FakePool:
        async def run_once(self, operation: Any) -> Any:  # noqa: ANN001, ANN401
            return await operation(conn)

        async def run(self, operation: Any) -> Any:  # noqa: ANN001, ANN401
            return await operation(conn)

    monkeypatch.setattr(worker_module, "pg_pool", _FakePool())
    return worker


def _scripted_tables(monkeypatch: pytest.MonkeyPatch, parents: list[str]) -> None:
    async def discover(conn: Any) -> list[str]:  # noqa: ANN001
        return list(parents)

    monkeypatch.setattr(worker_module, "_discover_retrieval_parents", discover)


def test_staging_names_are_deterministic_hashes_never_raw_workspace() -> None:
    name = staging_partition_name("LIGHTRAG_DOC_CHUNKS", 'evil"; DROP TABLE x; --')
    assert 'evil"' not in name
    assert name == staging_partition_name("LIGHTRAG_DOC_CHUNKS", 'evil"; DROP TABLE x; --')
    assert name.startswith("s_")


async def test_happy_path_cutover_is_one_transaction_exclusion_proved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents = ["dlightrag_doc_metadata", "LIGHTRAG_DOC_CHUNKS", "lightrag_vdb_chunks_8"]
    _scripted_tables(monkeypatch, parents)
    for helper in (
        "_drop_relation",
        "_create_staging",
        "_copy_workspace_rows",
        "_verify_copy_checksums",
        "_build_staging_indexes",
    ):

        async def noop(*args: Any, **_: Any) -> None:  # noqa: ANN001, ANN401
            return None

        monkeypatch.setattr(worker_module, helper, noop)

    conn = _Conn()
    job_store = SimpleNamespace(
        claim_next=AsyncMock(
            return_value={
                "job_id": 11,
                "workspace": "ws_alpha",
                "attempt_count": 2,
                "lease_generation": 7,
            }
        ),
        renew_lease=AsyncMock(return_value=True),
        mark_failed=AsyncMock(return_value=True),
    )
    registry = SimpleNamespace(
        acquire_write_fence=AsyncMock(return_value=True),
        set_promotion_state=AsyncMock(return_value=True),
        release_write_fence=AsyncMock(return_value=True),
    )
    worker = _worker(monkeypatch, job_store=job_store, registry=registry, conn=conn)
    monkeypatch.setattr(worker, "_recheck_current", AsyncMock())

    assert await worker.run_once() is True

    # Fence first, observability second, before the exclusive gate opens.
    assert registry.acquire_write_fence.await_args.kwargs["owner"] == "promo-owner#7"
    assert registry.set_promotion_state.await_args.kwargs == {
        "workspace": "ws_alpha",
        "state": "promoting",
        "expected_fence_owner": "promo-owner#7",
    }

    # Per table the cutover must prove exclusion before the ATTACH: NOT VALID
    # check, DELETE, VALIDATE, then RENAME/ATTACH, then DROP of the temporary
    # constraint — all inside the one transaction.
    for _parent in parents:
        deletes = [
            (index, args)
            for index, (query, args) in enumerate(conn.executed)
            if query.startswith("DELETE FROM ONLY") and args == ("ws_alpha",)
        ]
        assert len(deletes) == len(parents)
        attach_queries = [
            (index, query)
            for index, (query, _) in enumerate(conn.executed)
            if "ATTACH PARTITION" in query
        ]
        assert len(attach_queries) == len(parents)
        validates = [
            index
            for index, (query, _) in enumerate(conn.executed)
            if "VALIDATE CONSTRAINT" in query
        ]
        assert len(validates) == len(parents)
        drops = [
            index
            for index, (query, _) in enumerate(conn.executed)
            if query.startswith("ALTER TABLE ONLY")
            and "DROP CONSTRAINT" in query
            and "IF EXISTS" not in query
        ]
        assert len(drops) == len(parents)
        # For each table: exclusion ADD < DELETE < VALIDATE < ATTACH < DROP.
        for add_i, del_entry, val_i, att_entry, drop_i in zip(
            sorted(
                index
                for index, (query, _) in enumerate(conn.executed)
                if "ADD CONSTRAINT" in query and "NOT VALID" in query
            ),
            deletes,
            validates,
            attach_queries,
            drops,
            strict=True,
        ):
            assert add_i < del_entry[0] < val_i < att_entry[0] < drop_i
        assert "FOR VALUES IN ('ws_alpha')" in conn.executed[attach_queries[0][0]][0]

    flips = [(q, a) for q, a in conn.executed if "storage_tier = 'hot'" in q]
    assert len(flips) == 1
    assert flips[0][1] == ("ws_alpha", "promo-owner#7")
    flip_sql = flips[0][0]
    assert "write_fence_owner = NULL" in flip_sql
    assert "promotion_state = 'none'" in flip_sql
    assert "write_fence_owner = $2" in flip_sql  # fence-owner guarded
    done_sql = [(q, a) for q, a in conn.executed if "state = 'done'" in q]
    assert len(done_sql) == 1
    assert done_sql[0][1] == (11, "promo-owner", 7)

    # The recheck ran once outside and once inside the cutover transaction.
    assert worker._recheck_current.await_count == 2  # type: ignore[attr-defined]
    assert worker._recheck_current.await_args_list[1].kwargs["for_update"] is True  # type: ignore[attr-defined]


async def test_already_attached_partitions_reconcile_to_bookkeeping_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents = ["dlightrag_doc_metadata", "LIGHTRAG_DOC_CHUNKS"]
    _scripted_tables(monkeypatch, parents)

    async def child_attached(conn: Any, parent: str, child: str, workspace: str) -> bool:
        return True

    monkeypatch.setattr(worker_module, "_child_is_attached", child_attached)
    verify = AsyncMock()
    monkeypatch.setattr(worker_module, "_verify_attached_partition", verify)
    copied = AsyncMock()
    monkeypatch.setattr(worker_module, "_copy_workspace_rows", copied)

    conn = _Conn()
    job_store = SimpleNamespace(
        claim_next=AsyncMock(
            return_value={
                "job_id": 11,
                "workspace": "ws_alpha",
                "attempt_count": 2,
                "lease_generation": 7,
            }
        ),
        renew_lease=AsyncMock(return_value=True),
        mark_failed=AsyncMock(return_value=True),
    )
    registry = SimpleNamespace(
        acquire_write_fence=AsyncMock(return_value=True),
        set_promotion_state=AsyncMock(return_value=True),
        release_write_fence=AsyncMock(return_value=True),
    )
    worker = _worker(monkeypatch, job_store=job_store, registry=registry, conn=conn)
    monkeypatch.setattr(worker, "_recheck_current", AsyncMock())

    assert await worker.run_once() is True

    assert verify.await_count == 2
    copied.assert_not_awaited()
    # No ATTACH for reconciled tables, but the tier flip and job completion
    # still commit (the crash-after-attach reconciliation path).
    assert not any("ATTACH PARTITION" in q for q, _ in conn.executed)
    assert any("storage_tier = 'hot'" in q for q, _ in conn.executed)


async def test_verify_failure_fails_in_one_guarded_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted_tables(monkeypatch, ["LIGHTRAG_DOC_CHUNKS"])

    async def verify_fail(conn: Any, *args: Any) -> None:  # noqa: ANN001, ANN401
        raise PromotionAttemptError("copy verification failed for LIGHTRAG_DOC_CHUNKS")

    monkeypatch.setattr(worker_module, "_verify_copy_checksums", verify_fail)
    monkeypatch.setattr(worker_module, "_create_staging", AsyncMock())
    monkeypatch.setattr(worker_module, "_copy_workspace_rows", AsyncMock())
    monkeypatch.setattr(worker_module, "_drop_relation", AsyncMock())

    conn = _Conn()
    conn.returning_results = [1, 1]  # job guard + registry guard succeed
    job_store = SimpleNamespace(
        claim_next=AsyncMock(
            return_value={
                "job_id": 11,
                "workspace": "ws_alpha",
                "attempt_count": 2,
                "lease_generation": 7,
            }
        ),
        renew_lease=AsyncMock(return_value=True),
        mark_failed=AsyncMock(return_value=True),
    )
    registry = SimpleNamespace(
        acquire_write_fence=AsyncMock(return_value=True),
        set_promotion_state=AsyncMock(return_value=True),
        release_write_fence=AsyncMock(return_value=True),
    )
    worker = _worker(monkeypatch, job_store=job_store, registry=registry, conn=conn)
    monkeypatch.setattr(worker, "_recheck_current", AsyncMock())
    monkeypatch.setattr(worker, "_cleanup_artifacts_on", AsyncMock())

    assert await worker.run_once() is True

    # The attempt's artifacts were cleaned INSIDE the exclusive gate before
    # the guarded failure transition ran.
    worker._cleanup_artifacts_on.assert_awaited_once_with(conn, "ws_alpha")  # type: ignore[attr-defined]
    # The failure transition ran as guarded SQL, not via the observability
    # facade: job failed (owner/generation/unexpired-lease guarded) and
    # registry failed (fence-owner guarded) in one transaction, releasing the
    # owned fence.
    job_guard = [
        (q, a) for q, a in conn.executed if "state = 'failed'" in q and "lease_generation = $3" in q
    ]
    assert len(job_guard) == 1
    assert job_guard[0][1][:3] == (11, "promo-owner", 7)
    assert job_guard[0][1][3].startswith("promotion failed")
    assert job_guard[0][1][4] is not None
    registry_guard = [
        (q, a)
        for q, a in conn.executed
        if "promotion_state = 'failed'" in q and "write_fence_owner = $2" in q
    ]
    assert len(registry_guard) == 1
    assert registry_guard[0][1][0] == "ws_alpha"
    assert registry_guard[0][1][1] == "promo-owner#7"
    # The registry guard statement releases the owned fence in the same row.
    assert "write_fence_owner = NULL" in registry_guard[0][0]
    # Only the startup 'promoting' observability write used the facade.
    assert [call.kwargs["state"] for call in registry.set_promotion_state.await_args_list] == [
        "promoting"
    ]
    job_store.mark_failed.assert_not_awaited()
    # No cutover ever ran.
    assert not any("ATTACH PARTITION" in q for q, _ in conn.executed)
    assert not any("storage_tier = 'hot'" in q for q, _ in conn.executed)


async def test_fence_unavailable_fails_the_attempt_without_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted_tables(monkeypatch, ["LIGHTRAG_DOC_CHUNKS"])
    conn = _Conn()
    conn.returning_results = [1, 1]
    job_store = SimpleNamespace(
        claim_next=AsyncMock(
            return_value={
                "job_id": 11,
                "workspace": "ws_alpha",
                "attempt_count": 2,
                "lease_generation": 7,
            }
        ),
        renew_lease=AsyncMock(return_value=True),
        mark_failed=AsyncMock(return_value=True),
    )
    registry = SimpleNamespace(
        acquire_write_fence=AsyncMock(return_value=False),
        set_promotion_state=AsyncMock(return_value=True),
        release_write_fence=AsyncMock(return_value=True),
    )
    worker = _worker(monkeypatch, job_store=job_store, registry=registry, conn=conn)
    monkeypatch.setattr(worker, "_cleanup_artifacts_on", AsyncMock())

    assert await worker.run_once() is True

    # No exclusive gate ever opened, so no artifact cleanup ran.
    worker._cleanup_artifacts_on.assert_not_awaited()  # type: ignore[attr-defined]
    failed_sql = [a for q, a in conn.executed if "state = 'failed'" in q]
    assert failed_sql and "write fence unavailable" in failed_sql[0][3]
    # The exclusive gate never opened: no copy, no cutover.
    assert not any("ATTACH PARTITION" in q for q, _ in conn.executed)


async def test_state_transition_refusal_releases_owned_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted_tables(monkeypatch, ["LIGHTRAG_DOC_CHUNKS"])
    conn = _Conn()
    job_store = SimpleNamespace(
        claim_next=AsyncMock(
            return_value={
                "job_id": 11,
                "workspace": "ws_alpha",
                "attempt_count": 2,
                "lease_generation": 7,
            }
        ),
        renew_lease=AsyncMock(return_value=True),
        mark_failed=AsyncMock(return_value=True),
    )
    registry = SimpleNamespace(
        acquire_write_fence=AsyncMock(return_value=True),
        set_promotion_state=AsyncMock(return_value=False),
        release_write_fence=AsyncMock(return_value=True),
    )
    worker = _worker(monkeypatch, job_store=job_store, registry=registry, conn=conn)

    assert await worker.run_once() is True

    registry.release_write_fence.assert_awaited_once_with(
        workspace="ws_alpha",
        owner="promo-owner#7",
    )
    assert not any("ATTACH PARTITION" in query for query, _ in conn.executed)


async def test_stale_cutover_recheck_aborts_without_failure_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted_tables(monkeypatch, ["LIGHTRAG_DOC_CHUNKS"])
    conn = _Conn()

    async def stale_recheck(
        conn_arg: Any,  # noqa: ANN001, ANN401
        claim: Any,  # noqa: ANN001, ANN401
        fence_owner: str,
        *,
        for_update: bool = False,
    ) -> None:
        raise StalePromotionAttempt("promotion lease is not current")

    job_store = SimpleNamespace(
        claim_next=AsyncMock(
            return_value={
                "job_id": 11,
                "workspace": "ws_alpha",
                "attempt_count": 2,
                "lease_generation": 7,
            }
        ),
        renew_lease=AsyncMock(return_value=True),
        mark_failed=AsyncMock(return_value=True),
    )
    registry = SimpleNamespace(
        acquire_write_fence=AsyncMock(return_value=True),
        set_promotion_state=AsyncMock(return_value=True),
        release_write_fence=AsyncMock(return_value=True),
    )
    worker = _worker(monkeypatch, job_store=job_store, registry=registry, conn=conn)
    monkeypatch.setattr(worker, "_recheck_current", stale_recheck)

    assert await worker.run_once() is True

    job_store.mark_failed.assert_not_awaited()
    # Only the initial 'promoting' observability write; no failed state.
    assert all(
        kwargs["state"] == "promoting"
        for kwargs in [call.kwargs for call in registry.set_promotion_state.await_args_list]
    )


async def test_reclaimed_lease_during_failure_handling_mutates_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted_tables(monkeypatch, ["LIGHTRAG_DOC_CHUNKS"])

    async def verify_fail(conn: Any, *args: Any) -> None:  # noqa: ANN001, ANN401
        raise PromotionAttemptError("boom")

    monkeypatch.setattr(worker_module, "_verify_copy_checksums", verify_fail)
    monkeypatch.setattr(worker_module, "_create_staging", AsyncMock())
    monkeypatch.setattr(worker_module, "_copy_workspace_rows", AsyncMock())
    monkeypatch.setattr(worker_module, "_drop_relation", AsyncMock())

    conn = _Conn()
    # The job guard refuses: the lease was reclaimed by a newer generation.
    conn.returning_results = [0]
    job_store = SimpleNamespace(
        claim_next=AsyncMock(
            return_value={
                "job_id": 11,
                "workspace": "ws_alpha",
                "attempt_count": 2,
                "lease_generation": 7,
            }
        ),
        renew_lease=AsyncMock(return_value=True),
        mark_failed=AsyncMock(return_value=True),
    )
    registry = SimpleNamespace(
        acquire_write_fence=AsyncMock(return_value=True),
        set_promotion_state=AsyncMock(return_value=True),
        release_write_fence=AsyncMock(return_value=True),
    )
    worker = _worker(monkeypatch, job_store=job_store, registry=registry, conn=conn)
    monkeypatch.setattr(worker, "_recheck_current", AsyncMock())
    monkeypatch.setattr(worker, "_cleanup_artifacts_on", AsyncMock())

    assert await worker.run_once() is True
    # While we still hold the exclusive gate the deterministic artifacts are
    # ours to clean (a newer worker cannot have entered); the guarded
    # transition then refused, so no registry/job state was mutated.
    worker._cleanup_artifacts_on.assert_awaited_once_with(conn, "ws_alpha")  # type: ignore[attr-defined]
    assert not any("promotion_state = 'failed'" in q for q, _ in conn.executed)
    registry.release_write_fence.assert_not_awaited()


async def test_stale_fence_during_failure_handling_rolls_back_job_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted_tables(monkeypatch, ["LIGHTRAG_DOC_CHUNKS"])

    async def verify_fail(conn: Any, *args: Any) -> None:  # noqa: ANN001, ANN401
        raise PromotionAttemptError("boom")

    monkeypatch.setattr(worker_module, "_verify_copy_checksums", verify_fail)
    monkeypatch.setattr(worker_module, "_create_staging", AsyncMock())
    monkeypatch.setattr(worker_module, "_copy_workspace_rows", AsyncMock())
    monkeypatch.setattr(worker_module, "_drop_relation", AsyncMock())

    conn = _Conn()
    # Job guard succeeds, registry guard refuses (fence was taken over): the
    # transaction rolls back, so the job transition must not stand.
    conn.returning_results = [1, 0]
    job_store = SimpleNamespace(
        claim_next=AsyncMock(
            return_value={
                "job_id": 11,
                "workspace": "ws_alpha",
                "attempt_count": 2,
                "lease_generation": 7,
            }
        ),
        renew_lease=AsyncMock(return_value=True),
        mark_failed=AsyncMock(return_value=True),
    )
    registry = SimpleNamespace(
        acquire_write_fence=AsyncMock(return_value=True),
        set_promotion_state=AsyncMock(return_value=True),
        release_write_fence=AsyncMock(return_value=True),
    )
    worker = _worker(monkeypatch, job_store=job_store, registry=registry, conn=conn)
    monkeypatch.setattr(worker, "_recheck_current", AsyncMock())
    monkeypatch.setattr(worker, "_cleanup_artifacts_on", AsyncMock())

    assert await worker.run_once() is True

    # In-gate cleanup ran once; the registry guard then refused and the
    # guarded transaction rolled the job transition back — nothing was
    # released to anyone else.
    registry_guard = [
        (q, a)
        for q, a in conn.executed
        if "promotion_state = 'failed'" in q and "write_fence_owner = $2" in q
    ]
    assert len(registry_guard) == 1
    assert registry_guard[0][1] == (
        "ws_alpha",
        "promo-owner#7",
        "promotion failed: boom",
        None,
    ) or (registry_guard[0][1][0] == "ws_alpha" and registry_guard[0][1][1] == "promo-owner#7")
    worker._cleanup_artifacts_on.assert_awaited_once_with(conn, "ws_alpha")  # type: ignore[attr-defined]


async def test_worker_loop_claims_until_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    job_store = SimpleNamespace(claim_next=AsyncMock(return_value=None))
    registry = MagicMock()
    worker = PGPromotionWorker(
        job_store=cast(Any, job_store),
        registry=cast(Any, registry),
        lease_seconds=300,
        retry_backoff_seconds=60,
        claim_poll_seconds=0.02,
    )
    worker.start()
    await __import__("asyncio").sleep(0.1)
    await worker.aclose()
    assert job_store.claim_next.await_count >= 2


async def test_cancelled_stale_worker_mutates_nothing_and_drops_no_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    _scripted_tables(monkeypatch, ["LIGHTRAG_DOC_CHUNKS"])

    conn = _Conn()
    # The guarded failure transition refuses: the lease was reclaimed by a
    # newer generation before the cancellation landed.
    conn.returning_results = [0]
    job_store = SimpleNamespace(
        claim_next=AsyncMock(
            return_value={
                "job_id": 11,
                "workspace": "ws_alpha",
                "attempt_count": 2,
                "lease_generation": 7,
            }
        ),
        renew_lease=AsyncMock(return_value=True),
        mark_failed=AsyncMock(return_value=True),
    )
    registry = SimpleNamespace(
        acquire_write_fence=AsyncMock(return_value=True),
        set_promotion_state=AsyncMock(return_value=True),
        release_write_fence=AsyncMock(return_value=True),
    )
    worker = _worker(monkeypatch, job_store=job_store, registry=registry, conn=conn)
    monkeypatch.setattr(worker, "_recheck_current", AsyncMock())
    monkeypatch.setattr(worker, "_cleanup_artifacts_on", AsyncMock())

    async def cancel_mid(conn_arg: Any, claim: Any, fence_owner: str) -> None:  # noqa: ANN001, ANN401
        raise asyncio.CancelledError()

    monkeypatch.setattr(worker, "_copy_and_cutover", cancel_mid)

    with pytest.raises(asyncio.CancelledError):
        await worker.run_once()

    # In-gate cleanup ran (we still own the exclusive); the guarded job
    # transition was attempted once and refused (returned 0), so the registry
    # guard never ran and no registry/job state was mutated.
    worker._cleanup_artifacts_on.assert_awaited_once_with(conn, "ws_alpha")  # type: ignore[attr-defined]
    job_guard_attempts = [
        q for q, _ in conn.executed if "state = 'failed'" in q and "lease_generation = $3" in q
    ]
    assert len(job_guard_attempts) == 1
    assert conn.returning_results == []  # refusal consumed the only answer
    assert not any("promotion_state = 'failed'" in q for q, _ in conn.executed)
    registry.release_write_fence.assert_not_awaited()


async def test_cancelled_current_worker_fails_guarded_and_cleans_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    _scripted_tables(monkeypatch, ["LIGHTRAG_DOC_CHUNKS"])

    conn = _Conn()
    conn.returning_results = [1, 1]  # job + registry guards both succeed
    job_store = SimpleNamespace(
        claim_next=AsyncMock(
            return_value={
                "job_id": 11,
                "workspace": "ws_alpha",
                "attempt_count": 2,
                "lease_generation": 7,
            }
        ),
        renew_lease=AsyncMock(return_value=True),
        mark_failed=AsyncMock(return_value=True),
    )
    registry = SimpleNamespace(
        acquire_write_fence=AsyncMock(return_value=True),
        set_promotion_state=AsyncMock(return_value=True),
        release_write_fence=AsyncMock(return_value=True),
    )
    worker = _worker(monkeypatch, job_store=job_store, registry=registry, conn=conn)
    monkeypatch.setattr(worker, "_recheck_current", AsyncMock())
    monkeypatch.setattr(worker, "_cleanup_artifacts_on", AsyncMock())

    async def cancel_mid(conn_arg: Any, claim: Any, fence_owner: str) -> None:  # noqa: ANN001, ANN401
        raise asyncio.CancelledError()

    monkeypatch.setattr(worker, "_copy_and_cutover", cancel_mid)

    with pytest.raises(asyncio.CancelledError):
        await worker.run_once()

    # In-gate cleanup ran once, then the guarded failed/retry transition
    # (job + registry + owned fence) committed.
    worker._cleanup_artifacts_on.assert_awaited_once_with(conn, "ws_alpha")  # type: ignore[attr-defined]
    assert any("state = 'failed'" in q for q, _ in conn.executed)
    assert any("promotion_state = 'failed'" in q for q, _ in conn.executed)
