# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit contracts for the automatic promotion worker state machine.

The worker runs with scripted stores and a scripted connection, which pin the
orchestration a database cannot easily stage: renewal loss aborting the copy, a
refused state transition releasing its fence, and stale or cancelled attempts
yielding without releasing what they do not own. The cutover, its atomicity,
staging cleanup and the guarded failed/retry transition run against PostgreSQL
in tests/integration/test_promotion_worker_pg.py.
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
    PromotionJobClaim,
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
    """Scripted connection: answers the worker's reads and guarded transitions."""

    def __init__(self, *, workspace: str = "ws_alpha") -> None:
        self.workspace = workspace
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
        if "quote_literal" in query:
            return f"'{self.workspace}'"
        if "to_regclass" in query:
            return None
        if "RETURNING 1" in query:
            return self.returning_results.pop(0) if self.returning_results else 0
        return None

    async def fetchrow(self, query: str, *args: Any) -> Any:
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
        return []

    async def execute(self, query: str, *args: Any) -> str:
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


async def test_renewal_fence_loss_signals_copy_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    async def immediate_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(worker_module.asyncio, "sleep", immediate_sleep)
    job_store = SimpleNamespace(renew_lease=AsyncMock(return_value=True))
    registry = SimpleNamespace(acquire_write_fence=AsyncMock(return_value=False))
    worker = PGPromotionWorker(job_store=cast(Any, job_store), registry=cast(Any, registry))
    claim = PromotionJobClaim(
        job_id=11,
        workspace="ws_alpha",
        attempt_count=2,
        lease_generation=7,
        owner="promo-owner",
    )
    renewal_lost = asyncio.Event()

    await worker._renew_while(
        claim=claim,
        fence_owner="promo-owner#7",
        renewal_lost=renewal_lost,
    )

    assert renewal_lost.is_set()
    job_store.renew_lease.assert_awaited_once()
    registry.acquire_write_fence.assert_awaited_once()


async def test_copy_aborts_before_work_when_renewal_was_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    job_store = SimpleNamespace()
    registry = SimpleNamespace()
    worker = PGPromotionWorker(job_store=cast(Any, job_store), registry=cast(Any, registry))
    recheck = AsyncMock()
    monkeypatch.setattr(worker, "_recheck_current", recheck)
    renewal_lost = asyncio.Event()
    renewal_lost.set()

    with pytest.raises(StalePromotionAttempt, match="lost during copy"):
        await worker._copy_and_cutover(
            _Conn(),
            PromotionJobClaim(
                job_id=11,
                workspace="ws_alpha",
                attempt_count=2,
                lease_generation=7,
                owner="promo-owner",
            ),
            "promo-owner#7",
            renewal_lost,
        )

    recheck.assert_not_awaited()


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
    )
    registry = SimpleNamespace(
        acquire_write_fence=AsyncMock(return_value=True),
        set_promotion_state=AsyncMock(return_value=True),
        release_write_fence=AsyncMock(return_value=True),
    )
    worker = _worker(monkeypatch, job_store=job_store, registry=registry, conn=conn)
    monkeypatch.setattr(worker, "_recheck_current", stale_recheck)

    assert await worker.run_once() is True

    # Only the initial 'promoting' observability write; no failed state.
    assert all(
        kwargs["state"] == "promoting"
        for kwargs in [call.kwargs for call in registry.set_promotion_state.await_args_list]
    )


async def test_reclaimed_lease_during_failure_handling_releases_no_fence(
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
    registry.release_write_fence.assert_not_awaited()


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


async def test_cancelled_stale_worker_releases_no_fence(
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
    )
    registry = SimpleNamespace(
        acquire_write_fence=AsyncMock(return_value=True),
        set_promotion_state=AsyncMock(return_value=True),
        release_write_fence=AsyncMock(return_value=True),
    )
    worker = _worker(monkeypatch, job_store=job_store, registry=registry, conn=conn)
    monkeypatch.setattr(worker, "_recheck_current", AsyncMock())
    monkeypatch.setattr(worker, "_cleanup_artifacts_on", AsyncMock())

    async def cancel_mid(
        conn_arg: Any,  # noqa: ANN401
        claim: Any,  # noqa: ANN401
        fence_owner: str,
        renewal_lost: asyncio.Event,
    ) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(worker, "_copy_and_cutover", cancel_mid)

    with pytest.raises(asyncio.CancelledError):
        await worker.run_once()

    # In-gate cleanup ran (we still own the exclusive); the guarded job
    # transition was attempted once and refused (returned 0), so the registry
    # guard never ran and no registry/job state was mutated.
    worker._cleanup_artifacts_on.assert_awaited_once_with(conn, "ws_alpha")  # type: ignore[attr-defined]
    registry.release_write_fence.assert_not_awaited()


async def test_cancelled_current_worker_cleans_its_staging_once(
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
    )
    registry = SimpleNamespace(
        acquire_write_fence=AsyncMock(return_value=True),
        set_promotion_state=AsyncMock(return_value=True),
        release_write_fence=AsyncMock(return_value=True),
    )
    worker = _worker(monkeypatch, job_store=job_store, registry=registry, conn=conn)
    monkeypatch.setattr(worker, "_recheck_current", AsyncMock())
    monkeypatch.setattr(worker, "_cleanup_artifacts_on", AsyncMock())

    async def cancel_mid(
        conn_arg: Any,  # noqa: ANN401
        claim: Any,  # noqa: ANN401
        fence_owner: str,
        renewal_lost: asyncio.Event,
    ) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(worker, "_copy_and_cutover", cancel_mid)

    with pytest.raises(asyncio.CancelledError):
        await worker.run_once()

    # In-gate cleanup ran once; the guarded failed/retry transition it then
    # commits runs against PostgreSQL in test_promotion_worker_pg.py.
    worker._cleanup_artifacts_on.assert_awaited_once_with(conn, "ws_alpha")  # type: ignore[attr-defined]
