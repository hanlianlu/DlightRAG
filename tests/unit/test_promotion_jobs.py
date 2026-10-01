# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Promotion-job transition inputs the adapter refuses before any statement runs.

Enqueue, claims, leases, fenced transitions and the table's constraints run
against PostgreSQL in tests/integration/test_promotion_foundation_pg.py and
test_promotion_worker_pg.py.
"""

from typing import Any

import pytest

from dlightrag.adapters.postgres.corpus import promotion_jobs
from dlightrag.adapters.postgres.corpus.promotion_jobs import PGPromotionJobStore


class _NoStatements:
    """A connection that refuses every statement, so a refusal shows it ran none."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"no statement may run before the inputs are validated: {name}")


class _Pool:
    def acquire(self) -> Any:
        raise AssertionError("no connection may be taken before the inputs are validated")


async def test_transition_identity_and_retry_inputs_are_validated() -> None:
    conn = _NoStatements()
    store = PGPromotionJobStore(pool=_Pool())

    with pytest.raises(ValueError, match="lease owner"):
        await store.claim_next(owner=" ", lease_until="2026-04-01T00:00:00Z")
    with pytest.raises(ValueError, match="job_id"):
        await promotion_jobs.mark_done_in(conn, job_id=0, owner="worker", lease_generation=1)
    with pytest.raises(ValueError, match="lease_generation"):
        await promotion_jobs.mark_done_in(conn, job_id=1, owner="worker", lease_generation=0)
    with pytest.raises(ValueError, match="next_retry_at"):
        await promotion_jobs.mark_failed_in(
            conn,
            job_id=1,
            owner="worker",
            lease_generation=1,
            error="failed",
            next_retry_at=None,
        )
