# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Which Run holds each Agent Browser endpoint, in PostgreSQL.

Every process that runs Query workers may execute a Research Run, so the pool's
leases are shared state (ADR 0032). A lease is live exactly while its holder's Run
lease is: the same worker and fencing epoch on a running Run whose lease has not
expired. The Run's own heartbeat therefore renews it without a write of its own, and a
terminal, deferred, or reclaimed Run, like a dead worker, frees the slot with nothing
left to release. The table belongs to the ``runs`` scope, so the Run store creates
and verifies it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dlightrag.adapters.postgres.core._migrations import TableRequirement
from dlightrag.adapters.postgres.core._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.engine.answer.agent_browser import BrowserHolder
from dlightrag.engine.runtime.records import parse_run_id

_CREATE_AGENT_BROWSER_LEASES = """
CREATE TABLE IF NOT EXISTS dlightrag_agent_browser_leases (
    endpoint      TEXT        NOT NULL,
    owner_id      TEXT,
    run_id        UUID,
    lease_owner   TEXT,
    fencing_epoch BIGINT,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (endpoint),
    CONSTRAINT dlightrag_agent_browser_leases_holder_check CHECK (
        (owner_id IS NULL) = (run_id IS NULL)
        AND (run_id IS NULL) = (lease_owner IS NULL)
        AND (lease_owner IS NULL) = (fencing_epoch IS NULL))
)
"""

AGENT_BROWSER_LEASES_DDL = (_CREATE_AGENT_BROWSER_LEASES,)

AGENT_BROWSER_LEASES_SCHEMA_TABLE = TableRequirement(
    name="dlightrag_agent_browser_leases",
    columns=("endpoint", "owner_id", "run_id", "lease_owner", "fencing_epoch", "updated_at"),
    primary_key=("endpoint",),
    checks=("dlightrag_agent_browser_leases_holder_check",),
)

_REGISTER_ENDPOINTS = """
INSERT INTO dlightrag_agent_browser_leases (endpoint)
SELECT unnest($1::text[])
ON CONFLICT (endpoint) DO NOTHING
"""

# One statement, so a claim is atomic: the holder's own Run must still hold its lease; a
# holder that already has one of these endpoints gets it back; otherwise the
# least recently used free endpoint is taken, skipping rows a concurrent claim holds. An
# endpoint is free when it has no holder or its holder's Run no longer holds that lease.
# Freedom is a correlated NOT EXISTS, not a join: under READ COMMITTED a row that a
# concurrent claim just took is rechecked after it is locked, and a join would recheck
# it against the Run row it had when the endpoint was free (none), so a stale verdict
# of "free" would survive and two Runs would hold one endpoint. The subquery runs again
# for the locked row's new holder.
_CLAIM = """
WITH claimer AS (
    SELECT 1 FROM dlightrag_runs
    WHERE owner_id = $1 AND run_id = $2 AND lease_owner = $3 AND fencing_epoch = $4
      AND status = 'running' AND lease_expires_at > NOW()
),
held AS (
    SELECT l.endpoint FROM dlightrag_agent_browser_leases l
    WHERE l.owner_id = $1 AND l.run_id = $2 AND l.lease_owner = $3 AND l.fencing_epoch = $4
      AND l.endpoint = ANY($5::text[]) AND NOT (l.endpoint = ANY($6::text[]))
      AND EXISTS (SELECT 1 FROM claimer)
),
candidate AS (
    SELECT l.endpoint FROM dlightrag_agent_browser_leases l
    WHERE l.endpoint = ANY($5::text[]) AND NOT (l.endpoint = ANY($6::text[]))
      AND EXISTS (SELECT 1 FROM claimer) AND NOT EXISTS (SELECT 1 FROM held)
      AND (l.run_id IS NULL OR NOT EXISTS (
            SELECT 1 FROM dlightrag_runs r
            WHERE r.owner_id = l.owner_id AND r.run_id = l.run_id
              AND r.status = 'running' AND r.lease_owner = l.lease_owner
              AND r.fencing_epoch = l.fencing_epoch AND r.lease_expires_at > NOW()))
    ORDER BY l.updated_at, l.endpoint
    LIMIT 1
    FOR UPDATE OF l SKIP LOCKED
),
claimed AS (
    UPDATE dlightrag_agent_browser_leases l
    SET owner_id = $1, run_id = $2, lease_owner = $3, fencing_epoch = $4, updated_at = NOW()
    FROM candidate
    WHERE l.endpoint = candidate.endpoint
    RETURNING l.endpoint
)
SELECT endpoint FROM held
UNION ALL
SELECT endpoint FROM claimed
LIMIT 1
"""

_RELEASE = """
UPDATE dlightrag_agent_browser_leases
SET owner_id = NULL, run_id = NULL, lease_owner = NULL, fencing_epoch = NULL, updated_at = NOW()
WHERE endpoint = $1 AND owner_id = $2 AND run_id = $3 AND lease_owner = $4 AND fencing_epoch = $5
"""


class PGAgentBrowserLeaseStore(PostgresOperationRunner):
    """Claim and release the Agent Browser endpoints a Run leases."""

    def __init__(self, *, pool: ConnectionPool | None = None) -> None:
        super().__init__(pool=pool)

    async def register_endpoints(self, endpoints: Sequence[str]) -> None:
        """Make every configured endpoint claimable; one already registered stays as it is."""

        async def operation(conn: Any) -> None:
            await conn.execute(_REGISTER_ENDPOINTS, list(endpoints))

        await self._run(operation)

    async def claim(
        self,
        holder: BrowserHolder,
        endpoints: Sequence[str],
        exclude: Sequence[str] = (),
    ) -> str | None:
        """The endpoint ``holder`` now leases, or None when none is free.

        A holder that already leases one of ``endpoints`` is given that one again. An
        endpoint named in ``exclude`` is neither returned nor taken.
        """
        run_id = parse_run_id(holder.run_id)
        if run_id is None:
            return None

        async def operation(conn: Any) -> str | None:
            row = await conn.fetchrow(
                _CLAIM,
                holder.owner_id,
                run_id,
                holder.worker_id,
                holder.fencing_epoch,
                list(endpoints),
                list(exclude),
            )
            return None if row is None else str(row["endpoint"])

        return await self._run(operation)

    async def release(self, holder: BrowserHolder, endpoint: str) -> None:
        """Give ``endpoint`` back; a lease the holder no longer holds is left alone."""
        run_id = parse_run_id(holder.run_id)
        if run_id is None:
            return

        async def operation(conn: Any) -> None:
            await conn.execute(
                _RELEASE,
                endpoint,
                holder.owner_id,
                run_id,
                holder.worker_id,
                holder.fencing_epoch,
            )

        await self._run(operation)


__all__ = [
    "AGENT_BROWSER_LEASES_DDL",
    "AGENT_BROWSER_LEASES_SCHEMA_TABLE",
    "PGAgentBrowserLeaseStore",
]
