# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The fenced row lock a worker takes on the Run it still holds."""

from typing import Any

# Locks the Run row only while this worker's claim is still the live one: the
# same owner and fencing epoch, still running, lease unexpired. Every fenced
# write that is more than one statement takes this lock first, so a reclaimed
# Run refuses a stale worker and one lock order holds across writers.
_HOLD_RUN_LEASE = """
SELECT 1
FROM dlightrag_runs
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
FOR UPDATE
"""


async def hold_run_lease(
    conn: Any, owner_id: str, run_id: Any, lease_owner: str, fencing_epoch: int
) -> bool:
    """Lock the Run row inside the caller's transaction if the lease is still held."""
    held = await conn.fetchval(_HOLD_RUN_LEASE, owner_id, run_id, lease_owner, fencing_epoch)
    return held is not None


__all__ = ["hold_run_lease"]
