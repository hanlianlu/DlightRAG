# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""An adopted earlier-Run Resource, made durable under the consuming Run's lease."""

from collections.abc import Sequence
from typing import Any

from dlightrag.adapters.postgres.answer.session_repository import write_fetched_resources
from dlightrag.adapters.postgres.runtime._lease import hold_run_lease
from dlightrag.engine.runtime.coordinator import LeaseLostError
from dlightrag.engine.runtime.settlements import FetchedResourceSettlementUpdate


async def record_lineage_adoption(
    conn: Any,
    *,
    owner_id: str,
    run_id: Any,
    worker_id: str,
    fencing_epoch: int,
    resources: Sequence[FetchedResourceSettlementUpdate],
) -> None:
    """Write one adoption's rows and Blobs in one transaction under the Run lease.

    The Run row is locked first, as every fenced write of more than one statement
    does, so a reclaimed Run refuses a stale worker and nothing is written. The rows
    are written the way settlement writes them: recording one adoption again, or
    settling its view again later, only merges aliases, and a row naming other
    bytes rolls the whole adoption back.
    """
    async with conn.transaction():
        if not await hold_run_lease(conn, owner_id, run_id, worker_id, fencing_epoch):
            raise LeaseLostError
        await write_fetched_resources(conn, owner_id=owner_id, run_id=run_id, updates=resources)


__all__ = ["record_lineage_adoption"]
