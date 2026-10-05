# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Real Child Sessions in a claimed Run, created through the run store's own writes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpawnedChild:
    """One claimed Child Session with its first running Operation."""

    child_session_id: str
    parent_session_id: str
    operation_id: str
    child_epoch: int


async def spawn_child(
    store: Any,
    *,
    owner_id: str,
    run_id: str,
    worker_id: str,
    fencing_epoch: int,
    objective: str = "inspect sources",
) -> SpawnedChild:
    """Create one Child with its first Operation and claim it, as the parent's spawn does."""
    parent_id = str(uuid.uuid7())
    child_id = str(uuid.uuid7())
    assert await store.upsert_child_session(
        owner_id=owner_id,
        run_id=run_id,
        child_session_id=child_id,
        parent_session_id=parent_id,
        parent_call_id=f"call-{child_id}",
        parent_intent_id=str(uuid.uuid7()),
        objective=objective,
        context_mode="isolated",
        model_role="query",
        tools=("search_knowledge_base",),
        depth=1,
        context_snapshot={
            "parent_session_id": parent_id,
            "parent_entry_id": str(uuid.uuid7()),
            "depth": 0,
            "messages": [],
            "evidence_state": {},
        },
        plan={"schema_version": 2, "tools": ["search_knowledge_base"]},
        budget={"provider_attempt_limit": 2},
        worker_id=worker_id,
        fencing_epoch=fencing_epoch,
    )
    child_epoch = await store.claim_child_session(
        owner_id=owner_id,
        run_id=run_id,
        child_session_id=child_id,
        worker_id=worker_id,
        fencing_epoch=fencing_epoch,
    )
    child = await store.load_child_session(
        owner_id=owner_id, run_id=run_id, child_session_id=child_id
    )
    assert child is not None and child_epoch is not None
    return SpawnedChild(child_id, parent_id, str(child["operation_id"]), child_epoch)


async def settle_child(
    store: Any,
    child: SpawnedChild,
    *,
    owner_id: str,
    run_id: str,
    worker_id: str,
    fencing_epoch: int,
    status: str = "succeeded",
) -> None:
    """Finish the Child's running Operation with the outcome the Child would report."""
    assert await store.finish_child_session(
        owner_id=owner_id,
        run_id=run_id,
        child_session_id=child.child_session_id,
        status=status,
        summary="done",
        usage={"input_tokens": 3},
        outcome={
            "status": status,
            "summary": "done",
            "handles": [],
            "usage": {"input_tokens": 3},
            "child_session_id": child.child_session_id,
            "operation_id": child.operation_id,
            "evidence_state": {"contexts": {}},
        },
        worker_id=worker_id,
        fencing_epoch=fencing_epoch,
    )
