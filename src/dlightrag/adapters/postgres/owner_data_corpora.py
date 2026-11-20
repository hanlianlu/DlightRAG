# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Delivering an owner erase's Corpus Workspace deletions (ADR 0049).

An erase never deletes corpus tables itself, and it must not: the corpus belongs
to the customer, and every corpus write goes through the Corpus Mutation lane,
where the Workspace write fence and the mutation window keep a concurrent ingest
from writing back what a delete removed. So a covered Workspace becomes one
Workspace Delete Run, delivered through the same application service the browser's
Corpus Workspace delete calls (`application/corpus_admin/mutations.py`) — one
implementation, two surfaces.

That makes the deletion asynchronous, and the erase says so: it records the Run
ids in the receipt, and the 30-day promise is checked against their terminal
states rather than against the erase's own.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol

from dlightrag.adapters.postgres.corpus.workspaces import PGWorkspaceRegistry
from dlightrag.adapters.postgres.corpus_mutations import build_corpus_mutations
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.corpus_admin.errors import WorkspaceNotFoundError
from dlightrag.application.corpus_admin.mutations import CorpusMutationService

logger = logging.getLogger(__name__)


class CorpusDeletionDeliverer(Protocol):
    """Deliver one Corpus Workspace's deletion, returning the Run that will do it.

    This is the erase's seam, declared where its implementation lives (the sweep
    declares `SweptPackage` the same way). An erase never deletes corpus tables
    itself: it hands the Workspace to this, which accepts a Workspace Delete Run on
    the Corpus Mutation lane. `None` means the Workspace was already gone, which is
    the outcome the delivery asked for.
    """

    async def __call__(self, *, workspace: str, submitted_by: str) -> str | None: ...


class DetachedScheduler:
    """The scheduler an operator process offers: it accepts, the deployment runs.

    `dlightrag-owner-data` is not a worker. It accepts the Workspace Delete into
    the store the deployment shares, and a coordinator claims it from there — the
    claim loop polls, so `wake` is only a nudge *inside* the accepting process and
    this one has nothing to nudge. `is_started` is the one question admission asks,
    "can this deployment run what it is about to accept?", and no process can answer
    it for another. So the erase promises the delivery and records the Run ids; the
    deletion itself is checked against their terminal states before `erase_due_at`,
    which is what the receipt exists for.
    """

    #: Not a claim about this process: refusing every delivery because a CLI cannot
    #: see the deployment's coordinator would make the operation impossible rather
    #: than safe, and the receipt is how the wait is bounded.
    is_started = True

    @asynccontextmanager
    async def admission(self) -> AsyncIterator[bool]:
        yield True

    def wake(self) -> None:
        """Nothing to wake: this process executes no Run; the coordinator polls."""


class _CorpusMutationDeliverer:
    """`CorpusMutationService.create_workspace_delete`, narrowed to the erase's port."""

    def __init__(self, *, service: CorpusMutationService) -> None:
        self._service = service

    async def __call__(self, *, workspace: str, submitted_by: str) -> str | None:
        try:
            creation = await self._service.create_workspace_delete(
                workspace=workspace, submitted_by=submitted_by
            )
        except WorkspaceNotFoundError:
            # The Workspace row is gone, so a concurrent deletion already took it:
            # the delivery has happened, and there is no Run left to name.
            logger.info("Corpus Workspace %s is already gone; nothing to deliver", workspace)
            return None
        return str(creation.run.run_id)


def corpus_deletion_deliverer(*, pool: Any, config: DlightragConfig) -> CorpusDeletionDeliverer:
    """Build the delivery an erase reaches a Corpus Workspace through.

    The service itself is the deployment's — `build_corpus_mutations` wires it, so
    this and the browser's Workspace delete cannot drift apart. What is this
    process's own is the scheduler: an operator accepts the Run and the deployment
    executes it.
    """
    return _CorpusMutationDeliverer(
        service=build_corpus_mutations(
            config=config,
            store=PGRunStore(pool=pool),
            workspace_exists=PGWorkspaceRegistry(pool=pool).exists,
            coordinator=DetachedScheduler(),
        )
    )


__all__ = ["CorpusDeletionDeliverer", "DetachedScheduler", "corpus_deletion_deliverer"]
