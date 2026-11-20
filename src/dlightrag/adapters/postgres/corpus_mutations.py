# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Composing the Corpus Mutation service against a PostgreSQL deployment.

The application service takes its Run store, its scheduler, its Workspace catalog
and its upload limits as arguments. This is the one place that says which concrete
ones a PostgreSQL deployment uses, so the browser's corpus surfaces and an
operator's `dlightrag-owner-data` cannot drift into two differently wired
services — the corpus-write rules live in the service, and where it gets its
dependencies is stated once.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.corpus_admin.mutations import (
    CorpusMutationService,
    CorpusMutationStore,
    UploadLimits,
)
from dlightrag.application.runs import RunAdmissionScheduler


def build_corpus_mutations(
    *,
    config: DlightragConfig,
    store: CorpusMutationStore,
    workspace_exists: Callable[[str], Awaitable[bool]],
    coordinator: RunAdmissionScheduler,
) -> CorpusMutationService:
    """The service every PostgreSQL corpus-write surface submits through.

    `writable` follows the service role: a reader registers no Corpus Mutation
    executor, so it refuses a write at acceptance rather than staging bytes for a
    Run it can never execute.
    """
    return CorpusMutationService(
        source_root=config.input_dir_path,
        corpus_root=config.corpus_dir_path,
        store=store,
        coordinator=coordinator,
        upload_limits=UploadLimits(
            file_bytes=config.corpus.ingestion.max_upload_bytes,
            request_bytes=config.max_upload_batch_bytes,
        ),
        workspace_exists=workspace_exists,
        writable=not config.is_reader,
        default_workspace=config.deployment.workspace_id,
    )


__all__ = ["build_corpus_mutations"]
