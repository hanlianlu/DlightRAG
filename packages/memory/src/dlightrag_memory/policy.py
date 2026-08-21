# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closed Memory Write checklist and the product's fixed safety bounds.

The 500-character atomic-record bound and the recall caps bound the standing
block reserved at acceptance and are coupled: changing either requires
updating the acceptance reservation math. There is no fixed record-count
ceiling — growth is bounded by supersede folding, explicit forget/clear, and
the deployment storage quota the host composes in.
"""

from __future__ import annotations

import re

from dlightrag_memory.errors import MemoryUnavailableError, MemoryWriteRejectedError
from dlightrag_memory.models import MemoryWrite

MEMORY_BODY_LIMIT = 500
MEMORY_RECALL_LIMIT = 12
MEMORY_RECALL_KIND_LIMIT = 8
#: Superseded profile history lives at least this long before purge. The
#: deployment retention clock (e.g. DlightRAG RuntimeConfig's
#: answer_run_retention_days) overrides it in composed applications.
MEMORY_SUPERSEDE_RETENTION_DAYS = 365

_CITATION_MARK = re.compile(r"\[\d+(?:-\d+)?\]")


def memory_owner_allowed(auth_mode: str) -> bool:
    """JWT principals may write and auto-recall; deployment buckets may not."""
    return auth_mode == "jwt"


def evaluate_memory_write(write: MemoryWrite) -> None:
    """Accept one Memory Write or raise a public checklist error."""
    if not memory_owner_allowed(write.auth_mode):
        raise MemoryUnavailableError()
    body = write.body.strip()
    if write.action == "forget":
        if not (write.supersedes_id or "").strip() and not body:
            raise MemoryWriteRejectedError("A forget must name a memory or quote its body.")
        return
    if write.kind not in {"preference", "fact"}:
        raise MemoryWriteRejectedError("Memory kind must be preference or fact.")
    if not body:
        raise MemoryWriteRejectedError("Memory body cannot be empty.")
    if len(body) > MEMORY_BODY_LIMIT:
        raise MemoryWriteRejectedError(f"Memory body cannot exceed {MEMORY_BODY_LIMIT} characters.")
    if not 0 < write.confidence <= 1:
        raise MemoryWriteRejectedError("Memory confidence must be in (0, 1].")
    if not write.provenance.run_id.strip() or not write.provenance.session_id.strip():
        raise MemoryWriteRejectedError("A remember needs run and session provenance.")
    if _CITATION_MARK.search(body):
        raise MemoryWriteRejectedError("Memory body cannot carry citation markers.")


__all__ = [
    "MEMORY_BODY_LIMIT",
    "MEMORY_RECALL_KIND_LIMIT",
    "MEMORY_RECALL_LIMIT",
    "MEMORY_SUPERSEDE_RETENTION_DAYS",
    "evaluate_memory_write",
    "memory_owner_allowed",
]
