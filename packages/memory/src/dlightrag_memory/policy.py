# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closed Memory Write checklist and the product's fixed safety bounds.

The 500-character atomic-record bound caps one assertion; recall packing is
bounded by ``RECALL_TOP_K`` (count) plus ``RECALL_CHAR_BUDGET`` (block size),
matching the industry shape (MemMachine counts, MemoraX chars). Growth is
absorbed by supersede folding, explicit forget/clear, natural write rates,
and cheap storage — no fixed record ceiling, no quota (Pi/Kimi/MemMachine
bound nothing; see the retention ADR).
"""

from __future__ import annotations

import re

from dlightrag_memory.errors import MemoryWriteRejectedError
from dlightrag_memory.models import MemoryWrite

MEMORY_BODY_LIMIT = 500
RECALL_TOP_K = 10
RECALL_CHAR_BUDGET = 4000
#: Superseded profile history lives at least this long before purge. The
#: deployment retention clock (e.g. DlightRAG RuntimeConfig's
#: answer_run_retention_days) overrides it in composed applications.
MEMORY_SUPERSEDE_RETENTION_DAYS = 365

_CITATION_MARK = re.compile(r"\[\d+(?:-\d+)?\]")


def evaluate_memory_write(write: MemoryWrite) -> None:
    """Accept one Memory Write or raise a public checklist error.

    Caller eligibility is host policy; the package never judges identity.
    """
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
    "MEMORY_SUPERSEDE_RETENTION_DAYS",
    "RECALL_CHAR_BUDGET",
    "RECALL_TOP_K",
    "evaluate_memory_write",
]
