# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closed Profile Memory operation checklist and fixed safety bounds."""

from __future__ import annotations

import re

from dlightrag_memory.errors import MemoryWriteRejectedError
from dlightrag_memory.models import MemoryOperation

MEMORY_BODY_LIMIT = 500
RECALL_TOP_K = 10
RECALL_CHAR_BUDGET = 4000
MEMORY_SUPERSEDE_RETENTION_DAYS = 365

_CITATION_MARK = re.compile(r"\[\d+(?:-\d+)?\]")
_PRIVATE_KEY_MARK = re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----", re.IGNORECASE)
_TOKEN_MARK = re.compile(
    r"(?:\bAKIA[0-9A-Z]{16}\b|\bgh[opusr]_[A-Za-z0-9_]{20,}\b|"
    r"\bgithub_pat_[A-Za-z0-9_]{20,}\b|\bxox[baprs]-[A-Za-z0-9-]{20,}\b|"
    r"\bsk-[A-Za-z0-9_-]{20,}\b)"
)


def evaluate_memory_operation(operation: MemoryOperation) -> None:
    """Validate one operation before any storage mutation."""
    if not operation.owner_id.strip():
        raise MemoryWriteRejectedError("A Memory operation needs an owner.")
    if not operation.idempotency_key.strip():
        raise MemoryWriteRejectedError("A Memory operation needs an idempotency key.")
    if len(operation.idempotency_key) > 255:
        raise MemoryWriteRejectedError("Memory idempotency keys cannot exceed 255 characters.")
    if not operation.provenance.origin_id.strip():
        raise MemoryWriteRejectedError("A Memory operation needs trusted provenance.")
    if operation.mutation_limit is not None and operation.mutation_limit < 1:
        raise MemoryWriteRejectedError("A Memory mutation limit must be positive.")
    if (operation.mutation_scope is None) != (operation.mutation_limit is None):
        raise MemoryWriteRejectedError("Memory mutation scope and limit must be provided together.")

    body = operation.body.strip()
    if operation.action == "remember":
        if operation.kind not in {"preference", "fact"}:
            raise MemoryWriteRejectedError("Memory kind must be preference or fact.")
        if not body:
            raise MemoryWriteRejectedError("Memory body cannot be empty.")
        if len(body) > MEMORY_BODY_LIMIT:
            raise MemoryWriteRejectedError(
                f"Memory body cannot exceed {MEMORY_BODY_LIMIT} characters."
            )
        if operation.memory_id is not None or operation.target_change_id is not None:
            raise MemoryWriteRejectedError("Remember received an incompatible target.")
        if _CITATION_MARK.search(body):
            raise MemoryWriteRejectedError("Memory body cannot carry citation markers.")
        if _PRIVATE_KEY_MARK.search(body) or _TOKEN_MARK.search(body):
            raise MemoryWriteRejectedError("Credentials and private keys cannot be remembered.")
        return

    if operation.action == "forget":
        selectors = sum(bool(value and value.strip()) for value in (operation.memory_id, body))
        if selectors != 1:
            raise MemoryWriteRejectedError("Forget needs exactly one memory id or exact body.")
        if any((operation.kind, operation.supersedes_id, operation.target_change_id)):
            raise MemoryWriteRejectedError("Forget received an incompatible target.")
        return

    if operation.action == "undo":
        if not (operation.target_change_id or "").strip():
            raise MemoryWriteRejectedError("Undo needs a change id.")
        if any((operation.kind, body, operation.memory_id, operation.supersedes_id)):
            raise MemoryWriteRejectedError("Undo received an incompatible target.")
        return

    raise MemoryWriteRejectedError("Unknown Memory operation.")


__all__ = [
    "MEMORY_BODY_LIMIT",
    "MEMORY_SUPERSEDE_RETENTION_DAYS",
    "RECALL_CHAR_BUDGET",
    "RECALL_TOP_K",
    "evaluate_memory_operation",
]
