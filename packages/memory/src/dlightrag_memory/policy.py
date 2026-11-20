# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closed Profile Memory operation checklist and fixed safety bounds."""

from __future__ import annotations

import re

from dlightrag_memory.errors import MemoryWriteRejectedError
from dlightrag_memory.models import MEMORY_KINDS, MemoryOperation
from dlightrag_memory.normalize import single_spaced

MEMORY_BODY_LIMIT = 500
# The owner's own words behind a memory: a short excerpt, never the whole turn.
MEMORY_QUOTE_LIMIT = 120
# One recall fills its relevant section first — at most RECALL_TOP_K records of
# any kind the query's search found, within RELEVANT_CHAR_BUDGET — and its
# standing section of preferences and experiences takes the rest of
# RECALL_CHAR_BUDGET, from at most STANDING_FETCH_LIMIT newest standing records.
# Budgets count rendered lines: each record's body plus RECALL_LINE_OVERHEAD,
# which every renderer's bullet, label and line break must fit inside, and for a record that has
# one, its quote plus RECALL_QUOTE_OVERHEAD.
RECALL_TOP_K = 10
RECALL_CHAR_BUDGET = 8000
RELEVANT_CHAR_BUDGET = RECALL_CHAR_BUDGET // 2
STANDING_FETCH_LIMIT = 100
# A record reaches a model as "- (kind, time) body" plus a line break. The time is
# when it was last written, to the hour in UTC, so the framing at its longest is
# "- (" 3 + kind 10 + ", " 2 + time 16 + ") " 2 + line break 1 = 34.
RECALL_TIME_FORMAT = "%Y-%m-%d %H:00"
RECALL_TIME_NOTE = "times are UTC, when each was last written"
RECALL_LINE_OVERHEAD = 34
# A record's quote is the line under it: two spaces, "owner said:", a space, the words between
# double quotes, and a line break - 2 + 11 + 1 + 1 + 1 + 1 = 17 characters around the words.
RECALL_QUOTE_OVERHEAD = 17
MEMORY_SUPERSEDE_RETENTION_DAYS = 365
MEMORY_FUSE_MIN_SOURCES = 2
MEMORY_FUSE_MAX_SOURCES = 10

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
    if operation.action != "fuse" and operation.source_ids:
        raise MemoryWriteRejectedError("Only fuse names source records.")
    if operation.quote is not None:
        if operation.action not in ("remember", "fuse"):
            raise MemoryWriteRejectedError("Only remember and fuse carry a quote.")
        if not is_storable_quote(operation.quote):
            raise MemoryWriteRejectedError(
                f"A quote is one single-spaced line of 1 to {MEMORY_QUOTE_LIMIT} characters, "
                "free of citation markers and credentials."
            )

    if operation.action == "remember":
        if operation.kind not in MEMORY_KINDS:
            raise MemoryWriteRejectedError("Memory kind must be preference, fact or experience.")
        _require_storable_body(body)
        if operation.memory_id is not None or operation.target_change_id is not None:
            raise MemoryWriteRejectedError("Remember received an incompatible target.")
        return

    if operation.action == "forget":
        selectors = sum(bool(value and value.strip()) for value in (operation.memory_id, body))
        if selectors != 1:
            raise MemoryWriteRejectedError("Forget needs exactly one memory id or exact body.")
        if any((operation.kind, operation.supersedes_id, operation.target_change_id)):
            raise MemoryWriteRejectedError("Forget received an incompatible target.")
        return

    if operation.action == "undo":
        # A caller holding a change names it; one holding a record names that,
        # and the store resolves it to the change behind the record's current form.
        selectors = sum(
            bool(value and value.strip())
            for value in (operation.target_change_id, operation.memory_id, body)
        )
        if selectors != 1:
            raise MemoryWriteRejectedError(
                "Undo needs exactly one change id, memory id or exact body."
            )
        if any((operation.kind, operation.supersedes_id)):
            raise MemoryWriteRejectedError("Undo received an incompatible target.")
        return

    if operation.action == "fuse":
        if operation.kind is not None and operation.kind not in MEMORY_KINDS:
            raise MemoryWriteRejectedError("Memory kind must be preference, fact or experience.")
        _require_storable_body(body)
        sources = operation.source_ids
        if not MEMORY_FUSE_MIN_SOURCES <= len(sources) <= MEMORY_FUSE_MAX_SOURCES:
            raise MemoryWriteRejectedError(
                f"Fuse needs {MEMORY_FUSE_MIN_SOURCES} to {MEMORY_FUSE_MAX_SOURCES} records."
            )
        if len(set(sources)) != len(sources) or any(not source.strip() for source in sources):
            raise MemoryWriteRejectedError("Fuse needs distinct record ids.")
        if any((operation.memory_id, operation.supersedes_id, operation.target_change_id)):
            raise MemoryWriteRejectedError("Fuse received an incompatible target.")
        return

    raise MemoryWriteRejectedError("Unknown Memory operation.")


def is_storable_quote(quote: str) -> bool:
    """Whether ``quote`` may stand behind a memory.

    It is one line with single spaces, so that it renders as the one line it is charged for, of
    one to ``MEMORY_QUOTE_LIMIT`` characters; and it carries nothing a body may not: a citation
    marker or a credential, which the owner's own words may hold as well as a model's.
    """
    return (
        0 < len(quote) <= MEMORY_QUOTE_LIMIT
        and quote == single_spaced(quote)
        and not _CITATION_MARK.search(quote)
        and not _PRIVATE_KEY_MARK.search(quote)
        and not _TOKEN_MARK.search(quote)
    )


def _require_storable_body(body: str) -> None:
    """The checks every body a record will carry must pass, whoever wrote it."""
    if not body:
        raise MemoryWriteRejectedError("Memory body cannot be empty.")
    if len(body) > MEMORY_BODY_LIMIT:
        raise MemoryWriteRejectedError(f"Memory body cannot exceed {MEMORY_BODY_LIMIT} characters.")
    if _CITATION_MARK.search(body):
        raise MemoryWriteRejectedError("Memory body cannot carry citation markers.")
    if _PRIVATE_KEY_MARK.search(body) or _TOKEN_MARK.search(body):
        raise MemoryWriteRejectedError("Credentials and private keys cannot be remembered.")


__all__ = [
    "MEMORY_BODY_LIMIT",
    "MEMORY_FUSE_MAX_SOURCES",
    "MEMORY_FUSE_MIN_SOURCES",
    "MEMORY_QUOTE_LIMIT",
    "MEMORY_SUPERSEDE_RETENTION_DAYS",
    "RECALL_CHAR_BUDGET",
    "RECALL_LINE_OVERHEAD",
    "RECALL_QUOTE_OVERHEAD",
    "RECALL_TIME_FORMAT",
    "RECALL_TIME_NOTE",
    "RECALL_TOP_K",
    "RELEVANT_CHAR_BUDGET",
    "STANDING_FETCH_LIMIT",
    "evaluate_memory_operation",
    "is_storable_quote",
]
