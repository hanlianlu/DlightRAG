# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral projections of Memory application results."""

from collections.abc import Callable
from typing import Any

from dlightrag_memory import MemoryOperationReceipt, MemoryRunChange


def memory_receipt_payload(
    receipt: MemoryOperationReceipt,
    *,
    mask_reader_text: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Project one operation receipt without leaking package internals.

    ``body`` is the memory's own text. A writer may have stored a passage it read — a
    protected Skill document's words reach Memory the same way they reach any other carrier —
    so the projection a reader receives passes it through the reader's content gate.
    """
    body = receipt.body
    if mask_reader_text is not None:
        body = mask_reader_text(body)
    return {
        "action": receipt.action,
        "body": body,
        "change_id": receipt.change_id,
        "kind": receipt.kind,
        "memory_ids": list(receipt.memory_ids),
        "outcome": receipt.outcome,
        "supersedes_id": receipt.supersedes_id,
        "target_change_id": receipt.target_change_id,
    }


def memory_run_changes_payload(
    changes: tuple[MemoryRunChange, ...],
    *,
    mask_reader_text: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Project what one Run's writers left: each record, its change, and who wrote it.

    ``writer`` is ``conversation`` for a record the Session Memory Pass wrote from the
    conversation and ``agent`` for one the Agent's own memory tool wrote during the Run.
    Each ``body`` is reader text, so it is gated exactly as a receipt's is.
    """
    return {
        "changes": [
            {
                "memory_id": change.record.memory_id,
                "kind": change.record.kind,
                "body": (
                    change.record.body
                    if mask_reader_text is None
                    else mask_reader_text(change.record.body)
                ),
                "action": change.action,
                "change_id": change.change_id,
                "writer": (
                    "conversation"
                    if change.record.provenance.origin_kind == "session_pass"
                    else "agent"
                ),
            }
            for change in changes
        ]
    }


__all__ = ["memory_receipt_payload", "memory_run_changes_payload"]
