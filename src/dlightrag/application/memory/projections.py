# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral projections of Memory application results."""

from typing import Any

from dlightrag_memory import MemoryOperationReceipt


def memory_receipt_payload(receipt: MemoryOperationReceipt) -> dict[str, Any]:
    """Project one operation receipt without leaking package internals."""
    return {
        "action": receipt.action,
        "body": receipt.body,
        "change_id": receipt.change_id,
        "kind": receipt.kind,
        "memory_ids": list(receipt.memory_ids),
        "outcome": receipt.outcome,
        "supersedes_id": receipt.supersedes_id,
        "target_change_id": receipt.target_change_id,
    }


__all__ = ["memory_receipt_payload"]
