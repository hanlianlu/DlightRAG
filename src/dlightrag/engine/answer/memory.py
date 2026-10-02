# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-side Memory policy and context rendering.

The canonical Memory shapes, checklist, recall selection, and storage contract
live in the independently installable ``dlightrag_memory`` package; this
module re-exports them for Answer callers and keeps the root-owned concerns:
owner eligibility policy and rendering the non-citable standing block.
"""

from dataclasses import dataclass

from dlightrag_memory import (
    MemoryKind,
    MemoryOperation,
    MemoryOperationReceipt,
    MemoryProvenance,
    MemoryRecord,
    MemoryStatus,
    RecallResult,
)
from dlightrag_memory.errors import MemoryUnavailableError
from dlightrag_memory.policy import (
    MEMORY_BODY_LIMIT,
    MEMORY_SUPERSEDE_RETENTION_DAYS,
    RECALL_CHAR_BUDGET,
    RECALL_TOP_K,
    evaluate_memory_operation,
)

from dlightrag.engine.answer.owner import is_personal_auth_mode


@dataclass(frozen=True, slots=True)
class MemoryCapability:
    """One owner's durable activation state and invalidation epoch."""

    enabled: bool
    epoch: int


# The densest script the token estimator knows; worst-case reserves are CJK.
_DENSEST_CHAR = "记"


def recall_sections(recalled: RecallResult, *, ids: bool = False) -> list[str]:
    """The labeled preference and fact lines of one recall."""
    lines: list[str] = []
    for label, records in (
        ("Standing preferences:", recalled.preferences),
        ("Relevant facts:", recalled.facts),
    ):
        if records:
            lines.append(label)
            lines.extend(
                f"- {record.memory_id} {record.body}" if ids else f"- {record.body}"
                for record in records
            )
    return lines


def render_auto_recall(recalled: RecallResult) -> str:
    """Standing non-citable block, or empty when there is nothing to inject."""
    if not recalled.records:
        return ""
    return "\n".join(
        (
            "Remembered about this owner (context, not citable; "
            "the current request takes priority):",
            *recall_sections(recalled),
        )
    )


def reserved_auto_recall_text() -> str:
    """Worst-case standing block one JWT accept must leave room for.

    Recall injects at most ``RECALL_TOP_K`` preferences plus ``RECALL_TOP_K``
    facts whose bodies total at most ``RECALL_CHAR_BUDGET`` characters. That
    many records written in CJK, the densest script for the token estimator,
    bound every block execution can inject — never less.
    """
    count = 2 * RECALL_TOP_K
    total = min(RECALL_CHAR_BUDGET, count * MEMORY_BODY_LIMIT)
    records = tuple(
        MemoryRecord(
            owner_id="reserve",
            memory_id=f"{index:02d}",
            kind="preference" if index < RECALL_TOP_K else "fact",
            body=_DENSEST_CHAR * (total // count + (index < total % count)),
            provenance=MemoryProvenance(
                origin_kind="answer_run",
                origin_id="reserve",
                run_id="reserve",
                session_id="reserve",
            ),
        )
        for index in range(count)
    )
    return render_auto_recall(
        RecallResult(preferences=records[:RECALL_TOP_K], facts=records[RECALL_TOP_K:])
    )


def standing_memory_for_acceptance(auth_mode: str) -> str:
    """Reserve full auto-recall at accept so execute cannot overflow after 202."""
    if not is_personal_auth_mode(auth_mode):
        return ""
    return reserved_auto_recall_text()


def standing_memory_message(memory_text: str) -> dict[str, str] | None:
    """The low-authority injection message, or None when there is nothing.

    Pi and Kimi both append injected context as a user-role message after the
    current request instead of mixing it into the system prompt; the ordering
    plus the block's framing is the authority mechanism.
    """
    if not memory_text:
        return None
    return {"role": "user", "content": memory_text}


__all__ = [
    "MEMORY_BODY_LIMIT",
    "MEMORY_SUPERSEDE_RETENTION_DAYS",
    "RECALL_TOP_K",
    "MemoryCapability",
    "MemoryKind",
    "MemoryOperation",
    "MemoryOperationReceipt",
    "MemoryProvenance",
    "MemoryRecord",
    "MemoryStatus",
    "MemoryUnavailableError",
    "RecallResult",
    "evaluate_memory_operation",
    "recall_sections",
    "render_auto_recall",
    "reserved_auto_recall_text",
    "standing_memory_for_acceptance",
    "standing_memory_message",
]
