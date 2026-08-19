# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-owned routing record written at accept and resolved later."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from dlightrag.answer.mode import canonical_answer_mode


@dataclass(frozen=True, slots=True)
class RoutingAcceptance:
    """Facts persisted on the Routing Record when a run is accepted."""

    requested_mode: str
    valid_modes: tuple[str, ...]
    context_policy_revision: str
    model_fingerprints: Mapping[str, Any]
    resolved_mode: str | None = None
    research_session_id: str | None = None

    @classmethod
    def fallback(cls, prepared_input: Mapping[str, Any]) -> RoutingAcceptance:
        """Best-effort row for accept seams that do not yet pass policy."""
        from dlightrag_ai.capacity import CONTEXT_POLICY_REVISION

        return cls(
            requested_mode=canonical_answer_mode(str(prepared_input.get("mode") or "") or None),
            valid_modes=("fast", "research"),
            context_policy_revision=str(
                prepared_input.get("context_policy_revision") or CONTEXT_POLICY_REVISION
            ),
            model_fingerprints={},
        )


class AnswerRoutingStore(Protocol):
    """Lease-fenced CAS for Resolved Mode. Unused until Task 3."""

    async def resolve(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        resolved_mode: str,
        research_session_id: str | None = None,
    ) -> str | None: ...


__all__ = ["AnswerRoutingStore", "RoutingAcceptance"]
