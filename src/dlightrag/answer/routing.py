# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-owned routing record written at accept and resolved later."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from dlightrag.answer.mode import ResolvedMode, canonical_answer_mode


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
        from dlightrag.ai.capacity import CONTEXT_POLICY_REVISION

        return cls(
            requested_mode=canonical_answer_mode(str(prepared_input.get("mode") or "") or None),
            valid_modes=("fast", "research"),
            context_policy_revision=str(
                prepared_input.get("context_policy_revision") or CONTEXT_POLICY_REVISION
            ),
            model_fingerprints={},
        )


@dataclass(frozen=True, slots=True)
class RoutingRecord:
    """Durable routing row as the worker reads it."""

    requested_mode: str
    valid_modes: tuple[str, ...]
    resolved_mode: str | None
    research_session_id: str | None


def decide_resolved_mode(
    *,
    requested_mode: str,
    valid_modes: frozenset[str],
) -> ResolvedMode | None:
    """Return a mode that needs no LLM, or None when the router must run."""
    requested = canonical_answer_mode(requested_mode)
    if requested in valid_modes:
        return requested  # type: ignore[return-value]
    if requested == "auto" and len(valid_modes) == 1:
        only = next(iter(valid_modes))
        if only in {"fast", "research"}:
            return only  # type: ignore[return-value]
    if requested == "auto" and valid_modes >= {"fast", "research"}:
        return None
    raise ValueError(f"cannot resolve {requested} against {sorted(valid_modes)}")


class AnswerRoutingStore(Protocol):
    """Lease-fenced load and CAS for Resolved Mode."""

    async def load_routing(self, *, owner_id: str, run_id: str) -> RoutingRecord | None: ...

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


__all__ = [
    "AnswerRoutingStore",
    "RoutingAcceptance",
    "RoutingRecord",
    "decide_resolved_mode",
]
