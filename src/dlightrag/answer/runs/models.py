# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The restorable research state of one durable Answer run.

A checkpoint is restorable agent state, never a second lifecycle record: the run
row stays the sole authority for status, phase, turn count, cancellation, lease,
result, and terminal error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlightrag_agent.session.fold import SessionEpisode

    from dlightrag.answer.evidence import EvidenceLedger
    from dlightrag.answer.resources.registry import ResourceRegistry
    from dlightrag.answer.tools import ExactCallCache


@dataclass(slots=True)
class AgentRunState:
    """The research memory one control turn advances and a checkpoint restores."""

    evidence: EvidenceLedger
    episode: SessionEpisode
    tool_cache: ExactCallCache
    registry: ResourceRegistry | None = None
    trace: dict[str, Any] = field(default_factory=dict)
    completed_turns: int = 0
    stop_reason: str = "model_stop"


__all__ = [
    "AgentRunState",
]
