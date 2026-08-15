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
    from dlightrag.core.memory.episode import RunEpisode
    from dlightrag.core.memory.evidence import EvidenceLedger
    from dlightrag.core.resources.registry import ResourceRegistry
    from dlightrag.core.tools import ExactCallCache


@dataclass(slots=True)
class AgentRunState:
    """The research memory one control turn advances and a checkpoint restores."""

    evidence: EvidenceLedger
    episode: RunEpisode
    tool_cache: ExactCallCache
    registry: ResourceRegistry | None = None
    trace: dict[str, Any] = field(default_factory=dict)
    completed_turns: int = 0
    stop_reason: str = "model_stop"


__all__ = [
    "AgentRunState",
]
