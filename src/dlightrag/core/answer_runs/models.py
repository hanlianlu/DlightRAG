# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The restorable state of one durable Answer run, and how it can fail to load.

A checkpoint is restorable agent state, never a second lifecycle record: the run
row stays the sole authority for status, phase, turn count, cancellation, lease,
result, and terminal error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from dlightrag.core.memory.episode import RunEpisode
    from dlightrag.core.memory.evidence import EvidenceLedger
    from dlightrag.core.resources.registry import ResourceRegistry
    from dlightrag.core.tools import ExactCallCache

#: Every worker sharing a database writes and reads this checkpoint schema.
CHECKPOINT_SCHEMA_VERSION = 1
#: Compact UTF-8 JSON bound, measured after image-reference substitution.
MAX_CHECKPOINT_BYTES = 8 * 1024 * 1024

type CheckpointErrorKind = Literal[
    "checkpoint_incompatible",
    "checkpoint_corrupt",
    "checkpoint_too_large",
]


class CheckpointError(RuntimeError):
    """One durable checkpoint cannot be written or restored.

    Every kind is terminal for its run: a worker fails the run with this public
    kind instead of guessing at state or retrying the same deterministic turn.
    """

    def __init__(self, kind: CheckpointErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind: CheckpointErrorKind = kind
        self.public_message = message


class AnswerRunCancelledError(RuntimeError):
    """The run this caller waited on was cancelled by its owner."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Answer run {run_id} was cancelled")
        self.run_id = run_id


class AnswerRunFailedError(RuntimeError):
    """The run this caller waited on failed; only its public kind is exposed."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.error_kind = kind
        self.public_message = message


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
    "CHECKPOINT_SCHEMA_VERSION",
    "MAX_CHECKPOINT_BYTES",
    "AgentRunState",
    "AnswerRunCancelledError",
    "AnswerRunFailedError",
    "CheckpointError",
    "CheckpointErrorKind",
]
