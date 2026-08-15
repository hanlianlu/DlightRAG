# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage- and policy-neutral durable runtime failures."""

from typing import Literal

#: Every worker sharing a database writes and reads this checkpoint schema.
CHECKPOINT_SCHEMA_VERSION = 1
#: Compact UTF-8 JSON bound, measured after image-reference substitution.
MAX_CHECKPOINT_BYTES = 8 * 1024 * 1024

type CheckpointErrorKind = Literal[
    "checkpoint_incompatible",
    "checkpoint_corrupt",
    "checkpoint_too_large",
]


class RunSchemaError(RuntimeError):
    """The durable run schema is incompatible with this Runtime revision."""


class CheckpointError(RuntimeError):
    """One durable checkpoint cannot be safely written or restored.

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
    """The run this caller waited on failed with one public error."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.error_kind = kind
        self.public_message = message


class RunExecutionError(RuntimeError):
    """An owning executor's already-classified public run failure."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.public_message = message


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "MAX_CHECKPOINT_BYTES",
    "AnswerRunCancelledError",
    "AnswerRunFailedError",
    "CheckpointError",
    "CheckpointErrorKind",
    "RunExecutionError",
    "RunSchemaError",
]
