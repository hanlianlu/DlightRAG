# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider- and storage-neutral durable runtime contracts."""

from typing import Literal, TypeAlias

AnswerRunStatus: TypeAlias = Literal[  # noqa: UP040 - preserve the inline OpenAPI enum
    "queued", "running", "succeeded", "failed", "cancelled"
]
ANSWER_RUN_PHASES: tuple[str, ...] = (
    "routing",
    "planning",
    "searching",
    "researching",
    "generating",
)
AnswerRunPhase: TypeAlias = Literal[  # noqa: UP040 - preserve the inline OpenAPI enum
    "routing", "planning", "searching", "researching", "generating"
]

__all__ = ["ANSWER_RUN_PHASES", "AnswerRunPhase", "AnswerRunStatus"]
