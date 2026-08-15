# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider- and storage-neutral durable runtime contracts."""

from typing import Literal, TypeAlias

AnswerRunStatus: TypeAlias = Literal[  # noqa: UP040 - preserve the inline OpenAPI enum
    "queued", "running", "succeeded", "failed", "cancelled"
]
AnswerRunPhase: TypeAlias = Literal[  # noqa: UP040 - preserve the inline OpenAPI enum
    "planning", "searching", "researching", "generating"
]

__all__ = ["AnswerRunPhase", "AnswerRunStatus"]
