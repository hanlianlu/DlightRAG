# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer execution facade: acceptance planning and accepted-run execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .acceptance import research_history_input_measure
    from .executor import (
        AnswerExecutionStore,
        AnswerExecutor,
        AnswerExecutorSettings,
        AnswerResourceResolver,
        AnswerResourceSettings,
        OrchestratorRun,
        ResolvedAnswerResources,
        answer_trace_output,
    )

__all__ = [
    "AnswerExecutionStore",
    "AnswerExecutor",
    "AnswerExecutorSettings",
    "AnswerResourceResolver",
    "AnswerResourceSettings",
    "OrchestratorRun",
    "ResolvedAnswerResources",
    "answer_trace_output",
    "research_history_input_measure",
]


def __getattr__(name: str) -> Any:
    # Importing neutral execution contracts must not initialize the executor.
    if name == "research_history_input_measure":
        from .acceptance import research_history_input_measure

        return research_history_input_measure
    if name in {
        "AnswerExecutionStore",
        "AnswerExecutor",
        "AnswerExecutorSettings",
        "AnswerResourceResolver",
        "AnswerResourceSettings",
        "OrchestratorRun",
        "ResolvedAnswerResources",
        "answer_trace_output",
    }:
        from . import executor

        return getattr(executor, name)
    raise AttributeError(name)
