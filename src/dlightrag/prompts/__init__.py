# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Centralized prompts for DlightRAG."""

from .planner import (
    PLANNER_HISTORY_TEMPLATE,
    PLANNER_NO_HISTORY_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
)
from .rag import (
    ANSWER_CORE,
    HIGHLIGHT_SYSTEM_PROMPT,
    HIGHLIGHT_USER_PROMPT,
    VISUAL_RERANK_PROMPT,
    get_answer_system_prompt,
)

__all__ = [
    # planner
    "PLANNER_SYSTEM_PROMPT",
    "PLANNER_HISTORY_TEMPLATE",
    "PLANNER_NO_HISTORY_TEMPLATE",
    # rag
    "ANSWER_CORE",
    "get_answer_system_prompt",
    "VISUAL_RERANK_PROMPT",
    "HIGHLIGHT_SYSTEM_PROMPT",
    "HIGHLIGHT_USER_PROMPT",
]
