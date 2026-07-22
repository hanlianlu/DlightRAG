# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Centralized prompts for DlightRAG."""

from .guidance import (
    ANSWER_CONTEXT_GUIDANCE,
    CITATION_GUIDANCE,
    HIGHLIGHT_BATCH_USER_PROMPT,
    HIGHLIGHT_GUIDANCE,
    LISTWISE_RERANK_SYSTEM_PROMPT,
    PLANNER_GUIDANCE,
    RERANK_GUIDANCE,
)
from .identity import CORE_IDENTITY
from .planner import (
    PLANNER_IMAGE_CONTEXT_GUIDANCE,
    PLANNER_SYSTEM_PROMPT,
)
from .rag import (
    ANSWER_CORE,
    HIGHLIGHT_SYSTEM_PROMPT,
    get_answer_system_prompt,
)
from .web_planner import (
    WEB_PLANNER_SYSTEM_PROMPT,
)

__all__ = [
    # identity
    "CORE_IDENTITY",
    # guidance
    "ANSWER_CONTEXT_GUIDANCE",
    "CITATION_GUIDANCE",
    "PLANNER_GUIDANCE",
    "RERANK_GUIDANCE",
    "LISTWISE_RERANK_SYSTEM_PROMPT",
    "HIGHLIGHT_GUIDANCE",
    "HIGHLIGHT_BATCH_USER_PROMPT",
    # planner
    "PLANNER_SYSTEM_PROMPT",
    "PLANNER_IMAGE_CONTEXT_GUIDANCE",
    # web planner
    "WEB_PLANNER_SYSTEM_PROMPT",
    # rag
    "ANSWER_CORE",
    "get_answer_system_prompt",
    "HIGHLIGHT_SYSTEM_PROMPT",
]
