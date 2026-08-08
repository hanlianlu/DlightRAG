# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Centralized prompts for DlightRAG."""

from .guidance import (
    HIGHLIGHT_BATCH_USER_PROMPT,
    HIGHLIGHT_SYSTEM_PROMPT,
    LISTWISE_RERANK_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)
from .planner import (
    PLANNER_EXTERNAL_SEARCH_GUIDANCE,
    PLANNER_IMAGE_CONTEXT_GUIDANCE,
)
from .rag import (
    ANSWER_CORE,
)
from .web_planner import (
    WEB_PLANNER_SYSTEM_PROMPT,
)

__all__ = [
    # guidance
    "LISTWISE_RERANK_SYSTEM_PROMPT",
    "HIGHLIGHT_BATCH_USER_PROMPT",
    # planner
    "PLANNER_SYSTEM_PROMPT",
    "PLANNER_IMAGE_CONTEXT_GUIDANCE",
    "PLANNER_EXTERNAL_SEARCH_GUIDANCE",
    # web planner
    "WEB_PLANNER_SYSTEM_PROMPT",
    # rag
    "ANSWER_CORE",
    "HIGHLIGHT_SYSTEM_PROMPT",
]
