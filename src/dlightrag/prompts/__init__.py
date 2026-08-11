# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Centralized prompts for DlightRAG: one module per consumer.

A prompt lives in the module named after the call that sends it; `identity.py` holds the
only fragment shared across calls. This facade exports the complete prompts.
"""

from .agent import agent_control_prompt
from .answer import answer_core
from .highlight import HIGHLIGHT_BATCH_USER_PROMPT, HIGHLIGHT_SYSTEM_PROMPT
from .planner import PLANNER_IMAGE_CONTEXT_GUIDANCE, PLANNER_SYSTEM_PROMPT
from .rerank import LISTWISE_RERANK_SYSTEM_PROMPT

__all__ = [
    "HIGHLIGHT_BATCH_USER_PROMPT",
    "HIGHLIGHT_SYSTEM_PROMPT",
    "LISTWISE_RERANK_SYSTEM_PROMPT",
    "PLANNER_IMAGE_CONTEXT_GUIDANCE",
    "PLANNER_SYSTEM_PROMPT",
    "agent_control_prompt",
    "answer_core",
]
