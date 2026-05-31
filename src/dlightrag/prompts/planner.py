# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Query Planning and Analysis prompts."""

from .guidance import PLANNER_GUIDANCE
from .identity import CORE_IDENTITY

PLANNER_SYSTEM_PROMPT = "\n\n".join([CORE_IDENTITY, PLANNER_GUIDANCE])

PLANNER_HISTORY_TEMPLATE = """\
Conversation history:
{history_text}

Current follow-up message: {query}

"""

PLANNER_NO_HISTORY_TEMPLATE = """\
Query: {query}

"""
