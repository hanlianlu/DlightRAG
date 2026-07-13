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

PLANNER_WEB_SELECTION_TEMPLATE = """\

You are also given a catalog of images shared earlier in this conversation. When \
the current message refers to earlier images (e.g. "the second revenue chart", \
"the diagram from before"), select the matching image ids by comparing the user's \
wording against each image's description. Never invent ids; choose only from the \
catalog below. Return them in `selected_history_image_ids`, most relevant first, at \
most {allowed_count}. Still return the rewritten standalone query and any filters as usual.

Prior images (id | turn | ordinal | description):
{catalog_lines}
"""
