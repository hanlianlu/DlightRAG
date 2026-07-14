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
most {allowed_count}. When you select an image, fold its salient visual details \
(objects, visible text, chart/table cues from its description) into the rewritten \
standalone query so retrieval can find related documents. Return the rewritten \
standalone query and any filters as usual.

Prior images (id | turn | ordinal | description):
{catalog_lines}
"""


PLANNER_CURRENT_IMAGE_TEMPLATE = """\

The user attached image(s) with the current message (described below). Fold their \
salient content -- visible text, objects, chart/table cues, identifiers -- into the \
rewritten standalone query, the bm25 keywords, and any metadata filters you are confident \
about, even when there is no conversation history. Treat the query as image-grounded.

Current image descriptions:
{image_lines}
"""
