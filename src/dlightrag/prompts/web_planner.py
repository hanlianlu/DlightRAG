# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web conversation planner prompts.

Pure prompt strings/templates for the Web-only conversation planner contract.
The Web planner reuses the shared ``PLANNER_GUIDANCE`` (which carries the
``{schema_section}{custom_keys_hint}{history_section}`` placeholders) and adds
guidance for selecting scoped history documents/images.
"""

from .guidance import PLANNER_GUIDANCE
from .identity import CORE_IDENTITY

WEB_PLANNER_EXTRA_GUIDANCE = """\
You are planning one Web conversation turn. In addition to the workspace query
above, also produce these keys:

- "selected_history_attachment_ids": Ids of prior document attachments (from the
  catalog below) that the current message refers to. Never invent ids; choose
  only from the catalog. Omit when the current message does not reference prior
  attachments -- current attachments are already in scope.
- "selected_history_image_ids": Ids of prior images (from the image catalog
  below) that the current message refers to. Same rules as attachment ids."""

WEB_PLANNER_SYSTEM_PROMPT = "\n\n".join(
    [CORE_IDENTITY, PLANNER_GUIDANCE, WEB_PLANNER_EXTRA_GUIDANCE]
)

WEB_PLANNER_HISTORY_ATTACHMENT_TEMPLATE = """\

You are given a catalog of document attachments shared earlier in this
conversation. When the current message refers to an earlier attachment (e.g.
"the contract from before", "that report"), select the matching ids by comparing
the user's wording against each attachment's filename and summary. Never invent
ids; choose only from the catalog below. Return them in
`selected_history_attachment_ids`, most relevant first, at most {allowed_count}.

Prior document attachments (id | turn | ordinal | filename | summary):
{catalog_lines}
"""

WEB_PLANNER_CURRENT_ATTACHMENT_TEMPLATE = """\

The user attached document(s) with the current message (listed below). These are
already in scope for this turn -- do not add them to
`selected_history_attachment_ids`. Fold their salient content into the standalone
query when relevant.

Current document attachments (id | filename | summary):
{catalog_lines}
"""
