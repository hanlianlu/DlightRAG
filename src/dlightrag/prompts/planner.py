# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Query Planning and Analysis prompts."""

from .guidance import PLANNER_GUIDANCE

PLANNER_SYSTEM_PROMPT = PLANNER_GUIDANCE

PLANNER_IMAGE_CONTEXT_GUIDANCE = """\
When `prior_images` is present, select only referenced ids in
`selected_history_image_ids`, ordered by relevance and capped by
`limits.history_images`. Use `current_images` as current-turn context; do not select
them as history. Fold relevant visual details into the standalone and BM25 queries.
"""
