# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Query Planning and Analysis prompts."""

PLANNER_IMAGE_CONTEXT_GUIDANCE = """\
When `prior_images` is present, select only referenced ids in
`selected_history_image_ids`, ordered by relevance and capped by
`limits.history_images`. Use `current_images` as current-turn context; do not select
them as history. Fold relevant visual details into the standalone and BM25 queries.
"""

PLANNER_EXTERNAL_SEARCH_GUIDANCE = """\
Also set `external_query` to a search query when the answer turns on a moment -- current
events, prices, releases, "latest", "now" -- or when the request explicitly asks to look
something up. Otherwise leave it null. Nothing here describes the indexed corpus or the
date, so judge the request alone and never guess at what the corpus covers.
"""
