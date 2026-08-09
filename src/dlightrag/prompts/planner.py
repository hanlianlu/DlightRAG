# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Query Planning and Analysis prompts."""

PLANNER_IMAGE_CONTEXT_GUIDANCE = """\
Use `current_images` as current-turn context. Fold relevant visual details into the
standalone and BM25 queries.
"""
