# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Query Planning and Analysis prompts."""

PLANNER_IMAGE_CONTEXT_GUIDANCE = """\
Use `current_images` as current-turn retrieval context. When `preserve_query` is true,
keep `standalone_query` unchanged and use relevant visual details only for BM25 terms or
metadata filters. Otherwise fold relevant details into the standalone and BM25 queries.
"""
