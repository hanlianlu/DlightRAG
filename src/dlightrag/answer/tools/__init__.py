# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""DlightRAG tool adapters, caching, and per-run composition.

The resource adapters stay behind ``dlightrag.answer.tools.resources`` because
importing them pulls the document conversion and visual inspection stack, which
a run without registered resources never needs.
"""

from dlightrag.answer.tools.cache import ExactCallCache
from dlightrag.answer.tools.composition import compose_research_tools
from dlightrag.answer.tools.search import KnowledgeRetrieval, SearchInput, WebSearch

__all__ = [
    "ExactCallCache",
    "KnowledgeRetrieval",
    "SearchInput",
    "WebSearch",
    "compose_research_tools",
]
