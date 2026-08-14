# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""DlightRAG tool adapters, caching, and per-run composition.

The resource adapters stay behind ``dlightrag.core.tools.resources`` because
importing them pulls the document conversion and visual inspection stack, which
a run without registered resources never needs.
"""

from dlightrag.core.tools.cache import ExactCallCache
from dlightrag.core.tools.composition import compose_research_tools
from dlightrag.core.tools.search import KnowledgeRetrieval, SearchInput, WebSearch

__all__ = [
    "ExactCallCache",
    "KnowledgeRetrieval",
    "SearchInput",
    "WebSearch",
    "compose_research_tools",
]
