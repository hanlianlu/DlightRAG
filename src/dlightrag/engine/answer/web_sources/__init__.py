# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Public Web resource acquisition owned by Answer."""

from dlightrag.engine.answer.web_sources.contracts import (
    WebEffort,
    WebExtractProvider,
    WebExtractResult,
    WebSearchHit,
    WebSearchProvider,
    WebSearchRequest,
    WebSearchResult,
    WebSourceUnavailable,
)
from dlightrag.engine.answer.web_sources.exa import ExaWebSource
from dlightrag.engine.answer.web_sources.service import WebSourceService
from dlightrag.engine.answer.web_sources.tavily import TavilyWebSource

__all__ = [
    "ExaWebSource",
    "TavilyWebSource",
    "WebEffort",
    "WebExtractProvider",
    "WebExtractResult",
    "WebSearchHit",
    "WebSearchProvider",
    "WebSearchRequest",
    "WebSearchResult",
    "WebSourceService",
    "WebSourceUnavailable",
]
