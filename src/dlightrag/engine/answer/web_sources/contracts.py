# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-neutral contracts for public Web search and URL extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

type WebEffort = Literal["fast", "balanced", "deep"]


class WebSourceUnavailable(Exception):
    """One provider operation failed with a stable, model-safe reason."""

    def __init__(self, provider: str, operation: str, reason: str) -> None:
        self.provider = provider
        self.operation = operation
        self.reason = reason
        super().__init__(f"{provider} {operation} unavailable: {reason}")


@dataclass(frozen=True, slots=True)
class WebSearchRequest:
    """One provider-neutral search request."""

    query: str
    max_results: int = 10
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()
    start_date: str | None = None
    end_date: str | None = None
    effort: WebEffort = "balanced"


@dataclass(frozen=True, slots=True)
class WebSearchHit:
    """One immediately citable passage returned by Web search."""

    url: str
    title: str
    text: str
    published_date: str | None = None
    image_url: str | None = None
    acquisition: Literal["exa_search", "tavily_search"] = "exa_search"


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    """Normalized search output; malformed provider items may be dropped."""

    hits: tuple[WebSearchHit, ...]
    cost_dollars: float = 0.0
    provider: str = ""
    dropped_results: int = 0
    degradation: str | None = None


@dataclass(frozen=True, slots=True)
class WebExtractResult:
    """Normalized text extracted from one known public URL."""

    url: str
    text: str
    title: str | None = None
    provider: str = ""
    acquisition: Literal["exa_extract", "tavily_extract"] = "exa_extract"
    dropped_results: int = 0
    degradation: str | None = None


class WebSearchProvider(Protocol):
    name: str

    async def search(self, request: WebSearchRequest) -> WebSearchResult: ...
    async def aclose(self) -> None: ...


class WebExtractProvider(Protocol):
    name: str

    async def extract(self, url: str, *, effort: WebEffort) -> WebExtractResult: ...
    async def aclose(self) -> None: ...


__all__ = [
    "WebEffort",
    "WebExtractProvider",
    "WebExtractResult",
    "WebSearchHit",
    "WebSearchProvider",
    "WebSearchRequest",
    "WebSearchResult",
    "WebSourceUnavailable",
]
