# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Ordered provider failover for public Web search and URL extraction."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

from dlightrag.engine.answer.web_sources.contracts import (
    WebEffort,
    WebExtractProvider,
    WebExtractResult,
    WebSearchProvider,
    WebSearchRequest,
    WebSearchResult,
    WebSourceUnavailable,
)
from dlightrag.engine.public_http import public_network_admission

logger = logging.getLogger(__name__)


class WebSourceService:
    """Run independent, explicitly ordered Search and Extract chains."""

    def __init__(
        self,
        *,
        search_providers: tuple[WebSearchProvider, ...] = (),
        extract_providers: tuple[WebExtractProvider, ...] = (),
    ) -> None:
        self._search_providers = search_providers
        self._extract_providers = extract_providers
        self._closed = False

    @property
    def search_enabled(self) -> bool:
        return bool(self._search_providers)

    @property
    def extract_enabled(self) -> bool:
        return bool(self._extract_providers)

    async def search(self, request: WebSearchRequest) -> WebSearchResult:
        failures: list[WebSourceUnavailable] = []
        for provider in self._search_providers:
            try:
                async with public_network_admission():
                    result = await provider.search(request)
            except asyncio.CancelledError:
                raise
            except WebSourceUnavailable as exc:
                failures.append(exc)
                continue
            except Exception as exc:
                failures.append(WebSourceUnavailable(provider.name, "search", "error"))
                logger.warning("Web search provider %s failed", provider.name, exc_info=exc)
                continue
            return replace(result, degradation=_degradation(failures, provider.name))
        raise _exhausted("search", failures)

    async def extract(self, url: str, *, effort: WebEffort = "balanced") -> WebExtractResult:
        failures: list[WebSourceUnavailable] = []
        for provider in self._extract_providers:
            try:
                async with public_network_admission():
                    result = await provider.extract(url, effort=effort)
            except asyncio.CancelledError:
                raise
            except WebSourceUnavailable as exc:
                failures.append(exc)
                continue
            except Exception as exc:
                failures.append(WebSourceUnavailable(provider.name, "extract", "error"))
                logger.warning("Web extract provider %s failed", provider.name, exc_info=exc)
                continue
            return replace(result, degradation=_degradation(failures, provider.name))
        raise _exhausted("extract", failures)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        providers: list[WebSearchProvider | WebExtractProvider] = []
        seen: set[int] = set()
        for provider in (*self._search_providers, *self._extract_providers):
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            providers.append(provider)
        results = await asyncio.gather(
            *(provider.aclose() for provider in providers),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("Failed to close Web source provider", exc_info=result)


def _degradation(failures: list[WebSourceUnavailable], provider: str) -> str | None:
    if not failures:
        return None
    failed = ", ".join(f"{item.provider} ({item.reason})" for item in failures)
    return f"Provider fallback: {failed}; used {provider}."


def _exhausted(operation: str, failures: list[WebSourceUnavailable]) -> WebSourceUnavailable:
    if not failures:
        return WebSourceUnavailable("none", operation, "not_configured")
    reason = ",".join(f"{item.provider}:{item.reason}" for item in failures)
    return WebSourceUnavailable("all", operation, reason)


__all__ = ["WebSourceService"]
