# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Ordered provider failover for public Web search and URL extraction."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from dlightrag.engine.answer.web_sources.contracts import (
    WebExtractProvider,
    WebExtractResult,
    WebSearchProvider,
    WebSearchRequest,
    WebSearchResult,
    WebSourceUnavailable,
)
from dlightrag.engine.public_http import public_network_admission

logger = logging.getLogger(__name__)

#: How long a provider that said it is out of reach is left alone. Long enough that a burst of
#: traffic does not pay a dead provider's round trip on every call, short enough that a top-up or a
#: lifted rate limit is picked up within a minute of it happening.
_COOLDOWN_SECONDS = 60.0

#: The reasons that mean "not now" rather than "not this once": a provider that refused on
#: credentials, credit, or rate has nothing to say to the next call either. A timeout, an
#: unreachable host, or one malformed body is about the call, and the next one may well answer.
#: ``throttled`` is absent on purpose: it is a per-second limit the adapter has already waited out
#: once, so the next call may well pass, and a minute's silence would cost more than it saves.
_COOLING_REASONS = frozenset({"unauthorized", "payment_required", "rate_limited"})


class WebSourceService:
    """Run independent, explicitly ordered Search and Extract chains.

    A chain is walked in order and the first provider that answers supplies the whole result, so
    the fallback is availability, never a blend of two rankings. A provider that is out of reach is
    left alone for a minute (see ``_COOLING_REASONS``) and the chain reports it as cooling, which
    is what keeps a provider that has run out of credit from being asked on every single call.
    """

    def __init__(
        self,
        *,
        search_providers: tuple[WebSearchProvider, ...] = (),
        extract_providers: tuple[WebExtractProvider, ...] = (),
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._search_providers = search_providers
        self._extract_providers = extract_providers
        self._clock = clock or time.monotonic
        #: ``(operation, provider) -> the monotonic time it may be asked again``. Per operation, so
        #: a provider out of search credit still serves extraction.
        self._cooling: dict[tuple[str, str], float] = {}
        self._closed = False

    @property
    def search_enabled(self) -> bool:
        return bool(self._search_providers)

    async def search(self, request: WebSearchRequest) -> WebSearchResult:
        return await self._walk(
            operation="search",
            providers=self._search_providers,
            attempt=lambda provider: provider.search(request),
        )

    async def extract(
        self,
        url: str,
        *,
        providers: tuple[str, ...],
    ) -> WebExtractResult:
        """Extract ``url`` with the configured providers that ``providers`` names, in order.

        A chain that places the Agent Browser between hosted providers asks each run of
        them separately, so the browser sits where the order puts it.
        """
        return await self._walk(
            operation="extract",
            providers=tuple(item for item in self._extract_providers if item.name in providers),
            attempt=lambda provider: provider.extract(url),
        )

    async def _walk[R: (WebSearchResult, WebExtractResult)](
        self,
        *,
        operation: str,
        providers: tuple[Any, ...],
        attempt: Callable[[Any], Awaitable[R]],
    ) -> R:
        """Ask each provider in the chain's order and hand back the first answer.

        Every failed attempt is logged once, here, with the provider, the operation, the reason
        and the status it answered with: that line is what a dashboard counts and what an operator
        reads when search goes quiet. When every provider of an operation is cooling, the one that
        recovers first is asked anyway — that probe is how a top-up or a lifted rate limit is
        noticed without waiting for a timer nobody reads.
        """
        failures: list[WebSourceUnavailable] = []
        for provider in self._answering(operation, providers, failures):
            try:
                async with public_network_admission():
                    result = await attempt(provider)
            except asyncio.CancelledError:
                raise
            except WebSourceUnavailable as exc:
                failures.append(exc)
                self._cool_after(operation, exc)
                logger.warning(
                    "Web %s provider unavailable",
                    operation,
                    extra={
                        "provider": exc.provider,
                        "operation": operation,
                        "reason": exc.reason,
                        "status": exc.status,
                    },
                )
                continue
            except Exception as exc:
                failures.append(WebSourceUnavailable(provider.name, operation, "error"))
                logger.warning(
                    "Web %s provider failed",
                    operation,
                    exc_info=exc,
                    extra={
                        "provider": provider.name,
                        "operation": operation,
                        "reason": "error",
                    },
                )
                continue
            self._cooling.pop((operation, provider.name), None)
            return replace(result, degradation=_degradation(failures, provider.name))
        raise _exhausted(operation, failures)

    def _answering(
        self,
        operation: str,
        providers: tuple[Any, ...],
        failures: list[WebSourceUnavailable],
    ) -> tuple[Any, ...]:
        """The providers to ask now, in order, recording the ones left out as cooling."""
        now = self._clock()
        cooling = {
            provider.name: self._cooling[(operation, provider.name)]
            for provider in providers
            if self._cooling.get((operation, provider.name), 0.0) > now
        }
        if not cooling:
            return providers
        if len(cooling) == len(providers):
            # Everything is out, so exactly one probe goes out: whoever recovers first, with the
            # chain's own order settling a tie. Without it the chain would stay dark until a timer
            # nobody reads expired, and a provider that came back a second after a burst would
            # still be skipped.
            probed = min(cooling, key=lambda name: cooling[name])
            failures.extend(
                WebSourceUnavailable(provider.name, operation, "cooling")
                for provider in providers
                if provider.name != probed
            )
            return tuple(provider for provider in providers if provider.name == probed)
        failures.extend(
            WebSourceUnavailable(provider.name, operation, "cooling")
            for provider in providers
            if provider.name in cooling
        )
        return tuple(provider for provider in providers if provider.name not in cooling)

    def _cool_after(self, operation: str, failure: WebSourceUnavailable) -> None:
        """Leave a provider alone for a while when it said it is out of reach, not out of luck."""
        if failure.reason not in _COOLING_REASONS:
            return
        self._cooling[(operation, failure.provider)] = self._clock() + _COOLDOWN_SECONDS

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
