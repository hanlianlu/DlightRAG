# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One provider, several interchangeable credentials.

A provider adapter is written for one API key. This wrapper gives the same provider
name a pool of them: Search and Extract each rotate over the pool, and the ordered
provider chain falls through to the next provider only once every credential this
operation could try has refused. That keeps one source's quota multiplication inside
the source, where the chain's own contract — first provider that answers supplies the
whole result — needs no change.

The wrapper is separate from the adapters on purpose: it knows the provider-neutral
contract and the two failure classes, never a provider's request shape, so another
provider joins by passing its adapter factory and its secrets.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, TypeVar

from dlightrag.engine.answer.web_sources.contracts import (
    WebExtractResult,
    WebSearchRequest,
    WebSearchResult,
    WebSourceUnavailable,
)
from dlightrag.engine.credential_rotation import (
    DEFAULT_COOLDOWN_SECONDS,
    Credential,
    CredentialPool,
)

logger = logging.getLogger(__name__)

#: The reasons that mean "not this credential" rather than "not this call": a key that
#: refused on credentials, credit, or a rate has nothing to say to the next call either.
#: A timeout, an unreachable host, or one malformed body is about the call, so the
#: credential stays eligible and only the rotation moves on.
_CREDENTIAL_COOLING_REASONS = frozenset({"unauthorized", "payment_required", "rate_limited"})

#: What one provider's adapter offers. The pool builds one per credential, so closing the
#: wrapper closes them all.
ResultT = TypeVar("ResultT", WebSearchResult, WebExtractResult)


class KeyedWebSourceAdapter(Protocol):
    """The part of a Web source adapter the rotation drives, for either operation."""

    async def search(self, request: WebSearchRequest) -> WebSearchResult: ...
    async def extract(self, url: str) -> WebExtractResult: ...
    async def aclose(self) -> None: ...


class RotatingWebSource:
    """A Web source served by a pool of interchangeable API keys.

    ``name`` is the provider name the chain and the acquisition metadata already use; the
    credential that answered is an operator fact that rides the log line, never the
    model's text. Search and Extract rotate independently because a provider out of
    search credit may still serve an extraction, as the chain already assumes.
    """

    def __init__(
        self,
        *,
        name: str,
        secrets: Sequence[str],
        build: Callable[[str], KeyedWebSourceAdapter],
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.name = name
        self._search_pool = CredentialPool(secrets, cooldown_seconds=cooldown_seconds, clock=clock)
        self._extract_pool = CredentialPool(
            tuple(credential.secret for credential in self._search_pool.credentials),
            cooldown_seconds=cooldown_seconds,
            clock=clock,
        )
        # Both pools are built from the same secrets and numbers, so one index names the
        # same key in each and one map of clients serves both operations.
        self._adapters = {
            credential.index: build(credential.secret)
            for credential in self._search_pool.credentials
        }
        self._closed = False

    async def search(self, request: WebSearchRequest) -> WebSearchResult:
        return await self._run(
            operation="search",
            pool=self._search_pool,
            attempt=lambda adapter: adapter.search(request),
        )

    async def extract(self, url: str) -> WebExtractResult:
        return await self._run(
            operation="extract",
            pool=self._extract_pool,
            attempt=lambda adapter: adapter.extract(url),
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        results = await asyncio.gather(
            *(adapter.aclose() for adapter in self._adapters.values()),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("Failed to close Web source credential", exc_info=result)

    async def _run(
        self,
        *,
        operation: str,
        pool: CredentialPool,
        attempt: Callable[[KeyedWebSourceAdapter], Awaitable[ResultT]],
    ) -> ResultT:
        """Ask the pool's credentials in rotation order and hand back the first answer.

        When every credential is cooling the pool yields one probe, so the provider's
        recovery is noticed without waiting for a timer nobody reads; a failed probe ends
        the walk, because nothing else this operation can try is eligible. Each failed
        attempt is logged once, with the credential's position — never its secret — so an
        operator can tell a dead key from a dead provider.
        """
        failures: list[WebSourceUnavailable] = []
        for lease in pool.attempts():
            adapter = self._adapters[lease.credential.index]
            try:
                result = await attempt(adapter)
            except asyncio.CancelledError:
                raise
            except WebSourceUnavailable as exc:
                failures.append(exc)
                pool.release(lease.credential, cooling=exc.reason in _CREDENTIAL_COOLING_REASONS)
                logger.warning(
                    "Web %s credential unavailable",
                    operation,
                    extra={
                        "provider": self.name,
                        "operation": operation,
                        "reason": exc.reason,
                        "status": exc.status,
                        "credential": self._label(lease.credential),
                        "probe": lease.probe,
                    },
                )
                continue
            except Exception as exc:
                failures.append(WebSourceUnavailable(self.name, operation, "error"))
                pool.release(lease.credential, cooling=False)
                logger.warning(
                    "Web %s credential failed",
                    operation,
                    exc_info=exc,
                    extra={
                        "provider": self.name,
                        "operation": operation,
                        "reason": "error",
                        "credential": self._label(lease.credential),
                        "probe": lease.probe,
                    },
                )
                continue
            pool.release(lease.credential, cooling=False)
            return result
        raise _exhausted(self.name, operation, failures)

    def _label(self, credential: Credential) -> str:
        return f"{self.name}#{credential.index}"


def _exhausted(
    name: str, operation: str, failures: list[WebSourceUnavailable]
) -> WebSourceUnavailable:
    """The one bounded summary the chain receives for a provider whose keys all refused.

    One failure speaks for itself, so a single-key provider keeps the adapter's own
    reason. When every credential of a multi-key pool refused on credentials, credit, or
    rate, ``exhausted`` is the condition to park the provider on; a pool that also saw a
    failure about the call keeps the mixed reasons, because another credential may still
    answer next time and the chain should not leave the whole provider alone for a minute.
    """
    if not failures:
        return WebSourceUnavailable(name, operation, "not_configured")
    if len(failures) == 1:
        return WebSourceUnavailable(name, operation, failures[0].reason, status=failures[0].status)
    reasons = list(dict.fromkeys(failure.reason for failure in failures))
    if all(reason in _CREDENTIAL_COOLING_REASONS for reason in reasons):
        return WebSourceUnavailable(name, operation, "exhausted")
    return WebSourceUnavailable(name, operation, "+".join(reasons))


__all__ = ["KeyedWebSourceAdapter", "RotatingWebSource"]
