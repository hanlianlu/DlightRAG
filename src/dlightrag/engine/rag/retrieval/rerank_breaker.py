# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Circuit breaker that stops queries waiting on a reranker that keeps failing."""

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from dlightrag.engine.ai.providers.base import provider_status_code

logger = logging.getLogger(__name__)

# Stable low-level mechanics: code defaults, not configuration.
_FAILURES_TO_OPEN = 3
_OPEN_SECONDS = 30.0
_MAX_OPEN_SECONDS = 300.0


class RerankCircuitOpen(RuntimeError):
    """The reranker is being left alone for a while, so this call never reached it."""


@dataclass(frozen=True, slots=True)
class _Permit:
    """One admitted call: the period it was admitted in, and whether it is the trial."""

    period: int
    trial: bool


def _is_rejection(exc: Exception) -> bool:
    """Whether the reranker answered at once with a 4xx that refuses this one request.

    A 4xx says the request was too long, malformed, or not authorized; the reranker
    is up and replied. 408 and 429 are the exceptions: the reranker is asking to be
    left alone.
    """
    status = provider_status_code(exc)
    return status is not None and status < 500 and status not in (408, 429)


class RerankBreaker:
    """Stop calling a failing reranker for a while, then let one call test it.

    Closed, every call goes through and the breaker counts consecutive failures. The
    third opens it: for 30 s calls fail at once with ``RerankCircuitOpen``, so queries
    stop waiting on a reranker whose own timeout has just run out three times in a row.
    When the time is up exactly one call goes through as a trial while the others still
    fail at once. A trial that is answered closes the breaker; one that fails opens it
    again for twice as long, up to five minutes. Any answer resets the count and the wait.

    A 4xx other than 408 and 429 counts as an answer, never as a failure: the reranker
    replied at once, so it resets the count and can close a trial. Every other exception
    is a failure, a deadline's ``TimeoutError`` among them. A cancelled call gives no
    verdict, and a cancelled trial gives its place to the next call. A call counts only
    in the period it was admitted in, so the slow failures of calls that began before
    the breaker opened change nothing afterwards.

    One breaker serves every caller of one reranker endpoint. It belongs to one event
    loop and changes state only between awaits, so it takes no lock.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._failures = 0
        self._open_seconds = _OPEN_SECONDS
        # None while closed; otherwise when the open time ends, which a trial may outlive.
        self._open_until: float | None = None
        self._trial_running = False
        # Advances whenever the breaker opens or closes.
        self._period = 0

    async def call[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run one reranker call, or refuse it with ``RerankCircuitOpen``."""
        permit = self._admit()
        # True: the reranker answered. False: it failed. None: no verdict (cancelled).
        answered: bool | None = None
        try:
            result = await operation()
            answered = True
            return result
        except Exception as exc:
            answered = _is_rejection(exc)
            raise
        finally:
            self._settle(permit, answered)

    def _admit(self) -> _Permit:
        if self._open_until is None:
            return _Permit(self._period, trial=False)
        if self._trial_running:
            raise RerankCircuitOpen("rerank circuit is half-open: a trial call is in flight")
        remaining = self._open_until - self._clock()
        if remaining > 0:
            raise RerankCircuitOpen(f"rerank circuit is open for another {remaining:.0f}s")
        self._trial_running = True
        logger.info("Rerank circuit half-open; one call will test the reranker")
        return _Permit(self._period, trial=True)

    def _settle(self, permit: _Permit, answered: bool | None) -> None:
        if permit.trial:
            self._trial_running = False
            if answered is None:
                # The open time has run out, so the next call becomes the trial.
                return
            if answered:
                self._close()
            else:
                self._open_seconds = min(self._open_seconds * 2, _MAX_OPEN_SECONDS)
                self._open_until = self._clock() + self._open_seconds
                logger.warning(
                    "Rerank trial call failed; skipping rerank for another %.0fs",
                    self._open_seconds,
                )
            return
        if answered is None or permit.period != self._period:
            return
        if answered:
            self._failures = 0
            return
        self._failures += 1
        if self._failures >= _FAILURES_TO_OPEN:
            self._period += 1
            self._open_until = self._clock() + self._open_seconds
            logger.warning(
                "Rerank circuit opened after %d consecutive failures; skipping rerank for %.0fs",
                self._failures,
                self._open_seconds,
            )

    def _close(self) -> None:
        self._period += 1
        self._open_until = None
        self._failures = 0
        self._open_seconds = _OPEN_SECONDS
        logger.info("Rerank circuit closed; the reranker answered the trial call")


__all__ = ["RerankBreaker", "RerankCircuitOpen"]
