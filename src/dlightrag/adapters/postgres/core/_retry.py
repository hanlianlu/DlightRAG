# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded exponential backoff for transient PostgreSQL failures."""

import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


async def retry_transient[T](
    operation: Callable[[], Awaitable[T]],
    *,
    is_transient: Callable[[Exception], bool],
    attempts: int,
    backoff: float,
    backoff_max: float,
    subject: str,
) -> T:
    """Run ``operation``, repeating it after a transient failure with a doubling pause.

    The pause before attempt ``n + 1`` is ``backoff * 2 ** (n - 1)``, capped at
    ``backoff_max``. A failure ``is_transient`` does not name propagates at once, and so does
    the last attempt's, whatever it is. ``subject`` names the caller in the warning each
    retried failure logs.
    """
    for attempt in range(1, attempts):
        try:
            return await operation()
        except Exception as exc:
            if not is_transient(exc):
                raise
            logger.warning(
                "%s transient PostgreSQL error on attempt %d/%d: %r",
                subject,
                attempt,
                attempts,
                exc,
            )
            sleep_for = min(backoff * (2 ** (attempt - 1)), backoff_max)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
    return await operation()
