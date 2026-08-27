# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Process health state shared by composition and status interfaces."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping

type ReadinessProbe = Callable[[], Awaitable[str | None]]


class _ReadinessCache:
    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = max(0.0, float(ttl_seconds))
        self._deadline = 0.0
        self._detail: str | None = None
        self._probe: asyncio.Task[str | None] | None = None
        self._generation = 0

    async def detail(self, probe: ReadinessProbe) -> str | None:
        if time.monotonic() < self._deadline:
            return self._detail
        probing = self._probe
        if probing is None:
            generation = self._generation
            probing = self._probe = asyncio.ensure_future(probe())
            probing.add_done_callback(
                lambda task, generation=generation: self._memoize(task, generation)
            )
        return await asyncio.shield(probing)

    def _memoize(self, probing: asyncio.Task[str | None], generation: int) -> None:
        if self._probe is probing:
            self._probe = None
        if probing.cancelled() or probing.exception() is not None:
            return
        if generation != self._generation:
            return
        self._detail = probing.result()
        self._deadline = time.monotonic() + self._ttl

    def invalidate(self) -> None:
        self._generation += 1
        self._deadline = 0.0
        self._probe = None


class ApplicationHealth:
    """Own process state and aggregate it with one injected readiness probe."""

    def __init__(
        self,
        *,
        readiness_probe: ReadinessProbe | None,
        readiness_cache_seconds: float = 2.0,
    ) -> None:
        self._readiness_probe = readiness_probe
        self._readiness = _ReadinessCache(readiness_cache_seconds)
        self._ready = False
        self._degraded = False
        self._closed = False
        self._warnings: list[str] = []
        self._answer_image_capability: dict[str, object] = {
            "status": "unknown",
            "effective_max_images": 0,
            "configured_ceiling": 0,
            "model": None,
        }

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def is_degraded(self) -> bool:
        return self._degraded

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    @property
    def answer_image_capability(self) -> Mapping[str, object]:
        return dict(self._answer_image_capability)

    def add_warning(self, warning: str) -> None:
        if warning and warning not in self._warnings:
            self._warnings.append(warning)

    def mark_ready(self) -> None:
        if self._closed:
            return
        self._ready = True
        self._degraded = False
        self._readiness.invalidate()

    def mark_degraded(self, warning: str | None = None) -> None:
        if self._closed:
            return
        self._ready = False
        self._degraded = True
        if warning:
            self.add_warning(warning)
        self._readiness.invalidate()

    def mark_closed(self) -> None:
        self._ready = False
        self._closed = True
        self._readiness.invalidate()

    def set_answer_image_capability(self, summary: Mapping[str, object]) -> None:
        self._answer_image_capability = dict(summary)

    async def readiness_detail(self) -> str | None:
        if not self._ready or self._closed:
            self._readiness.invalidate()
            return "RAG service is not ready"
        if self._readiness_probe is None:
            return None
        return await self._readiness.detail(self._readiness_probe)


__all__ = ["ApplicationHealth", "ReadinessProbe"]
