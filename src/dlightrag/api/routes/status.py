# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""System status and health API routes.

``/health`` is liveness only: it answers from in-process state and never touches
PostgreSQL, so an unauthenticated poll loop cannot turn it into database load.
``/ready`` owns the database and corpus verdict and memoizes it for a few
seconds, which bounds that load without hiding a role, schema, or session-mode
change for more than the cache window.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dlightrag.api.models import HealthResponse, ReadinessResponse
from dlightrag.app_state import request_config
from dlightrag.config import DlightragConfig
from dlightrag.contracts import ServiceRole
from dlightrag.core.answer.capability import answer_image_capability_summary
from dlightrag.storage.lightrag_readonly import verify_reader_corpus_session

from .deps import get_manager

logger = logging.getLogger(__name__)

router = APIRouter()

#: How long one process reuses its last readiness verdict.
READINESS_CACHE_SECONDS = 2.0


class ReadinessProbeCache:
    """Short-lived memo of one process's database and corpus readiness verdict.

    Concurrent polls that find the memo cold share one probe, so a burst costs a
    single round trip, and the probe is shielded: a client that disconnects while
    waiting cannot cancel the probe its peers are still waiting on.
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._deadline = 0.0
        self._detail: str | None = None
        self._probe: asyncio.Task[str | None] | None = None
        self._generation = 0

    async def detail(self, probe: Callable[[], Awaitable[str | None]]) -> str | None:
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
        """Drop the memo so the next probe reflects a completed transition."""
        self._generation += 1
        self._deadline = 0.0
        # Do not cancel a probe another waiter may still need. Disown it so the
        # next readiness request starts a fresh post-transition probe instead.
        self._probe = None


def _readiness_cache(request: Request) -> ReadinessProbeCache:
    cache = getattr(request.app.state, "readiness_cache", None)
    if cache is None:
        cache = ReadinessProbeCache(READINESS_CACHE_SECONDS)
        request.app.state.readiness_cache = cache
    return cache


def _not_ready(*, service_role: ServiceRole, detail: str) -> JSONResponse:
    payload = ReadinessResponse(
        status="not_ready",
        service_role=service_role,
        detail=detail,
    )
    return JSONResponse(status_code=503, content=payload.model_dump(exclude_none=True))


async def _postgres_not_ready_detail(config: DlightragConfig) -> str | None:
    """Return why PostgreSQL is unusable for this role, or ``None`` when it is usable.

    Both roles write DlightRAG operational state, so the domain session must be
    writable. Probing the session default avoids a destructive write. A reader
    additionally proves its corpus session is still read-only and still resolves
    the corpus. Details are fixed strings: ``/ready`` is unauthenticated.
    """
    from dlightrag.storage.pool import pg_pool

    try:
        read_only = await pg_pool.run_once(lambda conn: conn.fetchval("SHOW transaction_read_only"))
        if str(read_only).lower() != "off":
            raise RuntimeError("domain pool session is read-only")
    except Exception:
        logger.warning("Domain PostgreSQL readiness probe failed", exc_info=True)
        return "DlightRAG domain database session is not writable"

    if not config.is_reader:
        return None

    try:
        await verify_reader_corpus_session()
    except Exception:
        logger.warning("Reader corpus PostgreSQL readiness probe failed", exc_info=True)
        return "Reader corpus database session is not read-only or is unavailable"
    return None


@router.get("/health", response_model=HealthResponse, response_model_exclude_none=True)
async def health(request: Request) -> dict[str, object]:
    """Report process liveness and the capabilities this build exposes."""
    config = request_config(request)
    manager = get_manager(request)

    warnings = manager.get_warnings()
    status: dict[str, object] = {
        "status": "degraded" if manager.is_degraded() else "healthy",
        "rag_initialized": manager.is_ready(),
        "service_role": config.service_role,
        "crafted_by": "hllyu",
        "maintained_by": "HanlianLyu",
        "storage": {
            "vector": config.vector_storage,
            "graph": config.graph_storage,
            "kv": config.kv_storage,
        },
        "answer_image_capability": answer_image_capability_summary(manager.answer_image_capability),
    }
    if warnings:
        status["warnings"] = warnings
    return status


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    response_model_exclude_none=True,
    responses={503: {"model": ReadinessResponse}},
)
async def readiness(request: Request) -> ReadinessResponse | JSONResponse:
    """Return whether this process can accept query traffic."""
    config = request_config(request)
    manager = get_manager(request)
    cache = _readiness_cache(request)
    if not manager.is_ready():
        # Startup and schema transitions must never be answered from a verdict
        # this process reached before the manager finished coming up.
        cache.invalidate()
        return _not_ready(
            service_role=config.service_role,
            detail="RAG service is not ready",
        )

    detail = await cache.detail(lambda: _postgres_not_ready_detail(config))
    if detail is not None:
        return _not_ready(service_role=config.service_role, detail=detail)

    return ReadinessResponse(status="ready", service_role=config.service_role)
