# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""FastAPI app factory for DlightRAG REST server.

Entry point: dlightrag-api
All endpoint logic lives in routes.py; this module handles app lifecycle,
middleware, exception handlers, and router mounting.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dlightrag.api.middleware import RequestIdLogFilter, RequestIdMiddleware
from dlightrag.api.models import ErrorDetail
from dlightrag.api.routes import router
from dlightrag.core.servicemanager import RAGServiceManager, RAGServiceUnavailableError

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Lifespan
# ═══════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        _app.state.manager = await RAGServiceManager.create()
    except Exception:
        logger.exception("Failed to initialize RAG service manager")
        raise
    yield
    await _app.state.manager.close()


# ═══════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════


def create_app(*, include_web: bool = True) -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="dlightrag",
        description="DlightRAG - Dual-mode (Caption based & Unified representation based) multi-modal RAG service",
        version=__import__("dlightrag").__version__,
        lifespan=lifespan,
    )

    # -- Request ID middleware (outermost — runs first) --
    application.add_middleware(RequestIdMiddleware)

    # -- CORS middleware --
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Exception handlers --
    @application.exception_handler(RAGServiceUnavailableError)
    async def rag_unavailable_handler(
        request: Request,  # noqa: ARG001
        exc: RAGServiceUnavailableError,
    ) -> JSONResponse:
        body = ErrorDetail(detail=exc.detail, error_type="unavailable")
        return JSONResponse(status_code=503, content=body.model_dump())

    # -- API routes --
    application.include_router(router)

    # -- Web frontend (optional — only if jinja2 is installed) --
    if include_web:
        try:
            import importlib.util

            if importlib.util.find_spec("jinja2"):
                from fastapi.staticfiles import StaticFiles

                from dlightrag.web.deps import _TEMPLATE_DIR
                from dlightrag.web.routes import router as web_router

                application.include_router(web_router)
                _static_dir = _TEMPLATE_DIR.parent / "static"
                if _static_dir.exists():
                    application.mount(
                        "/static", StaticFiles(directory=str(_static_dir)), name="static"
                    )
        except ImportError:
            pass  # Web frontend not installed

    return application


# Backward compat: module-level app for uvicorn and tests
app = create_app()


def get_app() -> FastAPI:
    """ASGI factory entry point (e.g. uvicorn dlightrag.api.server:get_app --factory)."""
    return create_app()


def main() -> None:
    """Entry point for dlightrag-api."""
    import argparse

    import uvicorn
    from dotenv import load_dotenv

    from dlightrag.config import get_config

    parser = argparse.ArgumentParser(description="dlightrag REST API server")
    parser.add_argument("--env-file", help="Path to .env configuration file")
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file, override=True)

    config = get_config()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s",
    )
    # Attach request ID filter to root logger so all loggers inherit it
    logging.getLogger().addFilter(RequestIdLogFilter())

    uvicorn.run(
        "dlightrag.api.server:app",
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level,
    )


__all__ = ["app", "create_app", "get_app", "main"]
