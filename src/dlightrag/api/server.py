# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""FastAPI app factory for DlightRAG REST server.

Entry point: dlightrag-api
All endpoint logic lives in routes.py; this module handles app lifecycle,
middleware, exception handlers, and router mounting.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dlightrag.api.middleware import RequestIdMiddleware, install_request_id_log_record_factory
from dlightrag.api.models import ErrorDetail
from dlightrag.api.routes import router
from dlightrag.app_state import request_config
from dlightrag.core.servicemanager import RAGServiceManager, RAGServiceUnavailableError

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Lifespan
# ═══════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        manager = await RAGServiceManager.acreate(config=request_config(_app))
    except Exception:
        logger.exception("Failed to initialize RAG service manager")
        raise
    _app.state.manager = manager
    try:
        conversation_service = getattr(_app.state, "web_conversation_service", None)
        if conversation_service is not None:
            await conversation_service.initialize()
        yield
    finally:
        await manager.aclose()


# ═══════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════


def create_app(*, include_web: bool = True) -> FastAPI:
    """Create and configure the FastAPI application."""
    from dlightrag.config import get_config

    cfg = get_config()

    application = FastAPI(
        title="dlightrag",
        description="DlightRAG - LightRAG-main unified multimodal RAG service",
        version=__import__("dlightrag").__version__,
        lifespan=lifespan,
    )
    application.state.config = cfg

    # -- Request ID middleware (outermost — runs first) --
    application.add_middleware(RequestIdMiddleware)

    # -- CORS middleware (config-driven; see DlightragConfig.cors_allow_origins) --
    # allow_credentials toggles based on origin list: browsers refuse '*' +
    # credentials, so we only enable credentials when origins are explicit.
    allow_credentials = cfg.cors_allow_origins != ["*"]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Exception handlers --

    @application.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,  # noqa: ARG001
        exc: HTTPException,
    ) -> JSONResponse:
        """Wrap every HTTPException in ErrorDetail for a uniform response schema."""
        status = exc.status_code
        if 400 <= status < 500:
            error_type = "validation" if status in {400, 422} else "auth"
        else:
            error_type = "internal"
        body = ErrorDetail(detail=str(exc.detail), error_type=error_type)
        return JSONResponse(status_code=status, content=body.model_dump())

    @application.exception_handler(RAGServiceUnavailableError)
    async def rag_unavailable_handler(
        request: Request,  # noqa: ARG001
        exc: RAGServiceUnavailableError,
    ) -> JSONResponse:
        body = ErrorDetail(detail=exc.detail, error_type="unavailable")
        return JSONResponse(status_code=503, content=body.model_dump())

    @application.exception_handler(PermissionError)
    async def permission_error_handler(
        request: Request,  # noqa: ARG001
        exc: PermissionError,
    ) -> JSONResponse:
        body = ErrorDetail(detail=str(exc), error_type="auth")
        return JSONResponse(status_code=403, content=body.model_dump())

    # -- API routes --
    application.include_router(router)

    # -- Web frontend --
    if include_web:
        from dlightrag.storage.web_conversations import PGWebConversationStore
        from dlightrag.web.auth import WebAuthMiddleware
        from dlightrag.web.conversations import WebConversationService
        from dlightrag.web.deps import _TEMPLATE_DIR
        from dlightrag.web.routes import router as web_router
        from dlightrag.web.static_files import NoCacheStaticFiles

        application.state.web_conversation_service = WebConversationService(
            store=PGWebConversationStore(),
            max_turns=cfg.web_conversations.max_turns,
            ttl_days=cfg.web_conversations.ttl_days,
        )
        application.add_middleware(WebAuthMiddleware, config_getter=lambda cfg=cfg: cfg)
        application.include_router(web_router)
        _static_dir = _TEMPLATE_DIR.parent / "static"
        if _static_dir.exists():
            application.mount(
                "/static",
                NoCacheStaticFiles(directory=str(_static_dir)),
                name="static",
            )

    return application


def get_app() -> FastAPI:
    """ASGI factory entry point (e.g. uvicorn dlightrag.api.server:get_app --factory)."""
    return create_app()


def main() -> None:
    """Entry point for dlightrag-api."""
    import argparse

    import uvicorn

    from dlightrag.config import get_config, load_config, set_config

    parser = argparse.ArgumentParser(
        description="dlightrag REST API server",
        suggest_on_error=True,
    )
    parser.add_argument("--env-file", help="Path to .env configuration file")
    args = parser.parse_args()

    if args.env_file:
        config = load_config(args.env_file)
        set_config(config)
    else:
        config = get_config()
    install_request_id_log_record_factory()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s",
    )

    uvicorn.run(
        "dlightrag.api.server:get_app",
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level,
        factory=True,
    )


__all__ = ["create_app", "get_app", "main"]
