# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""API routers package."""

from fastapi import APIRouter

from .files import router as files_router
from .files import serve_file
from .metadata import router as metadata_router
from .rag import router as rag_router
from .status import router as status_router

router = APIRouter()
router.include_router(status_router)
router.include_router(rag_router)
router.include_router(files_router)
router.include_router(metadata_router)

__all__ = ["router", "serve_file"]
