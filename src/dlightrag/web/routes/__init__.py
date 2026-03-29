# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routers package."""

from fastapi import APIRouter

from .chat import router as chat_router
from .files import router as files_router
from .workspaces import router as workspaces_router

router = APIRouter(prefix="/web", tags=["web"])
router.include_router(chat_router)
router.include_router(files_router)
router.include_router(workspaces_router)

__all__ = ["router"]
