# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routers package."""

from fastapi import APIRouter

from dlightrag.web.auth import router as auth_router

from .chat import router as chat_router
from .conversations import router as conversations_router
from .files import router as files_router
from .images import router as images_router
from .workspaces import router as workspaces_router

router = APIRouter(prefix="/web", tags=["web"])
router.include_router(auth_router)
router.include_router(chat_router)
router.include_router(conversations_router)
router.include_router(images_router)
router.include_router(files_router)
router.include_router(workspaces_router)

__all__ = ["router"]
