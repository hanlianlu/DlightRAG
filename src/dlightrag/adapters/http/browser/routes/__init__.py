# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routers package."""

from fastapi import APIRouter

from dlightrag.adapters.http.browser.auth import router as auth_router

from .agent_accounts import router as agent_accounts_router
from .bootstrap import router as bootstrap_router
from .chat import page_router as chat_page_router
from .chat import router as chat_api_router
from .client_errors import router as client_errors_router
from .connections import callback_router as connections_callback_router
from .connections import router as connections_router
from .conversations import router as conversations_router
from .corpus_runs import router as corpus_runs_router
from .files import router as files_router
from .images import router as images_router
from .live_browser import router as live_browser_router
from .memory import router as memory_router
from .model_catalogue import router as model_catalogue_router
from .overview import router as overview_router
from .privacy import router as privacy_router
from .run_activity import router as run_activity_router
from .runtime_policies import router as runtime_policies_router
from .schedules import router as schedules_router
from .skills import router as skills_router
from .step_search import router as step_search_router
from .video_playback import router as video_playback_router
from .workspaces import router as workspaces_router

router = APIRouter(prefix="/web", tags=["web"])
router.include_router(auth_router)
router.include_router(connections_callback_router)
router.include_router(chat_page_router)

api_router = APIRouter(prefix="/api")
api_router.include_router(bootstrap_router)
api_router.include_router(chat_api_router)
api_router.include_router(conversations_router)
api_router.include_router(run_activity_router)
api_router.include_router(corpus_runs_router)
api_router.include_router(images_router)
api_router.include_router(live_browser_router)
api_router.include_router(files_router)
api_router.include_router(memory_router)
api_router.include_router(schedules_router)
api_router.include_router(connections_router)
api_router.include_router(agent_accounts_router)
api_router.include_router(privacy_router)
api_router.include_router(workspaces_router)
api_router.include_router(model_catalogue_router)
api_router.include_router(overview_router)
api_router.include_router(runtime_policies_router)
api_router.include_router(skills_router)
api_router.include_router(step_search_router)
api_router.include_router(video_playback_router)
api_router.include_router(client_errors_router)
router.include_router(api_router)

__all__ = ["router"]
