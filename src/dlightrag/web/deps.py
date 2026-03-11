"""FastAPI dependency injection for web routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import Cookie, Request
from fastapi.templating import Jinja2Templates

_TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

DEFAULT_WORKSPACE = "default"


def get_workspace(dlightrag_workspace: str = Cookie(default=DEFAULT_WORKSPACE)) -> str:
    """Read current workspace from cookie."""
    return dlightrag_workspace


def get_manager(request: Request):
    """Get RAGServiceManager from app state."""
    return request.app.state.manager
