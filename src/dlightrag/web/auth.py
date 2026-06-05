# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web UI authentication using the global DlightRAG auth mode."""

from __future__ import annotations

from collections.abc import Callable
from urllib.parse import quote

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from dlightrag.api.auth import verify_bearer_token
from dlightrag.config import DlightragConfig, get_config
from dlightrag.web.deps import templates

WEB_AUTH_COOKIE = "dlightrag_web_auth"
_PUBLIC_WEB_PATHS = {"/web/login", "/web/logout"}
_WEB_COOKIE_PATH = "/web"

router = APIRouter()


def _safe_next_path(value: str | None) -> str:
    """Return a same-origin web path for post-login redirects."""
    if not value:
        return "/web/"
    cleaned = value.replace("\r", "").replace("\n", "").strip()
    if cleaned == "/web" or cleaned.startswith("/web/"):
        return cleaned
    return "/web/"


def _login_url(next_path: str) -> str:
    return f"/web/login?next={quote(_safe_next_path(next_path), safe='')}"


def _bearer_from_header(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    if auth_header:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return None


def _token_from_request(request: Request) -> tuple[str | None, str | None]:
    raw = _bearer_from_header(request)
    if raw is not None:
        return raw, "header"
    raw = request.cookies.get(WEB_AUTH_COOKIE)
    if raw:
        return raw, "cookie"
    return None, None


def _set_auth_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        key=WEB_AUTH_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path=_WEB_COOKIE_PATH,
    )


def _clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(key=WEB_AUTH_COOKIE, path=_WEB_COOKIE_PATH)


def _browser_missing_auth_response(request: Request) -> Response:
    if request.method.upper() == "GET":
        return RedirectResponse(_login_url(str(request.url.path)), status_code=303)
    return PlainTextResponse("Authentication required", status_code=401)


class WebAuthMiddleware(BaseHTTPMiddleware):
    """Protect `/web/*` with the same auth mode used by REST/MCP."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        config_getter: Callable[[], DlightragConfig] = get_config,
    ) -> None:
        super().__init__(app)
        self._config_getter = config_getter

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith("/web") or path in _PUBLIC_WEB_PATHS:
            return await call_next(request)

        cfg = self._config_getter()
        if cfg.auth_mode == "none":
            return await call_next(request)

        source: str | None = None
        try:
            raw_token, source = _token_from_request(request)
            if not raw_token:
                return _browser_missing_auth_response(request)
            verify_bearer_token(
                raw_token,
                cfg,
                default_user_id=request.headers.get("X-User-Id", "anonymous"),
            )
        except HTTPException as exc:
            if source == "cookie" and request.method.upper() == "GET":
                response = RedirectResponse(_login_url(path), status_code=303)
                _clear_auth_cookie(response)
                return response
            return PlainTextResponse(str(exc.detail), status_code=exc.status_code)

        return await call_next(request)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/web/"):
    """Render the web login form when global auth is enabled."""
    cfg = get_config()
    target = _safe_next_path(next)
    if cfg.auth_mode == "none":
        return RedirectResponse(target, status_code=303)
    return templates.TemplateResponse(
        request,
        "login.html",
        {"auth_mode": cfg.auth_mode, "next": target, "error": None},
    )


@router.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    token: str = Form(default=""),
    next: str = Form(default="/web/"),
):
    """Validate a bearer token and store it in an HttpOnly web cookie."""
    cfg = get_config()
    target = _safe_next_path(next)
    if cfg.auth_mode == "none":
        return RedirectResponse(target, status_code=303)
    try:
        verify_bearer_token(token, cfg)
    except HTTPException as exc:
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "auth_mode": cfg.auth_mode,
                "next": target,
                "error": str(exc.detail),
            },
            status_code=exc.status_code,
        )

    response = RedirectResponse(target, status_code=303)
    _set_auth_cookie(response, request, token)
    return response


@router.post("/logout")
async def logout() -> RedirectResponse:
    """Clear the web auth cookie."""
    response = RedirectResponse("/web/login", status_code=303)
    _clear_auth_cookie(response)
    return response


__all__ = ["WEB_AUTH_COOKIE", "WebAuthMiddleware", "router"]
