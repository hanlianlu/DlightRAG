# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web UI authentication using the global DlightRAG auth mode."""

import base64
import binascii
import secrets
from collections.abc import Callable
from typing import Literal
from urllib.parse import quote, urlencode, urlsplit

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import SecretStr
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import HTTPConnection
from starlette.types import ASGIApp

from dlightrag.adapters.http.browser.app_shell import app_html_response
from dlightrag.adapters.http.browser.edge_identity import EdgeIdentityError, authenticate_edge
from dlightrag.adapters.http.errors import error_response
from dlightrag.application.access.authentication import (
    AuthenticationError,
    authenticate_bearer_token,
)
from dlightrag.application.access.principal import UserContext
from dlightrag.application.config.loading import get_config
from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.settings.projections import (
    authentication_settings,
    web_identity_settings,
)

WEB_AUTH_COOKIE = "dlightrag_web_auth"
WEB_CSRF_COOKIE = "dlightrag_web_csrf"
CSRF_HEADER = "X-CSRF-Token"
_PUBLIC_WEB_PATHS = {
    "/web/login",
    "/web/logout",
    # An authorization server fetches this one without a cookie. It is a static document that
    # reveals only what an authorization redirect already does; see client_metadata.py.
    "/web/oauth/connections/mcp/client-metadata",
}
_WEB_COOKIE_PATH = "/web"
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
#: What an upgrade's own scheme is called in the vocabulary of the Origin it carries.
_SAME_ORIGIN_SCHEMES = {"http": "http", "https": "https", "ws": "http", "wss": "https"}

router = APIRouter()


def _authenticate_bearer(
    raw_token: str,
    config: DlightragConfig,
    *,
    default_user_id: str = "anonymous",
) -> UserContext:
    try:
        return authenticate_bearer_token(
            raw_token,
            authentication_settings(config),
            default_user_id=default_user_id,
        )
    except AuthenticationError as exc:
        status_code = 500 if exc.kind == "verifier_misconfigured" else 401
        raise HTTPException(status_code=status_code, detail=str(exc)) from None


def _safe_next_path(value: str | None) -> str:
    """Return a same-origin web path for post-login redirects."""
    if not value:
        return "/web/"
    cleaned = value.replace("\r", "").replace("\n", "").strip()
    if cleaned == "/web":
        return "/web"
    if cleaned.startswith("/web/"):
        return "/web/" + cleaned.removeprefix("/web/")
    return "/web/"


def _login_url(next_path: str) -> str:
    return f"/web/login?next={quote(_safe_next_path(next_path), safe='')}"


def _request_next_path(request: Request) -> str:
    path = str(request.url.path)
    query = str(request.url.query)
    return f"{path}?{query}" if query else path


def _origin_tuple(value: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            return None
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None
    return parsed.scheme, parsed.hostname.lower(), port


def _request_origin(request: HTTPConnection) -> tuple[str, str, int] | None:
    # An upgrade's own URL is ws or wss, while the Origin a browser sends on it is http or
    # https: the two name the same front door, and this is what compares them.
    scheme = _SAME_ORIGIN_SCHEMES.get(request.url.scheme)
    if scheme is None:
        return None
    try:
        hostname = request.url.hostname
        if hostname is None:
            return None
        port = request.url.port or (443 if scheme == "https" else 80)
    except ValueError:
        return None
    return scheme, hostname.lower(), port


def _has_exact_same_origin(request: HTTPConnection) -> bool:
    origin = request.headers.get("Origin")
    return origin is not None and _origin_tuple(origin) == _request_origin(request)


def _bearer_from_header(request: HTTPConnection) -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    if auth_header:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return None


def _token_from_request(request: HTTPConnection) -> str | None:
    raw = _bearer_from_header(request)
    if raw is not None:
        return raw
    raw = request.cookies.get(WEB_AUTH_COOKIE)
    return _decode_cookie_token(raw) if raw else None


def _encode_cookie_token(token: str) -> str:
    return base64.urlsafe_b64encode(token.encode()).decode().rstrip("=")


def _decode_cookie_token(value: str) -> str | None:
    try:
        padded = value + ("=" * (-len(value) % 4))
        return base64.b64decode(padded, altchars=b"-_", validate=True).decode()
    except binascii.Error, UnicodeDecodeError, ValueError:
        return None


def _set_auth_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        key=WEB_AUTH_COOKIE,
        value=_encode_cookie_token(token),
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path=_WEB_COOKIE_PATH,
    )


def _clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(key=WEB_AUTH_COOKIE, path=_WEB_COOKIE_PATH)


def _ensure_csrf_cookie(request: Request, response: Response) -> None:
    """Issue the JS-readable double-submit token once per browser."""
    if request.cookies.get(WEB_CSRF_COOKIE):
        return
    response.set_cookie(
        key=WEB_CSRF_COOKIE,
        value=secrets.token_urlsafe(32),
        httponly=False,
        samesite="strict",
        secure=request.url.scheme == "https",
        path=_WEB_COOKIE_PATH,
    )


def _csrf_header_matches(request: Request) -> bool:
    cookie_token = request.cookies.get(WEB_CSRF_COOKIE)
    if not cookie_token:
        return True
    header_token = request.headers.get(CSRF_HEADER, "")
    return secrets.compare_digest(cookie_token, header_token)


def _cross_origin_rejected() -> Response:
    """Refuse a request whose origin could not be verified, in the shared envelope."""
    return error_response(403, "Cross-origin request rejected", error_kind="cross_origin_rejected")


def _reject_web_mutation(request: Request) -> bool:
    """Return True when one unsafe /web request must be rejected.

    Browsers that carry the double-submit cookie must echo it in the header;
    cookie-authenticated (paste) mutations must additionally prove exact
    same-origin via an Origin header, and any Origin header must match.
    Scripted clients without any web cookie are left to their bearer
    credentials.
    """
    if request.method.upper() not in _UNSAFE_METHODS:
        return False
    if not request.url.path.startswith("/web"):
        return False
    if not _csrf_header_matches(request):
        return True
    bearer_present = request.headers.get("Authorization", "").startswith("Bearer ")
    if (
        WEB_AUTH_COOKIE in request.cookies
        and not bearer_present
        and request.headers.get("Origin") is None
    ):
        return True
    origin = request.headers.get("Origin")
    return origin is not None and not _has_exact_same_origin(request)


def _is_top_level_navigation(request: Request) -> bool:
    """Whether the browser is loading this URL as the page it shows, a download included.

    Fetch metadata says so, wherever the URL points; a client that sends none is
    judged by the path, where only ``/web/api`` is not a page. A nested frame is
    never top-level, and a bearer header is a scripted client's credential.
    """
    if request.method.upper() != "GET" or "Authorization" in request.headers:
        return False
    mode = request.headers.get("Sec-Fetch-Mode")
    if mode is None:
        return not request.url.path.startswith("/web/api/")
    return mode == "navigate" and request.headers.get("Sec-Fetch-Dest") not in {"iframe", "frame"}


def _unauthenticated(request: Request, detail: str = "Authentication required") -> Response:
    """Send a top-level navigation to sign in; refuse anything else with a 401 envelope.

    ``fetch`` follows a redirect without telling the app, which would read the login
    page as its answer, so every other request is refused for the app to explain.
    The navigation drops a cookie that stopped working; a 401 leaves cookies alone,
    so a request still in flight cannot sign out another tab that has just signed in.
    """
    if _is_top_level_navigation(request):
        response = RedirectResponse(_login_url(_request_next_path(request)), status_code=303)
        if WEB_AUTH_COOKIE in request.cookies:
            _clear_auth_cookie(response)
        return response
    return error_response(401, detail)


class WebSocketHandshakeRefused(RuntimeError):
    """Why a WebSocket upgrade carries no caller: the kind names the close code a route sends."""

    def __init__(self, kind: Literal["unauthenticated", "cross_origin"], message: str) -> None:
        super().__init__(message)
        self.kind: Literal["unauthenticated", "cross_origin"] = kind


def authenticate_web_socket(connection: HTTPConnection, cfg: DlightragConfig) -> UserContext:
    """Resolve the Web caller of a WebSocket upgrade, or refuse the handshake.

    A browser cannot set a header on an upgrade, so the session cookie is the credential
    and the ``Origin`` is the only evidence that the page behind the socket is this
    application's own — the same two rules a cookie-authenticated write already follows.
    The middleware never sees a WebSocket scope, so the deployment's mode is applied here
    from the same settings: an edge-asserted identity where there is one, the paste-token
    cookie otherwise, and no credential at all where the deployment runs with auth off.
    """
    if not _has_exact_same_origin(connection):
        # Deliberately stricter than the write path: a socket has no CSRF header to add,
        # so an upgrade from anywhere but this application's own origin is refused.
        raise WebSocketHandshakeRefused("cross_origin", "Origin is not this application's")
    if cfg.access.auth_mode == "none":
        return UserContext(user_id="anonymous", auth_mode="none")
    if cfg.access.web_identity.edge is not None:
        try:
            return authenticate_edge(
                connection, edge=cfg.access.web_identity.edge, settings=web_identity_settings(cfg)
            )
        except EdgeIdentityError as exc:
            raise WebSocketHandshakeRefused("unauthenticated", str(exc)) from None
    raw_token = _token_from_request(connection)
    if not raw_token:
        raise WebSocketHandshakeRefused("unauthenticated", "Authentication required")
    try:
        return _authenticate_bearer(
            raw_token, cfg, default_user_id=connection.headers.get("X-User-Id", "anonymous")
        )
    except HTTPException as exc:
        raise WebSocketHandshakeRefused("unauthenticated", str(exc.detail)) from None


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
        path = request.scope["path"]
        if path == "/web/oauth/connections/mcp/callback":
            request.state.connection_oauth_query = SecretStr(
                request.scope.get("query_string", b"").decode("utf-8", errors="replace")
            )
            request.scope["query_string"] = b""
        if not path.startswith("/web"):
            return await call_next(request)
        if path in _PUBLIC_WEB_PATHS:
            # Login/logout CSRF: browser-driven POSTs must be exact same-origin.
            if path in {"/web/login", "/web/logout"} and request.method.upper() == "POST":
                origin = request.headers.get("Origin")
                if origin is not None and not _has_exact_same_origin(request):
                    return _cross_origin_rejected()
            return await call_next(request)

        cfg = self._config_getter()
        if cfg.access.auth_mode == "none":
            request.state.user_context = UserContext(user_id="anonymous", auth_mode="none")
            if (
                path.startswith(
                    (
                        "/web/api/connections/mcp",
                        "/web/api/agent-accounts",
                        "/web/api/memory",
                        "/web/api/privacy",
                    )
                )
                or path == "/web/api/video-playback"
            ):
                if _reject_web_mutation(request):
                    return _cross_origin_rejected()
                return await self._finish_web_response(request, call_next)
            return await call_next(request)

        if cfg.access.web_identity.edge is not None:
            return await self._dispatch_edge_identity(cfg, request, call_next)

        try:
            raw_token = _token_from_request(request)
            if not raw_token:
                return _unauthenticated(request)
            request.state.user_context = _authenticate_bearer(
                raw_token,
                cfg,
                default_user_id=request.headers.get("X-User-Id", "anonymous"),
            )
        except HTTPException as exc:
            if exc.status_code != 401:
                # A misconfigured verifier is the server's fault, not the caller's sign-in.
                return error_response(exc.status_code, str(exc.detail))
            return _unauthenticated(request, str(exc.detail))

        if _reject_web_mutation(request):
            return _cross_origin_rejected()

        return await self._finish_web_response(request, call_next)

    async def _dispatch_edge_identity(self, cfg, request: Request, call_next) -> Response:
        """Resolve the Web caller from the configured edge credential only."""
        try:
            request.state.user_context = authenticate_edge(
                request, edge=cfg.access.web_identity.edge, settings=web_identity_settings(cfg)
            )
        except EdgeIdentityError as exc:
            if exc.kind == "misconfigured":
                return error_response(500, str(exc))
            # The edge owns sign-in, so not even a page load goes to the paste form.
            return error_response(401, "Authentication required")
        if _reject_web_mutation(request):
            return _cross_origin_rejected()
        return await self._finish_web_response(request, call_next)

    async def _finish_web_response(self, request: Request, call_next) -> Response:
        """Run the request and issue the double-submit cookie on GET responses."""
        response = await call_next(request)
        if request.method.upper() == "GET":
            _ensure_csrf_cookie(request, response)
        return response


@router.get("/login", response_class=FileResponse)
async def login_page(request: Request, next: str = "/web/"):
    """Serve the static paste-token form when global auth is enabled."""
    from dlightrag.adapters.http.application import get_application

    cfg = get_application(request).config
    target = _safe_next_path(next)
    if cfg.access.auth_mode == "none" or cfg.access.web_identity.edge is not None:
        # The edge owns login; the paste form is the no-edge development hatch.
        return RedirectResponse(target, status_code=303)
    return app_html_response("login.html")


@router.post("/login")
async def login(
    request: Request,
    token: str = Form(default=""),
    next: str = Form(default="/web/"),
):
    """Validate a bearer token and store it in an HttpOnly web cookie."""
    from dlightrag.adapters.http.application import get_application

    cfg = get_application(request).config
    target = _safe_next_path(next)
    if cfg.access.auth_mode == "none" or cfg.access.web_identity.edge is not None:
        return RedirectResponse(target, status_code=303)
    try:
        _authenticate_bearer(token, cfg)
    except HTTPException:
        query = urlencode({"next": target, "error": "Authentication failed"})
        return RedirectResponse(f"/web/login?{query}", status_code=303)

    response = RedirectResponse(target, status_code=303)
    _set_auth_cookie(response, request, token)
    return response


@router.post("/logout")
async def logout() -> RedirectResponse:
    """Clear the web auth cookie."""
    response = RedirectResponse("/web/login", status_code=303)
    _clear_auth_cookie(response)
    return response


__all__ = [
    "WEB_AUTH_COOKIE",
    "WEB_CSRF_COOKIE",
    "WebAuthMiddleware",
    "WebSocketHandshakeRefused",
    "authenticate_web_socket",
    "router",
]
