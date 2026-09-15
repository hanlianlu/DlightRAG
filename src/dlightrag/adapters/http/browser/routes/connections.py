# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Same-origin owner-only Settings commands; credentials are write-only."""

from dataclasses import asdict
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from dlightrag.adapters.http.browser.deps import get_application
from dlightrag.application.access import owner_id_from_user
from dlightrag.application.connections import ConnectionCommand, ConnectionsError


class _SecretSafeRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def safe(request: Request):
            try:
                return await handler(request)
            except RequestValidationError:
                raise HTTPException(422, "Invalid Connection command") from None

        return safe


router = APIRouter(prefix="/connections/mcp", route_class=_SecretSafeRoute)


class RevisionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    expected_revision: str = Field(min_length=1, max_length=128)


class CreateInput(RevisionInput):
    label: str = Field(min_length=1, max_length=100)
    endpoint: str = Field(min_length=1, max_length=2048)


class EditInput(RevisionInput):
    kind: Literal["edit", "enable", "disable"]
    label: str | None = Field(default=None, min_length=1, max_length=100)
    endpoint: str | None = Field(default=None, min_length=1, max_length=2048)
    consent_version: Literal[1] | None = None


class BearerInput(RevisionInput):
    endpoint: str | None = Field(default=None, min_length=1, max_length=2048)
    bearer: SecretStr = Field(min_length=1, max_length=8192, repr=False)


def _owner(request: Request) -> dict[str, str]:
    user = request.state.user_context
    return {"owner_id": owner_id_from_user(user), "auth_mode": user.auth_mode}


async def _change(
    request: Request, body: RevisionInput, command: ConnectionCommand
) -> dict[str, Any]:
    try:
        view = await get_application(request).connections.change(
            **_owner(request), expected_revision=body.expected_revision, command=command
        )
        return asdict(view)
    except ConnectionsError as exc:
        raise HTTPException(exc.status, {"kind": exc.kind, "message": str(exc)}) from None


@router.get("")
async def read_connections(request: Request) -> dict[str, Any]:
    try:
        return asdict(await get_application(request).connections.read(**_owner(request)))
    except ConnectionsError as exc:
        raise HTTPException(exc.status, {"kind": exc.kind, "message": str(exc)}) from None


@router.post("")
async def create_connection(request: Request, body: CreateInput) -> dict[str, Any]:
    return await _change(
        request, body, ConnectionCommand(kind="create", label=body.label, endpoint=body.endpoint)
    )


@router.patch("/{connection_id}")
async def edit_connection(connection_id: str, request: Request, body: EditInput) -> dict[str, Any]:
    return await _change(
        request,
        body,
        ConnectionCommand(
            connection_id=connection_id,
            kind=body.kind,
            label=body.label,
            endpoint=body.endpoint,
            consent_version=body.consent_version,
        ),
    )


@router.delete("/{connection_id}")
async def delete_connection(
    connection_id: str, request: Request, body: RevisionInput
) -> dict[str, Any]:
    return await _change(
        request, body, ConnectionCommand(kind="delete", connection_id=connection_id)
    )


@router.post("/{connection_id}/probe")
async def probe_connection(
    connection_id: str, request: Request, body: RevisionInput
) -> dict[str, Any]:
    return await _change(
        request, body, ConnectionCommand(kind="probe", connection_id=connection_id)
    )


@router.post("/{connection_id}/revoke")
async def revoke_connection(
    connection_id: str, request: Request, body: RevisionInput
) -> dict[str, Any]:
    return await _change(
        request, body, ConnectionCommand(kind="revoke", connection_id=connection_id)
    )


@router.put("/{connection_id}/bearer")
async def replace_bearer(connection_id: str, request: Request, body: BearerInput) -> dict[str, Any]:
    try:
        return asdict(
            await get_application(request).connections.replace_bearer(
                **_owner(request),
                connection_id=connection_id,
                expected_revision=body.expected_revision,
                bearer=body.bearer,
                endpoint=body.endpoint,
            )
        )
    except ConnectionsError as exc:
        raise HTTPException(exc.status, {"kind": exc.kind, "message": str(exc)}) from None


class OAuthInput(RevisionInput):
    endpoint: str | None = Field(default=None, min_length=1, max_length=2048)


@router.post("/{connection_id}/oauth")
async def begin_authorization(
    connection_id: str, request: Request, body: OAuthInput, response: Response
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    try:
        return asdict(
            await get_application(request).connections.begin_authorization(
                **_owner(request),
                connection_id=connection_id,
                expected_revision=body.expected_revision,
                endpoint=body.endpoint,
            )
        )
    except ConnectionsError as exc:
        raise HTTPException(exc.status, {"kind": exc.kind, "message": str(exc)}) from None


callback_router = APIRouter(route_class=_SecretSafeRoute)


@callback_router.get("/oauth/connections/mcp/client-metadata")
async def published_client_metadata(request: Request) -> Response:
    """Publish this deployment's Client ID Metadata Document.

    Public by protocol and read-only: it reads no cookie, owner, or Connection, and returns only
    the few facts an authorization redirect already reveals. A deployment without a public https
    callback publishes none, and its Connections register dynamically instead.
    """
    document = get_application(request).connections.published_client_metadata()
    if document is None:
        raise HTTPException(404, "No client metadata published")
    return JSONResponse(document, headers={"Cache-Control": "public, max-age=300"})


@callback_router.get("/oauth/connections/mcp/callback")
async def authorization_callback(request: Request):
    from urllib.parse import parse_qs

    from fastapi.responses import RedirectResponse

    # WebAuthMiddleware removes sensitive query data before downstream logging.
    secret = getattr(request.state, "connection_oauth_query", SecretStr(""))
    target = "/web/?settings=connections"
    try:
        raw = secret.get_secret_value()
        if len(raw) > 16384:
            raise ValueError
        query = parse_qs(raw, keep_blank_values=True, strict_parsing=True, max_num_fields=8)
        if any(len(values) != 1 for values in query.values()):
            raise ValueError
        await get_application(request).connections.authorization_callback(
            **_owner(request),
            state=query.get("state", [""])[0],
            code=query.get("code", [None])[0],
            issuer=query.get("iss", [None])[0],
            error=query.get("error", [None])[0],
        )
    except ConnectionsError, ValueError:
        target += "&authorization=restart"
    return RedirectResponse(
        target,
        status_code=303,
        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )
