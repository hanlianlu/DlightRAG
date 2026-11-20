# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Same-origin owner-only Settings commands; credentials are write-only."""

import logging
from dataclasses import asdict
from typing import Annotated, Any, Literal
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from dlightrag.adapters.http.browser.deps import get_application
from dlightrag.application.access.principal import (
    owner_id_from_user,
)
from dlightrag.application.connections.models import ConnectionCommand, ConnectionsError
from dlightrag.application.connections.policy import (
    MAX_ENDPOINT_HEADERS,
    MAX_HEADER_NAME_CHARS,
    MAX_HEADER_PREFIX_CHARS,
    MAX_HEADER_VALUE_CHARS,
    OAUTH_CALLBACK_PATH,
)

logger = logging.getLogger(__name__)

#: ``[[name, value], ...]`` as the form sends it: a short statement about an endpoint, never a
#: secret. The module's policy judges the pairs, and these are that policy's own numbers, so a
#: request this route accepts is never one the policy would refuse on size alone.
HeaderPair = tuple[
    Annotated[str, Field(max_length=MAX_HEADER_NAME_CHARS)],
    Annotated[str, Field(max_length=MAX_HEADER_VALUE_CHARS)],
]
EndpointHeaders = Annotated[list[HeaderPair], Field(max_length=MAX_ENDPOINT_HEADERS)]


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
    endpoint_headers: EndpointHeaders | None = None


class EditInput(RevisionInput):
    kind: Literal["edit", "enable", "disable"]
    label: str | None = Field(default=None, min_length=1, max_length=100)
    endpoint: str | None = Field(default=None, min_length=1, max_length=2048)
    # None leaves them as they are; an empty list clears them.
    endpoint_headers: EndpointHeaders | None = None
    consent_version: Literal[1] | None = None


class BearerInput(RevisionInput):
    endpoint: str | None = Field(default=None, min_length=1, max_length=2048)
    bearer: SecretStr = Field(min_length=1, max_length=8192, repr=False)


class HeaderCredentialInput(RevisionInput):
    """A credential that travels in a header of its own, named beside the token that carries it."""

    endpoint: str | None = Field(default=None, min_length=1, max_length=2048)
    header: str = Field(min_length=1, max_length=MAX_HEADER_NAME_CHARS)
    prefix: str = Field(default="", max_length=MAX_HEADER_PREFIX_CHARS)
    token: SecretStr = Field(min_length=1, max_length=8192, repr=False)


def _reached_callback_url(request: Request) -> str:
    """The OAuth callback route on the origin this request reached.

    For a browser that is its own origin, which the Web's same-origin guard checks on
    every write; a provider sends it back there with its session.
    """
    return str(request.url.replace(path=OAUTH_CALLBACK_PATH, query="", fragment=""))


def _owner(request: Request) -> str:
    return owner_id_from_user(request.state.user_context)


def _headers(pairs: list[HeaderPair] | None) -> tuple[tuple[str, str], ...] | None:
    """The form's stated headers as the command's pairs; None means "leave them as they are"."""
    return None if pairs is None else tuple((name, value) for name, value in pairs)


def _refused_callback_state(raw: str) -> str | None:
    """The one state a refused callback named, when it named exactly one usable value.

    Read leniently on purpose: the callback is already being refused, and this is only what lets
    its own flow be finished rather than left ``pending`` for the rest of its lifetime.
    """
    try:
        query = parse_qs(raw, keep_blank_values=True, max_num_fields=8)
    except ValueError:
        return None
    states = query.get("state", [])
    if len(states) != 1 or not states[0] or len(states[0]) > 256:
        return None
    return states[0]


async def _abandon_refused_callback(request: Request, raw: str) -> None:
    """Finish the flow a refused callback named, so its Connection reports the refusal at once.

    Best effort: the browser is returned to Settings either way, and a callback naming no flow
    this deployment can match leaves every live flow alone.
    """
    state = _refused_callback_state(raw)
    if state is None:
        return
    try:
        await get_application(request).connections.abandon_authorization(
            owner_id=_owner(request), state=state
        )
    except Exception:
        logger.warning("Refused authorization callback could not finish its flow")


async def _change(
    request: Request, body: RevisionInput, command: ConnectionCommand
) -> dict[str, Any]:
    try:
        view = await get_application(request).connections.change(
            owner_id=_owner(request), expected_revision=body.expected_revision, command=command
        )
        return asdict(view)
    except ConnectionsError as exc:
        raise HTTPException(exc.status, {"kind": exc.kind, "message": str(exc)}) from None


@router.get("")
async def read_connections(request: Request) -> dict[str, Any]:
    try:
        return asdict(await get_application(request).connections.read(owner_id=_owner(request)))
    except ConnectionsError as exc:
        raise HTTPException(exc.status, {"kind": exc.kind, "message": str(exc)}) from None


@router.post("")
async def create_connection(request: Request, body: CreateInput) -> dict[str, Any]:
    return await _change(
        request,
        body,
        ConnectionCommand(
            kind="create",
            label=body.label,
            endpoint=body.endpoint,
            endpoint_headers=_headers(body.endpoint_headers),
        ),
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
            endpoint_headers=_headers(body.endpoint_headers),
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
                owner_id=_owner(request),
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


@router.put("/{connection_id}/header")
async def replace_header_credential(
    connection_id: str, request: Request, body: HeaderCredentialInput
) -> dict[str, Any]:
    try:
        return asdict(
            await get_application(request).connections.replace_header_credential(
                owner_id=_owner(request),
                connection_id=connection_id,
                expected_revision=body.expected_revision,
                header=body.header,
                prefix=body.prefix,
                token=body.token,
                endpoint=body.endpoint,
            )
        )
    except ConnectionsError as exc:
        raise HTTPException(exc.status, {"kind": exc.kind, "message": str(exc)}) from None


@router.post("/{connection_id}/oauth")
async def begin_authorization(
    connection_id: str, request: Request, body: OAuthInput, response: Response
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    try:
        return asdict(
            await get_application(request).connections.begin_authorization(
                owner_id=_owner(request),
                connection_id=connection_id,
                expected_revision=body.expected_revision,
                callback_url=_reached_callback_url(request),
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
    document = get_application(request).connections.published_client_metadata(
        callback_url=_reached_callback_url(request)
    )
    if document is None:
        raise HTTPException(404, "No client metadata published")
    return JSONResponse(document, headers={"Cache-Control": "public, max-age=300"})


@callback_router.get("/oauth/connections/mcp/callback")
async def authorization_callback(request: Request):
    from fastapi.responses import RedirectResponse

    # WebAuthMiddleware removes sensitive query data before downstream logging.
    secret = getattr(request.state, "connection_oauth_query", SecretStr(""))
    raw = secret.get_secret_value()
    target = "/web/?settings=connections"
    try:
        if len(raw) > 16384:
            raise ValueError
        query = parse_qs(raw, keep_blank_values=True, strict_parsing=True, max_num_fields=8)
        if any(len(values) != 1 for values in query.values()):
            raise ValueError
        await get_application(request).connections.authorization_callback(
            owner_id=_owner(request),
            state=query.get("state", [""])[0],
            code=query.get("code", [None])[0],
            issuer=query.get("iss", [None])[0],
            error=query.get("error", [None])[0],
        )
    except (ConnectionsError, ValueError) as exc:
        # Refusals were silent, and a silent refusal is indistinguishable from a flow that never
        # arrived. The category is ours or the exception's class: never the callback's query.
        logger.warning(
            "Refused an authorization callback: %s",
            str(exc) if isinstance(exc, ConnectionsError) else type(exc).__name__,
        )
        # A refused callback still names the flow it came from: finish it there, so the card
        # reports the refusal instead of staying pending for the flow's whole lifetime.
        await _abandon_refused_callback(request, raw)
        target += "&authorization=restart"
    return RedirectResponse(
        target,
        status_code=303,
        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )
