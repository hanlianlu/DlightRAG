# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""SDK Settings authorization and fenced token-only refresh preflight."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from urllib.parse import parse_qs, urlsplit, urlunsplit

import httpx2
from mcp.client.auth.oauth2 import OAuthClientProvider, TokenStorage
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
    ProtectedResourceMetadata,
)
from pydantic import AnyHttpUrl, SecretStr

from dlightrag.application.connections.client_metadata import CLIENT_NAME, client_metadata_url
from dlightrag.application.connections.models import ConnectionsError, OAuthResult
from dlightrag.application.connections.policy import ConnectionPolicy
from dlightrag.engine.network_admission import _normalize_host_patterns, _resolve_public_target

from .personal_http import (
    _PRIVATE_SESSION,
    PersonalMcpClient,
    _AdmittedTransport,
    _protect_sdk_logs,
)


class _FlowTokenStorage(TokenStorage):
    def __init__(self, save: Callable[[SecretStr], Awaitable[None]]) -> None:
        self.consented_scopes: set[str] = set()
        self.tokens: OAuthToken | None = None
        self.client_info: OAuthClientInformationFull | None = None
        self.expires_at: float | None = None
        self._save = save

    def credentials(self) -> SecretStr:
        return SecretStr(
            json.dumps(
                {
                    "tokens": self.tokens.model_dump(mode="json") if self.tokens else None,
                    "client_info": self.client_info.model_dump(mode="json")
                    if self.client_info
                    else None,
                    "expires_at": self.expires_at,
                }
            )
        )

    async def get_tokens(self) -> OAuthToken | None:
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        if not set((tokens.scope or "").split()) <= self.consented_scopes:
            raise ConnectionsError("OAuth scope requires new provider consent", 401)
        self.tokens = tokens
        self.expires_at = time.time() + tokens.expires_in if tokens.expires_in is not None else None
        await self._save(self.credentials())

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self.client_info = client_info
        await self._save(self.credentials())


class PersonalOAuthClient:
    def __init__(
        self, *, transport_factory: Callable[[], httpx2.AsyncBaseTransport] | None = None
    ) -> None:
        self._mcp = PersonalMcpClient(transport_factory=transport_factory)
        self._transport_factory = self._mcp._transport_factory

    async def refresh(
        self,
        *,
        endpoint: str,
        credentials: SecretStr,
        scopes: tuple[str, ...],
        policy: ConnectionPolicy,
        save: Callable[[SecretStr], Awaitable[None]],
    ) -> SecretStr:
        """Drive the locked SDK refresh flow, stopping BEFORE its original request.

        The host holds a fenced Grant lease and re-gates the effect afterward.
        No failed refresh may enter SDK discovery, registration or consent.
        """
        _protect_sdk_logs()
        token = _PRIVATE_SESSION.set(True)
        flow = None
        try:
            raw = json.loads(credentials.get_secret_value())
            saved: SecretStr | None = None

            async def persist(value: SecretStr) -> None:
                nonlocal saved
                updated = {**raw, **json.loads(value.get_secret_value())}
                result = SecretStr(json.dumps(updated))
                await save(result)  # Fenced CAS failure discards the new token.
                saved = result

            storage = _FlowTokenStorage(persist)
            storage.consented_scopes = set(scopes)
            storage.tokens = OAuthToken.model_validate(raw["tokens"])
            storage.client_info = OAuthClientInformationFull.model_validate(raw["client_info"])
            metadata = OAuthMetadata.model_validate(raw["oauth_metadata"])
            if not storage.tokens.refresh_token or not metadata.token_endpoint:
                raise ConnectionsError("OAuth refresh unavailable", 401)
            provider = OAuthClientProvider(
                server_url=endpoint,
                client_metadata=OAuthClientMetadata(
                    redirect_uris=storage.client_info.redirect_uris or [],
                    scope=" ".join(scopes),
                ),
                storage=storage,
            )
            provider.context.oauth_metadata = metadata
            if raw.get("resource_metadata"):
                resource = ProtectedResourceMetadata.model_validate(raw["resource_metadata"])
                await provider._validate_resource_match(resource)
                provider.context.protected_resource_metadata = resource
            provider.context.token_expiry_time = raw["expires_at"]
            original = httpx2.Request(
                "POST", endpoint, headers={"mcp-protocol-version": "2025-11-25"}
            )
            flow = provider.async_auth_flow(original)
            outgoing = await anext(flow)
            async with (
                asyncio.timeout(policy.discovery_timeout),
                httpx2.AsyncClient(
                    transport=_AdmittedTransport(
                        endpoint, None, policy, self._transport_factory(), oauth=True
                    ),
                    trust_env=False,
                    follow_redirects=False,
                    timeout=httpx2.Timeout(policy.idle_timeout, connect=policy.connect_timeout),
                ) as client,
            ):
                for _ in range(6):
                    if outgoing is original:
                        break
                    # SDK owns request construction and same-origin redirect policy.
                    # Only the persisted token origin may receive refresh secrets.
                    target = httpx2.URL(str(metadata.token_endpoint))
                    if outgoing.method != "POST" or (
                        outgoing.url.scheme,
                        outgoing.url.host,
                        outgoing.url.port,
                    ) != (target.scheme, target.host, target.port):
                        raise ConnectionsError("OAuth refresh target rejected", 401)
                    response = await client.send(outgoing)
                    try:
                        outgoing = await flow.asend(response)
                    finally:
                        await response.aclose()
                if outgoing is not original or saved is None:
                    raise ConnectionsError("OAuth refresh needs authorization", 401)
                return saved
        except asyncio.CancelledError:
            raise
        except Exception:
            raise ConnectionsError("OAuth refresh needs authorization", 401) from None
        finally:
            if flow is not None:
                await flow.aclose()
            _PRIVATE_SESSION.reset(token)

    async def authorize(
        self,
        *,
        endpoint: str,
        callback_url: str,
        policy: ConnectionPolicy,
        redirect: Callable[[str], Awaitable[None]],
        callback: Callable[[], Awaitable[SecretStr]],
        save: Callable[[SecretStr], Awaitable[None]],
    ) -> OAuthResult:
        storage = _FlowTokenStorage(save)
        redirected = False
        redactions: list[str] = []

        async def admitted_redirect(url: str) -> None:
            nonlocal redirected
            # A provider step-up needs a fresh Settings flow/new Grant, not an
            # invisible second redirect from this task.
            if redirected or len(url) > 8192:
                raise ConnectionsError("Authorization must restart from Settings", 401)
            parts = urlsplit(url)
            query = parse_qs(parts.query, strict_parsing=True)
            if (
                parts.username
                or parts.password
                or parts.fragment
                or (
                    (policy.require_https or urlsplit(endpoint).scheme == "https")
                    and parts.scheme != "https"
                )
                or len(query.get("state", [])) != 1
            ):
                raise ConnectionsError("Authorization target rejected")
            target = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
            async with asyncio.timeout(policy.connect_timeout):
                await _resolve_public_target(
                    target, allow_private_hosts=_normalize_host_patterns(policy.allow_private_hosts)
                )
            redactions.append(query["state"][0])
            storage.consented_scopes = set(query.get("scope", [""])[0].split())
            redirected = True
            await redirect(url)

        async def sdk_callback() -> AuthorizationCodeResult:
            raw = json.loads((await callback()).get_secret_value())
            if raw.get("error"):
                raise ConnectionsError("Authorization rejected; restart from Settings", 401)
            if raw.get("code"):
                redactions.append(raw["code"])
            return AuthorizationCodeResult(
                code=raw.get("code"), state=raw.get("state"), iss=raw.get("iss")
            )

        provider = OAuthClientProvider(
            server_url=endpoint,
            client_metadata=OAuthClientMetadata(
                client_name=CLIENT_NAME,
                redirect_uris=[AnyHttpUrl(callback_url)],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                token_endpoint_auth_method="client_secret_basic",  # noqa: S106 - SDK method identifier, not a secret
            ),
            storage=storage,
            # The SDK uses this URL as the client_id when the authorization server advertises
            # client_id_metadata_document_supported, and registers dynamically when it does not.
            client_metadata_url=client_metadata_url(callback_url),
            redirect_handler=admitted_redirect,
            callback_handler=sdk_callback,
        )
        try:
            tools = await self._mcp.discover_authorized(
                endpoint=endpoint, bearer=None, policy=policy, auth=provider
            )
            if storage.tokens is None or not redirected:
                raise ConnectionsError("Server did not complete SDK OAuth authorization", 401)
            # SDK metadata is retained for the fenced refresh preflight. Never
            # infer token endpoints from a changed Connection audience.
            raw = json.loads(storage.credentials().get_secret_value())
            raw["oauth_metadata"] = (
                provider.context.oauth_metadata.model_dump(mode="json")
                if provider.context.oauth_metadata
                else None
            )
            raw["resource_metadata"] = (
                provider.context.protected_resource_metadata.model_dump(mode="json")
                if provider.context.protected_resource_metadata
                else None
            )
            redactions.extend(
                value
                for value in (
                    storage.tokens.access_token,
                    storage.tokens.refresh_token,
                    storage.client_info.client_secret if storage.client_info else None,
                )
                if value
            )
            encoded_tools = json.dumps(tools)
            for value in redactions:
                encoded_tools = encoded_tools.replace(json.dumps(value)[1:-1], "[redacted]")
            return OAuthResult(
                SecretStr(json.dumps(raw)),
                tuple(sorted(set((storage.tokens.scope or "").split()))),
                json.loads(encoded_tools),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            raise ConnectionsError(
                "OAuth authorization failed; restart from Settings", 401
            ) from None
