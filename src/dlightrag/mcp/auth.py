# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""MCP OAuth resource-server integration over DlightRAG authentication."""

from mcp.server.auth.provider import AccessToken

from dlightrag.access import AuthenticationError, authenticate_bearer_token
from dlightrag.config import DlightragConfig
from dlightrag.model_settings import authentication_settings


def _token_scopes(claims: dict[str, object]) -> list[str]:
    value = claims.get("scope", claims.get("scp"))
    if isinstance(value, str):
        return list(dict.fromkeys(value.split()))
    if isinstance(value, list):
        scopes = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return list(dict.fromkeys(scopes))
    return []


class DlightRAGTokenVerifier:
    """MCP TokenVerifier backed by DlightRAG's configured JWT verifier."""

    def __init__(self, config: DlightragConfig, *, resource: str | None = None) -> None:
        self._settings = authentication_settings(config, audience=resource)
        self._resource = resource

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            user = authenticate_bearer_token(token, self._settings)
        except AuthenticationError as exc:
            if exc.kind != "verifier_misconfigured":
                return None
            raise

        claims = dict(user.claims)
        client_id = claims.get("azp") or claims.get("client_id") or user.user_id
        expires_at = claims.get("exp")
        return AccessToken(
            token=token,
            client_id=str(client_id),
            scopes=_token_scopes(user.claims),
            expires_at=expires_at if isinstance(expires_at, int) else None,
            resource=self._resource,
            subject=user.user_id,
            claims=claims,
        )
