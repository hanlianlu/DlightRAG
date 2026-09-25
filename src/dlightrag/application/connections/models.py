# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner commands, redacted projections, and private persistence/discovery ports."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from dlightrag.engine.agent.tools import ToolDeclaration, ToolResult, ToolRuntime
from dlightrag.engine.answer.execution.connection_binding import (
    ResearchToolClaim,
    RunConnectionBinding,
)

from .policy import ConnectionPolicy


@dataclass(frozen=True, slots=True)
class BoundResearchConnections:
    tools: tuple[ToolDeclaration, ...] = ()
    bindings: tuple[RunConnectionBinding, ...] = ()


class ConnectionsError(RuntimeError):
    """A safe management rejection; never includes remote error text or credentials."""

    def __init__(
        self,
        message: str,
        status: int = 409,
        *,
        kind: Literal["rejected", "requires_reauthorization"] = "rejected",
    ) -> None:
        super().__init__(message)
        self.status = status
        self.kind = kind


class ConnectionCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["create", "edit", "enable", "disable", "delete", "probe", "revoke"]
    connection_id: str | None = None
    label: str | None = Field(default=None, min_length=1, max_length=100)
    endpoint: str | None = Field(default=None, max_length=2048)
    consent_version: Literal[1] | None = None


@dataclass(frozen=True)
class CatalogueTool:
    remote_name: str
    local_name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ConnectionView:
    """The Settings projection: exactly what Settings renders, and nothing else.

    Catalogue facts and raw error kinds stay server-side. Activation epoch and generation remain
    because the integration suite reads them here as the authoritative read model: re-enable must
    publish a newer epoch, and a refresh must publish a newer generation.
    """

    connection_id: str
    label: str
    endpoint: str
    enabled: bool
    activation_epoch: int
    generation: int
    authentication: str
    status: str
    authorization_status: str | None = None


@dataclass(frozen=True)
class PresetView:
    """One create-form starter: a label, an endpoint, and the tab that fits its tier.

    Presets carry no credential, no authority, and no catalogue: filling the form is all a
    Preset can do, and the ordinary create command still validates and owns the Connection.
    """

    preset_id: str
    label: str
    endpoint: str
    default_authentication: str


@dataclass(frozen=True)
class ConnectionsView:
    revision: str
    connections: tuple[ConnectionView, ...]
    presets: tuple[PresetView, ...] = ()


@dataclass(frozen=True, slots=True)
class PinnedToolFact:
    """One pinned Connection tool's display facts: identity plus owner-visible names.

    The local name is dispatch identity; the two names are what a person reads.
    Only a reader that already owns the Run may obtain these facts, and they
    authorize nothing.
    """

    local_name: str
    connection_label: str
    remote_name: str


@dataclass(frozen=True)
class StoredConnection:
    owner_id: str
    connection_id: str
    revision: int
    label: str
    endpoint: str
    enabled: bool
    activation_epoch: int
    generation: int
    grant_id: str | None
    status: str
    catalogue: tuple[CatalogueTool, ...]
    catalogue_created_at: str | None
    last_error_kind: str | None
    authorization_status: str | None = None
    authentication: str = "none"
    secret_version: int = 0
    grant_refresh_epoch: int = 0
    envelope: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class RefreshClaim:
    connection: StoredConnection
    worker_id: str
    epoch: int


class McpClientPort(Protocol):
    async def call(
        self,
        *,
        endpoint: str,
        bearer: SecretStr | None,
        policy: ConnectionPolicy,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult: ...

    async def discover(
        self, *, endpoint: str, bearer: SecretStr | None, policy: ConnectionPolicy
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class DispatchCredentials:
    connection_id: str
    generation: int
    activation_epoch: int
    endpoint: str
    grant_id: str | None
    secret_version: int
    envelope: str | None = field(repr=False)
    authentication: str = "none"


@dataclass(frozen=True)
class StoredGrant:
    owner_id: str
    connection_id: str
    grant_id: str
    endpoint: str
    secret_version: int
    refresh_epoch: int
    scopes: tuple[str, ...]
    key_id: str
    envelope: str = field(repr=False)


@dataclass(frozen=True)
class GrantRefreshClaim:
    grant: StoredGrant
    worker_id: str


class ConnectionsStore(Protocol):
    async def claim_grant_refresh(
        self,
        *,
        owner_id: str,
        connection_id: str,
        grant_id: str,
        endpoint: str,
        worker_id: str,
        lease_seconds: float,
    ) -> GrantRefreshClaim | None: ...
    async def save_grant_refresh(
        self, *, claim: GrantRefreshClaim, key_id: str, envelope: str
    ) -> bool: ...
    async def release_grant_refresh(self, *, claim: GrantRefreshClaim) -> None: ...
    async def rotation_candidates(
        self, *, active_key_id: str, limit: int
    ) -> tuple[StoredGrant, ...]: ...
    async def reencrypt_grant(self, *, grant: StoredGrant, key_id: str, envelope: str) -> bool: ...
    async def collect_garbage(self, *, limit: int) -> int: ...

    async def wait_oauth_callback(self, *, worker_id: str, timeout: float) -> None: ...
    async def create_oauth_flow(
        self, *, flow: OAuthFlow, lifetime: float, lease: float
    ) -> None: ...
    async def renew_oauth_flow(self, *, flow: OAuthFlow, lease: float) -> bool: ...
    async def oauth_redirect(self, *, flow: OAuthFlow, state_hash: str) -> None: ...
    async def oauth_credentials(self, *, flow: OAuthFlow, envelope: str) -> None: ...
    async def oauth_callback_flow(self, *, owner_id: str, state_hash: str) -> OAuthFlow: ...
    async def deposit_oauth_callback(
        self, *, flow: OAuthFlow, state_hash: str, envelope: str
    ) -> None: ...
    async def consume_oauth_callback(self, *, flow: OAuthFlow) -> str | None: ...
    async def finish_oauth_flow(self, *, flow: OAuthFlow) -> None: ...
    async def publish_authorization(
        self,
        *,
        owner_id: str,
        connection_id: str,
        expected_revision: str,
        endpoint: str,
        grant_id: str,
        kind: str,
        key_id: str,
        envelope: str,
        scopes: tuple[str, ...],
        catalogue: tuple[CatalogueTool, ...],
        policy: ConnectionPolicy,
        flow: OAuthFlow | None = None,
    ) -> None: ...

    async def dispatch_gate(
        self,
        *,
        claim: ResearchToolClaim,
        bindings: tuple[RunConnectionBinding, ...],
        runtime: ToolRuntime,
        tool: CatalogueTool,
        arguments: dict[str, Any],
    ) -> DispatchCredentials: ...
    async def dispatch_alive(
        self, *, claim: ResearchToolClaim, runtime: ToolRuntime, dispatch: DispatchCredentials
    ) -> bool: ...
    async def observe_call(
        self, *, owner_id: str, dispatch: DispatchCredentials, error: str | None
    ) -> None: ...
    async def wait_dispatch_change(self, timeout: float) -> None: ...

    async def research_catalogues(
        self, owner_id: str
    ) -> tuple[tuple[RunConnectionBinding, tuple[CatalogueTool, ...]], ...]: ...
    async def pinned_catalogues(
        self, *, owner_id: str, run_id: str, bindings: tuple[RunConnectionBinding, ...]
    ) -> tuple[CatalogueTool, ...]: ...
    async def pinned_tool_facts(
        self, *, owner_id: str, run_id: str
    ) -> tuple[PinnedToolFact, ...]: ...
    async def start_notifications(self) -> None: ...
    async def stop_notifications(self) -> None: ...
    async def wait_refresh(self, timeout: float) -> None: ...
    async def initialize(self, *, validate_only: bool) -> None: ...
    async def read(self, owner_id: str) -> tuple[str, tuple[StoredConnection, ...]]: ...
    async def change(
        self,
        *,
        owner_id: str,
        expected_revision: str,
        command: ConnectionCommand,
        policy: ConnectionPolicy,
        candidate: tuple[CatalogueTool, ...] | None = None,
    ) -> None: ...
    async def replace_bearer(
        self,
        *,
        owner_id: str,
        connection_id: str,
        expected_revision: str,
        grant_id: str,
        key_id: str,
        envelope: str,
    ) -> None: ...
    async def claim(
        self,
        *,
        worker_id: str,
        lease_seconds: float,
        owner_id: str | None = None,
        connection_id: str | None = None,
    ) -> RefreshClaim | None: ...
    async def publish(
        self,
        *,
        claim: RefreshClaim,
        catalogue: tuple[CatalogueTool, ...] | None,
        error: str | None,
        retry_seconds: float,
        policy: ConnectionPolicy,
    ) -> bool: ...


@dataclass(frozen=True)
class AuthorizationStart:
    authorization_url: str = field(repr=False)


@dataclass(frozen=True)
class OAuthFlow:
    flow_id: str
    owner_id: str
    connection_id: str
    flow_owner: str
    endpoint: str
    expected_revision: str


@dataclass(frozen=True)
class OAuthResult:
    credentials: SecretStr = field(repr=False)
    scopes: tuple[str, ...]
    tools: list[dict[str, Any]]


class OAuthPort(Protocol):
    async def refresh(
        self,
        *,
        endpoint: str,
        credentials: SecretStr,
        scopes: tuple[str, ...],
        policy: ConnectionPolicy,
        save: Callable[[SecretStr], Awaitable[None]],
    ) -> SecretStr: ...

    async def authorize(
        self,
        *,
        endpoint: str,
        callback_url: str,
        policy: ConnectionPolicy,
        redirect: Callable[[str], Awaitable[None]],
        callback: Callable[[], Awaitable[SecretStr]],
        save: Callable[[SecretStr], Awaitable[None]],
    ) -> OAuthResult: ...
