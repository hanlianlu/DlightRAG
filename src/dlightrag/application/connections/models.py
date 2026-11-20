# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner commands, redacted projections, and private persistence/discovery ports."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from dlightrag.engine.agent.tools.contracts import (
    ToolDeclaration,
    ToolResult,
    ToolRuntime,
)
from dlightrag.engine.answer.execution.connection_binding import (
    ResearchToolClaim,
    RunConnectionBinding,
)

from .policy import Authentication, ConnectionPolicy, GrantKind, StatedHeaders


@dataclass(frozen=True, slots=True)
class CredentialAccess:
    """One stored credential, ready to be sent: the token, and the header it travels in.

    The token is the secret; the header and prefix are how this Connection was told to present
    it. A port takes this one value rather than three fields that always travel together.
    """

    token: SecretStr = field(repr=False)
    header: str = "Authorization"
    prefix: str = "Bearer "


@dataclass(frozen=True, slots=True)
class BoundConnectionSurface:
    """One enabled Connection as a Research Run pins it for resources and guidance.

    Secret-free and display/authority metadata only: the connection identity the Run's pins
    already cover, the token the model names it by, its owner-visible label, and the bounded
    initialize instructions it published with the generation it was accepted against.
    """

    connection_id: str
    token: str
    label: str
    instructions: str = ""


@dataclass(frozen=True, slots=True)
class PinnedCatalogue:
    """One enabled Connection's published generation as acceptance binds it.

    The binding is the authority; the label and bounded instructions are the display and
    prompt facts pinned beside it. Content and credential never ride along.
    """

    binding: RunConnectionBinding
    tools: tuple[CatalogueTool, ...]
    label: str
    instructions: str = ""


@dataclass(frozen=True, slots=True)
class BoundResearchConnections:
    tools: tuple[ToolDeclaration, ...] = ()
    bindings: tuple[RunConnectionBinding, ...] = ()
    surface: tuple[BoundConnectionSurface, ...] = ()
    #: Whether this deployment serves the on-demand resource transport. The pinned surface
    #: still carries instructions when it does not; the three shared tools are not composed.
    resource_tools_available: bool = False


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


class UnsupportedResultError(ConnectionsError):
    """A tool call completed, but its result holds content a Connection cannot carry.

    ``kinds`` are MCP content types such as ``image``, never remote text.
    """

    def __init__(self, kinds: tuple[str, ...]) -> None:
        super().__init__("MCP result holds content that cannot be shown")
        self.kinds = kinds


class ConnectionCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["create", "edit", "enable", "disable", "delete", "probe", "revoke"]
    connection_id: str | None = None
    label: str | None = Field(default=None, min_length=1, max_length=100)
    endpoint: str | None = Field(default=None, max_length=2048)
    #: What this endpoint's every request states, or None to leave it as it is. Never a secret:
    #: a credential travels in its own sealed Grant, and policy refuses to overlap the two.
    endpoint_headers: StatedHeaders | None = None
    consent_version: Literal[1] | None = None

    @model_validator(mode="after")
    def _label_is_stated_text(self) -> ConnectionCommand:
        """A label is display text: stated, bounded spaces, no control characters.

        The resource tools name a Connection by this label, so a blank or newline-bearing
        label would leave the model no stable name to pass.
        """
        if self.label is not None:
            stated = self.label.strip()
            if not stated or any(ord(char) < 32 for char in stated):
                raise ValueError("Connection label must be stated text")
        return self


@dataclass(frozen=True)
class RemoteTool:
    """One validated tool definition as its server publishes it, before it has a local name."""

    remote_name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class CatalogueTool(RemoteTool):
    """A published tool: its server's definition and the name a model calls it by.

    Publication gives the name (see ``naming``), because only the owner-locked publication sees
    the names the owner's other Connections already publish.
    """

    local_name: str = field(kw_only=True)


@dataclass(frozen=True)
class ConnectionView:
    """The Settings projection: exactly what Settings renders, and nothing else.

    Catalogue content and raw error kinds stay server-side; only its aggregate size and
    declaration cost are projected. Activation epoch and generation remain
    because the integration suite reads them here as the authoritative read model: re-enable must
    publish a newer epoch, and a refresh must publish a newer generation.
    """

    connection_id: str
    label: str
    endpoint: str
    enabled: bool
    activation_epoch: int
    generation: int
    authentication: Authentication
    status: str
    endpoint_headers: StatedHeaders = ()
    #: The header a ``header`` credential travels in, and the prefix its value carries, when the
    #: deployment can read them back. Both are non-secret: the token stays sealed, and a
    #: deployment without the key ring answers None rather than failing a read.
    credential_header: str | None = None
    credential_prefix: str | None = None
    authorization_status: str | None = None
    #: The published catalogue's size and what its declarations cost one Research request,
    #: estimated with the same token estimator the runtime accounts with. None until a
    #: catalogue is published; a disabled Connection's cost is hypothetical and still shown.
    tool_count: int | None = None
    declaration_tokens: int | None = None


@dataclass(frozen=True)
class PresetView:
    """One create-form starter: a label, an endpoint, and the tab that fits its tier.

    Presets carry no credential, no authority, and no catalogue: filling the form is all a
    Preset can do, and the ordinary create command still validates and owns the Connection.
    ``credential_header`` and ``credential_prefix`` are the tab's own defaults, which an
    endpoint that authenticates with a vendor header needs stated before anything is saved.
    """

    preset_id: str
    label: str
    endpoint: str
    default_authentication: Authentication
    endpoint_headers: StatedHeaders = ()
    credential_header: str = "Authorization"
    credential_prefix: str = "Bearer "


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
    endpoint_headers: StatedHeaders = ()
    authentication: Authentication = "none"
    secret_version: int = 0
    grant_refresh_epoch: int = 0
    envelope: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class RefreshClaim:
    connection: StoredConnection
    worker_id: str
    epoch: int


@dataclass(frozen=True, slots=True)
class McpDiscovery:
    """One discovery session's result: the tool catalogue and the server's own instructions."""

    tools: tuple[dict[str, Any], ...]
    instructions: str | None = None


class McpClientPort(Protocol):
    async def call(
        self,
        *,
        endpoint: str,
        access: CredentialAccess | None,
        policy: ConnectionPolicy,
        name: str,
        arguments: dict[str, Any],
        endpoint_headers: StatedHeaders = (),
    ) -> ToolResult: ...

    async def discover(
        self,
        *,
        endpoint: str,
        access: CredentialAccess | None,
        policy: ConnectionPolicy,
        endpoint_headers: StatedHeaders = (),
    ) -> McpDiscovery: ...


class McpResourcePort(Protocol):
    """One Connection's on-demand document surface, over the same bounded transport.

    A resource call is a read: it mutates nothing on the server, so it may be retried after a
    fault and may run beside other read-only calls. Each call opens its own session; there is
    no shared resource cache.
    """

    async def list_resources(
        self,
        *,
        endpoint: str,
        access: CredentialAccess | None,
        policy: ConnectionPolicy,
        endpoint_headers: StatedHeaders = (),
        cursor: str | None = None,
    ) -> ToolResult: ...

    async def list_resource_templates(
        self,
        *,
        endpoint: str,
        access: CredentialAccess | None,
        policy: ConnectionPolicy,
        endpoint_headers: StatedHeaders = (),
        cursor: str | None = None,
    ) -> ToolResult: ...

    async def read_resource(
        self,
        *,
        endpoint: str,
        access: CredentialAccess | None,
        policy: ConnectionPolicy,
        endpoint_headers: StatedHeaders = (),
        uri: str,
    ) -> ToolResult: ...


@dataclass(frozen=True)
class DispatchCredentials:
    connection_id: str
    generation: int
    activation_epoch: int
    endpoint: str
    grant_id: str | None
    secret_version: int
    envelope: str | None = field(repr=False)
    authentication: Authentication = "none"
    endpoint_headers: StatedHeaders = ()


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
        self, *, key_ids: Sequence[str], limit: int
    ) -> tuple[StoredGrant, ...]: ...
    async def reencrypt_grant(self, *, grant: StoredGrant, key_id: str, envelope: str) -> bool: ...
    async def collect_garbage(self, *, limit: int) -> int: ...

    async def wait_oauth_callback(self, *, worker_id: str, timeout: float) -> None: ...
    async def create_oauth_flow(
        self, *, flow: OAuthFlow, expected_revision: str, lifetime: float, lease: float
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
    async def abandon_oauth_flow(self, *, owner_id: str, state_hash: str) -> None: ...
    async def publish_authorization(
        self,
        *,
        owner_id: str,
        connection_id: str,
        expected_revision: str,
        endpoint: str,
        grant_id: str,
        kind: GrantKind,
        key_id: str,
        envelope: str,
        scopes: tuple[str, ...],
        catalogue: tuple[RemoteTool, ...],
        policy: ConnectionPolicy,
        instructions: str | None = None,
    ) -> None: ...
    async def complete_authorization(
        self,
        *,
        flow: OAuthFlow,
        key_id: str,
        envelope: str,
        scopes: tuple[str, ...],
        catalogue: tuple[RemoteTool, ...],
        policy: ConnectionPolicy,
        instructions: str | None = None,
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

    async def research_catalogues(self, owner_id: str) -> tuple[PinnedCatalogue, ...]: ...
    async def resource_gate(
        self, *, claim: ResearchToolClaim, runtime: ToolRuntime, connection_id: str
    ) -> DispatchCredentials: ...
    async def pinned_catalogues(
        self, *, owner_id: str, run_id: str, bindings: tuple[RunConnectionBinding, ...]
    ) -> tuple[CatalogueTool, ...]: ...
    async def pinned_tool_facts(
        self, *, owner_id: str, run_id: str
    ) -> tuple[PinnedToolFact, ...]: ...
    async def start_notifications(self) -> None: ...
    async def stop_notifications(self) -> None: ...
    async def wait_refresh(self, timeout: float) -> None: ...
    async def seconds_until_refresh(self) -> float | None: ...
    async def initialize(self, *, validate_only: bool) -> None: ...
    async def read(self, owner_id: str) -> tuple[str, tuple[StoredConnection, ...]]: ...
    async def change(
        self,
        *,
        owner_id: str,
        expected_revision: str,
        command: ConnectionCommand,
        policy: ConnectionPolicy,
        candidate: tuple[RemoteTool, ...] | None = None,
        candidate_instructions: str | None = None,
    ) -> None: ...
    async def install_credential(
        self,
        *,
        owner_id: str,
        connection_id: str,
        expected_revision: str,
        grant_id: str,
        kind: GrantKind,
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
        catalogue: tuple[RemoteTool, ...] | None,
        error: str | None,
        retry_seconds: float,
        policy: ConnectionPolicy,
        instructions: str | None = None,
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


@dataclass(frozen=True)
class OAuthResult:
    credentials: SecretStr = field(repr=False)
    scopes: tuple[str, ...]
    tools: list[dict[str, Any]]
    instructions: str | None = None


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
        endpoint_headers: StatedHeaders = (),
    ) -> OAuthResult: ...
