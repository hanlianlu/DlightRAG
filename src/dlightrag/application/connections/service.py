# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Personal Connections: publication, consent, isolation, and bounded refresh."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any, Literal
from urllib.parse import parse_qs, urlsplit

from jsonschema import Draft202012Validator
from pydantic import BaseModel, RootModel, SecretStr, model_validator

from dlightrag.application.connections.policy import ConnectionPolicy, oauth_callback_problem
from dlightrag.engine.agent.tools.contracts import (
    AgentTool,
    ToolDeclaration,
    ToolExecute,
    ToolResult,
    ToolRuntime,
    declaration_tokens,
)
from dlightrag.engine.ai.messages import ToolDefinition
from dlightrag.engine.answer.execution.connection_binding import (
    ResearchToolClaim,
    RunConnectionBinding,
)
from dlightrag.engine.credential_cipher import CredentialCipher
from dlightrag.engine.network_admission import (
    validate_credential_free_query,
    validate_public_http_url,
)

from .client_metadata import client_metadata_document, client_metadata_url
from .credentials import (
    CredentialTransport,
    GrantCipher,
    credential_access,
    credential_transport,
    header_credential,
)
from .models import (
    AuthorizationStart,
    BoundConnectionSurface,
    BoundResearchConnections,
    CatalogueTool,
    ConnectionCommand,
    ConnectionsError,
    ConnectionsStore,
    ConnectionsView,
    ConnectionView,
    CredentialAccess,
    DispatchCredentials,
    McpClientPort,
    McpResourcePort,
    OAuthFlow,
    OAuthPort,
    PinnedToolFact,
    PresetView,
    RefreshClaim,
    RemoteTool,
    StoredConnection,
    StoredGrant,
    UnsupportedResultError,
)
from .policy import GrantKind, StatedHeaders, credential_header_problem, endpoint_headers_problem
from .presets import PRESETS

logger = logging.getLogger(__name__)


def _surface_tokens(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """The token each Connection is named by: its label, made unique and bounded.

    A label is the owner's own display name. Two labels compare as the prompt renders them
    (whitespace collapsed, at most 80 characters), so a collision is decided on that form and
    both sides take the identity's first eight hex characters; if even those collide the
    prefix grows. The long token is bounded to the tool argument's own ceiling, and no two
    Connections share one, because the host resolves a token to exactly one authority.
    """
    bases: dict[str, str] = {}
    for connection_id, label in pairs:
        normalized = " ".join(label.split())[:80]
        bases[connection_id] = normalized or f"Connection {connection_id[:8]}"
    counts = Counter(bases.values())
    tokens: dict[str, str] = {}
    used: set[str] = set()
    for connection_id in sorted(bases):
        base = bases[connection_id]
        token = base if counts[base] == 1 else ""
        width = 8
        while not token or token in used:
            if width > 32:
                raise ConnectionsError("Connection tokens cannot be made unique")
            token = f"{base[: max(1, 80 - width - 3)]} ({connection_id[:width]})"
            width += 4
        used.add(token)
        tokens[connection_id] = token
    return tokens


def _failure_name(exc: BaseException) -> str:
    """Our own message for a failure we raised, a class name for anyone else's.

    Never a foreign exception's message: an SDK failure can carry an endpoint, a code, or a token.
    """
    return str(exc) if isinstance(exc, ConnectionsError) else type(exc).__name__


# A refresh loop with nothing to claim sleeps until the next refresh falls due or the store wakes
# it on a change. The bound covers what no wake reports, such as a claim that lapsed or a change
# made while wakes could not be delivered: it waits at most this long.
_MAX_REFRESH_SLEEP_SECONDS = 30.0
# How soon a loop retries a store that failed.
_STORE_RETRY_SECONDS = 1.0

# The catalogue is static, so the projection is built once and shared by every read.
_PRESET_VIEWS: tuple[PresetView, ...] = tuple(
    PresetView(
        preset_id=preset.preset_id,
        label=preset.label,
        endpoint=preset.endpoint,
        default_authentication=preset.default_authentication,
        endpoint_headers=preset.endpoint_headers,
        credential_header=preset.credential_header,
        credential_prefix=preset.credential_prefix,
    )
    for preset in PRESETS
)


class Connections:
    def __init__(
        self,
        *,
        store: ConnectionsStore,
        mcp: McpClientPort,
        cipher: CredentialCipher,
        policy: ConnectionPolicy | None = None,
        oauth: OAuthPort | None = None,
        resources: McpResourcePort | None = None,
    ) -> None:
        self._oauth = oauth
        self._authorizations: dict[str, asyncio.Task[None]] = {}
        self._authorization_slots = asyncio.Semaphore(4)
        self._cipher = GrantCipher(cipher)
        self._store = store
        self._mcp = mcp
        self._resources = resources
        self._policy = policy or ConnectionPolicy()
        self._call_slots = asyncio.Semaphore(self._policy.call_concurrency)
        self._calls: dict[tuple[str, str, str, str], asyncio.Task[ToolResult]] = {}
        self._discovery_slots = asyncio.Semaphore(self._policy.discovery_concurrency)
        self._worker = uuid.uuid4().hex
        self._tasks: list[asyncio.Task[None]] = []

    async def bind_research(self, *, owner_id: str) -> BoundResearchConnections:
        """Read only complete enabled local definitions; never discover on acceptance."""
        catalogues = await self._store.research_catalogues(owner_id)
        tokens = _surface_tokens(
            (catalogue.binding.connection_id, catalogue.label) for catalogue in catalogues
        )
        surface = tuple(
            BoundConnectionSurface(
                connection_id=catalogue.binding.connection_id,
                token=tokens[catalogue.binding.connection_id],
                label=catalogue.label,
                instructions=catalogue.instructions,
            )
            for catalogue in catalogues
        )
        return BoundResearchConnections(
            tools=tuple(
                _catalogue_declaration(tool) for catalogue in catalogues for tool in catalogue.tools
            ),
            bindings=tuple(catalogue.binding for catalogue in catalogues),
            surface=surface,
            resource_tools_available=self._resources is not None,
        )

    async def restore_research(
        self, *, bindings: tuple[RunConnectionBinding, ...], claim: ResearchToolClaim
    ) -> tuple[AgentTool, ...]:
        """Restore accepted generations, never current heads or credentials."""
        await claim.check_cancelled()
        tools = await self._store.pinned_catalogues(
            owner_id=claim.owner_id, run_id=claim.run_id, bindings=bindings
        )
        restored = []
        for tool in tools:

            async def execute(
                raw: BaseModel, runtime: ToolRuntime, tool: CatalogueTool = tool
            ) -> ToolResult:
                return await self._dispatch(tool, claim, bindings, raw, runtime)

            restored.append(_catalogue_tool(tool, execute=execute))
        return tuple(restored)

    async def pinned_tool_labels(self, *, owner_id: str, run_id: str) -> Mapping[str, str]:
        """Label this Run's pinned Connection tools for a human reader.

        Display, not authority: the caller may only render these names for a Run
        it already owns, and every failure mode -- a Run with no pins, a retained
        catalogue that no longer resolves -- returns the same empty mapping, so a
        missing label can never fail or blank a trace.
        """
        facts = await self._store.pinned_tool_facts(owner_id=owner_id, run_id=run_id)
        return {fact.local_name: label for fact in facts if (label := _tool_label(fact)) != ""}

    async def _gate_access(
        self,
        *,
        claim: ResearchToolClaim,
        runtime: ToolRuntime,
        dispatch: DispatchCredentials,
        gate: Callable[[], Awaitable[DispatchCredentials]],
    ) -> tuple[CredentialAccess | None, DispatchCredentials]:
        """The credential one gated request presents, re-gated over any OAuth refresh.

        ``gate`` is the caller's own authority check against the Run's pins. It runs again
        after the refresh preflight and once more before the credential is returned, so a
        refresh that races a revoke, disable, or re-enable cannot carry the old authority
        into the request. Callers gate once before this too; this method only refreshes.
        """
        if dispatch.grant_id is None:
            return None, dispatch
        if dispatch.envelope is None:
            raise ConnectionsError("Connection credential unavailable", 503)
        secret = self._cipher.decrypt(
            dispatch.envelope,
            owner_id=claim.owner_id,
            connection_id=dispatch.connection_id,
            grant_id=dispatch.grant_id,
        )

        async def authorize_refresh(grant: StoredGrant | None) -> None:
            await claim.check_cancelled()
            current = await gate()
            if grant is not None and (
                current.grant_id != grant.grant_id or current.secret_version != grant.secret_version
            ):
                raise ConnectionsError("OAuth refresh lost authority", 401)

        await self._access_credentials(
            owner_id=claim.owner_id,
            connection_id=dispatch.connection_id,
            grant_id=dispatch.grant_id,
            endpoint=dispatch.endpoint,
            authentication=dispatch.authentication,
            secret=secret,
            authorize_refresh=authorize_refresh,
        )
        refreshed = await gate()
        if refreshed.envelope is None or refreshed.grant_id is None:
            raise ConnectionsError("Connection credential unavailable", 401)
        return (
            credential_access(
                self._cipher.decrypt(
                    refreshed.envelope,
                    owner_id=claim.owner_id,
                    connection_id=refreshed.connection_id,
                    grant_id=refreshed.grant_id,
                ),
                authentication=refreshed.authentication,
            ),
            refreshed,
        )

    async def _dispatch(
        self,
        tool: CatalogueTool,
        claim: ResearchToolClaim,
        bindings: tuple[RunConnectionBinding, ...],
        raw: BaseModel,
        runtime: ToolRuntime,
    ) -> ToolResult:
        await claim.check_cancelled()
        arguments = raw.model_dump()
        if len(json.dumps(arguments).encode()) > self._policy.max_call_argument_bytes:
            return _call_failure(tool, unknown=False)
        dispatch = None
        error = "transport"
        try:
            async with asyncio.timeout(self._policy.call_timeout), self._call_slots:
                await claim.check_cancelled()
                dispatch = await self._store.dispatch_gate(
                    claim=claim,
                    bindings=bindings,
                    runtime=runtime,
                    tool=tool,
                    arguments=arguments,
                )
                self._validate_endpoint(dispatch.endpoint)
                access, dispatch = await self._gate_access(
                    claim=claim,
                    runtime=runtime,
                    dispatch=dispatch,
                    gate=lambda: self._store.dispatch_gate(
                        claim=claim,
                        bindings=bindings,
                        runtime=runtime,
                        tool=tool,
                        arguments=arguments,
                    ),
                )
                await claim.check_cancelled()
                key = (
                    claim.owner_id,
                    claim.run_id,
                    runtime.execution_scope,
                    runtime.intent_id.value,
                )
                if key in self._calls:
                    return _call_failure(tool, unknown=True)
                task = asyncio.create_task(
                    self._call_mcp(
                        endpoint=dispatch.endpoint,
                        access=access,
                        endpoint_headers=dispatch.endpoint_headers,
                        name=tool.remote_name,
                        arguments=arguments,
                    )
                )
                self._calls[key] = task
                watcher = asyncio.create_task(self._watch_call(task, claim, runtime, dispatch))
                try:
                    result = await task
                    if len(result.text_content.encode()) > self._policy.max_result_bytes:
                        raise ConnectionsError("Connection result quota exceeded")
                    error = "transport" if result.is_error else None
                    return result if not result.is_error else _call_failure(tool, unknown=True)
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
                    return _call_failure(tool, unknown=True)
                finally:
                    watcher.cancel()
                    task.cancel()
                    await asyncio.gather(watcher, task, return_exceptions=True)
                    self._calls.pop(key, None)
        except asyncio.CancelledError:
            raise
        except UnsupportedResultError as exc:
            # The call completed and only its result cannot be shown, so the Connection is
            # still observed as degraded (`error` stays "transport") but nothing is unknown.
            return _unshowable_result(tool, exc.kinds)
        except ConnectionsError as exc:
            error = "authentication" if exc.status == 401 else "transport"
            return _call_failure(tool, unknown=dispatch is not None)
        except TimeoutError:
            return _call_failure(tool, unknown=dispatch is not None)
        finally:
            if dispatch is not None:
                try:
                    async with asyncio.timeout(2):
                        await self._store.observe_call(
                            owner_id=claim.owner_id, dispatch=dispatch, error=error
                        )
                except Exception:
                    logger.warning("Connection observation unavailable")

    async def _call_mcp(
        self,
        *,
        endpoint: str,
        access: CredentialAccess | None,
        endpoint_headers: StatedHeaders,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        try:
            return await self._mcp.call(
                endpoint=endpoint,
                access=access,
                policy=self._policy,
                name=name,
                arguments=arguments,
                endpoint_headers=endpoint_headers,
            )
        except asyncio.CancelledError:
            raise
        except ConnectionsError:
            raise
        except Exception:
            raise ConnectionsError("MCP call failed") from None

    async def _watch_call(
        self,
        task: asyncio.Task[ToolResult],
        claim: ResearchToolClaim,
        runtime: ToolRuntime,
        dispatch: DispatchCredentials,
    ) -> None:
        try:
            while not task.done():
                if not await self._store.dispatch_alive(
                    claim=claim, runtime=runtime, dispatch=dispatch
                ):
                    task.cancel()
                    return
                await self._store.wait_dispatch_change(0.25)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Losing local authority visibility cannot authorize further I/O.
            task.cancel()

    # ------------------------------------------------------------------
    # On-demand resources
    # ------------------------------------------------------------------

    async def list_resources(
        self,
        *,
        claim: ResearchToolClaim,
        runtime: ToolRuntime,
        connection_id: str,
        cursor: str | None = None,
    ) -> ToolResult:
        """One page of the pinned Connection's MCP resource list."""
        return await self._resource_call(
            "list_resources",
            claim=claim,
            runtime=runtime,
            connection_id=connection_id,
            cursor=cursor,
        )

    async def list_resource_templates(
        self,
        *,
        claim: ResearchToolClaim,
        runtime: ToolRuntime,
        connection_id: str,
        cursor: str | None = None,
    ) -> ToolResult:
        """One page of the pinned Connection's MCP resource templates."""
        return await self._resource_call(
            "list_resource_templates",
            claim=claim,
            runtime=runtime,
            connection_id=connection_id,
            cursor=cursor,
        )

    async def read_resource(
        self,
        *,
        claim: ResearchToolClaim,
        runtime: ToolRuntime,
        connection_id: str,
        uri: str,
    ) -> ToolResult:
        """One on-demand read of a resource from the pinned Connection."""
        return await self._resource_call(
            "read_resource",
            claim=claim,
            runtime=runtime,
            connection_id=connection_id,
            uri=uri,
        )

    async def _resource_call(
        self,
        operation: Literal["list_resources", "list_resource_templates", "read_resource"],
        *,
        claim: ResearchToolClaim,
        runtime: ToolRuntime,
        connection_id: str,
        cursor: str | None = None,
        uri: str | None = None,
    ) -> ToolResult:
        """Gate then read one resource operation; a read mutates nothing remote.

        The gate runs before the request and again over an OAuth refresh, the same order the
        tool-call path uses: authority is checked against the Run's pinned generation, never
        the caller's words. While the read is in flight the same watcher a tool call gets
        re-checks the head, Grant, and Run/Child fence, so a revoke or a moved lease cancels
        the read rather than letting it settle after its authority ended. The transport owns
        every bound: timeouts, admission, headers, credential redaction and result size.
        """
        resources = self._resources
        if resources is None:
            raise ConnectionsError("MCP resources unavailable", 503)
        await claim.check_cancelled()
        dispatch: DispatchCredentials | None = None
        error = "transport"
        try:
            async with asyncio.timeout(self._policy.call_timeout), self._call_slots:
                dispatch = await self._store.resource_gate(
                    claim=claim, runtime=runtime, connection_id=connection_id
                )
                self._validate_endpoint(dispatch.endpoint)
                access, dispatch = await self._gate_access(
                    claim=claim,
                    runtime=runtime,
                    dispatch=dispatch,
                    gate=lambda: self._store.resource_gate(
                        claim=claim,
                        runtime=runtime,
                        connection_id=connection_id,
                    ),
                )
                await claim.check_cancelled()
                task = asyncio.create_task(
                    self._read_mcp(
                        operation,
                        endpoint=dispatch.endpoint,
                        access=access,
                        endpoint_headers=dispatch.endpoint_headers,
                        cursor=cursor,
                        uri=uri,
                        resources=resources,
                    )
                )
                watcher = asyncio.create_task(self._watch_call(task, claim, runtime, dispatch))
                try:
                    result = await task
                    if len(result.text_content.encode()) > self._policy.max_result_bytes:
                        raise ConnectionsError("Connection result quota exceeded")
                    error = "transport" if result.is_error else None
                    return result
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
                    raise ConnectionsError(
                        "MCP resource request cancelled; no automatic retry"
                    ) from None
                finally:
                    watcher.cancel()
                    task.cancel()
                    await asyncio.gather(watcher, task, return_exceptions=True)
        except asyncio.CancelledError:
            raise
        except ConnectionsError as exc:
            error = "authentication" if exc.status == 401 else "transport"
            raise
        except Exception:
            # A read has no dispatch ledger and its result is model-visible: anything the
            # transport did not already sanitize becomes one safe product line, and the
            # observation below records the health fact.
            error = "transport"
            raise ConnectionsError("MCP resource request failed") from None
        finally:
            if dispatch is not None:
                try:
                    async with asyncio.timeout(2):
                        await self._store.observe_call(
                            owner_id=claim.owner_id, dispatch=dispatch, error=error
                        )
                except Exception:
                    logger.warning("Connection observation unavailable")

    async def _read_mcp(
        self,
        operation: Literal["list_resources", "list_resource_templates", "read_resource"],
        *,
        endpoint: str,
        access: CredentialAccess | None,
        endpoint_headers: StatedHeaders,
        cursor: str | None,
        uri: str | None,
        resources: McpResourcePort,
    ) -> ToolResult:
        if operation == "read_resource":
            if uri is None:
                raise ConnectionsError("MCP resource URI is required", 422)
            return await resources.read_resource(
                endpoint=endpoint,
                access=access,
                policy=self._policy,
                endpoint_headers=endpoint_headers,
                uri=uri,
            )
        if operation == "list_resources":
            return await resources.list_resources(
                endpoint=endpoint,
                access=access,
                policy=self._policy,
                endpoint_headers=endpoint_headers,
                cursor=cursor,
            )
        return await resources.list_resource_templates(
            endpoint=endpoint,
            access=access,
            policy=self._policy,
            endpoint_headers=endpoint_headers,
            cursor=cursor,
        )

    def published_client_metadata(self, *, callback_url: str) -> dict[str, Any] | None:
        """The Client ID Metadata Document this deployment publishes, or None.

        Public by protocol: an authorization server fetches it without a credential, so it carries
        only what an authorization redirect already reveals. ``callback_url`` is the callback on
        the origin the request reached; a configured override wins.
        """
        callback_url = self._policy.oauth_callback_url or callback_url
        metadata_url = client_metadata_url(callback_url)
        if metadata_url is None or oauth_callback_problem(callback_url):
            return None
        return client_metadata_document(metadata_url=metadata_url, oauth_callback_url=callback_url)

    async def read(self, *, owner_id: str) -> ConnectionsView:
        revision, items = await self._store.read(owner_id)
        return ConnectionsView(
            revision=revision,
            connections=tuple(self._connection_view(item) for item in items),
            presets=_PRESET_VIEWS,
        )

    def _connection_view(self, item: StoredConnection) -> ConnectionView:
        """What Settings renders for one Connection, including what a reload would otherwise lose.

        The credential's own header and prefix are read back here -- they are not secrets -- so a
        card can say which header its credential travels in after the form that typed it is gone.
        Whether the deployment can read them is the deployment's business: no key ring means
        None, and the card says so instead of showing a blank that looks unconfigured.
        """
        transport = self._credential_transport_of(item)
        declarations = tuple(
            ToolDefinition(
                name=tool.local_name,
                description=tool.description,
                parameters=tool.input_schema,
            )
            for tool in item.catalogue
        )
        return ConnectionView(
            connection_id=item.connection_id,
            label=item.label,
            endpoint=item.endpoint,
            enabled=item.enabled,
            activation_epoch=item.activation_epoch,
            generation=item.generation,
            authentication=item.authentication,
            authorization_status=item.authorization_status,
            endpoint_headers=item.endpoint_headers,
            credential_header=transport.header if transport is not None else None,
            credential_prefix=transport.prefix if transport is not None else None,
            status=item.status,
            tool_count=len(item.catalogue) if item.catalogue_created_at is not None else None,
            declaration_tokens=(
                declaration_tokens(declarations)
                if item.catalogue_created_at is not None and item.catalogue
                else 0
                if item.catalogue_created_at is not None
                else None
            ),
        )

    async def change(
        self, *, owner_id: str, expected_revision: str, command: ConnectionCommand
    ) -> ConnectionsView:
        if command.kind == "create" and (not command.label or not command.endpoint):
            raise ConnectionsError("Label and endpoint are required", 422)
        candidate: tuple[RemoteTool, ...] | None = None
        candidate_instructions: str | None = None
        if command.endpoint:
            self._validate_endpoint(command.endpoint)
        if command.endpoint_headers is not None:
            self._validate_endpoint_headers(command.endpoint_headers)
        if command.kind == "edit" and (command.endpoint or command.endpoint_headers is not None):
            revision, items = await self._store.read(owner_id)
            item = next(
                (item for item in items if item.connection_id == command.connection_id), None
            )
            if item is None:
                raise ConnectionsError("Connection not found", 404)
            if revision != expected_revision:
                raise ConnectionsError("Connections revision changed")
            if command.endpoint_headers is not None:
                self._validate_endpoint_headers(
                    command.endpoint_headers, credential_header=self._credential_header_of(item)
                )
            if command.endpoint and item.endpoint != command.endpoint:
                if item.grant_id:
                    raise ConnectionsError(
                        "Endpoint candidate needs a new grant", 409, kind="requires_reauthorization"
                    )
                stated = (
                    item.endpoint_headers
                    if command.endpoint_headers is None
                    else command.endpoint_headers
                )
                candidate, candidate_instructions = await self._discover(
                    command.endpoint, endpoint_headers=stated
                )
        await self._store.change(
            owner_id=owner_id,
            expected_revision=expected_revision,
            command=command,
            policy=self._policy,
            candidate=candidate,
            candidate_instructions=candidate_instructions,
        )
        if command.kind == "probe":
            claim = await self._store.claim(
                worker_id=self._worker,
                lease_seconds=2 * self._policy.discovery_timeout + 5,
                owner_id=owner_id,
                connection_id=command.connection_id,
            )
            if claim is not None:
                await self._refresh(claim)
        return await self.read(owner_id=owner_id)

    async def replace_bearer(
        self,
        *,
        owner_id: str,
        connection_id: str,
        bearer: SecretStr,
        expected_revision: str | None = None,
        endpoint: str | None = None,
    ) -> ConnectionsView:
        """Store a personal bearer: the default instance of the one credential rule.

        A bearer is a token in ``Authorization: Bearer``, and nothing about this command
        changes for it: the envelope still holds the token alone, so every stored row and every
        pinned generation reads exactly as it did.
        """
        self._read_credential(bearer, label="bearer")
        return await self._replace_credential(
            owner_id=owner_id,
            connection_id=connection_id,
            kind="bearer",
            credential=bearer,
            expected_revision=expected_revision,
            endpoint=endpoint,
        )

    async def replace_header_credential(
        self,
        *,
        owner_id: str,
        connection_id: str,
        header: str,
        prefix: str,
        token: SecretStr,
        expected_revision: str | None = None,
        endpoint: str | None = None,
    ) -> ConnectionsView:
        """Store a credential that travels in a header of its own.

        The header name and prefix are not Connection columns: they are sealed with the token
        into the one envelope, so a Connection states which header it uses only by being
        decrypted at dispatch. Which is what makes an endpoint that wants ``X-Lark-MCP-TAT``
        (and no prefix) an ordinary Connection rather than a special case.
        """
        if (problem := credential_header_problem(header, prefix)) is not None:
            raise ConnectionsError(problem, 422)
        self._read_credential(token, label="header")
        return await self._replace_credential(
            owner_id=owner_id,
            connection_id=connection_id,
            kind="header",
            credential=header_credential(header, prefix, token),
            expected_revision=expected_revision,
            endpoint=endpoint,
        )

    @staticmethod
    def _read_credential(secret: SecretStr, *, label: str) -> None:
        """One token, in the shape every outbound header can carry: no control characters."""
        value = secret.get_secret_value()
        if (
            not value
            or len(value) > 8192
            or any(ord(char) < 33 or ord(char) > 126 for char in value)
        ):
            raise ConnectionsError(f"Invalid {label} credential", 422)

    async def _replace_credential(
        self,
        *,
        owner_id: str,
        connection_id: str,
        kind: GrantKind,
        credential: SecretStr,
        expected_revision: str | None,
        endpoint: str | None,
    ) -> ConnectionsView:
        revision, items = await self._store.read(owner_id)
        item = next((item for item in items if item.connection_id == connection_id), None)
        if item is None:
            raise ConnectionsError("Connection not found", 404)
        access = credential_access(credential, authentication=kind)
        # A credential is applied after what the endpoint states, so the two may not name the
        # same header. Checked here, where both are known, and again by the transport itself.
        self._validate_endpoint_headers(item.endpoint_headers, credential_header=access.header)
        grant_id = uuid.uuid4().hex
        key_id, envelope = self._cipher.encrypt(
            credential, owner_id=owner_id, connection_id=connection_id, grant_id=grant_id
        )
        if endpoint is not None:
            self._validate_endpoint(endpoint)
            if expected_revision is None or revision != expected_revision:
                raise ConnectionsError("Connections revision changed")
            catalogue, instructions = await self._discover(
                endpoint, access=access, endpoint_headers=item.endpoint_headers
            )
            await self._store.publish_authorization(
                owner_id=owner_id,
                connection_id=connection_id,
                expected_revision=expected_revision,
                endpoint=endpoint,
                grant_id=grant_id,
                kind=kind,
                key_id=key_id,
                envelope=envelope,
                scopes=(),
                catalogue=catalogue,
                policy=self._policy,
                instructions=instructions,
            )
            return await self.read(owner_id=owner_id)
        await self._store.install_credential(
            owner_id=owner_id,
            connection_id=connection_id,
            expected_revision=expected_revision if expected_revision is not None else revision,
            grant_id=grant_id,
            kind=kind,
            key_id=key_id,
            envelope=envelope,
        )
        claim = await self._store.claim(
            worker_id=self._worker,
            lease_seconds=2 * self._policy.discovery_timeout + 5,
            owner_id=owner_id,
            connection_id=connection_id,
        )
        if claim is not None:
            await self._refresh(claim)
        return await self.read(owner_id=owner_id)

    async def begin_authorization(
        self,
        *,
        owner_id: str,
        connection_id: str,
        expected_revision: str,
        callback_url: str,
        endpoint: str | None = None,
    ) -> AuthorizationStart:
        """Start authorizing one Connection.

        ``callback_url`` is the callback route on the origin this request reached, the
        browser's own (the Web's same-origin guard ties the two): the provider sends that
        browser, with its session, back there. A configured override wins.
        """
        if self._oauth is None:
            raise ConnectionsError("OAuth callback deployment is not configured", 503)
        if len(self._authorizations) >= 4:
            raise ConnectionsError("Authorization quota exceeded", 429)
        callback_url = self._policy.oauth_callback_url or callback_url
        if problem := oauth_callback_problem(callback_url):
            raise ConnectionsError(problem, 422)
        revision, items = await self._store.read(owner_id)
        item = next((item for item in items if item.connection_id == connection_id), None)
        if item is None:
            raise ConnectionsError("Connection not found", 404)
        if revision != expected_revision:
            raise ConnectionsError("Connections revision changed")
        candidate = endpoint if endpoint is not None else item.endpoint
        self._validate_endpoint(candidate)
        flow = OAuthFlow(uuid.uuid4().hex, owner_id, connection_id, self._worker, candidate)
        # Fail missing keys before any external registration or flow mutation.
        self._cipher.encrypt(
            SecretStr("{}"), owner_id=owner_id, connection_id=connection_id, grant_id=flow.flow_id
        )
        await self._store.create_oauth_flow(
            flow=flow,
            expected_revision=expected_revision,
            lifetime=self._policy.oauth_timeout,
            lease=10,
        )
        ready: asyncio.Future[AuthorizationStart] = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            self._authorize_flow(flow, callback_url, ready, endpoint_headers=item.endpoint_headers)
        )
        self._authorizations[flow.flow_id] = task
        task.add_done_callback(lambda _: self._authorizations.pop(flow.flow_id, None))
        try:
            async with asyncio.timeout(self._policy.discovery_timeout):
                return await asyncio.shield(ready)
        except BaseException as exc:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if ready.done() and not ready.cancelled():
                ready.exception()
            if isinstance(exc, TimeoutError):
                raise ConnectionsError(
                    "OAuth authorization timed out; restart from Settings", 401
                ) from None
            raise

    async def authorization_callback(
        self,
        *,
        owner_id: str,
        state: str,
        code: str | None = None,
        issuer: str | None = None,
        error: str | None = None,
    ) -> None:
        if (
            not state
            or len(state) > 256
            or (bool(code) == bool(error))
            or any(len(value) > 8192 for value in (code, issuer, error) if value)
        ):
            raise ConnectionsError("Authorization callback invalid; restart from Settings", 400)
        state_hash = hashlib.sha256(state.encode()).hexdigest()
        flow = await self._store.oauth_callback_flow(owner_id=owner_id, state_hash=state_hash)
        _, envelope = self._cipher.encrypt(
            SecretStr(
                json.dumps({"state": state, "code": code, "iss": issuer, "error": bool(error)})
            ),
            owner_id=owner_id,
            connection_id=flow.connection_id,
            grant_id=flow.flow_id + ":callback",
        )
        await self._store.deposit_oauth_callback(
            flow=flow, state_hash=state_hash, envelope=envelope
        )

    async def abandon_authorization(self, *, owner_id: str, state: str) -> None:
        """Finish the flow a refused callback named, so its Connection reports the failure at once.

        Only a live flow that took no deposit is finished, so a callback that won the race keeps
        its exchange, and a state this deployment cannot match leaves every flow as it was.
        """
        await self._store.abandon_oauth_flow(
            owner_id=owner_id, state_hash=hashlib.sha256(state.encode()).hexdigest()
        )

    async def _authorize_flow(
        self,
        flow: OAuthFlow,
        callback_url: str,
        ready: asyncio.Future[AuthorizationStart],
        *,
        endpoint_headers: StatedHeaders = (),
    ) -> None:
        async def redirect(url: str) -> None:
            state = parse_qs(urlsplit(url).query)["state"][0]
            await self._store.oauth_redirect(
                flow=flow, state_hash=hashlib.sha256(state.encode()).hexdigest()
            )
            if not ready.done():
                ready.set_result(AuthorizationStart(url))

        async def callback() -> SecretStr:
            while True:
                envelope = await self._store.consume_oauth_callback(flow=flow)
                if envelope is not None:
                    return self._cipher.decrypt(
                        envelope,
                        owner_id=flow.owner_id,
                        connection_id=flow.connection_id,
                        grant_id=flow.flow_id + ":callback",
                    )
                await self._store.wait_oauth_callback(worker_id=self._worker, timeout=0.25)

        async def save(credentials: SecretStr) -> None:
            _, envelope = self._cipher.encrypt(
                credentials,
                owner_id=flow.owner_id,
                connection_id=flow.connection_id,
                grant_id=flow.flow_id,
            )
            await self._store.oauth_credentials(flow=flow, envelope=envelope)

        task = asyncio.current_task()

        async def heartbeat() -> None:
            try:
                while True:
                    await asyncio.sleep(2)
                    if not await self._store.renew_oauth_flow(flow=flow, lease=10):
                        break
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Authorization lease unavailable; cancelling flow")
            if task is not None:
                task.cancel()

        watcher = asyncio.create_task(heartbeat())
        try:
            async with asyncio.timeout(self._policy.oauth_timeout), self._authorization_slots:
                oauth = self._oauth
                if oauth is None:
                    raise ConnectionsError("OAuth unavailable", 503)
                result = await oauth.authorize(
                    endpoint=flow.endpoint,
                    callback_url=callback_url,
                    policy=self._policy,
                    redirect=redirect,
                    callback=callback,
                    save=save,
                    endpoint_headers=endpoint_headers,
                )
                catalogue = self._validate_catalogue(result.tools)
                instructions = self._validate_instructions(result.instructions)
                key_id, envelope = self._cipher.encrypt(
                    result.credentials,
                    owner_id=flow.owner_id,
                    connection_id=flow.connection_id,
                    grant_id=flow.flow_id,
                )
                # Publishes only if the Connection is still where the flow began; otherwise the
                # flow ends as changed and the owner's later command stands.
                await self._store.complete_authorization(
                    flow=flow,
                    key_id=key_id,
                    envelope=envelope,
                    scopes=result.scopes,
                    catalogue=catalogue,
                    policy=self._policy,
                    instructions=instructions,
                )
        except (Exception, asyncio.CancelledError) as exc:
            # The one place that sees every way an authorization can end without publishing: a
            # cancelled task, a failure the SDK wrapped, and a failure of our own. Silence here
            # left an operator with a card that said nothing and a flow row that said less.
            logger.warning("Authorization ended without publishing: %s", _failure_name(exc))
            if not ready.done():
                ready.set_exception(
                    ConnectionsError("OAuth authorization failed; restart from Settings", 401)
                )
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            try:
                await self._store.finish_oauth_flow(flow=flow)
            except Exception:
                logger.warning("Authorization cleanup unavailable; flow will expire")

    def _validate_endpoint(self, endpoint: str) -> None:
        try:
            validate_public_http_url(endpoint, allow_private_hosts=self._policy.allow_private_hosts)
            parts = urlsplit(endpoint)
            validate_credential_free_query(endpoint)
            if (
                "#" in endpoint
                or any(ord(char) < 32 or char.isspace() for char in endpoint)
                or (self._policy.require_https and parts.scheme != "https")
            ):
                raise ValueError
        except ValueError:
            raise ConnectionsError("Endpoint rejected by network policy", 422) from None

    def _validate_endpoint_headers(
        self,
        headers: StatedHeaders,
        *,
        credential_header: str | None = None,
    ) -> None:
        """Refuse stated headers this deployment cannot send, or that a credential claims.

        Nothing here is a secret, and nothing may become one: a credential travels in the
        sealed Grant, and the one header it names is refused as an endpoint header.
        """
        problem = endpoint_headers_problem(headers, credential_header=credential_header)
        if problem is not None:
            raise ConnectionsError(problem, 422)

    def _credential_transport_of(self, item: StoredConnection) -> CredentialTransport | None:
        """The header a stored credential travels in and its prefix, when they can be read.

        Read for a ``header`` credential alone: ``bearer`` and ``oauth`` are the default instance
        and need no reading. A credential no key opens needs authorization before it can be used,
        and the owner's next credential replaces it -- so this answers None, and the Settings
        card says the deployment cannot read the key ring rather than failing its own read.
        """
        if item.authentication != "header" or item.grant_id is None or not item.envelope:
            return None
        try:
            secret = self._cipher.decrypt(
                item.envelope,
                owner_id=item.owner_id,
                connection_id=item.connection_id,
                grant_id=item.grant_id,
            )
            return credential_transport(secret, authentication=item.authentication)
        except ConnectionsError:
            return None

    def _credential_header_of(self, item: StoredConnection) -> str | None:
        """The header this Connection's credential travels in, for the overlap check."""
        transport = self._credential_transport_of(item)
        return transport.header if transport is not None else None

    async def _discover(
        self,
        endpoint: str,
        *,
        access: CredentialAccess | None = None,
        endpoint_headers: StatedHeaders = (),
    ) -> tuple[tuple[RemoteTool, ...], str | None]:
        self._validate_endpoint(endpoint)
        async with asyncio.timeout(self._policy.discovery_timeout), self._discovery_slots:
            discovery = await self._mcp.discover(
                endpoint=endpoint,
                access=access,
                policy=self._policy,
                endpoint_headers=endpoint_headers,
            )
        return self._validate_catalogue(list(discovery.tools)), self._validate_instructions(
            discovery.instructions
        )

    def _validate_instructions(self, value: object) -> str | None:
        """The server's own initialize instructions, bounded; empty is no instructions."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ConnectionsError("MCP instructions invalid")
        if len(value.encode()) > self._policy.max_instruction_bytes:
            raise ConnectionsError("MCP instructions too large")
        text = value.strip()
        return text or None

    def _catalogue_shape(self, raw: list[dict[str, Any]]) -> str:
        """What a catalogue weighed, for the line a rejection writes: counts and bytes, no content.

        A rejection that only says "catalogue" leaves an operator comparing four limits against
        nothing; the numbers say which one the server's own catalogue ran past, and by how much.
        """
        items = [item for item in raw if isinstance(item, dict)]
        descriptions = [len(str(item.get("description", "")).encode()) for item in items]
        schemas = [
            len(json.dumps(item.get("input_schema"), default=str).encode()) for item in items
        ]
        return (
            f"{len(raw)} tools, {len(json.dumps(raw, default=str).encode())} bytes total, largest "
            f"description {max(descriptions, default=0)} bytes, largest schema "
            f"{max(schemas, default=0)} bytes (limits {self._policy.max_tools} tools, "
            f"{self._policy.max_catalogue_bytes} bytes, {self._policy.max_description_bytes} "
            f"description bytes, {self._policy.max_schema_bytes} schema bytes)"
        )

    def _validate_catalogue(self, raw: list[dict[str, Any]]) -> tuple[RemoteTool, ...]:
        """Validate a server's catalogue whole; publication names the tools."""
        shape = self._catalogue_shape(raw)
        if len(raw) > self._policy.max_tools:
            raise ConnectionsError(f"catalogue: too many tools; {shape}")
        tools: list[RemoteTool] = []
        names: set[str] = set()
        for item in raw:
            name = item.get("name")
            description = item.get("description", "")
            schema = item.get("input_schema")
            if (
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name)
                or name in names
            ):
                raise ConnectionsError(
                    f"catalogue: a tool name is missing, repeated, or not allowed; {shape}"
                )
            if not isinstance(description, str):
                raise ConnectionsError(f"catalogue: a tool has no description text; {shape}")
            if len(description.encode()) > self._policy.max_description_bytes:
                raise ConnectionsError(f"catalogue: a description is too long; {shape}")
            if not isinstance(schema, dict) or schema.get("type") != "object":
                raise ConnectionsError(f"catalogue: a tool has no object input schema; {shape}")
            if len(json.dumps(schema).encode()) > self._policy.max_schema_bytes:
                raise ConnectionsError(f"catalogue: a schema is too large; {shape}")
            Draft202012Validator.check_schema(schema)
            names.add(name)
            tools.append(RemoteTool(name, description, schema))
        if len(json.dumps(raw, default=str).encode()) > self._policy.max_catalogue_bytes:
            raise ConnectionsError(f"catalogue: too large; {shape}")
        return tuple(sorted(tools, key=lambda tool: tool.remote_name))

    async def _access_credentials(
        self,
        *,
        owner_id: str,
        connection_id: str,
        grant_id: str,
        endpoint: str,
        authentication: str,
        secret: SecretStr,
        authorize_refresh: Callable[[StoredGrant | None], Awaitable[None]] | None = None,
        on_grant: Callable[[StoredGrant], None] | None = None,
    ) -> tuple[CredentialAccess, StoredGrant | None]:
        try:
            return credential_access(secret, authentication=authentication), None
        except ConnectionsError as exc:
            if authentication != "oauth" or exc.status != 401 or self._oauth is None:
                raise
        async with asyncio.timeout(self._policy.discovery_timeout):
            while True:
                if authorize_refresh is not None:
                    await authorize_refresh(None)
                lease = await self._store.claim_grant_refresh(
                    owner_id=owner_id,
                    connection_id=connection_id,
                    grant_id=grant_id,
                    endpoint=endpoint,
                    worker_id=self._worker,
                    lease_seconds=self._policy.discovery_timeout + 5,
                )
                if lease is not None:
                    break
                await self._store.wait_dispatch_change(0.1)
            grant = lease.grant
            if on_grant is not None:
                on_grant(grant)
            try:
                current = self._cipher.decrypt(
                    grant.envelope,
                    owner_id=owner_id,
                    connection_id=connection_id,
                    grant_id=grant_id,
                )
                try:
                    return credential_access(current, authentication="oauth"), grant
                except ConnectionsError:
                    pass

                async def save(value: SecretStr) -> None:
                    nonlocal grant
                    key_id, envelope = self._cipher.encrypt(
                        value, owner_id=owner_id, connection_id=connection_id, grant_id=grant_id
                    )
                    if not await self._store.save_grant_refresh(
                        claim=lease, key_id=key_id, envelope=envelope
                    ):
                        raise ConnectionsError("OAuth refresh lost authority", 401)
                    grant = replace(
                        grant,
                        secret_version=grant.secret_version + 1,
                        envelope=envelope,
                        key_id=key_id,
                    )
                    if on_grant is not None:
                        on_grant(grant)

                async with self._discovery_slots:
                    if authorize_refresh is not None:
                        await authorize_refresh(grant)
                    updated = await self._oauth.refresh(
                        endpoint=endpoint,
                        credentials=current,
                        scopes=grant.scopes,
                        policy=self._policy,
                        save=save,
                    )
                return credential_access(updated, authentication="oauth"), grant
            finally:
                await self._store.release_grant_refresh(claim=lease)

    async def _refresh(self, claim: RefreshClaim) -> None:
        catalogue: tuple[RemoteTool, ...] | None = None
        instructions: str | None = None
        error = None
        try:
            item = claim.connection
            access = None
            if item.grant_id:
                if not item.envelope:
                    raise ConnectionsError("Credential deployment cannot read grant", 503)
                secret = self._cipher.decrypt(
                    item.envelope,
                    owner_id=item.owner_id,
                    connection_id=item.connection_id,
                    grant_id=item.grant_id,
                )

                def on_grant(grant: StoredGrant) -> None:
                    nonlocal claim
                    claim = replace(
                        claim,
                        connection=replace(
                            item,
                            secret_version=grant.secret_version,
                            grant_refresh_epoch=grant.refresh_epoch,
                        ),
                    )

                access, _ = await self._access_credentials(
                    owner_id=item.owner_id,
                    connection_id=item.connection_id,
                    grant_id=item.grant_id,
                    endpoint=item.endpoint,
                    authentication=item.authentication,
                    secret=secret,
                    on_grant=on_grant,
                )
            catalogue, instructions = await self._discover(
                item.endpoint, access=access, endpoint_headers=item.endpoint_headers
            )
        except asyncio.CancelledError:
            raise
        except ConnectionsError as exc:
            error = (
                "deployment"
                if exc.status == 503
                else "authentication"
                if exc.status == 401
                else "discovery"
            )
        except Exception:
            error = "discovery"
        await self._store.publish(
            claim=claim,
            catalogue=catalogue,
            error=error,
            retry_seconds=self._policy.refresh_seconds,
            policy=self._policy,
            instructions=instructions,
        )

    async def maintain(self) -> dict[str, int]:
        """One bounded writer maintenance pass; no remote I/O or key-source reads.

        Grants under a retired key move to the active one. A grant no key opens is
        left to its owner's next authorization, and never stops collection.
        """
        rotated = 0
        if retired := self._cipher.retired_key_ids:
            grants = await self._store.rotation_candidates(key_ids=retired, limit=100)
            for grant in grants:
                try:
                    secret = self._cipher.decrypt(
                        grant.envelope,
                        owner_id=grant.owner_id,
                        connection_id=grant.connection_id,
                        grant_id=grant.grant_id,
                    )
                except ConnectionsError:
                    continue
                key_id, envelope = self._cipher.encrypt(
                    secret,
                    owner_id=grant.owner_id,
                    connection_id=grant.connection_id,
                    grant_id=grant.grant_id,
                )
                rotated += await self._store.reencrypt_grant(
                    grant=grant, key_id=key_id, envelope=envelope
                )
        collected = await self._store.collect_garbage(limit=100)
        return {"reencrypted": rotated, "collected": collected}

    async def _maintain_forever(self) -> None:
        while True:
            try:
                await self.maintain()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Connection maintenance unavailable; check credential deployment and storage"
                )
            await asyncio.sleep(min(60, self._policy.refresh_seconds))

    async def start(self, *, validate_only: bool = False) -> None:
        await self._store.initialize(validate_only=validate_only)
        await self._store.start_notifications()
        if not self._tasks:
            self._tasks = [
                asyncio.create_task(self._refresh_forever())
                for _ in range(self._policy.discovery_concurrency)
            ]
            if not validate_only:
                self._tasks.append(asyncio.create_task(self._maintain_forever()))

    async def _refresh_forever(self) -> None:
        while True:
            sleep = _STORE_RETRY_SECONDS
            try:
                claim = await self._store.claim(
                    worker_id=self._worker, lease_seconds=2 * self._policy.discovery_timeout + 5
                )
                if claim is not None:
                    await self._refresh(claim)
                    continue
                due = await self._store.seconds_until_refresh()
                sleep = (
                    _MAX_REFRESH_SLEEP_SECONDS
                    if due is None
                    else min(due, _MAX_REFRESH_SLEEP_SECONDS)
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Connection refresh store unavailable; retrying")
            await self._store.wait_refresh(sleep)

    async def stop_refresh(self) -> None:
        tasks, self._tasks = [*self._tasks, *self._authorizations.values()], []
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._store.stop_notifications()

    async def aclose(self) -> None:
        await self.stop_refresh()
        tasks: tuple[asyncio.Task[Any], ...] = (
            *self._calls.values(),
            *self._authorizations.values(),
        )
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


_MAX_TOOL_LABEL_CHARS = 96


def _tool_label(fact: PinnedToolFact) -> str:
    """One line naming an owner's remote tool, without exposing its connection identity.

    The Connection label is owner-authored and the remote name is remote-authored, so
    the result is bounded here rather than at the wire, and whitespace is collapsed
    before it reaches a single-line activity row.
    """
    label = " ".join(fact.connection_label.split())
    remote = fact.remote_name.strip()
    if not label:
        return remote[:_MAX_TOOL_LABEL_CHARS]
    if not remote:
        return label[:_MAX_TOOL_LABEL_CHARS]
    text = f"{label} · {remote}"
    return text[:_MAX_TOOL_LABEL_CHARS]


def _call_failure(tool: CatalogueTool, *, unknown: bool) -> ToolResult:
    outcome = "Outcome may be unknown" if unknown else "No call was sent"
    return ToolResult.text(
        f"Connection tool {tool.local_name} unavailable. {outcome}. Do not retry automatically; identify the requested part not completed in the final Answer.",
        is_error=True,
    )


def _unshowable_result(tool: CatalogueTool, kinds: tuple[str, ...]) -> ToolResult:
    return ToolResult.text(
        f"Connection tool {tool.local_name} completed, but its result holds "
        f"{', '.join(kinds)} content that cannot be shown here. Do not retry; say in the "
        "final Answer that this result could not be shown.",
        is_error=True,
    )


def _catalogue_declaration(tool: CatalogueTool) -> ToolDeclaration:
    schema = deepcopy(tool.input_schema)

    class CatalogueArguments(RootModel[dict[str, Any]]):
        @model_validator(mode="after")
        def validate_catalogue(self):
            if not Draft202012Validator(schema).is_valid(self.root):
                raise ValueError("Arguments do not match the accepted tool schema")
            return self

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            # Published, not generated: a schema generator passed in does not apply.
            return deepcopy(schema)

    return ToolDeclaration(
        name=tool.local_name,
        description=tool.description,
        input_model=CatalogueArguments,
        replay_policy="never",
    )


def _catalogue_tool(tool: CatalogueTool, *, execute: ToolExecute) -> AgentTool:
    used: set[tuple[str, str]] = set()

    async def call(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        key = (runtime.execution_scope, runtime.intent_id.value)
        if key in used:
            return _call_failure(tool, unknown=True)
        used.add(key)
        return await execute(raw, runtime)

    return _catalogue_declaration(tool).bind(call)
