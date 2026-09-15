# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Personal Connections: publication, consent, isolation, and bounded refresh."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import replace
from typing import Any
from urllib.parse import parse_qs, urlsplit

from jsonschema import Draft202012Validator
from pydantic import BaseModel, RootModel, SecretStr, model_validator

from dlightrag.application.connections.policy import ConnectionPolicy
from dlightrag.engine.agent.tools import AgentTool, ToolResult, ToolRuntime
from dlightrag.engine.answer.execution.connection_binding import (
    ResearchToolClaim,
    RunConnectionBinding,
)
from dlightrag.engine.network_admission import (
    validate_credential_free_query,
    validate_public_http_url,
)

from .client_metadata import client_metadata_document, client_metadata_url
from .credentials import CredentialCipher, access_bearer
from .models import (
    AuthorizationStart,
    BoundResearchConnections,
    CatalogueTool,
    ConnectionCommand,
    ConnectionsError,
    ConnectionsStore,
    ConnectionsView,
    ConnectionView,
    DispatchCredentials,
    McpClientPort,
    OAuthFlow,
    OAuthPort,
    PresetView,
    RefreshClaim,
    StoredGrant,
)
from .presets import PRESETS

logger = logging.getLogger(__name__)

# The catalogue is static, so the projection is built once and shared by every read.
_PRESET_VIEWS: tuple[PresetView, ...] = tuple(
    PresetView(
        preset_id=preset.preset_id,
        label=preset.label,
        endpoint=preset.endpoint,
        default_authentication=preset.default_authentication,
    )
    for preset in PRESETS
)


class Connections:
    def __init__(
        self,
        *,
        store: ConnectionsStore,
        mcp: McpClientPort,
        policy: ConnectionPolicy | None = None,
        cipher: CredentialCipher | None = None,
        oauth: OAuthPort | None = None,
    ) -> None:
        self._oauth = oauth
        self._authorizations: dict[str, asyncio.Task[None]] = {}
        self._authorization_slots = asyncio.Semaphore(4)
        self._cipher = cipher or CredentialCipher(None)
        self._store = store
        self._mcp = mcp
        self._policy = policy or ConnectionPolicy()
        self._call_slots = asyncio.Semaphore(self._policy.call_concurrency)
        self._calls: dict[tuple[str, str, str, str], asyncio.Task[ToolResult]] = {}
        self._discovery_slots = asyncio.Semaphore(self._policy.discovery_concurrency)
        self._worker = uuid.uuid4().hex
        self._tasks: list[asyncio.Task[None]] = []

    @staticmethod
    def eligible(auth_mode: str) -> bool:
        return auth_mode in {"jwt", "none"}

    def _authorize(self, owner_id: str, auth_mode: str) -> None:
        if not owner_id or not self.eligible(auth_mode):
            raise ConnectionsError(
                "Personal Connections unavailable for this authentication mode", 403
            )

    async def bind_research(self, *, owner_id: str, auth_mode: str) -> BoundResearchConnections:
        """Read only complete enabled local definitions; never discover on acceptance."""
        if not owner_id or not self.eligible(auth_mode):
            return BoundResearchConnections()
        catalogues = await self._store.research_catalogues(owner_id)
        return BoundResearchConnections(
            tools=tuple(_catalogue_tool(tool) for _, tools in catalogues for tool in tools),
            bindings=tuple(binding for binding, _ in catalogues),
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
                bearer = None
                if dispatch.grant_id is not None:
                    if dispatch.envelope is None:
                        raise ConnectionsError("Connection credential unavailable", 503)
                    bearer = self._cipher.decrypt(
                        dispatch.envelope,
                        owner_id=claim.owner_id,
                        connection_id=dispatch.connection_id,
                        grant_id=dispatch.grant_id,
                    )

                    async def authorize_refresh(grant: StoredGrant | None) -> None:
                        await claim.check_cancelled()
                        current = await self._store.dispatch_gate(
                            claim=claim,
                            bindings=bindings,
                            runtime=runtime,
                            tool=tool,
                            arguments=arguments,
                        )
                        if grant is not None and (
                            current.grant_id != grant.grant_id
                            or current.secret_version != grant.secret_version
                        ):
                            raise ConnectionsError("OAuth refresh lost authority", 401)

                    bearer, _ = await self._access_credentials(
                        owner_id=claim.owner_id,
                        connection_id=dispatch.connection_id,
                        grant_id=dispatch.grant_id,
                        endpoint=dispatch.endpoint,
                        authentication=dispatch.authentication,
                        secret=bearer,
                        authorize_refresh=authorize_refresh,
                    )
                    # Refresh is preflight only. Recheck every pending-effect,
                    # Run/Child, activation and Grant authority before MCP I/O.
                    dispatch = await self._store.dispatch_gate(
                        claim=claim,
                        bindings=bindings,
                        runtime=runtime,
                        tool=tool,
                        arguments=arguments,
                    )
                    if dispatch.envelope is None or dispatch.grant_id is None:
                        raise ConnectionsError("Connection credential unavailable", 401)
                    bearer = access_bearer(
                        self._cipher.decrypt(
                            dispatch.envelope,
                            owner_id=claim.owner_id,
                            connection_id=dispatch.connection_id,
                            grant_id=dispatch.grant_id,
                        ),
                        authentication=dispatch.authentication,
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
                        bearer=bearer,
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
        self, *, endpoint: str, bearer: SecretStr | None, name: str, arguments: dict[str, Any]
    ) -> ToolResult:
        try:
            return await self._mcp.call(
                endpoint=endpoint,
                bearer=bearer,
                policy=self._policy,
                name=name,
                arguments=arguments,
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

    def published_client_metadata(self) -> dict[str, Any] | None:
        """The Client ID Metadata Document this deployment publishes, or None.

        Public by protocol: an authorization server fetches it without a credential, so it carries
        only what an authorization redirect already reveals.
        """
        callback_url = self._policy.oauth_callback_url
        metadata_url = client_metadata_url(callback_url)
        if metadata_url is None or callback_url is None:
            return None
        return client_metadata_document(metadata_url=metadata_url, oauth_callback_url=callback_url)

    async def read(self, *, owner_id: str, auth_mode: str) -> ConnectionsView:
        self._authorize(owner_id, auth_mode)
        revision, items = await self._store.read(owner_id)
        return ConnectionsView(
            revision=revision,
            connections=tuple(
                ConnectionView(
                    connection_id=item.connection_id,
                    label=item.label,
                    endpoint=item.endpoint,
                    enabled=item.enabled,
                    activation_epoch=item.activation_epoch,
                    generation=item.generation,
                    authentication=item.authentication,
                    authorization_status=item.authorization_status,
                    status=item.status,
                )
                for item in items
            ),
            presets=_PRESET_VIEWS,
        )

    async def change(
        self, *, owner_id: str, auth_mode: str, expected_revision: str, command: ConnectionCommand
    ) -> ConnectionsView:
        self._authorize(owner_id, auth_mode)
        if command.kind == "create" and (not command.label or not command.endpoint):
            raise ConnectionsError("Label and endpoint are required", 422)
        candidate = None
        if command.endpoint:
            self._validate_endpoint(command.endpoint)
        if command.kind == "edit" and command.endpoint:
            revision, items = await self._store.read(owner_id)
            item = next(
                (item for item in items if item.connection_id == command.connection_id), None
            )
            if item is None:
                raise ConnectionsError("Connection not found", 404)
            if revision != expected_revision:
                raise ConnectionsError("Connections revision changed")
            if item.endpoint != command.endpoint:
                if item.grant_id:
                    raise ConnectionsError(
                        "Endpoint candidate needs a new grant", 409, kind="requires_reauthorization"
                    )
                candidate = await self._discover(item.connection_id, command.endpoint)
        await self._store.change(
            owner_id=owner_id,
            expected_revision=expected_revision,
            command=command,
            policy=self._policy,
            candidate=candidate,
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
        return await self.read(owner_id=owner_id, auth_mode=auth_mode)

    async def replace_bearer(
        self,
        *,
        owner_id: str,
        auth_mode: str,
        connection_id: str,
        bearer: SecretStr,
        expected_revision: str | None = None,
        endpoint: str | None = None,
    ) -> ConnectionsView:
        self._authorize(owner_id, auth_mode)
        revision, items = await self._store.read(owner_id)
        item = next((item for item in items if item.connection_id == connection_id), None)
        if item is None:
            raise ConnectionsError("Connection not found", 404)
        if item.envelope and item.grant_id:
            self._cipher.decrypt(
                item.envelope,
                owner_id=owner_id,
                connection_id=connection_id,
                grant_id=item.grant_id,
            )
        value = bearer.get_secret_value()
        if (
            not value
            or len(value) > 8192
            or any(ord(char) < 33 or ord(char) > 126 for char in value)
        ):
            raise ConnectionsError("Invalid bearer credential", 422)
        grant_id = uuid.uuid4().hex
        key_id, envelope = self._cipher.encrypt(
            bearer, owner_id=owner_id, connection_id=connection_id, grant_id=grant_id
        )
        if endpoint is not None:
            self._validate_endpoint(endpoint)
            if expected_revision is None or revision != expected_revision:
                raise ConnectionsError("Connections revision changed")
            catalogue = await self._discover(connection_id, endpoint, bearer)
            await self._store.publish_authorization(
                owner_id=owner_id,
                connection_id=connection_id,
                expected_revision=expected_revision,
                endpoint=endpoint,
                grant_id=grant_id,
                kind="bearer",
                key_id=key_id,
                envelope=envelope,
                scopes=(),
                catalogue=catalogue,
                policy=self._policy,
            )
            return await self.read(owner_id=owner_id, auth_mode=auth_mode)
        await self._store.replace_bearer(
            owner_id=owner_id,
            connection_id=connection_id,
            expected_revision=expected_revision if expected_revision is not None else revision,
            grant_id=grant_id,
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
        return await self.read(owner_id=owner_id, auth_mode=auth_mode)

    async def begin_authorization(
        self,
        *,
        owner_id: str,
        auth_mode: str,
        connection_id: str,
        expected_revision: str,
        endpoint: str | None = None,
    ) -> AuthorizationStart:
        self._authorize(owner_id, auth_mode)
        if self._oauth is None or self._policy.oauth_callback_url is None:
            raise ConnectionsError("OAuth callback deployment is not configured", 503)
        if len(self._authorizations) >= 4:
            raise ConnectionsError("Authorization quota exceeded", 429)
        callback_url = self._policy.oauth_callback_url
        self._validate_endpoint(callback_url)
        if (
            urlsplit(callback_url).path != "/web/oauth/connections/mcp/callback"
            or urlsplit(callback_url).query
        ):
            raise ConnectionsError("OAuth callback deployment is invalid", 503)
        revision, items = await self._store.read(owner_id)
        item = next((item for item in items if item.connection_id == connection_id), None)
        if item is None:
            raise ConnectionsError("Connection not found", 404)
        if revision != expected_revision:
            raise ConnectionsError("Connections revision changed")
        if item.envelope and item.grant_id:
            self._cipher.decrypt(
                item.envelope,
                owner_id=owner_id,
                connection_id=connection_id,
                grant_id=item.grant_id,
            )
        candidate = endpoint if endpoint is not None else item.endpoint
        self._validate_endpoint(candidate)
        flow = OAuthFlow(
            uuid.uuid4().hex, owner_id, connection_id, self._worker, candidate, expected_revision
        )
        # Fail missing keys before any external registration or flow mutation.
        self._cipher.encrypt(
            SecretStr("{}"), owner_id=owner_id, connection_id=connection_id, grant_id=flow.flow_id
        )
        await self._store.create_oauth_flow(
            flow=flow, lifetime=self._policy.oauth_timeout, lease=10
        )
        ready: asyncio.Future[AuthorizationStart] = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(self._authorize_flow(flow, callback_url, ready))
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
        auth_mode: str,
        state: str,
        code: str | None = None,
        issuer: str | None = None,
        error: str | None = None,
    ) -> None:
        self._authorize(owner_id, auth_mode)
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

    async def _authorize_flow(
        self, flow: OAuthFlow, callback_url: str, ready: asyncio.Future[AuthorizationStart]
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
                )
                catalogue = self._validate_catalogue(flow.connection_id, result.tools)
                key_id, envelope = self._cipher.encrypt(
                    result.credentials,
                    owner_id=flow.owner_id,
                    connection_id=flow.connection_id,
                    grant_id=flow.flow_id,
                )
                await self._store.publish_authorization(
                    owner_id=flow.owner_id,
                    connection_id=flow.connection_id,
                    expected_revision=flow.expected_revision,
                    endpoint=flow.endpoint,
                    grant_id=flow.flow_id,
                    kind="oauth",
                    key_id=key_id,
                    envelope=envelope,
                    scopes=result.scopes,
                    catalogue=catalogue,
                    policy=self._policy,
                    flow=flow,
                )
        except Exception, asyncio.CancelledError:
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

    async def _discover(
        self, identity: str, endpoint: str, bearer: SecretStr | None = None
    ) -> tuple[CatalogueTool, ...]:
        self._validate_endpoint(endpoint)
        async with asyncio.timeout(self._policy.discovery_timeout), self._discovery_slots:
            raw = await self._mcp.discover(endpoint=endpoint, bearer=bearer, policy=self._policy)
        return self._validate_catalogue(identity, raw)

    def _validate_catalogue(
        self, identity: str, raw: list[dict[str, Any]]
    ) -> tuple[CatalogueTool, ...]:
        if len(raw) > self._policy.max_tools:
            raise ConnectionsError("catalogue")
        tools: list[CatalogueTool] = []
        names: set[str] = set()
        for item in raw:
            name = item.get("name")
            description = item.get("description", "")
            schema = item.get("input_schema")
            if (
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name)
                or name in names
                or not isinstance(description, str)
                or len(description.encode()) > self._policy.max_description_bytes
                or not isinstance(schema, dict)
                or schema.get("type") != "object"
                or len(json.dumps(schema).encode()) > self._policy.max_schema_bytes
            ):
                raise ConnectionsError("catalogue")
            Draft202012Validator.check_schema(schema)
            names.add(name)
            local = "mcp_" + identity + "_" + hashlib.sha256(name.encode()).hexdigest()[:24]
            if any(tool.local_name == local for tool in tools):
                raise ConnectionsError("catalogue")
            tools.append(CatalogueTool(name, local, description, schema))
        if len(json.dumps(raw).encode()) > self._policy.max_catalogue_bytes:
            raise ConnectionsError("catalogue")
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
    ) -> tuple[SecretStr, StoredGrant | None]:
        try:
            return access_bearer(secret, authentication=authentication), None
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
                    return access_bearer(current, authentication="oauth"), grant
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
                return access_bearer(updated, authentication="oauth"), grant
            finally:
                await self._store.release_grant_refresh(claim=lease)

    async def _refresh(self, claim: RefreshClaim) -> None:
        catalogue = None
        error = None
        try:
            item = claim.connection
            bearer = None
            if item.grant_id:
                if not item.envelope:
                    raise ConnectionsError("Credential deployment cannot read grant", 503)
                bearer = self._cipher.decrypt(
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

                bearer, _ = await self._access_credentials(
                    owner_id=item.owner_id,
                    connection_id=item.connection_id,
                    grant_id=item.grant_id,
                    endpoint=item.endpoint,
                    authentication=item.authentication,
                    secret=bearer,
                    on_grant=on_grant,
                )
            catalogue = await self._discover(item.connection_id, item.endpoint, bearer)
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
        )

    async def maintain(self) -> dict[str, int]:
        """One bounded writer maintenance pass; no remote I/O or key-source reads."""
        rotated = 0
        if self._cipher.active_key_id is not None:
            grants = await self._store.rotation_candidates(
                active_key_id=self._cipher.active_key_id, limit=100
            )
            for grant in grants:
                secret = self._cipher.decrypt(
                    grant.envelope,
                    owner_id=grant.owner_id,
                    connection_id=grant.connection_id,
                    grant_id=grant.grant_id,
                )
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
            try:
                claim = await self._store.claim(
                    worker_id=self._worker, lease_seconds=2 * self._policy.discovery_timeout + 5
                )
                if claim is not None:
                    await self._refresh(claim)
                    continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Connection refresh store unavailable; retrying")
            await self._store.wait_refresh(min(1.0, self._policy.refresh_seconds))

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


def _call_failure(tool: CatalogueTool, *, unknown: bool) -> ToolResult:
    outcome = "Outcome may be unknown" if unknown else "No call was sent"
    return ToolResult.text(
        f"Connection tool {tool.local_name} unavailable. {outcome}. Do not retry automatically; identify the requested part not completed in the final Answer.",
        is_error=True,
    )


def _catalogue_tool(
    tool: CatalogueTool,
    *,
    execute: Callable[[BaseModel, ToolRuntime], Awaitable[ToolResult]] | None = None,
) -> AgentTool:
    schema = deepcopy(tool.input_schema)

    class CatalogueArguments(RootModel[dict[str, Any]]):
        @model_validator(mode="after")
        def validate_catalogue(self):
            if not Draft202012Validator(schema).is_valid(self.root):
                raise ValueError("Arguments do not match the accepted tool schema")
            return self

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return deepcopy(schema)

    used: set[tuple[str, str]] = set()

    async def call(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        if execute is None:
            raise RuntimeError("Acceptance definitions cannot execute")
        key = (runtime.execution_scope, runtime.intent_id.value)
        if key in used:
            return _call_failure(tool, unknown=True)
        used.add(key)
        return await execute(raw, runtime)

    return AgentTool(
        name=tool.local_name,
        description=tool.description,
        input_model=CatalogueArguments,
        execute=call,
        replay_policy="never",
    )
