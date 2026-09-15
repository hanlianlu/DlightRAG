# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner-scoped immutable catalogues and short, fenced publication transactions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from dlightrag.adapters.postgres.core._migrations import (
    ForeignKeyRequirement,
    Migration,
    TableRequirement,
    apply_migrations,
    verify_migrations,
)
from dlightrag.adapters.postgres.core._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.application.connections.models import (
    CatalogueTool,
    ConnectionCommand,
    ConnectionsError,
    DispatchCredentials,
    GrantRefreshClaim,
    OAuthFlow,
    PinnedToolFact,
    RefreshClaim,
    StoredConnection,
    StoredGrant,
)
from dlightrag.application.connections.policy import ConnectionPolicy
from dlightrag.engine.agent.session.effects import canonical_json, schema_digest
from dlightrag.engine.agent.session.operation import ToolEffectPending, decode_operation_state
from dlightrag.engine.agent.tools import ToolRuntime
from dlightrag.engine.answer.execution.connection_binding import (
    ResearchToolClaim,
    RunConnectionBinding,
    StaleConnectionBindingError,
    decode_connection_bindings,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE dlightrag_connection_heads (
 owner_id TEXT NOT NULL, connection_id TEXT NOT NULL, revision BIGINT NOT NULL DEFAULT 1,
 label TEXT NOT NULL, enabled BOOLEAN NOT NULL DEFAULT FALSE,
 activation_epoch BIGINT NOT NULL DEFAULT 1, head_generation BIGINT NOT NULL DEFAULT 0,
 consent_version INTEGER, tombstoned_at TIMESTAMPTZ,
 refresh_due_at TIMESTAMPTZ NOT NULL DEFAULT now(), refresh_owner TEXT,
 refresh_epoch BIGINT NOT NULL DEFAULT 0, refresh_expires_at TIMESTAMPTZ,
 observed_status TEXT NOT NULL DEFAULT 'disabled', last_attempt_at TIMESTAMPTZ,
 refresh_failures INTEGER NOT NULL DEFAULT 0,
 last_error_kind TEXT, PRIMARY KEY(owner_id, connection_id)
);
CREATE TABLE dlightrag_connection_grants (
 owner_id TEXT NOT NULL, connection_id TEXT NOT NULL, grant_id TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('bearer','oauth')), audience_digest TEXT NOT NULL,
 consented_scopes JSONB NOT NULL DEFAULT '[]', status TEXT NOT NULL DEFAULT 'active',
 secret_version BIGINT NOT NULL DEFAULT 1, encrypted_envelope TEXT, key_id TEXT NOT NULL,
 refresh_owner TEXT, refresh_epoch BIGINT NOT NULL DEFAULT 0,
 refresh_expires_at TIMESTAMPTZ, updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 PRIMARY KEY(owner_id, grant_id), UNIQUE(owner_id, connection_id, grant_id),
 FOREIGN KEY(owner_id, connection_id) REFERENCES dlightrag_connection_heads(owner_id, connection_id)
);
CREATE TABLE dlightrag_connection_generations (
 owner_id TEXT NOT NULL, connection_id TEXT NOT NULL, generation BIGINT NOT NULL,
 endpoint_json JSONB NOT NULL, endpoint_digest TEXT NOT NULL, grant_id TEXT,
 catalogue_json JSONB, catalogue_digest TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 PRIMARY KEY(owner_id, connection_id, generation),
 FOREIGN KEY(owner_id, connection_id) REFERENCES dlightrag_connection_heads(owner_id, connection_id),
 FOREIGN KEY(owner_id, connection_id, grant_id) REFERENCES dlightrag_connection_grants(owner_id, connection_id, grant_id)
);
ALTER TABLE dlightrag_connection_heads ADD CONSTRAINT connection_head_generation_fk
 FOREIGN KEY(owner_id, connection_id, head_generation)
 REFERENCES dlightrag_connection_generations(owner_id, connection_id, generation)
 DEFERRABLE INITIALLY DEFERRED;
CREATE INDEX connection_refresh_due ON dlightrag_connection_heads(refresh_due_at)
 WHERE tombstoned_at IS NULL;
"""
_PIN_SCHEMA = """
CREATE TABLE dlightrag_answer_connection_pins (
 owner_id TEXT NOT NULL, run_id UUID NOT NULL, connection_id TEXT NOT NULL,
 generation BIGINT NOT NULL, activation_epoch BIGINT NOT NULL, catalogue_digest TEXT NOT NULL,
 PRIMARY KEY(owner_id,run_id,connection_id),
 FOREIGN KEY(owner_id,run_id) REFERENCES dlightrag_runs(owner_id,run_id) ON DELETE CASCADE,
 FOREIGN KEY(owner_id,connection_id,generation)
 REFERENCES dlightrag_connection_generations(owner_id,connection_id,generation)
);
CREATE INDEX connection_generation_pins ON dlightrag_answer_connection_pins(owner_id,connection_id,generation);
"""
_OAUTH_SCHEMA = """
CREATE TABLE dlightrag_connection_oauth_flows (
 flow_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, connection_id TEXT NOT NULL,
 flow_owner TEXT NOT NULL, flow_lease_expires_at TIMESTAMPTZ NOT NULL,
 endpoint TEXT NOT NULL, expected_revision TEXT NOT NULL,
 state_hash TEXT UNIQUE, encrypted_result TEXT, encrypted_credentials TEXT,
 expires_at TIMESTAMPTZ NOT NULL, deposited_at TIMESTAMPTZ, consumed_at TIMESTAMPTZ,
 finished_at TIMESTAMPTZ, succeeded BOOLEAN NOT NULL DEFAULT FALSE,
 FOREIGN KEY(owner_id,connection_id) REFERENCES dlightrag_connection_heads(owner_id,connection_id)
);
CREATE INDEX connection_oauth_expiry ON dlightrag_connection_oauth_flows(expires_at);
"""
_MIGRATIONS = (
    Migration("personal_connections", "Owner Connections and immutable catalogues", (_SCHEMA,)),
    Migration("answer_connection_pins", "Atomic owner Run generation pins", (_PIN_SCHEMA,)),
    Migration("connection_oauth_inbox", "Live initiator encrypted OAuth inbox", (_OAUTH_SCHEMA,)),
)
_TABLES = (
    TableRequirement(
        name="dlightrag_connection_oauth_flows",
        columns=(
            "flow_id",
            "owner_id",
            "connection_id",
            "flow_owner",
            "endpoint",
            "expected_revision",
            "flow_lease_expires_at",
            "state_hash",
            "encrypted_result",
            "encrypted_credentials",
            "expires_at",
            "deposited_at",
            "consumed_at",
            "finished_at",
            "succeeded",
        ),
        primary_key=("flow_id",),
        unique=(("state_hash",),),
        indexes=("connection_oauth_expiry",),
        foreign_keys=(
            ForeignKeyRequirement(("owner_id", "connection_id"), "dlightrag_connection_heads"),
        ),
    ),
    TableRequirement(
        name="dlightrag_answer_connection_pins",
        columns=(
            "owner_id",
            "run_id",
            "connection_id",
            "generation",
            "activation_epoch",
            "catalogue_digest",
        ),
        primary_key=("owner_id", "run_id", "connection_id"),
        indexes=("connection_generation_pins",),
        foreign_keys=(
            ForeignKeyRequirement(("owner_id", "run_id"), "dlightrag_runs"),
            ForeignKeyRequirement(
                ("owner_id", "connection_id", "generation"), "dlightrag_connection_generations"
            ),
        ),
    ),
    TableRequirement(
        name="dlightrag_connection_heads",
        columns=(
            "owner_id",
            "connection_id",
            "revision",
            "label",
            "enabled",
            "activation_epoch",
            "head_generation",
            "consent_version",
            "tombstoned_at",
            "refresh_due_at",
            "refresh_owner",
            "refresh_epoch",
            "refresh_expires_at",
            "observed_status",
            "last_attempt_at",
            "last_error_kind",
            "refresh_failures",
        ),
        primary_key=("owner_id", "connection_id"),
        indexes=("connection_refresh_due",),
        foreign_keys=(
            ForeignKeyRequirement(
                ("owner_id", "connection_id", "head_generation"), "dlightrag_connection_generations"
            ),
        ),
    ),
    TableRequirement(
        name="dlightrag_connection_generations",
        columns=(
            "owner_id",
            "connection_id",
            "generation",
            "endpoint_json",
            "endpoint_digest",
            "grant_id",
            "catalogue_json",
            "catalogue_digest",
            "created_at",
        ),
        primary_key=("owner_id", "connection_id", "generation"),
        foreign_keys=(
            ForeignKeyRequirement(("owner_id", "connection_id"), "dlightrag_connection_heads"),
            ForeignKeyRequirement(
                ("owner_id", "connection_id", "grant_id"), "dlightrag_connection_grants"
            ),
        ),
    ),
    TableRequirement(
        name="dlightrag_connection_grants",
        columns=(
            "owner_id",
            "connection_id",
            "grant_id",
            "kind",
            "audience_digest",
            "consented_scopes",
            "status",
            "secret_version",
            "encrypted_envelope",
            "key_id",
            "refresh_owner",
            "refresh_epoch",
            "refresh_expires_at",
            "updated_at",
        ),
        primary_key=("owner_id", "grant_id"),
        unique=(("owner_id", "connection_id", "grant_id"),),
        foreign_keys=(
            ForeignKeyRequirement(("owner_id", "connection_id"), "dlightrag_connection_heads"),
        ),
    ),
)
_SELECT = """SELECT h.*, g.endpoint_json, g.catalogue_json, g.created_at,
 g.grant_id, r.kind, r.secret_version, r.encrypted_envelope, r.refresh_epoch AS grant_refresh_epoch,
 (SELECT CASE WHEN f.succeeded THEN 'succeeded'
     WHEN f.finished_at IS NOT NULL OR f.expires_at<=clock_timestamp() OR f.flow_lease_expires_at<=clock_timestamp() THEN 'failed'
     ELSE 'pending' END FROM dlightrag_connection_oauth_flows f
     WHERE f.owner_id=h.owner_id AND f.connection_id=h.connection_id ORDER BY f.expires_at DESC LIMIT 1) AS authorization_status
 FROM dlightrag_connection_heads h
 JOIN dlightrag_connection_generations g ON
 (g.owner_id,g.connection_id,g.generation)=(h.owner_id,h.connection_id,h.head_generation)
 LEFT JOIN dlightrag_connection_grants r ON
 (r.owner_id,r.connection_id,r.grant_id)=(g.owner_id,g.connection_id,g.grant_id)
 """


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _stored(row: Any) -> StoredConnection:
    catalogue = _json(row["catalogue_json"])
    return StoredConnection(
        owner_id=row["owner_id"],
        connection_id=row["connection_id"],
        revision=row["revision"],
        label=row["label"],
        endpoint=_json(row["endpoint_json"])["url"],
        enabled=row["enabled"],
        activation_epoch=row["activation_epoch"],
        generation=row["head_generation"],
        grant_id=row["grant_id"],
        status=row["observed_status"],
        catalogue=tuple(CatalogueTool(**item) for item in catalogue or []),
        catalogue_created_at=row["created_at"].isoformat() if catalogue is not None else None,
        last_error_kind=row["last_error_kind"],
        authorization_status=row["authorization_status"],
        authentication=row["kind"] or "none",
        secret_version=row["secret_version"] or 0,
        grant_refresh_epoch=row["grant_refresh_epoch"] or 0,
        envelope=row["encrypted_envelope"],
    )


async def _revision(conn: Any, owner: str) -> str:
    rows = await conn.fetch(
        "SELECT connection_id, revision FROM dlightrag_connection_heads WHERE owner_id=$1 ORDER BY connection_id",
        owner,
    )
    return (
        hashlib.sha256(json.dumps([tuple(row) for row in rows]).encode()).hexdigest()
        if rows
        else "0"
    )


async def _owner_lock(conn: Any, owner: str) -> None:
    # Serializes owner quotas and revision CAS across otherwise independent heads.
    await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1, 71421))", owner)


class PGConnectionsStore(PostgresOperationRunner):
    def __init__(self, *, pool: ConnectionPool | None = None) -> None:
        super().__init__(pool=pool)
        self._oauth_wakes: dict[str, asyncio.Event] = {}
        self._wake = asyncio.Event()
        self._dispatch_wake = asyncio.Event()
        self._listener: asyncio.Task[None] | None = None

    async def start_notifications(self) -> None:
        if self._listener is None:
            self._listener = asyncio.create_task(self._listen_forever())

    async def _listen_forever(self) -> None:
        def changed(*_args: Any) -> None:
            self._wake.set()
            self._dispatch_wake.set()

        def oauth_changed(_conn: Any, _pid: int, _channel: str, worker_id: str) -> None:
            wake = self._oauth_wakes.get(worker_id)
            if wake is not None:
                wake.set()

        async def listen(conn: Any) -> None:
            await conn.add_listener("dlightrag_connection_oauth", oauth_changed)
            await conn.add_listener("dlightrag_connections_changed", changed)
            self._wake.set()  # Reconnect/startup scan recovers missed notifications.
            try:
                while not conn.is_closed():
                    await asyncio.sleep(1)
            finally:
                if not conn.is_closed():
                    await conn.remove_listener("dlightrag_connections_changed", changed)
                    await conn.remove_listener("dlightrag_connection_oauth", oauth_changed)

        while True:
            try:
                await self._run_once(listen)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Connection notification listener unavailable; retrying")
            await asyncio.sleep(1)

    async def wait_refresh(self, timeout: float) -> None:
        try:
            await asyncio.wait_for(self._wake.wait(), timeout)
        except TimeoutError:
            pass
        self._wake.clear()

    async def stop_notifications(self) -> None:
        task, self._listener = self._listener, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def initialize(self, *, validate_only: bool = False) -> None:
        async def operation(conn: Any) -> None:
            if validate_only:
                await verify_migrations(
                    conn,
                    scope="connections",
                    migrations=_MIGRATIONS,
                    tables=_TABLES,
                    schema_error=ConnectionsError,
                )
            else:
                await apply_migrations(
                    conn, scope="connections", migrations=_MIGRATIONS, schema_error=ConnectionsError
                )

        await self._run(operation)

    async def read(self, owner_id: str) -> tuple[str, tuple[StoredConnection, ...]]:
        async def operation(conn: Any) -> tuple[str, tuple[StoredConnection, ...]]:
            async with conn.transaction(isolation="repeatable_read", readonly=True):
                revision = await _revision(conn, owner_id)
                rows = await conn.fetch(
                    _SELECT
                    + " WHERE h.owner_id=$1 AND h.tombstoned_at IS NULL ORDER BY h.connection_id",
                    owner_id,
                )
                return revision, tuple(_stored(row) for row in rows)

        return await self._run(operation)

    async def research_catalogues(
        self, owner_id: str
    ) -> tuple[tuple[RunConnectionBinding, tuple[CatalogueTool, ...]], ...]:
        async def operation(
            conn: Any,
        ) -> tuple[tuple[RunConnectionBinding, tuple[CatalogueTool, ...]], ...]:
            rows = await conn.fetch(
                """SELECT h.connection_id,h.head_generation,h.activation_epoch,
                g.catalogue_digest,g.catalogue_json FROM dlightrag_connection_heads h
                JOIN dlightrag_connection_generations g ON
                (g.owner_id,g.connection_id,g.generation)=(h.owner_id,h.connection_id,h.head_generation)
                LEFT JOIN dlightrag_connection_grants r ON (r.owner_id,r.grant_id)=(g.owner_id,g.grant_id)
                WHERE h.owner_id=$1 AND h.enabled AND h.consent_version=1
                AND h.tombstoned_at IS NULL AND g.catalogue_json IS NOT NULL
                AND (g.grant_id IS NULL OR r.status='active') ORDER BY h.connection_id""",
                owner_id,
            )
            return tuple(
                (
                    RunConnectionBinding(
                        owner_id,
                        row["connection_id"],
                        row["head_generation"],
                        row["activation_epoch"],
                        row["catalogue_digest"],
                    ),
                    tuple(CatalogueTool(**tool) for tool in _json(row["catalogue_json"])),
                )
                for row in rows
            )

        return await self._run(operation)

    async def pinned_catalogues(
        self, *, owner_id: str, run_id: str, bindings: tuple[RunConnectionBinding, ...]
    ) -> tuple[CatalogueTool, ...]:
        if any(binding.owner_id != owner_id for binding in bindings):
            raise ValueError("Run Connection owner mismatch")

        async def operation(conn: Any) -> tuple[CatalogueTool, ...]:
            rows = await conn.fetch(
                """SELECT p.*,g.catalogue_json,g.catalogue_digest AS stored_digest
                FROM dlightrag_answer_connection_pins p JOIN dlightrag_connection_generations g
                ON (g.owner_id,g.connection_id,g.generation)=(p.owner_id,p.connection_id,p.generation)
                WHERE p.owner_id=$1 AND p.run_id=$2 ORDER BY p.connection_id""",
                owner_id,
                uuid.UUID(run_id),
            )
            stored = tuple(
                RunConnectionBinding(
                    owner_id,
                    row["connection_id"],
                    row["generation"],
                    row["activation_epoch"],
                    row["catalogue_digest"],
                )
                for row in rows
            )
            if stored != tuple(sorted(bindings, key=lambda b: b.connection_id)) or any(
                row["stored_digest"] != row["catalogue_digest"] or row["catalogue_json"] is None
                for row in rows
            ):
                raise ValueError("Run Connection pins unavailable or mismatched")
            return tuple(
                CatalogueTool(**tool) for row in rows for tool in _json(row["catalogue_json"])
            )

        return await self._run(operation)

    async def pinned_tool_facts(self, *, owner_id: str, run_id: str) -> tuple[PinnedToolFact, ...]:
        """Read the display facts of this Run's pinned tools, tolerating a lost head.

        The pin still names the generation whose definitions the Run accepted; a
        disabled, revoked, or deleted Connection keeps its head row (tombstoned), so
        the label survives, while a pin whose generation was never retained simply
        contributes nothing.
        """

        async def operation(conn: Any) -> tuple[PinnedToolFact, ...]:
            rows = await conn.fetch(
                """SELECT h.label,g.catalogue_json
                FROM dlightrag_answer_connection_pins p
                JOIN dlightrag_connection_generations g ON
                (g.owner_id,g.connection_id,g.generation)=(p.owner_id,p.connection_id,p.generation)
                LEFT JOIN dlightrag_connection_heads h ON
                (h.owner_id,h.connection_id)=(p.owner_id,p.connection_id)
                WHERE p.owner_id=$1 AND p.run_id=$2 ORDER BY p.connection_id""",
                owner_id,
                uuid.UUID(run_id),
            )
            return tuple(
                PinnedToolFact(
                    local_name=str(tool.get("local_name") or ""),
                    connection_label=str(row["label"] or ""),
                    remote_name=str(tool.get("remote_name") or ""),
                )
                for row in rows
                for tool in _json(row["catalogue_json"]) or ()
                if tool.get("local_name")
            )

        return await self._run(operation)

    async def dispatch_gate(
        self,
        *,
        claim: ResearchToolClaim,
        bindings: tuple[RunConnectionBinding, ...],
        runtime: ToolRuntime,
        tool: CatalogueTool,
        arguments: dict[str, Any],
    ) -> DispatchCredentials:
        async def operation(conn: Any) -> DispatchCredentials:
            async with conn.transaction():
                # Locate via the accepted pin, not a caller-supplied connection id.
                rows = await conn.fetch(
                    """SELECT p.*,g.catalogue_json FROM dlightrag_answer_connection_pins p
                    JOIN dlightrag_connection_generations g USING(owner_id,connection_id,generation)
                    WHERE p.owner_id=$1 AND p.run_id=$2 ORDER BY connection_id""",
                    claim.owner_id,
                    uuid.UUID(claim.run_id),
                )
                selected = None
                for row in rows:
                    binding = RunConnectionBinding(
                        claim.owner_id,
                        row["connection_id"],
                        row["generation"],
                        row["activation_epoch"],
                        row["catalogue_digest"],
                    )
                    if binding in bindings and any(
                        CatalogueTool(**value) == tool
                        for value in _json(row["catalogue_json"]) or []
                    ):
                        selected = binding
                        break
                if selected is None or runtime.tool_name != tool.local_name:
                    raise ConnectionsError("Connection dispatch denied")
                b = selected
                head = await conn.fetchrow(
                    """SELECT * FROM dlightrag_connection_heads
                    WHERE owner_id=$1 AND connection_id=$2 FOR UPDATE""",
                    claim.owner_id,
                    b.connection_id,
                )
                generation = await conn.fetchrow(
                    """SELECT * FROM dlightrag_connection_generations
                    WHERE owner_id=$1 AND connection_id=$2 AND generation=$3 FOR UPDATE""",
                    claim.owner_id,
                    b.connection_id,
                    b.generation,
                )
                if (
                    head is None
                    or not head["enabled"]
                    or head["tombstoned_at"] is not None
                    or head["consent_version"] != 1
                    or head["activation_epoch"] != b.activation_epoch
                    or generation is None
                    or generation["catalogue_digest"] != b.catalogue_digest
                ):
                    raise ConnectionsError("Connection dispatch denied")
                grant = None
                if generation["grant_id"] is not None:
                    grant = await conn.fetchrow(
                        """SELECT * FROM dlightrag_connection_grants
                        WHERE owner_id=$1 AND connection_id=$2 AND grant_id=$3 FOR UPDATE""",
                        claim.owner_id,
                        b.connection_id,
                        generation["grant_id"],
                    )
                    if (
                        grant is None
                        or grant["status"] != "active"
                        or grant["kind"] not in {"bearer", "oauth"}
                        or grant["audience_digest"] != generation["endpoint_digest"]
                        or grant["secret_version"] < 1
                        or grant["encrypted_envelope"] is None
                    ):
                        raise ConnectionsError("Connection grant unavailable")
                endpoint = _json(generation["endpoint_json"])["url"]
                if hashlib.sha256(endpoint.encode()).hexdigest() != generation["endpoint_digest"]:
                    raise ValueError("Connection endpoint digest mismatch")
                if not await self._live_execution(conn, claim=claim, runtime=runtime, lock=True):
                    raise ConnectionsError("Connection execution fence denied")
                # Parent lock serializes settlement and child cancellation. Check
                # existing runtime authority; never create a second effect ledger.
                states = await conn.fetch(
                    """SELECT payload_json FROM dlightrag_agent_session_registers
                    WHERE owner_id=$1 AND session_id=$2 AND register_kind='operation_state'""",
                    claim.owner_id,
                    uuid.UUID(runtime.execution_scope),
                )
                pending = []
                for row in states:
                    state = decode_operation_state(_json(row["payload_json"]))
                    if isinstance(state, ToolEffectPending):
                        item = state.batch.items[state.source_index]
                        if item.intent_id == runtime.intent_id:
                            pending.append(item)
                if len(pending) != 1:
                    raise ConnectionsError("Connection effect is not pending")
                item = pending[0]
                args = await conn.fetchval(
                    """SELECT payload_json FROM dlightrag_agent_session_registers
                    WHERE owner_id=$1 AND session_id=$2 AND register_kind='tool_arguments' AND register_key=$3""",
                    claim.owner_id,
                    uuid.UUID(runtime.execution_scope),
                    runtime.intent_id.value,
                )
                if (
                    item.tool_name != tool.local_name
                    or item.call_id != runtime.call_id
                    or item.replay_policy != "never"
                    or item.contract_version != 2
                    or item.input_schema_digest != schema_digest(tool.input_schema)
                    or item.effective_input_digest
                    != hashlib.sha256(canonical_json(arguments).encode()).hexdigest()
                    or args is None
                    or _json(args).get("arguments") != arguments
                ):
                    raise ConnectionsError("Connection pending effect mismatch")
                if not await self._live_execution(conn, claim=claim, runtime=runtime, lock=False):
                    raise ConnectionsError("Connection execution lease expired")
                return DispatchCredentials(
                    b.connection_id,
                    b.generation,
                    b.activation_epoch,
                    endpoint,
                    generation["grant_id"],
                    grant["secret_version"] if grant else 0,
                    grant["encrypted_envelope"] if grant else None,
                    grant["kind"] if grant else "none",
                )

        # A lost COMMIT acknowledgement must not silently rerun an effect gate.
        return await self._run_once(operation)

    @staticmethod
    async def _live_execution(
        conn: Any, *, claim: ResearchToolClaim, runtime: ToolRuntime, lock: bool
    ) -> bool:
        run = await conn.fetchrow(
            """SELECT prepared_input_json FROM dlightrag_runs
            WHERE owner_id=$1 AND run_id=$2 AND lease_owner=$3 AND fencing_epoch=$4
            AND status='running' AND lease_expires_at>clock_timestamp() AND cancel_requested_at IS NULL"""  # noqa: S608 - only a constant lock clause is appended
            + (" FOR UPDATE" if lock else ""),
            claim.owner_id,
            uuid.UUID(claim.run_id),
            claim.worker_id,
            claim.fencing_epoch,
        )
        if run is None or runtime.fencing_epoch < 1:
            return False
        parent_scope = _json(run["prepared_input_json"]).get("agent_session_id")
        if runtime.execution_scope == parent_scope:
            if runtime.fencing_epoch != claim.fencing_epoch:
                return False
        else:
            child = await conn.fetchval(
                """SELECT 1 FROM dlightrag_answer_child_sessions
                WHERE owner_id=$1 AND run_id=$2 AND child_session_id=$3 AND lease_owner=$4
                AND fencing_epoch=$5 AND status='running' AND lease_expires_at>clock_timestamp()
                AND cancel_requested_at IS NULL"""  # noqa: S608 - only a constant lock clause is appended
                + (" FOR UPDATE" if lock else ""),
                claim.owner_id,
                uuid.UUID(claim.run_id),
                uuid.UUID(runtime.execution_scope),
                claim.worker_id,
                runtime.fencing_epoch,
            )
            if child is None:
                return False
        return bool(
            await conn.fetchval(
                """SELECT 1 FROM dlightrag_agent_sessions
            WHERE owner_id=$1 AND session_id=$2 AND lease_run_id=$3 AND fencing_epoch=$4""",
                claim.owner_id,
                uuid.UUID(runtime.execution_scope),
                uuid.UUID(claim.run_id),
                runtime.fencing_epoch,
            )
        )

    async def dispatch_alive(
        self, *, claim: ResearchToolClaim, runtime: ToolRuntime, dispatch: DispatchCredentials
    ) -> bool:
        # Gate-first effects retain authorization across routine secret changes.
        # The gate/preflight and write CAS still fence credential versions.
        async def operation(conn: Any) -> bool:
            active = await conn.fetchval(
                """SELECT 1 FROM dlightrag_connection_heads h
                LEFT JOIN dlightrag_connection_grants g ON g.owner_id=h.owner_id
                    AND g.connection_id=h.connection_id AND g.grant_id=$4
                WHERE h.owner_id=$1 AND h.connection_id=$2 AND h.enabled AND h.activation_epoch=$3
                AND h.tombstoned_at IS NULL AND h.consent_version=1
                AND ($4::text IS NULL OR g.status='active')""",
                claim.owner_id,
                dispatch.connection_id,
                dispatch.activation_epoch,
                dispatch.grant_id,
            )
            return bool(active) and await self._live_execution(
                conn, claim=claim, runtime=runtime, lock=False
            )

        return await self._run_once(operation)

    async def observe_call(
        self, *, owner_id: str, dispatch: DispatchCredentials, error: str | None
    ) -> None:
        if error not in {None, "authentication", "transport"}:
            raise ValueError("Invalid Connection observation")

        async def operation(conn: Any) -> None:
            await conn.execute(
                """UPDATE dlightrag_connection_heads SET
                observed_status=$5,last_error_kind=$6,last_attempt_at=now()
                WHERE owner_id=$1 AND connection_id=$2 AND head_generation=$3 AND activation_epoch=$4
                AND enabled AND tombstoned_at IS NULL
                AND ($7::text IS NULL OR EXISTS(SELECT 1 FROM dlightrag_connection_grants r
                    WHERE r.owner_id=$1 AND r.grant_id=$7 AND r.status='active' AND r.secret_version=$8))""",
                owner_id,
                dispatch.connection_id,
                dispatch.generation,
                dispatch.activation_epoch,
                "ready"
                if error is None
                else "needs-auth"
                if error == "authentication"
                else "degraded",
                error,
                dispatch.grant_id,
                dispatch.secret_version,
            )

        await self._run_once(operation)

    async def wait_dispatch_change(self, timeout: float) -> None:
        try:
            await asyncio.wait_for(self._dispatch_wake.wait(), timeout)
        except TimeoutError:
            pass
        self._dispatch_wake.clear()

    async def change(
        self,
        *,
        owner_id: str,
        expected_revision: str,
        command: ConnectionCommand,
        policy: ConnectionPolicy,
        candidate: tuple[CatalogueTool, ...] | None = None,
    ) -> None:
        async def operation(conn: Any) -> None:
            async with conn.transaction():
                await _owner_lock(conn, owner_id)
                row = None
                if command.kind != "create":
                    row = await conn.fetchrow(
                        _SELECT
                        + " WHERE h.owner_id=$1 AND h.connection_id=$2 AND h.tombstoned_at IS NULL FOR UPDATE OF h",
                        owner_id,
                        command.connection_id,
                    )
                    if row is None:
                        raise ConnectionsError("Connection not found", 404)
                if await _revision(conn, owner_id) != expected_revision:
                    raise ConnectionsError("Connections revision changed")
                if command.kind == "create":
                    count = await conn.fetchval(
                        "SELECT count(*) FROM dlightrag_connection_heads WHERE owner_id=$1 AND tombstoned_at IS NULL",
                        owner_id,
                    )
                    if count >= policy.max_connections:
                        raise ConnectionsError("Connection quota exceeded")
                    identity = uuid.uuid4().hex
                    await conn.execute(
                        "INSERT INTO dlightrag_connection_heads(owner_id,connection_id,label) VALUES($1,$2,$3)",
                        owner_id,
                        identity,
                        command.label,
                    )
                    await self._generation(
                        conn, owner_id, identity, 0, command.endpoint or "", None, None
                    )
                else:
                    if row is None:
                        raise ConnectionsError("Connection not found", 404)
                    identity = row["connection_id"]
                    if command.kind == "enable":
                        if command.consent_version != 1:
                            raise ConnectionsError("Whole-Connection consent is required")
                        if row["catalogue_json"] is None or row["observed_status"] == "revoked":
                            raise ConnectionsError("Probe a complete catalogue before enabling")
                        await self._check_tool_quota(
                            conn, owner_id, identity, len(_json(row["catalogue_json"])), policy
                        )
                        await conn.execute(
                            "UPDATE dlightrag_connection_heads SET enabled=TRUE,consent_version=1,refresh_due_at=now() WHERE owner_id=$1 AND connection_id=$2",
                            owner_id,
                            identity,
                        )
                    elif command.kind in {"disable", "delete", "revoke"}:
                        await conn.execute(
                            "UPDATE dlightrag_connection_heads SET enabled=FALSE,activation_epoch=activation_epoch+1, tombstoned_at=CASE WHEN $3='delete' THEN now() ELSE tombstoned_at END, observed_status=CASE WHEN $3='revoke' THEN 'revoked' ELSE observed_status END WHERE owner_id=$1 AND connection_id=$2",
                            owner_id,
                            identity,
                            command.kind,
                        )
                        if command.kind in {"delete", "revoke"}:
                            await conn.execute(
                                "UPDATE dlightrag_connection_grants SET status='retired',encrypted_envelope=NULL,secret_version=secret_version+1,refresh_epoch=refresh_epoch+1 WHERE owner_id=$1 AND connection_id=$2",
                                owner_id,
                                identity,
                            )
                    elif command.kind == "edit":
                        if (
                            command.endpoint
                            and command.endpoint != _json(row["endpoint_json"])["url"]
                        ):
                            if row["grant_id"]:
                                raise ConnectionsError(
                                    "Endpoint candidate needs a new grant",
                                    kind="requires_reauthorization",
                                )
                            if candidate is None:
                                raise ConnectionsError("Endpoint candidate required")
                            if row["enabled"]:
                                await self._check_tool_quota(
                                    conn, owner_id, identity, len(candidate), policy
                                )
                            generation = row["head_generation"] + 1
                            await self._generation(
                                conn,
                                owner_id,
                                identity,
                                generation,
                                command.endpoint,
                                None,
                                candidate,
                            )
                            await conn.execute(
                                "UPDATE dlightrag_connection_heads SET head_generation=$3,observed_status='ready' WHERE owner_id=$1 AND connection_id=$2",
                                owner_id,
                                identity,
                                generation,
                            )
                        if command.label:
                            await conn.execute(
                                "UPDATE dlightrag_connection_heads SET label=$3 WHERE owner_id=$1 AND connection_id=$2",
                                owner_id,
                                identity,
                                command.label,
                            )
                    await conn.execute(
                        "UPDATE dlightrag_connection_heads SET revision=revision+1,refresh_epoch=refresh_epoch+1,refresh_owner=NULL,refresh_expires_at=NULL,refresh_due_at=now() WHERE owner_id=$1 AND connection_id=$2",
                        owner_id,
                        identity,
                    )
                await conn.execute("SELECT pg_notify('dlightrag_connections_changed',$1)", owner_id)

        await self._run_once(operation)

    async def replace_bearer(
        self,
        *,
        owner_id: str,
        connection_id: str,
        expected_revision: str,
        grant_id: str,
        key_id: str,
        envelope: str,
    ) -> None:
        async def operation(conn: Any) -> None:
            async with conn.transaction():
                await _owner_lock(conn, owner_id)
                row = await conn.fetchrow(
                    _SELECT
                    + " WHERE h.owner_id=$1 AND h.connection_id=$2 AND h.tombstoned_at IS NULL FOR UPDATE OF h",
                    owner_id,
                    connection_id,
                )
                if row is None:
                    raise ConnectionsError("Connection not found", 404)
                if await _revision(conn, owner_id) != expected_revision:
                    raise ConnectionsError("Connections revision changed")
                endpoint = _json(row["endpoint_json"])["url"]
                await conn.execute(
                    "UPDATE dlightrag_connection_grants SET status='retired',encrypted_envelope=NULL,secret_version=secret_version+1,refresh_epoch=refresh_epoch+1 WHERE owner_id=$1 AND connection_id=$2",
                    owner_id,
                    connection_id,
                )
                await conn.execute(
                    """INSERT INTO dlightrag_connection_grants
                    (owner_id,connection_id,grant_id,kind,audience_digest,encrypted_envelope,key_id)
                    VALUES($1,$2,$3,'bearer',$4,$5,$6)""",
                    owner_id,
                    connection_id,
                    grant_id,
                    hashlib.sha256(endpoint.encode()).hexdigest(),
                    envelope,
                    key_id,
                )
                generation = row["head_generation"] + 1
                await self._generation(
                    conn, owner_id, connection_id, generation, endpoint, grant_id, None
                )
                await conn.execute(
                    """UPDATE dlightrag_connection_heads SET head_generation=$3,
                    enabled=FALSE,activation_epoch=activation_epoch+1,revision=revision+1,
                    refresh_epoch=refresh_epoch+1,refresh_owner=NULL,refresh_expires_at=NULL,
                    refresh_due_at=now(),observed_status='disabled',last_error_kind=NULL
                    WHERE owner_id=$1 AND connection_id=$2""",
                    owner_id,
                    connection_id,
                    generation,
                )
                await conn.execute("SELECT pg_notify('dlightrag_connections_changed',$1)", owner_id)

        await self._run_once(operation)

    async def claim_grant_refresh(
        self,
        *,
        owner_id: str,
        connection_id: str,
        grant_id: str,
        endpoint: str,
        worker_id: str,
        lease_seconds: float,
    ) -> GrantRefreshClaim | None:
        async def operation(conn: Any) -> GrantRefreshClaim | None:
            async with conn.transaction():
                head = await conn.fetchval(
                    "SELECT 1 FROM dlightrag_connection_heads WHERE owner_id=$1 AND connection_id=$2 AND tombstoned_at IS NULL FOR UPDATE",
                    owner_id,
                    connection_id,
                )
                if not head:
                    raise ConnectionsError("Connection grant unavailable", 401)
                row = await conn.fetchrow(
                    "SELECT *,refresh_expires_at>clock_timestamp() AS leased FROM dlightrag_connection_grants WHERE owner_id=$1 AND connection_id=$2 AND grant_id=$3 FOR UPDATE",
                    owner_id,
                    connection_id,
                    grant_id,
                )
                if (
                    row is None
                    or row["status"] != "active"
                    or row["kind"] != "oauth"
                    or row["encrypted_envelope"] is None
                    or row["audience_digest"] != hashlib.sha256(endpoint.encode()).hexdigest()
                ):
                    raise ConnectionsError("Connection grant unavailable", 401)
                if row["leased"]:
                    return None
                epoch = await conn.fetchval(
                    "UPDATE dlightrag_connection_grants SET refresh_owner=$4,refresh_epoch=refresh_epoch+1,refresh_expires_at=clock_timestamp()+$5*interval '1 second' WHERE owner_id=$1 AND connection_id=$2 AND grant_id=$3 RETURNING refresh_epoch",
                    owner_id,
                    connection_id,
                    grant_id,
                    worker_id,
                    lease_seconds,
                )
                return GrantRefreshClaim(
                    StoredGrant(
                        owner_id,
                        connection_id,
                        grant_id,
                        endpoint,
                        row["secret_version"],
                        epoch,
                        tuple(_json(row["consented_scopes"])),
                        row["key_id"],
                        row["encrypted_envelope"],
                    ),
                    worker_id,
                )

        return await self._run_once(operation)

    async def save_grant_refresh(
        self, *, claim: GrantRefreshClaim, key_id: str, envelope: str
    ) -> bool:
        grant = claim.grant

        async def operation(conn: Any) -> bool:
            async with conn.transaction():
                await conn.fetchval(
                    "SELECT 1 FROM dlightrag_connection_heads WHERE owner_id=$1 AND connection_id=$2 FOR UPDATE",
                    grant.owner_id,
                    grant.connection_id,
                )
                changed = await conn.fetchval(
                    """UPDATE dlightrag_connection_grants SET encrypted_envelope=$7,key_id=$8,secret_version=secret_version+1,updated_at=now()
                    WHERE owner_id=$1 AND connection_id=$2 AND grant_id=$3 AND refresh_owner=$4 AND refresh_epoch=$5 AND secret_version=$6
                    AND status='active' AND refresh_expires_at>clock_timestamp() RETURNING 1""",
                    grant.owner_id,
                    grant.connection_id,
                    grant.grant_id,
                    claim.worker_id,
                    grant.refresh_epoch,
                    grant.secret_version,
                    envelope,
                    key_id,
                )
                if changed:
                    await conn.execute(
                        "SELECT pg_notify('dlightrag_connections_changed',$1)", grant.owner_id
                    )
                return bool(changed)

        return await self._run_once(operation)

    async def release_grant_refresh(self, *, claim: GrantRefreshClaim) -> None:
        grant = claim.grant

        async def operation(conn: Any) -> None:
            await conn.execute(
                "UPDATE dlightrag_connection_grants SET refresh_owner=NULL,refresh_expires_at=NULL WHERE owner_id=$1 AND grant_id=$2 AND refresh_owner=$3 AND refresh_epoch=$4",
                grant.owner_id,
                grant.grant_id,
                claim.worker_id,
                grant.refresh_epoch,
            )

        await self._run_once(operation)

    async def rotation_candidates(
        self, *, active_key_id: str, limit: int
    ) -> tuple[StoredGrant, ...]:
        async def operation(conn: Any) -> tuple[StoredGrant, ...]:
            rows = await conn.fetch(
                """SELECT r.*,g.endpoint_json FROM dlightrag_connection_grants r
                JOIN dlightrag_connection_heads h USING(owner_id,connection_id)
                JOIN dlightrag_connection_generations g ON (g.owner_id,g.connection_id,g.generation)=(h.owner_id,h.connection_id,h.head_generation)
                WHERE r.status='active' AND r.encrypted_envelope IS NOT NULL AND r.key_id<>$1
                AND (r.refresh_expires_at IS NULL OR r.refresh_expires_at<=clock_timestamp())
                AND g.grant_id=r.grant_id ORDER BY r.owner_id,r.connection_id,r.grant_id LIMIT $2""",
                active_key_id,
                limit,
            )
            return tuple(
                StoredGrant(
                    row["owner_id"],
                    row["connection_id"],
                    row["grant_id"],
                    _json(row["endpoint_json"])["url"],
                    row["secret_version"],
                    row["refresh_epoch"],
                    tuple(_json(row["consented_scopes"])),
                    row["key_id"],
                    row["encrypted_envelope"],
                )
                for row in rows
            )

        return await self._run_once(operation)

    async def reencrypt_grant(self, *, grant: StoredGrant, key_id: str, envelope: str) -> bool:
        # Cosmetic rotation must not discard an externally rotated refresh token.
        # A lease can start after selection; recheck it in the head-locked CAS.
        async def operation(conn: Any) -> bool:
            async with conn.transaction():
                await conn.fetchval(
                    "SELECT 1 FROM dlightrag_connection_heads WHERE owner_id=$1 AND connection_id=$2 FOR UPDATE",
                    grant.owner_id,
                    grant.connection_id,
                )
                changed = await conn.fetchval(
                    """UPDATE dlightrag_connection_grants SET encrypted_envelope=$5,key_id=$6,
                    secret_version=secret_version+1,refresh_epoch=refresh_epoch+1,
                    refresh_owner=NULL,refresh_expires_at=NULL,updated_at=now()
                    WHERE owner_id=$1 AND connection_id=$2 AND grant_id=$3 AND secret_version=$4
                    AND status='active' AND encrypted_envelope=$7
                    AND (refresh_expires_at IS NULL OR refresh_expires_at<=clock_timestamp()) RETURNING 1""",
                    grant.owner_id,
                    grant.connection_id,
                    grant.grant_id,
                    grant.secret_version,
                    envelope,
                    key_id,
                    grant.envelope,
                )
                if changed:
                    await conn.execute(
                        "SELECT pg_notify('dlightrag_connections_changed',$1)", grant.owner_id
                    )
                return bool(changed)

        return await self._run_once(operation)

    async def collect_garbage(self, *, limit: int) -> int:
        """Bounded head-first collection; normalized FKs arbitrate pin/GC races."""

        async def operation(conn: Any) -> int:
            expired = await conn.fetch(
                """DELETE FROM dlightrag_connection_oauth_flows WHERE flow_id IN (
                SELECT flow_id FROM dlightrag_connection_oauth_flows WHERE expires_at<=clock_timestamp()
                ORDER BY expires_at FOR UPDATE SKIP LOCKED LIMIT $1) RETURNING flow_id""",
                limit,
            )
            async with conn.transaction():
                heads = await conn.fetch(
                    """SELECT h.* FROM dlightrag_connection_heads h
                    WHERE (h.refresh_expires_at IS NULL OR h.refresh_expires_at<=clock_timestamp())
                    AND NOT EXISTS(SELECT 1 FROM dlightrag_connection_oauth_flows f WHERE f.owner_id=h.owner_id AND f.connection_id=h.connection_id)
                    AND NOT EXISTS(SELECT 1 FROM dlightrag_connection_grants r WHERE r.owner_id=h.owner_id AND r.connection_id=h.connection_id AND r.refresh_expires_at>clock_timestamp())
                    AND ((h.tombstoned_at IS NOT NULL AND NOT EXISTS(SELECT 1 FROM dlightrag_answer_connection_pins p WHERE p.owner_id=h.owner_id AND p.connection_id=h.connection_id)) OR EXISTS(
                        SELECT 1 FROM dlightrag_connection_generations g WHERE g.owner_id=h.owner_id AND g.connection_id=h.connection_id
                        AND g.generation<>h.head_generation AND NOT EXISTS(
                            SELECT 1 FROM dlightrag_answer_connection_pins p WHERE (p.owner_id,p.connection_id,p.generation)=(g.owner_id,g.connection_id,g.generation))))
                    ORDER BY h.owner_id,h.connection_id FOR UPDATE OF h SKIP LOCKED LIMIT $1""",
                    limit,
                )
                removed = len(expired)
                budget = limit
                for head in heads:
                    owner, identity = head["owner_id"], head["connection_id"]
                    generations = await conn.fetch(
                        """SELECT g.generation FROM dlightrag_connection_generations g WHERE g.owner_id=$1 AND g.connection_id=$2
                        AND g.generation<>$3 AND NOT EXISTS(SELECT 1 FROM dlightrag_answer_connection_pins p
                        WHERE (p.owner_id,p.connection_id,p.generation)=(g.owner_id,g.connection_id,g.generation))
                        ORDER BY g.generation FOR UPDATE SKIP LOCKED LIMIT $4""",
                        owner,
                        identity,
                        head["head_generation"],
                        budget,
                    )
                    for row in generations:
                        await conn.execute(
                            "DELETE FROM dlightrag_connection_generations WHERE owner_id=$1 AND connection_id=$2 AND generation=$3",
                            owner,
                            identity,
                            row["generation"],
                        )
                    budget -= len(generations)
                    removed += len(generations)
                    # Tombstones have no live authority. Delete their final head
                    # only after ALL retained Run pins and temporary flows end.
                    if head["tombstoned_at"] is not None and not await conn.fetchval(
                        "SELECT 1 FROM dlightrag_answer_connection_pins WHERE owner_id=$1 AND connection_id=$2 LIMIT 1",
                        owner,
                        identity,
                    ):
                        remaining = await conn.fetchval(
                            "SELECT count(*) FROM dlightrag_connection_generations WHERE owner_id=$1 AND connection_id=$2",
                            owner,
                            identity,
                        )
                        if remaining == 1 and budget > 0:
                            await conn.execute(
                                "DELETE FROM dlightrag_connection_generations WHERE owner_id=$1 AND connection_id=$2",
                                owner,
                                identity,
                            )
                            await conn.execute(
                                "DELETE FROM dlightrag_connection_grants WHERE owner_id=$1 AND connection_id=$2",
                                owner,
                                identity,
                            )
                            await conn.execute(
                                "DELETE FROM dlightrag_connection_heads WHERE owner_id=$1 AND connection_id=$2",
                                owner,
                                identity,
                            )
                            budget -= 1
                            removed += 1
                    await conn.execute(
                        """DELETE FROM dlightrag_connection_grants r WHERE r.owner_id=$1 AND r.connection_id=$2 AND r.status='retired'
                        AND NOT EXISTS(SELECT 1 FROM dlightrag_connection_generations g WHERE g.owner_id=r.owner_id AND g.grant_id=r.grant_id)""",
                        owner,
                        identity,
                    )
                    if budget == 0:
                        break
                return removed

        return await self._run_once(operation)

    async def wait_oauth_callback(self, *, worker_id: str, timeout: float) -> None:
        wake = self._oauth_wakes.setdefault(worker_id, asyncio.Event())
        try:
            await asyncio.wait_for(wake.wait(), timeout)
        except TimeoutError:
            pass
        wake.clear()

    async def create_oauth_flow(self, *, flow: OAuthFlow, lifetime: float, lease: float) -> None:
        async def operation(conn: Any) -> None:
            async with conn.transaction():
                await _owner_lock(conn, flow.owner_id)
                found = await conn.fetchval(
                    "SELECT 1 FROM dlightrag_connection_heads WHERE owner_id=$1 AND connection_id=$2 AND tombstoned_at IS NULL FOR UPDATE",
                    flow.owner_id,
                    flow.connection_id,
                )
                if not found:
                    raise ConnectionsError("Connection not found", 404)
                if await _revision(conn, flow.owner_id) != flow.expected_revision:
                    raise ConnectionsError("Connections revision changed")
                recent = await conn.fetchval(
                    "SELECT count(*) FROM dlightrag_connection_oauth_flows WHERE owner_id=$1 AND expires_at>clock_timestamp()",
                    flow.owner_id,
                )
                if recent >= 128:
                    raise ConnectionsError("Authorization quota exceeded", 429)
                active = await conn.fetchval(
                    "SELECT count(*) FROM dlightrag_connection_oauth_flows WHERE owner_id=$1 AND connection_id<>$2 AND finished_at IS NULL AND expires_at>clock_timestamp() AND flow_lease_expires_at>clock_timestamp()",
                    flow.owner_id,
                    flow.connection_id,
                )
                if active >= 4:
                    raise ConnectionsError("Authorization quota exceeded", 429)
                # Restart supersedes only this owner's pending authorization, not
                # the live head or Grant. A dead initiator is never taken over.
                await conn.execute(
                    "UPDATE dlightrag_connection_oauth_flows SET finished_at=now(),encrypted_result=NULL,encrypted_credentials=NULL WHERE owner_id=$1 AND connection_id=$2 AND finished_at IS NULL",
                    flow.owner_id,
                    flow.connection_id,
                )
                await conn.execute(
                    """INSERT INTO dlightrag_connection_oauth_flows
                    (flow_id,owner_id,connection_id,flow_owner,endpoint,expected_revision,flow_lease_expires_at,expires_at)
                    VALUES($1,$2,$3,$4,$5,$6,clock_timestamp()+$7*interval '1 second',clock_timestamp()+$8*interval '1 second')""",
                    flow.flow_id,
                    flow.owner_id,
                    flow.connection_id,
                    flow.flow_owner,
                    flow.endpoint,
                    flow.expected_revision,
                    lease,
                    lifetime,
                )

        await self._run_once(operation)

    async def renew_oauth_flow(self, *, flow: OAuthFlow, lease: float) -> bool:
        async def operation(conn: Any) -> bool:
            return bool(
                await conn.fetchval(
                    """UPDATE dlightrag_connection_oauth_flows SET flow_lease_expires_at=clock_timestamp()+$4*interval '1 second'
                WHERE flow_id=$1 AND owner_id=$2 AND flow_owner=$3 AND finished_at IS NULL
                AND flow_lease_expires_at>clock_timestamp() AND expires_at>clock_timestamp() RETURNING 1""",
                    flow.flow_id,
                    flow.owner_id,
                    flow.flow_owner,
                    lease,
                )
            )

        return await self._run_once(operation)

    async def oauth_redirect(self, *, flow: OAuthFlow, state_hash: str) -> None:
        async def operation(conn: Any) -> None:
            found = await conn.fetchval(
                """UPDATE dlightrag_connection_oauth_flows SET state_hash=$4
                WHERE flow_id=$1 AND owner_id=$2 AND flow_owner=$3 AND state_hash IS NULL AND finished_at IS NULL
                AND flow_lease_expires_at>clock_timestamp() AND expires_at>clock_timestamp() RETURNING 1""",
                flow.flow_id,
                flow.owner_id,
                flow.flow_owner,
                state_hash,
            )
            if not found:
                raise ConnectionsError("Authorization expired; restart from Settings")

        await self._run_once(operation)

    async def oauth_credentials(self, *, flow: OAuthFlow, envelope: str) -> None:
        async def operation(conn: Any) -> None:
            found = await conn.fetchval(
                """UPDATE dlightrag_connection_oauth_flows SET encrypted_credentials=$4
                WHERE flow_id=$1 AND owner_id=$2 AND flow_owner=$3 AND finished_at IS NULL
                AND flow_lease_expires_at>clock_timestamp() AND expires_at>clock_timestamp() RETURNING 1""",
                flow.flow_id,
                flow.owner_id,
                flow.flow_owner,
                envelope,
            )
            if not found:
                raise ConnectionsError("Authorization expired; restart from Settings")

        await self._run_once(operation)

    async def oauth_callback_flow(self, *, owner_id: str, state_hash: str) -> OAuthFlow:
        async def operation(conn: Any) -> OAuthFlow:
            row = await conn.fetchrow(
                """SELECT flow_id,owner_id,connection_id,flow_owner,endpoint,expected_revision
                FROM dlightrag_connection_oauth_flows WHERE owner_id=$1 AND state_hash=$2
                AND deposited_at IS NULL AND finished_at IS NULL AND flow_lease_expires_at>clock_timestamp() AND expires_at>clock_timestamp()""",
                owner_id,
                state_hash,
            )
            if row is None:
                raise ConnectionsError(
                    "Authorization expired or invalid; restart from Settings", 400
                )
            return OAuthFlow(**dict(row))

        return await self._run_once(operation)

    async def deposit_oauth_callback(
        self, *, flow: OAuthFlow, state_hash: str, envelope: str
    ) -> None:
        async def operation(conn: Any) -> None:
            async with conn.transaction():
                found = await conn.fetchval(
                    """UPDATE dlightrag_connection_oauth_flows SET encrypted_result=$4,deposited_at=now()
                    WHERE flow_id=$1 AND owner_id=$2 AND state_hash=$3 AND deposited_at IS NULL AND finished_at IS NULL
                    AND flow_lease_expires_at>clock_timestamp() AND expires_at>clock_timestamp() RETURNING 1""",
                    flow.flow_id,
                    flow.owner_id,
                    state_hash,
                    envelope,
                )
                if not found:
                    raise ConnectionsError(
                        "Authorization expired or invalid; restart from Settings", 400
                    )
                await conn.execute(
                    "SELECT pg_notify('dlightrag_connection_oauth',$1)", flow.flow_owner
                )

        await self._run_once(operation)

    async def consume_oauth_callback(self, *, flow: OAuthFlow) -> str | None:
        async def operation(conn: Any) -> str | None:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """SELECT encrypted_result FROM dlightrag_connection_oauth_flows
                    WHERE flow_id=$1 AND owner_id=$2 AND flow_owner=$3 AND consumed_at IS NULL AND finished_at IS NULL
                    AND flow_lease_expires_at>clock_timestamp() AND expires_at>clock_timestamp() FOR UPDATE""",
                    flow.flow_id,
                    flow.owner_id,
                    flow.flow_owner,
                )
                if row is None:
                    raise ConnectionsError("Authorization expired; restart from Settings")
                result = row["encrypted_result"]
                if result is not None:
                    await conn.execute(
                        "UPDATE dlightrag_connection_oauth_flows SET encrypted_result=NULL,consumed_at=now() WHERE flow_id=$1",
                        flow.flow_id,
                    )
                return result

        return await self._run_once(operation)

    async def finish_oauth_flow(self, *, flow: OAuthFlow) -> None:
        async def operation(conn: Any) -> None:
            await conn.execute(
                "UPDATE dlightrag_connection_oauth_flows SET finished_at=coalesce(finished_at,now()),encrypted_result=NULL,encrypted_credentials=NULL WHERE flow_id=$1 AND owner_id=$2 AND flow_owner=$3",
                flow.flow_id,
                flow.owner_id,
                flow.flow_owner,
            )

        await self._run_once(operation)

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
    ) -> None:
        async def operation(conn: Any) -> None:
            async with conn.transaction():
                await _owner_lock(conn, owner_id)
                row = await conn.fetchrow(
                    _SELECT
                    + " WHERE h.owner_id=$1 AND h.connection_id=$2 AND h.tombstoned_at IS NULL FOR UPDATE OF h",
                    owner_id,
                    connection_id,
                )
                if row is None:
                    raise ConnectionsError("Connection not found", 404)
                if await _revision(conn, owner_id) != expected_revision:
                    raise ConnectionsError("Connections revision changed")
                if flow is not None:
                    valid = await conn.fetchval(
                        "SELECT 1 FROM dlightrag_connection_oauth_flows WHERE flow_id=$1 AND owner_id=$2 AND flow_owner=$3 AND flow_lease_expires_at>clock_timestamp() AND expires_at>clock_timestamp() AND finished_at IS NULL AND consumed_at IS NOT NULL FOR UPDATE",
                        flow.flow_id,
                        owner_id,
                        flow.flow_owner,
                    )
                    if not valid:
                        raise ConnectionsError("Authorization expired; restart from Settings")
                if row["enabled"]:
                    await self._check_tool_quota(
                        conn, owner_id, connection_id, len(catalogue), policy
                    )
                await conn.execute(
                    "UPDATE dlightrag_connection_grants SET status='retired',encrypted_envelope=NULL,secret_version=secret_version+1,refresh_epoch=refresh_epoch+1 WHERE owner_id=$1 AND connection_id=$2",
                    owner_id,
                    connection_id,
                )
                await conn.execute(
                    """INSERT INTO dlightrag_connection_grants
                    (owner_id,connection_id,grant_id,kind,audience_digest,encrypted_envelope,key_id,consented_scopes)
                    VALUES($1,$2,$3,$4,$5,$6,$7,$8::jsonb)""",
                    owner_id,
                    connection_id,
                    grant_id,
                    kind,
                    hashlib.sha256(endpoint.encode()).hexdigest(),
                    envelope,
                    key_id,
                    json.dumps(scopes),
                )
                generation = row["head_generation"] + 1
                await self._generation(
                    conn, owner_id, connection_id, generation, endpoint, grant_id, catalogue
                )
                await conn.execute(
                    """UPDATE dlightrag_connection_heads SET head_generation=$3,revision=revision+1,
                    refresh_epoch=refresh_epoch+1,refresh_owner=NULL,refresh_expires_at=NULL,
                    refresh_due_at=now()+$4*interval '1 second',observed_status=CASE WHEN enabled THEN 'ready' ELSE 'disabled' END,last_error_kind=NULL
                    WHERE owner_id=$1 AND connection_id=$2""",
                    owner_id,
                    connection_id,
                    generation,
                    policy.refresh_seconds,
                )
                if flow is not None:
                    await conn.execute(
                        "UPDATE dlightrag_connection_oauth_flows SET finished_at=now(),succeeded=TRUE,encrypted_credentials=NULL,encrypted_result=NULL WHERE flow_id=$1",
                        flow.flow_id,
                    )
                await conn.execute("SELECT pg_notify('dlightrag_connections_changed',$1)", owner_id)

        await self._run_once(operation)

    async def _generation(
        self,
        conn: Any,
        owner: str,
        identity: str,
        generation: int,
        endpoint: str,
        grant: str | None,
        catalogue: tuple[CatalogueTool, ...] | None,
    ) -> None:
        encoded = (
            json.dumps([asdict(tool) for tool in catalogue], sort_keys=True, separators=(",", ":"))
            if catalogue is not None
            else None
        )
        await conn.execute(
            """INSERT INTO dlightrag_connection_generations
            (owner_id,connection_id,generation,endpoint_json,endpoint_digest,grant_id,catalogue_json,catalogue_digest)
            VALUES($1,$2,$3,$4::jsonb,$5,$6,$7::jsonb,$8)""",
            owner,
            identity,
            generation,
            json.dumps({"url": endpoint}),
            hashlib.sha256(endpoint.encode()).hexdigest(),
            grant,
            encoded,
            hashlib.sha256(encoded.encode()).hexdigest() if encoded is not None else None,
        )

    async def _check_tool_quota(
        self, conn: Any, owner: str, identity: str, count: int, policy: ConnectionPolicy
    ) -> None:
        existing = await conn.fetchval(
            """SELECT coalesce(sum(jsonb_array_length(g.catalogue_json)),0)
            FROM dlightrag_connection_heads h JOIN dlightrag_connection_generations g
            ON (g.owner_id,g.connection_id,g.generation)=(h.owner_id,h.connection_id,h.head_generation)
            WHERE h.owner_id=$1 AND h.connection_id<>$2 AND h.enabled AND h.tombstoned_at IS NULL""",
            owner,
            identity,
        )
        if existing + count > policy.max_enabled_tools:
            raise ConnectionsError("Enabled tool quota exceeded")

    async def claim(
        self,
        *,
        worker_id: str,
        lease_seconds: float,
        owner_id: str | None = None,
        connection_id: str | None = None,
    ) -> RefreshClaim | None:
        async def operation(conn: Any) -> RefreshClaim | None:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """SELECT owner_id,connection_id FROM dlightrag_connection_heads
                    WHERE tombstoned_at IS NULL AND observed_status<>'revoked'
                    AND (refresh_expires_at IS NULL OR refresh_expires_at < now())
                    AND (($1::text IS NULL AND enabled AND refresh_due_at<=now())
                        OR (owner_id=$1 AND connection_id=$2))
                    ORDER BY refresh_due_at,owner_id,connection_id FOR UPDATE SKIP LOCKED LIMIT 1""",
                    owner_id,
                    connection_id,
                )
                if row is None:
                    return None
                epoch = await conn.fetchval(
                    """UPDATE dlightrag_connection_heads SET refresh_owner=$3,
                    refresh_epoch=refresh_epoch+1,refresh_expires_at=now()+$4*interval '1 second',
                    observed_status='refreshing',last_attempt_at=now()
                    WHERE owner_id=$1 AND connection_id=$2 RETURNING refresh_epoch""",
                    row["owner_id"],
                    row["connection_id"],
                    worker_id,
                    lease_seconds,
                )
                loaded = await conn.fetchrow(
                    _SELECT + " WHERE h.owner_id=$1 AND h.connection_id=$2",
                    row["owner_id"],
                    row["connection_id"],
                )
                return RefreshClaim(_stored(loaded), worker_id, epoch)

        return await self._run_once(operation)

    async def publish(
        self,
        *,
        claim: RefreshClaim,
        catalogue: tuple[CatalogueTool, ...] | None,
        error: str | None,
        retry_seconds: float,
        policy: ConnectionPolicy,
    ) -> bool:
        item = claim.connection

        async def operation(conn: Any) -> bool:
            async with conn.transaction():
                await _owner_lock(conn, item.owner_id)
                head = await conn.fetchrow(
                    """SELECT * FROM dlightrag_connection_heads WHERE owner_id=$1 AND connection_id=$2
                    AND refresh_owner=$3 AND refresh_epoch=$4 AND revision=$5
                    AND refresh_expires_at>now() AND tombstoned_at IS NULL FOR UPDATE""",
                    item.owner_id,
                    item.connection_id,
                    claim.worker_id,
                    claim.epoch,
                    item.revision,
                )
                if head is None:
                    return False
                generation_row = await conn.fetchrow(
                    """SELECT grant_id FROM dlightrag_connection_generations
                    WHERE owner_id=$1 AND connection_id=$2 AND generation=$3 FOR UPDATE""",
                    item.owner_id,
                    item.connection_id,
                    item.generation,
                )
                if generation_row is None or generation_row["grant_id"] != item.grant_id:
                    return False
                if item.grant_id:
                    grant = await conn.fetchrow(
                        """SELECT status,secret_version,refresh_epoch
                        FROM dlightrag_connection_grants WHERE owner_id=$1 AND connection_id=$2 AND grant_id=$3 FOR UPDATE""",
                        item.owner_id,
                        item.connection_id,
                        item.grant_id,
                    )
                    if (
                        grant is None
                        or grant["status"] != "active"
                        or grant["secret_version"] != item.secret_version
                        or grant["refresh_epoch"] != item.grant_refresh_epoch
                    ):
                        return False
                active_error = error
                if catalogue is not None and head["enabled"]:
                    try:
                        await self._check_tool_quota(
                            conn, item.owner_id, item.connection_id, len(catalogue), policy
                        )
                    except ConnectionsError:
                        active_error = "quota"
                generation = item.generation
                if catalogue is not None and active_error is None:
                    generation += 1
                    await self._generation(
                        conn,
                        item.owner_id,
                        item.connection_id,
                        generation,
                        item.endpoint,
                        item.grant_id,
                        catalogue,
                    )
                status = (
                    "ready"
                    if active_error is None
                    else "needs-auth"
                    if active_error == "authentication"
                    else "degraded"
                )
                failures = min(10, head["refresh_failures"] + 1) if active_error else 0
                delay = (
                    retry_seconds
                    if not failures
                    else min(retry_seconds, 2**failures) * secrets.SystemRandom().uniform(0.8, 1.0)
                )
                await conn.execute(
                    """UPDATE dlightrag_connection_heads SET head_generation=$3,revision=revision+1,
                    observed_status=$4,last_error_kind=$5,refresh_owner=NULL,refresh_expires_at=NULL,
                    refresh_due_at=now()+$6*interval '1 second',refresh_failures=$7 WHERE owner_id=$1 AND connection_id=$2""",
                    item.owner_id,
                    item.connection_id,
                    generation,
                    status,
                    active_error,
                    delay,
                    failures,
                )
                await conn.execute(
                    "SELECT pg_notify('dlightrag_connections_changed',$1)", item.owner_id
                )
                return True

        return await self._run_once(operation)


class PGConnectionPinWriter:
    """Purpose-built participant in the accepting transaction, never its owner."""

    @staticmethod
    async def validate_in(
        conn: Any,
        *,
        owner_id: str,
        payload: Mapping[str, Any],
        bindings: tuple[RunConnectionBinding, ...],
    ) -> None:
        if decode_connection_bindings(payload.get("run_connection_bindings", [])) != bindings:
            raise ValueError("Run Connection input and normalized pins differ")
        if bindings and (
            payload.get("auth_mode", "none") not in {"none", "jwt"} or payload.get("mode") == "fast"
        ):
            raise ValueError("Run Connections require eligible Research acceptance")
        ordered = sorted(bindings, key=lambda b: (b.owner_id, b.connection_id))
        heads = []
        # All heads precede generations/Grants; revoke and publication use the
        # same head -> generation -> Grant order, with no network-held locks.
        for binding in ordered:
            if binding.owner_id != owner_id:
                raise ValueError("Run Connection owner mismatch")
            head = await conn.fetchrow(
                """SELECT * FROM dlightrag_connection_heads
                WHERE owner_id=$1 AND connection_id=$2 FOR UPDATE""",
                owner_id,
                binding.connection_id,
            )
            if (
                head is None
                or not head["enabled"]
                or head["tombstoned_at"] is not None
                or head["consent_version"] != 1
                or head["head_generation"] != binding.generation
                or head["activation_epoch"] != binding.activation_epoch
            ):
                raise StaleConnectionBindingError("Connections changed during Answer acceptance")
            heads.append(binding)
        for binding in heads:
            generation = await conn.fetchrow(
                """SELECT * FROM dlightrag_connection_generations
                WHERE owner_id=$1 AND connection_id=$2 AND generation=$3 FOR UPDATE""",
                owner_id,
                binding.connection_id,
                binding.generation,
            )
            if (
                generation is None
                or generation["catalogue_json"] is None
                or generation["catalogue_digest"] != binding.catalogue_digest
            ):
                raise StaleConnectionBindingError(
                    "Connection generation changed during Answer acceptance"
                )
            if generation["grant_id"] is not None:
                grant = await conn.fetchrow(
                    """SELECT status FROM dlightrag_connection_grants
                    WHERE owner_id=$1 AND connection_id=$2 AND grant_id=$3 FOR UPDATE""",
                    owner_id,
                    binding.connection_id,
                    generation["grant_id"],
                )
                if grant is None or grant["status"] != "active":
                    raise StaleConnectionBindingError(
                        "Connection grant changed during Answer acceptance"
                    )

    @staticmethod
    async def insert_in(
        conn: Any, *, owner_id: str, run_id: uuid.UUID, bindings: tuple[RunConnectionBinding, ...]
    ) -> None:
        for binding in bindings:
            await conn.execute(
                """INSERT INTO dlightrag_answer_connection_pins
                (owner_id,run_id,connection_id,generation,activation_epoch,catalogue_digest)
                VALUES($1,$2,$3,$4,$5,$6)""",
                owner_id,
                run_id,
                binding.connection_id,
                binding.generation,
                binding.activation_epoch,
                binding.catalogue_digest,
            )
