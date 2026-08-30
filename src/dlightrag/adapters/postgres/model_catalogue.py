# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL active-overlay store and NOTIFY wake adapter."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from typing import Any

import asyncpg

from dlightrag.adapters.postgres.core._migrations import (
    Migration,
    TableRequirement,
    apply_migrations,
    verify_migrations,
)
from dlightrag.adapters.postgres.core._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.application.model_catalogue import (
    ModelCatalogueSchemaError,
    StoredModelCatalogue,
)

MODEL_CATALOGUE_CHANNEL = "dlightrag_model_catalogue_changed"
MODEL_CATALOGUE_MIGRATION_SCOPE = "model_catalogue"
_RECONNECT_BASE_SECONDS = 1.0
_RECONNECT_MAX_SECONDS = 30.0

logger = logging.getLogger(__name__)

_CREATE_MODEL_CATALOGUE = """
CREATE TABLE IF NOT EXISTS dlightrag_model_catalogue (
    singleton  BOOLEAN     NOT NULL DEFAULT TRUE,
    revision   TEXT        NOT NULL,
    overlay    JSONB       NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by TEXT        NOT NULL,
    PRIMARY KEY (singleton),
    CONSTRAINT dlightrag_model_catalogue_singleton_check CHECK (singleton),
    CONSTRAINT dlightrag_model_catalogue_revision_check
        CHECK (revision ~ '^sha256:[0-9a-f]{64}$'),
    CONSTRAINT dlightrag_model_catalogue_overlay_check
        CHECK (jsonb_typeof(overlay) = 'array')
)
"""

MODEL_CATALOGUE_MIGRATIONS = (
    Migration(
        "model_catalogue",
        "Create the active runtime model catalogue overlay",
        (_CREATE_MODEL_CATALOGUE,),
    ),
)

MODEL_CATALOGUE_SCHEMA_TABLES = (
    TableRequirement(
        name="dlightrag_model_catalogue",
        columns=(
            "singleton",
            "revision",
            "overlay",
            "updated_at",
            "updated_by",
        ),
        primary_key=("singleton",),
        checks=(
            "dlightrag_model_catalogue_singleton_check",
            "dlightrag_model_catalogue_revision_check",
            "dlightrag_model_catalogue_overlay_check",
        ),
    ),
)

_INSERT_INITIAL = """
INSERT INTO dlightrag_model_catalogue (
    singleton, revision, overlay, updated_by)
VALUES (TRUE, $1, $2::jsonb, 'system:bootstrap')
ON CONFLICT (singleton) DO NOTHING
"""

_LOAD = """
SELECT revision, overlay::text AS overlay
FROM dlightrag_model_catalogue
WHERE singleton = TRUE
"""

_PUBLISH = """
UPDATE dlightrag_model_catalogue
SET revision = $2,
    overlay = $3::jsonb,
    updated_at = NOW(),
    updated_by = $4
WHERE singleton = TRUE AND revision = $1
RETURNING revision
"""


class PGModelCatalogueStore(PostgresOperationRunner):
    """Persist one active overlay and wake every process after publication."""

    def __init__(
        self,
        *,
        initial_revision: str,
        pool: ConnectionPool | None = None,
        open_connection: Callable[[], Awaitable[Any]] | None = None,
        connection_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(pool=pool)
        self._initial_revision = initial_revision
        self._connection_kwargs = dict(connection_kwargs) if connection_kwargs is not None else None
        self._open_connection = open_connection or self._connect
        self._listener_task: asyncio.Task[None] | None = None
        self._closing = False

    async def initialize(self, *, validate_only: bool) -> None:
        async def operation(conn: Any) -> None:
            if validate_only:
                await verify_migrations(
                    conn,
                    scope=MODEL_CATALOGUE_MIGRATION_SCOPE,
                    migrations=MODEL_CATALOGUE_MIGRATIONS,
                    tables=MODEL_CATALOGUE_SCHEMA_TABLES,
                    schema_error=ModelCatalogueSchemaError,
                )
            else:
                await apply_migrations(
                    conn,
                    scope=MODEL_CATALOGUE_MIGRATION_SCOPE,
                    migrations=MODEL_CATALOGUE_MIGRATIONS,
                    schema_error=ModelCatalogueSchemaError,
                )
                await conn.execute(_INSERT_INITIAL, self._initial_revision, "[]")
            if await conn.fetchrow(_LOAD) is None:
                raise ModelCatalogueSchemaError(
                    "runtime model catalogue row is missing; initialize it on a writer"
                )

        await self._run(operation)

    async def load(self) -> StoredModelCatalogue:
        async def operation(conn: Any) -> StoredModelCatalogue:
            row = await conn.fetchrow(_LOAD)
            if row is None:
                raise ModelCatalogueSchemaError("runtime model catalogue row is missing")
            try:
                overlay = json.loads(str(row["overlay"]))
            except json.JSONDecodeError as exc:
                raise ModelCatalogueSchemaError(
                    "runtime model catalogue overlay is not valid JSON"
                ) from exc
            return StoredModelCatalogue(
                revision=str(row["revision"]),
                overlay=overlay,
            )

        return await self._run(operation)

    async def publish(
        self,
        *,
        expected_revision: str,
        revision: str,
        overlay: object,
        actor: str,
    ) -> bool:
        encoded = json.dumps(
            overlay,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        async def operation(conn: Any) -> bool:
            async with conn.transaction():
                row = await conn.fetchrow(
                    _PUBLISH,
                    expected_revision,
                    revision,
                    encoded,
                    actor,
                )
                if row is None:
                    return False
                await conn.execute("SELECT pg_notify($1, $2)", MODEL_CATALOGUE_CHANNEL, revision)
                return True

        return await self._run_once(operation)

    async def start_listener(self, on_change: Callable[[], Awaitable[None]]) -> None:
        if self._listener_task is not None:
            return
        self._closing = False
        self._listener_task = asyncio.create_task(
            self._listen_forever(on_change),
            name="model-catalogue-listener",
        )

    async def _listen_forever(self, on_change: Callable[[], Awaitable[None]]) -> None:
        backoff = _RECONNECT_BASE_SECONDS
        while not self._closing:
            try:
                await self._listen_once(on_change)
                backoff = _RECONNECT_BASE_SECONDS
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Model catalogue listener failed; retrying in %.1fs",
                    backoff,
                    exc_info=True,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RECONNECT_MAX_SECONDS)

    async def _listen_once(self, on_change: Callable[[], Awaitable[None]]) -> None:
        connection = await self._open_connection()
        queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        terminated = asyncio.Event()

        def changed(_conn: object, _pid: int, channel: str, _payload: str) -> None:
            if channel == MODEL_CATALOGUE_CHANNEL and queue.empty():
                queue.put_nowait(None)

        def disconnected(_conn: object) -> None:
            terminated.set()
            if queue.empty():
                queue.put_nowait(None)

        try:
            await connection.add_listener(MODEL_CATALOGUE_CHANNEL, changed)
            connection.add_termination_listener(disconnected)
            # Startup/reconnect synchronization closes every missed-NOTIFY gap.
            await on_change()
            while not self._closing:
                await queue.get()
                if terminated.is_set():
                    return
                try:
                    await on_change()
                except Exception:
                    logger.warning("Model catalogue notification reload failed", exc_info=True)
        finally:
            with suppress(Exception):
                await connection.remove_listener(MODEL_CATALOGUE_CHANNEL, changed)
            with suppress(Exception):
                await connection.close()

    async def _connect(self) -> Any:
        if self._connection_kwargs is None:
            from dlightrag.application.config import get_config

            kwargs = get_config().pg_connection_kwargs()
        else:
            kwargs = self._connection_kwargs
        return await asyncpg.connect(**kwargs)

    async def aclose(self) -> None:
        self._closing = True
        task, self._listener_task = self._listener_task, None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


__all__ = [
    "MODEL_CATALOGUE_CHANNEL",
    "MODEL_CATALOGUE_MIGRATION_SCOPE",
    "MODEL_CATALOGUE_MIGRATIONS",
    "MODEL_CATALOGUE_SCHEMA_TABLES",
    "PGModelCatalogueStore",
]
