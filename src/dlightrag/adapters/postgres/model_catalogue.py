# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL active-overlay store and NOTIFY wake adapter."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from dlightrag.adapters.postgres.core._channels import MODEL_CATALOGUE_CHANNEL
from dlightrag.adapters.postgres.core._migrations import (
    Migration,
    TableRequirement,
    apply_migrations,
    verify_migrations,
)
from dlightrag.adapters.postgres.core._notifications import ChannelWatcher, PGNotificationHub
from dlightrag.adapters.postgres.core._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.application.model_catalogue import (
    ModelCatalogueSchemaError,
    StoredModelCatalogue,
)

MODEL_CATALOGUE_MIGRATION_SCOPE = "model_catalogue"

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
        notifications: PGNotificationHub | None = None,
    ) -> None:
        super().__init__(pool=pool, notifications=notifications)
        self._initial_revision = initial_revision
        self._listener: ChannelWatcher | None = None

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
                    tables=MODEL_CATALOGUE_SCHEMA_TABLES,
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
        """Reload after every publication, and whenever the hub resynchronizes."""
        if self._listener is not None:
            return
        listener = ChannelWatcher(
            self._notification_hub,
            MODEL_CATALOGUE_CHANNEL,
            on_change,
            name="Model catalogue reload",
        )
        await listener.start()
        self._listener = listener

    async def aclose(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            await listener.aclose()


__all__ = [
    "MODEL_CATALOGUE_MIGRATION_SCOPE",
    "MODEL_CATALOGUE_MIGRATIONS",
    "MODEL_CATALOGUE_SCHEMA_TABLES",
    "PGModelCatalogueStore",
]
