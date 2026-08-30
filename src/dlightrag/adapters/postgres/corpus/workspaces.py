# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL workspace registry.

The registry is the durable source of truth for workspace existence and
display labels. LightRAG stores own document/KG/vector data per workspace;
this table owns the user-facing workspace list, including empty workspaces.
"""

from typing import Any

from dlightrag.adapters.postgres.core._migrations import (
    Migration,
    TableRequirement,
    apply_migrations,
    verify_migrations,
)
from dlightrag.adapters.postgres.core._operations import PostgresOperationRunner
from dlightrag.application.corpus_admin import (
    WORKSPACE_CATALOG_PAGE_MAX_LIMIT,
    WorkspaceCatalogRowPage,
)
from dlightrag.engine.rag.workspace.ports import CorpusSchemaError
from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id

_CREATE = """
CREATE TABLE IF NOT EXISTS dlightrag_workspace_meta (
    workspace       TEXT PRIMARY KEY,
    display_name    TEXT NOT NULL DEFAULT '',
    embedding_model TEXT NOT NULL DEFAULT '',
    ingested_docs_total    BIGINT NOT NULL DEFAULT 0,
    ingested_chunks_total  BIGINT NOT NULL DEFAULT 0,
    storage_tier           TEXT NOT NULL DEFAULT 'shared',
    promotion_state        TEXT NOT NULL DEFAULT 'none',
    promotion_last_error   TEXT,
    promotion_retry_count  INTEGER NOT NULL DEFAULT 0,
    promotion_next_retry_at TIMESTAMPTZ,
    write_fence_owner      TEXT,
    write_fence_until      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT dlightrag_workspace_meta_counters_nonnegative
        CHECK (ingested_docs_total >= 0 AND ingested_chunks_total >= 0),
    CONSTRAINT dlightrag_workspace_meta_tier
        CHECK (storage_tier IN ('shared', 'hot')),
    CONSTRAINT dlightrag_workspace_meta_promotion_state
        CHECK (promotion_state IN ('none', 'pending', 'promoting', 'failed')),
    CONSTRAINT dlightrag_workspace_meta_retry_nonnegative
        CHECK (promotion_retry_count >= 0),
    CONSTRAINT dlightrag_workspace_meta_fence_pair
        CHECK ((write_fence_owner IS NULL) = (write_fence_until IS NULL)),
    CONSTRAINT dlightrag_workspace_meta_failed_error
        CHECK ((promotion_state = 'failed') = (promotion_last_error IS NOT NULL)),
    CONSTRAINT dlightrag_workspace_meta_retry_state
        CHECK ((promotion_state = 'failed') = (promotion_next_retry_at IS NOT NULL))
)
"""

_STORAGE_TIERS = frozenset({"shared", "hot"})
_PROMOTION_STATES = frozenset({"none", "pending", "promoting", "failed"})

_UPSERT = """
INSERT INTO dlightrag_workspace_meta (workspace, display_name, embedding_model)
VALUES ($1, $2, $3)
ON CONFLICT (workspace)
DO UPDATE SET display_name = EXCLUDED.display_name,
              embedding_model = EXCLUDED.embedding_model,
              updated_at = NOW()
"""

_LIST = """
SELECT workspace, display_name, embedding_model, created_at, updated_at
FROM dlightrag_workspace_meta
ORDER BY workspace
"""

# The empty string is the before-first key: every canonical workspace id is
# non-empty ASCII, so "" < workspace for every row. The cursor predicate rides
# the table primary key; no OFFSET is ever used.
_LIST_PAGE = """
SELECT workspace, display_name, embedding_model, created_at, updated_at
FROM dlightrag_workspace_meta
WHERE workspace > $1
ORDER BY workspace ASC
LIMIT $2
"""

_EXISTS = """
SELECT EXISTS (
    SELECT 1 FROM dlightrag_workspace_meta WHERE workspace = $1
)
"""

_DELETE = "DELETE FROM dlightrag_workspace_meta WHERE workspace = $1"

_ADD_INGESTED_COUNTS = """
UPDATE dlightrag_workspace_meta
SET ingested_docs_total = ingested_docs_total + $2,
    ingested_chunks_total = ingested_chunks_total + $3,
    updated_at = NOW()
WHERE workspace = $1
"""

_SET_STORAGE_TIER = """
UPDATE dlightrag_workspace_meta
SET storage_tier = $2, updated_at = NOW()
WHERE workspace = $1
  AND (storage_tier = $2 OR (storage_tier = 'shared' AND $2 = 'hot'))
"""

_SET_PROMOTION_STATE = """
UPDATE dlightrag_workspace_meta
SET promotion_state = $2,
    promotion_last_error = CASE WHEN $2 = 'failed' THEN $3 ELSE NULL END,
    promotion_retry_count = CASE WHEN $2 = 'failed' THEN promotion_retry_count + 1
                                 ELSE promotion_retry_count END,
    promotion_next_retry_at = CASE WHEN $2 = 'failed' THEN $4::timestamptz ELSE NULL END,
    updated_at = NOW()
WHERE workspace = $1
  AND ($5::text IS NULL OR write_fence_owner = $5)
"""

_ACQUIRE_WRITE_FENCE = """
UPDATE dlightrag_workspace_meta
SET write_fence_owner = $2, write_fence_until = $3::timestamptz, updated_at = NOW()
WHERE workspace = $1
  AND $3::timestamptz > NOW()
  AND (write_fence_owner IS NULL
       OR write_fence_owner = $2
       OR write_fence_until <= NOW())
"""

_RELEASE_WRITE_FENCE = """
UPDATE dlightrag_workspace_meta
SET write_fence_owner = NULL, write_fence_until = NULL, updated_at = NOW()
WHERE workspace = $1 AND (write_fence_owner = $2 OR write_fence_owner IS NULL)
"""

_GET_ROW = """
SELECT workspace, display_name, embedding_model,
       ingested_docs_total, ingested_chunks_total,
       storage_tier, promotion_state, promotion_last_error,
       promotion_retry_count, promotion_next_retry_at,
       write_fence_owner, write_fence_until,
       created_at, updated_at
FROM dlightrag_workspace_meta
WHERE workspace = $1
"""

_WORKSPACE_CHECK_EXPRESSIONS = (
    (
        "dlightrag_workspace_meta_counters_nonnegative",
        "ingested_docs_total >= 0 AND ingested_chunks_total >= 0",
    ),
    ("dlightrag_workspace_meta_tier", "storage_tier IN ('shared', 'hot')"),
    (
        "dlightrag_workspace_meta_promotion_state",
        "promotion_state IN ('none', 'pending', 'promoting', 'failed')",
    ),
    ("dlightrag_workspace_meta_retry_nonnegative", "promotion_retry_count >= 0"),
    (
        "dlightrag_workspace_meta_fence_pair",
        "(write_fence_owner IS NULL) = (write_fence_until IS NULL)",
    ),
    (
        "dlightrag_workspace_meta_failed_error",
        "(promotion_state = 'failed') = (promotion_last_error IS NOT NULL)",
    ),
    (
        "dlightrag_workspace_meta_retry_state",
        "(promotion_state = 'failed') = (promotion_next_retry_at IS NOT NULL)",
    ),
)


_ADD_WORKSPACE_CHECKS = """
DO $dlightrag$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_constraint
                   WHERE conrelid = 'dlightrag_workspace_meta'::regclass
                     AND conname = 'dlightrag_workspace_meta_counters_nonnegative') THEN
        ALTER TABLE dlightrag_workspace_meta
        ADD CONSTRAINT dlightrag_workspace_meta_counters_nonnegative
        CHECK (ingested_docs_total >= 0 AND ingested_chunks_total >= 0);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_constraint
                   WHERE conrelid = 'dlightrag_workspace_meta'::regclass
                     AND conname = 'dlightrag_workspace_meta_tier') THEN
        ALTER TABLE dlightrag_workspace_meta
        ADD CONSTRAINT dlightrag_workspace_meta_tier
        CHECK (storage_tier IN ('shared', 'hot'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_constraint
                   WHERE conrelid = 'dlightrag_workspace_meta'::regclass
                     AND conname = 'dlightrag_workspace_meta_promotion_state') THEN
        ALTER TABLE dlightrag_workspace_meta
        ADD CONSTRAINT dlightrag_workspace_meta_promotion_state
        CHECK (promotion_state IN ('none', 'pending', 'promoting', 'failed'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_constraint
                   WHERE conrelid = 'dlightrag_workspace_meta'::regclass
                     AND conname = 'dlightrag_workspace_meta_retry_nonnegative') THEN
        ALTER TABLE dlightrag_workspace_meta
        ADD CONSTRAINT dlightrag_workspace_meta_retry_nonnegative
        CHECK (promotion_retry_count >= 0);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_constraint
                   WHERE conrelid = 'dlightrag_workspace_meta'::regclass
                     AND conname = 'dlightrag_workspace_meta_fence_pair') THEN
        ALTER TABLE dlightrag_workspace_meta
        ADD CONSTRAINT dlightrag_workspace_meta_fence_pair
        CHECK ((write_fence_owner IS NULL) = (write_fence_until IS NULL));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_constraint
                   WHERE conrelid = 'dlightrag_workspace_meta'::regclass
                     AND conname = 'dlightrag_workspace_meta_failed_error') THEN
        ALTER TABLE dlightrag_workspace_meta
        ADD CONSTRAINT dlightrag_workspace_meta_failed_error
        CHECK ((promotion_state = 'failed') = (promotion_last_error IS NOT NULL));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_constraint
                   WHERE conrelid = 'dlightrag_workspace_meta'::regclass
                     AND conname = 'dlightrag_workspace_meta_retry_state') THEN
        ALTER TABLE dlightrag_workspace_meta
        ADD CONSTRAINT dlightrag_workspace_meta_retry_state
        CHECK ((promotion_state = 'failed') = (promotion_next_retry_at IS NOT NULL));
    END IF;
END
$dlightrag$
"""


_SCHEMA_MIGRATIONS = (
    Migration(
        "workspace_meta",
        "Create and migrate workspace registry",
        (_CREATE,),
    ),
    Migration(
        "workspace_meta_promotion_counters",
        "Add monotonic ingestion counters",
        (
            "ALTER TABLE dlightrag_workspace_meta "
            "ADD COLUMN IF NOT EXISTS ingested_docs_total BIGINT NOT NULL DEFAULT 0",
            "ALTER TABLE dlightrag_workspace_meta "
            "ADD COLUMN IF NOT EXISTS ingested_chunks_total BIGINT NOT NULL DEFAULT 0",
        ),
    ),
    Migration(
        "workspace_meta_storage_tier",
        "Add storage tier",
        (
            "ALTER TABLE dlightrag_workspace_meta "
            "ADD COLUMN IF NOT EXISTS storage_tier TEXT NOT NULL DEFAULT 'shared'",
        ),
    ),
    Migration(
        "workspace_meta_promotion_state",
        "Add promotion observability fields",
        (
            "ALTER TABLE dlightrag_workspace_meta "
            "ADD COLUMN IF NOT EXISTS promotion_state TEXT NOT NULL DEFAULT 'none'",
            "ALTER TABLE dlightrag_workspace_meta "
            "ADD COLUMN IF NOT EXISTS promotion_last_error TEXT",
            "ALTER TABLE dlightrag_workspace_meta "
            "ADD COLUMN IF NOT EXISTS promotion_retry_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE dlightrag_workspace_meta "
            "ADD COLUMN IF NOT EXISTS promotion_next_retry_at TIMESTAMPTZ",
        ),
    ),
    Migration(
        "workspace_meta_write_fence",
        "Add promotion write-fence facts",
        (
            "ALTER TABLE dlightrag_workspace_meta ADD COLUMN IF NOT EXISTS write_fence_owner TEXT",
            "ALTER TABLE dlightrag_workspace_meta "
            "ADD COLUMN IF NOT EXISTS write_fence_until TIMESTAMPTZ",
        ),
    ),
    Migration(
        "workspace_meta_promotion_constraints",
        "Install registry counter, tier, state, retry, and fence invariants",
        (_ADD_WORKSPACE_CHECKS,),
    ),
)

_SCHEMA_TABLES = (
    TableRequirement(
        name="dlightrag_workspace_meta",
        columns=(
            "workspace",
            "display_name",
            "embedding_model",
            "ingested_docs_total",
            "ingested_chunks_total",
            "storage_tier",
            "promotion_state",
            "promotion_last_error",
            "promotion_retry_count",
            "promotion_next_retry_at",
            "write_fence_owner",
            "write_fence_until",
            "created_at",
            "updated_at",
        ),
        primary_key=("workspace",),
        checks=tuple(name for name, _expression in _WORKSPACE_CHECK_EXPRESSIONS),
    ),
)


class PGWorkspaceRegistry(PostgresOperationRunner):
    """Durable workspace registry backed by PostgreSQL."""

    async def initialize(self, *, validate_only: bool = False) -> None:
        """Create/migrate the registry table, or validate it (reader)."""

        async def _operation(conn: Any) -> None:
            if validate_only:
                await verify_migrations(
                    conn,
                    scope="workspace_registry",
                    migrations=_SCHEMA_MIGRATIONS,
                    tables=_SCHEMA_TABLES,
                    schema_error=CorpusSchemaError,
                )
                return
            await apply_migrations(
                conn,
                scope="workspace_registry",
                migrations=_SCHEMA_MIGRATIONS,
                schema_error=CorpusSchemaError,
            )

        await self._run(_operation)

    async def upsert(
        self,
        *,
        workspace: str,
        display_name: str | None,
        embedding_model: str,
    ) -> None:
        """Insert or update one workspace registry row."""
        workspace_id = _workspace_id(workspace)
        label = (display_name or workspace).strip() or workspace_id

        async def _operation(conn: Any) -> None:
            await conn.execute(_UPSERT, workspace_id, label, embedding_model)

        await self._run(_operation)

    async def list(self) -> list[dict[str, Any]]:
        """Return all registered workspaces."""

        async def _operation(conn: Any) -> list[Any]:
            return await conn.fetch(_LIST)

        rows = await self._run(_operation)
        return [dict(row) for row in rows]

    async def list_page(
        self,
        *,
        after_workspace: str | None,
        limit: int,
    ) -> WorkspaceCatalogRowPage:
        """Return one bounded ascending workspace-keyset page via the primary key."""
        if after_workspace is None:
            after = ""
        else:
            after = after_workspace.strip()
            canonical = require_canonical_workspace_id(after)
            if canonical != after:
                raise ValueError("workspace-catalog cursor workspace must be canonical")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= WORKSPACE_CATALOG_PAGE_MAX_LIMIT
        ):
            raise ValueError("workspace-catalog page limit must be between 1 and 100")
        fetch_limit = limit + 1

        async def _operation(conn: Any) -> WorkspaceCatalogRowPage:
            rows = await conn.fetch(_LIST_PAGE, after, fetch_limit)
            fetched_rows = len(rows)
            return WorkspaceCatalogRowPage(
                items=tuple(dict(row) for row in rows[:limit]),
                has_more=fetched_rows > limit,
                fetched_rows=fetched_rows,
            )

        return await self._run(_operation)

    async def exists(self, workspace: str) -> bool:
        """Return whether one canonical workspace registry row exists."""
        workspace_id = _workspace_id(workspace)

        async def _operation(conn: Any) -> bool:
            return bool(await conn.fetchval(_EXISTS, workspace_id))

        return await self._run(_operation)

    async def delete(self, workspace: str) -> bool:
        """Delete one workspace registry row."""
        workspace_id = _workspace_id(workspace)

        async def _operation(conn: Any) -> bool:
            result = await conn.execute(_DELETE, workspace_id)
            return result != "DELETE 0"

        return await self._run(_operation)

    # -- Promotion control-plane state ---------------------------------------
    # Ingestion and the promotion worker update these counters and lifecycle fields.

    async def add_ingested_counts(
        self,
        *,
        workspace: str,
        docs: int,
        chunks: int,
    ) -> bool:
        """Add non-negative ingestion counts; the stored totals never decrease."""
        workspace_id = _workspace_id(workspace)
        docs_delta = int(docs)
        chunks_delta = int(chunks)
        if docs_delta < 0 or chunks_delta < 0:
            raise ValueError("ingested count deltas must be non-negative")

        async def _operation(conn: Any) -> str:
            return await conn.execute(_ADD_INGESTED_COUNTS, workspace_id, docs_delta, chunks_delta)

        return (await self._run(_operation)) != "UPDATE 0"

    async def set_storage_tier(self, *, workspace: str, tier: str) -> bool:
        """Set the observed storage tier; dedicated workspaces never auto-demote."""
        workspace_id = _workspace_id(workspace)
        if tier not in _STORAGE_TIERS:
            raise ValueError(f"storage tier must be one of {sorted(_STORAGE_TIERS)}")

        async def _operation(conn: Any) -> str:
            return await conn.execute(_SET_STORAGE_TIER, workspace_id, tier)

        return (await self._run(_operation)) != "UPDATE 0"

    async def set_promotion_state(
        self,
        *,
        workspace: str,
        state: str,
        error: str | None = None,
        next_retry_at: Any = None,
        expected_fence_owner: str | None = None,
    ) -> bool:
        """Record promotion observability, optionally fenced by owner."""
        workspace_id = _workspace_id(workspace)
        if state not in _PROMOTION_STATES:
            raise ValueError(f"promotion state must be one of {sorted(_PROMOTION_STATES)}")
        if state == "failed" and not error:
            raise ValueError("a failed promotion state must record its error")
        if state == "failed" and next_retry_at is None:
            raise ValueError("a failed promotion state must schedule its retry time")

        fence_owner = (
            str(expected_fence_owner).strip() if expected_fence_owner is not None else None
        )
        if expected_fence_owner is not None and not fence_owner:
            raise ValueError("expected fence owner cannot be empty")

        async def _operation(conn: Any) -> str:
            return await conn.execute(
                _SET_PROMOTION_STATE,
                workspace_id,
                state,
                error,
                next_retry_at,
                fence_owner,
            )

        return (await self._run(_operation)) != "UPDATE 0"

    async def acquire_write_fence(
        self,
        *,
        workspace: str,
        owner: str,
        until: Any,
    ) -> bool:
        """Take or renew the promotion write fence; an expired fence is free."""
        workspace_id = _workspace_id(workspace)
        owner_id = str(owner).strip()
        if not owner_id:
            raise ValueError("write-fence owner cannot be empty")

        async def _operation(conn: Any) -> str:
            return await conn.execute(_ACQUIRE_WRITE_FENCE, workspace_id, owner_id, until)

        return (await self._run(_operation)) != "UPDATE 0"

    async def release_write_fence(self, *, workspace: str, owner: str) -> bool:
        """Release a fence the caller owns (or one that is already gone)."""
        workspace_id = _workspace_id(workspace)
        owner_id = str(owner).strip()
        if not owner_id:
            raise ValueError("write-fence owner cannot be empty")

        async def _operation(conn: Any) -> str:
            return await conn.execute(_RELEASE_WRITE_FENCE, workspace_id, owner_id)

        return (await self._run(_operation)) != "UPDATE 0"

    async def get_row(self, workspace: str) -> dict[str, Any] | None:
        """Return the full registry row, including control-plane facts."""
        workspace_id = _workspace_id(workspace)

        async def _operation(conn: Any) -> Any:
            return await conn.fetchrow(_GET_ROW, workspace_id)

        row = await self._run(_operation)
        return dict(row) if row is not None else None


def _workspace_id(workspace: str) -> str:
    workspace_id = str(workspace).strip()
    if not workspace_id:
        raise ValueError("workspace cannot be empty")
    return workspace_id


__all__ = ["PGWorkspaceRegistry"]
