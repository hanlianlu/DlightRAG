# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Workspace registry inputs refused before any statement runs.

The registry's rows, constraints, fences and keyset pages run against PostgreSQL
in tests/integration/test_promotion_foundation_pg.py and test_pg_storage.py.
"""

from typing import Any

import pytest

from dlightrag.adapters.postgres.corpus.workspaces import PGWorkspaceRegistry


class _Pool:
    def acquire(self) -> Any:
        raise AssertionError("no connection may be taken before the inputs are validated")


def _registry() -> PGWorkspaceRegistry:
    return PGWorkspaceRegistry(pool=_Pool())


async def test_workspace_registry_rejects_an_empty_workspace() -> None:
    registry = _registry()

    with pytest.raises(ValueError, match="workspace cannot be empty"):
        await registry.exists("  ")
    with pytest.raises(ValueError, match="workspace cannot be empty"):
        await registry.upsert(
            workspace="  ",
            display_name="Empty",
            embedding_model="voyage-multimodal-3.5",
        )


async def test_promotion_state_is_validated() -> None:
    registry = _registry()

    with pytest.raises(ValueError, match="must record its error"):
        await registry.set_promotion_state(workspace="research", state="failed")
    with pytest.raises(ValueError, match="retry time"):
        await registry.set_promotion_state(
            workspace="research", state="failed", error="invariant mismatch"
        )
    with pytest.raises(ValueError, match="promotion state"):
        await registry.set_promotion_state(workspace="research", state="cutover")


async def test_write_fence_owner_cannot_be_empty() -> None:
    registry = _registry()

    with pytest.raises(ValueError, match="owner cannot be empty"):
        await registry.acquire_write_fence(
            workspace="research", owner="  ", until="2026-04-01T00:00:00Z"
        )
    with pytest.raises(ValueError, match="owner cannot be empty"):
        await registry.release_write_fence(workspace="research", owner="  ")


async def test_workspace_registry_list_page_rejects_invalid_inputs() -> None:
    registry = _registry()

    with pytest.raises(ValueError, match="canonical"):
        await registry.list_page(after_workspace="Finance!", limit=1)
    with pytest.raises(ValueError, match="limit"):
        await registry.list_page(after_workspace=None, limit=0)
    with pytest.raises(ValueError, match="limit"):
        await registry.list_page(after_workspace=None, limit=101)
