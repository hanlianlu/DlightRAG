# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL-backed answer metadata schema lookup."""

from collections.abc import Sequence
from typing import Any

from dlightrag.adapters.postgres.pg_metadata_index import PGMetadataIndex


class PGWorkspaceSchemaLookup:
    """Read the current metadata schema for one concrete workspace set."""

    def __init__(self, *, default_workspace: str) -> None:
        self._metadata_index = PGMetadataIndex(workspace=default_workspace)

    async def __call__(self, workspaces: Sequence[str]) -> dict[str, Any]:
        return await self._metadata_index.get_field_schema(workspaces=tuple(workspaces))


__all__ = ["PGWorkspaceSchemaLookup"]
