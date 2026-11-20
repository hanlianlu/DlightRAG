# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Run-side host for the three shared MCP resource tools.

The tools carry a model-worded Connection token; the host is the one place that turns it
into the identity a pinned Run accepted and hands the read to the Connection owner's live,
gated resource port. Engine code never sees a credential, an endpoint, or the store: only
the port this host holds.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from dlightrag.engine.agent.tools.contracts import (
    ToolResult,
    ToolRuntime,
)
from dlightrag.engine.answer.execution.connection_binding import (
    ConnectionResourceOperations,
    ConnectionSurface,
    ResearchToolClaim,
)


class RunMcpResourceHost:
    """Resolve one Run's pinned Connection tokens and perform its resource reads."""

    def __init__(
        self,
        *,
        surface: Sequence[ConnectionSurface],
        operations: ConnectionResourceOperations,
        claim: ResearchToolClaim,
    ) -> None:
        self._by_token = {entry.token: entry for entry in surface}
        self._operations = operations
        self._claim = claim

    def _connection_id(self, connection: str) -> str | None:
        entry = self._by_token.get(connection)
        return None if entry is None else entry.connection_id

    async def list_resources(
        self,
        *,
        connection: str,
        cursor: str | None,
        runtime: ToolRuntime,
    ) -> ToolResult:
        return await self._read(
            connection,
            lambda connection_id: self._operations.list_resources(
                claim=self._claim,
                runtime=runtime,
                connection_id=connection_id,
                cursor=cursor,
            ),
        )

    async def list_resource_templates(
        self,
        *,
        connection: str,
        cursor: str | None,
        runtime: ToolRuntime,
    ) -> ToolResult:
        return await self._read(
            connection,
            lambda connection_id: self._operations.list_resource_templates(
                claim=self._claim,
                runtime=runtime,
                connection_id=connection_id,
                cursor=cursor,
            ),
        )

    async def read_resource(
        self,
        *,
        connection: str,
        uri: str,
        runtime: ToolRuntime,
    ) -> ToolResult:
        return await self._read(
            connection,
            lambda connection_id: self._operations.read_resource(
                claim=self._claim,
                runtime=runtime,
                connection_id=connection_id,
                uri=uri,
            ),
        )

    async def _read(
        self,
        connection: str,
        operation: Callable[[str], Awaitable[ToolResult]],
    ) -> ToolResult:
        connection_id = self._connection_id(connection)
        if connection_id is None:
            return ToolResult.text(
                f"No enabled Connection named {connection!r} is pinned for this Run. Use a "
                "Connection name exactly as the system prompt lists it.",
                is_error=True,
            )
        try:
            return await operation(connection_id)
        except Exception:
            # The port's failures are already safe product copy; never re-raise remote text
            # or foreign diagnostics into the model-visible result.
            return ToolResult.text(
                "The MCP resource request failed. Its outcome may be unknown; do not retry "
                "automatically. Continue with the evidence you have or report the gap.",
                is_error=True,
            )


__all__ = ["RunMcpResourceHost"]
