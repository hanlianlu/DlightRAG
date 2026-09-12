# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner-managed personal MCP Connections; no transport or database imports."""

from typing import TYPE_CHECKING, Any

from .policy import ConnectionPolicy

if TYPE_CHECKING:
    from .models import (
        AuthorizationStart,
        BoundResearchConnections,
        CatalogueTool,
        ConnectionCommand,
        ConnectionsError,
        ConnectionsStore,
        ConnectionsView,
        ConnectionView,
        McpClientPort,
    )
    from .service import Connections

__all__ = [
    "AuthorizationStart",
    "BoundResearchConnections",
    "CatalogueTool",
    "ConnectionCommand",
    "ConnectionPolicy",
    "Connections",
    "ConnectionsError",
    "ConnectionsStore",
    "ConnectionsView",
    "ConnectionView",
    "McpClientPort",
]


def __getattr__(name: str) -> Any:
    # Config consumes policy without eagerly loading Answer execution contracts.
    if name == "Connections":
        from .service import Connections

        return Connections
    if name in __all__ and name != "ConnectionPolicy":
        from . import models

        return getattr(models, name)
    raise AttributeError(name)
