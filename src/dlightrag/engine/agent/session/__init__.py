# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Deep durable Agent Session interfaces.

Imports are lazy so low-level Tool contracts can depend on Session value types
without creating a package-initialization cycle.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlightrag.engine.agent.session.repository import (
        AgentSessionCursor,
        AgentSessionRepository,
        AgentSessionSnapshot,
    )
    from dlightrag.engine.agent.session.runtime import AgentSessionRuntime
    from dlightrag.engine.agent.session.transactions import SessionTransactionPort
    from dlightrag.engine.agent.session.tree import AgentSessionTree

__all__ = [
    "AgentSessionCursor",
    "AgentSessionRepository",
    "AgentSessionRuntime",
    "AgentSessionSnapshot",
    "AgentSessionTree",
    "SessionTransactionPort",
]


def __getattr__(name: str) -> Any:
    if name == "AgentSessionRuntime":
        from dlightrag.engine.agent.session.runtime import AgentSessionRuntime

        return AgentSessionRuntime
    if name in {"AgentSessionCursor", "AgentSessionRepository", "AgentSessionSnapshot"}:
        from dlightrag.engine.agent.session.repository import (
            AgentSessionCursor,
            AgentSessionRepository,
            AgentSessionSnapshot,
        )

        return {
            "AgentSessionCursor": AgentSessionCursor,
            "AgentSessionRepository": AgentSessionRepository,
            "AgentSessionSnapshot": AgentSessionSnapshot,
        }[name]
    if name == "AgentSessionTree":
        from dlightrag.engine.agent.session.tree import AgentSessionTree

        return AgentSessionTree
    if name == "SessionTransactionPort":
        from dlightrag.engine.agent.session.transactions import SessionTransactionPort

        return SessionTransactionPort
    raise AttributeError(name)
