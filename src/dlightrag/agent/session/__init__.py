# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Deep durable Agent Session interfaces.

Imports are lazy so low-level Tool contracts can depend on Session value types
without creating a package-initialization cycle.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlightrag.agent.session.repository import AgentSessionRepository, AgentSessionSnapshot
    from dlightrag.agent.session.runtime import AgentSessionRuntime
    from dlightrag.agent.session.transactions import SessionTransactionPort
    from dlightrag.agent.session.tree import AgentSessionTree

__all__ = [
    "AgentSessionRepository",
    "AgentSessionRuntime",
    "AgentSessionSnapshot",
    "AgentSessionTree",
    "SessionTransactionPort",
]


def __getattr__(name: str) -> Any:
    if name == "AgentSessionRuntime":
        from dlightrag.agent.session.runtime import AgentSessionRuntime

        return AgentSessionRuntime
    if name in {"AgentSessionRepository", "AgentSessionSnapshot"}:
        from dlightrag.agent.session.repository import AgentSessionRepository, AgentSessionSnapshot

        return {
            "AgentSessionRepository": AgentSessionRepository,
            "AgentSessionSnapshot": AgentSessionSnapshot,
        }[name]
    if name == "AgentSessionTree":
        from dlightrag.agent.session.tree import AgentSessionTree

        return AgentSessionTree
    if name == "SessionTransactionPort":
        from dlightrag.agent.session.transactions import SessionTransactionPort

        return SessionTransactionPort
    raise AttributeError(name)
