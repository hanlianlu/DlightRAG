# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Test fixture over the production in-memory Agent Session adapter."""

from dataclasses import dataclass

from dlightrag.engine.agent.session.memory import MemoryAgentSessionRepository


@dataclass(frozen=True, slots=True)
class NoHostUpdate:
    """Session settlement mutates no external Host state."""


class InMemoryAgentSessionRepository(MemoryAgentSessionRepository[NoHostUpdate]):
    """Compatibility test name while Answer tests migrate to the public adapter."""

    def __init__(self, *, initial_version: int = 0) -> None:
        if initial_version not in {0}:
            raise ValueError("clean-break Memory Session Store starts at sequence zero")
        super().__init__()


__all__ = ["InMemoryAgentSessionRepository", "NoHostUpdate"]
