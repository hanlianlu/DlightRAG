# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The complete trusted Python extension surface for Agent 3.0.

Extensions are host-loaded code and therefore already trusted. They may only
register tools, contribute model context, or provide an execution adapter.
There are deliberately no lifecycle, authorization, approval, or transport
hooks here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from dlightrag.agent.context import ContextContribution
from dlightrag.agent.environment.execution import ExecutionEnvironmentAdapter, ExecutionMode
from dlightrag.agent.tools.registry import ToolRegistry


class ToolExtension(Protocol):
    """Register model-visible tools into one run-local registry."""

    def register_tools(self, registry: ToolRegistry) -> None: ...


class ContextExtension(Protocol):
    """Return immutable context contributions for one projection."""

    def context_contributions(self) -> tuple[ContextContribution, ...]: ...


class ExecutionExtension(Protocol):
    """Provide an adapter for exactly one configured execution mode."""

    @property
    def execution_mode(self) -> ExecutionMode: ...

    def execution_adapter(self) -> ExecutionEnvironmentAdapter: ...


@dataclass(frozen=True, slots=True)
class TrustedExtensions:
    """Host-composed extension set with no implicit package discovery."""

    tools: tuple[ToolExtension, ...] = ()
    context: tuple[ContextExtension, ...] = ()
    execution: tuple[ExecutionExtension, ...] = ()

    def register_tools(self, registry: ToolRegistry) -> None:
        for extension in self.tools:
            extension.register_tools(registry)

    def context_contributions(self) -> tuple[ContextContribution, ...]:
        contributions = tuple(
            contribution
            for extension in self.context
            for contribution in extension.context_contributions()
        )
        if any(contribution.citable for contribution in contributions):
            raise ValueError("extension context cannot bypass the host Evidence ledger")
        return contributions

    def execution_adapter(self, mode: ExecutionMode) -> ExecutionEnvironmentAdapter | None:
        matches = [
            extension.execution_adapter()
            for extension in self.execution
            if extension.execution_mode == mode
        ]
        if len(matches) > 1:
            raise ValueError(f"multiple trusted execution adapters configured for {mode}")
        return matches[0] if matches else None


__all__ = [
    "ContextExtension",
    "ExecutionExtension",
    "ToolExtension",
    "TrustedExtensions",
]
