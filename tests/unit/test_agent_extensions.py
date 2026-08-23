# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Trusted extensions have exactly tools, context, and execution seams."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from dlightrag.agent.context import ContextContribution
from dlightrag.agent.environment import TrustExecutionAdapter
from dlightrag.agent.extensions import TrustedExtensions
from dlightrag.agent.tools import AgentTool, ToolResult
from dlightrag.agent.tools.registry import ToolRegistry


class _Args(BaseModel):
    value: str = ""


class _ToolExtension:
    def register_tools(self, registry: ToolRegistry) -> None:
        async def execute(_raw: BaseModel) -> ToolResult:
            return ToolResult(content="ok")

        registry.register(AgentTool("extension_tool", "trusted tool", _Args, execute))


class _ContextExtension:
    def __init__(self, *, citable: bool = False) -> None:
        self._citable = citable

    def context_contributions(self) -> tuple[ContextContribution, ...]:
        return (
            ContextContribution(
                source="extension.context",
                authority="evidence" if self._citable else "reference",
                messages=({"role": "user", "content": "extension"},),
                citable=self._citable,
            ),
        )


class _ExecutionExtension:
    execution_mode = "trust"

    def execution_adapter(self) -> TrustExecutionAdapter:
        return TrustExecutionAdapter()


def test_trusted_extension_set_registers_only_declared_seams(tmp_path: Path) -> None:
    del tmp_path
    extensions = TrustedExtensions(
        tools=(_ToolExtension(),),
        context=(_ContextExtension(),),
        execution=(_ExecutionExtension(),),  # type: ignore[arg-type]
    )
    registry = ToolRegistry()

    extensions.register_tools(registry)

    assert registry.names == ("extension_tool",)
    assert extensions.context_contributions()[0].authority == "reference"
    assert extensions.execution_adapter("trust") is not None
    assert not hasattr(extensions, "lifecycle_hooks")
    assert not hasattr(extensions, "permission_hooks")


def test_extension_context_cannot_bypass_evidence_ledger() -> None:
    extensions = TrustedExtensions(context=(_ContextExtension(citable=True),))

    with pytest.raises(ValueError, match="Evidence ledger"):
        extensions.context_contributions()
