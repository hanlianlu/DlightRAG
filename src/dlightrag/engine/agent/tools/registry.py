# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Deterministic registry for one run's available Agent tools."""

from collections.abc import Iterable

from dlightrag.engine.agent.tools.contracts import AgentTool


class DuplicateToolError(ValueError):
    """Two providers attempted to register the same model-visible tool name."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names
        super().__init__(f"duplicate Agent tool names: {', '.join(names)}")


class ToolRegistry:
    """Own a run-local ordered tool set behind a small registration interface."""

    def __init__(self, tools: Iterable[AgentTool] = ()) -> None:
        self._tools: dict[str, AgentTool] = {}
        self.extend(tools)

    def register(self, tool: AgentTool) -> None:
        if tool.name in self._tools:
            raise DuplicateToolError((tool.name,))
        self._tools[tool.name] = tool

    def extend(self, tools: Iterable[AgentTool]) -> None:
        incoming = tuple(tools)
        names = [tool.name for tool in incoming]
        duplicates = sorted(
            {name for name in names if names.count(name) > 1 or name in self._tools}
        )
        if duplicates:
            raise DuplicateToolError(tuple(duplicates))
        self._tools.update((tool.name, tool) for tool in incoming)

    def resolve(
        self,
        names: Iterable[str] | None = None,
        *,
        exclude: Iterable[str] = (),
    ) -> tuple[AgentTool, ...]:
        excluded = frozenset(exclude)
        if names is None:
            return tuple(tool for name, tool in self._tools.items() if name not in excluded)
        selected: list[AgentTool] = []
        missing: list[str] = []
        for name in names:
            tool = self._tools.get(name)
            if tool is None:
                missing.append(name)
            elif name not in excluded:
                selected.append(tool)
        if missing:
            raise KeyError(f"unknown Agent tools: {', '.join(missing)}")
        return tuple(selected)

    def inherited(
        self,
        *,
        names: Iterable[str] | None = None,
        exclude: Iterable[str] = (),
    ) -> ToolRegistry:
        """Return an immutable-by-copy child registry selection."""
        return ToolRegistry(self.resolve(names, exclude=exclude))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)


__all__ = ["DuplicateToolError", "ToolRegistry"]
