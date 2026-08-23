# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Model-visible ``read`` and ``inspect`` resource branches.

``read`` returns bounded deterministic text; ``inspect``
returns bounded VLM-derived visual evidence. The inspect tool is registered only
when a verified visual capability exists, so a text-only deployment never
advertises a tool it cannot serve. These adapters depend inward on the resource
domain; that domain knows nothing about tools.
"""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from dlightrag.agent.tools import AgentTool, ToolResult
from dlightrag.answer.resources.formatting import (
    format_resource_read,
    resource_read_continuation,
)
from dlightrag.answer.resources.models import (
    ResourceRegistryError,
    TextWindowBudget,
)
from dlightrag.answer.resources.registry import ResourceRegistry
from dlightrag.answer.resources.visual import (
    InspectionLocator,
    ResourceInspectionResult,
    ResourceInspector,
)

_READ_DESCRIPTION = (
    "Read bounded text from a registered resource. An "
    "optional focus reranks the resource's windows so the most relevant one is "
    "returned first; pass the returned cursor to continue with that same focus. The result "
    "includes a structural locator, the text, any inspectable visual handles, and "
    "a continuation cursor when more remains."
)
_INSPECT_DESCRIPTION = (
    "Visually inspect a registered resource for evidence that text extraction cannot give "
    "you: a source image, a PDF page, or an embedded figure by its visual handle. "
    "Provide a focus describing exactly what to look for. Set locator to a PDF page "
    "number or a visual handle id; omit it for a low-resolution PDF overview. Locator "
    "and cursor are mutually exclusive. Use "
    "cursor to page through an overview. The result is VLM-derived evidence tagged "
    "derived_by_vlm with the exact page/sheet/cell/visual locator — treat it as "
    "evidence to cite, never as the final answer."
)


class _ReadResourceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    resource_id: str = Field(min_length=1, description="Identifier of a registered resource.")
    focus: str | None = Field(
        default=None,
        min_length=1,
        description="Optional query to rerank the resource's text windows.",
    )
    cursor: str | None = Field(
        default=None,
        min_length=1,
        description="Continuation cursor returned by a previous read.",
    )


class _InspectResourceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    resource_id: str = Field(min_length=1, description="Identifier of a registered resource.")
    focus: str = Field(min_length=1, description="What to look for in the visual content.")
    locator: str | None = Field(
        default=None,
        min_length=1,
        description="A PDF page number or a visual handle id to target.",
    )
    cursor: str | None = Field(
        default=None,
        min_length=1,
        description="Continuation cursor for a PDF overview.",
    )


def _legacy_resource_read_tool(
    registry: ResourceRegistry,
    text_window_budget: TextWindowBudget,
) -> AgentTool:
    async def execute(args: BaseModel) -> ToolResult:
        read_args = cast(_ReadResourceArgs, args)
        try:
            result = await registry.read(
                read_args.resource_id,
                max_window_tokens=text_window_budget.tokens,
                focus=read_args.focus,
                cursor=read_args.cursor,
            )
        except ResourceRegistryError:
            raise
        except Exception as exc:
            raise ResourceRegistryError("resource read failed") from exc
        return ToolResult(
            content=format_resource_read(result),
            details={
                "resource_id": result.resource_id,
                **registry.evidence_source(result.resource_id),
            },
            protected_suffix=resource_read_continuation(result),
        )

    return AgentTool(
        name="read",
        description=_READ_DESCRIPTION,
        input_model=_ReadResourceArgs,
        execute=execute,
    )


def _inspect_tool(inspector: ResourceInspector) -> AgentTool:
    async def execute(args: BaseModel) -> ToolResult:
        inspect_args = cast(_InspectResourceArgs, args)
        try:
            result = await inspector.inspect(
                inspect_args.resource_id,
                inspect_args.focus,
                locator=inspect_args.locator,
                cursor=inspect_args.cursor,
            )
        except ResourceRegistryError:
            raise
        except Exception as exc:
            raise ResourceRegistryError("visual inspection failed") from exc
        return ToolResult(
            content=_format_inspection(result),
            details={
                "resource_id": result.resource_id,
                **inspector.evidence_source(result.resource_id),
            },
            protected_suffix=_inspection_continuation(result),
        )

    return AgentTool(
        name="inspect",
        description=_INSPECT_DESCRIPTION,
        input_model=_InspectResourceArgs,
        execute=execute,
    )


def build_resource_tools(
    registry: ResourceRegistry,
    *,
    text_window_budget: TextWindowBudget,
    inspector: ResourceInspector | None = None,
    visual_supported: bool = False,
) -> list[AgentTool]:
    """Return the resource peer tools; inspect only for a verified capability."""
    tools: list[AgentTool] = []
    if inspector is not None and visual_supported:
        tools.append(_inspect_tool(inspector))
    return tools


def _format_inspection(result: ResourceInspectionResult) -> str:
    parts = [f"[derived_by_vlm | {_describe_inspection_locator(result.locator)}]", result.content]
    if continuation := _inspection_continuation(result):
        parts.append(continuation)
    return "\n".join(parts)


def _inspection_continuation(result: ResourceInspectionResult) -> str:
    if result.has_more and result.next_cursor:
        return f"[more pages; cursor={result.next_cursor}]"
    return ""


def _describe_inspection_locator(locator: InspectionLocator) -> str:
    if locator.kind == "image":
        return "source image"
    if locator.kind == "pdf_page":
        return f"page {locator.page}"
    if locator.kind == "pdf_overview":
        return f"pages {locator.page_start}-{locator.page_end} overview"
    handle = locator.handle_id or "visual"
    return f"{handle} @ {locator.anchor}" if locator.anchor else handle


def make_resource_reader(
    registry: ResourceRegistry,
    text_window_budget: TextWindowBudget,
):
    """Adapt the registry into the agent-core resource-read callback."""

    async def read_registered(resource_id: str, cursor: str | None) -> ToolResult:
        result = await registry.read(
            resource_id,
            max_window_tokens=text_window_budget.tokens,
            cursor=cursor,
        )
        return ToolResult(
            content=format_resource_read(result),
            details={
                "resource_id": result.resource_id,
                **registry.evidence_source(result.resource_id),
            },
            protected_suffix=resource_read_continuation(result),
        )

    return read_registered


__all__ = ["build_resource_tools", "make_resource_reader"]
