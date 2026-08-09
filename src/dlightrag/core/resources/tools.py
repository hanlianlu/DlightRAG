# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Model-visible ``read_resource`` and ``inspect_resource`` peer tools.

``read_resource`` returns bounded deterministic text; ``inspect_resource``
returns bounded VLM-derived visual evidence. The inspect tool is registered only
when a verified visual capability exists, so a text-only deployment never
advertises a tool it cannot serve.
"""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from dlightrag.core.agent.tool_loop import AgentTool, ToolResult
from dlightrag.core.resources.models import ResourceReadResult, TextWindowLocator, VisualHandle
from dlightrag.core.resources.registry import ResourceRegistry
from dlightrag.core.resources.visual import (
    InspectionLocator,
    ResourceInspectionResult,
    ResourceInspector,
)

_READ_DESCRIPTION = (
    "Read bounded text (at most 16K tokens) from a registered attachment. An "
    "optional focus reranks the attachment's windows so the most relevant one is "
    "returned first; pass the returned cursor to continue reading. The result "
    "includes a structural locator, the text, any inspectable visual handles, and "
    "a continuation cursor when more remains."
)
_INSPECT_DESCRIPTION = (
    "Visually inspect an attachment for evidence that text extraction cannot give "
    "you: a source image, a PDF page, or an embedded figure by its visual handle. "
    "Provide a focus describing exactly what to look for. Set locator to a PDF page "
    "number or a visual handle id; omit it for a low-resolution PDF overview. Use "
    "cursor to page through an overview. The result is VLM-derived evidence tagged "
    "derived_by_vlm with the exact page/sheet/cell/visual locator — treat it as "
    "evidence to cite, never as the final answer."
)


class _ReadResourceArgs(BaseModel):
    resource_id: str = Field(description="Identifier of a registered attachment.")
    focus: str | None = Field(
        default=None, description="Optional query to rerank the attachment's text windows."
    )
    cursor: str | None = Field(
        default=None, description="Continuation cursor returned by a previous read."
    )


class _InspectResourceArgs(BaseModel):
    resource_id: str = Field(description="Identifier of a registered attachment.")
    focus: str = Field(description="What to look for in the visual content.")
    locator: str | None = Field(
        default=None, description="A PDF page number or a visual handle id to target."
    )
    cursor: str | None = Field(default=None, description="Continuation cursor for a PDF overview.")


def read_resource_tool(registry: ResourceRegistry) -> AgentTool:
    async def execute(args: BaseModel) -> ToolResult:
        read_args = cast(_ReadResourceArgs, args)
        result = await registry.read(
            read_args.resource_id, focus=read_args.focus, cursor=read_args.cursor
        )
        return ToolResult(
            content=_format_read(result),
            details={
                "resource_id": result.resource_id,
                "has_more": result.has_more,
                "next_cursor": result.next_cursor,
            },
        )

    return AgentTool(
        name="read_resource",
        description=_READ_DESCRIPTION,
        input_model=_ReadResourceArgs,
        execute=execute,
    )


def inspect_resource_tool(inspector: ResourceInspector) -> AgentTool:
    async def execute(args: BaseModel) -> ToolResult:
        inspect_args = cast(_InspectResourceArgs, args)
        result = await inspector.inspect(
            inspect_args.resource_id,
            inspect_args.focus,
            locator=inspect_args.locator,
            cursor=inspect_args.cursor,
        )
        return ToolResult(
            content=_format_inspection(result),
            details={
                "resource_id": result.resource_id,
                "derived_by_vlm": result.derived_by_vlm,
                "has_more": result.has_more,
                "next_cursor": result.next_cursor,
            },
        )

    return AgentTool(
        name="inspect_resource",
        description=_INSPECT_DESCRIPTION,
        input_model=_InspectResourceArgs,
        execute=execute,
    )


def build_resource_tools(
    registry: ResourceRegistry,
    *,
    inspector: ResourceInspector | None = None,
    visual_supported: bool = False,
) -> list[AgentTool]:
    """Return the resource peer tools; inspect only for a verified capability."""
    tools = [read_resource_tool(registry)]
    if inspector is not None and visual_supported:
        tools.append(inspect_resource_tool(inspector))
    return tools


def _format_read(result: ResourceReadResult) -> str:
    parts: list[str] = []
    if result.locator is not None:
        parts.append(_describe_text_locator(result.locator))
    parts.append(result.content)
    if result.visual_handles:
        parts.append(f"[visual handles: {_describe_handles(result.visual_handles)}]")
    if result.has_more and result.next_cursor:
        parts.append(f"[more text available; cursor={result.next_cursor}]")
    return "\n".join(parts)


def _describe_text_locator(locator: TextWindowLocator) -> str:
    if locator.char_start is not None:
        return (
            f"[lines {locator.start}-{locator.end}, chars {locator.char_start}-{locator.char_end}]"
        )
    return f"[lines {locator.start}-{locator.end}]"


def _describe_handles(handles: tuple[VisualHandle, ...]) -> str:
    return ", ".join(
        handle.handle_id + (f" ({handle.label})" if handle.label else "") for handle in handles
    )


def _format_inspection(result: ResourceInspectionResult) -> str:
    parts = [f"[derived_by_vlm | {_describe_inspection_locator(result.locator)}]", result.content]
    if result.has_more and result.next_cursor:
        parts.append(f"[more pages; cursor={result.next_cursor}]")
    return "\n".join(parts)


def _describe_inspection_locator(locator: InspectionLocator) -> str:
    if locator.kind == "image":
        return "source image"
    if locator.kind == "pdf_page":
        return f"page {locator.page}"
    if locator.kind == "pdf_overview":
        return f"pages {locator.page_start}-{locator.page_end} overview"
    handle = locator.handle_id or "visual"
    return f"{handle} @ {locator.anchor}" if locator.anchor else handle


__all__ = [
    "build_resource_tools",
    "inspect_resource_tool",
    "read_resource_tool",
]
