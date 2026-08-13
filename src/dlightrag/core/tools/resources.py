# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Model-visible ``read_resource`` and ``inspect_resource`` peer tools.

``read_resource`` returns bounded deterministic text; ``inspect_resource``
returns bounded VLM-derived visual evidence. The inspect tool is registered only
when a verified visual capability exists, so a text-only deployment never
advertises a tool it cannot serve. These adapters depend inward on the resource
domain; that domain knows nothing about tools.
"""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from dlightrag.core.resources.models import (
    ResourceReadResult,
    ResourceRegistryError,
    TextWindowLocator,
    VisualHandle,
)
from dlightrag.core.resources.registry import ResourceRegistry
from dlightrag.core.resources.visual import (
    InspectionLocator,
    ResourceInspectionResult,
    ResourceInspector,
)
from dlightrag.core.tools.models import AgentTool, ToolResult

_READ_DESCRIPTION = (
    "Read bounded text (at most 16K tokens) from a registered resource. An "
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


def read_resource_tool(registry: ResourceRegistry) -> AgentTool:
    async def execute(args: BaseModel) -> ToolResult:
        read_args = cast(_ReadResourceArgs, args)
        try:
            result = await registry.read(
                read_args.resource_id, focus=read_args.focus, cursor=read_args.cursor
            )
        except ResourceRegistryError:
            raise
        except Exception as exc:
            raise ResourceRegistryError("resource read failed") from exc
        return ToolResult(
            content=_format_read(result),
            details={
                "resource_id": result.resource_id,
                **registry.evidence_source(result.resource_id),
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
        parts.append(f"[{_describe_text_locator(result.locator)}]")
    parts.append(result.content)
    if result.visual_handles:
        parts.append(f"[visual handles: {_describe_handles(result.visual_handles)}]")
    if result.has_more and result.next_cursor:
        parts.append(f"[more text available; cursor={result.next_cursor}]")
    return "\n".join(parts)


def _describe_text_locator(locator: TextWindowLocator) -> str:
    if locator.char_start is not None:
        return f"lines {locator.start}-{locator.end}, chars {locator.char_start}-{locator.char_end}"
    return f"lines {locator.start}-{locator.end}"


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
