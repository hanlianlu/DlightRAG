# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Model-visible ``read`` and ``inspect`` resource branches.

``read`` returns bounded deterministic text; ``inspect``
returns bounded VLM-derived visual evidence. The inspect tool is registered only
when a verified visual capability exists, so a text-only deployment never
advertises a tool it cannot serve. These adapters depend inward on the resource
domain; that domain knows nothing about tools.
"""

from __future__ import annotations

import hashlib
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from dlightrag.engine.agent.tool_content import ToolResourceAttachmentPart, ToolTextPart
from dlightrag.engine.agent.tools import (
    AgentTool,
    EvidenceSourceFact,
    ResourceAttachmentBytes,
    ToolEffects,
    ToolResult,
    ToolRuntime,
)
from dlightrag.engine.agent.tools.files import ResourceReadRequest
from dlightrag.engine.answer.resources.formatting import (
    format_resource_read,
    resource_read_continuation,
)
from dlightrag.engine.answer.resources.models import (
    ResourceRegistryError,
    TextWindowBudget,
)
from dlightrag.engine.answer.resources.registry import ResourceEffectOwner, ResourceRegistry
from dlightrag.engine.answer.resources.visual import (
    InspectionLocator,
    ResourceInspectionResult,
    ResourceInspector,
)
from dlightrag.engine.public_http import PublicHttpPresentation

_INSPECT_DESCRIPTION = (
    "Visually inspect one run-scoped Resource whose id starts with res-. "
    "Never pass a filename, source_uri, or local:// corpus locator. "
    "focus says what to look for. locator is a PDF page or visual handle; omit it "
    "for a PDF overview. locator and cursor are mutually exclusive. "
    "The result is VLM evidence tagged derived_by_vlm — cite it, do not treat it as "
    "the final answer."
)


class _InspectResourceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    resource_id: str = Field(
        min_length=1,
        description="This turn's res- id from the registered list. Not a path or local:// URI.",
    )
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


def _inspect_tool(inspector: ResourceInspector) -> AgentTool:
    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        inspect_args = cast(_InspectResourceArgs, args)
        try:
            result = await inspector.inspect(
                inspect_args.resource_id,
                inspect_args.focus,
                locator=inspect_args.locator,
                cursor=inspect_args.cursor,
                effect_owner=_effect_owner(runtime),
            )
        except ResourceRegistryError:
            raise
        except Exception as exc:
            raise ResourceRegistryError("visual inspection failed") from exc
        return ToolResult.text(
            _format_inspection(result),
            protected_text=_inspection_continuation(result),
            effects=_evidence_effects(
                result.resource_id,
                inspector.evidence_source(result.resource_id),
            ),
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

    async def read_registered(
        request: ResourceReadRequest,
        runtime: ToolRuntime,
    ) -> ToolResult:
        resource_id = request.resource_id
        if resource_id is None:
            if request.url is None:  # pragma: no cover - Agent Core validates this
                raise ResourceRegistryError("resource read target is missing")
            try:
                resource_id = registry.register_agent_url(
                    request.url,
                    presentation=PublicHttpPresentation(
                        user_agent=request.user_agent,
                        accept=request.accept,
                        accept_language=request.accept_language,
                    ),
                )
            except Exception as exc:
                raise ResourceRegistryError("resource read failed") from exc
        target = None
        if request.cursor is None and registry.is_declared_image(resource_id):
            try:
                target = await registry.inspection_target(
                    resource_id,
                    effect_owner=_effect_owner(runtime),
                )
            except Exception:
                # Let the text read path apply direct-HTTP -> Extract fallback and
                # return its bounded no-evidence summary when acquisition fails.
                target = None
        if target is not None and target.kind == "image":
            resource_id = registry.canonical_resource_id(resource_id)
            # A verified original image snapshot goes straight to the query
            # model as an attachment instead of a text window.
            digest = hashlib.sha256(target.content).hexdigest()
            attachment = ToolResourceAttachmentPart(
                resource_id=resource_id,
                safe_name=_registry_filename(registry, resource_id),
                media_type=target.media_type or "image/png",
                content_digest=digest,
                size_bytes=len(target.content),
                data=target.content,
            )
            evidence = _evidence_effects(resource_id, registry.evidence_source(resource_id))
            return ToolResult(
                parts=(
                    ToolTextPart(
                        f"image attachment: {attachment.safe_name} "
                        f"({attachment.media_type}, {attachment.size_bytes} bytes); "
                        "the original snapshot is attached to this message"
                    ),
                    attachment,
                ),
                effects=ToolEffects(
                    evidence_sources=evidence.evidence_sources,
                    attached_resources=(
                        ResourceAttachmentBytes(
                            resource_id=resource_id,
                            filename=_registry_filename(registry, resource_id),
                            mime_type=attachment.media_type,
                            source_locator=resource_id,
                            content=target.content,
                        ),
                    ),
                ),
            )
        try:
            result = await registry.read(
                resource_id,
                max_window_tokens=text_window_budget.tokens,
                focus=request.focus,
                cursor=request.cursor,
                effect_owner=_effect_owner(runtime),
            )
        except ResourceRegistryError:
            raise
        except Exception as exc:
            raise ResourceRegistryError("resource read failed") from exc
        effects = (
            _evidence_effects(result.resource_id, registry.evidence_source(result.resource_id))
            if result.evidence_available
            else ToolEffects()
        )
        return ToolResult.text(
            format_resource_read(result),
            protected_text=resource_read_continuation(result),
            effects=effects,
        )

    return read_registered


def _registry_filename(registry: ResourceRegistry, resource_id: str) -> str:
    source = registry.evidence_source(resource_id)
    return source.get("title") or "image"


def _evidence_effects(resource_id: str, source: dict[str, str]) -> ToolEffects:
    return ToolEffects(
        evidence_sources=(
            EvidenceSourceFact(
                resource_id=resource_id,
                source_type=source.get("source_type", "unknown"),
                source_uri=source.get("source_uri", resource_id),
                title=source.get("title", resource_id),
                attributes=tuple(
                    (name, source[name])
                    for name in ("resource_kind", "admission_origin", "acquisition")
                    if source.get(name)
                ),
            ),
        )
    )


def _effect_owner(runtime: ToolRuntime) -> ResourceEffectOwner:
    return ResourceEffectOwner(
        execution_scope=runtime.execution_scope,
        intent_id=runtime.intent_id,
    )


__all__ = ["build_resource_tools", "make_resource_reader"]
