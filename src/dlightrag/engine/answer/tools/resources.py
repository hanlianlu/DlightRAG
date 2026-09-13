# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-owned text and located-pixel callbacks for Agent read/view tools."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, replace

from dlightrag.engine.agent.tool_content import (
    ToolResourceAttachmentPart,
    ToolTextPart,
    VisualSource,
)
from dlightrag.engine.agent.tools import (
    EvidenceSourceFact,
    ResourceAttachmentBytes,
    ToolEffects,
    ToolResult,
    ToolRuntime,
)
from dlightrag.engine.agent.tools.files import ImagePreparer, ResourceReadRequest, ViewArgs
from dlightrag.engine.answer.resources.converters import ConversionLimitError, UnsafeArchiveError
from dlightrag.engine.answer.resources.formatting import (
    format_resource_read,
    resource_read_continuation,
)
from dlightrag.engine.answer.resources.models import ResourceAdmissionError, TextWindowBudget
from dlightrag.engine.answer.resources.registry import ResourceEffectOwner, ResourceRegistry
from dlightrag.engine.answer.resources.visual import (
    ResourceViewError,
    pdf_page_count,
    render_pdf_page,
)
from dlightrag.engine.public_http import PublicHttpPresentation


def make_resource_reader(registry: ResourceRegistry, text_window_budget: TextWindowBudget):
    async def read_registered(request: ResourceReadRequest, runtime: ToolRuntime) -> ToolResult:
        resource_id = request.resource_id
        if resource_id is None:
            resource_id = registry.register_agent_url(
                request.url or "",
                presentation=PublicHttpPresentation(
                    user_agent=request.user_agent,
                    accept=request.accept,
                    accept_language=request.accept_language,
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
        except UnsafeArchiveError, ConversionLimitError, ResourceAdmissionError, MemoryError:
            return ToolResult.text(
                "extraction_status=safety_refused; no evidence admitted. Do not retry another parser or renderer around the restriction.",
                is_error=True,
                effects=ToolEffects(attached_resources=registry.conversion_effects(resource_id)),
            )
        effects = (
            _evidence_effects(result.resource_id, registry.evidence_source(result.resource_id))
            if result.evidence_available
            else ToolEffects()
        )
        effects = replace(
            effects, attached_resources=registry.conversion_effects(result.resource_id)
        )
        return ToolResult.text(
            format_resource_read(result),
            protected_text=resource_read_continuation(result),
            effects=effects,
        )

    return read_registered


def make_resource_viewer(registry: ResourceRegistry):
    async def view_registered(
        args: ViewArgs, runtime: ToolRuntime, prepare: ImagePreparer
    ) -> ToolResult:
        resource_id = args.resource_id
        if resource_id is None:
            options = args.http
            resource_id = registry.register_agent_url(
                args.url or "",
                presentation=PublicHttpPresentation(
                    user_agent=options.user_agent if options else None,
                    accept=options.accept if options else None,
                    accept_language=options.accept_language if options else None,
                ),
            )
        owner = _effect_owner(runtime)
        target = await registry.visual_target(resource_id, effect_owner=owner)
        resource_id = target.resource_id
        parts = []
        attached = []
        continuation = ""

        async def attach(data: bytes, source: VisualSource, label: str) -> bool:
            prepared = await asyncio.to_thread(prepare, data, label)
            if prepared is None:
                return False
            digest = hashlib.sha256(prepared.data).hexdigest()
            identity = hashlib.sha256(
                (json.dumps(asdict(source), sort_keys=True) + digest).encode()
            ).hexdigest()
            attachment_id = f"res-view-{identity[:32]}"
            parts.extend(
                (
                    ToolTextPart(f"[resource: {resource_id} | {label}]"),
                    ToolResourceAttachmentPart(
                        resource_id=attachment_id,
                        safe_name=label,
                        media_type=prepared.media_type,
                        content_digest=digest,
                        size_bytes=len(prepared.data),
                        data=prepared.data,
                        source=source,
                    ),
                )
            )
            attached.append(
                ResourceAttachmentBytes(
                    resource_id=attachment_id,
                    filename=label,
                    mime_type=prepared.media_type,
                    source_locator=resource_id,
                    content=prepared.data,
                    source=source,
                )
            )
            return True

        if target.kind == "image":
            if args.locator is not None or args.cursor is not None:
                raise ResourceViewError("source image does not accept locator or cursor")
            await attach(target.content, VisualSource(resource_id, "image"), "source image")
        elif target.kind == "pdf":
            count = await asyncio.to_thread(pdf_page_count, target.content)
            if args.locator is not None:
                if not args.locator.isascii() or not args.locator.isdigit():
                    raise ResourceViewError("locator must be a 1-based physical PDF page number")
                page = int(args.locator)
                raw = await asyncio.to_thread(render_pdf_page, target.content, page, overview=False)
                await attach(
                    raw, VisualSource(resource_id, "pdf_page", page=page), f"physical page {page}"
                )
            else:
                start = (
                    registry.resolve_visual_cursor(args.cursor, resource_id, "overview")
                    if args.cursor
                    else 0
                )
                if start >= count:
                    raise ResourceViewError("overview cursor has no remaining physical pages")
                end = start
                for page in range(start + 1, min(count, start + 8) + 1):
                    raw = await asyncio.to_thread(
                        render_pdf_page, target.content, page, overview=True
                    )
                    if not await attach(
                        raw,
                        VisualSource(resource_id, "pdf_page", page=page, overview=True),
                        f"physical page {page} overview",
                    ):
                        break
                    end = page
                if end < count:
                    cursor = registry.visual_cursor(resource_id, end, "overview")
                    continuation = f"More physical pages: view(resource_id={resource_id!r}, cursor={cursor!r})."
                parts.insert(
                    0,
                    ToolTextPart(
                        f"PDF overview covers physical pages {start + 1}-{end} of {count} only. "
                        "Low-resolution thumbnails are not reliable small-text transcription; select locator=<page> for detail."
                    ),
                )
        elif (
            target.kind == "document"
            and args.locator is not None
            and args.locator.startswith("vis-")
        ):
            asset = await registry.visual_asset(resource_id, args.locator, effect_owner=owner)
            source = VisualSource(
                resource_id,
                "embedded_image",
                handle_id=asset.handle_id,
                anchor=asset.anchor,
                origin_part=asset.origin_part,
            )
            await attach(
                asset.data,
                source,
                f"{asset.handle_id}" + (f" @ {asset.anchor}" if asset.anchor else ""),
            )
        else:
            raise ResourceViewError(
                "No directly viewable target. Read this resource for supported embedded-image handles; workspace documents and hosted extraction text cannot be rendered."
            )
        if not attached:
            raise ResourceViewError(
                "image cannot fit the remaining model image budget; no pixels were attached"
            )
        if continuation:
            parts.append(ToolTextPart(continuation))
        evidence = _evidence_effects(resource_id, registry.evidence_source(resource_id))
        return ToolResult(
            parts=tuple(parts),
            protected_text=continuation,
            effects=replace(
                evidence, attached_resources=(*registry.conversion_effects(resource_id), *attached)
            ),
        )

    return view_registered


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
    return ResourceEffectOwner(execution_scope=runtime.execution_scope, intent_id=runtime.intent_id)


__all__ = ["make_resource_reader", "make_resource_viewer"]
