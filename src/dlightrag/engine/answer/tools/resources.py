# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-owned text and located-pixel callbacks for Agent read/view tools."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, replace
from functools import partial

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
from dlightrag.engine.answer.resources.lineage import (
    LineageResourceLoader,
    LineageSnapshotError,
    adopt_lineage_resource,
    lineage_adoption_effects,
)
from dlightrag.engine.answer.resources.models import (
    ResourceAdmissionError,
    ResourceCursorError,
    ResourceNotFoundError,
    TextWindowBudget,
)
from dlightrag.engine.answer.resources.registry import ResourceEffectOwner, ResourceRegistry
from dlightrag.engine.answer.resources.visual import (
    ResourceViewError,
    pdf_page_count,
    render_pdf_page,
)
from dlightrag.engine.public_http import PublicHttpPresentation

logger = logging.getLogger(__name__)


def _run_scoped_handle_refusal(exc: ResourceNotFoundError) -> str:
    """Name the one rule an unusable handle breaks, and the way forward.

    Adoption covers a handle this Session's lineage admits; everything else is a
    caller mistake, and reporting it that way keeps the model from retrying, which
    an unknown internal failure would invite.
    """
    return (
        f"{exc}. Resource ids and cursors belong to the run that registered them, so a "
        "handle from an earlier turn is historical and cannot be read or viewed here. "
        "Re-attach the document, or work from the images already replayed in this context."
    )


def _stale_cursor_refusal(exc: ResourceCursorError) -> str:
    """Cursors are this Run's own view state, never a durable handle."""
    return (
        f"{exc}. A cursor continues this Run's own view of a Resource; read the resource "
        "again for a current continuation."
    )


def _unconverted_refusal(filename: str) -> str:
    """Reading an earlier document whose view this Session never built.

    Converting it now would select a parser and produce a view the earlier Run never
    recorded, so the model is told what is true instead: the pixels are adoptable,
    the text is not.
    """
    return (
        f"The earlier Run never extracted text from {filename}, so this Run will not "
        "convert it again. View its pages for pixels, or re-read it from its URL or a "
        "fresh attachment."
    )


async def _adopt_earlier_then_retry(
    retry: Callable[[], Awaitable[ToolResult]],
    *,
    resource_id: str | None,
    lineage: LineageResourceLoader | None,
    registry: ResourceRegistry,
    refusal: str,
    requires_stored_view: bool = False,
) -> ToolResult:
    """Give one earlier Run's handle the chance to become this Run's Resource.

    The loader owns the lineage rule, so a handle it will not admit keeps the ordinary
    refusal. Adoption effects ride the retried call's own settlement, which is what
    pins the adopted bytes under this Run before the model sees the content. A text
    read additionally requires the earlier Run's own view, because converting the
    document here would record a history that Run never had.
    """
    if lineage is None or not resource_id:
        return ToolResult.text(refusal, is_error=True)
    loaded = await lineage.load(resource_id)
    if loaded is None:
        return ToolResult.text(refusal, is_error=True)
    if requires_stored_view and loaded.conversion_snapshot is None:
        return ToolResult.text(_unconverted_refusal(loaded.filename), is_error=True)
    try:
        adopted = adopt_lineage_resource(registry, loaded)
    except LineageSnapshotError as exc:
        return ToolResult.text(f"{exc}; the document was not converted again.", is_error=True)
    logger.info(
        "Adopted an earlier Run Resource",
        extra={
            "resource_id": adopted.resource_id,
            "origin_resource_id": loaded.resource_id,
            "origin_run_id": loaded.origin_run_id,
            "filename": loaded.filename,
            "source_url": loaded.source_url,
            "reused_conversion_view": adopted.snapshot is not None,
        },
    )
    effects = lineage_adoption_effects(loaded, adopted)
    try:
        result = await retry()
    except ResourceNotFoundError:
        return ToolResult.text(refusal, is_error=True)
    return replace(
        result,
        effects=replace(
            result.effects,
            attached_resources=_settled_once(result.effects.attached_resources, effects),
        ),
    )


def _settled_once(
    first: tuple[ResourceAttachmentBytes, ...], then: tuple[ResourceAttachmentBytes, ...]
) -> tuple[ResourceAttachmentBytes, ...]:
    """Keep one settlement entry per Resource, preferring the tool's own.

    A view of an adopted document names the same stored snapshot the adoption
    pinned, and settling it twice would write the same bytes twice for no gain.
    """
    seen = {effect.resource_id: effect for effect in then}
    for effect in first:
        seen[effect.resource_id] = effect
    return tuple(seen.values())


def make_resource_reader(
    registry: ResourceRegistry,
    text_window_budget: TextWindowBudget,
    *,
    lineage: LineageResourceLoader | None = None,
):
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

    async def read(request: ResourceReadRequest, runtime: ToolRuntime) -> ToolResult:
        try:
            return await read_registered(request, runtime)
        except ResourceNotFoundError as exc:
            return await _adopt_earlier_then_retry(
                partial(read_registered, request, runtime),
                resource_id=request.resource_id,
                lineage=lineage,
                registry=registry,
                refusal=_run_scoped_handle_refusal(exc),
                requires_stored_view=True,
            )
        except ResourceCursorError as exc:
            return ToolResult.text(_stale_cursor_refusal(exc), is_error=True)

    return read


def make_resource_viewer(
    registry: ResourceRegistry, *, lineage: LineageResourceLoader | None = None
):
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

    async def view(args: ViewArgs, runtime: ToolRuntime, prepare: ImagePreparer) -> ToolResult:
        try:
            return await view_registered(args, runtime, prepare)
        except ResourceNotFoundError as exc:
            return await _adopt_earlier_then_retry(
                partial(view_registered, args, runtime, prepare),
                resource_id=args.resource_id,
                lineage=lineage,
                registry=registry,
                refusal=_run_scoped_handle_refusal(exc),
            )
        except ResourceCursorError as exc:
            return ToolResult.text(_stale_cursor_refusal(exc), is_error=True)

    return view


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
