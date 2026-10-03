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
from dlightrag.engine.answer.agent_browser import AgentBrowserError
from dlightrag.engine.answer.resources.converters import (
    ConversionLimitError,
    UnsafeArchiveError,
    conversion_format,
)
from dlightrag.engine.answer.resources.formatting import (
    format_resource_read,
    resource_read_continuation,
)
from dlightrag.engine.answer.resources.lineage import (
    LineageAdoptionConflict,
    LineageResourceLoader,
    LineageSnapshotError,
    adopt_lineage_resource,
)
from dlightrag.engine.answer.resources.models import (
    RenderedReadTargetError,
    ResourceAdmissionError,
    ResourceCursorError,
    ResourceNotConvertedError,
    ResourceNotFoundError,
    ResourceRegistryError,
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
        f"{exc}. This run neither holds that handle nor can adopt it from an earlier "
        "turn of this conversation, so it cannot be read or viewed here. Re-attach the "
        "document, or work from the images already replayed in this context."
    )


def _stale_cursor_refusal(exc: ResourceCursorError) -> str:
    """Cursors are this Run's own view state, never a durable handle."""
    return (
        f"{exc}. A cursor continues this Run's own view of a Resource; call read or view "
        "on the resource again for a current continuation."
    )


def _unconverted_refusal(filename: str, media_type: str | None) -> str:
    """Reading an earlier document whose view this Session never built.

    Converting it now would select a parser and produce a view the earlier Run never
    recorded, so the model is told what is true instead: a PDF's pages are adoptable
    as pixels, and no format's text is.
    """
    pages = conversion_format(filename, media_type) == "pdf"
    return (
        f"The earlier Run never extracted text from {filename}, and this Run does not "
        "build a text view the earlier Run never had. "
        + ("View its pages for pixels, or re-read" if pages else "Re-read")
        + " it from its URL or a fresh attachment."
    )


async def _adopt_earlier_then_retry(
    retry: Callable[[], Awaitable[ToolResult]],
    *,
    resource_id: str | None,
    lineage: LineageResourceLoader | None,
    registry: ResourceRegistry,
    refusal: str,
    runtime: ToolRuntime,
    needs_text: bool = False,
) -> ToolResult:
    """Give one earlier Run's handle the chance to become this Run's Resource.

    The loader owns the lineage rule, so a handle it will not admit keeps the ordinary
    refusal. The adoption is recorded under this Run's fence before the handle
    resolves, so it holds whatever the retried call does next, and that call's own
    result carries nothing on its behalf. It spends this Run's attachment allowance,
    so it waits for the calls before it in the batch.

    A read of a *convertible* resource requires a stored view, the earlier Run's or one
    this Run already holds for the same bytes, because converting it here would record
    a parse history that Run never had. A resource with no conversion route — a
    published Markdown report, a fetched text page — is read by decoding the adopted
    bytes, so demanding a view for it would refuse the very read the handle teaches.
    """
    if lineage is None or not resource_id:
        return ToolResult.text(refusal, is_error=True)
    loaded = await lineage.load(resource_id)
    if loaded is None:
        return ToolResult.text(refusal, is_error=True)
    await runtime.in_source_order()
    try:
        adopted = await adopt_lineage_resource(
            registry,
            loaded,
            record=partial(lineage.record, owner=_effect_owner(runtime)),
            needs_text=needs_text,
        )
    except LineageSnapshotError as exc:
        return ToolResult.text(f"{exc}; the document was not converted again.", is_error=True)
    except ResourceNotConvertedError as exc:
        return ToolResult.text(_unconverted_refusal(exc.filename, exc.media_type), is_error=True)
    except (ResourceAdmissionError, LineageAdoptionConflict) as exc:
        # Adoption spends this Run's own attachment allowance, so a spent allowance
        # refuses the earlier document the way it refuses one more attachment; the
        # store refuses a second view of a Resource the same way.
        return ToolResult.text(
            f"{exc}; the earlier document was not adopted into this run.", is_error=True
        )
    logger.info(
        "Adopted an earlier Run Resource",
        extra={
            "resource_id": adopted,
            "origin_resource_id": loaded.resource_id,
            "origin_run_id": loaded.origin_run_id,
            "resource_filename": loaded.filename,
            "source_url": loaded.source_url,
            "reused_conversion_view": registry.has_conversion_snapshot(adopted),
        },
    )
    try:
        return await retry()
    except ResourceNotFoundError as exc:
        # The handle is held now, so what the retried call did not find is inside
        # it, such as an embedded-image handle.
        return ToolResult.text(
            f"{exc}. Read the resource again for the handles it holds.", is_error=True
        )
    except ResourceNotConvertedError as exc:
        return ToolResult.text(_unconverted_refusal(exc.filename, exc.media_type), is_error=True)
    except ResourceCursorError as exc:
        return ToolResult.text(_stale_cursor_refusal(exc), is_error=True)
    except ResourceRegistryError as exc:
        return ToolResult.text(str(exc), is_error=True)


def make_resource_reader(
    registry: ResourceRegistry,
    max_window_tokens: int,
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
        elif registry.loads_on_read(resource_id):
            await runtime.in_source_order()
        try:
            result = await registry.read(
                resource_id,
                max_window_tokens=max_window_tokens,
                focus=request.focus,
                cursor=request.cursor,
                rendered=request.rendered,
                effect_owner=_effect_owner(runtime),
            )
        except UnsafeArchiveError, ConversionLimitError, ResourceAdmissionError, MemoryError:
            return ToolResult.text(
                "extraction_status=safety_refused; no evidence admitted. Do not retry another parser or renderer around the restriction.",
                is_error=True,
                effects=ToolEffects(
                    attached_resources=(
                        *registry.conversion_effects(resource_id),
                        *registry.rendered_effects(resource_id),
                    )
                ),
            )
        effects = (
            _evidence_effects(
                result.resource_id,
                registry.evidence_source(result.resource_id, text=True, rendered=result.rendered),
            )
            if result.evidence_available
            else ToolEffects()
        )
        # A rendering settles with the read that returned it, so recovery restores it.
        effects = replace(
            effects,
            attached_resources=(
                *registry.conversion_effects(result.resource_id),
                *(registry.rendered_effects(result.resource_id) if result.rendered else ()),
            ),
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
                runtime=runtime,
                needs_text=True,
            )
        except ResourceCursorError as exc:
            return ToolResult.text(_stale_cursor_refusal(exc), is_error=True)
        except ResourceNotConvertedError as exc:
            return ToolResult.text(
                _unconverted_refusal(exc.filename, exc.media_type), is_error=True
            )
        except RenderedReadTargetError as exc:
            return ToolResult.text(str(exc), is_error=True)
        except AgentBrowserError as exc:
            # A render the browser could not give leaves nothing admitted.
            return ToolResult.text(exc.public_message, is_error=True)

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
        elif registry.loads_on_read(resource_id):
            await runtime.in_source_order()
        owner = _effect_owner(runtime)
        held = (
            registry.held_visual_asset(resource_id, args.locator)
            if args.locator is not None and args.locator.startswith("vis-")
            else None
        )
        if held is None:
            target = await registry.visual_target(resource_id, effect_owner=owner)
            resource_id = target.resource_id
            held_rendering = False
        else:
            # The Run already holds this image, possibly from a rendering it never fetched.
            target = None
            resource_id = registry.canonical_resource_id(resource_id)
            held_rendering = held[1]
        provenance = registry.evidence_source(resource_id, rendered=held_rendering)
        # Each label names the document too: a later turn, or a Fast follow-up that
        # sees the image without this call, has no manifest that maps the id to it.
        name = provenance["title"]
        parts = []
        attached = []
        continuation = ""

        async def attach(data: bytes, source: VisualSource, label: str) -> bool:
            # Preparing spends the Run's shared image budget.
            await runtime.in_source_order()
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

        if held is not None:
            asset = held[0]
            await attach(
                asset.data,
                VisualSource(
                    resource_id,
                    "embedded_image",
                    handle_id=asset.handle_id,
                    anchor=asset.anchor,
                    origin_part=asset.origin_part,
                ),
                f"{name}, {asset.handle_id}" + (f" @ {asset.anchor}" if asset.anchor else ""),
            )
        elif target is not None and target.kind == "image":
            if args.locator is not None or args.cursor is not None:
                raise ResourceViewError("source image does not accept locator or cursor")
            await attach(target.content, VisualSource(resource_id, "image"), name)
        elif target is not None and target.kind == "pdf":
            count = await asyncio.to_thread(pdf_page_count, target.content)
            if args.locator is not None:
                if not args.locator.isascii() or not args.locator.isdigit():
                    raise ResourceViewError("locator must be a 1-based physical PDF page number")
                page = int(args.locator)
                raw = await asyncio.to_thread(render_pdf_page, target.content, page, overview=False)
                await attach(
                    raw,
                    VisualSource(resource_id, "pdf_page", page=page),
                    f"{name}, physical page {page}",
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
                        f"{name}, physical page {page} overview",
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
            target is not None
            and target.kind == "document"
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
                f"{name}, {asset.handle_id}" + (f" @ {asset.anchor}" if asset.anchor else ""),
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
        evidence = _evidence_effects(resource_id, provenance)
        return ToolResult(
            parts=tuple(parts),
            protected_text=continuation,
            effects=replace(
                evidence,
                attached_resources=(
                    *registry.conversion_effects(resource_id),
                    *(registry.rendered_effects(resource_id) if held_rendering else ()),
                    *attached,
                ),
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
                runtime=runtime,
            )
        except ResourceCursorError as exc:
            return ToolResult.text(_stale_cursor_refusal(exc), is_error=True)
        except ResourceNotConvertedError as exc:
            return ToolResult.text(
                _unconverted_refusal(exc.filename, exc.media_type), is_error=True
            )
        except ResourceViewError as exc:
            return ToolResult.text(str(exc), is_error=True)

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
