# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Focused visual inspection of answer resources through the VLM role.

Inspection is evidence only: every result is marked ``derived_by_vlm`` and
carries the exact source/page/sheet/cell/visual locator it was taken from, so
the model can cite where a claim came from and never treats a VLM description as
the final answer. Images are bounded through the one canonical image path and
``ImagePayloadBudget``; PDFs are rasterized with pypdfium2 off the event loop as
a bounded low-resolution overview and, on request, one higher-resolution page.
"""

from __future__ import annotations

import asyncio
import base64
import io
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import pypdfium2 as pdfium

from dlightrag.core.answer.images import AnswerImageBudget, AnswerImagePolicy
from dlightrag.core.resources.models import ResourceNotFoundError, ResourceRegistryError
from dlightrag.core.resources.registry import ResourceRegistry

type VLMFunc = Callable[..., Awaitable[Any]]


class ResourceInspectionError(ResourceRegistryError):
    """Raised when a resource cannot be visually inspected."""


@dataclass(frozen=True)
class InspectionLocator:
    """Exact origin of a piece of visual evidence."""

    kind: Literal["image", "pdf_overview", "pdf_page", "visual"]
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    handle_id: str | None = None
    anchor: str | None = None


@dataclass(frozen=True)
class ResourceInspectionResult:
    """VLM-derived visual evidence for one inspection call."""

    resource_id: str
    locator: InspectionLocator
    content: str
    derived_by_vlm: bool = True
    has_more: bool = False
    next_cursor: str | None = None


class ResourceInspector:
    """Turn a registered resource's pixels into bounded, located VLM evidence."""

    def evidence_source(self, resource_id: str) -> dict[str, str]:
        return self._registry.evidence_source(resource_id)

    def __init__(
        self,
        registry: ResourceRegistry,
        *,
        vlm_func: VLMFunc,
        image_policy: AnswerImagePolicy,
        overview_max_px: int = 900,
        overview_scale: int = 1,
        page_scale: int = 2,
        overview_page_limit: int = 8,
    ) -> None:
        self._registry = registry
        self._vlm_func = vlm_func
        self._image_policy = image_policy
        self._overview_max_px = max(1, int(overview_max_px))
        self._overview_scale = max(1, int(overview_scale))
        self._page_scale = max(1, int(page_scale))
        self._overview_page_limit = max(1, int(overview_page_limit))
        self._cursors: dict[str, tuple[str, str | None, int]] = {}

    async def inspect(
        self,
        resource_id: str,
        focus: str,
        *,
        locator: str | None = None,
        cursor: str | None = None,
    ) -> ResourceInspectionResult:
        focus = (focus or "").strip()
        if not focus:
            raise ResourceInspectionError("inspect requires a non-empty focus")
        handle = (locator or "").strip() or None
        cursor = (cursor or "").strip() or None
        if handle is not None and cursor is not None:
            raise ResourceInspectionError("locator and cursor are mutually exclusive")
        if handle and handle.startswith("vis-"):
            return await self._inspect_visual(resource_id, focus, handle)
        if handle is not None and _parse_page(handle) is None:
            raise ResourceInspectionError("locator must be a PDF page number or visual handle")
        target = await self._registry.inspection_target(resource_id)
        if target.kind == "image":
            if handle is not None or cursor is not None:
                raise ResourceInspectionError(
                    "source image inspection does not accept a locator or cursor"
                )
            return await self._inspect_image(resource_id, focus, target.content)
        if target.kind == "pdf":
            return await self._inspect_pdf(resource_id, focus, target.content, handle, cursor)
        raise ResourceInspectionError(
            "resource has no directly inspectable visual content; read it and "
            "inspect one of its visual handles"
        )

    async def _inspect_image(
        self, resource_id: str, focus: str, content: bytes
    ) -> ResourceInspectionResult:
        budget = self._image_policy.new_budget()
        block = await asyncio.to_thread(
            self._bound,
            budget,
            content,
            label=f"{resource_id}:image",
        )
        if block is None:
            raise ResourceInspectionError("source image is too large to inspect")
        text = await self._ask_vlm([block], focus, "image")
        return ResourceInspectionResult(resource_id, InspectionLocator(kind="image"), text)

    async def _inspect_visual(
        self, resource_id: str, focus: str, handle_id: str
    ) -> ResourceInspectionResult:
        try:
            asset = await self._registry.visual_asset(resource_id, handle_id)
        except ResourceNotFoundError as exc:
            raise ResourceInspectionError(str(exc)) from exc
        budget = self._image_policy.new_budget()
        block = await asyncio.to_thread(
            self._bound,
            budget,
            asset.data,
            label=f"{resource_id}:{handle_id}",
        )
        if block is None:
            raise ResourceInspectionError("embedded visual is too large to inspect")
        text = await self._ask_vlm([block], focus, "embedded figure")
        locator = InspectionLocator(kind="visual", handle_id=handle_id, anchor=asset.anchor)
        return ResourceInspectionResult(resource_id, locator, text)

    async def _inspect_pdf(
        self,
        resource_id: str,
        focus: str,
        content: bytes,
        locator: str | None,
        cursor: str | None,
    ) -> ResourceInspectionResult:
        page = _parse_page(locator)
        count = await asyncio.to_thread(_pdf_page_count, content)
        if page is not None:
            if page < 1 or page > count:
                raise ResourceInspectionError(f"page {page} is out of range (1-{count})")
            raw = await asyncio.to_thread(_render_pdf_page, content, page - 1, self._page_scale)
            budget = self._image_policy.new_budget()
            block = await asyncio.to_thread(
                self._bound,
                budget,
                raw,
                label=f"{resource_id}:p{page}",
            )
            if block is None:
                raise ResourceInspectionError("rendered page is too large to inspect")
            text = await self._ask_vlm([block], focus, f"page {page}")
            return ResourceInspectionResult(
                resource_id, InspectionLocator(kind="pdf_page", page=page), text
            )

        start = self._resolve_cursor(cursor, resource_id, focus) if cursor is not None else 0
        end = min(count, start + self._overview_page_limit)
        raws = await asyncio.to_thread(
            _render_pdf_pages, content, list(range(start, end)), self._overview_scale
        )
        budget = self._image_policy.new_budget(max_px=self._overview_max_px)
        blocks: list[dict[str, Any]] = []
        for offset, raw in enumerate(raws):
            block = await asyncio.to_thread(
                self._bound,
                budget,
                raw,
                label=f"{resource_id}:overview{start + offset + 1}",
            )
            if block is None:
                break
            blocks.append(block)
        if not blocks:
            raise ResourceInspectionError("no page could be rendered for the overview")
        actual_end = start + len(blocks)
        text = await self._ask_vlm(
            blocks,
            focus,
            f"pages {start + 1}-{actual_end} of a {count}-page document",
        )
        has_more = actual_end < count
        next_cursor = self._mint_cursor(resource_id, focus, actual_end) if has_more else None
        return ResourceInspectionResult(
            resource_id,
            InspectionLocator(kind="pdf_overview", page_start=start + 1, page_end=actual_end),
            text,
            has_more=has_more,
            next_cursor=next_cursor,
        )

    def _bound(
        self, budget: AnswerImageBudget, data: bytes, *, label: str
    ) -> dict[str, Any] | None:
        return budget.add_base64(base64.b64encode(data).decode("ascii"), label=label)

    async def _ask_vlm(self, blocks: list[dict[str, Any]], focus: str, subject: str) -> str:
        content: list[dict[str, Any]] = [*blocks]
        content.append({"type": "text", "text": _prompt(focus, subject)})
        try:
            response = await self._vlm_func(messages=[{"role": "user", "content": content}])
        except Exception as exc:  # noqa: BLE001 - any VLM failure is one tool error
            raise ResourceInspectionError("visual inspection failed") from exc
        text = response.strip() if isinstance(response, str) else str(response).strip()
        if not text:
            raise ResourceInspectionError("visual inspection returned no content")
        return text

    def _mint_cursor(self, resource_id: str, focus: str, next_start: int) -> str:
        token = secrets.token_urlsafe(18)
        self._cursors[token] = (resource_id, focus, next_start)
        return token

    def _resolve_cursor(self, cursor: str, resource_id: str, focus: str) -> int:
        state = self._cursors.get(cursor)
        if state is None or state[0] != resource_id or state[1] != focus:
            raise ResourceInspectionError("cursor is not valid for this inspection")
        return state[2]


def _prompt(focus: str, subject: str) -> str:
    return (
        f"Inspect this {subject} as visual evidence only. Focus: {focus}. "
        "Report exactly what is visible relevant to the focus — transcribe any "
        "text, numbers, and chart or table structure verbatim. Do not speculate "
        "beyond the pixels and do not answer the user's overall question."
    )


def _parse_page(locator: str | None) -> int | None:
    if not locator:
        return None
    text = locator.strip().lower()
    for prefix in ("page", "p", "#"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    return int(text) if text.isdigit() else None


def _pdf_page_count(data: bytes) -> int:
    pdf = pdfium.PdfDocument(data)
    try:
        return len(pdf)
    finally:
        pdf.close()


def _render_pdf_page(data: bytes, index: int, scale: int) -> bytes:
    pdf = pdfium.PdfDocument(data)
    try:
        return _png_bytes(pdf[index].render(scale=scale).to_pil())
    finally:
        pdf.close()


def _render_pdf_pages(data: bytes, indices: list[int], scale: int) -> list[bytes]:
    pdf = pdfium.PdfDocument(data)
    try:
        return [_png_bytes(pdf[index].render(scale=scale).to_pil()) for index in indices]
    finally:
        pdf.close()


def _png_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


__all__ = [
    "InspectionLocator",
    "ResourceInspectionError",
    "ResourceInspectionResult",
    "ResourceInspector",
]
