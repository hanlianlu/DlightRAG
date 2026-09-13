# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Deterministic binary resource conversion with OOXML preflight.

Only HTML, CSV, PDF, DOCX, PPTX, and XLSX are admitted. PDF, DOCX and XLSX text
use firecrawl-anydoc 0.2.4 with local OCR rejection. MarkItDown runs with
plugins disabled and never fetches the network: DlightRAG hands it admitted bytes
and explicit :class:`StreamInfo`. A fresh converter is built per call so no
mutable state is shared between concurrent conversions. OOXML archives pass a
central-directory size preflight before any converter opens them, so a zip bomb
is rejected without decompressing attack-sized data. DOCX images come from typed asset occurrences independently of Markdown;
incumbent Office images are replaced with compact handles;
XLSX images are pulled from the workbook with their sheet/cell anchor.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import time
import zipfile
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Event
from types import ModuleType

import openpyxl
from markdown_it import MarkdownIt
from markitdown import MarkItDown, StreamInfo
from openpyxl.utils import get_column_letter

from dlightrag.engine.answer.resources.docx_assets import (
    AssetBindingError,
    bind_docx_fallback_images,
    docx_asset_occurrences,
)
from dlightrag.engine.answer.resources.models import ResourceRegistryError

# Physical archive-safety limits. These are internal decompression bounds, not
# the public attachment-size quotas, so an OOXML file that is admissible by byte
# size can still be rejected here if its internal expansion looks like a bomb.
_MAX_OOXML_ENTRIES = 10_000
_MAX_OOXML_ENTRY_BYTES = 100 * 1024 * 1024
_MAX_OOXML_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_OOXML_EXPANSION_RATIO = 100
# Shared across preflight, candidate and (at most one) fallback.
# This is an adoption/start-work deadline, not a native thread kill mechanism.
_MAX_CONVERSION_SECONDS = 120.0
_ANYDOC_VERSION = "0.2.4"

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class ResourceConversionError(ResourceRegistryError):
    """Raised when an admitted resource cannot be converted deterministically."""

    def __init__(
        self,
        message: str,
        *,
        converter: str = "markitdown",
        converter_version: str | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.converter = converter
        self.converter_version = converter_version or version(
            "dlightrag" if converter == "resource-host" else converter
        )
        self.fallback_reason = fallback_reason


class ConversionLimitError(ResourceConversionError):
    """Terminal conversion resource/deadline/cancellation refusal; never retry."""


class UnsafeArchiveError(ResourceConversionError):
    """Raised when an OOXML archive exceeds a physical decompression limit."""

    def __init__(self, message: str) -> None:
        super().__init__(message, converter="resource-host")


@dataclass(frozen=True)
class ExtractedVisual:
    """An image pulled out of a converted resource with its anchor and bytes."""

    handle_id: str
    anchor: str | None
    media_type: str
    data: bytes
    origin_part: str | None = None


@dataclass(frozen=True)
class ConvertedResource:
    """Deterministic text view plus the visuals extracted from one resource."""

    text: str
    visuals: tuple[ExtractedVisual, ...]
    extraction_status: str = "usable_text_unverified_coverage"
    converter: str = "markitdown"
    converter_version: str = version("markitdown")
    fallback_reason: str | None = None
    known_ocr_pages: tuple[int, ...] = ()
    known_page_count: int | None = None
    note: str | None = None


@dataclass(frozen=True)
class _ConversionBudget:
    deadline: float
    cancelled: Event

    def check(self, *, converter: str = "resource-host") -> None:
        if self.cancelled.is_set() or time.monotonic() >= self.deadline:
            raise ConversionLimitError("conversion total budget exhausted", converter=converter)


@dataclass(frozen=True)
class _Route:
    extension: str
    mimetype: str
    is_ooxml: bool
    is_xlsx: bool


_ROUTES: dict[str, _Route] = {
    ".html": _Route(".html", "text/html", False, False),
    ".htm": _Route(".htm", "text/html", False, False),
    ".csv": _Route(".csv", "text/csv", False, False),
    ".pdf": _Route(".pdf", "application/pdf", False, False),
    ".docx": _Route(".docx", _DOCX_MIME, True, False),
    ".pptx": _Route(".pptx", _PPTX_MIME, True, False),
    ".xlsx": _Route(".xlsx", _XLSX_MIME, True, True),
}
_MIME_TO_ROUTE: dict[str, _Route] = {route.mimetype: route for route in _ROUTES.values()}


def _resolve_route(filename: str | None, declared_mime: str | None) -> _Route | None:
    if filename:
        suffix = Path(filename).suffix.lower()
        route = _ROUTES.get(suffix)
        if route is not None:
            return route
    if declared_mime:
        return _MIME_TO_ROUTE.get(declared_mime.split(";", 1)[0].strip().lower())
    return None


def conversion_format(filename: str | None, declared_mime: str | None) -> str | None:
    """The admitted converter format (without a dot), using conversion's route precedence."""
    route = _resolve_route(filename, declared_mime)
    return route.extension[1:] if route is not None else None


def is_convertible(filename: str | None, declared_mime: str | None) -> bool:
    """Return whether an admitted suffix/MIME pair routes to a binary converter."""
    return conversion_format(filename, declared_mime) is not None


async def convert_resource(
    content: bytes,
    *,
    filename: str | None,
    declared_mime: str | None,
) -> ConvertedResource:
    """Convert admitted binary *content* to text and extracted visuals off-loop."""
    route = _resolve_route(filename, declared_mime)
    if route is None:
        raise ResourceConversionError("resource type is not an admitted binary format")
    budget = _ConversionBudget(time.monotonic() + _MAX_CONVERSION_SECONDS, Event())
    work = asyncio.create_task(asyncio.to_thread(_convert_sync, content, route, budget))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        # Cancellation cannot stop native work. Signal no more parser starts or
        # output adoption, then consume repeated cancellation requests until the
        # worker is joined. The caller-facing Registry task remains cancelled.
        budget.cancelled.set()
        current = asyncio.current_task()
        while current is not None and current.cancelling():
            current.uncancel()
        while not work.done():
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError:
                budget.cancelled.set()
                while current is not None and current.cancelling():
                    current.uncancel()
        return work.result()


def _convert_sync(content: bytes, route: _Route, budget: _ConversionBudget) -> ConvertedResource:
    budget.check()
    if route.is_ooxml:
        _preflight_ooxml(content)
    budget.check()
    if route.extension in {".docx", ".pdf", ".xlsx"}:
        return _convert_anydoc(content, route, budget)
    budget.check()
    result = _convert_markitdown(content, route)
    budget.check()
    return result


def _load_anydoc() -> ModuleType:
    # A different distribution/version is configuration failure, not fallback.
    try:
        installed = version("firecrawl-anydoc")
    except PackageNotFoundError as exc:
        raise RuntimeError("required firecrawl-anydoc distribution is missing") from exc
    if installed != _ANYDOC_VERSION:
        raise RuntimeError("qualified firecrawl-anydoc 0.2.4 is required")
    import anydoc

    return anydoc


def _convert_anydoc(content: bytes, route: _Route, budget: _ConversionBudget) -> ConvertedResource:
    engine = "firecrawl-anydoc"
    budget.check(converter=engine)
    reason: str | None = None
    try:
        candidate = _load_anydoc()
    except (ImportError, OSError) as exc:
        if isinstance(exc, OSError) and exc.errno == 12:
            raise MemoryError("native initialization exhausted memory") from exc
        reason = f"firecrawl-anydoc@{_ANYDOC_VERSION}:init:{type(exc).__name__}"
    else:
        budget.check(converter=engine)
        try:
            text = candidate.to_markdown_bytes(content, format=route.extension[1:], ocr="reject")
            budget.check(converter=engine)
            note = None
            visuals: tuple[ExtractedVisual, ...] = ()
            if route.extension == ".docx":
                document = candidate.to_document(content, format="docx")
                budget.check(converter=engine)
                occurrences, note = docx_asset_occurrences(document, content)
                visuals = tuple(
                    ExtractedVisual(
                        handle_id=f"vis-{index + 1}",
                        anchor=None,  # Package membership is not physical placement.
                        media_type=media,
                        data=data,
                        origin_part=part,
                    )
                    for index, (part, media, data) in enumerate(occurrences)
                )
            elif route.is_xlsx:
                # Text display values and anchored source images are independent.
                # No structured candidate parse (it supplies no XLSX assets), no
                # formula evaluation, external fetch or Office-page rendering.
                try:
                    visuals = tuple(_extract_xlsx_visuals(content))
                except MemoryError:
                    raise
                except Exception as exc:  # noqa: BLE001 - never adopt text without required assets
                    budget.check(converter=engine)
                    raise ResourceConversionError(
                        "XLSX source image extraction failed", converter=engine
                    ) from exc
        except candidate.NeedsOcrError as exc:
            budget.check(converter=engine)
            return ConvertedResource(
                text="",
                visuals=(),
                extraction_status="known_incomplete",
                converter=engine,
                converter_version=_ANYDOC_VERSION,
                known_ocr_pages=tuple(exc.pages),
                known_page_count=exc.page_count,
                note="Known OCR requirement; no partial text supplied by converter.",
            )
        except AssetBindingError as exc:
            raise ConversionLimitError(
                "candidate asset verification refused", converter=engine
            ) from exc
        except candidate.ResourceLimitError as exc:
            raise ConversionLimitError(
                "native conversion resource limit", converter=engine
            ) from exc
        except (candidate.MalformedError, candidate.MissingPartError) as exc:
            reason = f"firecrawl-anydoc@{_ANYDOC_VERSION}:parse:{type(exc).__name__}"
        except candidate.UnsupportedError:
            budget.check(converter=engine)
            return ConvertedResource(
                text="",
                visuals=(),
                extraction_status="known_incomplete",
                converter=engine,
                converter_version=_ANYDOC_VERSION,
                note="Candidate cannot represent this source; no partial output was adopted.",
            )
        except candidate.EncryptedError as exc:
            raise ResourceConversionError("encrypted document", converter=engine) from exc
        else:
            budget.check(converter=engine)
            return ConvertedResource(
                text=text,
                visuals=visuals,
                converter=engine,
                converter_version=_ANYDOC_VERSION,
                extraction_status="known_incomplete"
                if note
                else ("usable_text_unverified_coverage" if text.strip() else "no_extracted_text"),
                note=note,
            )
    # Only reached after candidate import/native work has ended. No empty retry,
    # reverse loop, speculative parse or renewed time allowance.
    budget.check(converter=engine)
    try:
        result = _convert_markitdown(content, route)
    except ResourceConversionError as exc:
        exc.fallback_reason = reason
        raise
    budget.check()
    return replace(result, fallback_reason=reason)


def _convert_markitdown(content: bytes, route: _Route) -> ConvertedResource:
    # One converter per call: registered converters and detector are never shared
    # across threads, so concurrent conversions cannot cross-contaminate.
    try:
        converter = MarkItDown(enable_plugins=False)
        result = converter.convert_stream(
            io.BytesIO(content),
            stream_info=StreamInfo(mimetype=route.mimetype, extension=route.extension),
            keep_data_uris=True,
        )
    except MemoryError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any converter failure uniformly
        raise ResourceConversionError("resource conversion failed") from exc

    text, embedded = _extract_embedded_visuals(result.markdown)
    visuals = list(embedded)
    if route.is_xlsx:
        visuals.extend(_extract_xlsx_visuals(content))
    note = None
    if route.extension == ".docx":
        try:
            parts, note = bind_docx_fallback_images(
                content, [(v.media_type, v.data) for v in visuals]
            )
        except AssetBindingError as exc:
            raise ConversionLimitError("fallback asset verification refused") from exc
        visuals = [replace(v, origin_part=part) for v, part in zip(visuals, parts, strict=True)]
    return ConvertedResource(
        text=text,
        visuals=tuple(visuals),
        extraction_status="known_incomplete"
        if note
        else ("usable_text_unverified_coverage" if text.strip() else "no_extracted_text"),
        note=note,
    )


def _preflight_ooxml(content: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            infos = archive.infolist()
            if len({info.filename for info in infos}) != len(infos):
                raise UnsafeArchiveError("ambiguous duplicate OOXML entries")
            if any(info.flag_bits & 1 for info in infos):
                raise UnsafeArchiveError("encrypted OOXML entries are not admitted")
            sizes = [(info.file_size, info.compress_size) for info in infos]
    except zipfile.BadZipFile as exc:
        raise ResourceConversionError(
            "resource is not a valid OOXML archive", converter="resource-host"
        ) from exc
    _validate_archive_sizes(sizes)


def _validate_archive_sizes(sizes: list[tuple[int, int]]) -> None:
    """Reject zip-bomb shapes from central-directory ``(uncompressed, compressed)`` sizes."""
    if len(sizes) > _MAX_OOXML_ENTRIES:
        raise UnsafeArchiveError("OOXML archive has too many entries")
    total_uncompressed = 0
    total_compressed = 0
    for uncompressed, compressed in sizes:
        if uncompressed > _MAX_OOXML_ENTRY_BYTES:
            raise UnsafeArchiveError("OOXML entry exceeds the per-entry size limit")
        total_uncompressed += uncompressed
        total_compressed += compressed
    if total_uncompressed > _MAX_OOXML_TOTAL_BYTES:
        raise UnsafeArchiveError("OOXML total uncompressed size exceeds the limit")
    if total_compressed > 0 and total_uncompressed / total_compressed > _MAX_OOXML_EXPANSION_RATIO:
        raise UnsafeArchiveError("OOXML expansion ratio exceeds the limit")


def _extract_embedded_visuals(text: str) -> tuple[str, list[ExtractedVisual]]:
    """Replace base64 image nodes with compact handles using markdown-it tokens."""
    parser = MarkdownIt("zero").enable("image")
    visuals: list[ExtractedVisual] = []
    for token in parser.parse(text):
        if token.type != "inline" or not token.children:
            continue
        for child in token.children:
            if child.type != "image":
                continue
            src = child.attrGet("src")
            if not isinstance(src, str) or not src.startswith("data:"):
                continue
            decoded = _decode_data_uri(src)
            if decoded is None:
                continue
            media_type, data = decoded
            handle_id = f"vis-{len(visuals) + 1}"
            visuals.append(
                ExtractedVisual(
                    handle_id=handle_id,
                    anchor=(child.content[:256] or None),
                    media_type=media_type,
                    data=data,
                )
            )
            text = text.replace(src, f"visual://{handle_id}", 1)
    return text, visuals


def _decode_data_uri(uri: str) -> tuple[str, bytes] | None:
    header, _, payload = uri[len("data:") :].partition(",")
    if not payload:
        return None
    parameters = header.split(";")
    if "base64" not in parameters[1:]:
        return None
    media_type = parameters[0] or "application/octet-stream"
    try:
        data = base64.b64decode(payload, validate=True)
    except binascii.Error, ValueError:
        return None
    return media_type, data


def _extract_xlsx_visuals(content: bytes) -> list[ExtractedVisual]:
    workbook = openpyxl.load_workbook(io.BytesIO(content), keep_links=False)
    visuals: list[ExtractedVisual] = []
    for worksheet in workbook.worksheets:
        for image in getattr(worksheet, "_images", []):
            data = image._data()
            visuals.append(
                ExtractedVisual(
                    handle_id=f"vis-{len(visuals) + 1}",
                    anchor=_xlsx_anchor(worksheet.title, image),
                    media_type=f"image/{(image.format or 'png').lower()}",
                    data=data,
                )
            )
    workbook.close()
    return visuals


def _xlsx_anchor(sheet_title: str, image: object) -> str | None:
    marker = getattr(getattr(image, "anchor", None), "_from", None)
    if marker is None:
        return None
    cell = f"{get_column_letter(marker.col + 1)}{marker.row + 1}"
    return f"{sheet_title}!{cell}"


__all__ = [
    "ConvertedResource",
    "ConversionLimitError",
    "ExtractedVisual",
    "ResourceConversionError",
    "UnsafeArchiveError",
    "conversion_format",
    "convert_resource",
    "is_convertible",
]
