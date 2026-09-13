# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""On-demand physical PDF rendering, independent of text conversion and inference."""

from __future__ import annotations

import io
import threading

import pypdfium2 as pdfium

from dlightrag.engine.answer.resources.models import ResourceAdmissionError, ResourceRegistryError

# PDFium is not thread-safe, even across independent documents.
_PDF_LOCK = threading.Lock()
_MAX_RENDER_PIXELS = 40_000_000


class ResourceViewError(ResourceRegistryError):
    """An admitted resource cannot provide the requested visual target."""


def pdf_page_count(data: bytes) -> int:
    with _PDF_LOCK:
        try:
            with pdfium.PdfDocument(data) as pdf:
                return len(pdf)
        except Exception as exc:
            raise ResourceViewError("PDF page inventory unavailable") from exc


def render_pdf_page(data: bytes, page: int, *, overview: bool) -> bytes:
    """Render exactly one 1-based physical page, closing all native objects."""
    with _PDF_LOCK:
        try:
            with pdfium.PdfDocument(data) as pdf:
                if page < 1 or page > len(pdf):
                    raise ResourceViewError(f"page {page} is out of range (1-{len(pdf)})")
                source = pdf[page - 1]
                try:
                    width, height = source.get_size()
                    if width <= 0 or height <= 0:
                        raise ResourceViewError("PDF page has invalid dimensions")
                    scale = min(1.0, 900 / max(width, height)) if overview else 2.0
                    if width * height * scale * scale > _MAX_RENDER_PIXELS:
                        raise ResourceAdmissionError("PDF page exceeds rendering pixel limit")
                    # PDFium accepts fractional scale; its Python annotation says int.
                    bitmap = source.render(scale=scale)  # pyright: ignore[reportArgumentType]
                    try:
                        with bitmap.to_pil() as image:
                            buffer = io.BytesIO()
                            image.save(buffer, format="PNG")
                            return buffer.getvalue()
                    finally:
                        bitmap.close()
                finally:
                    source.close()
        except ResourceRegistryError:
            raise
        except Exception as exc:
            raise ResourceViewError("PDF page rendering failed") from exc


__all__ = ["ResourceViewError", "pdf_page_count", "render_pdf_page"]
