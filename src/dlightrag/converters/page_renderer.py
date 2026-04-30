# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Page rendering for unified representational RAG.

Converts PDF, Office, and image files into page images for visual embedding.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pypdfium2 as pdfium
from PIL import Image

if TYPE_CHECKING:
    from dlightrag.converters.office import LibreOfficeConverter

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp", ".gif"})
_OFFICE_EXTENSIONS = frozenset({".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls"})


@dataclass
class RenderResult:
    """Result of rendering a document to page images."""

    pages: list[tuple[int, Image.Image]]  # (page_index, image)
    metadata: dict[str, str | int] = field(default_factory=dict)
    # metadata keys: title, author, creation_date, page_count, original_format


class PageRenderer:
    """Renders PDF, Office, and image files into page images.

    Each document is converted to a list of ``(page_index, PIL.Image)`` tuples
    suitable for downstream visual embedding.

    Parameters
    ----------
    dpi:
        Rendering resolution in dots per inch. Default 250 balances quality
        and memory usage for typical A4/Letter documents.
    converter:
        Optional ``LibreOfficeConverter`` instance. When provided,
        ``_render_office`` delegates to its ``file_to_pdf_bytes`` method,
        enabling optimized conversion (e.g. adaptive page width for Excel).
        When omitted, a bare ``libreoffice --headless`` call is used.
    """

    def __init__(
        self,
        dpi: int = 250,
        converter: LibreOfficeConverter | None = None,
    ) -> None:
        self.dpi = dpi
        self._converter = converter

    async def render_file(self, path: str | Path) -> RenderResult:
        """Render a file to page images, dispatching by extension.

        Parameters
        ----------
        path:
            Path to the input file (PDF, image, or Office document).

        Returns
        -------
        RenderResult
            Page images and associated metadata.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file extension is not supported.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()

        if ext == ".pdf":
            return await self._render_pdf(path)
        if ext in _IMAGE_EXTENSIONS:
            return await self._load_image(path)
        if ext in _OFFICE_EXTENSIONS:
            return await self._render_office(path)

        supported = sorted({".pdf"} | _IMAGE_EXTENSIONS | _OFFICE_EXTENSIONS)
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported formats: {', '.join(supported)}"
        )

    async def render_file_batched(
        self,
        path: str | Path,
        batch_size: int = 20,
    ) -> AsyncIterator[RenderResult]:
        """Render a file to page images in batches, dispatching by extension.

        Yields one :class:`RenderResult` per batch of up to *batch_size* pages.
        This avoids loading an entire large document into memory at once.

        Parameters
        ----------
        path:
            Path to the input file (PDF, image, or Office document).
        batch_size:
            Maximum number of pages per yielded batch.

        Yields
        ------
        RenderResult
            A batch of rendered pages with associated metadata.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file extension is not supported.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()

        if ext == ".pdf":
            async for batch in self._render_pdf_batched(path, batch_size):
                yield batch
            return
        if ext in _IMAGE_EXTENSIONS:
            yield await self._load_image(path)
            return
        if ext in _OFFICE_EXTENSIONS:
            async for batch in self._render_office_batched(path, batch_size):
                yield batch
            return

        supported = sorted({".pdf"} | _IMAGE_EXTENSIONS | _OFFICE_EXTENSIONS)
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported formats: {', '.join(supported)}"
        )

    # ------------------------------------------------------------------
    # Internal renderers
    # ------------------------------------------------------------------

    async def _render_pdf(self, path: Path) -> RenderResult:
        """Render all pages of a PDF to images using pypdfium2.

        pypdfium2 is not thread-safe, so the synchronous work is wrapped
        in ``asyncio.to_thread()``.
        """
        return await asyncio.to_thread(self._render_pdf_sync, path)

    def _render_pdf_sync(self, path: Path) -> RenderResult:
        """Synchronous PDF rendering (called via ``to_thread``)."""
        return self._render_pdfium_doc(str(path))

    async def _load_image(self, path: Path) -> RenderResult:
        """Load a single image file as a one-page result."""

        def _load() -> RenderResult:
            img = Image.open(path)
            img.load()  # force read so file handle is released
            return RenderResult(
                pages=[(0, img)],
                metadata={"original_format": path.suffix.lower().lstrip(".")},
            )

        return await asyncio.to_thread(_load)

    def _render_pdf_from_bytes(self, pdf_bytes: bytes) -> RenderResult:
        """Render PDF from raw bytes (called via to_thread by _render_office)."""
        return self._render_pdfium_doc(pdf_bytes)

    def _render_pdfium_doc(self, source: str | bytes) -> RenderResult:
        """Render all pages of a PDF from a pdfium-compatible source (file path or bytes)."""
        total = (
            self._get_page_count(source)  # type: ignore[arg-type]
            if isinstance(source, str)
            else self._get_page_count_from_bytes(source)
        )
        return self._render_pdfium_doc_range(source, 0, total, total)

    def _render_pdfium_doc_range(
        self,
        source: str | bytes,
        start: int,
        end: int,
        total: int,
    ) -> RenderResult:
        """Render pages [start, end) from a pdfium-compatible source.

        This is the single shared render method used by both
        ``_render_pdf_batched`` and ``_render_office_batched``.

        Parameters
        ----------
        source:
            File path string or raw PDF bytes.
        start:
            First page index (inclusive, 0-based).
        end:
            Last page index (exclusive).
        total:
            Total page count of the document (used for metadata).

        Returns
        -------
        RenderResult
            Pages with global page indices and full document metadata.
        """
        doc = pdfium.PdfDocument(source)
        try:
            scale = self.dpi / 72
            pages: list[tuple[int, Image.Image]] = []
            for idx in range(start, end):
                page = doc[idx]
                bitmap = page.render(scale=scale)  # type: ignore[arg-type]
                pil_image = bitmap.to_pil()
                pages.append((idx, pil_image))

            metadata: dict[str, str | int] = {"original_format": "pdf", "page_count": total}
            try:
                meta = doc.get_metadata_dict()
                if meta.get("Title"):
                    metadata["title"] = meta["Title"]
                if meta.get("Author"):
                    metadata["author"] = meta["Author"]
                if meta.get("CreationDate"):
                    metadata["creation_date"] = meta["CreationDate"]
            except Exception:  # noqa: BLE001
                logger.debug("Could not extract PDF metadata from %s", source)

            return RenderResult(pages=pages, metadata=metadata)
        finally:
            doc.close()

    def _get_page_count(self, source: str) -> int:
        """Return the total page count for a PDF file path."""
        doc = pdfium.PdfDocument(source)
        try:
            return len(doc)
        finally:
            doc.close()

    def _get_page_count_from_bytes(self, data: bytes) -> int:
        """Return the total page count for PDF bytes."""
        doc = pdfium.PdfDocument(data)
        try:
            return len(doc)
        finally:
            doc.close()

    async def _render_office(self, path: Path) -> RenderResult:
        """Convert an Office document to PDF via LibreOffice, then render.

        When a converter is provided, delegates to its ``file_to_pdf_bytes``
        method for optimized conversion (adaptive page width for Excel).
        Otherwise falls back to a bare ``libreoffice --headless`` call.

        Raises
        ------
        RuntimeError
            If LibreOffice is not installed or the conversion fails.
        """
        if self._converter:
            converter = self._converter  # bind for closure

            def _convert_and_render(p: Path) -> RenderResult:
                pdf_bytes = converter.file_to_pdf_bytes(p)
                return self._render_pdf_from_bytes(pdf_bytes)

            render_result = await asyncio.to_thread(_convert_and_render, path)
            render_result.metadata["original_format"] = path.suffix.lower().lstrip(".")
            return render_result

        # Fallback: bare LibreOffice (no converter provided)
        lo_bin = shutil.which("libreoffice") or shutil.which("soffice")
        if lo_bin is None:
            raise RuntimeError(
                "LibreOffice is required to render Office documents but was not found on PATH. "
                "Install it (e.g. `apt install libreoffice-core`) and try again."
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    lo_bin,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    tmp_dir,
                    str(path),
                ],
                capture_output=True,
                timeout=120,
            )

            if result.returncode != 0:
                stderr = result.stderr.decode(errors="replace").strip()
                raise RuntimeError(
                    f"LibreOffice conversion failed (exit {result.returncode}): {stderr}"
                )

            pdf_path = Path(tmp_dir) / (path.stem + ".pdf")
            if not pdf_path.exists():
                raise RuntimeError(
                    f"LibreOffice conversion produced no output PDF "
                    f"(expected {pdf_path.name} in temp dir)"
                )

            render_result = self._render_pdf_sync(pdf_path)

        # Override format metadata to reflect the original Office format.
        render_result.metadata["original_format"] = path.suffix.lower().lstrip(".")
        return render_result

    async def _render_pdf_batched(self, path: Path, batch_size: int) -> AsyncIterator[RenderResult]:
        """Yield batches of rendered pages from a PDF file."""
        total = await asyncio.to_thread(self._get_page_count, str(path))
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = await asyncio.to_thread(
                self._render_pdfium_doc_range, str(path), start, end, total
            )
            yield batch

    async def _render_office_batched(
        self, path: Path, batch_size: int
    ) -> AsyncIterator[RenderResult]:
        """Convert an Office document to PDF bytes, then yield batches of rendered pages.

        Re-uses ``_render_pdfium_doc_range`` to avoid duplicating rendering logic.

        Raises
        ------
        RuntimeError
            If LibreOffice is not installed or the conversion fails.
        """
        if self._converter:
            converter = self._converter

            def _convert(p: Path) -> bytes:
                return converter.file_to_pdf_bytes(p)

            pdf_bytes: bytes = await asyncio.to_thread(_convert, path)
        else:
            lo_bin = shutil.which("libreoffice") or shutil.which("soffice")
            if lo_bin is None:
                raise RuntimeError(
                    "LibreOffice is required to render Office documents but was not found on PATH. "
                    "Install it (e.g. `apt install libreoffice-core`) and try again."
                )

            with tempfile.TemporaryDirectory() as tmp_dir:
                result = await asyncio.to_thread(
                    subprocess.run,
                    [lo_bin, "--headless", "--convert-to", "pdf", "--outdir", tmp_dir, str(path)],
                    capture_output=True,
                    timeout=120,
                )

                if result.returncode != 0:
                    stderr = result.stderr.decode(errors="replace").strip()
                    raise RuntimeError(
                        f"LibreOffice conversion failed (exit {result.returncode}): {stderr}"
                    )

                pdf_path = Path(tmp_dir) / (path.stem + ".pdf")
                if not pdf_path.exists():
                    raise RuntimeError(
                        f"LibreOffice conversion produced no output PDF "
                        f"(expected {pdf_path.name} in temp dir)"
                    )

                pdf_bytes = pdf_path.read_bytes()

        original_format = path.suffix.lower().lstrip(".")
        total = await asyncio.to_thread(self._get_page_count_from_bytes, pdf_bytes)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = await asyncio.to_thread(
                self._render_pdfium_doc_range, pdf_bytes, start, end, total
            )
            batch.metadata["original_format"] = original_format
            yield batch
