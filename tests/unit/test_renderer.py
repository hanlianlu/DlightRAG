# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PageRenderer (unified page-image rendering)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from dlightrag.unifiedrepresent.renderer import PageRenderer, RenderResult


class TestRenderPdf:
    """Test PDF rendering via _render_pdf_sync."""

    def test_render_pdf_sync_basic(self) -> None:
        """Render a 2-page PDF mock and verify pages + metadata."""
        renderer = PageRenderer(dpi=144)

        mock_pil_img_0 = Image.new("RGB", (100, 100), "white")
        mock_pil_img_1 = Image.new("RGB", (100, 100), "blue")

        mock_bitmap_0 = MagicMock()
        mock_bitmap_0.to_pil.return_value = mock_pil_img_0
        mock_bitmap_1 = MagicMock()
        mock_bitmap_1.to_pil.return_value = mock_pil_img_1

        mock_page_0 = MagicMock()
        mock_page_0.render.return_value = mock_bitmap_0
        mock_page_1 = MagicMock()
        mock_page_1.render.return_value = mock_bitmap_1

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.__getitem__ = MagicMock(side_effect=[mock_page_0, mock_page_1])
        mock_doc.get_metadata_dict.return_value = {
            "Title": "Test Doc",
            "Author": "Tester",
            "CreationDate": "D:20250101",
        }

        with patch("dlightrag.unifiedrepresent.renderer.pdfium.PdfDocument", return_value=mock_doc):
            result = renderer._render_pdf_sync(Path("/fake/doc.pdf"))

        assert isinstance(result, RenderResult)
        assert len(result.pages) == 2
        assert result.pages[0][0] == 0
        assert result.pages[1][0] == 1
        assert result.pages[0][1] is mock_pil_img_0
        assert result.pages[1][1] is mock_pil_img_1

        assert result.metadata["original_format"] == "pdf"
        assert result.metadata["page_count"] == 2
        assert result.metadata["title"] == "Test Doc"
        assert result.metadata["author"] == "Tester"
        assert result.metadata["creation_date"] == "D:20250101"

        # _render_pdfium_doc now delegates: _get_page_count (open+close) then
        # _render_pdfium_doc_range (open+close), so close is called twice.
        assert mock_doc.close.call_count == 2

    def test_render_pdf_sync_uses_correct_scale(self) -> None:
        """Verify that page.render() is called with scale = dpi / 72."""
        renderer = PageRenderer(dpi=144)

        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = Image.new("RGB", (10, 10))

        mock_page = MagicMock()
        mock_page.render.return_value = mock_bitmap

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.get_metadata_dict.return_value = {}

        with patch("dlightrag.unifiedrepresent.renderer.pdfium.PdfDocument", return_value=mock_doc):
            renderer._render_pdf_sync(Path("/fake/doc.pdf"))

        mock_page.render.assert_called_once_with(scale=144 / 72)

    def test_render_pdf_sync_metadata_extraction_failure(self) -> None:
        """If get_metadata_dict raises, the result still contains base metadata."""
        renderer = PageRenderer()

        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = Image.new("RGB", (10, 10))

        mock_page = MagicMock()
        mock_page.render.return_value = mock_bitmap

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.get_metadata_dict.side_effect = RuntimeError("corrupt metadata")

        with patch("dlightrag.unifiedrepresent.renderer.pdfium.PdfDocument", return_value=mock_doc):
            result = renderer._render_pdf_sync(Path("/fake/doc.pdf"))

        assert result.metadata["original_format"] == "pdf"
        assert result.metadata["page_count"] == 1
        assert "title" not in result.metadata


class TestLoadImage:
    """Test _load_image with a real temporary PNG."""

    async def test_load_image_returns_single_page(self, tmp_path: Path) -> None:
        """Create a real 10x10 PNG and load it."""
        img = Image.new("RGB", (10, 10), "red")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        renderer = PageRenderer()
        result = await renderer._load_image(img_path)

        assert isinstance(result, RenderResult)
        assert len(result.pages) == 1
        assert result.pages[0][0] == 0

        loaded_img = result.pages[0][1]
        assert loaded_img.size == (10, 10)
        assert result.metadata["original_format"] == "png"

    async def test_load_image_jpeg_format(self, tmp_path: Path) -> None:
        """Verify format metadata for JPEG images."""
        img = Image.new("RGB", (10, 10), "green")
        img_path = tmp_path / "photo.jpg"
        img.save(img_path)

        renderer = PageRenderer()
        result = await renderer._load_image(img_path)

        assert result.metadata["original_format"] == "jpg"


class TestRenderOffice:
    """Test _render_office dispatching to LibreOffice."""

    async def test_raises_when_libreoffice_not_found(self, tmp_path: Path) -> None:
        """RuntimeError when shutil.which returns None for both libreoffice/soffice."""
        renderer = PageRenderer()
        doc_path = tmp_path / "doc.docx"
        doc_path.touch()

        with patch("dlightrag.unifiedrepresent.renderer.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="LibreOffice is required"):
                await renderer._render_office(doc_path)

    async def test_calls_subprocess_with_correct_args(self, tmp_path: Path) -> None:
        """Verify subprocess.run is called with correct LibreOffice arguments."""
        renderer = PageRenderer()
        doc_path = tmp_path / "report.docx"
        doc_path.touch()

        mock_run_result = MagicMock()
        mock_run_result.returncode = 0

        # Prepare a mock for _render_pdf_sync to avoid real PDF rendering.
        mock_render_result = RenderResult(
            pages=[(0, Image.new("RGB", (10, 10)))],
            metadata={"original_format": "pdf", "page_count": 1},
        )

        # We need the fake subprocess.run to create the expected PDF inside the
        # temporary directory that _render_office creates internally.
        def fake_subprocess_run(cmd, **kwargs):
            # The --outdir argument is followed by the temp dir path.
            outdir_idx = cmd.index("--outdir")
            outdir = cmd[outdir_idx + 1]
            # Create the expected output PDF so the existence check passes.
            (Path(outdir) / "report.pdf").touch()
            return mock_run_result

        with (
            patch(
                "dlightrag.unifiedrepresent.renderer.shutil.which",
                side_effect=lambda cmd: "/usr/bin/libreoffice" if cmd == "libreoffice" else None,
            ),
            patch(
                "dlightrag.unifiedrepresent.renderer.subprocess.run",
                side_effect=fake_subprocess_run,
            ) as mock_run,
            patch.object(renderer, "_render_pdf_sync", return_value=mock_render_result),
        ):
            result = await renderer._render_office(doc_path)

        # Verify subprocess.run was called with the right command.
        call_args = mock_run.call_args
        cmd_list = call_args[0][0]
        assert cmd_list[0] == "/usr/bin/libreoffice"
        assert "--headless" in cmd_list
        assert "--convert-to" in cmd_list
        assert "pdf" in cmd_list
        assert str(doc_path) in cmd_list

        # Original format should be overridden to "docx".
        assert result.metadata["original_format"] == "docx"

    async def test_raises_on_nonzero_exit(self, tmp_path: Path) -> None:
        """RuntimeError when LibreOffice exits with non-zero code."""
        renderer = PageRenderer()
        doc_path = tmp_path / "broken.pptx"
        doc_path.touch()

        mock_run_result = MagicMock()
        mock_run_result.returncode = 1
        mock_run_result.stderr = b"conversion error"

        with (
            patch(
                "dlightrag.unifiedrepresent.renderer.shutil.which",
                return_value="/usr/bin/libreoffice",
            ),
            patch(
                "dlightrag.unifiedrepresent.renderer.subprocess.run",
                return_value=mock_run_result,
            ),
        ):
            with pytest.raises(RuntimeError, match="LibreOffice conversion failed"):
                await renderer._render_office(doc_path)


class TestRenderFile:
    """Test render_file dispatching by extension."""

    async def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """Unsupported extensions raise ValueError."""
        renderer = PageRenderer()
        txt_path = tmp_path / "notes.txt"
        txt_path.touch()

        with pytest.raises(ValueError, match="Unsupported file extension '.txt'"):
            await renderer.render_file(txt_path)

    async def test_nonexistent_file_raises(self) -> None:
        """Nonexistent file raises FileNotFoundError."""
        renderer = PageRenderer()

        with pytest.raises(FileNotFoundError, match="File not found"):
            await renderer.render_file(Path("/does/not/exist.pdf"))


class TestRenderOfficeWithConverter:
    """Test _render_office delegation to LibreOfficeConverter."""

    async def test_delegates_to_converter(self, tmp_path):
        """When converter is provided, _render_office uses file_to_pdf_bytes."""
        mock_converter = MagicMock()
        renderer = PageRenderer(dpi=144, converter=mock_converter)

        doc_path = tmp_path / "doc.xlsx"
        doc_path.touch()

        mock_converter.file_to_pdf_bytes.return_value = b"%PDF-fake"

        mock_render_result = RenderResult(
            pages=[(0, Image.new("RGB", (10, 10)))],
            metadata={"original_format": "pdf", "page_count": 1},
        )

        with patch.object(renderer, "_render_pdf_from_bytes", return_value=mock_render_result):
            result = await renderer._render_office(doc_path)

        mock_converter.file_to_pdf_bytes.assert_called_once_with(doc_path)
        assert result.metadata["original_format"] == "xlsx"

    async def test_no_converter_uses_fallback(self, tmp_path):
        """Without converter, falls back to bare LibreOffice."""
        renderer = PageRenderer(dpi=144)  # no converter

        doc_path = tmp_path / "doc.docx"
        doc_path.touch()

        mock_run_result = MagicMock()
        mock_run_result.returncode = 0
        mock_render_result = RenderResult(
            pages=[(0, Image.new("RGB", (10, 10)))],
            metadata={"original_format": "pdf", "page_count": 1},
        )

        def fake_run(cmd, **kwargs):
            outdir_idx = cmd.index("--outdir")
            outdir = cmd[outdir_idx + 1]
            (Path(outdir) / "doc.pdf").touch()
            return mock_run_result

        with (
            patch(
                "dlightrag.unifiedrepresent.renderer.shutil.which",
                return_value="/usr/bin/libreoffice",
            ),
            patch("dlightrag.unifiedrepresent.renderer.subprocess.run", side_effect=fake_run),
            patch.object(renderer, "_render_pdf_sync", return_value=mock_render_result),
        ):
            result = await renderer._render_office(doc_path)

        assert result.metadata["original_format"] == "docx"


class TestRenderPdfFromBytes:
    """Test _render_pdf_from_bytes."""

    def test_renders_from_bytes(self):
        """Verify PDF bytes are rendered to page images."""
        renderer = PageRenderer(dpi=72)

        mock_pil_img = Image.new("RGB", (100, 100), "white")
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = mock_pil_img

        mock_page = MagicMock()
        mock_page.render.return_value = mock_bitmap

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.get_metadata_dict.return_value = {}

        with patch(
            "dlightrag.unifiedrepresent.renderer.pdfium.PdfDocument", return_value=mock_doc
        ) as mock_cls:
            result = renderer._render_pdf_from_bytes(b"%PDF-fake")

        # PdfDocument called twice: _get_page_count_from_bytes + _render_pdfium_doc_range
        assert mock_cls.call_count == 2
        # Both calls should use bytes, not a path
        for call_args in mock_cls.call_args_list:
            assert call_args[0][0] == b"%PDF-fake"
        assert len(result.pages) == 1
        assert result.pages[0][1] is mock_pil_img
        assert mock_doc.close.call_count == 2


class TestOfficeExtensionsExpanded:
    """Test that legacy Office formats are accepted."""

    async def test_xls_routes_to_render_office(self, tmp_path):
        """xls files should be handled by _render_office, not raise ValueError."""
        renderer = PageRenderer(dpi=144)
        xls_path = tmp_path / "old.xls"
        xls_path.touch()

        mock_result = RenderResult(
            pages=[(0, Image.new("RGB", (10, 10)))],
            metadata={"original_format": "xls"},
        )

        with patch.object(renderer, "_render_office", return_value=mock_result) as mock:
            result = await renderer.render_file(xls_path)

        mock.assert_called_once_with(xls_path)
        assert result.metadata["original_format"] == "xls"


# ---------------------------------------------------------------------------
# TestRenderFileBatched
# ---------------------------------------------------------------------------


class TestRenderFileBatched:
    """Test render_file_batched() streaming batch rendering."""

    async def test_pdf_batched_yields_batches(self, tmp_path: Path) -> None:
        """Render a 5-page PDF with batch_size=2 -> 3 batches (2+2+1)."""
        renderer = PageRenderer(dpi=72)
        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.touch()

        # Mock _get_page_count and _render_pdfium_doc_range
        with (
            patch.object(renderer, "_get_page_count", return_value=5),
            patch.object(
                renderer,
                "_render_pdfium_doc_range",
                side_effect=lambda src, start, end, total: RenderResult(
                    pages=[(idx, Image.new("RGB", (10, 10))) for idx in range(start, end)],
                    metadata={"original_format": "pdf", "page_count": total},
                ),
            ),
        ):
            batches = []
            async for batch in renderer.render_file_batched(str(fake_pdf), batch_size=2):
                batches.append(batch)

        assert len(batches) == 3

        # Batch 0: pages 0,1
        assert [idx for idx, _ in batches[0].pages] == [0, 1]
        assert batches[0].metadata["page_count"] == 5

        # Batch 1: pages 2,3
        assert [idx for idx, _ in batches[1].pages] == [2, 3]

        # Batch 2: page 4
        assert [idx for idx, _ in batches[2].pages] == [4]

    async def test_image_batched_yields_single_batch(self, tmp_path: Path) -> None:
        """Image files yield exactly one batch with one page."""
        img = Image.new("RGB", (10, 10), "red")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        renderer = PageRenderer()
        batches = []
        async for batch in renderer.render_file_batched(str(img_path), batch_size=5):
            batches.append(batch)

        assert len(batches) == 1
        assert len(batches[0].pages) == 1
        assert batches[0].pages[0][0] == 0
        assert batches[0].metadata["original_format"] == "png"

    async def test_batched_global_indices_preserved(self, tmp_path: Path) -> None:
        """Global page indices are correct across batches."""
        renderer = PageRenderer(dpi=72)
        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.touch()

        with (
            patch.object(renderer, "_get_page_count", return_value=3),
            patch.object(
                renderer,
                "_render_pdfium_doc_range",
                side_effect=lambda src, start, end, total: RenderResult(
                    pages=[(idx, Image.new("RGB", (10, 10))) for idx in range(start, end)],
                    metadata={"original_format": "pdf", "page_count": total},
                ),
            ),
        ):
            all_indices = []
            async for batch in renderer.render_file_batched(str(fake_pdf), batch_size=1):
                for idx, _ in batch.pages:
                    all_indices.append(idx)

        assert all_indices == [0, 1, 2]

    async def test_batched_nonexistent_file_raises(self) -> None:
        """Nonexistent file raises FileNotFoundError."""
        renderer = PageRenderer()
        with pytest.raises(FileNotFoundError, match="File not found"):
            async for _ in renderer.render_file_batched("/does/not/exist.pdf"):
                pass

    async def test_batched_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """Unsupported extensions raise ValueError."""
        txt_path = tmp_path / "notes.txt"
        txt_path.touch()

        renderer = PageRenderer()
        with pytest.raises(ValueError, match="Unsupported file extension '.txt'"):
            async for _ in renderer.render_file_batched(str(txt_path)):
                pass

    async def test_batched_metadata_consistent(self, tmp_path: Path) -> None:
        """All batches carry the same metadata (page_count=total)."""
        renderer = PageRenderer(dpi=72)
        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.touch()

        with (
            patch.object(renderer, "_get_page_count", return_value=4),
            patch.object(
                renderer,
                "_render_pdfium_doc_range",
                side_effect=lambda src, start, end, total: RenderResult(
                    pages=[(idx, Image.new("RGB", (10, 10))) for idx in range(start, end)],
                    metadata={
                        "original_format": "pdf",
                        "page_count": total,
                        "title": "Test Doc",
                    },
                ),
            ),
        ):
            batches = []
            async for batch in renderer.render_file_batched(str(fake_pdf), batch_size=3):
                batches.append(batch)

        assert len(batches) == 2
        for batch in batches:
            assert batch.metadata["page_count"] == 4
            assert batch.metadata["title"] == "Test Doc"
