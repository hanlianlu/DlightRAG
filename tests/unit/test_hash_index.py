# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for HashIndex (JSON-based)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlightrag.ingestion.hash_index import HashIndex, compute_file_hash


class TestComputeFileHash:
    """Test file hash computation."""

    def test_computes_sha256(self, tmp_path: Path) -> None:
        """Test SHA256 hash computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hash_value = compute_file_hash(test_file)
        assert hash_value.startswith("sha256:")
        assert len(hash_value) == 71  # "sha256:" + 64 hex chars

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        """Test that identical content produces identical hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("same content")
        file2.write_text("same content")

        assert compute_file_hash(file1) == compute_file_hash(file2)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Test that different content produces different hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content A")
        file2.write_text("content B")

        assert compute_file_hash(file1) != compute_file_hash(file2)


class TestContentAwarePdfHash:
    """Test PDF content-aware hashing."""

    def test_pdf_same_content_different_metadata_same_hash(self, tmp_path: Path) -> None:
        """Two PDFs with same text but different CreationDate produce same hash."""
        _create_pdf_with_metadata(tmp_path / "pdf1.pdf", "Hello World", "2026-01-01")
        _create_pdf_with_metadata(tmp_path / "pdf2.pdf", "Hello World", "2026-06-15")

        assert compute_file_hash(tmp_path / "pdf1.pdf") == compute_file_hash(tmp_path / "pdf2.pdf")

    def test_pdf_different_content_different_hash(self, tmp_path: Path) -> None:
        """PDFs with different text produce different hashes."""
        _create_pdf_with_metadata(tmp_path / "pdf1.pdf", "Content Alpha", "2026-01-01")
        _create_pdf_with_metadata(tmp_path / "pdf2.pdf", "Content Beta", "2026-01-01")

        assert compute_file_hash(tmp_path / "pdf1.pdf") != compute_file_hash(tmp_path / "pdf2.pdf")

    def test_pdf_no_text_falls_back_to_byte_hash(self, tmp_path: Path) -> None:
        """Scanned/image-only PDF falls back to byte hash without error."""
        pdf_bytes = _minimal_pdf_no_text()
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(pdf_bytes)

        result = compute_file_hash(pdf_path)
        assert result.startswith("sha256:")
        assert len(result) == 71

    def test_pdf_pypdfium2_unavailable_falls_back(self, tmp_path: Path, monkeypatch: Any) -> None:
        """When pypdfium2 import fails, falls back to byte hash."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pypdfium2":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(_minimal_pdf_no_text())

        result = compute_file_hash(pdf_path)
        assert result.startswith("sha256:")


class TestContentAwareOfficeHash:
    """Test Office (ZIP-based) content-aware hashing."""

    def test_docx_same_content_different_metadata_same_hash(self, tmp_path: Path) -> None:
        """Two DOCX files with same document.xml but different docProps produce same hash."""
        _create_docx(tmp_path / "a.docx", "<w:body>Hello</w:body>", "2026-01-01T00:00:00Z")
        _create_docx(tmp_path / "b.docx", "<w:body>Hello</w:body>", "2026-06-15T12:00:00Z")

        assert compute_file_hash(tmp_path / "a.docx") == compute_file_hash(tmp_path / "b.docx")

    def test_docx_different_content_different_hash(self, tmp_path: Path) -> None:
        """DOCX files with different document.xml produce different hashes."""
        _create_docx(tmp_path / "a.docx", "<w:body>Alpha</w:body>", "2026-01-01T00:00:00Z")
        _create_docx(tmp_path / "b.docx", "<w:body>Beta</w:body>", "2026-01-01T00:00:00Z")

        assert compute_file_hash(tmp_path / "a.docx") != compute_file_hash(tmp_path / "b.docx")

    def test_corrupted_zip_falls_back_to_byte_hash(self, tmp_path: Path) -> None:
        """Corrupted DOCX (invalid ZIP) falls back to byte hash."""
        bad_docx = tmp_path / "bad.docx"
        bad_docx.write_bytes(b"this is not a zip file")

        result = compute_file_hash(bad_docx)
        assert result.startswith("sha256:")
        assert len(result) == 71

    def test_non_office_extension_uses_byte_hash(self, tmp_path: Path) -> None:
        """Non-PDF, non-Office files use raw byte hash (unchanged behavior)."""
        txt = tmp_path / "readme.txt"
        txt.write_text("hello world")

        result = compute_file_hash(txt)
        assert result.startswith("sha256:")

        from dlightrag.ingestion.hash_index import _hash_file_bytes
        assert result == _hash_file_bytes(txt)


class TestHashIndex:
    """Test JSON-based hash index."""

    async def test_register_and_lookup(self, tmp_path: Path) -> None:
        """Test registering and looking up hashes."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        index = HashIndex(tmp_path, sources_dir)

        content_hash = "sha256:abc123"
        await index.register(content_hash, "doc-001", "/path/to/file.pdf")

        entry = index.lookup(content_hash)
        assert entry is not None
        assert entry["doc_id"] == "doc-001"
        assert entry["file_path"] == "/path/to/file.pdf"

    def test_lookup_missing(self, tmp_path: Path) -> None:
        """Test lookup for non-existent hash."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        index = HashIndex(tmp_path, sources_dir)
        assert index.lookup("sha256:nonexistent") is None

    async def test_remove(self, tmp_path: Path) -> None:
        """Test removing a hash entry."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        index = HashIndex(tmp_path, sources_dir)
        content_hash = "sha256:abc123"

        await index.register(content_hash, "doc-001", "/path/to/file.pdf")
        assert index.lookup(content_hash) is not None

        result = await index.remove(content_hash)
        assert result is True
        assert index.lookup(content_hash) is None

    async def test_remove_missing(self, tmp_path: Path) -> None:
        """Test removing non-existent hash."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        index = HashIndex(tmp_path, sources_dir)
        assert await index.remove("sha256:nonexistent") is False

    async def test_list_all(self, tmp_path: Path) -> None:
        """Test listing all entries."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        index = HashIndex(tmp_path, sources_dir)
        await index.register("sha256:aaa", "doc-001", "/path/a.pdf")
        await index.register("sha256:bbb", "doc-002", "/path/b.pdf")

        entries = await index.list_all()
        assert len(entries) == 2

    async def test_invalidate(self, tmp_path: Path) -> None:
        """Test cache invalidation."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        index = HashIndex(tmp_path, sources_dir)
        await index.register("sha256:aaa", "doc-001", "/path/a.pdf")

        # Invalidate should clear cache without error
        index.invalidate()

        # Lookup should still work after invalidation
        entry = index.lookup("sha256:aaa")
        assert entry is not None

    async def test_should_skip_file_new(self, tmp_path: Path) -> None:
        """Test should_skip_file for a new file."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        test_file = tmp_path / "new_file.txt"
        test_file.write_text("new content")

        index = HashIndex(tmp_path, sources_dir)
        should_skip, content_hash, reason = await index.should_skip_file(test_file, replace=False)

        assert not should_skip
        assert content_hash is not None
        assert content_hash.startswith("sha256:")

    async def test_should_skip_file_duplicate(self, tmp_path: Path) -> None:
        """Test should_skip_file for a duplicate file."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        test_file = tmp_path / "file.txt"
        test_file.write_text("same content")

        index = HashIndex(tmp_path, sources_dir)

        # Register the hash first
        content_hash = compute_file_hash(test_file)
        await index.register(content_hash, "doc-001", str(test_file))

        # Now check — should skip
        should_skip, _, reason = await index.should_skip_file(test_file, replace=False)
        assert should_skip
        assert reason is not None


def _create_pdf_with_metadata(path: Path, text: str, date_str: str) -> None:
    """Create a minimal PDF with given text and CreationDate metadata."""
    content = f"BT /F1 12 Tf 100 700 Td ({text}) Tj ET".encode("latin-1")

    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>>endobj\n"
        b"4 0 obj<</Length " + str(len(content)).encode() + b">>\n"
        b"stream\n" + content + b"\nendstream\nendobj\n"
        b"5 0 obj<</CreationDate(D:" + date_str.replace("-", "").encode() + b"000000+00'00')>>endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000306 00000 n \n"
        b"0000000000 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R/Info 5 0 R>>\n"
        b"startxref\n0\n%%EOF"
    )
    Path(path).write_bytes(pdf)


def _minimal_pdf_no_text() -> bytes:
    """Return bytes for a minimal valid PDF with no extractable text."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n0\n%%EOF"
    )


def _create_docx(path: Path, body_xml: str, modified_time: str) -> None:
    """Create a minimal DOCX (ZIP) with given content and metadata timestamp."""
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", f'<?xml version="1.0"?>{body_xml}')
        zf.writestr(
            "docProps/core.xml",
            f'<?xml version="1.0"?><cp:coreProperties>'
            f"<dcterms:modified>{modified_time}</dcterms:modified>"
            f"</cp:coreProperties>",
        )
        zf.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships/>')
        zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
