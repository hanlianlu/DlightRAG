# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Hash-based content deduplication for RAG ingestion.

Provides PostgreSQL-backed content-addressable storage tracking via SHA256
hashes.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any

from dlightrag.storage.protocols import HashIndexProtocol

logger = logging.getLogger(__name__)


def derive_source_type(file_path: str) -> str:
    """Derive source type from a file path or URI.

    Returns "azure_blobs", "s3", "local", or "unknown".
    Remote sources use explicit URI schemes; all plain paths are local.
    """
    if not file_path:
        return "unknown"
    if file_path.startswith("azure://"):
        return "azure_blobs"
    if file_path.startswith("s3://"):
        return "s3"
    # Default for local absolute/relative paths
    if file_path.startswith("/") or file_path.startswith(".") or "://" not in file_path:
        return "local"
    return "unknown"


# --- Content-aware hash extensions ---
_PDF_EXTENSIONS = {".pdf"}
_OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}

_OFFICE_CONTENT_PREFIXES: dict[str, tuple[str, ...]] = {
    ".docx": ("word/document.xml", "word/media/"),
    ".pptx": ("ppt/slides/", "ppt/media/"),
    ".xlsx": ("xl/worksheets/", "xl/sharedStrings.xml"),
}


def _hash_pdf_content(file_path: Path) -> str | None:
    """Extract text from PDF pages via pypdfium2 and hash it.

    Returns sha256 hash string, or None to signal fallback to byte hash
    (when no text is extractable or pypdfium2 is unavailable).
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.debug("pypdfium2 not available, falling back to byte hash for %s", file_path.name)
        return None

    try:
        doc = pdfium.PdfDocument(str(file_path))
        sha256_hash = hashlib.sha256()
        has_text = False

        for i in range(len(doc)):
            page = doc[i]
            textpage = page.get_textpage()
            text = textpage.get_text_bounded()
            if text.strip():
                has_text = True
            sha256_hash.update(text.encode("utf-8"))

        if not has_text:
            logger.debug("No text extracted from %s, falling back to byte hash", file_path.name)
            return None

        return f"sha256:{sha256_hash.hexdigest()}"
    except Exception as exc:
        logger.debug("PDF content extraction failed for %s: %s", file_path.name, exc)
        return None


def _hash_office_content(file_path: Path) -> str | None:
    """Hash only content entries from Office ZIP archive, skipping metadata.

    Returns sha256 hash string, or None to signal fallback to byte hash.
    """
    import zipfile

    ext = file_path.suffix.lower()
    prefixes = _OFFICE_CONTENT_PREFIXES.get(ext)
    if not prefixes:
        return None

    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            sha256_hash = hashlib.sha256()
            has_content = False

            for name in sorted(zf.namelist()):
                if any(name.startswith(p) or name == p for p in prefixes):
                    sha256_hash.update(name.encode("utf-8"))
                    sha256_hash.update(zf.read(name))
                    has_content = True

            if not has_content:
                logger.debug(
                    "No content entries found in %s, falling back to byte hash", file_path.name
                )
                return None

            return f"sha256:{sha256_hash.hexdigest()}"
    except Exception as exc:
        logger.debug("Office content extraction failed for %s: %s", file_path.name, exc)
        return None


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of file content.

    For PDF and Office formats, hashes only the actual content (text, media)
    and skips volatile metadata (timestamps, IDs) that change on every
    download/export. Falls back to raw byte hash if content extraction fails.

    Returns:
        Hash string in format "sha256:<hex_digest>"
    """
    suffix = file_path.suffix.lower()

    # PDF: extract text via pypdfium2
    if suffix in _PDF_EXTENSIONS:
        result = _hash_pdf_content(file_path)
        if result is not None:
            return result

    # Office: hash ZIP content entries only
    if suffix in _OFFICE_EXTENSIONS:
        result = _hash_office_content(file_path)
        if result is not None:
            return result

    # Fallback: raw byte hash
    return _hash_file_bytes(file_path, chunk_size)


def _hash_file_bytes(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of raw file bytes (original behavior)."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return f"sha256:{sha256_hash.hexdigest()}"


class PGHashIndex:
    """PostgreSQL-backed content hash index for deduplication.

    Multi-worker safe — uses PostgreSQL transactions instead of file locks.
    Table: dlightrag_file_hashes (created by init.sql or auto-created).
    Uses LightRAG's shared ClientManager for connection pooling.
    """

    TABLE = "dlightrag_file_hashes"

    def __init__(
        self,
        workspace: str = "default",
        pool: Any = None,
    ) -> None:
        self._pool = pool  # asyncpg.Pool — set via initialize() or passed directly
        self._workspace = workspace

    def _get_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PGHashIndex not initialized — call initialize() first")
        return self._pool

    async def initialize(self) -> None:
        """Create table using DlightRAG's dedicated pool (idempotent DDL)."""
        if self._pool is None:
            from dlightrag.storage.pool import pg_pool

            self._pool = await pg_pool.get()

        async with self._get_pool().acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    content_hash TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    workspace TEXT NOT NULL DEFAULT 'default',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_file_hashes_doc_id
                ON {self.TABLE}(doc_id)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_file_hashes_workspace
                ON {self.TABLE}(workspace)
            """)

    async def clear(self) -> None:
        """Remove all hash entries for this workspace (used by reset)."""
        async with self._get_pool().acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.TABLE} WHERE workspace = $1",
                self._workspace,
            )
            logger.info("PGHashIndex cleared for workspace %s: %s", self._workspace, result)

    async def check_exists(self, content_hash: str) -> tuple[bool, str | None]:
        async with self._get_pool().acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT doc_id FROM {self.TABLE} WHERE content_hash = $1 AND workspace = $2",
                content_hash,
                self._workspace,
            )
            return (True, row["doc_id"]) if row else (False, None)

    async def register(self, content_hash: str, doc_id: str, file_path: str) -> None:
        async with self._get_pool().acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {self.TABLE} (content_hash, doc_id, file_path, workspace)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (content_hash) DO UPDATE SET
                     doc_id = EXCLUDED.doc_id,
                     file_path = EXCLUDED.file_path""",
                content_hash,
                doc_id,
                file_path,
                self._workspace,
            )

    async def remove(self, content_hash: str) -> bool:
        async with self._get_pool().acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.TABLE} WHERE content_hash = $1 AND workspace = $2",
                content_hash,
                self._workspace,
            )
            return result != "DELETE 0"

    async def should_skip_file(
        self,
        file_path: Path,
        replace: bool,
    ) -> tuple[bool, str | None, str | None]:
        content_hash = await asyncio.to_thread(compute_file_hash, file_path)
        if replace:
            return (False, content_hash, None)
        exists, doc_id = await self.check_exists(content_hash)
        if exists:
            logger.info(f"Skipping duplicate: {file_path.name} (hash matches doc_id={doc_id})")
            return (True, content_hash, f"Duplicate of {doc_id}")
        return (False, content_hash, None)

    @staticmethod
    def generate_doc_id_from_path(file_path: Path) -> str:
        return file_path.stem

    async def find_by_name(self, filename: str) -> tuple[str | None, str | None, str | None]:
        async with self._get_pool().acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT content_hash, doc_id, file_path FROM {self.TABLE} "
                f"WHERE file_path LIKE $1 AND workspace = $2 LIMIT 1",
                f"%/{filename}",
                self._workspace,
            )
            if row:
                return (row["doc_id"], row["content_hash"], row["file_path"])
            return (None, None, None)

    async def find_by_path(self, file_path: str) -> tuple[str | None, str | None, str | None]:
        async with self._get_pool().acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT content_hash, doc_id, file_path FROM {self.TABLE} "
                f"WHERE file_path = $1 AND workspace = $2 LIMIT 1",
                file_path,
                self._workspace,
            )
            if row:
                return (row["doc_id"], row["content_hash"], row["file_path"])
            return (None, None, None)

    async def find_by_hash(self, content_hash: str) -> tuple[str | None, str | None, str | None]:
        async with self._get_pool().acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT doc_id, file_path FROM {self.TABLE} "
                f"WHERE content_hash = $1 AND workspace = $2 LIMIT 1",
                content_hash,
                self._workspace,
            )
            if row:
                return (row["doc_id"], content_hash, row["file_path"])
            return (None, None, None)

    async def list_all(self) -> list[dict[str, Any]]:
        async with self._get_pool().acquire() as conn:
            rows = await conn.fetch(
                f"SELECT content_hash, doc_id, file_path, created_at "
                f"FROM {self.TABLE} WHERE workspace = $1",
                self._workspace,
            )
            results = []
            for row in rows:
                file_path = row["file_path"]
                source_type = derive_source_type(file_path)
                results.append(
                    {
                        "file_path": file_path,
                        "doc_id": row["doc_id"],
                        "source_type": source_type,
                        "file_name": Path(file_path).name,
                        "content_hash": row["content_hash"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else "",
                    }
                )
            return results

    def invalidate(self) -> None:
        """No-op for PG backend (no cache to invalidate)."""


__all__ = [
    "HashIndexProtocol",
    "PGHashIndex",
    "compute_file_hash",
    "derive_source_type",
]
