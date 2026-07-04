# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Upload staging helpers shared by REST and Web routes."""

from __future__ import annotations

import uuid
from pathlib import Path, PureWindowsPath
from typing import Any

from dlightrag.core.ingestion.paths import UPLOADS_DIR_NAME

CHUNK_SIZE = 1024 * 1024
IGNORED_UPLOAD_FILENAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


class UploadTooLargeError(ValueError):
    """Raised after a streamed upload exceeds its byte cap."""


def upload_batch_dir(input_root: Path) -> Path:
    """Return a fresh explicit upload batch directory under a workspace input root."""
    root = input_root / UPLOADS_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def safe_upload_relative_path(filename: str) -> Path:
    """Sanitize a browser-provided relative upload path."""
    if not filename or "\0" in filename:
        raise ValueError(f"Unsafe filename: {filename!r}")
    candidate = Path(filename)
    windows_candidate = PureWindowsPath(filename)
    if candidate.is_absolute() or windows_candidate.is_absolute() or windows_candidate.drive:
        raise ValueError(f"Unsafe filename: {filename!r}")
    parts = candidate.parts
    windows_parts = windows_candidate.parts
    if not parts or ".." in parts or ".." in windows_parts:
        raise ValueError(f"Unsafe filename: {filename!r}")
    if len(parts) == 1 and len(windows_parts) > 1:
        raise ValueError(f"Unsafe filename: {filename!r}")
    return Path(*parts)


def safe_upload_basename(filename: str) -> str:
    """Return a safe single filename, rejecting nested paths."""
    relative = safe_upload_relative_path(filename)
    if len(relative.parts) != 1:
        raise ValueError(f"Unsafe filename: {filename!r}")
    return relative.name


def safe_upload_destination(dest_dir: Path, filename: str) -> Path:
    """Return a destination under dest_dir, rejecting escapes after resolution."""
    root = dest_dir.resolve()
    dest = (root / safe_upload_relative_path(filename)).resolve()
    if not dest.is_relative_to(root):
        raise ValueError(f"Unsafe filename: {filename!r}")
    return dest


def ignored_upload(filename: str) -> bool:
    basename = filename.rsplit("/", 1)[-1]
    return basename in IGNORED_UPLOAD_FILENAMES or basename.startswith("._")


async def write_upload_stream(
    upload: Any,
    dest: Path,
    *,
    max_bytes: int,
    bytes_written: int = 0,
) -> int:
    """Stream one UploadFile-like object to disk and return the aggregate byte count."""
    import aiofiles  # noqa: PLC0415

    dest.parent.mkdir(parents=True, exist_ok=True)
    too_large = False
    async with aiofiles.open(dest, "wb") as out_file:
        while chunk := await upload.read(CHUNK_SIZE):
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                too_large = True
                break
            await out_file.write(chunk)
    if too_large:
        dest.unlink(missing_ok=True)
        raise UploadTooLargeError
    return bytes_written


__all__ = [
    "UploadTooLargeError",
    "ignored_upload",
    "safe_upload_basename",
    "safe_upload_destination",
    "safe_upload_relative_path",
    "upload_batch_dir",
    "write_upload_stream",
]
