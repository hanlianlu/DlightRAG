# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Upload staging helpers shared by REST and Web routes."""

import uuid
from pathlib import Path, PureWindowsPath

from dlightrag.engine.rag.corpus.ingestion.paths import UPLOADS_DIR_NAME


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


__all__ = [
    "UploadTooLargeError",
    "safe_upload_basename",
    "safe_upload_relative_path",
    "upload_batch_dir",
]
