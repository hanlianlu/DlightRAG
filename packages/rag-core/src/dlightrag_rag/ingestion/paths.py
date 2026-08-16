# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Product-layer path policy for LightRAG ingestion.

LightRAG owns document canonicalization and parser sidecars. DlightRAG only
owns the source-to-staged-input boundary needed by REST/web/remote ingestion.
Keep those rules centralized here so service orchestration does not grow
ad-hoc directory handling.
"""

import hashlib
import os
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath

from lightrag.constants import PARSED_DIR_NAME

UPLOADS_DIR_NAME = "__uploads__"
REMOTE_INGEST_DIR_NAME = "__remote_ingest__"
REMOTE_SOURCES_DIR_NAME = "__remote_sources__"


def workspace_input_root(input_dir: Path, workspace: str) -> Path:
    """Return the persistent LightRAG input root for one workspace."""
    return input_dir / workspace


def iter_ingestable_files(path: Path) -> list[Path]:
    """Resolve a local ingest target into concrete source files.

    Broad directory scans skip LightRAG parser sidecars and DlightRAG web upload
    staging. Explicit web upload batch directories remain ingestable because the
    upload route passes ``.../__uploads__/<batch-id>`` directly.
    """
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Local ingest path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Local ingest path is not a file or directory: {path}")

    explicit_upload_batch = is_explicit_upload_batch_dir(path)
    files = [
        item
        for item in sorted(
            (p for p in path.rglob("*") if p.is_file()),
            key=lambda p: p.relative_to(path).as_posix(),
        )
        if _is_ingestable_child(item, scan_root=path, explicit_upload_batch=explicit_upload_batch)
    ]
    if not files:
        raise ValueError(f"Local ingest directory contains no files: {path}")
    return files


def is_explicit_upload_batch_dir(path: Path) -> bool:
    """Return True for ``.../__uploads__/<batch>`` style explicit batch dirs."""
    return path.name != UPLOADS_DIR_NAME and UPLOADS_DIR_NAME in {p.name for p in path.parents}


def staged_input_path(
    *,
    input_root: Path,
    file_path: Path,
    relative_to: Path | None = None,
) -> Path:
    """Return where a source file should live under the workspace input root."""
    if relative_to is None:
        relative_path = Path(file_path.name)
    else:
        try:
            relative_path = file_path.resolve().relative_to(relative_to.resolve())
        except ValueError:
            relative_path = Path(file_path.name)
        if not relative_path.parts:
            relative_path = Path(file_path.name)
    return input_root / relative_path


def stage_input_file(
    *,
    input_root: Path,
    file_path: Path,
    relative_to: Path | None = None,
) -> Path:
    """Copy a source file into LightRAG's persistent workspace input root."""
    source = file_path.resolve()
    target = staged_input_path(input_root=input_root, file_path=file_path, relative_to=relative_to)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.resolve() == source:
        return target

    tmp_target = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    shutil.copy2(source, tmp_target)
    os.replace(tmp_target, target)
    return target


def remote_ingest_batch_root(
    *,
    input_root: Path,
    source_type: str,
    batch_id: str,
) -> Path:
    """Return the remote parser input root for one ingest batch.

    The directory lives under the workspace input root because LightRAG writes
    parser sidecars relative to the source file parent. DlightRAG removes the
    temporary source file after parsing and keeps only the generated artifacts.
    """
    return input_root / REMOTE_INGEST_DIR_NAME / source_type / batch_id


def remote_parser_input_path(
    *,
    batch_root: Path,
    source_uri: str,
    key: str,
) -> Path:
    """Return an extension-preserving, URI-stable parser input path.

    LightRAG 1.5 pending-parse APIs still require local files and derive doc
    IDs from canonicalized file names. Hashing the full remote URI avoids
    collisions for same-basename objects in different prefixes while keeping
    parser routing extension-based.
    """
    return batch_root / _remote_source_filename(source_uri=source_uri, key=key)


def retained_remote_source_path(
    *,
    input_root: Path,
    source_type: str,
    source_uri: str,
    key: str,
) -> Path:
    """Return the persistent workspace path for a retained remote source file."""
    return (
        input_root
        / REMOTE_SOURCES_DIR_NAME
        / _safe_filename_stem(source_type)
        / _remote_source_filename(source_uri=source_uri, key=key)
    )


def lightrag_archived_source_path(source_path: Path) -> Path:
    """Return LightRAG's deterministic post-parse location for a source file."""
    path = Path(source_path)
    if path.parent.name == PARSED_DIR_NAME:
        return path
    return path.parent / PARSED_DIR_NAME / path.name


def _is_ingestable_child(
    item: Path,
    *,
    scan_root: Path,
    explicit_upload_batch: bool,
) -> bool:
    relative_parts = item.relative_to(scan_root).parts
    parent_names = {p.name for p in item.parents}
    if PARSED_DIR_NAME in parent_names:
        return False
    if not explicit_upload_batch and UPLOADS_DIR_NAME in parent_names:
        return False
    if any(part.startswith(".") for part in relative_parts):
        return False
    if REMOTE_INGEST_DIR_NAME in parent_names or REMOTE_SOURCES_DIR_NAME in parent_names:
        return False
    return True


def _remote_source_filename(*, source_uri: str, key: str) -> str:
    parts = [part for part in PurePosixPath(key).parts if part not in {"", ".", ".."}]
    if not parts:
        raise ValueError("remote object key is empty")
    filename = PurePosixPath(*parts).name
    suffix = Path(filename).suffix.lower()
    stem = Path(filename).stem or "document"
    safe_stem = _safe_filename_stem(stem)
    digest = hashlib.sha256(source_uri.encode("utf-8")).hexdigest()[:12]
    return f"{safe_stem}__{digest}{suffix}"


def _safe_filename_stem(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return cleaned[:96] or "document"
