# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Product-layer path policy for LightRAG ingestion.

LightRAG owns document canonicalization and parser sidecars. DlightRAG only
owns the source-to-staged-input boundary needed by REST/web/remote ingestion.
Keep those rules centralized here so service orchestration does not grow
ad-hoc directory handling.
"""

import hashlib
import logging
import os
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath

from lightrag.constants import PARSED_DIR_NAME

logger = logging.getLogger(__name__)

UPLOADS_DIR_NAME = "__uploads__"
REMOTE_INGEST_DIR_NAME = "__remote_ingest__"
REMOTE_SOURCES_DIR_NAME = "__remote_sources__"


def workspace_input_root(input_dir: Path, workspace: str) -> Path:
    """Return the persistent LightRAG input root for one workspace."""
    return input_dir / workspace


def iter_ingestable_files(path: Path) -> list[Path]:
    """Resolve a local ingest target into concrete source files.

    Broad directory scans skip LightRAG parser sidecars, dot-prefixed paths,
    remote ingest/source staging, and the ``__uploads__`` staging that earlier
    releases left under workspace inputs. A directory inside ``__uploads__`` stays
    ingestable when a caller names it explicitly.
    """
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Local ingest path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Local ingest path is not a file or directory: {path}")

    explicit_upload_batch = _is_explicit_upload_batch_dir(path)
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


def excluded_from_directory_scan(name: str, *, is_dir: bool) -> bool:
    """Whether a scan of a tree copied elsewhere skips this entry and all below it.

    A copy of a source tree keeps none of the source's ancestors, so the scan of
    the copy skips exactly this: dot-prefixed entries, parser sidecars, and remote
    ingest, remote source and ``__uploads__`` staging directories. A caller that
    snapshots a tree for ingestion can leave these out of the copy.
    """
    return name.startswith(".") or (
        is_dir
        and name
        in {PARSED_DIR_NAME, UPLOADS_DIR_NAME, REMOTE_INGEST_DIR_NAME, REMOTE_SOURCES_DIR_NAME}
    )


def _is_explicit_upload_batch_dir(path: Path) -> bool:
    """Return True for ``.../__uploads__/<batch>`` style explicit batch dirs."""
    return path.name != UPLOADS_DIR_NAME and UPLOADS_DIR_NAME in {p.name for p in path.parents}


def parser_input_path(input_root: Path, source: Path) -> Path:
    """Where LightRAG resolves the parser input of ``source``: ``input_root/<basename>``.

    LightRAG keeps only a document's basename (its document id is derived from it)
    and, when it parses, looks the file up by that name in its workspace input
    directory, before its other candidates: the workspace's ``__parsed__`` archive,
    the input root itself, and a bare name or an ``inputs`` folder relative to the
    current directory. A file anywhere below the workspace directory is never
    found, so every parser input is placed flat, here.
    """
    return input_root / source.name


def place_parser_input(source: Path, input_root: Path) -> Path:
    """Copy ``source`` to its flat parser-input path, unless it is already there.

    The copy replaces an earlier input of the same name atomically: that name is
    the same LightRAG document, which one Run at a time ingests in a Workspace.
    """
    target = parser_input_path(input_root, source)
    if target.exists() and os.path.samefile(source, target):
        return target
    input_root.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def discard_parser_input(parser_path: Path) -> None:
    """Remove a transient parser input and the copy LightRAG archived beside it.

    The parser sidecar (``__parsed__/<name>.parsed/``) stays: retrieval reads it.
    """
    for candidate in (parser_path, lightrag_archived_source_path(parser_path)):
        try:
            if candidate.is_file():
                candidate.unlink()
        except OSError:
            logger.debug("Failed to remove parser input: %s", candidate, exc_info=True)


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
