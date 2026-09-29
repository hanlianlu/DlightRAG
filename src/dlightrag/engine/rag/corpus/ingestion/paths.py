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
from lightrag.utils_pipeline import normalize_document_file_path

logger = logging.getLogger(__name__)

UPLOADS_DIR_NAME = "__uploads__"
REMOTE_INGEST_DIR_NAME = "__remote_ingest__"
REMOTE_SOURCES_DIR_NAME = "__remote_sources__"
#: Where a workspace's Run stages live in its corpus directory, one folder per Run.
RUN_STAGES_DIR_NAME = ".runs"


def workspace_input_root(input_dir: Path, workspace: str) -> Path:
    """Return the persistent LightRAG input root for one workspace."""
    return input_dir / workspace


def document_name(filename: str | Path) -> str:
    """The name LightRAG stores a document under, and derives its id from."""
    return normalize_document_file_path(Path(filename).name)


#: The folders a Workspace's corpus directory keeps beside its documents: LightRAG's
#: parser archive, fetched remote sources, and staging folders earlier releases wrote.
_CORPUS_FOLDER_NAMES = frozenset(
    {PARSED_DIR_NAME, UPLOADS_DIR_NAME, REMOTE_INGEST_DIR_NAME, REMOTE_SOURCES_DIR_NAME}
)


def reserved_corpus_name(name: str) -> bool:
    """Whether a file of this name would take an entry the corpus directory keeps.

    Dot entries there are Run stages and temporary copies; the rest are its own
    folders. A document of such a name is refused, and a folder listing or a folder
    upload skips such an entry, and everything below it.
    """
    return name.startswith(".") or name in _CORPUS_FOLDER_NAMES


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
    the same LightRAG document, which one Run at a time ingests in a Workspace. A
    name the corpus directory keeps for itself is refused: the copy would take
    that entry's place.
    """
    if reserved_corpus_name(source.name):
        raise ValueError(f"a document cannot be named {source.name!r}: the name is reserved")
    target = parser_input_path(input_root, source)
    if target.exists() and os.path.samefile(source, target):
        return target
    input_root.mkdir(parents=True, exist_ok=True)
    # Named apart from the document, so any name that fits the folder fits the copy.
    temporary = input_root / f".{uuid.uuid4().hex}.part"
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def clear_archived_source(parser_input: Path) -> None:
    """Remove the file that holds LightRAG's archive name for ``parser_input``.

    LightRAG archives a parsed input to ``__parsed__/<name>``, or to
    ``__parsed__/<stem>_001<ext>`` while that name is taken, and a document's
    locator names the former. Right before an input is enqueued, what holds that
    name is an earlier version of the same document, already deleted, or an
    orphan; when it was the input's own source, it was already copied out.
    """
    lightrag_archived_source_path(parser_input).unlink(missing_ok=True)


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


def remote_parser_input_path(
    *,
    input_root: Path,
    source_uri: str,
    key: str,
) -> Path:
    """Return the extension-preserving, URI-stable flat parser input of a download.

    LightRAG 1.5 pending-parse APIs still require local files and derive doc
    IDs from canonicalized file names. Hashing the full remote URI avoids
    collisions for same-basename objects in different prefixes while keeping
    parser routing extension-based.
    """
    return input_root / _remote_source_filename(source_uri=source_uri, key=key)


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
