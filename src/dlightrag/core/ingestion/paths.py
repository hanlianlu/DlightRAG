# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Product-layer path policy for LightRAG ingestion.

LightRAG owns document canonicalization and parser sidecars. DlightRAG only
owns the source-to-staged-input boundary needed by REST/web/remote ingestion.
Keep those rules centralized here so service orchestration does not grow
ad-hoc directory handling.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path, PurePosixPath

from lightrag.constants import PARSED_DIR_NAME

UPLOADS_DIR_NAME = "__uploads__"


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


def remote_namespace_root(
    *,
    working_dir: Path,
    source_type: str,
    namespace: str,
) -> Path:
    """Return the managed local cache root for one remote source namespace."""
    return working_dir / "sources" / source_type / _safe_namespace(namespace)


def remote_object_path(
    *,
    working_dir: Path,
    source_type: str,
    namespace: str,
    key: str,
) -> Path:
    """Return a managed local path for a remote object key."""
    parts = [part for part in PurePosixPath(key).parts if part not in {"", ".", ".."}]
    if not parts:
        raise ValueError("remote object key is empty")
    return remote_namespace_root(
        working_dir=working_dir,
        source_type=source_type,
        namespace=namespace,
    ) / Path(*parts)


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
    return True


def _safe_namespace(namespace: str) -> str:
    return namespace.replace("/", "_").replace("\\", "_") or "default"
