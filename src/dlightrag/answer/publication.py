# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Scan staged workspace artifacts and decide empty-answer / primary report."""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ArtifactKind = Literal["primary_report", "published_artifact"]
PRIMARY_REPORT_NAME = "report.md"


class PublicationScanError(ValueError):
    """artifacts/ contained a symlink, special file, or unreadable path."""


@dataclass(frozen=True, slots=True)
class StagedArtifact:
    """One regular file under artifacts/ that may be published."""

    relative_path: str
    kind: ArtifactKind
    media_type: str
    size_bytes: int
    path: Path


def is_substantive_text(text: str) -> bool:
    """True when citation-cleaned text still has a non-whitespace character."""
    return bool(text.strip())


def is_empty_answer(*, answer: str, has_primary_report: bool) -> bool:
    """Fail closed only when there is neither a report nor visible answer text."""
    return not has_primary_report and not is_substantive_text(answer)


def scan_artifact_directory(artifacts_root: Path) -> tuple[StagedArtifact, ...]:
    """List publishable regular files. Blank report.md is omitted."""
    if not artifacts_root.exists():
        return ()
    if artifacts_root.is_symlink() or not artifacts_root.is_dir():
        raise PublicationScanError("artifacts path must be a real directory")
    staged: list[StagedArtifact] = []
    for path in sorted(artifacts_root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        _reject_special(path)
        relative = path.relative_to(artifacts_root).as_posix()
        if relative == PRIMARY_REPORT_NAME and not _report_has_body(path):
            continue
        kind: ArtifactKind = (
            "primary_report" if relative == PRIMARY_REPORT_NAME else "published_artifact"
        )
        staged.append(
            StagedArtifact(
                relative_path=relative,
                kind=kind,
                media_type=_media_type(relative),
                size_bytes=path.stat().st_size,
                path=path,
            )
        )
    return tuple(staged)


def _report_has_body(path: Path) -> bool:
    try:
        return is_substantive_text(path.read_text(encoding="utf-8"))
    except OSError, UnicodeDecodeError:
        return False


def _reject_special(path: Path) -> None:
    if path.is_symlink():
        raise PublicationScanError(f"refusing to publish symlink {path}")
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise PublicationScanError(f"refusing to publish special file {path}")


def _media_type(relative: str) -> str:
    if relative.endswith(".md"):
        return "text/markdown"
    if relative.endswith(".txt"):
        return "text/plain"
    if relative.endswith(".json"):
        return "application/json"
    if relative.endswith(".csv"):
        return "text/csv"
    return "application/octet-stream"


__all__ = [
    "PRIMARY_REPORT_NAME",
    "ArtifactKind",
    "PublicationScanError",
    "StagedArtifact",
    "is_empty_answer",
    "is_substantive_text",
    "scan_artifact_directory",
]
