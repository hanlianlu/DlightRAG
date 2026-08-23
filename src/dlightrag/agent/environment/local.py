# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Local trusted execution: rooted files plus explicit child processes."""

from __future__ import annotations

import asyncio
import os
import signal
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dlightrag.agent.environment.errors import (
    WORKSPACE_MAX_BYTES,
    WORKSPACE_MAX_ENTRIES,
    PathRejected,
    WorkspaceQuotaExceeded,
)


@dataclass(frozen=True, slots=True)
class DirectoryEntry:
    """One listing row: relative name, type, and size in bytes."""

    name: str
    kind: str
    size: int


@dataclass(frozen=True, slots=True)
class CompletedProcess:
    """One child-process result: exit code and captured text."""

    returncode: int
    stdout: str
    stderr: str


class LocalExecutionEnvironment:
    """POSIX workspace rooted at one directory. Not a security boundary."""

    def __init__(self, root: Path) -> None:
        resolved = root.expanduser().resolve()
        if not resolved.is_absolute():
            raise ValueError("execution environment root must be absolute")
        resolved.mkdir(parents=True, exist_ok=True)
        self._root = resolved

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, relative: str) -> Path:
        candidate = relative.strip()
        if not candidate or candidate.startswith(("/", "~")) or "\x00" in candidate:
            raise PathRejected("path must be a relative workspace path")
        parts = Path(candidate).parts
        if any(part in {".", ".."} for part in parts if part == ".."):
            raise PathRejected("path must not escape the workspace")
        if ".." in parts:
            raise PathRejected("path must not escape the workspace")
        current = self._root
        for part in parts:
            current = current / part
            if current.is_symlink():
                target = current.resolve()
                if not target.is_relative_to(self._root):
                    raise PathRejected("path must not follow a symlink out of the workspace")
        resolved = (self._root / candidate).resolve()
        if not resolved.is_relative_to(self._root):
            raise PathRejected("path must stay inside the workspace")
        if resolved.exists() and not (resolved.is_file() or resolved.is_dir()):
            raise PathRejected("path must name a regular file or directory")
        if resolved.exists() and stat.S_ISLNK(resolved.lstat().st_mode) is False:
            mode = resolved.lstat().st_mode
            if (
                stat.S_ISFIFO(mode)
                or stat.S_ISCHR(mode)
                or stat.S_ISBLK(mode)
                or stat.S_ISSOCK(mode)
            ):
                raise PathRejected("path must name a regular file or directory")
        return resolved

    def stat_kind(self, path: Path) -> str:
        if not path.exists():
            return "missing"
        if path.is_dir():
            return "directory"
        if path.is_file():
            return "file"
        raise PathRejected("path must name a regular file or directory")

    def list_directory(self, path: Path) -> tuple[DirectoryEntry, ...]:
        if not path.is_dir():
            raise PathRejected("directory listing requires a directory")
        entries: list[DirectoryEntry] = []
        for child in sorted(path.iterdir(), key=lambda item: item.name):
            if child.is_symlink():
                raise PathRejected("workspace listing rejects symbolic links")
            kind = "directory" if child.is_dir() else "file"
            size = child.stat().st_size if child.is_file() else 0
            entries.append(DirectoryEntry(name=child.name, kind=kind, size=size))
        return tuple(entries)

    def read_bytes(self, path: Path) -> bytes:
        if not path.is_file():
            raise PathRejected("read requires a regular file")
        return path.read_bytes()

    def write_bytes(self, path: Path, data: bytes) -> None:
        if path.exists() and path.is_dir():
            raise PathRejected("cannot overwrite a directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._check_quota(path, len(data))
        fd, tmp_name = tempfile.mkstemp(prefix=".dlightrag-write-", dir=path.parent)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    async def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> CompletedProcess:
        if not argv:
            raise ValueError("process argv cannot be empty")
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd or self._root),
            env=dict(env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout_seconds
            )
        except TimeoutError:
            self._terminate_group(process)
            stdout_bytes, stderr_bytes = await process.communicate()
            return CompletedProcess(
                returncode=process.returncode or -signal.SIGKILL,
                stdout=_decode_output(stdout_bytes),
                stderr=_decode_output(stderr_bytes),
            )
        except asyncio.CancelledError:
            self._terminate_group(process)
            await asyncio.shield(process.communicate())
            raise
        return CompletedProcess(
            returncode=process.returncode or 0,
            stdout=_decode_output(stdout_bytes),
            stderr=_decode_output(stderr_bytes),
        )

    def _terminate_group(self, process: object) -> None:
        pid = getattr(process, "pid", None)
        if pid is None:
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return

    def _check_quota(self, destination: Path, new_size: int) -> None:
        entries = 0
        total = 0
        for current, dirnames, filenames in os.walk(self._root):
            entries += len(dirnames) + len(filenames)
            for name in filenames:
                file_path = Path(current) / name
                if file_path == destination:
                    continue
                try:
                    total += file_path.stat().st_size
                except OSError:
                    continue
        if destination.exists() and destination.is_file():
            entries -= 1
        entries += 1
        total += new_size
        if entries > WORKSPACE_MAX_ENTRIES or total > WORKSPACE_MAX_BYTES:
            raise WorkspaceQuotaExceeded("workspace quota exceeded")


def _decode_output(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


__all__ = ["CompletedProcess", "DirectoryEntry", "LocalExecutionEnvironment"]
