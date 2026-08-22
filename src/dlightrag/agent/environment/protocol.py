# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The file-and-process port used by path tools, spill, and staging."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


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


class ExecutionEnvironment(Protocol):
    """Rooted filesystem and process host. Not a sandbox."""

    @property
    def root(self) -> Path:
        """Return the model-visible workspace directory."""
        ...

    def resolve(self, relative: str) -> Path:
        """Return a real path inside the root, or reject the model path."""
        ...

    def stat_kind(self, path: Path) -> str:
        """Return ``file``, ``directory``, or ``missing``."""
        ...

    def list_directory(self, path: Path) -> Sequence[DirectoryEntry]:
        """Return a sorted one-level listing."""
        ...

    def read_bytes(self, path: Path) -> bytes:
        """Return the file's bytes."""
        ...

    def write_bytes(self, path: Path, data: bytes) -> None:
        """Create parents and replace the file atomically."""
        ...

    async def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> CompletedProcess:
        """Run argv with an explicit environment and optional timeout."""
        ...

    def terminate_group(self, process: object) -> None:
        """Signal a process group with SIGTERM then SIGKILL."""
        ...
