# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Execution modes and the trusted adapter boundary.

A Run's Agent either has an environment or it does not. The kernel defines the seam
and ships the local adapter, which confines every Agent process to its Agent
Workspace under the deployment's policy; isolation stronger than the host kernel is
the environment the application is deployed in, not a mode (ADR 0024).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol
from weakref import WeakSet

from dlightrag.engine.agent.environment.confinement import ConfinementPolicy
from dlightrag.engine.agent.environment.local import (
    CompletedProcess,
    DirectoryEntry,
    LocalExecutionEnvironment,
    ProcessOutputSink,
)

type ExecutionMode = Literal["disabled", "trust"]


class ExecutionEnvironment(Protocol):
    """The filesystem/process operations base tools may request."""

    @property
    def root(self) -> Path: ...

    @property
    def integrity_violations(self) -> tuple[str, ...]: ...

    @property
    def quota_violation(self) -> str | None: ...

    def refresh_integrity(self) -> tuple[str, ...]: ...

    def prepare_process_directories(self) -> tuple[Path, Path]: ...

    def resolve(self, relative: str) -> Path: ...

    def stat_kind(self, path: Path) -> str: ...

    def list_directory(self, path: Path) -> tuple[DirectoryEntry, ...]: ...

    def read_bytes(self, path: Path) -> bytes: ...

    def write_bytes(self, path: Path, data: bytes) -> None: ...

    async def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
        on_output: ProcessOutputSink | None = None,
    ) -> CompletedProcess: ...


class ExecutionEnvironmentAdapter(Protocol):
    """Trusted host seam that binds one already-admitted workspace."""

    def create(self, workspace: Path, *, owner_id: str | None = None) -> ExecutionEnvironment: ...

    async def aclose(self) -> None: ...


class TrustExecutionAdapter:
    """Bind DlightRAG's rooted host environment under a confinement policy.

    The policy is required: it is what makes ``trust`` mean the deployment is
    trusted rather than the Agent being unconfined (ADR 0024). Tests that exercise
    the environment's own mechanics build a ``LocalExecutionEnvironment`` directly.
    """

    def __init__(self, confinement: ConfinementPolicy) -> None:
        self._confinement = confinement
        self._environments: WeakSet[LocalExecutionEnvironment] = WeakSet()
        self._closed = False

    def create(self, workspace: Path, *, owner_id: str | None = None) -> LocalExecutionEnvironment:
        if self._closed:
            raise RuntimeError("execution adapter is closed")
        environment = LocalExecutionEnvironment(
            workspace,
            confinement=self._confinement.for_workspace(workspace, owner_id=owner_id),
        )
        self._environments.add(environment)
        return environment

    async def aclose(self) -> None:
        self._closed = True
        await asyncio.gather(*(environment.aclose() for environment in tuple(self._environments)))


def resolve_execution_adapter(
    mode: ExecutionMode,
    *,
    confinement: ConfinementPolicy,
) -> ExecutionEnvironmentAdapter | None:
    """Resolve one mode: an Agent either has an environment or it does not."""
    if mode == "disabled":
        return None
    return TrustExecutionAdapter(confinement)


__all__ = [
    "ExecutionEnvironment",
    "ExecutionEnvironmentAdapter",
    "ExecutionMode",
    "TrustExecutionAdapter",
    "resolve_execution_adapter",
]
