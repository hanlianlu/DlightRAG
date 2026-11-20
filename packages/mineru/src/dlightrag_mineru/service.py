# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The background service: one specification, two platform renderers.

The service supervises the launcher installed inside the service venv
(:func:`dlightrag_mineru.install.launcher_command`), never this tool: launchd's
``KeepAlive`` and systemd's ``Restart=on-failure`` must keep acting on the process
that parses documents, and a manager in the middle would change what "restart"
means.

Behaviour here is deliberately the same as the shell component it replaces, apart
from the removed Gradio WebUI and the deviations ADR 0054 records: the same labels,
units, ordering, log locations and exit codes. ``tests/unit/test_mineru_package.py`` holds both the rendered
artifacts and the command sequences to that.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from dlightrag_mineru import launchd, systemd

SERVICE_NAME = "api"
SERVICE_DESCRIPTION = "DlightRAG MinerU API sidecar"
LABEL_PREFIX = "com.hanlianlyu.dlightrag.mineru-"
UNIT_PREFIX = "dlightrag-mineru-"

ACTIONS = ("install", "start", "stop", "status", "logs", "uninstall")

# Actions that need the launcher the installer placed in the venv.
NEEDS_LAUNCHER = ("install", "start")

USAGE = "Usage: dlightrag-mineru service {install|start|stop|status|logs|uninstall}"

LAUNCHER_MISSING = """the MinerU service env is not installed at {environment}.

Install it first:
  dlightrag-mineru install
"""

UNSUPPORTED_OS = (
    "Unsupported OS for a MinerU background service: {system}.\n"
    "Run the sidecar in the foreground instead:  dlightrag-mineru api"
)


@dataclass(frozen=True)
class ServiceSpec:
    """Everything a platform renderer needs, and nothing platform-specific."""

    name: str
    description: str
    environment: Path
    program: tuple[str, ...]
    working_directory: Path
    env_file: Path
    stdout_log: Path
    stderr_log: Path

    @property
    def label(self) -> str:
        return f"{LABEL_PREFIX}{self.name}"

    @property
    def unit(self) -> str:
        return f"{UNIT_PREFIX}{self.name}.service"


def spec(
    *,
    environment: Path,
    working_directory: Path,
    env_file: Path,
    launchd_home: Path | None = None,
) -> ServiceSpec:
    """The API service for this sidecar venv."""
    home = launchd_home if launchd_home is not None else _launchd_home()
    log_directory = home / "Library" / "Logs" / "dlightrag"
    return ServiceSpec(
        name=SERVICE_NAME,
        description=SERVICE_DESCRIPTION,
        environment=environment,
        program=install_launcher(environment),
        working_directory=working_directory,
        env_file=env_file,
        stdout_log=log_directory / f"mineru-{SERVICE_NAME}.out.log",
        stderr_log=log_directory / f"mineru-{SERVICE_NAME}.err.log",
    )


def install_launcher(environment: Path) -> tuple[str, ...]:
    """Imported lazily so this module stays importable without the install half."""
    from dlightrag_mineru.install import launcher_command

    return launcher_command(environment)


def platform_module(system: str | None = None):
    """The renderer for this OS, or ``None`` where a user service cannot run."""
    name = platform.system() if system is None else system
    if name == "Darwin":
        return launchd
    if name == "Linux":
        return systemd
    return None


def run(action: str, *, spec: ServiceSpec, system: str | None = None) -> int:
    """Dispatch one service action; returns the exit code to use."""
    if action not in ACTIONS:
        print(USAGE, file=sys.stderr)
        return 64
    # Bringing a service up needs the launcher the installer placed inside the venv.
    # Tearing one down must keep working when the venv is already gone, which is the
    # case an uninstall is usually run in.
    if action in NEEDS_LAUNCHER and not Path(spec.program[-1]).is_file():
        print(LAUNCHER_MISSING.format(environment=spec.environment), file=sys.stderr)
        return 127
    module = platform_module(system)
    if module is None:
        name = platform.system() if system is None else system
        print(UNSUPPORTED_OS.format(system=name), file=sys.stderr)
        return 64
    return module.run(action, spec)


def _launchd_home() -> Path:
    """One definition of that home, shared with the renderer that writes into it."""
    return launchd.launchd_home()
