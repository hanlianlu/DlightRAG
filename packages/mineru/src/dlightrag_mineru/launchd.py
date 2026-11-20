# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""macOS launchd user agent for the MinerU sidecar.

The artifacts and command sequences match the shell manager this replaces
(``launch_agent.sh``): same label, same plist keys, same log paths, and the same
retry around ``bootstrap``, because ``bootout`` returns before the old job has
finished tearing down and re-installing over a running agent otherwise fails with
"5: Input/output error".
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from dlightrag_mineru.service import ServiceSpec

BOOTSTRAP_ATTEMPTS = 5
BOOTSTRAP_BACKOFF_SECONDS = 0.5
LOGS_TAIL_LINES = "200"


def launchd_home() -> Path:
    """The home whose ``Library`` holds the agents and the logs.

    ``MINERU_LAUNCHD_HOME`` moves both together, which is what lets a test install an
    agent into a temporary home — and what the shell managed with the same variable.
    """
    override = os.environ.get("MINERU_LAUNCHD_HOME")
    return Path(override).expanduser() if override else Path.home()


def agents_directory() -> Path:
    return launchd_home() / "Library" / "LaunchAgents"


def plist_path(spec: ServiceSpec) -> Path:
    return agents_directory() / f"{spec.label}.plist"


def domain() -> str:
    return f"gui/{os.getuid()}"


def render(spec: ServiceSpec) -> bytes:
    """The plist for this service, byte-for-byte the plist the shell manager wrote."""
    payload = {
        "Label": spec.label,
        "ProgramArguments": list(spec.program),
        "WorkingDirectory": str(spec.working_directory),
        "RunAtLoad": True,
        "KeepAlive": True,
        "EnvironmentVariables": {"MINERU_ENV_FILE": str(spec.env_file)},
        "StandardOutPath": str(spec.stdout_log),
        "StandardErrorPath": str(spec.stderr_log),
    }
    return plistlib.dumps(payload)


def run(action: str, spec: ServiceSpec) -> int:
    handlers = {
        "install": _install,
        "start": _start,
        "stop": _stop,
        "status": _status,
        "logs": _logs,
        "uninstall": _uninstall,
    }
    return handlers[action](spec)


def _install(spec: ServiceSpec) -> int:
    _bootout(spec.label)
    path = plist_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render(spec))
    return _bootstrap_with_retry(path)


def _start(spec: ServiceSpec) -> int:
    return _bootstrap_with_retry(plist_path(spec))


def _stop(spec: ServiceSpec) -> int:
    _bootout(spec.label)
    return 0


def _status(spec: ServiceSpec) -> int:
    """Report the job's state through the exit code.

    The shell manager piped this through ``|| true``, which made a status check that
    cannot fail; systemd's status always propagated, so this also ends the asymmetry.
    """
    return _launchctl("print", f"{domain()}/{spec.label}")


def _logs(spec: ServiceSpec) -> int:
    spec.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "tail",
        "-n",
        LOGS_TAIL_LINES,
        "-f",
        str(spec.stdout_log),
        str(spec.stderr_log),
    ]
    os.execvp(argv[0], argv)  # noqa: S606 - argv, no shell
    return 127  # unreachable: execvp replaces this process


def _uninstall(spec: ServiceSpec) -> int:
    _bootout(spec.label)
    plist_path(spec).unlink(missing_ok=True)
    return 0


def _bootstrap_with_retry(plist: Path) -> int:
    """Five quiet attempts with backoff, then one attempt that surfaces the error."""
    for attempt in range(BOOTSTRAP_ATTEMPTS):
        if _bootstrap(plist, quiet=True) == 0:
            return 0
        if attempt < BOOTSTRAP_ATTEMPTS - 1:
            time.sleep(BOOTSTRAP_BACKOFF_SECONDS)
    return _bootstrap(plist, quiet=False)


def _bootstrap(plist: Path, *, quiet: bool) -> int:
    return _launchctl("bootstrap", domain(), str(plist), quiet=quiet)


def _bootout(label: str) -> int:
    return _launchctl("bootout", f"{domain()}/{label}", quiet=True)


def _launchctl(*arguments: str, quiet: bool = False) -> int:
    suppressed = subprocess.DEVNULL if quiet else None
    completed = subprocess.run(  # noqa: S603 - fixed launchctl argv
        ["launchctl", *arguments],  # noqa: S607 - launchctl resolves on PATH by design
        check=False,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=suppressed,
    )
    return completed.returncode


def usage() -> int:
    print("Usage: launchd service {install|start|stop|status|logs|uninstall}", file=sys.stderr)
    return 64
