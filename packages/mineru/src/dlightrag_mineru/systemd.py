# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Linux / WSL2 ``systemd --user`` units for the MinerU sidecar.

Mirrors :mod:`dlightrag_mineru.launchd` and the shell manager it replaces
(``systemd_service.sh``): same unit names and text, the same write ->
``daemon-reload`` -> ``enable --now`` ordering, and the same refusal to guess when
``systemctl`` is absent (a container, or a WSL2 distribution without systemd).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from dlightrag_mineru.service import ServiceSpec

LOGS_TAIL_LINES = "200"

SYSTEMCTL_MISSING = (
    "systemctl not found; cannot manage a background service.\n"
    "Run the sidecar in the foreground instead:  dlightrag-mineru api"
)


def unit_directory() -> Path:
    override = os.environ.get("XDG_CONFIG_HOME")
    base = Path(override).expanduser() if override else Path.home() / ".config"
    return base / "systemd" / "user"


def unit_path(spec: ServiceSpec) -> Path:
    return unit_directory() / spec.unit


def quote(value: str) -> str:
    """A unit-file value, quoted and escaped.

    A venv or env-file path comes from `.env.mineru`, so it is untrusted input: an
    unquoted space injects an argument into ``ExecStart``, and a newline injects a
    whole directive. Everything interpolated here goes through this, and a control
    character is refused rather than escaped.
    """
    if any(character in value for character in "\n\r\0"):
        msg = f"a service path may not contain a control character: {value!r}"
        raise ValueError(msg)
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'"{escaped}"'


def path_value(value: str) -> str:
    """A unit-file path setting, such as ``WorkingDirectory=``.

    A path setting takes the rest of the line as the path, spaces included, and does not
    unquote: a quoted value reads as a relative path that begins with ``"``. So the path
    goes in bare, with only the specifier ``%`` escaped, and the same control characters
    are refused as for a quoted value.
    """
    if any(character in value for character in "\n\r\0"):
        msg = f"a service path may not contain a control character: {value!r}"
        raise ValueError(msg)
    if not value.startswith("/"):
        msg = f"a service path must be absolute: {value!r}"
        raise ValueError(msg)
    return value.replace("%", "%%")


def render(spec: ServiceSpec) -> str:
    """The unit for this service, textually the unit the shell manager wrote."""
    program = " ".join(quote(argument) for argument in spec.program)
    return (
        "[Unit]\n"
        f"Description={spec.description}\n"
        "After=network.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"WorkingDirectory={path_value(str(spec.working_directory))}\n"
        f"Environment=MINERU_ENV_FILE={quote(str(spec.env_file))}\n"
        f"ExecStart={program}\n"
        "Restart=on-failure\n"
        "RestartSec=3\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def run(action: str, spec: ServiceSpec) -> int:
    handlers = {
        "install": _install,
        "start": _start,
        "stop": _stop,
        "status": _status,
        "logs": _logs,
        "uninstall": _uninstall,
    }
    if not _require_systemctl():
        return 127
    return handlers[action](spec)


def _install(spec: ServiceSpec) -> int:
    _disable_now(spec.unit)
    (unit_directory() / spec.unit).unlink(missing_ok=True)
    path = unit_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(spec), encoding="utf-8")
    code = _systemctl("daemon-reload")
    if code != 0:
        return code
    return _systemctl("enable", "--now", spec.unit)


def _start(spec: ServiceSpec) -> int:
    return _systemctl("start", spec.unit)


def _stop(spec: ServiceSpec) -> int:
    _systemctl("stop", spec.unit, quiet=True)
    return 0


def _status(spec: ServiceSpec) -> int:
    return _systemctl("status", spec.unit)


def _logs(spec: ServiceSpec) -> int:
    argv = [
        "journalctl",
        "--user",
        "-u",
        spec.unit,
        "-n",
        LOGS_TAIL_LINES,
        "-f",
    ]
    os.execvp(argv[0], argv)  # noqa: S606 - argv, no shell
    return 127  # unreachable: execvp replaces this process


def _uninstall(spec: ServiceSpec) -> int:
    _disable_now(spec.unit)
    (unit_directory() / spec.unit).unlink(missing_ok=True)
    return _systemctl("daemon-reload")


def _disable_now(unit: str) -> int:
    return _systemctl("disable", "--now", unit, quiet=True)


def _require_systemctl() -> bool:
    if shutil.which("systemctl") is not None:
        return True
    print(SYSTEMCTL_MISSING, file=sys.stderr)
    return False


def _systemctl(*arguments: str, quiet: bool = False) -> int:
    completed = subprocess.run(  # noqa: S603 - fixed systemctl argv
        ["systemctl", "--user", *arguments],  # noqa: S607 - systemctl resolves on PATH by design
        check=False,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )
    return completed.returncode
