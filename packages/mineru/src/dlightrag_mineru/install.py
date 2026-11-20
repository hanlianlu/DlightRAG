# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Create the service virtualenv, install MinerU into it, and place the launcher.

The venv is the service's whole runtime. Everything the background service needs is
written *inside* it -- the interpreter-startup shim, the env reader, and the launcher
that execs MinerU -- so launchd and systemd point at absolute paths in a directory
that has nothing to do with the application's environment (ADR 0054).
"""

from __future__ import annotations

import importlib.resources
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dlightrag_mineru import env, version_range

# The values this command answers to, environment first, file second.
INSTALL_KEYS = (
    "MINERU_SERVICE_VENV",
    "MINERU_PYTHON",
    "MINERU_VERSION",
    "MINERU_MIN_VERSION",
    "MINERU_MAX_VERSION",
    "MINERU_INSTALL_EXTRAS",
)

# MinerU supports CPython 3.10-3.13 only. Pin the sidecar venv to a compatible
# interpreter (uv downloads it if missing) so a host whose default python is 3.14+
# still resolves wheels -- onnxruntime ships no cp314 build.
DEFAULT_PYTHON = "3.13"

# Inside the venv: what the background service runs.
SHIM_DIRECTORY = "dlightrag-mineru"
LAUNCHER_NAME = "launcher.py"
ENV_MODULE_NAME = "_env.py"
ENV_SOURCE_NAME = "env.py"
LAUNCHER_SOURCE = "_shim/launcher.py"
# Inside site-packages: the interpreter-startup guards, imported by a .pth at every
# start of this venv's interpreter — not only when this launcher starts it (ADR 0054).
GUARD_MODULE = "_dlightrag_mineru_shim.py"
GUARD_SOURCE = "_shim/_dlightrag_mineru_shim.py"
PTH_NAME = "dlightrag-mineru.pth"
MANIFEST_NAME = "installed.json"
PTH_CONTENT = "import _dlightrag_mineru_shim\n"

UV_MISSING = "uv was not found on PATH; install uv before installing the MinerU service env."


class InstallError(RuntimeError):
    """The install cannot place what the service needs."""


@dataclass(frozen=True)
class Refusal:
    """Why this install must not proceed, and the exit code that says so."""

    message: str
    code: int = 64


@dataclass(frozen=True)
class InstallPlan:
    """What this install will do, resolved before anything is written."""

    environment: Path
    python_version: str
    package: str
    family: str
    client_source: Path | None

    @property
    def interpreter(self) -> Path:
        return self.environment / "bin" / "python"

    @property
    def launcher(self) -> Path:
        return self.environment / SHIM_DIRECTORY / LAUNCHER_NAME


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def plan(
    *,
    environ: Mapping[str, str] | None = None,
    start: Path | None = None,
    project_environment: Path | None = None,
) -> InstallPlan | Refusal:
    """Resolve this install, or return the refusal to print (exit 64).

    A value that is set and empty wins over the derived one: ``MINERU_MAX_VERSION=``
    lifts the ceiling deliberately, while an unset key keeps the derived default.
    """
    values = env.effective(INSTALL_KEYS, environ=environ)
    family, client = version_range.resolve(
        explicit=project_environment, environ=environ, start=start
    )
    derived_min, derived_max, derived_extras = version_range.derived_range(
        family, is_apple_silicon=is_apple_silicon()
    )

    version = values.get("MINERU_VERSION", "")
    minimum = values.get("MINERU_MIN_VERSION", derived_min)
    maximum = values.get("MINERU_MAX_VERSION", derived_max)
    extras = values.get("MINERU_INSTALL_EXTRAS", derived_extras)

    refusal = version_range.unsupported_pin(version, maximum, family)
    if refusal is not None:
        return Refusal(refusal)

    constraints = []
    if version:
        constraints.append(f"=={version}")
    if minimum:
        constraints.append(f">={minimum}")
    if maximum:
        constraints.append(f"<{maximum}")
    package = f"mineru[{extras}]" if extras else "mineru"
    package += ",".join(constraints)

    environment = env.service_environment(environ=environ, start=start)

    return InstallPlan(
        environment=environment,
        python_version=values.get("MINERU_PYTHON", DEFAULT_PYTHON) or DEFAULT_PYTHON,
        package=package,
        family=family,
        client_source=client,
    )


def install(
    *,
    environ: Mapping[str, str] | None = None,
    start: Path | None = None,
    project_environment: Path | None = None,
) -> int:
    """Run the install; returns the exit code to use."""
    resolved = plan(environ=environ, start=start, project_environment=project_environment)
    if isinstance(resolved, Refusal):
        print(resolved.message, file=sys.stderr)
        return resolved.code

    uv = shutil.which("uv")
    if uv is None:
        print(UV_MISSING, file=sys.stderr)
        return 127

    print(
        "mineru-install: the project's LightRAG client speaks the MinerU "
        f"{resolved.family} API: installing {resolved.package}"
    )

    code = _run([uv, "venv", "--python", resolved.python_version, str(resolved.environment)])
    if code != 0:
        return code
    # Only now, once the venv exists: `uv pip install` runs this interpreter, and site
    # would import the guards against a package that is being replaced.
    remove_guards(resolved.environment)
    code = _run(
        [uv, "pip", "install", "--python", str(resolved.interpreter), "-U", resolved.package]
    )
    if code != 0:
        # A failed upgrade must not leave the venv less guarded than it found it.
        try:
            materialize(resolved.environment)
        except InstallError:
            pass
        return code

    if not (resolved.environment / "pyvenv.cfg").is_file():
        print(
            f"{resolved.environment} is not a virtual environment; refusing to place the "
            "service files there.",
            file=sys.stderr,
        )
        return 1
    try:
        launcher = materialize(resolved.environment)
    except InstallError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"==> Sidecar ready: {resolved.environment}")
    print(f"    launcher : {launcher}")
    print("    service  : dlightrag-mineru service install")
    return 0


def materialize(environment: Path) -> Path:
    """Place the launcher, the env reader and the startup guards in the service venv.

    ``env.py`` is copied as ``_env.py`` rather than reimplemented, so the launcher and
    the manager can never drift apart on what ``.env.mineru`` means. The guards go into
    site-packages behind a ``.pth``, which makes them active for every start of this
    venv's interpreter — including a direct ``mineru-api`` run that bypasses the
    launcher. Returns the launcher path, which is what the service units name.
    """
    directory = environment / SHIM_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    package = importlib.resources.files("dlightrag_mineru")

    _write_atomically(
        directory / LAUNCHER_NAME, package.joinpath(LAUNCHER_SOURCE).read_text(encoding="utf-8")
    )
    _write_atomically(
        directory / ENV_MODULE_NAME, package.joinpath(ENV_SOURCE_NAME).read_text(encoding="utf-8")
    )

    site_packages = installed_site_packages(environment)
    if site_packages is None:
        msg = f"{environment} has no site-packages directory; not a virtual environment?"
        raise InstallError(msg)
    _write_atomically(
        site_packages / GUARD_MODULE, package.joinpath(GUARD_SOURCE).read_text(encoding="utf-8")
    )
    _write_atomically(site_packages / PTH_NAME, PTH_CONTENT)
    _write_atomically(
        directory / MANIFEST_NAME,
        json.dumps(
            {"tool": "dlightrag-mineru", "guards": GUARD_MODULE, "hook": PTH_NAME},
            indent=2,
        )
        + "\n",
    )

    launcher = directory / LAUNCHER_NAME
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return launcher


def _write_atomically(destination: Path, text: str) -> None:
    """Replace a placed file in one step, so a running service never execs a half file."""
    temporary = destination.with_name(f".{destination.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def installed_site_packages(environment: Path) -> Path | None:
    """The venv's site-packages, POSIX or Windows, if it has one."""
    return next((path for path in version_range.site_packages(environment) if path.is_dir()), None)


def remove_guards(environment: Path) -> None:
    """Drop the ``.pth`` before the venv's packages change underneath it.

    ``uv pip install`` runs this venv's interpreter, and site imports the guards at
    startup: mid-upgrade, that is a half-installed MinerU they would patch.
    """
    site_packages = installed_site_packages(environment)
    if site_packages is not None:
        (site_packages / PTH_NAME).unlink(missing_ok=True)


def purge(
    environment: Path,
    *,
    confirmed: bool = False,
    dry_run: bool = False,
    report: Callable[[str], None] = print,
) -> int:
    """Remove a service venv this tool installed, and nothing else.

    The venv is 1.6 GB on a working host, so leaving removal to ``rm -rf`` invites the
    mistake this refuses: the path must be a virtual environment that carries our
    launcher, and it may not be a root, a home, or the project itself.
    """
    resolved = environment.resolve()
    marker = resolved / SHIM_DIRECTORY / LAUNCHER_NAME
    manifest = resolved / SHIM_DIRECTORY / MANIFEST_NAME
    if not (resolved / "pyvenv.cfg").is_file() or not marker.is_file() or not manifest.is_file():
        report(
            f"refusing: {resolved} is not a MinerU service environment "
            f"(no pyvenv.cfg, {SHIM_DIRECTORY}/{LAUNCHER_NAME}, and {MANIFEST_NAME})"
        )
        return 1
    try:
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        report(f"refusing: {manifest} is unreadable; not a MinerU service environment")
        return 1
    if recorded.get("tool") != "dlightrag-mineru":
        report(f"refusing: {manifest} does not name this tool")
        return 1
    forbidden = {Path(resolved.anchor), Path.home().resolve(), env.project_root().resolve()}
    if resolved in forbidden or resolved.parent == resolved:
        report(f"refusing: {resolved} is not a directory this tool may remove")
        return 1

    megabytes = _directory_bytes(resolved) / 1024 / 1024
    if dry_run:
        report(f"would remove {resolved} ({megabytes:.0f} MiB)")
        return 0
    if not confirmed:
        report(f"{resolved} ({megabytes:.0f} MiB) will be removed; re-run with --yes")
        return 64

    shutil.rmtree(resolved)
    report(f"==> Removed {resolved}")
    report("    run `dlightrag-mineru service uninstall` if a unit is still installed")
    return 0


def _directory_bytes(directory: Path) -> int:
    """What this tree holds, without following a link out of it (lstat, not stat)."""
    total = 0
    for path in directory.rglob("*"):
        try:
            info = path.lstat()
        except OSError:  # pragma: no cover - a file that vanished mid-walk
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


def launcher_command(environment: Path) -> tuple[str, ...]:
    """How launchd and systemd invoke the sidecar: absolute interpreter, absolute file."""
    return (str(environment / "bin" / "python"), str(environment / SHIM_DIRECTORY / LAUNCHER_NAME))


def _run(argv: Sequence[str]) -> int:
    return subprocess.run(argv, check=False).returncode  # noqa: S603 - the installer's own plan
