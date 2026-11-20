# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Resolve and read the sidecar's dotenv file.

The shell component's ``env.sh`` rules, in Python: the file is ``MINERU_ENV_FILE`` when
it is set and non-empty, else ``.env.mineru`` beside the project root; a relative
value resolves against that root. Reading is by key name, never wholesale -- the
file holds title-aided credentials the parser process has no business seeing in its
environment.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

DEFAULT_ENV_FILE = ".env.mineru"


#: A missing file is "absent"; anything else (including NotADirectoryError) is a
#: configuration error the caller must hear about.
_ABSENT_FROM_DISK = (FileNotFoundError,)


class EnvFileError(RuntimeError):
    """The env file exists but cannot be read or decoded.

    Kept apart from "absent" on purpose: a service that silently falls back to
    host/port defaults because its config is unreadable looks like it is working.
    """


#: Where the service virtualenv lives unless ``MINERU_SERVICE_VENV`` moves it.
DEFAULT_SERVICE_VENV = ".venv-mineru"

# A value is unquoted only when the whole value is wrapped, and surrounding
# whitespace is dropped: an env file is hand-written, and `PORT=8210 ` is a typo,
# not a port.
_QUOTES = "\"'"


def project_root(start: Path | None = None) -> Path:
    """The nearest ancestor holding a ``pyproject.toml``, else ``start`` itself.

    The shell read the repository root out of its own file location; a wheel cannot,
    so the project the command runs inside names it instead. Every documented use
    runs from the repository root, and this also makes a subdirectory work.
    """
    origin = Path.cwd() if start is None else start
    for candidate in (origin, *origin.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return origin


def env_file_path(
    *,
    environ: Mapping[str, str] | None = None,
    start: Path | None = None,
) -> Path:
    """Where ``MINERU_ENV_FILE`` says, else ``<project root>/.env.mineru``."""
    source = os.environ if environ is None else environ
    raw = source.get("MINERU_ENV_FILE") or DEFAULT_ENV_FILE
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return project_root(start) / path


def read_env_file(path: Path) -> dict[str, str]:
    """Every ``KEY=value`` the file sets; the last occurrence of a key wins."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except _ABSENT_FROM_DISK:
        return values
    except (OSError, UnicodeDecodeError) as exc:
        raise EnvFileError(f"cannot read {path}: {exc}") from exc
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        if not key:
            continue
        values[key] = _unquote(value.strip())
    return values


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTES:
        return value[1:-1]
    return value


def service_environment(
    *,
    environ: Mapping[str, str] | None = None,
    start: Path | None = None,
) -> Path:
    """The service venv every command resolves first, absolute and real.

    One definition, because ``install`` bakes this path into the service units and a
    later ``service install`` must name the same directory.
    """
    source = os.environ if environ is None else environ
    raw = source.get("MINERU_SERVICE_VENV") or ""
    if not raw:
        raw = read_env_file(env_file_path(environ=source, start=start)).get(
            "MINERU_SERVICE_VENV", ""
        )
    environment = Path(raw or DEFAULT_SERVICE_VENV).expanduser()
    if not environment.is_absolute():
        environment = project_root(start) / environment
    return environment.resolve()


def effective(
    keys: Iterable[str],
    *,
    environ: Mapping[str, str] | None = None,
    path: Path | None = None,
) -> dict[str, str]:
    """The values these keys resolve to: the environment wins, the file fills gaps.

    Keys absent from both are absent from the result, which is what lets a caller
    tell "unset" from "deliberately empty" -- ``MINERU_MAX_VERSION=`` lifts the
    ceiling, an unset one keeps the derived default.
    """
    source = os.environ if environ is None else environ
    values = read_env_file(path if path is not None else env_file_path(environ=source))
    resolved: dict[str, str] = {}
    for key in keys:
        from_environment = source.get(key)
        if from_environment:
            resolved[key] = from_environment
        elif key in values:
            resolved[key] = values[key]
        elif key in source:
            # Set and empty in the environment, absent from the file: "deliberately
            # empty", which ``MINERU_MAX_VERSION=`` relies on to lift the ceiling. The
            # shell read this through ``${VAR-default}``, which defaults on unset only.
            resolved[key] = ""
    return resolved


def apply(
    keys: Iterable[str],
    *,
    environ: dict[str, str] | None = None,
    path: Path | None = None,
) -> dict[str, str]:
    """Resolve these keys and export the missing ones into ``environ``."""
    target = os.environ if environ is None else environ
    resolved = effective(keys, environ=target, path=path)
    for key, value in resolved.items():
        if not target.get(key):
            target[key] = value
    return resolved
