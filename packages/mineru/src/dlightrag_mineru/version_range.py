# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Which MinerU API generation the pinned LightRAG client speaks.

MinerU 4.0 replaced the local task API (``POST /tasks``, ``GET /tasks/{id}``,
``GET /tasks/{id}/result``) with ``/v1/parse/jobs`` and friends. Which of the two
this project can drive is a property of the LightRAG client in the project
environment, not of a number someone remembers to raise: upgrading LightRAG is then
the whole of a MinerU 4.x upgrade here. The client's *source* is read, never
imported -- the client may not be importable in the environment running this tool.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dlightrag_mineru.env import project_root

V3 = "v3"
V4 = "v4"
UNKNOWN = "unknown"

CLIENT_SUFFIX = Path("lightrag/parser/external/mineru/client.py")
V4_MARKERS = ("/v1/parse/jobs", "/v1/uploads")
V3_MARKERS = ("/tasks",)

# The range each family can be installed from. MinerU 4.x dropped the core/mlx
# extras, so its derived extras are empty rather than a different default.
V3_RANGE = ("3.4.5", "4.0")
V4_RANGE = ("4.0", "")
# No client found: the range every client has spoken so far.
FALLBACK_RANGE = ("3.4.5", "4.0")


def project_environment(
    *,
    explicit: Path | None = None,
    environ: Mapping[str, str] | None = None,
    start: Path | None = None,
) -> Path:
    """The environment whose LightRAG client decides the range.

    ``--project-env`` first, then uv's override, then the environment a ``uv run``
    activated, then the project's own ``.venv``.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    source = os.environ if environ is None else environ
    for name in ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        value = source.get(name)
        if value:
            return Path(value).expanduser()
    return project_root(start) / ".venv"


def site_packages(environment: Path) -> list[Path]:
    """Every site-packages this environment might hold, POSIX and Windows."""
    return sorted(environment.glob("lib/python*/site-packages")) + [
        environment / "Lib/site-packages"
    ]


def client_source(environment: Path) -> tuple[str, Path] | None:
    """The pinned client's source and where it was read from, if it is installed."""
    for packages in site_packages(environment):
        path = packages / CLIENT_SUFFIX
        if path.is_file():
            return path.read_text(encoding="utf-8"), path
    return None


def classify(source: str) -> str:
    """Class the client by the endpoints it calls, newest API first."""
    if any(marker in source for marker in V4_MARKERS):
        return V4
    if any(marker in source for marker in V3_MARKERS):
        return V3
    return UNKNOWN


def resolve(
    *,
    explicit: Path | None = None,
    environ: Mapping[str, str] | None = None,
    start: Path | None = None,
) -> tuple[str, Path | None]:
    """The API family this project's client speaks, and where that was read."""
    found = client_source(project_environment(explicit=explicit, environ=environ, start=start))
    if found is None:
        return UNKNOWN, None
    source, path = found
    return classify(source), path


def derived_range(family: str, *, is_apple_silicon: bool) -> tuple[str, str, str]:
    """``(min, max, extras)`` for this family, before any override.

    Platform extras follow the platform's accelerator: Apple Silicon gets MLX, every
    other host the cross-platform core. A family we could not identify keeps the
    conservative range and the platform extras.
    """
    if family == V4:
        return (*V4_RANGE, "")
    minimum, maximum = V3_RANGE if family == V3 else FALLBACK_RANGE
    return minimum, maximum, ("core,mlx" if is_apple_silicon else "core")


def unsupported_pin(version: str, maximum: str, family: str) -> str | None:
    """Why an exact pin is outside what this project can drive, if it is.

    An exact pin outside the range fails here rather than resolving to a sidecar the
    client cannot call, which ``uv`` would only report as an unsatisfiable specifier.
    """
    if not version or not maximum:
        return None
    if _major(version) < _major(maximum):
        return None
    return (
        f"MinerU {version} is not supported: the pinned LightRAG client speaks the\n"
        f"MinerU {family} API, and {maximum} and newer serve a different one.\n"
        "Upgrade LightRAG first, or set MINERU_MAX_VERSION to override the ceiling."
    )


def _major(version: str) -> int:
    head = version.split(".", 1)[0].strip()
    return int(head) if head.isdigit() else 0
