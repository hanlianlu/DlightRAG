#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Load the sidecar's environment, then exec MinerU.

`dlightrag-mineru install` copies this file and the package's env reader (as
``_env.py``) into ``<service venv>/dlightrag-mineru/``, installs the
interpreter-startup guards into that venv's site-packages behind a ``.pth``, and
points launchd/systemd at ``<venv>/bin/python <that dir>/launcher.py``.

The service therefore reads ``.env.mineru`` at every start (a restart applies an
edit, as it always has), the guards are active for any start of that interpreter,
and the service depends on nothing from the application's environment. No wrapper
process stays on the supervised path: this file execs MinerU, so ``KeepAlive`` /
``Restart=on-failure`` keep acting on MinerU itself (ADR 0054).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType

# The keys the parser process legitimately needs. The title-aided credential stays
# out of it: the shim reads that config from ~/mineru.json, and a secret in a parse
# process's environment is exposure with no benefit.
ENVIRONMENT_KEYS = (
    "MINERU_API_HOST",
    "MINERU_API_PORT",
    "MINERU_HYBRID_EFFORT",
    "MINERU_TITLE_AIDED_ATTEMPT_TIMEOUT_SECONDS",
    "MINERU_TITLE_AIDED_MAX_ATTEMPTS",
)

# Keys that must never reach the parser, even when the invoking shell exported them:
# the token configures title correction, which the shim reads from ~/mineru.json.
FORBIDDEN_KEYS = ("MINERU_TITLE_AIDED_API_KEY",)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = "8210"

BAD_BINDING = (
    "MINERU_API_HOST/MINERU_API_PORT are not a host and a port: host={host!r} port={port!r}.\n"
    "Fix them in .env.mineru; under launchd/systemd a bad value would crash-loop."
)

MISSING_BINARY = """mineru-api was not found in the service environment.

Install the dedicated MinerU service env first:
  dlightrag-mineru install

Override MinerU install extras when the machine needs them:
  MINERU_INSTALL_EXTRAS=core,mlx dlightrag-mineru install
  MINERU_INSTALL_EXTRAS=core,vllm dlightrag-mineru install
  MINERU_INSTALL_EXTRAS=core,lmdeploy dlightrag-mineru install
"""


def _load_env_module(here: Path) -> ModuleType:
    """Import the env reader installed beside this file, by path."""
    path = here / "_env.py"
    spec = importlib.util.spec_from_file_location("_dlightrag_mineru_env", path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreadable install
        raise SystemExit(f"{path} is missing; re-run `dlightrag-mineru install`.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve().parent
    venv = here.parent

    env_module = _load_env_module(here)
    try:
        env_module.apply(ENVIRONMENT_KEYS)
    except env_module.EnvFileError as exc:
        # A service whose configuration cannot be read must not start on defaults and
        # look healthy; the supervisor will report the failure.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for key in FORBIDDEN_KEYS:
        os.environ.pop(key, None)

    binary = venv / "bin" / "mineru-api"
    target = str(binary) if binary.is_file() else (shutil.which("mineru-api") or "")
    if not target:
        print(MISSING_BINARY, file=sys.stderr)
        return 127

    host = os.environ.get("MINERU_API_HOST") or DEFAULT_HOST
    port = os.environ.get("MINERU_API_PORT") or DEFAULT_PORT
    if not host or not port.isdigit():
        print(BAD_BINDING.format(host=host, port=port), file=sys.stderr)
        return 64
    os.execv(target, [target, "--host", host, "--port", port, *argv])  # noqa: S606 - argv, no shell
    return 127  # unreachable: execv replaces this process


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
