#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Prepare local Langfuse headless initialization for DlightRAG.

This helper keeps the local Langfuse stack's headless project keys and
DlightRAG's Langfuse SDK keys aligned before either service starts.
"""

import argparse
import os
import re
import secrets
import sys
from pathlib import Path
from typing import NamedTuple

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")

DEFAULT_PUBLIC_KEY = "pk-lf-dlight-local"
DEFAULT_ORG_ID = "dlight-local-org"
DEFAULT_ORG_NAME = "DlightRAG_Local"
DEFAULT_PROJECT_ID = "dlight-local-project"
DEFAULT_PROJECT_NAME = "DlightRAG_Local"
DEFAULT_USER_EMAIL = "admin@localhost.local"
DEFAULT_USER_NAME = "local-admin"


class EnvFile(NamedTuple):
    path: Path
    lines: list[str]
    values: dict[str, str]
    positions: dict[str, int]


class BootstrapResult(NamedTuple):
    langfuse_env: Path
    dlightrag_env: Path
    public_key: str
    host: str


def _clean_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def read_env(path: Path) -> EnvFile:
    """Read a simple dotenv file while preserving original lines."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else []
    values: dict[str, str] = {}
    positions: dict[str, int] = {}
    for index, line in enumerate(lines):
        match = _ENV_LINE.match(line)
        if not match:
            continue
        key = match.group(1)
        values[key] = _clean_value(match.group(2))
        positions.setdefault(key, index)
    return EnvFile(path=path, lines=lines, values=values, positions=positions)


def _set_value(env_file: EnvFile, key: str, value: str, *, overwrite: bool) -> None:
    if key in env_file.values and not overwrite:
        return
    line = f"{key}={value}\n"
    if key in env_file.positions:
        env_file.lines[env_file.positions[key]] = line
    else:
        if env_file.lines and not env_file.lines[-1].endswith("\n"):
            env_file.lines[-1] += "\n"
        env_file.lines.append(line)
        env_file.positions[key] = len(env_file.lines) - 1
    env_file.values[key] = value


def write_env(env_file: EnvFile) -> None:
    """Write a dotenv file, creating parent directories as needed."""
    env_file.path.parent.mkdir(parents=True, exist_ok=True)
    env_file.path.write_text("".join(env_file.lines), encoding="utf-8")


def _secret_key() -> str:
    return f"sk-lf-{secrets.token_urlsafe(24)}"


def bootstrap(
    *,
    langfuse_env: Path,
    dlightrag_env: Path,
    host: str,
    public_key: str | None = None,
    secret_key: str | None = None,
) -> BootstrapResult:
    """Sync local Langfuse headless init keys into DlightRAG's .env."""
    langfuse = read_env(langfuse_env)
    dlightrag = read_env(dlightrag_env)

    resolved_public = (
        public_key
        or os.environ.get("LANGFUSE_INIT_PROJECT_PUBLIC_KEY")
        or langfuse.values.get("LANGFUSE_INIT_PROJECT_PUBLIC_KEY")
        or dlightrag.values.get("DLIGHTRAG_LANGFUSE_PUBLIC_KEY")
        or DEFAULT_PUBLIC_KEY
    )
    resolved_secret = (
        secret_key
        or os.environ.get("LANGFUSE_INIT_PROJECT_SECRET_KEY")
        or langfuse.values.get("LANGFUSE_INIT_PROJECT_SECRET_KEY")
        or dlightrag.values.get("DLIGHTRAG_LANGFUSE_SECRET_KEY")
        or _secret_key()
    )

    langfuse_defaults = {
        "LANGFUSE_INIT_ORG_ID": DEFAULT_ORG_ID,
        "LANGFUSE_INIT_ORG_NAME": DEFAULT_ORG_NAME,
        "LANGFUSE_INIT_PROJECT_ID": DEFAULT_PROJECT_ID,
        "LANGFUSE_INIT_PROJECT_NAME": DEFAULT_PROJECT_NAME,
        "LANGFUSE_INIT_USER_EMAIL": DEFAULT_USER_EMAIL,
        "LANGFUSE_INIT_USER_NAME": DEFAULT_USER_NAME,
        "LANGFUSE_INIT_USER_PASSWORD": _secret_key(),
    }
    for key, value in langfuse_defaults.items():
        _set_value(langfuse, key, value, overwrite=False)
    _set_value(langfuse, "LANGFUSE_INIT_PROJECT_PUBLIC_KEY", resolved_public, overwrite=True)
    _set_value(langfuse, "LANGFUSE_INIT_PROJECT_SECRET_KEY", resolved_secret, overwrite=True)

    _set_value(dlightrag, "DLIGHTRAG_LANGFUSE_PUBLIC_KEY", resolved_public, overwrite=True)
    _set_value(dlightrag, "DLIGHTRAG_LANGFUSE_SECRET_KEY", resolved_secret, overwrite=True)
    _set_value(dlightrag, "DLIGHTRAG_LANGFUSE_HOST", host, overwrite=True)
    _set_value(dlightrag, "DLIGHTRAG_LANGFUSE_EXPORT_EXTERNAL_SPANS", "false", overwrite=False)

    write_env(langfuse)
    write_env(dlightrag)

    return BootstrapResult(
        langfuse_env=langfuse_env,
        dlightrag_env=dlightrag_env,
        public_key=resolved_public,
        host=host,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare local Langfuse headless project keys for DlightRAG",
        suggest_on_error=True,
    )
    parser.add_argument("--langfuse-env", type=Path, required=True)
    parser.add_argument("--dlightrag-env", type=Path, default=Path(".env"))
    parser.add_argument("--host", default="http://localhost:3300")
    parser.add_argument("--public-key", default=None)
    parser.add_argument("--secret-key", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = bootstrap(
        langfuse_env=args.langfuse_env,
        dlightrag_env=args.dlightrag_env,
        host=args.host,
        public_key=args.public_key,
        secret_key=args.secret_key,
    )
    print(
        "Langfuse headless project keys synced "
        f"(public key: {result.public_key}, host: {result.host})"
    )
    print(f"  Langfuse env: {result.langfuse_env}")
    print(f"  DlightRAG env: {result.dlightrag_env}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
