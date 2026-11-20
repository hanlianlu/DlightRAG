# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Write MinerU's title-aided LLM config into ``~/mineru.json``.

Only ``llm-aided-config.title_aided`` is overwritten; every other top-level key
(``models-dir``, ``model-source``, ...) survives, so this can run against a file
MinerU wrote itself.

The reasoning fields arrive through :func:`from_environment`'s ``resolver``. Which
dialect this deployment's endpoint speaks is DlightRAG's knowledge, resolved by the
application and injected here, so this module states no product vocabulary (ADR 0054).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dlightrag_mineru import env

CONFIG_SECTION = "llm-aided-config"
TITLE_SECTION = "title_aided"

TITLE_KEYS = (
    "MINERU_TITLE_AIDED_ENABLE",
    "MINERU_TITLE_AIDED_API_KEY",
    "MINERU_TITLE_AIDED_BASE_URL",
    "MINERU_TITLE_AIDED_MODEL",
    "MINERU_TITLE_AIDED_ENABLE_THINKING",
)

DEFAULT_TARGET = Path.home() / "mineru.json"

# The file holds an LLM credential, so it is never group- or world-readable, and it
# is replaced atomically: a reader (MinerU, or a person) sees the old file or the new
# one, never a half-written one.
_FILE_MODE = 0o600

# ``(base_url, model) -> request fields``: the caller's knowledge of how this endpoint
# turns reasoning off.
ReasoningResolver = Callable[[str, str], dict[str, Any]]

REQUIRED = (
    ("MINERU_TITLE_AIDED_API_KEY", "api_key"),
    ("MINERU_TITLE_AIDED_BASE_URL", "base_url"),
    ("MINERU_TITLE_AIDED_MODEL", "model"),
)


def _is_true(value: str) -> bool:
    """``true`` in any case, as the shell's ``.lower()`` comparison read it."""
    return value.strip().lower() == "true"


def _print_lines(lines: Sequence[str]) -> None:
    """The default report: one line per line, never a list repr."""
    for line in lines:
        print(line)


@dataclass(frozen=True)
class Request:
    """One title-aided configuration, fully resolved."""

    target: Path = DEFAULT_TARGET
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    enable_thinking: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)


def from_environment(
    *,
    environ: Mapping[str, str] | None = None,
    target: Path | None = None,
    disable: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    enable_thinking: bool = False,
    extra_body: dict[str, Any] | None = None,
    resolver: ReasoningResolver | None = None,
    report: Callable[[Sequence[str]], None] = _print_lines,
) -> int:
    """Read ``.env.mineru``, resolve the reasoning fields, and write the config.

    Explicit arguments win over the file, and the environment wins over both, as
    everywhere else in this tool. ``resolver`` supplies the request fields that keep
    the title model from reasoning; without one, and without ``extra_body``, the
    request carries none and the provider's default applies. Returns the exit code.
    """
    values = env.effective(TITLE_KEYS, environ=environ)
    enabled = values.get("MINERU_TITLE_AIDED_ENABLE", "true") == "true"
    destination = target if target is not None else DEFAULT_TARGET

    if disable or not enabled:
        report(disable_config(destination))
        return 0

    resolved = {
        "api_key": api_key or values.get("MINERU_TITLE_AIDED_API_KEY", ""),
        "base_url": base_url or values.get("MINERU_TITLE_AIDED_BASE_URL", ""),
        "model": model or values.get("MINERU_TITLE_AIDED_MODEL", ""),
    }
    if missing := [name for name, key in REQUIRED if not resolved[key]]:
        report([f"ERROR: {name} is not set." for name in missing])
        report(["Add it to .env.mineru or export it before running this command."])
        return 1

    thinking = enable_thinking or _is_true(values.get("MINERU_TITLE_AIDED_ENABLE_THINKING", ""))
    fields = extra_body
    if fields is None:
        # Thinking on leaves reasoning to the provider's default.
        if thinking or resolver is None:
            fields = {}
        else:
            fields = resolver(resolved["base_url"], resolved["model"])

    report(
        write_config(
            Request(
                target=destination,
                api_key=resolved["api_key"],
                base_url=resolved["base_url"],
                model=resolved["model"],
                enable_thinking=thinking,
                extra_body=fields or {},
            )
        )
    )
    return 0


def load_existing(path: Path) -> tuple[dict[str, Any], str | None]:
    """The parsed file and any warning about it; missing is not a warning."""
    if not path.is_file():
        return {}, None
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}, None
        parsed = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        return {}, f"WARNING: could not parse {path} — starting fresh ({exc})"
    if not isinstance(parsed, dict):
        return {}, f"WARNING: could not parse {path} — starting fresh (not a JSON object)"
    return parsed, None


def write_config(request: Request) -> list[str]:
    """Enable title-aided correction and report what was written.

    The report never echoes key material, not even partially: presence and length
    only.
    """
    existing, warning = load_existing(request.target)
    title_aided: dict[str, Any] = {
        "api_key": request.api_key,
        "base_url": request.base_url,
        "model": request.model,
        "extra_body": request.extra_body,
        "enable": True,
    }
    lines = [warning] if warning else []
    lines.append(f"==> Wrote {request.target}")
    lines.append(f"    model    : {request.model}")
    lines.append(f"    url      : {request.base_url}")
    lines.append(
        f"    key      : {'set (' + str(len(request.api_key)) + ' chars)' if request.api_key else 'MISSING'}"
    )
    rendered = json.dumps(request.extra_body) if request.extra_body else "provider default"
    lines.append(f"    reasoning: {rendered}")
    _merge_and_write(request.target, existing, title_aided)
    return lines


def disable_config(path: Path) -> list[str]:
    """Turn title-aided correction off, keeping every other key."""
    existing, warning = load_existing(path)
    lines = [warning] if warning else []
    _merge_and_write(path, existing, {"enable": False})
    lines.append(f"==> Disabled title-aided config in {path}")
    return lines


def _merge_and_write(path: Path, existing: dict[str, Any], title_aided: dict[str, Any]) -> None:
    section = existing.get(CONFIG_SECTION)
    if not isinstance(section, dict):
        section = {}
    section[TITLE_SECTION] = title_aided
    existing[CONFIG_SECTION] = section
    _write_json_atomically(path, existing)


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    # O_EXCL refuses to follow or reuse a pre-planted temporary file, and the cleanup
    # keeps a failed write from leaving key material on disk.
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, _FILE_MODE
        )
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, _FILE_MODE
        )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    os.chmod(path, _FILE_MODE)
