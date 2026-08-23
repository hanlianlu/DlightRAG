#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Verify lockstep release metadata and removed Agent surface names."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_RETIRED_AGENT_TERMS = (
    "delegate_research",
    "SessionEpisode",
    "local_trusted",
    "agent.scope",
)


def _project(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def verify_repository(root: Path = ROOT) -> None:
    manifests = (root / "pyproject.toml", root / "packages/memory/pyproject.toml")
    projects = [_project(path) for path in manifests]
    versions = {str(project["version"]) for project in projects}
    if len(versions) != 1:
        raise ValueError(f"workspace versions are not lockstep: {sorted(versions)}")
    (version,) = versions
    dependency = f"dlightrag-memory=={version}"
    if dependency not in projects[0].get("dependencies", []):
        raise ValueError(f"root distribution must depend on {dependency}")

    frontend = json.loads((root / "frontend/package.json").read_text(encoding="utf-8"))
    frontend_lock = json.loads((root / "frontend/package-lock.json").read_text(encoding="utf-8"))
    if frontend.get("version") != version:
        raise ValueError("frontend package version is not lockstep")
    if (
        frontend_lock.get("version") != version
        or frontend_lock["packages"][""].get("version") != version
    ):
        raise ValueError("frontend lock version is not lockstep")

    memory_init = (root / "packages/memory/src/dlightrag_memory/__init__.py").read_text(
        encoding="utf-8"
    )
    if f'__version__ = "{version}"' not in memory_init:
        raise ValueError("Memory runtime version is not lockstep")

    major = version.split(".", 1)[0]
    migration = root / f"docs/migration-{major}.0.md"
    if not migration.is_file():
        raise ValueError(f"missing current migration guide: {migration.relative_to(root)}")
    config_text = (root / "config.yaml").read_text(encoding="utf-8")
    if not re.search(
        rf"^# DlightRAG {re.escape(major)}\.0 canonical configuration$", config_text, re.M
    ):
        raise ValueError("config.yaml release header is stale")
    if (
        "execution_environment: disabled" not in config_text
        or "outbound_mcp: []" not in config_text
    ):
        raise ValueError("config.yaml does not expose the safe Agent defaults")

    source_root = root / "src"
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for term in _RETIRED_AGENT_TERMS:
            if term in text:
                raise ValueError(f"retired Agent term {term!r} remains in {path.relative_to(root)}")


def main() -> None:
    verify_repository()
    print("release contract verification passed")


if __name__ == "__main__":
    main()
