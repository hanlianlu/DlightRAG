#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Verify lockstep release metadata."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _project(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def _verify_playwright_pins(root: Path, dependencies: list[str]) -> None:
    """The Agent Browser client and its pool image run one Playwright version (ADR 0032).

    The pool's server refuses a client whose major or minor version differs, and the
    pin fixes the patch too, so every place that names the version must agree.
    """
    pinned = [dependency for dependency in dependencies if dependency.startswith("playwright==")]
    if len(pinned) != 1:
        raise ValueError("root distribution must pin playwright== exactly once")
    version = pinned[0].removeprefix("playwright==")

    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    locked = {package["name"]: package["version"] for package in lock["package"]}.get("playwright")
    if locked != version:
        raise ValueError(f"uv.lock resolves playwright {locked}, not {version}")

    pool = root / "agent-browser/browser"
    declared = re.search(
        r"^ARG PLAYWRIGHT_VERSION=(\S+)$", (pool / "Dockerfile").read_text(encoding="utf-8"), re.M
    )
    if declared is None or declared.group(1) != version:
        raise ValueError(f"the pool Dockerfile does not pin Playwright {version}")
    manifest = json.loads((pool / "package.json").read_text(encoding="utf-8"))
    if manifest.get("dependencies", {}).get("playwright") != version:
        raise ValueError(f"the pool package.json does not pin Playwright {version}")
    package_lock = json.loads((pool / "package-lock.json").read_text(encoding="utf-8"))["packages"]
    if (
        package_lock[""].get("dependencies", {}).get("playwright") != version
        or package_lock.get("node_modules/playwright", {}).get("version") != version
    ):
        raise ValueError(f"the pool package-lock.json does not lock Playwright {version}")
    image = re.search(
        r"^\s*image: dlightrag-agent-browser:(\S+)$",
        (root / "docker-compose.yml").read_text(encoding="utf-8"),
        re.M,
    )
    if image is None or image.group(1) != version:
        raise ValueError(f"the Compose pool image is not tagged {version}")


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

    _verify_playwright_pins(root, projects[0].get("dependencies", []))

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
    # Historical migration guides were retired: there are no external users
    # and the single-baseline schema makes upgrade narration meaningless.
    config_text = (root / "config.yaml").read_text(encoding="utf-8")
    if not re.search(
        rf"^# DlightRAG {re.escape(major)}\.0 canonical configuration$", config_text, re.M
    ):
        raise ValueError("config.yaml release header is stale")
    if "execution_environment: trust" not in config_text or "connections:" not in config_text:
        raise ValueError("config.yaml does not expose the canonical Agent defaults")


def main() -> None:
    verify_repository()
    print("release contract verification passed")


if __name__ == "__main__":
    main()
