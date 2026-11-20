#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Write a product version in every place that names it, then check the contract.

A release used to edit six files by hand and then wait for CI to say which one was missed.
The version lives in four manifests now, because everything else derives it: each
workspace runtime reads its installed distribution, the frontend names no version, and
``uv.lock`` resolves from the pins. This writes those manifests, re-resolves the lock,
and runs the release contract — so ``make release VERSION=x.y.z`` is the release. The
configuration baseline in ``config.yaml`` names the config schema a deployment writes
against, not the product version, so a release never rewrites it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = "Write a product version everywhere it is named, then check the contract"
_RELEASE = re.compile(r"^\d+\.\d+\.\d+$")
_VERSION_LINE = re.compile(r'^version = "\d+\.\d+\.\d+"$', re.MULTILINE)
#: Every workspace distribution the root pins, in the order a release rewrites them.
WORKSPACE_PINS = ("dlightrag-memory", "dlightrag-observability")
_WORKSPACE_PIN = re.compile(r'"dlightrag-(memory|observability)==\d+\.\d+\.\d+"')

#: What a release commit may carry. A release stages these and nothing else, so work in
#: progress elsewhere stays uncommitted instead of riding along with the version.
RELEASE_PATHS = (
    "pyproject.toml",
    "packages/memory/pyproject.toml",
    "packages/mineru/pyproject.toml",
    "packages/observability/pyproject.toml",
    "uv.lock",
)


def parse_version(text: str) -> tuple[int, int, int]:
    if not _RELEASE.fullmatch(text):
        raise ValueError(f"a version is three numbers, like 2.2.0: {text!r}")
    major, minor, patch = (int(part) for part in text.split("."))
    return major, minor, patch


def current_version(root: Path = ROOT) -> str:
    manifest = (root / "pyproject.toml").read_text(encoding="utf-8")
    found = _VERSION_LINE.search(manifest)
    if found is None:
        raise ValueError("pyproject.toml names no version")
    return found.group(0).split('"')[1]


def bump_manifest(text: str, version: str) -> str:
    """The ``version`` line of one workspace manifest."""
    bumped, replaced = _VERSION_LINE.subn(f'version = "{version}"', text, count=1)
    if replaced != 1:
        raise ValueError("a workspace manifest names no single version line")
    return bumped


def bump_workspace_pins(text: str, version: str) -> str:
    """Every exact workspace pin the root distribution carries."""
    bumped, replaced = _WORKSPACE_PIN.subn(
        lambda match: f'"dlightrag-{match.group(1)}=={version}"', text
    )
    if replaced != len(WORKSPACE_PINS):
        raise ValueError("the root manifest does not name every workspace pin")
    return bumped


def _writes(version: str) -> tuple[tuple[str, Callable[[str], str]], ...]:
    """The files a release writes. The configuration baseline is not one of them: it
    names the schema a deployment's config.yaml is written against, not the product
    version, so a version reset never rewrites it."""
    return (
        (
            "pyproject.toml",
            lambda text: bump_workspace_pins(bump_manifest(text, version), version),
        ),
        ("packages/memory/pyproject.toml", lambda text: bump_manifest(text, version)),
        ("packages/mineru/pyproject.toml", lambda text: bump_manifest(text, version)),
        ("packages/observability/pyproject.toml", lambda text: bump_manifest(text, version)),
    )


def is_ahead(current: str, target: str) -> bool:
    """Whether ``target`` moves the version forward, which a release normally does."""
    return parse_version(target) > parse_version(current)


def _run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603 - argv list, no shell


def main() -> int:
    parser = argparse.ArgumentParser(description=SUMMARY, suggest_on_error=True)
    parser.add_argument("version", help="the version to release, like 2.2.0")
    parser.add_argument("--commit", action="store_true", help="commit the release")
    parser.add_argument("--tag", action="store_true", help="tag it v<version>")
    parser.add_argument("--push", action="store_true", help="push the commit and the tag")
    parser.add_argument("--dry-run", action="store_true", help="say what would be written")
    parser.add_argument(
        "--allow-downgrade",
        action="store_true",
        help="write a version that is not ahead of the current one, as a reset is",
    )
    args = parser.parse_args()

    try:
        current = current_version()
        parse_version(args.version)
        parse_version(current)
    except ValueError as error:
        print(f"release: {error}", file=sys.stderr)
        return 64
    if not args.allow_downgrade and not is_ahead(current, args.version):
        print(
            f"release: {args.version} is not ahead of {current}; pass --allow-downgrade to reset",
            file=sys.stderr,
        )
        return 64

    for name, bump in _writes(args.version):
        path = ROOT / name
        bumped = bump(path.read_text(encoding="utf-8"))
        if args.dry_run:
            print(f"release: would write {name}")
        else:
            path.write_text(bumped, encoding="utf-8")
            print(f"release: wrote {name}")
    if args.dry_run:
        return 0

    _run("uv", "lock")
    _run("uv", "run", "python", "scripts/verify_release_contract.py", "--scope", "versions")
    if args.commit or args.tag or args.push:
        _run("git", "add", *RELEASE_PATHS)
        _run("git", "commit", "-m", f"chore: release {args.version}")
    if args.tag or args.push:
        _run("git", "tag", f"v{args.version}")
    if args.push:
        _run("git", "push", "origin", "main")
        _run("git", "push", "origin", f"v{args.version}")
    print(f"release: {args.version} is written and the contract holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
