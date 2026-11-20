# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The release contract holds the Agent Browser's Playwright pin in every place it is written."""

import shutil
from pathlib import Path

import pytest

from scripts.verify_release_contract import verify_repository

_ROOT = Path(__file__).resolve().parents[2]
_FILES = (
    "pyproject.toml",
    "uv.lock",
    "config.yaml",
    "docs/configuration.md",
    "src/dlightrag/engine/rag/workspace/settings.py",
    "docker-compose.yml",
    "packages/memory/pyproject.toml",
    "packages/memory/src/dlightrag_memory/__init__.py",
    "packages/mineru/pyproject.toml",
    "packages/mineru/src/dlightrag_mineru/__init__.py",
    "packages/observability/pyproject.toml",
    "packages/observability/src/dlightrag_observability/__init__.py",
    "frontend/package.json",
    "frontend/package-lock.json",
    "agent-browser/browser/Dockerfile",
    "agent-browser/browser/package.json",
    "agent-browser/browser/package-lock.json",
)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    for name in _FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(_ROOT / name, tmp_path / name)
    return tmp_path


def _edit(root: Path, name: str, old: str, new: str) -> None:
    path = root / name
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} is not in {name}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_the_repository_as_committed_holds_one_playwright_version(repository: Path) -> None:
    verify_repository(repository)


@pytest.mark.parametrize(
    ("name", "old", "new", "complaint"),
    [
        ("pyproject.toml", '"playwright==1.65.1"', '"playwright==1.65.2"', "uv.lock resolves"),
        ("pyproject.toml", '"playwright==1.65.1"', '"playwright>=1.65.1"', "exactly once"),
        (
            "uv.lock",
            'name = "playwright"\nversion = "1.65.1"',
            'name = "playwright"\nversion = "1.62.0"',
            "uv.lock resolves",
        ),
        (
            "agent-browser/browser/Dockerfile",
            "ARG PLAYWRIGHT_VERSION=1.65.1",
            "ARG PLAYWRIGHT_VERSION=1.62.0",
            "Dockerfile",
        ),
        (
            "agent-browser/browser/package.json",
            '"playwright":"1.65.1"',
            '"playwright":"1.62.0"',
            "package.json",
        ),
        (
            "agent-browser/browser/package-lock.json",
            '"node_modules/playwright": {\n      "version": "1.65.1"',
            '"node_modules/playwright": {\n      "version": "1.62.0"',
            "package-lock.json",
        ),
        (
            "docker-compose.yml",
            "image: dlightrag-agent-browser:1.65.1",
            "image: dlightrag-agent-browser:1.62.0",
            "Compose pool image",
        ),
    ],
)
def test_a_playwright_version_that_disagrees_anywhere_fails_the_release_check(
    repository: Path, name: str, old: str, new: str, complaint: str
) -> None:
    _edit(repository, name, old, new)

    with pytest.raises(ValueError, match=complaint):
        verify_repository(repository)
