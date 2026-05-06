# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Dependency constraint policy tests."""

from __future__ import annotations

import tomllib
from pathlib import Path


def _dependencies() -> list[str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["dependencies"]


def test_lightrag_dependency_has_no_upper_bound() -> None:
    """LightRAG should be minimum-pinned for 1.5+ without an upper cap."""
    dependencies = _dependencies()
    lightrag_deps = [dep for dep in dependencies if dep.startswith("lightrag-hku")]

    assert lightrag_deps == ["lightrag-hku>=1.5.0rc1"]


def test_raganything_dependency_has_no_upper_bound() -> None:
    """RAGAnything should be minimum-pinned for 1.3+ without an upper cap."""
    dependencies = _dependencies()
    raganything_deps = [dep for dep in dependencies if dep.startswith("raganything")]

    assert raganything_deps == ["raganything[all]>=1.3.0"]


def test_langfuse_dependency_has_no_upper_bound() -> None:
    """Langfuse should require the v4 SDK API without an upper cap."""
    dependencies = _dependencies()
    langfuse_deps = [dep for dep in dependencies if dep.startswith("langfuse")]

    assert langfuse_deps == ["langfuse>=4.0.0"]


def test_raganything_uses_registry_release_not_git_main() -> None:
    """RAGAnything should not drift on the GitHub main branch."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["tool"]["uv"].get("sources", {}).get("raganything") is None
