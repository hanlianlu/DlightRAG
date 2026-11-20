# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One command writes the version: the manifests that name it, and nothing else."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from scripts.release import (
    RELEASE_PATHS,
    _writes,
    bump_manifest,
    bump_workspace_pins,
    current_version,
    is_ahead,
    parse_version,
)
from scripts.verify_release_contract import verify_versions

_ROOT = Path(__file__).resolve().parents[2]
_FILES = (
    "pyproject.toml",
    "packages/memory/pyproject.toml",
    "packages/mineru/pyproject.toml",
    "packages/observability/pyproject.toml",
)
#: The prose says how many manifests there are in words, so the count is checked too.
_COUNT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def test_a_version_is_three_numbers() -> None:
    assert parse_version("2.2.0") == (2, 2, 0)
    assert current_version(_ROOT) == current_version()

    with pytest.raises(ValueError, match="three numbers"):
        parse_version("v2.2")


def test_only_a_forward_move_is_a_release_unless_it_says_otherwise() -> None:
    assert is_ahead("2.1.0", "2.1.1")
    assert is_ahead("2.1.0", "3.0.0")
    assert not is_ahead("2.1.0", "2.1.0")
    assert not is_ahead("2.1.0", "0.1.0")


def test_a_release_commit_carries_every_file_the_release_writes() -> None:
    """Work in progress elsewhere must not ride along with the version."""
    written = {name for name, _ in _writes("9.9.9")} | {"uv.lock"}

    assert written == set(RELEASE_PATHS)


def test_the_operations_doc_names_every_manifest_the_release_writes() -> None:
    """A reader follows the prose rather than `_writes`, so the two must name one set.

    The prose said three manifests and omitted `packages/mineru` while the release wrote
    four, which is the drift a hand-bump reads as permission to skip one.
    """
    prose = (_ROOT / "docs/operations.md").read_text(encoding="utf-8")
    assert "The version lives in " in prose
    paragraph = prose.split("The version lives in ", 1)[1].split("\n\n", 1)[0]

    named = set(re.findall(r"`([^`]*pyproject\.toml)`", paragraph))
    manifests = {name for name in RELEASE_PATHS if name.endswith("pyproject.toml")}

    assert named == manifests
    assert paragraph.split()[0] == _COUNT_WORDS[len(manifests)]


def test_the_root_manifest_is_written_in_both_places_that_name_the_version() -> None:
    manifest = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    bumped = bump_workspace_pins(bump_manifest(manifest, "9.9.9"), "9.9.9")

    assert 'version = "9.9.9"' in bumped
    assert '"dlightrag-memory==9.9.9"' in bumped
    assert '"dlightrag-observability==9.9.9"' in bumped
    assert bumped.count("9.9.9") == 3


def test_a_version_reset_leaves_the_configuration_baseline_alone() -> None:
    """The baseline names the config schema, not the product version."""
    config = (_ROOT / "config.yaml").read_text(encoding="utf-8")

    assert {name for name, _ in _writes("0.1.0")} == set(_FILES)
    assert "config.yaml" not in {name for name, _ in _writes("0.1.0")}
    assert config.startswith("# DlightRAG 2.0 canonical configuration")


def test_a_file_without_the_expected_line_is_refused() -> None:
    empty = "nothing to bump\n"

    with pytest.raises(ValueError, match="names no single version line"):
        bump_manifest(empty, "9.9.9")
    with pytest.raises(ValueError, match="does not name every workspace pin"):
        bump_workspace_pins(empty, "9.9.9")


def test_a_tree_the_release_wrote_holds_the_version_contract(tmp_path: Path) -> None:
    for name in _FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(_ROOT / name, tmp_path / name)
    for name, bump in _writes("9.9.9"):
        path = tmp_path / name
        path.write_text(bump(path.read_text(encoding="utf-8")), encoding="utf-8")

    verify_versions(tmp_path)


def test_a_tree_whose_manifests_disagree_fails_the_version_contract(tmp_path: Path) -> None:
    for name in _FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(_ROOT / name, tmp_path / name)
    memory_manifest = tmp_path / "packages/memory/pyproject.toml"
    memory_manifest.write_text(
        bump_manifest(memory_manifest.read_text(encoding="utf-8"), "9.9.9"), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="not lockstep"):
        verify_versions(tmp_path)
