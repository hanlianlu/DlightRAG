# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Publication scan: optional report, no symlinks, empty-answer rule."""

from pathlib import Path

import pytest

from dlightrag.answer.publication import (
    PublicationScanError,
    is_empty_answer,
    scan_artifact_directory,
)


def test_missing_artifacts_dir_is_empty(tmp_path: Path) -> None:
    assert scan_artifact_directory(tmp_path / "artifacts") == ()


def test_blank_report_is_not_published(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.md").write_text("  \n", encoding="utf-8")
    (root / "table.csv").write_text("a,b\n", encoding="utf-8")
    staged = scan_artifact_directory(root)
    assert [item.kind for item in staged] == ["published_artifact"]
    assert staged[0].relative_path == "table.csv"


def test_report_and_extra_file_are_both_staged(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.md").write_text("# Findings\n", encoding="utf-8")
    (root / "notes.txt").write_text("extra", encoding="utf-8")
    kinds = {item.relative_path: item.kind for item in scan_artifact_directory(root)}
    assert kinds == {"notes.txt": "published_artifact", "report.md": "primary_report"}


def test_symlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "link.md").symlink_to(tmp_path / "outside.md")
    with pytest.raises(PublicationScanError, match="symlink"):
        scan_artifact_directory(root)


def test_empty_answer_requires_neither_report_nor_text() -> None:
    assert is_empty_answer(answer="  ", has_primary_report=False) is True
    assert is_empty_answer(answer="42", has_primary_report=False) is False
    assert is_empty_answer(answer="", has_primary_report=True) is False
