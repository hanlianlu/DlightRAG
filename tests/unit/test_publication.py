# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Publication scan: optional report, no symlinks, empty-answer rule."""

from pathlib import Path

import pytest

from dlightrag.answer.publication import (
    PublicationLimits,
    PublicationScanError,
    is_empty_answer,
    scan_artifact_directory,
    validate_publication,
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
    assert (
        is_empty_answer(
            answer="[View report](artifact:unavailable-report)",
            has_primary_report=False,
        )
        is True
    )


def test_only_explicitly_referenced_files_are_settled_to_stable_ids(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.md").write_text(
        "# Report\n\n[Download data](artifact:data.csv)", encoding="utf-8"
    )
    (root / "data.csv").write_text("name,value\na,1\n", encoding="utf-8")
    (root / "scratch.txt").write_text("private intermediate", encoding="utf-8")

    plan = validate_publication(root, answer="Done. [View report](artifact:report.md)")

    assert plan.outcome == {"status": "complete", "issues": []}
    assert [item.relative_path for item in plan.artifacts] == ["report.md", "data.csv"]
    assert all("scratch" not in str(item) for item in plan.descriptors)
    assert "artifact:report.md" not in plan.answer
    assert "artifact:artifact-" in plan.answer
    assert b"artifact:data.csv" not in plan.artifacts[0].content


def test_missing_reference_becomes_an_unavailable_part_issue(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()

    plan = validate_publication(root, answer="[Chart](artifact:missing.png)")

    assert plan.outcome["status"] == "failed"
    assert plan.issues[0].kind == "missing_file"
    assert plan.descriptors[0]["status"] == "unavailable"
    assert str(plan.descriptors[0]["resource_id"]) in plan.answer
    assert "missing.png" not in plan.answer


def test_primary_report_slot_is_unique_and_must_be_referenced(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.md").write_text("# Markdown", encoding="utf-8")
    (root / "report.html").write_text(
        "<!doctype html><html><body>HTML</body></html>", encoding="utf-8"
    )

    ignored = validate_publication(root, answer="Text only")
    attempted = validate_publication(root, answer="[Report](artifact:report.md)")

    assert ignored.artifacts == ()
    assert ignored.outcome["status"] == "failed"
    assert ignored.issues[0].kind == "multiple_primary_reports"
    assert attempted.outcome["status"] == "failed"
    assert attempted.issues[0].kind == "multiple_primary_reports"


def test_media_and_publication_budgets_reject_whole_files(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "bad.pdf").write_text("not a pdf", encoding="utf-8")
    (root / "large.txt").write_text("0123456789", encoding="utf-8")

    mismatch = validate_publication(root, answer="[PDF](artifact:bad.pdf)")
    over_limit = validate_publication(
        root,
        answer="[Text](artifact:large.txt)",
        limits=PublicationLimits(max_file_bytes=4),
    )

    assert mismatch.issues[0].kind == "media_mismatch"
    assert over_limit.issues[0].kind == "file_too_large"
    assert mismatch.artifacts == over_limit.artifacts == ()


def test_svg_static_projection_removes_scripts_events_and_external_links(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "chart.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" onload="steal()">'
        '<script>steal()</script><image href="https://evil.test/x.png"/>'
        '<rect width="10" height="10"/></svg>',
        encoding="utf-8",
    )

    plan = validate_publication(root, answer="![Chart](artifact:chart.svg)")

    assert plan.outcome["status"] == "complete"
    settled = plan.artifacts[0].content.decode("utf-8")
    assert "script" not in settled
    assert "onload" not in settled
    assert "evil.test" not in settled


def test_active_html_must_be_self_contained_and_within_preview_budget(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.html").write_text(
        '<!doctype html><html><body><script src="https://evil.test/x.js"></script></body></html>',
        encoding="utf-8",
    )
    external = validate_publication(root, answer="[Report](artifact:report.html)")
    (root / "report.html").write_text(
        "<!doctype html><html><body><script>document.body.dataset.ok='1'</script></body></html>",
        encoding="utf-8",
    )
    oversized = validate_publication(
        root,
        answer="[Report](artifact:report.html)",
        limits=PublicationLimits(active_html_max_bytes=8),
    )

    assert external.issues[0].kind == "media_mismatch"
    assert oversized.issues[0].kind == "active_preview_too_large"
