# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Artifact publication validates structured roots and relative dependency links."""

import hashlib
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image

from dlightrag.engine.answer.publication import (
    ArtifactAttachment,
    ArtifactValidationError,
    PublicationLimits,
    is_empty_answer,
    prepare_artifact_attachment,
    validate_publication,
)


def _attachment(root: Path, path: str, *, label: str = "") -> ArtifactAttachment:
    try:
        return prepare_artifact_attachment(root, path=path, label=label)
    except ArtifactValidationError:
        try:
            content = (root / path).read_bytes()
        except OSError:
            content = b""
        suffix = Path(path).suffix.casefold()
        presentation = {
            ".md": "markdown",
            ".html": "html",
            ".pdf": "pdf",
            ".png": "image",
            ".svg": "image",
            ".txt": "text",
        }.get(suffix, "download")
        return ArtifactAttachment(
            relative_path=path,
            label=label or Path(path).name,
            content_digest=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            presentation=presentation,  # type: ignore[arg-type]
        )


def _validate(
    root: Path,
    *,
    answer: str,
    attached: tuple[str, ...] = (),
    limits: PublicationLimits | None = None,
):
    return validate_publication(
        root,
        answer=answer,
        attachments=tuple(_attachment(root, path) for path in attached),
        limits=limits,
    )


def _pdf_bytes(*, visual: bool) -> bytes:
    output = BytesIO()
    document = pdfium.PdfDocument.new()
    page = document.new_page(100, 100)
    try:
        if visual:
            source = BytesIO()
            Image.new("RGB", (1, 1), "black").save(source, format="JPEG")
            source.seek(0)
            image = pdfium.PdfImage.new(document)
            image.load_jpeg(source)
            image.set_matrix(pdfium.PdfMatrix().scale(10, 10).translate(5, 5))
            page.insert_obj(image)
            page.gen_content()
        document.save(output)
    finally:
        page.close()
        document.close()
    return output.getvalue()


def test_missing_artifacts_dir_has_no_publication(tmp_path: Path) -> None:
    plan = validate_publication(tmp_path / "artifacts", answer="Text only")

    assert plan.artifacts == ()
    assert plan.outcome == {"status": "complete", "issues": []}


def test_correction_feedback_preserves_markdown_citation_contract(tmp_path: Path) -> None:
    plan = validate_publication(
        tmp_path / "artifacts",
        answer="[Open analysis](artifact:analysis.md)",
    )

    feedback = plan.correction_feedback()
    assert "call attach_artifact after its final modification" in feedback
    assert "same inline Citation Contract" in feedback
    assert "independently for each Markdown Artifact" in feedback


def test_attached_artifact_satisfies_answer_output(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "analysis.md").write_text("  \n", encoding="utf-8")

    plan = _validate(
        root,
        answer="[Open analysis](artifact:analysis.md)",
        attached=("analysis.md",),
    )

    assert [item.relative_path for item in plan.artifacts] == ["analysis.md"]
    assert plan.outcome == {"status": "complete", "issues": []}
    assert is_empty_answer(answer=plan.answer, has_artifacts=bool(plan.artifacts)) is False


def test_attached_root_is_placed_when_answer_omits_it(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "analysis.md").write_text("Analysis", encoding="utf-8")
    attachment = _attachment(root, "analysis.md", label="Open analysis")

    plan = validate_publication(root, answer="Done.", attachments=(attachment,))

    assert [item.relative_path for item in plan.artifacts] == ["analysis.md"]
    assert "Done.\n\n[Open analysis](artifact:artifact-" in plan.answer


def test_omitted_roots_are_placed_in_attachment_order_without_placing_dependencies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "beta.md").write_text("[Data](artifact:data.csv)", encoding="utf-8")
    (root / "alpha.md").write_text("Alpha", encoding="utf-8")
    (root / "data.csv").write_text("value\n1\n", encoding="utf-8")
    attachments = (
        _attachment(root, "beta.md", label="Beta report"),
        _attachment(root, "alpha.md", label="Alpha report"),
    )

    plan = validate_publication(root, answer="Done.", attachments=attachments)

    assert [item.relative_path for item in plan.artifacts] == [
        "beta.md",
        "alpha.md",
        "data.csv",
    ]
    assert plan.answer.index("[Beta report]") < plan.answer.index("[Alpha report]")
    assert "[Data]" not in plan.answer


def test_attached_root_modified_after_attachment_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    path = root / "analysis.md"
    path.write_text("first", encoding="utf-8")
    attachment = _attachment(root, "analysis.md")
    path.write_text("second", encoding="utf-8")

    plan = validate_publication(root, answer="Done.", attachments=(attachment,))

    assert plan.artifacts == ()
    assert plan.issues[0].kind == "stale_attachment"


def test_attachment_digest_binds_raw_content_before_svg_sanitization(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    path = root / "chart.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><script>a()</script><rect/></svg>',
        encoding="utf-8",
    )
    attachment = _attachment(root, "chart.svg")
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><script>b()</script><rect/></svg>',
        encoding="utf-8",
    )

    plan = validate_publication(root, answer="Done.", attachments=(attachment,))

    assert plan.artifacts == ()
    assert plan.issues[0].kind == "stale_attachment"


def test_valid_empty_html_and_pdf_are_published_when_referenced(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "page.html").write_text("<!doctype html><html><body></body></html>", encoding="utf-8")
    (root / "document.pdf").write_bytes(_pdf_bytes(visual=False))

    plan = _validate(
        root,
        answer="[Open page](artifact:page.html) [Open PDF](artifact:document.pdf)",
        attached=("page.html", "document.pdf"),
    )

    assert [item.relative_path for item in plan.artifacts] == ["page.html", "document.pdf"]
    assert plan.outcome == {"status": "complete", "issues": []}


@pytest.mark.parametrize("filename", ["analysis.md", "page.html"])
def test_malformed_text_artifact_is_rejected_as_media_mismatch(
    tmp_path: Path, filename: str
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / filename).write_bytes(b"\xff\xfe")

    plan = _validate(
        root,
        answer=f"[Open Artifact](artifact:{filename})",
        attached=(filename,),
    )

    assert plan.artifacts == ()
    assert plan.issues[0].kind == "media_mismatch"


def test_malformed_pdf_artifact_is_rejected_as_media_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "document.pdf").write_bytes(b"%PDF-1.7\n%%EOF")

    plan = _validate(
        root,
        answer="[Open PDF](artifact:document.pdf)",
        attached=("document.pdf",),
    )

    assert plan.artifacts == ()
    assert plan.issues[0].kind == "media_mismatch"


def test_visual_pdf_artifact_is_published(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "document.pdf").write_bytes(_pdf_bytes(visual=True))

    plan = _validate(
        root,
        answer="[Open PDF](artifact:document.pdf)",
        attached=("document.pdf",),
    )

    assert plan.outcome == {"status": "complete", "issues": []}
    assert [item.relative_path for item in plan.artifacts] == ["document.pdf"]


def test_oversized_artifact_is_rejected_before_content_validation(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "analysis.md").write_bytes(b"large")

    plan = _validate(
        root,
        answer="[Open analysis](artifact:analysis.md)",
        attached=("analysis.md",),
        limits=PublicationLimits(max_file_bytes=4),
    )

    assert plan.artifacts == ()
    assert plan.issues[0].kind == "file_too_large"


def test_symlink_makes_publication_unavailable(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "link.md").symlink_to(tmp_path / "outside.md")

    plan = _validate(
        root,
        answer="[Open link](artifact:link.md)",
        attached=("link.md",),
    )

    assert plan.artifacts == ()
    assert plan.issues[0].kind == "unsafe_file"


def test_empty_answer_requires_neither_artifact_nor_text() -> None:
    assert is_empty_answer(answer="  ", has_artifacts=False) is True
    assert is_empty_answer(answer="42", has_artifacts=False) is False
    assert is_empty_answer(answer="", has_artifacts=True) is False
    assert (
        is_empty_answer(
            answer="[Open Artifact](artifact:unavailable-artifact)",
            has_artifacts=False,
        )
        is True
    )


def test_any_markdown_artifact_can_publish_linked_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "peer_analysis.md").write_text(
        "# Analysis\n\n[Download data](artifact:data.csv)", encoding="utf-8"
    )
    (root / "data.csv").write_text("name,value\na,1\n", encoding="utf-8")
    (root / "scratch.txt").write_text("private intermediate", encoding="utf-8")

    plan = _validate(
        root,
        answer="Done. [Open analysis](artifact:peer_analysis.md)",
        attached=("peer_analysis.md",),
    )

    assert plan.outcome == {"status": "complete", "issues": []}
    assert [item.relative_path for item in plan.artifacts] == ["peer_analysis.md", "data.csv"]
    assert all("scratch" not in str(item) for item in plan.descriptors)
    assert "artifact:peer_analysis.md" not in plan.answer
    assert "artifact:artifact-" in plan.answer
    assert b"artifact:data.csv" not in plan.artifacts[0].content


def test_any_html_artifact_can_publish_linked_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "dashboard.html").write_text(
        '<!doctype html><html><body><a href="artifact:data.csv">Data</a></body></html>',
        encoding="utf-8",
    )
    (root / "data.csv").write_text("name,value\na,1\n", encoding="utf-8")

    plan = _validate(
        root,
        answer="[Open dashboard](artifact:dashboard.html)",
        attached=("dashboard.html",),
    )

    assert [item.relative_path for item in plan.artifacts] == ["dashboard.html", "data.csv"]
    assert b"artifact:data.csv" not in plan.artifacts[0].content


def test_invalid_nested_markdown_reference_is_settled_to_an_unavailable_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "analysis.md").write_text("[Unsafe](artifact:../secret.txt)", encoding="utf-8")

    plan = _validate(
        root,
        answer="[Open analysis](artifact:analysis.md)",
        attached=("analysis.md",),
    )

    assert [item.relative_path for item in plan.artifacts] == ["analysis.md"]
    unavailable = next(item for item in plan.descriptors if item["status"] == "unavailable")
    assert unavailable["label"] == "Unsafe"
    assert b"artifact:../secret.txt" not in plan.artifacts[0].content
    assert f"artifact:{unavailable['resource_id']}".encode() in plan.artifacts[0].content


def test_invalid_nested_html_reference_is_settled_to_an_unavailable_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "dashboard.html").write_text(
        '<!doctype html><html><body><a href="artifact:../secret.txt">Unsafe</a></body></html>',
        encoding="utf-8",
    )

    plan = _validate(
        root,
        answer="[Open dashboard](artifact:dashboard.html)",
        attached=("dashboard.html",),
    )

    assert [item.relative_path for item in plan.artifacts] == ["dashboard.html"]
    unavailable = next(item for item in plan.descriptors if item["status"] == "unavailable")
    assert b"artifact:../secret.txt" not in plan.artifacts[0].content
    assert f"artifact:{unavailable['resource_id']}".encode() in plan.artifacts[0].content


def test_nested_artifact_reference_cycles_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "a.md").write_text("[B](artifact:b.md)", encoding="utf-8")
    (root / "b.md").write_text("[A](artifact:a.md)", encoding="utf-8")

    plan = _validate(root, answer="[A](artifact:a.md)", attached=("a.md",))

    assert plan.artifacts == ()
    assert {issue.kind for issue in plan.issues} == {"reference_cycle"}


def test_cycles_do_not_consume_the_artifact_admission_limit(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "a.md").write_text("[B](artifact:b.md)", encoding="utf-8")
    (root / "b.md").write_text("[A](artifact:a.md)", encoding="utf-8")
    (root / "c.md").write_text("[D](artifact:d.md)", encoding="utf-8")
    (root / "d.md").write_text("D", encoding="utf-8")

    plan = _validate(
        root,
        answer="[A](artifact:a.md) [C](artifact:c.md)",
        attached=("a.md", "c.md"),
        limits=PublicationLimits(max_artifacts=2),
    )

    assert [item.relative_path for item in plan.artifacts] == ["c.md", "d.md"]
    assert {issue.kind for issue in plan.issues} == {"reference_cycle"}
    assert b"artifact:d.md" not in plan.artifacts[0].content


def test_unattached_reference_does_not_authorize_an_existing_file(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.md").write_text("private draft", encoding="utf-8")

    plan = validate_publication(root, answer="[Report](artifact:report.md)")

    assert plan.outcome["status"] == "failed"
    assert plan.artifacts == ()
    assert plan.issues[0].kind == "unattached_reference"
    assert plan.descriptors[0]["status"] == "unavailable"
    assert "role" not in plan.descriptors[0]
    assert str(plan.descriptors[0]["resource_id"]) in plan.answer
    assert "missing.png" not in plan.answer


def test_multiple_artifacts_with_report_like_names_are_independent(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.md").write_text("# Markdown", encoding="utf-8")
    (root / "report.html").write_text(
        "<!doctype html><html><body>HTML</body></html>", encoding="utf-8"
    )
    (root / "report.pdf").write_bytes(_pdf_bytes(visual=True))

    plan = _validate(
        root,
        answer=(
            "[Markdown](artifact:report.md) [HTML](artifact:report.html) [PDF](artifact:report.pdf)"
        ),
        attached=("report.md", "report.html", "report.pdf"),
    )

    assert [item.relative_path for item in plan.artifacts] == [
        "report.md",
        "report.html",
        "report.pdf",
    ]
    assert plan.outcome == {"status": "complete", "issues": []}
    assert all("role" not in descriptor for descriptor in plan.descriptors)


def test_media_and_publication_budgets_reject_whole_files(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "bad.pdf").write_text("not a pdf", encoding="utf-8")
    (root / "large.txt").write_text("0123456789", encoding="utf-8")

    mismatch = _validate(root, answer="[PDF](artifact:bad.pdf)", attached=("bad.pdf",))
    over_limit = _validate(
        root,
        answer="[Text](artifact:large.txt)",
        attached=("large.txt",),
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

    plan = _validate(
        root,
        answer="![Chart](artifact:chart.svg)",
        attached=("chart.svg",),
    )

    assert plan.outcome["status"] == "complete"
    settled = plan.artifacts[0].content.decode("utf-8")
    assert "script" not in settled
    assert "onload" not in settled
    assert "evil.test" not in settled


def test_svg_static_projection_rejects_nested_svg_data_but_keeps_raster_data(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "chart.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<image id="active" href="data:image/svg+xml,%3Csvg%20onload%3Dsteal()%3E"/>'
        '<image id="raster" href="data:image/png;base64,iVBORw0KGgo="/>'
        "</svg>",
        encoding="utf-8",
    )

    plan = _validate(
        root,
        answer="![Chart](artifact:chart.svg)",
        attached=("chart.svg",),
    )

    assert plan.outcome["status"] == "complete"
    settled = plan.artifacts[0].content.decode("utf-8")
    assert "data:image/svg+xml" not in settled
    assert "data:image/png;base64,iVBORw0KGgo=" in settled


def test_active_html_must_be_self_contained_and_within_preview_budget(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "page.html").write_text(
        '<!doctype html><html><body><script src="https://evil.test/x.js"></script></body></html>',
        encoding="utf-8",
    )
    external = _validate(
        root,
        answer="[Open page](artifact:page.html)",
        attached=("page.html",),
    )
    (root / "page.html").write_text(
        "<!doctype html><html><body><script>document.body.dataset.ok='1'</script></body></html>",
        encoding="utf-8",
    )
    oversized = _validate(
        root,
        answer="[Open page](artifact:page.html)",
        attached=("page.html",),
        limits=PublicationLimits(active_html_max_bytes=8),
    )

    assert external.issues[0].kind == "media_mismatch"
    assert oversized.issues[0].kind == "active_preview_too_large"
