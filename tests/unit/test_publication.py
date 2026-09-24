# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Artifact publication validates structured roots and relative dependency links."""

import base64
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
    artifact_link,
    artifact_resource_id,
    is_empty_answer,
    prepare_artifact_attachment,
    validate_publication,
)
from dlightrag.engine.answer.results import answer_parts_from_markdown


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


# One real H.264/MP4 Artifact, 32x32 and 0.2s, so video publication is proved
# against actual container bytes instead of a hand-built header. Regenerate with:
#   ffmpeg -f lavfi -i color=c=black:size=32x32:rate=5 -t 0.2 -c:v libx264 \
#          -pix_fmt yuv420p -crf 51 -preset ultrafast tiny.mp4
_MP4_BYTES = base64.b64decode(
    "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAAm5tZGF0AAACUwYF//9P3EXpvebZSLeW"
    "LNgg2SPu73gyNjQgLSBjb3JlIDE2NSByMzIyMiBiMzU2MDVhIC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENv"
    "cHlsZWZ0IDIwMDMtMjAyNSAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNh"
    "YmFjPTAgcmVmPTEgZGVibG9jaz0wOjA6MCBhbmFseXNlPTA6MCBtZT1kaWEgc3VibWU9MCBwc3k9MSBwc3lfcmQ9"
    "MS4wMDowLjAwIG1peGVkX3JlZj0wIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MCA4eDhkY3Q9MCBj"
    "cW09MCBkZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0wIHRocmVhZHM9MSBsb29r"
    "YWhlYWRfdGhyZWFkcz0xIHNsaWNlZF90aHJlYWRzPTAgbnI9MCBkZWNpbWF0ZT0xIGludGVybGFjZWQ9MCBibHVy"
    "YXlfY29tcGF0PTAgY29uc3RyYWluZWRfaW50cmE9MCBiZnJhbWVzPTAgd2VpZ2h0cD0wIGtleWludD0yNTAga2V5"
    "aW50X21pbj01IHNjZW5lY3V0PTAgaW50cmFfcmVmcmVzaD0wIHJjPWNyZiBtYnRyZWU9MCBjcmY9NTEuMCBxY29t"
    "cD0wLjYwIHFwbWluPTAgcXBtYXg9NjkgcXBzdGVwPTQgaXBfcmF0aW89MS40MCBhcT0wAIAAAAALZYiEOiYoABWT"
    "rrwAAAMObW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAAMgAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAA"
    "AAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAjl0cmFrAAAAXHRr"
    "aGQAAAADAAAAAAAAAAAAAAABAAAAAAAAAMgAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAA"
    "AAAAAAAAAABAAAAAACAAAAAgAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAADIAAAAAAABAAAAAAGxbWRpYQAA"
    "ACBtZGhkAAAAAAAAAAAAAAAAAAAoAAAACABVxAAAAAAALWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRl"
    "b0hhbmRsZXIAAAABXG1pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1"
    "cmwgAAAAAQAAARxzdGJsAAAAuHN0c2QAAAAAAAAAAQAAAKhhdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAACAA"
    "IABIAAAASAAAAAAAAAABFUxhdmM2Mi4xMS4xMDAgbGlieDI2NAAAAAAAAAAAAAAAGP//AAAALmF2Y0MBQsAK/+EA"
    "FmdCwAraJbARAAADAAEAAAMACg8SJqABAAVozgGXIAAAABBwYXNwAAAAAQAAAAEAAAAUYnRydAAAAAAAAF/wAAAA"
    "AAAAABhzdHRzAAAAAAAAAAEAAAABAAAIAAAAABxzdHNjAAAAAAAAAAEAAAABAAAAAQAAAAEAAAAUc3RzegAAAAAA"
    "AAJmAAAAAQAAABRzdGNvAAAAAAAAAAEAAAAwAAAAYXVkdGEAAABZbWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAbWRp"
    "cmFwcGwAAAAAAAAAAAAAAAAsaWxzdAAAACSpdG9vAAAAHGRhdGEAAAABAAAAAExhdmY2Mi4zLjEwMA=="
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


def test_video_artifact_is_published_for_native_playback(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "clip.mp4").write_bytes(_MP4_BYTES)

    plan = _validate(root, answer="![clip](artifact:clip.mp4)", attached=("clip.mp4",))

    assert plan.outcome == {"status": "complete", "issues": []}
    (artifact,) = plan.artifacts
    assert (artifact.media_type, artifact.presentation) == ("video/mp4", "video")
    assert artifact.descriptor()["presentation"] == "video"


@pytest.mark.parametrize("filename", ["clip.mp4", "clip.mov", "clip.webm"])
def test_video_container_that_contradicts_its_extension_is_rejected(
    tmp_path: Path, filename: str
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / filename).write_bytes(b"\x00" * 4096)

    plan = _validate(
        root,
        answer=f"![clip](artifact:{filename})",
        attached=(filename,),
    )

    assert plan.artifacts == ()
    assert plan.issues[0].kind == "media_mismatch"


def test_video_container_does_not_substitute_for_another_container(tmp_path: Path) -> None:
    """An MP4 renamed to ``.webm`` is refused rather than handed to a player."""
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "clip.webm").write_bytes(_MP4_BYTES)

    plan = _validate(
        root,
        answer="![clip](artifact:clip.webm)",
        attached=("clip.webm",),
    )

    assert plan.artifacts == ()
    assert plan.issues[0].kind == "media_mismatch"


def test_omitted_video_affordance_stays_a_card(tmp_path: Path) -> None:
    """A video the answer never placed is offered as a card, not a player.

    Inline playback is the Answer's own explicit placement, so the framework's
    trailing affordance must not spend the reading column on a player.
    """
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "clip.mp4").write_bytes(_MP4_BYTES)

    plan = _validate(root, answer="See the recording.", attached=("clip.mp4",))

    assert plan.outcome == {"status": "complete", "issues": []}
    assert "[clip.mp4](artifact:artifact-" in plan.answer
    assert "![clip.mp4]" not in plan.answer


def test_answer_placed_video_is_inline(tmp_path: Path) -> None:
    """The Answer's own placement marker is what makes a video play in place."""
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "clip.mp4").write_bytes(_MP4_BYTES)

    plan = _validate(root, answer="![clip](artifact:clip.mp4)", attached=("clip.mp4",))

    (part,) = [
        part
        for part in answer_parts_from_markdown(
            plan.answer,
            artifacts=[item.descriptor() for item in plan.artifacts],
            evidence_images=[],
            artifact_bindings=plan.artifact_bindings,
        )
        if part["type"] == "artifact"
    ]

    assert part["inline"] is True
    assert part["artifact"]["presentation"] == "video"


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
    assert plan.answer == "Done. [Open analysis](artifact:peer_analysis.md)"
    assert plan.artifact_bindings == {"artifact:peer_analysis.md": plan.artifacts[0].resource_id}
    assert plan.artifacts[0].content == (root / "peer_analysis.md").read_bytes()
    assert plan.artifacts[0].artifact_bindings == {
        "artifact:data.csv": plan.artifacts[1].resource_id
    }


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
    assert plan.artifacts[0].content == (root / "dashboard.html").read_bytes()
    assert plan.artifacts[0].artifact_bindings == {
        "artifact:data.csv": plan.artifacts[1].resource_id
    }


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
    assert b"artifact:../secret.txt" in plan.artifacts[0].content
    assert plan.artifacts[0].artifact_bindings == {
        "artifact:../secret.txt": unavailable["resource_id"]
    }


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
    assert b"artifact:../secret.txt" in plan.artifacts[0].content
    assert plan.artifacts[0].artifact_bindings == {
        "artifact:../secret.txt": unavailable["resource_id"]
    }


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
    assert plan.artifacts[0].content == (root / "c.md").read_bytes()
    assert plan.artifacts[0].artifact_bindings == {"artifact:d.md": plan.artifacts[1].resource_id}


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
    assert plan.answer == "[Report](artifact:report.md)"
    assert plan.artifact_bindings["artifact:report.md"] == plan.descriptors[0]["resource_id"]
    assert "missing.png" not in plan.answer


@pytest.mark.parametrize(
    "answer",
    [
        "**[artifacts/report.html](artifacts/report.html)**",
        "[Report](./artifacts/report.html)",
        "[Report][report]\n\n[report]: artifacts/report.html",
        "![Report](artifacts/report.html)",
    ],
)
def test_workspace_links_require_publication_correction(tmp_path: Path, answer: str) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.html").write_text("<!doctype html><html><body>Report</body></html>")

    plan = validate_publication(root, answer=answer)

    assert plan.repairable
    assert plan.outcome["status"] == "failed"
    assert plan.artifacts == ()
    assert plan.issues[0].kind == "invalid_reference"
    assert "attach_artifact" in plan.correction_feedback()
    assert plan.descriptors[0]["status"] == "unavailable"
    parts = answer_parts_from_markdown(
        plan.answer,
        artifacts=plan.descriptors,
        evidence_images=(),
        artifact_bindings=plan.artifact_bindings,
    )
    assert any(
        part["type"] == "artifact"
        and part["artifact"]["resource_id"] == plan.descriptors[0]["resource_id"]
        for part in parts
    )


def test_workspace_link_correction_still_requires_explicit_attachment(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.html").write_text("<!doctype html><html><body>Report</body></html>")

    unattached = validate_publication(root, answer="[Report](artifact:report.html)")
    assert unattached.issues[0].kind == "unattached_reference"
    assert unattached.artifacts == ()

    corrected = _validate(root, answer="[Report](artifact:report.html)", attached=("report.html",))
    assert not corrected.repairable
    assert len(corrected.artifacts) == 1


@pytest.mark.parametrize(
    "answer",
    [
        "`[Report](artifacts/report.html)`",
        "```markdown\n[Report](artifacts/report.html)\n```",
        "    [Report](artifacts/report.html)",
        "[unused]: artifacts/report.html",
        "Use artifacts/report.html as a workspace path.",
        "[Example](https://example.com/artifacts/report.html)",
    ],
)
def test_workspace_path_examples_do_not_request_publication(tmp_path: Path, answer: str) -> None:
    plan = validate_publication(tmp_path / "artifacts", answer=answer)

    assert not plan.repairable
    assert plan.answer == answer
    assert plan.descriptors == ()


@pytest.mark.parametrize(
    "answer",
    [
        "`[Report](artifact:report.html)`",
        "```markdown\n[Report](artifact:report.html)\n```",
        "    [Report](artifact:report.html)",
        "[unused]: artifact:report.html",
        "$[Report](artifact:report.html)$",
    ],
)
def test_artifact_examples_are_inert_across_publication_and_parts(
    tmp_path: Path, answer: str
) -> None:
    plan = validate_publication(tmp_path / "artifacts", answer=answer)

    assert plan.outcome == {"status": "complete", "issues": []}
    assert plan.answer == answer
    assert answer_parts_from_markdown(
        plan.answer,
        artifacts=[{"resource_id": "report.html", "label": "Report"}],
        evidence_images=(),
    ) == [{"type": "markdown", "text": answer}]


@pytest.mark.parametrize(
    "answer",
    [
        "[Report][r]\n\n[r]: artifact:report.html",
        "[Report][]\n\n[Report]: artifact:report.html",
        "[Report]\n\n[Report]: artifact:report.html",
        "[**Report**](artifact:report.html)",
        "[Report \\[final\\]](artifact:report.html)",
    ],
)
def test_all_markdown_artifact_references_require_attachment(tmp_path: Path, answer: str) -> None:
    plan = validate_publication(tmp_path / "artifacts", answer=answer)

    assert plan.outcome["status"] == "failed"
    assert [issue.kind for issue in plan.issues] == ["unattached_reference"]
    parts = answer_parts_from_markdown(
        plan.answer,
        artifacts=plan.descriptors,
        evidence_images=(),
        artifact_bindings=plan.artifact_bindings,
    )
    assert len([part for part in parts if part["type"] == "artifact"]) == 1


@pytest.mark.parametrize("target", ["./report.html", "docs/report.html", "/tmp/report.html"])
def test_unresolvable_answer_links_fail_in_place(tmp_path: Path, target: str) -> None:
    answer = f"Before [Report]({target}) after.\n\n`[Report]({target})`"
    plan = validate_publication(tmp_path / "artifacts", answer=answer)

    assert plan.outcome["status"] == "failed"
    assert plan.artifacts == ()
    assert f"`[Report]({target})`" in plan.answer
    parts = answer_parts_from_markdown(
        plan.answer,
        artifacts=plan.descriptors,
        evidence_images=(),
        artifact_bindings=plan.artifact_bindings,
    )
    assert plan.answer == answer
    assert [part["type"] for part in parts] == ["markdown", "artifact"]
    assert parts[0]["text"] == answer
    assert parts[1]["target"] == target
    assert parts[1]["slot"] == 0


def test_reference_style_dependencies_publish_but_examples_do_not(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.md").write_text(
        "[Data][d]\n\n[d]: artifact:data.csv\n\n`[Example](artifact:missing.csv)`"
    )
    (root / "data.csv").write_text("value\n1\n")

    plan = _validate(root, answer="[Report][r]\n\n[r]: artifact:report.md", attached=("report.md",))

    assert plan.outcome == {"status": "complete", "issues": []}
    assert [item.relative_path for item in plan.artifacts] == ["report.md", "data.csv"]
    assert b"`[Example](artifact:missing.csv)`" in plan.artifacts[0].content
    parts = answer_parts_from_markdown(
        plan.answer,
        artifacts=plan.descriptors,
        evidence_images=(),
        artifact_bindings=plan.artifact_bindings,
    )
    assert len([part for part in parts if part["type"] == "artifact"]) == 1


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


def test_attachment_receipt_round_trips_without_rewriting_or_duplicate_placement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report [final].md").write_text("Report")
    attachment = _attachment(root, "report [final].md", label="Report [final]")
    answer = "Ready. " + artifact_link(attachment)

    plan = validate_publication(root, answer=answer, attachments=(attachment,))

    assert plan.answer == answer
    assert plan.outcome["status"] == "complete"
    assert plan.artifact_bindings == {
        f"artifact:{artifact_resource_id(attachment.relative_path)}": plan.artifacts[0].resource_id
    }


def test_canonical_dependency_ids_resolve_within_the_authorized_closure(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    dependency_id = artifact_resource_id("data.csv")
    (root / "report.md").write_text(f"[Data](artifact:{dependency_id})")
    (root / "data.csv").write_text("value\n1\n")
    (root / "private.txt").write_text("private")
    answer = f"[Report](artifact:report.md) [Data](artifact:{dependency_id})"

    plan = _validate(root, answer=answer, attached=("report.md",))

    assert [item.relative_path for item in plan.artifacts] == ["report.md", "data.csv"]
    assert plan.answer == answer
    assert plan.artifact_bindings[f"artifact:{dependency_id}"] == dependency_id
    assert plan.artifacts[0].artifact_bindings == {f"artifact:{dependency_id}": dependency_id}
    assert not _validate(
        root, answer=f"[Private](artifact:{artifact_resource_id('private.txt')})"
    ).artifacts


def test_document_scopes_bind_identical_relative_targets_independently(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    for folder in ("one", "two"):
        (root / folder).mkdir(parents=True)
        (root / folder / "report.md").write_text("[Data](artifact:data.csv)")
        (root / folder / "data.csv").write_text(f"value\n{folder}\n")

    plan = _validate(root, answer="Reports", attached=("one/report.md", "two/report.md"))
    reports = {item.relative_path: item for item in plan.artifacts}

    assert plan.outcome["status"] == "complete"
    for folder in ("one", "two"):
        report = reports[f"{folder}/report.md"]
        assert report.content == b"[Data](artifact:data.csv)"
        assert report.artifact_bindings == {
            "artifact:data.csv": reports[f"{folder}/data.csv"].resource_id
        }
        assert report.descriptor()["artifact_bindings"] == report.artifact_bindings


def test_invalid_markdown_dependency_keeps_source_and_binds_failure_in_document(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    source = "[Download](./data.csv)\n\n`[Example](./data.csv)`"
    (root / "report.md").write_text(source)
    (root / "data.csv").write_text("private draft")

    plan = _validate(root, answer="[Report](artifact:report.md)", attached=("report.md",))

    assert plan.outcome["status"] == "partial"
    assert [item.relative_path for item in plan.artifacts] == ["report.md"]
    report = plan.artifacts[0]
    failure = next(item for item in plan.descriptors if item["status"] == "unavailable")
    assert report.content.decode() == source
    assert report.artifact_bindings == {"./data.csv": failure["resource_id"]}
    assert "./data.csv" not in plan.artifact_bindings


def test_html_dependencies_come_from_attributes_not_comments_or_script_text(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    source = (
        '<html><body><!-- <a href="artifact:missing.csv">ignored</a> -->'
        "<script>const example = '<a href=\"artifact:also-missing.csv\">example</a>';</script>"
        '<a href="artifact:data&#46;csv">Data</a></body></html>'
    )
    (root / "report.html").write_text(source)
    (root / "data.csv").write_text("value\n1\n")

    plan = _validate(root, answer="[Report](artifact:report.html)", attached=("report.html",))

    assert plan.outcome["status"] == "complete"
    assert [item.relative_path for item in plan.artifacts] == ["report.html", "data.csv"]
    assert plan.artifacts[0].content.decode() == source
    assert plan.artifacts[0].artifact_bindings == {
        "artifact:data.csv": plan.artifacts[1].resource_id
    }


def test_unfinished_fence_cannot_hide_an_automatically_placed_root(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.txt").write_text("Report")
    answer = "Example:\n\n```markdown\n[Report](artifact:report.txt)"

    plan = _validate(root, answer=answer, attached=("report.txt",))

    assert plan.outcome["status"] == "complete"
    assert plan.answer.endswith(answer)
    assert plan.answer.startswith("[report.txt](artifact:artifact-")
    assert len(plan.artifact_bindings) == 1
    parts = answer_parts_from_markdown(
        plan.answer,
        artifacts=plan.descriptors,
        evidence_images=(),
        artifact_bindings=plan.artifact_bindings,
    )
    assert len([part for part in parts if part["type"] == "artifact"]) == 1


def test_unsafe_inventory_binds_actual_references_without_changing_examples(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "unsafe").symlink_to(tmp_path / "outside")
    answer = "[Report](artifact:report.md) [Local](./report.md)\n\n`[Example](artifact:example.md)`"

    plan = validate_publication(root, answer=answer)

    assert plan.answer == answer
    assert plan.outcome["status"] == "failed"
    assert set(plan.artifact_bindings) == {"artifact:report.md", "./report.md"}
    assert len(plan.descriptors) == 2


@pytest.mark.parametrize(
    ("answer", "empty"),
    [
        ("[Report][r]\n\n[r]: artifact:report.md", True),
        ("**[Report](artifact:report.md)**", True),
        ("`[Report](artifact:report.md)`", False),
        ("```md\n[Report](artifact:report.md)\n```", False),
        ("[unused]: artifact:report.md", True),
        ("![](https://example.com/chart.png)", False),
    ],
)
def test_answer_emptiness_uses_visible_markdown_semantics(answer: str, empty: bool) -> None:
    assert is_empty_answer(answer=answer, has_artifacts=False) is empty
