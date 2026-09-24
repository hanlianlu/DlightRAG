# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Reference consumers share actual grammar occurrences and target classification."""

import pytest

from dlightrag.engine.answer.markdown import answer_markdown
from dlightrag.engine.answer.reference import (
    Reference,
    classify_target,
    html_references,
    inline_references,
    markdown_references,
    resolve_artifact_target,
)


def test_markdown_references_preserve_order_labels_and_reference_style_targets() -> None:
    text = (
        '[A **bold** `label`](artifact:report%20one.md "title")\n\n'
        "[ref][report]\n\n![image](artifact:chart.png)\n\n"
        "[repeat][report]\n\n[report]: artifact:report%20one.md\n"
    )
    assert markdown_references(text) == (
        Reference("artifact:report%20one.md", "A bold label", False),
        Reference("artifact:report%20one.md", "ref", False),
        Reference("artifact:chart.png", "image", True),
        Reference("artifact:report%20one.md", "repeat", False),
    )


def test_escaped_label_and_balanced_destination_use_parser_values() -> None:
    assert markdown_references(r"[A \] bracket](artifact:report(one).md)") == (
        Reference("artifact:report(one).md", "A ] bracket", False),
    )


@pytest.mark.parametrize(
    "text",
    [
        "`[report](artifact:report.md)`",
        "```md\n[report](artifact:report.md)\n```",
        "    [report](artifact:report.md)",
        "$[report](artifact:report.md)$",
        r"\([report](artifact:report.md)\)",
        "[unused]: artifact:report.md",
        '<a href="artifact:report.md">literal HTML</a>',
        "An artifact:report.md mentioned in text.",
    ],
)
def test_non_references_stay_inert(text: str) -> None:
    assert markdown_references(text) == ()


def test_linked_images_are_real_but_image_alt_does_not_create_links() -> None:
    text = (
        "[![chart](artifact:chart.png)](https://example.com) "
        "![alt [child](artifact:child.md)](artifact:image.png)"
    )
    assert markdown_references(text) == (
        Reference("https://example.com", "chart", False),
        Reference("artifact:chart.png", "chart", True),
        Reference("artifact:image.png", "alt [child](artifact:child.md)", True),
    )
    children = next(
        block.children for block in answer_markdown().parse(text) if block.type == "inline"
    )
    assert children is not None
    outer, nested, standalone = inline_references(children)
    assert outer.link_span is None
    assert nested.link_span == (outer.start, outer.end)
    assert children[nested.start].type == "image"
    assert standalone.link_span is None


def test_inline_ranges_cover_whole_occurrence_without_consuming_other_tokens() -> None:
    blocks = answer_markdown().parse("**before [open](artifact:report.md) after** ![x](evidence:x)")
    children = next(block.children for block in blocks if block.type == "inline")
    assert children is not None
    references = tuple(inline_references(children))
    assert len(references) == 2
    assert [token.type for token in children[references[0].start : references[0].end]] == [
        "link_open",
        "text",
        "link_close",
    ]
    assert [token.type for token in children[references[1].start : references[1].end]] == ["image"]
    covered = {index for reference in references for index in range(reference.start, reference.end)}
    remaining = "".join(
        token.content for index, token in enumerate(children) if index not in covered
    )
    assert remaining == "before  after "


def test_html_reads_attributes_and_ignores_comments_scripts_and_text() -> None:
    text = """
    <!-- <img src="artifact:comment.png"> -->
    <script>const example = '<a href="artifact:script.md">';</script>
    <style>.x::after { content: '<img src="artifact:style.png">'; }</style>
    <p title='src="artifact:title.png"' data-src="artifact:data.png">href="artifact:text.md"</p>
    <a HREF='artifact:report.md?x=1&amp;y=2'>Report</a>
    <img src=artifact:chart.png />
    <a href="https://example.com"><img src="artifact:nested.png"></a>
    """
    assert html_references(text) == (
        Reference("artifact:report.md?x=1&y=2", "", False),
        Reference("artifact:chart.png", "", True),
        Reference("https://example.com", "", False),
        Reference("artifact:nested.png", "", True),
    )


@pytest.mark.parametrize(
    ("target", "image", "kind"),
    [
        ("ArTiFaCt:report.md", False, "artifact"),
        ("artifact:", False, "artifact"),
        ("EVIDENCE:shot", True, "evidence"),
        ("https://example.com/path?q=x#part", False, "external"),
        ("HTTP://localhost:8000/file", True, "external"),
        ("https://[::1]:8080/", False, "external"),
        ("https://", False, "unsupported"),
        ("https://example.com:bad/path", False, "unsupported"),
        ("https://example.com:70000/path", False, "unsupported"),
        ("https://exam ple.com/path", False, "unsupported"),
        ("https://example.com\\other/path", False, "unsupported"),
        ("artifacts/report.html", False, "unsupported"),
        ("./artifacts/report.html", True, "unsupported"),
        ("/tmp/report.html", False, "unsupported"),
        ("file:///tmp/report.html", False, "unsupported"),
        ("//example.com/report.html", False, "unsupported"),
        ("mailto:user@example.com", False, "unsupported"),
        ("data:image/png;base64,YQ==", True, "embedded"),
        ("DATA:IMAGE/PNG;BASE64,YQ==", True, "embedded"),
        ("data:image/avif;base64,YQ==", True, "unsupported"),
        ("data:image/bmp;base64,YQ==", True, "unsupported"),
        ("data:image/png;base64,YQ==", False, "unsupported"),
        ("data:image/svg+xml;base64,YQ==", True, "unsupported"),
        ("data:text/html;base64,YQ==", True, "unsupported"),
        ("blob:https://example.com/image-id", True, "embedded"),
        ("blob:https://example.com/image-id", False, "unsupported"),
        ("blob:", True, "unsupported"),
        ("", False, "unsupported"),
    ],
)
def test_target_classification(target: str, image: bool, kind: str) -> None:
    assert classify_target(target, image=image) == kind


def test_resolution_uses_explicit_bindings_then_artifact_id_without_guessing_paths() -> None:
    bindings = {"artifact:report.md": "artifact-123", "artifacts/missing.md": "unavailable-123"}
    assert resolve_artifact_target("artifact:report.md", bindings) == "artifact-123"
    assert resolve_artifact_target("artifacts/missing.md", bindings) == "unavailable-123"
    assert resolve_artifact_target("ARTIFACT:artifact-456", bindings) == "artifact-456"
    assert resolve_artifact_target("artifact:unbound.md", bindings) == "unbound.md"
    assert resolve_artifact_target("artifacts/unbound.md", bindings) is None
    assert resolve_artifact_target("https://example.com/report.md", bindings) is None
