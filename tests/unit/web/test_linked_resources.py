# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A real linked image is validated without nesting resource controls in anchors."""

from pathlib import Path

import pytest
from PIL import Image

from dlightrag.adapters.http.browser.presentation import build_answer_presentation
from dlightrag.engine.answer.publication import (
    PublicationPlan,
    prepare_artifact_attachment,
    validate_publication,
)
from dlightrag.engine.answer.results import project_answer_result


def _present(plan: PublicationPlan):
    projected = project_answer_result(
        {
            "answer": plan.answer,
            "artifacts": plan.descriptors,
            "artifact_bindings": plan.artifact_bindings,
            "artifact_outcome": plan.outcome,
        },
        run_id="run",
        artifact_url_prefix="/web/api/answer",
    )
    return build_answer_presentation(
        answer=projected["answer"],
        sources=[],
        evidence_images=[],
        artifacts=projected["artifacts"],
        artifact_bindings=projected["artifact_bindings"],
        artifact_outcome=projected["artifact_outcome"],
    )


@pytest.mark.parametrize(
    "answer",
    [
        "Before [![Report](artifacts/report.png)](https://example.com) after.",
        "Before [![Report][image]][outside] after.\n\n"
        "[image]: artifacts/report.png\n[outside]: https://example.com",
    ],
)
def test_invalid_linked_image_surfaces_one_failure_without_nested_anchor(
    tmp_path: Path, answer: str
) -> None:
    plan = validate_publication(tmp_path / "artifacts", answer=answer)
    assert plan.repairable
    assert plan.outcome["status"] == "failed"
    assert plan.answer == answer
    assert "artifacts/report.png" in plan.artifact_bindings

    presentation = _present(plan)
    rendered = presentation.parts[0].html or ""
    assert '<span class="answer-resource-slot-0"></span>' in rendered
    assert "<a " not in rendered
    assert "<img" not in rendered
    assert len(presentation.parts) == 2
    assert presentation.parts[1].artifact is not None
    assert presentation.parts[1].artifact.status == "unavailable"


def test_valid_linked_artifact_is_a_passive_image_inside_its_original_external_anchor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    Image.new("RGB", (2, 2), "blue").save(root / "report.png")
    attachment = prepare_artifact_attachment(root, path="report.png", label="Report")
    answer = 'Before [![Report](artifact:report.png "chart")](https://example.com "site") after.'
    plan = validate_publication(root, answer=answer, attachments=(attachment,))
    assert not plan.repairable
    assert plan.answer == answer
    assert len(plan.artifacts) == 1

    presentation = _present(plan)
    rendered = presentation.parts[0].html or ""
    assert '<a href="https://example.com"' in rendered
    assert 'title="site"' in rendered
    assert f'<img src="/web/api/answer/run/artifacts/{plan.artifacts[0].resource_id}"' in rendered
    assert 'alt="Report"' in rendered
    assert "answer-resource-slot-" not in rendered
    assert len(presentation.parts) == 1


def test_linked_non_image_artifact_becomes_a_normal_actionable_resource(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "table.csv").write_text("a,b\n1,2\n")
    attachment = prepare_artifact_attachment(root, path="table.csv", label="Table")
    plan = validate_publication(
        root,
        answer="[![Table](artifact:table.csv)](https://example.com)",
        attachments=(attachment,),
    )

    presentation = _present(plan)
    assert not plan.repairable
    assert len(presentation.parts) == 2
    assert presentation.parts[1].artifact is not None
    assert presentation.parts[1].artifact.status == "available"
    assert not presentation.parts[1].inline
    assert "<a " not in (presentation.parts[0].html or "")
    assert "<img" not in (presentation.parts[0].html or "")


def test_linked_evidence_uses_passive_image_and_is_not_duplicated_in_default_region() -> None:
    presentation = build_answer_presentation(
        answer="[![Evidence](evidence:shot)](https://example.com)",
        sources=[],
        evidence_images=[
            {
                "id": "shot",
                "chunk_id": "chunk",
                "source_ref": "1",
                "url": "/web/api/images/shot",
                "thumbnail_url": "/web/api/images/shot?size=thumb",
                "label": "Evidence",
            }
        ],
    )
    rendered = presentation.parts[0].html or ""
    assert '<a href="https://example.com"' in rendered
    assert '<img src="/web/api/images/shot"' in rendered
    assert "answer-resource-slot-" not in rendered
    assert presentation.evidence_images == []


@pytest.mark.parametrize(
    "answer",
    [
        "`[![Report](artifacts/report.png)](https://example.com)`",
        "```md\n[![Report](artifacts/report.png)](https://example.com)\n```",
    ],
)
def test_linked_image_code_examples_do_not_validate_or_place_resources(
    tmp_path: Path, answer: str
) -> None:
    plan = validate_publication(tmp_path / "artifacts", answer=answer)
    assert not plan.repairable
    assert plan.artifact_bindings == {}
    presentation = _present(plan)
    rendered = presentation.parts[0].html or ""
    assert "<code" in rendered
    assert "<img" not in rendered
    assert "<a " not in rendered
    assert "answer-resource-slot-" not in rendered
    assert len(presentation.parts) == 1
