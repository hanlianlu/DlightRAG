# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical Answer results expose shared usage and Evidence summaries."""

from dlightrag.engine.agent.skill_protection import MASK_PLACEHOLDER, ProtectedTextIndex
from dlightrag.engine.answer.citations.contracts import ChunkSnippet, SourceReference
from dlightrag.engine.answer.results import (
    project_answer_result,
    project_artifact_sources,
    restore_answer_result,
    store_answer_result,
)


def test_settled_bindings_round_trip_and_place_only_real_references() -> None:
    from dlightrag.adapters.http.browser.answer_events import render_done_event
    from dlightrag.adapters.http.client.client import AnswerResult
    from dlightrag.adapters.http.rest.models import AnswerResponse

    answer = "Before **[Report][r]** after.\n\n`[Report](./report.html)`\n\n[r]: ./report.html"
    binding = {"./report.html": "unavailable-report"}
    artifact = {
        "resource_id": "unavailable-report",
        "media_type": "application/octet-stream",
        "label": "Report",
        "filename": "report.html",
        "byte_size": 0,
        "digest": "",
        "presentation": "download",
        "status": "unavailable",
    }
    stored = store_answer_result(
        answer=answer,
        contexts={},
        sources=[],
        evidence_images=[],
        trace={},
        image_descriptions=[],
        artifacts=[artifact],
        artifact_bindings=binding,
        artifact_outcome={"status": "failed", "issues": []},
    )

    projected = project_answer_result(stored, run_id="run-1")
    assert projected["answer"] == answer
    assert projected["artifact_bindings"] == binding
    assert projected["parts"][0] == {"type": "markdown", "text": answer}
    assert [part["type"] for part in projected["parts"]] == ["markdown", "artifact"]
    assert projected["parts"][1]["target"] == "./report.html"
    assert projected["parts"][1]["slot"] == 0
    assert restore_answer_result(stored).artifact_bindings == binding
    client_result = AnswerResult.from_payload(projected)
    assert client_result.artifact_bindings == binding
    assert client_result.parts[1].target == "./report.html"
    assert client_result.parts[1].slot == 0
    assert AnswerResponse.model_validate(projected).artifact_bindings == binding

    done = render_done_event(
        {"result": stored}, downloadable_workspaces=None, visual_workspaces=None, run_id="run-1"
    )
    assert done.presentation is not None
    assert done.presentation.answer_text == answer
    assert len(done.presentation.parts) == 2
    rendered = done.presentation.parts[0].html
    assert 'Before <strong><span class="answer-resource-slot-0"></span></strong> after.' in rendered
    assert "<code>[Report](./report.html)</code>" in rendered
    assert "<a " not in rendered
    assert done.presentation.parts[1].artifact is not None
    assert done.presentation.parts[1].artifact.status == "unavailable"


def test_usage_and_evidence_round_trip_on_every_projection() -> None:
    stored = store_answer_result(
        answer="Grounded answer.",
        contexts={
            "chunks": [{"chunk_id": "c1", "content": "fact", "metadata": {}}],
            "entities": [{"entity": "one"}],
            "relationships": [],
        },
        sources=[],
        evidence_images=[],
        trace={"usage": {"usage_details": {"total_tokens": 12}}},
        image_descriptions=[],
    )

    assert stored["usage"] == {"usage_details": {"total_tokens": 12}}
    assert stored["evidence"] == {
        "chunks": 1,
        "entities": 1,
        "relationships": 0,
        "sources": 0,
    }
    restored = restore_answer_result(stored)
    projected = project_answer_result(stored)
    assert restored.usage == stored["usage"]
    assert restored.evidence == stored["evidence"]
    assert projected["usage"] == stored["usage"]
    assert projected["evidence"] == stored["evidence"]
    assert restored.artifact_outcome.status == "complete"
    assert projected["parts"] == [{"type": "markdown", "text": "Grounded answer."}]


def test_artifact_source_snapshots_round_trip_by_resource_id() -> None:
    source = SourceReference(
        id="2",
        title="Appendix source",
        type="document",
        source_uri="local://default/appendix.pdf",
        workspace="default",
        document_id="doc-appendix",
        download_locator="/private/appendix.pdf",
        cited_chunk_ids=["chunk-2"],
        chunks=[
            ChunkSnippet(
                chunk_id="chunk-2",
                chunk_idx=1,
                page_number=4,
                content="Appendix evidence.",
            )
        ],
    )
    stored = store_answer_result(
        answer="Artifact ready.",
        contexts={},
        sources=[],
        evidence_images=[],
        trace={},
        image_descriptions=[],
        artifact_sources={"artifact-appendix": [source]},
    )

    projected = project_artifact_sources(
        stored,
        resource_id="artifact-appendix",
    )

    assert list(stored["artifact_sources"]) == ["artifact-appendix"]
    assert [value.id for value in projected] == ["2"]
    assert projected[0].chunks is not None
    assert projected[0].chunks[0].content == "Appendix evidence."
    assert (
        project_artifact_sources(
            stored,
            resource_id="other-artifact",
        )
        == []
    )


def test_parts_derive_artifact_and_inline_evidence_placements() -> None:
    artifact = {
        "resource_id": "artifact-report",
        "media_type": "text/markdown",
        "label": "Report",
        "filename": "report.md",
        "byte_size": 10,
        "digest": "a" * 64,
        "presentation": "markdown",
        "status": "available",
    }
    stored = {
        "answer": (
            "Intro. [View report](artifact:artifact-report) ![Inline chart](evidence:chart-1) End."
        ),
        "sources": [],
        "contexts": {},
        "evidence_images": [
            {
                "id": "chart-1",
                "chunk_id": "chunk-1",
                "workspace": "default",
                "source_ref": "1",
                "label": "Chart",
            }
        ],
        "artifacts": [artifact],
        "artifact_outcome": {"status": "complete", "issues": []},
    }

    projected = project_answer_result(
        stored,
        run_id="run-1",
        artifact_url_prefix="/answer",
    )

    assert [part["type"] for part in projected["parts"]] == [
        "markdown",
        "artifact",
        "evidence_image",
    ]
    assert "role" not in projected["parts"][1]["artifact"]
    assert projected["parts"][1]["artifact"]["data_url"].endswith(
        "/run-1/artifacts/artifact-report"
    )
    assert projected["parts"][0]["text"] == stored["answer"]
    assert projected["parts"][1]["target"] == "artifact:artifact-report"
    assert projected["parts"][1]["slot"] == 0
    assert projected["parts"][2]["evidence_image"]["source_ref"] == "1"
    assert projected["parts"][2]["slot"] == 1


_PROTECTED_DOCUMENT = (
    "Every claim in a draft is checked against the passage it cites before the answer leaves "
    "the Run. When the passage does not support the claim, the sentence is removed instead of "
    "softened, and the gap is named in the same paragraph. The check runs again on the final "
    "draft in the same order, so a reader can follow what changed between two versions "
    "without reading the whole ledger. Nothing is filled in from memory: an absence is stated, "
    "not guessed, because a confident sentence that no evidence carries is the one failure this "
    "method exists to prevent."
)


def test_a_reader_projection_masks_a_protected_document_in_the_answer_and_contexts() -> None:
    """The reader's project is masked; the record the Run continues from is not (ADR 0061)."""
    from dlightrag.engine.agent.skill_protection import MASK_PLACEHOLDER, ProtectedTextIndex

    index = ProtectedTextIndex.build(
        [("method", "references/method.md", _PROTECTED_DOCUMENT.encode("utf-8"))]
    )
    protected_source = SourceReference(
        id="1",
        title="method.md",
        type="document",
        source_uri="local://default/method.md",
        workspace="default",
        document_id="doc-method",
        download_locator="/private/method.md",
        cited_chunk_ids=["c1"],
        chunks=[
            ChunkSnippet(
                chunk_id="c1",
                chunk_idx=1,
                page_number=2,
                content=_PROTECTED_DOCUMENT,
            )
        ],
    )
    ordinary_source = SourceReference(
        id="2",
        title="filings.pdf",
        type="document",
        source_uri="local://default/filings.pdf",
        workspace="default",
        document_id="doc-filings",
        download_locator="/private/filings.pdf",
        cited_chunk_ids=["c2"],
        chunks=[ChunkSnippet(chunk_id="c2", content="Revenue of twelve million.")],
    )
    stored = store_answer_result(
        answer=f"Here is the method:\n\n{_PROTECTED_DOCUMENT}\n\nThat is it.",
        contexts={
            "chunks": [
                {"chunk_id": "c1", "content": _PROTECTED_DOCUMENT, "metadata": {}},
            ],
            "entities": [],
            "relationships": [],
        },
        sources=[protected_source, ordinary_source],
        evidence_images=[],
        trace={},
        image_descriptions=[],
    )

    projected = project_answer_result(stored, mask_reader_text=index.mask)

    assert _PROTECTED_DOCUMENT not in projected["answer"]
    assert MASK_PLACEHOLDER in projected["answer"]
    assert projected["contexts"]["chunks"][0]["content"] == MASK_PLACEHOLDER
    # The source panel renders the same passage through a second field: its chunk text is the
    # document, so it passes the same gate while the facts around it stay readable.
    projected_sources = projected["sources"]
    assert projected_sources[0]["id"] == "1"
    assert projected_sources[0]["title"] == "method.md"
    assert projected_sources[0]["source_uri"] == "local://default/method.md"
    assert projected_sources[0]["chunks"][0]["content"] == MASK_PLACEHOLDER
    assert projected_sources[0]["chunks"][0]["chunk_id"] == "c1"
    assert projected_sources[0]["chunks"][0]["page_number"] == 2
    assert projected["references"] == [
        {"id": "1", "title": "method.md"},
        {"id": "2", "title": "filings.pdf"},
    ]
    assert projected_sources[1]["chunks"][0]["content"] == "Revenue of twelve million."
    # The bytes stay whole in the record: masking is a display action.
    assert _PROTECTED_DOCUMENT in stored["answer"]
    assert _PROTECTED_DOCUMENT in stored["contexts"]["chunks"][0]["content"]
    assert _PROTECTED_DOCUMENT in stored["sources"][0]["chunks"][0]["content"]


def test_a_projection_without_contexts_does_not_pay_the_gate_for_them() -> None:
    """The conversation page draws no retrieval contexts, so it neither projects nor masks them."""
    from dlightrag.engine.agent.skill_protection import MASK_PLACEHOLDER, ProtectedTextIndex

    index = ProtectedTextIndex.build(
        [("method", "references/method.md", _PROTECTED_DOCUMENT.encode("utf-8"))]
    )
    stored = store_answer_result(
        answer=f"Here is the method:\n\n{_PROTECTED_DOCUMENT}\n\nThat is it.",
        contexts={
            "chunks": [
                {"chunk_id": "c1", "content": _PROTECTED_DOCUMENT, "metadata": {}},
            ],
            "entities": [],
            "relationships": [],
        },
        sources=[
            SourceReference(
                id="1",
                title="method.md",
                source_uri="local://default/method.md",
                workspace="default",
                document_id="doc-method",
                download_locator="/private/method.md",
                chunks=[ChunkSnippet(chunk_id="c1", content=_PROTECTED_DOCUMENT)],
            )
        ],
        evidence_images=[],
        trace={},
        image_descriptions=[],
    )

    projected = project_answer_result(stored, mask_reader_text=index.mask, include_contexts=False)

    assert projected["contexts"] == {}
    assert MASK_PLACEHOLDER in projected["answer"]
    assert projected["sources"][0]["chunks"][0]["content"] == MASK_PLACEHOLDER


def test_artifact_source_citations_pass_the_reader_content_gate() -> None:
    """An Artifact's citation list is another copy of the evidence passages (ADR 0061)."""
    from dlightrag.engine.agent.skill_protection import MASK_PLACEHOLDER, ProtectedTextIndex

    index = ProtectedTextIndex.build(
        [("method", "references/method.md", _PROTECTED_DOCUMENT.encode("utf-8"))]
    )
    protected_source = SourceReference(
        id="1",
        title="method.md",
        source_uri="local://default/method.md",
        workspace="default",
        document_id="doc-method",
        download_locator="/private/method.md",
        chunks=[ChunkSnippet(chunk_id="c1", page_number=2, content=_PROTECTED_DOCUMENT)],
    )
    ordinary_source = SourceReference(
        id="2",
        title="filings.pdf",
        source_uri="local://default/filings.pdf",
        workspace="default",
        document_id="doc-filings",
        download_locator="/private/filings.pdf",
        chunks=[ChunkSnippet(chunk_id="c2", content="Revenue of twelve million.")],
    )
    stored = store_answer_result(
        answer="Artifact ready.",
        contexts={},
        sources=[],
        evidence_images=[],
        trace={},
        image_descriptions=[],
        artifact_sources={"artifact-report": [protected_source, ordinary_source]},
    )

    projected = project_artifact_sources(
        stored,
        resource_id="artifact-report",
        mask_reader_text=index.mask,
    )

    assert projected[0].id == "1"
    assert projected[0].chunks is not None
    assert projected[0].chunks[0].content == MASK_PLACEHOLDER
    assert projected[0].chunks[0].page_number == 2
    assert projected[1].chunks is not None
    assert projected[1].chunks[0].content == "Revenue of twelve million."
    # A caller that does not pass a gate -- the model's own reads -- still gets the record.
    ungated = project_artifact_sources(stored, resource_id="artifact-report")
    assert ungated[0].chunks is not None
    assert ungated[0].chunks[0].content == _PROTECTED_DOCUMENT


def test_a_reproduction_split_into_markdown_links_is_withheld() -> None:
    """The gate reads the page, not the source: links split a run the reader sees whole."""
    document = (
        "The run ledger is the first thing a calculation states: every input carries the period "
        "it was observed in, the source that stated it, and the mechanism by which it reaches "
        "the conclusion. An input nobody can date is an input nobody can defend."
    )
    index = ProtectedTextIndex.build([("method", "references/method.md", document.encode())])
    linked = "".join(f"[{document[offset : offset + 40]}](#)" for offset in range(0, 400, 40))
    stored = {"answer": linked, "sources": [], "contexts": {"chunks": []}}

    projected = project_answer_result(stored, mask_reader_text=index.mask)

    assert projected["answer"] == MASK_PLACEHOLDER
